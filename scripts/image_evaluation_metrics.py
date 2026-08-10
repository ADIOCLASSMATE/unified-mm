"""Shared Inception feature, FID, and IS utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def build_inception_extractor(
    feature: int,
    weights_path,
    device,
):
    from torchmetrics.image.fid import NoTrainInceptionV3

    class FeaturesAndLogitsInceptionV3(NoTrainInceptionV3):
        def forward(self, images):
            return self._torch_fidelity_forward(images)

    extractor = FeaturesAndLogitsInceptionV3(
        name="inception-v3-compat",
        features_list=[str(int(feature)), "logits_unbiased"],
        feature_extractor_weights_path=(
            str(weights_path) if weights_path is not None else None
        ),
        antialias=True,
    )
    return extractor.to(device).eval()


def metric_accumulation_dtype(device):
    """Choose a distributed-reduction dtype supported by the accelerator."""
    import torch

    # HCCL does not implement float64 all-reduce. Accumulate and reduce the
    # sufficient statistics in fp32 on Ascend, then convert the resulting
    # means/covariances to CPU float64 in ``frechet_distance`` below.
    return (
        torch.float32
        if torch.device(device).type == "npu"
        else torch.float64
    )


def extract_inception_features(extractor, images):
    """Return FID features and unbiased logits in a reducible dtype."""
    import torch

    uint8_images = (
        images.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .to(torch.uint8)
    )
    outputs = extractor(uint8_images)
    if isinstance(outputs, torch.Tensor):
        raise RuntimeError(
            "Inception extractor returned one tensor; expected FID features "
            "and logits."
        )
    if isinstance(outputs, Mapping):
        feature_keys = [
            key for key in outputs if key != "logits_unbiased"
        ]
        feature = (
            outputs.get(feature_keys[0])
            if len(feature_keys) == 1
            else None
        )
        logits = outputs.get("logits_unbiased")
    else:
        feature, logits = outputs
    if feature is None or logits is None:
        raise RuntimeError(
            "Inception extractor did not return features and "
            "logits_unbiased."
        )
    dtype = metric_accumulation_dtype(feature.device)
    return (
        feature.reshape(feature.shape[0], -1).to(dtype=dtype),
        logits.reshape(logits.shape[0], -1).to(dtype=dtype),
    )


@dataclass
class FeatureMoments:
    count: Any
    sum: Any
    outer_sum: Any

    @classmethod
    def zeros(cls, dimension: int, device):
        import torch

        dtype = metric_accumulation_dtype(device)

        return cls(
            count=torch.zeros((), dtype=torch.long, device=device),
            sum=torch.zeros(
                int(dimension),
                dtype=dtype,
                device=device,
            ),
            outer_sum=torch.zeros(
                int(dimension),
                int(dimension),
                dtype=dtype,
                device=device,
            ),
        )

    def update(self, features) -> None:
        features = features.to(
            device=self.sum.device,
            dtype=self.sum.dtype,
        )
        self.count += int(features.shape[0])
        self.sum += features.sum(dim=0)
        self.outer_sum += features.T @ features

    def all_reduce_(self) -> None:
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return
        dist.all_reduce(self.count)
        dist.all_reduce(self.sum)
        dist.all_reduce(self.outer_sum)

    def mean_cov(self):
        count = int(self.count.item())
        if count < 2:
            raise ValueError(
                f"at least two features are required, found {count}"
            )
        mean = self.sum / count
        covariance = (
            self.outer_sum
            - count * mean[:, None] * mean[None, :]
        ) / (count - 1)
        return mean, (covariance + covariance.T) * 0.5


@dataclass
class InceptionScoreMoments:
    count: Any
    probability_sum: Any
    probability_log_probability_sum: Any

    @classmethod
    def zeros(cls, splits: int, classes: int, device):
        import torch

        dtype = metric_accumulation_dtype(device)

        return cls(
            count=torch.zeros(
                int(splits),
                dtype=torch.long,
                device=device,
            ),
            probability_sum=torch.zeros(
                int(splits),
                int(classes),
                dtype=dtype,
                device=device,
            ),
            probability_log_probability_sum=torch.zeros(
                int(splits),
                dtype=dtype,
                device=device,
            ),
        )

    def update(
        self,
        logits,
        global_indices: Sequence[int],
        total_samples: int,
    ) -> None:
        import torch

        probabilities = logits.to(
            device=self.probability_sum.device,
            dtype=self.probability_sum.dtype,
        ).softmax(dim=-1)
        indices = torch.as_tensor(
            global_indices,
            device=logits.device,
            dtype=torch.long,
        )
        split_ids = torch.div(
            indices * int(self.count.numel()),
            int(total_samples),
            rounding_mode="floor",
        ).clamp_max(self.count.numel() - 1)
        p_log_p = (
            probabilities
            * probabilities.clamp_min(
                torch.finfo(probabilities.dtype).tiny
            ).log()
        ).sum(dim=-1)
        for split in split_ids.unique().tolist():
            mask = split_ids == int(split)
            self.count[split] += int(mask.sum().item())
            self.probability_sum[split] += probabilities[mask].sum(dim=0)
            self.probability_log_probability_sum[split] += p_log_p[
                mask
            ].sum()

    def all_reduce_(self) -> None:
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return
        dist.all_reduce(self.count)
        dist.all_reduce(self.probability_sum)
        dist.all_reduce(self.probability_log_probability_sum)

    def compute(self) -> tuple[float, float, list[float]]:
        import torch

        scores = []
        for split in range(int(self.count.numel())):
            count = int(self.count[split].item())
            if count <= 0:
                raise ValueError(
                    f"Inception Score split {split} is empty"
                )
            marginal = self.probability_sum[split] / count
            expected_p_log_p = (
                self.probability_log_probability_sum[split] / count
            )
            marginal_entropy_term = (
                marginal
                * marginal.clamp_min(
                    torch.finfo(marginal.dtype).tiny
                ).log()
            ).sum()
            scores.append(
                float(
                    torch.exp(
                        expected_p_log_p - marginal_entropy_term
                    )
                )
            )
        values = torch.tensor(scores, dtype=torch.float64)
        return (
            float(values.mean()),
            float(values.std(unbiased=False)),
            scores,
        )


def frechet_distance(
    real_mean,
    real_cov,
    fake_mean,
    fake_cov,
) -> float:
    """Numerically stable FID using symmetric eigendecompositions."""
    import torch

    real_mean = real_mean.detach().cpu().double()
    fake_mean = fake_mean.detach().cpu().double()
    real_cov = (
        real_cov.detach().cpu().double()
        + real_cov.T.detach().cpu().double()
    ) * 0.5
    fake_cov = (
        fake_cov.detach().cpu().double()
        + fake_cov.T.detach().cpu().double()
    ) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(real_cov)
    real_sqrt = (
        eigenvectors
        * eigenvalues.clamp_min(0).sqrt().unsqueeze(0)
    ) @ eigenvectors.T
    middle = real_sqrt @ fake_cov @ real_sqrt
    middle = (middle + middle.T) * 0.5
    trace_sqrt_product = (
        torch.linalg.eigvalsh(middle)
        .clamp_min(0)
        .sqrt()
        .sum()
    )
    difference = real_mean - fake_mean
    fid = (
        difference.dot(difference)
        + torch.trace(real_cov)
        + torch.trace(fake_cov)
        - 2.0 * trace_sqrt_product
    )
    return float(fid.clamp_min(0))
