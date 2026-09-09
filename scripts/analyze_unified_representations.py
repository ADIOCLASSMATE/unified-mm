"""CPU analysis for the frozen B representation diagnostic (not a benchmark)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def normalize(features):
    return F.normalize(features.float(), dim=-1, eps=1e-8)


def retrieval_scores(scores, image_groups, text_groups):
    image_groups = torch.as_tensor(image_groups)
    text_groups = torch.as_tensor(text_groups)
    positives = image_groups[:, None].eq(text_groups[None, :])
    if not bool(positives.any(0).all() and positives.any(1).all()):
        raise ValueError("every query must have a positive in the candidate pool")
    result = {}
    for label, values, matches in (
        ("i2t", scores, positives),
        ("t2i", scores.T, positives.T),
    ):
        # Stable sorting gives deterministic handling of identical/collapsed features.
        ordered = values.argsort(dim=1, descending=True, stable=True)
        for k in (1, 5, 10):
            hit = matches.gather(1, ordered[:, : min(k, values.shape[1])]).any(1)
            result[f"{label}_r{k}"] = float(hit.float().mean() * 100)
    result["positive_cosine"] = float(scores[positives].mean())
    result["negative_cosine"] = float(scores[~positives].mean())
    result["cosine_margin"] = result["positive_cosine"] - result["negative_cosine"]
    return result


def retrieval(images, texts, image_groups, text_groups):
    return retrieval_scores(
        normalize(images) @ normalize(texts).T, image_groups, text_groups
    )


def ridge_fit(x, y, strength=0.1):
    x, y = x.float(), y.float()
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    covariance = xc.T @ xc
    penalty = max(float(covariance.trace() / covariance.shape[0]) * strength, 1e-6)
    covariance.diagonal().add_(penalty)
    weight = torch.linalg.solve(covariance, xc.T @ yc)
    return mx, my, weight


def ridge_predict(fitted, x, input_mean=None):
    mx, my, weight = fitted
    return (x - (mx if input_mean is None else input_mean)) @ weight + my


def top1(logits, labels):
    return float(logits.argmax(-1).eq(labels).float().mean() * 100)


def linear_cka(x, y):
    x, y = x - x.mean(0), y - y.mean(0)
    gx, gy = x @ x.T, y @ y.T
    denominator = gx.square().sum().sqrt() * gy.square().sum().sqrt()
    return float((gx * gy).sum() / denominator.clamp_min(1e-12))


def load_features(root, dataset, expected, source):
    paths = sorted((root / "features").glob(f"{dataset}-rank-*-of-*.pt"))
    if not paths:
        raise FileNotFoundError(dataset)
    payloads = [
        torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        for path in paths
    ]
    indices = torch.cat([row["indices"] for row in payloads])
    if sorted(indices.tolist()) != list(range(expected)):
        raise ValueError(f"missing or duplicate sample indices: {dataset}")
    if len(paths) != payloads[0]["world_size"]:
        raise ValueError("incomplete rank coverage")
    if any(row["source"] != source for row in payloads):
        raise ValueError("mixed checkpoint sources")
    features = torch.empty(
        (expected, *payloads[0]["features"].shape[1:]), dtype=torch.bfloat16
    )
    for row in payloads:
        features[row["indices"]] = row["features"]
    return features


def imagenet_probe(images, texts, image_rows, text_rows):
    labels_i = torch.tensor([row["group"] for row in image_rows])
    labels_t = torch.tensor([row["group"] for row in text_rows])
    fit_i = torch.tensor([row["fit"] for row in image_rows])
    fit_t = torch.tensor([row["fit"] for row in text_rows])
    xi, xt = normalize(images), normalize(texts)
    yi = F.one_hot(labels_i, 1000).float()
    yt = F.one_hot(labels_t, 1000).float()
    image_model = ridge_fit(xi[fit_i], yi[fit_i])
    text_model = ridge_fit(xt[fit_t], yt[fit_t])
    result = {
        "image_within_top1": top1(
            ridge_predict(image_model, xi[~fit_i]), labels_i[~fit_i]
        ),
        "text_within_top1": top1(
            ridge_predict(text_model, xt[~fit_t]), labels_t[~fit_t]
        ),
        "text_to_image_top1": top1(
            ridge_predict(text_model, xi[~fit_i]), labels_i[~fit_i]
        ),
        "image_to_text_top1": top1(
            ridge_predict(image_model, xt[~fit_t]), labels_t[~fit_t]
        ),
        "text_to_image_centered_top1": top1(
            ridge_predict(text_model, xi[~fit_i], xi[fit_i].mean(0)), labels_i[~fit_i]
        ),
        "image_to_text_centered_top1": top1(
            ridge_predict(image_model, xt[~fit_t], xt[fit_t].mean(0)), labels_t[~fit_t]
        ),
    }
    prototypes = torch.stack(
        [texts[fit_t & labels_t.eq(index)].mean(0) for index in range(1000)]
    )
    result["text_prototype_top1"] = top1(
        normalize(images[~fit_i]) @ normalize(prototypes).T, labels_i[~fit_i]
    )
    mi, mt = images[fit_i].mean(0), texts[fit_t].mean(0)
    result["text_prototype_centered_top1"] = top1(
        normalize(images[~fit_i] - mi) @ normalize(prototypes - mt).T, labels_i[~fit_i]
    )
    return result


def flickr_alignment(images, texts, image_rows, text_rows):
    fit_i = torch.tensor([row["fit"] for row in image_rows])
    fit_t = torch.tensor([row["fit"] for row in text_rows])
    groups_i = torch.tensor([row["group"] for row in image_rows])
    groups_t = torch.tensor([row["group"] for row in text_rows])
    xi, xt = normalize(images), normalize(texts)
    if texts.shape[0] != images.shape[0] * 5:
        raise ValueError("Flickr mapping expects five grouped captions per image")
    targets = xt.reshape(len(images), 5, -1).mean(1)
    fitted = ridge_fit(xi[fit_i], targets[fit_i])
    predicted = ridge_predict(fitted, xi[~fit_i])
    mean_i, mean_t = images[fit_i].mean(0), texts[fit_t].mean(0)
    raw = retrieval(images[~fit_i], texts[~fit_t], groups_i[~fit_i], groups_t[~fit_t])
    centered = retrieval(
        images[~fit_i] - mean_i,
        texts[~fit_t] - mean_t,
        groups_i[~fit_i],
        groups_t[~fit_t],
    )
    mapped = retrieval(predicted, xt[~fit_t], groups_i[~fit_i], groups_t[~fit_t])
    # Fit an equally flexible map to permuted training pairs as a negative control.
    generator = torch.Generator().manual_seed(424242)
    permutation = torch.randperm(int(fit_i.sum()), generator=generator)
    shuffled = ridge_fit(xi[fit_i], targets[fit_i][permutation])
    shuffled_metrics = retrieval(
        ridge_predict(shuffled, xi[~fit_i]),
        xt[~fit_t],
        groups_i[~fit_i],
        groups_t[~fit_t],
    )
    held_images, held_text = xi[~fit_i], targets[~fit_i]
    permutation = torch.randperm(len(held_images), generator=generator)
    return {
        "raw": raw,
        "centered": centered,
        "ridge": mapped,
        "shuffled_pair_ridge": shuffled_metrics,
        "paired_linear_cka": linear_cka(held_images, held_text),
        "shuffled_linear_cka": linear_cka(held_images, held_text[permutation]),
    }


def input_controls(args, data, samples, protocol):
    from safetensors import safe_open

    from utils.evaluation.multimodal_likelihood import PosteriorCache
    from utils.research.representation_protocol import FLICKR

    with safe_open(
        str(Path(protocol["model"]["path"]) / "model.safetensors"), framework="pt"
    ) as handle:
        weight = handle.get_tensor("model.image_token_embedder.z_proj.weight").float()
        bias = handle.get_tensor("model.image_token_embedder.z_proj.bias").float()
    cache = PosteriorCache(
        FLICKR / "vae_posterior_mar_kl16/shards",
        expected_image_tokens=256,
        expected_latent_dim=16,
        seed=protocol["seed"],
    )
    latents = torch.stack(
        [
            cache.sample(row["image_id"]).float().mean(0)
            for row in samples["flickr_images"]
        ]
    )
    texts = data["flickr_texts"][:, 0, 0].float()
    gi = [row["group"] for row in samples["flickr_images"]]
    gt = [row["group"] for row in samples["flickr_texts"]]
    random_results = []
    for seed in range(5):
        generator = torch.Generator().manual_seed(424242 + seed)
        random_weight = torch.randn(weight.shape, generator=generator) * weight.std()
        random_images = latents @ random_weight.T
        random_results.append(
            {"seed": 424242 + seed, **retrieval(random_images, texts, gi, gt)}
        )
    return {
        "image_projector_shape": list(weight.shape),
        "singular_values": torch.linalg.svdvals(weight).tolist(),
        "bias_norm": float(bias.norm()),
        "random_projector_fixed_trained_text_embeddings": random_results,
        "note": "Random-projector controls are not exact step-zero models; VAE and Qwen are pretrained.",
    }


def plot_results(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for pool, color in ((0, "#2364aa"), (1, "#d66a20")):
        subset = [
            row
            for row in rows
            if row["pool_index"] == pool and row["layer"] != "final_norm"
        ]
        x = [int(row["layer"]) for row in subset]
        label = "Content mean" if pool == 0 else "Shared query"
        axes[0, 0].plot(
            x, [row["flickr_raw"]["i2t_r1"] for row in subset], label=label, color=color
        )
        axes[0, 1].plot(
            x,
            [row["flickr_centered"]["i2t_r1"] for row in subset],
            label=label,
            color=color,
        )
        axes[1, 0].plot(
            x,
            [row["imagenet"]["image_within_top1"] for row in subset],
            label=label + ": image probe",
            color=color,
        )
        axes[1, 0].plot(
            x,
            [row["imagenet"]["text_to_image_centered_top1"] for row in subset],
            label=label + ": text-to-image, centered",
            color=color,
            linestyle="--",
        )
        axes[1, 1].plot(
            x,
            [row["flickr_heldout"]["ridge"]["i2t_r1"] for row in subset],
            label=label + ": fitted mapping",
            color=color,
        )
        axes[1, 1].plot(
            x,
            [row["flickr_heldout"]["shuffled_pair_ridge"]["i2t_r1"] for row in subset],
            label=label + ": shuffled pairs",
            color=color,
            linestyle="--",
        )
    titles = [
        "Flickr30K I2T: raw cosine retrieval",
        "Flickr30K I2T: modality-centered cosine",
        "ImageNet-1K: frozen linear readout",
        "Flickr30K I2T: held-out linear alignment",
    ]
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xlabel("Backbone layer (0 = input embedding)")
        axis.set_ylabel("Recall@1 / top-1 (%)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0, 0].axhline(0.1, color="gray", linestyle=":")
    axes[0, 1].axhline(0.1, color="gray", linestyle=":")
    axes[1, 0].axhline(0.1, color="gray", linestyle=":")
    axes[1, 1].axhline(0.5, color="gray", linestyle=":")
    fig.suptitle(
        "B final EMA, step 95,415 | independent image/text encoding | frozen model"
    )
    fig.savefig(output / "layer_curves.png", dpi=170)
    fig.savefig(output / "layer_curves.pdf")
    plt.close(fig)


def main(args):
    from utils.research.representation_protocol import emit, write_json

    torch.set_num_threads(args.threads)
    protocol = json.loads((args.output_dir / "protocol.json").read_text())
    samples = json.loads((args.output_dir / "samples.json").read_text())
    data = {
        key: load_features(args.output_dir, key, len(value), protocol["model"])
        for key, value in samples.items()
    }
    emit("features_verified", counts={key: len(value) for key, value in data.items()})
    gi = [row["group"] for row in samples["flickr_images"]]
    gt = [row["group"] for row in samples["flickr_texts"]]
    layers = [str(value) for value in range(29)] + ["final_norm"]
    results = []
    for layer_index, layer in enumerate(layers):
        for pool in range(2):
            result_path = (
                args.output_dir / "analysis" / f"layer-{layer}-pool-{pool}.json"
            )
            if result_path.exists():
                results.append(json.loads(result_path.read_text()))
                continue
            fi, ft, ii, it = [
                data[key][:, layer_index, pool].float() for key in samples
            ]
            raw = retrieval(fi, ft, gi, gt)
            centered = retrieval(fi - fi.mean(0), ft - ft.mean(0), gi, gt)
            row = {
                "layer": layer,
                "pool_index": pool,
                "pool": protocol["pools"][pool],
                "flickr_raw": raw,
                "flickr_centered": centered,
                "image_feature_variance": float(fi.var(0).sum()),
                "text_feature_variance": float(ft.var(0).sum()),
                "modality_mean_cosine": float(
                    F.cosine_similarity(fi.mean(0), ft.mean(0), dim=0)
                ),
                "flickr_heldout": flickr_alignment(
                    fi, ft, samples["flickr_images"], samples["flickr_texts"]
                ),
                "imagenet": imagenet_probe(
                    ii, it, samples["imagenet_images"], samples["imagenet_texts"]
                ),
            }
            write_json(result_path, row)
            results.append(row)
            emit(
                "layer_analyzed",
                layer=layer,
                pool=pool,
                raw_r1=raw["i2t_r1"],
                centered_r1=centered["i2t_r1"],
                image_probe=row["imagenet"]["image_within_top1"],
                transfer=row["imagenet"]["text_to_image_centered_top1"],
            )
    controls = input_controls(args, data, samples, protocol)
    summary = {
        "schema": "unified_representation_diagnostic_results_v1",
        "complete": True,
        "protocol": protocol,
        "input_controls": controls,
        "layers": results,
    }
    write_json(args.output_dir / "results.json", summary)
    flat = []
    for row in results:
        flat.append(
            {
                "layer": row["layer"],
                "pool": row["pool"],
                **{
                    "flickr_raw_" + key: value
                    for key, value in row["flickr_raw"].items()
                },
                **{
                    "flickr_centered_" + key: value
                    for key, value in row["flickr_centered"].items()
                },
                **row["imagenet"],
                "heldout_ridge_i2t_r1": row["flickr_heldout"]["ridge"]["i2t_r1"],
                "heldout_shuffled_i2t_r1": row["flickr_heldout"]["shuffled_pair_ridge"][
                    "i2t_r1"
                ],
                "cka": row["flickr_heldout"]["paired_linear_cka"],
                "shuffled_cka": row["flickr_heldout"]["shuffled_linear_cka"],
            }
        )
    with (args.output_dir / "layer_metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    plot_results(results, args.output_dir)
    emit("analysis_complete", output=str(args.output_dir), layers=len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    main(parser.parse_args())
