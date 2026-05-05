"""
Minimal XQ-GAN VP2 wrapper for image tokenization.

Uses the ImageFolder codebase from GitHub to load the VP2-16384 checkpoint.
Encodes 256x256 images to discrete tokens and decodes back for verification.

Usage:
    from scripts.xqgan_wrapper import XQGANWrapper
    wrapper = XQGANWrapper("public/models/xqgan_vp2_16384")
    tokens = wrapper.encode(pil_image)  # -> torch.LongTensor [512] (2 quantizers × 256 positions)
    image = wrapper.decode(tokens)       # -> PIL.Image
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T


def _find_imagefolder_path() -> Path:
    """Find the ImageFolder codebase in uv git cache or from env var."""
    env_path = os.environ.get("IMAGEFOLDER_PATH")
    if env_path:
        return Path(env_path)

    checkouts_dir = Path(os.path.expanduser("~/.cache/uv/git-v0/checkouts"))
    if checkouts_dir.exists():
        for checkout_dir in checkouts_dir.iterdir():
            if not checkout_dir.is_dir():
                continue
            for inner_dir in checkout_dir.iterdir():
                marker = inner_dir / "tokenizer" / "tokenizer_image" / "xqgan_model.py"
                if marker.exists():
                    return inner_dir

    raise FileNotFoundError(
        "ImageFolder codebase not found. Set IMAGEFOLDER_PATH env var "
        "or clone https://github.com/lxa9867/ImageFolder and point to it."
    )


IMAGEFOLDER_PATH = _find_imagefolder_path()


class XQGANWrapper:
    """Wrapper around XQ-GAN VP2 16384 for image encode/decode."""

    def __init__(
        self,
        model_dir: str = "public/models/xqgan_vp2_16384",
        device: str = "cuda",
    ):
        self.device = device
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_absolute():
            # Resolve relative to project root (where scripts/ lives)
            project_root = Path(__file__).resolve().parent.parent
            self.model_dir = project_root / self.model_dir
        self.image_size = 256
        self.num_latent_tokens = 256
        self.product_quant = 2
        self.codebook_size = 16384
        # Total tokens per image: product_quant × num_latent_tokens
        self.total_tokens = self.product_quant * self.num_latent_tokens  # 512

        self.model = self._build_model()
        self.model.to(device)
        self.model.eval()

    def _build_model(self):
        """Build VQModel from ImageFolder codebase and load checkpoint."""
        # Add ImageFolder to path for imports
        if str(IMAGEFOLDER_PATH) not in sys.path:
            sys.path.insert(0, str(IMAGEFOLDER_PATH))

        orig_cwd = os.getcwd()
        os.chdir(IMAGEFOLDER_PATH)
        try:
            from tokenizer.tokenizer_image.xqgan_model import VQModel, ModelArgs

            config = ModelArgs(
                codebook_size=self.codebook_size,
                codebook_embed_dim=8,
                codebook_l2_norm=True,
                v_patch_nums=[16],
                enc_type="dinov2",
                dec_type="dinov2",
                semantic_guide="none",
                detail_guide="none",
                num_latent_tokens=self.num_latent_tokens,
                encoder_model="vit_base_patch14_dinov2.lvd142m",
                decoder_model="vit_base_patch14_dinov2.lvd142m",
                abs_pos_embed=True,
                product_quant=self.product_quant,
                share_quant_resi=4,
                codebook_drop=0.0,
                half_sem=False,
                start_drop=3,
                sem_loss_weight=0.1,
                test_model=True,
            )

            model = VQModel(config)

            ckpt_path = self.model_dir / "vq-16384" / "best_ckpt.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)

            # Only warn about structural missing keys, not training-only components
            structural_missing = [
                k for k in missing
                if not any(x in k for x in [
                    "semantic", "detail", "sem_linear", "sem_loss",
                    "clip_norm", "disc", "loss", "denormalize", "normalize",
                ])
            ]
            if structural_missing:
                print(f"Warning: {len(structural_missing)} structural keys missing from checkpoint")

            return model
        finally:
            os.chdir(orig_cwd)

    @torch.no_grad()
    def encode(self, image) -> torch.LongTensor:
        """
        Encode a PIL Image or tensor to discrete codebook indices.

        Args:
            image: PIL.Image (any size, will be resized) or torch.Tensor [3, 256, 256]

        Returns:
            torch.LongTensor of shape [512] — interleaved sub-codes:
                [q0_pos0, q1_pos0, q0_pos1, q1_pos1, ..., q0_pos255, q1_pos255]
            Values in [0, 16383].
        """
        if isinstance(image, Image.Image):
            image = self._preprocess(image)

        if image.dim() == 3:
            image = image.unsqueeze(0)  # [1, 3, 256, 256]

        image = image.to(self.device)

        # img_to_idxBl returns list of 2 elements (one per quantizer),
        # each is a list of 1 tensor of shape [256]
        idx_list = self.model.img_to_idxBl(image)

        # Interleave sub-codes: [q0_0, q1_0, q0_1, q1_1, ...]
        q0 = idx_list[0][0].cpu()  # [256]
        q1 = idx_list[1][0].cpu()  # [256]
        interleaved = torch.stack([q0, q1], dim=1).reshape(-1)  # [512]

        return interleaved

    @torch.no_grad()
    def encode_batch(self, images: torch.Tensor) -> List[torch.LongTensor]:
        """Encode a batch of images. Returns list of [512] tensors."""
        results = []
        for i in range(images.shape[0]):
            results.append(self.encode(images[i]))
        return results

    @torch.no_grad()
    def decode(self, tokens: torch.LongTensor) -> Image.Image:
        """
        Decode interleaved codebook indices back to an image.

        Args:
            tokens: torch.LongTensor of shape [512] (interleaved sub-codes)

        Returns:
            PIL.Image of size 256x256
        """
        tokens = tokens.to(self.device)
        # De-interleave: [q0_0, q1_0, q0_1, q1_1, ...] -> q0 [256], q1 [256]
        q0 = tokens[0::2]  # [256]
        q1 = tokens[1::2]  # [256]

        # Build idx_list format expected by the model
        idx_list = [[q0], [q1]]

        # Use the reconstruction pipeline
        pixel_values = self.model.img_to_reconstructed_img(
            torch.randn(1, 3, 256, 256, device=self.device)  # dummy, not used
        )

        # Actually use decode via fhat path
        # Convert idx to variable input
        f_hat_list = []
        for i, idx_bl_list in enumerate(idx_list):
            embedding = self.model.quantizes[i].embedding
            idx = idx_bl_list[0]  # [256]
            f_hat = embedding(idx)  # [256, 8]
            f_hat = f_hat.reshape(1, -1, 8)  # [1, 256, 8]
            permuted = f_hat.permute(0, 2, 1).unsqueeze(-1)  # [1, 8, 256, 1]
            f_hat_list.append(permuted)

        f_hat_cat = torch.cat(f_hat_list, dim=1)  # [1, 16, 256, 1]
        f_hat_cat = f_hat_cat.view(1, 16, 16, 16)

        pixel_values = self.model.fhat_to_img(f_hat_cat)  # [1, 3, 256, 256]
        pixel_values = pixel_values.clamp(-1, 1)
        pixel_values = (pixel_values + 1) / 2  # [-1,1] -> [0,1]
        pixel_values = pixel_values.squeeze(0).cpu()

        img = T.ToPILImage()(pixel_values)
        return img

    @torch.no_grad()
    def decode_batch(self, tokens_list: List[torch.LongTensor]) -> List[Image.Image]:
        """Decode a batch of token sequences."""
        return [self.decode(t) for t in tokens_list]

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL Image for model input."""
        transform = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return transform(image)

    def compute_reconstruction_metrics(
        self, images: List[Image.Image]
    ) -> dict:
        """
        Compute tokenizer reconstruction quality metrics on a set of images.
        Reports rFID placeholder and per-image PSNR.
        """
        psnr_values = []
        for img in images:
            tokens = self.encode(img)
            reconstructed = self.decode(tokens)
            original = T.ToTensor()(img.resize((256, 256)))
            recon_tensor = T.ToTensor()(reconstructed)
            mse = F.mse_loss(original, recon_tensor)
            psnr = 10 * torch.log10(1.0 / mse)
            psnr_values.append(psnr.item())

        return {
            "psnr_mean": sum(psnr_values) / len(psnr_values),
            "psnr_min": min(psnr_values),
            "psnr_max": max(psnr_values),
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="public/models/xqgan_vp2_16384")
    parser.add_argument("--test_image", type=str, default=None,
                        help="Path to a test image for encode/decode verification")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    wrapper = XQGANWrapper(args.model_dir, device=args.device)
    print(f"XQ-GAN VP2 loaded: {wrapper.total_tokens} tokens/image, "
          f"codebook_size={wrapper.codebook_size}, product_quant={wrapper.product_quant}")

    if args.test_image:
        img = Image.open(args.test_image).convert("RGB")
        tokens = wrapper.encode(img)
        print(f"Encoded: {tokens.shape}, range=[{tokens.min().item()}, {tokens.max().item()}]")
        recon = wrapper.decode(tokens)
        out_path = "/tmp/xqgan_recon.png"
        recon.save(out_path)
        print(f"Reconstructed saved to {out_path}")
    else:
        # Quick test with random noise
        print("No test image provided, running random noise sanity check...")
        dummy = torch.randn(3, 256, 256)
        tokens = wrapper.encode(dummy)
        assert tokens.shape == (512,), f"Expected [512], got {tokens.shape}"
        assert tokens.min() >= 0 and tokens.max() < 16384, \
            f"Token range [{tokens.min()}, {tokens.max()}] out of [0, 16384)"
        # Decode not tested with random noise (meaningless)
        print("Sanity check PASSED: encode returns [512] tokens in [0, 16384]")
