"""
Open-MAGVIT2 wrapper for image tokenization.

Uses the SEED-Voken codebase from GitHub to load the 262K LFQ checkpoint.
Encodes 256x256 images to discrete tokens and decodes back for verification.

Codebook: 262,144 (2^18) via Lookup-Free Quantization (LFQ)
Tokens per image: 256 (16x16 grid, single quantizer, no product quantization)

Usage:
    from scripts.magvit2_wrapper import MAGVIT2Wrapper
    wrapper = MAGVIT2Wrapper("public/models/open-magvit2-262144")
    tokens = wrapper.encode(pil_image)  # -> torch.LongTensor [256] (indices in [0, 262143])
    image = wrapper.decode(tokens)       # -> PIL.Image
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T


def _find_seed_voken_path() -> Path:
    """Find the SEED-Voken codebase in tmp or from env var."""
    env_path = os.environ.get("SEED_VOKEN_PATH")
    if env_path:
        return Path(env_path)

    candidate = Path("/tmp/SEED-Voken")
    if candidate.exists() and (candidate / "src" / "Open_MAGVIT2").exists():
        return candidate

    raise FileNotFoundError(
        "SEED-Voken codebase not found. Set SEED_VOKEN_PATH env var "
        "or clone https://github.com/TencentARC/SEED-Voken to /tmp/SEED-Voken"
    )


SEED_VOKEN_PATH = _find_seed_voken_path()


class MAGVIT2Wrapper:
    """Wrapper around Open-MAGVIT2 262K LFQ for image encode/decode."""

    def __init__(
        self,
        model_dir: str = "public/models/open-magvit2-262144",
        device: str = "cuda",
    ):
        self.device = device
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            self.model_dir = project_root / self.model_dir

        self.image_size = 256
        self.codebook_size = 262144  # 2^18
        self.embed_dim = 18  # log2(262144)
        self.num_latent_tokens = 256  # 16x16 grid
        self.total_tokens = 256

        self.model = self._build_model()
        self.model.to(device)
        self.model.eval()

        # Image preprocessing: resize to 256x256, normalize to [-1, 1]
        self._preprocess = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _build_model(self):
        """Build VQModel from SEED-Voken codebase and load checkpoint."""
        if str(SEED_VOKEN_PATH) not in sys.path:
            sys.path.insert(0, str(SEED_VOKEN_PATH))

        orig_cwd = os.getcwd()
        os.chdir(SEED_VOKEN_PATH)
        try:
            from src.Open_MAGVIT2.models.lfqgan_pretrain import VQModel
            from src.Open_MAGVIT2.modules.losses.vqperceptual import VQLPIPSWithDiscriminator

            ddconfig = {
                "double_z": False,
                "z_channels": self.embed_dim,
                "resolution": 256,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                "ch_mult": [1, 1, 2, 2, 4],
                "num_res_blocks": 4,
            }

            lossconfig = {
                "target": "src.Open_MAGVIT2.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
                "params": {
                    "disc_conditional": False,
                    "disc_in_channels": 3,
                    "disc_start": 0,
                    "disc_num_layers": 3,
                    "disc_weight": 0.8,
                    "gen_loss_weight": 0.1,
                    "lecam_loss_weight": 0.05,
                    "codebook_weight": 0.1,
                    "commit_weight": 0.25,
                    "codebook_enlarge_ratio": 0,
                    "codebook_enlarge_steps": 2000,
                    "disc_loss": "hinge",
                    "disc_num_channels": 3,
                    "disc_num_stages": 3,
                    "disc_hidden_channels": 128,
                    "blur_resample": True,
                    "blur_kernel_size": 4,
                },
            }

            model = VQModel(
                ddconfig=ddconfig,
                lossconfig=lossconfig,
                n_embed=self.codebook_size,
                embed_dim=self.embed_dim,
                sample_minimization_weight=1.0,
                batch_maximization_weight=1.0,
                ckpt_path=str(self.model_dir / "pretrain256_262144.ckpt"),
                use_ema=True,
            )

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
            torch.LongTensor of shape [256] — spatial raster-scan order tokens.
            Values in [0, 262143].
        """
        if isinstance(image, Image.Image):
            image = self._preprocess(image)

        if image.dim() == 3:
            image = image.unsqueeze(0)  # [1, 3, 256, 256]

        image = image.to(self.device, non_blocking=True)

        # encode returns (quant, emb_loss, info, loss_breakdown)
        # info is the flattened token indices [256]
        _, _, indices, _ = self.model.encode(image)

        return indices.cpu()

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> Image.Image:
        """
        Decode discrete codebook indices back to PIL Image.

        Args:
            tokens: torch.LongTensor of shape [256] or [B, 256]

        Returns:
            PIL.Image
        """
        tokens = tokens.to(self.device)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)  # [B, 256]

        B = tokens.shape[0]

        # LFQ.decode: indices -> {-1,1}^18 float features [B, 256, 18]
        quant = self.model.quantize.decode(tokens)
        # Reshape to spatial: [B, 256, 18] -> [B, 18, 16, 16]
        quant = quant.reshape(B, 16, 16, 18).permute(0, 3, 1, 2).contiguous()

        # Decoder: [B, 18, 16, 16] -> [B, 3, 256, 256] in [-1, 1]
        recon = self.model.decoder(quant)

        # Convert [-1, 1] -> [0, 1] -> PIL
        recon = (recon + 1.0) / 2.0
        recon = recon.clamp(0, 1)
        recon = recon.squeeze(0).cpu()

        return T.ToPILImage()(recon)


if __name__ == "__main__":
    # Quick test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Path to test image")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    wrapper = MAGVIT2Wrapper(device=args.device)
    print(f"Open-MAGVIT2 loaded: codebook={wrapper.codebook_size}, tokens/img={wrapper.total_tokens}")

    if args.image:
        img = Image.open(args.image).convert("RGB")
        tokens = wrapper.encode(img)
        print(f"Encoded tokens: shape={tokens.shape}, range=[{tokens.min()}, {tokens.max()}]")
        recon = wrapper.decode(tokens)
        recon.save("magvit2_recon_test.png")
        print("Reconstruction saved to magvit2_recon_test.png")
