import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open


def _add_mar_to_path(mar_root: Path) -> None:
    mar_root = mar_root.resolve()
    if str(mar_root) not in sys.path:
        sys.path.insert(0, str(mar_root))


def _load_diffloss(mar_root: Path, weights_path: Path, device: torch.device, sampling_steps: str):
    _add_mar_to_path(mar_root)
    from models.diffloss import DiffLoss

    diffloss = DiffLoss(
        target_channels=16,
        z_channels=768,
        depth=6,
        width=1024,
        num_sampling_steps=sampling_steps,
        grad_checkpointing=False,
    )

    state = {}
    prefix = "diffloss."
    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(prefix):
                state[key[len(prefix):]] = f.get_tensor(key)
    missing, unexpected = diffloss.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"DiffLoss load_state_dict missing={missing}, unexpected={unexpected}")

    diffloss = diffloss.to(device=device, dtype=torch.float32).eval()
    for param in diffloss.parameters():
        param.requires_grad_(False)
    return diffloss


def _load_vae(mar_root: Path, vae_path: Path, device: torch.device):
    _add_mar_to_path(mar_root)
    from models.vae import AutoencoderKL

    vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path=str(vae_path))
    vae = vae.to(device=device, dtype=torch.float32).eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    return vae


def _to_uint8_image(decoded: torch.Tensor) -> Image.Image:
    image = decoded.detach().cpu().clamp(-1, 1)
    image = ((image + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    image = image[0].permute(1, 2, 0).numpy()
    return Image.fromarray(image)


@torch.no_grad()
def generate_zero_condition_image(diffloss, vae, args, device: torch.device) -> Path:
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    z = torch.zeros(args.image_tokens, 768, device=device)
    noise = torch.randn(args.image_tokens, 16, device=device, generator=generator) * args.temperature
    sample = diffloss.gen_diffusion.p_sample_loop(
        diffloss.net.forward,
        noise.shape,
        noise,
        clip_denoised=False,
        model_kwargs={"c": z},
        progress=False,
        temperature=args.temperature,
    )
    side = int(args.image_tokens ** 0.5)
    latents = sample.view(1, side, side, 16).permute(0, 3, 1, 2).contiguous()
    decoded = vae.decode(latents / args.scaling_factor)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    _to_uint8_image(decoded).save(args.output_png)
    return args.output_png


@torch.no_grad()
def compute_zero_condition_mse(diffloss, args, device: torch.device) -> dict:
    cache = torch.load(args.imagenet_cache, map_location="cpu")
    latents = cache["latents"].float()
    num_images = min(args.num_images, latents.shape[0])
    target = latents[:num_images].reshape(-1, 16)
    if args.max_tokens > 0:
        target = target[:args.max_tokens]
    target = target.to(device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1)
    noise = torch.randn(target.shape, device=device, generator=generator)
    t = torch.randint(
        0,
        diffloss.train_diffusion.num_timesteps,
        (target.shape[0],),
        device=device,
        generator=generator,
    )
    x_t = diffloss.train_diffusion.q_sample(target, t, noise=noise)
    z = torch.zeros(target.shape[0], 768, device=device)
    model_output = diffloss.net(x_t, t, z)
    eps_pred = model_output[:, :16]
    mse = F.mse_loss(eps_pred, noise).item()
    zero_pred_mse = F.mse_loss(torch.zeros_like(noise), noise).item()
    pred_rms = eps_pred.pow(2).mean().sqrt().item()
    noise_rms = noise.pow(2).mean().sqrt().item()
    return {
        "model": "jadechoghari/mar mar-base.safetensors",
        "head": "MAR-B DiffLoss",
        "condition": "zeros",
        "num_images": int(num_images),
        "num_tokens": int(target.shape[0]),
        "diffusion_steps": int(diffloss.train_diffusion.num_timesteps),
        "epsilon_mse": mse,
        "zero_predictor_epsilon_mse": zero_pred_mse,
        "pred_epsilon_rms": pred_rms,
        "noise_rms": noise_rms,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mar_root", type=Path, default=Path("../mar"))
    parser.add_argument("--weights", type=Path, default=Path("../mar/pretrained_models/mar/mar_base_hf/mar-base.safetensors"))
    parser.add_argument("--vae", type=Path, default=Path("public/vae/mar-kl16/kl16.ckpt"))
    parser.add_argument("--imagenet_cache", type=Path, default=Path("public/datasets/imagenet_full/vae_latents_mar_kl16/flow_latents_all_fp16.pt"))
    parser.add_argument("--output_png", type=Path, default=Path("output/mar_zero_condition/mar_base_zero_condition.png"))
    parser.add_argument("--metrics_json", type=Path, default=Path("output/mar_zero_condition/mar_base_zero_condition_metrics.json"))
    parser.add_argument("--sampling_steps", default="100")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--scaling_factor", type=float, default=0.2325)
    parser.add_argument("--image_tokens", type=int, default=256)
    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    diffloss = _load_diffloss(args.mar_root, args.weights, device, args.sampling_steps)
    vae = _load_vae(args.mar_root, args.vae, device)

    png_path = generate_zero_condition_image(diffloss, vae, args, device)
    metrics = compute_zero_condition_mse(diffloss, args, device)
    metrics["output_png"] = str(png_path)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
