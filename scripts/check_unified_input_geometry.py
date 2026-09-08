"""Check token-level input MaxSim and uncertainty of the fixed final readout."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_unified_representations import (
    load_features,
    normalize,
    retrieval_scores,
)
from scripts.probe_unified_representations import FLICKR, emit, write_json


def token_maxsim(latents, word_vectors, weight, bias, caption_tokens):
    # Algebraically exact for FP32 e_image = Wz+b. The 16-channel factorization
    # avoids materializing every 1024-dimensional image/token pair.
    word_vectors = normalize(word_vectors)
    projected_words = word_vectors @ weight
    offsets = word_vectors @ bias
    maxima = []
    for start in range(0, len(latents), 8):
        z = latents[start : start + 8]
        norms = (z @ weight.T + bias).norm(dim=-1, keepdim=True).clamp_min(1e-8)
        similarities = (z @ projected_words.T + offsets) / norms
        maxima.append(similarities.amax(dim=1))
    maxima = torch.cat(maxima)
    scores = torch.empty(len(latents), len(caption_tokens))
    for index, tokens in enumerate(caption_tokens):
        scores[:, index] = maxima[:, tokens].mean(1)
    return scores


def recall1_interval(scores, image_groups, text_groups, seed):
    positives = image_groups[:, None].eq(text_groups[None, :])
    i_hit = positives.gather(1, scores.argmax(1)[:, None]).squeeze(1).float()
    t_hit = positives.T.gather(1, scores.argmax(0)[:, None]).squeeze(1).float()
    grouped_t_hit = torch.stack(
        [t_hit[text_groups.eq(group)].mean() for group in image_groups]
    )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        len(image_groups), (2000, len(image_groups)), generator=generator
    )
    result = {}
    for name, hits in (("i2t", i_hit), ("t2i", grouped_t_hit)):
        means = hits[indices].mean(1) * 100
        result[name] = {
            "r1": float(hits.mean() * 100),
            "bootstrap_95_interval": torch.quantile(
                means, torch.tensor([0.025, 0.975])
            ).tolist(),
        }
    result["bootstrap_unit"] = (
        "image; all five associated captions remain grouped; candidate pool fixed"
    )
    n, p, z = len(i_hit), float(i_hit.mean()), 1.959963984540054
    denominator = 1 + z * z / n
    middle = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    result["i2t"]["wilson_95_interval"] = [
        max(0.0, middle - radius) * 100,
        min(1.0, middle + radius) * 100,
    ]
    result["uncertainty_scope"] = (
        "query resampling with fixed candidates and feature means; zero-hit bootstrap "
        "intervals are degenerate, so use the I2T Wilson interval for that case"
    )
    return result


def main(args):
    from scripts.evaluate_multimodal_likelihood_benchmarks import PosteriorCache

    torch.set_num_threads(8)
    protocol = json.loads((args.output_dir / "protocol.json").read_text())
    samples = json.loads((args.output_dir / "samples.json").read_text())
    model_path = Path(protocol["model"]["path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    caption_ids = [
        tokenizer.encode(row["text"], add_special_tokens=False)
        for row in samples["flickr_texts"]
    ]
    unique_ids = sorted({token for ids in caption_ids for token in ids})
    lookup = {token: index for index, token in enumerate(unique_ids)}
    caption_tokens = [[lookup[token] for token in ids] for ids in caption_ids]
    with safe_open(str(model_path / "model.safetensors"), framework="pt") as handle:
        weight = handle.get_tensor("model.image_token_embedder.z_proj.weight").float()
        bias = handle.get_tensor("model.image_token_embedder.z_proj.bias").float()
        words = handle.get_tensor("model.embed_tokens.weight")[unique_ids].float()
    cache = PosteriorCache(
        FLICKR / "vae_posterior_mar_kl16/shards",
        expected_image_tokens=256,
        expected_latent_dim=16,
        seed=protocol["seed"],
    )
    latents = torch.stack(
        [cache.sample(row["image_id"]).float() for row in samples["flickr_images"]]
    )
    gi = torch.tensor([row["group"] for row in samples["flickr_images"]])
    gt = torch.tensor([row["group"] for row in samples["flickr_texts"]])
    emit("maxsim_started", unique_tokens=len(unique_ids))
    scores = token_maxsim(latents, words, weight, bias, caption_tokens)
    exact = normalize(latents[0] @ weight.T + bias) @ normalize(words[:16]).T
    factored = (
        latents[0] @ (normalize(words[:16]) @ weight).T + normalize(words[:16]) @ bias
    )
    factored /= (latents[0] @ weight.T + bias).norm(dim=-1, keepdim=True)
    torch.testing.assert_close(exact, factored, atol=1e-6, rtol=1e-5)
    result = {
        "schema": "unified_input_token_geometry_v1",
        "source": protocol["model"],
        "arithmetic": "FP32 stored EMA weights and cached latents; no backbone or learned readout",
        "formula": "mean_over_caption_tokens(max_over_image_patches(cos(Wz+b, E[token])))",
        "scope": "all caption subword tokens, no stopword filtering or text prompt",
        "unique_text_tokens": len(unique_ids),
        "trained_projector": retrieval_scores(scores, gi, gt),
    }
    emit("trained_maxsim_complete", **result["trained_projector"])
    generator = torch.Generator().manual_seed(424242)
    random_weight = torch.randn(weight.shape, generator=generator) * weight.std()
    random_scores = token_maxsim(
        latents, words, random_weight, torch.zeros_like(bias), caption_tokens
    )
    result["random_projector_same_trained_text"] = retrieval_scores(
        random_scores, gi, gt
    )
    del scores, random_scores
    images = load_features(args.output_dir, "flickr_images", 1000, protocol["model"])
    texts = load_features(args.output_dir, "flickr_texts", 5000, protocol["model"])
    intervals = {}
    for label, layer, pool in (
        ("input_content_mean", 0, 0),
        ("fixed_final_query", 29, 1),
        ("exploratory_layer27_query", 27, 1),
    ):
        i, t = images[:, layer, pool].float(), texts[:, layer, pool].float()
        for centered in (False, True):
            if centered:
                i, t = i - i.mean(0), t - t.mean(0)
            score = normalize(i) @ normalize(t).T
            key = label + ("_centered" if centered else "_raw")
            intervals[key] = recall1_interval(score, gi, gt, protocol["seed"])
    result["retrieval_uncertainty"] = intervals
    write_json(args.output_dir / "input_token_geometry.json", result)
    emit("input_geometry_complete", output=str(args.output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args())
