# Legacy selfless flow latent-mix model

This archive preserves the model code that matches the old
`selfless-flow-stage0-imagenet-full-from-qwen3base-latentmix` run.

## Provenance

- Closest committed source: `4a354f7 Add latent mix flow training path`
- W&B recorded commit for the run: `d1b738ee620e27efde786565ce2d7a6174600662`
- The W&B commit did not contain the latent-mix files, so the run appears to
  have been launched from a dirty worktree. The files in this archive are the
  later committed version that exactly matches the saved flow-head weights.

## Weight Match

The archived `mar_flow_latentmix.FlowLoss` matches both:

- `output/selfless-flow-stage0-imagenet-full-from-qwen3base-latentmix/hf_model-20000/model.safetensors`
- `output/selfless-flow-stage0-imagenet-full-from-qwen3base-latentmix/image_flow_adapter-20000.pt`

Observed key/shape match:

```text
old_4a354f7 expected 90 saved 90 missing 0 unexpected 0 mismatched 0
```

The saved flow-head keys contain:

```text
image_flow_head.net.input_mixer.*
image_flow_head.net.res_blocks.*
```

They do not match the current contextual flow head keys:

```text
image_flow_head.net.blocks.*.cross_q/cross_k/cross_v/...
```

## Architecture

The legacy head is a one-shot latent mixer followed by MLP residual blocks:

```text
x_t
  -> input_proj
  -> CausalLatentInputMixer(context_latents)
  -> ResBlock MLP-AdaLN x 8
  -> FinalLayer
```

The run config used:

```yaml
image_flow_depth: 8
image_flow_width: 1280
image_flow_mlp_ratio: 1.0
image_input_noise_strength: 0.01
```

This directory is for comparison only. It is intentionally not imported by the
current training path.
