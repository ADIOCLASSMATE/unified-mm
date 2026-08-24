# ImageNet-1K Caption + T2I joint-training conclusion

The retained production configuration is
`configs/selfless/imagenet1k_caption_joint_10ep_ascend16_b1024.yaml`:

- initialization: `output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep/hf_model-final-ema`
- initialization weights SHA256: `76b319f8094554b4022879a887c57f72ab584eda06398e01928b03b8d1b19baf`
- initialization config SHA256: `4ad2b5308fc7c47e1807a4fa6b726b0d576cf190aa808df0d8568d01535e760c`
- synthetic-caption manifest SHA256: `c74f17cd6f8f85e74ae23606a4f0c3cc3eb1d4979413c910b5a63b2b327f6c3b`
- backbone, LM head, special-token, image-projector, and flow-head LR: `2e-5`
- `lambda_text=0.05`, `lambda_image=1.0`
- 16 Ascend NPUs, per-rank batch 16, gradient accumulation 4, global batch 1024
- 10-epoch WSD horizon and 12,020 optimizer steps

## Evidence retained from the completed sweep

The LR sweep's validation rank placed `b2e5-f2e5` first at step 12,020. Its
final validation text/image-flow losses were `2.1930861473` and
`0.5774435997`; the fixed probe had median shared-backbone
`g_image/g_text=0.02847458` and median cosine `-0.00694069`. The slightly
negative cosine is evidence of mild task conflict, so the loss weight should
not be interpreted as eliminating interference.

After fixing that LR, the generation finalists at step 4,808 were:

| lambda_text | Caption CLIP (1,000 classes) | FID (50,000) | IS (50,000) | generation mean-rank |
|---:|---:|---:|---:|---:|
| 0.05 | 0.22847543 | 21.53690287 | 44.82762070 | 1.6667 |
| 0.10 | 0.23105135 | 24.32990464 | 38.60260658 | 2.0000 |
| 0.20 | 0.23343236 | 27.84991786 | 32.70475826 | 2.3333 |

The original strict finalizer returned no winner because all three candidates
regressed against the class-conditioned initialization (`FID=18.99694403`,
`IS=450.06854248`). For the intended class-to-caption/prompt domain transition,
that exclusion was explicitly waived. Applying the already defined equal
mean-rank rule over Caption CLIP (higher), FID (lower), and IS (higher) then
selects `lambda_text=0.05`. This is a practical default under that domain-shift
decision, not a claim that T2I retained the class-conditioned baseline.

The deleted source reports can be identified by these SHA256 digests:

- final selection: `d142c7650cf2d135fa679199c05834794b76a4701449084fd459a4ece51a2f4a`
- LR validation ranking: `e9c1f65f026096ac21deeddcffd33ea1699ac659ae8ebf4ca6a1f7aa70b02cf7`
- lambda validation ranking: `5afbf9dec0ec9e390eb74c955a3347c7ea6f818f43cdea942e9bd433f332cd66`

The Qwen text-backbone plus image-adapter run was a short control, not part of
the final selection: it had no canonical Caption CLIP or 50,000-sample FID/IS
evaluation. Its config, launcher, probes, and checkpoint are intentionally not
retained.

## Stable entry points

- train: `script/selfless/pretraining_imagenet1k_caption_joint_ascend16.sh`
- validate assets/config: `scripts/validate_ascend_imagenet1k_caption_joint.py`
- evaluate: `script/selfless/evaluate_imagenet1k_caption_joint_ascend16.sh`

The default config uses `log_grad_norm_every=1200`, aligned with
`log_every=50`, so pre-clip gradient norms are persisted.
