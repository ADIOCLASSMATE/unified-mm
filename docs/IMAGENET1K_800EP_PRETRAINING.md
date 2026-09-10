# 历史 ImageNet-1K：800 epoch 预训练

ImageNet class-conditioned 研究。当前 Unified A/B 见 [实验定义](EXPERIMENTS.md)。

## 训练配方

| Item | Formal value |
| --- | ---: |
| Ascend 910B NPUs | 64 (`4 x 16`) |
| Per-rank batch | 16 |
| Gradient accumulation | 1 |
| Global batch | 1024 |
| Backbone / Special-token LR | `30e-5` |
| Flow-head / Projector LR | `4e-5` |
| ImageNet train rows | 1,281,167 |
| Samples used per epoch | 1,281,024 |
| Randomly omitted rows per epoch | 143 |
| Optimizer steps per epoch | 1,251 |
| Epochs | 800 |
| Total optimizer steps | 1,000,800 |

每 epoch 确定性重排，使用1281024行，剩余143行随 epoch 变化。EMA 为 step0 起的 FP32 rank-sharded，decay0.9999；半衰期约6931步 / 5.54 epochs。

WSD 为 warmup5 epochs / 6255步，stable595 / 744345步，decay200 / 250200步，最终 LR scale0.1。

每10 epoch 保存恢复 checkpoint，滚动保留3份；每100 epoch 保留里程碑，每20 epoch 导出完整 raw/EMA 配对；最终导出 raw、EMA 和 image-flow adapter。验证每10 epoch，标量每50步，梯度范数每epoch。训练诊断的50图/类仍在训练池中；正式 FID reference 为官方 val50000。

## 资产与运行

配置为 `configs/selfless/imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024.yaml`。需要完整 KL16 posterior、固定 Inception 和 ImageNet-val moments，路径见 [数据](DATA.md)。准备命令：

```bash
bash script/selfless/prepare_imagenet1k_cache_ascend16.sh
```

```bash
bash script/selfless/prepare_imagenet1k_fid_stats_ascend16.sh
```

配置、资产和端到端检查：

```bash
python scripts/validate_ascend_imagenet1k_pretraining.py --config_only
```

```bash
python scripts/validate_ascend_imagenet1k_pretraining.py --deep_cache_scan
```

```bash
bash script/selfless/smoke_imagenet1k_train_val_eval_ascend16.sh
```

正式启动：

```bash
bash script/selfless/pretraining_imagenet1k_class_ascend_64npu_bs1024_800ep.sh
```

恢复时给同一 launcher 设置 `RESUME_FROM=<complete-checkpoint-dir>`。周期 `hf_model-<step>-eval` / `hf_model-<step>-ema-eval` 用于离线评测，DeepSpeed checkpoint 用于恢复训练。

10-step 评测入口如下；下节历史最终分数使用100步：

```bash
torchrun --standalone --nproc_per_node=16 \
  scripts/evaluate_single_stream_fid_is.py \
  --config configs/selfless/imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024.yaml \
  --model_path_override output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep/hf_model-final-ema \
  --output_dir output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep-heun10-fid-is \
  --device npu --model_dtype bf16 \
  --samples 50000 --batch_size 4096 --vae_decode_batch_size 16 \
  --sampling_steps 10 --temperature 1.0 --cfg 3.5 \
  --cfg_schedule constant --flow_solver heun \
  --parallel_rate 1 --strategies spatial_halton \
  --inception_weights_path public/models/torch-fidelity/weights-inception-2015-12-05-6726825d.pth \
  --real_stats_path public/datasets/imagenet_full/fid_stats/inception_v3_2048_imagenet_val50000_256.pt \
  --require_formal_protocol --canonical_pairing \
  --resume_progress --resume_checkpoint_interval_batches 1
```

## 最终 EMA 结果

2026-08-19，50000样本、CFG3.5、100-step Heun、Halton、canonical pairing、项目 ImageNet-val moments：

| Metric | Result |
| --- | ---: |
| FID | `18.996944032440638` |
| Inception Score | `450.06854248046875 ± 4.259687366514454` |
| Generation throughput | `4.228181407150334 samples/s` |

结果：`output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep-fid-is/metrics.json`。

### Position-wise head

2026-08-22，相同生成协议：

| Metric | Position-wise | Baseline | Change |
| --- | ---: | ---: | ---: |
| FID | `18.2875253165069` | `18.996944032440638` | `-0.7094187159337366` (`-3.734%`) |
| Inception Score | `438.955712890625 ± 3.8873937344382083` | `450.06854248046875 ± 4.259687366514454` | `-11.112829589843727` (`-2.469%`) |
| Generation throughput | `18.023885168137856 samples/s` | `4.228181407150334 samples/s` | `4.263x` |

结果：`output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep-positionwise-head-fid-is/metrics.json`。

### Sequential sigma

2026-08-24，训练与生成均使用 sequential；其余50000样本、CFG3.5、Heun100和 reference 相同：

| Metric | Seq-sigma (`sequential`) | Baseline (`spatial_halton`) | Descriptive change |
| --- | ---: | ---: | ---: |
| FID | `8.633703493770327` | `18.996944032440638` | `-10.363240538670311` (`-54.552%`) |
| Inception Score | `247.52949981689454 ± 3.36128814432888` | `450.06854248046875 ± 4.259687366514454` | `-202.5390426635742` (`-45.002%`) |
| Generation throughput | `4.224009451221147 samples/s` | `4.228181407150334 samples/s` | `0.999x` |

此对照同时改变训练 sigma 和推理顺序。结果：`output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep-seq-sigma-fid-is/metrics.json`。

ImageNet-only Dynamic-XT 配方已停用；当前 D 从 Qwen Base 开始，定义见 [实验](EXPERIMENTS.md)。
