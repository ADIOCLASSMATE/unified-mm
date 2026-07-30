# Unified-MM

Unified-MM 当前只维护最终选定的 Selfless-Flow 图像生成架构：

- Qwen two-stream backbone 与 dynamic dual-stream contextual flow head；
- backbone 和 flow head 都固定使用 row/column pure 2D RoPE；
- 不使用 additive image position；
- attention output gate 保留单一接口，但默认关闭；
- ImageNet latent dataloader 只支持 `class` 与 `caption` 两种条件模式。

完整的历史选择依据压缩在
[消融结论](docs/ABLATION_CONCLUSIONS.md)；实验矩阵、proposal 和兼容接口不再
属于运行时仓库。

## 训练

Class-conditioned：

```bash
bash script/selfless/pretraining_imagenet_flow_full_from_qwen3base.sh
```

Caption-conditioned 使用
`configs/selfless/imagenet_flow_caption_from_qwen3base.yaml`：

```bash
CONFIG=configs/selfless/imagenet_flow_caption_from_qwen3base.yaml \
  bash script/selfless/pretraining_imagenet_flow_full_from_qwen3base.sh
```

Caption manifest 必须完整覆盖 latent manifest；loader 不静默回退到 class，
也不截断超出配置上下文长度的 caption。训练集可使用确定性 segment packing，
validation 保持一条样本一行。

## 评测

```bash
bash script/selfless/evaluate_imagenet_flow.sh
```

默认正式协议是 8×H100、10K samples、BF16、CFG 3.5、100-step Heun、
`spatial_halton`。评测脚本中的 batch 是 shard 前的全局 batch；默认
`4096 = 8 × 512`。可以通过 `NUM_GPUS`、`BATCH_SIZE_PER_GPU` 或
`BATCH_SIZE` 显式覆盖。每个 global batch 会先按 rank 切分，再进入 dataset
collation；各 rank 不再重复构造完整的 4096-row CPU batch。

正式 FID 需要传入与目标数据分布匹配的 real-stat cache：

```bash
REAL_STATS_PATH=/path/to/inception_stats.pt \
  bash script/selfless/evaluate_imagenet_flow.sh
```

## 代码入口

- `models/modeling_model/modeling_selfless_flow.py`：two-stream backbone。
- `models/modeling_model/image_flow_loss.py`：pure-2D dynamic flow head。
- `utils/dataset_imagenet_flow_cache.py`：class/caption dataset 与 dataloader。
- `utils/multimodal_segment_packing.py`：caption 变长样本 packing。
- `scripts/evaluate_single_stream_fid_is.py`：FID/IS evaluator。

## 验证

```bash
PYTHONPATH=. TORCH_COMPILE_DISABLE=1 \
  .venv/bin/python -m pytest -q tests
uvx ruff check models utils pretrain scripts tests
```

平台路径和资源约束见 [INSPIRE.md](INSPIRE.md)。
