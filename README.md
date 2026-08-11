# Unified-MM

Unified-MM 当前以华为昇腾 Ascend NPU 为唯一生产后端，维护最终选定的
Selfless-Flow 图像生成架构：

- Qwen two-stream backbone 与 dynamic dual-stream contextual flow head；
- backbone 和 flow head 都固定使用 row/column pure 2D RoPE；
- 不使用 additive image position；
- attention output gate 保留单一接口，但默认关闭；
- ImageNet latent dataloader 只支持 `class` 与 `caption` 两种条件模式。

完整的架构选择依据压缩在 [消融结论](docs/ABLATION_CONCLUSIONS.md)，最终训练
超参数见 [ImageNet-100 超参数结论](docs/IMAGENET100_HYPERPARAMETER_CONCLUSION.md)，
正式预训练合同见 [ImageNet-1K 800-epoch 配置](docs/IMAGENET1K_800EP_PRETRAINING.md)。
实验矩阵、旧配置和兼容入口不属于运行时仓库。

## NPU 环境

项目只使用根目录 `.venv`。`uv sync` 会安装已经过当前 CANN 环境验证的
Python 3.11、PyTorch CPU host runtime 和 `torch-npu`，不会安装 CUDA、NCCL、
Triton 或 `flash-attn`：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
uv sync --frozen
```

之后直接使用 `uv run`，无需再激活或维护 `.venv-npu`：

```bash
uv run --frozen python -c \
  'import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available(), torch.npu.device_count())'
```

当前锁定的核心配套是 Python 3.11、PyTorch 2.6.0+cpu、torch-npu
2.6.0.post5、torchvision 0.21.0+cpu 和 NumPy 1.26.4。CANN 属于系统运行时，
必须在调用 `uv run` 前加载 `set_env.sh`。

## 训练

正式 ImageNet-1K 预训练使用 64×Ascend 910B、global batch 1024、800 epochs、
HCCL 和 DeepSpeed ZeRO-2。唯一训练入口是：

```bash
bash script/selfless/pretraining_imagenet1k_class_ascend_64npu_bs1024_800ep.sh
```

配置固定 Backbone/Special-token LR `30e-5`、Flow-head/Projector LR `4e-5`，
EMA decay 为 `0.9999`。训练前必须通过完整数据缓存、模型哈希和官方 ImageNet-val
FID real-stat 前检；准备方法及恢复训练命令见正式预训练合同。

Caption manifest 必须完整覆盖 latent manifest；loader 不静默回退到 class，
也不截断超出配置上下文长度的 caption。训练集可使用确定性 segment packing，
validation 保持一条样本一行。

## 评测

正式 ImageNet-1K 协议是 50K samples、BF16、CFG 3.5、100-step Heun、
`spatial_halton`。
评测核心入口是 `scripts/evaluate_single_stream_fid_is.py`；global batch 会先按
rank 切分，再进入 dataset collation。

正式 FID 必须传入与目标数据分布匹配的 real-stat cache。

## 代码入口

- `models/modeling_model/modeling_selfless_flow.py`：two-stream backbone。
- `models/modeling_model/image_flow_loss.py`：pure-2D dynamic flow head。
- `utils/dataset_imagenet_flow_cache.py`：class/caption dataset 与 dataloader。
- `utils/multimodal_segment_packing.py`：caption 变长样本 packing。
- `scripts/evaluate_single_stream_fid_is.py`：FID/IS evaluator。

## 验证

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
PYTHONPATH="${PWD}:${PYTHONPATH:-}" TORCH_COMPILE_DISABLE=1 \
  uv run --frozen python -m pytest -q tests
uvx ruff check models utils pretrain scripts tests
```

平台路径和资源约束见 [INSPIRE.md](INSPIRE.md)。
