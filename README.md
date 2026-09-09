# Unified-MM

Unified-MM 以华为昇腾 Ascend NPU 为生产后端，研究 ClimbMix 文本与 ImageNet
图文联合训练。当前主对照是 B X0-content，并保留 A/C/D/E/F 与单源控制组。
基础架构约定：

- Qwen two-stream backbone 与 dynamic dual-stream contextual flow head；
- backbone 和 flow head 都固定使用 row/column pure 2D RoPE；
- 不使用 additive image position；
- attention output gate 保留单一接口，但默认关闭；
- ImageNet latent dataloader 只支持 `class` 与 `caption` 两种条件模式。

消融模型使用各自 checkpoint 声明的 attention、flow condition 和生成顺序。
当前协议、研究结果与早期 ImageNet 实验分别列在 [文档索引](docs/README.md)。

所有模型的评测统一放在 [`output/evaluation/`](output/evaluation/)，
打开 [评测总览](output/evaluation/index.html) 查看指标与 T2I/I2T/文本逐样本对照。
训练权重、优化器和续训状态保留在各自训练目录。

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

Unified 0.6B 的正式消融使用 64×Ascend 910B、HCCL 和 DeepSpeed ZeRO-2。
各训练臂由 [100B 消融协议](configs/protocols/unified_ablation_100b_ascend64.yaml)
固定。B 与 D 的入口例如：

```bash
bash script/selfless/pretraining_unified_ablation_b_0p6b_formal_ascend64.sh
bash script/selfless/pretraining_unified_ablation_d_on_b_0p6b_formal_ascend64.sh
```

Unified 的运行时 hashing 与 W&B 保持关闭；启动前校验数据、模型来源、评测缓存和
实验合同。训练中快评见 [训练中下游验证](docs/TRAINING_DOWNSTREAM_VALIDATION.md)。
早期 class-only 800-epoch 配方见 [历史训练合同](docs/IMAGENET1K_800EP_PRETRAINING.md)。

Caption manifest 必须完整覆盖 latent manifest；loader 不静默回退到 class，
也不截断超出配置上下文长度的 caption。训练集可使用确定性 segment packing，
validation 保持一条样本一行。

### Checkpoint 与评测模型的默认约定

一般训练实验统一保留最近 **3 个普通续训 checkpoint**，并额外永久保留每
**100 个图像 epoch** 的完整 checkpoint（含优化器、调度器、随机状态及 EMA
状态）。里程碑独立触发保存，不占普通 checkpoint 的 3 个滚动名额。

每 **20 个图像 epoch** 永久导出一对完整 BF16 Hugging Face 评测模型：
`hf_model-<step>-eval/`（raw）和 `hf_model-<step>-ema-eval/`（EMA），并写入
`hf_model-<step>-eval-pair.json` 完成标记。这些导出不参与续训 checkpoint 轮换。

| 配置族 | 每图像 epoch 步数 | 100 epoch 里程碑 | 20 epoch raw + EMA 导出 |
| --- | ---: | ---: | ---: |
| unified、ImageNet class | 1,251 | 125,100 | 25,020 |
| ImageNet T2I、caption-joint | 1,202 | 120,200 | 24,040 |

纯文本单源实验使用 unified 的参考步数对齐。配置字段
`checkpoints_total_limit`、`checkpoint_milestone_every`、`save_ema_eval_every`
和 `save_model_with_ema_eval` 分别控制上述策略；步数从训练起点累计，调整数据量或
全局 batch 后应重新换算。普通续训保存频率由 `save_every` 单独控制。
新实验默认遵守此约定；短程 smoke、LR sweep 等可通过显式覆盖调整保存策略。

## 评测

当前 ImageNet-1K 生成评测配方是 50K samples、BF16、CFG 3.5、10-step
Heun、`spatial_halton`；除 50K 指标覆盖外，采样参数可按后续实验调整，并应随结果
完整记录。
评测核心入口是 `scripts/evaluate_single_stream_fid_is.py`；global batch 会先按
rank 切分，再进入 dataset collation。

正式 FID 必须传入与目标数据分布匹配的 real-stat cache。

完整项目评测使用 `script/selfless/evaluate_unified_native_full_checkpoint_ascend16.sh`，
输出目录放在 `output/evaluation/` 下。已有结果的汇总页在 CPU 上更新：

```bash
python3 scripts/build_evaluation_report.py
```

目录约定与结果身份见 [评测结构](docs/EVALUATION_STRUCTURE.md)。

最终 checkpoint 另外全量报告官方 GenEval、DPG-Bench 和 MJHQ-30K clean-fid；
生成在 Ascend 上执行，三套官方评分器使用各自独立 CUDA 环境。固定版本、命令和
分辨率可比边界见 [官方图像生成评测](docs/OFFICIAL_T2I_BENCHMARKS.md)。

## 代码入口

- `models/modeling_model/modeling_selfless_flow.py`：two-stream backbone。
- `models/modeling_model/image_flow_loss.py`：pure-2D dynamic flow head。
- `utils/dataset_imagenet_flow_cache.py`：class/caption dataset 与 dataloader。
- `utils/multimodal_segment_packing.py`：caption 变长样本 packing。
- `scripts/evaluate_single_stream_fid_is.py`：FID/IS evaluator。
- `scripts/evaluate_official_t2i_benchmarks.py`：GenEval、DPG-Bench、MJHQ 官方评分适配器。

## 验证

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash script/check_repo.sh
```

此入口使用现有根目录 `.venv` 和 PATH 中的 Ruff，按 `pyproject.toml` 的
正确性规则检查后运行完整 CPU 回归，不会同步或修改训练环境。需要多卡的保存、
恢复和生成验收使用固定开发机，命令见 [重构验收记录](docs/REPO_REFACTOR_20260909.md)。

报告端依赖单独列在 `scripts/evaluation_report_requirements.txt`，包括网页和独立
loss 图所需的 PyYAML、NumPy、Matplotlib；研究环境的额外依赖保留在各自
`scripts/geometry_v5_*_requirements.txt`。正式作业运行期间不要重装共享环境。

平台路径和资源约束见 [INSPIRE.md](INSPIRE.md)。
