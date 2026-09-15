# Unified-MM

研究随机序图像建模与文本、图文联合训练。模型以 Qwen3 为 backbone，使用 MAR-KL16 latent 和 flow head；生产环境为 Ascend 910B。

正式 A/B 比较 content attention 的对角线可见性，C–F 为 B 的架构消融；另有任务 only、flow-head 深度扩展和 S2/SigLIP 对照。

| 内容 | 入口 |
| --- | --- |
| 统一多模态预测的观察与双流设计动机 | [设计动机](docs/DUAL_STREAM_DESIGN_MOTIVATION.md) |
| 实验定义、配置与训练预算 | [实验](docs/EXPERIMENTS.md) · [训练](docs/TRAINING.md) |
| 数据集与合成文本 | [旧消融数据](docs/DATA.md) · [当前 B512 合成流程](docs/DATA_SYNTHESIS.md) |
| 指标、样例与训练曲线 | [评测总览](output/evaluation/index.html) |
| 评测命令与输出位置 | [评测](docs/EVALUATION_STRUCTURE.md) |
| 平台、资源与共享路径 | [Inspire](INSPIRE.md) |
| 表征研究与历史结果 | [文档索引](docs/README.md) |

## 环境与运行

根目录 `.venv` 由 `uv` 管理。环境配套为 Python 3.11、PyTorch 2.6.0+cpu、torch-npu 2.6.0.post5、torchvision 0.21.0+cpu、NumPy 1.26.4。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
uv sync --frozen
```

正式 A/B 训练：

```bash
bash script/selfless/pretraining_unified_ablation_a_0p6b_formal_ascend64.sh
bash script/selfless/pretraining_unified_ablation_b_0p6b_formal_ascend64.sh
```

已有环境中的代码检查与报告更新：

```bash
bash script/check_repo.sh
python3 scripts/build_evaluation_report.py
```

训练入口为 `pretrain/train_selfless_flow.py`，模型位于 `models/modeling_model/`，数据与训练公共实现位于 `utils/`。任务提交和 NPU 验证使用 [Inspire 操作约定](INSPIRE.md)。
