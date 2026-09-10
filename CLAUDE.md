# 项目约定

## 信息入口

- [实验定义](docs/EXPERIMENTS.md)：A/B、C–F、only、深度扩展、S2/SigLIP。
- [训练](docs/TRAINING.md)、[数据](docs/DATA.md)、[评测](docs/EVALUATION_STRUCTURE.md)。
- [Inspire](INSPIRE.md)：项目归属、共享路径、永久开发机和提交流程。
- [文档索引](docs/README.md)：研究协议与历史结果。

## 实验与实现

- 展示名称使用正式 A/B；持久 ID、run 名和 checkpoint 路径保持原值。名称与分组来自 `configs/protocols/experiment_registry.json`。
- B 使用共享参数的 X0/content、XT/query 两流，query 按 `sigma_kv < sigma_q` 读取，content 按 `sigma_kv <= sigma_q` 读取。A 的 content 也用严格小于。
- 图像为 256 个 16 维 KL16 latent token；backbone 与 flow head 使用 row/column 2D RoPE，实际 backbone 基数为 10,000。
- 文本和 caption 按从左到右训练，图像按配置的 sigma 顺序训练。各架构的标签与生成路径见实验定义。
- 使用现有 ClimbMix、ImageNet caption/T2I 数据集；only 对齐 B 对应任务的数据和累计曝光。
- S2/SigLIP 按 Show-o2 范式检验本项目的双路径必要性；文档列明当前实现与原训练阶段的差异。

## 运行与验证

- 使用根目录 `.venv` 和现有 CANN 环境；共享环境变更安排在训练空闲期。
- 加入仓库到 `PYTHONPATH` 时保留原有 CANN 路径。
- Ascend 为生产后端；设备 smoke 使用固定 `dev-wjx-ascend`。
- Unified 运行时 hashing 和 W&B 关闭，来源、配置、进度及指标写入本地文件。
- 保存、续训、EMA 和生成改动按 [训练文档](docs/TRAINING.md)验证；CPU 检查入口为 `bash script/check_repo.sh`。
- 评测产物写入 `output/evaluation/`，训练状态写入 `output/<run>/`。

## 文档

文档写设计、实际配置、结果和复现入口。共享设置集中维护，其他页面链接引用。历史结果注明模型、数据和采样条件；保留关键条件，省去过程叙述和重复声明。
