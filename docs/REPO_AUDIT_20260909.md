# 仓库正确性与重构审计 · 2026-09-09

下文保留审计时的状态；后续修复和开发机验收见 [实施记录](REPO_REFACTOR_20260909.md)。

本轮发现并复现了 **5 个正确性问题**，其中 **4 个已在工作区修复**，caption 队列的锁所有权问题仍需处理。最有价值的重构是 checkpoint 保存、队列并发控制、训练与离线评测的公共实现，以及模型/实验身份管理。

审计覆盖了修改前的 274 个 Python 文件（约 10.1 万行）、shell 入口、配置和文档，运行了完整 CPU 测试，并重点人工检查训练、数据、保存/恢复、权重加载、评测和报告生成路径。这是静态检查、测试与重点代码审查，未逐一执行所有历史实验或证明所有并发路径正确。

本轮新增的训练保存修改只在工作区。正在运行的 depth16/depth30 正式作业继续使用此前经过开发机验收的 `source-r2`；本轮未修改该快照、重启作业或提交新作业。后续采用保存逻辑修改前，仍需开发机多卡保存/恢复验收。

## 已修复的问题

| 编号 | 影响 | 修复前的可复现行为 | 当前处理 |
| --- | --- | --- | --- |
| R1 · 高 | 旧整模型加载入口可能产生混合权重，影响使用该入口的评测 | 给 `load_model_state` 一个缺少 bias 的 state dict，函数成功返回；bias 仍来自初始化 | 加载前校验全部键、张量类型和形状；仅允许真实共享参数的别名省略；不完整输入在修改模型前报错 |
| R2 · 高 | 保存失败可能损失最近的可恢复状态 | 保留上限为 1 时，保存 checkpoint-20 失败，但 checkpoint-10 已被删除 | 普通备份轮换推迟到新 checkpoint 的 Accelerate 状态、数据游标、元数据、EMA 和完成标记全部写入之后 |
| R3 · 中 | 新增容量消融无法使用默认评测入口 | 用真实 depth16 导出配置套用默认评测配置，报 `checkpoint=16, evaluation=8` | flow head 宽度/深度取自 checkpoint，保留合法性检查；数据维度、目标函数和特殊架构约束仍严格检查 |
| R4 · 中 | 定性模型集合可能混入临时实验 | 模型发现会纳入带 smoke/debug/replay 名称的最终导出 | 定性评测和训练曲线共用临时实验过滤函数，在解析权重之前过滤 |

修复位置：

- R1：[utils/image_generation_io.py](../utils/image_generation_io.py)，测试 [test_image_generation_io.py](../tests/test_image_generation_io.py)。有意支持部分模块覆盖的 adapter 接口保留原语义；不能将它与整模型加载混为一谈。
- R2：[utils/utils.py](../utils/utils.py) 和 [pretrain/train_selfless_flow.py](../pretrain/train_selfless_flow.py)，测试 [test_checkpoint_retention.py](../tests/test_checkpoint_retention.py)。覆盖 Accelerate、数据、元数据、EMA、完成标记五个失败点，以及成功后的轮换。
- R3：[utils/evaluation_model_source.py](../utils/evaluation_model_source.py)，测试 [test_evaluation_model_source.py](../tests/test_evaluation_model_source.py)。覆盖 depth16/depth30、非法容量值，并保留原有模型合同测试。
- R4：[utils/evaluation_paths.py](../utils/evaluation_paths.py)、[scripts/generate_unified_qualitative.py](../scripts/generate_unified_qualitative.py) 和 [scripts/evaluation_report_training.py](../scripts/evaluation_report_training.py)，新增定性模型发现回归测试。

这些复现证明相关入口存在缺陷，不能据此认定历史训练或已发布指标都受到了影响。R1 涉及旧整模型加载入口；正式 EMA 评测另有严格的权重来源校验。

## 已复现、尚未修复：caption 队列锁所有权

**R5 · 高优先级。** [caption_farm/io.py](../caption_farm/io.py) 中，`DirectoryLock` 用目录修改时间回收过期锁；`refresh()` 和 `release()` 只检查本地 `acquired` 布尔值，不核对当前目录是否仍属于这个持有者。

可复现顺序：A 获得锁 → 锁过期 → B 回收并获得同一路径 → A 恢复运行。此时 A 的 `refresh()` 会刷新 B 的目录，`release()` 会删除 B 的锁。本轮在临时目录中复现了两个行为，没有操作正式 caption 队列。

```bash
PYTHONPATH="$PWD" .venv/bin/python \
  output/repo-audit/20260909/reproduce_caption_lock_ownership.py
# 当前输出：{'replacement_lock_survived': False}
```

这会破坏队列互斥保证；尚无证据说明现有 caption 数据已经因此损坏。不能仅凭过期时间假定旧进程永久死亡。

修复需要覆盖锁和受保护的写入：明确所有者身份，失去所有权后拒绝续租/释放及提交，通过单写者或能拒绝旧持有者写入的版本协议保证提交顺序。只给 `owner.json` 加一个 token，无法完整解决“检查通过后锁被接管”的竞争窗口。验收应包含进程暂停/恢复、持锁者退出、锁过期接管、批量提交中断和共享文件系统上的多进程竞争。

## 重构顺序与验收边界

### 1. 先收拢 checkpoint 保存事务和队列提交协议

训练保存已分散在 `utils.utils`、训练入口、sharded EMA、混合数据 loader。R2 的修复解决了旧备份提前删除，但当前保存流程仍有两个需要处理的边界：

- `_begin_checkpoint_write()` 会删除同名目的目录。同一步重试若命中已有完整 checkpoint，仍存在先删后写窗口，应使用独立暂存目录和完成后的发布流程。
- 部分 rank 局部/主 rank 文件写入之后直接进入 barrier。写入异常传播仍依赖进程组和启动器处理，应该统一阶段结果传播和超时诊断。

建议用一个明确的保存接口管理：暂存 → 所有 rank 写入 → 校验模型/优化器/RNG/数据/EMA 一致性 → 发布完成标记 → 更新可恢复入口 → 清理旧状态。ordinary checkpoint、周期 raw/EMA 配对导出和 final 导出共享发布/失败处理的基础组件，各自保留格式规则。

验收：故障注入后旧完整状态仍可恢复；同一步重试不破坏已发布状态；损坏/缺失 rank 状态拒绝恢复；开发机多卡完成保存、恢复下一步、raw/EMA 实际加载与生成。caption 队列的 R5 同属本优先级，但应独立修改和验证。

### 2. 将评测公共实现从 CLI 脚本下沉为库

AST 检查发现训练侧 `utils` 有 **11 处导入 `scripts.*`**，主要在 `training_downstream_validation.py`，涉及候选文本打分、ImageNet 原生评测、纯文本 benchmark、语言先验以及 JSON 写入。

训练和离线评测复用同一评分实现有助于保证协议一致。目前问题是公共实现放在命令行入口里，训练依赖随 CLI 扩张，测试边界也难稳定。

建议先迁移数据选择、评分、聚合、结果结构和原子写入到专用库模块；训练快评和离线 CLI 都调用它。沿用现有 `training_unified_loss_validation`、`training_validation` 等拆分，不再新增一份等价算法。

验收：固定输入、权重、随机种子下，任务计数/分母/聚合与数值不变；2,000 张均衡 ImageNet 清单和纯文本验证协议不变；训练路径不再导入命令行入口。函数迁移和数值优化应分开进行。

### 3. 统一模型来源、实验身份和结果选择

R1、R3、R4 反映了相同的维护问题：多个入口分别解释“哪个模型、哪些权重、是否正式实验”。临时名称过滤这次已统一，但部分模型标签、任务训练状态和发现规则仍由脚本及目录名分别推断。

建议逐步建立共同的实验记录：实验 ID、任务启用状态、配置/权重来源、架构合同、数据协议、采样参数、step、raw/EMA、临时/正式用途。明确区分完整权重加载与有意的模块覆盖；以现有 `evaluation_model_source` 为基础收拢公共校验。报告、定性生成、指标汇总和 launcher 消费相同记录。

验收：未知模型无需向多处硬编码表补项；临时/不完整/协议不兼容结果不能进入正式比较；保留旧记录的明确迁移规则，避免静默改写历史实验含义。

### 4. 按职责继续拆训练入口，再拆生成入口

修改前 `pretrain/train_selfless_flow.py` 共 4,678 行，`main()` 约 1,832 行；`evaluate_single_stream_fid_is.py` 的 `main()` 约 1,285 行；`generate_image()` 约 1,219 行。行数只是定位线索，更直接的问题是初始化、数据状态、训练循环、保存、验证和结果写入相互交织。

优先让训练 `main` 只组织初始化与循环，把数据进度、保存和验证生命周期交给明确组件。上述第 1、2 项完成后再拆，避免把当前耦合机械搬到更多文件。

生成代码最后拆：按序列准备、顺序规划、backbone/cache、flow 求解、CFG 和输出整理划边界。这里涉及 strict sigma、2D RoPE、X0/XT 条件及缓存合同，需要固定种子数值对照、各架构合同测试和 NPU 实测；不宜夹带算法改动。

### 5. 小范围维护整理

- `dataset_imagenet_flow_cache` 与 `imagenet_flow_dataloaders` 存在双向导入，当前靠局部导入运行。将兼容入口/构造职责移到单向依赖位置即可，不需要重写 loader。
- `scripts.prepare_geometry_v5_assets` 被 30 个生产模块导入、`scripts.probe_unified_representations` 被 18 个导入。可按第 2 项迁移被共享的公共函数，同时保留研究版本入口。
- 将报告和开发依赖明确分组，统一 README 的运行方式与报告 requirements。当前工作区运行环境不应在正式作业期间重装。
- 固定仓库自己的 lint 规则与回归入口。广泛 Ruff 扫描的 637 条结果主要是格式、导入和现代化建议，不能等同于 637 个逻辑错误；避免将批量格式修改混入数值/infra 修复。

V2–V5 研究版本回答的问题不同，V5 仍复用部分旧实现。保留版本和协议依据，提取公共实现；没有依据删除旧目录。

## 检查记录

- 修复前完整 CPU 测试：666 passed，2 skipped。
- 四处修复的定向回归：110 passed。
- 修复后完整 CPU 测试：**686 passed，2 skipped，75.06 秒**。两项跳过均要求 CUDA FlexAttention，见 [完整日志](../output/repo-audit/20260909/pytest-after-fixes.log)。
- 真实导出配置集成检查：depth16/depth30 均成功接入共用评测配置；正式定性模型发现返回 14 个模型，无 smoke/debug/replay。此检查读取配置和权重元信息，没有加载模型张量。
- 评测网页重新构建成功：14 个定性模型、3,584 条记录、1,792 张图像通过构建校验；9 个正式指标模型均完整。此项验证报告生成，没有重新计算评测分数。
- Python 语法检查、shell `bash -n` 检查通过；Ruff 未定义名称、重复定义及明确语义错误规则通过；`git diff --check` 通过。
- 本轮未运行 NPU smoke、多节点训练、官方生成 benchmark 或真实 caption 队列故障注入。CPU 测试不能替代这些验收。

证据保存在 [output/repo-audit/20260909/](../output/repo-audit/20260909/)：文件清单、依赖图、五个复现结果、完整回归与集成检查。建议实施顺序为：R5 和保存失败边界 → 评测公共库 → 实验身份 → 训练入口 → 生成入口。
