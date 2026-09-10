# 仓库审查：2026-09-09

审查覆盖274个 Python 文件及训练、恢复、评测入口。发现的五项问题已在后续重构中处理；实现与16-NPU验证见 [重构记录](REPO_REFACTOR_20260909.md)。

| 问题 | 复现行为 | 处理 |
| --- | --- | --- |
| 整模型加载不完整 | 缺少 bias 的 state dict 加载成功，残留初始化参数 | 加载前校验键、类型和形状，共享参数别名单独处理 |
| checkpoint 提前轮换 | 新保存失败时，旧恢复点已删除 | 暂存、完成标记、发布成功后再轮换 |
| 深度评测配置 | depth16 被默认 depth8 拒绝 | 从 checkpoint 读取合法 head 容量 |
| 临时实验发现 | smoke/debug/replay 被纳入正式定性集合 | 共用实验身份与临时用途过滤 |
| caption 锁所有权 | 过期旧持有者可删除新持有者的锁 | 单机多进程改用 POSIX flock |

对应入口：[权重加载](../utils/image_generation_io.py)、[checkpoint](../utils/training_checkpoint.py)、[评测来源](../utils/evaluation_model_source.py)、[实验身份](../utils/experiment_registry.py)、[caption 锁](../caption_farm/io.py)。

当日修复前 CPU 回归666 passed / 2 skipped；第一轮修复后686 passed / 2 skipped，耗时75.06秒。日志位于 `output/repo-audit/20260909/pytest-after-fixes.log`。后续完整开发机回归706 passed / 2 skipped，详见实施记录。

训练/评测共享函数已移入 `utils/evaluation/`，研究公共实现移入 `utils/research/`；训练保存、恢复、统计和生成 cache/query 已按职责拆分。
