# 仓库重构实施与验收

回退基线：`05e9ef7`，包含统一 loss 协议、网页更新和审计时的四处修复。正式作业继续使用原 `source-r2`，开发机使用本次改动独立验收。

## Caption 队列锁

`DirectoryLock` 保留调用接口，内部改为 POSIX `flock`。锁文件永久保留，不按 mtime 回收、不在释放时删除。持有者暂停时继续互斥，退出或被 kill 后由文件描述符关闭释放。fork 子进程关闭继承的描述符，不能释放父进程的锁。`refresh` 只校验文件身份和更新诊断时间。

这使整个受锁保护的队列写入期间保持同一个所有者；无需通过一次 token 检查猜测旧进程是否还能提交。`stale_seconds` 仅保留参数兼容，不再决定锁所有权。底层共享文件系统必须支持跨进程/跨主机的 POSIX 文件锁。

旧运行目录升级时必须先停止全部旧 worker/controller，再统一使用新代码。遇到遗留的目录锁会明确拒绝运行，不自动删除；确认旧进程停止后才能移除该目录。新协议的锁文件始终保留，禁止在运行期间删除或替换。任务 lease 的过期/心跳机制不变，只有保护 allocator/commit 的互斥锁改变。

本地 GPFS 上通过 42 项 caption 队列测试，包括暂停持锁者、SIGKILL、fork、写入失败和多 worker 竞争。开发机验收记录将在本文件补充。

## Checkpoint 与导出生命周期

保存/恢复、EMA 和 HF 导出的 24 个函数迁到 `utils/training_checkpoint.py`，训练入口保留兼容导出。`checkpoint_transaction.py` 管理目录发布，`distributed_io.py` 在进入下一阶段前向所有 rank 传播局部文件错误。

续训状态先写入 `.checkpoint-<step>.partial`，各 rank 的 Accelerate、RNG、数据游标、EMA 写入结束后检查文件清单，再写 v2 完成标记并发布为正式目录，最后才执行保留策略。同一步已有 checkpoint 时明确拒绝覆盖，原目录保持不变；不同训练轨迹应使用新输出目录。恢复 v2 时检查文件存在性和长度，缺失或截断直接报错；v1 历史格式保留原有检查。不新增大权重 hashing。

raw final 和 EMA final 共用目录发布基础实现；替换失败会恢复上一份完整导出。原先缺少 metadata 的 raw final 可在保留备份的前提下升级。

开发机已通过 123 项 infra 测试及 16-rank HCCL 故障传播：末 rank 写入失败、主 rank 写入失败均被全部 rank 观察到，之后进程组仍可完成下一次调用。后续新增 raw final 发布/回滚和模块迁移的 60 项定向回归通过。DeepSpeed 内部集合通信中的进程退出仍由后端超时与 torchrun 处理；局部文件阶段的错误传播不能替代进程组故障处理。

## 评测公共实现

ImageNet 原生理解、纯文本、多模态 likelihood 和语言先验校准的共享类/函数迁到 `utils/evaluation/`。CLI 保留参数解析、任务组织及历史导出接口，训练和其他研究脚本直接依赖公共库。`utils`/`models`/`pretrain` 对 `scripts.*` 的依赖已清零。原子文本写入另外下沉到不加载模型的 `utils/atomic_io.py`，两份相同实现合为一份。

迁移后的评分函数/类与迁移前 AST 完全相同；完整回归为 698 passed、2 skipped（CUDA FlexAttention）。三个 CLI 的参数入口保持兼容。固定数据、评分分母、MC 随机种子及缓存合同均保持原定义。
