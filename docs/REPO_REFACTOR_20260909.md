# 仓库重构实施与验收

回退基线：`05e9ef7`，包含统一 loss 协议、网页更新和审计时的四处修复。正式作业继续使用原 `source-r2`，开发机使用本次改动独立验收。

## Caption 队列锁

`DirectoryLock` 保留调用接口，内部改为 POSIX `flock`。锁文件永久保留，不按 mtime 回收、不在释放时删除。持有者暂停时继续互斥，退出或被 kill 后由文件描述符关闭释放。fork 子进程关闭继承的描述符，不能释放父进程的锁。`refresh` 只校验文件身份和更新诊断时间。

这使整个受锁保护的队列写入期间保持同一个所有者；无需通过一次 token 检查猜测旧进程是否还能提交。`stale_seconds` 仅保留参数兼容，不再决定锁所有权。用户已明确数据合成只在单机进行，本模块只保证单机多进程互斥。跨主机记录锁适配已撤回，不引入相关线程锁和协议要求。

旧运行目录升级时必须先停止全部旧 worker/controller，再统一使用新代码。遇到遗留的目录锁会明确拒绝运行，不自动删除；确认旧进程停止后才能移除该目录。新协议的锁文件始终保留，禁止在运行期间删除或替换。任务 lease 的过期/心跳机制不变，只有保护 allocator/commit 的互斥锁改变。

本地 GPFS 上通过 42 项 caption 队列测试，包括暂停持锁者、SIGKILL、fork、写入失败和多 worker 竞争。开发机已完成同机测试；后续验收集中于训练、保存/恢复和生成。

## Checkpoint 与导出生命周期

保存/恢复、EMA 和 HF 导出的 24 个函数迁到 `utils/training_checkpoint.py`，训练入口保留兼容导出。`checkpoint_transaction.py` 管理目录发布，`distributed_io.py` 在进入下一阶段前向所有 rank 传播局部文件错误。

续训状态先写入 `.checkpoint-<step>.partial`，各 rank 的 Accelerate、RNG、数据游标、EMA 写入结束后检查文件清单，再写 v2 完成标记并发布为正式目录，最后才执行保留策略。同一步已有 checkpoint 时明确拒绝覆盖，原目录保持不变；不同训练轨迹应使用新输出目录。恢复 v2 时检查文件存在性和长度，缺失或截断直接报错；v1 历史格式保留原有检查。不新增大权重 hashing。

raw final 和 EMA final 共用目录发布基础实现；替换失败会恢复上一份完整导出。原先缺少 metadata 的 raw final 可在保留备份的前提下升级。

开发机已通过 123 项 infra 测试及 16-rank HCCL 故障传播：末 rank 写入失败、主 rank 写入失败均被全部 rank 观察到，之后进程组仍可完成下一次调用。后续新增 raw final 发布/回滚和模块迁移的 60 项定向回归通过。DeepSpeed 内部集合通信中的进程退出仍由后端超时与 torchrun 处理；局部文件阶段的错误传播不能替代进程组故障处理。

## 评测公共实现

ImageNet 原生理解、纯文本、多模态 likelihood 和语言先验校准的共享类/函数迁到 `utils/evaluation/`。CLI 保留参数解析、任务组织及历史导出接口，训练和其他研究脚本直接依赖公共库。`utils`/`models`/`pretrain` 对 `scripts.*` 的依赖已清零。原子文本写入另外下沉到不加载模型的 `utils/atomic_io.py`，两份相同实现合为一份。

迁移后的评分函数/类与迁移前 AST 完全相同；完整回归为 698 passed、2 skipped（CUDA FlexAttention）。三个 CLI 的参数入口保持兼容。固定数据、评分分母、MC 随机种子及缓存合同均保持原定义。

## 实验身份与任务状态

历史模型的 ID、标签、分组和启用任务统一保存在 `configs/protocols/experiment_registry.json`。新实验可在配置中声明 `experiment.identity`，训练将其与实际任务 schedule 写入不可变的 `experiment_identity.json`；launcher、训练曲线和定性模型发现共用解析器。架构、完整权重、raw/EMA、step 和评测数据协议仍由各自已有的严格来源合同验证，身份记录不代替模型合同。

没有记录任务的未知模型明确显示“训练任务未记录”，不再从普通目录名猜测所有任务都已训练。临时用途可以显式声明，历史 smoke/debug/replay 名称仍按临时实验处理。flow depth 配置中的展示身份不会改变科学参数一致性检查。

48 项定向回归通过。网页重建后仍有 14 个定性模型、3,584 条记录、1,792 张图像和 9 个完整正式指标模型；未重新计算历史分数。

## 训练与生成职责拆分

训练入口另外拆出 `training_setup.py` 的优化器/调度器构造、`training_checkpoint.py` 的恢复预检与状态恢复，以及 `training_reporting.py` 的跨 rank 运行统计。优化器参数分组和 WSD 算法原样迁移；恢复预检由主 rank 检查完成标记、文件清单和配置合同后共享结果，再加载可变状态。数据游标、NPU RNG 和 EMA 分片读取中的局部错误向所有 rank 传播，运行报告也共用原子写入和主 rank 错误传播。

生成侧将静态 K/V cache 和 image backbone query 分别移到 `modeling_selfless_cache.py` 与 `image_generation_backbone.py`。序列张量的引用保持稳定，pending content 位置按每次调用显式传入，以保留置信度探测的提交时机。CFG 分支排列、strict sigma、2D RoPE、X0/XT 条件、缓存写入与采样算法均沿用原实现。迁移前后 cache 类与 query 计算的 AST 一致。

训练相关 98 项定向回归、恢复预检 8 项、生成合同 102 项回归通过。真实权重的生成前后数值对照在开发机完整验收时进行。

## 维护入口

ImageNet 数据集模块不再反向导入 loader 构造；构造统一从 `utils.imagenet_flow_dataloaders` 或 `utils.dataset_utils` 进入。仓库测试已迁移到该入口。

V5 公共资产协议、下载函数以及早期表征诊断的协议/特征提取器迁入 `utils/research/`，原 CLI 保留兼容导出，研究脚本改为依赖公共库。保留 V2–V5 各自的方法、结果和版本入口，不改历史模型身份或分数。

`pyproject.toml` 固定少量正确性 lint 规则，`bash script/check_repo.sh` 复用已有环境运行 lint 和 CPU 回归。报告及研究依赖仍在各自 requirements 文件中，不改变正在使用的共享环境。

完整开发机验收命令：

```bash
bash script/selfless/validate_repo_refactor_ascend16.sh \
  /absolute/path/to/report all refactor-r1
```

必须使用新的 label；该命令依次运行完整回归、16-rank HCCL 文件故障传播、NPU BF16 loss/梯度对照和 depth30/depth16 的完整保存/恢复验收。每个模型验证两次完整的 2,000 张均衡 ImageNet 样本及 400 条 ClimbMix，覆盖冷/热缓存、11 项下游任务、step 2/4/5/6 checkpoint、step 2/4 raw/EMA 配对导出和 final raw/EMA 实际重载生成。EMA 生成还与提交 `1ad1670` 的原生成实现使用相同权重/噪声逐位比较。
