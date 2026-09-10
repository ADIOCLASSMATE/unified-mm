# 仓库重构：2026-09-09

完成 checkpoint 事务、caption 单机互斥、评测公共库、实验身份和训练/生成职责拆分。问题来源见 [审查记录](REPO_AUDIT_20260909.md)。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `utils/training_checkpoint.py` | 保存/恢复预检、分阶段写入与错误传播、完成标记、发布及轮换 |
| `utils/training_setup.py` | 优化器分组与 WSD 构造 |
| `utils/training_reporting.py` | 跨 rank 运行统计 |
| `utils/evaluation/` | 纯文本、原生理解、多模态 likelihood、先验校准共享实现 |
| `utils/atomic_io.py` | 不加载模型的原子文本写入 |
| `utils/experiment_registry.py` | 配置/注册表中的实验身份与任务信息 |
| `models/modeling_model/modeling_selfless_cache.py` | 静态 backbone K/V cache |
| `models/modeling_model/image_generation_backbone.py` | 图像 backbone query |
| `utils/imagenet_flow_dataloaders.py` | ImageNet loader 构造 |
| `utils/research/` | 研究资产、协议和特征提取公共实现 |

`utils` / `models` / `pretrain` 对 `scripts.*` 的依赖清零。历史 CLI 保留参数及兼容导出。

## 保存与队列

checkpoint 使用独立暂存目录；所有 rank 写完模型/优化器、RNG、数据游标、EMA 后写 completed v2 文件清单，发布成功再轮换。文件阶段错误向各 rank 传播；raw/EMA 配对发布支持回滚。恢复先检查完成标记和文件清单。

caption `DirectoryLock` 使用 POSIX flock，适用单机多进程。父进程 fork 后，子进程关闭继承 FD 时不释放父锁；`stale_seconds` 仅保留调用兼容。锁文件持续保留。旧目录锁迁移在相关 worker 停止后进行。

实验注册表为 `configs/protocols/experiment_registry.json`；新 run 将 `experiment.identity` 与实际任务 schedule 写入 `experiment_identity.json`。未知任务显示“训练任务未记录”。

## 验证

完整开发机命令，最后一个参数为 depth，可取16、30或both：

```bash
bash script/selfless/validate_repo_refactor_ascend16.sh   /absolute/path/to/report all <new-label> 30
```

2026-09-09，版本 `7f90800`，16×910B，镜像v1.4。depth30 完整结果如下；depth16 只完成两步保存和配对导出。

| 检查 | 结果 |
| --- | --- |
| 开发机完整回归 | 706 passed、2 skipped；两项跳过均需要 CUDA FlexAttention |
| 16-rank HCCL 文件故障传播 | 末 rank、主 rank 写入故障被所有 rank 观察到，之后进程组继续可用 |
| NPU BF16 共享 content 对照 | loss 逐位一致；最大梯度相对 L2 误差 0.00973779，低于预定 0.03 容差 |
| depth30 完整验证 | step 2 / 4 均完成 11 项下游任务；耗时 409.19 / 400.52 秒 |
| 统一 loss 与缓存 | 当前权重、train_no_grad；冷 / 热耗时 8.80 / 5.10 秒；第二轮复用准备缓存 |
| 样本一致性 | loss 和轻量评测共用相同 2,000 张图像、1,000 类各 2 张；纯文本 400 条固定记录 |
| Checkpoint 与续训 | step 2 / 4 / 5 完成校验，恢复 5→6，step 6 保存通过；NPU RNG、数据游标、EMA 成功恢复 |
| 周期导出 | step 2 / 4 的 raw / EMA BF16 完整配对导出通过 |
| Final 更新与重载 | raw / EMA 的 source_global_step 均为 6；分别核对 927 / 926 个已保存权重键的取样值，均实际生成完整图像 |
| 生成重构前后对照 | 同一 EMA 权重、同一初始噪声、CFG 3.5、10-step Heun、Halton；latent 和生成顺序逐位一致 |
| 网页 | 开发机重建成功；14 个定性模型、3,584 条记录、1,792 张图像、9 个完整正式指标模型；浏览器 10 个路由无脚本或资源加载错误 |


以上为当日验证记录。smoke 权重、图片、日志和临时目录已按要求清理，本页保留结果；正式产物保留。日常检查入口为 `bash script/check_repo.sh`。

## 提交索引

`05e9ef7` 实施前基线；`4e0b4c8` caption 单机锁；`08c1418` checkpoint；`2539696` 评测库；`1ad1670` 实验身份；`14694d8` 训练职责；`463bbe5` 生成 cache/query；`7f90800` 研究公共实现。跨机器 caption 锁适配由 `a970173` 撤回。
