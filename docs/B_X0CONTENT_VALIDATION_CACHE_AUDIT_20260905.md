# B 训练验证与独立生成不一致排查

日期：2026-09-05 UTC。

模型现统一称为正式 B；文件名保留历史检索用途，见[实验定义](EXPERIMENTS.md)。

训练目录：`output/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1`。

## 结论与证据边界

发现并在 CPU 最小模型上复现了一个实际的生成 runtime bug：FlowLoss 的时间嵌入缓存跨训练更新保留，而 DeepSpeed ZeRO-2 的参数更新不能由该缓存使用的 `Parameter._version` 可靠检测。因此，训练内验证可能使用“当前权重 + 旧时间嵌入”，独立加载 checkpoint 则重新计算时间嵌入。

这能解释“验证 loss 可以回放，但验证生成图不同”的现象。尚未读取正在运行的训练进程内缓存，也未在该进程清缓存后完成同噪声图像对照；不能把最小模型的复现称为该 Job 原图的逐像素复现，或据此排除其他生成差异。

## 现场核对

- 平台 Job 为 `umm-b-x0content-0p6b-100b-64-s42-r1`。通过 `inspire job command` 查询，其启动命令在当前共享仓库执行，并检查提交 `f1b5d14c9c26167b120ff06729fdc7cc4df75013`；不是另一个 runtime 目录。
- 本次修复前，`git diff f1b5d14 -- models pretrain utils` 为空。没有发现训练核心代码与工作区发生版本漂移的证据。
- 配置为 raw-model 验证（`ema_validate: false`）、seed 424242、CFG 3.5、10-step Heun、`spatial_halton`，每 2000 step 生成两张图。
- 已有 step-22000 回放报告加载全部 487 个 state keys，missing/unexpected 均为空；target PNG 完全一致，生成图与记录图的像素 MSE 为 0.04445769265294075。报告位于 `output/evaluation/diagnostics/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/generation-diagnostic/step-00022000-correct-vs-recorded/report.json`。这是此前保存的证据，不是本次重新执行的 NPU 回放。
- 已有 step-30000 loss 审计中，记录的加权 loss 为 0.7814149856567383，回放为 0.7814070241445245，差 7.9615e-6；image-flow loss 差 4.1127e-6。见 `output/running-validation-compat-audit-20260904/b-x0-step-30000/audit.json`。

## 根因机制

1. `models/modeling_model/image_flow_loss.py` 的 `_inference_time_embeddings()` 缓存 `net.time_embed` 在 ODE 时间网格上的输出。缓存键包含参数 `_version`，不包含训练步数。
2. 当前环境的 DeepSpeed `runtime/zero/stage_1_and_2.py` 使用 `bit16_partitions[partition_id].data.copy_(fp32_partition.data)` 写回参数，并在 `_update_model_bit16_weights()` 中执行 `p.data = q.data`。不能依靠 Parameter 的版本计数发现这些底层更新。
3. 原来的 `FlowLoss` 只在 `_apply()`（例如设备或 dtype 转换）时清缓存，没有在 `train()` / `eval()` 时清理。
4. `pretrain/train_selfless_flow.py` 的 `validate()` 在验证前调用 `model.eval()`，结束后调用 `model.train()`；原实现中的这些切换不会使时间嵌入缓存失效。
5. 在采样配置和 batch shape 不变、没有额外失效操作时，首次验证后缓存可以一直命中。该 Job 首次验证发生在 step 2000；“现场仍持有 step-2000 时间嵌入”是据执行路径推断，不是进程内存实测。

独立重新加载权重的新模型没有这份历史缓存，所以仅对齐 checkpoint、seed、CFG 和采样步数仍不足以复现旧进程的错误状态。

## 本次实测

- CPU 最小复现：先计算时间嵌入，再对时间 MLP bias 做 `.data.add_(1.0)`，参数 `_version` 更新前后均为 `[3, 2, 3, 2]`。原实现仍返回旧缓存，与当前权重直接计算结果的最大误差为 1.0。
- 完整 tiny FlowLoss 采样回归：固定权重和初始噪声，对照“已验证、继续更新的模型”和“重新加载相同 state_dict 的模型”。修复前两个测试都失败，16/16 个 latent 元素不一致，最大绝对误差 1.1358023881912231；修复后结果逐元素相等。
- 实际 checkpoint-40000 与 checkpoint-44000 的四个时间嵌入参数 tensor 全部发生变化。在 CPU 上按 BF16、11 个时间点计算的嵌入最大绝对差为 0.109375，RMS 差为 0.017099745571613312。这不是未取得的 step-2000 缓存，也不是 NPU 图像比较。

## 修复与验证

在 `FlowLoss.train(mode)` 中清空 `_inference_time_embedding_cache`，然后交给父类切换模式。`model.eval()` 也会递归触发它。与权重无关的时间网格缓存继续保留，同一次生成中各 token 仍复用时间嵌入。

回归覆盖训练/验证切换、重复进入 eval、同轮采样复用，以及训练 loss 不读取该推理缓存。相关测试命令：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_dual_stream_flow_head.py \
  tests/test_selfless_flow_behavior.py \
  tests/test_backbone_kv_cache.py \
  tests/test_dynamic_xt_contract.py \
  tests/test_validation_flow_memory.py \
  tests/test_generation_sampling_defaults.py \
  tests/test_training_forward_invariance.py
```

结果：80 passed，1 skipped；涉及文件的 `git diff --check` 通过。

## 影响与后续操作

- 此缓存只用于 flow 采样，不用于训练/验证 loss 的 forward，也不进入 `state_dict`。本 bug 本身不会写坏保存的权重，不构成从零重训的理由。
- 已生成的训练内 T2I 图片及其生成指标不能当作对应 checkpoint 的干净离线生成结果。这里的生成指标与 validation loss 是不同的产物。
- 修复共享盘源码不会替换已运行 Python 进程中的类定义。本次没有停止、重启或热修改训练 Job。下一次受控地从完整 checkpoint 恢复时，需使用修复后的代码；不要用新配置启动一个 step-zero 训练覆盖现有目录。
- 原平台命令带有对旧提交 `f1b5d14` 的 clean-diff 检查，不能不加修改地重跑它来启用补丁；恢复命令需同时对齐修复后的代码版本和 resume checkpoint。
- 离线对照应使用同 step 的 raw 权重，不要混用 EMA；还需对齐输入、初始噪声、VAE 和精度。训练验证 JSON 的生成指标是跨 rank 平均，PNG 仅来自 rank 0，不能直接与单 rank 数字比较。
- 本次尝试连接固定开发 Notebook 做 NPU 复核，但连接未建立，没有执行新的 NPU 推理。已停止本次启动的 `dev-wjx-ascend` 并确认 `STOPPED`。
