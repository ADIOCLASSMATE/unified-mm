# B 训练验证生成缓存：2026-09-05

FlowLoss 的时间嵌入缓存曾跨训练更新保留。DeepSpeed ZeRO-2 通过 `.data` 写回参数，缓存键使用的 `Parameter._version` 无法可靠发现更新，导致训练内生成可能读取旧时间嵌入。

## 修复

`FlowLoss.train(mode)` 清空 `_inference_time_embedding_cache`，再调用父类；`model.eval()` 也触发该路径。与权重无关的时间网格继续缓存，同次生成仍复用时间嵌入。

缓存只用于采样，不进入训练 loss 或 `state_dict`。历史训练内图片和指标按当时 runtime 解释；同一步 checkpoint 的离线生成重新计算时间嵌入。

## 证据

| 检查 | 结果 |
| --- | --- |
| CPU 最小复现 | time MLP bias 经 `.data.add_(1)` 更新后，版本号不变；旧缓存相对新权重直接计算最大差1.0 |
| tiny FlowLoss 固定噪声采样 | 修复前16/16 latent元素不同，最大差1.1358023881912231；修复后逐元素相同 |
| checkpoint40000 / 44000 时间参数 | 四个张量均变化；11个时间点的 BF16 嵌入最大差0.109375，RMS差0.017099745571613312 |
| 已保存 step22000 回放 | 487个state键完整，target PNG相同，生成图像素MSE为0.04445769265294075 |
| 已保存 step30000 loss回放 | 记录0.7814149856567383，回放0.7814070241445245，差7.9615e-6 |

前两项为CPU复现，后两项为此前保存的NPU报告。该次审查未完成运行进程内部缓存读取或清缓存后的图像重放。

相关回归80 passed / 1 skipped，覆盖模式切换、同轮复用及训练 loss 独立性。实现见 [image_flow_loss.py](../models/modeling_model/image_flow_loss.py)。

## 历史记录

- 图像回放：`output/evaluation/diagnostics/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/generation-diagnostic/step-00022000-correct-vs-recorded/report.json`
- loss回放：`output/running-validation-compat-audit-20260904/b-x0-step-30000/audit.json`

回放使用同step的raw权重，并对齐输入、初始噪声、VAE和精度。历史训练JSON为跨rank平均，PNG来自rank0。
