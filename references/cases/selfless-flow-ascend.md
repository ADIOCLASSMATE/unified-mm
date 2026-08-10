# Selfless-Flow Ascend NPU 迁移与验收报告

## 基本信息

| 字段 | 内容 |
|---|---|
| 模型 | Selfless-Flow（Qwen3-0.6B + dynamic dual-stream flow head） |
| 类型 | ImageNet-100 类条件生成训练与 FID/IS 评测 |
| 仓库 | unified-mm（本地工作区） |
| 日期 | 2026-08-10 |
| 状态 | NPU 训练/validation/评测基础设施已通过；80 epoch 正式结果待运行完成 |

## 环境信息

- NPU：16 × Ascend 910B2C，64 GiB HBM/卡
- CANN/npu-smi：25.0.rc1.1
- Python：3.11.15
- PyTorch：2.6.0+cpu（torch_npu PrivateUse1 后端）
- torch_npu：2.6.0.post5
- 分布式后端：HCCL

## 迁移过程

### 快速尝试

没有使用 `transfer_to_npu`。训练和评测都包含显式的设备分支、融合注意力、
DeepSpeed ZeRO-2、HCCL 指标归约与确定性评测协议；全局 monkey patch 无法安全
表达这些语义，因此采用按 tensor device 条件分派的最小改动。

### 关键修复

- validation 单流缓存中的 `uint8.masked_fill` 改为保持 uint8 的 `torch.where`。
- validation 与 FID/IS 的 HCCL 归约从不支持的 float64 改为 fp32；最终 FID
  仍在 CPU float64 中计算。
- 评测器增加 `npu[:index]`、HCCL、NPU 同步与显存统计支持，不再把 NPU
  请求静默降级到 CPU。
- flow head 与 backbone attention 改为按实际 tensor device 分派：NPU 使用
  `npu_fusion_attention`，CPU/CUDA 参考路径使用 SDPA/BlockMask。
- 新增 16-NPU 正式评测入口、10,000 张固定原图 real-stat 构建器与严格结果验收。
- 训练 DataLoader 增加精确 epoch 采样预算：每 epoch 无放回抽取 114,688 张，
  保证 16 rank × batch 16 × GA 2 的 224 个 optimizer step 全部为 global batch 512，
  不再在 epoch 尾部同步半个梯度累积 step。
- 移除会在 pytest 收集阶段直接调用外部模型 API 的旧实验文件 `test_api.py`。

### 依赖修复

NPU 虚拟环境补装了 `torch-fidelity 0.4.0`、`torchmetrics 1.9.0` 与
`pytest 9.1.1`；`pyproject.toml` 显式声明 torchmetrics。Inception 权重来自
torch-fidelity v0.2.0 release，SHA-256 为
`6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`。

## 验证结果

### 测试套件

```text
143 passed, 2 skipped, 4 warnings in 14.15s
PASS native_gqa prepared_mask uint8_token_type_fill single_stream_cache flow_bsnd prepared_flow_mask span_gather
PASS npu_fid_is_metrics world=16 backend=hccl dtype=torch.float32
```

### 16 卡训练 validation gate

精确生产模型、数据、KL16 VAE 和单流 100-step validation 已通过：

```text
[Validation] Step 1 | ImageFlow: 1.9670
```

### 16 卡精确 epoch gate

真实模型、真实训练集与 HCCL 完成一个无保存的 224-step epoch：

```text
samples_per_epoch=114688
prepared_microbatches_per_rank=448
optimizer_steps_per_epoch=224
global_step=224
finite_loss_microbatches_checked=448
train_samples_per_second=448.7751
```

### 16 卡端到端评测 gate

每卡 16 样本、全局 batch 256、CFG 3.5、Heun、真实 KL16 VAE 与 Inception：

```text
count=256
generation_samples_per_second=2.8499290542084283
peak_device_allocated_mib=3902.8095703125
peak_device_reserved_mib=8216.0
```

该 gate 使用随机初始化 flow head 和 1 个采样步，只验证容量与基础设施，不作为
模型质量指标。正式协议固定为训练后 EMA、10,000 样本、100 采样步。

### Real-stat cache

固定 validation 集合为 100 类 × 每类 100 张，共 10,000 张；2048 维统计量全有限。

```text
cache_sha256=917e4d6af770d91118e592d9aa71ed082025b52ea3236fd2a3c858607e87e232
accumulation_dtype=torch.float32
```

## 问题与解决

1. Ascend 不支持 uint8 `masked_fill`：改用 `torch.where`，保持 token type dtype。
2. HCCL 不支持 float64 all-reduce：设备端累加/归约使用 fp32，CPU 最终矩阵计算
   使用 float64，并把精度协议写入 metrics。
3. 旧迁移将 NPU 算子硬编码到 CPU 路径：按 `tensor.device.type` 分派，恢复参考测试。
4. 配置中的 real-stat 路径无法显式关闭：支持 `--real_stats_path none`，仅供 smoke；
   正式入口仍强制冻结 cache。
5. 原 DataLoader 的 115,000 张训练 split 在分片后产生每 rank 449 个微批；GA=2
   使每个数据 epoch 的尾部更新只有 global batch 256。日志在 step
   450/900/1350/... 出现精确半 loss，最终验收的 finite-microbatch 计数也会失败。
   现按固定种子每 epoch 无放回抽取 114,688 张，并在 accelerator prepare 后强制
   断言 448 microbatches/rank、224 steps/epoch、80 epochs 与 17,920 steps 完全一致。

## 待优化项

- 完成 17,920 steps（80 epoch）正式训练、最终 EMA 导出与 10k FID/IS。
- 正式评测完成后记录真实 100-step 吞吐、FID 与 Inception Score。
- 如果评测时间成为主要瓶颈，可在不改变样本顺序和随机数合同的前提下继续上探
  evaluation batch；当前每卡 16 已有安全实测依据。
