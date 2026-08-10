# Selfless-Flow Ascend NPU 迁移与验收报告

## 基本信息

| 字段 | 内容 |
|---|---|
| 模型 | Selfless-Flow（Qwen3-0.6B + dynamic dual-stream flow head） |
| 类型 | ImageNet-100 类条件生成训练与 FID/IS 评测 |
| 仓库 | unified-mm（本地工作区） |
| 日期 | 2026-08-10 |
| 状态 | 80 epoch 正式训练与 10k/100-step 最终评测均已完成并严格验收 |

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
- 评测 flow content cache 改为条件/无条件统一批处理并一次预分配 256 token；
  attention 始终使用固定长度和 mask，避免 Ascend 为 256 种动态长度累计工作区。
- VAE 与 Inception 特征提取均以每卡 16 的微批流式执行，生成 batch 可安全提高到
  每卡 256，而不改变 FID/IS 数学协议。
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
151 passed, 2 skipped, 4 warnings in 13.77s
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

### 16 卡 80 epoch 正式训练

正式训练固定为每卡 batch 16、梯度累积 2、global batch 512、DataLoader workers 0，
完成 80 个精确 epoch 和 17,920 个 optimizer step。最终训练 checkpoint、EMA 与
16 个 rank 的 optimizer/RNG 分片齐全，严格验收状态为 `ok`。

```text
samples_per_epoch=114688
optimizer_steps_per_epoch=224
epochs=80
global_step=17920
finite_loss_microbatches_checked=35840
final_loss=0.6527
train_samples_per_second=400.8510
final_validation_image_flow=0.6733
```

### 16 卡端到端评测 gate

最终 EMA、每卡 256 样本、全局 batch 4096、CFG 3.5、Heun、真实 KL16 VAE
与 Inception 的 1-step 容量/吞吐 gate：

```text
count=4096
generation_samples_per_second=24.578289963356266
peak_device_allocated_mib=25131.1640625
peak_device_reserved_mib=30668.0
```

该 gate 使用正式训练后的 EMA 和 1 个采样步，只验证容量与基础设施，不作为模型
质量指标。外部 HBM 观测稳定在约 32.7/65.5 GiB；正式协议仍固定为 10,000 样本、
100 采样步。

### 16 卡 100-step 最终评测

最终评测使用 FP32 EMA checkpoint、模型 BF16、flow integrator/VAE/指标累加 FP32，
固定每卡 batch 256、global batch 4096、CFG 3.5、100-step Heun、canonical noise
pairing 与冻结的 10,000 张原图 real-stat。10000/10000 个样本完成，生成 latent
有限率为 1.0，严格验收状态为 `ok`。

```text
FID=25.016273944757813
Inception_Score=69.01506576538085 +/- 0.8990247253689196
generation_samples_per_second=4.076489107832094
generation_wall_seconds=2453.09130859375
peak_device_allocated_mib=25521.693359375
peak_device_reserved_mib=30674.0
```

完整指标位于
`output/selfless-flow-im100-class-ascend16-b16ga2-b4e5-f1e4-fid-is/metrics.json`；
最终训练与评测 acceptance JSON 均为 `status: ok`。

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
6. 大 evaluation batch 下，旧 flow cache 对每个样本分别 `torch.cat` 增长，再按
   token 重新 stack；即使预分配 K/V，只要向 attention 暴露 1..256 的动态长度，
   Ascend 仍会累计多套编译/工作区缓存并把 HBM 推到 65 GiB。现统一为批量、固定
   256 长度的 K/V 与 mask，1-step 全批验收期间 HBM 不再随 token 数增长。

## 最终结论

- 正式训练固定 DataLoader workers 0、global batch 512，已完成 80 epoch 严格验收。
- 正式评测固定每卡 batch 256、global batch 4096，不再继续上探 batch。
- 10k/100-step Heun FID/IS 已完成严格验收；训练、评测与进程清理均已闭环。
