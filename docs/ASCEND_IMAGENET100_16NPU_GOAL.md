# Ascend ImageNet-100 16-NPU 训练目标

## 目标

在 Ascend 910B 环境中完成 Selfless-Flow 的数据准备、生产训练入口适配和
ImageNet-100 class-conditioned 训练。正式运行必须使用 16 张 NPU，保持已经在
H100 上调定的模型、数据、global batch、学习率和 80 epoch 训练语义，并产出
可恢复 checkpoint、final model 与 EMA model。

正式训练直接运行在 Ascend 开发环境中，不提交 Inspire Job。数据准备和 smoke
可以先使用当前可见卡，但不得将 8 卡训练冒充正式的 16 卡实验。

## 当前基线

文档创建时的代码基线为：

- 分支：`npu`
- NPU 实现 commit：`824db060520f20fb808b2266b5e009ea8bd46772`
- PyTorch：`2.6.0+cpu`
- torch-npu：`2.6.0.post5`
- Accelerate：`1.14.0`
- DeepSpeed：`0.19.4`
- 硬件：Ascend 910B2C，单卡 64 GiB HBM
- 当前容器只暴露 8 张 NPU；正式训练仍受 16 卡资源门槛阻塞

该分支已经包含并验证：

- `torch_npu.npu_fusion_attention`；
- fully-masked query row 严格输出 0；
- native GQA，不再使用 `repeat_interleave` 扩展 KV heads；
- image hidden-state `index_select`/`torch.gather` 路径；
- flow attention 的 NPU BSND 布局，移除显式 layout 往返；
- NPU memory bookkeeping、HCCL-safe INT64/FP32 gather 与 NPU RNG 保存恢复；
- 算子、跨设备 parity、profiler、单卡和多卡 benchmark 脚本。

不得为了训练准备回退这些优化，也不得在缺少新的端到端证据时引入 QKV/MLP
参数融合或自定义 AscendC kernel。

## 路径与已有资产

仓库：

```text
/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/code/unified-mm
```

仓库中的 `public` 是项目公共目录的软链接：

```text
public -> /inspire/sj-ssd3/project/high-dimensionaldata/public
```

已验证的原始 ImageNet：

```text
public/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/train
public/dataset/imagenet/v1/LOC_synset_mapping.txt
```

训练集必须包含 1,000 类、1,281,167 张图像。

已验证的 backbone：

```text
public/models/Qwen--Qwen3-0.6B-Base
```

`model.safetensors` SHA256：

```text
cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba
```

必须使用 Base checkpoint，不得误用 `Qwen3-0.6B` instruction model，也不得在
已有权重完整时重复下载。

## 需要补齐的资产

### MAR KL16 VAE 代码

与既有实验一致的代码来源：

```text
Repository: https://github.com/ADIOCLASSMATE/mar.git
Commit: c6d53f7fa6427634b5850ebed771b7c2d19ea21f
Destination: public/code/mar
models/vae.py SHA256:
95e9d47d017817cd86858d78587786c931a9ba9596fe3eb6d6dce4136580112b
```

训练和预处理只需要 KL16 VAE 代码，不需要 MAR-B、MAR-L 或 MAR-H 权重。

### MAR KL16 checkpoint

目标位置：

```text
public/vae/mar-kl16/kl16.ckpt
```

参考下载地址：

```text
https://www.dropbox.com/scl/fi/hhmuvaiacrarfg28qxhwz/kl16.ckpt?rlkey=l44xipsezc8atcffdp4q7mwmh&dl=1
```

完整性合同：

```text
Size: 265900046 bytes
SHA256: 34ce001bcfffb7af67ec8af1e683a30d7bd45760855ddc7deedc1330f2cfd38f
```

若 Ascend 环境不能访问公网，应通过平台批准的文件路径补齐资产；不得建立代理、
VPN 或额外隧道绕过网络限制。

## ImageNet-100 membership 与 split

正式实验固定使用 100 类、每类 1,250 张，共 125,000 张图像。选择过程必须可
复现：

1. 按 synset 目录名排序，再按类内文件名排序；`img_id` 从 1 开始。
2. eligible class 至少包含 1,250 张图像。
3. 使用 stratified class selection 和 seed 42 选出 100 类。
4. 使用同一个 `torch.Generator(seed=43)` 依次打乱每类成员并取前 1,250 张。
5. 使用 seed 42 做 class-stratified split，每类固定 100 张 validation。
6. 最终必须得到 115,000 train 和 10,000 validation 样本。

历史选择算法可从以下文件的旧版本只读恢复：

```text
git show 6f7a16a:scripts/build_imagenet_flow_cache_subset.py
git show 6f7a16a:scripts/build_imagenet100_split_manifest.py
git show 6f7a16a:script/ablation/build_imagenet_100c_balanced_cache.sh
```

只复用 membership/split 算法，不恢复旧的 frozen-latent Dataset。

Canonical 参考：

| Artifact | Rows | SHA256 |
| --- | ---: | --- |
| Full ImageNet manifest | 1,281,167 | `9d165263e8cf4ba6d537d084a8cc3b87af2eaf5ef9a5b59e1360a6228c840759` |
| ImageNet-100 manifest | 125,000 | `6c6fc84e6ec9cb8b92421659d44abbb24d1cc34558213d9fb9db7e4ec44f1c3a` |
| ImageNet-100 split | 125,000 | `02e5c67c058f95bcca46c82f3c1fc81086f61dcec62ce25049843f09d930a5ba` |

目标位置：

```text
public/datasets/imagenet_ablation_100c_balanced/manifest.jsonl
public/datasets/imagenet_ablation_100c_balanced/split_seed42_val100.jsonl
```

Canonical split 依赖“按类 shuffle 后的 selected image-id order”。不得直接在按
`img_id` 排序后的 posterior cache 上重新执行 split RNG，否则 validation
membership 会发生静默变化。

## KL16 posterior cache 合同

需要缓存的是 posterior statistics，不是固定的一次性 latent，也不是 prior。
训练 Dataset 必须能按 image/epoch 的稳定 seed 从 mean/std 重新采样 latent。

目标文件：

```text
public/datasets/imagenet_ablation_100c_balanced/
  vae_posterior_mar_kl16/posterior_stats_100c_1250pc_fp16.pt
```

Tensor 与 metadata 合同：

| Field | Contract |
| --- | --- |
| `posterior_stats` | FP16 `[125000, 256, 32]` |
| `posterior_stats[..., :16]` | scaled posterior mean |
| `posterior_stats[..., 16:]` | scaled posterior std |
| `img_ids` | INT64 `[125000]`，严格递增且唯一 |
| scaling factor | `0.2325` |
| format | `imagenet_kl16_scaled_posterior_v1` |
| stats layout | `scaled_mean_then_scaled_std` |

应先生成 100 类 membership，再只编码选中的 125,000 张图像；不得为这个目标先
编码全部 1,281,167 张图像。编码需要支持 NPU、分 shard、原子写入、断点复用，
并在 merge 后检查 shape、dtype、finite、non-negative std、唯一递增 IDs 与
manifest membership。

Canonical H100 cache 的参考 SHA256 为：

```text
d30e916aefe7944815b2975f4fa23d87f7d2462a749748a36673d0eddf8e8cc7
```

如果公共目录中出现该精确 cache，应直接复用。若在 NPU 上重新编码，二进制 SHA
可能不同；必须记录新 SHA，并对固定图像的 posterior mean/std 和 VAE decode
做数值与有限性检查，不能为了匹配哈希修改 tensor。

## 必须完成的 NPU 训练入口

以 `configs/selfless/imagenet100_class_base_80ep.yaml` 为算法基线，新增或整理
NPU 专用配置和 launcher。至少解决：

- Qwen model path 与 Ascend 公共目录实际布局不一致；
- synset mapping 路径仍是旧的 H100 布局；
- validation VAE module root 硬编码 H100 路径；
- `script/offline_env.sh` 只识别 `.venv`，而 Ascend 使用 `.venv-npu`；
- class training launcher 硬编码 8 GPU、`CUDA_VISIBLE_DEVICES` 和 CUDA allocator；
- VAE encoder 中 CUDA-only device、backend、dtype 与 pin-memory 分支；
- 缺少 16-NPU Accelerate/DeepSpeed Zero2 配置；
- runtime memory metric 仍使用 CUDA-only 名称。

优先新增 NPU 专用文件或做显式 backend dispatch，不破坏 H100 路径。不得运行
`uv sync`，不得替换现有 PyTorch/torch-npu，也不得安装 CUDA `flash-attn`。

## 正式训练合同

训练语义必须保持：

| Item | Value |
| --- | ---: |
| conditioning | `class` |
| backbone | pretrained Qwen3-0.6B-Base |
| `from_scratch` | `false` |
| dtype | BF16 forward，FP32 gradient accumulation |
| sequence length | 320 |
| flow multiplier | 4 |
| global batch | 512 |
| optimizer steps/epoch | 224 |
| epochs | 80 |
| max optimizer steps | 17,920 |
| backbone/special-token LR | `4e-5` |
| flow/projector LR | `1e-4` |
| warmup steps | 1,000 |
| WSD decay steps | 4,480 |
| save interval | 4,480 steps |
| validation interval | 2,240 steps |
| EMA | enabled |
| activation checkpointing | disabled unless required by measured HBM |

16 卡的候选 batch 方案：

```text
microbatch 32 x 16 ranks x GA 1 = global batch 512
microbatch 16 x 16 ranks x GA 2 = global batch 512
```

先用真实模型、真实 cache 和生产 DeepSpeed Zero2 路径测量。若 B32 没有安全的
HBM 余量，则使用已知更稳妥的 B16/GA2。不得改变 global batch 或
`max_train_steps`，也不得因卡数翻倍而改变 80 epoch 的优化步数。

生产多卡必须使用 HCCL，并验证 `HCCL_INTRA_ROCE_ENABLE=1`。不得将
`tests/bench_npu_8x.py` 视为生产分布式验证，因为该 benchmark 使用手工
all-reduce 而非正式 Accelerator/DeepSpeed 路径。

## 分阶段门槛

### 1. 资产与 CPU 预检

- Qwen config、tokenizer、weights 可完全离线加载；
- ImageNet count、manifest、split、posterior cache 合同全部通过；
- 配置插值后，world size、microbatch、GA、global batch、epoch 与 step 数一致。

### 2. 单 NPU

- 使用真实 pretrained model 和真实 cache 完成 forward/backward/update；
- loss、gradient norm、parameter update 均 finite；
- VAE validation decode 至少生成一张有效图；
- native GQA、fully-masked row、gather、BSND 与 runtime regression 全部通过。

### 3. 当前可见多卡

- HCCL all-reduce smoke 通过；
- 正式 `pretrain/train_selfless_flow.py` + Accelerator + DeepSpeed Zero2 至少运行
  3 个 optimizer steps；
- 保存完整 checkpoint，并从 `checkpoint_complete.json` 对应 checkpoint 恢复
  后再运行至少 1 步；
- rank、loss、LR、throughput、HBM 和 checkpoint metadata 正常。

### 4. 正式 16 卡门槛

启动正式运行前必须同时满足：

```text
torch.npu.is_available() == True
torch.npu.device_count() == 16
distributed world_size == 16
```

若环境仍只暴露 8 卡，应完成前述准备和 smoke，然后报告资源阻塞；不得启动降格
的正式实验，也不得自行提交 Job。

## 正式运行与恢复

正式训练在独立 `tmux` 或等价持久会话中启动，使用唯一 run name 和新的输出
目录。启动前记录：

- Git commit 与 diff；
- PyTorch、torch-npu、CANN、driver、Accelerate、DeepSpeed 版本；
- `npu-smi` 与 16 个逻辑 rank；
- model、VAE、manifest、split、posterior cache SHA256；
- 完整配置快照、启动命令和环境变量。

先观察至少 3 个 optimizer steps，确认 16 ranks、loss、throughput 和 HBM 稳定，
再让会话持续运行。不得启动重复的正式 run。中断后只允许从包含
`checkpoint_complete.json` 的完整 checkpoint 恢复。

## 完成判据

只有全部满足以下条件才可宣布目标完成：

- 真实 `world_size=16`；
- `global_step=17920`，对应 canonical 80 epoch；
- 全程无 NaN/Inf、无 rank 丢失、无静默数据跳过；
- final checkpoint 完整；
- final Hugging Face model 完整；
- final EMA Hugging Face model 完整；
- 记录训练 wall time、samples/s、每 rank 峰值 HBM 和 final loss；
- 验证所有产物可以离线重新加载；
- 给出 final、EMA 和可恢复 checkpoint 的绝对路径。

本目标不包含正式 10K FID/IS；训练完成后的正式生成评测需要用户另行确认。

全过程记录到：

```text
/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/
  npu-parity-audit/ASCEND_TRAINING_LOG.md
```

日志必须包括资产哈希、实际修改、所有 smoke、失败与恢复、正式启动参数、训练
进度摘要和最终产物。仅仅启动进程、看到部分 step，或完成 8 卡运行，都不构成
本目标完成。
