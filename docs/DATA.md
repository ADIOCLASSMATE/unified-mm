# 数据

| 任务 | 训练来源 | 规模与组织 |
| --- | --- | --- |
| 纯文本 | `public/ClimbMix/*.jsonl` | 在线分词，2,048-token segment packing |
| I2T | ImageNet train + 六条合成 caption | 1,281,167 图；每图 Qwen / MiniMax 各三条，按 epoch 轮换 |
| T2I | ImageNet train + 十二条 prompt | 同一图像库，按 epoch 轮换 prompt |
| 图像验证 | ImageNet val | 50,000 图，每类 50 图 |

I2T 资产还包含 original caption，当前 `caption_include_original=false`。T2I 使用现有十二类 prompt；每条都配该图的 VAE posterior。训练与 only 对照使用同一套资产。

## 资产路径

| 资产 | 路径 |
| --- | --- |
| 图像身份 | `public/datasets/imagenet_full/manifest.jsonl`、`manifest_val.jsonl` |
| 训练 posterior | `public/datasets/imagenet_full/vae_posterior_mar_kl16/posterior_stats_imagenet1k_train_fp16.pt` |
| 验证 posterior | 同目录 `posterior_stats_imagenet1k_val_fp16.pt` |
| 七 caption 资产 | `public/datasets/imagenet1k_synthetic_v1/captions/imagenet1k_train_7captions.jsonl` |
| 验证描述 | 同目录 `imagenet1k_val_visual_descriptions.jsonl` |
| 训练／验证文本索引 | `public/datasets/imagenet1k_synthetic_v1/indexed/{train,val}/manifest.json` |
| ClimbMix 固定探针 | `public/datasets/climbmix_validation_v1/manifest.json` |
| FID / IS Inception | `public/models/torch-fidelity/weights-inception-2015-12-05-6726825d.pth` |
| ImageNet-val FID moments | `public/datasets/imagenet_full/fid_stats/inception_v3_2048_imagenet_val50000_256.pt` |

MAR-KL16 表示为 `[N,256,16]`，对应 256×256 图像。Posterior 按样本身份与 epoch 采样，文本与图像身份通过 manifest 对齐。Loader 支持 `class` 和 `caption`，完整保留 caption，训练可 packing，验证按独立样本构造。

图文前缀：

```text
T2I: Generate an image matching this description:
I2T: Describe this image in one detailed caption:
```

## 合成来源

I2T 合成来源为 Qwen3.6-35B-A3B-FP8 与 MiniMax-M3；T2I 使用 GPT-5.6-Luna 的十二类 prompt。来源、模板、样例及全量计数见 [数据来源页](../output/evaluation/data-provenance/README.md)，源配置为 [evaluation_data_sources.json](../configs/protocols/evaluation_data_sources.json)。

更新资产统计与网页：

```bash
python3 scripts/audit_evaluation_data_provenance.py
python3 scripts/build_evaluation_report.py
```

## 历史 caption 队列

入口为 `scripts/imagenet_qwen_caption_farm.py`，配置为 `configs/caption_farm/imagenet1k_qwen36_35b_a3b_fp8.json`。该配置保留历史HDD路径；发布目标由 `output.published_jsonl` 指定。按实际队列目录查询状态：

```bash
PYTHONPATH=. .venv/bin/python scripts/imagenet_qwen_caption_farm.py queue status --run-dir <run-dir>
```

队列采用单机多进程 `flock`、任务 lease 和原子提交；每图三个 Qwen slot。发布条件为 3,843,501 slot 完成，输出 1,281,167 行。实现见 [维护说明](REPO_REFACTOR_20260909.md)。
