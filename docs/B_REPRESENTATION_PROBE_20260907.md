# B 表征诊断 V1：输入与 backbone

2026-09-07，冻结 B step 95415 final EMA，在 16 张 910B 上以 BF16 采集。最终 backbone query 已有跨模态可读方向；输入平均与局部匹配的直接检索接近随机。后续扩展见 [V2](B_SEMANTIC_EMERGENCE_V2_RESULTS_20260907.md)。

## 协议

Flickr30K 使用 1000 test 图 / 5000 caption，全候选集检索；额外映射按图分 800 fit / 200 test。ImageNet 使用 1000 类 × 10 图，每类 6 fit / 4 test；8 个类名模板分 6 fit / 2 test。

两侧独立输入，共同前缀为 `Describe this image in one detailed caption:`，后缀为 `\nThe main subject is`，随后放置固定文本 query。图像侧不含类名或配对描述。采集输入、28 个 block、最终 RMSNorm，读出为正文 Content mean 与可访问完整输入的共有文本 query。

分类使用逐样本 L2 与闭式 ridge，相对正则强度固定 0.1。完整检索的中心化使用候选池模态均值；200 图映射仅用 fit 均值。跨模态分类的均值修正版本使用目标模态 fit 均值。

## 输入层

Flickr30K 全候选集 R@1（%）：

| 输入层评分 | 图 → 文 | 文 → 图 |
| --- | ---: | ---: |
| 随机期望 | 0.10 | 0.10 |
| 输入 embedding 平均后原始余弦 | 0.00 | 0.04 |
| 输入 embedding 平均后按模态中心化 | 0.40 | 0.14 |
| 文本 token → 最相似图像 patch，随后平均 | 0.00 | 0.08 |
| 随机 image projector 的同一局部匹配控制 | 0.10 | 0.08 |

局部匹配使用全部 4768 个实际文本 subword：`mean_text_token max_image_patch cos(Wz+b, E[token])`，按 FP32 EMA 在 CPU 计算。输入图像平均 embedding 的域内分类为 2.275%；1024×16 projector 的 16 个奇异值均非零，约 0.105–0.327。

## 最终 backbone

固定最终 RMSNorm 后 query，Flickr30K：

| 评分 | 图 → 文 R@1 | 文 → 图 R@1 |
| --- | ---: | ---: |
| 原始余弦 | **10.60%** | **2.66%** |
| 仅按模态中心化 | **17.90%** | **12.70%** |

图→文原始 / 中心化 R@1 的身份级 95% 区间为 [8.70%, 12.50%] / [15.60%, 20.30%]。探索性层扫描在 block27 得到中心化 26.40% / 17.36%，原始 12.60% / 4.02%。该层配对/非配对平均余弦为 0.9588 / 0.9466，中心化后为 0.3129 / 约 0。

ImageNet 最终 query 的线性分类（%）：

| 分类器拟合域 → 测试域 | 原始跨域使用 | 额外修正目标模态均值 |
| --- | ---: | ---: |
| 文本 → 图像 | **25.80** | **31.40** |
| 图像 → 文本 | **41.75** | **51.85** |
| 图像 → 留出的图像 | **52.05** | — |
| 文本 → 留出的文本模板 | **100.00** | — |

200 图测试池中，配对线性映射的 R@1 为 26.50% / 27.10%，打乱 fit 为 0.00% / 0.60%；同池仅中心化为 35.00% / 25.20%。这些指标描述固定读出的表征诊断。

## 产物和复现

16-rank 索引完整，64 组隐藏目标替换逐元素一致；3 项 CPU 指标测试通过。

[完整结果](../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/step-95415-ema-20260907/results.json)、[逐层 CSV](../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/step-95415-ema-20260907/layer_metrics.csv)、[输入局部匹配与区间](../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/step-95415-ema-20260907/input_token_geometry.json)、[曲线](../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/step-95415-ema-20260907/layer_curves.png)、[协议](../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/step-95415-ema-20260907/protocol.json)。

准备与分析命令在仓库根目录执行；NPU 启动脚本在 `dev-wjx-ascend` 内运行：

```bash
MODEL=output/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/hf_model-final-ema
REPR_OUTPUT=output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/new-run

TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=. .venv/bin/python \
  scripts/probe_unified_representations.py prepare \
  --model-source "$MODEL" --output-dir "$REPR_OUTPUT"

bash script/selfless/probe_unified_representations_dev_ascend16.sh "$MODEL" "$REPR_OUTPUT"

TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=. .venv/bin/python \
  scripts/analyze_unified_representations.py --output-dir "$REPR_OUTPUT"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=. .venv/bin/python \
  scripts/check_unified_input_geometry.py --output-dir "$REPR_OUTPUT"
```
