# 评分协议

模型、checkpoint、raw/EMA、step、dtype、数据版本和采样设置随每份结果保存。准确率与 recall 在 JSON 中使用 0–1，表格显示为百分数。

## 文本

Selfless 使用同位置 query 评分；C 的纯文本与 S2 使用 next-token AR 对齐。

| 任务 | Split | 主指标 |
| --- | --- | --- |
| ARC-Easy / Challenge、OpenBookQA | test | 按原始选项 Unicode 字符数归一化的 accuracy |
| HellaSwag、PIQA | validation | 同上 |
| WinoGrande | validation | 给定 prefix + option 后，共享 suffix 的似然 accuracy |
| BoolQ | validation | accuracy |
| MMLU | test，5-shot | 57 个 subject 的 macro accuracy |

八任务算术均值作为项目摘要。实现位于 `utils/evaluation/text_benchmarks.py`，协议为 `selfless_text_benchmark_v3`；历史重算结果见 [文本修正](TEXT_P1_CORRECTION_20260905.md)。

## 图文候选似然

定义候选文本的平均 token log-likelihood 为：

```text
S[i,c] = mean_token log P(text_c | image_i)
b[c]   = logsumexp_i S[i,c] - log(N)
S'[i,c] = S[i,c] - b[c]      # alpha = 1
```

ImageNet 分类与 COCO/Flickr 检索在全体评测图像上估计列先验，使用 `S'` 排名。这是基于评测图像分布的语言先验校准。I2T 横向比较不同文本，校准会改变排序；T2I 对固定文本减去同一常数，排序保持一致。

SugarCrepe / ARO 使用三张固定 Gaussian null image 估计先验：VAE 输入空间 `mean=0, std=0.25`，clamp 至 `[-1,1]`。正负例按严格胜率汇总，并单列平局。

| 任务 | 规模 | 主指标 |
| --- | --- | --- |
| ImageNet val | 50,000 图 × 1,000 类；`a photo of a {class_name}.` | 校准 Top-1 / Top-5 |
| COCO Karpathy | 5,000 图、25,010 caption | I2T / T2I R@1/5/10 |
| Flickr30K Karpathy | 1,000 图、5,000 caption | I2T / T2I R@1/5/10 |
| SugarCrepe | 7,511 对，七类 | 总体及分类别严格胜率 |
| ARO Relation / Attribution | 23,937 / 28,748 对 | 严格胜率 |
| MMBench Dev-EN / SEED image | 4,329 / 14,233 条 | 项目内候选似然诊断 |

ImageNet 使用 CLIP 类名表和单模板，分数来自本模型的生成式似然。MMBench / SEED 的评分方式是候选似然，区别于官方答案生成协议。资产和命令见 [图像理解](PRETRAINING_NATIVE_EVALUATION.md)。

## 生成

| 指标 | 协议 |
| --- | --- |
| ImageNet FID / IS | 50K 生成图；ImageNet-val 50K real stats；torch-fidelity Inception；10 个 synset 分层 IS split |
| KL16 reconstruction FID | 同一 val reference；posterior sample 为主指标，posterior mean 为诊断 |
| GenEval | 553×4 图；六任务 accuracy 的无权平均 |
| DPG-Bench | 1,065×4 图，2×2 grid；官方 mPLUG VQA score |
| MJHQ-30K | 30K reference / generated；clean-fid overall 与十类 FID |

ImageNet 结果按本项目 val-reference 协议比较。MJHQ 结果记录实际生成分辨率，1024×1024 与本项目原生 256×256 分列。官方版本和命令见 [生成评测](OFFICIAL_T2I_BENCHMARKS.md)。

## 参考

- [CLIP 类名与模板](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb)
- [VisualGPTScore](https://arxiv.org/abs/2306.01879)
- [Karpathy retrieval split](https://cs.stanford.edu/people/karpathy/deepimagesent/)
- [SugarCrepe](https://github.com/RAIVNLab/sugar-crepe) · [ARO](https://github.com/mertyg/vision-language-models-are-bows)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
