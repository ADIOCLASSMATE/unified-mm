# 评测协议检查报告

## 结论

当前正式协议已统一为“只发布去语言先验后的分数”，旧的 raw likelihood、
ImageNet 1K/5K 自定义检索和 generated-caption CLIP score 均不再有可运行或可复用
的正式路径。需要特别区分两件事：OpenAI CLIP 的标准 zero-shot 是归一化图文
embedding 的相似度与 prompt ensemble；本项目则只采用 CLIP 整理的 ImageNet 类别
名称，模型分数仍是 Selfless 的生成式 token likelihood。因此，新的 ImageNet 指标
是**固定、可复现的 transductive calibrated generative zero-shot protocol**，不是
“OpenAI CLIP accuracy”，也不能与 CLIP cosine-similarity leaderboard 数字直接横比。
图像生成质量现另行加入固定官方版本的 GenEval、DPG-Bench 和 MJHQ-30K
clean-fid；生成参数保持可调并逐次记录，官方数据覆盖、目录和聚合定义不可更改。

## 正式指标逐项检查

| 指标 | 实现与检查结论 | 学术报告口径 |
| --- | --- | --- |
| ImageNet-1K Top-1/Top-5 | 完整 val 50K × 1,000 类；CLIP 类名经两个重复 synset 消歧；单模板；每图每类取 mean token log-likelihood；在全部 50K 无标签图像上做 column-wise log-mean-exp prior；固定 alpha=1；只输出去先验 Top-1/Top-5 | 可报告为“transductive calibrated generative zero-shot classification”；不可称为标准 OpenAI CLIP zero-shot |
| COCO 5K I2T/T2I R@1/5/10 | Karpathy test；5,000 图、源 split 中全部 25,010 caption；完整矩阵；每图多正 caption；去先验后才计算 recall | R@K 和 split 是标准检索口径；必须披露 Selfless likelihood 与 calibration，不能与 cosine retrieval 无条件等价比较 |
| Flickr30K 1K I2T/T2I R@1/5/10 | Karpathy test；1,000 图、5,000 caption；完整矩阵；去先验后计算 recall | 同上 |
| SugarCrepe | 官方 7 类、7,511 对；正负 caption 严格 win/tie/loss；3 张固定无标签 Gaussian null image 估计 prior | 可作 compositional matching；必须报告各类别、null-image 设定和 alpha；tie 不算 win |
| ARO VG Relation/Attribution | 官方 pair，分别 23,937/28,748 条；同一去先验 strict pairwise 协议 | 可作 ARO pair ranking；需声明生成式 scorer/calibration，不冒充原论文 encoder similarity |
| MMBench/SEED | 全部候选也已去先验 | 仅内部 ablation；semantic candidate likelihood 不等于官方 free-form leaderboard 协议 |
| ImageNet-val FID/IS | 50K fake；torch-fidelity-compatible Inception；real reference 是 ImageNet val 50K；IS 采用按 synset 分层的 10 splits | 项目内同协议可比；不是 ADM/DiT 常用 train-reference evaluator，禁止标作其 leaderboard-comparable FID |
| GenEval | 官方 553 prompts × 4 images；固定原始 GenEval commit；直接运行官方 Mask2Former/OpenCLIP evaluator；Overall 为六个 task image accuracy 的无权平均 | 学术界常用 compositional generation 指标；必须同时披露六类分数，不能以总体 image accuracy 替代 Overall |
| DPG-Bench | 官方 1,065 prompts × 4 images，组成无间隔 2×2 grid；固定 ELLA commit；直接运行官方 mPLUG VQA evaluator | 报告 DPG score 及 L1/L2；适配器要求 1,065 grid 全部成功，不能接受官方脚本捕获异常后产生的残缺结果 |
| MJHQ-30K FID | 官方 30,000 prompts/reference；固定 MJHQ revision；clean-fid 0.1.35、`clean` mode、Inception-v3；overall 加十类 FID | FID 越低越好；只有 reference/generated 都是 30K 张 1024×1024 时才可标为 MJHQ 1024 横榜可比 |
| MAR KL16 reconstruction FID | ImageNet val 50K；固定 seed 的 posterior sample 为 primary；posterior mean 仅 diagnostic；同一 Inception/real moments | 可以报告为明确限定的 val-rFID；样本路径与 mean 路径不能混为一个主结果 |
| ARC/OpenBookQA | 已改为 test split；ARC/OpenBookQA primary 为 length-normalized accuracy | 与固定 lm-eval 参考版本对齐；Selfless 必须使用 same-position adapter |
| MMLU | test、5-shot、57 subject macro accuracy | subject macro 为 primary；example micro 与跨任务 macro 均不得替代它 |

## 去先验公式和方向不对称

所有 dense image-text 任务先形成矩阵

`S[i,c] = mean_token_log P(text_c | image_i)`，

再计算

`b[c] = logsumexp_i S[i,c] - log(N)`，

最终只用 `S'[i,c] = S[i,c] - b[c]` 排名（`alpha=1`）。实现使用
probability-space mean 对应的 `logmeanexp`，不是错误的 mean-log-score。
分类 accuracy、检索 recall 与 benchmark accuracy 在 JSON 中统一保存为
`unit_interval`（例如 `0.382`），论文 Markdown 表格才乘以 100 显示为百分数
（例如 `38.20%`），避免 0--1 与 0--100 混用。

这也解释了此前 I2T/T2I 的巨大差异：I2T 固定一张图并横向比较不同文本，减去
不同的 `b[c]` 会改变排序；T2I 固定一个文本并纵向比较图像，所有候选同时减去同一
常数，因此排名与 R@K 数学上完全不变。若要同时改变 T2I，需要另行定义 image-side
prior 或对称 PMI，但那不是用户指定的 `log P(text|image)-log P(text)`，不能悄悄加入。

## 协议边界与风险披露

1. ImageNet prior 使用全部 50K evaluation images，虽不读取标签，仍属于
   transductive/test-distribution calibration；论文方法与表头必须明确写出。
2. 单模板由协议预先冻结。不能在 ImageNet val 标签上试多个模板后挑最高分，否则
   zero-shot 声明会受到 validation-set tuning 污染。若未来比较模板，应使用独立
   development set 并在正式评测前冻结。
3. `alpha=1` 是固定方法选择而非当前 checkpoint 上调优出的最优超参。不同论文若
   使用 raw cosine、不同 alpha 或不同 null-image estimator，数值不应直接横比。
4. SugarCrepe/ARO 的三张 null image 采用模型实际 VAE 输入空间中的
   `N(0,0.25)` 并 clamp 到 `[-1,1]`。这是面向 Selfless 的明确适配；并非声称所有
   VisualGPTScore 模型都使用完全相同的像素均值。
5. 运行时哈希按实验约束关闭，因此 text contamination check 也明确为 disabled；
   这不是“已证明无污染”。
6. 生成的采样策略、步数、CFG、solver、seed 和计算精度是可调实验参数，结果文件会
   记录实际取值，但这些参数不作为“正式协议”的硬编码门禁。比较 FID/IS 时仍应保证
   两次运行采用相同参数；配置文件中的数值只是当前运行配方。
7. GenEval/DPG-Bench 的判别器本身也有模型偏差，因此它们衡量的是可检测的
   compositional alignment，不等同于人工整体美学偏好；这正是同时保留细分类分数和
   MJHQ distributional FID 的原因。
8. MJHQ 官方表格使用 1024×1024。将当前原生 256×256 结果交给同一个 clean-fid
   实现，并不会自动使它具有 1024 横榜可比性；代码会据实际文件分辨率强制标注。

## 已删除或硬拒绝的旧协议

- ImageNet-val class-balanced 1K/5K exact-instance retrieval；
- generated-caption CLIP score evaluator 及其测试/launcher 分支；
- raw/uncalibrated retrieval 或 hard-negative likelihood；
- legacy visual calibration 与 custom ImageNet caption negatives；
- ARC validation、OpenBookQA validation、MMLU example-micro primary；
- 旧 multimodal/text/retrieval result schemas；
- 把 ImageNet-val-reference FID 标成 “official” 或 ADM/DiT comparable 的字段。

## 实现中的强制门禁

- ImageNet 正式运行必须恰好 50,000 图、1,000 类、固定类名文件和模板，最终文件不
  保存 raw score variant。
- COCO/Flickr 只有完整、无重复 query coverage 的矩阵才能 merge；旧 schema
  不能复用。
- multimodal manifest v2 固定 3 个 null ID、3 个 seed、像素空间、均值/方差、
  clamp 和 lossless PNG；cache v2 也必须显式包含这些 ID，旧 manifest/cache 会
  直接失败。
- formal summarizer 校验数据条数、模型来源、checkpoint step、calibration、schema
  与 finite metrics；不会把缺失 benchmark 当成零。
- rFID formal gate 固定 50K、16 shards、2048-d Inception、sample+mean、seed 42、
  FP32 decoder，并把 sample rFID 指定为唯一 primary。
- GenEval、DPG-Bench、MJHQ checkout 必须分别命中固定 commit；prompt 数必须为
  553/1,065/30,000，GenEval 与 DPG 必须各生成 4 张，MJHQ 十类必须各 3,000 张。
- 三项评分都读取生成 manifest，校验 checkpoint provenance、实际生成参数、图片
  分辨率和完整 ID 覆盖；最终 summary 拒绝混用不同 checkpoint。
- GenEval 结果必须完整包含 2,212 image rows；DPG 必须完整包含 1,065 grid rows；
  MJHQ reference/generated 必须各自与 30,000 metadata ID 一一对应。

## 参考依据

- [OpenAI CLIP ImageNet prompt/class-name notebook](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb)
- [OpenAI CLIP paper](https://arxiv.org/abs/2103.00020)
- [Revisiting the Role of Language Priors in Vision-Language Models (VisualGPTScore)](https://arxiv.org/abs/2306.01879)
- [Karpathy image-caption retrieval split source](https://cs.stanford.edu/people/karpathy/deepimagesent/)
- [SugarCrepe official repository](https://github.com/RAIVNLab/sugar-crepe)
- [ARO official repository](https://github.com/mertyg/vision-language-models-are-bows)
- [lm-evaluation-harness official repository](https://github.com/EleutherAI/lm-evaluation-harness)
- [OpenAI guided-diffusion evaluator reference](https://github.com/openai/guided-diffusion/tree/main/evaluations)
- [MAR official repository](https://github.com/LTH14/mar)
- [GenEval official repository](https://github.com/djghosh13/geneval)
- [ELLA / DPG-Bench official repository](https://github.com/TencentQQGYLab/ELLA)
- [MJHQ-30K official dataset](https://huggingface.co/datasets/playgroundai/MJHQ-30K)
- [clean-fid official repository](https://github.com/GaParmar/clean-fid)

## 2026-09-05：文本 P1 修正（v3）

此前“length-normalized accuracy 已对齐”的结论不完整：实现错误地除以
token 数。现在改为固定 lm-eval 版本的 `len(original_choice)`，即原始选项的
Unicode 字符数；不包含编码时新增的分隔空格，也不是 UTF-8 字节数。
ARC-Easy/Challenge、HellaSwag、PIQA、OpenBookQA 的主指标受此修正影响。

WinoGrande 改为 `log P(shared_suffix | prefix + option)`，仅对共享后缀打分，
不再把候选 option 自身的似然混进目标。它的主指标仍为未归一化 accuracy。
同位置 Selfless query-stream 打分和 A/B 的 attention contract 保持不变。

文本协议升至 v3，sample/metrics 升至 v2，summary/run/rank-complete 升至 v4；
完整评测协议升至 v10。旧分片、旧 summary 和旧 core 结果不能被正式续跑、归档、
趋势汇总当作新协议复用。旧推理 LL 可由
`scripts/repair_text_benchmark_results.py --text_dir <已有文本结果目录>` 原地重算，
原始结果保留在该目录的 `legacy-before-p1/`。WinoGrande 旧 LL 无法恢复共享后缀
分数，必须重新推理；在传入 `--winogrande_text_dir <新协议重评目录>` 合并之前，
其结果与八任务 macro 均不可作为有效当前指标，summary 标记为 incomplete。

## 历史验证状态（2026-09-02，文本结论由上面的 P1 修正覆盖）

本次完整 CPU suite 为 `384 passed, 2 skipped`（两项是 CUDA FlexAttention
集成测试）。CPU 控制节点必须设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，因为该节点
没有 CANN/`libascend_hal.so`；这只影响本地测试收集，不影响 Ascend 正式作业。
全部正式多模态 likelihood 作业已在 16×Ascend 910B 环境中完成。

数据门禁也已对当前资产实检：ImageNet val 为 50,000 图/1,000 类且每类 50 图；
固定 CLIP 类名文件与所钉住 notebook 除两项消歧外逐项一致；COCO/Flickr 分别为
5,000 图/25,010 caption 和 1,000 图/5,000 caption；multimodal asset v2 为
61,036 张唯一图；rFID 的 16 个 posterior shard 与 real-stat 均完整覆盖 50,000 图。
此外已对固定 GenEval 与 ELLA checkout 实检 553/1,065 个 prompt、GenEval 六类
分布和 DPG prompt-file/CSV ID 一致性；MJHQ 官方 metadata 实检为 30,000 条、十类
各 3,000 条。

正式结果均直接来自 A/B 的 step 95,415 `hf_model-final-ema`：ImageNet 50K×1,000
分类、ImageNet 50K 生成、COCO/Flickr 全量检索、8 项文本任务，以及 FP32、MC=64
的 SugarCrepe、ARO Relation、ARO Attribution、MMBench 和 SEED 均已完成。后五项
分别严格覆盖 7,511、23,937、28,748、4,329、14,233 条记录，每项 16 个 rank
shard；formal 合并器验证 `project_formal_protocol=true`、三张固定 null image、
`alpha=1`、仅输出去先验分数。A/B 共 157,516 条预测记录中未出现 `NaN` 或
`Infinity` 字面量，运行时也由 fail-fast 门禁逐 token 检查非有限值。

开发机上还用完全撤回猜测性 attention 补丁后的原始模型代码，对曾失败的 B/ARO
Relation 前 400 条连续复跑两次；两次预测文件 SHA256 均为
`6d9314471248a456187a485eb4df0bb3d8bcfee9a0285f1fd4da5037384c83e6`。因此没有把
“全 mask attention 行”写成未经证实的根因，也没有保留相关模型补丁；正式全量
复跑同样通过。

官方生成集也已完整出图并校验 ID/目录：每个模型 GenEval 2,212 张、DPG-Bench
4,260 张样本加 1,065 张官方 2×2 grid、MJHQ-30K 30,000 张。GenEval
Mask2Former、DPG mPLUG、MJHQ 30K clean-fid 尚未在各自 CUDA 环境执行，所以三项
分数继续严格留空，不以 0 或替代指标填充。
