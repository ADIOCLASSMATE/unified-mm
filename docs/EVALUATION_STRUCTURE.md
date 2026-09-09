# 评测结构

统一入口是 [`output/evaluation/index.html`](../output/evaluation/index.html)。
所有模型的正式分数、定性输出、训练验证与表征分析都在 `output/evaluation/`；
权重、优化器、随机状态、续训 checkpoint 和训练日志保留在训练目录。

```text
output/evaluation/
├── index.html                         # 研究摘要、模型/采样消融、训练验证、样例与协议
├── summary.json / selection.json      # 可读汇总与明确的结果选择
├── unified-*/                         # 各模型正式评测与历史 checkpoint 结果
├── qualitative/<run>/                 # 全模型逐样本输出、原报告及 ZIP
├── training-validation/<training-run>/ # 训练阶段验证，含历史 sweep
├── training-loss/                     # 所有统一消融的 loss JSON/CSV 与 PNG/SVG
├── data-provenance/                   # 数据来源说明、合成协议、全量来源统计
├── research/                          # 表征、语义与 geometry 研究
├── comparisons/                       # 比较报告与 FID 复核
├── diagnostics/ / audits/             # 诊断与评测审计
└── migrations/                        # 迁移映射、核验和原始 metadata
```

运行 `python3 scripts/build_evaluation_report.py` 更新总览，不启动设备或重新评分。
选取依据在 `configs/protocols/evaluation_report.json`；脚本直接读取原始完成结果，
校验模型身份、checkpoint step、协议和样本覆盖。未完成项保持空白，失效结果拒绝纳入。
旧 B 的路径重命名使用单独的控制组身份记录，D 只选修复后的 r2。
报告是生成时刻的快照；同一命令会重新读取新增指标与训练日志。

网页按六个入口组织：

| 入口 | 内容 |
| --- | --- |
| 研究总览 `#overview` | 研究摘要，以及后续五节各自的总表、结论和详情导航 |
| 模型消融 `#matrix` | 模型结构与训练变体的评测分数；按指标组、生成协议、模型范围切换 |
| 采样与解码消融 `#sampling/parameters` | CFG / Heun 步数扫描、B_x0 解码策略、跨模型换序三个子页；各自保留分数、曲线与配对样例 |
| 训练与验证 Loss `#training` | 所有训练实验的总 loss 及 T2I / I2T / 纯文本曲线 |
| 定性样例 `#qualitative/t2i` | T2I、I2T、文本续写三个子页，共用模型与样本筛选 |
| 数据与协议 `#sources` | 数据来源、合成模型与模板、来源核验及全部文件入口 |

首页与模型消融页默认使用 CFG=2.0、Heun=10 的原生顺序 FID / IS：
E 使用 Sequential，其余使用 Halton，直接链接到对应的原始完成结果。
可切换至历史 CFG=3.5；不改变文本、图像理解、检索等自身协议的分数。
CFG=2.0 源自 B_x0 参数扫描，未对各模型分别选优；缺少该协议的分数保持空白。
同配置下的解码顺序比较集中在 `#sampling/cross-model`，不混入模型指标表。
旧链接 `#sweep` / `#order` 和 `#t2i` / `#i2t` / `#text` 会跳到对应的新子页。

“训练与验证 Loss”独立扫描 `output/unified-*` 的正式训练配置和 `training_metrics.jsonl`，
覆盖主要消融、单任务/历史对照、1.7B LR sweep，以及尚未导出 EMA 的 flow-depth
实验；不将 smoke/debug/replay 纳入正式曲线。T2I、I2T、ClimbMix 分别读取
`train/loss_t2i`、`train/loss_i2t`、`train/loss_climbmix`，并保留总 loss 和三项加权贡献。
混合 `train/loss_text` 不冒充纯文本 loss，缺失任务保持空白。原始点完整导出；平滑仅作用于显示。
状态依据 runtime 的已到步数及配置中的 `stop_after_steps`/`max_train_steps`，不推断平台实时状态。
训练用实线，验证用虚线与圆点，可分别选择；验证不做平滑。验证来源为
`training-validation/<run>/validation_metrics_step_*.json`，兼容未迁移的训练目录；
独立最终 EMA 评测不加入训练期间曲线。目标数为零的任务不画零值。
历史 `val/loss_text` 为 I2T caption CE，不冒充纯文本验证。
新纯文本验证文件 `validation_climbmix_metrics_step_*.json` 也会自动收录，并保留
源记录是否排除的信息，见 [纯文本验证 loss](CLIMBMIX_VALIDATION.md)。
新协议文件 `validation_unified_loss_metrics_step_*.json` 使用与训练一致的任务组成、
权重、microbatch 平均和调度比例，见[统一 loss 验证](UNIFIED_LOSS_VALIDATION.md)。
其中 T2I / I2T 与轻量分类评测共用每类 2 张、共 2,000 张的有序 ImageNet-val 清单。
网页可按新/旧验证协议筛选，同一步采用完整新记录，协议切换处断开曲线。
原始训练 CSV 为 `training-loss/curves.csv`，验证 CSV 为 `training-loss/validation.csv`，
`curves.json` 包含两者及协议元数据。历史 ImageNet 验证总 loss 的聚合口径保持原样。

“数据与协议”的来源说明维护在
[`evaluation_data_sources.json`](../configs/protocols/evaluation_data_sources.json)，
Qwen/MiniMax 模板直接读取仓库配置与函数，记录示例及全量计数保存在
`data-provenance/audit.json`。首次构建或源文件更新后运行一次
`python3 scripts/audit_evaluation_data_provenance.py`；它读取全部已发布 caption/T2I
记录，统计真实模型与协议版本，不做推理或重新计算哈希。后续构建只核验文件大小与修改时间。

额外传入 `--plots` 可刷新独立 PNG/SVG；默认构建始终刷新交互曲线和原始 CSV/JSON，
静态图标出自己的导出时间。CPU 绘图依赖见
[`evaluation_report_requirements.txt`](../scripts/evaluation_report_requirements.txt)。例如使用独立环境：

```bash
uv venv /tmp/unified-evaluation-report-venv --python .venv/bin/python
uv pip install --python /tmp/unified-evaluation-report-venv/bin/python -r scripts/evaluation_report_requirements.txt
/tmp/unified-evaluation-report-venv/bin/python scripts/build_evaluation_report.py --plots
```

训练入口自动设置 `experiment.validation_output_dir`；独立评测与回放显式覆盖该字段，
因此已有训练配置中的路径不会把评测写回训练目录。旧的执行日志与代码快照保留历史
路径，迁移映射见 `output/evaluation/migrations/20260908-consolidation/`。

`output/evaluation-checkpoints/` 仅含历史评测输入权重，属于 checkpoint 资产，
不存放评测结果。

当前 final 协议直接加载训练结束导出的 FP32 EMA Hugging Face 目录
`hf_model-final-ema`。协议不做下游微调、指令微调或线性探针，运行时不计算
内容哈希。模型来源 metadata 决定 architecture variant、attention contract 和
checkpoint step。

## 论文可报告指标

- ImageNet-1K zero-shot classification：完整 50,000 张 validation 图像、1,000
  个 OpenAI CLIP 类别名称、单模板，报告 `alpha=1` 去文本先验后的 Top-1/Top-5。
- MSCOCO Karpathy 5K 与 Flickr30K Karpathy 1K：完整候选集合上的 I2T/T2I
  R@1/R@5/R@10，只报告去文本先验分数。
- SugarCrepe、ARO VG Relation、ARO VG Attribution：只报告三张固定 Gaussian
  null image 估计先验后的严格 pairwise win rate。
- 纯文本：ARC-Easy/Challenge 与 OpenBookQA 使用 test split，HellaSwag、PIQA、
  WinoGrande、BoolQ 使用 validation，MMLU 使用 5-shot test 和 subject-macro
  accuracy。跨任务 macro 只作内部摘要。
- VAE reconstruction FID：posterior sample 是主指标，posterior mean 只作诊断。
- 图像生成质量：官方 GenEval（553×4）、DPG-Bench（1,065×4）和 MJHQ-30K
  clean-fid（30,000）均在全量官方 prompts 上报告；三套评分器使用固定官方版本。

所有图文候选分数均来自 Selfless same-position query-stream 的平均 token
log-likelihood，不套用普通 causal LM 的 one-token shift。

## 生成指标的边界

项目生成评测生成 50,000 张 ImageNet-val 条件图，使用 torch-fidelity-compatible
Inception、ImageNet-val 50K real statistics，以及按 synset 分层的 10-split IS。
该结果适合项目内、同协议 checkpoint 比较；由于 real reference 和标准 ADM/DiT
评测生态不同，不得标为 ADM/DiT leaderboard-comparable FID。

最终 checkpoint 还单独执行官方生成 benchmark。GenEval 主指标是六任务 image
accuracy 的无权平均；DPG-Bench 主指标是官方 mPLUG VQA score；MJHQ 主指标是
clean-fid overall FID，并补充十类 FID。MJHQ 公开结果以 1024×1024 生成图为口径；
若本模型原生输出 256×256，结果文件必须保持
`leaderboard_comparable_at_1024px=false`，只能做相同分辨率、相同协议比较。

## 诊断指标

ImageNet held-out T2I/I2T loss、text PPL、生成图片与生成 caption 用于训练健康
检查。MMBench Dev-EN circular 与 SEED-Bench image 只用于内部消融；它们不是
官方 free-form leaderboard 分数。

## 已删除且 schema 会拒绝的协议

- ImageNet-val class-balanced 1K/5K I2T/T2I retrieval；
- raw/uncalibrated image-text likelihood 与旧 visual calibration；
- generated-caption CLIP score、ImageNet ReaL 和自定义 caption negatives；
- POPE、COCO Caption PPL，以及未具备完整官方资产的 benchmark 结果；
- ARC validation、OpenBookQA validation、MMLU example-micro primary；
- 将 ImageNet-val-reference FID 描述成“official/ADM/DiT comparable”的旧标签。

## 统一入口

```bash
EVAL_PROFILE=formal \
  script/selfless/evaluate_unified_native_full_checkpoint_ascend16.sh \
  /path/to/hf_model-final-ema /path/to/evaluation-output
```

完整协议定义见
`configs/protocols/unified_full_evaluation_ascend16.yaml`；评测审计与可比性说明见
`docs/EVALUATION_PROTOCOL_AUDIT.md`。三项官方生成 benchmark 因评分依赖 CUDA
Mask2Former、ModelScope mPLUG 和 clean-fid，作为独立的 final-checkpoint release
stage 运行，命令见 `docs/OFFICIAL_T2I_BENCHMARKS.md`。
