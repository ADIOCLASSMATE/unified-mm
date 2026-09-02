# 评测结构

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
