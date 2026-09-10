# 官方生成评测：GenEval、DPG-Bench、MJHQ-30K

最终 checkpoint 全量生成。生成使用16张910B；官方评分在各自独立 CUDA 环境运行。正式 A/B 尚无完整官方评分，历史模型已生成图片的状态见评测总览。

## 协议

| Benchmark | 固定数据与输出 | 主指标 |
| --- | --- | --- |
| GenEval | 官方 553 prompts，每条 4 张，共 2,212 张 | 六类 task image accuracy 的无权平均 Overall；同时保留六类分数 |
| DPG-Bench | 官方 1,065 prompts，每条 4 张并组成无间隔 2×2 grid | 官方 mPLUG VQA 的 DPG-Bench score；同时保留 L1/L2 类别分数 |
| MJHQ-30K | 官方 30,000 prompts，与原始 30,000 张 reference 一一对应 | clean-fid `mode=clean`、Inception-v3 的 overall FID；同时保留十类 FID |

固定版本：GenEval `af4902f24d3ca90ebbb446dd9891a59e0f82725f`；ELLA/DPG `3c228f1dc6c4d3cad0a47493816151a419f14db3`；MJHQ `15b0a659e066e763d0e9a6cd8f00e25f8af5e084`；clean-fid `e88c4d6269a4bbf04c04deeb578475b57719acee`（0.1.35）。

采样 CFG、steps、seed、solver、strategy、dtype 写入 `generation_manifest.json`。DPG 分数保存为0–1；MJHQ 同时记录生成/参考分辨率，256px 结果按同分辨率比较，1024px 官方口径单列。

## 准备资产

```bash
git clone https://github.com/djghosh13/geneval.git /path/to/geneval
git -C /path/to/geneval checkout af4902f24d3ca90ebbb446dd9891a59e0f82725f

git clone https://github.com/TencentQQGYLab/ELLA.git /path/to/ELLA
git -C /path/to/ELLA checkout 3c228f1dc6c4d3cad0a47493816151a419f14db3

git clone https://huggingface.co/datasets/playgroundai/MJHQ-30K /path/to/MJHQ-30K
git -C /path/to/MJHQ-30K checkout 15b0a659e066e763d0e9a6cd8f00e25f8af5e084
git -C /path/to/MJHQ-30K lfs pull --include=mjhq30k_imgs.zip
unzip /path/to/MJHQ-30K/mjhq30k_imgs.zip -d /path/to/mjhq-reference
```

GenEval 按官方 README 准备 Mask2Former；DPG 按 ELLA 的 `requirements-for-dpg_bench.txt` 准备 mPLUG。clean-fid 使用独立环境：

```bash
python -m venv /path/to/cleanfid-env
/path/to/cleanfid-env/bin/pip install \
  'git+https://github.com/GaParmar/clean-fid.git@e88c4d6269a4bbf04c04deeb578475b57719acee'
```

## 生成

以下路径替换为目标模型、官方资产和新的输出目录。模型序列保留训练前缀，官方 prompt 原文保持完整。

```bash
script/selfless/generate_official_t2i_benchmark_ascend16.sh \
  geneval configs/selfless/unified_baseline_100b_ascend_64npu.yaml \
  /path/to/hf_model-final-ema /path/to/geneval output/evaluation/official-generation/geneval

script/selfless/generate_official_t2i_benchmark_ascend16.sh \
  dpgbench configs/selfless/unified_baseline_100b_ascend_64npu.yaml \
  /path/to/hf_model-final-ema /path/to/ELLA output/evaluation/official-generation/dpgbench

script/selfless/generate_official_t2i_benchmark_ascend16.sh \
  mjhq configs/selfless/unified_baseline_100b_ascend_64npu.yaml \
  /path/to/hf_model-final-ema /path/to/MJHQ-30K output/evaluation/official-generation/mjhq
```

## 评分与汇总

```bash
python scripts/evaluate_official_t2i_benchmarks.py geneval \
  --repository /path/to/geneval \
  --image_dir output/evaluation/official-generation/geneval/generation/images \
  --detector_dir /path/to/geneval-detector \
  --python /path/to/geneval-env/bin/python \
  --output_dir output/evaluation/official-generation/geneval/metrics

python scripts/evaluate_official_t2i_benchmarks.py dpgbench \
  --repository /path/to/ELLA \
  --image_dir output/evaluation/official-generation/dpgbench/generation/images \
  --python /path/to/dpg-env/bin/python \
  --processes 1 \
  --output_dir output/evaluation/official-generation/dpgbench/metrics

python scripts/evaluate_official_t2i_benchmarks.py mjhq \
  --repository /path/to/MJHQ-30K \
  --reference_dir /path/to/mjhq-reference \
  --image_dir output/evaluation/official-generation/mjhq/generation/images \
  --python /path/to/cleanfid-env/bin/python \
  --output_dir output/evaluation/official-generation/mjhq/metrics
```

```bash
python scripts/evaluate_official_t2i_benchmarks.py summary \
  --geneval output/evaluation/official-generation/geneval/metrics/metrics.json \
  --dpgbench output/evaluation/official-generation/dpgbench/metrics/metrics.json \
  --mjhq output/evaluation/official-generation/mjhq/metrics/metrics.json \
  --output output/evaluation/official-generation/summary.json
```

评分器核对 revision、样本数、ID 和图片覆盖。官方说明：[GenEval](https://github.com/djghosh13/geneval)、[DPG-Bench](https://github.com/TencentQQGYLab/ELLA)、[MJHQ](https://huggingface.co/datasets/playgroundai/MJHQ-30K)、[clean-fid](https://github.com/GaParmar/clean-fid)。
