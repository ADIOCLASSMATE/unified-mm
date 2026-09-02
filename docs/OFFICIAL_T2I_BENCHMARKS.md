# 官方图像生成评测：GenEval、DPG-Bench、MJHQ-30K

这三项只在最终 checkpoint 上全量评测。模型生成仍在 16 张 Ascend 910B 上完成；
GenEval、DPG-Bench 和 clean-fid 分别在独立 CUDA 环境中运行官方代码，不把它们的
互相冲突依赖加入项目主环境。

## 固定协议

| Benchmark | 固定数据与输出 | 主指标 |
| --- | --- | --- |
| GenEval | 官方 553 prompts，每条 4 张，共 2,212 张 | 六类 task image accuracy 的无权平均 Overall；同时保留六类分数 |
| DPG-Bench | 官方 1,065 prompts，每条 4 张并组成无间隔 2×2 grid | 官方 mPLUG VQA 的 DPG-Bench score；同时保留 L1/L2 类别分数 |
| MJHQ-30K | 官方 30,000 prompts，与原始 30,000 张 reference 一一对应 | clean-fid `mode=clean`、Inception-v3 的 overall FID；同时保留十类 FID |

固定版本如下：

- GenEval `djghosh13/geneval@af4902f24d3ca90ebbb446dd9891a59e0f82725f`
- DPG-Bench `TencentQQGYLab/ELLA@3c228f1dc6c4d3cad0a47493816151a419f14db3`
- MJHQ-30K `playgroundai/MJHQ-30K@15b0a659e066e763d0e9a6cd8f00e25f8af5e084`
- clean-fid `GaParmar/clean-fid@e88c4d6269a4bbf04c04deeb578475b57719acee`
  （package version `0.1.35`）

脚本会校验 git revision、prompt 数量、类别分布、ID 覆盖和图片目录，不接受抽样结果
冒充正式指标。CFG、采样步数、seed、solver、strategy 和 dtype 不是 benchmark 定义，
可通过 launcher 环境变量调整；每次实际取值都会写入 `generation_manifest.json`。

## 1. 准备官方仓库

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

GenEval 按其 README 建独立环境并下载 Mask2Former；DPG-Bench 按 ELLA 的
`requirements-for-dpg_bench.txt` 建独立环境并准备 ModelScope mPLUG 权重。MJHQ
环境安装固定 clean-fid：

```bash
python -m venv /path/to/cleanfid-env
/path/to/cleanfid-env/bin/pip install \
  'git+https://github.com/GaParmar/clean-fid.git@e88c4d6269a4bbf04c04deeb578475b57719acee'
```

## 2. 在 Ascend 上生成官方目录

三个 benchmark 分别运行一次；下面的生成参数只是当前配方，后续可用同名环境变量
调整。

```bash
script/selfless/generate_official_t2i_benchmark_ascend16.sh \
  geneval configs/selfless/unified_baseline_100b_ascend_64npu.yaml \
  /path/to/hf_model-final-ema /path/to/geneval output/geneval

script/selfless/generate_official_t2i_benchmark_ascend16.sh \
  dpgbench configs/selfless/unified_baseline_100b_ascend_64npu.yaml \
  /path/to/hf_model-final-ema /path/to/ELLA output/dpgbench

script/selfless/generate_official_t2i_benchmark_ascend16.sh \
  mjhq configs/selfless/unified_baseline_100b_ascend_64npu.yaml \
  /path/to/hf_model-final-ema /path/to/MJHQ-30K output/mjhq
```

输出分别严格采用 GenEval prompt-folder、DPG 2×2 grid、MJHQ category-folder 格式。
模型内部仍使用训练时的
`Generate an image matching this description: ...` 序列化前缀；官方 prompt 文本本身
不会截断或改写。

## 3. 在各自官方 CUDA 环境评分

```bash
python scripts/evaluate_official_t2i_benchmarks.py geneval \
  --repository /path/to/geneval \
  --image_dir output/geneval/generation/images \
  --detector_dir /path/to/geneval-detector \
  --python /path/to/geneval-env/bin/python \
  --output_dir output/geneval/metrics

python scripts/evaluate_official_t2i_benchmarks.py dpgbench \
  --repository /path/to/ELLA \
  --image_dir output/dpgbench/generation/images \
  --python /path/to/dpg-env/bin/python \
  --processes 1 \
  --output_dir output/dpgbench/metrics

python scripts/evaluate_official_t2i_benchmarks.py mjhq \
  --repository /path/to/MJHQ-30K \
  --reference_dir /path/to/mjhq-reference \
  --image_dir output/mjhq/generation/images \
  --python /path/to/cleanfid-env/bin/python \
  --output_dir output/mjhq/metrics
```

最后只做结果汇总，不重新计算指标：

```bash
python scripts/evaluate_official_t2i_benchmarks.py summary \
  --geneval output/geneval/metrics/metrics.json \
  --dpgbench output/dpgbench/metrics/metrics.json \
  --mjhq output/mjhq/metrics/metrics.json \
  --output output/official-t2i-summary.json
```

## 报告边界

- GenEval 报告 `overall` 及六个 task score；不能用总体 image accuracy 替代官方
  “六任务无权平均”。
- DPG-Bench 报告 0--1 形式的分数，论文表格显示时再乘 100。适配器会检查全部
  1,065 grids 都被官方脚本成功处理，不能容忍其异常捕获后静默漏图。
- MJHQ FID 越低越好。官方公开结果使用 1024×1024；适配器会记录 reference 和
  generated resolution，只有两侧均为 30,000 张 1024×1024 时才把
  `leaderboard_comparable_at_1024px` 标为 true。当前模型若原生输出 256×256，仍可
  得到 clean-fid 数值，但只能作为本模型/同分辨率消融比较，不能冒充 MJHQ 1024
  leaderboard 可比结果。

官方依据：[GenEval](https://github.com/djghosh13/geneval)、
[ELLA / DPG-Bench](https://github.com/TencentQQGYLab/ELLA)、
[MJHQ-30K](https://huggingface.co/datasets/playgroundai/MJHQ-30K)、
[clean-fid](https://github.com/GaParmar/clean-fid)。
