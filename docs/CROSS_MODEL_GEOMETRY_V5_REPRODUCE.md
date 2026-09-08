# V5 复现与继续计算

本文件是命令入口，不是完成声明。数据、模型、原始／修订特征全部保留；
默认分析读取冻结契约，缺失内容或契约不一致时应报错，不可改成跳过模型。

## CPU 分析（已有全部特征后）

从仓库根目录执行，使用已有 `.venv`；不要将绘图依赖覆盖进模型环境。
先确认没有同一设置／层分片的写入者，再恢复断点。已有完整 JSON 结果会复用；
不能因观察连接超时而重复启动仍然存活的任务。

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=.

.venv/bin/python scripts/prepare_geometry_v4_views.py \
  --output-dir output/evaluation/research/cross-model-geometry-v5-20260907/f-v4
.venv/bin/python scripts/prepare_geometry_v5_views.py --workers 1
.venv/bin/python scripts/analyze_geometry_v5.py --workers 12
.venv/bin/python scripts/bootstrap_geometry_v5.py --workers 3
.venv/bin/python scripts/analyze_geometry_v5_robustness.py --workers 3
.venv/bin/python scripts/audit_geometry_v5_results.py
.venv/bin/python scripts/compare_geometry_v5.py --workers 4
.venv/bin/python scripts/audit_geometry_v5_comparisons.py --workers 4
.venv/bin/python scripts/audit_geometry_v5_summary.py
.venv/bin/python scripts/explain_geometry_v5_perturbations.py

PYTHONPATH=public/models/_dependencies/geometry-v5-plots:. \
  .venv/bin/python scripts/plot_geometry_v5.py
.venv/bin/python scripts/report_geometry_v5.py
```

`--settings` 可限定尚未完成的设置；分析的 `--pair-indices` 仅用于事先确定的
执行分片，不按得分删层。分片调用增加 `--no-summary`，最后统一执行
`analyze_geometry_v5.py --summarize-only`。共享汇总文件不能由并发进程竞争写入。

```bash
.venv/bin/python scripts/audit_geometry_v5_samples.py
.venv/bin/python scripts/audit_geometry_v5_parity.py
.venv/bin/python scripts/audit_geometry_v5_model_sources.py
.venv/bin/python scripts/audit_geometry_v5_features.py --workers 2
.venv/bin/python -m pytest tests/test_geometry_v3.py tests/test_geometry_v4.py \
  tests/test_geometry_v5.py -q \
  --junitxml=output/evaluation/research/cross-model-geometry-v5-20260907/audits/tests-v3-v4-v5.xml
```

正式生成的图表位于 `figures/`，机器可读索引 `figures/index.json`；
`figures-preflight/` 明确是部分模型的绘图预检，不能当完整交付。
`RESULTS_ZH.md` 只有全部结果和配对差值审计通过后才能生成。
配对计算与审计可用 `--workers 1` 串行复算；并行只改变执行分配，不改变
配对身份、随机抽样或统计定义。16 组真实端点的串行／4-worker 数值和
全部区间已逐项验证完全一致，预检证据为 `audits/paired-parallel-preflight.json`。

## 模型与依赖

权重及官方代码的精确 revision、文件范围与下载字节数分别位于
`asset-manifest.json`、`audits/model-sources-and-training.json`。这些命令
默认是离线分析，不能重新下载不同版本“覆盖修复”。原始 Qwen3 Base 是
已有本地权重，不是 B 的初始化诊断参考。

若需要重建绘图依赖，仅写独立 public overlay，下载严格禁用代理：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY='*' no_proxy='*' UV_NO_CONFIG=1 \
  UV_CACHE_DIR=public/models/_dependencies/uv-cache \
  uv pip install --python .venv/bin/python \
  --target public/models/_dependencies/geometry-v5-plots \
  --index-url https://pypi.org/simple --no-build \
  -r scripts/geometry_v5_plot_requirements.txt
```

模型适配器依赖见 `geometry_v5_flow_requirements.txt` 和
`geometry_v5_common_requirements.txt`。绘图 overlay 不用于特征或数值拟合。

## 重新采集才需要 NPU

仅使用固定 `dev-wjx-ascend`，遵循 `INSPIRE.md` 和 Inspire 当前 CLI Help；
不要新建 Notebook 或改用其他显卡。已经完成的特征不需重提。
采集前用 cal-only 验证序列、目标替换和批量数值一致性；禁止用 test/ARO
选择精度、提示、噪声或读出。适配器变化要先版本化并保留旧产物。

Show-o2 的正式版本是 FP32 计算、FP32 特征存储及数值等价的稳定输入池化。
旧 BF16 全量特征保存在 `calibration-archive/showo2-fp32-cal-revision-2/`。
修订工具 `revise_geometry_v5_showo_precision.py` 只允许未计分的 Show-o2
做该次限定修订；已有正式得分后不能重跑该修订。

主采集监督入口是 `script/selfless/launch_cross_model_geometry_v5_dev.sh`，
分为 `--stage-set f`、`--stage-set encoders`、`--stage-set flow`；flow 必须
显式 `--models janusflow,showo2`。16-rank 完成标记、精确参数加载与特征
审计才是完成证据，启动日志和进程退出本身不足够。

NPU 采集与数值检查结束后停止固定开发机，保留 Notebook 对象与所有数据。
真实平台停止响应和 STOPPED 状态记录在 `resource-cleanup.json` 及 logs/；
CPU 分析尚在运行时不能把计算资源清理误当作完整实验已经完成。

## 最终验收

重新运行单元测试与lint、检查全部正式PNG和中文报告，按原八项要求写出
完成审计。完成检查还要求当前Notebook为同一永久对象且已停止；可按当前
CLI Help，用禁代理的只读status命令更新 `logs/notebook-final-status.json`。
之后运行：

验收使用当前环境已有的 `pdfinfo` 解析全部PDF；它不是模型／拟合依赖，
也不会下载或改变模型环境。

```bash
.venv/bin/python scripts/audit_geometry_v5_completion.py
```

输出 `audits/completion-artifacts.json` 只证明记录的产物覆盖；不能替代
逐项科学解释复核。不要将旧阶段性WORKSTATE或进行中笔记当成当前终态。
