# V5 复现入口

从仓库根目录执行。产物根目录为 `output/evaluation/research/cross-model-geometry-v5-20260907/`；现有特征足以完成 CPU 分析。

## CPU 分析和绘图

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

`--settings` 限定设置，`--pair-indices` 分配层对分片；分片使用 `--no-summary`，最后执行 `analyze_geometry_v5.py --summarize-only`。同一分片及共享汇总各保留一个写入进程，已有完整 JSON 会复用。

## 检查

```bash
.venv/bin/python scripts/audit_geometry_v5_samples.py
.venv/bin/python scripts/audit_geometry_v5_parity.py
.venv/bin/python scripts/audit_geometry_v5_model_sources.py
.venv/bin/python scripts/audit_geometry_v5_features.py --workers 2
.venv/bin/python -m pytest tests/test_geometry_v3.py tests/test_geometry_v4.py \
  tests/test_geometry_v5.py -q \
  --junitxml=output/evaluation/research/cross-model-geometry-v5-20260907/audits/tests-v3-v4-v5.xml
```

配对统计与检查可用 `--workers 1` 串行重算。正式图表位于 `figures/`，索引为 `figures/index.json`；数值报告为 `RESULTS_ZH.md`。

## 依赖

权重 revision、源码和字节数见 `asset-manifest.json` 与 `audits/model-sources-and-training.json`。绘图依赖放在独立 overlay：

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

模型适配器依赖为 `scripts/geometry_v5_flow_requirements.txt` 和 `scripts/geometry_v5_common_requirements.txt`。

## NPU 特征采集

在 [INSPIRE.md](../INSPIRE.md) 指定的 `dev-wjx-ascend` 上运行 `script/selfless/launch_cross_model_geometry_v5_dev.sh`。阶段为 `--stage-set f`、`encoders`、`flow`；flow 显式指定 `--models janusflow,showo2`。

先完成 cal 的序列、目标替换和批量一致性检查，再冻结适配器契约并采集。正式 Show-o2 使用 FP32 计算与存储，数值设置见 [方法](CROSS_MODEL_GEOMETRY_V5_METHODS_20260907.md)。每个阶段保存 16-rank 完成记录、参数加载检查和特征检查；采集结束停止开发机，保留对象与产物。

## 产物检查

```bash
.venv/bin/python scripts/audit_geometry_v5_completion.py
```

该命令检查正式产物覆盖，使用已有 `pdfinfo` 解析 PDF。数量与证据入口见 [产物清单](CROSS_MODEL_GEOMETRY_V5_COMPLETION_AUDIT_20260907.md)。
