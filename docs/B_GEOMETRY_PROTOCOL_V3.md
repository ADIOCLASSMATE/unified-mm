# B 关系几何 V3 协议

复用 [V2](B_SEMANTIC_EMERGENCE_PROTOCOL_V2.md) 特征，在 CPU 上检验留出类别、场景和跨数据集的关系对应。结果见 [V3 报告](B_GEOMETRY_V3_RESULTS_20260907.md)。

## 样本与读出

| 数据 | fit / dev / test | 主几何候选集 |
| --- | --- | --- |
| ImageNet | 600 / 200 / 200 类；每类 3 图、2 模板构造原型 | 200 test 类 |
| COCO | 500 / 500 / 1000 图，另有 500 cal | 固定 500 test 场景；映射测试用全部 1000 图 |

ImageNet 的独立 3 图 / 2 模板视图用于重复参考；校准仅使用 fit 类。原型先平均原始特征，再处理几何。

状态和提示沿用 V2 的 11 个组合，读出为 Content mean / query，每组 30 层，共 660 条记录。初始化 seed 是项目初始化参考。

## 指标与拟合

分别报告中心化欧氏空间和中心化单位球空间。无拟合指标为线性 CKA、RSA、kNN@5/10/20 身份重合；kNN 排除自身，机会水平为 `k/(n−1)`。

映射只在 fit 估计两侧均值、独立 PCA、整体 RMS 单位及正交矩阵 Q：

`min ||XQ − Y||², QᵀQ = I`，`R² = 1 − ||X_test Q − Y_test||² / ||Y_test||²`。

Y 按 fit 均值居中；R²=0 对应预测 fit 均值，负数保留。维度为 32、128、完整 1024；不逐轴白化。full 的 fit 秩上限在 ImageNet / COCO 为 599 / 499。

源 dev 按最小 NRMSE 选择层和维度，平分时先选浅层、再选低维。跨域复用源域全部参数。另存打乱 fit 配对、整体尺度补充和 PCA 方差覆盖。

## 统计

199 次语义身份置换沿层共用，保存 max-over-layer 零分布；最小可达 p 值为 0.005。固定拟合与 dev 选择后，按测试语义单位做 2000 次 bootstrap；caption 随图归组。区间描述当前权重和拟合下的测试抽样。

## 复现

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python scripts/analyze_unified_geometry_v3.py \
  --source-dir output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/semantic-v2-20260907 \
  --output-dir output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v3-20260907 \
  --state final_ema --profiles bare,native,neutral --workers 8 --threads 1
```

raw 和 init42/43/44 使用 `--profiles bare,native`。执行分片用 `--skip-summary`，全部完成后用 `--summarize-only` 汇总；`--audit-only` 检查完整性，`--neighbor-examples-only` 导出邻居示例。

绘图入口为 [plot_unified_geometry_v3.py](../scripts/plot_unified_geometry_v3.py)。
