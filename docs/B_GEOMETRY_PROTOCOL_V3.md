# B 跨模态几何实验 V3

日期：2026-09-07。本轮回答的是“文本和图像的语义关系能否在换一套坐标后保持一致”，而不是要求配对向量原本就接近。复用 [V2](B_SEMANTIC_EMERGENCE_V2_RESULTS_20260907.md) 已核验的逐层缓存，不新增模型前向、训练或开发机资源。

V2 结果已经被查看，因此 V3 是**有固定规则的后续探索**，不是独立、未触碰测试集上的确认性研究。机器可读协议在 V3 得分产生前写入输出目录的 `protocol.json`。

## 1. 三个强度不同的问题

1. **关系共享**：若图像里的猫靠近狗，文本里的猫是否也靠近狗？比较同一批类别/场景在两侧的距离关系和近邻身份，不要求图文向量直接接近。
2. **形状一致**：是否存在一个全局的正交变换 Q，使图像点云与文本点云对应？允许平移和整体单位换算，不允许任意逐轴拉伸。Q 允许旋转和反射；不是对每个类别单独拟合。
3. **跨域可复用**：只在 ImageNet 上拟合的变换能否直接应用于 COCO，反向是否也成立？不在目标域重估均值、PCA、尺度或 Q。

问题 1 的成功不推出问题 2，低维子空间的成功也不推出整个 1024 维表示等价。语义还包括属性、关系、组合与模态专有信息；本轮主实验只覆盖类别/自然场景层次。

这一分层借鉴 [Generalized Shape Metrics（NeurIPS 2021）](https://proceedings.neurips.cc/paper/2021/hash/252a3dbaeb32e7690242ad3b556e626b-Abstract.html) 的表征形状视角，以及 [Platonic Representation Hypothesis（ICML 2024）](https://phillipi.github.io/prh/) 中“比较两侧内部关系而不是直接坐标”的问题设定。CKA 采用 [Kornblith et al.（ICML 2019）](https://proceedings.mlr.press/v97/kornblith19a.html) 的线性中心核形式；本实验不把任何单一相似性分数当成完整语义的证明。

## 2. 模型、条件和读出

主模型是指定 run 的 `hf_model-final-ema`，step 95415。另测 final raw 及 init42/43/44 初始化流程参考。初始化参考包含预训练 Qwen backbone 和预训练 VAE，不是从零随机网络，也不是该 run 的真实历史 step 0；三个初始化种子不是三个训练重复。

- 每个输入记录输入层 0、28 个 block 输出与最终 RMSNorm，共 30 个位置。
- `content_mean`：已观察正文/图像 latent 的 X0 均值，排除提示、边界与目标。
- `query_native`：该条件的首目标 query。仅 `native` 条件是图像上下文的文本 query 对文本上下文的图像 query；`bare`、`neutral` 两侧均为文本 query。
- EMA 测 `bare` / `native` / `neutral`，raw 和三个初始化参考测 `bare` / `native`。条件语义及精确提示沿用 V2。

共 11 个状态/条件组合 × 30 个位置 × 2 种读出 = 660 条逐层记录。

## 3. 语义单位与数据隔离

先在原始特征上平均成语义单位，再做本轮的中心化、归一化和几何计算。不能先把每张图/每条 caption 分别投上单位球，再声称得到原始类别点云。

| 项目 | ImageNet | COCO |
| --- | --- | --- |
| 语义单位 | 一个类别：3 张图的均值，对 2 个类名模板的均值 | 一个场景：1 张图，对其全部 5/6 条 caption 的均值 |
| 拟合 | 600 类 | 500 场景 |
| 开发 | 200 个不同类别 | 500 个不同场景 |
| 测试 | 200 个不同类别 | 1000 个不同场景 |
| 校准均值 | 仅 600 个拟合类别的独立 cal 图/模板 | 独立 500 场景，caption 先按场景平均 |
| 无映射几何计算 | 全部 200 测试类别 | 固定随机选定的 500 测试场景 |
| 同模态重复参考 | 同 200 类的另外 3 张图/2 个模板，与测试视图不重叠 | 同场景前 2 条 caption 对剩余 caption；主缓存没有独立图像重复视图 |

ImageNet 每类原有 2 张 cal、3 张 fit、3 张 test 图及 6 个文本模板，V3 使用原有类别级 `mapping_split`（600/200/200）。开发类别用 fit 视图；测试类别用 test 视图，其 fit 视图仅作重复参考。

“未见类别”只指对齐映射未见过，不是 B 训练未见过。COCO 是相对于本次 B 的 ImageNet 图像训练的跨数据集评估，不声明排除了 LM/VAE 全部预训练污染。类别原型与复杂场景本身有粒度差异，跨域失败不能独立证明没有共享语义。

## 4. 中心化与两种几何

每个状态、条件、层和模态独立估计校准均值。

- 主分析 `centered_euclidean`：仅减模态校准均值，保留不同样本的长度差异。
- 补充 `centered_unit_sphere`：减均值后逐语义单位做 L2 归一化。这是另一种几何，不能与原始点云等距混称。

欧氏距离和中心化 CKA 本身对整体平移不敏感，因此在不做逐行归一化时，减模态常量均值不会改变这些无映射关系指标。球面分析则依赖中心点的位置。映射还会减去拟合集均值；欧氏模式下这一额外拟合平移吸收前面的校准均值，球面模式不具有这种抵消。

## 5. 不拟合跨模态映射的指标

每侧独立计算同一批语义单位的两两关系，再按对应身份比较：

- 线性 CKA：中心化 Gram 矩阵的归一化内积。
- 距离 Pearson：只用欧氏距离矩阵的非对角上三角。
- RSA Spearman：上述距离的平均并列秩相关。
- kNN 一致率：每个点在两侧的 k 个近邻身份交集占比，k = 5、10、20，排除自己；它比较“谁是邻居”，不是图文直接检索。

kNN 的随机期望是 k/(n−1)，另报告扣除机会水平后的值。每条曲线使用 199 次一致的语义身份置换：置换行列，不独立打乱距离矩阵元素。报告零分布均值、95% 分位和经验尾概率；最小可报告 p 为 0.005。

同一次置换跨层复用，另比较整条曲线的层间最大值与相应置换最大值。这个校正只针对一条固定曲线，不是整个实验所有条件/读出/指标的家族错误率控制。常量表示记为无效，不视为完美相似。

同模态重复几何是**参考**，不是严格的可达上界；不同重复视图的信息量和噪声不相等。

## 6. 在拟合集求解，在新类别/场景检验旋转

给定图像 X、文本 Y：

1. 按固定模式预处理，仅用拟合集求中心；可选各自独立的 PCA，固定维度 32、128，或者完整 1024 维。
2. 每侧用拟合点云的一个全局 RMS 半径换算单位，不逐轴白化，也不逐点缩放（球面补充除外）。
3. 对 XᵀY 做 SVD，求最小化 ||XQ−Y||² 的正交 Q。同样拟合一个打乱配对身份的 Q 作为容量匹配对照。
4. 额外报告在正确/打乱拟合配对上各自估计的一个全局尺度 s，即 similarity Procrustes；主结果仍用不额外收缩的正交版本。
5. 冻结所有参数，评估 fit、dev、test，以及另一个数据集的 test。

误差定义：`NRMSE = sqrt(Σ||XQ−Y||² / Σ||Y||²)`，Y 已减拟合中心；`R² = 1 − NRMSE²`。这里 R² 的基线是预测拟合均值，不是测试均值；域外尤其要保留这一区别。R² 的负数不截断；R²=0/NRMSE=1 表示与这个均值基线相当，而不是“随机旋转”。

全 1024 维旋转用 600/500 个居中点不能被完整识别；报告数值秩和有效维度，不声称求得唯一全空间 Q。PCA 在拟合例上独立学习，报告测试/域外保留方差：低维成功只能支持该子空间，低保留率不能推广为整体成功。若拟合数值秩不足指定维度则记无效。

按开发集正交 NRMSE 联合选层与维度，同分时优先较早层、再较低维。固定 final RMSNorm 另外做 2000 次类别/场景 bootstrap，报告 R² 与相对打乱拟合的优势区间；区间以当前拟合映射和原型样本为条件，不覆盖训练种子、映射拟合或原型采样全部不确定性。

## 7. 实现与运行

分析器：[analyze_unified_geometry_v3.py](../scripts/analyze_unified_geometry_v3.py)。自测：[test_geometry_v3.py](../tests/test_geometry_v3.py)，包含已知等距变换、任意拉伸负例、退化表示和测试数据不影响拟合参数的检查。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python scripts/analyze_unified_geometry_v3.py \
  --source-dir output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/semantic-v2-20260907 \
  --output-dir output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v3-20260907 \
  --state final_ema --profiles bare,native,neutral --workers 8 --threads 1
```

其余状态分别用 `--state final_raw/init42/init43/init44 --profiles bare,native`。并行运行状态时加 `--skip-summary`，总 worker × threads 按容器配额控制，本环境实际为 16 核，不是主机显示的 192 核。全部完成后单次 `--summarize-only` 汇总，`--audit-only` 检查完整性。已存在的逐层 JSON 可恢复跳过；固定协议不一致时拒绝覆盖。

后续定性展示用 `--neighbor-examples-only`，导出 native 的固定最终层及 query 开发集选出的第 13 层，全部 200 个测试类的邻居；不据这些例子重新选择层或修改主指标。绘图器 `scripts/plot_unified_geometry_v3.py --output-dir ...` 读取完整结果，需要 NumPy / Matplotlib。

本轮不据此因果断言“纯重建从零涌现全部语义”或“head 只负责采样”。B 的实际训练含预训练部件、配对条件、文本 CE 与图像 flow matching；需要另设训练对照及 head 干预才能回答因果分工问题。
