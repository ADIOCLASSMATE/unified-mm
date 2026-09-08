# B 的输入 embedding 与跨模态语义表征实测

日期：2026-09-07。结论来自本次开发机实测，不是依据模型结构推断。

B 的 backbone 后段已经出现明显的、可以跨模态读取的语义方向；本次没有观察到输入 embedding 能直接构成有效的图文余弦检索空间。结果支持部分共享语义表征，但不能证明所有语义计算都在 backbone、flow head 只负责采样。

## 权重、设备与产物

- 权重：`output/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/hf_model-final-ema`。
- `ema_export_metadata.json` 确认来源为最终 step **95,415**，FP32 EMA 导出，487 个 state keys / 486 个 safetensors 实体键；差异为 tied embeddings。
- 所有模型前向在 `dev-wjx-ascend` 的 16 张 Ascend 910B 上执行，使用 BF16；PyTorch `2.6.0+cpu`，torch-npu `2.6.0.post5`。统计分析在 CPU 上完成。
- 模型权重全部冻结；没有修改训练代码、checkpoint 或正式评测协议。线性探针及映射只属于诊断分析。
- 开发机最初为 STOPPED；本次启动、采集后停止，并重新查询确认 STOPPED。
- 产物目录：[representation-diagnostic/step-95415-ema-20260907](../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/step-95415-ema-20260907)。

## 预先固定的实验

Flickr30K 使用完整 Karpathy test：1,000 张图和 5,000 条描述，每张图的五条描述均为正例。主检索实验始终使用完整候选集。额外线性映射实验才按图划分 800 张拟合、200 张测试；同图的五条描述不跨集合。这是新的表征诊断，不是仓库既有的生成似然检索协议。

ImageNet 覆盖全部 1,000 类，从官方 val 每类固定随机抽取 10 张，共 10,000 张：6 张用于图像探针拟合，4 张用于测试。文本侧每类使用八个预先固定模板，六个拟合、两个测试。类别名称来自项目已有类名资产，没有加载 CLIP 编码器。探针的类别标签来自评测数据；不能将其准确率称为原生生成式零样本 benchmark。

两种模态分别独立前向。共有前缀为 `Describe this image in one detailed caption:`；输入图像或文本后追加共有后缀 `\nThe main subject is`，再加入一个不允许 query 读取的占位 token。图像前向不含配对描述、类名或类别标签。位置与 attention 保留 B 的 random image sigma、严格 query mask 和包含对角线的 content mask。

逐层采集两个读出：

1. **X0 content mean**：只平均图像 latent token 或描述正文 token，排除前后缀、特殊 token、padding。
2. **XT shared query**：取共有后缀后、可访问完整输入的固定文本 query。

记录入口、28 个 backbone block 输出、最终训练得到的 RMSNorm 输出，共 30 个位置、两种读出。入口 query 是相同的 mask embedding，作为无样本信息的控制。不同读出的上下文覆盖范围不同，因此它们的差距不能单独归因为两条流的优劣。

所有探针统一先做逐样本 L2 归一化，用闭式 ridge、固定相对正则强度 0.1 拟合，没有使用测试准确率选择超参数。原始跨模态迁移使用同一个分类器，不适配目标模态；centered 版本另用目标模态拟合集的无标签均值修正偏移。完整检索的 centered 版本使用候选池的模态均值，属于无标签但依赖测试分布的处理；200 张测试的映射实验只使用拟合集均值。

## 输入层：平均与局部匹配均接近随机

Flickr30K 完整候选集 R@1，单位为百分比：

| 输入层评分 | 图 → 文 | 文 → 图 |
| --- | ---: | ---: |
| 随机期望 | 0.10 | 0.10 |
| 输入 embedding 平均后原始余弦 | 0.00 | 0.04 |
| 输入 embedding 平均后按模态中心化 | 0.40 | 0.14 |
| 文本 token → 最相似图像 patch，随后平均 | 0.00 | 0.08 |
| 随机 image projector 的同一局部匹配控制 | 0.10 | 0.08 |

局部匹配额外使用全部 4,768 个实际出现的文本 subword token，计算 `mean_text_token max_image_patch cos(Wz+b, E[token])`，没有经过 backbone、拟合读出或过滤停用词。该项以保存的 FP32 EMA 权重和相同后验样本在 CPU 上计算；利用 16 通道分解加速，并对照显式 1024 维余弦验证数值一致，因此与 BF16 前向的低层数值口径略有区别。

这里的结论是“没有看到可直接使用的跨模态语义距离”，不是“输入不含任何语义”。输入图像平均 embedding 的图像域内线性分类仍有 **2.275%**，高于 0.1% 随机值；VAE 的视觉信息已经存在。`1024 × 16` projector 的 16 个奇异值都非零，约为 0.105–0.327，也没有投影塌缩的证据。每个图像 token 的低维线性空间不能等同于整张图只有 16 维信息。

## Backbone 后段：跨模态关联明显增强

预先固定的最终 RMSNorm 后 query，Flickr30K 完整候选集：

| 评分 | 图 → 文 R@1 | 文 → 图 R@1 |
| --- | ---: | ---: |
| 原始余弦 | **10.60%** | **2.66%** |
| 仅按模态中心化 | **17.90%** | **12.70%** |

固定最终 query 的原始图→文 R@1，按图分组 bootstrap 95% 区间为 **8.70%–12.50%**；中心化后为 **15.60%–20.30%**。Bootstrap 保留固定候选池与特征均值，并把同图五条描述作为一组，因此反映固定候选池下的 query 抽样不确定性。

层扫描中，第 27 个 block 后 query 的中心化 R@1 为 **26.40% / 17.36%**，原始余弦为 **12.60% / 4.02%**。这些中间层数字是探索性结果，不能当作通过独立验证集选择后的最终性能。完整 60 组结果全部保留，没有只保留最优层。

均值方向对距离影响很大。例如第 27 层 query 的配对/非配对平均原始余弦为 0.9588 / 0.9466；中心化后为 0.3129 / 约 0。这说明直接余弦受到强公共方向及模态均值的影响。中心化提升本身不需要额外学习配对映射。

ImageNet-1K 固定最终 query 的线性分类，单位为百分比：

| 分类器拟合域 → 测试域 | 原始跨域使用 | 额外修正目标模态均值 |
| --- | ---: | ---: |
| 文本 → 图像 | **25.80** | **31.40** |
| 图像 → 文本 | **41.75** | **51.85** |
| 图像 → 留出的图像 | **52.05** | — |
| 文本 → 留出的文本模板 | **100.00** | — |

对应输入层文本→图像为 **0.10%**。文本域的 100% 主要证明类名在不同模板下易于线性辨识，并不是语言理解 benchmark。更关键的是：只在文本表征上拟合的分类权重，原样用于图像表征，仍得到 25.80%；这支持两种输入在 backbone 后段拥有部分共享的可读语义方向。

800/200 图像分组的额外线性映射实验，在最终 query 上拟合图→文映射后，200 张独立测试图的 R@1 为 **26.50% / 27.10%**；打乱拟合图文配对的同容量控制为 **0.00% / 0.60%**，随机期望为 0.50%。但同一小候选集上不拟合映射、只中心化已经有 **35.00% / 25.20%**。因此不能声称拟合映射必然优于中心化，也不能把这个 200 图候选集与完整 1,000 图结果直接相减。该结果用于检查跨模态关系能否泛化及打乱配对控制是否失效。

![逐层曲线](../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/step-95415-ema-20260907/layer_curves.png)

## 对原假设的证据边界

- **支持**：当前 B 的 backbone 后段有明显跨模态语义关联；同一线性读出方向可以部分跨模态复用。
- **本次未支持**：文本输入 embedding 与图像 latent 过 linear 后，已经自然成为实用的图文余弦语义空间。平均与局部最大匹配均接近随机，但不能据此排除所有低层关联或其他非线性读出。
- **尚未验证**：所有语义计算在 backbone、flow head 仅负责采样。这里使用的共同 query 是文本读出位置，没有对 flow head 的图像 query 接口做等价性检验，也没有干预 head 或对其职责进行因果消融。
- **不能归因为从零涌现**：B 初始化自 Qwen3-0.6B-Base，输入来自预训练 VAE。本次只检查最终 EMA，没有同协议的原始初始化/早期 checkpoint 曲线；层间增长不是训练时间上的增长。随机 projector 控制也不是精确的 step-zero 模型。
- **任务范围有限**：使用单一固定文本读出提示、一个后验样本与一个 image sigma seed，类别探针衡量对象类别，不能覆盖全部属性、关系、组合推理或不同提示下的稳定性。

## 完整性、验证与复现

16 个 rank、四个数据集的样本索引均完整且无重复；64 组隐藏目标替换检查逐元素通过。每批末层 hook query 与模型实际输出逐元素相同；所有特征有限。线性拟合只在指定的拟合集进行。三项 CPU 测试验证多正例检索、塌缩表征不能报告完美检索、以及线性映射对独立合成样本的恢复。相关文件通过 Ruff 与 shell 语法检查。

数据与所有分析口径见产物中的 `protocol.json`、`samples.json`；完整指标为 `results.json`、`layer_metrics.csv`，低层补充与区间为 `input_token_geometry.json`，NPU 证据为 `extraction.log` 和各 rank 完成记录。曲线同时提供 PNG 与 PDF。Bootstrap 在零命中时会退化，不应用其 `[0,0]` 宣称总体错误概率为零；I2T 另提供 Wilson 区间。

源码入口：

- [样本准备与 NPU 提取](../scripts/probe_unified_representations.py)
- [开发机 16 卡启动脚本](../script/selfless/probe_unified_representations_dev_ascend16.sh)
- [CPU 逐层分析](../scripts/analyze_unified_representations.py)
- [输入 token 局部匹配与不确定性](../scripts/check_unified_input_geometry.py)
- [指标正确性测试](../tests/test_representation_diagnostics.py)

在仓库根目录使用一个新的产物路径，避免覆盖本次固定样本协议：

```bash
MODEL=output/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/hf_model-final-ema
REPR_OUTPUT=output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/new-run

TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=. .venv/bin/python \
  scripts/probe_unified_representations.py prepare \
  --model-source "$MODEL" --output-dir "$REPR_OUTPUT"

# 仅在已确认 RUNNING 的 dev-wjx-ascend 内执行：
bash script/selfless/probe_unified_representations_dev_ascend16.sh "$MODEL" "$REPR_OUTPUT"

# CPU；绘图需要 matplotlib。本次将绘图依赖装在临时目录，未修改项目环境或锁文件。
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=. .venv/bin/python \
  scripts/analyze_unified_representations.py --output-dir "$REPR_OUTPUT"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=. .venv/bin/python \
  scripts/check_unified_input_geometry.py --output-dir "$REPR_OUTPUT"
```

平台会话初次请求遇到代理环境影响；仅对本次 Inspire 命令取消大小写 HTTP(S)/ALL_PROXY 变量后恢复。Notebook 的 CANN 环境通过 `set_env.sh` 加载，启动脚本保留其 `PYTHONPATH`。没有修改全局代理配置、平台账号配置或训练进程。
