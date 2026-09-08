# V5 完成审计：逐项证据与解释边界

日期：2026-09-07。验收范围从用户目标、[V4 基准协议](B_GEOMETRY_PROTOCOL_V4.md)
和 [V5 的八项完成要求](CROSS_MODEL_GEOMETRY_PROTOCOL_V5.md)推导，不以已有
分数高低定义完成。核验对象是正式特征、原始分析、冻结映射、身份级统计、
当前图表和平台状态；启动记录、阶段性进度和单一 passed 标志不构成完整证明。

产物根目录：`output/evaluation/research/cross-model-geometry-v5-20260907`（下文相对路径均以此为根）。
交付说明见[中文结果报告](CROSS_MODEL_GEOMETRY_V5_RESULTS_20260907.md)、
[完整数值报告](../output/evaluation/research/cross-model-geometry-v5-20260907/RESULTS_ZH.md)、
[方法](CROSS_MODEL_GEOMETRY_V5_METHODS_20260907.md)和
[复现命令](CROSS_MODEL_GEOMETRY_V5_REPRODUCE.md)。

## 要求 1：全部模型、正确来源、冻结前向

结论：已满足。B 使用指定 x0content run 的 step95415 final EMA；F 使用
自己独立的 final EMA 与 `PositionwiseFlowOnBQwen3ForCausalLM`。DINOv2
ViT-B/14、MAE ViT-B/16 分别配原始 Qwen3-0.6B-Base，不借用 B 的初始化
诊断缓存。JanusFlow-1.3B、Show-o2-1.5B、SigLIP-so400m-patch14-384 均为
声明的官方权重，不混用其他大小、HQ 或新版本。

证据与实际核验范围：

- [模型来源及训练差异](../output/evaluation/research/cross-model-geometry-v5-20260907/audits/model-sources-and-training.json)：
  逐参数加载覆盖／值检查、实际模型配置、EMA 步数、参数范围、B/F 训练配置差异。
- [资产清单](../output/evaluation/research/cross-model-geometry-v5-20260907/asset-manifest.json)：
  官方 revision、下载字节数、源码快照及 public/models 路径。下载器显式使用
  空 ProxyHandler，curl 禁配置与环境代理；依赖下载也清空代理，正式前向离线。
- `audits/features-{b,f,qwen_text,dinov2,mae,siglip,janusflow,showo2}.json`
  覆盖八种实际特征来源。B/F 各480分片，Qwen/DINOv2/MAE 各48，SigLIP96，
  两种 flow 各352；包含全部主设置与预定扰动，逐分片检查身份、契约及有限值。
- 抽取代码使用冻结参数、eval 与 inference_mode；没有 optimizer 更新、新训练
  或重新训练连接器。拟合 PCA／正交 Q 是预定诊断，不是更新模型。

保留限制：原始 Qwen Base 是已有本地资产，没有可信的上游 revision 记录；
不事后补造。外部模型名称中的参数量不等于完整模型参数量；Show-o2 的共享
embedding／lm_head 存储不重复计数。历史下载的物理路由没有抓包证明，直连
依据是实际执行的禁代理下载实现与保留日志，不把清单中的字符串单独当证明。

## 要求 2：完整复用数据、身份和裁剪

结论：已满足。ImageNet 32000 图／12000 模板，COCO 11776 场景／58909
caption，ARO 868 图／1736 候选文本；顺序及分割均保留。

[样本与输入审计](../output/evaluation/research/cross-model-geometry-v5-20260907/audits/samples-and-preprocessing.json)
的代码逐行比较非 ARO V4 清单，重新推导600/200/200类别、512/8192/1024/2048
场景划分，检查同图 caption 归组、全部图像路径及 ARO 原裁剪注册表。
ARO 的868个原图身份重新由 VG→COCO 作者元数据推导，不是仅相信过滤清单；
其中451属性、417关系。所有外部文本都未发生截断，flow 不执行截断操作。

像素等价检查在每个图像数据集前32张，共96张，验证共同 Resize256／
CenterCrop256 输入实现；不是声称逐像素比对了所有原图。主数据的身份及
索引覆盖则是全量检查。共享内容裁剪后才适配各模型原生分辨率。

保留限制：身份互斥不等于内容去重或预训练数据去污染；COCO 也不对所有
外部模型都严格域外。V4 更宽松的2000/1924 ARO结果保留，但没有混入V5主表。

## 要求 3：全部实际层与读出

结论：已满足。[冻结比较契约](../output/evaluation/research/cross-model-geometry-v5-20260907/comparison-contract.json)
列明真实图／文层对与池化。每侧输入、各 block、最终 norm 均进入预定相对
深度网格；不是按测试成绩选层配对。14个路径／提示设置、41条读出曲线、
1171个层对×读出组合全部保留。

| 设置 | 层对×读出行数 | 固定／dev端点记录数 |
| --- | --- | --- |
| B native / bare / neutral | 各90 | 各120 |
| F native / bare / neutral | 各90 | 各120 |
| DINOv2＋Qwen / MAE＋Qwen | 各90 | 各120 |
| SigLIP 内容读出 | 58 | 80 |
| SigLIP 原生 pooler | 1 | 40 |
| JanusFlow 理解 / 生成 | 78 / 104 | 120 / 160 |
| Show-o2 理解 / 生成 | 90 / 120 | 120 / 160 |
| 合计 | 1171 | 1640 |

`audits/results-*.json` 不是只计行数：它对比全部预期文件名、层对、池化、
模式和维度组合。共同内容均值、末token、异构原生任务位置分开；没有 query
的模型不补零伪造。SigLIP 训练后的 pooler 单列。B/F query mask 分别为
文本151669、图像151672，主报告明确它们不是同一 token。

## 要求 4：理解／生成路径、上游语义与噪声

结论：已满足。`adapter-contracts/janusflow.json`、`showo2.json` 与两组
352分片来源审计互相核对。理解图像保存上游编码，生成两侧保存生成编码；
Show-o2 额外保存1152维语义路径、1536维低层路径和1536维融合输入。
理解文本没有视觉上游是正确的；其他应有分支不得用空字典通过。

主生成探针是 clean-image posterior mean/t=1 与 text/fixed-noise/t=0
两次独立前向，不将真图混入文本输入。所有分片首批替换相反模态目标，
全部层读出保持不变。noise seed 对所有语义身份共享，另有两个固定种子
与图像t=0.5；只在预定32类／512场景的鲁棒子集检查，不据其分数选主设置。

B/F sigma及后验均值控制、JanusFlow／Show-o2 的种子／时间控制共1680行，
全部读出、两域与固定／dev映射均保留。B/F sigma还移动原生query位置；
JanusFlow 前部内容看不到后部噪声，其不变性不能当非平凡鲁棒性。

另有明确标注的[48条事后残差分解](../output/evaluation/research/cross-model-geometry-v5-20260907/audits/perturbation-error-decomposition.json)：
从原始特征及冻结矩阵独立复算正式误差，拆分整体平移和中心化变形，没有
重定位预测或修改主成绩。它用于解释 Show-o2 的坐标失稳，不代替正式控制。

## 要求 5：关系结构、正交子空间及可识别性

结论：已满足。1171行均包含两域×欧氏／单位球，kNN@5/10/20、RSA、线性
CKA及199次身份置换。ImageNet原型1/3/5/10/15、a/b与模板参考，COCO
caption1/3/5及独立caption分组参考均保留；无效方差显式标记。

拟合实现复核了源域中心、PCA、每侧整体RMS和正交Q的来源；不逐轴白化。
32/128/512是共同预算，full仅允许原始宽度相同。结果保留PCA测试方差覆盖、
样本秩、跨协方差谱／秩／条件数和正交误差；“每侧满秩”不自动等于Q可识别。
`geometry_v5_math.py` 的独立单元测试覆盖旋转、不同宽度与不可识别情形。

[B−V4精确复现审计](../output/evaluation/research/cross-model-geometry-v5-20260907/audits/b-v4-parity.json)
核对270行、1080个几何bundle、63936个映射R²与1152个严格ARO正确率；
映射R²最大绝对差为0。V4的raw和三个初始化流程参考继续保留为历史基准，
不被冒充为新的独立训练种子或原始Qwen编码。

## 要求 6：留出、冻结迁移、拟合规模与困难负例

结论：已满足。每行映射覆盖ImageNet600、COCO512/2048/8192与四种维度，
两种几何合计32个记录。全部37472个记录含明确无效项，没有丢弃负R²。
有效映射保留fit/dev/test、跨域、打乱fit及整体尺度补充。

`audits/results-*.json` 实际载入冻结矩阵和身份级误差，检查源域、fit点数、
target_adaptation=false，以及测试身份顺序和误差重构；COCO映射测试用全2048
场景，不用512场景几何子集冒充。跨域不重估目标均值、PCA、RMS或Q。

ARO使用COCO8192映射，无任何ARO拟合／选层；属性／关系、余弦／距离、
配对fit／打乱fit／打乱图像均保留。不能仅靠高于50%或高CKA声称组合语义成功。

## 要求 7：源dev选择、配对不确定性与选层偏差

结论：已满足。[统计合同](../output/evaluation/research/cross-model-geometry-v5-20260907/statistics-contract.json)
先固定主对照与选择规则；每个预定读出、几何和fit规模各自只用源dev选择。
[汇总审计](../output/evaluation/research/cross-model-geometry-v5-20260907/audits/summary-tables.json)
逐值检查37472行CSV，重新推导全部328个dev选择、984个层搜索零分布检验。

[配对差值审计](../output/evaluation/research/cross-model-geometry-v5-20260907/audits/paired-model-differences.json)
用身份级数组重算1260组对照及全部2000-bootstrap区间，含16个预定主对照。
相同身份抽样同时用于两模型，各模型R²仍使用自己的分母。另有16组真实端点
串行／4-worker精确一致预检，不把这个小预检冒充完整1260组审计。

保留限制：区间条件于现有模型、源fit和dev选择，不包含重新训练、fit重抽样
或重新选层的不确定性；它们是逐项区间而非整个实验的同时区间。kNN区间
重采样query身份，固定候选图。没有用test、ARO或跨域成绩选择赢家。

## 要求 8：完整审计、图表、中文报告、停机保留

结论：实验与交付证据已齐备，最终机器覆盖检查由
`scripts/audit_geometry_v5_completion.py` 执行并生成 `audits/completion-artifacts.json`；
该检查失败时不得仅凭本段声明标记目标完成。

- 八组全层cal数值检查及实际16-NPU算术检查保留。局部BF16敏感性导致的
  SigLIP／JanusFlow FP32计算、Show-o2 FP32计算与存储修订均在其语义计分前
  根据cal决定，旧特征／契约／日志保留。B/F neutral局部警告未隐藏。
- V3/V4/V5目前34项单元测试通过；JUnit与lint日志保留。覆盖匹配几何、
  拟合与迁移、分组统计、配对分母、主对照分类和残差分解等，不把CPU单元
  测试当成NPU数值一致性的替代。
- [图表索引](../output/evaluation/research/cross-model-geometry-v5-20260907/figures/index.json)含25对
  正式PNG／PDF。主助手逐张打开25张PNG检查坐标、标签、域、读出、负值和
  区间显示，[目视记录](../output/evaluation/research/cross-model-geometry-v5-20260907/audits/visual-review.json)
  列出全清单。程序检查全部PNG解码，用pdfinfo解析全部25个PDF并验证单页、
  未加密、无解析警告；没有声称单独渲染审阅PDF。
- 主中文报告的40个主表数值单元与当前端点逐一核对。异常噪声解释、B/F
  主对照与原生读出差异、训练教师及因果边界均回到原始结果和模型论文检查。
  不用较强关系结构证明“head不做语义”，也不把外部预训练优势归因于架构。
- [资源收尾](../output/evaluation/research/cross-model-geometry-v5-20260907/resource-cleanup.json)含
  17个抽取阶段完成、实际RUNNING→停止响应→STOPPED，以及停止前无剩余
  本实验NPU进程的证据。之后又[直连实时重查](../output/evaluation/research/cross-model-geometry-v5-20260907/logs/notebook-final-status.json)
  同名、同created_at、同Workspace和910B资源对象，仍为STOPPED。
  遵循Inspire与固定Ascend技能，保留永久Notebook对象，没有重启或删除资产。

## 最终判断

原始七组模型、全部预定数据／路径／读出／统计范围没有缩减。实验结果允许
判断B有共享关系结构和部分低维对应，但没有普遍跨域同构或B独有结构的证据；
B相对F的优势具有读出、层和任务条件。这是实验的有效结论，不是完成失败。
预训练污染无法全排除、缺少独立训练重复、不能证明head因果分工，是报告
明确保留的解释边界，不以新增训练或未授权扩展替代本轮任务。
