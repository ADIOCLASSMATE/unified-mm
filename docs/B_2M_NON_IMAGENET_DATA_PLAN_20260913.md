**B 大规模训练的数据选择：ImageNet 之外至少 200 万张，配额为起点，优先复用文本（2026-09-13）**

执行入口与运行合同已统一到 [DATA_SYNTHESIS.md](DATA_SYNTHESIS.md) 和 [b512_sii_v1.json](../configs/data_synthesis/b512_sii_v1.json)。本文维护来源选择及覆盖目标；完整下载、512px 验收与资格报告仍是正式批量合成的前置条件。

确定的目标是：**完整 ImageNet train 底座 + 至少 2,000,000 张非 ImageNet、去重且通过 512px 验收的原图**。下表是起始覆盖目标，各来源和总图池均不设 200 万表内配额对应的硬上限；有价值的额外合格原图继续保留。按 ImageNet 原始 train 规模计算，200 万基线对应 3,281,167 张，最终报告额外合格项和底座损坏/排除项。已有 105,322 张非 ImageNet 首轮产物包含在该目标内，符合新方案和质量要求的图像、文本可以复用。不同 caption、不同裁剪和重复 epoch 均不增加原图数。ClimbMix 不变。

这是训练前的来源与配额计划，尚不是已发布的 200 万训练集，也不是经过下游实验验证的最优比例。当前先确定非 ImageNet 部分，ImageNet 全量 512px 扩展另行接续。机器可读协议见 [unified_b_non_imagenet_2m_v1.json](../configs/protocols/unified_b_non_imagenet_2m_v1.json)。

**推荐以原配额为起点，调整三个获取入口**

| 最终归属桶 | 起始合格原图目标，非上限 | 获取与筛选决定 | 文本优先级 |
| --- | ---: | --- | --- |
| PixMo-Cap | 500,000 | 扫描完整 train 元数据，分层选多主体/动作/属性组合；先扣除分给专项桶的原图 | 原 caption 优先，可同文用于 I2T/T2I |
| Open Images 关系 + Localized Narratives | 300,000 | 建议约 10 万对象间关系图 + 20 万叙述场景图，限制常见关系，保留尾部关系 | 先按原图 ID 匹配人工叙述和已有文本，缺失再补 |
| PixMo-Points | 180,000 | 按数量、可辨认性、实例位置分层，保留完整人工指代/点标注 | 与 Cap 等按原图连接 caption；准确点/计数可构造聚焦描述 |
| AnyWord-3M | 350,000 | 中文约 20 万、英文约 15 万；文字载体和布局分层 | 原 caption + OCR/区域，优先核查 AnyText2 更新标注 |
| JourneyDB train 风格池 | 300,000 | 保留条件目标；入口/原始 train 映射不成立时，由 MONET synthetic 同风格补位 | 原 prompt/描述作为候选，核对已实现的内容 |
| WikiArt | 70,000 | 全 72 分片作为候选；风格、题材、作者平衡 | 复用现成合格文本；单有类别标签的画作需要看图补描述 |
| BLIP3o-60k + ShareGPT-4o-Image T2I | 80,000 | 建议 4.5 万 + 3.5 万；按动作/文字/组合筛选，只取 T2I | 原配对优先，错误或未实现的要求才修订 |
| TextCaps + DOCCI train | 30,000 | 尽量完整保留合格 train；这组原图先于通用 OI/OCR 桶分配 | 人工 caption 优先，可补接现有 TextOCR-GPT4V |
| PixMo-Docs / ChartQA train | 40,000 | 优先 DIM 已提供 caption 的候选；约 3 万简单图表/表格/图解 + 1 万 ChartQA | 复用 DIM caption；缺失时再用结构化源数据补齐 |
| 通用真实场景 | 150,000 | **MONET 的 CC12M 分支 10 万 + PD3M 5 万起步**；前者获取不顺时优先用 LLaVA-ReCap-CC3M | 已重描述 caption 优先，避免从原始 CC12M 全部 URL 重做 |
| **起始目标合计** | **2,000,000，可超过** | **按原图互斥归属，各桶保留多种能力标签** | **不要求每张图重新生成两条文本** |

三个入口调整是：CC12M 优先使用已筛选/重描述的分支；图表优先匹配现成 caption；JourneyDB 明确准备同能力替代来源。其余沿用用户建议。45 万风格/生成组合、35 万文字、4 万结构化图形是来源预算；“关系 + 计数”48 万中包含叙述场景，不能把全部 48 万称为已验证的关系/计数真值。

**已找到的精选与可复用资源**

- **[MONET](https://huggingface.co/datasets/jasperai/monet)**：发布者已经完成图像筛选、近重复处理和多模型重描述，并提供来源、哈希、聚类及检测字段。当前文件树有独立 `v1.2.0/cc12m/` 和 `v1.2.0/synthetic/` 分支，可以只选相关候选分片，不必处理整个约一亿图仓库。原始 CC12M 的 10 万预算优先从此获取；这仍需要针对 B 的主题分层。**Parquet 中是 384px 缩略图，训练必须取完整图像；已存 SANA latent 不能复用为本项目 KL16。** 读取必要元数据列即可，避免为了 10 万图下载全量 embedding/latent。
- **[DIM-T2I 的按来源 caption 文件](https://github.com/showlab/DIM/blob/main/data/T2I_DATASET.md)**：提供 PixMo-Docs 109,142、ChartQA 15,389 条 caption 候选，还有 TextOCR-GPT4V 等对应文本。建议只获取相关文件，并按 `image_path` 与上游 ID 连接、再次限定 train。它发布的是文本和匹配键，**不会额外带来一份原图**；超长 caption 也不能直接塞进当前序列。其 caption 发布协议为 CC BY-NC 4.0，源图沿用各自协议。
- **[PD3M](https://huggingface.co/datasets/Spawning/PD3M)**：确实是 PD12M 的审美精选子集，约 330 万图文对，保留 5 万补题材有价值。其精选目标偏审美；关系、拥挤计数等仍应在专项桶保留，不按统一审美阈值淘汰。
- **[TextOCR-GPT4V](https://huggingface.co/datasets/jimmycarter/textocr-gpt4v)**：已有 `caption_image`、`caption_text`、`caption_condensed`，可为现有 TextOCR 图片补充可复用文字。只获取 caption 元数据再连接本地图像，不重复下载同一批图；其 HF `train` 包装不替代原始 TextOCR split，OCR 结果仍可能错误。
- **[BLIP3o-60k](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k)** 和 **[ShareGPT-4o-Image](https://github.com/FreedomIntelligence/ShareGPT-4o-Image)**：已有生成图与 prompt，适合复用组合、文字和设计监督。ShareGPT 的 T2I 为 45,717 条；46,539 条编辑样本依赖输入图，不计入这次单图 T2I 配额。BLIP3o 的类别包括 JourneyDB、人物、常见物体、文字等，须检查测试 prompt 和实际 tar 成员，不能把 viewer 展示的 7,103 文本行当全部图像。
- **[LLaVA-ReCap-CC3M](https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-CC3M)**：约 286 万图像已有详细描述，是通用预算的省事备用入口。它是重描述语料，未证明每张都属于 B 的难例，不直接用最前面的 N 条充当精选。
- **[CoSyn-400K](https://huggingface.co/datasets/allenai/CoSyn-400K)**：PixMo-Docs 作者推荐的更新版本，含 `data`、`code`、QA 和图像。适合作为缺失结构化图形的备用；为了尽量复用文本，本轮仍优先使用已有 DIM caption 的候选。QA、代码和完整 caption 是不同字段。

[Recap-DataComp-1B](https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B) 暂不设配额。当前 `default` 与 `condition_diverse_topk` 均列出 940,890,257 条 train 数据；`topk` 名称不表示一个已经筛好的小型图像训练集，1,000 条 preview 也不能当专项精选。未核实到官方针对这些 corner cases 的小精选，不为 15 万通用预算处理其全库。

**需要在冻结来源数前解决的供给约束**

1. **Cap / Points 已完成全量 URL 并集统计，内容去重仍待图片下载后执行。** PixMo-Cap 的 717,042 条标注对应 716,551 个唯一 URL；Points 的 2,376,222 条对应 228,080 个。两者仅有 181 个完全相同 URL，URL 并集为 944,450，对 68 万起始目标有候选余量。但 URL 不同也可能是同图不同尺寸/转载，不能当作 944,450 张验收原图。按完全相同 URL 只能给 181 张 Points 候选接到 Cap caption，**不能假设大部分 Points 已有 Cap 描述**；其余仍需其他同原图文本、可靠标注转写或看图补标。超过 50 万 / 18 万的优质项可以保留。[本地逐分片统计与版本](../public/data_preparation/unified_b_corners_api_v3/quota_2m_20260913/pixmo_url_union_audit.json)、[PixMo-Cap](https://huggingface.co/datasets/allenai/pixmo-cap)、[PixMo-Points](https://huggingface.co/datasets/allenai/pixmo-points)
2. **Open Images 的属性与对象间关系分开计数。** 本地完整关系 CSV 实数为 568,539 张原图；排除 `RelationshipLabel=is` 的属性行后，含对象间关系的原图为 126,368 张。约 10 万关系图目标已有合理余量，其余由 Open Images 的 [Localized Narratives](https://google.github.io/localized-narratives/) train 补充，不能用全套叙述中的 COCO/Flickr 数据替换。
3. **TextCaps + DOCCI 的 3 万接近来源上限。** TextCaps train 21,953 与本地逐行核实的 DOCCI train 9,647 合计 31,600 张，去重/512px 可读性检查后可能不足。TextCaps、TextOCR、Open Images 共享原图时只占一个名额。缺口应记录，并从同类文字/详细描述候选补齐；不能多裁几张凑数。[TextCaps 论文](https://arxiv.org/abs/2003.12462)、[DOCCI](https://google.github.io/docci/)
4. **JourneyDB 30 万仍是条件池。** 官方 HF 仓库当前要求访问条件；另外找到了公开的 [BLIP3o-Pretrain-JourneyDB](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-JourneyDB) 分片，但它是重新打包的预训练库，预览可见同 prompt 多个变体，原始 train 映射还没有核实。不能称它为已经验收的 30 万精选，也不应先下载其约 3.13 TB 全库。若无法落实，使用 MONET synthetic 的同风格候选补位，并保存实际来源配额差异。

补充检查已按 Flickr photo ID、Pinterest 图像标识、Reddit/Imgur 静态图片 ID 等做保守 URL 归一化，Cap / Points 的交集仅增加到 189 个来源键；跨站转载仍须内容去重。Cap 本身还有 10,354 个 `ai-gen/` 路径候选，其实际风格和原 caption 可以一并利用，不额外重复计数。[来源键与域名统计](../public/data_preparation/unified_b_corners_api_v3/quota_2m_20260913/pixmo_conservative_source_keys.json)

访问探测中，Open Images Localized Narratives 元数据、MONET 示例分片及 BLIP3o JourneyDB 重打包均返回有效的直连 Range 响应；旧官方 TextCaps train JSON URL 返回 403，仍需落实可用的官方元数据入口或有原图映射的镜像。入口可访问不代表完整原图、train 归属或质量已验收。[直连探测记录](../public/data_preparation/unified_b_corners_api_v3/quota_2m_20260913/source_discovery/direct_access_probes.json)

**配额不是上限。** 非 ImageNet 以至少 200 万合格原图为规模目标；各来源表内数字仅作为起始覆盖参考。TextCaps、DOCCI、WikiArt、BLIP3o-60k、ShareGPT T2I 等有限来源优先保留全部合格 train，超过表内数也继续保留；Points 等稀缺标注图和带可靠 caption 的优质图同样可以追加。大库仍按能力缺口与下载/处理成本选择候选范围。

实际不足时用同能力候选补位并记录变更，保持文字/风格/关系/计数覆盖。如果同能力候选也不足，应明确报告缺额，不能降低质量门槛或用普通单主体图填平数字。**图池保留量和训练抽样分布分别确定**：保存实际合格原图及全部可用标注；训练按能力/来源采样权重控制曝光，避免某个易获取的大来源随着入库增长主导训练。训练任务权重与来源权重另行冻结，不把磁盘百分比当 token 或 loss 百分比。

**复用协议：一条好 caption 可以服务两个方向**

先建立 `原图身份 → 全部现有 caption / prompt / 结构化标注` 的索引，再处理 512px 视图。默认使用同一条忠实 caption 作为 I2T 答案和 T2I 条件；loader 自行添加任务前缀。已有多条合格人工描述可保存，但不同文本不会增加图片权重，也不强制每图多生成同义句。

按以下顺序处理每个文本候选：

| 路径 | 处理 | 是否调用大模型 |
| --- | --- | --- |
| 原样复用 | 原图匹配、最终视图内容仍可见、文字在预算内 | 否 |
| 规则整理 | 清理格式/生成参数；选择完整且自足的句子，保留原文与变换记录 | 否 |
| 已验证标注转写 | 将准确的 referent + count、可读文字串与区域关系转成聚焦正描述，标记为 annotation-derived | 通常否 |
| 缺失/错误修复 | 缺 caption、事实冲突、关键约束遗漏，或无法安全压缩的长描述 | 仅此队列调用通过前置验收的 SII 模型 |

不使用“最少几十词”排掉准确的短 caption，也不要求 I2T 与 T2I 两条字符串必须不同。上限仍按当前 tokenizer 检查，单文本保守控制在 960 tokens 内。不能截断半句、拆散指代关系，或为了顺畅增添新事实。

数量预算上，**Points 18 万 + 关系图约 10 万 + WikiArt 7 万，共约 35 万图所在的来源通常没有现成完整 caption**，应优先检查跨源可复用文本。这个数不是实际 API 调用量或上限：同图 caption 连接会降低需求，AnyWord/生成 prompt 的错误修复又可能增加需求。当前没有逐图复用审核结果，因此不承诺“复用率 80%”或固定 token 节省比例。

已有首轮合格 sol 文本可以按原图和 view SHA256 精确复用，不再重新调用 sol 复核。原始作者、源 caption、采用字段、文字哈希、处理方式及模型版本分别保留。近重复图的 caption 不自动跨图复制；仅同一原图身份/内容匹配可以直接连接。

**后续真正需要合成的 corner-case 文本**

- **I2T 关系与属性归属**：描述中补上容易交换的主语、宾语和各自属性；只从图像或可靠关系标注取事实。二维框可支持部分左右/上下关系，不能凭框猜前后、接触或动作意图。
- **I2T 计数与指代**：精确保留被计数的 referent、范围及数量。`三个人站着` 不等于 `全图只有三个人`；点坐标本身不提供对象尺寸和完整可见性。0 数量/否定只在有充分覆盖证据时加入。
- **I2T 文字与位置**：把文字串绑定到牌子、包装或具体区域；小字缩小后不可读就删除相应确定转写或淘汰此专项候选。无需把中文画面文字翻译成英文。
- **T2I 组合约束**：补真实存在的数量、颜色归属、左右位置和动作关系，减少泛泛描述。原 prompt 要求五个而图片只有四个时，以真实图像为准修复，不能继续当“五个”的正样本。
- **T2I 风格**：WikiArt 补实际媒介、笔触和构图；生成图库删除未实现的风格/内容要求。不能让原照片继续配水彩/3D 改写文本。
- **简单图表/图解**：在可读的最终视图中，以表格、渲染源和标注为核对依据形成描述。保留 QA 作为将来的指令数据，不声称普通 caption 训练已经覆盖 VQA 或数值推理。

本轮不额外生成目标图片，也不把错误 caption/反事实 prompt 当作普通 I2T 正目标训练。若将来增加负例排序、QA 或多图任务，应独立定义 loss 和 serializer。

**冻结与执行顺序**

1. **元数据与复用盘点**：先读完整来源索引和相关 caption 文件，连接已有 105,322 张非 ImageNet 产物；检查 Cap/Points/OCR/OI 的交集。为稀缺人工标注、计数和 OCR 先分配身份。记录原始 train、caption 版本和可追溯的 parent ID。
2. **带余量的候选清单**：按能力桶冻结候选及替补序列，范围以明确 ID/分片列出。候选量根据各来源实际成功率计算，不能把初始候选 URL 数直接当最终配额。对有限小库以全部 train 为候选；大库跨主题/分片选择，避免取开头 N 行。
3. **完成下载和预处理**：下载与 CPU 512px 预处理可独立并行，但批量 API 暂不启动。复用本地原图；直连客户端禁用代理环境，保留 Codex 的全局代理连接。原图、caption、embedding 分开按需获取，不整库下载 MONET/Recap。
4. **验收至少 200 万原图，保留有价值的额外合格项**：排除 ImageNet 原图重叠、B 评测身份、精确/视觉近重复、测试 prompt、损坏图；专项按 512px 可见性验收。关系/计数/OCR 优先全幅保留，确需裁剪时同步变换全部标注，所有文本以最终视图为准。缺额按同能力替补补下载；完成本轮选定候选后，按实际数量冻结原图池，不在某来源刚达到配额时丢弃余下合格图片。
5. **文本复用与定点修复**：先出复用/规则整理/标注转写清单，再给真正缺失的样本调用 API。由 GPT-6 做一次分层前置图文评审，锁定合成模型及提示；后续常规流程不逐图调用 GPT-5.6-sol 生成或审核；仅当同一图的 SII 尝试达到持久化失败上限，才允许现有 Codex CLI 以 sol low 作有界最终兜底。并发从 16、32、64 逐级实测，可稳定再升 128；按错误率、延迟和服务限额调整，保存断点、原始响应和隔离队列。
6. **发布并接训练配置**：原图总数、各来源实际配额、I2T/T2I 文本可用数、复用/新增数、拒绝原因、哈希及 loader 全量验收分别报告。后验缓存只按最终视图生成一次，不因两个任务重复编码。

当前 SII 实测已经确认：`qwen3.8-max` 在同一个 SII URL/key 的 OpenAI 格式接口上能描述真实图像；但成功样本是原图 URL 和缩略图诊断，**尚未完成完整 512px 成对输出的分层质量验收**。`deepseek-v4-pro-0813` 在多个有效附图请求中返回无法看到图片，暂不作为视觉教师。它能否用于已有可靠事实的纯文本整理，应单独测试；规则能处理的情况不必调用 API。不会将 HTTP 200 或 JSON 合法率当作图文质量通过率。

512px 使用 1,024 个图像 token、16 个 latent 通道、图像分支序列 2,048；mean/std 缓存存储 32 通道。200 万图的 fp16 mean/std 纯张量约 131 GB，这是计算估计，不含原图、预处理图、索引及文件开销。MONET 自带的其他 VAE 缓存不进入该缓存。

**当前产物与状态**

- [推荐配额与复用策略 JSON](../configs/protocols/unified_b_non_imagenet_2m_v1.json)
- 来源版本/文件目录和直连探测：`public/data_preparation/unified_b_corners_api_v3/quota_2m_20260913/source_discovery/`
- 图像目录：`public/datasets/unified_image_pool_512_v3/`；预计独立文本目录：`public/datasets/unified_image_text_512_api_v3/`，尚未发布新的 200 万文本集。
- 现有可复用首轮 release：`public/datasets/unified_image_text_512_sol_v1/releases/sol_100k_plus_corners_v1/`。
- 上一版“8 来源全文件下载”任务已暂停并保留断点。暂停时校验完成 22/121 个文件、约 3.115 GB，其中包含已有文件复用；本次进程实际新下载约 0.551 GB。该清单被本配额方案取代，不作为 200 万图完成证明。
- B 大模型训练未启动；现有评测分数仍属于旧消融协议。网页“数据与协议”同时保留旧消融、首轮实际产物及本次 200 万目标，避免将计划数与已完成数混合。

## 当前实现与连接约定

SII 连接只从环境变量或 bashrc/zshrc 的静态 `SII_API_KEY` / `SII_BASE_URL` export 读取，不读取 `test_api.py`。SII 客户端沿用 P-256 并固定 TLS 1.2 / HTTP/1.1，保持证书校验及 `trust_env=False, proxy=None`；全局代理不变。下载与 CPU 预处理独立并行，来源配额不控制接收上限，候选封闭后完成去重/评测排除，再进入复用或 API 队列。旧全来源归档生成器已隔离，当前归档下载必须使用显式冻结 catalogue；不会自动恢复此前暂停任务。

代码检查、真实 API 探测与发布样例是实现验证，不是新的 200 万图库或下游训练收益。运行状态与资格证据分别保存在当前准备目录。
