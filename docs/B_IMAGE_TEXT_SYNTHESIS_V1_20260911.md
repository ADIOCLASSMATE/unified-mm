**历史方案：B 512px Qwen 合成、Codex sol low 终审与补写**

> 历史记录：本文描述当时的协议和实际产物。当前执行以 [DATA_SYNTHESIS.md](DATA_SYNTHESIS.md) 为准：复用优先、SII 补缺纠错、重试耗尽后才用 Codex CLI；连接信息只从 SII 环境变量或静态 shell rc export 读取。

2026-09-12 更新：用户已要求停用 Qwen API，改由现有 Codex CLI 的 GPT-5.6-sol low
直接高并发合成。当前队列、目录和验收以
[Codex 直接合成方案](B_CODEX_IMAGE_SYNTHESIS_20260912.md)为准。下文保留旧方案与历史实验记录，
其中 Qwen 待执行队列及旧全量发布路径已被替代。

2026-09-11，模型路由修订 v2。按用户最终指定的流程执行：**确定图片 → 无代理下载与筛选 → 512×512 预处理 → qwen3.8-27b 合成 I2T/T2I → 现有 Codex CLI 的 GPT-5.6-sol low 终审，不合格样本当场补写 → 发布与 VAE 缓存**。Qwen API 同样必须直连。纯文本继续使用原有 ClimbMix。

已实现流水线和 512px 缓存/训练适配。截至 2026-09-12 01:19 UTC，最新完整发布为 `public/datasets/unified_image_text_512_v1/releases/calibration_v2`，**2,887 对 I2T/T2I**：1,057 张本地 ImageNet、1,229 张 PixMo、601 张 Open Images，已跨批次去重并统一编号。165 对旧合成文本原样复用，1,073 对新 Qwen 文本直接通过，1,649 对由 sol 修正；全量图像哈希、RGB 512px、文本索引、教师来源及真实 VAE posterior 检查通过。这仍是校准版本，完整大批次尚未完成，训练尚未提交。B 分项结果与图片来源证据见 [数据分析](B_DATA_SCALING_RECOMMENDATION_20260911.md)。

早期 `train/` 的 239 对和 `shards/calibration_expanded` 的 2,039 对快照均保留，不与新合并版重复计数。Open Images 原合成任务已完成，WikiArt 文本合成已自动启动，ImageNet 100K 文本排在其后；五个校准批次的图像编码均已完成，ImageNet 100K 图像编码继续并行推进。后续依次发布 `releases/calibration_with_styles_v1` 和 `releases/imagenet100k_plus_corners_v1`。

本地 ImageNet 的 **100K 候选已完成预处理**，覆盖全部 1,000 类，50K 为旧文本复用候选，50K 计划新标。99,291 张通过，46 张像素重复、86 张命中评测近重复过滤、577 张预处理失败。准备阶段耗时 295 秒，约 337 张通过图片/秒，不包含模型合成时间。Open Images 的 1,024 张关系候选已有 651 张通过双对象保留检查。合成依次执行扩展校准、Open Images、WikiArt、ImageNet 100K；每批完成后自动合并发布并逐条审计，实际数量以发布结果为准。

新合并版的真实 MAR-KL16 posterior 为 `[2887,1024,32]`，实际训练 tokenizer 和 I2T/T2I loader 已全量读取，每个任务完成 722 个 batch，监督掩码、padding 和同图同 epoch 的 latent 一致性全部通过。最大序列分别为 1,244 / 1,151 token，宽度上限为 2,048。二进制 posterior 与行映射放在图片池，文本目录仅有 JSON 引用。记录见 `public/data_preparation/unified_b_512_v2/posterior_publications/calibration_v2/{complete_dataset_audit,training_loader_audit}.json`。完整 512px 验证缓存、FID 参考与目标大模型卡上验证仍未完成。

[全量训练加载检查](../scripts/audit_b512_training_loaders.py) 已接入后续 WikiArt 合并版与 ImageNet 100K 完整版：等待各自的图文、posterior 完整审计结束，再逐条运行实际 I2T/T2I loader 和组批，验证监督掩码、padding、身份及同图 latent。调度状态位于准备目录 `training_loader_publications/<版本>/job_status.json`，通过结果写入 `posterior_publications/<版本>/training_loader_audit.json`。该工具已经完整读取现有 2,887 对，并验证错误版本的审计文件会被拒绝；它检查数据加载，不包含大模型 forward 或卡上训练。运行时只在内存中切换待审计的数据路径，不修改训练配置。

512px 完整验证数据已接入后续队列：本地 50,000 张 ImageNet-val 原图、1,000 类及每类 50 张均已核对，顺序与旧验证清单完全相同。两张真实图的 VAE 编码、I2T/T2I 验证加载、跨 epoch 固定 latent 和 Inception 特征提取已通过，VAE/FID 的 RGB、bicubic resize、中心裁剪也已对齐。新缓存与 FID 参考保存在 `public/datasets/unified_image_pool_512_v1/validation_imagenet512_v1`，训练配置已指向这里；原验证文本及消融数据保持原位。两个 VAE 进程和一个 FID 进程各使用 4 个计算线程，等待当前训练图像编码主任务及两个补充任务全部结束后启动。98 个 VAE 分片合并后，再核对全部 50,000 行 posterior、实际 I2T/T2I 加载与 FID 参考；状态和预检证据位于准备目录 `validation_512/`，最终检查为 `complete_validation_audit.json`。当前是已排队，不能算作完整验证缓存已经生成；这些验证图片不进入训练合成库。

文本可以把图中已有的数量、关系和风格表达得更明确。若图中不存在目标能力需要的视觉内容，就要先补对应图片：例如水彩要选真实水彩/水彩风格图，五物体计数要选确实可见五个实例的图，读字要选处理后仍能读出文字的图。

**先下载哪些图片**

建议先做新增 250K 不同图片的验收批次，打通下载、预处理、教师输出和质检；再扩为新增 2M 的配方验证库，加上已有 ImageNet 1.281M，共约 3.281M 不同图片。这里是通过解码、去重和评测排除后的目标数量，下载候选数会更大；不足时不以劣质图补齐数字。

| 图片来源 | 首批目标 | 扩大验证库目标 | 优先选取内容 |
| --- | ---: | ---: | --- |
| [Open Images V7 train](https://storage.googleapis.com/openimages/web/download_v7.html) | 100K | 800K | 有框/关系/Localized Narratives 的多物体场景，人物活动、工具使用、遮挡与相对位置 |
| [PixMo-Cap](https://huggingface.co/datasets/allenai/pixmo-cap) | 50K | 400K | 日常场景、人物、长尾物体、非典型构图；原 caption 用于候选筛选与后续交叉检查 |
| [CC12M](https://github.com/google-research-datasets/conceptual-12m) / [Recap-DataComp](https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B) 精选 | 50K | 400K | 补前三者不足的场景/对象；优先物体共现丰富且成像清晰的图片 |
| [JourneyDB train](https://github.com/JourneyDB/JourneyDB) 精选 | 30K | 300K | 水彩、插画、3D、像素风、设计图、非典型组合；按最终图像验证实际风格 |
| [AnyWord-3M](https://github.com/tyxsspa/AnyText) / [TextCaps train](https://arxiv.org/abs/2003.12462) | 20K | 100K | 牌子、包装、海报、菜单、招牌；文字大且清楚，包含计划支持的语言 |
| 合计新增 | 250K | 2M | 各来源按处理后图片去重计数 |

现有 ImageNet 保留，第一批可以选取 100K 分层样本，同样以处理后的训练视图让 Qwen 合成、sol low 终审忠实 T2I/I2T，作为新旧文本质量对照。全库是否重标由这批结果决定；已有合格 caption 可复用。

这个第一批包含风格图片，而不只下载自然照片，是因为当前 T2I 风格监督存在真实目标缺口。JourneyDB 的图片也是现成图片来源；其原 prompt 只作筛选线索，教师最终根据下载图片重新描述，避免保留原生成器未画出的要求。该数据下载按发布方流程申请；如暂时不可获取，可先用 [BLIP3o-60k](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k) 中通过来源排查的风格图片做小样验证，其规模不能代替 300K 配额。

已另准备 [WikiArt 的固定分片](https://huggingface.co/datasets/huggan/wikiart/tree/d559852d2b232e0fcf195e775866964f0564f2b5/data)：下载 `train-00000-of-00072.parquet`，521,983,739 字节，上游 LFS SHA256 `04a2de8091c0e25704f736bb64073366ca0db1bc50a6104288188f19d694d069` 校验通过。该分片实际有 1,132 行，从中选出 1,024 张原生短边至少 512px 的图片，全部通过预处理和评测近重复筛查，无放大，覆盖 15 个上游 style ID。风格标签只用于选图，重新根据最终像素合成文本。元数据保留[数据卡](https://huggingface.co/datasets/huggan/wikiart)标记的 `license: unknown` 和原作者版权信息。合成排在 Open Images 后、ImageNet 100K 前，先发布 `releases/calibration_with_styles_v1`，再纳入 100K 合并版本；当前已在合成并持续产生通过终审的标签，尚未完成整批发布。

验证之后再向约 20M 不同图片扩展：大部分增量来自 CC12M/Recap；Open Images、PixMo、Objects365 提供结构和实例覆盖；JourneyDB 提供风格；AnyWord 和文档/图表数据提供专项内容。各来源的有效数量必须以下载成功、排除重叠和验收后的统计为准。COCO/Flickr/VG/GQA 暂不作为第一批必需来源，先用独立来源改善覆盖，保留现有跨来源评测价值。

下载不能只按来源随机抽满，应同时维护多标签能力桶：多实例计数（2/3/4/5/6+）、两实体及以上的颜色归属、明确左右/前后/上下、人与物交互、文字、真实艺术风格、长尾物体和反常共现。每个桶至少先人工检查几百张处理后的图，确认候选筛选确实找到目标内容。数据源自带的框、点、文字与 caption 可以辅助选图，但不是最终合成标签。

**预处理以训练实际可见内容为准**

用户已确定首阶段直接采用 **512×512**。教师输入与 VAE 编码均读取最终 512×512 版本，原图保留用于溯源和必要的重新裁剪。预处理版本必须在正式合成前冻结；尺寸、裁剪或图像内容改变后，相关标签需要重新核验。当前 B 的 256px 图文数据只作为历史基线，不能把其旧 latent 直接标为 512px 使用。

| 步骤 | 规则 | 对 corner case 的保护 |
| --- | --- | --- |
| 解码与规范化 | 应用 EXIF 方向，统一 RGB/sRGB；透明图按固定背景合成并记录，排除损坏图片 | 教师和训练方向、颜色、透明背景一致 |
| 原图分辨率 | 新下载主池优先短边至少 512；低分辨率图单独标记，避免靠放大伪造细节 | 512px 训练需要可用的细节，不只需要文件尺寸符合要求 |
| 图片身份 | 保存原始 source ID/URL、原图和所有派生视图的关联 | 不让同一原图的 crop 分散到训练/评测 |
| 单主体图 | 保持比例缩放，选取不切主体的方形 crop | 防止主体、手、文字被裁掉 |
| 多主体/计数/关系图 | 根据已有实例框或检测结果检查候选 crop；关键实例及关系必须保留 | 不能用裁掉一只动物的图学习“有五只” |
| 带文字/文档图 | 检查处理后目标文字可读，文字区域完整 | 防止教师看原图能读、模型看缩略图不能读 |
| 无法安全方形裁剪的图 | 记录并进入保留全图的候选桶，供长宽比支持或 I2T 专门视图使用 | 避免把宝贵的复杂场景裁成又一张单主体图 |
| 最终产物 | 教师与 VAE 读取同一份冻结后的像素文件 | caption 不描述原图中已经被处理掉的内容 |

整图拉伸会改变几何和字体；大面积补边会影响 T2I 学到的画面。建议 T2I 主池使用自然的、保留语义的裁剪；补边图如用于 I2T，应单独标记并小比例采样，不自动进入 T2I 主池。若后续 B 支持长宽比 bucket，可直接用保留的原图重建相应训练视图。没有确认完整视野的图，不能据不完整的检测列表监督全图“没有某物”或精确总数。

对计数、相对位置与 OCR 的验收，除 RGB 图外，先抽查同样 VAE 的重建视图。数据量大的时候，图像压缩后仍可见的信息才是可学习监督；对高分辨率内容的无法辨认不应被误当成模型推理错误。

**每张图如何合成与终审**

历史默认每张图由 **qwen3.8-27b 一次生成 1 条 I2T caption + 1 条 T2I prompt**，使用 SII Anthropic Messages，`max_tokens=3200`、`thinking={type: enabled, budget_tokens: 1600}`。旧版从 `test_api.py` 解析配置的行为已经删除；包括保留的历史工具在内，连接信息现在只从 `SII_API_KEY` / `SII_BASE_URL` 环境变量或静态 bashrc/zshrc export 读取，不使用官方 Qwen 端点。历史 Qwen thinking 块没有写成训练标签；完整外层 JSON 围栏可由解析器移除。当前生产模型路由见本文顶部链接。

| 字段 | 内容约定 |
| --- | --- |
| `i2t` | 一条真实 caption，通常 50–100 词；简单图可更短，不凑字数 |
| `t2i` | 一条忠实于这张图片的 prompt，通常 30–70 词；不改变风格、数量、属性或关系 |
| `observations` | 候选计数、关系和可读文字，缺少证据时留空；不视为外部真值 |
| `capabilities` / `uncertainties` | 图片实际覆盖的能力与不可确认的信息 |
| `usable` | I2T/T2I 各自是否可用；当前配对发布要求二者都可用 |

首版输出英文，图中文字转写保留原语言。照片不改写为水彩，模糊文字不靠常识补全，检测列表不完整时不能声称精确总数或对象缺失。完整提示与 JSON Schema 集中在 [模型客户端](../utils/image_text_teacher.py)，图内文字和候选文本只作为待检查数据。

每个待发布样本都由 **现有 Codex CLI 的 `gpt-5.6-sol`、`model_reasoning_effort="low"`** 最终验图。默认每请求 4 张独立 512px 附件，逐图携带 ID，不缩成拼图。终审直接对照像素核对 I2T/T2I 的实体、属性归属、计数、位置、文字和实际风格：

| 终审结果 | 本次请求返回内容 | 后续状态与生成者 |
| --- | --- | --- |
| `accept` | 判定及问题列表，`replacement=null`；不重写好样本 | `ready`；生成者仍为 Qwen |
| `replace` | 判定及问题列表，并在同一次请求中直接给出正确 I2T/T2I | 替换文本经格式/长度检查且双任务可用后为 `ready`；生成者为 sol |
| `reject` | 图像不足以可靠标注，`replacement=null` | `review`，不发布 |

Qwen 的 JSON 错误、截断、过长文本或不可用标签也交给 sol 当场补写。Qwen 的连接、认证、额度等服务错误保留为失败任务，不触发 sol 全量代写。终审无法接受机械检查未通过的候选；sol 补写仍不合格时保留失败/复核状态。

程序保存实际生成者、终审模型/effort/后端、判定、问题和原始响应。`ready` 表示已有 sol 终审通过且结构满足训练契约，不能据此声称达到人工事实准确率目标。计数/OCR/关系桶经终审通过后可正常发布，无需逐图人工放行；人工抽检用于校准和验收整个配方。

**验收之后再决定是否扩大**

对已发布 `calibration_v2` 的 2,887 对进行了全量标注字段统计，并核对发布清单 SHA256：447 对包含某类对象数量至少 3 的计数记录，122 对至少 5，10 对至少 10；650 对包含非空可读文字记录。这里只统计经过终审的 `observations` 元数据，不证明这些事实独立正确，也不证明每条事实都写入了 I2T/T2I 训练文本；原样复用的旧文本没有该元数据，空字段不能判为图片没有对应内容。较复杂计数仍需定向选图，扩大 ImageNet 的总量不能替代该覆盖检查。记录位于 `public/data_preparation/unified_b_512_v2/calibration_v2_observation_coverage.json`，未增加模型调用。

先取约 1K–2K 张有效分桶样本校准，再处理首批 250K。每桶统计 Qwen 通过率、sol 替换率、最终剔除率、文本长度、真实 token 用量和合格图/小时。首个合并版本已有真实 ImageNet、照片、插画等图片的终审结果；其中 OCR、关系、计数仍很稀疏，来源与能力桶配额不能用候选数代替。

扩展校准的 ImageNet 部分已完成终审：508 对旧文本中 141 对原样通过、367 对补写；419 对收到有效调用结果的新 Qwen 文本中 226 对通过、193 对补写。另外 95 张失败、2 张近重复排除。这里的通过率分别为 27.8% 和 53.9%，说明应继续逐图核验，不能整库复制旧标签；也不能将 sol 的判定当成人工事实准确率。网络失败不属于“Qwen 文本质量不合格”。

除全量终审外，按来源与能力桶人工抽检，并用已有框/点/文字标注交叉检查。二维框不能直接验证三维前后关系；OCR、计数还应抽查相同 VAE 的重建图。建议验收起点：普通描述原子事实准确率至少 95%，精确计数/文字/属性绑定事实至少 98%，同时报告分桶样本量与不确定性；这些是目标，尚未测得。若某桶替换率高，先改善选图和提示，不盲目扩大 Qwen 调用量。

实际训练 tokenizer 对每段文本限制 960 token，不静默截断。当前 2,048 总序列中，1,024 为图像 token，另预留任务前缀与特殊 token。新增 QA、多图对话或长宽比需要 loader/模型支持，新增 JSON 字段本身不能补齐接口。

从旧配方到忠实配对、到新图库、到结构化文本逐步做同预算数据对照。I2T 看 B 的关系、计数、实例位置和跨数据集检索；T2I 看完整 prompt 约束、实际风格、文字，并补齐官方 GenEval/DPG 结果。评测图、其原图/派生图和测试 prompt 在选图阶段排除。

**高吞吐 pipeline 的具体组织**

冻结图像之后，VAE 编码与文本合成并行进行，发布时再按图像身份对齐：

```text
候选清单 → 直连下载 → CPU 512 预处理 → 冻结视图与原图 tar
                                            ├─ Qwen 合成 → sol 终审/补写 → 合格文本发布
                                            └─ 校验视图 SHA256 → VAE 分片编码
                                                            ↓
                              按视图 SHA256 对齐 → 全局编号/文本与 posterior 索引 → 全量审计
```

| 环节 | 已实现的默认起点 | 调优依据 |
| --- | --- | --- |
| 下载 | 64 个异步连接、每域名 8 个，连接池复用，有限退避 | 直连带宽、源站限速、有效字节/秒 |
| CPU | 8 个进程，spawn 避免继承 tokenizer 线程；每图处理一次 | 解码时间、CPU/磁盘吞吐 |
| Qwen | 默认 4 个 worker、原生 curl 直连 HTTP/2 流式请求；当前实跑限制 30 RPM / 600K TPM | 12 并发实跑出现大量约 60 秒断流，需按有效成功量调节，不能只增加 worker |
| sol 终审 | 统一组批后分发，每批最多 4 图、首图起最多等 10 秒；当前 4 个 CLI 并发，CLI 默认 2 | 按实际批次填充率、合格图/小时和串图率调节 |
| 状态/存储 | 有界队列、SQLite、约 512MiB tar、字节范围读取 | 避免小文件膨胀和未完成任务堆积 |

上述数字是起测参数，不能视为账户配额或测得吞吐。Qwen 按每请求至少 6,144 token 保守预留节流；真实 usage 保存在原始响应。Qwen 429/部分 5xx 有限退避，认证/模型不存在等错误停止任务；网络错误单独记账，连续大量失败会停下保存进度。

Codex 每批启动一个临时只读 `exec`，复用现有登录；明确指定模型、low、JSON Schema、图片附件，关闭该调用的网页搜索并忽略个人配置。用组批摊薄 CLI 启动和共同提示开销，不为合格文本再生成一遍，也不另起一次请求补写。全量终审可能成为瓶颈，应先测其吞吐再扩 Qwen 并发；尾批会正常排空。有界队列把终审的背压传回合成和下载，控制内存。

23:08 UTC 已将当前进程恢复到统一组批实现，已完成样本和已收到的响应保留。此前多个终审 worker 自行取下一张图，会互相分散陆续到达的样本；观测到 658 个请求中 329 个仅一张图。现在由一个组批器填充，再交给并行终审 worker，等待上限从首张图起算；错峰到达的 4 张图交给 4 个 worker 的回归测试验证只需一个 4 图请求。实际节省比例仍以运行统计为准，不能仅凭请求数推断账单。

每个输出目录只允许一个协调器，使用文件锁；多机各自使用固定分区和独立目录。持久化契约包含候选/排除清单摘要、预处理版本、tokenizer 路径、Qwen 公共配置、sol 配置及两级提示/schema 摘要。Qwen 原始响应、sol 整批原始响应都在解析前强制提交，恢复时可以直接重用；完成任务不再次调用模型。API 超时后服务端是否完成可能未知，显式重试仍可能重复计费。

实跑发现少量 sol 批次外层编号正确，但替换文本内部的 `image_id` 不匹配。流水线会拦截这种结果；新启动的批次现在对该单张样本重新终审一次，保持图片和 Qwen 候选不变，并保留两次原始响应及 `retry_of_batch_id`。其他合格样本不重审，再次不合格仍不发布。批次解析失败、服务失败和明确拒绝不由这条单样本规则自动重试。相关 22 项流水线测试通过。2026-09-12 00:31 UTC，已通过独立补跑修复扩展分区的 1 个旧失败样本，核对图片与 Qwen 原始响应未变；该分区现有 2,040 对 ready，但已发布的 `shards/calibration_expanded` 仍为 2,039 对快照，新增一对已纳入 calibration_v2。2026-09-12 01:04 UTC，Open Images 的 6 个历史终审失败样本也已补审通过（2 个编号错误、4 个超时），核对图片和 Qwen 原始响应均未变；这 6 对由下一版合并接收，记录在 `calibration_openimages_sol_repair/repair_audit.json`。

VAE 只读同一冻结视图，I2T/T2I 共用 posterior。[图像缓存调度器](../scripts/encode_b512_banks.py) 已运行：当前容器配额为 16 CPU，初始采用 2 个编码进程、每进程 4 个计算线程，每个分片约 512 图，优先处理校准批次。每张图编码前校验 SHA256，源清单也冻结并校验；最终未通过文本终审的图片不会进入训练索引。图像、posterior 和二进制行映射全部保存在图片池。

2026-09-12 00:12 UTC 增加 2 个补充编码进程，总计 4 个进程、每进程 4 个计算线程。原有任务从低编号开始，补充任务从 ImageNet 分片 193/192 向 99/98 处理各 48 个分片；每个分片在检查缓存、编码和原子写入期间持有同一文件锁。两个任务若遇到同一分片，后到者会等待并验证复用已完成缓存。补充任务不执行合并，原调度器仍逐个验证全部分片并发布唯一索引。状态位于准备目录 `posterior_encoding_supplement/worker_{0,1}/job_status.json`。45 秒观测窗口实测总吞吐 1.96 图/秒、平均使用 14.53 个 CPU 核，记录在 `posterior_encoding_supplement/throughput_observation.json`；此前双进程每个约 2 秒/图。这是短窗口观测，不是全任务耗时保证。分片互斥及异常释放检查已通过，相关 8 项测试通过，原真实缓存复用也已验证。

[索引合成器](../scripts/compose_b512_posterior_index.py) 使用磁盘索引按 `view_sha256` 连接发布文本与已编码图片，单独保存发布编号和分片内部编号；读取时继续检查分片内部身份，不复制 posterior 大张量。已用两个真实图像缓存重新拼接首批 239 张图，并验证全部 I2T/T2I 读取；与原先直接编码的最大绝对差为 `5.96e-08`，记录在 `posterior_bank_roundtrip_audit.json`。三个后续发布版本都有独立任务等待“文本发布完成 + 所需缓存完成”，然后自动合成 posterior 索引，并以 `--require-posterior` 全量审计。

大文件元数据使用独立的并行 Range 下载器，验证 206/Content-Range、保存部分块并续传。ImageNet、PixMo 和 Open Images 关系元数据已有候选适配；后者使用两个对象的归一化框保护裁剪。PixMo 第一批原站直连有效率较低，后续候选限制到已成功直连的 S3 域名。旧合成文本及已收到的模型响应使用独立缓存队列，不占等待 Qwen 网络请求的 worker。

WikiArt 的内嵌图片直接提取到 tar，并以字节范围作为本地输入，避免生成一批中间小图片文件。本地文件和 tar 输入均先限制单图字节数；读取期间保留文件句柄引用，防止其他并发读取淘汰缓存后关闭正在使用的描述符。

已对 117,036 张本地评测图片建立原图/中心视图共 229,603 个 pHash 索引，使用四段索引筛出 Hamming 距离不超过 3 的近重复。`run --near-exclude-index` 在模型调用前排除命中，`export --near-exclude-index` 在发布时再次筛查。它是保守的近重复过滤，不保证识别所有语义派生图。自动并发调优和独立 OCR/事实服务尚未接入。

**图片下载与 Qwen API 均不使用代理**

全局 mihomo 保持运行以维持 Codex 连接，禁止用 `clashctl off` 关闭整个服务。图片和元数据下载使用 `httpx.AsyncClient(trust_env=False, proxy=None, http2=True)`，忽略代理环境；本机 Python/OpenSSL 默认握手曾超时，下载客户端选用 P-256 密钥交换，保留 TLS 1.3 和证书/主机名校验。图片下载每次重定向仍使用同一直连客户端，并重新应用源站并发限制。

SII 的 Anthropic SDK 通过自定义传输层调用本机 curl，明确设置 `--disable --proxy '' --noproxy '*' --http2`，并从 curl 子进程环境移除大小写代理变量。URL、密钥、图片和请求体经 stdin 传递，不出现在命令行或临时请求文件中；不跟随 API 重定向。使用 Messages 流式响应并收集完整结果，只将 text 块作为候选。真实失败图片在流式调用中已有成功结果，但较高并发仍出现约 60 秒的 HTTP/2 断流，不能声称服务问题已完全解决。错误记录保留安全的传输类型/退出码。把所有代理环境变量设为不可达地址、清空 NO_PROXY 的本地回归已分别覆盖图片 GET 和 Anthropic `/v1/messages` 流式图像 POST。

启动时只读检查 Linux 路由中的已知隧道接口（tun/tap/wg/tailscale/ppp 等），发现时停止；这能检测可见接口，不能证明任意基础网络都没有透明转发。不会启用 VPN、代理镜像或改写用户全局网络配置。直连失败就记录失败并等待补跑。

Codex 终审使用现有 CLI 的登录和连接方式，不把 Qwen API 密钥放进提示、输出清单或命令行，并从 Codex 子进程环境移除 `SII_API_KEY`；保留 Codex 所需的代理与登录。本轮不修改 Codex 的全局认证与网络设置。服务保持运行和 API 直连已在同一轮真实调用中同时成立。对同一组失败图片的原生/Python 传输、HTTP/1.1/HTTP/2 和思考预算对照没有证明某个参数能稳定解决断流，当时继续沿用其既有协议参数。这是历史诊断结论；当前连接入口和 TLS 参数以 [DATA_SYNTHESIS.md](DATA_SYNTHESIS.md) 为准。

**512×512 对训练和缓存的适配目标**

| 项目 | 当前正式 B | 新配方目标 |
| --- | --- | --- |
| 图像像素 | 256×256 | 512×512 |
| KL16 latent 网格 | 16×16 | 32×32 |
| `image_tokens_per_img` | 256 | 1,024；模型、dataset、生成/验证配置一致 |
| `image_latent_dim` | 16 | 16 |
| posterior mean+std | `[N,256,32]` | `[N,1024,32]` |
| 图像任务 `max_seq_length` / `pad_to_length` | 512 | 建议起点 2,048；文本预算为 2,048−1,024−3=1,021 token |
| ClimbMix | 当前配置 | 沿用原配置 |
| 位置编码与图像顺序 | 16×16 空间位置 | 基于 32×32 正确生成 2D RoPE、Halton 和随机顺序 |
| 图像 microbatch / GA | 16 / 当前调度 | 由目标大模型显存与吞吐实测决定；保持有效图次记账 |
| FID 参考 | ImageNet-val 256px moments | 按新的 512px 预处理重算相应参考；保留同协议历史对照 |

已修改 [VAE 编码脚本](../scripts/imagenet_encode_kl16_vae.py) 的缓存分配、复用校验、shape 校验、reshape 与 metadata，按 `image_size/16` 推导网格；`--frozen_views` 要求教师与 VAE 读取同一份已处理像素。同步修改了合并/分片索引、训练 loader、解码检查、正式 FID 参考分辨率校验及配对评测初始噪声。旧 256px 默认仍可用于历史复现；独立的历史 `evaluate_vae_rfid.py` 专项协议仍限定 256px，不能把它当成新的 512px 入口。

图像 token 增加四倍，不应把当前 microbatch 直接照搬。新增 [B 512 配方验证配置](../configs/selfless/unified_b_x0_images512_v1_ascend64.yaml) 保留当前 0.6B backbone 以隔离数据/分辨率变量，图像 microbatch 暂取 4，GA=4 和 ClimbMix 完全沿用；64 卡每个图像任务 256 图/更新，是原 B 图次的四分之一，但图像 token 曝光相同。它不是已测得可承载的大模型配置；大 backbone、图次预算和卡上 batch 需要单独确定。配置在 955 步停下做配方验证，没有提交任务。CPU 已验证小型 B/X0 的 1,024 图像 token I2T/T2I forward/backward、分片身份，以及真实 MAR-KL16 权重的 512→32×32→512 编解码有限值；后者使用全零输入作尺寸探针，不是画质或吞吐评测。[CPU VAE 记录](../output/evaluation/data-strategy/b-20260911/kl16_512_cpu_smoke.json)。卡上生成、显存及吞吐仍需实测。

该配置的数据路径现已指向完整计划版本 `releases/imagenet100k_plus_corners_v1`；只有对应文本、posterior 和审计都完成后才可使用。首批 `train/` 的 239 对保留作校准记录。真实图片的跨缓存读取已验证，完整 512px 验证集缓存与 FID 参考仍需另行准备。

512px FP16 posterior 每图 `1,024×32×2 = 65,536 bytes`，20M 图仅 posterior 约 1.31TB，30M 图约 1.97TB；RGB/原图/文本另计。I2T/T2I 共用这份 posterior，避免重复编码和存储。首版已提供 `--index_only` 与按需 mmap 的训练读取，避免合并出 TB 级大张量；全量校验仍需顺序扫描各 shard 一次。原图和处理图顺序打包为 tar，训练当前仍沿用原有采样器；按 shard 优先打乱的训练采样策略是后续吞吐优化项，尚未实现。

**已实现入口与使用方式**

实际数据位置：原始 ImageNet 位于 `public/dataset/imagenet/v1`，新增/冻结图片归档位于 `public/datasets/unified_image_pool_512_v1`，最终合成文本发布到 `public/datasets/unified_image_text_512_v1`，运行状态、来源元数据与模型原始响应位于 `public/data_preparation/unified_b_512_v2`。最终文本目录不放图片或未通过终审的候选。

旧 ImageNet 只选合成 caption 和 `faithful_photo` T2I，核对原图 SHA256，再让 sol 对照新的 512px 视图终审。通过后原样保留两段文本和各自模型来源；若不符则由 sol 当场补写。旧图可按原生短边至少 224px 入选并放大到 512px，记录 `upsampled`，不声称获得新的细节；新增图库默认要求原生短边至少 512px。

[流水线脚本](../scripts/legacy/synthesize_image_text.py) 接收已筛选的候选清单。每行字段为 `source`、`source_id`、`url`（或本地 `local_path`），可带 `parent_id`、`split`、`capabilities` 和 EXIF 归一坐标系下的 `required_boxes: [[x0,y0,x1,y1], ...]`。来源元数据需要先转换成该清单，脚本不自动申请图库权限或抓取站点目录。

排除文件每行一个 `source:id` 或图片 SHA256，包含正式评测原图身份。空文件不能证明无评测重叠；实际任务还在预处理和发布时使用上述 pHash 索引排除跨来源近重复，其检测范围不等于全部语义派生图。

```bash
# 仓库根目录；只下载并固定视图，不调用任一模型，也不要求 API 密钥。
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python scripts/legacy/synthesize_image_text.py run \
  --manifest /data/candidates.jsonl --exclude /data/benchmark_exclusions.txt \
  --output public/data_preparation/unified_b_512_v2/runs/production --image-root public/datasets/unified_image_pool_512_v1/production --prepare-only

# 历史工具也只读取 SII_API_KEY / SII_BASE_URL 环境变量或静态 shell rc export。
# 复用上一命令下载/处理的图片，Qwen 合成，Codex sol low 终审与条件补写。
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python scripts/legacy/synthesize_image_text.py run \
  --manifest /data/candidates.jsonl --exclude /data/benchmark_exclusions.txt \
  --output public/data_preparation/unified_b_512_v2/runs/production --image-root public/datasets/unified_image_pool_512_v1/production \
  --download-workers 32 --per-host 8 --cpu-workers 8 --qwen-workers 4 \
  --teacher-workers 4 --judge-batch-size 4 --rpm 30 --tpm 600000

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python scripts/legacy/synthesize_image_text.py export \
  --run public/data_preparation/unified_b_512_v2/runs/production --output public/datasets/unified_image_text_512_v1/train

# 单个 VAE shard 示例，完成 0..63 的全部 shard 后再发布 posterior 索引。
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python scripts/imagenet_encode_kl16_vae.py \
  --source_mode manifest_jsonl \
  --source_manifest_jsonl public/datasets/unified_image_text_512_v1/train/manifest.jsonl \
  --cache_shard_dir public/datasets/unified_image_pool_512_v1/vae/production --image_size 512 --frozen_views \
  --num_shards 64 --shard_index 0 --batch_size 8 --num_workers 4 --no_hash

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python pretrain/merge_flow_latent_shards.py \
  --shard_dir public/datasets/unified_image_pool_512_v1/vae/production --index_only --no_hash \
  --row_index_path public/datasets/unified_image_pool_512_v1/vae/production/posterior.rows.pt \
  --manifest_jsonl public/datasets/unified_image_text_512_v1/train/manifest.jsonl \
  --output_path public/datasets/unified_image_text_512_v1/train/posterior_index.json
```

上述发布路径是新建目录的模板；已经发布的 `train/` 不覆盖，后续批次使用 `releases/` 下的新版本。实际任务的准备与合成命令均传入同一个 `--near-exclude-index public/data_preparation/unified_b_512_v2/benchmark_phash`，发布时再次传入，避免状态契约不一致。`--row_index_path` 把二进制行映射留在图片池，并在 JSON 中记录绝对路径。

[发布审计入口](../scripts/audit_image_text_publication.py) 在不调用模型的条件下逐图核对原图/视图 SHA256、RGB 512px、跨分片唯一身份、I2T/T2I seek 对齐、最终教师与复用文本哈希，并用本地 tokenizer 检查文本预算。后续 `calibration_v2` 和 `imagenet100k_plus_corners_v1` 的队列已接入该检查，审计结果写入对应准备目录的 `publication_audit.json`。

VAE 并行编码计划及总状态位于 `public/data_preparation/unified_b_512_v2/posterior_encoding/`；每个缓存的清单和状态位于 `posterior_banks/`。发布索引的依赖队列位于 `posterior_publications/<版本>/`，成功时产生 `complete_dataset_audit.json`。`--require-posterior` 额外核对发布清单哈希、连续全局编号、分片内身份、`[N,1024,32]` 形状及全部有限值/非负标准差。代码验证更新为 791 项通过、2 项跳过，日志为 `repo_checks_posterior_banks.log`。

`--retry-failed` 默认重新解析已收到的响应，缺少响应才调用对应阶段。确需重生成时，组合使用 `--retry-failed --regenerate-failed`：对标记为 sol 的失败只重做终审，包含尚未收到终审响应的超时，始终保留已收到的主生成结果；其他可重试失败按原阶段处理。缺失主生成记录的 sol 失败会报错，避免悄悄重新生成。该超时恢复分支已加入回归测试，相关流水线测试更新为 23 项通过。`review` 表示终审无法产生可用配对，不会自动补跑或通过手工 accept 字段绕过 sol。发布只接收 `ready` 且终审/生成者来源一致的配对，失败的发布不留下可见的半成品。

100K 批次混合旧文本复用与新 Qwen 合成，连续服务失败计数必须只由真实 Qwen 请求成功清零。已修复旧文本/已保存响应也清零该计数的问题；原代码在离线交错测试中错误地把全部 33 次失败请求跑完，修复后在连续 32 个样本调用失败时保存进度并停止，真实请求恢复成功则正常继续。网络失败没有转入 sol 补写。该分支与其余流水线共 25 项测试通过，记录在准备目录 `primary_outage_regression_{before,after}.log`。修复由之后启动的 100K 进程加载，不重启当前 WikiArt，也不改变 SII 参数和代理设置。

2026-09-12 的一次有界直连观察中，已失败的 Open Images 图片用原 SII 参数在 13.9 秒返回完整 SSE 结果，包含 `message_stop` 和 `end_turn`；它只证明这次重试成功，未定位其余断流原因。收到的有效 Qwen 候选已保存，I2T/T2I 分别为 105/66 token。2026-09-12 01:04 UTC，该恢复任务已在 Open Images 原任务和 sol 补跑结束后完成：持有状态库写锁，复用已收到的响应通过最终终审并原样保留 Qwen 文本，没有再次请求 Qwen。新增一对由后续合并版本接收，记录在 `sii_stream_observation_20260912/recovery_audit.json`。观察与恢复状态位于准备目录 `sii_stream_observation_20260912/`。

旧的全 sol 合成目录与 v2 的提示/模型契约不同，不能混用同一状态库；需新目录。原图/视图 tar 和已接收响应保留，正常恢复不重复下载。原始响应可能包含候选内容，按数据集产物保管；API 密钥不写入契约。

第一轮按 250K→2M 验证。20M 前仍需处理训练 loader 的全量 Python manifest 字典和 train/val 路径集合所带来的内存增长。`export` 已支持重复 `--run` 参数：使用磁盘去重索引，在多个已结束分区之间统一去重、编号和构建文本 seek 索引，原子发布到新目录；直接拼接分区 JSONL 或 latent 索引仍不正确。首个 239 对版本即由 ImageNet/PixMo 两个状态库合并发布。

早期附件/格式探针保存在 [v2 验证目录](../output/evaluation/data-strategy/b-20260911/qwen-sol-v2/)。真实数据的运行日志、原始响应和核验结果位于 `public/data_preparation/unified_b_512_v2`：`combined_publication_audit.json` 验证首个合并发布；各批次 `job_status.json` 记录执行状态；`implementation_validation.json` 记录代码验证。PixMo Parquet 已核对上游 LFS SHA256，Open Images 关系 CSV 已固定 GCS generation 并核对上游 MD5。512px 旧版 CPU 记录保留在 [前版验证记录](../output/evaluation/data-strategy/b-20260911/implementation_validation.json)。
