> 历史维护文档：此文件保留 2026-09-13 至 2026-09-15 的旧准备、冻结、发布与全图复核说明，不代表当前生产策略。当前入口见 [DATA_SYNTHESIS](DATA_SYNTHESIS.md)。

# 数据合成当前运行

当前使用 SII `deepseek-v4.1-flash` 逐图复核；启动与恢复命令见 [2026-09-15 全量迁移](B_DEEPSEEK_FULL_REVIEW_20260915.md)。Qwen 历史结果保留，Codex CLI 本轮关闭。

# 当前 B512 数据准备与合成

当前入口是 `python -m scripts.synthesize_image_text`，实现位于 `data_synthesis/`。运行参数只有一份：[b512_sii_v1.json](../configs/data_synthesis/b512_sii_v1.json)。[Long 主库协议](B_BLIP3O_LONG_DATA_PLAN_20260913.md)和[来源 JSON](../configs/protocols/unified_b_blip3o_long_v1.json)定义当前范围；本文定义实际执行方式。

2026-09-14 用户明确要求不再计算数据哈希，当前 `compute_hashes=false`。下载、预处理、冻结、模型附件、文本发布和 posterior 包装入口都使用该策略：不计算图片、归档、清单和文本的 SHA/MD5，不计算 pHash，也不执行内容哈希去重或评测近重复扫描。改用 `source/source_id`、URL、原图引用和 `view_id` 连接数据；保留长度、RGB 512px 解码、文本结构及逐行映射检查。历史回执和来源中已有的哈希仅保留为历史信息；既有任务 ID 与小型配置/提示协议指纹维持兼容。审计明确输出 `compute_hashes=false`，不会把跳过的内容校验写成通过。缺省当前配置之外的历史重放仍按其原协议运行。

## 训练范围与阶段

采用 **BLIP3o Long 主库 + 160 万专项起始覆盖目标 + 完整本地 ImageNet train**。Long 标称约 2,700 万条，账面相加约 2,988 万，最终数量按已知身份去重和 512px 验收清点；不计算或声称全库内容重复率。非 ImageNet 合格图至少 200 万的底线保留，各来源配额不是硬上限。ClimbMix 沿用现有语料。合格首轮非 ImageNet 105,322 张计在并集内。Short 与独立 JourneyDB 暂不全量下载；已选补充来源照常续传，有价值的独有图保留。

原始图像包、512px 视图、文本发布和准备状态都写入 `/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/unified-mm-b512/`。该新方案在 global_user 中只新增 posterior 缓存：`public/datasets/unified_b512_posterior_cache/`。旧 ImageNet/历史发布作为只读来源；旧 v3 图像、文本、准备目录迁移后保留兼容软链接，已有清单中的绝对路径和 SHA256 无需重写。

| 来源 | 起始合格图目标，非上限 | 优先复用和精选入口 |
| --- | ---: | --- |
| PixMo-Cap | 500,000 | 原人工 caption，场景/动作/属性分层 |
| Open Images 关系 + Localized Narratives | 300,000 | 同原图人工叙述、框与关系；关系专项优先 |
| PixMo-Points | 180,000 | 同图已有描述、完整点标注；不能假设大部分有 Cap caption |
| AnyWord-3M | 350,000 | 已有文字、区域和描述，约中文 200K / 英文 150K |
| WikiArt | 70,000 | 媒介/风格/题材元数据；标题不自动当完整 caption |
| BLIP3o + ShareGPT-4o-Image T2I | 80,000 | 保留真实实现的内容，检查原 prompt 与图片一致性 |
| TextCaps + DOCCI train | 30,000 | 尽量完整复用合格 train 人工 caption |
| PixMo-Docs / ChartQA train | 40,000 | DIM caption 精选标注可连接原图；只收 512px 可读视图 |
| PD3M | 50,000 | 补缺失摄影题材和长尾外观 |

原来的 JourneyDB 30 万、CC12M 10 万覆盖目标并入 Long 主库，不再额外相加；这不代表相应来源已经证明全部同图。Long 的 SA-1B / CC12M / JourneyDB 分支全取，共 2,891 个固定版本 tar，约 1.374 TB，当前检查版本与完整长度，不计算 SHA256。

```bash
.venv/bin/python -m scripts.download_blip3o_long init
.venv/bin/python -m scripts.download_blip3o_long download --files 16 --ranges 4
```

`init` 固定版本 `e4d07091a466d1a1e35a9b0c61caddc78d14a059`，从官方直连元数据或显式 `--metadata` 构造完整 catalogue；不隐式加入 Short/独立 JourneyDB。`download` 使用同一个严格 Range/校验下载器。下载归档与 512px 服务独立；归档通过校验后用以下两个进程接入，原图直接按 tar offset 引用，不再次复制一套原图：

```bash
.venv/bin/python -m scripts.intake_blip3o_long --watch
.venv/bin/python -m scripts.synthesize_image_text prepare \
  --supply-root /inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/unified-mm-b512/preparation/supply/blip3o_long_v1 \
  --root /inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/unified-mm-b512/preparation/prepared/blip3o_long_v1 \
  --workers 8 --exclude <benchmark-identities.txt>
```

Long intake 单次索引一个 tar，按 256 图封闭小批次，最多积压 64 批；已有 txt 原样保留，缺失文本留给后续补缺。实际图像解码、512px 全幅视图、评测排除由 prepare 执行；全部归档接入后才关闭输入，正式 API 合成仍等待整个图片池封闭及资格验收。低分辨率候选不绕过现有门槛，不按标称数量伪装验收成功。

配额中的替代来源共享对应预算，不另加重复计数。MONET Parquet 的 384px 缩略图不能直接充当原生 512px 训练图；取对应完整 JPEG，已存 SANA latent 不能代替 KL16。DIM / TextOCR-GPT4V 等 caption-only 精选必须连接原始 train 图像身份。Recap 暂作备用，来源核实与下载范围见来源选择协议。

流程为：`连接已有标注 → 冻结候选清单 → 原图下载与 512px 预处理并行 → 关闭所有输入批次 → 去重及评测排除 → 冻结图片池 → 复用/规则转写 → SII 补缺纠错 → 重试耗尽后 Codex 兜底 → 文本发布、posterior 和 loader 验收`。正式批量 API 必须在完整图片池冻结之后开始；小规模开发使用显式 `--pilot`。

## 网络与连接信息

SII 连接只取 `SII_API_KEY`、`SII_BASE_URL`。优先读取当前进程继承的环境变量；未继承时，解析当前用户 `~/.bashrc`、`~/.zshrc` 中对应的单行静态 `export NAME="value"`。**不读取、不导入、不执行 `test_api.py`；不 source 或执行启动文件。** 两份 rc 中同名缺失变量若有冲突则报错，调用方应明确 export 所选值；包含命令替换的声明必须先由用户正常 shell 提供环境变量。运行日志不输出 key。

SII 的独立 HTTP 连接池使用 `trust_env=False, proxy=None`，固定 P-256 密钥交换、TLS 1.2 上限和 HTTP/1.1，保留证书与主机名校验。此次实测该组合可返回 200；P-256 本身不足以保证该链路可用。图片下载也直连，但保留其原有 TLS 协商配置。所有修改局限于客户端，不调用 `clashctl off`，不改全局代理、路由或 DNS；Codex CLI 保留其需要的代理环境。

```bash
.venv/bin/python -m scripts.synthesize_image_text check-connection
.venv/bin/python -m scripts.synthesize_image_text plan
```

`check-connection` 只验证配置存在且 URL 合法，不发送请求。真实 API 探测见下文。

## 候选数据接口

下载入口读取规范化 JSONL，不依赖固定十个来源或固定数量上限。每行明确原始 `source/source_id/split=train`，附 `url` 或 `local_path`；不要把上游打包文件名中的 train 当作原图训练划分证明。

```json
{"source":"pixmo_cap","source_id":"original-image-id","split":"train","url":"https://source.example/image.jpg","view_policy":"fit_pad","capabilities":["relation"],"caption_candidates":[{"image_identity":"pixmo_cap:original-image-id","kind":"human_caption","author":"allenai/pixmo-cap","text":"A person holds a red umbrella beside a bicycle.","provenance":{"dataset":"allenai/pixmo-cap","revision":"pinned-upstream-revision","field":"caption"}}]}
```

- `caption_candidates` 保留原文本、作者、字段与版本；`kind` 可为 `human_caption`、`curated_caption`、`generation_prompt`。已有已验收发布通过 `freeze --release` 导入，生成绑定相同 `view_id` 的 `accepted_pair`，不重新调用教师。
- `parent_id` 和 `identity_aliases` 用于跨来源原图身份连接；同图多种标注保留。相同身份的标注行在一次 `ingest` 中使用磁盘 SQLite 聚合；冲突的 URL、视图或其他标量字段报错。跨 intake 的相同来源身份必须先合并，封闭清单时检查，避免下载时丢失额外标注。
- `annotations` 保存上游点、框、OCR 等原始标注。尚未经最终视图检查的原始标注不能标为 verified。规范化适配器应使用现有元数据、几何和可读性检查，不逐图调用 GPT 做入口筛选。
- 默认 `fit_pad` 保持全幅。预处理记录 EXIF 方向、crop、编码、尺寸、原图引用及最终 `view_id`；`source_sha256/view_sha256=null`、`hashes_computed=false`。计数和文字优先保留全部对象。被裁掉、缩小后不可读的内容不作确定监督。
- OCR/文档 caption 复用要求 `readability_view_id` 对应最终可读视图。原始 generation prompt 不能仅凭视图引用直接免审；只有已验收的该视图文本可直接复用。
- 可确定的正标注用 `verified_facts`：需要 `verified=true`、当前 `view_id` 和 `provenance`。计数还需 `entity/count/fully_visible/exhaustive_for_referent`，文字需 `text/carrier/readable`，关系需明确 `text`。标签不完整不能推断数量为零；字段不足则进入 API 补缺队列。
- 可用 `required_fact_text` 声明原 caption 必须含有的已知事实字符串。规则仅检查显式条件，不能证明所有描述都语义正确。

`prepare_b512_candidates.py` 的 PixMo 适配已保留原 caption。其他来源按其冻结版本转换为同一接口，不能声称所有远端入口已经下载或全部来源适配已经验收。

## 独立下载与 512px 预处理

以下路径示例是一批候选；后续补图用新的 cohort 名。`--config` 若需要覆盖，放在子命令之前。

```bash
SYNTH_PROJECT=/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/unified-mm-b512
SYNTH_SUPPLY="$SYNTH_PROJECT/preparation/supply/cohort-001"
SYNTH_PREP="$SYNTH_PROJECT/preparation/prepared/cohort-001"
SYNTH_POOL="$SYNTH_PROJECT/images"

.venv/bin/python -m scripts.synthesize_image_text ingest \
  --manifest <normalized-candidates.jsonl> --supply-root "$SYNTH_SUPPLY" --prefix selected
.venv/bin/python -m scripts.synthesize_image_text seal-candidates --supply-root "$SYNTH_SUPPLY"
```

封闭清单后，下面两个命令在两个独立进程/终端同时运行。下载默认 64 并发、每 host 8；预处理默认 8 个 CPU 进程。参数可按实际吞吐调整。下载以 256 图批次持续交给预处理，最多积压 32 个批次，不等待文本合成。下载结束仍需等全部预处理结束。

```bash
.venv/bin/python -m scripts.synthesize_image_text download \
  --supply-root "$SYNTH_SUPPLY" --image-root "$SYNTH_POOL" --workers 64 --per-host 8
```

```bash
.venv/bin/python -m scripts.synthesize_image_text prepare \
  --supply-root "$SYNTH_SUPPLY" --root "$SYNTH_PREP" --image-root "$SYNTH_POOL" \
  --workers 8 --exclude <benchmark-identities.txt>
```

`download_status.json`、`prepare_status.json` 分别统计成功、失败和预处理验收数量。下载器每轮最多导入 4 个候选批次并提交状态，再调度网络请求，避免大标注库阻塞首批下载或已有连接；完整封闭清单仍会逐步导入，不构成图片数上限。失败 URL 与低质量图片不计入目标，按缺失能力补新 cohort。每批发布独立原图归档和视图归档，避免每图小文件；本地原图优先引用，不重复下载。

PixMo-Points 使用 [pixmo_points.py](../data_synthesis/pixmo_points.py) 的 `point_rows(..., compute_hashes=false)` 归一化固定版本的原始 train Parquet。上游元数据已经声明同一 URL 有多个图片版本时，仍保留原来的冲突隔离记录，不重新计算图片哈希。下载不再因实际图片 SHA 与上游字段不同而拒绝；旧的此类失败已单独重入有界重试队列。原始点坐标保持不变并标为 `upstream_raw` / 未验证，不能直接充当 512px 视图坐标或已验证计数。未绑定最终视图的标注仍需补缺/纠错，不能因跳过校验自动变成可信监督。

大文件/归档的下载器仍是 `scripts.download_b512_corners_v3`，现在只接受**显式选取且版本固定**的 `--catalogue`，或续跑该 root 已冻结的 catalogue；不再自动扩展成旧版全来源下载。字段为 `files[{id,source,path,url,revision,bytes,sha256,reuse_existing}]` 和 `declared_bytes`，`path` 为绝对路径；保留 Range 断点、上游校验和和完成回执。客户端使用独立 P-256 / TLS 1.2 / HTTP/1.1 直连池，忽略环境代理并保持证书校验，连接最多空闲保留 60 秒以减少重复握手。连接中断后按落盘的实际字节续传，每个分片最多尝试 6 次，退避加入随机间隔，避免并发请求同时重试；重启优先续传失败文件，并将旧进程遗留的运行状态恢复为待处理。状态记录 `http2=false`、`tls_version=TLSv1.2`，重启继承上次落盘的传输字节计数；计数包含重试流量，不能当作已验收体积。归档完成不代表其所有图片已通过 512px 验收。旧暂停任务不会因新入口自动恢复。

## 冻结、复用和模型补缺

`freeze` 导入封闭 inbox（`--inbox` 可重复传入多个完成 cohort）、带 `batch.json` 的 `--prepared-run` 或已验收 `--release`。低于至少 200 万时返回 `needs_more_images`，保留已导入内容；补足后再冻结。只按已知 ID、URL 和原图引用连接、去重及合并标注，不计算内容 SHA/pHash；不同身份的相同像素不会被内容去重。首次输入顺序决定原图主归属，专项来源先导入，其他来源作为别名和标注保留。

```bash
.venv/bin/python -m scripts.synthesize_image_text freeze \
  --inbox "$SYNTH_PREP/input_queue" \
  --release public/datasets/unified_image_text_512_sol_v1/releases/sol_100k_plus_corners_v1 \
  --exclude <benchmark-and-imagenet-exclusions.txt> \
  --exclude-prompts <test-prompts.jsonl>
```

正式范围保留评测身份和测试 prompt 排除文件。`compute_hashes=false` 时不读取评测 pHash 索引，也不要求 `--near-exclude-index`；只排除已知身份，不声称完成评测内容近重复排除。prompt 文件每行可为纯文本、JSON 字符串或 `{"prompt":"..."}`，按 Unicode/空白/大小写归一后的原文本比较，不计算文本哈希。`freeze` 不自行猜测 benchmark 划分。只在开发小样本上使用 `--pilot`，其发布明确标为 pilot。

路由与默认限制从 runtime JSON 读取：

| 路由 | 触发 | 是否调用模型 |
| --- | --- | --- |
| reuse / normalize | 可信原 caption 与全幅几何、必要可读性检查一致，或已有精确 view 的合格 pair | 不调用；只允许换行和首尾空白整理 |
| annotation | 可信最终视图正标注可直接转成描述 | 不调用 |
| sii | 缺 caption、裁剪错位、不可验证原 prompt、事实冲突、关键条件缺失、超长文本 | SII `deepseek-v4.1-flash`，此历史复用入口每图最多 3 次；当前全图复核为 4 次 |
| codex_fallback | 同一图的 SII 尝试次数已经耗尽 | 当前关闭；仅后续用户单独启动时使用现有 Codex CLI |
| failed / quarantined | 最终仍无法产生合格配对文本 | 保留原始响应与错误，显式隔离，不伪装成完成 |

SII 从并发 256 自适应增加至 512，共用 HTTP 连接池，受 RPM=1,200 / TPM=6,000,000 的保守请求节奏限制；输出最多 3,200 tokens。连续传输失败触发熔断、降并发和单探测恢复；鉴权/端点错误停止补入新请求。重启不清零尝试次数，原始响应先持久化再校验；已接收响应在恢复时本地重放，不重复付费。Codex 配额耗尽保留待处理队列，需后续显式运行；不按日期重置单图失败上限。

SII 合格结果直接采用，**不逐图再调用 sol 审核**。同一忠实 caption 可以同时用于 I2T/T2I；无最低字数要求，不为不同措辞额外生成两遍。每条最长 960 个实际 tokenizer token，图片为固定 RGB 512×512。API 提示仅带少量相关候选和已验证事实，完整原始标注/作者/响应存状态库。

## 一次性前置资格验收

2026-09-15 起，当前模型为 SII `deepseek-v4.1-flash`。真实图片请求与错误 caption 对照已验证该入口能利用图片；用户明确要求全量切换，按 `user_accepted_for_generation` 保存授权与尚未通过的语义质量结论。旧 `deepseek-v4-pro-0813` 的不可看图结论不适用于这个新模型。当前全图复核入口与运行配置见 [DeepSeek 全量迁移](B_DEEPSEEK_FULL_REVIEW_20260915.md)。下列资格验收命令属于历史复用入口，HTTP 200、结构合格与事实质量仍分别统计。

```bash
.venv/bin/python -m scripts.probe_b512_api_models smoke --concurrency 2 --root <pilot-root>
.venv/bin/python -m scripts.probe_b512_api_models pilot --concurrency 4 --root <pilot-root>
```

探测使用与生产相同的客户端、图像字节、prompt/schema；32 图覆盖关系、2–4 计数、5+ 计数、中英文文字、绘画、图形风格和不确定场景。GPT-6 查看实际附件与输出一次性评审，报告列出事实错误、遗漏和各桶结果，再明确模型是否 qualified。`protocol.json` 给出 `runtime_sha256` 与 `contract_hash`，只有真实评审通过后才填写资格报告：`status=passed, reviewer=GPT-6, image_size=512, visual_review=true, models[模型名].qualified=true`，并绑定这两个 hash。更换模型、提示、TLS 或其他运行合同需要新报告/新状态目录，不能沿用旧 JSON 的通过标记。

```bash
.venv/bin/python -m scripts.synthesize_image_text run --qualification <qualification.json>
.venv/bin/python -m scripts.synthesize_image_text status
```

开发小池最多处理 256 图，默认单次 32；可设 `--max-items`。SIGINT/SIGTERM 停止补入任务并收完有界在途请求；未完整接收的尝试保留为 interrupted。资格报告是前置抽查证据，不代表百万级逐图语义审核。

## 发布、KL16 与训练绑定

所有输入必须完成或显式隔离才允许发布；无静默部分成功。`quarantine-failed` 仅处理已经耗尽模型尝试的终态失败，保留全部历史。正式发布在隔离后仍需满足至少 200 万非 ImageNet 合格图，否则补充一个新范围，不通过删行伪造完成。

```bash
.venv/bin/python -m scripts.synthesize_image_text quarantine-failed
.venv/bin/python -m scripts.synthesize_image_text export \
  --output /inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/unified-mm-b512/texts/releases/<release-name>
.venv/bin/python -m scripts.synthesize_image_text audit --dataset <release-path>
```

发布 `manifest.jsonl`、`captions.jsonl`、分片 T2I、N+1 offsets 和 `text_index.json`，兼容现有真实训练索引；默认每个 T2I shard 250,000 行。审计逐条核对冻结视图解码、图像引用、映射、文本/token、复用作者与原文、原始模型响应和 Codex 失败前置条件。文件记录字节数，不计算哈希；审计模式为 `id_reference_size_decode`。发布原子完成且不可覆盖。状态 SQLite/原始响应是审计依赖，保留在 `state_root`，迁移发布时一并迁移并校验路径。

图像编码独立于文本合成；池冻结后即可：

```bash
.venv/bin/python -m scripts.synthesize_image_text prepare-bank --output <bank-manifest-directory>
```

现有 `scripts.encode_b512_banks --plan`、`scripts.encode_b512_supply` 和 Ascend 加速包装入口读取当前策略，向 encoder/merge 传 `--no_hash`，不传 `--verify_view_hashes`。`scripts.compose_b512_posterior_index` 按同一冻结 `view_id`/图像引用连接 bank 与新发布，**不复制 posterior 张量**；VAE 路径、缩放、dtype、shape、行 ID、有限值与非负标准差仍要一致。旧缓存仅在引用及 VAE 合同兼容时复用。`scripts.synthesize_image_text audit --require-posterior --dataset ...` 完成全量检查后，再运行实际 I2T/T2I loader 验收。

512px 预处理是 CPU 解码、方向校正、等比缩放/补边并保存最终视图；posterior cache 是随后用 KL16 VAE 编码这些视图得到的 mean/std 张量，两者是独立阶段。关闭哈希不等于已经完成 posterior 编码。

模型每图 1,024 个、每 token 16 通道，posterior 存 mean+std 为 `[N,1024,32]`，序列长度 2,048。当前 [512px YAML](../configs/selfless/unified_b_x0_images512_v1_ascend64.yaml) 是 9 月 12 日首轮数据/分辨率验收配置，仍指向历史 205,755 图，并非已完成大模型训练配置。新训练配置应按最终发布同时绑定 `cache_path/manifest_jsonl/caption_jsonl/synthetic_text_index_manifest/expected_records`，再跑真实 loader 检查；ClimbMix 和正式旧消融配置不变。实际模型规模、任务比例、来源采样权重和训练预算在发布量确定后单独冻结。

## 目录与兼容性

| 用途 | 路径 |
| --- | --- |
| 当前代码与配置 | `data_synthesis/`、`configs/data_synthesis/b512_sii_v1.json` |
| 图片/视图 | 项目目录 `unified-mm-b512/images/` |
| 当前准备/运行状态 | 项目目录 `unified-mm-b512/preparation/` |
| 当前文本发布 | 项目目录 `unified-mm-b512/texts/` |
| posterior tensor 缓存 | `public/datasets/unified_b512_posterior_cache/` |

`encode_b512_supply` 与 `accelerate_b512_posteriors plan` 使用独立 `--posterior-root`，默认读取 runtime 中的全局缓存目录，不能再用 `--image-root` 表达缓存落点。文本默认每片 250,000 条，现有 uint8 分片编号可覆盖 6,400 万行，不把主库截断为原默认的 2,560 万行。

迁移前为已有 supply 写入 `cohort_identity.json`，之后以持久 ID 命名批次和归档；迁移不重写不可变的清单内容。恢复入口通过兼容链接定位旧路径，prepare 的 owner 比较使用实际目录，防止物理迁移误判成另一个数据源。
| 首轮可复用发布 | `public/datasets/unified_image_text_512_sol_v1/releases/sol_100k_plus_corners_v1/` |
| 原始 ImageNet | `public/dataset/imagenet/v1/` |
| 历史纯 Codex / Qwen27B 全量评审实现 | `scripts/legacy/`，只用于历史重放和审计 |
| 历史旧消融数据 | [DATA.md](DATA.md) 和 [网页的数据与协议](../output/evaluation/index.html#sources) |

旧状态不原地改造成新状态，不覆盖旧结果和教师记录。`scripts.distill_b512_codex` 仅保留历史 export / recover-completed；新生产统一进入 `scripts.synthesize_image_text`。来源适配与独立下载/编码脚本是可复用组件，不再自行决定“全图重写/全图教师复审”策略。

## 2026-09-14 全量复核试跑

2026-09-15 用户要求全部迁移到 SII `deepseek-v4.1-flash`，旧 Qwen 结果原样保留。当前入口为 `scripts.review_b512_sii`，配置为 `configs/data_synthesis/b512_deepseek_v41_flash_bulk_c512_20260915.json`，独立处理固定的 22,991,115 张图片，所有图片均真实附图复核。新结果保存在项目 `preparation/review/deepseek_v41_flash_bulk_c512_20260915/`，不混入旧 Qwen 结果。并发从 256 自适应至 512；不算哈希、不下载新图、不调用 Codex CLI。SII 使用 shell 的 `SII_API_KEY`/`SII_BASE_URL`、客户端直连与连接前 TCP MSS=512，全局代理保持。语义问题继续留给后续单独阶段，记录用户接受而不伪造质量通过。详见 [DeepSeek 全量迁移](B_DEEPSEEK_FULL_REVIEW_20260915.md)；[Qwen 试跑](B_SII_FULL_REVIEW_20260914.md)仅为历史证据。

## 本次重构的实测边界（2026-09-13）

真实旧数据取三个不同非 ImageNet 来源，已完成原文免调用复用、分片发布、原图/视图哈希审计、旧 posterior 按 view hash 复用，以及实际 I2T/T2I loader 全行检查。样例明确标记 pilot，不计为新增 200 万发布：[回放验收](../public/data_preparation/unified_b_corners_api_v3/refactor_loader_smoke_20260913/receipt.json)。I2T/T2I 最长实际序列为 1,162 / 1,109，均在 2,048 内，未执行模型 forward。

SII 直连的 P-256 / TLS 1.2 / HTTP/1.1 控制请求取得 200，日志显示真实图片请求的 TLS 握手与客户端上传阶段也能完成；真实 512px 自然图片的完整响应仍超时。约 38KB、含同一简单控制图及 JSON 空白填充的请求可返回 200，因此不能仅按总请求大小或代理配置解释剩余故障。当前仍未取得完整 512px 成对输出的模型资格报告，不将控制图成功视为自然图语义质量通过。原始记录位于 `public/data_preparation/unified_b_corners_api_v3/refactor_sii_tls_diagnostics_20260913/`，生产格式两图探测位于 `refactor_sii_tls12_20260913/`。

当前测试覆盖断点重放、SII 重试、Codex 上限、信号排空、失败隔离、跨来源/跨批次原图保护、文本发布和真实 loader 身份映射。每个补图 cohort 的原图、512px 视图及 VAE 归档有独立命名空间，避免下一批从编号零开始时覆盖上一批。

当前统一进度入口：`.venv/bin/python -m scripts.status_b512_data`，`--watch` 每 30 秒更新项目 `preparation/status.json`。归档下载量、URL 成功数、512px 预处理数分别报告，不混作去重后的最终训练图片数。
