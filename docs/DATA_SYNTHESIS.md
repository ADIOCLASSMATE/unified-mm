# 当前 B512 数据准备与合成

当前入口是 `python -m scripts.synthesize_image_text`，实现位于 `data_synthesis/`。运行参数只有一份：[b512_sii_v1.json](../configs/data_synthesis/b512_sii_v1.json)。[来源选择协议](B_2M_NON_IMAGENET_DATA_PLAN_20260913.md)和[配额 JSON](../configs/protocols/unified_b_non_imagenet_2m_v1.json)定义覆盖目标；本文定义实际执行方式。

## 训练范围与阶段

非 ImageNet 图库以**去重、512px 验收且最终图文合格后至少 200 万张原图**为基线，各来源配额不是硬上限。完整本地 ImageNet train 1,281,167 张是后续底座，当前先完成额外图库；ClimbMix 沿用现有语料。合格首轮非 ImageNet 105,322 张计入目标，不重复计数。来源数量、候选 URL 数、合成文本数和训练采样权重分别记录。

| 来源 | 起始合格图目标，非上限 | 优先复用和精选入口 |
| --- | ---: | --- |
| PixMo-Cap | 500,000 | 原人工 caption，场景/动作/属性分层 |
| Open Images 关系 + Localized Narratives | 300,000 | 同原图人工叙述、框与关系；关系专项优先 |
| PixMo-Points | 180,000 | 同图已有描述、完整点标注；不能假设大部分有 Cap caption |
| AnyWord-3M | 350,000 | 已有文字、区域和描述，约中文 200K / 英文 150K |
| JourneyDB train 风格池 | 300,000 | 原 prompt 是候选；访问及原始 train 映射待落实，MONET synthetic 可同风格补位 |
| WikiArt | 70,000 | 媒介/风格/题材元数据；标题不自动当完整 caption |
| BLIP3o + ShareGPT-4o-Image T2I | 80,000 | 保留真实实现的内容，检查原 prompt 与图片一致性 |
| TextCaps + DOCCI train | 30,000 | 尽量完整复用合格 train 人工 caption |
| PixMo-Docs / ChartQA train | 40,000 | DIM caption 精选标注可连接原图；只收 512px 可读视图 |
| CC12M + PD3M | 150,000 | CC12M 优先 MONET 精选分支；补缺失主题，PD3M 约 50K |

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

- `caption_candidates` 保留原文本、作者、字段与版本；`kind` 可为 `human_caption`、`curated_caption`、`generation_prompt`。已有已验收发布通过 `freeze --release` 导入，生成绑定精确 view SHA256 的 `accepted_pair`，不重新调用教师。
- `parent_id` 和 `identity_aliases` 用于跨来源原图身份连接；同图多种标注保留。相同身份的标注行在一次 `ingest` 中使用磁盘 SQLite 聚合；冲突的 URL、视图或其他标量字段报错。跨 intake 的相同来源身份必须先合并，封闭清单时检查，避免下载时丢失额外标注。
- `annotations` 保存上游点、框、OCR 等原始标注。尚未经最终视图检查的原始标注不能标为 verified。规范化适配器应使用现有元数据、几何和可读性检查，不逐图调用 GPT 做入口筛选。
- 默认 `fit_pad` 保持全幅。预处理记录原图 SHA256、EXIF 方向、crop、512px view SHA256、编码与原图引用；计数和文字优先保留全部对象。被裁掉、缩小后不可读的内容不作确定监督。
- OCR/文档 caption 复用要求 `readability_view_sha256` 对应最终可读视图。原始 generation prompt 即使附了 view hash 也不能因此直接免审；只有已验收的该视图文本可直接复用。
- 可确定的正标注用 `verified_facts`：需要 `verified=true`、当前 `view_sha256` 和 `provenance`。计数还需 `entity/count/fully_visible/exhaustive_for_referent`，文字需 `text/carrier/readable`，关系需明确 `text`。标签不完整不能推断数量为零；字段不足则进入 API 补缺队列。
- 可用 `required_fact_text` 声明原 caption 必须含有的已知事实字符串。规则仅检查显式条件，不能证明所有描述都语义正确。

`prepare_b512_candidates.py` 的 PixMo 适配已保留原 caption。其他来源按其冻结版本转换为同一接口，不能声称所有远端入口已经下载或全部来源适配已经验收。

## 独立下载与 512px 预处理

以下路径示例是一批候选；后续补图用新的 cohort 名。`--config` 若需要覆盖，放在子命令之前。

```bash
SYNTH_SUPPLY=public/data_preparation/unified_b_corners_api_v3/supply/cohort-001
SYNTH_PREP=public/data_preparation/unified_b_corners_api_v3/prepared/cohort-001
SYNTH_POOL=public/datasets/unified_image_pool_512_v3

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
  --workers 8 --exclude <benchmark-identities-and-hashes.txt> \
  --near-exclude-index <benchmark-phash-index>
```

`download_status.json`、`prepare_status.json` 分别统计成功、失败和预处理验收数量。失败 URL 与低质量图片不计入目标，按缺失能力补新 cohort。每批发布独立原图归档和视图归档，避免每图小文件；本地原图优先引用，不重复下载。

大文件/归档的下载器仍是 `scripts.download_b512_corners_v3`，现在只接受**显式选取且版本固定**的 `--catalogue`，或续跑该 root 已冻结的 catalogue；不再自动扩展成旧版全来源下载。字段为 `files[{id,source,path,url,revision,bytes,sha256,reuse_existing}]` 和 `declared_bytes`，`path` 为绝对路径；保留 Range 断点、上游校验和和完成回执。归档完成不代表其所有图片已通过 512px 验收。旧暂停任务不会因新入口自动恢复。

## 冻结、复用和模型补缺

`freeze` 导入封闭 inbox（`--inbox` 可重复传入多个完成 cohort）、带 `batch.json` 的 `--prepared-run` 或已验收 `--release`。低于至少 200 万时返回 `needs_more_images`，保留已导入内容；补足后再冻结。身份/SHA256 精确去重合并标注；图库近重复用 pHash 候选加严格 RGB 相似性筛查，近重复图不跨图复制标注。首次输入顺序决定原图主归属，专项来源先导入，重复原图的其他来源作为别名和标注保留。

```bash
.venv/bin/python -m scripts.synthesize_image_text freeze \
  --inbox "$SYNTH_PREP/input_queue" \
  --release public/datasets/unified_image_text_512_sol_v1/releases/sol_100k_plus_corners_v1 \
  --exclude <benchmark-and-imagenet-exclusions.txt> \
  --near-exclude-index <benchmark-phash-index> --exclude-prompts <test-prompts.jsonl>
```

正式范围必须有评测身份/内容 hash、评测近重复索引和测试 prompt 排除文件。prompt 文件每行可为纯文本、JSON 字符串或 `{"prompt":"..."}`，按 Unicode/空白/大小写归一后的精确文本排除；不声称覆盖所有改写和衍生图。`freeze` 不自行猜测 benchmark 划分。只在开发小样本上使用 `--pilot`，其发布明确标为 pilot。

路由与默认限制从 runtime JSON 读取：

| 路由 | 触发 | 是否调用模型 |
| --- | --- | --- |
| reuse / normalize | 可信原 caption 与全幅几何、必要可读性检查一致，或已有精确 view 的合格 pair | 不调用；只允许换行和首尾空白整理 |
| annotation | 可信最终视图正标注可直接转成描述 | 不调用 |
| sii | 缺 caption、裁剪错位、不可验证原 prompt、事实冲突、关键条件缺失、超长文本 | SII `qwen3.8-max`，每图最多 3 次持久化尝试 |
| codex_fallback | 同一图的 SII 尝试次数已经耗尽 | 现有 Codex CLI，`gpt-5.6-sol` / low，最多 1 次；并发 2，每次运行最多 128 图 |
| failed / quarantined | 最终仍无法产生合格配对文本 | 保留原始响应与错误，显式隔离，不伪装成完成 |

SII 从并发 16 自适应增加至 128，共用 HTTP 连接池，受 RPM=1,200 / TPM=6,000,000 的保守请求节奏限制；输出最多 3,200 tokens。连续传输失败触发熔断、降并发和单探测恢复；鉴权/端点错误停止补入新请求。重启不清零尝试次数，原始响应先持久化再校验；已接收响应在恢复时本地重放，不重复付费。Codex 配额耗尽保留待处理队列，需后续显式运行；不按日期重置单图失败上限。

SII 合格结果直接采用，**不逐图再调用 sol 审核**。同一忠实 caption 可以同时用于 I2T/T2I；无最低字数要求，不为不同措辞额外生成两遍。每条最长 960 个实际 tokenizer token，图片为固定 RGB 512×512。API 提示仅带少量相关候选和已验证事实，完整原始标注/作者/响应存状态库。

## 一次性前置资格验收

`qwen3.8-max` 是当前视觉候选；`deepseek-v4-pro-0813` 此前多次有效附图请求返回无法看到图片，保留为待单独验证的文本整理候选，不进入视觉合成队列。HTTP 200、JSON 合法与事实质量分别统计，尚未取得 512px 成对输出的分层资格报告前，不放行正式批量合成。

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
  --output public/datasets/unified_image_text_512_api_v3/releases/<release-name>
.venv/bin/python -m scripts.synthesize_image_text audit --dataset <release-path>
```

发布 `manifest.jsonl`、`captions.jsonl`、分片 T2I、N+1 offsets 和 `text_index.json`，兼容现有真实训练索引；默认每个 T2I shard 10 万行。审计逐条核对图片字节、原图 hash、映射、文本/token、复用作者与原文、原始模型响应和 Codex 失败前置条件。发布原子完成且不可覆盖。状态 SQLite/原始响应是审计依赖，保留在 `state_root`，迁移发布时一并迁移并校验路径。

图像编码独立于文本合成；池冻结后即可：

```bash
.venv/bin/python -m scripts.synthesize_image_text prepare-bank --output <bank-manifest-directory>
```

现有 `scripts.encode_b512_banks --plan`、`scripts.encode_b512_supply` 继续编码同一冻结 512px 视图；已验收旧缓存用 view SHA256 复用。`scripts.compose_b512_posterior_index` 将多个 bank 与新发布逐图连接，**不复制 posterior 张量**。`scripts.synthesize_image_text audit --require-posterior --dataset ...` 完成全量检查后，再运行 `scripts.audit_b512_training_loaders --dataset ... --publication-audit ... --output ... --config ...`。

模型每图 1,024 个、每 token 16 通道，posterior 存 mean+std 为 `[N,1024,32]`，序列长度 2,048。当前 [512px YAML](../configs/selfless/unified_b_x0_images512_v1_ascend64.yaml) 是 9 月 12 日首轮数据/分辨率验收配置，仍指向历史 205,755 图，并非已完成大模型训练配置。新训练配置应按最终发布同时绑定 `cache_path/manifest_jsonl/caption_jsonl/synthetic_text_index_manifest/expected_records`，再跑真实 loader 检查；ClimbMix 和正式旧消融配置不变。实际模型规模、任务比例、来源采样权重和训练预算在发布量确定后单独冻结。

## 目录与兼容性

| 用途 | 路径 |
| --- | --- |
| 当前代码与配置 | `data_synthesis/`、`configs/data_synthesis/b512_sii_v1.json` |
| 图片/视图/posterior | `public/datasets/unified_image_pool_512_v3/` |
| 当前准备/运行状态 | `public/data_preparation/unified_b_corners_api_v3/` |
| 当前发布根目录 | `public/datasets/unified_image_text_512_api_v3/` |
| 首轮可复用发布 | `public/datasets/unified_image_text_512_sol_v1/releases/sol_100k_plus_corners_v1/` |
| 原始 ImageNet | `public/dataset/imagenet/v1/` |
| 历史纯 Codex / Qwen27B 全量评审实现 | `scripts/legacy/`，只用于历史重放和审计 |
| 历史旧消融数据 | [DATA.md](DATA.md) 和 [网页的数据与协议](../output/evaluation/index.html#sources) |

旧状态不原地改造成新状态，不覆盖旧结果和教师记录。`scripts.distill_b512_codex` 仅保留历史 export / recover-completed；新生产统一进入 `scripts.synthesize_image_text`。来源适配与独立下载/编码脚本是可复用组件，不再自行决定“全图重写/全图教师复审”策略。

## 本次重构的实测边界（2026-09-13）

真实旧数据取三个不同非 ImageNet 来源，已完成原文免调用复用、分片发布、原图/视图哈希审计、旧 posterior 按 view hash 复用，以及实际 I2T/T2I loader 全行检查。样例明确标记 pilot，不计为新增 200 万发布：[回放验收](../public/data_preparation/unified_b_corners_api_v3/refactor_loader_smoke_20260913/receipt.json)。I2T/T2I 最长实际序列为 1,162 / 1,109，均在 2,048 内，未执行模型 forward。

SII 直连的 P-256 / TLS 1.2 / HTTP/1.1 控制请求取得 200，日志显示真实图片请求的 TLS 握手与客户端上传阶段也能完成；真实 512px 自然图片的完整响应仍超时。约 38KB、含同一简单控制图及 JSON 空白填充的请求可返回 200，因此不能仅按总请求大小或代理配置解释剩余故障。当前仍未取得完整 512px 成对输出的模型资格报告，不将控制图成功视为自然图语义质量通过。原始记录位于 `public/data_preparation/unified_b_corners_api_v3/refactor_sii_tls_diagnostics_20260913/`，生产格式两图探测位于 `refactor_sii_tls12_20260913/`。

当前测试覆盖断点重放、SII 重试、Codex 上限、信号排空、失败隔离、跨来源/跨批次原图保护、文本发布和真实 loader 身份映射。每个补图 cohort 的原图、512px 视图及 VAE 归档有独立命名空间，避免下一批从编号零开始时覆盖上一批。
