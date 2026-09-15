# 当前 B512 文本流程：Long 原文主干与定向 DeepSeek 修复

2026-09-15 用户确定：**Long 原文为主干，专项数据负责补能力，DeepSeek 只处理明确需要补缺或纠错的部分。** 此协议替代全量逐图 API 复核，图像池仍是已封存的 22,991,115 张 512px 视图。

## 路由

| 输入情况 | 行为 |
| --- | --- |
| Long 已有原 caption，图像身份、全幅几何、来源、文本格式与长度合格 | 保留原文，允许整理首尾空白与换行；同一描述用于 I2T/T2I，不调用 API |
| 专项来源已有符合相同条件的 caption | 同样直接复用 |
| 有明确绑定最终视图的可信标注，可安全转成文本 | 使用确定性标注转换，不调用 API |
| caption 缺失、无来源、身份不符、裁剪导致不对齐、超长/格式损坏、已确认事实错误 | 加入定向修复队列 |
| 定向修复项已有该图对应的 DeepSeek 合格响应 | 验证原始回复与图片绑定后复用，保留历史响应来源 |
| 没有可复用修复结果 | SII `deepseek-v4.1-flash` 真实附图处理，每图最多 4 次 |
| 同一样本连续两次 SII 超时 | 根据用户追加授权，转交独立 Codex CLI 队列，固定 `gpt-5.6-sol` / `low`；不等待 SII 第四次尝试 |
| 其他 SII 失败或 Codex 尝试失败 | 保存错误及所有尝试，留待后续阶段 |

Long 占 22,115,459 张，专项来源合计 875,656 张。免调用数量由实际本地检查决定，不预先声称全部 Long 均合格；专项来源也不一律调用 API。基础图像解码与 512px 筛选沿用封存时的结果，不为直接复用重新扫描所有图像字节。

复用检查不等于逐图语义审核。输出明确保留 `semantic_accuracy_independently_verified=false`，Long 文本作者仍是上游 `Qwen/Qwen2.5-VL-7B-Instruct`，不会改标为 DeepSeek。caption 中的数字不会自动转成精确计数标签。

可信计数标注需要最终视图绑定、目标完整可见、指定对象标注穷尽等条件；文字标注需要最终视图可读性。PixMo-Points 等尚未验证坐标或可见性的标注只作为 API 注意力提示，不能直接作为答案。原始大标注仍保留在源状态库，不反复发送到 API。

已人工确认的问题登记在 `configs/protocols/b512_known_caption_repairs_20260915.jsonl`，用原任务 key 和精确 view_id 指定。规则在运行目录中冻结，修改规则后需新运行版本。不要仅凭 caption 出现数字或引号就将所有 Long 送回模型。

## 并行实现与恢复

入口为 `scripts.synthesize_b512_reuse`，实现位于 `data_synthesis/reuse_farm.py`，配置为 `configs/data_synthesis/b512_reuse_first_20260915.json`。默认 `b512_sii_v1.json` 和 `data_synthesis.cli run` 也路由到此流程。

本地阶段有 **8 个 CPU worker**，按固定源批次分片处理，流式保存免调用结果和待修复队列。先启动各来源的代表分片，让专项修复尽早接入；之后继续固定顺序。API 阶段独立消费已落盘队列，不阻塞本地处理 Long。多个来源分片的待修复项会合并成约 2,048 张的调度组，短队列最多等待 5 秒，避免小分片让并发长期空闲；API 本身仍逐图请求。

API 并发起点 **256**、自适应上限 **512**，仍为单图请求；RPM=1200、TPM=6,000,000、180 秒超时及传输熔断保持。新的请求版本是 `b512-reuse-first-targeted-deepseek-v2`，原有复核版本保留用于历史结果检查。

2026-09-15 并发恢复修复使用 `sii-circuit-generation-v2`：每次请求绑定调度代次，一轮相关超时只开启一次熔断；该轮晚到的回复仍保存，但不能反复减半并发、延长等待或提前关闭新一轮熔断。只有恢复探测再次失败才延长退避；内容校验错误不降低网络并发。

用户随后明确要求“反复超时的直接用codex cli进行”“调用GPT-5.6-sol low”。当前根的 `timeout_fallback_policy.json` 将这项授权作为独立、不可变的追加规则，原始 SII-only 接受记录仍保留为历史证据。`data_synthesis/timeout_fallback.py` 在同一控制器中运行独立的 8 路 Codex 队列；只有连续两次实际传输超时才转交，HTTP 429、其他接口错误和内容错误不因此转交。每条 Codex 最多一次、300 秒超时，真实附图，检查模型/low 参数、退出状态、事件流末条消息和输出身份。超时样本离开 SII 活动批次，Codex 执行不阻塞后续 SII 批次。

全局运行锁、固定配置/选择范围/修复规则、分片原子写入与计数校验共同约束恢复。已封存本地分片跳过；已收到 API 回复优先本地重放；重试次数不清零。不计算数据哈希，不重新下载，不改全局代理。

SII 连接只读取环境或静态 bashrc/zshrc 中的 `SII_API_KEY` / `SII_BASE_URL`。客户端 `trust_env=false`，连接前 TCP MSS=512、TLS 1.2 上限和证书校验保持，不读取 `test_api.py`。

## 当前路径

项目根：`/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/unified-mm-b512`。

| 内容 | 项目根下路径 |
| --- | --- |
| 当前运行根 | `preparation/synthesis/reuse_first_20260915/` |
| 本地结果与修复队列 | `routing/编号/reused.jsonl.gz`、`repair_inputs.jsonl.gz`、`manifest.json`（相对当前运行根） |
| API 原始响应、重试、结果 | `repairs/编号/review.sqlite3`、`reviewed.jsonl.gz`、`manifest.json`（相对当前运行根） |
| 连续超时的 Codex 接管 | `timeout_fallback/review.sqlite3`、`status.json`、`reviewed.jsonl.gz`、`manifest.json`；父 SII 响应通过 `timeout_handoff.evidence_db` 引用 |
| 当前路由与 API 进度 | 当前运行根 `status.json`；活动 API 分片另有 `status.json` |
| 固定输入 | `operations/frozen_downloads_20260914/prepared_batches.jsonl` |
| 旧 DeepSeek 全图复核 | `preparation/review/deepseek_v41_flash_bulk_c512_20260915/`，已停机，保留历史结果 |
| 旧 Qwen 全图复核 | `preparation/review/qwen38max_bulk_20260914/`，保留历史结果 |

本地 `reused.jsonl.gz` 中的 `pair` 为直接复用/标注结果，`evidence` 保存原文来源或历史响应指针；API 导出中的 `result.pair` 为定向修复结果。最终根 `manifest.json` 列出两类分片，下一阶段统一整理成训练发布格式。

`routed_images` 是已经做完本地决策的图片数；`locally_published_images` 是免新增 API 调用且已保存的结果数；`routes.needs_sii` 是送入 API 队列的图片数。`completed_images` 还会计入已封存 API 分片中的失败/不可用记录，应结合 `repair_finalized_counts` 判断可训练数量。未封存 API 结果查活动分片，不能把待修复队列数量称为已合成数量。

`deferred_codex` 表示转交，不能算作 SII 成功。根 `timeout_fallback` 字段报告独立队列的实际并发、调用数和结果数；训练发布时按 key 用成功的 Codex 结果补充对应的转交/失败项，保留两份原始记录，不能将其作者标成 DeepSeek。既有 SII 成功项不再调用 Codex。全量结束需同时等待 SII 与 Codex 队列完成，根清单引用 `timeout_fallback/manifest.json`。

全量结束后保留失败、不可用与原始证据，状态如实为 `completed` 或 `needs_attention`。不自动开始 posterior 编码、训练或超出连续超时范围的 Codex 修复。当前视图总数沿用封存口径，尚未声称完成跨来源身份合并；训练发布准备仍为下一阶段。

## 启动和恢复

先检查活动注册文件，确认不存在仍在运行的同一控制器。保持原配置恢复：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -u -m scripts.synthesize_b512_reuse \
  --config configs/data_synthesis/b512_reuse_first_20260915.json

.venv/bin/python -m data_synthesis.cli status
```

默认根目录的 `user_acceptance.json` 保存本次用户指令及准确范围；真实抽查中已知的语义问题仍保留，不伪造模型通过结论。旧全图复核的启动命令属于历史维护接口，不用于当前生产任务。

当前生产运行已保存超时追加授权，按原入口恢复即可。新运行不会自动开启该能力：只有根目录中存在与自身范围绑定的 `timeout_fallback_policy.json` 才启用。Codex 子进程保留代理且不继承 SII key；SII 的 `trust_env=false` 继续绕过代理。无需关闭全局代理。

## 实测与验证

本次相关回归测试 **52 项通过**，覆盖原文/作者保留、身份和裁剪拦截、可信标注条件、历史 DeepSeek 响应复用、修复队列合并、导出和重启后不重复调用。真实八图试跑保存六条原文复用、一条 SII 修复及一条达到尝试上限的失败，均保留完整记录。

并发恢复与超时接管追加了故障代次、固定 Codex 参数/事件、连续超时限定、原始证据保存和恢复不重复计费的测试；全套 68 项通过，再加入包含 Codex 转交的整流程恢复用例后，27 项受影响测试通过，当前共 69 个相关用例。两条历史真实超时样本已用 `gpt-5.6-sol` / `low` 完成并保存，继续运行时直接复用。小样本结构/来源验收不等于全库语义正确。

生产启动后已核实超过百万条免调用文本落盘，定向 DeepSeek 请求真实附图、返回模型一致且客户端无代理。当前具体进度读取运行根 `status.json`；`migration_completion_verification.json` 保存迁移验收时的快照。此处不将局部处理完成表述为全量训练集已发布。
