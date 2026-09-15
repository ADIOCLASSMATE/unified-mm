# B512 当前图像池：2026-09-14 封存

用户已决定停止下载及失败重试，采用当前已完成 512px 预处理和筛选的 22,991,115 条图像视图记录。来源范围已封存；这不是已完成文本验收和 posterior 编码的训练发布包。

| 来源 | 合格 512px 图片数 | 比例 | 主要覆盖 |
|---|---:|---:|---|
| BLIP3o-Pretrain-Long-Caption | 22,115,459 | 96.19% | 通用图像、CC12M、SA-1B、JourneyDB |
| Open Images + Localized Narratives | 513,127 | 2.23% | 关系、位置、交互、场景叙述 |
| PixMo-Cap | 174,594 | 0.76% | 多主体、动作、属性、详细描述 |
| PixMo-Points | 89,047 | 0.39% | 计数、实例位置与指代 |
| PD3M | 96,603 | 0.42% | 摄影与长尾外观 |
| TextCaps | 2,285 | 0.01% | 文字与画面联合理解 |
| 合计 | 22,991,115 | 100% | 不含 ImageNet 和此前消融数据 |

Long 占 96.19%，专项补充共 875,656 张，占 3.81%。这是实际库存比例，不是已验证的训练采样比例；后续可以提高专项图片的采样权重，无需继续等待下载。TextCaps 专项只有 2,285 张，不能宣称 OCR 覆盖已达到原计划。

原始 Long 图片记录 29,353,470 条，已全部预处理：通过 22,115,459，拒绝 7,238,011。URL 来源成功下载 1,018,275 张，全部处理完：通过 875,656，拒绝 142,619。下载失败 113,518 条、未完成下载 393,182 条均排除，后者已改为 skipped_user_stop；保留原因记录，不再重试。

计数口径：合格 512×512 视图，按已完成批次数据库求和。未进行跨来源全局身份合并，未验证内容唯一性；遵照用户要求不计算哈希。ImageNet、公有目录、旧消融发布包和仅完成下载但未完成预处理的归档均未混入本次统计。AnyWord、WikiArt、BLIP3o-60k、ShareGPT-4o、DOCCI/ChartQA、PixMo-Docs、MONET 的已下载归档保留，但不属于当前已选图像池。

预处理为解码、方向修正、fit_pad 到 512×512、图像编码及既有尺寸/解码/已知评测 ID 筛选；不是 VAE posterior。文本质量、512px OCR/计数语义可用性尚不能仅凭该筛选判定合格。既有 caption/标注保留在各批次 state.sqlite3，后续先复用，再使用 SII 补缺或纠错；本轮没有调用合成 API。

## 封存清单与路径

- 机器可读协议：`configs/protocols/unified_b_prepared_frozen_20260914.json`
- 操作快照：`/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/unified-mm-b512/operations/frozen_downloads_20260914/dataset.json`
- 已固定批次清单：`/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/unified-mm-b512/operations/frozen_downloads_20260914/prepared_batches.jsonl`
- 每个批次清单列出 cohort、batch_id、records 和 source_run；只选对应 state.sqlite3 中 status=prepared 的记录。source_run 保留图像引用和原始文本/标注。
- 原图及 512px 图片在项目目录 `unified-mm-b512/images`；各来源的实际准备目录见下面。

- blip3o_long：`/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/unified-mm-b512/preparation/supply/blip3o_long_v1/prepared_batches`（22,115,459 张，116,497 批）
- pixmo_cap_a：`/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/unified-mm-b512/preparation/legacy_v3/supply/pixmo_cap_a_20260913/prepared_batches`（77,862 张，9,604 批）
- pixmo_cap_b：`/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/unified-mm-b512/preparation/legacy_v3/supply/pixmo_cap_b_20260913/prepared_batches`（96,732 张，12,319 批）
- pixmo_points：`/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/unified-mm-b512/preparation/legacy_v3/supply/pixmo_points_20260913_v2/prepared_batches`（89,047 张，11,211 批）
- openimages：`/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/unified-mm-b512/preparation/legacy_v3/supply/openimages_20260913/prepared_batches`（513,127 张，17,557 批）
- pd3m：`/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/unified-mm-b512/preparation/legacy_v3/supply/pd3m_20260913/prepared_batches`（96,603 张，11,826 批）
- textcaps：`/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/unified-mm-b512/preparation/legacy_v3/supply/textcaps_20260913/prepared_batches`（2,285 张，74 批）

## 停止验证

下载进程和 source_job 监督进程已退出；所有已下载批次均处理完成，输入队列均已封闭。各 URL 来源持久化 downloads.stopped.json；下载入口在触碰网络前拒绝重启已关闭来源。历史失败原因保留，未删除原图。停止保护 smoke 检查及 Ruff 通过。

后续训练发布仍需要已知 ID/URL/引用合并、已有文本复用与必要修正、posterior 编码和加载检查；任何新来源加入都应建立新版本，不修改本次固定批次范围。
