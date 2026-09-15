> 2026-09-14 更新：下载已按用户要求停止；当前采用 [已封存图像池](B_PREPARED_DATASET_FROZEN_20260914.md)，共 22,991,115 条合格 512px 视图。下文下载目标为历史规划，当前范围以封存协议为准。

# B512：Long 主库与分离存储

当前选择是 BLIP3o-Pretrain-Long-Caption 主库、160 万专项起始覆盖目标、完整 ImageNet train。Long 标称约 2,700 万条，账面合计 29,881,167，实际训练图片数由已知 ID/URL/原图引用去重、512px 可用性、原始 train/评测排除及最终文本验收确定。Short-Caption 和独立 JourneyDB 暂不全量下载。纯文本 ClimbMix 沿用现有数据。

[来源协议 JSON](../configs/protocols/unified_b_blip3o_long_v1.json)记录当前选择；[运行配置](../configs/data_synthesis/b512_sii_v1.json)是路径和模型路由的唯一当前配置；[执行步骤](DATA_SYNTHESIS.md)说明完整下载、独立预处理、复用优先及 SII 补缺。

2026-09-14 按用户要求关闭数据哈希：`compute_hashes=false`。下载、512px 预处理、图片池冻结、文本发布和 posterior 流程不再计算图像、归档或清单的 SHA/MD5/pHash，也不进行依赖内容哈希的去重及评测近重复扫描。保留 ID/URL/引用连接、长度、RGB 解码和逐行映射检查；最终计数是已知身份去重后的图片数，不代表经过内容去重的全库独有原图数。历史回执保留，不回填新哈希、不重算旧批次。任务 ID 和小型运行协议指纹继续兼容历史任务。

| 补充来源 | 起始目标，非上限 |
| --- | ---: |
| PixMo-Cap | 500,000 |
| Open Images 关系 + Localized Narratives | 300,000 |
| PixMo-Points | 180,000 |
| AnyWord-3M | 350,000 |
| WikiArt | 70,000 |
| BLIP3o-60k + ShareGPT-4o-Image T2I | 80,000 |
| TextCaps + DOCCI train | 30,000 |
| PixMo-Docs / ChartQA train | 40,000 |
| PD3M | 50,000 |

原 200 万方案中的 JourneyDB 30 万和 CC12M 10 万覆盖目标由 Long 承担；已下载 MONET 等数据中合格的独有图片仍保留。已有 105,322 张非 ImageNet 发布计在来源并集内。Long/Short 与 MONET 已确认有同图，但现有抽样不能估计全库重复率；不把不同 caption、编码、缩放视作新原图。

Long 固定 Hugging Face 版本 `e4d07091a466d1a1e35a9b0c61caddc78d14a059`，全部 2,891 个 tar：SA-1B 1,000、CC12M 1,472、JourneyDB 419；tar 合计 1,374,049,964,032 bytes。上游已有校验和可以保存在来源元数据中，但当前下载仅执行长度、版本和 Range 检查，不计算或比对文件内容哈希。下载 16 个文件并发、每文件 4 路 Range，直连 TLS 1.2 / P-256 / HTTP/1.1，不修改全局代理。

项目根目录为 `/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/unified-mm-b512`：

| 内容 | 目录 |
| --- | --- |
| Long 原始归档 | `images/source_archives/blip3o_long/` |
| 迁移的已有图像、归档与视图 | `images/legacy_v3/` |
| 新 512px 视图 | `images/prepared_supply/` |
| 新文本发布 | `texts/releases/` |
| 历史 v3 文本兼容入口 | `texts/legacy_api_v3/` |
| Long 下载状态 | `preparation/downloads/blip3o_long_v1/` |
| Long 原图引用批次 | `preparation/supply/blip3o_long_v1/` |
| Long 已预处理 inbox | `preparation/prepared/blip3o_long_v1/` |
| 迁移的旧 v3 准备状态 | `preparation/legacy_v3/` |
| 新合成状态 | `preparation/synthesis/long_v1/` |

新 posterior tensor 根目录为 `/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/datasets/unified_b512_posterior_cache`。I2T/T2I 共享同一图片与 posterior。按账面约 2,988 万计算，fp16 `[N,1024,32]` 净载荷约 1.958 TB；其他工作数据都在项目空间。旧 ImageNet 原图和历史消融是只读输入，本次不移动用户整个个人目录或改写历史实验。

此前迁移跨存储域，使用先复制、暂停已知写进程、补同步及内容校验、最后保留旧路径软链接的流程；不可变清单与来源哈希保持原值。该次迁移已完成，记录保存在项目根 `operations/migration_20260913/`，不作为当前重新计算哈希的要求。

2026-09-13 已完成既有 v3 图像、文本和准备状态的迁移，共 764,315,564,948 bytes；三项 checksum 同步均退出 0，原副本已清理，兼容软链接与五组续传服务已核对。`operations/migration_20260913/completion.json` 保存迁移、恢复与回归检查记录。实时汇总位于项目根 `preparation/status.json`，可运行 `.venv/bin/python -m scripts.status_b512_data` 刷新；下载、intake、prepare 的原始状态仍各自保留。

Long 的 intake 只创建原 tar 的 offset 引用和现有 txt，不再拷贝一份原图；独立 prepare 负责解码与 512px 视图。正式大规模 API 仍要等待整个图片池关闭与模型资格验收。SII 配置只读环境/静态 bashrc 或 zshrc 的 SII_API_KEY、SII_BASE_URL；不读 test_api.py。

图像任务内部 BLIP 75%、专项 20%、ImageNet 5% 为训练前建议，尚未用 B 验证或写入旧消融配置。旧 512px 配方仍是小规模验证配置，不因下载启动而声称大模型已开训。
