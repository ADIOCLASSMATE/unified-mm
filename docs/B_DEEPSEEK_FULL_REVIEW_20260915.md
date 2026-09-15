> 已被用户选择的 [复用优先协议](B_REUSE_FIRST_20260915.md) 替代；本页是保留的全图复核历史记录，不再是当前生产入口。

# DeepSeek 全量复核与合成（2026-09-15）

用户明确要求今后统一使用 SII `deepseek-v4.1-flash`，保留旧 Qwen 结果，并尝试更大并发。当前全量配置为 `configs/data_synthesis/b512_deepseek_v41_flash_bulk_c512_20260915.json`；默认 SII 配置和模型探测入口也已切换。旧 Qwen 配置保留供历史结果审计。

## 固定范围与运行协议

- 范围：`configs/protocols/unified_b_prepared_frozen_20260914.json` 中的 **22,991,115 张**已筛选 512×512 视图。下载保持停止，图片沿用现有项目存储；不复制全套图像、不计算哈希。
- 每图真实解码并附图。复用准确原 caption；重写错误、缺失或不可靠内容，输出 I2T/T2I 配对文本。每段不超过 960 tokenizer tokens；1024 image tokens 的训练约定不变。
- 唯一生产模型 `deepseek-v4.1-flash`，请求和返回模型都须完全一致。模型身份从各状态库固定配置读取，导出不会把历史 Qwen 标成 DeepSeek。
- 并发起点 **256**，自适应上限 **512**；RPM 1200、TPM 6,000,000、单次超时 180 秒、每图最多 4 次尝试。活动任务包含等待请求限速的任务，不能直接当作已发出的网络请求数。连续传输失败触发减半、熔断和单探测恢复。提高并发不等于已经证明吞吐提升。
- `SII_API_KEY` / `SII_BASE_URL` 只取环境或 bashrc/zshrc 的静态声明。客户端无代理、连接前 TCP MSS=512、TLS 上限 1.2、证书校验开启；不关闭全局代理，不读取 `test_api.py`。
- 本轮 **Codex CLI 关闭**。问题与失败保留给后续单独阶段；不自动运行 posterior 编码或训练。已记录用户接受语义局限，不填写虚假的质量通过结论。

## 当前结果与旧结果

项目根为 `/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/unified-mm-b512`。

| 内容 | 项目根下的路径 |
| --- | --- |
| 当前 DeepSeek 全量结果 | `preparation/review/deepseek_v41_flash_bulk_c512_20260915/` |
| DeepSeek 最初 128 并发启动记录 | `preparation/review/deepseek_v41_flash_bulk_20260915/` |
| 保留的 Qwen 全量结果 | `preparation/review/qwen38max_bulk_20260914/` |
| DeepSeek 对 Qwen 失败样本的试跑 | `preparation/review/deepseek_v41_flash_failed_probe_20260915/` |
| 当前任务注册信息 | `preparation/active_sii_review.json` |

Qwen 已停止补入并完成在途保存：已入队的状态库合计 **60,471 ready、411 failed、398 retry、2,573 pending**。其中前五个完整分片共 53,348 条已导出，其余已收到的结果保存在第六片 SQLite。`preserved_for_deepseek_migration_20260915.json` 保存停机快照。Qwen 文本不混入当前 DeepSeek 结果集，也不改写其模型来源。

较低并发 DeepSeek 启动阶段若已有结果，可在完全停止后按原身份与原始响应迁入新运行；仅调度并发不同，完整保留来源配置与迁移记录，避免重复调用。新运行与 Qwen 无此复用关系。

每分片目录含 `review.sqlite3`（原始响应与重试记录）、`status.json`、完成后的 `reviewed.jsonl.gz` 和 `manifest.json`。运行根目录的 `progress.json` 只统计已封存分片；尚在运行的分片需要结合其状态统计。最终全量完成后才产生总 `manifest.json`。

## 本次真实试跑结果

取 Qwen 首个已封存分片中全部 **64 条终态失败**，DeepSeek 每条最多两次；不是全数据集随机质量评测，也不是和 Qwen 等重试次数的总体比较。

| Qwen 失败类型 | 测试量 | DeepSeek 通过结构与来源校验 |
| --- | ---: | ---: |
| 声明 keep 但改动原文 | 56 | 44 |
| 回复图片 ID 不一致 | 4 | 4 |
| SII 图片内容检查拒绝 | 4 | 0 |
| 合计 | 64 | **48（75%）** |

共 94 次调用，包含前置三条探测；通过项中 25 条 keep、23 条 rewrite。另做四条故意使用空白图描述的错误 caption 对照，四条均按实际图像主体重写，证明这个 SII 入口能利用图像；这不能推断底层模型实现或总体精度。

当前助手直接查看七张自然图及四条对照结果：发现手机壳侧边按键判断错误，对照中将两颗镜头写成三颗，人物速写中也有不可靠姿势细节。因此 `visual_audit.json` 中 `semantic_quality_passed=false`。用户随后要求全量迁移，生产以 `user_accepted_for_generation` 授权运行，不把这次 75% 当作逐图语义合格率或整体优于 Qwen 的证明。

## 启动与恢复

先确认 `preparation/active_sii_review.json` 对应进程已停止，避免重复启动；同一运行的文件锁也会拒绝第二个控制器。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 .venv/bin/python -u -m scripts.review_b512_sii bulk \
  --config configs/data_synthesis/b512_deepseek_v41_flash_bulk_c512_20260915.json \
  --root /inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/unified-mm-b512/preparation/review/deepseek_v41_flash_bulk_c512_20260915 \
  --selection configs/protocols/unified_b_prepared_frozen_20260914.json \
  --qualification /inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/unified-mm-b512/preparation/review/deepseek_v41_flash_bulk_c512_20260915/user_acceptance.json
```

保持同一配置恢复；已封存分片跳过，已保存原始回复本地重放，未完成项延续尝试次数。修改模型或运行配置须创建新版本，不能覆盖历史状态的固定合同。

验证涵盖 DeepSeek/Qwen 各自的模型绑定、结果导出、断点回复重放，以及既有数据复用与发布协议；迁移代码相关测试 39 项通过。

## 高并发启动观测

本次从最初 DeepSeek 运行迁入 257 条 ready 与 47 条 retry，原始响应全部保留，257 条已接收结果逐条重放校验通过。新任务 PID 与完整命令保存在当前根目录的 `launch.json`，进程注册指向当前目录。

初始 60 秒观测窗口中，ready 从 544 增至 1,144（约 600 条/分钟），自适应活动任务上限从 256 增至 280。采样结束时，新运行已持久化的 1,051 次响应均为 HTTP 200，没有已返回的 429 或传输错误；仍有在途请求，不能据此断言长期无超时。结构不合格项仍按原协议重试，HTTP 200 不计作语义通过。

详细采样与模型/附件/无代理证据保存在 `concurrency_trial_samples.json` 和 `concurrency_trial_verification.json`。这是启动短窗口，不是 512 并发持续吞吐基准，也不据此推算全量完成时间。
