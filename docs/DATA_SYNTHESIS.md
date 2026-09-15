# 当前 B512 数据合成流程

当前策略是 **Long 原文为主干，专项数据负责补能力，DeepSeek 只处理明确需要补缺或纠错的部分**。完整路由、保存格式和恢复说明见 [复用优先协议](B_REUSE_FIRST_20260915.md)。原全图复核已停止，旧 Qwen/DeepSeek 结果与响应记录保留。

## 范围与存储

[封存协议](../configs/protocols/unified_b_prepared_frozen_20260914.json)包含 22,991,115 张已筛选 512×512 视图：Long 22,115,459 张，其他来源 875,656 张。本轮不包含 ImageNet、旧消融发布或尚未预处理的归档，下载保持停止。ClimbMix 不变。

图像、文本与运行状态留在 `/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/unified-mm-b512/`。只有 posterior 缓存写入 `/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/datasets/unified_b512_posterior_cache/`。不计算图片、归档、文本、清单的 SHA/MD5/pHash；已知身份连接与最终跨来源身份合并不能被表述为完成了内容去重。

## 当前入口

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -u -m scripts.synthesize_b512_reuse \
  --config configs/data_synthesis/b512_reuse_first_20260915.json

.venv/bin/python -m data_synthesis.cli status
.venv/bin/python -m scripts.status_b512_data
```

默认配置 [b512_sii_v1.json](../configs/data_synthesis/b512_sii_v1.json) 已同步到相同策略；`python -m scripts.synthesize_image_text run` 也会路由到复用优先流程。恢复同一运行必须使用原配置和固定修复规则，不覆盖历史状态。当前根目录是项目 `preparation/synthesis/reuse_first_20260915/`，活动注册信息是 `preparation/active_sii_review.json`。

## 执行方式

- 8 个 CPU worker 并行检查原 caption 的图像身份、全幅几何、来源、格式与 960-token 上限；合格描述直接形成 I2T/T2I 配对文本，不逐图请求模型。
- 可靠最终视图标注可以直接转成描述。仅未通过复用检查或已确认有问题的样本进入定向队列，按约 2,048 张组合调度；短队列最多等待 5 秒。
- SII 模型固定为 `deepseek-v4.1-flash`，单图真实附图。并发从 256 自适应至 512，受 RPM/TPM 限速和熔断保护。每图最多 4 次；同一轮传输故障只降速一次，过期请求不会继续拉长熔断。
- 用户追加授权：同一样本连续两次 SII 超时后，交给独立的 Codex CLI 队列，固定 `gpt-5.6-sol` / `low`，8 路并发、每条最多一次。转交后立即释放 SII 队列，其他格式/内容错误仍按原 SII 规则处理。授权范围保存在当前运行根 `timeout_fallback_policy.json`，配置模板见 [超时接管策略](../configs/data_synthesis/b512_timeout_codex_20260915.json)。原始 SII 配置和旧授权记录保留；此追加规则不启用旧通用 Codex fallback。
- SII 只读环境或 bashrc/zshrc 的静态 `SII_API_KEY`/`SII_BASE_URL`，不读 `test_api.py`。客户端无代理、连接前 TCP MSS=512、TLS 1.2/P-256、证书验证开启，全局代理不变。
- 本地复用与 API 消费并行；分片先保存再登记。原文保留原作者，历史 DeepSeek 修复保留原响应引用，新调用保留原始回复和重试历史。

Codex 保留父进程的代理环境，SII 通过客户端设置直连。超时接管结果与 CLI 原始事件保存到运行根 `timeout_fallback/`，完成前不会结束整轮流程；最终清单将它作为同 key 的补充结果引用，不覆盖旧 SII 响应。

本地结构合格不代表逐图语义通过。明确的图文问题登记在 [定向修复清单](../configs/protocols/b512_known_caption_repairs_20260915.jsonl)，不能凭模型改写比例推断质量提升。失败和不可用样本如实保留，整个流程结束后再统一整理训练发布、posterior 与后续质量修复，不自动开始下一阶段。

## 历史资料

- [封存图像池与统计](B_PREPARED_DATASET_FROZEN_20260914.md)
- [Long 来源与存储规划](B_BLIP3O_LONG_DATA_PLAN_20260913.md)
- [旧下载、冻结与发布接口](DATA_SYNTHESIS_LEGACY_20260913.md)：维护历史数据时使用，不用其中旧默认范围重新下载或全图复核。
- [DeepSeek 全图复核及并发试跑](B_DEEPSEEK_FULL_REVIEW_20260915.md)、[Qwen 全图复核试跑](B_SII_FULL_REVIEW_20260914.md)：保留历史证据。
- [旧消融数据](DATA.md)：不因当前模型或路由变化而改写。
