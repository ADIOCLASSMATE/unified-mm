# 纯文本验证 loss

`UnifiedMixedDataset` 在 `experiment.val_every` 触发时执行
[与训练同口径的 loss 验证](UNIFIED_LOSS_VALIDATION.md)。调度包含 `climbmix`
就计算纯文本 CE；联合训练和 text-only 都覆盖，未训练纯文本的实验不计算此项。
以下固定源记录和独立性协议继续沿用，联合汇总由新验证器完成。

新验证使用**当前训练权重**，保留训练模式的随机损失机制并关闭梯度，恢复模型模式和 Python / NumPy / Torch / 设备
随机数状态，不消费训练 DataLoader 或修改其断点游标。下游验证仍沿用原先的 EMA
协议及单独时间预算；纯文本验证耗时单独写入产物。本次执行了 CPU 测试，未测量 NPU 耗时。
已经运行的进程需要在正常重启/恢复时加载新版代码；编辑代码不会给历史训练补出验证点。

## 固定 ClimbMix 子集

默认 manifest 为 `public/datasets/climbmix_validation_v1/manifest.json`。
本次已从 100 个 ClimbMix 分片各固定选出 4 条，共 **400 条源记录**，seed 424242。
已用本地 Qwen3-0.6B tokenizer 实际检查：默认窗口共 **250,608 个有效目标 token**，
每条长度 8–2,048 token；这是数据准备检查，没有伪造模型验证分数。
每个分片随机选择字节位置，再取其后的完整非空 JSONL 记录，去除重复位置；
这是固定探针，不是按文档均匀抽样。manifest 保存分片路径、大小、mtime 与字节偏移，
不修改原始语料，不计算内容哈希。

```bash
python3 scripts/prepare_climbmix_validation.py
```

相同参数重复运行会核验并复用 manifest，不重新抽样或覆盖。语料发生变化会报错。
需要新协议时使用新的输出目录。

每条文本固定抽取一个最多 32,768 字符的片段，用当前模型 tokenizer 编码，
在长片段中固定选择最多 2,048 token 的窗口。长度可跟随训练源的 `sequence_length`。
文档片段末尾按训练约定追加 EOS；token 窗口截断处不额外追加 EOS。
一行只含一份文档，首 token 作为上下文不计 loss；短文本 padding 不计 loss。
所有 rank 使用相同的全局记录列表，按 `indices[rank::world_size]` 分片，不补齐、
不重复，支持空 rank。原始 CE 按全局有效目标 token 数汇总，不按 batch 均值再平均。

**独立性由训练排除记录决定。** 默认兼容已有训练的数据流；此前没有排除这 400 条，
新测 loss 会标记 `may_have_been_seen_in_training`，不能声称是独立留出集。
要让从 step 0 开始的新训练排除这些源记录，设置：

```yaml
dataset:
  params:
    sources:
      climbmix:
        validation_exclusion_manifest: public/datasets/climbmix_validation_v1/manifest.json
```

排除在读取源行后、tokenization 和预取前执行。manifest 身份进入 ClimbMix 游标；
续训必须保留相同排除规则，禁止给已有游标临时增加/删除排除项。
已有训练的配置及数据流没有在本次被改写。源行排除不等于全文去重：同样文本在其他
行、基座预训练或外部语料中是否出现，并未由此证明。

## 独立外部 JSONL 与配置

如果有独立验证文本，可以改用每行包含 `text` 字符串的 JSONL，跳过默认 manifest：

```yaml
experiment:
  climbmix_validation:
    enabled: true
    jsonl: /path/to/independent_validation.jsonl
    external_independent: true  # 数据提供者的独立性声明；不能由路径自动证明
    sequence_length: 2048
    batch_size: 4
    max_document_chars: 32768
    seed: 424242
```

显式拒绝把训练分片路径本身作为外部验证文件。没有独立性声明时仍标为可能见过。
默认 `enabled: true`。新联合协议不允许关闭一个活跃任务后继续发布总 loss；
若要关闭整组 loss 验证，使用 `experiment.loss_validation.enabled: false`。
`val_every=0` 同时停用定期验证。自定义 manifest 使用 `manifest` 字段，
与训练端的 `validation_exclusion_manifest` 配对时才会标记源行已排除。

## 日志与报告

新联合验证点原子写入：

```text
output/evaluation/training-validation/<run>/validation_unified_loss_metrics_step_<N>.json
```

schema 为 `selfless_unified_loss_validation_metrics_v1`，同时保存来源、独立性、当前权重
类型、训练步数、种子、全局样本数、有效 token 数和耗时。写入失败会在所有 rank 同步报错。
Tracker 记录：

- `val/loss_climbmix`：有效纯文本目标 token 的平均 CE。
- `val/ppl_climbmix`：`exp(min(CE, 100))`。
- `val/climbmix_target_tokens`：真实目标数。
- `val/weighted_contribution_climbmix`：纯文本在训练调度中的比例 × 已加权 microbatch loss 的均值。

新 `val/loss` 是全部活跃任务的贡献之和；text-only 的纯文本比例为 1。
历史 `validation_climbmix_metrics_step_*.json` 保留原来的 token 均值定义，
不会与历史 ImageNet 总 loss 相加冒充新协议。旧 `val/loss_text` 是 caption CE。

重新构建报告会读取新文件，与同一步的历史图文 loss 对齐；未执行的验证保留空白：

```bash
python3 scripts/build_evaluation_report.py
```

训练与验证的完整点、源文件路径及纯文本独立性同时保存在网页和 `training-loss/`
CSV / JSON 中。PNG / SVG 可用构建选项 `--plots` 刷新。
