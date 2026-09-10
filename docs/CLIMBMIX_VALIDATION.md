# ClimbMix 验证

训练调度包含 `climbmix` 时，周期验证计算纯文本 CE。它使用当前训练权重，作为 [统一 loss](UNIFIED_LOSS_VALIDATION.md)的一部分；下游文本 benchmark 使用 EMA。

## 固定探针

默认 manifest 为 `public/datasets/climbmix_validation_v1/manifest.json`：100 个训练分片各选四条源记录，共 400 条，seed 424242。抽样以随机字节位置后的完整记录为单位，保存路径、大小、mtime 与偏移。

```bash
python3 scripts/prepare_climbmix_validation.py
```

每条最多读取 32,768 字符，分词后固定一个最长 2,048-token 窗口。首 token 和 padding 不计 loss；文档结尾按训练规则追加 EOS。Qwen3-0.6B 默认窗口共有 250,608 个有效目标 token。

各 rank 分片同一全局记录列表，CE 按有效目标 token 总数归约。

## 来源设置

现有实验使用训练语料探针，结果标记 `may_have_been_seen_in_training`。新训练可通过源行排除建立留出：

```yaml
dataset:
  params:
    sources:
      climbmix:
        validation_exclusion_manifest: public/datasets/climbmix_validation_v1/manifest.json
```

排除规则写入数据游标，续训沿用相同规则。独立 JSONL 每行包含 `text`，配置为：

```yaml
experiment:
  climbmix_validation:
    enabled: true
    jsonl: /path/to/validation.jsonl
    external_independent: true
    sequence_length: 2048
    batch_size: 4
    max_document_chars: 32768
    seed: 424242
```

`external_independent` 记录数据来源声明。自定义固定清单使用 `manifest` 字段。

## 输出

`validation_unified_loss_metrics_step_<N>.json` 保存来源、源行排除状态、有效 token 数、CE、PPL 和任务贡献。路径为 `output/evaluation/training-validation/<run>/`。

Tracker 对应 `val/loss_climbmix`、`val/ppl_climbmix`、`val/climbmix_target_tokens` 和 `val/weighted_contribution_climbmix`。网页通过 `python3 scripts/build_evaluation_report.py` 更新。
