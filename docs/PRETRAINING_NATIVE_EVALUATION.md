# 图像理解评测

协议：[pretraining_native_understanding_evaluation_ascend16.yaml](../configs/protocols/pretraining_native_understanding_evaluation_ascend16.yaml)。评分公式统一见 [评分协议](EVALUATION_PROTOCOL_AUDIT.md)。输入为完整 final EMA，直接评测模型的图文候选似然。

| 任务 | 数据 | 输出 |
| --- | --- | --- |
| ImageNet 分类 | val 50K、1,000 类、单模板 | Top-1 / Top-5 |
| COCO 检索 | Karpathy 5K，25,010 caption | 双向 R@1/5/10 |
| Flickr30K 检索 | Karpathy 1K，5,000 caption | 双向 R@1/5/10 |
| SugarCrepe | 七类完整正负例 | 总体／分类别严格胜率、平局率 |
| ARO | Relation / Attribution 完整正负例 | 严格胜率、平局率 |
| MMBench / SEED | 完整候选集 | 项目内消融诊断 |

COCO 中 4,990 张图各有五条 caption，十张各有六条；I2T 接受同图的全部参考 caption。ImageNet 类名按 synset 区分 `projectile` / `missile`、`sunglass` / `sunglasses`。

## 资产与缓存

| 操作 | 入口 |
| --- | --- |
| 检索资产 | `scripts/prepare_cross_dataset_retrieval_assets.py` |
| 检索 posterior cache | `script/selfless/prepare_cross_dataset_retrieval_cache_ascend16.sh` |
| 检索评分 | `script/selfless/evaluate_cross_dataset_retrieval_ascend16.sh` |
| 图文正负例 cache | `script/selfless/prepare_multimodal_likelihood_cache_ascend16.sh` |
| ImageNet 分类 | `scripts/evaluate_imagenet_pretraining_native.py` |

图文资产与 cache 使用 v2，包含三张固定 null image；默认 cache 名为 `vae_posterior_mar_kl16_v2`。

## 完整运行

```bash
RUN_ROOT=output/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1
EVAL_PROFILE=formal \
  script/selfless/evaluate_unified_native_full_checkpoint_ascend16.sh \
  "${RUN_ROOT}/hf_model-final-ema" output/evaluation/unified-b-x0content-0p6b/<evaluation-name>
```

结果保存模型来源、候选覆盖、校准配置和每项分数，目录结构见 [评测与产物](EVALUATION_STRUCTURE.md)。
