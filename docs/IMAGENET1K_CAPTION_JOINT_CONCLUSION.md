# 历史 ImageNet-1K：Caption / T2I 联合训练

从800-epoch class-conditioned final EMA开始训练，配置为 `configs/selfless/imagenet1k_caption_joint_10ep_ascend16_b1024.yaml`。

## 选定配方

所有可训练参数 LR2e-5，文本/图像权重0.05 / 1.0；16×910B，每 rank batch16、GA4、全局1024；10-epoch WSD，共12020步。每图6条合成 caption、12条 T2I prompt。`log_every=50`、`log_grad_norm_every=1200`。

LR 扫描的选定点在12020步得到文本/图像验证 loss 2.1930861473 / 0.5774435997。固定梯度探针的共享 backbone `g_image/g_text` 中位数0.02847458，余弦中位数−0.00694069。

## 文本权重扫描

固定 LR 后，4808步候选结果：

| lambda_text | Caption CLIP (1,000 classes) | FID (50,000) | IS (50,000) | generation mean-rank |
|---:|---:|---:|---:|---:|
| 0.05 | 0.22847543 | 21.53690287 | 44.82762070 | 1.6667 |
| 0.10 | 0.23105135 | 24.32990464 | 38.60260658 | 2.0000 |
| 0.20 | 0.23343236 | 27.84991786 | 32.70475826 | 2.3333 |

按 Caption CLIP、FID、IS 等权平均排名，选 λ_text=0.05。该组对应 class→caption/prompt 数据条件转换；class 初始化的 FID18.99694403 / IS450.06854248 单列作为前序阶段结果。

## 入口

- 训练：`script/selfless/pretraining_imagenet1k_caption_joint_ascend16.sh`
- 配置/资产检查：`scripts/validate_ascend_imagenet1k_caption_joint.py`
- 评测：`script/selfless/evaluate_imagenet1k_caption_joint_ascend16.sh`

扫描和短程 Qwen＋adapter 控制的 checkpoint、一次性脚本及源报告已清理；此页保留选定配方与数值。当前 Unified 配方见 [训练](TRAINING.md)。
