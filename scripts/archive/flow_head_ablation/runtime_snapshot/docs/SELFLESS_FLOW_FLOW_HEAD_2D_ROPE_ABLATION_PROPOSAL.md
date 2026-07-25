# Selfless-Flow Flow-head 2D RoPE 消融：完成记录

Status: **completed and closed on 2026-07-24**

Matched seed-42 screen 已完成：

| ID | Position change | FID ↓ | IS ↑ |
| --- | --- | ---: | ---: |
| **FH0** | additive query/context；无 RoPE | **25.0669** | **61.5120** |
| FH1 | FH0 + row/column 2D RoPE | 25.2985 | 60.3427 |
| FH2 | FH0 去掉 context additive | 27.3799 | 57.4373 |
| FH3 | FH2 + row/column 2D RoPE | 25.4672 | 60.7045 |
| FH4 | FH3 再去掉 query additive | 26.1535 | 60.1577 |

FH0 是 FID/IS 的唯一 Pareto point。2D RoPE 能部分修复移除 context additive 造成的
退化，但没有超过现有 additive baseline；flow MSE 的排序也不追踪 FID。因此最终保留
FH0，关闭 FH1--FH4，不运行 multi-seed confirmation。

- 机器可读结果：
  [`../output/flow_head_position_ablation/screen_summary.json`](../output/flow_head_position_ablation/screen_summary.json)
- 最终 Markdown：
  [`../output/flow_head_position_ablation/screen_summary.md`](../output/flow_head_position_ablation/screen_summary.md)
- 完整预注册设计与历史正文：
  [`archive/SELFLESS_FLOW_FLOW_HEAD_2D_ROPE_ABLATION_PROPOSAL_HISTORICAL.md`](archive/SELFLESS_FLOW_FLOW_HEAD_2D_ROPE_ABLATION_PROPOSAL_HISTORICAL.md)
- 下一轮实验：
  [`SELFLESS_FLOW_DUAL_STREAM_FLOW_HEAD_ABLATION_PROPOSAL.md`](SELFLESS_FLOW_DUAL_STREAM_FLOW_HEAD_ABLATION_PROPOSAL.md)
