# Dynamic Dual-Stream Flow Head 消融：完成记录

Status: **completed and archived on 2026-07-25**

The experiment is closed. The active runtime baseline family is
`DF1-FH0 / DF1-FH4`.

The later matched backbone × flow-head screen selected
**`E2-Q0 + DF1-FH4`** as the current system default. This supersedes the
standalone flow-head default while preserving both contracts as runtime
options. See
[`SELFLESS_FLOW_BACKBONE_FLOW_HEAD_JOINT_ABLATION.md`](SELFLESS_FLOW_BACKBONE_FLOW_HEAD_JOINT_ABLATION.md).

- Sampling efficiency remains a reported diagnostic; there is no efficiency
  pass/fail gate.
- Active interface:
  [`SELFLESS_FLOW_HEAD_BASELINE.md`](SELFLESS_FLOW_HEAD_BASELINE.md)
- Full historical proposal and all candidate results:
  [`archive/SELFLESS_FLOW_DUAL_STREAM_FLOW_HEAD_ABLATION_PROPOSAL_HISTORICAL.md`](archive/SELFLESS_FLOW_DUAL_STREAM_FLOW_HEAD_ABLATION_PROPOSAL_HISTORICAL.md)
- Amended quality-only summary:
  `output/flow_head_ablation/dynamic_dual_stream_screen/evidence/amended_quality_only_summary.json`
- Data relocation manifest:
  `output/flow_head_ablation/relocation_manifest.json`

On 2026-07-26, six stopped static-position confirmation runs with neither a
final model nor metrics were pruned. Their compact audit evidence remains under
`output/flow_head_ablation/static_position_screen/evidence/pruned_partial_runs/`;
all completed runs are unchanged.
