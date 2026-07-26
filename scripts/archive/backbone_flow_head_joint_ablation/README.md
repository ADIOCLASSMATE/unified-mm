# Backbone × flow-head joint ablation archive

Status: historical; not an active training interface.

This directory preserves the matrix builder, smoke test, protocol tests, and
the two manifest-bound runtime source snapshots used by the completed
seed-42 study. Archived launchers and configs live under:

```text
script/ablation/archive/backbone_flow_head_joint_ablation/
configs/ablation/archive/backbone_flow_head_joint_ablation/
```

Run artifacts are stored under
`output/backbone_flow_head_joint_ablation/runs/`; evidence, smoke reports, and
the invalid first-attempt record remain alongside them. Historical absolute
paths in immutable result files are resolved by
`output/backbone_flow_head_joint_ablation/archive_manifest.json`.

Do not import this archive from active training code. The selected runtime
interface is `E2-Q0 + DF1-FH4`; see
`docs/SELFLESS_FLOW_BACKBONE_FLOW_HEAD_JOINT_ABLATION.md`.
