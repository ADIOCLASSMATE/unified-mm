# Flow-head ablation code archive

Status: historical, not an active runtime interface.

This directory preserves the matrix builders, launchers, evaluators, tests, and
the exact runtime source snapshot used by the completed flow-head studies.
Historical `DF0`, `DF2`, and `FH1/FH2/FH3` names in this tree are evidence, not
supported configuration values.

The active implementation supports only `DF1-FH0` and `DF1-FH4`; see
`docs/SELFLESS_FLOW_HEAD_BASELINE.md`.

Files outside `runtime_snapshot/` have been path-relocated for archival
inspection. The snapshot itself is intentionally byte-for-byte historical and
retains its original repository-relative imports. To reproduce the old
environment, restore the snapshot and archived configs/launchers into an
isolated checkout using `output/flow_head_ablation/relocation_manifest.json`.
Do not import this archive from active training or evaluation code.

The amended archived selector reports sampling wall time but does not use it as
a pass/fail gate. The immutable raw summary remains under
`output/flow_head_ablation/dynamic_dual_stream_screen/evidence/` for audit.
