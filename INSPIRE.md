# Inspire execution notes

## Shared paths

- Repository: `/inspire/hdd/global_user/wanjiaxin-253108030048/code/unified-mm`
- Shared user root: `/inspire/hdd/global_user/wanjiaxin-253108030048`
- ImageNet-100 data: `public/datasets/imagenet_ablation_100c_balanced`

## Official dataset mount

Any Notebook or Job that reads raw ImageNet must explicitly attach:

```text
Dataset ID: imagenet
Version ID: v1
Validated platform path: rclone-worker-1/imagenet/v1
Container path: /inspire/dataset/imagenet/v1
```

Verify that Job details contain non-empty `dataset_info`; shared storage does
not implicitly mount the official dataset.

## Resources

- Use `dev-wjx` for single-GPU micro-batch/smoke checks when it is running.
- Formal ImageNet-100 training and FID/IS evaluation use 8×H100.
- GPU Job priority is 4.
- Default project: `随机序语言建模-统一自回归与掩码扩散的随机顺序生成框架`.
- The main project permits at most 16 concurrent GPUs, so submit at most two
  8-GPU formal jobs together.
- Secondary project: `多模态大模型新架构评测探索与scaling-law`. It may be used
  after a live quota/availability check, with at most 32 concurrent GPUs assigned
  to this work. Do not keep duplicate runnable jobs in both projects.
- Image: `docker.sii.shaipower.online/inspire-studio/dev-wjx:v-2.1`.

## Waiting

After one initial configuration/status check, wait with one blocking process:

```bash
inspire --json job wait <job-name> \
  --workspace 分布式训练空间 \
  --interval 30 \
  --timeout 14400
```

Do not repeatedly poll status, events, logs, or GPU utilization while a formal
job is running.
