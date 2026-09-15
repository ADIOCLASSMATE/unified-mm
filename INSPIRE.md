# Inspire

## 平台与项目

| 项目 | 设置 |
| --- | --- |
| 账号 | `wjx-ascend` |
| Workspace | `昇腾卡公共空间` |
| Compute Group | `910B资源` |
| 设备 | Ascend 910B，64 GiB/卡 |
| Job 镜像 | `docker-t.sii.shaipower.online/inspire-studio/dev-wjx-ascend:v-1.3` |
| 首选整节点 quota | `16,128,1024`，提交时按平台可用配额选择 |
| 优先级 | 6 |
| 永久开发机 | `dev-wjx-ascend`，16 卡 |
| 开发机当前镜像 | `dev-wjx-ascend:v-1.4`（与 Job 首选镜像分别记录） |
| 大模型新架构项目上限 | 256 张并发 Ascend 卡 |
| 随机序语言建模项目上限 | 256 张并发 Ascend 卡（调度器项目配额） |

| 实验 | Project |
| --- | --- |
| A/B、C、LR sweep、only、B/F flow-head scaling、B+SigLIP | `随机序语言建模-统一自回归与掩码扩散的随机顺序生成框架` |
| D/E/F 基础档、S2-single、S2-dual-siglip | `多模态大模型新架构评测探索与scaling-law`（`high-dimensionaldata`） |
| Z | `多模态大模型新架构评测探索与scaling-law`（`high-dimensionaldata`） |

生产训练与评测使用 Ascend。开发机固定排除 `infra-gpu-npu-248.host.shzhisuan.com`，设备测试结束后停机并保留 Notebook 对象。

## 共享路径

| 资产 | 路径 |
| --- | --- |
| 仓库 | `/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/unified-mm` |
| 共享用户根目录 | `/inspire/sj-ssd3/global_user/wanjiaxin-253108030048` |
| `public` | 指向共享用户根目录 |
| 本地 ImageNet 原图 | `public/dataset/imagenet/v1` |
| ImageNet latent | `public/datasets/imagenet_full` |
| Caption / T2I 文本 | `public/datasets/imagenet1k_synthetic_v1` |
| 可定位文本索引 | `public/datasets/imagenet1k_synthetic_v1/indexed/train/manifest.json` |
| 当前 B512 复用 / SII 文本发布根目录 | `public/datasets/unified_image_text_512_api_v3`，至少 200 万目标尚未完成 |
| 当前 B512 图片及 VAE 缓存 | `public/datasets/unified_image_pool_512_v3` |
| 当前下载 / 合成状态 | `public/data_preparation/unified_b_corners_api_v3` |
| 9 月 12 日首轮图文数据（可复用历史） | `public/datasets/unified_image_text_512_sol_v1` |
| 9 月 12 日首轮图片及 VAE 缓存 | `public/datasets/unified_image_pool_512_v1`，与合成文本分开 |
| Qwen 基座 | `public/models/Qwen--Qwen3-0.6B-Base` |
| SigLIP | `public/models/google--siglip-so400m-patch14-384` |
| 训练状态 | `output/<run>/` |
| 评测与验证 | `output/evaluation/` |
| Job 配置、提交记录、冻结源码 | `output/experiments/` |

读取原始 ImageNet 的 Notebook / Job 挂载以下平台数据集：

```text
Dataset ID: imagenet
Version ID: v1
Platform path: rclone-worker-1/imagenet/v1
Container path: /inspire/dataset/imagenet/v1
```

挂载信息记录在 Job 的 `dataset_info`。

仓库的 `public/dataset/imagenet/v1` 已有共享盘原图副本；图文准备优先读取该目录。旧 manifest 中的 `/inspire/dataset/imagenet/v1` 是平台挂载路径，当前 Notebook 缺少该挂载不代表原图不存在。

## 提交与观察

CLI 参数以 `inspire <command> --help` 为准。每次提交先查询项目、配额、镜像、节点和活跃任务，再执行 dry-run，核对规格后提交。节点排除列表使用当前资源组中的有效节点名。

Notebook、`qz.sii.edu.cn` 和 `keycloak-inspire-prod.sii.edu.cn` 直连。当前 CAS 的内网连接会被重置，`cas.sii.edu.cn` 登录需走现有代理；已有账号的登录刷新使用：

```bash
bash script/inspire_wjx_login.sh
inspire config check
```

该脚本从现有 `wjx-ascend` 配置读取凭据，按域名处理 CAS 及其回跳的代理路由，保存标准 CLI 会话；只影响该进程。已有账号不需要再次 `account add`。

已登录后的 Notebook 连接与平台查询可在命令级清理代理：

Codex 所在 CPU Notebook 的全局 mihomo 服务必须保持运行。仅清理需要直连的命令/客户端环境，不调用全局 `clashctl off`；SII 模型与图像下载同样使用独立直连传输。

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    inspire --json job list --workspace 昇腾卡公共空间 --active --all
```

Job 显式填写 Workspace、Project、Group、quota、镜像、节点数和优先级。提交后依次检查 Events、Instances 和训练日志。

NPU smoke 使用 `dev-wjx-ascend`，完成保存、续训、raw/EMA 重载及生成验证后停机。CPU lint、配置检查和报告构建在本地执行。

等待长任务：

```bash
inspire --json job wait <job-name> \
  --workspace 昇腾卡公共空间 --interval 60 --timeout 2592000
```

任务进度、放置和临时故障记录在 `output/experiments/`。训练配置与入口见 [训练](docs/TRAINING.md)，S2 的执行配置见 [S2 infra](docs/S2_INFRA_20260910.md)，caption 合成见 [数据](docs/DATA.md)。
