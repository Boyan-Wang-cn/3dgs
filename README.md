# GS Dual-Critic 3DGS Baseline

这个目录是一个新的、独立的 3D Gaussian Splatting 压缩 baseline。它参考原 Dual-Critic RL 视频压缩代码的思想，但不修改 `Train`、`Test`、`Preprocess` 里的任何文件。

## 目标

第一版先做 voxel-level Dual-Critic 3DGS compression baseline：

- 一个 episode 对应一个 3DGS scene。
- 将 scene 里的 Gaussian 按 `x/y/z` 坐标划分到 voxel grid。
- 每个非空 voxel group 是一个 step。
- agent 对每个 group 选择一个离散 compression level，范围是 `0..4`。
- 所有 group 决策完成后，统一写出 compressed PLY。
- 第一版使用 dummy reward，不强制运行 3DGS render 或 CrossScore。

## Dual-Critic 映射关系

原视频压缩任务到 3DGS 压缩任务的对应关系：

- `GOP` -> `scene`
- `frame` -> `voxel group`
- `QP` -> `compression level`
- `rate critic` -> `size critic`
- `distortion critic` -> `quality critic`

第一版保留 dual-critic 思想：

- `QualityCritic` 学习 `reward_quality`
- `SizeCritic` 学习 `reward_size`
- 如果最终 `size_ratio > target_size_ratio`，actor 使用 size advantage 更新
- 否则 actor 使用 quality advantage 更新

## 当前已实现

- `gs_baseline/ply_utils.py`：读取 binary PLY，打印字段，识别常见 3DGS 字段，按原字段顺序写出新 PLY。
- `gs_baseline/voxel_grouping.py`：按 `x/y/z` 做 voxel 分组，并提取第一版 state 特征。
- `gs_baseline/compression_ops.py`：实现 pruning、SH-degree 近似降阶、SH float 量化、geometry float 量化。
- `gs_baseline/env.py`：实现 `GSCompressionEnv`，支持 reset/step、dummy reward、compressed PLY 输出。
- `gs_baseline/networks.py`：PyTorch 版 Actor、QualityCritic、SizeCritic。
- `gs_baseline/replay_buffer.py`：on-policy episode buffer。
- `gs_baseline/train.py`：dummy reward training 主循环。
- `gs_baseline/render_bridge.py`：3DGS renderer 命令桥接占位。
- `gs_baseline/crossscore_bridge.py`：CrossScore reward 占位。
- `scripts/inspect_ply.py`：PLY 字段检查工具。
- `scripts/smoke_test_env.py`：随机 action 跑完一个环境 episode。

## PLY 检查

在本目录下运行：

```powershell
python scripts/inspect_ply.py --ply D:/download/Code/data/train.ply
```

如果你的 PLY 在别的位置，把 `--ply` 换成自己的路径。

## Smoke Test

```powershell
python scripts/smoke_test_env.py --ply D:/download/Code/data/train.ply --out D:/download/Code/GS_DualCritic_3DGS_Baseline/outputs/smoke_test
```

这个命令会随机给每个 voxel group 选择 compression level，并输出一个 compressed PLY，同时打印 size ratio。

## Dummy Reward Training

```powershell
python -m gs_baseline.train --config configs/default.yaml
```

或者：

```powershell
.\scripts\run_train_dummy.ps1 -Config configs/default.yaml -Episodes 10
```

训练需要 PyTorch。当前代码不再使用原工程的 TensorFlow 1.x。

## Config

默认配置在 `configs/default.yaml`。当前仓库里真实存在的 PLY 是：

- `D:/download/Code/data/train.ply`
- `D:/download/Code/data/truck.ply`

所以默认 `ply_path` 已经指向这两个文件。若后续你使用的是 `D:/download/Code/data/tested_outputs/train.ply`、`truck.ply` 或其他 3DGS 输出，请修改：

```yaml
data:
  scenes:
    - name: train
      ply_path: your/train/point_cloud.ply
    - name: truck
      ply_path: your/truck/point_cloud.ply
```

## TODO

- 接入真实 3DGS render。
- 接入 `D:/download/Code/CrossScore-main/CrossScore-main` 的 CrossScore 推理。
- 从 dummy reward 切换到真实 CrossScore reward。
- 将单一 compression level 扩展为 multi-head action，例如 pruning、SH-degree、SH-bit、geometry-bit 分开决策。
- 适配官方 3DGS `render.py` 对 model 目录结构的要求。部分版本需要把 PLY 放在 `point_cloud/iteration_xxx/point_cloud.ply` 这类目录结构中。
