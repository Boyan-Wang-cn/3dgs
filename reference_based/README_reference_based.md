# Reference-Based Dual-Critic 3DGS Baseline

> Round-4 README update: this top section is the current source of truth.
> Some older second-round notes may remain below if the host terminal displays
> legacy text with the wrong encoding.

This directory is the reference-based 3DGS compression baseline migrated from
`Train/DRL_x265_TRAIN`. Round 4 keeps the Dual-Critic training skeleton and
connects the terminal path:

```text
compressed PLY -> model_path -> 3DGS render -> CrossScore -> reward_D
```

## Migrated Core

- `Network.py` -> `Network_GS.py`
- `Transition.py` -> `Transition_GS.py`
- `Environment.py` -> `Environment_GS.py`
- `trainRDO.py` -> `trainRDO_GS.py`

Still preserved:

- `ActorNetwork` / `CriticNetwork`
- `baseQP + deltaQP` mapped to `baseLevel + deltaLevel`
- `q_value_D` / `q_value_P`
- `reward_D` / `reward_P`
- Q(s,a) critics
- replay memory random sampling
- target networks
- soft update
- actor update through critic action gradients
- `left_bitbudget < 0` prefers size critic P; `left_bitbudget >= 0` prefers quality critic D

## Round-4 Modules

- `configs/default.yaml`: path, training, render, CrossScore, and quality settings.
- `model_path_utils.py`: prepares official GraphDeco model_path layout.
- `render_bridge.py`: optional `gaussian-splatting` render.py bridge.
- `crossscore_input_utils.py`: validates and adapts render/reference image pairs for CrossScore.
- `crossscore_bridge.py`: CrossScore repo inspection, real CLI adapter, score parsing, cache helpers, and placeholder mode.
- `pipeline_utils.py`: shared helpers for non-training terminal pipeline and baselines.
- `scripts/test_*.py`: step-by-step pipeline tests.
- `scripts/run_fixed_levels.py` and `scripts/run_random_baseline.py`: required baseline comparisons.

## Stage 0: Dummy Dual-Critic

```powershell
python trainRDO_GS.py --config configs/default.yaml --scene train --episodes 3 --use-dummy-reward
```

## Stage 1: Test PLY Compression And model_path

```powershell
python scripts/test_terminal_reward_pipeline.py --config configs/default.yaml --scene truck --level 2
```

```powershell
python scripts/test_prepare_model_path.py --ply data/gs_models/truck_original/point_cloud/iteration_30000/point_cloud.ply --out outputs/debug/truck_model_test --iteration 30000
```

## Stage 2: Test 3DGS Render

```powershell
python scripts/test_terminal_reward_pipeline.py --config configs/default.yaml --scene truck --level 2 --use-render
```

```powershell
python scripts/test_render_one_model.py --gaussian-splatting-dir gaussian-splatting-main --model-path data/gs_models/truck_original --source-path data/tandt/truck --iteration 30000 --resolution 4
```

## Stage 3: Inspect CrossScore

```powershell
python scripts/test_crossscore_inspect.py --crossscore-dir CrossScore-main
```

The detected CrossScore entrypoint is `task/predict.py`. The default adapter
passes:

- `trainer.ckpt_path_to_load`
- `data.dataset.query_dir`
- `data.dataset.reference_dir`
- `logger.predict.out_dir`
- `alias`

CrossScore's default score summary is parsed from `score_summary/**/*.csv`;
the default `pred_ssim_0_1` direction is higher-is-better.

## Stage 4: Test One CrossScore Pair

```powershell
python scripts/test_crossscore_one_pair.py --crossscore-dir CrossScore-main --render-dir <renders> --reference-dir <gt> --output-dir outputs/crossscore/debug_pair
```

If the command fails, the script prints the saved command, stdout/stderr paths,
and the output file listing.

## Stage 5: Terminal Pipeline With CrossScore

```powershell
python scripts/test_terminal_reward_pipeline.py --config configs/default.yaml --scene truck --level 2 --use-crossscore
```

This runs:

```text
compress PLY -> prepare compressed model_path -> render original/compressed -> CrossScore -> reward_D
```

## Stage 6: Train With CrossScore

```powershell
python trainRDO_GS.py --config configs/default.yaml --scene train --episodes 3 --use-crossscore
```

Important: the bundled `ckpt/CrossScore-v1.0.0.ckpt` may be a Git LFS pointer.
If so, real CrossScore will stop with a clear error. Run `git lfs pull` inside
CrossScore or set `crossscore.ckpt` to a real checkpoint.

## Reward_D Modes

- `dummy`: no render, no CrossScore; `reward_D = -mean_action / 4 * 0.1`.
- `render_only`: render original and compressed; `reward_D = 0` and render paths are logged.
- `crossscore`: render original and compressed, compute original/compressed CrossScore, then `reward_D = -quality_drop`.

`--use-crossscore` never silently falls back to placeholder. Placeholder scoring
requires both `quality.crossscore_mode: "placeholder"` and the explicit
`--allow-crossscore-placeholder` flag; do not use it for final quality results.

## Baseline Comparisons

```powershell
python scripts/run_fixed_levels.py --config configs/default.yaml --scene truck
python scripts/run_random_baseline.py --config configs/default.yaml --scene truck --trials 5
```

Add `--use-render` or `--use-crossscore` to log render/CrossScore quality
fields. The CSVs include `quality_mode`, `original_score`, `compressed_score`,
`quality_drop`, `crossscore_is_placeholder`, render dirs, `reference_dir`,
`reward_D`, and `reward_P`.

Outputs:

- `outputs/baselines/fixed_levels_<scene>.csv`
- `outputs/baselines/random_<scene>.csv`

These fixed/random baselines are required comparison items.

这个目录是第二轮版本：它不再只是“参考 Dual-Critic 思想”，而是按原始 `Train/DRL_x265_TRAIN` 的代码结构迁移到 3DGS 压缩任务。

## 和上一版 `gs_baseline` 的区别

上一版 `gs_baseline` 是一个简化的 PyTorch/on-policy smoke baseline，重点是先跑通 PLY、voxel grouping、compression env。

本版 `reference_based` 的重点是代码级迁移原 Dual-Critic RDO 结构：

- `Network.py` -> `Network_GS.py`
- `Transition.py` -> `Transition_GS.py`
- `Environment.py` -> `Environment_GS.py`
- `trainRDO.py` -> `trainRDO_GS.py`

## 保留的原始机制

- `ActorNetwork` / `CriticNetwork` 类结构。
- Actor 的 `baseQP + deltaQP` 分解，迁移为 `baseLevel + deltaLevel`。
- `q_value_D` / `q_value_P` 双 critic 命名。
- Q(s,a) critic：critic 输入包含 state 和 action。
- replay memory 随机采样，而不是 on-policy episode buffer。
- behavior network / target network。
- copy weights 和 soft update。
- actor 通过 critic action gradients 更新。
- `left_bitbudget` 驱动 size constraint：若 `left_bitbudget < 0`，说明模型仍超预算，actor 主要跟随 size critic P；否则跟随 quality critic D。

## 任务替换

| 原 Dual-Critic 视频压缩 | 3DGS baseline |
| --- | --- |
| sequence / GOP | 一个 3DGS scene episode |
| frame | voxel group |
| QP | compression level |
| baseQP | base compression level |
| deltaQP | delta compression level |
| bit budget | target model size ratio / target size bytes |
| left_bitbudget | target_size_bytes - compressed_size_bytes |
| distortion reward_D | dummy quality reward_D，后续可接 CrossScore |
| rate reward_P | size penalty reward_P |
| x265 encoder | 3DGS PLY compressor |

## 文件说明

- `Network_GS.py`：从原 `Network.py` 迁移，保留 Actor/Critic、dual critic、target/soft update、action gradient 接口。实现使用 NumPy，避免 TF1 环境依赖。
- `Transition_GS.py`：从原 `Transition.py` 迁移，保留 namedtuple 字段和 replay memory，并增加随机 `sample_batch`。
- `Environment_GS.py`：从原 `Environment.py` 迁移，保留 reset/getObservation/getTrainReward 组织方式，把 frame feature 改为 voxel group feature。
- `trainRDO_GS.py`：从原 `trainRDO.py` 迁移，替换 x265 socket 为 `env.step(action)`，保留 replay sampling、target Q、critic update、actor gradient update。
- `ply_utils.py`、`voxel_grouping.py`、`compression_ops.py`、`gs_compressor.py`：3DGS 工具层，替代原 x265 编码器。

## 运行命令

检查 PLY：

```powershell
cd reference_based
python ply_utils.py --ply ..\..\data\train.ply
```

训练一个 dummy episode smoke test：

```powershell
cd reference_based
python trainRDO_GS.py --episodes 1 --ply ..\..\data\train.ply --max-groups 4
```

默认训练 10 episodes：

```powershell
cd reference_based
python trainRDO_GS.py --episodes 10
```

输出位置：

- checkpoints: `reference_based/checkpoints`
- logs: `reference_based/logs`
- compressed PLY: `reference_based/outputs`

## 当前没有做的事

- 没有接入真实 CrossScore。
- 没有调用真实 3DGS render。
- 没有做 multi-head action。
- 没有改成 PPO 或 on-policy policy gradient。
