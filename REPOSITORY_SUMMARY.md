# UMI Data All Process 仓库概览

## 定位

本仓库收录了 UMI 双手数据从原始目录整理、手柄位姿转换、时间戳对齐、LeRobot v2.1 数据集生成、OpenPI 归一化统计，到 Quanta X2 + Revo2 MuJoCo 回放的脚本。

仓库重点不是通用 Python 包，而是一组带有明确数据契约和安全检查的命令行工具。多数流程面向固定的机器人、目录结构和服务器环境。

## 当前已提交组件

| 组件 | 作用 | 关键约束 |
| --- | --- | --- |
| `flatten_data.py` | 将多个采集父目录下的 episode 移动或复制到统一目录 | 默认只预览；必须显式使用 `--execute` 才会修改文件 |
| `controller_to_hand_pose.py` | 把左右 VR 手柄绝对位姿转换为 EEF 局部帧间增量 | 默认输出 X2 + Revo2 末端坐标系；第一帧是单位增量；默认只预览 |
| `conversion_tools/umi_to_lerobot.py` | 将 UMI 手部位姿、绝对手指状态和三路视频转换为 LeRobot v2.1 | 固定 10 Hz、H50、30 维 state/action；保留最长严格对齐段 |
| `conversion_tools/task_v1_to_lerobot.py` | `task_v1` 专用 LeRobot v2.1 转换器 | 固定 10 Hz、H50、30 维 state/action；一个有效连续段输出为一个 LeRobot episode |
| `conversion_tools/compute_task_v1_masked_norm_stats.py` | 排除 `action_is_pad` 后计算 OpenPI normalization | 当前代码硬编码为 10 Hz / H50 / 30 维契约，依赖 OpenPI 环境 |
| `random_replay_5090/random_replay_5090.py` | 随机或指定 episode，校验 CSV，传输回放包并调用 5090 主机回放 | 支持 seed 复现、`--prepare-only`、视频输出和步长警告 |
| `random_replay_5090/replay_x2_revo2_relative_eef.py` | 在 Quanta X2 + Revo2 MuJoCo 环境里累积局部 EEF 增量并求解 IK | 依赖 MuJoCo、SciPy 和 `quanta_x2_mujoco`；可交互显示或离屏生成 MP4 |
| `random_replay_5090/launch_random_replay_5090.sh` | 管理 5090 桌面回放进程 | 默认使用项目特定的 Python、assets、DISPLAY 和 Xauthority 路径 |

## 数据流

1. `flatten_data.py` 整理采集目录，但不解析 episode 内容。
2. `controller_to_hand_pose.py` 用标定外参将手柄位姿转为左右 EEF 局部增量：`D[t] = inverse(T[t-1]) @ T[t]`。
3. LeRobot 转换器根据全局时间戳对齐 E6、左右腕部相机和 Revo2 手指状态。
4. 输出 state 由左右 EEF 位姿增量和 12 维绝对手指状态组成，共 30 维。
5. 每个 action chunk 共 H50，所有 EEF 目标共用当前帧锚点；越过 episode 尾部的 slot 使用 `action_is_pad` 标记。
6. 归一化统计必须排除 padding slot，训练时也必须逐 slot 屏蔽。
7. MuJoCo 回放按 `T_target[t] = T_target[t-1] @ D[t]` 累积目标，用于检查坐标系、轨迹和 IK 可行性。

## 安全性与可复现性

- 目录整理和位姿转换默认 dry-run，需要 `--execute` 才写入。
- LeRobot 转换器提供 `inspect`、`convert` 和数学 `self-test`。
- 转换先在目标旁构建 staging 目录，完成后再原子发布，并拒绝覆盖非空目标。
- 对齐和转换过程记录原文件哈希、源帧索引、时间差、契约与审计数据。
- 随机回放记录实际 seed，便于重现同一 episode 的选择。

## 运行环境

- `flatten_data.py` 和 `controller_to_hand_pose.py` 仅依赖 Python 标准库。
- LeRobot 转换器需要 NumPy、FFmpeg/FFprobe，并在转换时需要兼容的 LeRobot Python 环境。
- normalization 脚本需要 NumPy、PyArrow 和 OpenPI。
- MuJoCo 回放需要 NumPy、SciPy、MuJoCo 和 `quanta_x2_mujoco`。
- 默认路径包含 `/home/dzq`、`/mnt/data/dzq` 和 `192.168.110.199`，在其他环境使用前应通过参数或环境变量覆盖。

## 快速自检

```bash
python3 -m py_compile \
  flatten_data.py \
  controller_to_hand_pose.py \
  conversion_tools/*.py \
  random_replay_5090/*.py

python3 conversion_tools/umi_to_lerobot.py self-test
python3 conversion_tools/task_v1_to_lerobot.py self-test
```

在处理真实数据前，先查看参数并执行只读预检：

```bash
python3 flatten_data.py --help
python3 controller_to_hand_pose.py --help
python3 conversion_tools/umi_to_lerobot.py inspect --help
python3 random_replay_5090/random_replay_5090.py --help
```

## 当前分支已知缺口

以 `main` 分支当前实际文件为准，存在以下文档/代码偏差：

- 主 README 引用了 `conversion_tools/umi_folder_to_lerobot.py`，但该文件尚未提交。
- 主 README 引用了 `review_tools/`、`training_tools/` 和 `archive/`，但这些目录尚未提交。
- 主 README 将 `compute_task_v1_masked_norm_stats.py` 描述为旧 30 Hz / H90 工具，而当前脚本实际常量和 schema 是 10 Hz / H50。
- 仓库尚无依赖锁定文件、正式测试目录和 CI；可执行的自检主要由转换器内置 `self-test` 提供。

建议先同步上述缺失文件，再以一个可配置的通用转换器取代两个硬编码的 10 Hz / H50 转换器，最后补充最小依赖文件和 CI 自检。
