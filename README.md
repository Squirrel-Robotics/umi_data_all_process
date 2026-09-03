# UMI 数据处理工具

用于整理 UMI 原始数据、生成末端增量、审核 RGB、转换 LeRobot 数据，以及在 RTX 5090 上回放。

[查看当前仓库的简明概览、数据契约与已知缺口](REPOSITORY_SUMMARY.md)。

## 目录

```text
data_deal/
├── flatten_data.py                 # 通用目录扁平化
├── controller_to_hand_pose.py      # 手柄绝对位姿 → 双手 EEF 局部增量
├── conversion_tools/               # LeRobot 转换和归一化统计
├── review_tools/                   # RGB 审核、首帧和裁剪检查
├── random_replay_5090/             # MuJoCo 回放及 MP4
├── training_tools/                 # OpenPI 训练启动和排队
└── archive/                        # 历史文档和备份
```

以下命令默认在数据服务器运行：

```bash
cd /home/dzq/data_deal
```

## 推荐流程

1. 用 `flatten_data.py` 整理采集目录。
2. 用 `controller_to_hand_pose.py` 生成 `camera/hand_pose.csv`。
3. 用 `review_tools/rgb_review_app.py` 筛选 RGB 数据。
4. 用 `conversion_tools/umi_folder_to_lerobot.py` 预检并转换。
5. 为该数据集的 FPS、action horizon 和 schema 生成匹配的 OpenPI normalization。

所有会移动或覆盖原始文件的通用脚本默认只预览；确认输出后再加 `--execute`。LeRobot 转换器始终只读源数据，并在目标旁的临时目录完成后原子发布。

## 整理采集目录

先预览：

```bash
python3 flatten_data.py /mnt/data/dzq/umi/data/task_v1 \
  --parent-pattern 'collector_run_*' \
  --remove-empty-parents \
  --verbose
```

确认后执行：

```bash
python3 flatten_data.py /mnt/data/dzq/umi/data/task_v1 \
  --parent-pattern 'collector_run_*' \
  --remove-empty-parents \
  --execute
```

## 生成 `hand_pose.csv`

先预览：

```bash
python3 controller_to_hand_pose.py /mnt/data/dzq/umi/data/task_v1 \
  --input-glob '**/camera/e6_rgb_controller_poses.csv' \
  --skip-existing \
  --verbose
```

确认后执行：

```bash
python3 controller_to_hand_pose.py /mnt/data/dzq/umi/data/task_v1 \
  --input-glob '**/camera/e6_rgb_controller_poses.csv' \
  --skip-existing \
  --execute
```

输出是上一帧 EEF 自身坐标系中的局部增量，可直接使用：

```python
T_target[t] = T_target[t - 1] @ D_robot[t]
```

第一帧是单位增量；不要把 `hand_pose.csv` 当作世界坐标系绝对位姿。

## LeRobot 转换

当前原生 UMI 布局统一使用：

```text
conversion_tools/umi_folder_to_lerobot.py
```

该脚本适用于 `task_v1`、`task_v1_new` 以及任意同目录布局的新数据集。旧的
`task_v1_to_lerobot.py` 和 `umi_to_lerobot.py` 仅保留作为历史工具，新数据不再使用它们。

### 通用规则

- `--source`、`--target`、`--repo-id`、`--task`、`--fps` 和 `--action-horizon` 均由命令行设置。
- 无 `--expected-episodes`；脚本自动扫描 `--source` 下的全部直接子目录。
- 源 episode 目录名必须匹配 `YYYYMMDD_HHMMSS_..._...`；可见的异常文件、目录或符号链接会使预检失败，不会被静默忽略。
- 每个源文件夹必须恰好生成一个 LeRobot episode，并保留原文件夹 ID。
- 若一个源文件夹内存在两段或更多可输出的连续数据，整体预检失败；不拆分、不跨断点拼接、不默认只取最长段。
- 任何源 episode 失败时，`convert` 会在视频编码和创建目标前整体终止，不会跳过问题数据后部分发布。
- 转换开始和原子发布前会两次校验 source snapshot，防止采集过程中数据变化。
- `--fps` 必须能整除源 E6 帧率。60 Hz 源数据可设为 10/15/20/30/60 Hz。
- 尾部不足的 action 使用 `action_is_pad` 逐 slot 标记；真实观测和三路视频不会因 action horizon 不足而被填充或删除。

转换器提供：

```bash
self-test   # 数学与数据契约自检
inspect     # 只读检查全部源 episode，不创建目标
convert     # 预检全部通过后才正式转换
```

### 1. 运行自检

```bash
/home/dzq/openpi/.venv/bin/python \
  conversion_tools/umi_folder_to_lerobot.py self-test
```

### 2. 先运行只读预检

以 `task_v1_new`、30 Hz、H90 为例：

```bash
/home/dzq/openpi/.venv/bin/python \
  conversion_tools/umi_folder_to_lerobot.py inspect \
  --source /mnt/data/dzq/umi/data/task_v1_new \
  --target /mnt/data/dzq/umi/datasets/task_v1_new_lerobot_30hz_h90 \
  --repo-id dzq/task_v1_new_lerobot_30hz_h90 \
  --task 'Put the object into the box.' \
  --fps 30 \
  --action-horizon 90 \
  --max-alignment-ms 100 \
  --max-hand-age-ms 100 \
  --hand-alignment nearest \
  --compact
```

`inspect` 会读取和校验每个源文件夹，但不创建 `--target`。只有输出中同时满足
`status: "ok"`、`failure_count: 0` 和 `one_source_one_output_episode: true` 时才应执行 `convert`。
若返回码为 1 且 `status: "failed"`，表示预检发现数据问题，不是脚本崩溃；所有问题会集中列在 `failures`。

### 3. 全部通过后正式转换

```bash
/home/dzq/openpi/.venv/bin/python \
  conversion_tools/umi_folder_to_lerobot.py convert \
  --source /mnt/data/dzq/umi/data/task_v1_new \
  --target /mnt/data/dzq/umi/datasets/task_v1_new_lerobot_30hz_h90 \
  --repo-id dzq/task_v1_new_lerobot_30hz_h90 \
  --task 'Put the object into the box.' \
  --fps 30 \
  --action-horizon 90 \
  --max-alignment-ms 100 \
  --max-hand-age-ms 100 \
  --hand-alignment nearest \
  --video-workers 4 \
  --confirm CREATE_LEROBOT_DATASET
```

目标目录默认必须不存在。只有目标是完全空目录时，才可显式增加
`--replace-empty-target`；该参数不会覆盖已有数据集。

转换其他同布局数据时，替换 `--source`、`--target`、`--repo-id` 和 `--task` 即可。
例如改为 10 Hz / H50，还需同时修改目标路径和 repo ID，并设置：

```bash
--fps 10 --action-horizon 50
```

### 数据契约

- 三路视觉：E6 右眼、cam0 左腕、cam1 右腕。
- 以时间戳最近邻对齐，相机和手部数据误差均不超过 100 ms。
- `observation.state` 为 `(30,)`。
- `action` 为 `(<action-horizon>, 30)`。
- `action_is_pad` 为 `(<action-horizon>,)`，训练 loss 必须逐 slot 屏蔽 padding。
- `state[t] = inverse(T[t-1]) @ T[t]`。
- `action[t,k] = inverse(T[t]) @ T[t+k]`，同一 action chunk 使用同一个当前帧锚点。
- 每个 episode 的第一个对齐点只用作构造首个局部 state 的基线，不单独输出为训练行。
- OpenPI 中保持 `action_sequence_keys = ()`，不再次切 future chunk，不再次计算 EEF delta。

### `task_v1_new` 当前预检状态（2026-08-31）

30 Hz / H90 只读预检已扫描 156 个源文件夹：148 个可生成，8 个失败。
其中 5 个为时间戳冲突或相机对齐异常，3 个存在多个有效连续段。
因为转换器严格执行“一个源文件夹 = 一个输出 episode”，所以当前预检返回失败，且未创建目标目录。
处理这 8 个源目录后必须重新运行 `inspect`；以新的预检结果为准。

### OpenPI normalization 注意事项

通用转换器会在 LeRobot 数据集中写入 `meta/stats.json`，其中 action 统计只包含
`action_is_pad=False` 的真实 slot。

`conversion_tools/compute_task_v1_masked_norm_stats.py` 仍是旧 `task_v1` 30 Hz / H90 专用工具：
它硬编码了旧 schema 和 `sample_valid_h90`，不能直接用于
`umi_folder_to_lerobot.py` 的新通用输出。在训练新数据集前，必须先使用或编写与
`umi-folder-dual-hand-pose-lerobot` schema 兼容的 normalization 工具，并确保：

- 从 `action_is_pad` 逐 slot 排除填充 action。
- 动态读取 action horizon，不硬编码 H90。
- 使用新的 asset ID，不覆盖或沿用旧 norm stats。
- OpenPI dataset config、model action horizon、action mask 处理与转换时的 FPS/Horizon 保持一致。

## RGB 快速审核

```bash
/home/dzq/openpi/.venv/bin/python review_tools/rgb_review_app.py \
  --root /mnt/data/dzq/umi/data/task_v1 \
  --host 0.0.0.0 \
  --port 8090
```

网站支持首帧总览、10 FPS MP4、左右眼切换，以及将 episode 移入可恢复的 `.review_trash/`。审核缓存位于 `review_tools/cache/`。

## 随机回放并生成 MP4

```bash
python3 random_replay_5090/random_replay_5090.py \
  /mnt/data/dzq/umi/data/task_v1 \
  --camera-view front \
  --no-viewer
```

视频默认保存到：

```text
/home/dzq/data_deal/random_replay_5090/videos/
```

## OpenPI 训练契约

以下内容对应已有的旧 `task_v1` 30 Hz / H90 训练配置。新的通用转换数据集
在完成 dataset config 和 normalization 迁移前，不要直接套用该启动命令。

30 Hz/H90 数据需要 `action_horizon=90`、`action_sequence_keys=()`，并使用
`action_is_pad` 逐 slot 屏蔽 loss。当前训练配置为：

```text
pi05_umi_task_v1_hand_pose_30hz_h90_masked_with_head_roi_v1
```

训练参数：Pi0.5、三路相机、head ROI、`action_dim=32`、H90、batch size 64、
8 张 GPU、20,000 steps，每 2,000 steps 保存一次。

先运行 CPU smoke test：

```bash
cd /home/dzq/data_deal
PYTHONPATH=/home/dzq/openpi/src \
  /home/dzq/openpi/.venv/bin/python \
  training_tools/smoke_test_task_v1_openpi.py
```

也可以只检查启动条件而不启动训练：

```bash
/home/dzq/openpi/.venv/bin/python \
  training_tools/launch_task_v1_training.py --check-only
```

确认通过后启动训练：

```bash
cd /home/dzq/data_deal
/home/dzq/openpi/.venv/bin/python \
  training_tools/launch_task_v1_training.py \
  --memory-fraction 0.90
```

8 卡训练实测 `0.95` 会在第一步 NCCL 通信时因缺少额外工作显存而 OOM；
当前稳定值为 `0.90`，不要仅为了提高预分配比例改回 `0.95`。

默认实验名为：

```text
task_v1_hand_pose_30hz_h90_masked_with_head_roi_20260828
```

训练日志位于 `/home/dzq/openpi/logs/<实验名>.log`，checkpoint 位于：

```text
/mnt/data/dzq/openpi/checkpoints/
  pi05_umi_task_v1_hand_pose_30hz_h90_masked_with_head_roi_v1/<实验名>/
```

如果已有训练任务占用 GPU，可使用安全排队器；它不会终止现有任务：

```bash
nohup /home/dzq/openpi/.venv/bin/python \
  training_tools/queue_task_v1_roi_training.py \
  >/home/dzq/data_deal/training_tools/logs/task_v1_h90_queue.nohup.log 2>&1 &
```

## 查看完整参数

```bash
python3 flatten_data.py --help
python3 controller_to_hand_pose.py --help
/home/dzq/openpi/.venv/bin/python conversion_tools/umi_folder_to_lerobot.py --help
/home/dzq/openpi/.venv/bin/python conversion_tools/umi_folder_to_lerobot.py inspect --help
/home/dzq/openpi/.venv/bin/python conversion_tools/umi_folder_to_lerobot.py convert --help
python3 random_replay_5090/random_replay_5090.py --help
```
