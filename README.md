# UMI 数据处理工具

用于整理 UMI 原始数据、生成末端增量、审核 RGB、转换 LeRobot 数据，以及在 RTX 5090 上回放。

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
4. 用 `conversion_tools/task_v1_to_lerobot.py` 预检并转换。
5. 用 `conversion_tools/compute_task_v1_masked_norm_stats.py` 生成 OpenPI normalization。

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

## `conversion_tools` 用法

### 脚本选择

| 脚本 | 适用场景 |
| --- | --- |
| `task_v1_to_lerobot.py` | 当前原生采集布局；E6、cam0、cam1、Revo2 串口数据；默认处理 `task_v1` |
| `umi_to_lerobot.py` | 旧版通用 UMI 布局；包含 `e6/hand_pose.csv` 和左右相机时间戳 |
| `compute_task_v1_masked_norm_stats.py` | 为已经转换的 H90 数据计算 OpenPI normalization |

两个转换器都提供三个子命令：

```bash
self-test   # 只运行数学单元测试
inspect     # 只读预检，不创建目标目录
convert     # 正式转换
```

### 当前原生数据：`task_v1_to_lerobot.py`

先运行数学检查：

```bash
/home/dzq/openpi/.venv/bin/python \
  conversion_tools/task_v1_to_lerobot.py self-test
```

只读检查全部源数据：

```bash
/home/dzq/openpi/.venv/bin/python \
  conversion_tools/task_v1_to_lerobot.py inspect \
  --source /mnt/data/dzq/umi/data/task_v1 \
  --target /mnt/data/dzq/umi/datasets/task_v1_lerobot_30hz_h90 \
  --expected-episodes 0 \
  --max-alignment-ms 100 \
  --max-hand-age-ms 100 \
  --hand-alignment nearest \
  --compact
```

`--expected-episodes 0` 表示预检时不限制数量。正式转换前，建议将它改成预检得到的实际源 episode 数，防止漏数据。

只检查一条数据：

```bash
/home/dzq/openpi/.venv/bin/python \
  conversion_tools/task_v1_to_lerobot.py inspect \
  --source /mnt/data/dzq/umi/data/task_v1 \
  --only 20260826_092507_421505161_65015
```

正式转换：

```bash
/home/dzq/openpi/.venv/bin/python \
  conversion_tools/task_v1_to_lerobot.py convert \
  --source /mnt/data/dzq/umi/data/task_v1 \
  --target /mnt/data/dzq/umi/datasets/task_v1_lerobot_30hz_h90 \
  --repo-id dzq/task_v1_lerobot_30hz_h90 \
  --task 'Put the object into the box.' \
  --expected-episodes 143 \
  --max-alignment-ms 100 \
  --max-hand-age-ms 100 \
  --hand-alignment nearest \
  --video-workers 4 \
  --confirm CREATE_LEROBOT_DATASET
```

当前 `task_v1` 全量预检得到 143 条源 episode；其中 1 条会因有效时间段中断拆成两段，所以预计输出 144 个 episode。目标目录必须不存在，或者是完全空目录并显式增加 `--replace-empty-target`。数据数量改变后，请重新运行 `inspect` 并更新 `--expected-episodes`。

转换契约：

- 三路视觉：E6 右眼、cam0 左腕、cam1 右腕。
- 以时间戳最近邻对齐，误差不超过 100 ms，输出 30 Hz。
- `observation.state` 为 `(30,)`。
- `action` 为 `(90, 30)`，即 90-step action horizon。
- `action_is_pad` 为 `(90,)`，终点以最后目标补齐但训练 loss 必须屏蔽 padding。
- 90 个 action 在 30 Hz 下覆盖 3 秒。
- `state[t] = inverse(T[t-1]) @ T[t]`。
- `action[t,k] = inverse(T[t]) @ T[t+k]`，整个 H90 使用同一个当前帧锚点。
- OpenPI 中保持 `action_sequence_keys = ()`，不要再次切 future chunk 或再次计算 EEF delta。

### 旧版布局：`umi_to_lerobot.py`

接口与上面一致，主要区别是读取旧版 `e6/hand_pose.csv` 和左右相机文件：

```bash
/home/dzq/openpi/.venv/bin/python \
  conversion_tools/umi_to_lerobot.py inspect \
  --source '/mnt/data/dzq/umi/data/最新100条' \
  --target /mnt/data/dzq/umi/datasets/umi_lerobot_h90 \
  --expected-episodes 0 \
  --compact
```

正式转换：

```bash
/home/dzq/openpi/.venv/bin/python \
  conversion_tools/umi_to_lerobot.py convert \
  --source '/mnt/data/dzq/umi/data/最新100条' \
  --target /mnt/data/dzq/umi/datasets/umi_lerobot_h90 \
  --repo-id dzq/umi_lerobot_h90 \
  --task 'Put the object into the box.' \
  --expected-episodes 0 \
  --confirm CREATE_LEROBOT_DATASET
```

### 计算 OpenPI normalization

转换成功后运行：

```bash
/home/dzq/openpi/.venv/bin/python \
  conversion_tools/compute_task_v1_masked_norm_stats.py \
  --dataset /mnt/data/dzq/umi/datasets/task_v1_lerobot_30hz_h90 \
  --asset-id umi_task_v1_hand_pose_30hz_h90_masked_v1
```

输出到：

```text
/mnt/data/dzq/openpi/data/assets/<asset-id>/norm_stats.json
/mnt/data/dzq/openpi/data/assets/<asset-id>/norm_stats_audit.json
```

全部 state 参与统计；action 只统计 `action_is_pad=False` 的真实 slot。

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

30 Hz/H90 数据需要 `action_horizon=90`、`action_sequence_keys=()`，并使用
`action_is_pad` 逐 slot 屏蔽 loss。现有 `training_tools/` 启动器属于旧的
10 Hz/H50 配置，不能直接用于这套数据。

## 查看完整参数

```bash
python3 flatten_data.py --help
python3 controller_to_hand_pose.py --help
/home/dzq/openpi/.venv/bin/python conversion_tools/task_v1_to_lerobot.py --help
/home/dzq/openpi/.venv/bin/python conversion_tools/task_v1_to_lerobot.py inspect --help
/home/dzq/openpi/.venv/bin/python conversion_tools/task_v1_to_lerobot.py convert --help
python3 random_replay_5090/random_replay_5090.py --help
```
