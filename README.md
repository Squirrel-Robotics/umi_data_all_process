# UMI 数据处理与仿真回放工具

这个仓库用于完成 UMI 数据的目录整理、手柄位姿转换、质量检查，以及在 RTX 5090 上的 Quanta X2 + Revo2 MuJoCo 回放和 MP4 渲染。

## 功能概览

| 工具 | 用途 |
| --- | --- |
| `flatten_data.py` | 将多层采集包装目录中的数据单元汇总到同一层 |
| `controller_to_hand_pose.py` | 将左右手柄的绝对位姿转换为可直接右乘到 X2 + Revo2 末端的局部增量 |
| `random_replay_5090/random_replay_5090.py` | 随机或指定数据单元，完成校验、5090 仿真回放和 MP4 生成 |
| `random_replay_5090/replay_x2_revo2_relative_eef.py` | 5090 上的 MuJoCo IK、可视化和离屏渲染实现 |

> 数据整理和转换工具默认只预览计划，不会修改数据。确认输出无误后再加 `--execute`。

## 快速开始

在数据服务器上进入仓库：

```bash
cd /home/dzq/data_deal
```

### 1. 整理 `taskv1` 目录

先预览：

```bash
python3 flatten_data.py \
  /mnt/data/dzq/umi/data/taskv1 \
  --parent-pattern 'collector_run_*' \
  --remove-empty-parents \
  --verbose
```

确认后执行：

```bash
python3 flatten_data.py \
  /mnt/data/dzq/umi/data/taskv1 \
  --parent-pattern 'collector_run_*' \
  --remove-empty-parents \
  --execute
```

### 2. 批量生成 `hand_pose.csv`

先预览：

```bash
python3 controller_to_hand_pose.py \
  /mnt/data/dzq/umi/data/taskv1 \
  --input-glob '**/camera/e6_rgb_controller_poses.csv' \
  --skip-existing \
  --verbose
```

确认后执行：

```bash
python3 controller_to_hand_pose.py \
  /mnt/data/dzq/umi/data/taskv1 \
  --input-glob '**/camera/e6_rgb_controller_poses.csv' \
  --skip-existing \
  --report /mnt/data/dzq/umi/data/taskv1/hand_pose_report.json \
  --execute
```

### 3. 抽样回放并生成 MP4

随机选择一条数据，从机器人正面渲染 MP4：

```bash
python3 random_replay_5090/random_replay_5090.py \
  /mnt/data/dzq/umi/data/taskv1 \
  --camera-view front \
  --no-viewer
```

默认视频保存在：

```text
/home/dzq/data_deal/replay_videos/<episode>_seed<seed>.mp4
```

## 1. 数据目录整理

### 通用用法

```bash
python3 flatten_data.py SOURCE \
  --target TARGET \
  --parent-pattern 'run_*' \
  --child-pattern '*' \
  --entry-type dirs \
  --mode copy \
  --conflict rename \
  --execute
```

| 参数 | 说明 |
| --- | --- |
| `--parent-pattern GLOB` | 要展开的父目录；可重复传入 |
| `--child-pattern GLOB` | 父目录中要收集的子项；默认为 `*` |
| `--entry-type dirs\|files\|all` | 只处理目录、文件或全部 |
| `--mode move\|copy` | 移动或复制；默认为 `move` |
| `--conflict error\|skip\|rename` | 目标同名时停止、跳过或自动改名 |
| `--remove-empty-parents` | 移动完成后删除已经为空的包装目录 |
| `--execute` | 实际执行；不加时只预览 |

## 2. 手柄位姿转换

### 输出增量的定义

对每只手，先在标准 v3 手部坐标系中计算相对上一帧的增量：

```text
D_canonical[t] = inverse(T_hand[t-1]) @ T_hand[t]
```

默认的 `--target-frame-profile x2-revo2` 会将增量转换到机器人 EEF 坐标系：

```text
X_left  = Rx(+90 deg)
X_right = Rx(-90 deg)
D_robot = inverse(X_side) @ D_canonical @ X_side
```

新生成的 `hand_pose.csv` 可直接右乘到机器人末端目标，不要再额外旋转坐标系：

```text
T_target[0] = T_sdk_home
T_target[t] = T_target[t-1] @ D_robot[t]
```

`T_sdk_home` 是 Quanta X2 + Revo2 的 SDK 零位，不是 CSV 第一帧的绝对位姿。

### 单文件转换

```bash
python3 controller_to_hand_pose.py \
  /path/to/e6_rgb_controller_poses.csv \
  --output /path/to/hand_pose.csv \
  --execute
```

### 通用批处理

```bash
python3 controller_to_hand_pose.py /data/root \
  --input-glob '**/controller.csv' \
  --output-name hand_pose.csv \
  --frame-column frame_id \
  --timestamp-column timestamp_ns \
  --output-root /optional/output/root \
  --skip-existing \
  --execute
```

`--output-root` 会保留输入目录的相对结构。数据集数量已知时，可用 `--expected-count N` 防止漏处理。

### 坐标和标定配置

| 参数 | 适用场景 |
| --- | --- |
| `--coordinate-mode ros` | 默认；使用 `(x, y, z) = (-z, -x, y)_UMI` |
| `--coordinate-mode native` | 保留 UMI 原生坐标基 |
| `--target-frame-profile x2-revo2` | 默认；输出可直接右乘到 X2 + Revo2 EEF |
| `--target-frame-profile canonical-v3` | 保留与机器人无关的 v3 标准增量 |
| `--calibration-json FILE` | 为其他手柄或法兰结构提供左右手外参 |

自定义标定 JSON 中，左右手都需要以毫米为单位提供 `origin_mm`、`forward_mm` 和 `up_mm`：

```json
{
  "left": {
    "origin_mm": [0.0, 0.0, 0.0],
    "forward_mm": [100.0, 0.0, 0.0],
    "up_mm": [0.0, 0.0, 100.0]
  },
  "right": {
    "origin_mm": [0.0, 0.0, 0.0],
    "forward_mm": [100.0, 0.0, 0.0],
    "up_mm": [0.0, 0.0, 100.0]
  }
}
```

### 输出 CSV

每只手包含一个有效标志、局部平移 `dp` 和局部旋转四元数 `dq`：

```text
left_relative_valid
left_local_dpx, left_local_dpy, left_local_dpz
left_local_dqx, left_local_dqy, left_local_dqz, left_local_dqw

right_relative_valid
right_local_dpx, right_local_dpy, right_local_dpz
right_local_dqx, right_local_dqy, right_local_dqz, right_local_dqw
```

平移单位为米，四元数顺序为 `(x, y, z, w)`。无效增量会写成单位变换，同时将 `relative_valid` 设为 `0`。

## 3. RTX 5090 仿真回放

### 回放流程

`random_replay_5090.py` 会自动完成：

1. 随机或按名称选择一个数据单元。
2. 识别已有的 `hand_pose*.csv`，或临时转换手柄 CSV。
3. 检查时间戳、四元数、单帧突变和静止/冻结轨迹。
4. 将最小回放包传到 5090。
5. 在 MuJoCo 中从 `sdk_home` 零位进行 IK 回放。
6. 离屏渲染并编码 MP4。
7. 将 MP4 下载回数据服务器。
8. 可选地在 5090 桌面打开循环回放窗口。

### 常用运行方式

只查看抽样结果：

```bash
python3 random_replay_5090/random_replay_5090.py \
  /mnt/data/dzq/umi/data/taskv1 \
  --dry-run
```

只抽样、转换和本地校验，不连接 5090：

```bash
python3 random_replay_5090/random_replay_5090.py \
  /mnt/data/dzq/umi/data/taskv1 \
  --seed 20260826 \
  --prepare-only
```

指定一条数据并从正面回放：

```bash
python3 random_replay_5090/random_replay_5090.py \
  /mnt/data/dzq/umi/data/taskv1 \
  --episode 20260825_105700_247429960_2371 \
  --camera-view front \
  --replace-running
```

指定输出文件和视频规格：

```bash
python3 random_replay_5090/random_replay_5090.py \
  /mnt/data/dzq/umi/data/taskv1 \
  --video-output /home/dzq/data_deal/replay_videos/check.mp4 \
  --video-width 1920 \
  --video-height 1080 \
  --video-fps 60 \
  --camera-view front \
  --no-viewer
```

### 直接回放已有的 `hand_pose_v3.csv`

`hand_pose_v3.csv` 保存的是“相对上一帧，并在上一帧自身 EEF 坐标系中表达”的增量，不是绝对位姿。

```bash
python3 random_replay_5090/random_replay_5090.py \
  /mnt/data/dzq/umi/data/最新100条 \
  --source-glob '*/e6/hand_pose_v3.csv' \
  --episode 20260811_071021_738576644_80808 \
  --camera-view front \
  --no-viewer
```

### EEF 坐标映射

| `--frame-profile` | 行为 |
| --- | --- |
| `auto` | 默认；对 `hand_pose_v3.csv` 应用左 `+90°`/右 `-90°`，其他 CSV 使用单位映射 |
| `identity` | 输入已经在 X2 + Revo2 EEF 坐标系中，不再转换 |
| `hand-pose-v3` | 强制按 v3 标准坐标左 `+90°`/右 `-90°` 转换 |

`controller_to_hand_pose.py` 默认生成的 `hand_pose.csv` 已经是 X2 + Revo2 EEF 增量，回放时应使用 `identity`；`auto` 也会对该文件名自动选择单位映射。

### 质量检查和回放参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--fail-on-warning` | 关闭 | 出现 QA 警告时不启动 5090 回放 |
| `--min-path-mm` | `1` | 单手累积路径过小时报告疑似静止/冻结；`0` 表示禁用 |
| `--max-step-mm` | `100` | 单帧最大平移阈值 |
| `--max-rotation-deg` | `90` | 单帧最大旋转阈值 |
| `--max-dt-ms` | `200` | 最大帧间隔阈值 |
| `--max-frames N` | 全部 | 只回放前 `N` 帧 |
| `--playback-rate N` | `1.0` | 回放速度倍率 |
| `--translation-scale N` | `1.0` | 平移增量缩放 |
| `--rotation-scale N` | `1.0` | 旋转增量缩放 |
| `--camera-view operator\|front` | `operator` | 操作者视角或机器人正面视角 |
| `--no-viewer` | 关闭 | 只生成 MP4，不打开交互窗口 |

### 停止交互回放

```bash
ssh dzq@192.168.110.199 \
  '/home/dzq/umi_x2_mujoco/random_replay/launch_random_replay_5090.sh --stop'
```

## 输出和缓存位置

| 位置 | 内容 | Git 跟踪 |
| --- | --- | --- |
| `/home/dzq/data_deal/.replay_cache/` | 数据服务器上的临时转换、QA 报告和回放包 | 否 |
| `/home/dzq/data_deal/replay_videos/` | 从 5090 下载回来的 MP4 | 否 |
| `/home/dzq/umi_x2_mujoco/random_replay/` | 5090 上的日志、PID、轨迹和临时 MP4 | 不在本仓库 |

5090 默认主机为 `dzq@192.168.110.199`。如果尚未配置 SSH 密钥，首次连接会请求输入密码，后续步骤会复用同一连接。

## 常见问题

### 脚本只打印计划，没有生成文件

`flatten_data.py` 和 `controller_to_hand_pose.py` 默认为预览模式。确认计划后加 `--execute`。

### 目标 `hand_pose.csv` 已经存在

- 保留旧文件并处理其他数据：使用 `--skip-existing`。
- 确认需要原子替换旧文件：使用 `--overwrite`。

### 视频在哪里

脚本结束时会打印最终 MP4 路径。未传入 `--video-output` 时，在 `/home/dzq/data_deal/replay_videos/` 中查找。

### 查看所有参数

```bash
python3 flatten_data.py --help
python3 controller_to_hand_pose.py --help
python3 random_replay_5090/random_replay_5090.py --help
```
