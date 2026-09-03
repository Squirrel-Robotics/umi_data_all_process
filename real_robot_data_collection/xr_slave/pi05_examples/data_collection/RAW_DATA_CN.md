# 原始数据(`raw_data/`)格式与解析说明

本文档说明在录制时开启 `keep_raw_data=True`(命令行 `--keep-raw-data`)后生成的
`raw_data/` 目录,以及如何解析它。

## `raw_data/` 是什么?

采集过程中,每一路数据都以 `(timestamp, value)` 记录序列写入临时文件。正常情况下
这些临时文件会被对齐到主相机(head camera)时间轴生成 `episode.json`,然后删除。
开启 `keep_raw_data` 后,会在清理前把这些**未对齐的原始流**原样拷贝到 episode 目录:

```
collected_data/episode_XXXX/
├── episode.json            # 对齐后的数据(重采样到相机时间轴)
├── head_camera.mp4 ...     # 视频 / 图像
└── raw_data/
    ├── manifest.json       # 索引:每个文件的记录条数 + 记录格式
    ├── left_arm_end_pose.pkl
    ├── right_arm_end_pose.pkl
    ├── waist_end_pose.pkl
    ├── left_arm_joint_states.pkl
    ├── right_arm_joint_states.pkl
    ├── ...
    ├── odometry.pkl
    ├── left_gripper_position.pkl
    └── ...(传感器 / 位姿 / 关节 / 动作 流)
```

> 相机帧**不**包含在 `raw_data/` 中(体积大,且已存为视频/图像)。

### `raw_data/` 与 `episode.json` 的区别

| | `raw_data/*.pkl` | `episode.json` |
|---|---|---|
| 时间轴 | 每路流**各自的原始频率**(如 ~100Hz) | 重采样到**主相机**时间轴(如 ~30Hz) |
| 对齐 | 无(原样保存) | 末端位姿:位置线性 + 四元数 SLERP;关节/动作:线性;其他:最近邻 |
| 时间戳 | 原始消息 header 时间戳 | 每帧的相机时间戳 |
| 用途 | 调试、重新对齐、延迟/抖动分析 | 训练 / 回放 |

> 注意:原始时间戳可能出现**短时间内完全相同**的情况(发布端有时会给一批消息打同一个
> 时间戳),这是正常现象;解析时不要假设时间戳严格递增。

## `manifest.json`

```json
{
  "episode_id": 5,
  "created": "2026-06-13T14:34:43.620901",
  "note": "Pre-alignment raw streams. Each .pkl is a sequence of pickled records; read with repeated pickle.load until EOF.",
  "files": {
    "left_arm_end_pose.pkl":     { "records": 1434, "record_format": "(timestamp, dict[position, orientation])" },
    "left_arm_joint_states.pkl": { "records": 2884, "record_format": "(timestamp, positions, velocities, efforts)" },
    "odometry.pkl":              { "records": 721,  "record_format": "(timestamp, dict[timestamp, data])" },
    "left_arm_actions.pkl":      { "records": 0,    "record_format": "empty" }
  }
}
```

- `records`:该文件中 `(timestamp, value)` 记录的条数。
- `record_format`:每条记录结构的提示(见下文)。

## 每个 `.pkl` 文件

`.pkl` **不是单个对象**,而是一条接一条、独立 pickle 的记录序列。读取方式是循环调用
`pickle.load(f)` 直到 `EOFError`:

```python
import pickle

def load_pkl(path):
    records = []
    with open(path, "rb") as f:
        while True:
            try:
                records.append(pickle.load(f))
            except EOFError:
                break
    return records
```

## 记录格式

共有三种具体结构。`timestamp` 始终是 `float`(Unix 秒,取自消息 header)。

### 1. 末端位姿 —— `*_end_pose.pkl`

```
(timestamp, {
    "position":    {"x": float, "y": float, "z": float},
    "orientation": {"x": float, "y": float, "z": float, "w": float},
})
```

```python
ts, pose = record
x, y, z = pose["position"]["x"], pose["position"]["y"], pose["position"]["z"]
qx, qy, qz, qw = (pose["orientation"][k] for k in "xyzw")
```

### 2. 关节状态 —— `*_joint_states.pkl`

```
(timestamp, positions, velocities, efforts)
```

`positions` / `velocities` / `efforts` 是一维 `numpy.ndarray`(`float32`)。
当消息未提供时,`velocities` / `efforts` 可能为 `None`。

```python
ts, positions, velocities, efforts = record
```

### 3. 通用传感器 —— `odometry.pkl`、`*_gripper_position.pkl` 等

```
(timestamp, {"timestamp": float, "data": <protobuf 消息对象>})
```

这里的 `data` 是从 SDK 收到的**原始 protobuf / betterproto2 消息对象**(如
`Odometry`)。反序列化它需要在同一环境里能 `import x2robot`,否则 `pickle.load`
会因找不到对应的类而失败。

```python
ts, payload = record
msg = payload["data"]          # protobuf 消息;直接访问字段,如 msg.pose ...
```

### 4. 动作 —— `*_actions.pkl`

通常为**空**(`records == 0`)。动作数据是在对齐阶段由下一帧的状态推导出来的,因此原始
动作流一般不写入数据。

## 完整示例:解析一个 episode 的 `raw_data/`

```python
import json
import pickle
from pathlib import Path

def load_pkl(path):
    out = []
    with open(path, "rb") as f:
        while True:
            try:
                out.append(pickle.load(f))
            except EOFError:
                break
    return out

def load_raw_data(episode_dir):
    raw_dir = Path(episode_dir) / "raw_data"
    manifest = json.loads((raw_dir / "manifest.json").read_text())

    streams = {}
    for fname, info in manifest["files"].items():
        if info["records"] == 0:
            continue
        streams[fname[:-4]] = load_pkl(raw_dir / fname)  # 去掉 ".pkl"
    return manifest, streams

# 用法
manifest, streams = load_raw_data("collected_data/episode_0005")

pose = streams["left_arm_end_pose"]
print(f"left_arm_end_pose: {len(pose)} 条记录")
ts0, p0 = pose[0]
print("第一条:", ts0, p0["position"])

js = streams["left_arm_joint_states"]
ts0, positions, velocities, efforts = js[0]
print("关节角:", positions)
```

> 要解析通用传感器的 `data` 字段(如 `odometry`),请在能 `import x2robot` 的环境
> (即采集时使用的同一个 venv)中运行。
