# Raw Data (`raw_data/`) Format & Parsing Guide

This document explains the `raw_data/` directory that is produced when recording
with `keep_raw_data=True` (CLI flag `--keep-raw-data`), and how to parse it.

## What is `raw_data/`?

During collection, every stream is written to a temporary file as a sequence of
`(timestamp, value)` records. Normally these temp files are aligned to the head
camera timeline (producing `episode.json`) and then deleted. When
`keep_raw_data` is enabled, the **pre-alignment** streams are copied verbatim
into the episode directory before cleanup:

```
collected_data/episode_XXXX/
├── episode.json            # aligned data (resampled to the camera timeline)
├── head_camera.mp4 ...     # video / images
└── raw_data/
    ├── manifest.json       # index: per-file record count + record format
    ├── left_arm_end_pose.pkl
    ├── right_arm_end_pose.pkl
    ├── waist_end_pose.pkl
    ├── left_arm_joint_states.pkl
    ├── right_arm_joint_states.pkl
    ├── ...
    ├── odometry.pkl
    ├── left_gripper_position.pkl
    └── ... (sensor / pose / joint / action streams)
```

> Camera frames are **not** included in `raw_data/` (they are large and already
> stored as video/images).

### `raw_data/` vs `episode.json`

| | `raw_data/*.pkl` | `episode.json` |
|---|---|---|
| Timeline | each stream's **own native rate** (e.g. ~100Hz) | resampled to the **head camera** timeline (e.g. ~30Hz) |
| Alignment | none (as received) | end pose: linear position + SLERP quaternion; joints/actions: linear; others: nearest-neighbor |
| Timestamps | original message header timestamps | per-frame camera timestamps |
| Use case | debugging, re-alignment, latency/jitter analysis | training / playback |

> Note: raw timestamps may contain short runs of **identical values** (the
> publisher sometimes stamps a burst of messages with the same time). This is
> expected; consumers should not assume strictly increasing timestamps.

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

- `records`: number of `(timestamp, value)` records in that file.
- `record_format`: a hint describing the shape of each record (see below).

## Each `.pkl` file

A `.pkl` is **not** a single object. It is a sequence of independently pickled
records appended one after another. Read it by calling `pickle.load(f)` in a loop
until `EOFError`:

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

## Record formats

There are three concrete record shapes. `timestamp` is always a `float` (Unix
seconds, from the message header).

### 1. End-effector pose — `*_end_pose.pkl`

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

### 2. Joint states — `*_joint_states.pkl`

```
(timestamp, positions, velocities, efforts)
```

`positions` / `velocities` / `efforts` are 1-D `numpy.ndarray` (`float32`).
`velocities` / `efforts` may be `None` if the message did not provide them.

```python
ts, positions, velocities, efforts = record
```

### 3. Generic sensors — `odometry.pkl`, `*_gripper_position.pkl`, ...

```
(timestamp, {"timestamp": float, "data": <protobuf message object>})
```

Here `data` is the **raw protobuf / betterproto2 message object** as received
from the SDK (e.g. `Odometry`). Unpickling it requires the `x2robot` package to
be importable in the same environment, otherwise `pickle.load` will fail to find
the class.

```python
ts, payload = record
msg = payload["data"]          # protobuf message; access fields directly, e.g. msg.pose ...
```

### 4. Actions — `*_actions.pkl`

Usually **empty** (`records == 0`). Action data is derived from the next frame's
state during alignment, so the raw action streams are typically not populated.

## Full example: parse one episode's `raw_data/`

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
        streams[fname[:-4]] = load_pkl(raw_dir / fname)  # strip ".pkl"
    return manifest, streams

# usage
manifest, streams = load_raw_data("collected_data/episode_0005")

pose = streams["left_arm_end_pose"]
print(f"left_arm_end_pose: {len(pose)} records")
ts0, p0 = pose[0]
print("first sample:", ts0, p0["position"])

js = streams["left_arm_joint_states"]
ts0, positions, velocities, efforts = js[0]
print("joint positions:", positions)
```

> To parse the generic-sensor `data` field (e.g. `odometry`), run inside an
> environment where `import x2robot` works (the same venv used for collection).
