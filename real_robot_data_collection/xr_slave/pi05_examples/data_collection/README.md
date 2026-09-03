# Data Collection Tools

This directory contains customizable data collection tools:

- `collection_config.py` - Collection configuration definitions and presets
- `data_collector.py` - Data collector implementation
- `RAW_DATA.md` - Format & parsing guide for the optional `raw_data/` output

## Usage

Users can directly modify these files to customize their own data collection workflow.

## Examples

See `data_collection_example.py`

## Aligned Data Accuracy (`episode.json`)

`episode.json` is what you ship for training. Each raw stream is resampled onto the
head-camera timeline, so accuracy depends on the interpolation method:

| Field | Method | Notes |
|---|---|---|
| End-pose **position** (x, y, z) | linear interpolation | raw ~100Hz ≫ target ~30Hz, so error is sub-millimeter |
| End-pose **orientation** (quaternion) | **SLERP** (shortest path) | component-wise linear interp of a quaternion is incorrect; SLERP keeps unit norm and constant angular velocity |
| Joint states / joint actions | linear interpolation | when `downsample_joint_states=True` |
| Other sensors | nearest-neighbor | |

These run only when `downsample_joint_states=True` (the default); otherwise every
stream falls back to nearest-neighbor (snapping to real samples).

### Action labels

`observation.<part>_end_pose` is the interpolated pose **at frame `i`**.
`action.<part>_end_pose_action` is the **target pose = the pose at frame `i+1`**
(the last frame reuses its own pose). So the action leads the observation by one
frame and is suitable as an absolute-target label for imitation learning.

## Raw Data

When recording with `--keep-raw-data` (or `keep_raw_data=True`), the pre-alignment
streams are saved under `episode_XXXX/raw_data/`. See [RAW_DATA.md](RAW_DATA.md)
for the directory layout, `manifest.json` schema, record formats, and a runnable
parsing example.
