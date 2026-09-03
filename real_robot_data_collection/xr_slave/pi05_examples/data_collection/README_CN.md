# 数据采集工具

这个目录包含可定制的数据采集工具：

- `collection_config.py` - 采集配置定义和预设
- `data_collector.py` - 数据采集器实现
- `RAW_DATA_CN.md` - 可选 `raw_data/` 输出的格式与解析说明

## 使用方法

用户可以直接修改这些文件来定制自己的数据采集流程。

## 示例

参见 `data_collection_example.py`

## 对齐数据的准确性(`episode.json`)

`episode.json` 是真正用于训练的产物。每条原始流都会被重采样到主相机时间轴上,
因此准确性取决于插值方式:

| 字段 | 插值方式 | 说明 |
|---|---|---|
| 末端位姿 **位置** (x, y, z) | 线性插值 | 原始 ~100Hz ≫ 目标 ~30Hz,误差在亚毫米级 |
| 末端位姿 **姿态** (四元数) | **SLERP**(最短路径) | 四元数逐分量线性插值是错误的;SLERP 保持单位模长和匀角速度 |
| 关节状态 / 关节动作 | 线性插值 | 当 `downsample_joint_states=True` 时 |
| 其他传感器 | 最近邻 | |

以上仅在 `downsample_joint_states=True`(默认)时生效;否则所有流都退化为最近邻
(贴到真实采样点)。

### 动作标签

`observation.<part>_end_pose` 是**第 `i` 帧**的插值位姿。
`action.<part>_end_pose_action` 是**目标位姿 = 第 `i+1` 帧的位姿**
(最后一帧复用自身位姿)。即动作比观测超前一帧,可作为模仿学习的绝对目标标签。

## 原始数据

录制时加上 `--keep-raw-data`(或 `keep_raw_data=True`),未对齐的原始流会保存到
`episode_XXXX/raw_data/` 下。目录结构、`manifest.json` 字段、记录格式以及可运行的
解析示例,详见 [RAW_DATA_CN.md](RAW_DATA_CN.md)。
