# XR 真机数采与 Revo2 遥操映射

## 这个分支包含什么

该目录整理了 2026-09-03 真机上正在使用的 XR 数采代码，包括：

- XR 主机上的 Meta Quest 控制器输入、侧板机 UDP 转发和不暂停兼容层。
- XR 从机上的 Revo2 双手串口控制、关节/触觉反馈和成功指令发布。
- 三路视频、双臂末端位姿、双手关节状态和双手实际指令的真机数据采集。
- 数采 Web 界面、事务型 episode 保存、systemd 服务和部署配置。

源码来自真机当前文件，没有收录备份、`__pycache__`、密钥、Web 凭据或数据集。为避免把设备唯一标识和内网地址写入公开仓库，E6/Revo2 序列号和主从机 IP 改为必填环境变量；其余采集和映射逻辑保持真机版本。

## 目录与真机路径

| 仓库目录 | 主机角色 | 真机对应路径 |
| --- | --- | --- |
| `xr_master/revo2_vr_bridge/` | XR 主机 | `/home/xr/robocontrol_ws/revo2_vr_bridge/` |
| `xr_slave/revo2_vr_bridge/` | X2 从机 | `/home/xr/robocontrol_ws/revo2_vr_bridge/` |
| `xr_slave/pi05_examples/` | X2 从机 | `/home/xr/pi05/sdk_robot/examples/` |
| `xr_slave/systemd/` | X2 从机 | `/home/xr/.config/systemd/user/` |
| `xr_slave/udev/` | X2 从机 | 根据系统策略安装到 `/etc/udev/rules.d/` |
| `deploy/` | 两端 | 环境变量样例和 XR 主机 Compose 覆盖片段 |

## 数据与控制流

```text
Meta Quest controller
  -> XR master xr_vr_driver_node
  -> /joy_trigger/state/{left,right}
     -> index trigger -> databridge -> slave Float64 trigger topic
     -> side grip -> UDP/39157 -> slave thumb-rotation input
  -> slave vr_trigger_revo2_bridge.py
  -> BrainCo Revo2 RS485
  -> /revo2/{state,tactile,command}/{left,right}
  -> vr_ros_command_bridge.py
  -> DataCollector raw topic streams

E6 right-eye H265 + left/right wrist MJPEG + dual EEF poses
  -> DataCollector
  -> staging episode
  -> operator Save or Discard
  -> atomic episode_xxxx publish
```

## VR 板机到 Revo2 六维的映射

Revo2 指令顺序是：

```text
[thumb_flex, thumb_rotation, index, middle, ring, pinky]
```

- 食指板机为 `0` 时，目标是 `[0, 0, 0, 0, 0, 0]`。
- 食指板机的 `0..0.4` 先将拇指屈曲预定位到 `400`。
- 食指板机的 `0.4..1.0` 继续将拇指屈曲到左手 `500`/右手 `517`，并将食指、中指同步到 `593`。
- 环指和小指保持 `0`。
- 侧板机独立控制第 2 维 `thumb_rotation`，`0..1` 平滑映射到 `0..814`。
- 两路输入都使用 `0.02/0.98` 端点死区和 smoothstep，指令频率不超过 10 Hz。

`xr_master/revo2_vr_bridge/vr_grip_udp_sender.py` 还会将侧板机置零后重发给 databridge，并过滤由侧板机导致的遥操停止请求。B/A 键、摇杆和食指板机保持原始值。

## 真机数采契约

Quanta X2 当前配置共保存 9 路数据：

1. `head_rgb_stream`：E6 双目 H.265，源画面 3200×1200 @ 60 Hz，只将右侧 1600×1200 定义为物理右目。
2. `left_arm_rgb_stream`：左腕 MJPEG，640×480 @ 30 Hz。
3. `right_arm_rgb_stream`：右腕 MJPEG，640×480 @ 30 Hz。
4. `left_arm_end_pose`。
5. `right_arm_end_pose`。
6. `left_revo2_joint_states`。
7. `right_revo2_joint_states`。
8. `vr_left_revo2_joint_commands`。
9. `vr_right_revo2_joint_commands`。

Revo2 动作是桥接程序已成功下发的六维指令，不是用下一帧测量状态替代。三路视频、双臂 EEF 和双手状态是 continuous 流，双手动作是 event-driven 流，保存时使用 carry-forward 提供起点上下文。

默认使用 `raw_topics_pickle_v1` 保存每路原始时间戳流。Web 流程为“开始预检 → 确认采集 → 停止 → 保存/丢弃”；保存前完成 flush/fsync 和完整性校验，成功后以原子 rename 发布 episode。

## 配置与部署

先复制 [`deploy/xr-real-data-collection.env.example`](deploy/xr-real-data-collection.env.example)，用真机当前值替换所有 `CHANGE_ME`。该文件不应提交真实值。

XR 主机：

1. 将 `xr_master/revo2_vr_bridge/` 复制到 `/home/xr/robocontrol_ws/revo2_vr_bridge/`。
2. 将 `vr_driver_no_side_pause.launch.py` 同步到 `/opt/xr/config/revo2_vr_bridge/`。
3. 在原有 `cx002_master` Compose 上应用 [`deploy/xr_master_compose.override.yml`](deploy/xr_master_compose.override.yml)。
4. 导出 `REVO2_GRIP_DESTINATION=<XR_SLAVE_IP>` 后运行 `vr_grip_sender_control.sh start`。

X2 从机：

1. 将 `xr_slave/pi05_examples/` 内容同步到 `/home/xr/pi05/sdk_robot/examples/`。
2. 将 `xr_slave/revo2_vr_bridge/` 同步到 `/home/xr/robocontrol_ws/revo2_vr_bridge/`。
3. 导出 Revo2 双手序列号、串口路径和 `REVO2_GRIP_SOURCE_IP=<XR_MASTER_IP>`，然后运行 `vr_trigger_control.sh start`。
4. 将用户级 `sol-collection-web.service` 安装到 `~/.config/systemd/user/`，在 `~/.config/xr-real-data-collection.env` 填入 E6 和 ROS bridge 配置后重载并启动。
5. 通过 `http://<XR_SLAVE_IP>:8000/` 进入数采 Web 界面。

## 验证顺序

1. XR 主机的 `adb devices -l` 必须将 Quest 显示为 `device`，而不是 `unauthorized`。
2. `xr_vr_driver_node` 日志不应持续出现 `Connection to VR device failed`。
3. `/joy_trigger/state/right` 与 `/joy_trigger/state/left` 的 `trigger/grip` 应在实际按压时连续变化。
4. X2 从机应在 UDP/39157 监听，Revo2 bridge 启动日志应明确验证左右手型号。
5. `/revo2/state/{left,right}` 应持续发布六维反馈，板机变化时 `/revo2/command/{left,right}` 应出现成功指令。
6. Web 页面只有在 9 路契约全部通过预检后才能确认采集。

## 安全边界

- Revo2 串口必须只有一个 Modbus owner；`vr_trigger_control.sh` 会先停止已知的竞争容器并等待串口释放。
- 上电或更换手后先用 `--probe-only` 验证左右手；无需打开串口时可用 `--dry-run` 检查 ROS 映射。
- 设备序列号、手型、Normalized 模式和五指触觉使能任一不符都会 fail closed。
- 当前 XR 主机配置会抑制“按侧板机导致遥操暂停”的请求；使用者必须另行保留可靠的急停方式。
- Web 页面的“停止采集”只停止记录，不是机器人急停。
- 数采期间不要并行启动旧的命令行 collector；Web 服务和输出目录锁会拒绝大部分重复启动。

## 本地自检

```bash
cd real_robot_data_collection
python3 -m py_compile xr_slave/pi05_examples/sol_data_collection.py xr_slave/pi05_examples/data_collection/*.py xr_slave/pi05_examples/collection_web/server.py xr_slave/revo2_vr_bridge/*.py xr_master/revo2_vr_bridge/*.py
bash -n xr_slave/revo2_vr_bridge/vr_trigger_control.sh
bash -n xr_master/revo2_vr_bridge/vr_grip_sender_control.sh
node --check xr_slave/pi05_examples/collection_web/static/app.js
```

Revo2 单元测试：

```bash
cd real_robot_data_collection/xr_slave/revo2_vr_bridge
REVO2_LEFT_SERIAL=test-left REVO2_RIGHT_SERIAL=test-right python3 -m unittest -v test_vr_trigger_revo2_bridge.py
```
