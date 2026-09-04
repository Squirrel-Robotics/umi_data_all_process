# XR Pi0.5 数据采集网页

这是 `sol_data_collection.py` 的轻量局域网控制台。它直接导入现有
`DataCollector`，不通过终端输入模拟操作。

## 采集流程

1. 输入任务名称，点击“开始预检”；预检只监测数据，不写入 Episode。
   网页和命令行的默认任务名均为
   `Put the object on the box, then take it down.`。
2. 预检全部通过后点击“确认开始采集”。通过六阶段流程条确认当前处于准备、
   预检、有效起点、正式采集、复核还是提交阶段，并实时查看三路相机和
   9 路 topic 的计数、频率、当前值、
   时间戳来源、相对 head stamp 差值及共同覆盖范围。
3. 点击“停止采集”，全部 producer 线程停止并将临时文件
   flush/fsync，进入待审核状态。
4. 点击“保存本条”才原子提交 Episode 并更新 metadata；点击
   “丢弃本条”则清理 staging，Episode 编号不增加。
5. 保存或丢弃后等待配置的冷却时间（服务默认 10 秒），才能开始下一条。

## 9 路采集契约

- 三路相机：`head_rgb_stream`、`left_arm_rgb_stream`、
  `right_arm_rgb_stream`。`head_rgb_stream` 只允许接入 USB KONA 设备上
  `com.ssnwt.e6stream.debug` 在 TCP 8554 输出的 E6 双目 H.265，并把
  3200×1200 画面的右侧 1600×1200 定义为物理右目；机器人遥操作视频和
  腕部相机均不得代替该流。正式采集逐帧直存压缩 H.265 access unit，
  不做实时全帧解码；网页只解码约每秒一张关键帧并裁出右目用于预检。
  E6 帧的启动时钟会在每次 TCP Session 建立时映射到 XR 墙钟，避免头环
  系统时钟偏差影响与机器人数据的时间对齐。
  左右腕为鱼眼相机，采集格式为 MJPEG，分辨率
  640×480，频率 30 Hz。网页预览缓存始终只保留最新一帧，慢客户端会
  丢弃旧预览而不会反压采集线程。
- 两路臂末端位姿：`left_arm_end_pose`、`right_arm_end_pose`。
- 两路 Revo2 关节反馈：`left_revo2_joint_states`、
  `right_revo2_joint_states`。当前任务不订阅、不预检、也不保存 Revo2
  触觉流。
- 两路 Revo2 动作：`vr_left_revo2_joint_commands`、
  `vr_right_revo2_joint_commands`。

两路动作命令按 event-driven 流处理：有效起点前必须至少收到一条，之后
以 carry-forward 方式提供上下文，不用于截断有效区间终点。三路相机、
两路臂末端位姿和两路 Revo2 关节反馈均为 continuous 流，保存前会检查有效
起点后的新样本、尾部间隔和采集期间最大中断。旧的左右标量夹爪流不再
属于网页采集契约。

网页只显示在线“可对齐性”，不会对各 raw topic 做跨流重采样。训练数据
仍由后续转换程序按 10 Hz 和 50-step action horizon 生成。

E6 本体通过环境变量 `E6_SERIAL` 指定的 ADB 序列号管理。若头环本体未连接
或未授权，预检会保持重连且不会回退到任何机器人视频源。USB 重新插拔后
需要确保 XR 用户仍对对应 `/dev/bus/usb/...` 节点具有读写权限。

页面三路相机标题旁提供“恢复 E6”按钮，只能在数采关闭或预检阶段使用。
它会清理并重建 ADB server、对配置序列号对应的 E6 最多执行一次 USB
reset、重启 `com.ssnwt.e6stream.debug`，并重新建立 TCP 端口转发。若设备
仍处于 `05c6:f000` KONA-MTP 模式，页面会提示在头环开发者选项中
重新开启 USB 调试、重插 USB 并允许此电脑；若状态为 `unauthorized`，
页面会提示在头环内确认 USB 调试。
正式 Episode 采集中按钮被锁定，不会中断有效数据。

为了允许非 root 的网页服务只重置指定头环，需要先安装仓库中的
`xr_slave/udev/99-e6-adb.rules`，将其中的占位值替换为实际 `E6_SERIAL`，
然后重载规则并重新插拔一次 E6。规则不会向网页服务开放其他 USB 设备；
它还会取消 Ubuntu 针对 `05c6:f000` 的通用调制解调器
`usb_modeswitch`，避免干扰该 E6 从 KONA-MTP 阶段进入 Android/ADB。

## XR 运行

```bash
cd /home/xr/pi05/sdk_robot/examples
source /opt/ros/jazzy/setup.bash

/home/xr/robocontrol_ws/.venv/bin/python -u \
  collection_web/server.py \
  --host 0.0.0.0 \
  --port 8000
```

必须保持单进程、单 collector。服务会对输出目录持有文件锁，重复启动
会被拒绝。网页在开始前也会检查旧版 `sol_data_collection.py` 进程并
拒绝并行采集；仍不要在网页录制过程中手工启动旧脚本。

## 安全行为

- 网页无需口令，首次打开时自动建立短期同源 Session。
- POST 控制接口仍需要 Session 和 CSRF token。
- 重复点击和过期 `run_id` 不会重复执行。
- 浏览器断线不会停止正在进行的采集。
- 保存成功后会显示校验结果；下一条预检通过后可从成功提示直接开始。
- “最近保存”每 0.5 秒跟随状态轮询，并核对输出目录中真实存在的
  `episode_xxxx/episode.json`；手工删除的 Episode 不会继续显示。
- Episode 的 `duration` 仅统计正式采集有效区间，预检到停止的总耗时另存为
  `capture_duration`。
- 页面隐藏或离开活动采集阶段时会主动断开三路预览，返回后自动重连。
- 服务退出时，未保存的活动或待审核 Episode 默认丢弃，绝不自动保存。
- “停止采集”不是机器人急停。
- “恢复 E6”只操作匹配 `E6_SERIAL` 的 USB/ADB/视频链路，不操作机械臂、
  Revo2 或遥操控制模式。
