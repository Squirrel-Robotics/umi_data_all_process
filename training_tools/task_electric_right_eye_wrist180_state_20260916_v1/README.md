# task_electric：E6 右眼 + 双腕 180° + state

选取 `/mnt/data/dzq/umi_v2/data/task_electric/pass_sessions.txt` 中全部 **115 条** session；预检无失败，不拆段、不排除任何选中 session。输出 **24,182 帧**，10 Hz / H50。原始数据只读。

英文任务指令：`Insert the battery into the empty slot in the middle.`

## 数据与图像

输出：`/mnt/data/dzq/umi_v2/datasets/task_electric_lerobot_10hz_h50_right_eye_wrist180_v1`

- `cam_high`：E6 实际解码图像的完整右半幅，不能用过时的 metadata 宽度；保持比例、letterbox 到 640×480，不加固定 ROI。
- **左腕：cam1，右腕：cam0**（本批数据由用户确认）。仅修正视频的左右映射，state/action 左右手通道不交换。两路先 resize，再各自旋转 180°，再编码；训练时不重复旋转。
- 三路 H264 / YUV420P / CRF 0 无损编码。每帧与选帧、裁剪、旋转后的源像素比较，并验证输出时间戳为精确 10 Hz。
- 三路相机按时间戳对齐，最大误差 ≤100 ms。预检左腕（cam1）/右腕（cam0）实际最大误差约 82.23 / 84.08 ms；手指状态最近邻时间误差也 ≤100 ms。

数值规则沿用之前转换：state 30 维为双手相对上一帧的自身坐标系位姿增量（位置 + rot6d），加双手绝对手指角度；action 为 `[50,30]`，每个未来目标共用当前帧 EEF 锚点。尾部以 `action_is_pad` 屏蔽 padding，不删 H50 不足的观测，不再次做 delta，不按未来 action 行二次切片。

## 训练

| 项目 | 设置 |
|---|---|
| 模型 | Pi0.5，`discrete_state_input=True` |
| GPU | 0–7，FSDP 8 |
| Batch | 全局 64，每卡 8 |
| DataLoader | 32 workers |
| 步数 | 20,000 |
| Checkpoint | 每 1,000 步保存并保留 |
| 初始化 | 从 `/mnt/data/checkpoints/pi05_base/params` 新训练 |
| 学习率 | warmup 500，peak 2.5e-5，decay 20k，final 2.5e-6 |
| 显存 | XLA 预分配比例 0.90 |

state 先归一化、离散化后拼入 prompt token；接口输入补齐到 32 维。使用本任务重新计算的 masked norm stats，不复用旧任务统计：

`/mnt/data/dzq/openpi/data/assets/umi_task_electric_10hz_h50_right_eye_wrist180_masked_v1`

配置：`pi05_umi_task_electric_10hz_h50_wrist180_state_20k_b64_w32_8gpu_v1`

实验：`task_electric_right_eye_wrist180_state_10hz_h50_20k_b64_w32_8gpu_20260916_v1`

Checkpoint 根目录：

```text
/mnt/data/dzq/openpi/checkpoints/pi05_umi_task_electric_10hz_h50_wrist180_state_20k_b64_w32_8gpu_v1/task_electric_right_eye_wrist180_state_10hz_h50_20k_b64_w32_8gpu_20260916_v1/
```

## 查看进度

工具目录：`/home/dzq/data_deal/training_tools/task_electric_right_eye_wrist180_state_20260916_v1`

```bash
# 当前阶段
cat /home/dzq/data_deal/training_tools/task_electric_right_eye_wrist180_state_20260916_v1/pipeline.status.json
# 转换日志
tail -f /home/dzq/data_deal/training_tools/task_electric_right_eye_wrist180_state_20260916_v1/conversion.log
# 训练日志：实际训练进程启动后才会出现
tail -f /home/dzq/openpi/logs/task_electric_right_eye_wrist180_state_10hz_h50_20k_b64_w32_8gpu_20260916_v1.log
```

流程为转换 → 新归一化统计 → CPU smoke → 等待八卡空闲 → 训练。`continue_pipeline.py` 已负责自动衔接，**不要重复启动**。任何转换/校验失败都会阻止训练；不杀其他 GPU 进程、不自动重试、不覆盖旧数据或 checkpoint。

`smoke_training.py` 会从原始数据重新核对全部数值目标、三路视频的来源/处理/像素证据、归一化，以及真实 dataloader 和 state token 输入。`smoke.report.json` 与训练日志旁 `.launch.json` 记录核验和实际启动信息。

## 转换脚本与来源

`converter_electric.py` 基于现有 strict 转换器，增加根目录清单支持、本批数据的 cam1→左腕 / cam0→右腕映射、可选腕图 180°、逐帧无损证据和安全发布；未改 state/action 计算函数。`inspect` 不写数据，`convert` 要求明确确认参数。

```bash
cd /home/dzq/data_deal/training_tools/task_electric_right_eye_wrist180_state_20260916_v1
/home/dzq/openpi/.venv/bin/python converter_electric.py --help
```

清单 SHA256：`0145c1b2005d0a7d93f02e465dba74d156fc9c01474d55ae15712f32b362aa16`。

为加快解码，并发由 4 提升至 16；旧的未完成临时输出和日志保留在 `conversion.launch.json` 所记录的位置，原始数据未修改。旧配置备份为 `backups/config.before_electric.py`。

2026-09-16 按用户确认，仅修正相机映射为 **cam1→左腕、cam0→右腕**。`camera_mapping_fix_validation.json` 逐字节比较了修正前后全部 115 条、24,182 帧的 state、action、padding mask、选帧索引和对齐时间差，均完全一致。错误映射的未完成输出保存在 `.task_electric_lerobot_10hz_h50_right_eye_wrist180_v1.abandoned-wrong-camera-map-k5tf4xre`，旧代码与日志保存在 `backups/before_camera_mapping_fix`；没有删除原始数据或 checkpoint。修正预览为 `camera_mapping_preview/corrected_three_cameras.png`。

## 推理必须匹配预处理

配套入口为本目录 `serve_wrist180.py`。`--input-orientation raw` 对左右腕原图各旋转一次；若调用端已旋转或回放此新数据集，使用 `--input-orientation rotated180`，不能转两次。

输入使用 `images.cam_high`、`images.cam_left_wrist`、`images.cam_right_wrist`、正确语义的 `state` 和本任务 `prompt`。如果推理仍使用这批采集设备的 ID，**cam1 填入 cam_left_wrist，cam0 填入 cam_right_wrist**；若调用端已经按物理左右手命名，不可再次交换。state/action 保持原有左右手顺序。其中 **cam_high 必须已经是 E6 右眼裁剪并 letterbox 后的画面**，此入口不将完整双目图自动裁成右眼。

训练完成后可使用对应已写完的 checkpoint 启动服务，例如：

```bash
cd /home/dzq/openpi
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/dzq/openpi/src \
  /home/dzq/openpi/.venv/bin/python \
  /home/dzq/data_deal/training_tools/task_electric_right_eye_wrist180_state_20260916_v1/serve_wrist180.py \
  --checkpoint /mnt/data/dzq/openpi/checkpoints/pi05_umi_task_electric_10hz_h50_wrist180_state_20k_b64_w32_8gpu_v1/task_electric_right_eye_wrist180_state_10hz_h50_20k_b64_w32_8gpu_20260916_v1/20000 \
  --input-orientation raw --host 127.0.0.1 --port 8000
```

本次没有启动推理服务，没有执行机器人动作。
