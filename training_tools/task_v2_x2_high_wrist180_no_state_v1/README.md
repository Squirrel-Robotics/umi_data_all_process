# 双腕 180° / 无 state 输入对照训练

> 2026-09-11 更新：本八卡任务已按用户要求停止（PID 4006142），改为有 / 无 state 两版各四卡、每版 batch 32 同时训练。请使用 `/home/dzq/data_deal/training_tools/task_v2_x2_high_wrist180_state_ablation_4gpu_v1/README.md`。本目录保留数据转换工具和原八卡配置记录。

85 个完整 episode，9995 帧。左右腕**各自旋转 180°，不交换相机**；头部右眼画面、state/action、帧顺序和时间戳均不变。源数据、旧训练、checkpoint 保留。

数据集：`/mnt/data/dzq/umi_v2/datasets/task_v2_x2_high_lerobot_10hz_h50_right_eye_v2_wrist180`

## 训练

参数与之前无 state 版本一致：Pi0.5，10 Hz / H50，GPU 0–7，global batch 64，num_workers 32，10k 步，每 1k 保存；从 pi05_base 新训练，XLA 显存预分配 90%。Prompt：`Put the two objects into the box.`

`discrete_state_input=False`，已检查本配置的 Pi0.5 路径不使用本体 state 数值进行条件输入；接口仍保留 state 字段。未修改现有数据增强。数据已旋转，训练管线不再旋转。

```bash
bash /home/dzq/data_deal/training_tools/task_v2_x2_high_wrist180_no_state_v1/start_training.sh
```

启动器先全量 CPU 校验，八卡空闲后启动；拒绝重复实验名、覆盖 checkpoint 和自动重试，不杀其他进程。当前实验已启动时不要重复执行。

```bash
tail -f /home/dzq/openpi/logs/task_v2_x2_high_wrist180_no_state_10hz_h50_10k_b64_w32_8gpu_20260911_v1.log
```

Checkpoint 根目录：`/mnt/data/dzq/openpi/checkpoints/pi05_umi_task_v2_x2_high_10hz_h50_wrist180_no_state_10k_b64_w32_8gpu_v1/task_v2_x2_high_wrist180_no_state_10hz_h50_10k_b64_w32_8gpu_20260911_v1/`

## 推理时必须匹配方向

**不要直接用普通 serving 入口接收未旋转的腕部原图。** 本目录提供推理专用入口，对原始左右腕图各转一次 180°，不改 head/state/action；训练不使用此变换。输入结构沿用 CX002：`images.cam_high`、`images.cam_left_wrist`、`images.cam_right_wrist`、`state`、`prompt`。

训练完成后，选择已写完的 checkpoint（下面以 10000 为例）：

```bash
cd /home/dzq/openpi
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/dzq/openpi/src \
  /home/dzq/openpi/.venv/bin/python \
  /home/dzq/data_deal/training_tools/task_v2_x2_high_wrist180_no_state_v1/serve_wrist180.py \
  --checkpoint /mnt/data/dzq/openpi/checkpoints/pi05_umi_task_v2_x2_high_10hz_h50_wrist180_no_state_10k_b64_w32_8gpu_v1/task_v2_x2_high_wrist180_no_state_10hz_h50_10k_b64_w32_8gpu_20260911_v1/10000 \
  --input-orientation raw --host 127.0.0.1 --port 8000
```

如果调用端已经旋转，或回放新数据集，必须改为 `--input-orientation rotated180`，避免转两次。无法从任意画面自动判断方向；调用端需明确选择。默认只监听本机，跨机部署需自行明确监听地址和访问限制。此脚本仅提供模型服务，不执行机器人动作；本次没有启动推理服务。

## 校验与来源

- `rotate_wrist_dataset.py`：独立新数据集，libx264 CRF 0 无损编码；逐帧 YUV hash、时间戳、原文件 hash 校验；同步更新腕图统计。
- `meta/wrist_rotation.json`：完整旋转审计；新数据契约 SHA256：`9fdc693f5141fb124e2be7e3c9f4f226a015570cc2dcd6b5cf03f7c2c3b4b768`。
- 新 norm asset：`/mnt/data/dzq/openpi/data/assets/umi_task_v2_x2_high_hand_pose_10hz_h50_right_eye_wrist180_masked_v1`；统计值与旋转前逐字节一致，审计绑定新数据集。
- `smoke_training.py`：85 条选择、对齐、头部右眼、170 个腕图、归一化、实际 dataloader、无 state token、H50 padding mask 检查。
- `test_training_inference_orientation.py`：真实数据首/中/末帧的训练与在线输入方向对照。
- 2026-09-11 实测第 0、4997、9994 帧：通过配套入口后的三路图像与训练模型输入逐像素一致，RGB 最大误差为 0，prompt token 一致。
- `backups/config.before_wrist180.py`：新增配置前的原配置备份。

能否提升效果需要与上一版在相同任务条件下比较实机成功率，训练 loss 本身不能证明提升。
