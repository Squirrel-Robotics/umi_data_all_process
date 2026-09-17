# task_v2_x2_high 完整右眼修复版

## 修复范围

85 条原始 E6 视频实际为 3840×1200 双目，但 session 元数据写成 3200×1200。旧版固定裁剪 `1600:1200:1600:0` 混入左眼并截断右眼。本版探测真实视频尺寸，取完整右半：`1920:1200:1920:0`。

保留完整右眼后等比例缩放至 640×400，再上下各补 40 像素到 640×480；不拉伸、不额外裁任务 ROI。原有 Pi0.5 训练增强保持不变：头图 95% 随机裁剪、±5° 随机旋转、颜色增强。

仅重新编码 85 条头图视频并更新对应图像统计、manifest 和校验值。Parquet 中的 state/action、动作 padding、时间和对齐索引，以及两路腕视频与旧版逐文件字节一致。85 条清单及排除的 10 条不变，不拆 episode、不删除原始数据。

## 数据与训练

- 数据：`/mnt/data/dzq/umi_v2/datasets/task_v2_x2_high_lerobot_10hz_h50_right_eye_v2`
- 任务：`Put the two objects into the box.`
- 10 Hz、H50、85 episodes、9,995 帧、三路相机、本体 state 输入（`discrete_state_input=True`）。
- 8 卡，batch 64，workers 32，10,000 步，每 1,000 步保存并保留 checkpoint。
- 从 `/mnt/data/checkpoints/pi05_base/params` 重新训练，**不续训错误裁剪版的模型**；旧数据和旧 checkpoint 均保留。
- 学习率、优化器、FSDP 与之前相同，XLA 显存预分配比例 0.90。

## 运行和状态

本次 `run_pipeline.py` 已完成视频编码和校验，但 NFS 不支持其 `renameat2(RENAME_NOREPLACE)` 发布操作，因此原流程的 `pipeline_status.json` 保留为失败审计。使用 `publish_repaired_dataset.py` 重新核验完整暂存数据后，以独占空目录预留和原子重命名发布；不重新编码、不覆盖已有数据。

发布后由 `continue_repaired_training.py` 计算独立归一化统计，再调用 `start_training.sh` 实际加载验证、等待 8 张卡全部空闲并启动训练。当前流程看 `continuation_status.json`，训练排队看 `.queue.json`。任一步失败都会停止，不会自动重试；不要重复启动入口。

```bash
cd /home/dzq/data_deal/training_tools/task_v2_x2_high_right_eye_v2
cat continuation_status.json
tail -n 20 continuation.log
tail -n 20 normalization.log
```

训练日志：

```text
/home/dzq/openpi/logs/task_v2_x2_high_right_eye_10hz_h50_state_10k_b64_w32_8gpu_20260909_v2.log
```

同前缀 `.queue.json` 表示启动/排队状态，`.smoke.log` 为验证结果，`.launch.json` 为精确进程启动收据；训练进度以 `.log` 为准。

Checkpoint：

```text
/mnt/data/dzq/openpi/checkpoints/pi05_umi_task_v2_x2_high_10hz_h50_right_eye_state_10k_b64_w32_8gpu_v2/task_v2_x2_high_right_eye_10hz_h50_state_10k_b64_w32_8gpu_20260909_v2
```

`meta/head_video_repair.json` 保存实际输入尺寸、裁剪坐标、选帧索引、原始码流哈希、原/新视频哈希和保留文件证明；`meta/head_video_publication.json` 单独记录 NFS 发布恢复的校验证据，不修改原报告。训练前还独立核验全部 85 条实际输出首帧。
