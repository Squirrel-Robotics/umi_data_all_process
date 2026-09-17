# 腕部 180°：有 / 无 state，各四卡同时训练

> 启动核验（2026-09-11 10:29）：两版均已正常通过 step 50。无 state PID 4013191，GPU 0–3；有 state PID 4013192，GPU 4–7。此处是核验快照，后续进度以训练日志为准。

用户已确认每版 batch 32。两版使用相同 85 条完整数据、9995 帧，左右腕各旋转 180°、不交换；头部右眼及 state/action 不变。

| 版本 | 物理 GPU | state 输入 | batch | worker |
|---|---|---|---|---|
| no_state | 0,1,2,3 | `discrete_state_input=False` | 32 | 32 |
| state | 4,5,6,7 | `discrete_state_input=True` | 32 | 32 |

共同设置：Pi0.5，10 Hz / H50，FSDP 4，10k 步，每 1k 保存，seed 42；从相同 pi05_base 基础权重新训，不续训旧 checkpoint；XLA 预分配 90%。LR、优化器、数据增强、归一化和 action padding 策略一致。Prompt：`Put the two objects into the box.`

数据：`/mnt/data/dzq/umi_v2/datasets/task_v2_x2_high_lerobot_10hz_h50_right_eye_v2_wrist180`

训练模型与数据管线只切换 state 条件输入。接口两版都保留 state 字段；有 state 版使用实际归一化本体状态生成离散 token，无 state 版不使用这些数值进行条件输入。

## 启动和日志

```bash
bash /home/dzq/data_deal/training_tools/task_v2_x2_high_wrist180_state_ablation_4gpu_v1/start_training.sh
```

已启动后不要重复运行。启动器要求八卡空闲、拒绝同名实验及 checkpoint 覆盖，先校验后启动两个独立进程。不会自动重启失败训练或杀其他进程。

```bash
# 无 state
tail -f /home/dzq/openpi/logs/task_v2_x2_high_wrist180_no_state_10hz_h50_10k_b32_w32_4gpu_20260911_v1.log
# 有 state
tail -f /home/dzq/openpi/logs/task_v2_x2_high_wrist180_state_10hz_h50_10k_b32_w32_4gpu_20260911_v1.log
```

配置名：

```text
pi05_umi_task_v2_x2_high_10hz_h50_wrist180_no_state_10k_b32_w32_4gpu_v1
pi05_umi_task_v2_x2_high_10hz_h50_wrist180_state_10k_b32_w32_4gpu_v1
```

Checkpoint 位于 `/mnt/data/dzq/openpi/checkpoints/<配置名>/<对应实验名>/<步数>/`。完整路径和 PID 见两版日志旁的 `.launch.json`；双进程启动状态见本目录 `pair.status.json`。

## 推理方向必须一致

训练数据已经旋转，训练时不会再转。使用本目录 `serve_wrist180.py`，明确指定 `--variant no_state` 或 `--variant state`，以及对应版本已写完的 checkpoint。脚本会拒绝版本不匹配。

```bash
cd /home/dzq/openpi
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/dzq/openpi/src \
  /home/dzq/openpi/.venv/bin/python \
  /home/dzq/data_deal/training_tools/task_v2_x2_high_wrist180_state_ablation_4gpu_v1/serve_wrist180.py \
  --variant no_state \
  --checkpoint /mnt/data/dzq/openpi/checkpoints/pi05_umi_task_v2_x2_high_10hz_h50_wrist180_no_state_10k_b32_w32_4gpu_v1/task_v2_x2_high_wrist180_no_state_10hz_h50_10k_b32_w32_4gpu_20260911_v1/10000 \
  --input-orientation raw --host 127.0.0.1 --port 8000
```

输入原始腕图使用 `raw`（入口各转一次 180°）；若调用端已旋转或回放新数据集，使用 `rotated180`（入口不再转）。不能自动判断任意图像的方向。输入保持 CX002 格式：`images.cam_high`、`images.cam_left_wrist`、`images.cam_right_wrist`、`state`、`prompt`。有 state 版必须提供正确的本体状态。

本次没有启动推理服务，也没有执行机器人动作。提升与否需要通过相同场景下的推理/实机成功率对照验证。

## 校验与旧任务

- 复用已完成的 85 条 / 170 个腕图逐帧 YUV、时间戳及统计全量审计；先重算全部已记录文件 hash，文件变化即拒绝复用。
- 对两版分别执行真实 CPU dataloader，检查 H50 mask、三路图像，以及改变 state 时 token 是否按预期开启/关闭变化。
- `preflight.report.json` 保存校验结果和文件指纹；`backups/config.before_dual4gpu.py` 保存新增两个配置前的原配置。
- 原八卡训练 PID 4006142 已因本次切换请求终止，当时仍处于初始化；旧日志、数据、配置和 checkpoint 目录均保留。
