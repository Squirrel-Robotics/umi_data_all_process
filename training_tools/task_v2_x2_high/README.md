# task_v2_x2_high：85 条高质量数据训练

原始目录：`/mnt/data/dzq/umi_v2/data/task_v2_x2`。

使用原始 `track_pass_episodes.txt` 的 95 条，按用户确认整条排除有中间断点的 10 条，仅将 `selected_85_episodes.txt` 中的 85 条用于转换与训练。`excluded_10_episodes.txt` 和 `selection_audit.json` 记录排除清单与哈希。**不拆分 episode，不删除或修改原始数据。**

## 数据与训练参数

- 指令：`Put the two objects into the box.`
- LeRobot：`/mnt/data/dzq/umi_v2/datasets/task_v2_x2_high_lerobot_10hz_h50`
- 10 Hz、H50（5 秒），三路 RGB，图像不加 ROI。
- 严格相机/手部状态时间对齐 ≤100 ms，每个源 episode 对应一个输出 episode。开头无法对齐的采样和建立相对 state 所需的基准采样不输出；不拼接中间片段。
- 使用保存的 `camera/hand_pose.csv` 重建 EEF 轨迹，不重新计算控制器外参。state 包含相邻采样 EEF 增量与绝对手指状态；H50 action 使用同一个当前 EEF 锚点，尾部补齐由 `action_is_pad` 屏蔽。
- Pi0.5，`discrete_state_input=True`：归一化后的本体 state 编码进模型输入 token。
- GPU 0–7，FSDP 8，全局 batch 64，workers 32，10,000 步，每 1,000 步保存并保留 checkpoint。
- 复用 task_v3_high 的学习率与基础权重；XLA 预分配比例 0.90。归一化统计独立计算，排除补齐 action。

## 入口

在当前目录执行；不要重复启动已在运行的转换或排队进程：

```bash
cd /home/dzq/data_deal/training_tools/task_v2_x2_high
bash convert_dataset.sh
bash normalize_dataset.sh
bash start_training.sh --check-only
bash start_training.sh
```

三个阶段按上述顺序运行。转换和归一化拒绝覆盖已有输出；启动器先执行 CPU 实际数据加载验证，再等待全部 8 卡空闲。它不会停止其他进程，不会覆盖/续训旧实验，也不会自动重启失败训练。排队上限为 72 小时；超时或数据/代码变化会明确报错，需要重新检查后再启动。

`converter_strict.py` 是上次严格转换器的独立归档，本次没有改动 `conversion_tools/umi_folder_to_lerobot.py` 的默认行为。`openpi_config.prepared.py` 是新增本次配置后的完整快照，安装前的原配置保存在 `backups/openpi_config.before_x2.py`，不要用快照覆盖后续用户修改。

转换的视频并行阶段限制在 24 个 CPU、nice 10；本次视频全部编码完后，为避开繁忙核心，解码统计阶段放开到主机所有可用 CPU 调度，仍保留 nice 10。没有停止其他工作负载。

## 状态与输出

当前后台流程状态：本目录 `pipeline_status.json`，总日志：`pipeline.log`。`finish_and_queue.py` 只等待已启动的指定转换进程，通过发布校验后计算归一化统计，再调用训练启动器；任一步失败会停止，不会越过失败继续训练。

数据转换日志：本目录 `conversion.log`。归一化日志：本目录 `normalization.log`。

队列、CPU 验证、训练日志目录：`/home/dzq/openpi/logs/`，共同文件名前缀：

```text
task_v2_x2_high_10hz_h50_full_head_state_10k_b64_w32_8gpu_20260909
```

后缀 `.queue.json` 为队列状态、`.smoke.log` 为加载验证结果、`.log` 为训练日志、`.launch.json` 为实际训练启动凭据。只有 `waiting_gpu` 代表等待，并不代表已经开始训练；实际训练进度以 `.log` 为准。

Checkpoint 根目录：

```text
/mnt/data/dzq/openpi/checkpoints/pi05_umi_task_v2_x2_high_10hz_h50_masked_with_full_head_state_10k_b64_w32_8gpu_v1/task_v2_x2_high_10hz_h50_full_head_state_10k_b64_w32_8gpu_20260909
```

`README.preflight.md` 和 `data_preflight.json` 保留最初 95 条的历史只读检查；当前训练选择以本文件及 `selection_audit.json` 为准。
