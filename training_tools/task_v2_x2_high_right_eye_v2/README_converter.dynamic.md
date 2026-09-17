# 通用 UMI 文件夹转 LeRobot

`umi_folder_to_lerobot.py` 适用于与 `task_v1` / `task_v1_new` 相同采集布局的数据根目录。默认自动扫描根目录下的所有 episode 文件夹，不需要 `--expected-episodes`；也可以用 `--episode-list` 传入经过筛选的 episode basename 清单。

## E6 完整右眼（2026-09-09 修复）

先用 `ffprobe` 读取每条 `camera/e6_rgb.h265` 的实际解码宽高，再按左右并排布局裁完整右半幅。不再用 session 元信息或固定 `1600` 作为裁剪位置。

- 实际 3200×1200：`crop=1600:1200:1600:0`。
- 实际 3840×1200：`crop=1920:1200:1920:0`。
- 默认 `letterbox` 保持完整右眼与原始比例；3840 宽的双目裁右眼后为1920×1200，输出640×480时上下各补40像素，不拉伸、不混入左眼。
- 探测失败、非法尺寸或 YUV420 不兼容的奇数裁剪会失败，绝不退回猜测宽度。
- 每条 `head_video_preprocessing` 记录真实/声明尺寸、差异告警及裁剪坐标；原始 metadata 不会被修改。

## 版本与数据规则

`umi_folder_to_lerobot.py` 当前保留完整时间线，对齐误差是审计信号，不是自动剔除条件。新增 `umi_folder_to_lerobot_strict.py` 使用相同的动态右眼修复，但保留原严格对齐规则；严格版本要求一个源 episode 恰好有一段可输出的连续有效数据，否则拒绝、不拆分。使用哪个版本，应在预检和转换命令中保持一致。

- 只扫描 `--source` 的直接子目录，不递归搜索。
- `--episode-list` 必须是 UTF-8、每行一个 episode basename；空行、重复、格式错误或不存在的 ID 都会使预检失败。脚本只把每一行当作数据，不执行其内容。
- 使用清单时，清单 SHA-256、选中/未选中的 ID 都写入转换合同，原清单复制为 `meta/episode_selection.txt`，发布前会再次核对。
- 每个源文件夹必须恰好生成一个 LeRobot episode，输出保留原文件夹 ID。
- 严格版本遇到两段或更多可输出的连续数据时整体预检失败；完整时间线版本不会自动排除这些问题，训练前必须审查追踪失效和对齐标志。
- 任何源 episode 失败时，`convert` 会在视频编码和创建目标前整体终止，不会静默跳过。
- action 尾部使用 `action_is_pad` 逐位 mask，图像和观测不会因 action horizon 不足而被删除或填充。
- `--fps` 必须能整除源 E6 帧率。例如 60 Hz 源数据可设为 10/15/20/30/60 Hz。

## 先做只读预检

```bash
cd /home/dzq/data_deal

/home/dzq/openpi/.venv/bin/python \
  conversion_tools/umi_folder_to_lerobot.py inspect \
  --source /mnt/data/dzq/umi/data/task_v1_new \
  --target /mnt/data/dzq/umi/datasets/task_v1_new_lerobot_30hz_h90 \
  --repo-id dzq/task_v1_new_lerobot_30hz_h90 \
  --task 'Put the object into the box.' \
  --fps 30 \
  --action-horizon 90 \
  --max-alignment-ms 100 \
  --max-hand-age-ms 100 \
  --hand-alignment nearest \
  --compact
```

`inspect` 只读取和校验，不创建 `--target`。只有当输出的 `status` 为 `ok`、`failure_count` 为 0 时才应执行转换。

若只转换筛选后的 episode，在 `inspect` 和 `convert` 两条命令中同时加入：

```bash
--episode-list /absolute/path/pass_episodes.txt
```

## 执行转换

```bash
/home/dzq/openpi/.venv/bin/python \
  conversion_tools/umi_folder_to_lerobot.py convert \
  --source /mnt/data/dzq/umi/data/task_v1_new \
  --target /mnt/data/dzq/umi/datasets/task_v1_new_lerobot_30hz_h90 \
  --repo-id dzq/task_v1_new_lerobot_30hz_h90 \
  --task 'Put the object into the box.' \
  --fps 30 \
  --action-horizon 90 \
  --max-alignment-ms 100 \
  --max-hand-age-ms 100 \
  --hand-alignment nearest \
  --video-workers 4 \
  --confirm CREATE_LEROBOT_DATASET
```

要改为 10 Hz / H50，同时修改路径、repo ID，并设置：

```bash
--fps 10 --action-horizon 50
```

频率或 action horizon 改变后，训练侧的 dataset config、model action horizon 和 norm stats 必须使用同一组参数重新生成，不能沿用旧数据集的统计量。
