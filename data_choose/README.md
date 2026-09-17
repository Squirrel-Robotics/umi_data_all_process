# UMI 数据人工筛选网站

这个网站用于逐条检查 UMI episode 的三路相机画面，并记录“保留 / 跳过 / 隔离删除”等人工审核结果。默认数据根目录是：

```text
/mnt/data/dzq/umi/data/task_v1_new
```

## 三路相机映射

| 网页位置 | 数据文件 | 说明 |
| --- | --- | --- |
| 头部相机 | `camera/e6_rgb.h265` | E6 头部双目 RGB 裸 HEVC；网页预览使用右眼画面 |
| 左腕相机 | `extensions/customer_camera/cam0/media.mjpeg` | `cam0 -> left_wrist_rgb` |
| 右腕相机 | `extensions/customer_camera/cam1/media.mjpeg` | `cam1 -> right_wrist_rgb` |

网站只审核完整 episode。一个目录至少需要同时具备：

```text
.done
camera/hand_pose.csv
camera/e6_rgb.h265
extensions/customer_camera/cam0/media.mjpeg
extensions/customer_camera/cam1/media.mjpeg
```

`collector_run_*`、尚未完成的采集目录、缺失任一路视频或缺失 `hand_pose.csv` 的目录不会进入审核列表。不要在采集程序仍在写入某条数据时手工移动它。

## 启动

在数据服务器上运行：

```bash
cd /home/dzq/data_deal/data_choose
./run.sh
```

服务保持在前台运行，按 `Ctrl+C` 停止。默认只监听服务器本机：

```text
127.0.0.1:8091
```

可以在 `run.sh` 后追加参数覆盖默认值。例如改用 8092 端口：

```bash
./run.sh --port 8092
```

启动后，服务会在后台提前为所有完整 episode 生成三路浏览器预览缓存。已经有效的缓存会直接跳过，默认使用 1 个后台任务；总共 2 个 FFmpeg 槽中会为网页实时请求预留 1 个，避免首次预热期间打开未缓存 episode 时长时间排队。网页顶部会显示缓存数量、百分比和当前处理对象。首次预热期间仍可审核，但尚未缓存到的 episode 第一次打开时可能需要等待；进度显示“全部已缓存”后，切换 episode 会直接读取 MP4，不再现场调用 FFmpeg 解码。

完整预热结束后，服务默认每 60 秒重新扫描一次数据目录，因此采集程序后续写完的新 episode 也会自动进入缓存。可以按机器负载调整并发数（范围 1–4）：

```bash
./run.sh --precache-workers 1
```

如果确实需要加快预热，可同时提高总槽数和后台任务数，例如 `--ffmpeg-concurrency 3 --precache-workers 2`；服务仍会为交互请求保留一个转码槽。不要只提高后台任务数而不提高总槽数。

服务会拒绝 `后台任务数 >= FFmpeg 总槽数` 的参数组合，也会用 `runtime/server.lock` 阻止同一网站被重复启动。预热只会在端口成功绑定后开始。

如需关闭启动预热，或保留启动预热但关闭后续自动扫描：

```bash
./run.sh --no-precache
./run.sh --precache-rescan-seconds 0
```

网页中的“开始缓存 / 重试缓存”按钮也可以手工启动一次完整扫描。后台缓存失败不会停止网站，页面会显示失败对象和原因；修复源文件后点击“重试缓存”即可。

默认 token 文件为：

```text
/home/dzq/data_deal/data_choose/runtime/review_token
```

首次启动会生成 token，并在终端打印带 token 的完整访问 URL。之后请始终以终端实际输出的 URL 为准。

## 从自己的电脑访问

网页不会直接暴露到局域网，需要在自己的电脑上建立 SSH 端口转发：

```bash
ssh -N -L 8091:127.0.0.1:8091 server-dzq
```

如果本机没有 `server-dzq` SSH 别名，可以替换为实际登录信息：

```bash
ssh -N -L 8091:127.0.0.1:8091 dzq@服务器地址
```

保持这个终端窗口运行，然后在本机浏览器中打开服务器启动时打印的完整 URL，形式类似：

```text
http://127.0.0.1:8091/?token=<终端打印的token>
```

不要把 token、完整访问 URL 或 `review_token` 文件发给其他人。

## 审核操作与快捷键

三路画面使用同一条时间轴。点击输入框或其他可编辑控件时，快捷键会暂时避让，避免误操作。

| 快捷键 | 操作 |
| --- | --- |
| `空格` | 播放 / 暂停 |
| `J` / `L` | 后退 / 前进 1 秒 |
| `,` / `.` | 后退 / 前进一帧 |
| `←` / `→` | 上一条 / 下一条 episode |
| `1` / `2` / `3` | 聚焦头部 / 左腕 / 右腕画面 |
| `K` | 标记为保留 |
| `S` | 标记为跳过，稍后继续判断 |
| `D` | 打开“隔离删除”二次确认，不会立即永久删除 |

执行隔离前必须在二次确认界面再次确认。回收站视图提供“恢复”按钮，可把误隔离的数据放回原数据根目录。

## 缓存、审核状态和回收站

网站运行产生的文件均与原始媒体分开：

| 内容 | 位置 |
| --- | --- |
| 浏览器播放所需的转码缓存 | `/home/dzq/data_deal/data_choose/runtime/cache/` |
| SQLite 审核状态 | `/home/dzq/data_deal/data_choose/runtime/review.sqlite3` |
| 访问 token | `/home/dzq/data_deal/data_choose/runtime/review_token` |
| 被隔离的 episode | `/mnt/data/dzq/umi/data/task_v1_new/.data_choose_trash/` |
| 隔离/恢复审计日志 | `/mnt/data/dzq/umi/data/task_v1_new/.data_choose_trash/review_actions.jsonl` |

转码缓存不属于原始数据。停止网站后可以清理 `runtime/cache/`，但下次启动会重新执行全量预热。不要随意删除 SQLite 状态或审计文件，否则已完成的人工审核记录可能丢失。

网页上的“删除”实际是把整个 episode 原子移动到同一数据根目录下的 `.data_choose_trash/`，不是执行不可恢复的 `rm`。隔离后的目录名会追加时间戳和随机后缀以避免重名；原 episode 名仍保存在审核记录中。因此隔离操作速度快，也可以恢复。

## 恢复被隔离的数据

进入网页回收站，找到对应 episode 后点击“恢复”。网站会同时完成目录原子移动、SQLite 状态更新和审计记录；原位置已存在同名目录时会拒绝覆盖。

不要用手工 `mv` 恢复，因为那样不会同步 SQLite 审核状态。如果网页无法启动，请先备份 `runtime/review.sqlite3` 和 `.data_choose_trash/review_actions.jsonl`，再修复网站后从回收站执行恢复。

## 安全限制

- 服务默认仅绑定 `127.0.0.1`，建议始终通过 SSH 隧道访问；不要为了方便改成公网可访问的 `0.0.0.0`。
- 只有通过 token 校验的请求才能查看媒体或执行审核操作；token 应当按密码管理。
- 网站只允许操作配置的数据根目录内、通过完整性检查的 episode；不会跟随任意外部路径或把 `collector_run_*` 当作可删除数据。
- 隔离、恢复和审核状态变化都会写入审计记录。多人同时审核同一数据集时应先协调，避免相互覆盖判断。
- 隔离不是永久清理。确认回收站内容确实不再需要之前，不要手工执行 `rm`。
- 运行网站的账号需要读取三路媒体、写入 `runtime/`，以及在数据根目录中移动 episode 的权限；推荐使用 `dzq` 账号。
