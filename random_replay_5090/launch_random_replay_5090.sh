#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CSV_PATH=""
ASSETS_ROOT="/home/dzq/umi_x2_mujoco/assets/generated_revo2"
SIM_PYTHON="/mnt/mujoco/.venv/bin/python"
QUANTA_SRC="/mnt/mujoco/quanta-x2-mujoco/src"
PLAYBACK_RATE="1.0"
TRANSLATION_SCALE="1.0"
ROTATION_SCALE="1.0"
LEFT_FRAME_X_DEG="0.0"
RIGHT_FRAME_X_DEG="0.0"
CAMERA_VIEW="operator"
MAX_FRAMES=""
REPLACE=0
STOP=0

usage() {
  echo "Usage: $0 --csv PATH [--playback-rate N] [--camera-view operator|front] [--replace]"
  echo "       $0 --stop"
}

while (($#)); do
  case "$1" in
    --csv) CSV_PATH="$2"; shift 2 ;;
    --assets-root) ASSETS_ROOT="$2"; shift 2 ;;
    --python) SIM_PYTHON="$2"; shift 2 ;;
    --quanta-src) QUANTA_SRC="$2"; shift 2 ;;
    --playback-rate) PLAYBACK_RATE="$2"; shift 2 ;;
    --translation-scale) TRANSLATION_SCALE="$2"; shift 2 ;;
    --rotation-scale) ROTATION_SCALE="$2"; shift 2 ;;
    --left-frame-x-deg) LEFT_FRAME_X_DEG="$2"; shift 2 ;;
    --right-frame-x-deg) RIGHT_FRAME_X_DEG="$2"; shift 2 ;;
    --camera-view) CAMERA_VIEW="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --replace) REPLACE=1; shift ;;
    --stop) STOP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PID_FILE="$SCRIPT_DIR/viewer.pid"
LOG_FILE="$SCRIPT_DIR/viewer.log"
SUMMARY_FILE="$SCRIPT_DIR/viewer_summary.json"
TRAJECTORY_FILE="$SCRIPT_DIR/viewer_trajectory.npz"
RUNNER="$SCRIPT_DIR/replay_x2_revo2_relative_eef.py"

running_pid=""
if [[ -f "$PID_FILE" ]]; then
  candidate="$(<"$PID_FILE")"
  if [[ "$candidate" =~ ^[0-9]+$ ]] && kill -0 "$candidate" 2>/dev/null; then
    running_pid="$candidate"
  else
    rm -f "$PID_FILE"
  fi
fi

stop_running() {
  local pid="$1"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in {1..50}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      return 0
    fi
    sleep 0.2
  done
  echo "Replay process $pid did not stop after SIGTERM." >&2
  return 1
}

if ((STOP)); then
  if [[ -n "$running_pid" ]]; then
    stop_running "$running_pid"
    echo "Stopped random MuJoCo replay PID $running_pid."
  else
    echo "No random MuJoCo replay is running."
  fi
  exit 0
fi

if [[ -n "$running_pid" ]]; then
  if ((REPLACE)); then
    stop_running "$running_pid"
  else
    echo "Replay is already running with PID $running_pid; pass --replace to replace it." >&2
    exit 1
  fi
fi

if [[ -z "$CSV_PATH" ]]; then
  echo "--csv is required." >&2
  usage >&2
  exit 2
fi
for required in "$CSV_PATH" "$ASSETS_ROOT" "$SIM_PYTHON" "$RUNNER"; do
  if [[ ! -e "$required" ]]; then
    echo "Required path does not exist: $required" >&2
    exit 2
  fi
done
if [[ "$CAMERA_VIEW" != "operator" && "$CAMERA_VIEW" != "front" ]]; then
  echo "--camera-view must be operator or front." >&2
  exit 2
fi
if [[ ! -r /run/user/1002/gdm/Xauthority ]]; then
  echo "5090 desktop Xauthority is not readable." >&2
  exit 2
fi

runner_args=(
  "$SIM_PYTHON" "$RUNNER"
  --csv "$CSV_PATH"
  --assets-root "$ASSETS_ROOT"
  --translation-scale "$TRANSLATION_SCALE"
  --rotation-scale "$ROTATION_SCALE"
  --left-frame-x-deg "$LEFT_FRAME_X_DEG"
  --right-frame-x-deg "$RIGHT_FRAME_X_DEG"
  --playback-rate "$PLAYBACK_RATE"
  --camera-view "$CAMERA_VIEW"
  --summary "$SUMMARY_FILE"
  --trajectory-npz "$TRAJECTORY_FILE"
  --view
  --loop
)
if [[ -n "$MAX_FRAMES" ]]; then
  runner_args+=(--max-frames "$MAX_FRAMES")
fi

nohup setsid env \
  DISPLAY="${REPLAY_DISPLAY:-:4}" \
  XAUTHORITY="${REPLAY_XAUTHORITY:-/run/user/1002/gdm/Xauthority}" \
  XDG_RUNTIME_DIR="${REPLAY_XDG_RUNTIME_DIR:-/run/user/1002}" \
  DBUS_SESSION_BUS_ADDRESS="${REPLAY_DBUS_ADDRESS:-unix:path=/run/user/1002/bus}" \
  MUJOCO_GL=glfw \
  PYTHONPATH="$QUANTA_SRC" \
  "${runner_args[@]}" \
  >"$LOG_FILE" 2>&1 </dev/null &
pid=$!
echo "$pid" >"$PID_FILE"

sleep 5
if ! kill -0 "$pid" 2>/dev/null; then
  echo "MuJoCo replay failed to start. Last log lines:" >&2
  tail -n 100 "$LOG_FILE" >&2
  rm -f "$PID_FILE"
  exit 1
fi

echo "Random MuJoCo replay started on DISPLAY=${REPLAY_DISPLAY:-:4}."
echo "PID: $pid"
echo "CSV: $CSV_PATH"
echo "Log: $LOG_FILE"
echo "Summary: $SUMMARY_FILE"
