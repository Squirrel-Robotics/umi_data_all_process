"use strict";

const ui = {
  appView: document.querySelector("#appView"),
  connectionBadge: document.querySelector("#connectionBadge"),
  serverClock: document.querySelector("#serverClock"),
  episodeId: document.querySelector("#episodeId"),
  stateBadge: document.querySelector("#stateBadge"),
  flowTitle: document.querySelector("#flowTitle"),
  flowSteps: [...document.querySelectorAll("[data-flow-stage]")],
  taskName: document.querySelector("#taskName"),
  duration: document.querySelector("#duration"),
  frames10Hz: document.querySelector("#frames10Hz"),
  horizonWindows: document.querySelector("#horizonWindows"),
  runMessage: document.querySelector("#runMessage"),
  alignmentCard: document.querySelector("#alignmentCard"),
  alignmentTitle: document.querySelector("#alignmentTitle"),
  alignmentSummary: document.querySelector("#alignmentSummary"),
  commonDuration: document.querySelector("#commonDuration"),
  uncoveredPrefix: document.querySelector("#uncoveredPrefix"),
  topicTable: document.querySelector("#topicTable"),
  recentEpisodes: document.querySelector("#recentEpisodes"),
  diskFree: document.querySelector("#diskFree"),
  memoryFree: document.querySelector("#memoryFree"),
  swapUsed: document.querySelector("#swapUsed"),
  cameraPreviewInfo: document.querySelector("#cameraPreviewInfo"),
  e6RecoverButton: document.querySelector("#e6RecoverButton"),
  startButton: document.querySelector("#startButton"),
  recordButton: document.querySelector("#recordButton"),
  stopButton: document.querySelector("#stopButton"),
  saveButton: document.querySelector("#saveButton"),
  discardButton: document.querySelector("#discardButton"),
  closeButton: document.querySelector("#closeButton"),
  recordingDialog: document.querySelector("#recordingDialog"),
  reviewDialog: document.querySelector("#reviewDialog"),
  reviewSummary: document.querySelector("#reviewSummary"),
  reviewWarnings: document.querySelector("#reviewWarnings"),
  dialogSave: document.querySelector("#dialogSave"),
  dialogDiscard: document.querySelector("#dialogDiscard"),
  saveSuccessDialog: document.querySelector("#saveSuccessDialog"),
  saveSuccessSummary: document.querySelector("#saveSuccessSummary"),
  saveSuccessState: document.querySelector("#saveSuccessState"),
  saveSuccessNext: document.querySelector("#saveSuccessNext"),
  toast: document.querySelector("#toast"),
};

const DEFAULT_TASK = "Put the object on the box, then take it down.";

const cameras = [
  "head_rgb_stream",
  "left_arm_rgb_stream",
  "right_arm_rgb_stream",
];
const DEFAULT_TOPIC_COUNT = 9;

const app = {
  csrf: null,
  sessionTimer: null,
  status: null,
  statusTimer: null,
  cameraStreamsActive: false,
  cameraReconnectTimers: new Map(),
  cameraMjpegRetryAfter: new Map(),
  toastTimer: null,
  recordingNoticeRunId: null,
  reviewRunId: null,
  lastResultKey: null,
  statusRequestSequence: 0,
  appliedStatusSequence: 0,
  pollGeneration: 0,
  connected: false,
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatDuration(seconds) {
  const total = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(total / 60);
  const remaining = total - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remaining
    .toFixed(1)
    .padStart(4, "0")}`;
}

function formatNumber(value, digits = 1) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "—";
}

function formatDelta(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  const sign = numeric > 0 ? "+" : "";
  return `${sign}${numeric.toFixed(1)} ms`;
}

function formatAge(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  if (numeric < 1000) return `${Math.round(numeric)} ms`;
  return `${(numeric / 1000).toFixed(1)} s`;
}

function requiredTopicCount(status = app.status) {
  const candidates = [
    status?.start_gate?.required_topic_count,
    status?.alignment?.required_topic_count,
    status?.alignment?.topics?.length,
  ];
  for (const candidate of candidates) {
    const count = Number(candidate);
    if (Number.isInteger(count) && count > 0) return count;
  }
  return DEFAULT_TOPIC_COUNT;
}

function formatCompactNumber(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  if (Number.isInteger(numeric)) return String(numeric);
  return numeric.toFixed(Math.abs(numeric) >= 100 ? 1 : 2);
}

function formatCompactArray(values, limit = 6) {
  if (!Array.isArray(values)) return "";
  const shown = values
    .slice(0, limit)
    .map((value) => formatCompactNumber(value));
  return `${shown.join(", ")}${values.length > limit ? ", …" : ""}`;
}

function collectNumericLeaves(value, results = [], path = [], depth = 0) {
  if (results.length >= 2 || depth > 4 || value == null) return results;
  if (typeof value === "number" && Number.isFinite(value)) {
    results.push({ name: path.slice(-2).join("."), value });
    return results;
  }
  if (Array.isArray(value)) {
    for (let index = 0; index < value.length && results.length < 2; index += 1) {
      collectNumericLeaves(value[index], results, [...path, String(index)], depth + 1);
    }
    return results;
  }
  if (typeof value === "object") {
    for (const [key, item] of Object.entries(value)) {
      if (results.length >= 2) break;
      collectNumericLeaves(item, results, [...path, key], depth + 1);
    }
  }
  return results;
}

function formatLatestValue(rawValue) {
  if (rawValue == null) return "—";
  const value =
    typeof rawValue === "object" && rawValue !== null && "payload" in rawValue
      ? rawValue.payload
      : rawValue;
  if (typeof value === "number") return formatCompactNumber(value);
  if (typeof value === "string" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) return `[${formatCompactArray(value)}]`;
  if (typeof value !== "object" || value === null) return "—";

  const positions =
    value.positions ?? value.joint_positions ?? value.position_values;
  if (Array.isArray(positions)) {
    return `q [${formatCompactArray(positions)}]`;
  }

  const posePosition = value.position;
  if (
    posePosition &&
    typeof posePosition === "object" &&
    [posePosition.x, posePosition.y, posePosition.z].some((item) =>
      Number.isFinite(Number(item)),
    )
  ) {
    return `xyz ${[posePosition.x, posePosition.y, posePosition.z]
      .map((item) => formatCompactNumber(item))
      .join(", ")}`;
  }

  const fingers = value.fingers ?? value.finger_data ?? value.touch;
  if (fingers != null) {
    const fingerCount =
      typeof fingers === "number"
        ? fingers
        : Array.isArray(fingers)
          ? fingers.length
          : typeof fingers === "object"
            ? Object.keys(fingers).length
            : null;
    const sensorType = value.sensor_type ?? value.tactile_type ?? "触觉";
    const samples =
      typeof fingers === "number" ? [] : collectNumericLeaves(fingers);
    const sampleText = samples.length
      ? ` · ${samples
          .map(({ name, value: sample }) =>
            `${name || "值"} ${formatCompactNumber(sample)}`,
          )
          .join(" · ")}`
      : "";
    return `${sensorType}${fingerCount != null ? ` · ${fingerCount} 指` : ""}${sampleText}`;
  }

  try {
    const serialized = JSON.stringify(value);
    return serialized.length > 72 ? `${serialized.slice(0, 69)}…` : serialized;
  } catch {
    return String(value);
  }
}

function latestValueTitle(value) {
  if (value == null) return "尚未收到当前值";
  try {
    const serialized = JSON.stringify(value);
    return serialized.length > 1000
      ? `${serialized.slice(0, 997)}…`
      : serialized;
  } catch {
    return String(value);
  }
}

function showToast(message, isError = false) {
  ui.toast.textContent = message;
  ui.toast.className = `toast visible${isError ? " error" : ""}`;
  clearTimeout(app.toastTimer);
  app.toastTimer = setTimeout(() => {
    ui.toast.className = "toast";
  }, 4200);
}

async function request(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body && typeof options.body !== "string") {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(options.body);
  }
  if (options.method === "POST" && app.csrf) {
    headers["X-CSRF-Token"] = app.csrf;
  }
  const response = await fetch(path, {
    credentials: "same-origin",
    cache: "no-store",
    ...options,
    headers,
  });
  let payload = {};
  try {
    payload = await response.json();
  } catch {
    payload = {};
  }
  if (!response.ok) {
    const error = new Error(payload.error || `请求失败：${response.status}`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

async function establishSession() {
  clearTimeout(app.sessionTimer);
  app.sessionTimer = null;
  try {
    const payload = await request("/api/session");
    app.csrf = payload.csrf;
    startPolling();
  } catch {
    app.csrf = null;
    app.connected = false;
    renderConnection(false);
    app.sessionTimer = setTimeout(establishSession, 1000);
  }
}

function startPolling() {
  clearTimeout(app.statusTimer);
  app.statusTimer = null;
  const generation = ++app.pollGeneration;
  pollStatusLoop(generation);
}

async function pollStatusLoop(generation) {
  if (!app.csrf || generation !== app.pollGeneration) return;
  await pollStatus();
  if (!app.csrf || generation !== app.pollGeneration) return;
  app.statusTimer = setTimeout(
    () => pollStatusLoop(generation),
    document.hidden ? 2000 : 500,
  );
}

async function pollStatus() {
  const requestSequence = ++app.statusRequestSequence;
  try {
    const status = await request("/api/status");
    if (requestSequence < app.appliedStatusSequence) return;
    app.appliedStatusSequence = requestSequence;
    app.connected = true;
    app.status = status;
    renderStatus(status);
  } catch (error) {
    if (requestSequence < app.appliedStatusSequence) return;
    app.appliedStatusSequence = requestSequence;
    app.connected = false;
    renderConnection(false);
    if (error.status === 401) {
      app.csrf = null;
      establishSession();
    }
  }
}

function renderConnection(connected) {
  ui.connectionBadge.className = `status-pill ${connected ? "ok" : "bad"}`;
  ui.connectionBadge.querySelector("span:last-child").textContent = connected
    ? "XR 在线"
    : "连接中断";
}

function renderStatus(status) {
  renderConnection(true);
  ui.serverClock.textContent = new Date(status.server_time * 1000).toLocaleTimeString(
    "zh-CN",
    { hour12: false },
  );
  ui.episodeId.textContent = String(status.episode_id).padStart(4, "0");
  ui.stateBadge.textContent = status.state_label;
  ui.stateBadge.className = `state-badge ${status.state}`;
  ui.duration.textContent = formatDuration(status.duration_seconds);
  ui.frames10Hz.textContent = status.alignment.estimated_10hz_frames ?? 0;
  ui.horizonWindows.textContent = status.alignment.full_50_action_windows ?? 0;
  ui.taskName.disabled = status.state !== "ready";

  renderAlignment(status.alignment);
  renderTopics(status.alignment.topics || []);
  renderRecent(status.recent_episodes || []);
  renderSystem(status.system || {});
  renderButtons(status);
  renderRunMessage(status);
  renderFlow(status);
  renderCameraInfo(status.alignment.topics || []);
  syncCameraStreams(status);
  if (ui.cameraPreviewInfo) {
    const previewFps = Number(status.head_camera_source?.preview_fps || 1);
    const source = status.head_camera_source?.label || "E6 右侧流";
    ui.cameraPreviewInfo.textContent =
      `${source} + 双腕 30 Hz · E6 右目预览约 ${previewFps.toFixed(0)} FPS`;
  }
  maybeShowRecordingStarted(status);
  maybeShowReview(status);
  maybeShowResult(status);
  renderSaveSuccessState(status);
}

function renderAlignment(alignment) {
  const overall = alignment.overall || "neutral";
  ui.alignmentCard.className = `alignment-card panel ${overall}`;
  const titles = {
    ok: "可对齐",
    warn: "需要关注",
    bad: "暂不可用",
    neutral: "等待数据",
  };
  ui.alignmentTitle.textContent = titles[overall] || "等待数据";
  ui.alignmentSummary.textContent =
    alignment.summary || "开始预检后显示数据状态。";
  ui.commonDuration.textContent = `${formatNumber(
    alignment.common_duration_seconds,
    2,
  )} s`;
  const prefix = Number(alignment.uncovered_prefix_seconds);
  ui.uncoveredPrefix.textContent = Number.isFinite(prefix)
    ? `${prefix.toFixed(2)} s`
    : "—";
}

function sourceSummary(sources) {
  const entries = Object.entries(sources || {});
  if (!entries.length) return "—";
  return entries
    .map(([name, count]) => `${name.replaceAll("_", " ")} (${count})`)
    .join(" / ");
}

const TOPIC_GROUP_LABELS = {
  camera: "相机",
  state: "状态",
  tactile: "触觉",
  action: "动作",
};

function renderTopics(topics) {
  ui.topicTable.innerHTML = topics
    .map(
      (topic) => `
      <tr>
        <td>
          <div class="topic-name">
            <i class="${escapeHtml(topic.status)}"></i>
            <div>
              <strong title="${escapeHtml(topic.label)}">${escapeHtml(
                topic.label,
              )}</strong>
              <span class="topic-code">${escapeHtml(topic.name)}</span>
            </div>
          </div>
        </td>
        <td><span class="group-tag">${escapeHtml(
          TOPIC_GROUP_LABELS[topic.group] || topic.group,
        )}</span></td>
        <td>
          <span class="topic-status ${escapeHtml(topic.status)}">
            ${escapeHtml(topic.note)}
          </span>
        </td>
      </tr>`,
    )
    .join("");
}

function renderCameraInfo(topics) {
  const byName = Object.fromEntries(topics.map((topic) => [topic.name, topic]));
  for (const camera of cameras) {
    const topic = byName[camera];
    const target = document.querySelector(`#cameraInfo-${camera}`);
    if (!target) continue;
    target.textContent = topic
      ? `${Number(topic.count || 0).toLocaleString()} 帧 · ${formatNumber(
          topic.rate_hz,
          1,
        )} Hz`
      : "0 帧 · — Hz";
  }
}

function renderRecent(episodes) {
  if (!episodes.length) {
    ui.recentEpisodes.innerHTML =
      '<p class="run-message">还没有已保存 Episode。</p>';
    return;
  }
  ui.recentEpisodes.innerHTML = episodes
    .map(
      (episode) => `
      <div class="recent-item">
        <strong>EP ${String(episode.episode_id).padStart(4, "0")}</strong>
        <span title="${escapeHtml(episode.task)}">${escapeHtml(
          episode.task || "未命名任务",
        )}</span>
        <span>${formatNumber(episode.duration, 1)} s</span>
      </div>`,
    )
    .join("");
}

function renderSystem(system) {
  ui.diskFree.textContent = `${formatNumber(system.disk_free_gib, 1)} GiB`;
  ui.memoryFree.textContent = `${formatNumber(
    system.memory_available_gib,
    1,
  )} GiB`;
  ui.swapUsed.textContent = `${formatNumber(system.swap_used_gib, 1)} GiB`;
}

function renderButtons(status) {
  const state = status.state;
  const e6RecoveryRunning = Boolean(status.e6_recovery?.running);
  ui.startButton.disabled = state !== "ready";
  ui.recordButton.disabled = !(
    state === "armed" && status.start_gate?.confirm_allowed
  );
  ui.stopButton.disabled = state !== "recording";
  ui.saveButton.disabled = state !== "review" || !status.save_allowed;
  ui.discardButton.disabled = state !== "review";
  ui.closeButton.disabled = !new Set(["starting", "armed"]).has(state);
  ui.dialogSave.disabled = !status.save_allowed;
  ui.e6RecoverButton.disabled =
    e6RecoveryRunning || !new Set(["ready", "starting"]).has(state);
  ui.e6RecoverButton.textContent = e6RecoveryRunning
    ? "恢复中…"
    : "恢复 E6";
}

function renderRunMessage(status) {
  const gate = status.start_gate || {};
  const detected = Number(gate.detected_topic_count || 0);
  const required = requiredTopicCount(status);
  const blocked = gate.blocked_topics || [];
  const blockedHint = blocked.length
    ? `；待恢复：${blocked.join(", ")}`
    : "";
  const messages = {
    ready: "数采监听已关闭。准备好设备后点击“开始预检”。",
    starting:
      `正在预检 ${required} 路数据流：${detected}/${required}${blockedHint}。` +
      "准备阶段不要执行任务；保持 VR 双臂在线，并触发左右 Revo2 命令。",
    armed: status.start_gate?.confirm_allowed
      ? `${required} 路数据监测正常，可以点击“开始采集”。`
      : `数据监测中，当前暂不能开始采集${blockedHint}。`,
    canceling_start: "正在停止数据监听并清理预检缓存…",
    starting_recording: "正在记录本条 Episode 的有效起点…",
    recording:
      `${required} 路有效起点已记录，正在采集本条 Episode；完成后点击“停止本条”。`,
    stopping: "正在停止全部采集线程并封存临时流…",
    review: "本条已停止。请选择保存本条或删除本条。",
    saving: "正在保存并校验；完成后会自动继续预检下一条。",
    discarding: "正在删除本条；完成后会自动继续预检，Episode 编号不增加。",
    cooldown: `已处理完成，${status.cooldown_remaining_seconds} 秒后继续。`,
    closing: "正在关闭数采并停止所有数据监听…",
    error: status.last_error || "采集器发生错误，请查看服务日志。",
  };
  ui.runMessage.textContent = messages[status.state] || status.state_label;
}

const FLOW_STAGES = [
  "prepare",
  "preflight",
  "monitor",
  "record",
  "review",
  "close",
];

function renderFlow(status) {
  const stageByState = {
    initializing: "prepare",
    ready: "prepare",
    starting: "preflight",
    armed: "monitor",
    canceling_start: "close",
    starting_recording: "record",
    recording: "record",
    stopping: "review",
    review: "review",
    saving: "review",
    discarding: "review",
    cooldown: "preflight",
    closing: "close",
    error: "prepare",
  };
  const labels = {
    prepare: "准备设备",
    preflight: "开始预检",
    monitor: "数据监测",
    record: "Episode 采集",
    review: "保存或删除",
    close: "关闭数采",
  };
  const activeStage = stageByState[status.state] || "prepare";
  const activeIndex = FLOW_STAGES.indexOf(activeStage);
  ui.flowTitle.textContent =
    status.state === "error"
      ? "当前：流程异常"
      : `当前：${labels[activeStage]}`;
  for (const step of ui.flowSteps) {
    const index = FLOW_STAGES.indexOf(step.dataset.flowStage);
    step.classList.toggle("active", index === activeIndex);
    step.classList.toggle("complete", index < activeIndex);
    step.classList.toggle(
      "error",
      status.state === "error" && index === activeIndex,
    );
    step.setAttribute(
      "aria-current",
      index === activeIndex ? "step" : "false",
    );
  }
}

function maybeShowRecordingStarted(status) {
  if (status.state !== "recording") {
    if (ui.recordingDialog.open) ui.recordingDialog.close();
    return;
  }

  if (ui.saveSuccessDialog.open) ui.saveSuccessDialog.close();

  const runId = status.run_id;
  if (!runId || app.recordingNoticeRunId === runId) return;
  if (ui.recordingDialog.open) ui.recordingDialog.close();
  app.recordingNoticeRunId = runId;
  ui.recordingDialog.showModal();
}

function maybeShowReview(status) {
  if (status.state !== "review") {
    if (ui.reviewDialog.open) ui.reviewDialog.close();
    app.reviewRunId = null;
    return;
  }
  if (app.reviewRunId === status.run_id && ui.reviewDialog.open) return;
  app.reviewRunId = status.run_id;
  const alignment = status.alignment;
  const validation = status.pending_validation || {};
  const required = requiredTopicCount(status);
  ui.reviewSummary.textContent = status.save_allowed
    ? `有效采集区间的 ${required} 路数据已封存；准备阶段会在转换时自动跳过。`
    : "检测到缺失、中断或无效数据，本条不能保存，只能删除。";
  const warnings = [];
  const prefix = Number(alignment.uncovered_prefix_seconds);
  if (Number.isFinite(prefix) && prefix > 0.5) {
    warnings.push(
      `开头约 ${prefix.toFixed(2)} 秒没有完整 ${required} 路共同覆盖。`,
    );
  }
  if ((alignment.missing_topics || []).length) {
    warnings.push(`缺少：${alignment.missing_topics.join(", ")}`);
  }
  if ((alignment.invalid_topics || []).length) {
    warnings.push(`时间戳异常：${alignment.invalid_topics.join(", ")}`);
  }
  if ((validation.interrupted_topics || []).length) {
    warnings.push(
      `采集期间数据流中断：${validation.interrupted_topics.join(", ")}`,
    );
  }
  if ((validation.effective_missing_topics || []).length) {
    warnings.push(
      `有效起点后没有新样本：${validation.effective_missing_topics.join(", ")}`,
    );
  }
  if (
    validation.effective_interval_valid === false &&
    !(validation.effective_missing_topics || []).length
  ) {
    warnings.push("有效采集区间为空，或连续数据流没有完整覆盖该区间。");
  }
  if ((validation.collector_errors || []).length) {
    warnings.push("采集线程报告异常，本条禁止保存。");
  }
  if (!warnings.length) warnings.push("在线检查未发现阻断性问题。");
  ui.reviewWarnings.innerHTML = warnings
    .map((warning) => `<div class="review-warning">${escapeHtml(warning)}</div>`)
    .join("");
  if (!ui.reviewDialog.open) ui.reviewDialog.showModal();
}

function renderSaveSuccessState(status) {
  if (!ui.saveSuccessDialog.open) return;
  const nextReady = Boolean(
    status.state === "armed" && status.start_gate?.confirm_allowed,
  );
  ui.saveSuccessNext.disabled = !nextReady;
  ui.saveSuccessNext.textContent = nextReady
    ? "开始下一条"
    : "等待下一条预检";

  const restartResult = status.last_result?.monitoring_result;
  if (restartResult?.error) {
    ui.saveSuccessState.textContent =
      "下一条预检失败：" + restartResult.error
      + "。请关闭提示后重新预检。";
  } else if (nextReady) {
    ui.saveSuccessState.textContent =
      "下一条的 9 路数据预检通过，可以开始采集。";
  } else if (status.state === "starting") {
    ui.saveSuccessState.textContent =
      "保存已完成，正在预检下一条数据流，请稍候…";
  } else {
    ui.saveSuccessState.textContent =
      "保存已完成；请关闭提示后从页面开始下一条预检。";
  }
}

function maybeShowResult(status) {
  const result = status.last_result;
  if (!result) return;
  const resultId = result.episode_id ?? result.run_id ?? status.version;
  const key = `${result.kind}:${resultId}`;
  if (key === app.lastResultKey) return;
  app.lastResultKey = key;
  if (result.kind === "saved") {
    const episodeLabel = String(result.episode_id).padStart(4, "0");
    const duration = Number(
      result.effective_duration ?? result.duration,
    );
    const saveElapsed = Number(result.save_elapsed_seconds);
    const durationText = Number.isFinite(duration)
      ? "，有效时长 " + duration.toFixed(2) + " 秒"
      : "";
    const saveElapsedText = Number.isFinite(saveElapsed)
      ? "，保存校验耗时 " + saveElapsed.toFixed(2) + " 秒"
      : "";
    ui.saveSuccessSummary.textContent =
      "Episode " + episodeLabel + " 已完整落盘"
      + durationText + saveElapsedText + "。";
    if (!ui.saveSuccessDialog.open) {
      ui.saveSuccessDialog.showModal();
    }
    renderSaveSuccessState(status);
    showToast("Episode " + episodeLabel + " 已保存并校验");
  } else if (result.kind === "discarded") {
    showToast(`Episode ${String(result.episode_id).padStart(4, "0")} 已删除`);
  } else if (result.kind === "preflight_timeout") {
    const blocked = (result.blocked_topics || []).join(", ");
    showToast(`预检超时${blocked ? `，异常：${blocked}` : ""}`, true);
  } else if (result.kind === "preflight_failed") {
    showToast(`预检启动失败：${result.error || "请检查数据源"}`, true);
  } else if (result.kind === "collection_closed") {
    showToast("数采已关闭，所有数据监听已停止");
  }
}

const CAMERA_STREAM_STATES = new Set([
  "starting",
  "armed",
  "starting_recording",
  "recording",
  "stopping",
]);
const CAMERA_SNAPSHOT_INTERVAL_MS = 200;
const CAMERA_MJPEG_RETRY_MS = 10000;

function clearCameraReconnect(camera) {
  const timer = app.cameraReconnectTimers.get(camera);
  if (timer) clearTimeout(timer);
  app.cameraReconnectTimers.delete(camera);
}

function openCameraStream(camera) {
  if (!app.cameraStreamsActive || document.hidden) return;
  clearCameraReconnect(camera);
  const image = document.querySelector(`#camera-${camera}`);
  if (!image) return;
  const useSnapshot =
    Date.now() < Number(app.cameraMjpegRetryAfter.get(camera) || 0);
  const scheduleNext = (delay) => {
    if (!app.cameraStreamsActive || document.hidden) return;
    clearCameraReconnect(camera);
    app.cameraReconnectTimers.set(
      camera,
      setTimeout(() => openCameraStream(camera), delay),
    );
  };
  image.onload = () => {
    image.classList.add("ready");
    if (useSnapshot) scheduleNext(CAMERA_SNAPSHOT_INTERVAL_MS);
    else app.cameraMjpegRetryAfter.delete(camera);
  };
  image.onerror = () => {
    image.classList.remove("ready");
    if (!app.cameraStreamsActive || document.hidden) return;
    if (!useSnapshot) {
      app.cameraMjpegRetryAfter.set(
        camera,
        Date.now() + CAMERA_MJPEG_RETRY_MS,
      );
    }
    scheduleNext(useSnapshot ? 1000 : 0);
  };
  image.src = useSnapshot
    ? `/api/camera/${camera}.jpg?v=${Date.now()}`
    : `/api/camera/${camera}.mjpeg?v=${Date.now()}`;
}

function startCameraStreams() {
  if (app.cameraStreamsActive || document.hidden) return;
  app.cameraStreamsActive = true;
  for (const camera of cameras) openCameraStream(camera);
}

function stopCameraStreams() {
  app.cameraStreamsActive = false;
  for (const camera of cameras) {
    clearCameraReconnect(camera);
    const image = document.querySelector(`#camera-${camera}`);
    if (!image) continue;
    image.onload = null;
    image.onerror = null;
    image.removeAttribute("src");
    image.classList.remove("ready");
  }
}

function syncCameraStreams(status = app.status) {
  const shouldStream = Boolean(
    status && CAMERA_STREAM_STATES.has(status.state) && !document.hidden,
  );
  if (shouldStream) startCameraStreams();
  else stopCameraStreams();
}

document.addEventListener("visibilitychange", () => {
  syncCameraStreams();
  if (app.csrf) startPolling();
});

async function command(path, body = {}) {
  try {
    const payload = await request(path, {
      method: "POST",
      body: { ...body, run_id: app.status?.run_id || null },
    });
    const acceptedMessages = {
      saving: "正在保存并校验，请稍候…",
      starting_recording: "正在记录有效起点…",
      stopping: "正在停止并封存本条数据…",
    };
    if (payload.state) {
      showToast(
        acceptedMessages[payload.state]
          || "操作已接受：" + payload.state,
      );
    }
    await pollStatus();
    return true;
  } catch (error) {
    showToast(error.message, true);
    if (error.status === 409) await pollStatus();
    return false;
  }
}

ui.startButton.addEventListener("click", async () => {
  ui.startButton.disabled = true;
  const accepted = await command("/api/start", {
    task: ui.taskName.value.trim() || DEFAULT_TASK,
  });
  if (!accepted && app.status) renderButtons(app.status);
});

ui.recordButton.addEventListener("click", async () => {
  if (!window.confirm("数据检查已通过，确认开始采集本条 Episode？")) return;
  ui.recordButton.disabled = true;
  const accepted = await command("/api/confirm-start");
  if (!accepted && app.status) renderButtons(app.status);
});

ui.saveSuccessNext.addEventListener("click", async () => {
  if (
    app.status?.state !== "armed"
    || !app.status?.start_gate?.confirm_allowed
  ) {
    renderSaveSuccessState(app.status || {});
    return;
  }
  ui.saveSuccessNext.disabled = true;
  ui.saveSuccessNext.textContent = "正在开始…";
  const accepted = await command("/api/confirm-start");
  if (accepted && ui.saveSuccessDialog.open) {
    ui.saveSuccessDialog.close();
  } else if (app.status) {
    renderSaveSuccessState(app.status);
  }
});

ui.stopButton.addEventListener("click", () => {
  if (window.confirm("确认动作已完成并停止本条采集？停止后再选择保存或删除。")) {
    command("/api/stop");
  }
});

ui.closeButton.addEventListener("click", async () => {
  if (!window.confirm("确认关闭数采？这会停止相机和 Topic 数据监听。")) return;
  ui.closeButton.disabled = true;
  const accepted = await command("/api/close");
  if (!accepted && app.status) renderButtons(app.status);
});

ui.e6RecoverButton.addEventListener("click", async () => {
  if (!new Set(["ready", "starting"]).has(app.status?.state)) {
    showToast("只能在数采关闭或预检阶段恢复 E6", true);
    return;
  }
  ui.e6RecoverButton.disabled = true;
  ui.e6RecoverButton.textContent = "恢复中…";
  showToast("正在检测并恢复 E6，最多需要约 20 秒…");
  try {
    const payload = await request("/api/recover-e6", {
      method: "POST",
      body: { run_id: app.status?.run_id || null },
    });
    const result = payload.e6_recovery || {};
    showToast(result.message || "E6 恢复操作已完成", !result.ok);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    await pollStatus();
    if (app.status) renderButtons(app.status);
  }
});

function requestSave() {
  const alignment = app.status?.alignment;
  const hasWarning = alignment && alignment.overall !== "ok";
  if (
    hasWarning &&
    !window.confirm("在线可对齐性存在提示。仍要保存本条数据吗？")
  ) {
    return;
  }
  command("/api/save");
}

function requestDiscard() {
  if (
    window.confirm(
      "确定永久删除本条数据吗？Episode 编号不会增加，此操作无法撤销。",
    )
  ) {
    command("/api/discard");
  }
}

ui.saveButton.addEventListener("click", requestSave);
ui.dialogSave.addEventListener("click", (event) => {
  event.preventDefault();
  requestSave();
});
ui.discardButton.addEventListener("click", requestDiscard);
ui.dialogDiscard.addEventListener("click", (event) => {
  event.preventDefault();
  requestDiscard();
});

establishSession();
