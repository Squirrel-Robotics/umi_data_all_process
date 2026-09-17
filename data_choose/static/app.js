"use strict";

/*
 * Backend adaptation lives in api() and ENDPOINTS. The rest of the UI never
 * talks to fetch() directly, so alternate response shapes can be normalized in
 * one place without touching playback or review behavior.
 */

const API_BASE = document.documentElement.dataset.apiBase || "";
const TOKEN_STORAGE_KEY = "dataChooseReviewToken";

const ENDPOINTS = {
  config: () => "/api/config",
  stats: () => "/api/stats",
  precache: () => "/api/precache",
  startPrecache: () => "/api/precache/start",
  episodes: ({ status, query }) => {
    const params = new URLSearchParams();
    if (status) params.set("state", status);
    params.set("page", "1");
    params.set("limit", "500");
    if (query) {
      params.set("q", query);
      params.set("search", query);
    }
    const suffix = params.toString();
    return `/api/episodes${suffix ? `?${suffix}` : ""}`;
  },
  episode: (episodeId) => `/api/episodes/${encodeURIComponent(episodeId)}`,
  media: (episodeId, role) =>
    `/api/media/${encodeURIComponent(episodeId)}/${encodeURIComponent(role)}`,
  decision: (episodeId) => `/api/episodes/${encodeURIComponent(episodeId)}/decision`,
  trash: ({ query } = {}) => {
    const params = new URLSearchParams();
    if (query) {
      params.set("q", query);
      params.set("search", query);
    }
    const suffix = params.toString();
    return `/api/trash${suffix ? `?${suffix}` : ""}`;
  },
  restore: (trashName) => `/api/trash/${encodeURIComponent(trashName)}/restore`,
};

class ApiError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

function captureTokenFromUrl() {
  const url = new URL(window.location.href);
  const token = url.searchParams.get("token");
  if (!token) return;
  localStorage.setItem(TOKEN_STORAGE_KEY, token);
  url.searchParams.delete("token");
  window.history.replaceState({}, "", `${url.pathname}${url.search}${url.hash}`);
}

async function api(path, options = {}) {
  const {
    method = "GET",
    body,
    headers: customHeaders = {},
    signal,
  } = options;
  const headers = new Headers(customHeaders);
  headers.set("Accept", "application/json");

  const token = localStorage.getItem(TOKEN_STORAGE_KEY);
  if (token) headers.set("X-Review-Token", token);
  if (body !== undefined && !(body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body:
      body === undefined || body instanceof FormData ? body : JSON.stringify(body),
    credentials: "same-origin",
    cache: "no-store",
    signal,
  });

  const contentType = response.headers.get("content-type") || "";
  let payload = null;
  if (response.status !== 204) {
    payload = contentType.includes("application/json")
      ? await response.json().catch(() => null)
      : await response.text().catch(() => "");
  }

  if (!response.ok) {
    const detail =
      payload?.detail || payload?.error || payload?.message ||
      (typeof payload === "string" && payload) ||
      `HTTP ${response.status}`;
    throw new ApiError(String(detail), response.status, payload);
  }
  return payload;
}

const dom = {
  connectionBadge: document.querySelector("#connectionBadge"),
  datasetLabel: document.querySelector("#datasetLabel"),
  refreshButton: document.querySelector("#refreshButton"),
  reviewProgressText: document.querySelector("#reviewProgressText"),
  reviewProgressBar: document.querySelector("#reviewProgressBar"),
  precacheStatus: document.querySelector("#precacheStatus"),
  precacheText: document.querySelector("#precacheText"),
  precacheCurrent: document.querySelector("#precacheCurrent"),
  precacheButton: document.querySelector("#precacheButton"),
  statTotal: document.querySelector("#statTotal"),
  statPending: document.querySelector("#statPending"),
  statKept: document.querySelector("#statKept"),
  statSkipped: document.querySelector("#statSkipped"),
  statTrashed: document.querySelector("#statTrashed"),
  episodeOrdinal: document.querySelector("#episodeOrdinal"),
  episodeTitle: document.querySelector("#episodeTitle"),
  episodeStateBadge: document.querySelector("#episodeStateBadge"),
  handPoseBadge: document.querySelector("#handPoseBadge"),
  handPoseText: document.querySelector("#handPoseText"),
  mediaBadge: document.querySelector("#mediaBadge"),
  mediaText: document.querySelector("#mediaText"),
  warningBadge: document.querySelector("#warningBadge"),
  warningText: document.querySelector("#warningText"),
  viewerPanel: document.querySelector("#viewerPanel"),
  headVideo: document.querySelector("#headVideo"),
  leftVideo: document.querySelector("#leftVideo"),
  rightVideo: document.querySelector("#rightVideo"),
  headOverlay: document.querySelector("#headOverlay"),
  leftOverlay: document.querySelector("#leftOverlay"),
  rightOverlay: document.querySelector("#rightOverlay"),
  currentTimeLabel: document.querySelector("#currentTimeLabel"),
  durationLabel: document.querySelector("#durationLabel"),
  timeline: document.querySelector("#timeline"),
  playButton: document.querySelector("#playButton"),
  previousFrameButton: document.querySelector("#previousFrameButton"),
  nextFrameButton: document.querySelector("#nextFrameButton"),
  backOneSecondButton: document.querySelector("#backOneSecondButton"),
  forwardOneSecondButton: document.querySelector("#forwardOneSecondButton"),
  previousEpisodeButton: document.querySelector("#previousEpisodeButton"),
  nextEpisodeButton: document.querySelector("#nextEpisodeButton"),
  speedSelect: document.querySelector("#speedSelect"),
  keepButton: document.querySelector("#keepButton"),
  skipButton: document.querySelector("#skipButton"),
  deleteButton: document.querySelector("#deleteButton"),
  queueTitle: document.querySelector("#queueTitle"),
  queueCount: document.querySelector("#queueCount"),
  searchInput: document.querySelector("#searchInput"),
  clearSearchButton: document.querySelector("#clearSearchButton"),
  queueLoading: document.querySelector("#queueLoading"),
  queueEmpty: document.querySelector("#queueEmpty"),
  episodeList: document.querySelector("#episodeList"),
  queuePreviousButton: document.querySelector("#queuePreviousButton"),
  queueNextButton: document.querySelector("#queueNextButton"),
  queuePosition: document.querySelector("#queuePosition"),
  deleteDialog: document.querySelector("#deleteDialog"),
  deleteEpisodeId: document.querySelector("#deleteEpisodeId"),
  deleteReason: document.querySelector("#deleteReason"),
  deleteNote: document.querySelector("#deleteNote"),
  cancelDeleteButton: document.querySelector("#cancelDeleteButton"),
  confirmDeleteButton: document.querySelector("#confirmDeleteButton"),
  toastRegion: document.querySelector("#toastRegion"),
  filterTabs: [...document.querySelectorAll(".filter-tab")],
  focusButtons: [...document.querySelectorAll("[data-focus-video]")],
};

const videos = [dom.headVideo, dom.leftVideo, dom.rightVideo];
const followers = [dom.leftVideo, dom.rightVideo];
const streamDescriptors = [
  { video: dom.headVideo, overlay: dom.headOverlay, role: "head", label: "E6 右眼" },
  { video: dom.leftVideo, overlay: dom.leftOverlay, role: "left_wrist", label: "左腕 cam0" },
  { video: dom.rightVideo, overlay: dom.rightOverlay, role: "right_wrist", label: "右腕 cam1" },
];

const FILTER_META = {
  pending: { title: "待审核", backendStatus: "pending" },
  kept: { title: "已保留", backendStatus: "kept" },
  skipped: { title: "已跳过", backendStatus: "skipped" },
  trash: { title: "回收站", backendStatus: "trashed" },
};

const STATUS_META = {
  pending: { label: "待审核", className: "is-pending" },
  in_review: { label: "正在审核", className: "is-pending" },
  kept: { label: "已保留", className: "is-kept" },
  keep: { label: "已保留", className: "is-kept" },
  skipped: { label: "已跳过", className: "is-skipped" },
  skip: { label: "已跳过", className: "is-skipped" },
  trashed: { label: "回收站", className: "is-trash" },
  quarantined: { label: "回收站", className: "is-trash" },
  rejected: { label: "回收站", className: "is-trash" },
  delete: { label: "回收站", className: "is-trash" },
  error: { label: "异常", className: "is-error" },
};

const state = {
  filter: "pending",
  query: "",
  episodes: [],
  current: null,
  currentIndex: -1,
  stats: { total: 0, pending: 0, kept: 0, skipped: 0, trashed: 0, reviewed: 0 },
  previewFps: 15,
  listRequestId: 0,
  detailRequestId: 0,
  playbackRate: 1,
  syncFrame: null,
  mediaPollTimer: null,
  precache: null,
  precachePollTimer: null,
  precacheRequestId: 0,
  precacheStarting: false,
  focus: "all",
  busy: false,
};

function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null);
}

function asNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function normalizeStatus(rawStatus) {
  const status = String(rawStatus || "pending").toLowerCase();
  const aliases = {
    approved: "kept",
    retained: "kept",
    pass: "kept",
    deferred: "skipped",
    quarantined: "trashed",
    rejected: "trashed",
    deleted: "trashed",
    trash: "trashed",
  };
  return aliases[status] || status;
}

function normalizeEpisode(raw = {}, forcedStatus) {
  const nested = raw.episode && typeof raw.episode === "object" ? raw.episode : raw;
  const handPose = nested.hand_pose || nested.handPose || {};
  const media = nested.media || {};
  const id = String(
    firstDefined(
      nested.episode_id,
      nested.id,
      nested.name,
      nested.original_episode_id,
      raw.episode_id,
      raw.id,
      raw.name,
      "",
    ),
  );
  const status = normalizeStatus(
    firstDefined(forcedStatus, nested.status, nested.review_state, nested.decision),
  );
  const handPoseExists = Boolean(
    firstDefined(
      nested.hand_pose_exists,
      nested.handPoseExists,
      handPose.exists,
      handPose.present,
      false,
    ),
  );
  const rows = firstDefined(
    nested.hand_pose_rows,
    nested.handPoseRows,
    handPose.rows,
    handPose.row_count,
    null,
  );
  const mediaStatus = String(
    firstDefined(nested.media_status, nested.mediaState, media.status, "unknown"),
  ).toLowerCase();
  const quality = nested.quality || {};
  const problemCodesRaw = firstDefined(
    nested.problem_codes,
    nested.problemCodes,
    nested.health?.problem_codes,
    quality.problem_codes,
    nested.warnings,
    [],
  );
  const problemCodes = Array.isArray(problemCodesRaw)
    ? problemCodesRaw.map(String).filter(Boolean)
    : problemCodesRaw
      ? [String(problemCodesRaw)]
      : [];
  const warningValues = firstDefined(
    nested.warning,
    nested.warning_message,
    nested.health?.warning,
    quality.warnings,
    [],
  );
  const warningParts = Array.isArray(warningValues)
    ? warningValues.map(String).filter(Boolean)
    : warningValues
      ? [String(warningValues)]
      : [];
  if (quality.has_warning && !warningParts.length && !problemCodes.length) {
    const healthSummary = [quality.customer_camera_status, quality.health_status]
      .filter((value) => value && !["ok", "healthy"].includes(String(value).toLowerCase()));
    warningParts.push(healthSummary.join(" / ") || "数据健康检查异常");
  }
  const rawProgress = asNumber(
    firstDefined(nested.media_progress, media.progress, media.percent),
    0,
  );
  const mediaErrorValue = firstDefined(nested.media_error, media.error, nested.error, "");
  const mediaError = mediaErrorValue && typeof mediaErrorValue === "object"
    ? Object.entries(mediaErrorValue)
        .map(([key, value]) => `${key}: ${value}`)
        .join(" · ")
    : String(mediaErrorValue || "");

  return {
    raw: nested,
    id,
    status,
    version: asNumber(firstDefined(nested.version, nested.revision), 0),
    handPoseExists,
    handPoseRows: rows == null ? null : asNumber(rows, 0),
    mediaStatus,
    mediaProgress: rawProgress > 0 && rawProgress <= 1 ? rawProgress * 100 : rawProgress,
    mediaError,
    problemCodes,
    warning: warningParts.join(" · "),
    fps: asNumber(
      firstDefined(
        nested.preview_fps,
        media.preview_fps,
        nested.fps,
        media.fps,
        media.timeline_fps,
        nested.source_fps,
      ),
      15,
    ),
    duration: asNumber(
      firstDefined(nested.duration, nested.duration_seconds, media.duration),
      0,
    ),
    trashName: String(
      firstDefined(nested.trash_name, nested.quarantine_name, raw.trash_name, id),
    ),
    deletedAt: String(
      firstDefined(nested.deleted_at, nested.quarantined_at, raw.deleted_at, ""),
    ),
  };
}

function normalizeList(payload, forcedStatus) {
  const items = Array.isArray(payload)
    ? payload
    : firstDefined(payload?.items, payload?.episodes, payload?.trash, payload?.results, []);
  return Array.isArray(items)
    ? items.map((item) => normalizeEpisode(item, forcedStatus)).filter((item) => item.id)
    : [];
}

function normalizeStats(payload = {}) {
  const source = payload.stats || payload.counts || payload;
  const total = asNumber(firstDefined(source.total, source.all), 0);
  const pending = asNumber(firstDefined(source.pending, source.unreviewed), 0);
  const kept = asNumber(firstDefined(source.kept, source.keep, source.approved), 0);
  const skipped = asNumber(firstDefined(source.skipped, source.skip, source.deferred), 0);
  const trashed = asNumber(
    firstDefined(source.trashed, source.trash, source.quarantined, source.deleted),
    0,
  );
  return {
    total: total || pending + kept + skipped + trashed,
    pending,
    kept,
    skipped,
    trashed,
    reviewed: asNumber(firstDefined(source.reviewed), kept + skipped + trashed),
  };
}

function precacheMediaLabel(value) {
  const text = String(value || "");
  const labels = {
    head: "头部相机",
    left_wrist: "左腕相机",
    right_wrist: "右腕相机",
  };
  return labels[text] || text;
}

function describePrecacheItem(value) {
  if (typeof value === "string" || typeof value === "number") {
    const text = String(value);
    const separatorIndex = text.lastIndexOf("/");
    if (separatorIndex > 0 && separatorIndex < text.length - 1) {
      const episode = text.slice(0, separatorIndex);
      const media = precacheMediaLabel(text.slice(separatorIndex + 1));
      return `${episode} · ${media}`;
    }
    return precacheMediaLabel(text);
  }
  if (!value || typeof value !== "object") return "";
  const episode = firstDefined(value.episode_id, value.episode, value.id, "");
  const mediaValue = firstDefined(
    value.media,
    value.media_key,
    value.camera,
    value.role,
    "",
  );
  if (!episode && mediaValue) return describePrecacheItem(mediaValue);
  const media = precacheMediaLabel(mediaValue);
  return [episode, media].filter(Boolean).join(" · ");
}

function describePrecacheError(value) {
  if (typeof value === "string" || typeof value === "number") return String(value);
  if (!value || typeof value !== "object") return "";
  const target = describePrecacheItem(value);
  const error = firstDefined(value.error, value.message, value.detail, "");
  if (target && error) return `${target}：${error}`;
  return String(error || target || "未知缓存错误");
}

function normalizePrecache(payload = {}) {
  const total = Math.max(0, asNumber(payload.total_media, 0));
  const ready = Math.max(0, asNumber(payload.ready, 0));
  const completed = Math.max(0, asNumber(payload.completed, ready));
  const failed = Math.max(0, asNumber(payload.failed, 0));
  const percent = Math.max(
    0,
    Math.min(100, asNumber(payload.percent, total ? (completed / total) * 100 : 0)),
  );
  const currentValue = payload.current;
  let current = "";
  if (Array.isArray(currentValue)) {
    current = currentValue.map(describePrecacheItem).filter(Boolean).join("、");
  } else {
    current = describePrecacheItem(currentValue);
  }
  const rawErrors = Array.isArray(payload.errors)
    ? payload.errors
    : payload.errors
      ? [payload.errors]
      : [];
  const errors = rawErrors.map(describePrecacheError).filter(Boolean);
  return {
    running: Boolean(payload.running),
    total,
    ready,
    completed,
    failed,
    percent,
    current,
    errors,
    autoMonitor:
      Boolean(payload.auto_enabled) || Math.max(0, asNumber(payload.rescan_seconds, 0)) > 0,
    rescanSeconds: Math.max(0, asNumber(payload.rescan_seconds, 0)),
  };
}

function setConnection(status, label) {
  dom.connectionBadge.className = `connection-badge is-${status}`;
  dom.connectionBadge.querySelector("span:last-child").textContent = label;
}

function showToast(message, type = "info", duration = 3200) {
  const toast = document.createElement("div");
  toast.className = `toast is-${type}`;
  toast.textContent = message;
  dom.toastRegion.append(toast);
  window.setTimeout(() => toast.remove(), duration);
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "00:00.000";
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(3).padStart(6, "0")}`;
}

function statusMeta(status) {
  return STATUS_META[normalizeStatus(status)] || {
    label: status || "未知",
    className: "is-neutral",
  };
}

function updateStats() {
  const { total, pending, kept, skipped, trashed } = state.stats;
  const reviewed = asNumber(state.stats.reviewed, kept + skipped + trashed);
  const percent = total ? Math.min(100, (reviewed / total) * 100) : 0;
  dom.statTotal.textContent = total;
  dom.statPending.textContent = pending;
  dom.statKept.textContent = kept;
  dom.statSkipped.textContent = skipped;
  dom.statTrashed.textContent = trashed;
  dom.reviewProgressText.textContent = `${reviewed} / ${total}`;
  dom.reviewProgressBar.style.width = `${percent}%`;
}

function renderPrecache() {
  const cache = state.precache;
  dom.precacheStatus.className = "precache-status";
  dom.precacheCurrent.textContent = "";
  dom.precacheCurrent.removeAttribute("title");

  if (!cache) {
    dom.precacheStatus.classList.add("is-checking");
    dom.precacheText.textContent = "正在检查后台缓存";
    dom.precacheButton.textContent = "开始缓存";
    dom.precacheButton.disabled = true;
    return;
  }

  const allReady = cache.total > 0 && cache.failed === 0 && cache.ready >= cache.total;
  if (cache.running) {
    dom.precacheStatus.classList.add("is-running");
    dom.precacheText.textContent = `缓存 ${cache.ready}/${cache.total} · ${Math.round(cache.percent)}%`;
    if (cache.current) {
      dom.precacheCurrent.textContent = `正在处理 ${cache.current}`;
      dom.precacheCurrent.title = cache.current;
    }
    dom.precacheButton.textContent = "缓存中";
    dom.precacheButton.disabled = true;
  } else if (cache.failed > 0) {
    dom.precacheStatus.classList.add("is-failed");
    dom.precacheText.textContent = `有 ${cache.failed} 个失败 · 已缓存 ${cache.ready}/${cache.total}`;
    if (cache.errors.length) {
      const errorSummary = cache.errors.join(" · ");
      dom.precacheCurrent.textContent = errorSummary;
      dom.precacheCurrent.title = errorSummary;
    }
    dom.precacheButton.textContent = "重试缓存";
    dom.precacheButton.disabled = state.precacheStarting;
  } else if (allReady) {
    dom.precacheStatus.classList.add("is-complete");
    dom.precacheText.textContent = cache.autoMonitor
      ? "全部已缓存 · 自动监控新数据"
      : "全部已缓存";
    dom.precacheButton.textContent = "已完成";
    dom.precacheButton.disabled = true;
  } else {
    dom.precacheStatus.classList.add("is-idle");
    dom.precacheText.textContent = cache.total
      ? `缓存 ${cache.ready}/${cache.total} · ${Math.round(cache.percent)}%`
      : "后台缓存尚未开始";
    dom.precacheButton.textContent = cache.ready > 0 ? "继续缓存" : "开始缓存";
    dom.precacheButton.disabled = state.precacheStarting;
  }
  if (state.precacheStarting) {
    dom.precacheButton.textContent = "启动中";
    dom.precacheButton.disabled = true;
  }
}

function stopPrecachePoll() {
  if (state.precachePollTimer) window.clearTimeout(state.precachePollTimer);
  state.precachePollTimer = null;
}

function schedulePrecachePoll(delay) {
  stopPrecachePoll();
  state.precachePollTimer = window.setTimeout(() => {
    loadPrecache().catch(() => {});
  }, delay);
}

async function loadPrecache() {
  stopPrecachePoll();
  const requestId = ++state.precacheRequestId;
  try {
    const payload = await api(ENDPOINTS.precache());
    if (requestId !== state.precacheRequestId) return;
    state.precache = normalizePrecache(payload || {});
    renderPrecache();
    schedulePrecachePoll(state.precache.running ? 2000 : state.precache.failed > 0 ? 15000 : 30000);
  } catch (error) {
    if (requestId !== state.precacheRequestId) return;
    state.precache = null;
    dom.precacheStatus.className = "precache-status is-unavailable";
    dom.precacheText.textContent = "缓存状态暂不可用";
    dom.precacheCurrent.textContent = "";
    dom.precacheButton.textContent = "稍后重试";
    dom.precacheButton.disabled = true;
    schedulePrecachePoll(30000);
  }
}

async function startPrecache() {
  if (state.precacheStarting || state.precache?.running) return;
  state.precacheStarting = true;
  renderPrecache();
  try {
    await api(ENDPOINTS.startPrecache(), { method: "POST" });
    showToast("后台缓存已启动，可继续审核数据", "success");
    await loadPrecache();
  } catch (error) {
    showToast(`启动缓存失败：${error.message}`, "error", 5000);
  } finally {
    state.precacheStarting = false;
    renderPrecache();
  }
}

function updateQueuePosition() {
  const total = state.episodes.length;
  const position = state.currentIndex >= 0 ? state.currentIndex + 1 : 0;
  dom.queuePosition.textContent = total ? `${position} / ${total}` : "— / —";
  dom.queuePreviousButton.disabled = position <= 1;
  dom.previousEpisodeButton.disabled = position <= 1;
  dom.queueNextButton.disabled = position === 0 || position >= total;
  dom.nextEpisodeButton.disabled = position === 0 || position >= total;
}

function renderQueue() {
  dom.episodeList.replaceChildren();
  dom.queueCount.textContent = state.episodes.length;
  dom.queueTitle.textContent = FILTER_META[state.filter].title;
  dom.queueLoading.classList.add("is-hidden");
  dom.queueEmpty.classList.toggle("is-hidden", state.episodes.length > 0);

  state.episodes.forEach((episode, index) => {
    const li = document.createElement("li");
    li.className = "episode-item";
    if (state.current?.id === episode.id) li.classList.add("is-active");

    const row = document.createElement(state.filter === "trash" ? "div" : "button");
    if (row instanceof HTMLButtonElement) row.type = "button";
    row.className = "episode-item-row";
    row.dataset.episodeId = episode.id;

    const body = document.createElement("span");
    const id = document.createElement("span");
    id.className = "episode-item-id";
    id.textContent = episode.id;
    const meta = document.createElement("span");
    meta.className = "episode-item-meta";
    const handText = episode.handPoseExists
      ? `hand_pose${episode.handPoseRows !== null ? ` · ${episode.handPoseRows} 行` : ""}`
      : "hand_pose 缺失";
    meta.textContent = state.filter === "trash"
      ? `${handText}${episode.deletedAt ? ` · ${episode.deletedAt}` : ""}`
      : `${statusMeta(episode.status).label} · ${handText}`;
    body.append(id, meta);
    row.append(body);

    if (state.filter === "trash") {
      const restore = document.createElement("button");
      restore.type = "button";
      restore.className = "episode-item-action";
      restore.dataset.restore = episode.trashName;
      restore.dataset.episodeId = episode.id;
      restore.textContent = "恢复";
      row.append(restore);
    } else {
      const marker = document.createElement("span");
      marker.textContent = index === state.currentIndex ? "●" : "›";
      marker.setAttribute("aria-hidden", "true");
      row.append(marker);
    }

    li.append(row);
    dom.episodeList.append(li);
  });
  updateQueuePosition();
}

function setFactChip(element, className) {
  element.className = `fact-chip ${className}`;
}

function renderEpisodeHeader() {
  const episode = state.current;
  if (!episode) {
    dom.episodeOrdinal.textContent = "未选择 episode";
    dom.episodeTitle.textContent = "等待加载";
    dom.episodeStateBadge.textContent = "—";
    dom.episodeStateBadge.className = "state-badge is-neutral";
    dom.handPoseText.textContent = "hand_pose 未检查";
    dom.mediaText.textContent = "视频未准备";
    dom.warningBadge.classList.add("is-hidden");
    setFactChip(dom.handPoseBadge, "is-muted");
    setFactChip(dom.mediaBadge, "is-muted");
    return;
  }

  dom.episodeOrdinal.textContent = state.episodes.length
    ? `EPISODE ${state.currentIndex + 1} / ${state.episodes.length}`
    : "EPISODE";
  dom.episodeTitle.textContent = episode.id;
  dom.episodeTitle.title = episode.id;
  const meta = statusMeta(episode.status);
  dom.episodeStateBadge.textContent = meta.label;
  dom.episodeStateBadge.className = `state-badge ${meta.className}`;

  if (episode.handPoseExists) {
    dom.handPoseText.textContent = episode.handPoseRows === null
      ? "hand_pose 已生成"
      : `hand_pose · ${episode.handPoseRows.toLocaleString()} 行`;
    setFactChip(dom.handPoseBadge, "is-good");
  } else {
    dom.handPoseText.textContent = "hand_pose 缺失";
    setFactChip(dom.handPoseBadge, "is-error");
  }

  const mediaStatus = episode.mediaStatus;
  if (["ready", "complete", "completed", "ok", "unknown"].includes(mediaStatus)) {
    dom.mediaText.textContent = mediaStatus === "unknown" ? "正在加载视频" : "三路视频已就绪";
    setFactChip(dom.mediaBadge, mediaStatus === "unknown" ? "is-muted" : "is-good");
  } else if (["building", "preparing", "queued", "processing", "pending", "partial"].includes(mediaStatus)) {
    const percent = Math.max(0, Math.min(100, episode.mediaProgress));
    dom.mediaText.textContent = mediaStatus === "pending"
      ? "首次打开将生成预览"
      : `视频准备中${percent ? ` · ${Math.round(percent)}%` : ""}`;
    setFactChip(dom.mediaBadge, "is-warning");
  } else {
    dom.mediaText.textContent = episode.mediaError || "视频准备失败";
    setFactChip(dom.mediaBadge, "is-error");
  }

  const warningParts = [episode.warning, ...(episode.problemCodes || [])].filter(Boolean);
  if (warningParts.length) {
    const warningText = warningParts.join(" · ");
    dom.warningText.textContent = warningText;
    dom.warningBadge.title = warningText;
    dom.warningBadge.classList.remove("is-hidden");
  } else {
    dom.warningBadge.classList.add("is-hidden");
    dom.warningBadge.removeAttribute("title");
  }
}

function decisionAllowed() {
  if (!state.current || state.busy || state.filter === "trash") return false;
  return true;
}

function updateDecisionButtons() {
  const enabled = decisionAllowed();
  dom.keepButton.disabled = !enabled;
  dom.skipButton.disabled = !enabled;
  dom.deleteButton.disabled = !enabled;
}

function setBusy(busy) {
  state.busy = busy;
  dom.refreshButton.disabled = busy;
  updateDecisionButtons();
}

function setStreamOverlay(descriptor, mode, message) {
  const { overlay } = descriptor;
  overlay.className = "stream-overlay";
  overlay.replaceChildren();
  if (mode === "ready") {
    overlay.classList.add("is-ready");
    return;
  }
  if (mode === "error") overlay.classList.add("is-error");
  if (mode === "loading") {
    const spinner = document.createElement("span");
    spinner.className = "spinner";
    overlay.append(spinner);
  }
  const text = document.createElement("span");
  text.textContent = message;
  overlay.append(text);
}

function stopMediaPoll() {
  if (state.mediaPollTimer) window.clearTimeout(state.mediaPollTimer);
  state.mediaPollTimer = null;
}

function stopPlayback() {
  videos.forEach((video) => video.pause());
  if (state.syncFrame) cancelAnimationFrame(state.syncFrame);
  state.syncFrame = null;
  dom.playButton.textContent = "▶";
  dom.playButton.setAttribute("aria-label", "播放");
}

function clearVideos() {
  stopPlayback();
  stopMediaPoll();
  streamDescriptors.forEach((descriptor) => {
    descriptor.video.removeAttribute("src");
    descriptor.video.load();
    setStreamOverlay(descriptor, "loading", "等待视频");
  });
  dom.timeline.value = "0";
  dom.timeline.max = "1";
  dom.timeline.disabled = true;
  dom.currentTimeLabel.textContent = "00:00.000";
  dom.durationLabel.textContent = "00:00.000";
}

function mediaUrl(episodeId, role) {
  const url = new URL(`${API_BASE}${ENDPOINTS.media(episodeId, role)}`, window.location.origin);
  const token = localStorage.getItem(TOKEN_STORAGE_KEY);
  if (token) url.searchParams.set("token", token);
  return url.toString();
}

function loadVideos(episode) {
  clearVideos();
  dom.viewerPanel.classList.remove("is-empty");

  if (["error", "failed", "missing"].includes(episode.mediaStatus)) {
    streamDescriptors.forEach((descriptor) =>
      setStreamOverlay(descriptor, "error", episode.mediaError || "视频不可用"),
    );
    return;
  }

  streamDescriptors.forEach((descriptor) => {
    setStreamOverlay(descriptor, "loading", `${descriptor.label} 加载中`);
    descriptor.video.src = mediaUrl(episode.id, descriptor.role);
    descriptor.video.playbackRate = state.playbackRate;
    descriptor.video.load();
  });

  if (["building", "preparing", "queued", "processing", "pending", "partial"].includes(episode.mediaStatus)) {
    scheduleMediaPoll(episode.id);
  }
}

function scheduleMediaPoll(episodeId) {
  stopMediaPoll();
  state.mediaPollTimer = window.setTimeout(async () => {
    if (state.current?.id !== episodeId) return;
    try {
      const detail = normalizeEpisode(await api(ENDPOINTS.episode(episodeId)));
      if (state.current?.id !== episodeId) return;
      state.current = { ...state.current, ...detail };
      const listIndex = state.episodes.findIndex((item) => item.id === episodeId);
      if (listIndex >= 0) state.episodes[listIndex] = state.current;
      renderEpisodeHeader();
      if (["building", "preparing", "queued", "processing", "pending", "partial"].includes(detail.mediaStatus)) {
        scheduleMediaPoll(episodeId);
      } else if (["ready", "complete", "completed", "ok"].includes(detail.mediaStatus)) {
        const hasSources = streamDescriptors.every((descriptor) => descriptor.video.getAttribute("src"));
        if (!hasSources) loadVideos(state.current);
      } else if (["error", "failed", "missing"].includes(detail.mediaStatus)) {
        streamDescriptors.forEach((descriptor) =>
          setStreamOverlay(descriptor, "error", detail.mediaError || "视频准备失败"),
        );
      }
    } catch (error) {
      scheduleMediaPoll(episodeId);
    }
  }, 1600);
}

async function selectEpisodeByIndex(index, { fetchDetail = true } = {}) {
  if (index < 0 || index >= state.episodes.length) return;
  const requestId = ++state.detailRequestId;
  const base = state.episodes[index];
  state.currentIndex = index;
  state.current = base;
  renderQueue();
  renderEpisodeHeader();
  updateDecisionButtons();
  clearVideos();
  dom.viewerPanel.classList.remove("is-empty");
  streamDescriptors.forEach((descriptor) =>
    setStreamOverlay(descriptor, "loading", `${descriptor.label} 信息加载中`),
  );

  if (!fetchDetail || state.filter === "trash") {
    loadVideos(base);
    return;
  }
  try {
    const payload = await api(ENDPOINTS.episode(base.id));
    if (requestId !== state.detailRequestId || state.current?.id !== base.id) return;
    const detail = normalizeEpisode(payload);
    state.current = { ...base, ...detail };
    state.episodes[index] = state.current;
    renderQueue();
    renderEpisodeHeader();
    loadVideos(state.current);
  } catch (error) {
    if (requestId !== state.detailRequestId) return;
    showToast(`读取 episode 详情失败：${error.message}`, "error");
    loadVideos(base);
  }
}

function selectAdjacent(delta) {
  if (!state.episodes.length) return;
  const next = Math.max(0, Math.min(state.episodes.length - 1, state.currentIndex + delta));
  if (next !== state.currentIndex) selectEpisodeByIndex(next);
}

async function loadStats() {
  try {
    state.stats = normalizeStats(await api(ENDPOINTS.stats()));
    updateStats();
    setConnection("online", "服务已连接");
  } catch (error) {
    setConnection("offline", "连接失败");
    throw error;
  }
}

async function loadConfig() {
  try {
    const config = (await api(ENDPOINTS.config())) || {};
    const datasetName = firstDefined(
      config.dataset_name,
      config.dataset?.name,
      config.root?.split?.("/").filter(Boolean).pop(),
      config.dataset_root?.split?.("/").filter(Boolean).pop(),
      "task_v1_new",
    );
    dom.datasetLabel.textContent = `${datasetName} · 三路同步视频`;
    state.previewFps = asNumber(config.preview_fps, state.previewFps);
  } catch (error) {
    // Config is informative only; older backends can omit this endpoint.
  }
}

async function loadEpisodeList({ preserveId = state.current?.id } = {}) {
  const requestId = ++state.listRequestId;
  dom.queueLoading.classList.remove("is-hidden");
  dom.queueEmpty.classList.add("is-hidden");
  dom.episodeList.replaceChildren();
  const meta = FILTER_META[state.filter];
  const endpoint = state.filter === "trash"
    ? ENDPOINTS.trash({ query: state.query })
    : ENDPOINTS.episodes({ status: meta.backendStatus, query: state.query });

  try {
    const payload = await api(endpoint);
    if (requestId !== state.listRequestId) return;
    state.episodes = normalizeList(payload, state.filter === "trash" ? "trashed" : null);
    let index = preserveId
      ? state.episodes.findIndex((episode) => episode.id === preserveId)
      : -1;
    if (index < 0) index = state.episodes.length ? 0 : -1;
    state.currentIndex = index;
    state.current = index >= 0 ? state.episodes[index] : null;
    renderQueue();
    renderEpisodeHeader();
    updateDecisionButtons();
    if (index >= 0 && state.filter !== "trash") {
      await selectEpisodeByIndex(index);
    } else {
      clearVideos();
      dom.viewerPanel.classList.add("is-empty");
    }
    setConnection("online", "服务已连接");
  } catch (error) {
    if (requestId !== state.listRequestId) return;
    state.episodes = [];
    state.current = null;
    state.currentIndex = -1;
    renderQueue();
    clearVideos();
    dom.viewerPanel.classList.add("is-empty");
    setConnection("offline", "连接失败");
    showToast(`加载列表失败：${error.message}`, "error", 5000);
  }
}

async function refreshAll({ preserveId = state.current?.id, announce = false } = {}) {
  dom.refreshButton.classList.add("is-spinning");
  const results = await Promise.allSettled([
    loadConfig(),
    loadStats(),
    loadPrecache(),
    loadEpisodeList({ preserveId }),
  ]);
  dom.refreshButton.classList.remove("is-spinning");
  if (announce && results.every((result) => result.status === "fulfilled")) {
    showToast("数据已刷新", "success");
  }
}

function playbackDuration() {
  if (Number.isFinite(dom.headVideo.duration) && dom.headVideo.duration > 0) {
    return dom.headVideo.duration;
  }
  return state.current?.duration || 0;
}

function seekAll(targetTime) {
  const duration = playbackDuration();
  const value = Math.max(0, Math.min(duration || Infinity, Number(targetTime) || 0));
  videos.forEach((video) => {
    if (video.readyState >= HTMLMediaElement.HAVE_METADATA && Number.isFinite(video.duration)) {
      const followerEnd = Math.max(0, video.duration - 0.001);
      video.currentTime = Math.min(value, followerEnd);
    }
  });
  updateTimeline(value);
}

function updateTimeline(forcedTime) {
  const current = Number.isFinite(forcedTime) ? forcedTime : dom.headVideo.currentTime || 0;
  const duration = playbackDuration();
  dom.timeline.max = String(duration || 1);
  if (!dom.timeline.matches(":active")) dom.timeline.value = String(current);
  dom.timeline.disabled = !(duration > 0);
  dom.currentTimeLabel.textContent = formatTime(current);
  dom.durationLabel.textContent = formatTime(duration);
}

function syncFollowers() {
  if (dom.headVideo.paused || dom.headVideo.ended) {
    state.syncFrame = null;
    return;
  }
  const masterTime = dom.headVideo.currentTime;
  followers.forEach((video) => {
    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    const drift = video.currentTime - masterTime;
    if (Math.abs(drift) > 0.08) {
      video.currentTime = Math.min(masterTime, Math.max(0, video.duration - 0.001));
    }
    video.playbackRate = state.playbackRate;
    if (video.paused && !video.ended) video.play().catch(() => {});
  });
  updateTimeline(masterTime);
  state.syncFrame = requestAnimationFrame(syncFollowers);
}

async function playAll() {
  if (!state.current || dom.headVideo.readyState < HTMLMediaElement.HAVE_METADATA) {
    showToast("视频尚未准备完成", "info");
    return;
  }
  const start = dom.headVideo.currentTime;
  followers.forEach((video) => {
    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) video.currentTime = start;
  });
  videos.forEach((video) => { video.playbackRate = state.playbackRate; });
  const results = await Promise.allSettled(videos.map((video) => video.play()));
  if (results.every((result) => result.status === "rejected")) {
    showToast("浏览器阻止了视频播放", "error");
    return;
  }
  dom.playButton.textContent = "❚❚";
  dom.playButton.setAttribute("aria-label", "暂停");
  if (state.syncFrame) cancelAnimationFrame(state.syncFrame);
  state.syncFrame = requestAnimationFrame(syncFollowers);
}

function pauseAll() {
  stopPlayback();
  updateTimeline();
}

function togglePlayback() {
  if (dom.headVideo.paused) playAll();
  else pauseAll();
}

function stepFrame(direction) {
  pauseAll();
  const fps = Math.max(1, state.previewFps || state.current?.fps || 15);
  seekAll(dom.headVideo.currentTime + direction / fps);
}

function setFocus(target) {
  const classNames = ["focus-head", "focus-left", "focus-right"];
  dom.viewerPanel.classList.remove(...classNames);
  if (state.focus === target) {
    state.focus = "all";
  } else {
    state.focus = target;
    dom.viewerPanel.classList.add(`focus-${target}`);
  }
}

async function submitDecision(decision, extras = {}) {
  const episode = state.current;
  if (!episode || state.busy) return;
  const previousIndex = state.currentIndex;
  const preferredNextId =
    state.episodes[previousIndex + 1]?.id || state.episodes[previousIndex - 1]?.id || null;
  setBusy(true);
  pauseAll();
  try {
    await api(ENDPOINTS.decision(episode.id), {
      method: "POST",
      body: {
        action: decision === "delete" ? "quarantine" : decision,
        note: [extras.reason || "", extras.note || ""].filter(Boolean).join(": "),
      },
    });
    const labels = { keep: "已保留", skip: "已跳过", delete: "已移入回收站" };
    showToast(`${episode.id} ${labels[decision] || "已处理"}`, "success");
    state.current = null;
    await Promise.allSettled([
      loadStats(),
      loadEpisodeList({ preserveId: preferredNextId }),
    ]);
  } catch (error) {
    const conflict = error.status === 409 || error.status === 423;
    showToast(
      conflict ? `这条数据已被其他页面处理：${error.message}` : `操作失败：${error.message}`,
      "error",
      5000,
    );
    if (conflict) await refreshAll({ preserveId: episode.id });
  } finally {
    setBusy(false);
  }
}

function openDeleteDialog() {
  if (!decisionAllowed()) return;
  pauseAll();
  dom.deleteEpisodeId.textContent = state.current.id;
  dom.deleteReason.value = "camera_quality";
  dom.deleteNote.value = "";
  dom.deleteDialog.showModal();
  dom.confirmDeleteButton.focus();
}

async function confirmDelete() {
  if (!state.current) return;
  dom.confirmDeleteButton.disabled = true;
  dom.cancelDeleteButton.disabled = true;
  try {
    dom.deleteDialog.close();
    await submitDecision("delete", {
      reason: dom.deleteReason.value,
      note: dom.deleteNote.value.trim(),
    });
  } finally {
    dom.confirmDeleteButton.disabled = false;
    dom.cancelDeleteButton.disabled = false;
  }
}

async function restoreTrashItem(trashName, episodeId) {
  if (state.busy) return;
  setBusy(true);
  try {
    await api(ENDPOINTS.restore(trashName), {
      method: "POST",
    });
    showToast(`${episodeId} 已恢复到数据集`, "success");
    await Promise.allSettled([loadStats(), loadEpisodeList({ preserveId: null })]);
  } catch (error) {
    showToast(`恢复失败：${error.message}`, "error", 5000);
  } finally {
    setBusy(false);
  }
}

function isTypingTarget(target) {
  return target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement ||
    target?.isContentEditable;
}

function bindEvents() {
  dom.refreshButton.addEventListener("click", () => refreshAll({ announce: true }));
  dom.precacheButton.addEventListener("click", startPrecache);
  dom.previousEpisodeButton.addEventListener("click", () => selectAdjacent(-1));
  dom.nextEpisodeButton.addEventListener("click", () => selectAdjacent(1));
  dom.queuePreviousButton.addEventListener("click", () => selectAdjacent(-1));
  dom.queueNextButton.addEventListener("click", () => selectAdjacent(1));
  dom.playButton.addEventListener("click", togglePlayback);
  dom.backOneSecondButton.addEventListener("click", () => seekAll(dom.headVideo.currentTime - 1));
  dom.forwardOneSecondButton.addEventListener("click", () => seekAll(dom.headVideo.currentTime + 1));
  dom.previousFrameButton.addEventListener("click", () => stepFrame(-1));
  dom.nextFrameButton.addEventListener("click", () => stepFrame(1));
  dom.timeline.addEventListener("input", () => seekAll(Number(dom.timeline.value)));
  dom.speedSelect.addEventListener("change", () => {
    state.playbackRate = asNumber(dom.speedSelect.value, 1);
    videos.forEach((video) => { video.playbackRate = state.playbackRate; });
  });

  dom.keepButton.addEventListener("click", () => submitDecision("keep"));
  dom.skipButton.addEventListener("click", () => submitDecision("skip"));
  dom.deleteButton.addEventListener("click", openDeleteDialog);
  dom.cancelDeleteButton.addEventListener("click", () => dom.deleteDialog.close());
  dom.cancelDeleteButton.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      dom.deleteDialog.close();
    }
  });
  dom.confirmDeleteButton.addEventListener("click", confirmDelete);
  dom.deleteDialog.addEventListener("click", (event) => {
    if (event.target === dom.deleteDialog) dom.deleteDialog.close();
  });

  dom.filterTabs.forEach((tab) => {
    tab.addEventListener("click", async () => {
      const filter = tab.dataset.filter;
      if (filter === state.filter) return;
      state.filter = filter;
      state.current = null;
      dom.filterTabs.forEach((item) => {
        const active = item === tab;
        item.classList.toggle("is-active", active);
        item.setAttribute("aria-selected", String(active));
      });
      await loadEpisodeList({ preserveId: null });
    });
  });

  let searchTimer = null;
  dom.searchInput.addEventListener("input", () => {
    window.clearTimeout(searchTimer);
    searchTimer = window.setTimeout(() => {
      state.query = dom.searchInput.value.trim();
      loadEpisodeList({ preserveId: null });
    }, 280);
  });
  dom.clearSearchButton.addEventListener("click", () => {
    dom.searchInput.value = "";
    state.query = "";
    loadEpisodeList({ preserveId: null });
    dom.searchInput.focus();
  });

  dom.episodeList.addEventListener("click", (event) => {
    const restore = event.target.closest("[data-restore]");
    if (restore) {
      event.preventDefault();
      event.stopPropagation();
      restoreTrashItem(restore.dataset.restore, restore.dataset.episodeId);
      return;
    }
    const row = event.target.closest("[data-episode-id]");
    if (!row) return;
    const index = state.episodes.findIndex((item) => item.id === row.dataset.episodeId);
    if (index >= 0 && state.filter !== "trash") selectEpisodeByIndex(index);
  });

  dom.focusButtons.forEach((button) => {
    button.addEventListener("click", () => setFocus(button.dataset.focusVideo));
  });

  streamDescriptors.forEach((descriptor) => {
    descriptor.video.addEventListener("loadstart", () =>
      setStreamOverlay(descriptor, "loading", `${descriptor.label} 加载中`),
    );
    descriptor.video.addEventListener("canplay", () =>
      setStreamOverlay(descriptor, "ready", ""),
    );
    descriptor.video.addEventListener("error", () => {
      if (!descriptor.video.getAttribute("src")) return;
      setStreamOverlay(descriptor, "error", `${descriptor.label} 加载失败`);
    });
    descriptor.video.addEventListener("loadedmetadata", updateTimeline);
  });

  dom.headVideo.addEventListener("timeupdate", () => updateTimeline());
  dom.headVideo.addEventListener("ended", pauseAll);
  dom.headVideo.addEventListener("pause", () => {
    if (!dom.headVideo.ended) followers.forEach((video) => video.pause());
    dom.playButton.textContent = "▶";
    dom.playButton.setAttribute("aria-label", "播放");
  });
  dom.headVideo.addEventListener("seeking", () => {
    const time = dom.headVideo.currentTime;
    followers.forEach((video) => {
      if (video.readyState >= HTMLMediaElement.HAVE_METADATA) video.currentTime = time;
    });
  });

  document.addEventListener("keydown", (event) => {
    if (isTypingTarget(event.target)) return;
    if (dom.deleteDialog.open) {
      if (event.key === "Enter" && document.activeElement === dom.confirmDeleteButton) {
        event.preventDefault();
        confirmDelete();
      }
      return;
    }
    if (event.repeat && ["k", "s", "d"].includes(event.key.toLowerCase())) return;
    switch (event.key.toLowerCase()) {
      case " ":
        event.preventDefault();
        togglePlayback();
        break;
      case "arrowleft":
        event.preventDefault();
        selectAdjacent(-1);
        break;
      case "arrowright":
        event.preventDefault();
        selectAdjacent(1);
        break;
      case ",":
        event.preventDefault();
        stepFrame(-1);
        break;
      case ".":
        event.preventDefault();
        stepFrame(1);
        break;
      case "j":
        seekAll(dom.headVideo.currentTime - 1);
        break;
      case "l":
        seekAll(dom.headVideo.currentTime + 1);
        break;
      case "k":
        if (decisionAllowed()) submitDecision("keep");
        break;
      case "s":
        if (decisionAllowed()) submitDecision("skip");
        break;
      case "d":
        openDeleteDialog();
        break;
      case "1":
        setFocus("head");
        break;
      case "2":
        setFocus("left");
        break;
      case "3":
        setFocus("right");
        break;
      case "escape":
        state.focus = "all";
        dom.viewerPanel.classList.remove("focus-head", "focus-left", "focus-right");
        break;
      default:
        break;
    }
  });

  window.addEventListener("beforeunload", () => {
    stopPlayback();
    stopPrecachePoll();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) pauseAll();
  });
}

async function init() {
  captureTokenFromUrl();
  bindEvents();
  updateStats();
  renderEpisodeHeader();
  updateDecisionButtons();
  await refreshAll({ preserveId: null });
}

init().catch((error) => {
  setConnection("offline", "初始化失败");
  showToast(`初始化失败：${error.message}`, "error", 8000);
});
