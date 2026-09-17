#!/usr/bin/env python3
"""Small browser-based RGB review tool for UMI episode directories.

Raw E6 ``camera/e6_rgb.h265`` streams are transcoded to cached H.264 MP4
previews on demand. "Delete" is intentionally recoverable: an episode is
atomically moved into ``<dataset>/.review_trash`` and can be restored in the
same UI. The application never edits files inside an episode.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import threading
from typing import Any

try:
    from flask import Flask, Response, jsonify, request, send_file
except ImportError as error:  # pragma: no cover - environment guidance
    raise SystemExit(
        "Flask is required. Run with /home/dzq/openpi/.venv/bin/python."
    ) from error


DEFAULT_ROOT = Path("/mnt/data/dzq/umi/data/task_v1")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = SCRIPT_DIR / "cache"
DEFAULT_TOKEN_FILE = SCRIPT_DIR / ".review_token"
EPISODE_PATTERN = re.compile(r"^\d{8}_\d{6}_\d+_\d+$")
TRASH_PATTERN = re.compile(
    r"^(?P<episode>\d{8}_\d{6}_\d+_\d+)__(?P<stamp>\d{8}T\d{6}\.\d{6}Z)$"
)
VIEWS = {
    "full": "scale=960:360:flags=fast_bilinear",
    "left": "crop=1600:1200:0:0,scale=640:480:flags=fast_bilinear",
    "right": "crop=1600:1200:1600:0,scale=640:480:flags=fast_bilinear",
}
SOURCE_FPS_FALLBACK = 25.0
PREVIEW_FPS = 10
PREVIEW_CACHE_VERSION = 2


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>UMI RGB 数据筛选</title>
  <style>
    :root { color-scheme: dark; --bg:#111318; --panel:#1a1e25; --line:#303641;
      --text:#eef1f5; --muted:#9aa4b2; --blue:#4f91ff; --red:#eb5757; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,sans-serif; }
    header { height:58px; padding:10px 18px; border-bottom:1px solid var(--line);
      display:flex; align-items:center; gap:12px; background:#15181e; }
    header h1 { font-size:18px; margin:0; white-space:nowrap; }
    header .root { color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    main { height:calc(100vh - 58px); display:grid; grid-template-columns:320px 1fr; }
    aside { border-right:1px solid var(--line); display:flex; flex-direction:column; min-width:0; }
    .tabs,.search { display:flex; gap:8px; padding:10px; border-bottom:1px solid var(--line); }
    button,input { border:1px solid var(--line); border-radius:7px; background:#222731;
      color:var(--text); padding:8px 10px; }
    button { cursor:pointer; } button:hover { border-color:#566174; }
    button.active { background:#264a83; border-color:var(--blue); }
    input { width:100%; }
    #list { overflow:auto; flex:1; }
    .item { padding:10px 12px; border-bottom:1px solid #252a32; cursor:pointer;
      font-family:ui-monospace,monospace; font-size:12px; }
    .item:hover { background:#20252d; } .item.active { background:#21385b; color:#fff; }
    section { min-width:0; display:flex; flex-direction:column; }
    .toolbar { min-height:58px; padding:10px 14px; display:flex; gap:8px; align-items:center;
      border-bottom:1px solid var(--line); flex-wrap:wrap; }
    .episode { font-family:ui-monospace,monospace; margin-right:auto; }
    .danger { background:#662b30; border-color:#9f4149; }
    .restore { background:#24553c; border-color:#3c875e; }
    .viewer { flex:1; min-height:0; display:flex; align-items:center; justify-content:center;
      padding:14px; position:relative; }
    video { max-width:100%; max-height:100%; background:#000; border-radius:8px; box-shadow:0 8px 30px #0008; }
    #empty,#loading { color:var(--muted); font-size:16px; text-align:center; }
    #loading { position:absolute; background:#111c; padding:12px 16px; border-radius:8px; }
    .status { padding:8px 14px; border-top:1px solid var(--line); color:var(--muted); min-height:38px; }
    .status.error { color:#ff8585; }
    @media (max-width:800px) { main { grid-template-columns:1fr; }
      aside { height:35vh; border-right:0; border-bottom:1px solid var(--line); }
      section { height:65vh; } }
  </style>
</head>
<body>
<header><h1>UMI RGB 数据筛选</h1><span class="root" id="root"></span></header>
<main>
  <aside>
    <div class="tabs"><button id="activeTab" class="active">待筛选</button><button id="trashTab">回收站</button></div>
    <div class="search"><input id="search" placeholder="搜索 episode ID"></div>
    <div id="list"></div>
  </aside>
  <section>
    <div class="toolbar">
      <button id="prev">← 上一条</button><button id="next">下一条 →</button>
      <span class="episode" id="episode">请选择数据</span>
      <button class="view active" data-view="full">双目</button>
      <button class="view" data-view="left">左眼</button>
      <button class="view" data-view="right">右眼</button>
      <button id="gallery">首帧总览</button>
      <button id="delete" class="danger">删除到回收站</button>
      <button id="restore" class="restore" hidden>恢复</button>
    </div>
    <div class="viewer">
      <div id="empty">从左侧选择一条数据</div>
      <div id="loading" hidden>正在生成/加载 10 FPS H.264 预览…</div>
      <video id="video" controls autoplay muted loop playsinline hidden></video>
    </div>
    <div class="status" id="status">快捷键：←/→ 切换，空格播放/暂停，D 删除（仍需确认）。</div>
  </section>
</main>
<script>
const params = new URLSearchParams(location.search);
const token = params.get('token') || '';
let mode='active', items=[], filtered=[], details={}, current=null, view='full';
const $=id=>document.getElementById(id), video=$('video');
async function api(path, options={}) {
  options.headers={...(options.headers||{}),'X-Review-Token':token};
  const response=await fetch(path,options); const body=await response.json().catch(()=>({}));
  if(!response.ok) throw new Error(body.error||`HTTP ${response.status}`); return body;
}
function setStatus(text,error=false){ $('status').textContent=text; $('status').classList.toggle('error',error); }
function formatDuration(seconds){if(!Number.isFinite(seconds))return '时长未知';const total=Math.max(0,seconds);const minutes=Math.floor(total/60);const rest=(total-minutes*60).toFixed(1).padStart(4,'0');return `${String(minutes).padStart(2,'0')}:${rest}`}
async function loadItems(keep=true){
  const data=await api(mode==='active'?'/api/episodes':'/api/trash');
  $('root').textContent=`${data.root} · ${data.items.length} 条`;
  items=data.items; details=data.details||{}; applyFilter();
  if(keep && current && filtered.includes(current)) select(current); else if(filtered.length) select(filtered[0]); else clearViewer();
}
function applyFilter(){
  const q=$('search').value.trim().toLowerCase(); filtered=items.filter(x=>x.toLowerCase().includes(q));
  $('list').innerHTML=''; filtered.forEach(id=>{ const d=document.createElement('div'); d.className='item'+(id===current?' active':'');
    d.textContent=id; d.onclick=()=>select(id); $('list').appendChild(d); });
}
function clearViewer(){ current=null; video.pause(); video.removeAttribute('src'); video.hidden=true; $('empty').hidden=false;
  $('episode').textContent='没有数据'; $('delete').hidden=mode!=='active'; $('restore').hidden=mode!=='trash'; }
function select(id){
  current=id; localStorage.setItem(`umi-rgb-${mode}`,id); applyFilter(); const timing=details[id]; $('episode').textContent=timing?`${id} · ${formatDuration(timing.duration_seconds)}`:id;
  $('empty').hidden=true; $('delete').hidden=mode!=='active'; $('restore').hidden=mode!=='trash';
  if(mode==='trash'){ video.pause(); video.hidden=true; $('empty').hidden=false; $('empty').textContent='回收站数据不生成预览；可直接恢复。'; return; }
  $('loading').hidden=false; video.hidden=false;
  video.src=`/preview/${encodeURIComponent(id)}?view=${view}&token=${encodeURIComponent(token)}&v=1`;
  video.load(); video.play().catch(()=>{});
}
video.oncanplay=()=>{$('loading').hidden=true;const timing=details[current];setStatus(`${current} · ${view} · ${timing?'采集时长 '+formatDuration(timing.duration_seconds)+' · ':''}可播放`);};
video.onerror=()=>{$('loading').hidden=true; setStatus('预览生成或播放失败，请查看服务器日志。',true);};
function move(delta){ if(!filtered.length)return; let i=Math.max(0,filtered.indexOf(current)); i=Math.min(filtered.length-1,i+delta); select(filtered[i]); }
async function removeCurrent(){
  if(!current||mode!=='active')return;
  if(!confirm(`将 ${current} 移入 .review_trash？\n之后可以在“回收站”中恢复。`))return;
  const id=current; try{ const r=await api(`/api/delete/${encodeURIComponent(id)}`,{method:'POST'});
    setStatus(`${id} 已移动到回收站：${r.trash_name}`); current=null; await loadItems(false); }catch(e){setStatus(e.message,true);}
}
async function restoreCurrent(){
  if(!current||mode!=='trash')return; const id=current;
  try{ const r=await api(`/api/restore/${encodeURIComponent(id)}`,{method:'POST'});
    setStatus(`${r.episode_id} 已恢复`); current=null; await loadItems(false); }catch(e){setStatus(e.message,true);}
}
$('prev').onclick=()=>move(-1); $('next').onclick=()=>move(1); $('delete').onclick=removeCurrent; $('restore').onclick=restoreCurrent;
$('gallery').onclick=()=>{location.href=`/gallery?token=${encodeURIComponent(token)}`;};
$('search').oninput=applyFilter;
document.querySelectorAll('.view').forEach(b=>b.onclick=()=>{document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));
  b.classList.add('active'); view=b.dataset.view; if(current&&mode==='active')select(current);});
async function setMode(value){ mode=value; current=null; $('activeTab').classList.toggle('active',mode==='active');
  $('trashTab').classList.toggle('active',mode==='trash'); $('search').value=''; await loadItems(false);
  const saved=localStorage.getItem(`umi-rgb-${mode}`); if(saved&&filtered.includes(saved))select(saved); }
$('activeTab').onclick=()=>setMode('active'); $('trashTab').onclick=()=>setMode('trash');
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;
  if(e.key==='ArrowLeft')move(-1); else if(e.key==='ArrowRight')move(1);
  else if(e.key===' '){e.preventDefault(); video.paused?video.play():video.pause();}
  else if((e.key==='d'||e.key==='D')&&mode==='active')removeCurrent();});
loadItems(false).then(()=>{const requested=params.get('episode'); const saved=localStorage.getItem('umi-rgb-active');
  if(requested&&filtered.includes(requested))select(requested); else if(saved&&filtered.includes(saved))select(saved);
}).catch(e=>setStatus(e.message,true));
</script>
</body></html>"""


GALLERY_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>UMI RGB 首帧总览</title>
  <style>
    :root{color-scheme:dark;--bg:#111318;--panel:#1a1e25;--line:#343b47;--text:#eef1f5;--muted:#a1abba;--blue:#4f91ff;--red:#e4545e}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,sans-serif}
    header{position:sticky;top:0;z-index:5;background:#15181ef5;border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:9px;align-items:center;flex-wrap:wrap;backdrop-filter:blur(8px)}
    h1{font-size:18px;margin:0 10px 0 0}button,input{border:1px solid var(--line);border-radius:7px;background:#242a34;color:var(--text);padding:8px 10px}button{cursor:pointer}button:hover{border-color:#637087}.danger{background:#682c32;border-color:#a4424b}.active{background:#28518e;border-color:var(--blue)}
    #search{min-width:240px}.muted{color:var(--muted)}#status{margin-left:auto}.notice{padding:10px 16px;color:#e8c76a;border-bottom:1px solid #574d2a;background:#2b2718}
    #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;padding:14px}
    .card{background:var(--panel);border:1px solid var(--line);border-radius:9px;overflow:hidden;position:relative}.card.selected{border-color:var(--blue);box-shadow:0 0 0 2px #4f91ff44}.card img{width:100%;aspect-ratio:4/3;object-fit:contain;background:#050608;display:block;cursor:pointer}.card .body{padding:8px}.id{font:11px/1.35 ui-monospace,monospace;word-break:break-all}.meta{color:#8fc3ff;font:12px/1.4 ui-monospace,monospace;margin-top:5px}.row{display:flex;gap:7px;margin-top:8px;align-items:center}.row button{padding:5px 8px;font-size:12px}.pick{width:18px;height:18px}.error img{display:none}.error:before{content:'缩略图生成失败';display:flex;height:180px;align-items:center;justify-content:center;color:#ff8585}
  </style>
</head>
<body>
<header><h1>RGB 首帧总览</h1><button id="back">返回视频查看</button><button id="showAll" hidden>显示全部</button><input id="search" placeholder="搜索 episode ID"><button class="eye active" data-view="right">右眼</button><button class="eye" data-view="left">左眼</button><button class="eye" data-view="full">双目</button><button id="selectVisible">选中当前筛选</button><button id="clear">清空选择</button><button id="delete" class="danger">删除所选到回收站</button><span id="status" class="muted">加载中…</span></header>
<div class="notice">首帧仅用于粗筛黑屏、曝光、视角和场景问题；动作质量请点“查看视频”确认。删除会移入 <code>.review_trash</code>，可恢复。
</div>
<main id="grid"></main>
<script>
const params=new URLSearchParams(location.search), token=params.get('token')||'';
const requestedIds=new Set((params.get('ids')||'').split(',').map(x=>x.trim()).filter(Boolean));
let candidateOnly=requestedIds.size>0,items=[],details={},view='right',selected=new Set(); const grid=document.getElementById('grid'), search=document.getElementById('search'), status=document.getElementById('status');
async function api(path,options={}){options.headers={...(options.headers||{}),'X-Review-Token':token};const r=await fetch(path,options);const b=await r.json().catch(()=>({}));if(!r.ok)throw new Error(b.error||`HTTP ${r.status}`);return b}
function visible(){const q=search.value.trim().toLowerCase();return items.filter(x=>(!candidateOnly||requestedIds.has(x))&&x.toLowerCase().includes(q))}
function updateStatus(){status.textContent=`共 ${items.length} 条 · ${candidateOnly?'复查候选':'当前'} ${visible().length} 条 · 已选 ${selected.size} 条`}
function formatDuration(seconds){if(!Number.isFinite(seconds))return '时长未知';const total=Math.max(0,seconds);const minutes=Math.floor(total/60);const rest=(total-minutes*60).toFixed(1).padStart(4,'0');return `${String(minutes).padStart(2,'0')}:${rest}`}
function thumb(id){return `/thumbnail/${encodeURIComponent(id)}?view=${view}&token=${encodeURIComponent(token)}&v=1`}
function render(){grid.innerHTML='';for(const id of visible()){const card=document.createElement('article');card.className='card'+(selected.has(id)?' selected':'');card.dataset.id=id;
  const img=document.createElement('img');img.loading='lazy';img.src=thumb(id);img.alt=id;img.onerror=()=>card.classList.add('error');img.onclick=()=>toggle(id);
  const body=document.createElement('div');body.className='body';const label=document.createElement('div');label.className='id';label.textContent=id;
  const meta=document.createElement('div');meta.className='meta';const timing=details[id];meta.textContent=timing?`采集时长 ${formatDuration(timing.duration_seconds)} · ${timing.frame_count} 帧 · ${timing.capture_fps.toFixed(1)} FPS`:'采集时长未知';
  const row=document.createElement('div');row.className='row';const box=document.createElement('input');box.type='checkbox';box.className='pick';box.checked=selected.has(id);box.onchange=()=>toggle(id);
  const play=document.createElement('button');play.textContent='查看视频';play.onclick=()=>location.href=`/?token=${encodeURIComponent(token)}&episode=${encodeURIComponent(id)}`;
  const del=document.createElement('button');del.textContent='删除';del.className='danger';del.onclick=()=>remove([id]);row.append(box,play,del);body.append(label,meta,row);card.append(img,body);grid.append(card)}updateStatus()}
function toggle(id){selected.has(id)?selected.delete(id):selected.add(id);render()}
async function remove(ids){if(!ids.length)return;if(!confirm(`将所选 ${ids.length} 条 episode 移入回收站？`))return;status.textContent=`正在删除 0/${ids.length}…`;let done=0;for(const id of ids){try{await api(`/api/delete/${encodeURIComponent(id)}`,{method:'POST'});items=items.filter(x=>x!==id);selected.delete(id);done++;status.textContent=`正在删除 ${done}/${ids.length}…`}catch(e){alert(`${id}: ${e.message}`);break}}render()}
search.oninput=render;document.getElementById('back').onclick=()=>location.href=`/?token=${encodeURIComponent(token)}`;
document.getElementById('showAll').onclick=()=>{candidateOnly=false;document.getElementById('showAll').hidden=true;selected.clear();render()};
document.getElementById('selectVisible').onclick=()=>{visible().forEach(x=>selected.add(x));render()};document.getElementById('clear').onclick=()=>{selected.clear();render()};document.getElementById('delete').onclick=()=>remove([...selected]);
document.querySelectorAll('.eye').forEach(b=>b.onclick=()=>{document.querySelectorAll('.eye').forEach(x=>x.classList.remove('active'));b.classList.add('active');view=b.dataset.view;render()});
api('/api/episodes').then(d=>{items=d.items;details=d.details||{};if(candidateOnly){selected=new Set([...requestedIds].filter(x=>items.includes(x)));document.getElementById('showAll').hidden=false}render()}).catch(e=>status.textContent=e.message);
</script>
</body></html>"""


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def load_or_create_token(path: Path) -> str:
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if len(token) >= 24:
            return token
        raise ValueError(f"invalid token file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(24)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(token + "\n")
    return token


def episode_dirs(root: Path) -> list[str]:
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and not path.is_symlink() and EPISODE_PATTERN.fullmatch(path.name)
    )


def trash_dirs(trash: Path) -> list[str]:
    if not trash.exists():
        return []
    return sorted(
        (path.name for path in trash.iterdir() if path.is_dir() and not path.is_symlink()),
        reverse=True,
    )


def read_capture_timing(episode: Path) -> dict[str, float | int] | None:
    """Return timestamp-derived video timing without decoding the H.265 stream."""
    metadata = episode / "camera" / "e6_rgb_stream_metainfo.csv"
    if not metadata.is_file():
        return None

    first_ns: int | None = None
    last_ns: int | None = None
    frame_count = 0
    with metadata.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            raw_timestamp = (
                row.get("e6_mid_exposure_realtime_ns")
                or row.get("e6_mid_exposure_boot_ns")
            )
            if not raw_timestamp:
                continue
            try:
                timestamp_ns = int(raw_timestamp)
            except ValueError:
                continue
            if first_ns is None:
                first_ns = timestamp_ns
            last_ns = timestamp_ns
            frame_count += 1

    if first_ns is None or last_ns is None or frame_count == 0:
        return None
    if frame_count == 1 or last_ns <= first_ns:
        return {"duration_seconds": 0.0, "frame_count": frame_count, "capture_fps": 0.0}

    timestamp_span = (last_ns - first_ns) / 1_000_000_000
    frame_interval = timestamp_span / (frame_count - 1)
    return {
        "duration_seconds": round(timestamp_span + frame_interval, 3),
        "frame_count": frame_count,
        "capture_fps": round((frame_count - 1) / timestamp_span, 3),
    }


def validate_episode(root: Path, episode_id: str) -> Path:
    if not EPISODE_PATTERN.fullmatch(episode_id):
        raise ValueError("invalid episode ID")
    candidate = root / episode_id
    if not candidate.is_dir() or candidate.is_symlink():
        raise FileNotFoundError(episode_id)
    return candidate


def append_action_log(trash: Path, record: dict[str, Any]) -> None:
    trash.mkdir(parents=True, exist_ok=True)
    path = trash / "review_actions.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def transcode_preview(source: Path, target: Path, view: str, source_fps: float) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{secrets.token_hex(4)}.tmp.mp4")
    video_filter = (
        f"setpts=N/({source_fps:.9f}*TB),fps={PREVIEW_FPS},{VIEWS[view]}"
    )

    def command(hardware_decode: bool) -> list[str]:
        decoder = ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"] if hardware_decode else []
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *decoder,
            "-f", "hevc", "-i", str(source), "-an", "-vf", video_filter,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "29",
            "-pix_fmt", "yuv420p", "-r", str(PREVIEW_FPS),
            "-movflags", "+faststart", str(temporary),
        ]

    try:
        completed = subprocess.run(command(True), text=True, capture_output=True)
        if completed.returncode:
            temporary.unlink(missing_ok=True)
            completed = subprocess.run(command(False), text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or "ffmpeg failed")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("ffmpeg created an empty preview")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def extract_thumbnail(source: Path, target: Path, view: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{secrets.token_hex(4)}.tmp.jpg")

    def command(hardware_decode: bool) -> list[str]:
        decoder = ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"] if hardware_decode else []
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *decoder,
            "-f", "hevc", "-i", str(source), "-frames:v", "1",
            "-vf", VIEWS[view], "-q:v", "3", str(temporary),
        ]

    try:
        completed = subprocess.run(command(True), text=True, capture_output=True)
        if completed.returncode:
            temporary.unlink(missing_ok=True)
            completed = subprocess.run(command(False), text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or "ffmpeg thumbnail failed")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("ffmpeg created an empty thumbnail")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def create_app(root: Path, cache: Path, token: str) -> Flask:
    root = root.resolve()
    cache = cache.resolve()
    trash = root / ".review_trash"
    app = Flask(__name__)
    move_lock = threading.Lock()
    transcode_slots = threading.Semaphore(4)
    preview_locks: dict[tuple[str, str], threading.Lock] = {}
    preview_locks_guard = threading.Lock()
    timing_cache: dict[str, tuple[tuple[int, int], dict[str, float | int] | None]] = {}
    timing_cache_lock = threading.Lock()

    def capture_timing(episode_id: str) -> dict[str, float | int] | None:
        episode = root / episode_id
        metadata = episode / "camera" / "e6_rgb_stream_metainfo.csv"
        try:
            stat = metadata.stat()
        except FileNotFoundError:
            return None
        signature = (stat.st_mtime_ns, stat.st_size)
        with timing_cache_lock:
            cached = timing_cache.get(episode_id)
            if cached and cached[0] == signature:
                return cached[1]
        result = read_capture_timing(episode)
        with timing_cache_lock:
            timing_cache[episode_id] = (signature, result)
        return result

    def authorized() -> bool:
        supplied = request.headers.get("X-Review-Token") or request.args.get("token")
        return bool(supplied) and secrets.compare_digest(supplied, token)

    @app.before_request
    def require_token() -> Response | None:
        if request.path == "/health":
            return None
        if not authorized():
            return jsonify(error="missing or invalid review token"), 401
        return None

    @app.errorhandler(ValueError)
    def bad_request(error: ValueError) -> tuple[Response, int]:
        return jsonify(error=str(error)), 400

    @app.errorhandler(FileNotFoundError)
    def not_found(error: FileNotFoundError) -> tuple[Response, int]:
        return jsonify(error=f"not found: {error}"), 404

    @app.get("/")
    def index() -> Response:
        return Response(HTML, mimetype="text/html")

    @app.get("/gallery")
    def gallery() -> Response:
        return Response(GALLERY_HTML, mimetype="text/html")

    @app.get("/health")
    def health() -> Response:
        return jsonify(status="ok", root=str(root), episodes=len(episode_dirs(root)))

    @app.get("/api/episodes")
    def list_episodes() -> Response:
        items = episode_dirs(root)
        details = {
            episode_id: timing
            for episode_id in items
            if (timing := capture_timing(episode_id)) is not None
        }
        return jsonify(root=str(root), items=items, details=details)

    @app.get("/api/trash")
    def list_trash() -> Response:
        return jsonify(root=str(trash), items=trash_dirs(trash))

    @app.get("/preview/<episode_id>")
    def preview(episode_id: str) -> Response:
        view = request.args.get("view", "full")
        if view not in VIEWS:
            raise ValueError(f"unknown view: {view}")
        episode = validate_episode(root, episode_id)
        source = episode / "camera" / "e6_rgb.h265"
        if not source.is_file():
            raise FileNotFoundError(source)
        timing = capture_timing(episode_id)
        source_fps = float(timing["capture_fps"]) if timing else SOURCE_FPS_FALLBACK
        if source_fps <= 0:
            source_fps = SOURCE_FPS_FALLBACK
        target = cache / f"{episode_id}.{view}.v{PREVIEW_CACHE_VERSION}.mp4"
        key = (episode_id, view)
        with preview_locks_guard:
            lock = preview_locks.setdefault(key, threading.Lock())
        with lock:
            if not target.is_file() or target.stat().st_mtime_ns < source.stat().st_mtime_ns:
                with transcode_slots:
                    transcode_preview(source, target, view, source_fps)
        return send_file(target, mimetype="video/mp4", conditional=True)

    @app.get("/thumbnail/<episode_id>")
    def thumbnail(episode_id: str) -> Response:
        view = request.args.get("view", "right")
        if view not in VIEWS:
            raise ValueError(f"unknown view: {view}")
        episode = validate_episode(root, episode_id)
        source = episode / "camera" / "e6_rgb.h265"
        if not source.is_file():
            raise FileNotFoundError(source)
        target = cache / f"{episode_id}.first.{view}.jpg"
        key = (episode_id, f"first-{view}")
        with preview_locks_guard:
            lock = preview_locks.setdefault(key, threading.Lock())
        with lock:
            if not target.is_file() or target.stat().st_mtime_ns < source.stat().st_mtime_ns:
                with transcode_slots:
                    extract_thumbnail(source, target, view)
        return send_file(target, mimetype="image/jpeg", conditional=True)

    @app.post("/api/delete/<episode_id>")
    def delete_episode(episode_id: str) -> Response:
        with move_lock:
            source = validate_episode(root, episode_id)
            trash.mkdir(parents=True, exist_ok=True)
            trash_name = f"{episode_id}__{utc_stamp()}"
            target = trash / trash_name
            if target.exists():
                raise RuntimeError(f"trash destination already exists: {target}")
            os.replace(source, target)
            for view in VIEWS:
                (cache / f"{episode_id}.{view}.mp4").unlink(missing_ok=True)
                (cache / f"{episode_id}.{view}.v{PREVIEW_CACHE_VERSION}.mp4").unlink(missing_ok=True)
                (cache / f"{episode_id}.first.{view}.jpg").unlink(missing_ok=True)
            append_action_log(trash, {
                "action": "trash", "episode_id": episode_id,
                "trash_name": trash_name, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })
        return jsonify(status="trashed", episode_id=episode_id, trash_name=trash_name)

    @app.post("/api/restore/<trash_name>")
    def restore_episode(trash_name: str) -> Response:
        match = TRASH_PATTERN.fullmatch(trash_name)
        if not match:
            raise ValueError("invalid trash entry")
        with move_lock:
            source = trash / trash_name
            if not source.is_dir() or source.is_symlink():
                raise FileNotFoundError(trash_name)
            episode_id = match.group("episode")
            target = root / episode_id
            if target.exists():
                raise ValueError(f"cannot restore; episode already exists: {episode_id}")
            os.replace(source, target)
            append_action_log(trash, {
                "action": "restore", "episode_id": episode_id,
                "trash_name": trash_name, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })
        return jsonify(status="restored", episode_id=episode_id)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"dataset root does not exist: {args.root}")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    return args


def main() -> int:
    args = parse_args()
    token = load_or_create_token(args.token_file)
    app = create_app(args.root, args.cache, token)
    print(f"dataset: {args.root.resolve()}", flush=True)
    print(f"open: http://{args.host}:{args.port}/?token={token}", flush=True)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
