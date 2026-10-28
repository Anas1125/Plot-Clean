"""Plot Map Cleaner with review, manual editing, approval, and download."""
import io
import os
import json
import queue
import tempfile
import threading
import uuid
import zipfile
from datetime import datetime, timedelta, UTC

import numpy as np
from flask import Flask, abort, jsonify, render_template_string, request, send_file, url_for
from PIL import Image, ImageDraw
from cleaner import clean
import puzzle
import board

Image.MAX_IMAGE_PIXELS = None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

# In-memory job store for this local prototype. Jobs are lost when Flask restarts.
JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_TTL = timedelta(hours=3)
PREVIEW_MAX_SIDE = 1800
APPROVED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'approved_images')
os.makedirs(APPROVED_DIR, exist_ok=True)
PUZZLE_DIR = os.path.join(APPROVED_DIR, 'puzzle')
os.makedirs(PUZZLE_DIR, exist_ok=True)
# state of the (single) background puzzle build
PUZZLE = {"state": "idle", "stage": "", "fraction": 0.0, "message": "", "report": None, "error": None}
PUZZLE_LOCK = threading.Lock()
MANUAL_FILE = os.path.join(PUZZLE_DIR, 'manual_layout.json')

def _load_manual():
    try:
        with open(MANUAL_FILE, 'r', encoding='utf-8') as stream:
            data = json.load(stream)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}

def _save_manual(data):
    with open(MANUAL_FILE, 'w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2)

def _order_map():
    # 1 = first approved, 2 = second approved, ... (stable even when older sheets are deleted: ranks close up)
    records = sorted(_approved_records(), key=lambda r: (r.get('approved_at', ''), r.get('id', '')))
    return {r['id']: i + 1 for i, r in enumerate(records)}

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Plot Map Cleaner</title>
<style>
:root{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#202124;background:#f7f8fa}
*{box-sizing:border-box}body{max-width:1100px;margin:30px auto;padding:0 18px}
h1{font-size:28px;margin-bottom:8px}.muted{color:#62666d}
.panel{background:white;border:1px solid #e0e3e7;border-radius:14px;padding:22px;margin:18px 0}
.upload{text-align:center;border:2px dashed #aeb4bc}
button,.button{border:0;border-radius:9px;padding:11px 17px;font-size:15px;font-weight:600;cursor:pointer;background:#1769e0;color:white}
button:disabled{opacity:.55;cursor:wait}.secondary{background:#edf1f7;color:#202124}button.save-dirty{background:#e8710a;color:#fff}#save-state{font-weight:600;font-size:14px;align-self:center}.approve{background:#16804a}.download{background:#1769e0}
input[type=file]{max-width:100%}.options{display:flex;justify-content:center;gap:18px;flex-wrap:wrap;margin:18px 0}
.hidden{display:none!important}.toolbar{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin-bottom:14px}
.toolbar label{display:flex;gap:7px;align-items:center;background:#f1f3f5;padding:8px 10px;border-radius:8px}
.toolbar input[type=range]{width:110px}.toolbar button.active{outline:3px solid #9dc2ff}
.preview-wrap{background:#eceff3;border-radius:10px;min-height:220px;display:flex;align-items:center;justify-content:center;overflow:auto;padding:8px}
#editor{display:block;max-width:100%;height:auto;background:white;cursor:crosshair;touch-action:none}
.status{margin-top:12px;white-space:pre-wrap}
.topbar,.approved-heading{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap}
.topbar h1{margin-top:0}
.approved-list{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:14px;margin:18px 0}
.approved-card{border:1px solid #e0e3e7;border-radius:12px;padding:12px;min-width:0}
.approved-card img{width:100%;height:150px;object-fit:contain;background:#f0f2f5;border-radius:8px}
.approved-card h3{font-size:16px;overflow-wrap:anywhere;margin:10px 0 4px}
.approved-card .meta{font-size:12px;color:#666;overflow-wrap:anywhere}
.approved-card .actions{margin-top:10px}
.progress-shell{margin:18px auto 0;max-width:560px}
.progress-meta{display:flex;justify-content:space-between;gap:12px;font-size:13px;color:#555;margin-bottom:8px}
.progress-track{height:28px;padding:4px;border:2px solid #202124;background:#fff;display:grid;grid-template-columns:repeat(10,minmax(0,1fr));gap:4px;overflow:hidden}
.progress-track{display:block;position:relative;padding:4px}
.progress-segment{display:none}
.progress-fill{height:100%;width:0%;background:#111;transition:width .35s ease}
.queue-list{margin:16px auto 0;max-width:760px;display:grid;gap:8px}
.q-row{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(0,2fr) 52px auto;gap:10px;align-items:center;border:1px solid #e0e3e7;border-radius:10px;padding:8px 12px;font-size:13px}
.q-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:600}
.q-bar{height:10px;background:#eceff3;border-radius:6px;overflow:hidden}
.q-fill{height:100%;width:0%;background:#111;transition:width .35s ease}
.q-row.done .q-fill{background:#1a7f37}.q-row.error .q-fill{background:#c62828}
.q-row.cancelled{opacity:.6;background:#f3f4f6;border-style:dashed}.q-row.cancelled .q-fill{background:#9aa0a6}
button.danger{background:#fff;color:#b42318;border:1px solid #e0b4b0}button.danger:hover{background:#fdecea}
#orig-win{position:fixed;top:90px;right:24px;width:min(560px,62vw);height:min(480px,62vh);min-width:260px;min-height:180px;background:#fff;border:1px solid #80868b;border-radius:10px;box-shadow:0 12px 44px rgba(0,0,0,.35);z-index:2000;display:flex;flex-direction:column;resize:both;overflow:hidden}
#orig-head{display:flex;align-items:center;gap:6px;padding:6px 8px;background:#202124;color:#fff;cursor:move;user-select:none;-webkit-user-select:none;touch-action:none;font-size:13px}
#orig-head .title{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:600}
#orig-head button{padding:2px 9px;font-size:13px;background:#3c4043;color:#fff;border-radius:6px;cursor:pointer}
#orig-zoom-val{min-width:42px;text-align:center}
#orig-body{flex:1;overflow:auto;background:#eceff3}
#orig-body img{display:block;width:100%;max-width:none;background:#fff;user-select:none}
.batch-actions{display:flex;gap:10px;flex-wrap:wrap;align-items:center;justify-content:center;margin:14px auto 0;max-width:760px}.batch-actions .muted{font-size:13px}
.cancel-all-row{display:flex;justify-content:flex-end;margin:12px auto 0;max-width:760px}
.q-row.good{border-color:#1a7f37;background:#f1faf4}.q-row.good .q-fill{background:#1a7f37}.q-row.good .q-msg{color:#137333}
.q-row.warn{border-color:#e0a100;background:#fff8e1}.q-row.warn .q-fill{background:#e0a100}.q-row.warn .q-msg{color:#8a5a00;font-weight:600}
.q-row.bad,.q-row.error{border-color:#c62828;background:#fdecea;animation:q-blink 0.8s ease-in-out infinite alternate}
.q-row.bad .q-fill,.q-row.error .q-fill{background:#c62828}.q-row.bad .q-msg,.q-row.error .q-msg{color:#b42318;font-weight:700}
@keyframes q-blink{from{background:#fdecea;box-shadow:0 0 0 0 rgba(198,40,40,.0)}to{background:#ffb4ab;box-shadow:0 0 0 4px rgba(198,40,40,.45)}}
.q-pct{text-align:right;font-variant-numeric:tabular-nums}
.q-msg{grid-column:1 / -1;color:#666;font-size:12px;margin-top:-4px}
.q-row button{padding:4px 12px;font-size:12px}
.progress-segment.filled{background:#111}
.progress-segment.moving{animation:segment-pulse .75s ease-in-out infinite alternate}
@keyframes segment-pulse{from{opacity:.35}to{opacity:1}}
.progress-track.indeterminate .progress-segment{background:#fff}
.progress-track.indeterminate .progress-segment.moving{background:#111}
.progress-segment.filled{background:#111}
.progress-segment.moving{animation:segment-pulse .75s ease-in-out infinite alternate}
@keyframes segment-pulse{from{opacity:.35}to{opacity:1}}
.progress-track.indeterminate .progress-segment{background:#fff}
.progress-track.indeterminate .progress-segment.moving{background:#111}
.puzzle-box{border:1px solid #e0e3e7;border-radius:12px;padding:16px;margin:16px 0;background:#fafbfc}
.puzzle-bar{width:100%;height:14px;margin-top:10px}
.group-card{border:1px solid #e0e3e7;border-radius:12px;padding:14px;margin:14px 0;background:white}
.group-card img{max-width:100%;max-height:640px;display:block;margin:10px auto;background:#fff;border:1px solid #e0e3e7}
.map-tools{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0}
.map-tools button{padding:6px 12px}
.map-tools .zoomval{min-width:52px;text-align:center;color:#556}
.map-viewport{position:relative;height:78vh;min-height:420px;background:#eceff3;border:1px solid #d5d9df;border-radius:10px;overflow:hidden;cursor:grab;touch-action:none;user-select:none}
.map-viewport.dragging{cursor:grabbing}
.map-viewport:fullscreen{height:100vh;border-radius:0}
.map-stage{position:absolute;left:0;top:0;transform-origin:0 0;display:flex;gap:90px;align-items:flex-start;padding:30px;will-change:transform}
.map-stage figure{margin:0}
.map-stage figcaption{font:600 15px system-ui,sans-serif;color:#334;margin:0 0 8px}
#map-viewer .map-stage img{max-width:none;max-height:none;margin:0;border:1px solid #cfd4da;background:#fff;display:block;image-rendering:auto;-webkit-user-drag:none}
.link-table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}.link-table td,.link-table th{border-bottom:1px solid #eceff3;padding:4px 6px;text-align:left}
.badge{display:inline-block;border-radius:999px;padding:1px 8px;font-size:12px;font-weight:600}.badge.high{background:#e3f4ea;color:#137333}.badge.medium{background:#fff4d6;color:#8a5a00}.badge.low{background:#fde7e4;color:#b42318}
.attn{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:6px 0;font-size:14px}.attn input{width:110px;padding:6px}
.error{color:#b42318}.good{color:#137333}
.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}.hint{font-size:13px;color:#666;margin:10px 0}
@media(max-width:600px){body{margin:16px auto}.panel{padding:14px}.toolbar{gap:7px}.toolbar button{padding:9px 11px}}
</style>
</head>
<body>
<div class="topbar">
  <div><h1>Plot Map Cleaner</h1>
  <p class="muted">Clean survey sheets, review the result, make manual corrections, then approve or download.</p></div>
  <button type="button" id="show-approved" class="secondary">Approved Images <span id="approved-count">0</span></button>
</div>
<section id="approved-panel" class="panel hidden">
  <div class="approved-heading">
    <div><h2>Approved Images</h2><p class="muted">Approved survey sheets. Drag the pieces on the board to assemble them; edges snap together.</p></div>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <button type="button" id="delete-all-approved" class="danger hidden">Delete all</button>
      <button type="button" id="refresh-approved" class="secondary">Refresh</button>
    </div>
  </div>
  <div class="puzzle-box">
    <h3 style="margin:0 0 4px">Puzzle board</h3>
    <p class="muted" style="margin:0 0 6px">Each sheet is one coloured piece labelled with its survey number. Drag a piece close to another and it sticks edge to edge. Click a piece to open its cleaned image.</p>
    <div id="board-root"></div>
  </div>
  <details class="puzzle-box">
    <summary style="cursor:pointer;font-weight:600">Automatic match - press Build puzzle here first if you want to use the Auto-arrange pieces button</summary>
    <div class="approved-heading">
      <div><h3 style="margin:0">Automatic puzzle</h3>
      <p class="muted" style="margin:4px 0 0">Joins the approved sheets of a village along their shared thick boundaries, using each sheet's printed scale (1 : N) and header. Rotation is corrected automatically.</p></div>
      <button type="button" id="build-puzzle">Build puzzle</button>
    </div>
    <progress id="puzzle-bar" class="puzzle-bar hidden" max="100" value="0"></progress>
    <div id="puzzle-status" class="status" role="status" aria-live="polite"></div>
    <div id="puzzle-output"></div>
  </details>
  <div id="approved-list" class="approved-list"></div>
  <div id="approved-status" class="status" role="status" aria-live="polite"></div>
  <button type="button" id="back-to-cleaner" class="secondary">Back to Cleaner</button>
</section>

<section class="panel upload" id="upload-panel">
  <form id="upload-form">
    <input id="file-input" type="file" name="file" accept=".tif,.tiff,.png,.jpg,.jpeg" multiple required>
    <div id="file-count" class="muted" style="margin-top:6px"></div>
    <div class="options">
      <label><input type="checkbox" name="bw" id="bw" checked> Save as 1-bit black &amp; white (smaller file)</label>
      <label><input type="checkbox" name="dbg" id="dbg"> Generate debug image</label>
    </div>
    <button id="clean-button" type="submit">Clean images</button>
  </form>
  <div id="upload-status" class="status" role="status" aria-live="polite"></div>
  <div id="progress-shell" class="progress-shell hidden" aria-live="polite">
    <div class="progress-meta"><span id="progress-label">Preparing…</span><strong id="progress-percent">0%</strong></div>
    <div id="progress-track" class="progress-track" role="progressbar" aria-label="Cleaning progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
      <div id="progress-fill" class="progress-fill"></div>
    </div>
    <div id="cancel-all-row" class="cancel-all-row hidden"><button type="button" id="cancel-all" class="danger">Cancel all</button></div>
    <div id="queue-list" class="queue-list"></div>
    <div id="batch-actions" class="batch-actions hidden"><button type="button" id="approve-all" class="approve">Approve all</button><button type="button" id="download-all" class="download">Download all (.zip)</button><span id="batch-msg" class="muted"></span></div>
  </div>
</section>

<div id="orig-win" class="hidden" role="dialog" aria-label="Original image">
  <div id="orig-head"><span class="title" id="orig-title">Original</span>
    <button type="button" id="orig-zoom-out" title="Zoom out">−</button><span id="orig-zoom-val">100%</span><button type="button" id="orig-zoom-in" title="Zoom in">+</button><button type="button" id="orig-fit" title="Fit width">Fit</button><button type="button" id="orig-close" title="Close">✕</button></div>
  <div id="orig-body"><img id="orig-img" alt="Original sheet" draggable="false"></div>
</div>
<section id="review-panel" class="panel hidden">
  <h2>Review cleaned image</h2>
  <p class="muted" id="image-info"></p>
  <div class="toolbar" aria-label="Manual editing tools">
    <button type="button" id="tool-pan" class="secondary active" data-tool="pan">View</button>
    <button type="button" id="tool-erase" class="secondary" data-tool="erase">Brush erase</button>
    <button type="button" id="tool-line" class="secondary" data-tool="line">Add line</button>
    <button type="button" id="tool-text" class="secondary" data-tool="text" title="Type a plot number, then click where it belongs">Add text</button>
    <button type="button" id="tool-magicline" class="secondary" data-tool="magicline" title="Drag a box around a gap: the broken pieces are joined">✨ Magic line</button>
    <button type="button" id="tool-magicerase" class="secondary" data-tool="magicerase" title="Click a number or a piece of line to remove it">✨ Magic erase</button>
    <button type="button" id="show-original" class="secondary" title="Open the original sheet in a window you can drag aside">Show original</button>
    <label id="brush-wrap" class="hidden">Brush size <input id="brush-size" type="range" min="2" max="60" value="14"><span id="brush-value">14</span></label>
    <label id="line-wrap" class="hidden">Line size <input id="line-size" type="range" min="1" max="40" value="4"><span id="line-value">4</span></label>
    <label id="text-wrap" class="hidden">Text <input id="text-value" type="text" maxlength="20" placeholder="e.g. 12A" style="width:90px;padding:4px 6px"> Size <input id="text-size" type="range" min="4" max="40" value="10"><span id="text-size-value">10</span></label>
    <button type="button" id="undo" class="secondary" disabled>Undo</button>
    <button type="button" id="redo" class="secondary" disabled>Redo</button>
    <button type="button" id="reset-edits" class="secondary">Reset edits</button>
  </div>
  <p class="hint">Brush erase paints white over unwanted marks. Add line draws a black line between the start and end points. Undo/redo changes your edits. Edits are applied to the full-resolution image when downloading.</p>
  <div class="preview-wrap"><canvas id="editor"></canvas></div>
  <div class="actions">
    <button type="button" class="secondary" id="save-edits" title="Keep your edits for this image (Ctrl/Cmd+S)">Save edits</button>
    <span id="save-state"></span>
    <button type="button" class="approve" id="approve">Approve (add to puzzle)</button>
    <button type="button" class="download" id="download">Download final TIFF</button>
    <button type="button" class="secondary" id="new-image">Clean another image</button>
  </div>
  <div id="review-status" class="status" role="status" aria-live="polite"></div>
</section>

<script>
(() => {
  const $ = id => document.getElementById(id);
  const uploadForm = $('upload-form'), fileInput = $('file-input');
  const cleanButton = $('clean-button'), uploadStatus = $('upload-status');
  const progressShell = $('progress-shell'), cancelAllRow = $('cancel-all-row');
  const progressLabel = $('progress-label'), progressPercent = $('progress-percent');
  const progressTrack = $('progress-track');
  const progressFill = $('progress-fill');
  const queueList = $('queue-list');
  const reviewPanel = $('review-panel'), editor = $('editor'), ctx = editor.getContext('2d');
  const reviewStatus = $('review-status'), imageInfo = $('image-info');
  let jobId = null, baseImage = null, operations = [], redoStack = [];
  let batchItems = [];
  let tool = 'pan', drawing = false, activeOperation = null, lineStart = null, magicRect = null, magicBusy = false;
  let previewWidth = 0, previewHeight = 0;

  function status(el, message, kind='') { el.textContent = message; el.className = 'status ' + kind; }
  const approvedPanel = $('approved-panel');
  const uploadPanel = $('upload-panel');
  const approvedList = $('approved-list');
  const approvedStatus = $('approved-status');

  async function loadApprovedImages() {
    status(approvedStatus, 'Loading approved images…');
    try {
      const response = await fetch('/api/approved');
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not load approved images.');
      $('approved-count').textContent = data.images.length;
      $('delete-all-approved').classList.toggle('hidden', data.images.length <= 2);
      approvedList.replaceChildren();
      if (!data.images.length) {
        const empty = document.createElement('p');
        empty.className = 'muted';
        empty.textContent = 'No approved images yet. Clean a survey sheet and click Approve to add it here.';
        approvedList.appendChild(empty);
      } else {
        for (const item of data.images) {
          const card = document.createElement('article');
          card.className = 'approved-card';
          const image = document.createElement('img');
          image.src = item.preview_url;
          image.alt = 'Preview of ' + item.filename;
          const title = document.createElement('h3');
          title.textContent = (item.order ? '#' + item.order + ' · ' : '') + item.filename;
          const meta = document.createElement('p');
          meta.className = 'meta';
          meta.textContent = `${item.width.toLocaleString()} × ${item.height.toLocaleString()} px · Approved ${new Date(item.approved_at).toLocaleString()}`;
          const note = document.createElement('p');
          note.className = 'meta';
          note.textContent = item.puzzle_note || 'Status: Approved · not in a puzzle yet';
          const actions = document.createElement('div');
          actions.className = 'actions';
          const download = document.createElement('a');
          download.className = 'button';
          download.href = item.download_url;
          download.textContent = 'Download TIFF';
          download.style.textDecoration = 'none';
          actions.appendChild(download);
          const remove = el('button', 'secondary', 'Delete');
          remove.type = 'button';
          remove.style.background = '#fde7e4';
          remove.style.color = '#b42318';
          remove.addEventListener('click', async () => {
            if (!window.confirm(`Delete approved image "${item.filename}"? This removes it from future puzzle builds.`)) return;
            remove.disabled = true;
            try {
              const response = await fetch(`/api/approved/${encodeURIComponent(item.id)}`, {method:'DELETE'});
              const result = await response.json();
              if (!response.ok) throw new Error(result.error || 'Could not delete approved image.');
              status(approvedStatus, result.message || 'Approved image deleted.', 'good');
              puzzleOut.replaceChildren();
              puzzleBar.classList.add('hidden');
              status(puzzleStatus, 'Approved images changed. Build the puzzle again to refresh the result.');
              await loadApprovedImages();
              if (window.PlotBoard) PlotBoard.reload();
            } catch (error) {
              status(approvedStatus, error.message, 'error');
              remove.disabled = false;
            }
          });
          actions.appendChild(remove);
          card.append(image, title, meta, note, actions);
          approvedList.appendChild(card);
        }
      }
      status(approvedStatus, data.images.length ? `${data.images.length} approved image(s) saved.` : '');
    } catch (error) {
      status(approvedStatus, error.message, 'error');
    }
  }

  // ---------- automatic puzzle ----------
  const puzzleBtn = $('build-puzzle'), puzzleStatus = $('puzzle-status'), puzzleOut = $('puzzle-output'), puzzleBar = $('puzzle-bar');
  let puzzleTimer = null;

  // ---------- manual placement ----------
  async function manualCall(body, note) {
    status(puzzleStatus, note || 'Updating the puzzle…');
    document.querySelectorAll('#manual-box button').forEach(b => b.disabled = true);
    try {
      const r = await fetch('/api/puzzle/manual', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'Manual placement failed.');
      renderPuzzle(data.report);
      status(puzzleStatus, 'Placement updated.', 'good');
    } catch (error) {
      status(puzzleStatus, error.message, 'error');
      document.querySelectorAll('#manual-box button').forEach(b => b.disabled = false);
    }
  }
  const manualSel = {};
  function renderManual(report) {
    const old = $('manual-box'); if (old) old.remove();
    if (!report) return;
    const members = [];
    for (const g of report.groups) for (const s of g.sheets) members.push({id: s.id, survey: s.survey, name: s.name, order: s.order, manual: s.manual, placed: true});
    for (const u of report.unattached || []) members.push({id: u.id, survey: u.survey, name: u.name, placed: false});
    if (members.length < 2) return;
    const label = m => `${m.order ? '#' + m.order + ' · ' : ''}S.${m.survey || '?'} (${m.name})${m.placed ? '' : ' – not attached'}`;
    const box = el('section', 'group-card'); box.id = 'manual-box';
    box.appendChild(el('h3', '', 'Manual placement'));
    box.appendChild(el('p', 'muted', 'Pick a sheet. To attach a loose sheet, choose its neighbour and which side it goes on, then fine-tune with the arrows. Nudges are a share of the sheet size; each press redraws the group.'));
    const row1 = el('div', 'attn');
    const pick = el('select'); for (const m of members) { const o = el('option', '', label(m)); o.value = m.id; pick.appendChild(o); }
    const loose = members.find(m => !m.placed);
    if (manualSel.pick && members.some(m => m.id === manualSel.pick)) pick.value = manualSel.pick;
    else if (loose) pick.value = loose.id;
    manualSel.pick = pick.value;
    const near = el('select'); const side = el('select');
    const fillNear = () => { near.replaceChildren(); for (const m of members) if (m.id !== pick.value) { const o = el('option', '', label(m)); o.value = m.id; near.appendChild(o); } };
    fillNear();
    if (manualSel.near && [...near.options].some(o => o.value === manualSel.near)) near.value = manualSel.near;
    manualSel.near = near.value;
    pick.addEventListener('change', () => { manualSel.pick = pick.value; fillNear(); manualSel.near = near.value; });
    near.addEventListener('change', () => { manualSel.near = near.value; });
    for (const [v, t] of [['right', 'on its right'], ['left', 'on its left'], ['below', 'below it'], ['above', 'above it']]) { const o = el('option', '', t); o.value = v; side.appendChild(o); }
    if (manualSel.side) side.value = manualSel.side;
    side.addEventListener('change', () => { manualSel.side = side.value; });
    const put = el('button', '', 'Place here');
    put.addEventListener('click', () => manualCall({id: pick.value, to: near.value, side: side.value}, 'Placing the sheet…'));
    row1.append(el('span', '', 'Sheet'), pick, el('span', '', 'goes next to'), near, side, put);
    box.appendChild(row1);
    const row2 = el('div', 'attn');
    const step = el('select');
    for (const [v, t] of [['0.005', 'Fine (0.5%)'], ['0.02', 'Medium (2%)'], ['0.1', 'Coarse (10%)']]) { const o = el('option', '', t); o.value = v; step.appendChild(o); }
    step.value = manualSel.step || '0.02';
    step.addEventListener('change', () => { manualSel.step = step.value; });
    const rotStep = el('select');
    for (const [v, t] of [['0.25', '0.25°'], ['1', '1°'], ['5', '5°']]) { const o = el('option', '', t); o.value = v; rotStep.appendChild(o); }
    if (manualSel.rot) rotStep.value = manualSel.rot;
    rotStep.addEventListener('change', () => { manualSel.rot = rotStep.value; });
    const btn = (text, fn) => { const b = el('button', 'secondary', text); b.addEventListener('click', fn); return b; };
    const mv = (fx, fy) => manualCall({id: pick.value, fx: fx * Number(step.value), fy: fy * Number(step.value)}, 'Moving…');
    const rt = d => manualCall({id: pick.value, rot: d * Number(rotStep.value)}, 'Rotating…');
    row2.append(el('span', '', 'Move'), step, btn('◀', () => mv(-1, 0)), btn('▲', () => mv(0, -1)), btn('▼', () => mv(0, 1)), btn('▶', () => mv(1, 0)),
                el('span', '', 'Rotate'), rotStep, btn('⟲', () => rt(-1)), btn('⟳', () => rt(1)));
    box.appendChild(row2);
    const row3 = el('div', 'attn');
    row3.append(btn('Undo manual placement of this sheet', () => manualCall({id: pick.value, reset: true})),
                btn('Undo all manual placement', () => { if (window.confirm('Remove every manual placement?')) manualCall({reset_all: true}); }));
    box.appendChild(row3);
    puzzleOut.prepend(box);
  }
  function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
  let mapView = null;   // remembered zoom / pan so a redraw does not reset the view
  function buildMapViewer(report) {
    const card = el('article', 'group-card'); card.id = 'map-viewer';
    card.appendChild(el('h3', '', 'Whole map – every group in one place'));
    card.appendChild(el('p', 'muted', 'Scroll or pinch to zoom, drag to move around. Zoom in and the full-resolution drawing loads.'));
    const tools = el('div', 'map-tools');
    const vp = el('div', 'map-viewport'); const stage = el('div', 'map-stage'); vp.appendChild(stage);
    const zoomLabel = el('span', 'zoomval', '100%');
    const tb = (t, fn, title) => { const b = el('button', 'secondary', t); b.type = 'button'; b.title = title || ''; b.addEventListener('click', fn); return b; };
    const imgs = [];
    const stamp = Date.now();
    for (const g of report.groups) {
      const fig = el('figure');
      const surveys = g.sheets.map(s => 'S.' + (s.survey || '?')).join(' · ');
      fig.appendChild(el('figcaption', '', `Group ${g.group} – ${surveys}`));
      const img = el('img'); img.alt = `Puzzle group ${g.group}`; img.draggable = false;
      img.dataset.full = `/puzzle/${g.group}/download?fmt=png&t=${stamp}`;
      img.addEventListener('load', () => { if (!img.dataset.fixed) { img.style.width = img.naturalWidth + 'px'; img.dataset.fixed = '1'; } if (!mapView) fit(); });
      img.src = `/puzzle/${g.group}/preview?t=${stamp}`;
      fig.appendChild(img); stage.appendChild(fig); imgs.push(img);
    }
    let sc = 1, tx = 0, ty = 0, hires = false;
    const apply = () => {
      stage.style.transform = `translate(${tx}px,${ty}px) scale(${sc})`;
      zoomLabel.textContent = Math.round(sc * 100) + '%';
      mapView = {s: sc, x: tx, y: ty, n: report.groups.length};
      if (sc > 1.1 && !hires) { hires = true; for (const im of imgs) if (im.src.indexOf('/preview') !== -1) im.src = im.dataset.full; }
    };
    function fit() {
      const w = stage.offsetWidth, h = stage.offsetHeight, vw = vp.clientWidth, vh = vp.clientHeight;
      if (!w || !h) return;
      sc = Math.min(vw / w, vh / h) * 0.98; tx = (vw - w * sc) / 2; ty = (vh - h * sc) / 2; apply();
    }
    const zoomAt = (mx, my, f) => {
      const ns = Math.max(0.03, Math.min(12, sc * f)); f = ns / sc;
      tx = mx - (mx - tx) * f; ty = my - (my - ty) * f; sc = ns; apply();
    };
    const center = f => zoomAt(vp.clientWidth / 2, vp.clientHeight / 2, f);
    vp.addEventListener('wheel', e => {
      e.preventDefault();
      const r = vp.getBoundingClientRect();
      zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0018)));
    }, {passive: false});
    const pts = new Map(); let pinch = 0;
    vp.addEventListener('pointerdown', e => { vp.setPointerCapture(e.pointerId); pts.set(e.pointerId, {x: e.clientX, y: e.clientY}); vp.classList.add('dragging'); pinch = 0; });
    vp.addEventListener('pointermove', e => {
      const p = pts.get(e.pointerId); if (!p) return;
      if (pts.size === 2) {
        const [a, b] = [...pts.values()]; const d0 = Math.hypot(a.x - b.x, a.y - b.y);
        p.x = e.clientX; p.y = e.clientY;
        const [c, d] = [...pts.values()]; const d1 = Math.hypot(c.x - d.x, c.y - d.y);
        if (pinch && d0) { const r = vp.getBoundingClientRect(); zoomAt((c.x + d.x) / 2 - r.left, (c.y + d.y) / 2 - r.top, d1 / d0); }
        pinch = d1; return;
      }
      tx += e.clientX - p.x; ty += e.clientY - p.y; p.x = e.clientX; p.y = e.clientY; apply();
    });
    const up = e => { pts.delete(e.pointerId); if (!pts.size) vp.classList.remove('dragging'); pinch = 0; };
    vp.addEventListener('pointerup', up); vp.addEventListener('pointercancel', up);
    vp.addEventListener('dblclick', e => { const r = vp.getBoundingClientRect(); zoomAt(e.clientX - r.left, e.clientY - r.top, 2); });
    tools.append(tb('−', () => center(1 / 1.4), 'Zoom out'), zoomLabel, tb('+', () => center(1.4), 'Zoom in'),
                 tb('Fit all', fit, 'Show everything'), tb('100%', () => { sc = 1; tx = 20; ty = 20; apply(); }, 'Actual size'),
                 tb('Full screen', () => { if (vp.requestFullscreen) vp.requestFullscreen(); }, 'Full screen'));
    card.append(tools, vp);
    if (mapView && mapView.n === report.groups.length) { sc = mapView.s; tx = mapView.x; ty = mapView.y; requestAnimationFrame(apply); }
    else requestAnimationFrame(() => requestAnimationFrame(fit));
    return card;
  }
  function renderPuzzle(report) { renderPuzzleBase(report); renderManual(report); }
  function renderPuzzleBase(report) {
    puzzleOut.replaceChildren();
    if (!report) return;
    const sheetMetadataById = new Map((report.sheet_metadata || []).map(sheet => [sheet.id, sheet]));
    if (report.groups.length) puzzleOut.appendChild(buildMapViewer(report));
    for (const g of report.groups) {
      const card = el('article', 'group-card');
      card.appendChild(el('h3', '', `Group ${g.group} · ${g.village} · ${g.sheets.length} sheets`));
      const surveys = g.sheets.map(s => `${s.survey || '?'}${Math.abs(s.rotation_deg) > 0.05 ? ' (' + s.rotation_deg + '°)' : ''}`).join(', ');
      card.appendChild(el('p', 'meta muted', 'Survey nos.: ' + surveys));
      const incompleteHeaders = g.sheets
        .map(sheet => sheetMetadataById.get(sheet.id))
        .filter(meta => meta && (!meta.village || !meta.scale || !meta.survey));
      if (incompleteHeaders.length) {
        const missing = incompleteHeaders.map(meta => `${meta.name}: ${[
          !meta.survey ? 'survey number' : '', !meta.village ? 'village' : '', !meta.scale ? 'scale' : ''
        ].filter(Boolean).join(', ')}`).join('; ');
        card.appendChild(el('p', 'error', `Some header fields were not detected automatically: ${missing}.`));
        const ocrProblem = incompleteHeaders.find(meta => meta.ocr_error);
        if (ocrProblem) card.appendChild(el('p', 'muted', `OCR setup: ${ocrProblem.ocr_error}`));
        else card.appendChild(el('p', 'muted', 'Verify Tesseract OCR is installed and that the source image includes a readable header.'));
      }
      if (g.overlap_warning) card.appendChild(el('p', 'error', 'Warning: some sheets overlap in this group – check the joins below.'));
      if (g.low_confidence_links) card.appendChild(el('p', 'error', `${g.low_confidence_links} link(s) have low confidence (a straight border can slide) – please check them by eye.`));
      const actions = el('div', 'actions');
      for (const [label, fmt] of [['Download PNG', 'png'], ['Download TIFF (1-bit)', 'tif']]) {
        const a = el('a', 'button', label); a.href = `/puzzle/${g.group}/download?fmt=${fmt}`; a.style.textDecoration = 'none'; actions.appendChild(a);
      }
      card.appendChild(actions);
      const ids = new Set(g.sheets.map(s => s.id));
      const links = report.edges.filter(e => e.used && ids.has(e.a) && ids.has(e.b));
      if (links.length) {
        const table = el('table', 'link-table');
        const head = el('tr');
        for (const h of ['Joined', 'Shared border', 'Overlap', 'Reciprocal match', 'Relative rotation', 'Confidence']) head.appendChild(el('th', '', h));
        table.appendChild(head);
        for (const e of links) {
          const row = el('tr');
          row.appendChild(el('td', '', `S.${e.a_survey || '?'} + S.${e.b_survey || '?'}`));
          row.appendChild(el('td', '', e.shared_border_pct + '% of the smaller outline'));
          row.appendChild(el('td', '', e.overlap_pct + '%'));
          row.appendChild(el('td', '', e.match_balance == null ? '—' : `${Math.round(e.match_balance * 100)}%`));
          row.appendChild(el('td', '', e.rotation_deg == null ? '—' : `${e.rotation_deg}°`));
          const c = el('td'); c.appendChild(el('span', 'badge ' + e.confidence, e.confidence)); row.appendChild(c);
          table.appendChild(row);
        }
        card.appendChild(table);
      }
      puzzleOut.appendChild(card);
    }
    if (report.unattached.length) {
      const box = el('div', 'group-card');
      box.appendChild(el('h3', '', `Not attached (${report.unattached.length})`));
      for (const u of report.unattached) box.appendChild(el('p', 'meta', `S.${u.survey || '?'} – ${u.name}: ${u.reason}`));
      puzzleOut.appendChild(box);
    }
    if (report.needs_attention.length) {
      const box = el('div', 'group-card');
      box.appendChild(el('h3', '', `Needs attention (${report.needs_attention.length})`));
      for (const n of report.needs_attention) {
        const row = el('div', 'attn');
        row.appendChild(el('span', '', `${n.name}: ${n.reason}`));
        if (/scale/.test(n.reason)) {
          const input = el('input'); input.type = 'number'; input.placeholder = 'Scale 1 : N'; input.min = '50';
          const save = el('button', 'secondary', 'Save');
          save.addEventListener('click', async () => {
            if (!input.value) {
              status(puzzleStatus, 'Enter the denominator printed after 1 : before saving.', 'error');
              return;
            }
            save.disabled = true;
            try {
              const r = await fetch(`/api/approved/${encodeURIComponent(n.id)}/meta`, {
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body:JSON.stringify({scale:Number(input.value)})
              });
              const data = await r.json();
              if (!r.ok) throw new Error(data.error || 'Could not save the scale.');
              status(puzzleStatus, `Scale 1:${Number(input.value)} saved for ${n.name}. Click Build puzzle again.`, 'good');
            } catch (error) {
              status(puzzleStatus, error.message, 'error');
              save.disabled = false;
            }
          });
          row.append(input, save);
        }
        box.appendChild(row);
      }
      puzzleOut.appendChild(box);
    }

    // Explain what the matcher actually read and why candidate joins were rejected.
    if (!report.groups.length) {
      const diagnostic = el('section', 'group-card');
      diagnostic.appendChild(el('h3', '', 'Why no puzzle was created'));

      if (report.sheet_metadata && report.sheet_metadata.length) {
        diagnostic.appendChild(el('p', 'muted', 'Detected sheet metadata:'));
        const table = el('table', 'link-table');
        const head = el('tr');
        for (const label of ['File', 'Survey', 'Village', 'Detected scale', 'Outline', 'Header OCR']) head.appendChild(el('th', '', label));
        table.appendChild(head);
        for (const sheet of report.sheet_metadata) {
          const row = el('tr');
          row.appendChild(el('td', '', sheet.name || 'Unknown'));
          row.appendChild(el('td', '', sheet.survey || 'not detected'));
          row.appendChild(el('td', '', sheet.village || 'not detected'));
          row.appendChild(el('td', '', sheet.scale ? `1:${sheet.scale}` : 'missing'));
          row.appendChild(el('td', '', sheet.outline_points ? `${sheet.outline_points.toLocaleString()} points` : (sheet.error || 'not extracted')));
          row.appendChild(el('td', '', sheet.ocr_error || (sheet.scale ? 'Scale detected' : 'No text recognized')));
          table.appendChild(row);
        }
        diagnostic.appendChild(table);
      }

      const candidates = (report.edges || []).slice().sort((a,b) => (b.shared_border_pct || 0) - (a.shared_border_pct || 0));
      if (candidates.length) {
        diagnostic.appendChild(el('p', 'muted', 'Best candidate border matches:'));
        for (const edge of candidates.slice(0, 8)) {
          const title = `Survey ${edge.a_survey || '?'} + Survey ${edge.b_survey || '?'}`;
          const details = `Shared border ${edge.shared_border_pct}% · overlap ${edge.overlap_pct}% · ${edge.confidence} confidence.`;
          const why = edge.used ? 'Used in the puzzle.' :
            edge.accepted ? 'Accepted by pair validation but not selected during assembly.' :
            (edge.rejection_reason || 'Rejected by border validation.');
          diagnostic.appendChild(el('p', 'meta', `${title}: ${details} ${why}`));
        }
      } else {
        diagnostic.appendChild(el('p', 'error',
          'No candidate pairs were compared. Check that at least two readable sheets have valid scales and matching village metadata.'));
      }

      if (report.needs_attention && report.needs_attention.length) {
        diagnostic.appendChild(el('p', 'error', 'Some sheets still need attention above. Fix those fields and click Build puzzle again.'));
      } else if (!candidates.length) {
        diagnostic.appendChild(el('p', 'muted', 'If metadata looks correct, share the terminal output from this puzzle build so we can inspect outline extraction.'));
      }
      puzzleOut.appendChild(diagnostic);
    }

    if (report.unattached && report.unattached.length) {
      const unattachedPanel = el('section', 'group-card');
      unattachedPanel.appendChild(el('h3', '', 'Why some sheets were not attached'));
      const sheetsById = new Map((report.sheet_metadata || []).map(sheet => [sheet.id, sheet]));
      for (const item of report.unattached) {
        const sheet = sheetsById.get(item.id) || {};
        const heading = el('p', '', item.name || 'Unknown sheet');
        heading.style.fontWeight = '700';
        unattachedPanel.appendChild(heading);
        unattachedPanel.appendChild(el('p', 'meta',
          `Detected: survey ${sheet.survey || '?'}, village ${sheet.village || '?'}, scale ${sheet.scale ? '1:' + sheet.scale : 'missing'}, orientation ${sheet.orientation || 'unknown'}${sheet.width && sheet.height ? ` (${sheet.width.toLocaleString()} × ${sheet.height.toLocaleString()} px)` : ''}.`));
        const candidates = (report.edges || []).filter(edge => edge.a === item.id || edge.b === item.id);
        if (!candidates.length) {
          unattachedPanel.appendChild(el('p', 'error',
            'No candidate border match survived the initial search for this sheet. The updated matcher includes quarter-turns for landscape/portrait pairs; rebuild the puzzle to retry.'));
        } else {
          for (const edge of candidates.slice(0, 6)) {
            const otherId = edge.a === item.id ? edge.b : edge.a;
            const other = sheetsById.get(otherId) || {};
            const pairName = `${item.name || 'Sheet'} ↔ ${other.name || 'another sheet'}`;
            const outcome = edge.used ? 'Used in a puzzle group.' :
              edge.accepted ? 'Passed pair checks but was not selected into the final assembly.' :
              (edge.rejection_reason || 'Rejected by border checks.');
            unattachedPanel.appendChild(el('p', 'meta',
              `${pairName}: shared border ${edge.shared_border_pct}%, overlap ${edge.overlap_pct}%, ${edge.confidence} confidence. ${outcome}`));
          }
        }
      }
      puzzleOut.appendChild(unattachedPanel);
    }
  }
  async function pollPuzzle() {
    try {
      const r = await fetch('/api/puzzle/status'); const d = await r.json();
      if (d.state === 'running') {
        puzzleBar.classList.remove('hidden'); puzzleBar.value = Math.round(d.fraction * 100);
        status(puzzleStatus, `${d.message} (${Math.round(d.fraction * 100)}%)`);
        puzzleTimer = setTimeout(pollPuzzle, 1200); puzzleBtn.disabled = true; return;
      }
      puzzleBar.classList.add('hidden'); puzzleBtn.disabled = false;
      if (d.state === 'error') { status(puzzleStatus, d.error, 'error'); return; }
      if (d.state === 'done') {
        const n = d.report.groups.length;
        const needs = (d.report.needs_attention || []).length;
        const candidates = (d.report.edges || []).length;
        const unattached = (d.report.unattached || []).length;
        const message = n
          ? `Puzzle ready: ${n} group(s).${unattached ? ` ${unattached} sheet(s) remain unattached; see diagnostics below.` : ''}`
          : needs
            ? `No puzzle created: ${needs} sheet(s) need metadata or extraction fixes. See diagnostics below.`
            : candidates
              ? 'Sheets were read, but candidate joins failed validation. See border diagnostics below.'
              : 'No candidate matches were found. See detected sheet metadata and terminal logs below.';
        status(puzzleStatus, message, n ? 'good' : 'error');
        renderPuzzle(d.report); loadApprovedImages();
      }
    } catch (e) { puzzleBtn.disabled = false; status(puzzleStatus, e.message, 'error'); }
  }
  puzzleBtn.addEventListener('click', async () => {
    puzzleBtn.disabled = true; status(puzzleStatus, 'Starting…');
    const r = await fetch('/api/puzzle/build', {method:'POST'}); const d = await r.json();
    if (!r.ok) { puzzleBtn.disabled = false; status(puzzleStatus, d.error || 'Could not start.', 'error'); return; }
    pollPuzzle();
  });
  fetch('/api/puzzle/status').then(r => r.json()).then(d => { if (d.state === 'done') renderPuzzle(d.report); else if (d.state === 'running') pollPuzzle(); }).catch(() => {});
  // ---------- end puzzle ----------

  $('show-approved').addEventListener('click', () => {
    if (!confirmDiscard()) return;
    uploadPanel.classList.add('hidden');
    reviewPanel.classList.add('hidden');
    approvedPanel.classList.remove('hidden');
    loadApprovedImages();
    if (window.PlotBoard) setTimeout(() => PlotBoard.open(), 0);
    window.scrollTo({top:0, behavior:'smooth'});
  });
  $('refresh-approved').addEventListener('click', loadApprovedImages);
  $('delete-all-approved').addEventListener('click', async () => {
    const n = $('approved-count').textContent;
    if (!window.confirm(`Delete ALL ${n} approved images? This also clears the puzzle board and cannot be undone.`)) return;
    const btn = $('delete-all-approved'); btn.disabled = true;
    try {
      const response = await fetch('/api/approved', {method:'DELETE'});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Could not delete the approved images.');
      puzzleOut.replaceChildren();
      puzzleBar.classList.add('hidden');
      status(puzzleStatus, '');
      await loadApprovedImages();
      status(approvedStatus, result.message || 'All approved images deleted.', 'good');
      if (window.PlotBoard) PlotBoard.reload();
    } catch (error) {
      status(approvedStatus, error.message, 'error');
    } finally { btn.disabled = false; }
  });
  $('back-to-cleaner').addEventListener('click', () => {
    approvedPanel.classList.add('hidden');
    uploadPanel.classList.remove('hidden');
    window.scrollTo({top:0, behavior:'smooth'});
  });
  function setBusy(busy) {
    cleanButton.disabled = busy;
    cleanButton.textContent = busy ? 'Cleaning… please wait' : 'Clean images';
    if (!busy) cleanButton.disabled = false;
  }

  function setProgress(percent, label) {
    const value = Math.max(0, Math.min(100, Math.round(percent)));
    progressShell.classList.remove('hidden');
    progressLabel.textContent = label;
    progressPercent.textContent = `${value}%`;
    progressFill.style.width = value + '%';
    progressTrack.setAttribute('aria-valuenow', String(value));
  }

  // ---- saved edits: each image keeps its saved edits; unsaved ones trigger a warning before they are lost
  const savedOps = {};            // job id -> saved operations
  let savedSnapshot = '[]';
  const cloneOps = o => JSON.parse(JSON.stringify(o));
  function isDirty() { return !!jobId && JSON.stringify(operations) !== savedSnapshot; }
  function confirmDiscard() {
    if (!isDirty()) return true;
    if (!window.confirm('Oops - you forgot to press "Save edits".\n\nYour changes on this image are not saved yet.\nPress Cancel to go back and save them, or OK to discard them.')) return false;
    operations = cloneOps(savedOps[jobId] || []); redoStack = []; activeOperation = null; magicRect = null;
    if (baseImage) renderCanvas();
    return true;
  }
  window.addEventListener('beforeunload', e => { if (isDirty()) { e.preventDefault(); e.returnValue = ''; } });

  function updateHistoryButtons() {
    $('undo').disabled = operations.length === 0;
    $('redo').disabled = redoStack.length === 0;
    if (!activeOperation && !magicRect) {
      const dirty = isDirty();
      $('save-state').textContent = dirty ? '● Unsaved changes' : (jobId ? '✔ Saved' : '');
      $('save-state').style.color = dirty ? '#b42318' : '#137333';
      $('save-edits').classList.toggle('save-dirty', dirty);
    }
  }

  function renderCanvas() {
    if (!baseImage) return;
    ctx.clearRect(0, 0, editor.width, editor.height);
    ctx.drawImage(baseImage, 0, 0);
    for (const op of operations) drawOperation(ctx, op);
    if (activeOperation) drawOperation(ctx, activeOperation);
    if (magicRect) {
      ctx.save(); ctx.setLineDash([8, 5]); ctx.lineWidth = 2; ctx.strokeStyle = '#1769e0'; ctx.fillStyle = 'rgba(23,105,224,.10)';
      const rx = Math.min(magicRect.x0, magicRect.x1), ry = Math.min(magicRect.y0, magicRect.y1), rw = Math.abs(magicRect.x1 - magicRect.x0), rh = Math.abs(magicRect.y1 - magicRect.y0);
      ctx.fillRect(rx, ry, rw, rh); ctx.strokeRect(rx, ry, rw, rh); ctx.restore();
    }
    updateHistoryButtons();
  }

  function drawOperation(context, op) {
    context.save();
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.strokeStyle = op.type === 'erase' ? '#ffffff' : '#000000';
    context.fillStyle = context.strokeStyle;
    context.lineWidth = op.magic ? Math.max(op.size, 1.2) : op.size;
    if (op.type === 'text') {
      context.fillStyle = '#000000';
      context.font = `${op.size}px Arial, Helvetica, sans-serif`;
      context.textAlign = 'center'; context.textBaseline = 'middle';
      context.fillText(op.text, op.x, op.y);
    } else if (op.type === 'polys') {
      context.fillStyle = '#ffffff';
      for (const poly of op.polys) {
        if (poly.length < 3) continue;
        context.beginPath(); context.moveTo(poly[0][0], poly[0][1]);
        for (let i = 1; i < poly.length; i++) context.lineTo(poly[i][0], poly[i][1]);
        context.closePath(); context.fill();
        context.strokeStyle = '#ffffff'; context.lineWidth = 1.2; context.stroke();   // hides the soft grey halo of the shrunk preview
      }
    } else if (op.type === 'line') {
      context.beginPath();
      context.moveTo(op.points[0][0], op.points[0][1]);
      context.lineTo(op.points[1][0], op.points[1][1]);
      context.stroke();
    } else {
      const pts = op.points;
      if (pts.length === 1) {
        context.beginPath(); context.arc(pts[0][0], pts[0][1], op.size/2, 0, Math.PI*2); context.fill();
      } else {
        context.beginPath(); context.moveTo(pts[0][0], pts[0][1]);
        for (let i=1; i<pts.length; i++) context.lineTo(pts[i][0], pts[i][1]);
        context.stroke();
      }
    }
    context.restore();
  }

  function canvasPoint(event) {
    const rect = editor.getBoundingClientRect();
    return [
      Math.max(0, Math.min(editor.width, (event.clientX - rect.left) * editor.width / rect.width)),
      Math.max(0, Math.min(editor.height, (event.clientY - rect.top) * editor.height / rect.height))
    ];
  }

  function selectTool(nextTool) {
    tool = nextTool;
    document.querySelectorAll('[data-tool]').forEach(btn => btn.classList.toggle('active', btn.dataset.tool === tool));
    $('brush-wrap').classList.toggle('hidden', tool !== 'erase');
    $('line-wrap').classList.toggle('hidden', tool !== 'line');
    $('text-wrap').classList.toggle('hidden', tool !== 'text');
    if (tool === 'text') { status(reviewStatus, 'Add text: type the plot number, then click on the sheet where it belongs. The text is centred on your click. Undo if it is off.'); setTimeout(() => $('text-value').focus(), 0); }
    editor.style.cursor = tool === 'pan' ? 'default' : 'crosshair';
    if (tool === 'magicline') status(reviewStatus, 'Magic line: drag a box around the gap. Cover a bit of both broken pieces.');
    if (tool === 'magicerase') status(reviewStatus, 'Magic erase: click on a number or on a stray piece of line to remove it.');
  }
  document.querySelectorAll('[data-tool]').forEach(btn => btn.addEventListener('click', () => selectTool(btn.dataset.tool)));

  $('brush-size').addEventListener('input', e => $('brush-value').textContent = e.target.value);
  $('line-size').addEventListener('input', e => $('line-value').textContent = e.target.value);
  $('text-size').addEventListener('input', e => $('text-size-value').textContent = e.target.value);
  function addText(p) {
    const text = $('text-value').value.trim();
    if (!text) { status(reviewStatus, 'Type the plot number in the Text box first, then click on the sheet.', 'error'); $('text-value').focus(); return; }
    operations.push({type:'text', text, x:p[0], y:p[1], size:Number($('text-size').value)});
    redoStack = []; renderCanvas();
    status(reviewStatus, `Added "${text}". Click again to add another, or Undo.`, 'good');
  }

  editor.addEventListener('pointerdown', event => {
    if (!jobId || tool === 'pan') return;
    event.preventDefault();
    if (tool === 'magicerase') { magicErase(canvasPoint(event)); return; }
    if (tool === 'text') { addText(canvasPoint(event)); return; }
    editor.setPointerCapture(event.pointerId);
    drawing = true;
    if (tool === 'magicline') { const q = canvasPoint(event); magicRect = {x0: q[0], y0: q[1], x1: q[0], y1: q[1]}; renderCanvas(); return; }
    const p = canvasPoint(event);
    const size = Number(tool === 'line' ? $('line-size').value : $('brush-size').value);
    if (tool === 'line') {
      lineStart = p;
      activeOperation = {type:'line', points:[p,p], size};
    } else {
      activeOperation = {type:'erase', points:[p], size};
    }
    renderCanvas();
  });

  editor.addEventListener('pointermove', event => {
    if (drawing && magicRect) { const q = canvasPoint(event); magicRect.x1 = q[0]; magicRect.y1 = q[1]; renderCanvas(); return; }
    if (!drawing || !activeOperation) return;
    const p = canvasPoint(event);
    if (activeOperation.type === 'line') activeOperation.points[1] = p;
    else activeOperation.points.push(p);
    renderCanvas();
  });

  async function magicCall(body) {
    if (magicBusy || !jobId) return null;
    magicBusy = true; status(reviewStatus, 'Working…');
    try {
      const r = await fetch(`/job/${jobId}/magic`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(Object.assign({operations}, body))});
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || d.message || 'Magic tool failed.');
      if (!d.ops || !d.ops.length) { status(reviewStatus, d.message || 'Nothing changed.', 'error'); return null; }
      d.ops.forEach(op => operations.push(op)); redoStack = []; renderCanvas();
      status(reviewStatus, d.message, 'good');
    } catch (e) { status(reviewStatus, e.message, 'error'); }
    finally { magicBusy = false; }
  }
  function magicErase(p) { return magicCall({mode: 'erase', point: p}); }
  function magicLine(r) {
    return magicCall({mode: 'join', rect: [Math.min(r.x0, r.x1), Math.min(r.y0, r.y1), Math.max(r.x0, r.x1), Math.max(r.y0, r.y1)]});
  }

  function finishDrawing() {
    if (magicRect) {
      const r = magicRect; magicRect = null; drawing = false; renderCanvas();
      if (Math.abs(r.x1 - r.x0) > 3 && Math.abs(r.y1 - r.y0) > 3) magicLine(r);
      else status(reviewStatus, 'Drag a box around the gap you want to join.', 'error');
      return;
    }
    if (!drawing || !activeOperation) return;
    drawing = false;
    if (activeOperation.type !== 'line' || (lineStart && Math.hypot(activeOperation.points[1][0]-lineStart[0], activeOperation.points[1][1]-lineStart[1]) > 1)) {
      operations.push(activeOperation);
      redoStack = [];
    }
    activeOperation = null; lineStart = null;
    renderCanvas();
  }
  editor.addEventListener('pointerup', finishDrawing);
  editor.addEventListener('pointercancel', finishDrawing);

  $('undo').addEventListener('click', () => {
    if (operations.length) redoStack.push(operations.pop());
    renderCanvas();
  });
  $('redo').addEventListener('click', () => {
    if (redoStack.length) operations.push(redoStack.pop());
    renderCanvas();
  });
  $('reset-edits').addEventListener('click', () => {
    operations = []; redoStack = []; activeOperation = null; renderCanvas();
    status(reviewStatus, 'Manual edits cleared. The cleaned image is restored.');
  });

  function openReview(data) {
    if (data.job_id === jobId && !reviewPanel.classList.contains('hidden')) { reviewPanel.scrollIntoView({behavior:'smooth', block:'start'}); return; }
    if (!confirmDiscard()) return;
    jobId = data.job_id;
    if (!$('orig-win').classList.contains('hidden')) setTimeout(() => { $('orig-title').textContent = 'Original — ' + data.filename; $('orig-img').src = `/job/${data.job_id}/original?t=` + Date.now(); }, 0);
    previewWidth = data.preview_width; previewHeight = data.preview_height;
    operations = cloneOps(savedOps[data.job_id] || []); redoStack = [];
    savedSnapshot = JSON.stringify(operations);
    baseImage = new Image();
    baseImage.onload = () => {
      editor.width = previewWidth; editor.height = previewHeight;
      imageInfo.textContent = `${data.filename} · Original size ${data.width.toLocaleString()} × ${data.height.toLocaleString()} pixels`;
      renderCanvas();
      reviewPanel.classList.remove('hidden');
      status(reviewStatus, 'Choose Manual Edit tools if needed, then approve or download.');
      reviewPanel.scrollIntoView({behavior:'smooth', block:'start'});
    };
    baseImage.onerror = () => status(uploadStatus, 'Could not load the preview image.', 'error');
    baseImage.src = data.preview_url + '?t=' + Date.now();
  }

  fileInput.addEventListener('change', () => {
    const n = fileInput.files.length;
    $('file-count').textContent = n ? `${n} file${n > 1 ? 's' : ''} selected` : '';
  });

  let running = false;
  uploadForm.addEventListener('submit', async event => {
    event.preventDefault();
    if (!fileInput.files.length || running) return;
    if (!confirmDiscard()) return;
    running = true;
    const files = Array.from(fileInput.files);
    const items = files.map(file => ({file, state:'waiting', pct:0, msg:'Waiting…', taskId:null, result:null, xhr:null, approved:false}));
    batchItems = items; $('batch-actions').classList.add('hidden'); $('batch-msg').textContent = '';
    reviewPanel.classList.add('hidden');
    queueList.innerHTML = '';
    items.forEach((it, idx) => {
      const row = document.createElement('div'); row.className = 'q-row';
      row.innerHTML = '<div class="q-name"></div><div class="q-bar"><div class="q-fill"></div></div><div class="q-pct">0%</div><div class="q-act"></div><div class="q-msg"></div>';
      row.querySelector('.q-name').textContent = it.file.name;
      const cb = document.createElement('button'); cb.type = 'button'; cb.className = 'danger'; cb.textContent = 'Cancel';
      cb.addEventListener('click', () => cancelItem(it));
      row.querySelector('.q-act').appendChild(cb); it.cancelBtn = cb;
      it.row = row; queueList.appendChild(row);
    });
    setBusy(true);
    status(uploadStatus, `Cleaning ${items.length} image${items.length > 1 ? 's' : ''}. Several are cleaned at the same time, as far as your computer's memory allows.`);
    const showRow = it => {
      it.row.querySelector('.q-fill').style.width = it.pct + '%';
      it.row.querySelector('.q-pct').textContent = Math.round(it.pct) + '%';
      it.row.querySelector('.q-msg').textContent = it.msg;
      const lvl = (it.state === 'done' && it.quality) ? it.quality.level : '';
      it.row.classList.toggle('done', it.state === 'done');
      it.row.classList.toggle('error', it.state === 'error');
      it.row.classList.toggle('cancelled', it.state === 'cancelled');
      if (it.cancelBtn) it.cancelBtn.style.display = (it.state === 'waiting' || it.state === 'uploading' || it.state === 'cleaning') ? '' : 'none';
      it.row.classList.toggle('good', lvl === 'good');
      it.row.classList.toggle('warn', lvl === 'warn');
      it.row.classList.toggle('bad', lvl === 'bad');
    };
    const overall = () => {
      const live = items.filter(it => it.state !== 'cancelled');
      const total = live.length ? live.reduce((s, it) => s + it.pct, 0) / live.length : 100;
      const finished = items.filter(it => it.state === 'done' || it.state === 'error' || it.state === 'cancelled').length;
      const cur = items.find(it => it.state === 'uploading' || it.state === 'cleaning') || items.find(it => it.state === 'waiting');
      let label = finished === items.length ? (live.length ? 'All done' : 'Cancelled') : `File ${Math.min(finished + 1, items.length)} of ${items.length}` + (cur ? ` · ${cur.file.name} — ${cur.msg}` : '');
      setProgress(total, label);
    };
    const cancelItem = it => {
      if (!(it.state === 'waiting' || it.state === 'uploading' || it.state === 'cleaning')) return;
      const wasUploading = it.state === 'uploading';
      it.state = 'cancelled'; it.msg = 'Cancelled'; it.pct = Math.min(it.pct, 99);
      if (wasUploading && it.xhr) { try { it.xhr.abort(); } catch (e) {} }
      if (it.taskId) fetch('/api/task/' + it.taskId + '/cancel', {method: 'POST'}).catch(() => {});
      showRow(it); overall();
    };
    $('cancel-all').onclick = () => items.forEach(cancelItem);
    cancelAllRow.classList.toggle('hidden', items.length < 2);
    const upload = it => new Promise(resolve => {
      if (it.state === 'cancelled') { resolve(); return; }
      const fd = new FormData(); fd.append('file', it.file);
      if ($('bw').checked) fd.append('bw', 'on');
      const xhr = new XMLHttpRequest(); xhr.open('POST', '/api/clean'); xhr.responseType = 'json'; it.xhr = xhr;
      it.state = 'uploading'; it.msg = 'Uploading…';
      xhr.upload.addEventListener('progress', e => { if (e.lengthComputable) { it.pct = (e.loaded / e.total) * 4; showRow(it); overall(); } });
      xhr.addEventListener('abort', () => resolve());
      xhr.addEventListener('load', () => {
        const p = xhr.response || {};
        if (it.state === 'cancelled') {                       // cancelled while the upload was finishing: stop the job on the server too
          if (p.task_id) fetch('/api/task/' + p.task_id + '/cancel', {method: 'POST'}).catch(() => {});
          resolve(); return;
        }
        if (xhr.status < 200 || xhr.status >= 300) { it.state = 'error'; it.msg = p.error || `Upload failed (${xhr.status}).`; }
        else { it.taskId = p.task_id; it.state = 'cleaning'; it.pct = 4; it.msg = 'Queued…'; }
        showRow(it); overall(); resolve();
      });
      xhr.addEventListener('error', () => { it.state = 'error'; it.msg = 'Network error while uploading.'; showRow(it); resolve(); });
      xhr.send(fd);
    });
    const poll = async () => {
      for (const it of items) {
        if (it.state !== 'cleaning' || !it.taskId) continue;
        try {
          const r = await fetch('/api/task/' + it.taskId); const d = await r.json();
          if (!r.ok) throw new Error(d.error || 'Lost track of this job.');
          if (it.state !== 'cleaning') continue;                 // cancelled while this request was in flight
          if (d.state === 'cancelled') { it.state = 'cancelled'; it.msg = 'Cancelled'; }
          else if (d.state === 'error') { it.state = 'error'; it.msg = d.error; }
          else if (d.state === 'done') {
            it.state = 'done'; it.pct = 100; it.result = d.result; it.quality = (d.result && d.result.quality) || null;
            const q = it.quality;
            it.msg = !q ? 'Done' : (q.level === 'good' ? '✔ ' : q.level === 'warn' ? '⚠ Check this one: ' : '✖ Looks wrong: ') + q.message;
            const btn = document.createElement('button'); btn.type = 'button'; btn.className = 'secondary'; btn.textContent = 'Review';
            btn.addEventListener('click', () => openReview(it.result));
            it.row.querySelector('.q-act').appendChild(btn);
          } else { it.pct = 4 + 0.96 * d.percent; it.msg = d.message || (d.state === 'queued' ? 'Waiting in queue…' : 'Cleaning…'); }
        } catch (err) { it.state = 'error'; it.msg = err.message; }
        showRow(it);
      }
      overall();
    };
    const uploads = (async () => { for (const it of items) { if (it.state === 'cancelled') continue; await upload(it); } })();
    const timer = setInterval(poll, 1000);
    await uploads;
    while (items.some(it => it.state === 'cleaning' || it.state === 'uploading' || it.state === 'waiting')) {
      await new Promise(r => setTimeout(r, 500));
    }
    clearInterval(timer); await poll(); overall();
    const ok = items.filter(it => it.state === 'done');
    const nCancelled = items.filter(it => it.state === 'cancelled').length;
    const bad = items.length - ok.length - nCancelled;
    cancelAllRow.classList.add('hidden');
    $('batch-actions').classList.toggle('hidden', ok.length < 2);
    const nWarn = ok.filter(it => it.quality && it.quality.level === 'warn').length;
    const nBad = ok.filter(it => it.quality && it.quality.level === 'bad').length + bad;
    const cancelNote = nCancelled ? ` ${nCancelled} cancelled.` : '';
    if (!ok.length && !bad) {
      status(uploadStatus, nCancelled ? 'Cancelled.' : '');
    } else if (!bad && (nWarn || nBad)) {
      status(uploadStatus, `${ok.length} cleaned. ${nBad ? nBad + ' look completely wrong (red, blinking). ' : ''}${nWarn ? nWarn + ' may be wrong (yellow) – check them. ' : ''}${cancelNote}`, nBad ? 'error' : '');
    } else
    status(uploadStatus, (bad ? `${ok.length} cleaned, ${bad} failed. Press Review on a finished file.` : (ok.length > 1 ? `All ${ok.length} images cleaned. Press Review on any file to check and approve it.` : 'Cleaning complete. Review the result below.')) + cancelNote, bad ? 'error' : 'good');
    if (ok.length === 1) openReview(ok[0].result);
    running = false; setBusy(false);
  });

  $('download').addEventListener('click', async () => {
    if (!jobId) return;
    const button = $('download'); button.disabled = true;
    status(reviewStatus, 'Applying your edits to the full-resolution image and preparing the TIFF…');
    try {
      const response = await fetch(`/job/${jobId}/download`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({operations})
      });
      if (!response.ok) {
        let message = 'Download failed.';
        try { const data = await response.json(); message = data.error || message; } catch (_) {}
        throw new Error(message);
      }
      const blob = await response.blob();
      const disposition = response.headers.get('Content-Disposition') || '';
      const match = disposition.match(/filename="?([^";]+)"?/i);
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);
      link.href = url; link.download = match ? match[1] : 'cleaned.tif';
      document.body.appendChild(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      status(reviewStatus, 'Download started. You can keep editing or download again.', 'good');
    } catch (error) { status(reviewStatus, error.message, 'error'); }
    finally { button.disabled = false; }
  });

  $('save-edits').addEventListener('click', async () => {
    if (!jobId) return;
    const btn = $('save-edits'); btn.disabled = true;
    const id = jobId, snap = cloneOps(operations);
    try {
      const r = await fetch(`/job/${id}/save`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({operations: snap})});
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || 'Could not save the edits.');
      savedOps[id] = snap;
      if (jobId === id) { savedSnapshot = JSON.stringify(snap); updateHistoryButtons(); }
      status(reviewStatus, `Edits saved (${snap.length} change${snap.length === 1 ? '' : 's'}). They stay with this image while you work on others.`, 'good');
    } catch (error) { status(reviewStatus, error.message, 'error'); }
    finally { btn.disabled = false; }
  });
  document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's' && jobId && !reviewPanel.classList.contains('hidden')) { e.preventDefault(); $('save-edits').click(); }
  });

  $('approve').addEventListener('click', async () => {
    if (!jobId) return;
    const button = $('approve'); button.disabled = true; const jobIdAtClick = jobId;
    status(reviewStatus, 'Saving approval…');
    try {
      const response = await fetch(`/job/${jobId}/approve`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({operations})
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Approval failed.');
      status(reviewStatus, data.message, 'good');
      $('approved-count').textContent = data.count;
      savedOps[jobIdAtClick] = cloneOps(operations);
      if (jobId === jobIdAtClick) { savedSnapshot = JSON.stringify(operations); updateHistoryButtons(); }
      markApproved(jobIdAtClick);
    } catch (error) { status(reviewStatus, error.message, 'error'); }
    finally { button.disabled = false; }
  });

  // ---- several files: approve / download them all at once
  function markApproved(id) {
    for (const it of batchItems) {
      if (it.result && it.result.job_id === id) {
        it.approved = true;
        it.msg = it.msg.replace(/ · ✔ Approved$/, '') + ' · ✔ Approved';
        it.row.querySelector('.q-msg').textContent = it.msg;
      }
    }
  }
  const batchMsg = $('batch-msg');
  $('approve-all').addEventListener('click', async () => {
    const todo = batchItems.filter(it => it.state === 'done' && it.result && !it.approved);
    if (!todo.length) { batchMsg.textContent = 'Everything is already approved.'; return; }
    const lvl = it => (it.quality ? it.quality.level : 'good');
    const red = todo.filter(it => lvl(it) === 'bad'), yellow = todo.filter(it => lvl(it) === 'warn');
    const go = todo.filter(it => lvl(it) !== 'bad');
    if (!go.length) { batchMsg.textContent = 'The remaining files look wrong (red). Open them with Review first.'; return; }
    if (yellow.length && !window.confirm(`${yellow.length} file(s) may be wrong (yellow):\n` + yellow.map(it => it.file.name).join(', ') + `\n\nApprove them anyway?` + (red.length ? `\n(${red.length} red file(s) will be skipped.)` : ''))) return;
    const btns = [$('approve-all'), $('download-all')]; btns.forEach(b => b.disabled = true);
    let okN = 0, failN = 0;
    for (let i = 0; i < go.length; i++) {
      const it = go[i];
      batchMsg.textContent = `Approving ${i + 1} of ${go.length}…`;
      try {
        const r = await fetch(`/job/${it.result.job_id}/approve`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({operations: savedOps[it.result.job_id] || []})});
        const d = await r.json();
        if (!r.ok) throw new Error(d.error || 'failed');
        $('approved-count').textContent = d.count; markApproved(it.result.job_id); okN++;
      } catch (e) { failN++; }
    }
    btns.forEach(b => b.disabled = false);
    batchMsg.textContent = `${okN} approved.` + (failN ? ` ${failN} failed.` : '') + (red.length ? ` ${red.length} red skipped.` : '');
    batchMsg.className = 'muted';
  });
  $('download-all').addEventListener('click', async () => {
    const ids = batchItems.filter(it => it.state === 'done' && it.result).map(it => it.result.job_id);
    if (!ids.length) { batchMsg.textContent = 'Nothing to download.'; return; }
    const b = $('download-all'); b.disabled = true; batchMsg.textContent = 'Preparing the zip…';
    try {
      const r = await fetch('/api/download-all', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({job_ids: ids})});
      if (!r.ok) { let m = 'Download failed.'; try { m = (await r.json()).error || m; } catch (_) {} throw new Error(m); }
      const blob = await r.blob(), url = URL.createObjectURL(blob), a = document.createElement('a');
      a.href = url; a.download = 'cleaned_images.zip'; document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      batchMsg.textContent = `Downloaded ${ids.length} image(s) as cleaned_images.zip.`;
    } catch (e) { batchMsg.textContent = e.message; }
    finally { b.disabled = false; }
  });

  // ---- 'Show original': a window you can drag (by its title bar), resize (bottom-right corner) and zoom
  const origWin = $('orig-win'), origImg = $('orig-img'), origBody = $('orig-body');
  let origZoom = 100;
  function setOrigZoom(z) { origZoom = Math.max(30, Math.min(900, Math.round(z))); origImg.style.width = origZoom + '%'; $('orig-zoom-val').textContent = origZoom + '%'; }
  function loadOriginal() {
    if (!jobId) return;
    $('orig-title').textContent = 'Original — ' + ($('image-info').textContent.split(' · ')[0] || '');
    origImg.onerror = () => { origWin.classList.add('hidden'); status(reviewStatus, 'The original is not available for this file.', 'error'); };
    origImg.src = `/job/${jobId}/original?t=` + Date.now();
  }
  $('show-original').addEventListener('click', () => {
    if (!origWin.classList.contains('hidden')) { origWin.classList.add('hidden'); return; }
    setOrigZoom(100); origWin.classList.remove('hidden'); loadOriginal();
  });
  $('orig-close').addEventListener('click', () => origWin.classList.add('hidden'));
  $('orig-zoom-in').addEventListener('click', () => setOrigZoom(origZoom * 1.25));
  $('orig-zoom-out').addEventListener('click', () => setOrigZoom(origZoom / 1.25));
  $('orig-fit').addEventListener('click', () => { setOrigZoom(100); origBody.scrollTo(0, 0); });
  origBody.addEventListener('wheel', e => { if (e.ctrlKey || e.metaKey) { e.preventDefault(); setOrigZoom(origZoom * (e.deltaY < 0 ? 1.1 : 1 / 1.1)); } }, {passive: false});
  let dragWin = null;
  $('orig-head').addEventListener('pointerdown', e => {
    if (e.target.closest('button')) return;
    const r = origWin.getBoundingClientRect();
    origWin.style.left = r.left + 'px'; origWin.style.top = r.top + 'px'; origWin.style.right = 'auto';
    dragWin = {dx: e.clientX - r.left, dy: e.clientY - r.top};
    $('orig-head').setPointerCapture(e.pointerId);
  });
  $('orig-head').addEventListener('pointermove', e => {
    if (!dragWin) return;
    const w = origWin.offsetWidth;
    origWin.style.left = Math.max(-w + 80, Math.min(window.innerWidth - 80, e.clientX - dragWin.dx)) + 'px';
    origWin.style.top = Math.max(0, Math.min(window.innerHeight - 40, e.clientY - dragWin.dy)) + 'px';
  });
  const endDrag = () => { dragWin = null; };
  $('orig-head').addEventListener('pointerup', endDrag); $('orig-head').addEventListener('pointercancel', endDrag);

  $('new-image').addEventListener('click', () => {
    if (!confirmDiscard()) return;
    origWin.classList.add('hidden');
    jobId = null; baseImage = null; operations = []; redoStack = [];
    reviewPanel.classList.add('hidden'); approvedPanel.classList.add('hidden');
    uploadPanel.classList.remove('hidden'); fileInput.value = '';
    queueList.innerHTML = ''; progressShell.classList.add('hidden'); $('file-count').textContent = ''; batchItems = []; $('batch-actions').classList.add('hidden');
    status(uploadStatus, 'Choose images to start a new cleaning job.');
    window.scrollTo({top:0, behavior:'smooth'});
  });
})();
</script>
<script src="/board/board.js"></script>
</body>
</html>"""

def _cleanup_jobs():
    cutoff = datetime.now(UTC) - JOB_TTL
    with JOBS_LOCK:
        expired = [key for key, value in JOBS.items() if value["created"] < cutoff]
        for key in expired:
            del JOBS[key]

def _get_job(job_id):
    _cleanup_jobs()
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        abort(404, description="This cleaning session expired or does not exist. Please upload the image again.")
    return job

def _preview_image(image):
    image = image.convert("L")
    image.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)
    return buffer

_FONT_CACHE = {}

def _text_font(px):
    """A TrueType font of about `px` pixels (falls back to PIL's built-in font)."""
    from PIL import ImageFont
    px = max(6, int(px))
    if px in _FONT_CACHE:
        return _FONT_CACHE[px]
    font = None
    for name in ("Arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(name, px)
            break
        except OSError:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size=px)
        except TypeError:
            font = ImageFont.load_default()
    _FONT_CACHE[px] = font
    return font

def _apply_operations(base_array, operations, preview_width, preview_height, offset=(0, 0), full_size=None):
    """Replay preview-coordinate edits on a copy of the original-resolution grayscale image.
    offset/full_size let the same edits be replayed on a crop whose top-left corner is `offset` in the full image."""
    image = Image.fromarray(base_array.astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(image)
    full_w, full_h = full_size or image.size
    ox, oy = offset
    sx = full_w / max(1, preview_width)
    sy = full_h / max(1, preview_height)
    scale = (sx + sy) / 2.0

    for op in operations:
        if not isinstance(op, dict) or op.get("type") not in ("erase", "line", "polys", "text"):
            continue
        if op["type"] == "text":                                   # typed plot number
            try:
                txt = str(op.get("text", ""))[:40]
                tx = int(round(float(op["x"]) * sx)) - ox
                ty = int(round(float(op["y"]) * sy)) - oy
                fpx = float(op.get("size", 10)) * scale
            except (TypeError, ValueError, KeyError, OverflowError):
                continue
            if txt:
                font = _text_font(fpx)
                try:
                    draw.text((tx, ty), txt, fill=0, font=font, anchor="mm")
                except ValueError:
                    draw.text((tx, ty), txt, fill=0, font=font)
            continue
        if op["type"] == "polys":                                  # white-filled polygons (magic eraser)
            for poly in (op.get("polys") or [])[:5000]:
                try:
                    pp = [(int(round(float(x) * sx)) - ox, int(round(float(y) * sy)) - oy) for x, y in poly]
                except (TypeError, ValueError, IndexError, OverflowError):
                    continue
                if len(pp) >= 3:
                    draw.polygon(pp, fill=255)
            continue
        points = op.get("points", [])
        if not isinstance(points, list) or not points:
            continue
        try:
            pts = [(int(round(float(p[0]) * sx)) - ox, int(round(float(p[1]) * sy)) - oy) for p in points]
            size = max(1, int(round(float(op.get("size", 10)) * scale)))
        except (TypeError, ValueError, IndexError, OverflowError):
            continue
        colour = 255 if op["type"] == "erase" else 0
        if op["type"] == "line":
            if len(pts) >= 2:
                draw.line(pts[:2], fill=colour, width=size)
        else:
            radius = max(1, size // 2)
            if len(pts) == 1:
                x, y = pts[0]
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=colour)
            else:
                draw.line(pts, fill=colour, width=size, joint="curve")
                for x, y in (pts[0], pts[-1]):
                    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=colour)
    return image

# ---------------------------------------------------------------------------------- magic line / magic eraser
def _job_shape(job):
    return tuple(job["shape"]) if "packed" in job else job["base"].shape

def _job_crop(job, x0, y0, x1, y1):
    """Pixels x0..x1 / y0..y1 of the cleaned image, without unpacking the whole sheet."""
    h, w = _job_shape(job)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if "packed" in job:
        start, end = y0 * w, y1 * w
        b0, b1 = start // 8, (end + 7) // 8
        bits = np.unpackbits(job["packed"][b0:b1])
        off = start - b0 * 8
        bits = bits[off: off + (y1 - y0) * w].reshape(y1 - y0, w)
        return (bits[:, x0:x1] * 255).astype(np.uint8), (x0, y0)
    return job["base"][y0:y1, x0:x1].copy(), (x0, y0)

def _crop_with_edits(job, x0, y0, x1, y1, operations):
    crop, (ox, oy) = _job_crop(job, x0, y0, x1, y1)
    h, w = _job_shape(job)
    img = _apply_operations(crop, operations, job["preview_width"], job["preview_height"], offset=(ox, oy), full_size=(w, h))
    return np.array(img, dtype=np.uint8), (ox, oy)

def _nearest_pair(A, B, cap=1500):
    """Closest pixel pair between two point sets (n,2) -> (pa, pb, distance)."""
    def sub(P):
        return P if len(P) <= cap else P[np.linspace(0, len(P) - 1, cap).astype(int)]
    a, b = sub(A), sub(B)
    d = ((a[:, None, :].astype(np.int64) - b[None, :, :]) ** 2).sum(2)
    i, j = np.unravel_index(int(np.argmin(d)), d.shape)
    pa, pb = a[i], b[j]
    ra = max(25, 3 * int(np.ceil(len(A) / cap))); rb = max(25, 3 * int(np.ceil(len(B) / cap)))
    A2 = A[((A - pa) ** 2).sum(1) <= ra * ra]; B2 = B[((B - pb) ** 2).sum(1) <= rb * rb]
    a, b = sub(A2), sub(B2)
    d = ((a[:, None, :].astype(np.int64) - b[None, :, :]) ** 2).sum(2)
    i, j = np.unravel_index(int(np.argmin(d)), d.shape)
    return a[i], b[j], float(np.sqrt(d[i, j]))

def _local_width(dt, ridge, lab, cid, pt, radius=30):
    x, y = int(pt[0]), int(pt[1]); H, W = dt.shape
    x0, y0, x1, y1 = max(0, x - radius), max(0, y - radius), min(W, x + radius + 1), min(H, y + radius + 1)
    sel = ridge[y0:y1, x0:x1] & (lab[y0:y1, x0:x1] == cid)
    vals = dt[y0:y1, x0:x1][sel]
    return float(max(1.0, 2.0 * np.median(vals))) if vals.size else 3.0

def _magic_join(crop):
    """Join the broken pieces inside the box.  Returns ([(x0,y0,x1,y1,width)], message)."""
    import cv2
    ink = (crop < 128).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    ch, cw = ink.shape; big = max(ch, cw)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if a < 6: continue
        comps.append(dict(id=i, area=a, touch=(x <= 1 or y <= 1 or x + w >= cw - 1 or y + h >= ch - 1), long=max(w, h)))
    cand = [c for c in comps if c["touch"] or c["long"] >= 0.35 * big]
    if len(cand) < 2:
        cand = sorted(comps, key=lambda c: -c["area"])[:2]
    cand = sorted(cand, key=lambda c: -c["area"])[:8]
    if len(cand) < 2:
        return [], "I could not find two separate pieces inside that box. Draw the box around the gap, covering a bit of both pieces."
    pts = {c["id"]: np.argwhere(lab == c["id"])[:, ::-1] for c in cand}
    edges = []
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            pa, pb, d = _nearest_pair(pts[cand[i]["id"]], pts[cand[j]["id"]], cap=800)
            edges.append((d, cand[i]["id"], cand[j]["id"]))
    edges.sort()
    group = {c["id"]: c["id"] for c in cand}
    def find(a):
        while group[a] != a:
            group[a] = group[group[a]]; a = group[a]
        return a
    chosen = []
    for d, a, b in edges:
        if find(a) == find(b): continue
        if chosen and d > 0.6 * big: continue
        group[find(a)] = find(b); chosen.append((a, b))
        if len(chosen) >= 4: break
    if not chosen:
        return [], "Those pieces are already connected."
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (ink > 0) & (dt >= 1)
    lines = []
    for a, b in chosen:
        pa, pb, d = _nearest_pair(pts[a], pts[b])
        wa = _local_width(dt, ridge, lab, a, pa); wb = _local_width(dt, ridge, lab, b, pb)
        def centre(P, p, wloc):
            r = max(2.0, 1.25 * wloc)
            m = ((P - p) ** 2).sum(1) <= r * r
            return P[m].mean(0) if m.any() else p.astype(float)
        ca, cb = centre(pts[a], pa, wa), centre(pts[b], pb, wb)
        lines.append((float(ca[0]), float(ca[1]), float(cb[0]), float(cb[1]), max(1, int(round(min(wa, wb))))))
    return lines, ""

def _magic_erase(crop, cx, cy, big_full):
    """Remove the number (or the piece of line) under the click.  Returns (polygons in crop px, kind, message)."""
    import cv2
    ink = (crop < 128).astype(np.uint8)
    H, W = ink.shape
    rs = int(max(14, 0.003 * big_full))
    x0, y0, x1, y1 = max(0, cx - rs), max(0, cy - rs), min(W, cx + rs + 1), min(H, cy + rs + 1)
    ys, xs = np.nonzero(ink[y0:y1, x0:x1])
    if not len(xs):
        return [], "", "There is no ink under the click. Click right on the number or line."
    k = int(np.argmin((xs + x0 - cx) ** 2 + (ys + y0 - cy) ** 2))
    px, py = int(xs[k] + x0), int(ys[k] + y0)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    cid = int(lab[py, px])
    longs = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
    text_max = 0.011 * big_full
    if longs[cid] <= text_max:                                          # a number, a decimal point, a small mark
        textlike = (longs <= text_max); textlike[0] = False
        mask_text = textlike[lab].astype(np.uint8)
        # how far apart characters of ONE number are depends on the character size near the click
        cxs = st[:, cv2.CC_STAT_LEFT] + st[:, cv2.CC_STAT_WIDTH] / 2.0; cys = st[:, cv2.CC_STAT_TOP] + st[:, cv2.CC_STAT_HEIGHT] / 2.0
        near_t = textlike & (((cxs - px) ** 2 + (cys - py) ** 2) <= (2.5 * text_max) ** 2)
        T = float(longs[near_t].max()) if near_t.any() else 20.0
        kx = int(max(9, 0.5 * T)); ky = int(max(5, 0.25 * T))
        dil = cv2.dilate(mask_text, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
        _, dlab = cv2.connectedComponents(dil, connectivity=8)
        grp = (dlab == dlab[py, px]) & (mask_text > 0)
        ys2, xs2 = np.nonzero(grp)
        if (xs2.max() - xs2.min()) > 6 * text_max or (ys2.max() - ys2.min()) > 6 * text_max:
            grp = (lab == cid)
        m = cv2.dilate(grp.astype(np.uint8), np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = [cv2.approxPolyDP(c, 0.8, True)[:, 0, :].astype(float).tolist() for c in cnts if len(c) >= 3]
        return polys, "text", ""
    # a long line: follow it from the click in both directions until it ends or meets another line
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (ink > 0) & (dt >= 1)
    w = _local_width(dt, ridge, lab, cid, (px, py), radius=40)
    R = int(max(12, 2.5 * w))
    wx0, wy0, wx1, wy1 = max(0, px - R), max(0, py - R), min(W, px + R + 1), min(H, py + R + 1)
    yy, xx = np.nonzero(lab[wy0:wy1, wx0:wx1] == cid)
    P = np.stack([xx + wx0, yy + wy0], 1).astype(float)
    P = P[((P - [px, py]) ** 2).sum(1) <= R * R]
    if len(P) < 5:
        return [], "", "Could not tell the direction of that line."
    Pc = P - P.mean(0)
    ev, evec = np.linalg.eigh(np.cov(Pc.T))
    d0 = evec[:, int(np.argmax(ev))]; d0 = d0 / (np.linalg.norm(d0) + 1e-9)
    def run(sign):
        d = d0 * sign; n_ = np.array([-d[1], d[0]]); p = np.array([float(px), float(py)]); path = []; miss = 0
        S = int(w) + 6; offs = np.arange(-S, S + 1)
        for _ in range(3200):
            p = p + d
            xs_ = np.rint(p[0] + n_[0] * offs).astype(int); ys_ = np.rint(p[1] + n_[1] * offs).astype(int)
            ok = (xs_ >= 0) & (xs_ < W) & (ys_ >= 0) & (ys_ < H)
            if not ok[S]: break
            v = np.zeros(len(offs), bool); v[ok] = ink[ys_[ok], xs_[ok]] > 0
            idx = np.nonzero(v)[0]
            if not len(idx) or np.min(np.abs(idx - S)) > 2:
                miss += 1
                if miss > 3: break
                continue
            miss = 0
            c = idx[np.argmin(np.abs(idx - S))]
            a_ = c
            while a_ > 0 and v[a_ - 1]: a_ -= 1
            b_ = c
            while b_ < len(v) - 1 and v[b_ + 1]: b_ += 1
            if (b_ - a_ + 1) > 1.8 * w + 3: break                  # the line widens: another line crosses here
            p = p + n_ * (0.5 * ((a_ + b_) / 2.0 - S))
            path.append(p.copy())
        return path[:-1] if miss == 0 and path else path
    fwd, back = run(1), run(-1)
    if len(fwd) + len(back) < 3 * max(1.0, w):
        return [], "", "That piece of line is too short to remove this way. Use Brush erase for it."
    path = np.array(list(reversed(back)) + [[px, py]] + fwd, float)
    simp = cv2.approxPolyDP(path.astype(np.float32).reshape(-1, 1, 2), 1.5, False)[:, 0, :].astype(float)
    if len(simp) < 2: simp = np.array([path[0], path[-1]])
    h = w / 2.0 + 2.0
    nrm = np.array([-d0[1], d0[0]])
    poly = [(q + nrm * h).tolist() for q in simp] + [(q - nrm * h).tolist() for q in simp[::-1]]
    return [poly], "line", ""

@app.route("/job/<job_id>/magic", methods=["POST"])
def job_magic(job_id):
    job = _get_job(job_id)
    payload = request.get_json(silent=True) or {}
    operations = _get_operations()
    mode = payload.get("mode")
    H, W = _job_shape(job)
    pw, ph = job["preview_width"], job["preview_height"]
    sx, sy = W / max(1, pw), H / max(1, ph)
    scale = (sx + sy) / 2.0
    big_full = float(max(W, H))
    try:
        if mode == "join":
            r = [float(v) for v in payload.get("rect")]
            x0, y0, x1, y1 = int(r[0] * sx) - 6, int(r[1] * sy) - 6, int(r[2] * sx) + 6, int(r[3] * sy) + 6
            if (x1 - x0) > 4000 or (y1 - y0) > 4000:
                return jsonify(ops=[], message="That box is too big. Draw it around just the gap.")
            crop, (ox, oy) = _crop_with_edits(job, x0, y0, x1, y1, operations)
            lines, msg = _magic_join(crop)
            if not lines:
                return jsonify(ops=[], message=msg)
            ops = [{"type": "line", "magic": True, "size": round(w_ / scale, 4),
                    "points": [[round((a + ox) / sx, 3), round((b + oy) / sy, 3)], [round((c + ox) / sx, 3), round((d + oy) / sy, 3)]]}
                   for a, b, c, d, w_ in lines]
            return jsonify(ops=ops, message=f"Joined {len(ops)} gap{'s' if len(ops) > 1 else ''}. Undo if it is not right.")
        if mode == "erase":
            px_, py_ = [float(v) for v in payload.get("point")]
            cx, cy = int(px_ * sx), int(py_ * sy)
            R = 1500
            crop, (ox, oy) = _crop_with_edits(job, cx - R, cy - R, cx + R, cy + R, operations)
            polys, kind, msg = _magic_erase(crop, cx - ox, cy - oy, big_full)
            if not polys:
                return jsonify(ops=[], message=msg or "Nothing to remove there.")
            polys = [[[round((x + ox) / sx, 3), round((y + oy) / sy, 3)] for x, y in poly] for poly in polys]
            return jsonify(ops=[{"type": "polys", "magic": True, "polys": polys}],
                           message="Removed the number." if kind == "text" else "Removed that piece of line. Undo if it is not right.")
    except (TypeError, ValueError, IndexError) as exc:
        return jsonify(ops=[], message=f"Bad request: {type(exc).__name__}"), 400
    return jsonify(ops=[], message="Unknown tool."), 400

def _store_original_preview(result, original):
    """Keep a preview of the ORIGINAL sheet (same size as the cleaned preview) for the 'Show original' window."""
    pw, ph = result["preview_width"], result["preview_height"]
    g = original.convert("L")
    f = max(1, int(g.size[0] // max(1, pw * 2)))
    if f > 1: g = g.reduce(f)
    g = g.resize((pw, ph), Image.Resampling.LANCZOS)
    buf = io.BytesIO(); g.save(buf, format="PNG")
    with JOBS_LOCK:
        job = JOBS.get(result["job_id"])
        if job is not None: job["orig_preview"] = buf.getvalue()

@app.route("/job/<job_id>/original", methods=["GET"])
def job_original(job_id):
    job = _get_job(job_id)
    if not job.get("orig_preview"):
        abort(404, description="The original is not available for this file.")
    return send_file(io.BytesIO(job["orig_preview"]), mimetype="image/png", max_age=0)

@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/job/") or request.path == "/preview":
        return jsonify(error=getattr(error, "description", "Not found.")), 404
    return "Not found", 404

@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="The uploaded file is larger than the 1 GB limit."), 413

@app.route("/", methods=["GET"])
def index():
    return render_template_string(PAGE)

def _job_base(job):
    """Cleaned image of a job as a uint8 array (stored bit-packed when it is pure black & white)."""
    if "packed" in job:
        h, w = job["shape"]
        bits = np.unpackbits(job["packed"], count=h*w).reshape(h, w)
        return (bits * 255).astype(np.uint8)
    return job["base"]

# ---- result check: green = fine, warn = might be wrong, bad = completely wrong (thresholds are easy to tune here)
Q_WIPED = 0.25        # cleaned image keeps less than this share of the original ink  -> bad
Q_LOW_INK = 0.50      # ... less than this -> warning
Q_NOTHING = 0.985     # ... more than this = almost nothing was removed -> warning
Q_BORDER_BAD = 0.65   # share of the thick boundary that survived, below -> bad
Q_BORDER_WARN = 0.85  # ... below -> warning
Q_LINES_BAD = 0.60   # share of the long thin plot lines that survived, below -> bad
Q_LINES_WARN = 0.75  # ... below -> warning
Q_GAP_MIN = 0.045    # a missing stretch of border/line at least this long (share of the sheet's long side) counts as a gap
Q_GAP_WARN = 1       # this many gaps or more -> warning
Q_GAP_BAD = 4        # this many gaps (or one gap longer than Q_GAP_HUGE) -> bad
Q_GAP_HUGE = 0.20
Q_ADDED = 0.04        # new ink that was not in the original (share of original ink) -> warning

def _assess(original, cleaned):
    """Compare the original sheet with the cleaned one.  Returns dict(level, message, stats)."""
    import cv2
    try:
        def small(img):
            g = img.convert("L"); f = max(1, -(-max(g.size) // 2000))
            return np.array(g.reduce(f) if f > 1 else g, dtype=np.uint8)
        o = small(original)
        c = small(Image.fromarray(cleaned))
        if c.shape != o.shape:
            c = cv2.resize(c, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_AREA)
        oi, ci = o < 235, c < 235
        n_o, n_c = int(oi.sum()), int(ci.sum())
        if n_o < 500:
            return dict(level="warn", message="The original looks almost empty - nothing to check against.", stats={})
        ratio = n_c / n_o
        added = int((ci & ~cv2.dilate(oi.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)).sum()) / n_o
        k = np.ones((5, 5), np.uint8)
        to = cv2.morphologyEx(oi.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
        tc = cv2.morphologyEx(ci.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
        border = None
        if int(to.sum()) > 300:
            near = cv2.dilate(tc.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            border = float((to & near).sum()) / float(to.sum())
        # long thin lines (plot boundaries): opened with line-shaped kernels at many angles; a line cut into pieces fails the opening
        def long_lines(m):
            out = np.zeros(m.shape, bool); u = m.astype(np.uint8); L = 41
            for a in range(0, 180, 15):
                kk = np.zeros((L, L), np.uint8); r = a * np.pi / 180.0; cx = L // 2
                cv2.line(kk, (int(cx - cx * np.cos(r)), int(cx - cx * np.sin(r))), (int(cx + cx * np.cos(r)), int(cx + cx * np.sin(r))), 1, 1)
                out |= cv2.morphologyEx(u, cv2.MORPH_OPEN, kk).astype(bool)
            return out
        lo_, lc_ = long_lines(oi), long_lines(ci)
        lines = None
        if int(lo_.sum()) > 300:
            lines = float((lo_ & cv2.dilate(lc_.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)).sum()) / float(lo_.sum())
        # individual gaps: thick border / long lines that existed in the original but have a missing stretch now
        side = float(max(o.shape)); gaps = []
        miss = ((to | lo_) & ~cv2.dilate((tc | lc_).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)).astype(np.uint8)
        miss = cv2.dilate(miss, np.ones((5, 5), np.uint8))
        n_cc, _, st, _ = cv2.connectedComponentsWithStats(miss, connectivity=8)
        for i in range(1, n_cc):
            ln = float(np.hypot(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])) / side
            if ln >= Q_GAP_MIN: gaps.append(ln)
        stats = dict(gaps=len(gaps), longest_gap=round(max(gaps), 3) if gaps else 0, ink_kept=round(ratio, 3), border_kept=None if border is None else round(border, 3), lines_kept=None if lines is None else round(lines, 3), added_ink=round(added, 3))
        bad, warn = [], []
        if n_c == 0 or ratio < Q_WIPED: bad.append(f"most of the drawing is gone ({ratio:.0%} of the ink left)")
        elif ratio < Q_LOW_INK: warn.append(f"a lot was removed ({ratio:.0%} of the ink left)")
        if border is not None:
            if border < Q_BORDER_BAD: bad.append(f"the thick border is broken ({border:.0%} kept)")
            elif border < Q_BORDER_WARN: warn.append(f"part of the thick border is missing ({border:.0%} kept)")
        if lines is not None:
            if lines < Q_LINES_BAD: bad.append(f"many plot lines are missing or cut ({lines:.0%} kept)")
            elif lines < Q_LINES_WARN: warn.append(f"some plot lines are missing or cut ({lines:.0%} kept)")
        if gaps:
            msg = f"{len(gaps)} gap{'s' if len(gaps) > 1 else ''} in the border/plot lines (longest {max(gaps):.0%} of the sheet)"
            if len(gaps) >= Q_GAP_BAD or max(gaps) >= Q_GAP_HUGE: bad.append(msg)
            elif len(gaps) >= Q_GAP_WARN: warn.append(msg)
        if ratio > Q_NOTHING: warn.append("almost nothing was removed - the numbers may still be there")
        if added > Q_ADDED: warn.append("new marks appeared that were not in the original")
        if bad: return dict(level="bad", message="; ".join(bad + warn).capitalize() + ".", stats=stats)
        if warn: return dict(level="warn", message="; ".join(warn).capitalize() + ".", stats=stats)
        return dict(level="good", message="Cleaned well - border and plot lines kept.", stats=stats)
    except Exception as exc:
        return dict(level="warn", message=f"Could not check the result automatically ({type(exc).__name__}).", stats={})

def _make_job(cleaned, original_size, dpi, filename, bw):
    cleaned_image = Image.fromarray(cleaned).convert("L")
    if bw:
        cleaned_image = cleaned_image.point(lambda value: 255 if value > 140 else 0).convert("1").convert("L")
    base_array = np.array(cleaned_image, dtype=np.uint8)
    preview_bytes = _preview_image(cleaned_image).getvalue()
    preview_size = Image.open(io.BytesIO(preview_bytes)).size
    job_id = uuid.uuid4().hex
    safe_name = os.path.basename(filename)
    job = {
        "created": datetime.now(UTC), "filename": safe_name,
        "width": original_size[0], "height": original_size[1], "dpi": dpi,
        "preview_width": preview_size[0], "preview_height": preview_size[1],
        "preview": preview_bytes, "approved": False, "operations": [],
    }
    if bw:
        job["packed"] = np.packbits(base_array > 127); job["shape"] = base_array.shape
    else:
        job["base"] = base_array
    with JOBS_LOCK:
        JOBS[job_id] = job
    return dict(job_id=job_id, filename=safe_name, width=original_size[0], height=original_size[1],
                preview_width=preview_size[0], preview_height=preview_size[1],
                preview_url=f"/job/{job_id}/preview")

@app.route("/preview", methods=["POST"])
def create_preview():
    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename:
        return jsonify(error="Choose an image first."), 400
    try:
        original = Image.open(uploaded.stream)
        original.load()
        cleaned = clean(original, log=print)
        return jsonify(_make_job(cleaned, original.size, original.info.get("dpi"), uploaded.filename, "bw" in request.form))
    except Exception as exc:
        app.logger.exception("Failed to clean uploaded image")
        return jsonify(error=f"Could not process image: {type(exc).__name__}: {exc}"), 400

# ---- background cleaning queue: any number of images, one at a time (big sheets need several GB each) ----
TASKS = {}
TASK_QUEUE = queue.Queue()
_WORKER = {"thread": None}
UPLOAD_TMP = tempfile.mkdtemp(prefix="plotclean_")
MAX_PARALLEL = max(1, int(os.environ.get("PLOT_WORKERS", min(4, os.cpu_count() or 2))))
_SLOTS = threading.Condition()
_ACTIVE = {"n": 0, "mem": 0}

def _total_memory():
    try:
        import psutil
        return psutil.virtual_memory().total
    except Exception:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        pass
    try:
        import subprocess
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
    except Exception:
        return 8 * 2**30

def _need_bytes(path):
    """Rough peak memory of cleaning one sheet (~36 bytes per pixel, sheets above ~150 M px are cleaned on a reduced copy)."""
    try:
        with Image.open(path) as im:
            px = im.size[0] * im.size[1]
    except Exception:
        px = 140_000_000
    return int(min(px, 140_000_000) * 36)

class _Cancelled(Exception):
    pass

def _run_task(task, need):
    def check():
        if task.get("cancel"): raise _Cancelled()
    try:
        check()
        task["state"] = "running"; task["message"] = "Opening the image"; task["percent"] = 1
        def progress(pct, msg=""):
            check()
            task["percent"] = max(task["percent"], min(99, float(pct)))
            if msg: task["message"] = msg
        original = Image.open(task["path"]); original.load()
        size, dpi = original.size, original.info.get("dpi")
        cleaned = clean(original, log=lambda *a, **k: (check(), print(*a, **k)), progress=progress)
        check()
        task["message"] = "Checking the result"; task["percent"] = max(task["percent"], 95)
        quality = _assess(original, cleaned)
        task["message"] = "Preparing the preview"; task["percent"] = max(task["percent"], 97)
        task["result"] = _make_job(cleaned, size, dpi, task["filename"], task["bw"])
        task["result"]["quality"] = quality
        try: _store_original_preview(task["result"], original)
        except Exception: app.logger.exception("Could not keep the original preview")
        del original
        task["percent"] = 100; task["state"] = "done"; task["message"] = "Done"
    except _Cancelled:
        task["state"] = "cancelled"; task["message"] = "Cancelled"; task["result"] = None
    except Exception as exc:
        app.logger.exception("Cleaning failed")
        task["state"] = "error"; task["error"] = f"Could not process image: {type(exc).__name__}: {exc}"
    finally:
        try: os.remove(task["path"])
        except OSError: pass
        with _SLOTS:
            _ACTIVE["n"] -= 1; _ACTIVE["mem"] -= need
            _SLOTS.notify_all()

def _dispatcher():
    """Start queued images in parallel, as many as the CPU count and the machine's memory allow."""
    budget = 0.75 * _total_memory()
    while True:
        task = TASK_QUEUE.get()
        if task.get("cancel"):                       # cancelled while waiting in the queue: never start it
            try: os.remove(task["path"])
            except OSError: pass
            continue
        need = _need_bytes(task["path"])
        with _SLOTS:
            while _ACTIVE["n"] >= MAX_PARALLEL or (_ACTIVE["n"] > 0 and _ACTIVE["mem"] + need > budget):
                _SLOTS.wait()
            _ACTIVE["n"] += 1; _ACTIVE["mem"] += need
        threading.Thread(target=_run_task, args=(task, need), daemon=True).start()

def _ensure_worker():
    th = _WORKER["thread"]
    if th is None or not th.is_alive():
        th = threading.Thread(target=_dispatcher, daemon=True); th.start(); _WORKER["thread"] = th

@app.route("/api/clean", methods=["POST"])
def api_clean():
    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename:
        return jsonify(error="Choose an image first."), 400
    task_id = uuid.uuid4().hex
    path = os.path.join(UPLOAD_TMP, task_id + os.path.splitext(uploaded.filename)[1].lower())
    uploaded.save(path)
    task = {"id": task_id, "state": "queued", "percent": 0.0, "message": "Waiting in queue", "filename": uploaded.filename,
            "path": path, "bw": "bw" in request.form, "result": None, "error": None, "created": datetime.now(UTC)}
    TASKS[task_id] = task
    for k in [k for k, v in TASKS.items() if v["created"] < datetime.now(UTC) - JOB_TTL]: TASKS.pop(k, None)
    _ensure_worker(); TASK_QUEUE.put(task)
    return jsonify(task_id=task_id)

@app.route("/api/task/<task_id>/cancel", methods=["POST"])
def api_task_cancel(task_id):
    task = TASKS.get(task_id)
    if task is None:
        return jsonify(error="This cleaning job no longer exists."), 404
    if task["state"] in ("done", "error", "cancelled"):
        return jsonify(state=task["state"])
    task["cancel"] = True                            # a running job stops at its next progress step
    if task["state"] == "queued":
        task["state"] = "cancelled"; task["message"] = "Cancelled"
    return jsonify(state=task["state"])

@app.route("/api/task/<task_id>", methods=["GET"])
def api_task(task_id):
    task = TASKS.get(task_id)
    if task is None:
        return jsonify(error="This cleaning job no longer exists."), 404
    return jsonify(state=task["state"], percent=round(task["percent"], 1), message=task["message"],
                   error=task["error"], result=task["result"] if task["state"] == "done" else None)

@app.route("/job/<job_id>/preview", methods=["GET"])
def job_preview(job_id):
    job = _get_job(job_id)
    return send_file(io.BytesIO(job["preview"]), mimetype="image/png", max_age=0)

def _get_operations():
    payload = request.get_json(silent=True) or {}
    operations = payload.get("operations", [])
    if not isinstance(operations, list) or len(operations) > 10000:
        abort(400, description="Invalid edit history.")
    return operations

def _encode_job_tiff(job, operations):
    image = _apply_operations(_job_base(job), operations, job["preview_width"], job["preview_height"])
    buffer = io.BytesIO()
    options = {"compression": "group4"} if image.getextrema() in ((0, 255), (0, 0), (255, 255)) else {"compression": "tiff_lzw"}
    if options.get("compression") == "group4":
        image = image.point(lambda value: 255 if value > 140 else 0).convert("1")
    if job.get("dpi") and min(job["dpi"]) > 1:
        options["dpi"] = job["dpi"]
    image.save(buffer, format="TIFF", **options)
    buffer.seek(0)
    return buffer

@app.route("/api/download-all", methods=["POST"])
def download_all():
    """Every listed cleaned image (no manual edits) in one .zip, so the browser asks for a single download."""
    ids = (request.get_json(silent=True) or {}).get("job_ids") or []
    if not isinstance(ids, list) or not ids or len(ids) > 500:
        return jsonify(error="No images to download."), 400
    tmp = tempfile.SpooledTemporaryFile(max_size=256 * 1024 * 1024)
    used, n = set(), 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:             # TIFFs are already compressed
        for jid in ids:
            with JOBS_LOCK:
                job = JOBS.get(str(jid))
            if job is None:
                continue
            base = os.path.splitext(job["filename"])[0] or "survey"
            name, k = f"{base}_cleaned.tif", 2
            while name in used:
                name = f"{base}_cleaned_{k}.tif"; k += 1
            used.add(name)
            z.writestr(name, _encode_job_tiff(job, job.get("operations") or []).getvalue()); n += 1
    if not n:
        tmp.close()
        return jsonify(error="These images expired. Please clean them again."), 404
    tmp.seek(0)
    return send_file(tmp, mimetype="application/zip", as_attachment=True, download_name="cleaned_images.zip", max_age=0)

@app.route("/job/<job_id>/save", methods=["POST"])
def save_job_edits(job_id):
    """Keep the manual edits of one image (also used by 'Download all')."""
    job = _get_job(job_id)
    job["operations"] = _get_operations()
    return jsonify(ok=True, count=len(job["operations"]))

@app.route("/job/<job_id>/download", methods=["POST"])
def download_job(job_id):
    job = _get_job(job_id)
    operations = _get_operations()
    buffer = _encode_job_tiff(job, operations)
    base = os.path.splitext(job["filename"])[0] or "survey"
    return send_file(buffer, mimetype="image/tiff", as_attachment=True, download_name=f"{base}_cleaned.tif", max_age=0)

def _approved_records():
    os.makedirs(APPROVED_DIR, exist_ok=True)
    records = []
    for filename in os.listdir(APPROVED_DIR):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(APPROVED_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as stream:
                record = json.load(stream)
            if os.path.isfile(os.path.join(APPROVED_DIR, record.get("image_file", ""))):
                records.append(record)
        except (OSError, ValueError, TypeError):
            app.logger.exception("Could not read approved image metadata: %s", path)
    records.sort(key=lambda item: item.get("approved_at", ""), reverse=True)
    return records

@app.route("/api/approved", methods=["GET"])
def list_approved():
    records = _approved_records()
    notes = _puzzle_notes()
    order = _order_map()
    items = []
    for record in records:
        item_id = record["id"]
        items.append({
            "order": order.get(item_id),
            "puzzle_note": notes.get(item_id),
            "id": item_id,
            "filename": record["original_filename"],
            "width": record["width"],
            "height": record["height"],
            "approved_at": record["approved_at"],
            "preview_url": url_for("approved_preview", item_id=item_id),
            "download_url": url_for("approved_download", item_id=item_id),
        })
    return jsonify(images=items)

def _find_approved(item_id):
    for record in _approved_records():
        if record.get("id") == item_id:
            return record
    abort(404, description="Approved image not found.")

@app.route("/approved/<item_id>/preview", methods=["GET"])
def approved_preview(item_id):
    record = _find_approved(item_id)
    path = os.path.join(APPROVED_DIR, record["preview_file"])
    return send_file(path, mimetype="image/png", max_age=0)

@app.route("/approved/<item_id>/download", methods=["GET"])
def approved_download(item_id):
    record = _find_approved(item_id)
    path = os.path.join(APPROVED_DIR, record["image_file"])
    return send_file(path, mimetype="image/tiff", as_attachment=True, download_name=record["image_file"], max_age=0)

@app.route("/job/<job_id>/approve", methods=["POST"])
def approve_job(job_id):
    job = _get_job(job_id)
    operations = _get_operations()
    final_image = _apply_operations(_job_base(job), operations, job["preview_width"], job["preview_height"])
    approved_id = uuid.uuid4().hex
    base = os.path.splitext(job["filename"])[0] or "survey"
    safe_base = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in base)[:100]
    image_filename = f"{safe_base}_approved_{approved_id[:8]}.tif"
    preview_filename = f"{safe_base}_approved_{approved_id[:8]}.png"
    metadata_filename = f"{approved_id}.json"
    image_path = os.path.join(APPROVED_DIR, image_filename)
    preview_path = os.path.join(APPROVED_DIR, preview_filename)
    metadata_path = os.path.join(APPROVED_DIR, metadata_filename)

    try:
        save_image = final_image
        options = {"compression": "group4"} if save_image.getextrema() in ((0, 255), (0, 0), (255, 255)) else {"compression": "tiff_lzw"}
        if options.get("compression") == "group4":
            save_image = save_image.point(lambda value: 255 if value > 140 else 0).convert("1")
        if job.get("dpi") and min(job["dpi"]) > 1:
            options["dpi"] = job["dpi"]
        save_image.save(image_path, format="TIFF", **options)
        preview_buffer = _preview_image(final_image)
        with open(preview_path, "wb") as stream:
            stream.write(preview_buffer.getvalue())
        approved_at = datetime.now().astimezone().isoformat()
        record = {
            "id": approved_id,
            "original_filename": job["filename"],
            "image_file": image_filename,
            "preview_file": preview_filename,
            "width": job["width"],
            "height": job["height"],
            "approved_at": approved_at,
            "status": "approved",
            "puzzle_status": "pending_implementation",
            "operations": operations,
        }
        with open(metadata_path, "w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
    except Exception as exc:
        for path in (image_path, preview_path, metadata_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        app.logger.exception("Failed to save approved image")
        return jsonify(error=f"Could not save approved image: {type(exc).__name__}: {exc}"), 500

    job["operations"] = operations
    job["approved"] = True
    job["approved_at"] = approved_at
    job["approved_id"] = approved_id
    count = len(_approved_records())
    return jsonify(
        approved=True,
        count=count,
        message=f"Approved image saved. It is now available in Approved Images ({count} total). Open Approved Images and press Build puzzle to attach it to its neighbouring sheets."
    )

# ----------------------------------------------------------------------------------
# automatic puzzle
# ----------------------------------------------------------------------------------
def _update_record(item_id, mutate):
    record = _find_approved(item_id)
    path = os.path.join(APPROVED_DIR, f"{item_id}.json")
    mutate(record)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    return record

@app.route("/api/approved", methods=["DELETE"])
def delete_all_approved():
    with PUZZLE_LOCK:
        if PUZZLE["state"] == "running":
            return jsonify(error="A puzzle is currently building. Wait for it to finish, then delete the images."), 409
    records = _approved_records()
    deleted, failed = 0, 0
    board_dir = os.path.join(PUZZLE_DIR, "board")
    for record in records:
        item_id = record.get("id", "")
        paths = [
            os.path.join(APPROVED_DIR, record.get("image_file", "")),
            os.path.join(APPROVED_DIR, record.get("preview_file", "")),
            os.path.join(APPROVED_DIR, f"{item_id}.json"),
            os.path.join(PUZZLE_DIR, f"{item_id}.puzzle.pkl"),
            os.path.join(board_dir, f"{item_id}.json"),
            os.path.join(board_dir, f"{item_id}.png"),
        ]
        try:
            for path in paths:
                if os.path.isfile(path):
                    os.remove(path)
            deleted += 1
        except OSError:
            app.logger.exception("Failed to delete approved image %s", item_id)
            failed += 1
    try:
        latest_dir = os.path.join(PUZZLE_DIR, "latest")
        if os.path.isdir(latest_dir):
            for name in os.listdir(latest_dir):
                path = os.path.join(latest_dir, name)
                if os.path.isfile(path):
                    os.remove(path)
            os.rmdir(latest_dir)
        _save_manual({})
    except OSError:
        app.logger.exception("Could not clear the old puzzle output")
    with PUZZLE_LOCK:
        PUZZLE.update(state="idle", stage="", fraction=0.0,
                      message="Approved images changed. Build the puzzle again.", report=None, error=None)
    if failed:
        return jsonify(error=f"Deleted {deleted} image(s), but {failed} could not be deleted."), 500
    return jsonify(ok=True, deleted=deleted, message=f"Deleted all {deleted} approved images.")

@app.route("/api/approved/<item_id>", methods=["DELETE"])
def delete_approved(item_id):
    # Do not remove a source sheet while the background puzzle builder may be reading it.
    with PUZZLE_LOCK:
        if PUZZLE["state"] == "running":
            return jsonify(error="A puzzle is currently building. Wait for it to finish, then delete the image."), 409

    record = _find_approved(item_id)
    paths = [
        os.path.join(APPROVED_DIR, record.get("image_file", "")),
        os.path.join(APPROVED_DIR, record.get("preview_file", "")),
        os.path.join(APPROVED_DIR, f"{item_id}.json"),
        os.path.join(PUZZLE_DIR, f"{item_id}.puzzle.pkl"),
    ]
    try:
        for path in paths:
            # Only delete files directly in the expected application directories.
            if os.path.isfile(path):
                os.remove(path)

        # Existing assembled output is now stale; remove it so it cannot be mistaken
        # for a puzzle built from the current approved-image set.
        latest_dir = os.path.join(PUZZLE_DIR, "latest")
        if os.path.isdir(latest_dir):
            for name in os.listdir(latest_dir):
                path = os.path.join(latest_dir, name)
                if os.path.isfile(path):
                    os.remove(path)
            os.rmdir(latest_dir)

        with PUZZLE_LOCK:
            PUZZLE.update(
                state="idle", stage="", fraction=0.0,
                message="Approved images changed. Build the puzzle again.",
                report=None, error=None
            )
    except OSError as exc:
        app.logger.exception("Failed to delete approved image %s", item_id)
        return jsonify(error=f"Could not delete approved image: {exc}"), 500

    return jsonify(ok=True, message=f"Deleted {record['original_filename']}. It will not be included in the next puzzle build.")

@app.route("/api/approved/<item_id>/meta", methods=["POST"])
def set_approved_meta(item_id):
    payload = request.get_json(silent=True) or {}
    clean_meta = {}
    try:
        if payload.get("scale") not in (None, ""):
            scale = int(payload["scale"])
            if not 50 <= scale <= 200000:
                raise ValueError
            clean_meta["scale"] = scale
            clean_meta["scale_source"] = "manual"
    except (TypeError, ValueError):
        return jsonify(error="Scale must be a whole number such as 936."), 400
    if payload.get("survey"):
        clean_meta["survey"] = str(payload["survey"])[:20]
    if payload.get("village"):
        clean_meta["village"] = str(payload["village"])[:80]
    _update_record(item_id, lambda rec: rec.setdefault("meta", {}).update(clean_meta))
    return jsonify(ok=True)

def _puzzle_worker(items):
    def progress(stage, fraction, message):
        with PUZZLE_LOCK:
            PUZZLE.update(stage=stage, fraction=fraction, message=message)
    try:
        out_dir = os.path.join(PUZZLE_DIR, "latest_tmp")
        if os.path.isdir(out_dir):
            for name in os.listdir(out_dir):
                os.remove(os.path.join(out_dir, name))
        report = puzzle.build_puzzle(items, out_dir, progress=progress, cache_dir=PUZZLE_DIR, manual=_load_manual())
        # Persist successful OCR fields so the gallery, later puzzle builds, and labels share metadata.
        for detected in report.get("sheet_metadata", []):
            item_id = detected.get("id")
            if not item_id:
                continue
            try:
                record = _find_approved(item_id)
                existing = record.setdefault("meta", {})
                changed = False
                for key in ("scale", "survey", "village", "district", "taluk", "area_m2"):
                    value = detected.get(key)
                    if value not in (None, "") and existing.get(key) in (None, ""):
                        existing[key] = value
                        changed = True
                if changed:
                    with open(os.path.join(APPROVED_DIR, f"{item_id}.json"), "w", encoding="utf-8") as stream:
                        json.dump(record, stream, ensure_ascii=False, indent=2)
            except Exception:
                app.logger.exception("Could not persist detected header metadata for %s", item_id)
        final_dir = os.path.join(PUZZLE_DIR, "latest")
        if os.path.isdir(final_dir):
            for name in os.listdir(final_dir):
                os.remove(os.path.join(final_dir, name))
            os.rmdir(final_dir)
        os.replace(out_dir, final_dir)
        with PUZZLE_LOCK:
            PUZZLE.update(state="done", fraction=1.0, message="Puzzle ready.", report=report, error=None)
    except Exception as exc:
        app.logger.exception("Puzzle build failed")
        with PUZZLE_LOCK:
            PUZZLE.update(state="error", error=f"Puzzle failed: {type(exc).__name__}: {exc}")

@app.route("/api/puzzle/build", methods=["POST"])
def puzzle_build():
    records = _approved_records()
    if len(records) < 2:
        return jsonify(error="Approve at least two sheets first."), 400
    order = _order_map()
    items = [{"id": r["id"], "name": r["original_filename"], "path": os.path.join(APPROVED_DIR, r["image_file"]),
              "meta": r.get("meta") or {}, "order": order.get(r["id"], 0)} for r in records]
    items.sort(key=lambda it: it["order"])
    with PUZZLE_LOCK:
        if PUZZLE["state"] == "running":
            return jsonify(error="A puzzle build is already running."), 409
        PUZZLE.update(state="running", stage="read", fraction=0.0, message="Starting…", error=None)
    threading.Thread(target=_puzzle_worker, args=(items,), daemon=True).start()
    return jsonify(started=True, sheets=len(items))

@app.route("/api/puzzle/status", methods=["GET"])
def puzzle_status():
    with PUZZLE_LOCK:
        return jsonify(dict(PUZZLE))

def _swap_latest(out_dir):
    final_dir = os.path.join(PUZZLE_DIR, "latest")
    if os.path.isdir(final_dir):
        for name in os.listdir(final_dir):
            os.remove(os.path.join(final_dir, name))
        os.rmdir(final_dir)
    os.replace(out_dir, final_dir)

@app.route("/api/puzzle/manual", methods=["GET"])
def puzzle_manual_get():
    return jsonify(manual=_load_manual())

@app.route("/api/puzzle/manual", methods=["POST"])
def puzzle_manual_set():
    """Manual placement.  Body: {id, to?, side?, fx?, fy?, rot?, reset?}.
    to+side: put this sheet flush against another one.  fx/fy: nudge as a fraction of the sheet's long side
    (added to what is already stored).  rot: rotate by this many degrees about the sheet's centre.
    reset: forget the manual placement of this sheet (reset_all clears every sheet)."""
    payload = request.get_json(silent=True) or {}
    with PUZZLE_LOCK:
        if PUZZLE["state"] == "running":
            return jsonify(error="A puzzle build is running. Wait for it to finish."), 409
    manual = _load_manual()
    if payload.get("reset_all"):
        manual = {}
    else:
        sid = str(payload.get("id") or "")
        if not sid:
            return jsonify(error="Choose a sheet first."), 400
        if payload.get("reset"):
            manual.pop(sid, None)
        else:
            entry = manual.setdefault(sid, {})
            try:
                if payload.get("to"):
                    if payload.get("side") not in puzzle.SIDES:
                        return jsonify(error="Side must be right, left, above or below."), 400
                    if str(payload["to"]) == sid:
                        return jsonify(error="A sheet cannot be placed next to itself."), 400
                    entry.update(to=str(payload["to"]), side=payload["side"], force=True, fx=0.0, fy=0.0, rot=0.0)
                for key, limit in (("fx", 1.0), ("fy", 1.0), ("rot", 45.0)):
                    if payload.get(key) not in (None, ""):
                        entry[key] = max(-limit * 4, min(limit * 4, float(entry.get(key) or 0.0) + float(payload[key])))
            except (TypeError, ValueError):
                return jsonify(error="Nudge values must be numbers."), 400
    _save_manual(manual)
    out_dir = os.path.join(PUZZLE_DIR, "latest_tmp")
    try:
        report = puzzle.redraw(manual, out_dir)
        with PUZZLE_LOCK:
            _swap_latest(out_dir)
            PUZZLE.update(state="done", fraction=1.0, message="Puzzle updated with manual placement.", report=report, error=None)
    except Exception as exc:
        app.logger.exception("Manual puzzle redraw failed")
        return jsonify(error=f"{type(exc).__name__}: {exc}"), 500
    return jsonify(ok=True, report=report)

def _puzzle_file(group, suffix):
    try:
        group = int(group)
    except ValueError:
        abort(404, description="Unknown puzzle group.")
    path = os.path.join(PUZZLE_DIR, "latest", f"group_{group}{suffix}")
    if not os.path.isfile(path):
        abort(404, description="Puzzle image not found. Build the puzzle again.")
    return path

@app.route("/puzzle/<group>/preview", methods=["GET"])
def puzzle_preview(group):
    return send_file(_puzzle_file(group, "_preview.png"), mimetype="image/png", max_age=0)

@app.route("/puzzle/<group>/download", methods=["GET"])
def puzzle_download(group):
    if request.args.get("fmt") == "tif":
        return send_file(_puzzle_file(group, ".tif"), mimetype="image/tiff", as_attachment=True,
                         download_name=f"puzzle_group_{group}.tif", max_age=0)
    return send_file(_puzzle_file(group, ".png"), mimetype="image/png", as_attachment=True,
                     download_name=f"puzzle_group_{group}.png", max_age=0)

def _puzzle_notes():
    with PUZZLE_LOCK:
        report = PUZZLE.get("report")
    notes = {}
    if not report:
        return notes
    for g in report["groups"]:
        for sheet in g["sheets"]:
            notes[sheet["id"]] = f"Puzzle: group {g['group']} ({len(g['sheets'])} sheets)"
    for u in report["unattached"]:
        notes[u["id"]] = "Puzzle: not attached – " + u["reason"]
    for n in report["needs_attention"]:
        notes[n["id"]] = "Puzzle: needs attention – " + n["reason"]
    return notes

board.register(app, _approved_records, APPROVED_DIR, PUZZLE_DIR)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)