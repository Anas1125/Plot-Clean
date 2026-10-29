/* Manual puzzle board. Loaded by app.py from /board/board.js (kept out of the page template so that
   curly-brace sequences in this file can never collide with the template engine).
   World units are metres on the ground; screen = world * view.k + (view.ox, view.oy). */
(() => {
  'use strict';
  const SNAP_PX = 16;                    // pull-in distance, screen px
  const ROT_TOL = 12 * Math.PI / 180;     // edges up to this far from parallel are turned flush
  const SAVE_DELAY = 350;
  const OVERLAP_MAX = 0.03;               // max fraction of the smaller of two pieces allowed to lie on the other one

  const PB = window.PlotBoard = {};
  let root, canvas, ctx, statusEl, selBar, selName, selColor, snapBox, wrap, modal, modalImg, modalTitle, modalDl;
  let linesBox, fmtSel, pieces = [], selected = null, drag = null, pan = null, snapTarget = null;
  let view = { ox: 0, oy: 0, k: 1 }, started = false, loading = false, dpr = 1, W = 0, H = 0, raf = 0;
  let saveTimer = 0; const dirty = new Set();

  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const setStatus = (msg, kind) => { if (statusEl) { statusEl.textContent = msg || ''; statusEl.className = 'status ' + (kind || ''); } };
  const schedule = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); }); };

  // ------------------------------------------------------------------ geometry
  function refresh(p) {
    const a = p.rot * Math.PI / 180, c = Math.cos(a), s = Math.sin(a), n = p.poly.length;
    const wx = new Float64Array(n), wy = new Float64Array(n);
    let x0 = 1e18, y0 = 1e18, x1 = -1e18, y1 = -1e18;
    for (let i = 0; i < n; i++) {
      const px = p.poly[i][0] * p.mpp, py = p.poly[i][1] * p.mpp;
      const x = c * px - s * py + p.x, y = s * px + c * py + p.y;
      wx[i] = x; wy[i] = y;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    p.wx = wx; p.wy = wy; p.bb = { x0, y0, x1, y1 };
    let a2 = 0; for (let i = 0, j = n - 1; i < n; j = i++) a2 += (wx[j] + wx[i]) * (wy[j] - wy[i]);
    p.area = Math.abs(a2) / 2;
  }
  function pointIn(x, y, p) {
    const wx = p.wx, wy = p.wy, n = wx.length; let inside = false;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      if ((wy[i] > y) !== (wy[j] > y) && x < (wx[j] - wx[i]) * (y - wy[i]) / (wy[j] - wy[i]) + wx[i]) inside = !inside;
    }
    return inside;
  }
  function distToPoly(x, y, p) {
    const wx = p.wx, wy = p.wy, n = wx.length; let best = 1e18;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const ax = wx[j], ay = wy[j], dx = wx[i] - ax, dy = wy[i] - ay;
      const l2 = dx * dx + dy * dy;
      let t = l2 > 0 ? ((x - ax) * dx + (y - ay) * dy) / l2 : 0;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const ex = ax + t * dx - x, ey = ay + t * dy - y, d = ex * ex + ey * ey;
      if (d < best) best = d;
    }
    return Math.sqrt(best);
  }
  function perimeter(p) {
    let L = 0; const n = p.wx.length;
    for (let i = 0; i < n; i++) { const j = (i + 1) % n; L += Math.hypot(p.wx[j] - p.wx[i], p.wy[j] - p.wy[i]); }
    return L;
  }
  function sampleBoundary(p, count) {
    const n = p.wx.length, L = perimeter(p), step = L / count, out = [];
    let carry = 0;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n, dx = p.wx[j] - p.wx[i], dy = p.wy[j] - p.wy[i], len = Math.hypot(dx, dy);
      let t = carry;
      while (t < len) { out.push([p.wx[i] + dx * t / len, p.wy[i] + dy * t / len]); t += step; }
      carry = t - len;
    }
    return { pts: out, step };
  }
  // about n sample points inside P, at least `inset` away from its own outline
  function interiorSamples(P, inset, n) {
    const w = P.bb.x1 - P.bb.x0, h = P.bb.y1 - P.bb.y0;
    const g = Math.max(inset, Math.sqrt(w * h / n)), out = [];
    for (let x = P.bb.x0 + g / 2; x < P.bb.x1; x += g)
      for (let y = P.bb.y0 + g / 2; y < P.bb.y1; y += g)
        if (pointIn(x, y, P) && distToPoly(x, y, P) > inset) out.push([x, y]);
    return out;
  }
  function samplesOf(P, inset) {                       // cached until the piece moves, turns, resizes or the zoom changes
    const key = P.x + '|' + P.y + '|' + P.rot + '|' + P.mpp + '|' + inset;
    if (P._insKey !== key) { P._ins = interiorSamples(P, inset, 300); P._insKey = key; }
    return P._ins;
  }
  // If A were moved by candidate c: the worst fraction of the SMALLER piece that would lie on top of another piece.
  // (Measured against the smaller piece, so a big piece cannot sit on a small one and still look like a tiny overlap.)
  function overlapWorst(A, insA, c, cA, others, inset) {
    const cs = Math.cos(c.d), sn = Math.sin(c.d);
    const mv = (x, y) => [cA.x + cs * (x - cA.x) - sn * (y - cA.y) + c.tx, cA.y + sn * (x - cA.x) + cs * (y - cA.y) + c.ty];
    let x0 = 1e18, y0 = 1e18, x1 = -1e18, y1 = -1e18;
    for (const [x, y] of [[A.bb.x0, A.bb.y0], [A.bb.x1, A.bb.y0], [A.bb.x1, A.bb.y1], [A.bb.x0, A.bb.y1]]) {
      const [mx, my] = mv(x, y); if (mx < x0) x0 = mx; if (mx > x1) x1 = mx; if (my < y0) y0 = my; if (my > y1) y1 = my;
    }
    let worst = 0;
    for (const B of others) {
      if (x1 < B.bb.x0 || x0 > B.bb.x1 || y1 < B.bb.y0 || y0 > B.bb.y1) continue;
      let n = 0, total;
      if (A.area <= B.area) {                          // A is the smaller one: how much of A lands inside B
        total = insA.length;
        for (const [sx, sy] of insA) {
          const [x, y] = mv(sx, sy);
          if (x >= B.bb.x0 && x <= B.bb.x1 && y >= B.bb.y0 && y <= B.bb.y1 && pointIn(x, y, B)) n++;
        }
      } else {                                         // B is the smaller one: how much of B lands inside A's new place
        const insB = samplesOf(B, inset); total = insB.length;
        for (const [bx, by] of insB) {
          const dx = bx - cA.x - c.tx, dy = by - cA.y - c.ty;
          const x = cA.x + cs * dx + sn * dy, y = cA.y - sn * dx + cs * dy;      // undo the move
          if (x >= A.bb.x0 && x <= A.bb.x1 && y >= A.bb.y0 && y <= A.bb.y1 && pointIn(x, y, A)) n++;
        }
      }
      const f = total ? n / total : 0; if (f > worst) worst = f;
    }
    return worst;
  }

  // Visual centre of a polygon given in its own (pixel) coordinates: the inside point furthest from the outline.
  // Used for the S.n labels so they sit in the middle of the piece even when its outline is notched or L-shaped.
  function visualCentre(poly) {
    const n = poly.length; if (n < 3) return null;
    let x0 = 1e18, y0 = 1e18, x1 = -1e18, y1 = -1e18;
    for (const [x, y] of poly) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    const inPoly = (x, y) => {
      let ins = false;
      for (let i = 0, j = n - 1; i < n; j = i++) {
        const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
        if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) ins = !ins;
      }
      return ins;
    };
    const dist = (x, y) => {
      let best = 1e18;
      for (let i = 0, j = n - 1; i < n; j = i++) {
        const ax = poly[j][0], ay = poly[j][1], dx = poly[i][0] - ax, dy = poly[i][1] - ay, l2 = dx * dx + dy * dy;
        let t = l2 > 0 ? ((x - ax) * dx + (y - ay) * dy) / l2 : 0; t = t < 0 ? 0 : t > 1 ? 1 : t;
        const ex = ax + t * dx - x, ey = ay + t * dy - y, d = ex * ex + ey * ey; if (d < best) best = d;
      }
      return Math.sqrt(best);
    };
    let cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, half = Math.max(x1 - x0, y1 - y0) / 2, bx = cx, by = cy, bd = -1;
    for (let iter = 0; iter < 5; iter++) {
      for (let a = -8; a <= 8; a++) for (let b = -8; b <= 8; b++) {
        const x = cx + a * half / 8, y = cy + b * half / 8;
        if (!inPoly(x, y)) continue;
        const d = dist(x, y); if (d > bd) { bd = d; bx = x; by = y; }
      }
      cx = bx; cy = by; half /= 3;
    }
    return bd < 0 ? null : [bx, by];
  }

  // ------------------------------------------------------------------ edge snapping
  function candidates(A, B, tol, cA, rotTol, tol0) {
    const out = [], na = A.wx.length, nb = B.wx.length, minLen = tol0 * 1.5;
    for (let i = 0; i < na; i++) {
      const ax0 = A.wx[i], ay0 = A.wy[i], ax1 = A.wx[(i + 1) % na], ay1 = A.wy[(i + 1) % na];
      const la = Math.hypot(ax1 - ax0, ay1 - ay0); if (la < minLen) continue;
      const ta = Math.atan2(ay1 - ay0, ax1 - ax0);
      for (let j = 0; j < nb; j++) {
        const bx0 = B.wx[j], by0 = B.wy[j], bx1 = B.wx[(j + 1) % nb], by1 = B.wy[(j + 1) % nb];
        const lb = Math.hypot(bx1 - bx0, by1 - by0); if (lb < minLen) continue;
        let d = Math.atan2(by1 - by0, bx1 - bx0) - ta;
        d = (((d + Math.PI / 2) % Math.PI) + Math.PI) % Math.PI - Math.PI / 2;      // parallel in either direction
        if (Math.abs(d) > rotTol) continue;
        const cs = Math.cos(d), sn = Math.sin(d);
        const rx0 = cA.x + cs * (ax0 - cA.x) - sn * (ay0 - cA.y), ry0 = cA.y + sn * (ax0 - cA.x) + cs * (ay0 - cA.y);
        const rx1 = cA.x + cs * (ax1 - cA.x) - sn * (ay1 - cA.y), ry1 = cA.y + sn * (ax1 - cA.x) + cs * (ay1 - cA.y);
        const ux = (bx1 - bx0) / lb, uy = (by1 - by0) / lb, nx = -uy, ny = ux;
        const dist = ((rx0 + rx1) / 2 - bx0) * nx + ((ry0 + ry1) / 2 - by0) * ny;
        if (Math.abs(dist) > tol) continue;
        const p0 = (rx0 - bx0) * ux + (ry0 - by0) * uy, p1 = (rx1 - bx0) * ux + (ry1 - by0) * uy;
        const amin = Math.min(p0, p1), amax = Math.max(p0, p1);
        const ov = Math.min(amax, lb) - Math.max(amin, 0);
        if (ov < tol0 || ov < 0.2 * Math.min(la, lb)) continue;
        const shifts = [0];
        for (const target of [0, lb]) for (const e of [amin, amax]) { const s = target - e; if (Math.abs(s) <= 3 * tol) shifts.push(s); }
        for (const s of shifts) out.push({ d, tx: -dist * nx + s * ux, ty: -dist * ny + s * uy, ov });
      }
    }
    out.sort((a, b) => b.ov - a.ov);
    const keep = [];
    for (const c of out) {
      if (keep.every(k => Math.abs(k.d - c.d) > 1e-3 || Math.hypot(k.tx - c.tx, k.ty - c.ty) > tol * 0.25)) keep.push(c);
      if (keep.length >= 24) break;
    }
    // corner to corner (edges that are not parallel but meet at a corner)
    const corner = [];
    for (let i = 0; i < na; i++) for (let j = 0; j < nb; j++) {
      const d = Math.hypot(B.wx[j] - A.wx[i], B.wy[j] - A.wy[i]);
      if (d <= 2.5 * tol) corner.push({ d: 0, tx: B.wx[j] - A.wx[i], ty: B.wy[j] - A.wy[i], ov: 0, dist: d });
    }
    corner.sort((a, b) => a.dist - b.dist);
    return keep.concat(corner.slice(0, 6));
  }

  function findSnap(A, opt) {
    opt = opt || {};
    const tol = opt.tolWorld || SNAP_PX * (opt.tolMul || 1) / view.k, rotTol = opt.rotTol || ROT_TOL, cA = { x: (A.bb.x0 + A.bb.x1) / 2, y: (A.bb.y0 + A.bb.y1) / 2 };
    const tol0 = Math.min(tol, SNAP_PX / view.k);          // normal snap distance: edge length / contact limits use this, not the wide search distance
    const perim = perimeter(A), smp = sampleBoundary(A, 150);
    const inset = 4 / view.k, ins = samplesOf(A, inset);   // points inside A, used to reject placements that sit on top of other pieces
    const others = pieces.filter(q => q !== A && (!opt.only || opt.only.has(q)));
    let best = null;
    for (const B of pieces) {
      if (B === A || (opt.only && !opt.only.has(B))) continue;
      if (A.bb.x0 - tol > B.bb.x1 || A.bb.x1 + tol < B.bb.x0 || A.bb.y0 - tol > B.bb.y1 || A.bb.y1 + tol < B.bb.y0) continue;
      for (const c of candidates(A, B, tol, cA, rotTol, tol0)) {
        const cs = Math.cos(c.d), sn = Math.sin(c.d);
        let near = 0, inside = 0, flush = 0;
        for (const [sx, sy] of smp.pts) {
          const x = cA.x + cs * (sx - cA.x) - sn * (sy - cA.y) + c.tx, y = cA.y + sn * (sx - cA.x) + cs * (sy - cA.y) + c.ty;
          const dd = distToPoly(x, y, B);
          if (dd < tol * 0.7) { near++; const q = 1 - dd / (tol * 0.7); flush += q * q; }   // flush points count fully, near-misses little
          else if (pointIn(x, y, B)) inside++;
        }
        if (near * smp.step < Math.max(3.5 * tol0, 0.04 * perim)) continue;
        if (overlapWorst(A, ins, c, cA, others, inset) > OVERLAP_MAX) continue;   // reject placements that land on top of any piece
        const sc = (flush - 3 * inside) * smp.step - 0.05 * Math.hypot(c.tx, c.ty);
        if (!best || sc > best.sc) best = { sc, c, B };
      }
    }
    if (!best) return null;
    const { c, B } = best, cs = Math.cos(c.d), sn = Math.sin(c.d);
    return {
      x: cA.x + cs * (A.x - cA.x) - sn * (A.y - cA.y) + c.tx,
      y: cA.y + sn * (A.x - cA.x) + cs * (A.y - cA.y) + c.ty,
      rot: A.rot + c.d * 180 / Math.PI, target: B, sc: best.sc,
    };
  }
  function applySnap(A) {
    snapTarget = null;
    if (!snapBox.checked) return;
    const s = findSnap(A);
    if (!s) return;
    A.x = s.x; A.y = s.y; A.rot = s.rot; snapTarget = s.target; refresh(A);
  }


  // ------------------------------------------------------------------ background reference image
  // An uploaded picture (any image or TIFF, converted to PNG by the server) drawn under the pieces, in world
  // units like the pieces: (x, y) = top-left in metres, mpp = metres per picture pixel, rot = degrees.
  // Upload with the button, drop a file on the board, or paste (Ctrl/Cmd+V).
  let bg = null, bgImg = null, bgAdjust = false, bgDrag = null, bgSaveTimer = 0, bgBtn, bgAdjBtn, bgOpacity, bgInput;
  function bgFitToView() {
    if (!bg || !bg.w) return;
    const [x0, y0] = toWorld(0, 0), [x1, y1] = toWorld(W, H);
    bg.mpp = Math.min((x1 - x0) / bg.w, (y1 - y0) / bg.h) * 0.95;
    bg.x = (x0 + x1) / 2 - bg.w * bg.mpp / 2; bg.y = (y0 + y1) / 2 - bg.h * bg.mpp / 2; bg.rot = 0;
  }
  function bgSaveSoon() {
    clearTimeout(bgSaveTimer);
    bgSaveTimer = setTimeout(() => {
      if (!bg) return;
      fetch('/api/board/background/pose', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x: bg.x, y: bg.y, mpp: bg.mpp, rot: bg.rot, opacity: bg.opacity }) }).catch(() => {});
    }, 400);
  }
  function bgApply(meta, fresh) {
    if (!meta || !meta.exists) { bg = null; bgImg = null; updateBgUI(); schedule(); return; }
    bg = { w: meta.w, h: meta.h, x: meta.x, y: meta.y, mpp: meta.mpp, rot: meta.rot, opacity: meta.opacity == null ? 0.6 : meta.opacity };
    if (!(bg.mpp > 0)) { bgFitToView(); bgSaveSoon(); }
    bgImg = new Image(); bgImg.onload = () => schedule(); bgImg.src = meta.url;
    updateBgUI(); schedule();
    if (fresh) setStatus('Background loaded. Press "Move background" to line it up with your pieces.', 'good');
  }
  async function bgLoad() {
    try { const r = await fetch('/api/board/background'); bgApply(await r.json(), false); } catch (e) { /* no background */ }
  }
  async function bgUpload(file) {
    if (!file) return;
    setStatus('Uploading background… (large TIFFs can take a few seconds)');
    try {
      const fd = new FormData(); fd.append('file', file, file.name || 'pasted.png');
      const r = await fetch('/api/board/background', { method: 'POST', body: fd }), d = await r.json();
      if (!r.ok) throw new Error(d.error || 'Upload failed.');
      d.mpp = 0; bgApply(d, true);
    } catch (e) { setStatus(e.message, 'error'); }
  }
  async function bgClear() {
    if (!bg) return;
    bgAdjust = false; bg = null; bgImg = null; updateBgUI(); schedule();
    try { await fetch('/api/board/background', { method: 'DELETE' }); } catch (e) { /* ignore */ }
    setStatus('Background removed.');
  }
  function updateBgUI() {
    if (bgAdjBtn) {
      bgAdjBtn.disabled = !bg; bgAdjBtn.textContent = bgAdjust ? '✔ Done moving' : '✥ Move background'; bgAdjBtn.classList.toggle('on', bgAdjust);
      if (bgScaleBtn) bgScaleBtn.disabled = !bg;
      if (fitBtn2) fitBtn2.disabled = !bg; if (fitSel) fitSel.disabled = !bg;
      bgAdjBtn.title = 'Drag = move, mouse wheel = resize, Shift+wheel = rotate';
    }
    if (bgOpacity) { bgOpacity.disabled = !bg; if (bg) bgOpacity.value = Math.round(bg.opacity * 100); }
    if (bgClearBtn) bgClearBtn.disabled = !bg;
    if (wrap) wrap.classList.toggle('bgadjust', bgAdjust);
  }
  let bgClearBtn;
  function drawBackground() {
    if (!bg || !bgImg || !bgImg.complete || !bgImg.naturalWidth) return;
    ctx.save();
    ctx.globalAlpha = bg.opacity;
    ctx.translate(view.ox + bg.x * view.k, view.oy + bg.y * view.k);
    ctx.rotate(bg.rot * Math.PI / 180);
    const sc = bg.mpp * view.k; ctx.scale(sc, sc);
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bgImg, 0, 0, bg.w, bg.h);
    if (bgAdjust) { ctx.globalAlpha = 1; ctx.lineWidth = 2 / sc; ctx.strokeStyle = '#0b6bcb'; ctx.setLineDash([8 / sc, 6 / sc]); ctx.strokeRect(0, 0, bg.w, bg.h); }
    ctx.restore();
  }


  // ------------------------------------------------------------------ pin tools (calibrate against the background)
  // 1) "Pin-fit piece": click a spot on the selected piece, then the SAME spot on the background; do it twice (far apart).
  //    The piece is then moved, turned AND resized so both spots coincide.
  // 2) "Set bg scale": click two points on the background and type their real distance in metres (e.g. the scale bar).
  let pin = null, pinDown = null, pinBtn, bgScaleBtn;
  const PIN_TXT = ['Click a recognisable spot on the PIECE (a corner of its outline).',
                   'Now click the same spot on the BACKGROUND.',
                   'Click a second spot on the piece, far from the first one.',
                   'Click the same second spot on the BACKGROUND.'];
  function pinStart(kind) {
    if (kind === 'fit') {
      if (!selected) { setStatus('Select a piece first.', 'error'); return; }
      if (!bg) { setStatus('Add a background image first.', 'error'); return; }
      if (boardLocked) { lockedMsg(); return; }
      if (selected.locked) { pieceLockedMsg(selected); return; }
      pin = { kind, piece: selected, pts: [], pose: { x: selected.x, y: selected.y, rot: selected.rot, mpp: selected.mpp } };
      setStatus('Pin-fit S.' + selected.survey + ': ' + PIN_TXT[0] + ' (Esc cancels)');
    } else {
      if (!bg) { setStatus('Add a background image first.', 'error'); return; }
      pin = { kind, pts: [] };
      setStatus('Set background scale: click one end of a known distance (e.g. the scale bar). (Esc cancels)');
    }
    canvas.style.cursor = 'crosshair'; schedule();
  }
  const pinCancel = (msg) => { pin = null; pinDown = null; if (msg) setStatus(msg); schedule(); };
  function pinClick(sx, sy) {
    const [wx, wy] = toWorld(sx, sy);
    if (pin.kind === 'scale') {
      pin.pts.push([wx, wy]); schedule();
      if (pin.pts.length < 2) { setStatus('Now click the other end of that distance.'); return; }
      const [a, b] = pin.pts, d = Math.hypot(b[0] - a[0], b[1] - a[1]);
      const m = parseFloat(window.prompt('Real distance between the two points, in metres:', '100'));
      if (isFinite(m) && m > 0 && d > 0) {
        const k = m / d;                                              // new metres-per-world-unit factor for the picture
        const a0 = bg.rot * Math.PI / 180, lx = a[0] - bg.x, ly = a[1] - bg.y;
        const u = (Math.cos(a0) * lx + Math.sin(a0) * ly) / bg.mpp, v = (-Math.sin(a0) * lx + Math.cos(a0) * ly) / bg.mpp;
        bg.mpp *= k;                                                  // keep the first clicked point where it is
        bg.x = a[0] - (Math.cos(a0) * u - Math.sin(a0) * v) * bg.mpp; bg.y = a[1] - (Math.sin(a0) * u + Math.cos(a0) * v) * bg.mpp;
        bgSaveSoon(); pinCancel('Background scale set. Pieces drawn at the right scale should now match it in size.');
      } else pinCancel('Cancelled.');
      return;
    }
    const p = pin.piece, n = pin.pts.length;
    // clicks alternate piece, background, piece, background.  A spot on the piece starts a new pair; a pair that
    // has no background spot yet receives this click.  (n is the number of pairs, not clicks, so n % 2 was wrong.)
    if (n === 0 || pin.pts[n - 1].t) {                                // a spot on the piece, in the piece's own (picture-pixel) units
      const a = pin.pose.rot * Math.PI / 180, dx = wx - pin.pose.x, dy = wy - pin.pose.y;
      pin.pts.push({ q: [(Math.cos(a) * dx + Math.sin(a) * dy) / pin.pose.mpp, (-Math.sin(a) * dx + Math.cos(a) * dy) / pin.pose.mpp], w: [wx, wy] });
    } else { pin.pts[n - 1].t = [wx, wy]; }
    schedule();
    if (pin.pts.length < 2 || !pin.pts[1].t) { setStatus(PIN_TXT[pin.pts.length === 1 && !pin.pts[0].t ? 1 : (pin.pts[1] ? 3 : 2)]); return; }
    const [P1, P2] = pin.pts;
    const dq = [P2.q[0] - P1.q[0], P2.q[1] - P1.q[1]], dt = [P2.t[0] - P1.t[0], P2.t[1] - P1.t[1]];
    const lq = Math.hypot(dq[0], dq[1]), lt = Math.hypot(dt[0], dt[1]);
    if (lq < 5 || lt < 1e-6) { pinCancel('The two spots were too close together. Try again with spots far apart.'); return; }
    const m = lt / lq, th = Math.atan2(dt[1], dt[0]) - Math.atan2(dq[1], dq[0]);
    const f = m / p.mpp0;
    if (f < 0.2 || f > 5) { pinCancel(`That needs a size of x${f.toFixed(2)}, outside the allowed 0.2 to 5. Check that you clicked matching spots.`); return; }
    const before = snapshot();
    const c = Math.cos(th), s = Math.sin(th);
    p.rot = th * 180 / Math.PI; p.sz = f; p.mpp = m;
    p.x = P1.t[0] - (c * P1.q[0] - s * P1.q[1]) * m; p.y = P1.t[1] - (s * P1.q[0] + c * P1.q[1]) * m;
    refresh(p); rememberSize(p); pushUndo(before); queueSave(p); updateSel();
    pinCancel(`S.${p.survey} fitted to the background (size x${f.toFixed(2)}, rotation ${p.rot.toFixed(1)}°). Ctrl/Cmd+Z undoes it.`);
  }
  function drawPins() {
    if (!pin) return;
    const mark = (w, label, col) => {
      const [x, y] = toScreen(w[0], w[1]); ctx.save(); ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x - 9, y); ctx.lineTo(x + 9, y); ctx.moveTo(x, y - 9); ctx.lineTo(x, y + 9); ctx.stroke();
      ctx.font = '700 13px system-ui'; ctx.fillText(label, x + 8, y - 8); ctx.restore();
    };
    if (pin.kind === 'scale') { pin.pts.forEach((w, i) => mark(w, 'ABC'[i], '#0b6bcb')); return; }
    pin.pts.forEach((P, i) => { mark(P.w, 'P' + (i + 1), '#c62828'); if (P.t) mark(P.t, 'B' + (i + 1), '#0b6bcb'); });
  }


  // ------------------------------------------------------------------ auto size from neighbours
  // A piece's size comes from the 1:N printed on its sheet, which is sometimes misread. Two sheets that touch share a
  // stretch of boundary, and at the RIGHT size both ends of that stretch coincide (two corners meet, the edges run
  // together). sizeFit tries many sizes for a piece, lets it snap against the pieces it is compared with at each size,
  // and keeps the size where the most outline lies on outline AND corners coincide. sizeChain does this piece after piece,
  // so each fitted piece becomes a reference for the next one.
  const spanOf = p => Math.max(p.bb.x1 - p.bb.x0, p.bb.y1 - p.bb.y0);
  function cornersMeeting(A, B, tol) {
    let n = 0;
    for (let i = 0; i < A.wx.length; i++) {
      for (let k = 0; k < B.wx.length; k++) if (Math.hypot(A.wx[i] - B.wx[k], A.wy[i] - B.wy[k]) <= tol) { n++; break; }
    }
    return n;
  }
  function sizeFit(A, anchors, opt) {
    opt = opt || {};
    const o = { x: A.x, y: A.y, rot: A.rot, sz: A.sz, mpp: A.mpp };
    const span0 = spanOf(A);
    const tolWorld = 0.035 * span0, tolC = 0.012 * span0, tolF = 0.012 * span0;
    // Pivots: the piece's own corners that already sit closest to a neighbour.  Resizing about one of them keeps the
    // shared stretch of border in contact while it grows or shrinks along its own line.
    const pv = [];
    const cornersByDist = [];
    for (let i = 0; i < A.wx.length; i++) {
      let d = 1e18; for (const B of anchors) d = Math.min(d, distToPoly(A.wx[i], A.wy[i], B));
      if (d <= 0.15 * span0) cornersByDist.push({ d, p: [A.wx[i], A.wy[i]] });
    }
    cornersByDist.sort((a, b) => a.d - b.d);
    for (const c of cornersByDist) if (pv.length < 4 && pv.every(q => Math.hypot(q[0] - c.p[0], q[1] - c.p[1]) > 0.12 * span0)) pv.push(c.p);
    if (!pv.length) return null;
    const set = (r, dRot, P) => {
      A.sz = o.sz * r; A.mpp = o.mpp * r;
      const a = (dRot || 0) * Math.PI / 180, c = Math.cos(a), s = Math.sin(a), px = P[0] + r * (o.x - P[0]), py = P[1] + r * (o.y - P[1]);
      A.x = P[0] + c * (px - P[0]) - s * (py - P[1]); A.y = P[1] + s * (px - P[0]) + c * (py - P[1]); A.rot = o.rot + (dRot || 0); refresh(A);
    };
    const score = (r, dRot, P) => {
      if (o.sz * r < 0.2 || o.sz * r > 5) return null;
      set(r, dRot, P);
      const s = findSnap(A, { tolWorld, rotTol: 2.5 * Math.PI / 180, only: anchors });
      if (!s) return null;
      A.x = s.x; A.y = s.y; A.rot = s.rot; refresh(A);
      let corners = 0;
      for (const B of anchors) if (B !== A) corners = Math.max(corners, cornersMeeting(A, B, tolC));
      // How much of EACH outline lies on the other.  Only at the right size do both fractions peak together: too small and
      // the neighbour's edge is not used up, too big and the piece's own edge overhangs.
      const T = s.target, nA = 160, sa = sampleBoundary(A, nA).pts, sb = sampleBoundary(T, nA).pts;
      let ca = 0, cb = 0;
      for (const [x, y] of sa) if (distToPoly(x, y, T) <= tolF) ca++;
      for (const [x, y] of sb) if (distToPoly(x, y, A) <= tolF) cb++;
      const sc = ca / sa.length + cb / sb.length + 0.3 * Math.min(corners, 3) - 0.04 * Math.abs(Math.log(r)) - 0.01 * Math.abs(dRot || 0);
      return { sc, r, P, dRot: dRot || 0, pose: { x: A.x, y: A.y, rot: A.rot }, corners, target: s.target };
    };
    const lo = opt.lo || 0.55, hi = opt.hi || 1.8, steps = 29;
    let cand = [];
    for (const P of pv) {
      for (let i = 0; i < steps; i++) {
        const v = score(lo * Math.pow(hi / lo, i / (steps - 1)), 0, P); if (v) cand.push(v);
      }
      const v1 = score(1, 0, P); if (v1) cand.push(v1);          // always compare against the current size too
    }
    cand.sort((p, q) => q.sc - p.sc);
    let best = null;
    for (const c of cand.slice(0, 4)) {                        // refine the best few: finer size, a few degrees of tilt
      for (let k = -4; k <= 4; k++) {
        const r = c.r * Math.pow(1.04, k / 4);
        for (const d of (opt.noRot ? [0] : [-4, -2, 0, 2, 4])) {
          const v = score(r, d, c.P); if (v && (!best || v.sc > best.sc)) best = v;
        }
      }
    }
    if (best) {                                                // final pass: 1% steps around the winner
      const w = best;
      for (let k = -6; k <= 6; k++) { const v = score(w.r * Math.pow(1.01, k), w.dRot, w.P); if (v && v.sc > best.sc) best = v; }
    }
    if (!best || best.corners < 1) {                           // no coinciding corner at any size: too uncertain, keep as is
      A.sz = o.sz; A.mpp = o.mpp; A.x = o.x; A.y = o.y; A.rot = o.rot; refresh(A); return null;
    }
    set(best.r, best.dRot, best.P); A.x = best.pose.x; A.y = best.pose.y; A.rot = best.pose.rot; refresh(A);
    rememberSize(A); queueSave(A);
    return best;
  }
  const sizeMsg = (p, b, was) => `S.${p.survey}: size x${p.sz.toFixed(2)} (was x${was.toFixed(2)}), fitted against S.${b.target.survey}.`;
  function sizeOne(p) {
    if (!p) return; if (boardLocked) { lockedMsg(); return; }
    if (p.locked) { pieceLockedMsg(p); return; }
    const near = new Set(pieces.filter(q => q !== p && !(q.bb.x1 < p.bb.x0 - 0.3 * spanOf(p) || q.bb.x0 > p.bb.x1 + 0.3 * spanOf(p) || q.bb.y1 < p.bb.y0 - 0.3 * spanOf(p) || q.bb.y0 > p.bb.y1 + 0.3 * spanOf(p))));
    if (!near.size) { setStatus(`No piece near S.${p.survey}. Drag it next to its neighbour first.`, 'error'); return; }
    const before = snapshot(), was = p.sz;
    setStatus(`Checking sizes for S.${p.survey}…`);
    setTimeout(() => {
      const b = sizeFit(p, near);
      if (!b) { setStatus(`Could not find a size where S.${p.survey} meets its neighbours corner to corner. Drag it closer, or use Pin-fit.`, 'error'); schedule(); return; }
      pushUndo(before); updateSel(); schedule(); setStatus(sizeMsg(p, b, was), 'good');
    }, 20);
  }
  let chainBusy = false;
  async function sizeChain(root) {
    if (!root) { setStatus('Select the piece whose size you trust first (the chain starts there).', 'error'); return; }
    if (boardLocked) { lockedMsg(); return; }
    if (chainBusy) return; chainBusy = true;
    const before = snapshot(), fixed = new Set([root].concat(pieces.filter(q => q.locked))), failed = new Set(), changed = [];
    try {
      for (;;) {
        // the next piece = the unplaced one that sits closest to the pieces already fixed
        let pick = null, pd = 1e18;
        for (const q of pieces) {
          if (fixed.has(q) || failed.has(q)) continue;
          const m = 0.25 * spanOf(q);
          for (const f of fixed) {
            const gx = Math.max(0, f.bb.x0 - q.bb.x1, q.bb.x0 - f.bb.x1), gy = Math.max(0, f.bb.y0 - q.bb.y1, q.bb.y0 - f.bb.y1), g = Math.hypot(gx, gy);
            if (g <= m && g < pd) { pd = g; pick = q; }
          }
        }
        if (!pick) break;
        setStatus(`Auto-sizing: ${fixed.size} fixed, checking S.${pick.survey}…`);
        await new Promise(r => setTimeout(r, 15));
        const was = pick.sz, b = sizeFit(pick, fixed);
        if (b) { fixed.add(pick); changed.push(`S.${pick.survey} x${was.toFixed(2)}→x${pick.sz.toFixed(2)}`); schedule(); }
        else failed.add(pick);
      }
      pushUndo(before); updateSel(); schedule();
      const miss = pieces.filter(q => !fixed.has(q)).map(q => 'S.' + q.survey);
      setStatus(`Auto-size done: ${fixed.size - 1} piece(s) fitted from S.${root.survey}.` + (miss.length ? ` Not reached (not touching a fitted piece, or no matching corner): ${miss.slice(0, 12).join(', ')}${miss.length > 12 ? '…' : ''}.` : ''), miss.length ? '' : 'good');
    } catch (e) { setStatus('Auto-size failed: ' + e.message, 'error'); }
    chainBusy = false;
  }


  // ------------------------------------------------------------------ tray (pieces not placed on the board yet)
  // `pieces` is only what lies on the board; `tray` holds the pieces that are still waiting.  Drag a tray item onto the
  // board (or double-click it to drop it in the middle of the view); "Send to tray" puts a piece back.
  let tray = [], trayList, trayCount, trayFilter, stage;
  const surveyNum = p => { const m = String(p.survey || '').match(/\d+/); return m ? parseInt(m[0], 10) : 1e9; };
  function thumb(p) {
    const c = document.createElement('canvas'); c.width = 128; c.height = 96;
    const g = c.getContext('2d'); let x0 = 1e18, y0 = 1e18, x1 = -1e18, y1 = -1e18;
    for (const [x, y] of p.poly) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    const k = Math.min(116 / Math.max(1, x1 - x0), 84 / Math.max(1, y1 - y0));
    g.translate(64 - (x0 + x1) / 2 * k, 48 - (y0 + y1) / 2 * k); g.scale(k, k);
    g.beginPath(); p.poly.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); g.closePath();
    g.globalAlpha = 0.5; g.fillStyle = p.color; g.fill(); g.globalAlpha = 1; g.lineWidth = 2 / k; g.strokeStyle = p.color; g.stroke();
    return c;
  }
  function renderTray() {
    if (!trayList) return;
    const q = (trayFilter.value || '').trim().toLowerCase();
    trayCount.textContent = `Not placed yet (${tray.length})`;
    trayList.textContent = '';
    const list = tray.slice().sort((a, b) => surveyNum(a) - surveyNum(b)).filter(p => !q || ('s.' + p.survey).toLowerCase().includes(q) || String(p.name || '').toLowerCase().includes(q));
    if (!list.length) trayList.appendChild(el('div', 'empty', tray.length ? 'No match.' : 'Every piece is on the board.'));
    for (const p of list) {
      const it = el('div', 'it'); it.draggable = true; it.title = 'Drag onto the board, or double-click to drop it in the middle';
      it.appendChild(thumb(p)); it.appendChild(el('span', '', 'S.' + p.survey));
      it.addEventListener('dragstart', ev => { ev.dataTransfer.setData('application/x-plotpiece', p.id); ev.dataTransfer.setData('text/plain', p.id); ev.dataTransfer.effectAllowed = 'move'; });
      it.addEventListener('dblclick', () => placeFromTray(p, W / 2, H / 2));
      trayList.appendChild(it);
    }
  }
  function placeFromTray(p, sx, sy) {
    if (boardLocked) { lockedMsg(); return; }
    const i = tray.indexOf(p); if (i < 0) return;
    tray.splice(i, 1);
    if (lastSize != null && Math.abs(lastSize - p.sz) > 1e-9) setSize(p, lastSize);   // same size as the one you set last
    const [wx, wy] = toWorld(sx, sy);
    refresh(p);
    p.x += wx - (p.bb.x0 + p.bb.x1) / 2; p.y += wy - (p.bb.y0 + p.bb.y1) / 2; refresh(p);
    p.placed = true; pieces.push(p); select(p);
    snapTarget = null; if (snapBox.checked) applySnap(p);
    queueSave(p); renderTray(); schedule(); setStatus(snapTarget ? `S.${p.survey} placed next to S.${snapTarget.survey}.` : `S.${p.survey} placed on the board.`);
    snapTarget = null;
  }
  function sendToTray(p) {
    if (!p) return; if (boardLocked) { lockedMsg(); return; }
    if (p.locked) { pieceLockedMsg(p); return; }
    const i = pieces.indexOf(p); if (i < 0) return;
    pieces.splice(i, 1); p.placed = false; tray.push(p); if (selected === p) select(null);
    queueSave(p); renderTray(); schedule(); setStatus(`S.${p.survey} sent back to the tray.`);
  }
  // pieces whose outline touches no other piece are "loose"
  function collectLoose() {
    if (boardLocked) { lockedMsg(); return; }
    const loose = [];
    for (const p of pieces) {
      if (p.locked) continue;
      const tol = 0.02 * spanOf(p), pts = sampleBoundary(p, 120).pts; let touch = false;
      for (const B of pieces) {
        if (B === p || p.bb.x0 - tol > B.bb.x1 || p.bb.x1 + tol < B.bb.x0 || p.bb.y0 - tol > B.bb.y1 || p.bb.y1 + tol < B.bb.y0) continue;
        for (const [x, y] of pts) if (distToPoly(x, y, B) <= tol) { touch = true; break; }
        if (touch) break;
      }
      if (!touch) loose.push(p);
    }
    if (!loose.length) { setStatus('Every piece on the board touches another one.'); return; }
    if (!window.confirm(`Move ${loose.length} piece(s) that touch no other piece to the tray?\n` + loose.map(p => 'S.' + p.survey).join(', '))) return;
    for (const p of loose) { pieces.splice(pieces.indexOf(p), 1); p.placed = false; tray.push(p); if (selected === p) select(null); queueSave(p); }
    renderTray(); schedule(); setStatus(`${loose.length} loose piece(s) moved to the tray.`, 'good');
  }

  // ------------------------------------------------------------------ drawing
  const toScreen = (x, y) => [view.ox + x * view.k, view.oy + y * view.k];
  const toWorld = (sx, sy) => [(sx - view.ox) / view.k, (sy - view.oy) / view.k];

  function pathPoly(p) { ctx.beginPath(); p.poly.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.closePath(); }
  function drawPiece(p, outline) {
    ctx.save();
    ctx.translate(view.ox + p.x * view.k, view.oy + p.y * view.k);
    ctx.rotate(p.rot * Math.PI / 180);
    const sc = p.mpp * view.k; ctx.scale(sc, sc);
    pathPoly(p);
    if (!outline) {
      if (!bg) { ctx.fillStyle = '#fff'; ctx.fill(); }          // with a background the pieces stay see-through
      ctx.globalAlpha = bg ? 0.30 : 0.42; ctx.fillStyle = p.color; ctx.fill(); ctx.globalAlpha = 1;
      if (linesBox.checked) {                                   // inner lines are optional; only the outer outline by default
        if (!p.img) { p.img = new Image(); p.img.onload = () => { p.loaded = true; schedule(); }; p.img.src = p.imgUrl; }
        if (p.loaded) ctx.drawImage(p.img, 0, 0, p.w, p.h);
      }
      ctx.lineJoin = 'round';
      ctx.lineWidth = (p === selected ? 3.2 : 1.6) / sc; ctx.strokeStyle = p === selected ? '#111' : p.color; pathPoly(p); ctx.stroke();
    } else {
      ctx.lineJoin = 'round'; ctx.lineWidth = 4 / sc; ctx.strokeStyle = '#0b6bcb'; ctx.stroke();
    }
    ctx.restore();
  }
  // ------------------------------------------------------------------ fit check against the background
  // Every piece outline is walked in small steps; a step is "good" when a dark line of the background picture lies
  // within a few picture pixels of it. Stretches with no line under them are drawn in red, and a piece where less
  // than FIT_BAD of the outline sits on a line gets a red tint and a % badge.
  let fitOn = false, fitTolPx = 8, fitBtn2, fitSel, ink = null;
  const FIT_BAD = 0.6;
  let fitTimer = 0;
  const fitBlink = () => 0.15 + 0.85 * (0.5 + 0.5 * Math.sin(performance.now() / 1000 * Math.PI * 2 * 1.5));   // ~1.5 blinks per second
  function fitAnimate(on) { clearInterval(fitTimer); fitTimer = on ? setInterval(schedule, 50) : 0; }
  function buildInk() {
    if (!bg || !bgImg || !bgImg.complete || !bgImg.naturalWidth) return null;
    if (ink && ink.src === bgImg) return ink;
    const s = Math.min(1, 2400 / Math.max(bg.w, bg.h)), dw = Math.max(1, Math.round(bg.w * s)), dh = Math.max(1, Math.round(bg.h * s));
    const cv = document.createElement('canvas'); cv.width = dw; cv.height = dh;
    const g = cv.getContext('2d', { willReadFrequently: true });
    g.fillStyle = '#fff'; g.fillRect(0, 0, dw, dh); g.drawImage(bgImg, 0, 0, dw, dh);
    let data; try { data = g.getImageData(0, 0, dw, dh).data; } catch (e) { return null; }
    const INF = 1e9, d = new Float32Array(dw * dh);
    for (let i = 0; i < d.length; i++) { const j = i * 4; d[i] = (0.299 * data[j] + 0.587 * data[j + 1] + 0.114 * data[j + 2]) < 130 ? 0 : INF; }
    const A = 1, B = 1.4142;                                    // two-pass chamfer distance to the nearest dark pixel
    for (let y = 0; y < dh; y++) for (let x = 0; x < dw; x++) {
      const i = y * dw + x; let v = d[i];
      if (x > 0) v = Math.min(v, d[i - 1] + A);
      if (y > 0) { v = Math.min(v, d[i - dw] + A); if (x > 0) v = Math.min(v, d[i - dw - 1] + B); if (x < dw - 1) v = Math.min(v, d[i - dw + 1] + B); }
      d[i] = v;
    }
    for (let y = dh - 1; y >= 0; y--) for (let x = dw - 1; x >= 0; x--) {
      const i = y * dw + x; let v = d[i];
      if (x < dw - 1) v = Math.min(v, d[i + 1] + A);
      if (y < dh - 1) { v = Math.min(v, d[i + dw] + A); if (x < dw - 1) v = Math.min(v, d[i + dw + 1] + B); if (x > 0) v = Math.min(v, d[i + dw - 1] + B); }
      d[i] = v;
    }
    ink = { src: bgImg, s, dw, dh, d };
    return ink;
  }
  function inkDist(u, v) {                                      // picture px -> distance (picture px) to the nearest line, -1 = outside the picture
    const x = Math.round(u * ink.s), y = Math.round(v * ink.s);
    if (x < 0 || y < 0 || x >= ink.dw || y >= ink.dh) return -1;
    return ink.d[y * ink.dw + x] / ink.s;
  }
  function fitOf(p) {
    if (!buildInk()) return null;
    const key = [p.x, p.y, p.rot, p.mpp, bg.x, bg.y, bg.mpp, bg.rot, fitTolPx].join('|');
    if (p._fit && p._fit.key === key) return p._fit;
    const a = p.rot * Math.PI / 180, c = Math.cos(a), s = Math.sin(a), b = -bg.rot * Math.PI / 180, cb = Math.cos(b), sb = Math.sin(b);
    const P = p.poly, stepW = bg.mpp * 4, st = [];
    for (let i = 0; i < P.length; i++) {
      const A = P[i], B = P[(i + 1) % P.length];
      const n = Math.max(1, Math.min(400, Math.ceil(Math.hypot(B[0] - A[0], B[1] - A[1]) * p.mpp / stepW)));
      for (let k = 0; k < n; k++) {
        const t = k / n, x = (A[0] + (B[0] - A[0]) * t) * p.mpp, y = (A[1] + (B[1] - A[1]) * t) * p.mpp;
        const wx = c * x - s * y + p.x, wy = s * x + c * y + p.y, dx = wx - bg.x, dy = wy - bg.y;
        const d = inkDist((cb * dx - sb * dy) / bg.mpp, (sb * dx + cb * dy) / bg.mpp);
        st.push({ wx, wy, ok: d < 0 ? null : d <= fitTolPx });
      }
    }
    let good = 0, valid = 0; const runs = []; let run = [];
    const flush = () => { if (run.length >= 3) runs.push(run); run = []; };
    for (const q of st) {
      if (q.ok === null) { flush(); continue; }
      valid++;
      if (q.ok) { good++; flush(); } else run.push([q.wx, q.wy]);
    }
    flush();
    p._fit = { key, score: valid >= 10 ? good / valid : null, runs };
    return p._fit;
  }
  function drawFit(p) {
    const f = fitOf(p); if (!f) return;
    if (f.score != null && f.score < FIT_BAD) {
      ctx.save(); ctx.translate(view.ox + p.x * view.k, view.oy + p.y * view.k); ctx.rotate(p.rot * Math.PI / 180);
      const sc = p.mpp * view.k; ctx.scale(sc, sc); pathPoly(p); ctx.globalAlpha = fitBlink(); ctx.fillStyle = 'rgba(230,0,0,.28)'; ctx.fill(); ctx.restore();
    }
    ctx.save(); ctx.globalAlpha = fitBlink(); ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.strokeStyle = '#e00000'; ctx.lineWidth = 4;
    for (const r of f.runs) { ctx.beginPath(); r.forEach(([wx, wy], i) => { const [sx, sy] = toScreen(wx, wy); i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy); }); ctx.stroke(); }
    ctx.restore();
    if (f.score != null && f.score < FIT_BAD) {
      const a = p.rot * Math.PI / 180, cc = Math.cos(a), ss = Math.sin(a), lp = p.ctr || p.tag, lx = lp[0] * p.mpp, ly = lp[1] * p.mpp;
      const [sx, sy] = toScreen(cc * lx - ss * ly + p.x, ss * lx + cc * ly + p.y);
      const size = Math.max(10, Math.min(34, p.tag_r * p.mpp * view.k * 0.62));
      ctx.font = `700 ${Math.max(10, size * 0.7)}px system-ui, sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.save(); ctx.globalAlpha = fitBlink();
      const t = '⚠ ' + Math.round(f.score * 100) + '%';
      ctx.lineWidth = 4; ctx.strokeStyle = 'rgba(255,255,255,.95)'; ctx.strokeText(t, sx, sy + size * 0.95);
      ctx.fillStyle = '#e00000'; ctx.fillText(t, sx, sy + size * 0.95);
      ctx.restore();
    }
  }
  function fitReport() {
    if (!buildInk()) { setStatus('Fit check: could not read the background picture yet. Try again in a moment.', 'error'); return; }
    const rows = pieces.map(p => [p, fitOf(p)]).filter(([, f]) => f && f.score != null);
    const bad = rows.filter(([, f]) => f.score < FIT_BAD).sort((x, y) => x[1].score - y[1].score);
    setStatus(bad.length
      ? `Fit check: ${bad.length} of ${rows.length} piece(s) do not follow the background lines (red). Worst: ` + bad.slice(0, 10).map(([p, f]) => `S.${p.survey} ${Math.round(f.score * 100)}%`).join(', ') + (bad.length > 10 ? '…' : '') + '. Red stretches show where the outline leaves the lines.'
      : `Fit check: all ${rows.length} piece(s) follow the background lines.`, bad.length ? 'error' : 'good');
  }

  function drawLabel(p) {
    const a = p.rot * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
    const lp = p.ctr || p.tag, lx = lp[0] * p.mpp, ly = lp[1] * p.mpp;
    const [sx, sy] = toScreen(c * lx - s * ly + p.x, s * lx + c * ly + p.y);
    const r = p.tag_r * p.mpp * view.k;
    const size = Math.max(10, Math.min(34, r * 0.62));
    ctx.font = `700 ${size}px system-ui, sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    const t = (p.locked ? '🔒 ' : '') + 'S.' + p.survey;
    ctx.lineWidth = 4; ctx.strokeStyle = 'rgba(255,255,255,.92)'; ctx.strokeText(t, sx, sy);
    ctx.fillStyle = '#c62828'; ctx.fillText(t, sx, sy);
  }
  function draw() {
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#eceff3'; ctx.fillRect(0, 0, W, H);
    drawBackground();
    const vis = [];
    for (const p of pieces) {
      const x0 = view.ox + p.bb.x0 * view.k, x1 = view.ox + p.bb.x1 * view.k, y0 = view.oy + p.bb.y0 * view.k, y1 = view.oy + p.bb.y1 * view.k;
      if (x1 < 0 || y1 < 0 || x0 > W || y0 > H) continue;
      vis.push(p); drawPiece(p, false);
    }
    if (snapTarget && drag && drag.moved) drawPiece(snapTarget, true);
    if (fitOn && bg) for (const p of vis) drawFit(p);
    for (const p of vis) drawLabel(p);
    drawPins();
  }


  // ------------------------------------------------------------------ download the puzzle only (no background image)
  // Coloured pieces + survey labels on white. Inner lines are included only when "Show inner lines" is ticked.
  async function exportPuzzle(fmt) {
    fmt = fmt || (fmtSel && fmtSel.value) || 'png';
    if (!pieces.length) { setStatus('Nothing on the board to download yet.', 'error'); return; }
    const withLines = !!(linesBox && linesBox.checked);
    setStatus('Preparing the download…');
    if (withLines) {
      await Promise.all(pieces.map(p => new Promise(res => {
        if (!p.img) { p.img = new Image(); p.img.onload = () => { p.loaded = true; res(); }; p.img.onerror = () => res(); p.img.src = p.imgUrl; }
        else if (p.loaded || p.img.complete) { p.loaded = true; res(); }
        else { p.img.addEventListener('load', () => { p.loaded = true; res(); }, { once: true }); p.img.addEventListener('error', () => res(), { once: true }); }
      })));
    }
    let x0 = 1e18, y0 = 1e18, x1 = -1e18, y1 = -1e18;
    for (const p of pieces) { x0 = Math.min(x0, p.bb.x0); y0 = Math.min(y0, p.bb.y0); x1 = Math.max(x1, p.bb.x1); y1 = Math.max(y1, p.bb.y1); }
    const bw = Math.max(1e-6, x1 - x0), bh = Math.max(1e-6, y1 - y0);
    const minMpp = Math.min(...pieces.map(p => p.mpp));
    let k = 1 / minMpp;                                          // native resolution of the sheets
    const MAXSIDE = 8000;                                        // keep the canvas inside browser limits
    k = Math.min(k, MAXSIDE / Math.max(bw, bh));
    const pad = 20, cw = Math.ceil(bw * k) + pad * 2, ch = Math.ceil(bh * k) + pad * 2;
    const cv = document.createElement('canvas'); cv.width = cw; cv.height = ch;
    const g = cv.getContext('2d');
    g.fillStyle = '#fff'; g.fillRect(0, 0, cw, ch);
    const ox = pad - x0 * k, oy = pad - y0 * k;
    for (const p of pieces) {
      g.save();
      g.translate(ox + p.x * k, oy + p.y * k); g.rotate(p.rot * Math.PI / 180);
      const sc = p.mpp * k; g.scale(sc, sc);
      const path = () => { g.beginPath(); p.poly.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); g.closePath(); };
      path(); g.fillStyle = '#fff'; g.fill();
      g.globalAlpha = 0.42; g.fillStyle = p.color; g.fill(); g.globalAlpha = 1;
      if (withLines && p.loaded) {                                // keep the sheet's own lines inside the piece, so stray ink never pokes out past the boundary
        g.save(); path(); g.clip(); g.drawImage(p.img, 0, 0, p.w, p.h); g.restore();
      }
      g.lineJoin = 'round'; g.lineWidth = 1.6 * Math.max(1, k / 40) / sc * 1; g.strokeStyle = p.color; path(); g.stroke();
      g.restore();
    }
    // ---- boundaries: a dark line on every piece edge (including where pieces touch), plus a thicker outer line around the whole puzzle.
    {
      const bwid = Math.max(2, Math.round(Math.max(cw, ch) / 1600));          // line thickness in px, grows with image size
      // edges between touching pieces (and every piece's own edge): a thinner dark line, drawn centred on the edge
      for (const p of pieces) {
        g.save(); g.translate(ox + p.x * k, oy + p.y * k); g.rotate(p.rot * Math.PI / 180);
        const sc = p.mpp * k; g.scale(sc, sc);
        g.beginPath(); p.poly.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); g.closePath();
        g.lineJoin = 'round'; g.lineWidth = Math.max(1.5, bwid * 0.5) / sc; g.strokeStyle = '#111'; g.stroke();
        g.restore();
      }
      const m = document.createElement('canvas'); m.width = cw; m.height = ch;
      const mg = m.getContext('2d');
      const each = fn => {
        for (const p of pieces) {
          mg.save(); mg.translate(ox + p.x * k, oy + p.y * k); mg.rotate(p.rot * Math.PI / 180);
          const sc = p.mpp * k; mg.scale(sc, sc);
          mg.beginPath(); p.poly.forEach(([x, y], i) => i ? mg.lineTo(x, y) : mg.moveTo(x, y)); mg.closePath();
          fn(sc); mg.restore();
        }
      };
      mg.fillStyle = mg.strokeStyle = '#111'; mg.lineJoin = 'round';
      each(sc => { mg.lineWidth = 2 * bwid / sc; mg.stroke(); mg.fill(); });  // pieces grown by bwid
      mg.globalCompositeOperation = 'destination-out';
      each(() => mg.fill());                                                   // minus the pieces themselves -> ring outside the union
      g.drawImage(m, 0, 0);
    }
    for (const p of pieces) {
      const a = p.rot * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
      const lp = p.ctr || p.tag, lx = lp[0] * p.mpp, ly = lp[1] * p.mpp;
      const sx = ox + (c * lx - s * ly + p.x) * k, sy = oy + (s * lx + c * ly + p.y) * k;
      const size = Math.max(14, Math.min(120, p.tag_r * p.mpp * k * 0.62));
      g.font = `700 ${size}px system-ui, sans-serif`; g.textAlign = 'center'; g.textBaseline = 'middle';
      const t = 'S.' + p.survey;
      g.lineWidth = Math.max(4, size / 8); g.lineJoin = 'round'; g.strokeStyle = 'rgba(255,255,255,.92)'; g.strokeText(t, sx, sy);
      g.fillStyle = '#c62828'; g.fillText(t, sx, sy);
    }
    const base = withLines ? 'puzzle_with_inner_lines' : 'puzzle';
    const save = (blob, ext) => {
      const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = base + '.' + ext;
      document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 5000);
      setStatus(`Downloaded ${a.download} (${cw}×${ch}px, ${withLines ? 'with' : 'without'} inner lines).`, 'good');
    };
    cv.toBlob(async blob => {
      if (!blob) { setStatus('Could not create the image (it may be too large).', 'error'); return; }
      if (fmt === 'png') { save(blob, 'png'); return; }
      try {                                                      // PDF / TIF are converted by the server (Pillow)
        setStatus(`Converting to ${fmt.toUpperCase()}…`);
        const r = await fetch('/api/board/convert?fmt=' + encodeURIComponent(fmt), { method: 'POST', body: blob });
        if (!r.ok) { let m = 'Conversion failed.'; try { m = (await r.json()).error || m; } catch (e) { /* ignore */ } throw new Error(m); }
        save(await r.blob(), fmt);
      } catch (e) { setStatus(e.message, 'error'); }
    }, 'image/png');
  }

  // ------------------------------------------------------------------ view
  function resize() {
    const r = wrap.getBoundingClientRect(); if (r.width < 10 || r.height < 10) return false;
    dpr = window.devicePixelRatio || 1; W = r.width; H = r.height;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    canvas.style.width = W + 'px'; canvas.style.height = H + 'px'; schedule(); return true;
  }
  function fit() {
    if (!pieces.length || !W) return;
    let x0 = 1e18, y0 = 1e18, x1 = -1e18, y1 = -1e18;
    for (const p of pieces) { x0 = Math.min(x0, p.bb.x0); y0 = Math.min(y0, p.bb.y0); x1 = Math.max(x1, p.bb.x1); y1 = Math.max(y1, p.bb.y1); }
    const k = 0.92 * Math.min(W / Math.max(1e-6, x1 - x0), H / Math.max(1e-6, y1 - y0));
    view.k = k; view.ox = (W - (x1 - x0) * k) / 2 - x0 * k; view.oy = (H - (y1 - y0) * k) / 2 - y0 * k; schedule();
  }
  function zoomAt(sx, sy, f) {
    const [wx, wy] = toWorld(sx, sy);
    view.k = Math.max(1e-3, Math.min(1e3, view.k * f)); view.ox = sx - wx * view.k; view.oy = sy - wy * view.k; schedule();
  }

  // ------------------------------------------------------------------ data
  function addPiece(d) {
    const sz = d.sz_saved || savedSizes[d.id] || 1;
    const p = Object.assign({}, d, { imgUrl: d.img, img: null, loaded: false, mpp0: d.mpp, sz, mpp: d.mpp * sz });
    p.ctr = visualCentre(p.poly);
    refresh(p); return p;
  }
  async function load(keepView) {
    if (loading) return; loading = true;
    try {
      for (;;) {
        const r = await fetch('/api/board'); const d = await r.json();
        if (!r.ok) throw new Error(d.error || 'Could not load the board.');
        if (d.state === 'preparing') { setStatus(`Preparing pieces… ${d.done || 0} of ${d.total || '?'}`); await new Promise(res => setTimeout(res, 1000)); continue; }
        const keepSel = selected && selected.id;
        const all = d.pieces.map(addPiece);
        pieces = all.filter(p => p.placed !== false); tray = all.filter(p => p.placed === false);
        pieces.forEach(p => p.placed = true);
        selected = pieces.find(p => p.id === keepSel) || null;
        const bad = Object.keys(d.errors || {}).length, unk = all.filter(p => !p.scale_known).length;
        let msg = all.length ? `${pieces.length} piece(s) on the board, ${tray.length} waiting in the tray on the left. Drag a piece from the tray (or on the board) next to another and it snaps edge to edge. Click a piece (without dragging) to open its cleaned image.` : 'No approved images yet.';
        if (unk) msg += ` ${unk} piece(s) have no readable scale (1:N), so their size is a guess.`;
        if (bad) msg += ` ${bad} sheet(s) could not be read.`;
        setStatus(msg, bad ? 'error' : '');
        if (!keepView) fit();
        all.forEach(p => { if (p.sz_saved == null && Math.abs(p.sz - 1) > 1e-6) dirty.add(p.id); }); if (dirty.size) flushSave();
        if (all.some(p => !p.saved)) { all.forEach(p => dirty.add(p.id)); flushSave(); all.forEach(p => p.saved = true); }
        renderTray();
        autoBtn.disabled = false; updateSel(); schedule(); if (!bg) bgLoad(); break;
      }
    } catch (e) { setStatus(e.message, 'error'); }
    loading = false;
  }
  function flushSave() {
    clearTimeout(saveTimer);
    const body = {};
    for (const id of dirty) { const p = pieces.find(q => q.id === id) || tray.find(q => q.id === id); if (p) body[id] = { x: p.x, y: p.y, rot: p.rot, sz: p.sz, color: p.color, placed: p.placed !== false, locked: !!p.locked }; }
    dirty.clear();
    if (!Object.keys(body).length) return;
    fetch('/api/board/layout', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pieces: body }) })
      .catch(() => setStatus('Could not save the layout.', 'error'));
  }
  function queueSave(p) { dirty.add(p.id); clearTimeout(saveTimer); saveTimer = setTimeout(flushSave, SAVE_DELAY); }

  // ------------------------------------------------------------------ undo / redo
  // Each entry is a snapshot of every piece's pose and colour, taken just before a change is made.
  const undoStack = [], redoStack = [], HIST_MAX = 100;
  let undoBtn, redoBtn, lockBtn, selSize, colorPending = false;
  const snapshot = () => pieces.map(p => ({ id: p.id, x: p.x, y: p.y, rot: p.rot, color: p.color, sz: p.sz }));
  function updateHist() { if (undoBtn) undoBtn.disabled = !undoStack.length; if (redoBtn) redoBtn.disabled = !redoStack.length; }
  function pushUndo(snap) {
    undoStack.push(snap || snapshot()); if (undoStack.length > HIST_MAX) undoStack.shift();
    redoStack.length = 0; updateHist();
  }
  function restore(snap) {
    for (const q of snap) {
      const p = pieces.find(x => x.id === q.id); if (!p || p.locked) continue;
      if (q.sz != null && Math.abs(q.sz - p.sz) > 1e-9) { p.sz = q.sz; p.mpp = p.mpp0 * q.sz; rememberSize(p); }
      p.x = q.x; p.y = q.y; p.rot = q.rot; p.color = q.color; refresh(p); dirty.add(p.id);
    }
    snapTarget = null; flushSave(); updateSel(); schedule();
  }
  function undo() {
    if (!undoStack.length || drag) return;
    if (boardLocked) { lockedMsg(); return; }
    redoStack.push(snapshot()); restore(undoStack.pop()); updateHist(); setStatus('Undid the last change.');
  }
  function redo() {
    if (!redoStack.length || drag) return;
    if (boardLocked) { lockedMsg(); return; }
    undoStack.push(snapshot()); restore(redoStack.pop()); updateHist(); setStatus('Redid the change.');
  }
  function fitWithUndo(p) { if (!p) return; if (boardLocked) { lockedMsg(); return; } if (p.locked) { pieceLockedMsg(p); return; } const before = snapshot(); if (autoFit(p)) pushUndo(before); }

  // ------------------------------------------------------------------ lock
  // One switch for the whole board. While locked, no piece can be moved, rotated, auto-fitted, auto-arranged,
  // reset, or changed by undo/redo. Dragging then pans the view, and clicking a piece still opens its clean image.
  const LOCK_KEY = 'plotboard.boardLocked';
  let boardLocked = (() => { try { return localStorage.getItem(LOCK_KEY) === '1'; } catch (e) { return false; } })();
  function updateLockUI() {
    if (lockBtn) {
      lockBtn.textContent = boardLocked ? '🔒 Unlock board' : '🔓 Lock board'; lockBtn.classList.toggle('on', boardLocked);
      lockBtn.title = boardLocked ? 'The board is LOCKED: nothing can be moved. Click to unlock (key L)' : 'Lock the whole board so nothing can be moved by accident (key L)';
    }
    if (wrap) wrap.classList.toggle('locked', boardLocked);
  }
  function toggleLock() {
    boardLocked = !boardLocked;
    try { localStorage.setItem(LOCK_KEY, boardLocked ? '1' : '0'); } catch (e) { /* storage unavailable */ }
    updateLockUI(); schedule();
    setStatus(boardLocked ? 'Board locked. Nothing can be moved until you unlock it.' : 'Board unlocked. You can move pieces again.');
  }
  const pieceLocked = p => !!(p && p.locked);
  const pieceLockedMsg = p => setStatus(`S.${p.survey} is locked. Unlock it (key K) to move or change it.`, 'error');
  const lockedMsg = () => setStatus('The board is locked. Click Unlock (or press L) to move pieces.', 'error');

  // ------------------------------------------------------------------ piece size
  // Each piece's size comes from the 1:N scale printed on its sheet. When that scale was misread (or two sheets are
  // drawn at different scales) a piece looks too big or too small next to its neighbours, so every piece has a
  // size factor: 1 = as read from the sheet. Factors are remembered in this browser, per piece.
  const LASTSZ_KEY = 'plotboard.lastSize';
  let lastSize = (() => { try { const v = parseFloat(localStorage.getItem(LASTSZ_KEY)); return isFinite(v) && v > 0 ? v : null; } catch (e) { return null; } })();
  function setLastSize(f) { lastSize = f; try { localStorage.setItem(LASTSZ_KEY, String(f)); } catch (e) { /* storage unavailable */ } }
  const SIZE_KEY = 'plotboard.sizes';
  const savedSizes = (() => { try { return JSON.parse(localStorage.getItem(SIZE_KEY) || '{}') || {}; } catch (e) { return {}; } })();
  function rememberSize(p) {
    if (Math.abs(p.sz - 1) < 1e-6) delete savedSizes[p.id]; else savedSizes[p.id] = p.sz;
    try { localStorage.setItem(SIZE_KEY, JSON.stringify(savedSizes)); } catch (e) { /* storage unavailable */ }
    queueSave(p);
  }
  // resize about the piece's own centre, so it grows or shrinks in place
  function setSize(p, f) {
    f = Math.max(0.02, Math.min(5, f));
    const k = f / p.sz; if (Math.abs(k - 1) < 1e-9) return;
    const cx = (p.bb.x0 + p.bb.x1) / 2, cy = (p.bb.y0 + p.bb.y1) / 2;
    p.x = cx + k * (p.x - cx); p.y = cy + k * (p.y - cy);
    p.sz = f; p.mpp = p.mpp0 * f; refresh(p); rememberSize(p);
  }
  function sizeSel(f) {
    const p = selected; if (!p) return;
    if (boardLocked) { lockedMsg(); updateSel(); return; }
    if (p.locked) { pieceLockedMsg(p); updateSel(); return; }
    f = Math.max(0.02, Math.min(5, f));
    setLastSize(f);                                             // the next piece taken from the tray starts at this size
    if (Math.abs(f - p.sz) < 1e-6) { updateSel(); return; }
    const was = p.sz;
    pushUndo(); setSize(p, f); queueSave(p); updateSel(); schedule();
    setStatus(`S.${p.survey} size x${f.toFixed(2)} (was x${was.toFixed(2)}). Drag it against its neighbour to snap it.`, 'good');
  }

  // ------------------------------------------------------------------ selection / controls
  function select(p) { selected = p; updateSel(); schedule(); }
  // ------------------------------------------------------------------ nudge (move the selected piece a little)
  let lastNudge = 0;
  function nudge(dx, dy, big) {
    const p = selected; if (!p) return;
    if (boardLocked) { lockedMsg(); return; }
    if (p.locked) { pieceLockedMsg(p); return; }
    const now = Date.now();
    if (now - lastNudge > 700) { pushUndo(); setStatus(`Moving S.${p.survey} with the arrows (hold Shift for bigger steps). Ctrl/Cmd+Z undoes the whole move.`); }
    lastNudge = now;
    const px = (big ? 24 : 3) / view.k;                     // 3 screen pixels per step, 24 with Shift
    p.x += dx * px; p.y += dy * px; refresh(p); snapTarget = null; queueSave(p); schedule();
  }
  function mkNudge(label, dx, dy, title) {
    const b = el('button', 'secondary', label); b.type = 'button'; b.title = title; b.style.padding = '6px 10px';
    let t1 = 0, t2 = 0; const stop = () => { clearTimeout(t1); clearInterval(t2); };
    b.addEventListener('pointerdown', ev => { ev.preventDefault(); nudge(dx, dy, ev.shiftKey); t1 = setTimeout(() => { t2 = setInterval(() => nudge(dx, dy, ev.shiftKey), 50); }, 350); });
    for (const n of ['pointerup', 'pointerleave', 'pointercancel']) b.addEventListener(n, stop);
    return b;
  }
  function togglePieceLock(p) {
    if (!p) return;
    p.locked = !p.locked; queueSave(p); updateSel(); schedule();
    setStatus(p.locked ? `S.${p.survey} locked: it will not move, turn, resize or be changed by auto-arrange/undo. Press K to unlock.` : `S.${p.survey} unlocked.`);
  }
  let pieceLockBtn;
  function updateSel() {
    if (!selected) { selBar.classList.add('hidden'); return; }
    selBar.classList.remove('hidden');
    if (pieceLockBtn) { pieceLockBtn.textContent = selected.locked ? '🔒 Unlock piece' : '🔓 Lock piece'; pieceLockBtn.classList.toggle('on', !!selected.locked); }
    selName.textContent = 'S.' + selected.survey + ' · ' + (selected.name || '');
    selColor.value = selected.color;
    if (selSize) selSize.value = selected.sz.toFixed(2);
  }
  function rotateSel(deg) {
    if (boardLocked) { lockedMsg(); return; }
    if (selected && selected.locked) { pieceLockedMsg(selected); return; }
    if (selected) pushUndo();
    const p = selected; if (!p) return;
    const cx = (p.bb.x0 + p.bb.x1) / 2, cy = (p.bb.y0 + p.bb.y1) / 2, a = deg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
    const nx = cx + c * (p.x - cx) - s * (p.y - cy), ny = cy + s * (p.x - cx) + c * (p.y - cy);
    p.x = nx; p.y = ny; p.rot += deg; refresh(p); applySnap(p); snapTarget = null; queueSave(p); schedule();
  }
  // Try turning the piece through +-30 degrees and look for the placement where its edges lie flush on neighbours
  // with the least overlap; searches a wider area than the normal drag snap.
  function autoFit(p, opt) {
    opt = opt || {};
    if (!p || p.locked) return false;
    const maxRot = opt.maxRot == null ? 30 : opt.maxRot;
    const o = { x: p.x, y: p.y, rot: p.rot };
    const cx = (p.bb.x0 + p.bb.x1) / 2, cy = (p.bb.y0 + p.bb.y1) / 2;
    let best = null;
    const rotStep = opt.rotStep || 1;
    for (let dr = -maxRot; dr <= maxRot; dr += rotStep) {
      const a = dr * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
      p.x = cx + c * (o.x - cx) - s * (o.y - cy); p.y = cy + s * (o.x - cx) + c * (o.y - cy); p.rot = o.rot + dr; refresh(p);
      const r = findSnap(p, { tolMul: opt.tolMul || 5, tolWorld: opt.tolWorld, rotTol: 2.5 * Math.PI / 180 });
      if (r && (!best || r.sc > best.sc)) best = r;
    }
    if (!best) {
      p.x = o.x; p.y = o.y; p.rot = o.rot; refresh(p);
      if (!opt.quiet) setStatus(`No neighbouring edge found for S.${p.survey}. Drag it closer to the piece it belongs next to, then try again.`, 'error');
      schedule(); return false;
    }
    p.x = best.x; p.y = best.y; p.rot = best.rot; refresh(p);
    snapTarget = null; queueSave(p); schedule();
    if (!opt.quiet) setStatus(`S.${p.survey} fitted against S.${best.target.survey}. Drag or press Auto-fit again to adjust.`, 'good');
    return true;
  }
  // After the automatic layout: pieces are roughly right, so let every piece find its neighbours and sit flush.
  async function refineAll() {
    const sizes = pieces.map(p => Math.max(p.bb.x1 - p.bb.x0, p.bb.y1 - p.bb.y0)).sort((a, b) => a - b);
    const med = sizes[Math.floor(sizes.length / 2)] || 1;
    const area = p => (p.bb.x1 - p.bb.x0) * (p.bb.y1 - p.bb.y0);
    const order = pieces.slice().sort((a, b) => area(b) - area(a));        // big pieces first: they anchor the rest
    let moved = 0;
    // one piece at a time with a pause in between, so the board stays responsive and the pieces are already visible
    for (let pass = 0; pass < 2; pass++) for (const p of order) {
      if (autoFit(p, { maxRot: 6, rotStep: 2, tolWorld: 0.15 * med, quiet: true })) moved++;
      await new Promise(r => setTimeout(r, 0));
    }
    return moved;
  }
  function bringToFront(p) { const i = pieces.indexOf(p); if (i >= 0) { pieces.splice(i, 1); pieces.push(p); } }
  function hit(sx, sy) {
    const [x, y] = toWorld(sx, sy);
    for (let i = pieces.length - 1; i >= 0; i--) {
      const p = pieces[i];
      if (x < p.bb.x0 || x > p.bb.x1 || y < p.bb.y0 || y > p.bb.y1) continue;
      if (pointIn(x, y, p)) return p;
    }
    return null;
  }
  let viewerPiece = null, plotLblBox = null;
  function showViewerImage() {
    const p = viewerPiece; if (!p) return;
    const lab = !plotLblBox || plotLblBox.checked;                // S2-1 style labels (default) or the original numbers
    modalTitle.textContent = 'S.' + p.survey + ' · ' + (p.name || '');
    modalDl.href = lab ? `/board/labelled/${p.id}/download` : p.download;
    modalDl.textContent = lab ? 'Download TIFF (S' + p.survey + '-N)' : 'Download TIFF';
    modalImg.alt = 'Cleaned survey sheet';
    modalImg.onerror = null;
    if (lab) {
      modalImg.removeAttribute('src'); modalTitle.textContent += '  – labelling plots…';
      const url = `/board/labelled/${p.id}/preview`;
      fetch(url).then(r => { if (!r.ok) throw new Error(); return r.blob(); }).then(b => {
        if (viewerPiece !== p) return;
        modalTitle.textContent = 'S.' + p.survey + ' · ' + (p.name || '');
        modalImg.src = URL.createObjectURL(b);
      }).catch(() => {                                             // could not label: fall back to the plain cleaned image
        if (viewerPiece !== p) return;
        modalTitle.textContent = 'S.' + p.survey + ' · ' + (p.name || '') + '  (plot labels unavailable)';
        modalImg.src = p.clean + '?t=' + Date.now(); modalDl.href = p.download;
      });
    } else modalImg.src = p.clean + '?t=' + Date.now();
  }
  function openViewer(p) {
    viewerPiece = p;
    modal.classList.remove('hidden');
    showViewerImage();
  }
  // Viewer zoom: mouse wheel / trackpad pinch zooms about the cursor, drag pans, double-click toggles fit <-> close-up,
  // and the +, -, Fit and 100% buttons (or keys + - 0) do the same. Zoom goes far past the picture's own pixels.
  let mbEl = null; const vz = { s: 1, tx: 0, ty: 0, fit: 1 };
  const vApply = () => { modalImg.style.transform = `translate(${vz.tx}px,${vz.ty}px) scale(${vz.s})`; };
  function viewerFit() {
    const nw = modalImg.naturalWidth, nh = modalImg.naturalHeight; if (!nw || !mbEl) return;
    const r = mbEl.getBoundingClientRect(); if (r.width < 10 || r.height < 10) return;
    const s = Math.min((r.width - 32) / nw, (r.height - 32) / nh);
    vz.fit = s; vz.s = s; vz.tx = (r.width - nw * s) / 2; vz.ty = (r.height - nh * s) / 2; vApply();
  }
  function viewerZoomAt(cx, cy, f) {
    if (!modalImg.naturalWidth) return;
    const s = Math.max(vz.fit * 0.5, Math.min(40, vz.s * f)), k = s / vz.s;
    vz.tx = cx - (cx - vz.tx) * k; vz.ty = cy - (cy - vz.ty) * k; vz.s = s; vApply();
  }
  function viewerZoomCentre(f) { if (!mbEl) return; const r = mbEl.getBoundingClientRect(); viewerZoomAt(r.width / 2, r.height / 2, f); }
  function wireViewerZoom() {
    const at = ev => { const r = mbEl.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };
    let vd = null;
    mbEl.addEventListener('wheel', ev => { ev.preventDefault(); const [x, y] = at(ev); viewerZoomAt(x, y, Math.exp(-ev.deltaY * (ev.ctrlKey ? 0.01 : 0.0018))); }, { passive: false });
    mbEl.addEventListener('pointerdown', ev => { mbEl.setPointerCapture(ev.pointerId); vd = { x: ev.clientX, y: ev.clientY, tx: vz.tx, ty: vz.ty }; mbEl.classList.add('drag'); });
    mbEl.addEventListener('pointermove', ev => { if (!vd) return; vz.tx = vd.tx + ev.clientX - vd.x; vz.ty = vd.ty + ev.clientY - vd.y; vApply(); });
    const up = () => { vd = null; mbEl.classList.remove('drag'); };
    mbEl.addEventListener('pointerup', up); mbEl.addEventListener('pointercancel', up);
    mbEl.addEventListener('dblclick', ev => {
      const [x, y] = at(ev);
      if (vz.s > vz.fit * 1.05) viewerFit(); else viewerZoomAt(x, y, Math.max(1, vz.fit * 3) / vz.s);
    });
    modalImg.addEventListener('load', viewerFit);
    window.addEventListener('resize', () => { if (!modal.classList.contains('hidden')) viewerFit(); });
  }
  const closeViewer = () => { viewerPiece = null; modal.classList.add('hidden'); modalImg.removeAttribute('src'); };

  // ------------------------------------------------------------------ pointer events
  function wire() {
    const pos = ev => { const r = canvas.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };
    canvas.addEventListener('pointerdown', ev => {
      canvas.setPointerCapture(ev.pointerId);
      if (pin) { const [qx, qy] = pos(ev); pinDown = { sx: qx, sy: qy, ox: view.ox, oy: view.oy, panned: false }; return; }
      if (bgAdjust && bg) { const [bx, by] = pos(ev); bgDrag = { sx: bx, sy: by, x: bg.x, y: bg.y }; canvas.classList.add('panning'); return; }
      const [sx, sy] = pos(ev), p = hit(sx, sy);
      if (p) {
        bringToFront(p); select(p);
        const [wx, wy] = toWorld(sx, sy);
        drag = { p, wx, wy, x: p.x, y: p.y, sx, sy, moved: false, before: snapshot(), locked: boardLocked || !!p.locked, vox: view.ox, voy: view.oy };
      } else { select(null); pan = { sx, sy, ox: view.ox, oy: view.oy }; canvas.classList.add('panning'); }
    });
    canvas.addEventListener('pointermove', ev => {
      const [sx, sy] = pos(ev);
      if (pinDown) { if (pinDown.panned || Math.hypot(sx - pinDown.sx, sy - pinDown.sy) > 4) { pinDown.panned = true; view.ox = pinDown.ox + sx - pinDown.sx; view.oy = pinDown.oy + sy - pinDown.sy; schedule(); } return; }
      if (bgDrag) { bg.x = bgDrag.x + (sx - bgDrag.sx) / view.k; bg.y = bgDrag.y + (sy - bgDrag.sy) / view.k; schedule(); return; }
      if (drag) {
        if (!drag.moved && Math.hypot(sx - drag.sx, sy - drag.sy) < 4) return;
        drag.moved = true;
        if (drag.locked) { view.ox = drag.vox + sx - drag.sx; view.oy = drag.voy + sy - drag.sy; schedule(); return; }
        const [wx, wy] = toWorld(sx, sy), p = drag.p;
        p.x = drag.x + (wx - drag.wx); p.y = drag.y + (wy - drag.wy);
        refresh(p);
        drag.free = ev.altKey;
        if (!ev.altKey) applySnap(p); else snapTarget = null;
        schedule();
      } else if (pan) {
        view.ox = pan.ox + sx - pan.sx; view.oy = pan.oy + sy - pan.sy; schedule();
      } else { canvas.style.cursor = hit(sx, sy) ? 'pointer' : 'grab'; }
    });
    const end = () => {
      if (pinDown) { const dn = pinDown; pinDown = null; if (!dn.panned && pin) pinClick(dn.sx, dn.sy); return; }
      if (bgDrag) { bgDrag = null; canvas.classList.remove('panning'); bgSaveSoon(); return; }
      if (drag) {
        if (drag.moved && drag.locked) { /* locked piece: the drag only panned the view */ }
        else if (drag.moved) {
          const p = drag.p, t = snapTarget; pushUndo(drag.before);
          // on drop, turn the piece to lie flush against whatever it was dragged next to (up to 30 degrees)
          if (snapBox.checked && !drag.free && autoFit(p, { maxRot: 30, tolMul: 3, quiet: true })) setStatus(t ? `S.${p.survey} snapped to S.${t.survey} and turned to lie flush.` : `S.${p.survey} fitted edge to edge.`, 'good');
          else { queueSave(p); setStatus(t ? `S.${p.survey} snapped to S.${t.survey}.` : `S.${p.survey} moved.`); }
        }
        else openViewer(drag.p);
      }
      drag = null; pan = null; snapTarget = null; canvas.classList.remove('panning'); schedule();
    };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
    canvas.addEventListener('wheel', ev => {
      ev.preventDefault(); const [sx, sy] = pos(ev);
      if (bgAdjust && bg) {                                    // resize (Shift: rotate) the background about the cursor
        const [wx, wy] = toWorld(sx, sy), a0 = bg.rot * Math.PI / 180, lx = wx - bg.x, ly = wy - bg.y;
        const u = (Math.cos(a0) * lx + Math.sin(a0) * ly) / bg.mpp, v = (-Math.sin(a0) * lx + Math.cos(a0) * ly) / bg.mpp;   // picture px under the cursor
        if (ev.shiftKey) bg.rot += -Math.sign(ev.deltaY || ev.deltaX) * 0.25;
        else bg.mpp = Math.max(1e-4, bg.mpp * Math.exp(-ev.deltaY * 0.0015));
        const a1 = bg.rot * Math.PI / 180;
        bg.x = wx - (Math.cos(a1) * u - Math.sin(a1) * v) * bg.mpp; bg.y = wy - (Math.sin(a1) * u + Math.cos(a1) * v) * bg.mpp;
        schedule(); bgSaveSoon(); return;
      }
      zoomAt(sx, sy, Math.exp(-ev.deltaY * 0.0015));
    }, { passive: false });
    document.addEventListener('keydown', ev => {
      if (pin && ev.key === 'Escape') { pinCancel('Cancelled.'); return; }
      if (!modal.classList.contains('hidden')) {
        if (ev.key === 'Escape') closeViewer();
        else if (ev.key === '+' || ev.key === '=') viewerZoomCentre(1.4);
        else if (ev.key === '-' || ev.key === '_') viewerZoomCentre(1 / 1.4);
        else if (ev.key === '0') viewerFit();
        return;
      }
      if ((ev.metaKey || ev.ctrlKey) && !ev.altKey && wrap.offsetParent && !ev.target.closest('input,textarea,select')) {
        const k = ev.key.toLowerCase();
        if (k === 'z') { ev.preventDefault(); if (ev.shiftKey) redo(); else undo(); return; }
        if (k === 'y') { ev.preventDefault(); redo(); return; }
      }
      if (!selected || ev.target.closest('input,textarea,select') || !wrap.offsetParent) return;
      const arrows = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
      if (arrows[ev.key]) { ev.preventDefault(); nudge(arrows[ev.key][0], arrows[ev.key][1], ev.shiftKey); return; }
      const step = ev.shiftKey ? 90 : 1;
      if (ev.key === 'q' || ev.key === 'Q') rotateSel(-step);
      else if (ev.key === 'e' || ev.key === 'E') rotateSel(step);
      else if (ev.key === 'f' || ev.key === 'F') fitWithUndo(selected);
      else if (ev.key === 'l' || ev.key === 'L') toggleLock();
      else if (ev.key === 't' || ev.key === 'T') sendToTray(selected);
      else if (ev.key === 'k' || ev.key === 'K') togglePieceLock(selected);
      else if (ev.key === '[') sizeSel(selected.sz / 1.02);
      else if (ev.key === ']') sizeSel(selected.sz * 1.02);
    });
    new ResizeObserver(() => {
      if (resize() && !started) { started = true; load(false); }
    }).observe(wrap);
  }

  // ------------------------------------------------------------------ UI
  let autoBtn;
  function build() {
    root = document.getElementById('board-root'); if (!root) return false;
    const style = el('style'); style.textContent = `
      #board-root .bar{display:flex;gap:10px;align-items:stretch;flex-wrap:wrap;margin:10px 0}
      #board-root .grp{display:flex;gap:6px;align-items:center;flex-wrap:wrap;position:relative;padding:22px 10px 9px;background:#fff;border:1px solid #d5d9df;border-radius:10px}
      #board-root .grp .gt{position:absolute;top:6px;left:11px;font-size:10px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:#6b7280}
      #board-root .bar label,#board-root .sel label{font-size:13px;display:flex;gap:6px;align-items:center;color:#1f2933}
      #board-root .bar select,#board-root .sel select{padding:5px 6px;font-size:13px;border:1px solid #cfd4db;border-radius:8px;background:#fff}
      #board-root .bar button,#board-root .sel button{padding:6px 11px;font-size:13px;line-height:1.2;color:#1f2933;background:#fff;border:1px solid #cfd4db;border-radius:8px;cursor:pointer;white-space:nowrap}
      #board-root .bar button:hover:not(:disabled),#board-root .sel button:hover:not(:disabled){background:#f1f5fb;border-color:#0b6bcb}
      #board-root .bar button:disabled,#board-root .sel button:disabled{opacity:.4;cursor:not-allowed}
      #board-root button.b-pri{background:#0b6bcb;border-color:#0b6bcb;color:#fff;font-weight:600}
      #board-root .bar button.b-pri:hover:not(:disabled){background:#0957a6;border-color:#0957a6}
      #board-root button.b-danger{color:#c62828;border-color:#e6b8b8}
      #board-root .bar button.b-danger:hover:not(:disabled){background:#fdecea;border-color:#c62828}
      #board-root button.on{background:#e8f1fc;border-color:#0b6bcb;color:#0b4f96;font-weight:600}
      #board-root button.lock.on{background:#fdecea;border-color:#c62828;color:#c62828}
      #board-root .status{margin:6px 0}
      #board-root .stage{display:flex;gap:8px;align-items:stretch}
      #board-root:fullscreen{background:#fff;padding:10px;box-sizing:border-box;height:100vh;display:flex;flex-direction:column;overflow:auto}
      #board-root:fullscreen .stage{flex:1;min-height:0}
      #board-root:fullscreen .tray,#board-root:fullscreen .wrap{height:auto;min-height:0}
      #board-root .tray{width:190px;flex:none;height:74vh;min-height:420px;overflow:auto;background:#fff;border:1px solid #d5d9df;border-radius:10px;padding:8px;box-sizing:border-box}
      #board-root .tray h4{margin:0 0 6px;font-size:14px}
      #board-root .tray input{width:100%;box-sizing:border-box;margin-bottom:6px;padding:4px 6px}
      #board-root .tray .it{display:flex;gap:8px;align-items:center;padding:4px;border:1px solid #e3e6ea;border-radius:8px;margin-bottom:6px;cursor:grab;background:#fafbfc;user-select:none}
      #board-root .tray .it:hover{border-color:#0b6bcb}
      #board-root .tray .it canvas{width:64px;height:48px;flex:none}
      #board-root .tray .it span{font-weight:600;font-size:13px}
      #board-root .tray .empty{font-size:13px;color:#667}
      #board-root .wrap{flex:1;min-width:0;position:relative;height:74vh;min-height:420px;background:#eceff3;border:1px solid #d5d9df;border-radius:10px;overflow:hidden}
      #board-root .wrap:fullscreen{height:100vh;border-radius:0}
      #board-root .wrap.locked{border-color:#c62828;box-shadow:0 0 0 2px rgba(198,40,40,.25)}
      #board-root canvas{display:block;cursor:grab;touch-action:none}
      #board-root canvas.panning{cursor:grabbing}
      #board-root .wrap.bgadjust{border-color:#0b6bcb;box-shadow:0 0 0 2px rgba(11,107,203,.25)}
      #board-root .wrap.dropping{outline:3px dashed #0b6bcb;outline-offset:-6px}
      #board-root .sel{display:flex;gap:10px;align-items:stretch;flex-wrap:wrap;margin:10px 0}
      #board-root .sel.hidden{display:none}
      #board-root .sel input[type=color]{width:38px;height:30px;padding:0;border:1px solid #bbb;border-radius:6px;background:none}
      #board-root .sel .nm{font-weight:600;margin-right:6px}
      #board-modal{position:fixed;inset:0;z-index:9999;background:rgba(15,18,24,.78);display:flex;flex-direction:column}
      #board-modal.hidden{display:none}
      #board-modal .mh{display:flex;gap:10px;align-items:center;justify-content:space-between;padding:10px 16px;background:#fff}
      #board-modal .mb{flex:1;overflow:hidden;position:relative;cursor:grab;touch-action:none}
      #board-modal .mb.drag{cursor:grabbing}
      #board-modal img{position:absolute;left:0;top:0;max-width:none;max-height:none;transform-origin:0 0;background:#fff;box-shadow:0 4px 24px rgba(0,0,0,.4);user-select:none;-webkit-user-drag:none;will-change:transform}`;
    document.head.appendChild(style);

    const bar = el('div', 'bar'); let grp = bar;
    const group = (title) => { grp = el('div', 'grp'); grp.appendChild(el('span', 'gt', title)); bar.appendChild(grp); return grp; };
    const mk = (t, f, cls, tip) => { const b = el('button', cls || 'secondary', t); b.type = 'button'; if (tip) b.title = tip; b.addEventListener('click', f); grp.appendChild(b); return b; };

    group('View');
    mk('Fit all', fit, 'secondary', 'Zoom out so every piece is visible');
    mk('＋', () => zoomAt(W / 2, H / 2, 1.3), 'secondary', 'Zoom in (or use the mouse wheel)');
    mk('－', () => zoomAt(W / 2, H / 2, 1 / 1.3), 'secondary', 'Zoom out');
    const fsBtn = mk('⛶ Full screen', () => { if (document.fullscreenElement) document.exitFullscreen(); else root.requestFullscreen && root.requestFullscreen(); });
    document.addEventListener('fullscreenchange', () => {          // the button turns into "Exit full screen" while full screen is on (Esc works too)
      const on = document.fullscreenElement === root;
      fsBtn.textContent = on ? '⤡ Exit full screen' : '⛶ Full screen';
      fsBtn.title = on ? 'Leave full screen (Esc)' : 'Show the whole board, toolbar and tray full screen';
    });

    group('Edit');
    undoBtn = mk('↶ Undo', undo, 'secondary', 'Undo (Ctrl/Cmd+Z)');
    redoBtn = mk('↷ Redo', redo, 'secondary', 'Redo (Ctrl/Cmd+Shift+Z or Ctrl+Y)');
    lockBtn = mk('🔓 Lock board', toggleLock, 'secondary lock');
    updateHist();
    const sl = el('label'); snapBox = el('input'); snapBox.type = 'checkbox'; snapBox.checked = true;
    sl.append(snapBox, document.createTextNode('Snap edges')); sl.title = 'Pieces stick edge to edge when dragged close. Hold Alt while dragging to move freely.'; grp.appendChild(sl);
    const ll = el('label'); linesBox = el('input'); linesBox.type = 'checkbox'; linesBox.checked = false;
    linesBox.addEventListener('change', schedule);
    ll.append(linesBox, document.createTextNode('Inner lines')); ll.title = 'Also draw the lines inside each sheet (plot boundaries), not just its outer outline'; grp.appendChild(ll);

    group('Arrange');
    autoBtn = mk('✨ Auto-arrange', async () => {
      if (boardLocked) { lockedMsg(); return; }
      const before = snapshot();
      setStatus('Arranging the pieces automatically…');
      try {
        let r = await fetch('/api/board/auto', { method: 'POST' }), d = await r.json();
        if (r.status === 409 && d.preparing) {          // a newly approved sheet is still being prepared: wait, then retry
          loading = false; await load(true);
          r = await fetch('/api/board/auto', { method: 'POST' }); d = await r.json();
        }
        if (!r.ok) throw new Error(d.error === 'preparing' ? 'Pieces are still being prepared. Try again in a moment.' : (d.error || 'No automatic match available.'));
        for (const p of tray) { p.placed = true; pieces.push(p); } tray = []; renderTray();
        for (const p of pieces) { const q = d.poses[p.id]; if (q && !p.locked) { p.x = q.x; p.y = q.y; p.rot = q.rot; refresh(p); dirty.add(p.id); } }
        pushUndo(before); pieces.forEach(p => dirty.add(p.id));
        flushSave(); fit(); schedule();
        setStatus('Pieces placed. Fine-tuning the edges…');            // the pieces are on screen now; the edge fitting follows
        await new Promise(r => setTimeout(r, 30));
        await refineAll(); pieces.forEach(p => dirty.add(p.id));
        flushSave(); fit(); schedule(); setStatus(`Pieces auto-arranged (${d.joined} joined group(s)). Drag pieces to correct anything.`, 'good');
      } catch (e) { setStatus(e.message, 'error'); }
    }, 'secondary b-pri');
    autoBtn.title = 'Place all pieces automatically from the survey numbers, then fine-tune the edges';
    mk('Collect loose → tray', collectLoose, 'secondary', 'Send every piece that touches no other piece back to the tray on the left');
    mk('🎨 New colours', () => {
      pushUndo(); const off = Math.random();
      pieces.concat(tray).forEach((p, i) => { p.color = hsv((off + i * 0.61803398875) % 1, i % 2 ? 0.78 : 0.62, i % 3 ? 0.92 : 0.8); dirty.add(p.id); });
      flushSave(); updateSel(); renderTray(); schedule();
    }, 'secondary');
    group('Danger');
    mk('Reset layout', async () => {
      if (boardLocked) { lockedMsg(); return; }
      if (!window.confirm('Take every piece off the board and put it back in the tray? Your manual placement will be lost.')) return;
      const before = snapshot(); await fetch('/api/board/reset', { method: 'POST' }); await load(false); pushUndo(before);
    }, 'secondary b-danger');

    group('Background');
    bgInput = el('input'); bgInput.type = 'file'; bgInput.accept = 'image/*,.tif,.tiff'; bgInput.style.display = 'none';
    bgInput.addEventListener('change', () => { bgUpload(bgInput.files[0]); bgInput.value = ''; });
    grp.appendChild(bgInput);
    bgBtn = mk('🖼 Upload image…', () => bgInput.click(), 'secondary', 'Upload an image or TIFF (or drop / paste one on the board) to sit behind the pieces');
    bgAdjBtn = mk('✥ Move background', () => { if (!bg) return; bgAdjust = !bgAdjust; if (bgAdjust) select(null); updateBgUI(); schedule(); setStatus(bgAdjust ? 'Moving the background: drag to move, wheel to resize, Shift+wheel to rotate. Press Done when it lines up.' : 'Background fixed. Pieces can be moved again.'); });
    bgAdjBtn.title = 'Drag = move, mouse wheel = resize, Shift+wheel = rotate';
    bgScaleBtn = mk('📏 Set scale…', () => pinStart('scale'), 'secondary', 'Click two points on the background and enter their real distance in metres (use the scale bar)');
    const ol = el('label'); ol.append(document.createTextNode('Opacity'));
    bgOpacity = el('input'); bgOpacity.type = 'range'; bgOpacity.min = '5'; bgOpacity.max = '100'; bgOpacity.value = '60'; bgOpacity.style.width = '90px';
    bgOpacity.addEventListener('input', () => { if (bg) { bg.opacity = bgOpacity.value / 100; schedule(); bgSaveSoon(); } });
    ol.appendChild(bgOpacity); grp.appendChild(ol);
    bgClearBtn = mk('Remove', bgClear, 'secondary b-danger', 'Remove the background image');

    group('Check against background');
    fitBtn2 = mk('◎ Check fit', () => {
      if (!bg) { setStatus('Upload a background image first.', 'error'); return; }
      fitOn = !fitOn; fitBtn2.textContent = fitOn ? '■ Stop checking' : '◎ Check fit'; fitBtn2.classList.toggle('on', fitOn);
      if (fitOn) fitReport();
      fitAnimate(fitOn);
      schedule();
    }, 'secondary', 'Blink red on every stretch of a piece outline that does not sit on a line of the background, and flag pieces that do not match it');
    const fl = el('label'); fl.append(document.createTextNode('Tolerance'));
    fitSel = el('select'); fitSel.title = 'How close to a background line an outline must be to count as matching';
    [['4', 'Strict'], ['8', 'Normal'], ['16', 'Loose']].forEach(([v, t]) => { const o = el('option', '', t); o.value = v; if (v === '8') o.selected = true; fitSel.appendChild(o); });
    fitSel.addEventListener('change', () => { fitTolPx = +fitSel.value; if (fitOn) fitReport(); schedule(); });
    fl.appendChild(fitSel); grp.appendChild(fl);

    group('Export');
    fmtSel = el('select'); fmtSel.title = 'File type for the download';
    [['png', 'PNG'], ['pdf', 'PDF'], ['tif', 'TIF']].forEach(([v, t]) => { const o = el('option', '', t); o.value = v; fmtSel.appendChild(o); });
    grp.appendChild(fmtSel);
    mk('⬇ Download puzzle', () => exportPuzzle(fmtSel.value), 'secondary', 'Save the coloured puzzle (no background image) as PNG, PDF or TIF. Inner lines are included only if "Inner lines" is ticked');
    root.appendChild(bar);
    statusEl = el('div', 'status'); statusEl.setAttribute('role', 'status'); root.appendChild(statusEl);

    selBar = el('div', 'sel hidden'); selName = el('span', 'nm');
    selColor = el('input'); selColor.type = 'color'; selColor.title = 'Colour of this piece';
    selColor.addEventListener('input', () => { if (selected) { if (!colorPending) { pushUndo(); colorPending = true; } selected.color = selColor.value; queueSave(selected); schedule(); } });
    selColor.addEventListener('change', () => { colorPending = false; });
    const rb = (t, d, title) => { const b = el('button', 'secondary', t); b.type = 'button'; b.title = title; b.addEventListener('click', () => rotateSel(d)); return b; };
    const fitBtn = el('button', 'secondary', 'Auto-fit piece'); fitBtn.type = 'button'; fitBtn.title = 'Rotate and slide this piece so its edges sit flush on its neighbours (key F)'; fitBtn.addEventListener('click', () => fitWithUndo(selected));
    const open = el('button', 'secondary', '🔍 Open clean image'); open.type = 'button'; open.title = 'View the cleaned sheet of this piece'; open.addEventListener('click', () => selected && openViewer(selected));
    const szWrap = el('label'); szWrap.style.cssText = 'display:flex;gap:6px;align-items:center;font-size:14px';
    selSize = el('input'); selSize.type = 'number'; selSize.min = '0.02'; selSize.max = '5'; selSize.step = '0.01'; selSize.style.width = '80px';
    selSize.title = 'Size of this piece: 1 = as read from the sheet scale. Keys [ and ] change it by 2%.';
    selSize.addEventListener('change', () => { const f = parseFloat(selSize.value); if (selected && isFinite(f) && f > 0) sizeSel(f); else updateSel(); });
    szWrap.append(document.createTextNode('×'), selSize);
    const szReset = el('button', 'secondary', 'Reset size'); szReset.type = 'button'; szReset.title = 'Back to the size read from the sheet';
    szReset.addEventListener('click', () => sizeSel(1));
    pieceLockBtn = el('button', 'secondary lock', '🔓 Lock piece'); pieceLockBtn.type = 'button'; pieceLockBtn.title = 'Lock just this piece so it cannot be moved by accident (key K)'; pieceLockBtn.addEventListener('click', () => togglePieceLock(selected));
    const nudgeBox = el('span'); nudgeBox.style.cssText = 'display:inline-flex;gap:4px;align-items:center';
    nudgeBox.append(mkNudge('◀', -1, 0, 'Move left (←). Click-and-hold repeats, Shift = bigger steps'), mkNudge('▲', 0, -1, 'Move up (↑)'), mkNudge('▼', 0, 1, 'Move down (↓)'), mkNudge('▶', 1, 0, 'Move right (→)'));
    const toTray = el('button', 'secondary', '↩ Send to tray'); toTray.type = 'button'; toTray.title = 'Take this piece off the board and put it back in the tray (key T)'; toTray.addEventListener('click', () => sendToTray(selected));
    const szNb = el('button', 'secondary', 'Auto-size to neighbours'); szNb.type = 'button'; szNb.title = 'Try other sizes for this piece and keep the one where its outline and corners meet the pieces next to it';
    szNb.addEventListener('click', () => sizeOne(selected));
    const szCh = el('button', 'secondary', 'Auto-size chain from here'); szCh.type = 'button'; szCh.title = 'Treat this piece as correct, then fit the pieces touching it, then the ones touching those, and so on';
    szCh.addEventListener('click', () => sizeChain(selected));
    pinBtn = el('button', 'secondary', 'Pin-fit to background'); pinBtn.type = 'button'; pinBtn.title = 'Click the same two spots on this piece and on the background: the piece is moved, turned and resized to match';
    pinBtn.addEventListener('click', () => pinStart('fit'));
    const sg = (title) => { const g = el('div', 'grp'); g.appendChild(el('span', 'gt', title)); selBar.appendChild(g); return g; };
    sg('Selected piece').append(selName, selColor, pieceLockBtn, toTray, open);
    sg('Move').append(nudgeBox);
    sg('Rotate').append(rb('↺ 90°', -90, 'Rotate left 90° (Shift+Q)'), rb('↺ 1°', -1, 'Rotate left 1° (Q)'), rb('↻ 1°', 1, 'Rotate right 1° (E)'), rb('↻ 90°', 90, 'Rotate right 90° (Shift+E)'));
    sg('Size').append(szWrap, szReset);
    sg('Fit this piece').append(fitBtn, szNb, szCh, pinBtn);
    root.appendChild(selBar);

    wrap = el('div', 'wrap'); canvas = el('canvas'); wrap.appendChild(canvas);
    stage = el('div', 'stage');
    const trayBox = el('div', 'tray'); trayCount = el('h4', '', 'Not placed yet (0)');
    trayFilter = el('input'); trayFilter.type = 'search'; trayFilter.placeholder = 'Find S.number…'; trayFilter.addEventListener('input', renderTray);
    trayList = el('div', 'list'); trayBox.append(trayCount, trayFilter, trayList);
    stage.append(trayBox, wrap); root.appendChild(stage);
    ctx = canvas.getContext('2d');
    updateLockUI();
    updateBgUI();
    const hasFiles = ev => ev.dataTransfer && Array.from(ev.dataTransfer.types || []).includes('Files');
    const isPiece = ev => ev.dataTransfer && Array.from(ev.dataTransfer.types || []).includes('application/x-plotpiece');
    wrap.addEventListener('dragover', ev => { if (isPiece(ev)) ev.preventDefault(); });
    wrap.addEventListener('drop', ev => {
      if (!isPiece(ev)) return; ev.preventDefault();
      const id = ev.dataTransfer.getData('application/x-plotpiece'), p = tray.find(q => q.id === id), r = canvas.getBoundingClientRect();
      if (p) placeFromTray(p, ev.clientX - r.left, ev.clientY - r.top);
    });
    wrap.addEventListener('dragover', ev => { if (hasFiles(ev)) { ev.preventDefault(); wrap.classList.add('dropping'); } });
    wrap.addEventListener('dragleave', () => wrap.classList.remove('dropping'));
    wrap.addEventListener('drop', ev => {
      wrap.classList.remove('dropping');
      if (hasFiles(ev) && ev.dataTransfer.files.length) { ev.preventDefault(); bgUpload(ev.dataTransfer.files[0]); }
    });
    document.addEventListener('paste', ev => {
      if (!wrap.offsetParent || (ev.target && ev.target.closest && ev.target.closest('input,textarea,select'))) return;
      const it = Array.from((ev.clipboardData && ev.clipboardData.items) || []).find(i => i.kind === 'file' && i.type.startsWith('image/'));
      if (it) { ev.preventDefault(); bgUpload(it.getAsFile()); }
    });

    modal = el('div', 'hidden'); modal.id = 'board-modal';
    const mh = el('div', 'mh'); modalTitle = el('strong');
    const right = el('div'); right.style.cssText = 'display:flex;gap:8px';
    modalDl = el('a', 'button', 'Download TIFF'); modalDl.style.textDecoration = 'none';
    const close = el('button', 'secondary', 'Close'); close.type = 'button'; close.addEventListener('click', closeViewer);
    const pl = el('label'); pl.style.cssText = 'display:flex;gap:6px;align-items:center;font-size:14px';
    plotLblBox = el('input'); plotLblBox.type = 'checkbox'; plotLblBox.checked = true;
    plotLblBox.addEventListener('change', showViewerImage);
    pl.append(plotLblBox, document.createTextNode('Show plot labels as S-number'));
    pl.title = 'Plot numbers are shown as S<survey>-<plot>, e.g. S2-1, S2-2';
    const zb = (t, title, f) => { const b = el('button', 'secondary', t); b.type = 'button'; b.title = title; b.addEventListener('click', f); return b; };
    right.append(pl, zb('−', 'Zoom out (key -)', () => viewerZoomCentre(1 / 1.5)), zb('+', 'Zoom in (key +). Mouse wheel / pinch also zooms, drag moves the picture', () => viewerZoomCentre(1.5)),
                 zb('Fit', 'Fit the whole sheet in the window (key 0, or double-click)', viewerFit), zb('100%', 'Show the sheet at its real pixel size', () => viewerZoomCentre(1 / vz.s)), modalDl, close);
    mh.append(modalTitle, right);
    const mb = el('div', 'mb'); modalImg = el('img'); modalImg.alt = 'Cleaned survey sheet'; modalImg.draggable = false;
    mbEl = mb; mb.appendChild(modalImg); modal.append(mh, mb); root.appendChild(modal);
    wireViewerZoom();
    return true;
  }
  function hsv(h, s, v) {
    const i = Math.floor(h * 6), f = h * 6 - i, p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
    const [r, g, b] = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6];
    return '#' + [r, g, b].map(x => Math.round(x * 255).toString(16).padStart(2, '0')).join('');
  }

  // ------------------------------------------------------------------ "Back to Cleaner" button
  // The button sits at the very bottom of the page; move it up next to the Refresh button above the board.
  // It is found by its label, because the page template is not part of this file. The original element is moved
  // (not copied), so its link or click handler keeps working.
  function moveBackButton() {
    const label = e => (e.textContent || '').trim().toLowerCase();
    const all = Array.from(document.querySelectorAll('button, a'));
    const back = all.find(e => label(e) === 'back to cleaner');
    const refreshes = all.filter(e => label(e) === 'refresh');
    if (!back || !refreshes.length) return false;
    const board = document.getElementById('board-root');
    // the Refresh button that sits just above the puzzle board (falls back to the first one on the page)
    const above = board ? refreshes.filter(r => r.compareDocumentPosition(board) & Node.DOCUMENT_POSITION_FOLLOWING) : [];
    const refresh = above.length ? above[above.length - 1] : refreshes[0];
    if (back.nextElementSibling === refresh) return true;          // already in place
    back.style.marginRight = '8px';
    refresh.parentNode.insertBefore(back, refresh);
    return true;
  }
  function watchBackButton() {
    if (moveBackButton()) return;
    const mo = new MutationObserver(() => { if (moveBackButton()) mo.disconnect(); });   // the page may build these buttons later
    mo.observe(document.body, { childList: true, subtree: true });
    setTimeout(() => mo.disconnect(), 20000);
  }

  // ------------------------------------------------------------------ public
  PB.open = () => {
    if (!root && build()) wire();
    if (root) { if (!resize()) return; if (!started) { started = true; load(false); } else load(true); }   // picks up newly approved sheets
  };
  PB.reload = () => { if (started) load(true); };
  const boot = () => { if (!root && build()) wire(); watchBackButton(); };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();