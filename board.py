"""Manual puzzle board for approved survey sheets.

Every approved sheet becomes a *piece*: its filled outline (polygon) plus a transparent line-art picture.
The browser (board.js) lets the person drag pieces around; a piece that comes close to another piece sticks
edge-to-edge.  Each piece has its own colour (assigned automatically, changeable) and is labelled S.<survey no.>.
Clicking a piece opens its cleaned image.

Positions are stored in ground metres (pieces from different scales stay comparable), so a piece drawn from a
1:500 sheet is physically smaller than one from a 1:2000 sheet, exactly as on the ground.

Files written under <puzzle_dir>/board/:  <id>.png (line art), <id>.json (polygon etc.), and
<puzzle_dir>/board_layout.json (positions + colours that the person chose).
"""
import colorsys
import json
import math
import os
import threading
import time

import cv2
import numpy as np
from PIL import Image

import puzzle

Image.MAX_IMAGE_PIXELS = None
BOARD_VERSION = 5
PIECE_IMG_MAX = 1600          # longest side of a piece picture, px
DEFAULT_MPP = 0.5             # metres per piece-picture pixel when no sheet of the village has a readable scale

STATE = {'state': 'idle', 'done': 0, 'total': 0, 'message': '', 'errors': {}}
LOCK = threading.Lock()
LAYOUT_LOCK = threading.Lock()


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def color_for(index):
    """Distinct, pleasant colours: golden-angle hue steps."""
    h = (index * 0.61803398875) % 1.0
    s = 0.62 if index % 2 == 0 else 0.78
    v = 0.92 if index % 3 else 0.80
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))


def _paths(puzzle_dir, item_id):
    d = os.path.join(puzzle_dir, 'board')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'{item_id}.json'), os.path.join(d, f'{item_id}.png')


def _signature(item):
    st = os.stat(item['path'])
    meta = item.get('meta') or {}
    return [BOARD_VERSION, st.st_mtime_ns, st.st_size, str(meta.get('scale') or ''), str(meta.get('survey') or ''), str(meta.get('area_m2') or '')]


def _read_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_json(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(data, fh)
    os.replace(tmp, path)


def _survey_label(item, sheet):
    s = (sheet.get('meta') or {}).get('survey') or (item.get('meta') or {}).get('survey')
    if s:
        return str(s)
    stem = os.path.splitext(item.get('name') or '')[0]
    return stem or '?'



# --------------------------------------------------------------------------------------
# plot-number labels:  "1"  ->  "S2-1"   (only for the viewer / download inside the puzzle board)
# --------------------------------------------------------------------------------------
PREVIEW_SIDE = 1800


def _glyph_groups(gray, inside, log=print):
    """Find the plot-number strings of a cleaned sheet (black ink on white).
    Returns (groups, labels): groups = [{'ids': [component ids], 'box': (x0, y0, x1, y1), 'h': digit height}]."""
    H, W = gray.shape
    long_side = max(H, W)
    ink = (gray < 170).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return [], lab
    x, y, w, h, a = (st[:, k] for k in range(5))
    sz = np.maximum(w, h)
    lo, hi = 0.0025 * long_side, 0.035 * long_side          # a digit is small next to the line network
    cand = [i for i in range(1, n) if lo <= sz[i] <= hi and a[i] >= 0.08 * sz[i] * sz[i] and inside(cen[i][0], cen[i][1])]
    if not cand:
        return [], lab
    cand = np.array(cand)
    par = {int(i): int(i) for i in cand}

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for ii in cand:
        for jj in cand:
            if jj <= ii:
                continue
            hi_, hj = h[ii], h[jj]
            ratio = max(hi_, hj) / max(1.0, min(hi_, hj))
            if ratio > 2.0:
                continue
            ref = max(hi_, hj)
            gx = max(0, x[jj] - (x[ii] + w[ii]), x[ii] - (x[jj] + w[jj]))
            if gx > 0.55 * ref:
                continue
            # same text row: vertical centres close
            if abs((y[ii] + h[ii] / 2.0) - (y[jj] + h[jj] / 2.0)) > 0.45 * ref:
                continue
            par[find(int(ii))] = find(int(jj))
    groups = {}
    for i in cand:
        groups.setdefault(find(int(i)), []).append(int(i))
    out = []
    for ids in groups.values():
        x0 = min(x[i] for i in ids); y0 = min(y[i] for i in ids)
        x1 = max(x[i] + w[i] for i in ids); y1 = max(y[i] + h[i] for i in ids)
        out.append({'ids': ids, 'box': (int(x0), int(y0), int(x1), int(y1)), 'h': float(np.median([h[i] for i in ids]))})
    return out, lab


def _prefix_img(text, h_px, thick):
    """Render `text` (Hershey) as a dark-on-white strip whose capital height is h_px."""
    (tw, th), base = cv2.getTextSize('S2-', cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)
    scale = max(0.2, h_px / float(th))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = thick + 1
    img = np.full((th + base + 2 * pad, tw + 2 * pad), 255, np.uint8)
    cv2.putText(img, text, (pad, pad + th), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thick, cv2.LINE_AA)
    return img


def relabel_plots(gray, inside, survey, log=print):
    """Return a copy of `gray` where every plot number N is shown as S<survey>-N (the original digits are kept
    and moved, so no OCR is needed), plus the number of labels changed."""
    groups, lab = _glyph_groups(gray, inside, log)
    res = gray.copy()
    if not groups:
        return res, 0
    # 1) lift every found string out of the picture
    saved = []
    for g in groups:
        x0, y0, x1, y1 = g['box']
        m = np.isin(lab[y0:y1, x0:x1], g['ids'])
        patch = gray[y0:y1, x0:x1].copy()
        saved.append((g, m, patch))
        reg = res[y0:y1, x0:x1]
        reg[m] = 255
    # 2) put it back, shifted right, with "S<survey>-" in front (centred on the old position)
    text = f'S{survey}-'
    H, W = res.shape
    for g, m, patch in saved:
        x0, y0, x1, y1 = g['box']
        gh = max(6.0, g['h'])
        thick = max(1, int(round(gh * 0.085)))
        pre = _prefix_img(text, gh, thick)
        pw = pre.shape[1] - 2 * (thick + 1)
        gap = max(2, int(0.12 * gh))
        total = pw + gap + (x1 - x0)
        left = int(round((x0 + x1) / 2.0 - total / 2.0))
        # prefix, vertically aligned to the bottom of the digits
        py0 = int(round(y1 - gh - (thick + 1)))
        px0 = left - (thick + 1)
        ph, pwid = pre.shape
        ya, yb = max(0, py0), min(H, py0 + ph)
        xa, xb = max(0, px0), min(W, px0 + pwid)
        if yb > ya and xb > xa:
            sub = pre[ya - py0:yb - py0, xa - px0:xb - px0]
            res[ya:yb, xa:xb] = np.minimum(res[ya:yb, xa:xb], sub)
        # original digits
        nx0 = left + pw + gap
        ya, yb = max(0, y0), min(H, y1)
        xa, xb = max(0, nx0), min(W, nx0 + (x1 - x0))
        if yb > ya and xb > xa:
            mm = m[ya - y0:yb - y0, xa - nx0:xb - nx0]
            pp = patch[ya - y0:yb - y0, xa - nx0:xb - nx0]
            reg = res[ya:yb, xa:xb]
            reg[mm] = np.minimum(reg[mm], pp[mm])
    log(f'  relabelled {len(groups)} plot number(s) as {text}N')
    return res, len(groups)


# --------------------------------------------------------------------------------------
# one piece
# --------------------------------------------------------------------------------------
def prepare_piece(item, puzzle_dir):
    """Build <id>.png and <id>.json for one approved sheet.  item: dict(id, path, name, meta)."""
    jpath, ppath = _paths(puzzle_dir, item['id'])
    sig = _signature(item)
    old = _read_json(jpath)
    if old and old.get('sig') == sig and os.path.isfile(ppath):
        return old
    sheet = puzzle.load_sheet(item, puzzle_dir)
    if not sheet.get('ok'):
        raise RuntimeError(sheet.get('error') or 'no drawing found in this sheet')
    fill4 = (sheet['fill4'] > 127).astype(np.uint8)
    cnts, _ = cv2.findContours(fill4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError('could not find the outline of this sheet')
    c = max(cnts, key=cv2.contourArea)
    ap = cv2.approxPolyDP(c, 1.1, True)[:, 0, :].astype(np.float64)
    if len(ap) < 3:
        raise RuntimeError('outline too small')
    x0, y0, _, _ = sheet['zone']
    # fill4 pixel i covers analysis pixels 4i..4i+3, so its centre is 4i + 1.5
    poly_a = ap * 4.0 + 1.5 + np.array([x0, y0], float)
    full = puzzle._read_gray(item['path'])
    H, W = full.shape
    rf = max(H, W) / float(puzzle.AN_LONG)
    # show plot numbers as S<survey>-N on the piece picture too (same as the cleaned-sheet viewer), so the puzzle
    # download with inner lines carries the full labels
    try:
        poly_full = (poly_a * rf).astype(np.float32).reshape(-1, 1, 2)
        inside = lambda cx, cy: cv2.pointPolygonTest(poly_full, (float(cx), float(cy)), False) >= 0
        full, _ = relabel_plots(full, inside, _survey_label(item, sheet), log=lambda m: print('[board]' + m))
    except Exception as exc:                                   # never lose a piece because of a label problem
        print('[board] could not relabel plots:', exc)
    bx0, by0, bx1, by1 = sheet['bbox']
    lo = np.minimum(poly_a.min(0), [bx0, by0]) - 6
    hi = np.maximum(poly_a.max(0), [bx1, by1]) + 6
    cx0, cy0 = max(0, int(math.floor(lo[0] * rf))), max(0, int(math.floor(lo[1] * rf)))
    cx1, cy1 = min(W, int(math.ceil(hi[0] * rf))), min(H, int(math.ceil(hi[1] * rf)))
    crop = 255 - full[cy0:cy1, cx0:cx1]
    f = min(1.0, PIECE_IMG_MAX / float(max(crop.shape)))
    if f < 1.0:
        crop = cv2.resize(crop, (max(1, int(round(crop.shape[1] * f))), max(1, int(round(crop.shape[0] * f)))),
                          interpolation=cv2.INTER_AREA)
    # thin lines get faint when shrunk: boost the alpha so they stay visible
    alpha = np.clip(crop.astype(np.float32) * 2.4, 0, 255).astype(np.uint8)
    rgba = np.zeros(crop.shape + (4,), np.uint8)
    rgba[..., 3] = alpha
    Image.fromarray(rgba, 'RGBA').save(ppath, optimize=False, compress_level=3)
    # polygon and label point in piece-picture pixels:  full px = analysis * rf ; picture px = (full - crop0) * f
    to_pic = lambda p: ((np.asarray(p, float) * rf) - np.array([cx0, cy0], float)) * f
    poly = to_pic(poly_a)
    tag = to_pic(sheet.get('tag', sheet['centroid']))
    g = sheet.get('g')                       # ground metres per analysis px from the printed scale (paper size assumed)
    g_puzzle = g
    # The printed area (Hect/Ares) is a better yardstick than the scale: it does not depend on the paper size
    # the sheet happened to be scanned/printed on.  Outline area in analysis px^2 -> metres per px.
    area_m2 = (sheet.get('meta') or {}).get('area_m2')
    fill_px = float((sheet['fill4'] > 127).sum()) * 16.0
    try:
        area_m2 = float(area_m2) if area_m2 else None
    except (TypeError, ValueError):
        area_m2 = None
    if area_m2 and area_m2 > 50 and fill_px > 0:
        g_area = math.sqrt(area_m2 / fill_px)
        if g is None or 0.2 <= g_area / g <= 5.0:     # ignore a nonsense OCR'd area
            g = g_area
    cen_a = np.array(sheet.get('centroid', sheet['origin']), float)
    data = {
        'sig': sig, 'id': item['id'], 'name': item.get('name'), 'survey': _survey_label(item, sheet),
        'w': int(crop.shape[1]), 'h': int(crop.shape[0]),
        'poly': [[round(float(x), 1), round(float(y), 1)] for x, y in poly],
        'tag': [round(float(tag[0]), 1), round(float(tag[1]), 1)],
        'tag_r': round(float(sheet.get('tag_r', 20.0)) * rf * f, 1),
        'g': g, 'g_puzzle': g_puzzle, 'scale_known': bool(g), 'area_m2': area_m2,
        'centroid_px': [round(float(v), 1) for v in to_pic(cen_a)],
        'centroid_a': [float(cen_a[0]), float(cen_a[1])],
        'mpp': (g / (rf * f)) if g else None,
        # for turning the automatic result into board positions
        'crop_origin_a': [cx0 / rf, cy0 / rf], 'origin_a': [float(sheet['origin'][0]), float(sheet['origin'][1])],
    }
    _write_json(jpath, data)
    return data


def _items_from_records(records, approved_dir):
    items = []
    for r in records:
        items.append({'id': r['id'], 'name': r.get('original_filename') or r['id'],
                      'path': os.path.join(approved_dir, r['image_file']), 'meta': r.get('meta') or {}})
    return items


def _prepare_worker(items, puzzle_dir):
    errors = {}
    for k, it in enumerate(items):
        with LOCK:
            STATE.update(done=k, message=f'Preparing piece {k + 1} of {len(items)}…')
        try:
            prepare_piece(it, puzzle_dir)
        except Exception as exc:                                   # one broken sheet must not stop the others
            errors[it['id']] = f'{type(exc).__name__}: {exc}'
    with LOCK:
        STATE.update(state='ready', done=len(items), errors=errors, message='')


# --------------------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------------------
def _layout_file(puzzle_dir):
    return os.path.join(puzzle_dir, 'board_layout.json')


def load_layout(puzzle_dir):
    data = _read_json(_layout_file(puzzle_dir))
    return data if isinstance(data, dict) else {}


def save_layout(puzzle_dir, layout):
    with LAYOUT_LOCK:
        _write_json(_layout_file(puzzle_dir), layout)


def _world_poly(pm, x, y, rot_deg):
    a = math.radians(rot_deg)
    c, s = math.cos(a), math.sin(a)
    p = np.array(pm['poly'], float) * pm['mpp']
    return np.stack([c * p[:, 0] - s * p[:, 1] + x, s * p[:, 0] + c * p[:, 1] + y], 1)


def _shelf_pack(units, start_y=0.0, gap=None):
    """units: [(key, (minx, miny, maxx, maxy))] -> {key: (dx, dy)} shifts that tile the units in rows."""
    if not units:
        return {}
    areas = sum((b[2] - b[0]) * (b[3] - b[1]) for _, b in units)
    widest = max(b[2] - b[0] for _, b in units)
    if gap is None:
        gap = 0.04 * math.sqrt(areas / max(1, len(units)))
    max_w = max(1.6 * math.sqrt(areas), widest * 1.2)
    out = {}
    x = y = row_h = 0.0
    for key, (mnx, mny, mxx, mxy) in sorted(units, key=lambda u: -(u[1][2] - u[1][0]) * (u[1][3] - u[1][1])):
        w, h = mxx - mnx, mxy - mny
        if x > 0 and x + w > max_w:
            x, y, row_h = 0.0, y + row_h + gap, 0.0
        out[key] = (x - mnx, start_y + y - mny)
        x += w + gap
        row_h = max(row_h, h)
    return out


def _fill_mpp(pieces):
    known = [p['mpp'] for p in pieces if p.get('mpp')]
    fallback = float(np.median(known)) if known else DEFAULT_MPP
    for p in pieces:
        if not p.get('mpp'):
            p['mpp'] = fallback
            p['scale_known'] = False
    return pieces


def default_positions(pieces, layout):
    """Positions for pieces without a saved place: tiled below everything that is already placed."""
    start_y, loose = 0.0, []
    for p in pieces:
        lay = layout.get(p['id'])
        if lay and 'x' in lay:
            ys = _world_poly(p, lay['x'], lay['y'], lay.get('rot', 0.0))[:, 1]
            start_y = max(start_y, float(ys.max()))
        else:
            loose.append(p)
    units = []
    for p in loose:
        w = _world_poly(p, 0.0, 0.0, 0.0)
        units.append((p['id'], (w[:, 0].min(), w[:, 1].min(), w[:, 0].max(), w[:, 1].max())))
    shifts = _shelf_pack(units, start_y + (0.05 * start_y if start_y else 0.0))
    return {k: {'x': float(v[0]), 'y': float(v[1]), 'rot': 0.0} for k, v in shifts.items()}


def auto_positions(pieces):
    """Turn the automatic puzzle (puzzle.LAST) into board positions.  Returns ({id: pose}, n_joined) or (None, 0)."""
    if not puzzle.LAST or not puzzle.LAST.get('villages'):
        return None, 0
    by_id = {p['id']: p for p in pieces}
    poses, units, joined = {}, [], 0
    for v in puzzle.LAST['villages']:
        sheets = v['sheets']
        for place in v['groups']:
            unit = {}
            for k, (R, t) in place.items():
                pm = by_id.get(sheets[k].get('id'))
                gp = (pm or {}).get('g_puzzle')
                if pm is None or not gp:
                    continue
                R = np.asarray(R)
                # where the automatic puzzle put the sheet's centre, then keep that centre on the board
                ctr = R @ ((np.array(pm['centroid_a']) - np.array(pm['origin_a'])) * gp) + np.asarray(t)
                rot = math.degrees(math.atan2(R[1, 0], R[0, 0]))
                a = math.radians(rot)
                cp = np.array(pm['centroid_px']) * pm['mpp']
                off = np.array([math.cos(a) * cp[0] - math.sin(a) * cp[1], math.sin(a) * cp[0] + math.cos(a) * cp[1]])
                unit[pm['id']] = {'x': float(ctr[0] - off[0]), 'y': float(ctr[1] - off[1]), 'rot': rot}
            if not unit:
                continue
            allp = np.vstack([_world_poly(by_id[i], q['x'], q['y'], q['rot']) for i, q in unit.items()])
            units.append((tuple(unit), (allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max()), unit))
            joined += len(unit) > 1
    shifts = _shelf_pack([(u[0], u[1]) for u in units])
    for key, _, unit in units:
        dx, dy = shifts[key]
        for i, q in unit.items():
            poses[i] = {'x': q['x'] + dx, 'y': q['y'] + dy, 'rot': q['rot']}
    # sheets the automatic step could not place go in rows below
    rest = [p for p in pieces if p['id'] not in poses]
    if rest:
        poses.update(default_positions(rest, poses))
    return poses, joined


# --------------------------------------------------------------------------------------
# Flask routes
# --------------------------------------------------------------------------------------
def register(app, approved_records, approved_dir, puzzle_dir):
    """approved_records: callable returning the approved-image records (as app.py's _approved_records)."""
    from flask import jsonify, request, send_file, abort

    here = os.path.dirname(os.path.abspath(__file__))

    def stale_items():
        items = _items_from_records(approved_records(), approved_dir)
        todo = []
        for it in items:
            jpath, ppath = _paths(puzzle_dir, it['id'])
            old = _read_json(jpath)
            try:
                fresh = bool(old) and old.get('sig') == _signature(it) and os.path.isfile(ppath)
            except OSError:
                fresh = False
            if not fresh:
                todo.append(it)
        return items, todo

    def load_pieces(items):
        out = []
        for it in items:
            pm = _read_json(_paths(puzzle_dir, it['id'])[0])
            if pm:
                out.append(pm)
        return _fill_mpp(out)

    def kick(todo):
        """Start preparing pieces that are new or changed (no-op when already running).  True while running."""
        with LOCK:
            if STATE['state'] != 'running' and todo:
                STATE.update(state='running', done=0, total=len(todo), message='Preparing pieces…', errors={})
                threading.Thread(target=_prepare_worker, args=(todo, puzzle_dir), daemon=True).start()
            return STATE['state'] == 'running'

    @app.route('/board/board.js')
    def board_js():
        return send_file(os.path.join(here, 'board.js'), mimetype='text/javascript', max_age=0)

    @app.route('/api/board', methods=['GET'])
    def board_get():
        items, todo = stale_items()
        running = kick(todo)
        with LOCK:
            snap = dict(STATE)
        if running:
            return jsonify(state='preparing', done=snap['done'], total=snap['total'], message=snap['message'])
        pieces = load_pieces(items)
        layout = load_layout(puzzle_dir)
        order = {r['id']: i for i, r in enumerate(sorted(approved_records(), key=lambda r: (r.get('approved_at', ''), r.get('id', ''))))}
        pieces.sort(key=lambda p: order.get(p['id'], 0))
        defaults = default_positions(pieces, layout)
        out = []
        for i, p in enumerate(pieces):
            lay = layout.get(p['id']) or {}
            pose = {'x': lay['x'], 'y': lay['y'], 'rot': lay.get('rot', 0.0)} if 'x' in lay else defaults.get(p['id'], {'x': 0, 'y': 0, 'rot': 0})
            out.append({'id': p['id'], 'name': p['name'], 'survey': p['survey'], 'w': p['w'], 'h': p['h'],
                        'mpp': p['mpp'], 'poly': p['poly'], 'tag': p['tag'], 'tag_r': p['tag_r'],
                        'scale_known': p['scale_known'], 'saved': 'x' in lay,
                        'sz_saved': lay.get('sz'),
                        'placed': bool(lay.get('placed', 'x' in lay)),
                        'locked': bool(lay.get('locked', False)),
                        'color': lay.get('color') or color_for(order.get(p['id'], i)),
                        'img': f"/board/piece/{p['id']}.png?v={p['sig'][1]}",
                        'clean': f"/approved/{p['id']}/preview", 'download': f"/approved/{p['id']}/download",
                        **pose})
        return jsonify(state='ready', pieces=out, errors=snap['errors'],
                       has_auto=bool(puzzle.LAST and puzzle.LAST.get('villages')))

    @app.route('/board/piece/<item_id>.png')
    def board_piece(item_id):
        if not item_id.isalnum():
            abort(404)
        path = _paths(puzzle_dir, item_id)[1]
        if not os.path.isfile(path):
            abort(404)
        return send_file(path, mimetype='image/png', max_age=3600)

    @app.route('/api/board/layout', methods=['POST'])
    def board_layout_save():
        payload = request.get_json(silent=True) or {}
        pieces = payload.get('pieces')
        if not isinstance(pieces, dict):
            return jsonify(error='Nothing to save.'), 400
        layout = load_layout(puzzle_dir)
        for pid, v in pieces.items():
            try:
                entry = layout.setdefault(str(pid), {})
                entry['x'], entry['y'] = float(v['x']), float(v['y'])
                entry['rot'] = float(v.get('rot', 0.0))
                if v.get('sz') is not None:
                    entry['sz'] = float(v['sz'])
                if 'placed' in v:
                    entry['placed'] = bool(v['placed'])
                if 'locked' in v:
                    entry['locked'] = bool(v['locked'])
                col = str(v.get('color') or '')
                if len(col) == 7 and col.startswith('#'):
                    int(col[1:], 16)
                    entry['color'] = col
            except (KeyError, TypeError, ValueError):
                return jsonify(error='Bad layout data.'), 400
        save_layout(puzzle_dir, layout)
        return jsonify(ok=True)

    # ---------------------------------------------------------------- background reference image
    bg_png = os.path.join(puzzle_dir, 'board_bg.png')
    bg_json = os.path.join(puzzle_dir, 'board_bg.json')
    BG_MAX = 5000                                   # longest side kept, px

    def bg_meta():
        m = _read_json(bg_json)
        if not m or not os.path.isfile(bg_png):
            return {'exists': False}
        m['exists'] = True
        m['url'] = '/board/background.png?v=%s' % m.get('v', 0)
        return m

    def to_png(fileobj):
        """Any image or TIFF -> 8-bit RGB PNG, longest side <= BG_MAX."""
        im = Image.open(fileobj)
        im.seek(0)
        if im.mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F'):
            a = np.array(im).astype(np.float64)
            lo, hi = float(a.min()), float(a.max())
            im = Image.fromarray(((a - lo) * (255.0 / max(1e-9, hi - lo))).astype(np.uint8), 'L')
        if im.mode == 'P' and 'transparency' in im.info:
            im = im.convert('RGBA')
        if im.mode in ('RGBA', 'LA'):
            flat = Image.new('RGB', im.size, (255, 255, 255))
            flat.paste(im.convert('RGBA'), mask=im.convert('RGBA').split()[-1])
            im = flat
        elif im.mode == '1':
            im = im.convert('L')
        im.thumbnail((BG_MAX, BG_MAX), Image.LANCZOS, reducing_gap=3.0)
        return im.convert('RGB')

    @app.route('/api/board/background', methods=['GET'])
    def board_bg_get():
        return jsonify(bg_meta())

    @app.route('/api/board/background', methods=['POST'])
    def board_bg_upload():
        f = request.files.get('file')
        if f is None:
            return jsonify(error='No file received.'), 400
        try:
            im = to_png(f.stream)
        except Exception as exc:
            return jsonify(error='Could not read that image: %s' % exc), 400
        tmp = bg_png + '.tmp.png'
        im.save(tmp, format='PNG', optimize=False)
        os.replace(tmp, bg_png)
        meta = {'v': int(time.time() * 1000), 'w': im.size[0], 'h': im.size[1],
                'x': 0.0, 'y': 0.0, 'mpp': 0.0, 'rot': 0.0, 'opacity': 0.6}   # mpp 0 -> the browser fits it to the view
        _write_json(bg_json, meta)
        return jsonify(bg_meta())

    @app.route('/api/board/background/pose', methods=['POST'])
    def board_bg_pose():
        m = _read_json(bg_json)
        if not m:
            return jsonify(error='No background image.'), 400
        p = request.get_json(silent=True) or {}
        try:
            for k in ('x', 'y', 'mpp', 'rot', 'opacity'):
                if k in p:
                    m[k] = float(p[k])
        except (TypeError, ValueError):
            return jsonify(error='Bad background data.'), 400
        _write_json(bg_json, m)
        return jsonify(ok=True)

    @app.route('/api/board/background', methods=['DELETE'])
    def board_bg_delete():
        for path in (bg_png, bg_json):
            try:
                os.remove(path)
            except OSError:
                pass
        return jsonify(ok=True)

    @app.route('/board/background.png')
    def board_bg_img():
        if not os.path.isfile(bg_png):
            abort(404)
        return send_file(bg_png, mimetype='image/png', max_age=0)

    # ---- cleaned sheet with plot numbers shown as S<survey>-<plot> (viewer + download inside the puzzle board) ----
    LABEL_VERSION = 1
    label_lock = threading.Lock()

    def _item_for(item_id):
        for it in _items_from_records(approved_records(), approved_dir):
            if it['id'] == item_id:
                return it
        abort(404)

    def _outline_full(item):
        """Outline of the sheet in full-resolution pixels (same maths as prepare_piece)."""
        sheet = puzzle.load_sheet(item, puzzle_dir)
        if not sheet.get('ok'):
            raise RuntimeError(sheet.get('error') or 'no drawing found in this sheet')
        fill4 = (sheet['fill4'] > 127).astype(np.uint8)
        cnts, _ = cv2.findContours(fill4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        ap = cv2.approxPolyDP(c, 1.1, True)[:, 0, :].astype(np.float64)
        x0, y0, _, _ = sheet['zone']
        full = puzzle._read_gray(item['path'])
        rf = max(full.shape) / float(puzzle.AN_LONG)
        return ((ap * 4.0 + 1.5 + np.array([x0, y0], float)) * rf).astype(np.float32)

    def labelled_paths(item):
        import hashlib
        pm = _read_json(_paths(puzzle_dir, item['id'])[0]) or {}
        survey = str(pm.get('survey') or _survey_label(item, {}))
        key = hashlib.md5(f"{LABEL_VERSION}|{pm.get('sig')}|{survey}|{os.path.getmtime(item['path'])}".encode()).hexdigest()[:10]
        d = os.path.join(puzzle_dir, 'board')
        return survey, os.path.join(d, f"{item['id']}_lab_{key}.png"), os.path.join(d, f"{item['id']}_lab_{key}.tif")

    def build_labelled(item):
        survey, png_path, tif_path = labelled_paths(item)
        if os.path.isfile(png_path) and os.path.isfile(tif_path):
            return png_path, tif_path, survey
        with label_lock:
            if os.path.isfile(png_path) and os.path.isfile(tif_path):
                return png_path, tif_path, survey
            src = Image.open(item['path'])
            binary = src.mode == '1'
            dpi = src.info.get('dpi')
            gray = np.array(src.convert('L'))
            poly = _outline_full(item)
            inside = lambda cx, cy: cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(cx), float(cy)), False) >= 0
            res, count = relabel_plots(gray, inside, survey, log=lambda m: print('[board]' + m))
            out = Image.fromarray(res, 'L')
            opts = {}
            if dpi and min(dpi) > 1:
                opts['dpi'] = dpi
            if binary:
                out.point(lambda v: 255 if v > 140 else 0).convert('1').save(tif_path, format='TIFF', compression='group4', **opts)
            else:
                out.save(tif_path, format='TIFF', compression='tiff_lzw', **opts)
            prev = out.copy()
            prev.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE), Image.Resampling.LANCZOS)
            prev.save(png_path, format='PNG', optimize=True)
            return png_path, tif_path, survey

    @app.route('/board/labelled/<item_id>/preview')
    def board_labelled_preview(item_id):
        try:
            png, _, _ = build_labelled(_item_for(item_id))
        except Exception as e:
            return jsonify(error=f'Could not label the plots: {e}'), 500
        return send_file(png, mimetype='image/png', max_age=0)

    @app.route('/board/labelled/<item_id>/download')
    def board_labelled_download(item_id):
        item = _item_for(item_id)
        try:
            _, tif, survey = build_labelled(item)
        except Exception as e:
            return jsonify(error=f'Could not label the plots: {e}'), 500
        base = os.path.splitext(item['name'])[0] or 'survey'
        return send_file(tif, mimetype='image/tiff', as_attachment=True, download_name=f'{base}_S{survey}_plots.tif', max_age=0)

    @app.route('/api/board/convert', methods=['POST'])
    def board_convert():
        """Turn the PNG that the browser drew (puzzle only) into a PDF or a TIF for download."""
        import io
        fmt = (request.args.get('fmt') or '').lower()
        if fmt not in ('pdf', 'tif', 'tiff'):
            return jsonify(error='Unknown format.'), 400
        try:
            im = Image.open(io.BytesIO(request.get_data())).convert('RGB')
            out = io.BytesIO()
            if fmt == 'pdf':
                im.save(out, 'PDF', resolution=200.0)
                mime = 'application/pdf'
            else:
                im.save(out, 'TIFF', compression='tiff_lzw', dpi=(200, 200))
                mime = 'image/tiff'
            out.seek(0)
            return send_file(out, mimetype=mime)
        except Exception as e:
            return jsonify(error=f'Could not convert: {e}'), 500

    # ---------------------------------------------------------------- import sheet files straight onto the board
    @app.route('/api/board/import', methods=['POST'])
    def board_import():
        """Add already-cleaned sheet images (TIF/PNG/JPG...) as approved sheets, so they become board pieces
        without going through Upload -> Clean -> Review."""
        import re
        import uuid
        from datetime import datetime
        files = request.files.getlist('files')
        if not files:
            return jsonify(error='Choose one or more image files.'), 400
        survey_in = (request.form.get('survey') or '').strip()[:20]
        scale_in = None
        if (request.form.get('scale') or '').strip():
            try:
                scale_in = int(request.form['scale'])
                if not 50 <= scale_in <= 200000:
                    raise ValueError
            except ValueError:
                return jsonify(error='Scale must be a whole number such as 936.'), 400
        added, errors = [], {}
        for f in files:
            name = os.path.basename(f.filename or 'sheet')
            try:
                im = Image.open(f.stream)
                dpi = im.info.get('dpi')
                im.load()
                if im.mode in ('RGBA', 'LA', 'P') and ('A' in im.getbands() or 'transparency' in im.info):
                    im = im.convert('RGBA')
                    flat = Image.new('RGBA', im.size, (255, 255, 255, 255))
                    flat.alpha_composite(im)
                    im = flat
                gray = im.convert('L')
                arr = np.asarray(gray)
                if float(arr.mean()) < 110:                       # white lines on black -> black ink on white
                    gray = Image.fromarray(255 - arr)
                    arr = np.asarray(gray)
                w, h = gray.size
                stem = os.path.splitext(name)[0] or 'survey'
                safe = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in stem)[:100]
                rid = uuid.uuid4().hex
                img_file, prev_file = f'{safe}_approved_{rid[:8]}.tif', f'{safe}_approved_{rid[:8]}.png'
                opts = {'compression': 'tiff_lzw'}
                save_img = gray
                if len(np.unique(arr[::7, ::7])) <= 2 and set(np.unique(arr).tolist()) <= {0, 255}:
                    opts = {'compression': 'group4'}
                    save_img = gray.point(lambda v: 255 if v > 140 else 0).convert('1')
                if dpi and min(dpi) > 1:
                    opts['dpi'] = (float(dpi[0]), float(dpi[1]))
                save_img.save(os.path.join(approved_dir, img_file), format='TIFF', **opts)
                prev = gray.copy()
                prev.thumbnail((1800, 1800), Image.LANCZOS)
                prev.save(os.path.join(approved_dir, prev_file), format='PNG')
                meta = {}
                if len(files) == 1 and survey_in:
                    meta['survey'] = survey_in
                else:
                    m = re.match(r'^(\d+[A-Za-z]?)(?:[ _\-.(].*)?$', stem)
                    if m:
                        meta['survey'] = m.group(1)
                if scale_in:
                    meta['scale'] = scale_in
                    meta['scale_source'] = 'manual'
                record = {'id': rid, 'original_filename': name, 'image_file': img_file, 'preview_file': prev_file,
                          'width': int(w), 'height': int(h), 'approved_at': datetime.now().astimezone().isoformat(),
                          'status': 'approved', 'puzzle_status': 'pending_implementation', 'operations': [],
                          'imported': True, 'meta': meta}
                _write_json(os.path.join(approved_dir, f'{rid}.json'), record)
                added.append({'id': rid, 'name': name, 'survey': meta.get('survey')})
            except Exception as exc:                                   # one bad file must not stop the others
                errors[name] = f'{type(exc).__name__}: {exc}'
        if not added:
            return jsonify(error='No file could be imported. ' + '; '.join(f'{k}: {v}' for k, v in errors.items())), 400
        return jsonify(ok=True, added=added, errors=errors)

    @app.route('/api/board/reset', methods=['POST'])
    def board_reset():
        save_layout(puzzle_dir, {})
        return jsonify(ok=True)

    @app.route('/api/board/auto', methods=['POST'])
    def board_auto():
        items, todo = stale_items()
        if todo:
            kick(todo)                                   # a newly approved sheet: prepare it now
            return jsonify(error='preparing', preparing=True), 409
        pieces = load_pieces(items)
        poses, joined = auto_positions(pieces)
        if poses is None:
            return jsonify(error='No automatic match yet. Press Build puzzle in the "Automatic match" box below first, then try Auto-arrange again.'), 400
        return jsonify(ok=True, poses=poses, joined=joined)