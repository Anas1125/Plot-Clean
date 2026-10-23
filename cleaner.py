"""Plot-map cleaner v3 (colour-aware).
Sheets use: black = boundaries/dashes/arrows/icons, blue = text (dimensions + plot numbers),
red = point labels.  Linework is protected; plot numbers are picked out of the blue text by size."""
import cv2, numpy as np
from PIL import Image
from classmap import class_map, LINE, BLUE, RED
from netclean import clean_network
Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(1)

def _lum(im):
    if im.mode == 'P':
        pal = np.array((im.getpalette() or [])[:768] + [0]*768, np.uint8)[:768].reshape(-1, 3)[:256].astype(np.int32)
        lut = ((pal[:,0]*299 + pal[:,1]*587 + pal[:,2]*114)//1000).astype(np.uint8)
        return lut[np.array(im)]
    return np.array(im.convert('L'))

def _find_zone(lab, st, frames, H, W, s, log):
    """work zone = below the header rule, above the footer texts, inside the frame"""
    inner = min(frames, key=lambda i: st[i, 2]*st[i, 3])
    x, y, w, h, _ = st[inner]
    crop = (lab[y:y+h, x:x+w] == inner)
    rows = crop.sum(1) >= 0.9 * w
    ys = np.nonzero(rows)[0]
    groups = []
    for r in ys:
        if groups and r - groups[-1][1] <= 3: groups[-1][1] = r
        else: groups.append([r, r])
    top = None
    for g0, g1 in groups:
        if g0 > 150*s and g0 < 0.2*H: top = y + g1 + int(12*s); break
    if top is None:
        top = int(y + 0.105*min(H, W)) ; log('  header rule not found, using fallback')
    bot = y + h - int(700*s)
    return top, bot, x, x + w

def clean(im, log=print, debug=False):
    if isinstance(im, str): im = Image.open(im)
    cm = class_map(im)
    H, W = cm.shape
    s = min(H, W) / 9917.0
    S = lambda v: max(1, int(round(v*s)))
    log(f'{W}x{H}, scale {s:.2f}')

    lines = (cm == LINE).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(lines, connectivity=8, ltype=cv2.CV_16U)
    del lines
    frames = [i for i in range(1, n) if st[i, 2] >= 0.85*W and st[i, 3] >= 0.85*H]
    if not frames: frames = [int(np.argmax(st[1:, 4])) + 1]
    top, bot, zx0, zx1 = _find_zone(lab, st, frames, H, W, s, log)
    log(f'  work zone y {top}..{bot}')

    keep = np.zeros(n, bool)
    network = []
    small_free = []                                   # (x,y,w,h,area) for dash evidence
    for i in range(1, n):
        x, y, w, h, a = st[i]
        cy = y + h/2
        if i in frames or cy < top or cy > bot or x + w/2 < zx0 or x + w/2 > zx1:
            keep[i] = True; continue                  # header, footer, frame: untouched
        if max(w, h) > S(1000):
            keep[i] = True; network.append(i); continue
        small_free.append((i, x, y, w, h, a))
    # --- dash / dot fragments: evidence that a stub belongs to a dashed line ---
    dp = []
    for (i, x, y, w, h, a) in small_free:
        diag = float(np.hypot(w, h))
        if diag >= S(100) and a/diag <= S(16):
            dp.append((x + w/2, y + h/2, diag/2))
        elif w <= S(26) and h <= S(26) and a >= S(20):
            dp.append((x + w/2, y + h/2, 0.0))
    dpts = np.array(dp, np.float64) if dp else np.zeros((0, 3))

    # --- isolated closed polygons (parcels not touching the network) are boundaries ---
    kept_poly = 0
    for (i, x, y, w, h, a) in small_free:
        if min(w, h) < S(260) or max(w, h) > S(1000): continue
        sub = (lab[y:y+h, x:x+w] == i).astype(np.uint8)
        cnts, hier = cv2.findContours(sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = [k for k in range(len(cnts)) if hier[0][k][3] >= 0]
        if len(holes) == 1 and cv2.contourArea(cnts[holes[0]]) > 0.35*w*h:   # one big hole = polygon, not an icon
            keep[i] = True; kept_poly += 1
    log(f'  isolated polygons kept: {kept_poly}')

    m = keep[lab]                                      # kept line pixels (bool, full size)
    chains = _dash_chains(cm, lab, small_free, top, bot, s, log)
    # --- network cleaning ---
    for i in network:
        x, y, w, h, a = st[i]
        pad = 10
        x0, y0 = max(x-pad, 0), max(y-pad, 0); x1, y1 = min(x+w+pad, W), min(y+h+pad, H)
        reg = (lab[y0:y1, x0:x1] == i)
        log(f'network component {i}: {w}x{h}')
        loc = [(c - np.array([x0, y0]), v, t0, t1) for c, v, t0, t1 in chains]   # chains in crop coordinates
        cleaned = clean_network(reg, s, log, dash_pts=dpts - np.array([x0, y0, 0.0]), chains=loc)
        sub = m[y0:y1, x0:x1]
        sub[cv2.dilate(reg.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = False
        sub |= cleaned
    del lab

    # --- heal line pixels that were covered by labels drawn on top of the linework ---
    healed = _heal_under_labels(m, cm, s, log)
    m |= healed
    # --- stray dimension labels in the footer band: keep only real footer text lines / QR ---
    foot = _footer_keep(cm, bot, H, W, zx1, s)
    # --- plot numbers from the blue text ---
    kept_blue = _plot_numbers(cm, s, log)
    m |= cv2.dilate(kept_blue.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    lum = _lum(im)
    res = np.full_like(lum, 255)
    keep_px = m & (cm > 0)
    keep_px |= m & (lum < 250)
    res[keep_px] = lum[keep_px]
    res[healed] = np.minimum(res[healed], 20)
    # outside the work zone everything stays exactly as it was
    res[:top] = lum[:top]
    res[bot:] = np.where(foot, lum[bot:], 255)
    if debug:
        removed = ((cm > 0) & ~m)
        removed[:top] = False; removed[bot:] = False
        f = 4
        sg = cv2.resize(res, None, fx=1/f, fy=1/f, interpolation=cv2.INTER_AREA)
        rem = cv2.resize(removed.astype(np.uint8)*255, None, fx=1/f, fy=1/f, interpolation=cv2.INTER_AREA) > 20
        dbg = np.stack([sg]*3, -1); dbg[rem] = (255, 0, 0)
        return res, dbg
    return res


def _dash_chains(cm, lab, small_free, top, bot, s, log):
    """Find the long dash-dot construction lines (A-B, B-C, C-D, A-D, B-D ...).
    Their fragments are free dashes / dots, black or red depending on the sheet.
    Returns [(centre(x,y), unit_dir, t0, t1)] in full-image coordinates. Pieces glued to the
    network that lie on one of these lines are leftovers of the same line (see netclean)."""
    S = lambda v: max(1, int(round(v*s)))
    H, W = cm.shape
    # --- black fragments (free components of the line class) ---
    P, half, kind = [], [], []
    black_ids = []
    for (i, x, y, w, h, a) in small_free:
        if y + h/2 < top or y + h/2 > bot: continue
        diag = float(np.hypot(w, h))
        if diag >= S(100) and a/diag <= S(16): k = 2
        elif w <= S(26) and h <= S(26) and a >= S(20): k = 1
        else: continue
        black_ids.append(i); P.append((x + w/2, y + h/2)); half.append(diag/2 if k == 2 else 0.0); kind.append(k)
    # --- red fragments (some sheets draw the construction lines in red) ---
    red = (cm[top:bot] == RED).astype(np.uint8)
    n, rlab, rst, rcen = cv2.connectedComponentsWithStats(red, connectivity=8, ltype=cv2.CV_16U)
    rids = []
    for i in range(1, n):
        x, y, w, h, a = rst[i]
        diag = float(np.hypot(w, h))
        if diag >= S(80) and a/diag <= S(16): k = 2
        elif w <= S(26) and h <= S(26) and a >= S(12): k = 1
        else: continue
        rids.append(i); P.append((rcen[i][0], rcen[i][1] + top)); half.append(diag/2 if k == 2 else 0.0); kind.append(k)
    if len(P) < 6:
        return []
    f = max(1, int(round(4*s)))
    lutb = np.zeros(lab.max() + 1 if lab.size else 1, np.uint8); lutb[black_ids] = 255
    small = np.zeros((H//f + 1, W//f + 1), np.uint8)
    # rasterise at 1/f without allocating a second full-size image: block-wise
    B = 512*f
    for y0 in range(0, H, B):
        blk = lutb[lab[y0:y0+B]]
        sm = cv2.resize(blk, (W//f, max(1, blk.shape[0]//f)), interpolation=cv2.INTER_AREA)
        small[y0//f:y0//f + sm.shape[0], :sm.shape[1]] = sm
    if rids:
        lutr = np.zeros(n, np.uint8); lutr[rids] = 255
        rs = cv2.resize(lutr[rlab], (W//f, max(1, rlab.shape[0]//f)), interpolation=cv2.INTER_AREA)
        yy = top//f
        small[yy:yy + rs.shape[0], :rs.shape[1]] = np.maximum(small[yy:yy + rs.shape[0], :rs.shape[1]], rs)
    del rlab
    small = cv2.dilate((small > 20).astype(np.uint8)*255, np.ones((3, 3), np.uint8))
    segs = cv2.HoughLinesP(small, 1, np.pi/720, threshold=max(15, int(250*s/f)),
                           minLineLength=max(10, int(700*s/f)), maxLineGap=max(5, int(450*s/f)))
    if segs is None: return []
    P = np.array(P, np.float64); half = np.array(half); kind = np.array(kind)
    chains = []
    for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4) * float(f):
        c = np.array([(x1+x2)/2, (y1+y2)/2]); v = np.array([x2-x1, y2-y1]); L = np.hypot(*v)
        if L < 1: continue
        v /= L
        d = P - c; along = d @ v; perp = np.abs(d[:, 0]*v[1] - d[:, 1]*v[0])
        mem = (perp <= 10*s) & (along > -L/2 - 150*s) & (along < L/2 + 150*s)
        if mem.sum() < 6 or (kind[mem] == 2).sum() < 1: continue
        Q = P[mem]; c2 = Q.mean(0)
        v2 = np.linalg.svd(Q - c2, full_matrices=False)[2][0]
        if v2 @ v < 0: v2 = -v2
        t = (Q - c2) @ v2
        t0, t1 = float((t - half[mem]).min()), float((t + half[mem]).max())
        if t1 - t0 < 600*s: continue
        dup = False
        for c0, v0, a0, a1 in chains:
            if abs(v0 @ v2) > 0.9995 and abs((c2 - c0)[0]*v0[1] - (c2 - c0)[1]*v0[0]) < 12*s: dup = True; break
        if not dup: chains.append((c2, v2, t0, t1))
    log(f'  dash-dot construction lines found: {len(chains)}')
    return chains

def _footer_keep(cm, bot, H, W, zx1, s):
    """footer band: keep long text lines, the QR code and the page tag; drop short stray labels"""
    band = (cm[bot:] > 0).astype(np.uint8)
    keep = np.zeros(band.shape, bool)
    d = cv2.dilate(band, cv2.getStructuringElement(cv2.MORPH_RECT, (int(90*s), int(25*s))))
    n, lab, st, _ = cv2.connectedComponentsWithStats(d, connectivity=8)
    ok = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w >= 380*s or h >= 220*s or x + w > zx1 - 1600*s or y + h >= band.shape[0] - 40*s: ok[i] = True
    keep = ok[lab] & (band > 0)
    return cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

def _heal_under_labels(m, cm, s, log):
    """Red point labels / blue text are drawn over the black lines and replace the ink underneath.
    Fill only those label pixels that lie between line ink continuing on both sides."""
    H, W = m.shape
    txt = ((cm == RED) | (cm == BLUE)).astype(np.uint8)
    near = (cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & (txt > 0)
    out = np.zeros_like(m)
    if not near.any(): return out
    n, lab, st, _ = cv2.connectedComponentsWithStats(txt, connectivity=8, ltype=cv2.CV_16U)
    ids = np.unique(lab[near]); ids = ids[ids > 0]
    L = int(70*s) | 1
    angles = np.arange(0, 180, 10)
    kernels = []
    for a in angles:
        k = np.zeros((L, L), np.uint8)
        c = L // 2; dx = np.cos(np.deg2rad(a))*c; dy = np.sin(np.deg2rad(a))*c
        cv2.line(k, (int(round(c-dx)), int(round(c-dy))), (int(round(c+dx)), int(round(c+dy))), 1, 1)
        kernels.append(k)
    pad = L
    cnt = 0
    for i in ids:
        x, y, w, h, a = st[i]
        if max(w, h) > 120*s: continue                      # only glyph-sized things (labels), not long lines
        x0, y0, x1, y1 = max(x-pad, 0), max(y-pad, 0), min(x+w+pad, W), min(y+h+pad, H)
        sub = m[y0:y1, x0:x1].astype(np.uint8)
        lm = cv2.dilate((lab[y0:y1, x0:x1] == i).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        acc = np.zeros(sub.shape, bool)
        for k in kernels:
            acc |= (cv2.morphologyEx(sub, cv2.MORPH_CLOSE, k) > 0) & lm & (sub == 0)
        if acc.any():
            out[y0:y1, x0:x1] |= acc; cnt += 1
    log(f'  healed {cnt} labels sitting on lines')
    return out

def _plot_numbers(cm, s, log):
    S = lambda v: max(1, int(round(v*s)))
    blue = (cm == BLUE).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(blue, connectivity=8, ltype=cv2.CV_16U)
    del blue
    x, y, w, h, a = (st[1:, k] for k in range(5))
    sz = np.maximum(w, h)
    glyph = (a >= S(30)**2 // 6) & (sz >= S(30))
    dots = (sz <= S(20)) & (a >= S(15)) & ~glyph
    gs = np.nonzero(glyph)[0]
    # typical dimension-digit size (most common size in the small cluster)
    vals = sz[gs]; vals = vals[(vals >= S(40)) & (vals <= S(58))]
    H1 = float(np.bincount(vals).argmax()) if len(vals) else 50.0*s
    thr = H1 * 1.10
    log(f'  text sizes: dimension ~{H1:.0f}px, plot-number threshold {thr:.0f}px')
    # link glyphs of similar size that sit next to each other into strings
    cx = x[gs] + w[gs]/2; cy = y[gs] + h[gs]/2
    par = np.arange(len(gs))
    def find(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for ii in range(len(gs)):
        gx = np.maximum(0, np.maximum(x[gs] - (x[gs[ii]] + w[gs[ii]]), x[gs[ii]] - (x[gs] + w[gs])))
        gy = np.maximum(0, np.maximum(y[gs] - (y[gs[ii]] + h[gs[ii]]), y[gs[ii]] - (y[gs] + h[gs])))
        gap = np.hypot(gx, gy)
        same = np.abs(sz[gs] - sz[gs[ii]]) <= 0.09 * np.maximum(sz[gs], sz[gs[ii]])
        for jj in np.nonzero((gap <= 0.45*H1) & same)[0]:
            if jj != ii: par[find(jj)] = find(ii)
    root = np.array([find(i) for i in range(len(gs))])
    # dots belong to every string they touch (both neighbours of the "." in 52.0)
    hasdot = set()
    for d in np.nonzero(dots)[0]:
        dx, dy = x[d] + w[d]/2, y[d] + h[d]/2
        gx = np.maximum(0, np.maximum(x[gs] - dx, dx - (x[gs] + w[gs])))
        gy = np.maximum(0, np.maximum(y[gs] - dy, dy - (y[gs] + h[gs])))
        dist = np.hypot(gx, gy)
        for j in np.nonzero(dist <= 0.5*H1)[0]: hasdot.add(root[j])
    # parentheses: a paren-like glyph with neighbours on ONE side belongs to a dimension "(217.6)";
    # with neighbours on both sides it is the "/" of a plot name such as 3E/1
    def _parenlike(k):
        return max(w[k], h[k]) / max(1, min(w[k], h[k])) > 2.4 and a[k] / (w[k]*h[k]) < 0.5
    slashes = []
    for ii, k in enumerate(gs):
        if not _parenlike(k): continue
        near = np.nonzero((np.hypot(np.maximum(0, np.maximum(x[gs] - (x[k] + w[k]), x[k] - (x[gs] + w[gs]))),
                                   np.maximum(0, np.maximum(y[gs] - (y[k] + h[k]), y[k] - (y[gs] + h[gs])))) <= 0.6*H1)
                          & (np.arange(len(gs)) != ii))[0]
        left = [j for j in near if cx[j] < cx[ii]]; right = [j for j in near if cx[j] > cx[ii]]
        if left and right: slashes.append((ii, left + right))
        else:
            for j in left + right: hasdot.add(root[j])
    keep_ids = []
    for r in set(root.tolist()):
        mem = gs[root == r]
        if r in hasdot: continue
        if sz[mem].min() < thr or sz[mem].max() > 1.7*H1: continue
        ar = np.maximum(w[mem], h[mem]) / np.maximum(1, np.minimum(w[mem], h[mem]))
        fill = a[mem] / (w[mem]*h[mem])
        if (ar > 3.2).any() or (fill < 0.12).any(): continue         # thin dashes
        if len(mem) > 8: continue
        # dash / line segments: a straight thin stroke is far more elongated than any glyph
        bad = False
        for k in mem:
            x0_, y0_, w_, h_ = int(x[k]), int(y[k]), int(w[k]), int(h[k])
            sub = (lab[y0_:y0_+h_, x0_:x0_+w_] == k + 1).astype(np.uint8)
            pts = cv2.findNonZero(sub)
            (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
            if max(rw, rh) / max(1.0, min(rw, rh)) > 4.2: bad = True; break
        if bad: continue
        if w[mem].max() < 0.45*sz[mem].max(): continue       # slender brackets / ticks, not glyphs
        # plot numbers are upright & horizontal; rotated/slanted strings are dimensions
        C0 = np.stack([x[mem] + w[mem]/2, y[mem] + h[mem]/2], 1).astype(np.float64)
        if len(mem) > 1:
            ax = np.linalg.svd(C0 - C0.mean(0), full_matrices=False)[2][0]
            if abs(np.degrees(np.arctan2(ax[1], ax[0]))) % 180 > 25 and abs(np.degrees(np.arctan2(ax[1], ax[0]))) % 180 < 155: continue
        elif w[mem[0]] > 1.15*h[mem[0]]: continue
        if (w[mem] > 0.9*h[mem]).any() and (fill > 0.25).any() and (w[mem] > 0.9*h[mem]).sum() >= 1 and ((w[mem] > 0.9*h[mem]) & (h[mem] >= thr)).any(): continue   # square symbol (hatched building), not a digit
        # strings that start/end with a parenthesis are dimensions such as (186.2)
        if len(mem) >= 1:
            cxm = x[mem-0]*0 + (x[mem] + w[mem]/2); cym = y[mem] + h[mem]/2
            if len(mem) > 1:
                C = np.stack([cxm, cym], 1).astype(np.float64)
                axis = np.linalg.svd(C - C.mean(0), full_matrices=False)[2][0]
                order = np.argsort((C - C.mean(0)) @ axis)
                ends_ = [mem[order[0]], mem[order[-1]]]
            else:
                ends_ = [mem[0]]
            def parenlike(k):
                arr = max(w[k], h[k]) / max(1, min(w[k], h[k]))
                return arr > 2.4 and (a[k] / (w[k]*h[k])) < 0.5
            if any(parenlike(k) for k in ends_): continue
        keep_ids.extend((mem + 1).tolist())
    kept_roots = {root[np.nonzero(gs == (k - 1))[0][0]] for k in keep_ids if (k - 1) in set(gs.tolist())}
    for ii, nb in slashes:
        if any(root[j] in kept_roots for j in nb): keep_ids.append(int(gs[ii]) + 1)
    log(f'  plot-number glyphs kept: {len(keep_ids)}')
    if not keep_ids: return np.zeros(cm.shape, bool)
    lut = np.zeros(n, bool); lut[keep_ids] = True
    return lut[lab]

if __name__ == '__main__':
    import sys
    r = clean(sys.argv[1])
    Image.fromarray(r).save(sys.argv[2])