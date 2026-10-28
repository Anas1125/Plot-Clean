"""Clean the big connected line network: prune dash stubs, remove text glued to lines."""
import cv2, numpy as np
from skimage.morphology import skeletonize

K8 = np.ones((3,3), np.float32)

def _deg(sk):
    return (cv2.filter2D(sk.astype(np.float32), -1, K8, borderType=cv2.BORDER_CONSTANT) - 1) * sk


_OFFS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
def _shift(a, dy, dx):
    out = np.zeros_like(a)
    H, W = a.shape
    ys0, ys1 = max(0, -dy), min(H, H - dy); xs0, xs1 = max(0, -dx), min(W, W - dx)
    out[ys0:ys1, xs0:xs1] = a[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
    return out

def _endmask(sk, deg=None):
    """skeleton end pixels. Besides deg==1, a diagonal stroke often ends in a corner pixel with
    two neighbours that touch each other - that is an end too (otherwise a free-ended stub
    looks like an edge between two junctions and is never pruned)."""
    if deg is None: deg = _deg(sk)
    e = sk & (deg == 1)
    c = sk & (deg == 2)
    if c.any():
        nb = {o: _shift(sk, *o) for o in _OFFS}
        pairs = np.zeros(sk.shape, np.uint8)
        for o in _OFFS:
            for o2 in _OFFS:
                if o < o2 and max(abs(o[0]-o2[0]), abs(o[1]-o2[1])) == 1:
                    pairs += (nb[o] & nb[o2]).astype(np.uint8)
        e |= c & (pairs >= 1)
    return e

def _edges(sk):
    deg = _deg(sk)
    J = sk & (deg >= 3)
    e = (sk & ~J).astype(np.uint8)
    n, lab = cv2.connectedComponents(e, connectivity=8)
    return deg, J, n, lab

def _orient(dash_pts, v, hl):
    """A dash only counts for a piece running the same way (within 25 deg). Its reach along the piece's
    axis is its half length projected on that axis (a dash lying ACROSS the piece reaches ~0)."""
    if dash_pts.shape[1] < 5: return hl, np.ones(len(hl), bool)
    c = np.abs(dash_pts[:, 3]*v[0] + dash_pts[:, 4]*v[1])
    isdash = hl > 0
    ok = ~isdash | (c >= np.cos(np.deg2rad(25)))
    return np.where(isdash, hl*c, 0.0), ok


def _dash_evidence(ys, xs, dash_pts, s):
    """True if free dash/dot fragments lie along the axis of this edge (either side of it)."""
    if dash_pts is None or len(dash_pts) == 0: return False
    P = np.stack([xs, ys], 1).astype(np.float64)
    c = P.mean(0)
    if len(P) > 2:
        u, sv, vt = np.linalg.svd(P - c, full_matrices=False); v = vt[0]
    else:
        v = np.array([1.0, 0.0])
    t = (P - c) @ v
    t0, t1 = t.min(), t.max()
    d = dash_pts[:, :2] - c
    hl = dash_pts[:, 2] if dash_pts.shape[1] > 2 else np.zeros(len(d))
    along = d @ v
    perp = np.abs(d[:, 0]*v[1] - d[:, 1]*v[0])
    # fragments of a real dash-dot line sit ON its axis (a few px); 14 px off is somebody else's line
    near = perp <= np.where(hl > 0, 8*s, 5*s)
    hl, ok = _orient(dash_pts, v, hl)
    near &= ok
    # the next dash/dot of the pattern must follow closely (dash-gap-dot-gap rhythm), on either side
    beyond = (((along - hl) > t1 - 5*s) & ((along - hl) < t1 + 130*s)) | (((along + hl) < t0 + 5*s) & ((along + hl) > t0 - 130*s))
    return bool((near & beyond).any())


def _thick_mask(net0, wthin, s):
    """pixels belonging to HEAVY strokes (outer boundary, corner stubs). Heavy = survives an opening that
    is wider than a normal boundary line. wthin = measured width of a normal line on THIS sheet, so it
    works whatever line weights the survey was plotted with (the old fixed 6.5*s kernel only matched one)."""
    d = max(5, int(round(1.4*wthin))) | 1
    op = cv2.morphologyEx(net0.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(op, connectivity=8)
    big = np.zeros(n, bool)
    if n > 1: big[1:] = np.maximum(st[1:, 2], st[1:, 3]) >= 45*s
    return big[lab] & net0

def _tf(thickm, ys, xs):
    """fraction of the given pixels that lie on heavy strokes"""
    if thickm is None or len(ys) == 0: return 0.0
    return float(thickm[ys, xs].mean())


def _dash_evidence_strong(ys, xs, dash_pts, s):
    """Like _dash_evidence, but a single stray dot lying roughly in line with the piece is not enough
    (a chain line crossing near a boundary leaves such dots by accident). Strong evidence = at least
    two fragments following the piece, or one real dash (not a dot) right behind it."""
    if dash_pts is None or len(dash_pts) == 0: return False
    P = np.stack([xs, ys], 1).astype(np.float64)
    c = P.mean(0)
    v = np.linalg.svd(P - c, full_matrices=False)[2][0] if len(P) > 2 else np.array([1.0, 0.0])
    t = (P - c) @ v
    t0, t1 = t.min(), t.max()
    d = dash_pts[:, :2] - c
    hl = dash_pts[:, 2] if dash_pts.shape[1] > 2 else np.zeros(len(d))
    along = d @ v
    perp = np.abs(d[:, 0]*v[1] - d[:, 1]*v[0])
    # fragments of a real dash-dot line sit ON its axis (a few px); 14 px off is somebody else's line
    near = perp <= np.where(hl > 0, 8*s, 5*s)
    hl, ok = _orient(dash_pts, v, hl)
    near &= ok
    beyond = (((along - hl) > t1 - 5*s) & ((along - hl) < t1 + 130*s)) | (((along + hl) < t0 + 5*s) & ((along + hl) > t0 - 130*s))
    hit = near & beyond
    if hit.sum() >= 2: return True
    return bool((hit & (hl > 0)).any())


def _chain_hits(ys, xs, chains, s, tol=40, ox=0, oy=0, ang=None):
    """fraction of the given pixels lying inside the corridor (+-tol*s) of a red dash-dot chain
    line (best chain wins). With ang (degrees) the piece must also run parallel to that chain.
    chains: list of (centre[2], unit_dir[2], t0, t1) in the same coordinates as the pixels."""
    if not chains or len(ys) == 0: return 0.0
    P = np.stack([xs + ox, ys + oy], 1).astype(np.float64)
    pv = None
    if ang is not None:
        if len(P) < 8: return 0.0
        pv = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]
    best = 0.0
    for c, v, t0, t1 in chains:
        if pv is not None and abs(pv @ v) < np.cos(np.deg2rad(ang)): continue
        d = P - c
        along = d @ v
        perp = np.abs(d[:, 0]*v[1] - d[:, 1]*v[0])
        ok = (perp <= tol*s) & (along >= t0 - 60*s) & (along <= t1 + 60*s)
        best = max(best, float(ok.mean()))
    return best

def _on_chain(ys, xs, chains, s, frac=0.8, tol=40, ox=0, oy=0):
    """piece runs along a (red) dash-dot construction line -> leftover piece of that line"""
    return _chain_hits(ys, xs, chains, s, tol, ox, oy, ang=15) >= frac

def _parallel_chain(ys, xs, chains, s, tol=25, ox=0, oy=0):
    """piece runs parallel to (and within tol of) a construction line"""
    return _chain_hits(ys, xs, chains, s, tol, ox, oy, ang=15) > 0.0

def _touch_chain(ys, xs, chains, s, tol=25, ox=0, oy=0):
    """any pixel of the piece sits on a construction line (short V / tick marks)"""
    return _chain_hits(ys, xs, chains, s, tol, ox, oy) > 0.0


def _continues_both(sk1, sl, x, y, w, h, s):
    """True when the piece (mask sl inside bbox x,y,w,h) carries on in a straight line beyond BOTH of
    its ends (a real line that a dash was drawn over). A dash/stub stops at least at one end."""
    H, W = sk1.shape
    ys, xs = np.nonzero(sl)
    P = np.stack([xs + x, ys + y], 1).astype(np.float64)
    if len(P) < 8: return False
    c = P.mean(0)
    p = np.linalg.svd(P - c, full_matrices=False)[2][0]
    t = (P - c) @ p
    pad = int(90*s) + 10
    y0, y1, x0, x1 = max(y - pad, 0), min(y + h + pad, H), max(x - pad, 0), min(x + w + pad, W)
    other = sk1[y0:y1, x0:x1].copy()
    other[ys + y - y0, xs + x - x0] = False
    other = cv2.dilate(other.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    for e, sgn in ((t.min(), -1.0), (t.max(), 1.0)):
        ds = np.arange(10*s, 50*s, max(2.0, 3*s))
        hits = 0
        for d in ds:
            q = c + p*(e + sgn*d)
            qx, qy = int(round(q[0])) - x0, int(round(q[1])) - y0
            if 0 <= qy < other.shape[0] and 0 <= qx < other.shape[1] and other[qy, qx]: hits += 1
        if hits < 0.7*len(ds): return False          # a crossing line gives a hit or two, a real continuation gives most
    return True

def _continues_either(sk1, sl, x, y, w, h, s):
    """True when the piece carries on in a straight line into other linework beyond AT LEAST ONE of its
    ends. A tick / V-mark left by a dash crossing a line sits at an angle to what it touches; a real
    boundary segment (e.g. the last stretch of a plot edge that a construction line crosses) runs on into
    the rest of its edge."""
    ys, xs = np.nonzero(sl)
    P = np.stack([xs + x, ys + y], 1).astype(np.float64)
    if len(P) < 8: return False
    H, W = sk1.shape
    c = P.mean(0)
    p = np.linalg.svd(P - c, full_matrices=False)[2][0]
    t = (P - c) @ p
    pad = int(90*s) + 10
    y0, y1, x0, x1 = max(y - pad, 0), min(y + h + pad, H), max(x - pad, 0), min(x + w + pad, W)
    other = sk1[y0:y1, x0:x1].copy()
    other[ys + y - y0, xs + x - x0] = False
    other = cv2.dilate(other.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    for e, sgn in ((t.min(), -1.0), (t.max(), 1.0)):
        ds = np.arange(12*s, 55*s, max(2.0, 3*s))
        hits = 0
        for d in ds:
            q = c + p*(e + sgn*d)
            qx, qy = int(round(q[0])) - x0, int(round(q[1])) - y0
            if 0 <= qy < other.shape[0] and 0 <= qx < other.shape[1] and other[qy, qx]: hits += 1
        if hits >= 0.7*len(ds): return True
    return False


def prune_spurs(sk, dt, L, thin_w, iters=4, dash_pts=None, s=1.0, sure_L=0, chains=None):
    """Remove thin free-ended spurs. A spur is only removed when it is tiny (< sure_L)
    or when dashed-line fragments line up with it (positive evidence it is a dash)."""
    sk = sk.copy()
    for _ in range(iters):
        deg, J, n, lab = _edges(sk)
        if n <= 1: break
        cnt = np.bincount(lab.ravel(), minlength=n)
        ends = np.bincount(lab[_endmask(sk, deg) & (lab > 0)], minlength=n)
        Jd = cv2.dilate(J.astype(np.uint8), np.ones((3,3), np.uint8)) > 0
        touchJ = np.bincount(lab[Jd & (lab > 0)], minlength=n)
        ys, xs = np.nonzero(lab)
        l = lab[ys, xs]
        wsum = np.bincount(l, weights=dt[ys, xs], minlength=n)
        width = 2 * wsum / np.maximum(cnt, 1)
        cand = (ends >= 1) & (touchJ > 0) & (cnt < L) & (width < thin_w)
        cand[0] = False
        rm = np.zeros(n, bool)
        order = np.argsort(l, kind='stable'); ls = l[order]
        starts = np.searchsorted(ls, np.arange(n)); stops = np.searchsorted(ls, np.arange(n), 'right')
        for i in np.nonzero(cand)[0]:
            if cnt[i] < sure_L:
                rm[i] = True; continue
            idx = order[starts[i]:stops[i]]
            if _dash_evidence(ys[idx], xs[idx], dash_pts, s): rm[i] = True
            elif np.median(dt[ys[idx], xs[idx]]) < 6*s and _on_chain(ys[idx], xs[idx], chains, s, frac=0.8): rm[i] = True     # lies along a red dash-dot line
        if not rm.any(): break
        sk[rm[lab]] = False
    return sk

def fill_small_holes(net, s):
    """arrow heads are small closed triangles: fill them so they skeletonise to a plain wedge, and
    remember where they were (evidence that a thin stroke next to them is an arrow).
    Returns (filled net, mask of all filled holes, lut) where lut(wthin) gives the arrow-head holes only:
    holes smaller than a speck of the sheet's own stroke width are print noise, not arrow heads."""
    inv = (~net).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    H, W = net.shape
    small = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H: continue
        if 1 <= a <= 1800*s*s and max(w, h) <= 90*s: small[i] = True
    filled = small[lab]
    def arrow_holes(wthin):
        keep = small & (st[:, 4] >= max(1.0, (0.15*wthin)**2))
        return keep[lab]
    return net | filled, filled, arrow_holes


def _axis(P, away_from):
    c = P.mean(0)
    v = np.linalg.svd(P - c, full_matrices=False)[2][0] if len(P) > 2 else np.array([1.0, 0.0])
    if (c - away_from) @ v < 0: v = -v
    return c, v


def _ray_runs(other, p, u, s, d0, d1, step=2):
    """hit flags of ink along the ray p + u*d (d in d0..d1)"""
    hits = []
    for d in np.arange(d0, d1, step):
        q = p + u*d
        qx, qy = int(round(q[0])), int(round(q[1]))
        hits.append(bool(0 <= qy < other.shape[0] and 0 <= qx < other.shape[1] and other[qy, qx]))
    return np.array(hits, bool)


def _longest_run(h):
    best = cur = 0
    for v in h:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _stub_geometry(i, order, starts, stops, ys, xs, ends, s):
    """pixels, free end, outward direction at the free end, junction end and outward direction beyond the junction"""
    idx = order[starts[i]:stops[i]]
    py, px = ys[idx], xs[idx]
    P = np.stack([px, py], 1).astype(np.float64)
    e_in = ends[py, px]
    if not e_in.any() or len(P) < 8: return None
    pend = P[np.nonzero(e_in)[0][0]]
    def fit(pt, away_from):
        d = np.hypot(P[:, 0] - pt[0], P[:, 1] - pt[1])
        q = P[d <= max(25, 70*s)]
        if len(q) < 6: return None
        c = q.mean(0); v = np.linalg.svd(q - c, full_matrices=False)[2][0]
        if (pt - c) @ v < 0: v = -v
        return v / max(np.hypot(*v), 1e-9)
    u = fit(pend, None)
    dj = np.hypot(P[:, 0] - pend[0], P[:, 1] - pend[1]); pj = P[dj.argmax()]
    vj = fit(pj, None)
    if u is None or vj is None: return None
    return P, pend, u, pj, vj

def _straight_through(i, order, starts, stops, ys, xs, sk, ends, txt, s, step=2):
    g = _stub_geometry(i, order, starts, stops, ys, xs, ends, s)
    if g is None: return False
    P, pend, u, pj, vj = g
    H, W = sk.shape
    # free end at printed text
    r = int(30*s)
    ex, ey = int(pend[0]), int(pend[1])
    if txt[max(ey-r, 0):ey+r+1, max(ex-r, 0):ex+r+1].sum() < 40*s*s: return False
    # straight through the junction: kept ink carries on beyond the other end along the same line
    R2 = int(160*s)
    x0, y0 = max(int(pj[0]) - R2, 0), max(int(pj[1]) - R2, 0)
    x1, y1 = min(int(pj[0]) + R2, W), min(int(pj[1]) + R2, H)
    sub = sk[y0:y1, x0:x1].copy()
    inw = (P[:, 1] >= y0) & (P[:, 1] < y1) & (P[:, 0] >= x0) & (P[:, 0] < x1)
    sub[P[inw, 1].astype(int) - y0, P[inw, 0].astype(int) - x0] = False
    other = cv2.dilate(sub.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    org = pj - np.array([x0, y0])
    best = 0
    for ang in (0.0, -1.5, 1.5, -3.0, 3.0, -5.0, 5.0):
        ca, sa = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        ur = np.array([vj[0]*ca - vj[1]*sa, vj[0]*sa + vj[1]*ca])
        best = max(best, _longest_run(_ray_runs(other, org, ur, s, 3, 150*s)))
    return best >= max(8, int(30*s/step))

def _hidden_under_label(i, lab, order, starts, stops, ys, xs, sk, ends, txt, s, step=2):
    """A boundary hidden under a label leaves two stubs facing each other across the text. Keep a stub when
    another stub faces it through printed text and at least one of the two runs straight through its junction."""
    g = _stub_geometry(i, order, starts, stops, ys, xs, ends, s)
    if g is None: return False
    P, pend, u, pj, vj = g
    H, W = sk.shape
    rg = int(150*s)
    ex, ey = int(pend[0]), int(pend[1])
    x0, y0, x1, y1 = max(ex-rg, 0), max(ey-rg, 0), min(ex+rg+1, W), min(ey+rg+1, H)
    cy_, cx_ = np.nonzero(ends[y0:y1, x0:x1] & (lab[y0:y1, x0:x1] > 0) & (lab[y0:y1, x0:x1] != i))
    facing, seen = [], set()
    for yy, xx in zip(cy_ + y0, cx_ + x0):
        j = int(lab[yy, xx])
        if j in seen: continue
        v = np.array([xx - pend[0], yy - pend[1]], np.float64); d = np.hypot(*v)
        if d < 6 or d > rg: continue
        if (v / d) @ u < 0.5: continue                       # the other stub lies in front of this one
        ts_ = np.linspace(0, 1, 14)
        pts = (pend[None, :] + v[None, :]*ts_[:, None]).astype(int)
        cov = np.mean([txt[min(max(q[1]+dy, 0), H-1), min(max(q[0]+dx, 0), W-1)] for q in pts for dy in (-5, 0, 5) for dx in (-5, 0, 5)])
        if cov < 0.2: continue                               # the gap between them is a label
        seen.add(j); facing.append(j)
    if not facing: return False
    if _straight_through(i, order, starts, stops, ys, xs, sk, ends, txt, s, step): return True
    return any(_straight_through(j, order, starts, stops, ys, xs, sk, ends, txt, s, step) for j in facing)

def strip_dangling(sk, dt, arrow_zone, s, log, rounds=16, Lmax=420, thick_dt=6.0, thickm=None, txt=None, head_zone=None):
    """Remove arrows, ticks and other thin strokes that hang off the network with a free end.
    A dangling thin edge is KEPT only when it really is a boundary:
      - it is heavy (outer boundary stroke), or
      - it carries on in a straight line beyond its free end (line cut by a label), or
      - its free end almost touches other linework (a boundary that stops just short of another one)
    unless an arrow head (small closed triangle) sits on it - then it is always an arrow.
    A piece of an arrow lying between two boundaries is removed because it continues into
    removed arrow pieces."""
    sk = sk.copy()
    removed_total = np.zeros(sk.shape, bool)
    nrem = 0
    for rnd in range(rounds):
        deg, J, n, lab = _edges(sk)
        if n <= 1: break
        ends = _endmask(sk, deg)
        cnt = np.bincount(lab.ravel(), minlength=n)
        endcnt = np.bincount(lab[ends & (lab > 0)], minlength=n)
        Jd = cv2.dilate(J.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        touchJ = np.bincount(lab[Jd & (lab > 0)], minlength=n)
        cand = (endcnt >= 1) & (touchJ > 0) & (cnt <= Lmax*s) & (cnt >= 1)
        cand[0] = False
        ys, xs = np.nonzero(lab)
        l = lab[ys, xs]
        order = np.argsort(l, kind='stable'); ls = l[order]
        starts = np.searchsorted(ls, np.arange(n)); stops = np.searchsorted(ls, np.arange(n), 'right')
        rm = np.zeros(n, bool)
        R = int(200*s) + 12
        R2 = int(345*s) + 12
        H, W = sk.shape
        # an arrow head drawn as a small closed triangle is a loop with no free end, so it never counts as dangling:
        # an edge that lies wholly inside an arrow-head zone and closes on itself is part of the head
        inz = np.bincount(lab[head_zone & (lab > 0)], minlength=n) if head_zone is not None else np.zeros(n, int)
        loopc = (endcnt == 0) & (cnt >= 3) & (cnt <= 110*s) & (inz >= 0.95*cnt)
        loopc[0] = False
        if head_zone is not None and inz.any():
            nj_, jlab = cv2.connectedComponents(Jd.astype(np.uint8), connectivity=8)
            shaft = (endcnt == 0) & (touchJ >= 2) & (cnt >= 60*s) & (cnt <= Lmax*s)
            shaft[0] = False
            Hh, Ww = sk.shape
            for i in np.nonzero(shaft)[0]:
                idx = order[starts[i]:stops[i]]
                py, px = ys[idx], xs[idx]
                if np.median(dt[py, px]) >= thick_dt*s or _tf(thickm, py, px) >= 0.6: continue
                P = np.stack([px, py], 1).astype(np.float64)
                c0 = P.mean(0); v = np.linalg.svd(P - c0, full_matrices=False)[2][0]
                tt = (P - c0) @ v
                if np.hypot(*(P[tt.argmax()] - P[tt.argmin()])) / len(P) < 0.9: continue    # a straight shaft
                for e in (P[tt.argmin()], P[tt.argmax()]):
                    ex, ey = int(e[0]), int(e[1])
                    x0, y0, x1, y1 = max(ex-45, 0), max(ey-45, 0), min(ex+46, Ww), min(ey+46, Hh)
                    win = jlab[y0:y1, x0:x1]
                    near = win[max(ey-y0-4, 0):ey-y0+5, max(ex-x0-4, 0):ex-x0+5]
                    cl = np.unique(near[near > 0])
                    if len(cl) != 1: continue
                    cl_m = cv2.dilate((win == cl[0]).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                    others = np.unique(lab[y0:y1, x0:x1][cl_m]); others = others[(others > 0) & (others != i)]
                    # the head: only short edges lying wholly in an arrow-head zone meet at this end of the shaft
                    hz_ = (cnt[others] <= 110*s) & (inz[others] >= 0.5*cnt[others])
                    if len(others) >= 2 and hz_.all():
                        # the head lies beyond the tip of the shaft, along its axis (not alongside it)
                        u_out = (e - c0); u_out = u_out / max(np.hypot(*u_out), 1e-9)
                        ok_dir = True
                        for o in others:
                            io = order[starts[o]:stops[o]]
                            cen = np.array([xs[io].mean(), ys[io].mean()])
                            if (cen - e) @ u_out < -0.2*np.hypot(*(cen - e)): ok_dir = False
                        if not ok_dir: continue
                        rm[i] = True; rm[others] = True
                        break
        for i in np.nonzero(cand)[0]:
            idx = order[starts[i]:stops[i]]
            py, px = ys[idx], xs[idx]
            if (np.median(dt[py, px]) >= thick_dt*s or _tf(thickm, py, px) >= 0.6) and len(idx) >= 60*s: continue   # long heavy boundary stroke (short heavy stubs = arrow-head wedges)
            if arrow_zone[py, px].any():
                # a long line that merely grazes an arrow-head zone is a boundary, not the arrow's shaft
                if not len(idx) > 250*s:
                    rm[i] = True; continue                          # arrow head on it
            P = np.stack([px, py], 1).astype(np.float64)
            # free end pixel(s) of this edge
            e_in = ends[py, px]
            if not e_in.any(): continue
            pend = P[np.nonzero(e_in)[0][0]]
            # outward direction at the free end: line fitted to the last ~70*s px (an end-minus-centroid
            # estimate is several degrees off, which misses the far side of a label gap 150 px away)
            dd = np.hypot(P[:, 0] - pend[0], P[:, 1] - pend[1])
            near = P[dd <= max(25, 70*s)]
            if len(near) < 6: near = P[dd <= max(12, 25*s)]
            if len(near) < 4: near = P
            cn = near.mean(0)
            u = np.linalg.svd(near - cn, full_matrices=False)[2][0] if len(near) > 2 else pend - cn
            if (pend - cn) @ u < 0: u = -u
            nu = np.hypot(*u)
            if nu < 1e-6: continue
            u = u / nu
            x0, y0 = max(int(pend[0]) - R2, 0), max(int(pend[1]) - R2, 0)
            x1, y1 = min(int(pend[0]) + R2, W), min(int(pend[1]) + R2, H)
            sub = sk[y0:y1, x0:x1].copy()
            inw = (py >= y0) & (py < y1) & (px >= x0) & (px < x1)     # long edges can leave the window
            sub[py[inw] - y0, px[inw] - x0] = False
            # other tiny free-ended pieces (arrow-head leftovers) are not continuation evidence: they are about to be
            # judged themselves, and a stub direction measured over a few pixels is unreliable
            sub_lab = lab[y0:y1, x0:x1]
            small_c = cand & (cnt <= 40*s); small_c[i] = False
            sub &= ~small_c[sub_lab]
            # own edge's neighbours (junction pixels it hangs on) lie behind the end, harmless
            other = cv2.dilate(sub.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0     # +-4 px: direction is only known to ~1 deg
            step = 2
            org = pend - np.array([x0, y0])
            # does the line run into printed text right at its end? then it was cut by a label and may
            # continue on the far side of it - look further (labels are up to ~250 px long)
            reach = 170*s
            if txt is not None:
                th = _ray_runs(txt[y0:y1, x0:x1], org, u, s, 3, 120*s)
                if th.sum() >= 6: reach = 330*s
            best = None
            for ang in (0.0, -1.5, 1.5, -3.0, 3.0, -5.0, 5.0):
                ca, sa = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
                ur = np.array([u[0]*ca - u[1]*sa, u[0]*sa + u[1]*ca])
                hits = _ray_runs(other, org, ur, s, 3, reach)
                lr_ = _longest_run(hits)
                if best is None or lr_ > best[0]: best = (lr_, hits)
            hits = best[1]
            # a free end only a few pixels long has no measurable direction: whatever 'continuation' a ray finds from it is
            # a guess (it can point back along the line it hangs on). Such a speck is a leftover, not a boundary.
            if len(idx) < 12*s:
                rm[i] = True; continue
            # (1) carries on collinearly beyond a gap -> line cut by a label
            if _longest_run(hits) >= max(6, int(18*s/step)): continue
            # (2) free end almost touches other linework straight ahead
            if hits[:max(1, int(38*s/step))].any(): continue
            # a boundary hidden under a label: this piece runs straight through the junction it hangs on, and its free
            # end stops at printed text (the label sits ON the line) - or it faces such a piece across the label
            if txt is not None and _hidden_under_label(i, lab, order, starts, stops, ys, xs, sk, ends, txt, s, step):
                continue
            # a long straight line whose free end sits right beside printed text: the label hid the junction
            if txt is not None and len(idx) >= 150*s and np.hypot(*(P.max(0) - P.min(0))) / len(idx) > 0.9:
                r_ = int(45*s)
                ex, ey = int(pend[0]), int(pend[1])
                if txt[max(ey-r_, 0):ey+r_+1, max(ex-r_, 0):ex+r_+1].sum() >= 150*s*s: continue
            rm[i] = True
        # pieces lying between two boundaries that continue into removed arrow pieces
        if rm.any():
            removed_total |= rm[lab]
            sk[rm[lab]] = False
            nrem += int(rm.sum())
        else:
            break
    # middle pieces of an arrow shaft (between two boundary crossings)
    if removed_total.any():
        rz = cv2.dilate(removed_total.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        deg, J, n, lab = _edges(sk)
        cnt = np.bincount(lab.ravel(), minlength=n)
        H, W = sk.shape
        killm = np.zeros(n, bool)
        ys, xs = np.nonzero(lab)
        l = lab[ys, xs]
        order = np.argsort(l, kind='stable'); ls = l[order]
        starts = np.searchsorted(ls, np.arange(n)); stops = np.searchsorted(ls, np.arange(n), 'right')
        for i in range(1, n):
            if cnt[i] < 5 or cnt[i] > 300*s: continue
            idx = order[starts[i]:stops[i]]
            py, px = ys[idx], xs[idx]
            if dt[py, px].max() >= thick_dt*s or _tf(thickm, py, px) >= 0.3: continue
            P = np.stack([px, py], 1).astype(np.float64)
            c = P.mean(0); v = np.linalg.svd(P - c, full_matrices=False)[2][0]
            t = (P - c) @ v
            if np.hypot(*(P[t.argmax()] - P[t.argmin()])) / max(1, cnt[i]) < 0.9: continue   # straight only
            into = 0
            for e, sg in ((P[t.argmin()], -1.0), (P[t.argmax()], 1.0)):
                u = v*sg
                hh = _ray_runs(rz, e, u, s, 3, 40*s, step=1)
                if _longest_run(hh) >= 6: into += 1
            # between TWO crossings, both continuing into removed pieces; a boundary that merely runs on into a
            # removed extension line past its corner is not a shaft
            if into == 2: killm[i] = True
        if killm.any():
            sk[killm[lab]] = False
            nrem += int(killm.sum())
    log(f'  arrow/stub strip: removed {nrem} dangling pieces')
    return sk


def _trace_path(sl, start):
    """order the pixels of a simple skeleton path from end pixel start=(y, x); returns (x, y) array"""
    H, W = sl.shape
    seen = np.zeros_like(sl); y, x = int(start[0]), int(start[1]); out = []
    while True:
        seen[y, x] = True; out.append((x, y)); nxt = None
        for dy, dx in _OFFS:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and sl[yy, xx] and not seen[yy, xx]:
                nxt = (yy, xx); break
        if nxt is None: break
        y, x = nxt
    return np.array(out) if len(out) >= 0.9*sl.sum() else None

def _right_angle_path(sl, s):
    """True when the simple path `sl` bends only at (near) right angles - the corner of a plot outline.
    An arrow shaft that happens to touch a boundary stub bends at some other angle."""
    deg = _deg(sl)
    pts = np.argwhere(sl & (deg == 1))
    if len(pts) != 2: return False
    path = _trace_path(sl, pts[0])
    if path is None: return False
    ap = cv2.approxPolyDP(path.reshape(-1, 1, 2).astype(np.int32), 3*s, False).reshape(-1, 2).astype(np.float64)
    if len(ap) < 3: return True
    v = np.diff(ap, axis=0)
    for a_, b_ in zip(v[:-1], v[1:]):
        if min(np.hypot(*a_), np.hypot(*b_)) < 12*s: continue
        cosang = (a_ @ b_) / (np.hypot(*a_) * np.hypot(*b_))
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        if not (ang <= 15 or 72 <= ang <= 108): return False
    return True


def _bent_boundary_ends(sl, s):
    """A removed piece that is one simple, long path made of a few long straight runs (a bent
    boundary such as a plot outline with a kink) is not text.  Text glued to a line has junctions,
    loops and curves, which simplify into many short pieces.  Returns the two end points (x, y) or None."""
    a = int(sl.sum())
    if a < 100*s: return None
    deg = _deg(sl)
    if (deg >= 3).any(): return None
    pts = np.argwhere(sl & (deg == 1))
    if len(pts) != 2: return None
    path = _trace_path(sl, pts[0])
    if path is None: return None
    ap = cv2.approxPolyDP(path.reshape(-1, 1, 2).astype(np.int32), 3*s, False).reshape(-1, 2)
    L = np.hypot(*np.diff(ap, axis=0).T)
    if L[L >= 30*s].sum() < 0.9 * L.sum(): return None
    return pts[0][::-1].astype(int), pts[1][::-1].astype(int)

def reconstruct(net, sk, dt, slack=0.8):
    """pixels of net lying within local stroke-radius of skeleton sk"""
    if not sk.any(): return np.zeros_like(net)
    ys, xs = np.nonzero(sk)                         # raster order == label order
    dist, labs = cv2.distanceTransformWithLabels((~sk).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
    r = dt[ys, xs]
    rad = np.concatenate([[0], r]).astype(np.float32)[labs]
    del labs
    out = net & (dist <= rad + np.float32(slack))
    return out

def clean_network(net, s, log=print, dash_pts=None, chains=None, txt=None):
    """net: bool crop of black-core network pixels. s: scale factor vs 9917px-wide sheet."""
    net0 = net.copy()
    net, filled, arrow_holes = fill_small_holes(net0, s)            # arrow heads -> solid wedges
    dt = cv2.distanceTransform(net.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(net)
    dt0 = cv2.distanceTransform(net0.astype(np.uint8), cv2.DIST_L2, 5)   # true stroke width (head wedges are filled in dt)
    wthin = 2*float(np.median(dt0[sk])) if sk.any() else 9.5*s/1.41
    thickm = _thick_mask(net0, wthin, s)
    filled = arrow_holes(wthin); del arrow_holes
    thin_w = min(14*s, 1.5*wthin)
    log(f'  skeleton px: {int(sk.sum())}, normal line width ~{wthin:.1f}px, heavy-stroke px {int(thickm.sum())}')
    sk1 = prune_spurs(sk, dt, L=int(330*s), thin_w=thin_w, dash_pts=dash_pts, s=s, sure_L=int(45*s), chains=chains)
    arrow_zone = cv2.dilate(filled.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*int(16*s)+1,)*2)) > 0
    # arrow-head loops are thin slivers; a fat little triangle (two lines crossing a boundary) is a real plot corner
    nh_, lh_, _, _ = cv2.connectedComponentsWithStats(filled.astype(np.uint8), connectivity=8)
    dth_ = cv2.distanceTransform(filled.astype(np.uint8), cv2.DIST_L2, 3)
    mx_ = np.zeros(nh_, np.float32); np.maximum.at(mx_, lh_.ravel(), dth_.ravel())
    thin_h = (mx_ <= 0.8*wthin); thin_h[0] = False
    head_zone = cv2.dilate(thin_h[lh_].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*int(16*s)+1,)*2)) > 0
    del lh_, dth_
    sk_pre = sk1
    sk1 = strip_dangling(sk1, dt0, arrow_zone, s, log, thickm=thickm, txt=txt, head_zone=head_zone)
    del dt0, filled, head_zone
    stripped = cv2.dilate((sk_pre & ~sk1).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # arrows/ticks: never rescue
    sk1 = prune_spurs(sk1, dt, L=int(60*s), thin_w=thin_w, iters=2, s=s, sure_L=int(60*s))
    log(f'  after spur pruning: {int(sk1.sum())}')
    # long straight runs (survive being split by glued text)
    u8 = (sk1 * 255).astype(np.uint8)
    segs = cv2.HoughLinesP(u8, 1, np.pi/720, threshold=int(120*s), minLineLength=int(220*s), maxLineGap=int(25*s))
    Lm = np.zeros(sk1.shape, np.uint8)
    if segs is not None:
        for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
            cv2.line(Lm, (x1, y1), (x2, y2), 255, 1)
    line_sk = sk1 & (cv2.dilate(Lm, np.ones((7,7), np.uint8)) > 0)
    rest = sk1 & ~line_sk
    del u8, Lm
    n, lab, st, _ = cv2.connectedComponentsWithStats(rest.astype(np.uint8), connectivity=8)
    lineD = cv2.dilate(line_sk.astype(np.uint8), np.ones((11,11), np.uint8)) > 0
    kr = int(10*s) + 1
    keptD = cv2.dilate(line_sk.astype(np.uint8), np.ones((2*kr+1, 2*kr+1), np.uint8)) > 0
    deg = _deg(rest)
    del rest
    rm = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        sl = lab[y:y+h, x:x+w] == i
        touches = lineD[y:y+h, x:x+w][sl].any()
        junc = (deg[y:y+h, x:x+w][sl] >= 3).any()
        diag = np.hypot(w, h)
        _ty, _tx = np.nonzero(sl)
        tfr = _tf(thickm, _ty + y, _tx + x)
        thick = dt[y:y+h, x:x+w][sl].mean() >= 9*s or tfr >= 0.6
        if thick and touches and max(w, h) < 260*s: continue     # short thick boundary jog
        stroke_thick = dt[y:y+h, x:x+w][sl].max() >= 6*s or tfr >= 0.3   # any part of it is a heavy boundary stroke
        if not thick and not stroke_thick and chains:
            yy0, xx0 = np.nonzero(sl)
            par = _on_chain(yy0 + y, xx0 + x, chains, s, frac=0.6)
            free = not _continues_both(sk1, sl, x, y, w, h, s)   # stops at an end -> dash/stub; a line that carries on both sides is a real boundary
            # parallel to a red dash-dot line AND ending freely -> leftover stub of that line
            if par and free and diag < 400*s: rm[i] = True; continue
            # short non-parallel tick / V mark sitting on a red dash-dot line
            if (30*s <= diag < 200*s and not _parallel_chain(yy0 + y, xx0 + x, chains, s)
                    and _touch_chain(yy0 + y, xx0 + x, chains, s)
                    and not _continues_either(sk1, sl, x, y, w, h, s)): rm[i] = True; continue
        # a simple path (straight or with a few kinks) that starts and ends ON kept boundary lines is a
        # boundary itself - e.g. a plot edge with a bend, or an edge whose neighbour was a glued arrow.
        # Text has junctions / loops / curves, dash stubs have a free end, so neither passes this test.
        if not junc and a >= 100*s:
            be = _bent_boundary_ends(sl, s)
            if be is not None:
                e0, e1 = be
                def _on_kept(e):
                    ex, ey = int(e[0]) + x, int(e[1]) + y
                    return bool(keptD[ey, ex])
                if _on_kept(e0) and _on_kept(e1):
                    yb, xb = np.nonzero(sl)
                    if not _dash_evidence_strong(yb + y, xb + x, dash_pts, s):
                        continue
        straight = (not junc) and a > 0 and diag / a > 0.9
        if straight:
            # straight pieces are boundaries unless dashed-line fragments line up with them
            yy, xx = np.nonzero(sl)
            dashy = _dash_evidence(yy + y, xx + x, dash_pts, s)
            if not dashy and a >= 40*s: continue
            if not dashy and touches and a > 25*s: continue
            if dashy and a < 330*s: rm[i] = True; continue
        if max(w, h) > 330*s: continue
        # plot-number sized glyph glued to a line -> keep
        if 55*s <= h <= 80*s and w <= 100*s and not straight: continue
        rm[i] = True
    sk2 = sk1 & ~rm[lab]
    del lineD, keptD, deg
    # inside removed clusters, keep any long straight edge (boundaries share junctions with junk)
    rcl = sk1 & rm[lab]
    if rcl.any():
        degc = _deg(rcl)
        Jc = rcl & (degc >= 3)
        edg = (rcl & ~(cv2.dilate(Jc.astype(np.uint8), np.ones((3,3), np.uint8)) > 0)).astype(np.uint8)
        ne, le, ste, _ = cv2.connectedComponentsWithStats(edg, connectivity=8)
        keepe = np.zeros(ne, bool)
        for i in range(1, ne):
            x, y, w, h, a = ste[i]
            straight_e = a >= 30*s and np.hypot(w, h) / a > 0.93
            bent_e = (not straight_e) and a >= 100*s and _bent_boundary_ends(le[y:y+h, x:x+w] == i, s) is not None and _right_angle_path(le[y:y+h, x:x+w] == i, s) and not arrow_zone[np.nonzero(le[y:y+h, x:x+w] == i)[0] + y, np.nonzero(le[y:y+h, x:x+w] == i)[1] + x].any()   # plot corner: two long edges, no junction between them
            if (straight_e and (a >= 70*s or _continues_either(sk2, le[y:y+h, x:x+w] == i, x, y, w, h, s))) or bent_e:
                # a short straight stretch is still a boundary when it runs on into kept linework
                # (e.g. the last 80 px of a plot edge between an arrow crossing and the next corner)
                yy, xx = np.nonzero(le[y:y+h, x:x+w] == i)
                if not _dash_evidence_strong(yy + y, xx + x, dash_pts, s) and not _on_chain(yy + y, xx + x, chains, s, frac=0.6): keepe[i] = True
        if keepe.any():
            ke = keepe[le]
            sk2 |= rcl & (cv2.dilate(ke.astype(np.uint8), np.ones((7,7), np.uint8)) > 0)
            log(f'  kept {int(keepe.sum())} straight edges inside junk clusters')
    log(f'  removed {int(rm.sum())} glued/stub clusters')
    del arrow_zone
    sk3 = prune_spurs(sk2, dt, L=int(330*s), thin_w=thin_w, iters=3, s=s, sure_L=int(45*s), chains=chains)       # tiny ticks left on lines
    # --- boundary rescue: a removed straight piece that bridges two kept lines is a boundary ---
    removed = sk & ~sk3 & ~stripped
    nr, lr, str_, _ = cv2.connectedComponentsWithStats(removed.astype(np.uint8), connectivity=8)
    if nr > 1:
        dist = cv2.distanceTransform((~sk3).astype(np.uint8), cv2.DIST_L2, 3)
        resc = np.zeros(nr, bool)
        for i in range(1, nr):
            x, y, w, h, a = str_[i]
            if a < 12*s: continue
            sl = lr[y:y+h, x:x+w] == i
            yy, xx = np.nonzero(sl)
            P = np.stack([xx, yy], 1).astype(np.float64)
            if np.hypot(w, h) / a < 0.88:                   # not straight: only a bent boundary path survives
                be = _bent_boundary_ends(sl, s)
                if be is None: continue                     # curved -> text, leave removed
                e0, e1 = be
            else:
                c = P.mean(0); v = np.linalg.svd(P - c, full_matrices=False)[2][0]
                t = (P - c) @ v
                e0 = P[t.argmin()].astype(int); e1 = P[t.argmax()].astype(int)
            d0 = dist[y+e0[1], x+e0[0]]; d1 = dist[y+e1[1], x+e1[0]]
            # an end that stops short of other ink is a real boundary only if a label sits in the gap AND the
            # line, carried straight on, runs into that ink (a dimension leader pointing at its number does not)
            ok0 = d0 <= 10*s or (_gap_in_text(txt, y+e0[1], x+e0[0], d0, s) and _runs_into(sk3, sl, x, y, e0, d0, s))
            ok1 = d1 <= 10*s or (_gap_in_text(txt, y+e1[1], x+e1[0], d1, s) and _runs_into(sk3, sl, x, y, e1, d1, s))
            if ok0 and ok1 and (d0 <= 10*s or d1 <= 10*s or (d0 <= 130*s and d1 <= 130*s)):
                ys_, xs_ = np.nonzero(sl)
                if (not _dash_evidence_strong(ys_ + y, xs_ + x, dash_pts, s)
                        and not (30*s <= np.hypot(w, h) < 200*s and not _parallel_chain(ys_ + y, xs_ + x, chains, s)
                                 and _touch_chain(ys_ + y, xs_ + x, chains, s) and dt[y:y+h, x:x+w][sl].max() < 6*s and _tf(thickm, ys_ + y, xs_ + x) < 0.3
                                 and not _continues_either(sk3, sl, x, y, w, h, s))):
                    resc[i] = True
        if resc.any():
            log(f'  rescued {int(resc.sum())} boundary pieces')
            sk3 = sk3 | resc[lr]
    # --- dash pieces lying on a dash-dot construction line, cut at junctions (dash crossing a boundary) ---
    # several passes: once the dash segment between two lines is gone, the next one has no continuation either
    nkill = 0
    for _pass in range(4):
        if not chains: break
        deg3 = _deg(sk3)
        J3 = sk3 & (deg3 >= 3)
        e3 = (sk3 & ~(cv2.dilate(J3.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)).astype(np.uint8)
        ne, le, ste, _ = cv2.connectedComponentsWithStats(e3, connectivity=8)
        kill = np.zeros(ne, bool)
        for i in range(1, ne):
            x, y, w, h, a = ste[i]
            if a < 25*s or a > 400*s: continue
            sl = le[y:y+h, x:x+w] == i
            yy, xx = np.nonzero(sl)
            if dt[yy + y, xx + x].max() >= 6*s or _tf(thickm, yy + y, xx + x) >= 0.3: continue   # heavy boundary stroke
            if _chain_hits(yy, xx, chains, s, tol=20, ox=x, oy=y, ang=8) < 0.9: continue
            if _continues_both(sk3, sl, x, y, w, h, s): continue                 # real line a dash was drawn over
            kill[i] = True
            break                                                                 # remove one, re-evaluate the rest
        if not kill.any(): break
        sk3 = sk3 & ~kill[le]
        nkill += 1
    if nkill:
        sk3 = prune_spurs(sk3, dt, L=int(45*s), thin_w=thin_w, iters=2, s=s, sure_L=int(45*s))
        log(f'  removed {nkill} dash pieces lying on construction lines')
    out = reconstruct(net0, sk3, dt)
    # thick outer-boundary strokes are never touched
    out = out | thickm
    return bridge_gaps(out, s, log, txt=txt)


def _gap_in_text(txt, y, x, d, s):
    """True when a line end that stops short of other ink (distance d) has printed text next to it:
    the label sliced the line, so the gap is not real."""
    if txt is None or d > 130*s: return False
    r = int(d + 12*s) + 1
    H, W = txt.shape
    return bool(txt[max(y-r, 0):min(y+r+1, H), max(x-r, 0):min(x+r+1, W)].sum() >= 40*s*s)


def _runs_into(sk3, sl, x, y, e, d, s):
    """The piece `sl` (bbox origin x,y) ends at e=(ex,ey) in piece coords. Carry its local direction straight on:
    True when it meets kept skeleton within about the measured gap d."""
    ys, xs = np.nonzero(sl)
    P = np.stack([xs, ys], 1).astype(np.float64)
    e = np.asarray(e, np.float64)
    near = P[np.hypot(*(P - e).T) <= 60*s]
    if len(near) < 10: return False
    c = near.mean(0)
    u = e - c
    nu = np.hypot(*u)
    if nu < 1e-6: return False
    u /= nu
    H, W = sk3.shape
    reach = d*1.35 + 15*s
    for ang in (0.0, 3.0, -3.0, 6.0, -6.0):
        ca, sa = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        v = np.array([u[0]*ca - u[1]*sa, u[0]*sa + u[1]*ca])
        for r in np.arange(4*s, reach, 2.0):
            q = e + v*r
            qx, qy = int(round(q[0] + x)), int(round(q[1] + y))
            if not (0 <= qy < H and 0 <= qx < W): break
            y0, y1, x0, x1 = max(qy-3, 0), min(qy+4, H), max(qx-3, 0), min(qx+4, W)
            win = sk3[y0:y1, x0:x1]
            if win.any():
                # must be other ink, not the piece itself (it is not in sk3: it was removed)
                return True
    return False


def _end_directions(skel, ends, R):
    """outward unit direction at each skeleton end (from the local branch it terminates)"""
    H, W = skel.shape
    dirs = []
    for (y, x) in ends:
        y0, y1, x0, x1 = max(y-R, 0), min(y+R+1, H), max(x-R, 0), min(x+R+1, W)
        crop = skel[y0:y1, x0:x1].astype(np.uint8)
        n, lab = cv2.connectedComponents(crop, connectivity=8)
        l = lab[y-y0, x-x0]
        ys, xs = np.nonzero(lab == l)
        if len(ys) < 6: dirs.append((0.0, 0.0)); continue
        P = np.stack([xs + x0, ys + y0], 1).astype(np.float64)
        near = np.hypot(P[:,0]-x, P[:,1]-y) <= R*0.8
        P = P[near]
        c = P.mean(0)
        v = np.array([x, y], np.float64) - c
        nv = np.hypot(*v)
        dirs.append((v[0]/nv, v[1]/nv) if nv > 1e-6 else (0.0, 0.0))
    return np.array(dirs)

def bridge_gaps(mask, s, log=print, max_gap=75, ang_tol=14.0, txt=None, kink_tol=32.0):
    """Reconnect line ends that were cut by labels drawn on top of the line."""
    sk = skeletonize(mask)
    deg = _deg(sk)
    ends = np.argwhere(_endmask(sk, deg))
    if len(ends) < 2: return mask
    R = max(12, int(45*s))
    d = _end_directions(sk, ends, R)
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    P = ends[:, ::-1].astype(np.float64)                   # x, y
    out = mask.copy().astype(np.uint8)
    used = np.zeros(len(P), bool); drawn = 0
    cos_tol = np.cos(np.deg2rad(ang_tol))
    for i in range(len(P)):
        if used[i] or not d[i].any(): continue
        v = P - P[i]; dist = np.hypot(v[:,0], v[:,1])
        ok = (dist > 3) & (dist <= max_gap*s) & ~used
        ok[i] = False
        if not ok.any(): continue
        vn = v / np.maximum(dist[:, None], 1e-6)
        a_i = vn @ d[i]                                     # my end points toward j
        a_j = -(vn * d).sum(1)                              # j's end points toward me
        good = ok & (a_i >= cos_tol) & (a_j >= cos_tol)
        if not good.any() and txt is not None:
            # a boundary that kinks where a label sits on it: one piece is aimed squarely at the other, the other
            # leans off by up to kink_tol degrees. Allowed only when printed text sits in the gap.
            cos_k = np.cos(np.deg2rad(kink_tol)); cos_s = np.cos(np.deg2rad(14.0))
            lk = ok & (a_i >= cos_k) & (a_j >= cos_k) & (np.maximum(a_i, a_j) >= cos_s)
            H_, W_ = txt.shape
            for jj in np.nonzero(lk)[0]:
                mx, my = (P[i] + P[jj]) / 2
                r = int(dist[jj]/2 + 40*s)
                win = txt[max(int(my)-r, 0):min(int(my)+r+1, H_), max(int(mx)-r, 0):min(int(mx)+r+1, W_)]
                if win.sum() >= 40*s*s: good[jj] = True
        if not good.any(): continue
        j = np.nonzero(good)[0][dist[good].argmin()]
        # the corridor between the ends must be empty ink (a real gap, not a different line)
        w = int(round(2*max(dt[tuple(ends[i])], dt[tuple(ends[j])]))) + 1
        cv2.line(out, tuple(int(round(t)) for t in P[i]), tuple(int(round(t)) for t in P[j]), 1, max(w, 2))
        used[i] = used[j] = True; drawn += 1
    log(f'  bridged {drawn} gaps')
    return out > 0