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
    near = perp <= 10*s
    # the next dash/dot of the pattern must follow closely (dash-gap-dot-gap rhythm), on either side
    beyond = (((along - hl) > t1 - 5*s) & ((along - hl) < t1 + 130*s)) | (((along + hl) < t0 + 5*s) & ((along + hl) > t0 - 130*s))
    return bool((near & beyond).any())


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

def reconstruct(net, sk, dt, slack=0.8):
    """pixels of net lying within local stroke-radius of skeleton sk"""
    if not sk.any(): return np.zeros_like(net)
    ys, xs = np.nonzero(sk)                         # raster order == label order
    dist, labs = cv2.distanceTransformWithLabels((~sk).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
    r = dt[ys, xs]
    rad = np.concatenate([[0], r])[labs]
    return net & (dist <= rad + slack)

def clean_network(net, s, log=print, dash_pts=None, chains=None):
    """net: bool crop of black-core network pixels. s: scale factor vs 9917px-wide sheet."""
    net = net.copy()
    dt = cv2.distanceTransform(net.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(net)
    log(f'  skeleton px: {int(sk.sum())}')
    sk1 = prune_spurs(sk, dt, L=int(330*s), thin_w=14*s, dash_pts=dash_pts, s=s, sure_L=int(45*s), chains=chains)
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
    n, lab, st, _ = cv2.connectedComponentsWithStats(rest.astype(np.uint8), connectivity=8)
    lineD = cv2.dilate(line_sk.astype(np.uint8), np.ones((11,11), np.uint8)) > 0
    deg = _deg(rest)
    endm_sk = _endmask(sk1)
    rm = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        sl = lab[y:y+h, x:x+w] == i
        touches = lineD[y:y+h, x:x+w][sl].any()
        junc = (deg[y:y+h, x:x+w][sl] >= 3).any()
        diag = np.hypot(w, h)
        thick = dt[y:y+h, x:x+w][sl].mean() >= 9*s
        if thick and touches and max(w, h) < 260*s: continue     # short thick boundary jog
        stroke_thick = dt[y:y+h, x:x+w][sl].max() >= 6*s          # any part of it is a heavy boundary stroke
        if not thick and not stroke_thick and chains:
            yy0, xx0 = np.nonzero(sl)
            par = _on_chain(yy0 + y, xx0 + x, chains, s, frac=0.6)
            free = not _continues_both(sk1, sl, x, y, w, h, s)   # stops at an end -> dash/stub; a line that carries on both sides is a real boundary
            # parallel to a red dash-dot line AND ending freely -> leftover stub of that line
            if par and free and diag < 400*s: rm[i] = True; continue
            # short non-parallel tick / V mark sitting on a red dash-dot line
            if (30*s <= diag < 200*s and not _parallel_chain(yy0 + y, xx0 + x, chains, s)
                    and _touch_chain(yy0 + y, xx0 + x, chains, s)): rm[i] = True; continue
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
            if a >= 70*s and np.hypot(w, h) / a > 0.93:
                yy, xx = np.nonzero(le[y:y+h, x:x+w] == i)
                if not _dash_evidence(yy + y, xx + x, dash_pts, s) and not _on_chain(yy + y, xx + x, chains, s, frac=0.6): keepe[i] = True
        if keepe.any():
            ke = keepe[le]
            sk2 |= rcl & (cv2.dilate(ke.astype(np.uint8), np.ones((7,7), np.uint8)) > 0)
            log(f'  kept {int(keepe.sum())} straight edges inside junk clusters')
    log(f'  removed {int(rm.sum())} glued/stub clusters')
    sk3 = prune_spurs(sk2, dt, L=int(330*s), thin_w=14*s, iters=3, s=s, sure_L=int(45*s), chains=chains)       # tiny ticks left on lines
    # --- boundary rescue: a removed straight piece that bridges two kept lines is a boundary ---
    removed = sk & ~sk3
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
            if np.hypot(w, h) / a < 0.88: continue          # curved -> text, leave removed
            c = P.mean(0); v = np.linalg.svd(P - c, full_matrices=False)[2][0]
            t = (P - c) @ v
            e0 = P[t.argmin()].astype(int); e1 = P[t.argmax()].astype(int)
            d0 = dist[y+e0[1], x+e0[0]]; d1 = dist[y+e1[1], x+e1[0]]
            if d0 <= 10*s and d1 <= 10*s:
                ys_, xs_ = np.nonzero(sl)
                if (not _dash_evidence(ys_ + y, xs_ + x, dash_pts, s)
                        and not (30*s <= np.hypot(w, h) < 200*s and not _parallel_chain(ys_ + y, xs_ + x, chains, s)
                                 and _touch_chain(ys_ + y, xs_ + x, chains, s) and dt[y:y+h, x:x+w][sl].max() < 6*s)):
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
            if dt[yy + y, xx + x].max() >= 6*s: continue                         # heavy boundary stroke
            if _chain_hits(yy, xx, chains, s, tol=20, ox=x, oy=y, ang=8) < 0.9: continue
            if _continues_both(sk3, sl, x, y, w, h, s): continue                 # real line a dash was drawn over
            kill[i] = True
            break                                                                 # remove one, re-evaluate the rest
        if not kill.any(): break
        sk3 = sk3 & ~kill[le]
        nkill += 1
    if nkill:
        sk3 = prune_spurs(sk3, dt, L=int(45*s), thin_w=14*s, iters=2, s=s, sure_L=int(45*s))
        log(f'  removed {nkill} dash pieces lying on construction lines')
    out = reconstruct(net, sk3, dt)
    # thick outer-boundary strokes are never touched
    r = max(3, int(round(6.5*s)))
    thick = cv2.morphologyEx(net.astype(np.uint8), cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))) > 0
    nt, lt, stt, _ = cv2.connectedComponentsWithStats(thick.astype(np.uint8), connectivity=8)
    big_t = np.zeros(nt, bool)
    big_t[1:] = np.maximum(stt[1:, 2], stt[1:, 3]) > 120*s
    thick = big_t[lt]
    thick = (cv2.dilate(thick.astype(np.uint8), np.ones((5,5), np.uint8)) > 0) & net
    out = out | thick
    return bridge_gaps(out, s, log)


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

def bridge_gaps(mask, s, log=print, max_gap=75, ang_tol=14.0):
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
        if not good.any(): continue
        j = np.nonzero(good)[0][dist[good].argmin()]
        # the corridor between the ends must be empty ink (a real gap, not a different line)
        w = int(round(2*max(dt[tuple(ends[i])], dt[tuple(ends[j])]))) + 1
        cv2.line(out, tuple(int(round(t)) for t in P[i]), tuple(int(round(t)) for t in P[j]), 1, max(w, 2))
        used[i] = used[j] = True; drawn += 1
    log(f'  bridged {drawn} gaps')
    return out > 0