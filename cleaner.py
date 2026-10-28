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

BIG_PX = 150_000_000          # sheets above this are cleaned on a ~139 M px copy, then mapped back to full size
TARGET_PX = 139_000_000

def clean(im, log=print, debug=False, progress=None):
    """progress(percent 0-100, message) is called as the work advances."""
    if isinstance(im, str): im = Image.open(im)
    P = progress or (lambda p, m='': None)
    W0, H0 = im.size
    if W0*H0 <= BIG_PX:
        return _clean_core(im, log, debug, P)
    # --- very large sheet: clean a reduced copy (needs far less memory), then apply the decision to the full image ---
    f = (TARGET_PX / (W0*H0)) ** 0.5
    w1, h1 = int(round(W0*f)), int(round(H0*f))
    log(f'{W0}x{H0} is very large: cleaning a {w1}x{h1} working copy, result mapped back to full size')
    P(2, 'Preparing a working copy of the large sheet')
    small = im.convert('RGB').resize((w1, h1), Image.LANCZOS)
    extra = {}
    out = _clean_core(small, log, debug, lambda p, m='': P(4 + 0.86*p, m), extra)
    del small
    res_s, dbg = (out if debug else (out, None))
    P(91, 'Mapping the result back to full size')
    lum = _lum(im)
    keep = cv2.resize((res_s < 250).astype(np.uint8), (W0, H0), interpolation=cv2.INTER_LINEAR) > 0
    keep = cv2.dilate(keep.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    del res_s
    res = np.where(keep & (lum < 250), lum, 255).astype(np.uint8)
    del keep
    hl = cv2.resize(extra['healed'].astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST) > 0
    hl = cv2.dilate(hl.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    res[hl] = np.minimum(res[hl], 20)
    del hl, lum
    return (res, dbg) if debug else res

def _clean_core(im, log=print, debug=False, P=lambda p, m='': None, extra=None):
    cm = class_map(im)
    H, W = cm.shape
    s = max(H, W) / 9917.0
    S = lambda v: max(1, int(round(v*s)))
    log(f'{W}x{H}, scale {s:.2f}')
    P(3, 'Reading the sheet')

    lines = (cm == LINE).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(lines, connectivity=8, ltype=cv2.CV_16U)
    del lines
    frames = [i for i in range(1, n) if st[i, 2] >= 0.85*W and st[i, 3] >= 0.85*H]
    if not frames: frames = [int(np.argmax(st[1:, 4])) + 1]
    top, bot, zx0, zx1 = _find_zone(lab, st, frames, H, W, s, log)
    log(f'  work zone y {top}..{bot}')
    P(6, 'Finding the map area')

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
    # --- fragments of a boundary that a label was printed over -------------------------------------
    # A dimension such as "18.2" is plotted ON the boundary line and replaces its ink glyph by glyph, so
    # the line is left as a row of tiny black pieces squeezed between the strokes of the text. They look
    # exactly like the dots of a dash-dot line, but they have text on both sides. They must never count as
    # dash evidence (that is what made real dividers vanish), and they are put back into the line later.
    sliced = _text_sliced(cm, lab, small_free, s)
    log(f'  fragments squeezed between label strokes: {len(sliced)}')
    dash_free = [f for f in small_free if f[0] not in sliced]
    # --- dash / dot fragments: evidence that a stub belongs to a dashed line ---
    dp = []
    for (i, x, y, w, h, a) in dash_free:
        diag = float(np.hypot(w, h))
        if diag >= S(100) and a/diag <= S(16):
            ys_, xs_ = np.nonzero(lab[y:y+h, x:x+w] == i)          # direction of the dash: a dash only vouches for
            q = np.stack([xs_, ys_], 1).astype(np.float64)          # a line that runs the SAME way, never for one
            ux, uy = np.linalg.svd(q - q.mean(0), full_matrices=False)[2][0]   # that crosses it
            dp.append((x + w/2, y + h/2, diag/2, ux, uy))
        elif w <= S(26) and h <= S(26) and a >= S(20):
            dp.append((x + w/2, y + h/2, 0.0, 0.0, 0.0))
    dpts = np.array(dp, np.float64) if dp else np.zeros((0, 5))

    # --- isolated closed polygons (parcels not touching the network) are boundaries ---
    kept_poly = 0
    block_poly = []                                   # multi-cell parcel blocks: cleaned like the network
    for (i, x, y, w, h, a) in small_free:
        if min(w, h) < S(150) or max(w, h) > S(1500): continue      # was S(260): thin parcel pairs (e.g. 95/98 on sheet 7, 305 px tall) were thrown away
        sub = (lab[y:y+h, x:x+w] == i).astype(np.uint8)
        cnts, hier = cv2.findContours(sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # real cells only: the pinholes where a dash touches the outline are not plots
        sig = [cv2.contourArea(cnts[k]) for k in range(len(cnts))
               if hier[0][k][3] >= 0 and cv2.contourArea(cnts[k]) > max(0.02*w*h, S(60)**2)]
        if len(sig) == 1 and sig[0] > 0.35*w*h:          # one big hole = polygon, not an icon
            block_poly.append(i); kept_poly += 1
        elif len(sig) >= 1 and sum(sig) > 0.3*w*h:
            block_poly.append(i)                          # several cells / a block with attachments: clean like the network
        elif not sig:
            # an outline with a small gap (a label or a dash cut it): closed by a short bridge it encloses a parcel
            k_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*S(35)+1, 2*S(35)+1))
            pad_ = S(40)
            cl = cv2.morphologyEx(cv2.copyMakeBorder(sub, pad_, pad_, pad_, pad_, cv2.BORDER_CONSTANT, value=0), cv2.MORPH_CLOSE, k_)
            cc, hh = cv2.findContours(cl, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            hs = [cv2.contourArea(cc[k]) for k in range(len(cc)) if hh[0][k][3] >= 0]
            if hs and max(hs) > 0.35*w*h: block_poly.append(i)
    for i in block_poly:
        keep[i] = True; network.append(i)
    log(f'  isolated polygons kept: {kept_poly}, multi-cell blocks: {len(block_poly)}')
    # the work zone must reach below the lowest linework of the map: labels hanging under the last boundary
    # (20.4, 46.4 ...) belong to the map, not to the footer
    if network:
        nb = max(int(st[i, 1] + st[i, 3]) for i in network)
        fb = bot + S(700)
        nbot = int(min(max(bot, nb + S(160)), fb - S(250)))
        if nbot > bot:
            log(f'  work zone extended to y {nbot} (linework reaches y {nb})')
            bot = nbot

    block_poly_set = set(block_poly)
    m = keep[lab]                                      # kept line pixels (bool, full size)
    chains = _dash_chains(cm, lab, dash_free, top, bot, s, log)
    P(12, 'Looking for construction lines')
    # --- network cleaning ---
    _tot = float(sum(int(st[i, 2])*int(st[i, 3]) for i in network)) or 1.0
    _done = 0.0
    _NETF = (('skeleton px', .12, 'Tracing the linework'), ('arrow/stub strip', .40, 'Removing arrows and stubs'),
             ('after spur pruning', .55, 'Trimming spurs'), ('glued/stub clusters', .70, 'Removing dimension clutter'),
             ('rescued', .82, 'Rescuing boundary pieces'), ('bridged', .96, 'Closing boundary gaps'))
    for i in network:
        _wgt = st[i, 2]*st[i, 3]/_tot
        _b0 = 12 + 56*_done; _sp = 56*_wgt; _done += _wgt
        def nlog(msg, _b0=_b0, _sp=_sp):
            log(msg)
            for key, fr, label in _NETF:
                if key in msg: P(_b0 + _sp*fr, label); break
        x, y, w, h, a = st[i]
        pad = 10
        x0, y0 = max(x-pad, 0), max(y-pad, 0); x1, y1 = min(x+w+pad, W), min(y+h+pad, H)
        reg = (lab[y0:y1, x0:x1] == i)
        log(f'network component {i}: {w}x{h}')
        loc = [(c - np.array([x0, y0]), v, t0, t1) for c, v, t0, t1 in chains]   # chains in crop coordinates
        txt_crop = (cm[y0:y1, x0:x1] == BLUE) | (cm[y0:y1, x0:x1] == RED)
        cleaned = clean_network(reg, s, nlog, dash_pts=dpts - np.array([x0, y0, 0.0, 0.0, 0.0]), chains=loc, txt=txt_crop)
        if i in block_poly_set:
            # a free parcel block: its OUTER outline is the boundary. Whatever cleaning removed from that outline
            # (a dash line lying along an edge, a crossing arrow) is put back so the parcel stays closed.
            cleaned = _close_open_outline(cleaned | _outer_ring(reg), s)
        sub = m[y0:y1, x0:x1]
        sub[cv2.dilate(reg.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = False
        sub |= cleaned
        P(_b0 + _sp, 'Cleaning the network')
    slice_px = np.isin(lab, np.fromiter(sliced, np.int64)) if sliced else np.zeros(m.shape, bool)
    # heavy boundary strokes cut off from the network by something printed across them (a dash-dot line, a label):
    # long, much thicker than a dash. They are put back when they continue kept linework.
    heavy_ids = [i for (i, x, y, w, h, a) in small_free
                 if i not in sliced and np.hypot(w, h) >= S(100) and a/np.hypot(w, h) > S(18)]
    heavy_px = np.isin(lab, np.array(heavy_ids, np.int64)) if heavy_ids else np.zeros(m.shape, bool)
    del lab
    if heavy_px.any():
        nh, lh, sh_, _ = cv2.connectedComponentsWithStats(heavy_px.astype(np.uint8), connectivity=8)
        reach = S(45)
        backh = np.zeros(nh, bool)
        mk = m
        def _runs_on(px, py, v, sgn):
            """kept ink straight ahead of an end (px,py) along sgn*v: first contact within `reach`, then it must
            carry on along the same line (a boundary), not just cross it (a leader meeting a boundary)"""
            hit0 = None
            for d in range(2, reach + 1, 2):
                qx, qy = int(round(px + sgn*v[0]*d)), int(round(py + sgn*v[1]*d))
                if not (0 <= qy < H and 0 <= qx < W): return False
                if mk[max(qy-2, 0):qy+3, max(qx-2, 0):qx+3].any(): hit0 = d; break
            if hit0 is None: return False
            ok = n_ = 0
            for d in range(hit0, hit0 + S(60), 3):
                qx, qy = int(round(px + sgn*v[0]*d)), int(round(py + sgn*v[1]*d))
                if not (0 <= qy < H and 0 <= qx < W): break
                n_ += 1; ok += bool(mk[max(qy-3, 0):qy+4, max(qx-3, 0):qx+4].any())
            return n_ >= 5 and ok >= 0.75*n_
        for q in range(1, nh):
            qx, qy, qw, qh, qa = sh_[q]
            ys_, xs_ = np.nonzero(lh[qy:qy+qh, qx:qx+qw] == q)
            if len(ys_) < 20: continue
            Pq = np.stack([xs_ + qx, ys_ + qy], 1).astype(np.float64)
            cq = Pq.mean(0); vq = np.linalg.svd(Pq - cq, full_matrices=False)[2][0]
            tq = (Pq - cq) @ vq
            e0 = Pq[tq.argmin()]; e1 = Pq[tq.argmax()]
            if _runs_on(e0[0], e0[1], vq, -1.0) or _runs_on(e1[0], e1[1], vq, 1.0):
                backh[q] = True
        for q in np.nonzero(backh)[0]:
            qx, qy, qw, qh, _a = sh_[q]
            m[qy:qy+qh, qx:qx+qw] |= (lh[qy:qy+qh, qx:qx+qw] == q)
        log(f'  heavy boundary pieces cut off by crossing lines restored: {int(backh.sum())} of {nh-1}')
        P(72, 'Restoring cut boundary pieces')
        del lh
    del heavy_px
    # put label-sliced line pieces back - but only those that continue a kept line: grow outwards from the
    # kept linework, fragment by fragment (each one must start within ~one glyph stroke of the previous)
    if slice_px.any():
        nq, lq, sq, _ = cv2.connectedComponentsWithStats(slice_px.astype(np.uint8), connectivity=8)
        reach = S(14)
        back = np.zeros(nq, bool)
        for _round in range(14):
            grew = False
            for q in range(1, nq):
                if back[q]: continue
                qx, qy, qw, qh, _a = sq[q]
                x0_, y0_ = max(qx-reach, 0), max(qy-reach, 0)
                x1_, y1_ = min(qx+qw+reach, W), min(qy+qh+reach, H)
                near_ = m[y0_:y1_, x0_:x1_]
                piece = (lq[y0_:y1_, x0_:x1_] == q)
                if not near_.any(): continue
                if (cv2.dilate(near_.astype(np.uint8), np.ones((2*reach+1, 2*reach+1), np.uint8)) > 0)[piece].any():
                    back[q] = True; m[y0_:y1_, x0_:x1_] |= piece; grew = True
            if not grew: break
        log(f'  label-sliced line pieces restored: {int(back.sum())} of {nq-1}')
        del lq
    del slice_px

    # --- heal line pixels that were covered by labels drawn on top of the linework ---
    P(76, 'Restoring labels sitting on lines')
    healed = _heal_under_labels(m, cm, s, log)
    P(82, 'Joining lines hidden under labels')
    m |= healed
    j = _join_under_labels(m, cm, s, log)
    m |= j
    healed |= j
    P(88, 'Finishing lines')
    # --- stray dimension labels in the footer band: keep only real footer text lines / QR ---
    foot = _footer_keep(cm, bot, H, W, zx1, s)
    # --- plot numbers from the blue text ---
    P(90, 'Detecting plot numbers')
    kept_blue = _plot_numbers(cm, s, log)
    P(96, 'Writing the cleaned image')
    if extra is not None: extra['healed'] = healed
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


def _close_open_outline(cl, s):
    """cl: cleaned ink of one free parcel block. An outline left open (a corner hidden under labels) is closed:
    two free ends whose straight continuations meet at a corner, or face each other, are joined."""
    from skimage.morphology import skeletonize
    S = lambda v: max(1, int(round(v*s)))
    if not cl.any(): return cl
    sk = skeletonize(cl)
    sk8 = sk.astype(np.uint8)
    nb = cv2.filter2D(sk8, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk8
    ends = np.argwhere(sk & (nb == 1))
    if len(ends) < 2: return cl
    nl, ll = cv2.connectedComponents(sk8, connectivity=8)
    dtm = cv2.distanceTransform(cl.astype(np.uint8), cv2.DIST_L2, 3)
    wid = max(3, int(round(2*float(np.median(dtm[sk])))))
    info = []
    for (ey, ex) in ends:
        yy, xx = np.nonzero(ll == ll[ey, ex])
        P = np.stack([xx, yy], 1).astype(np.float64)
        if len(P) < S(90): continue                              # tiny stubs are not outline
        dd = np.hypot(P[:, 0]-ex, P[:, 1]-ey)
        q = P[dd <= S(80)]
        if len(q) < 10: continue
        cq = q.mean(0); u = np.linalg.svd(q - cq, full_matrices=False)[2][0]
        if (np.array([ex, ey]) - cq) @ u < 0: u = -u
        info.append((np.array([ex, ey], np.float64), u))
    out = cl.copy(); canvas = np.zeros(cl.shape, np.uint8)
    Dmax = S(450); used = set()
    for a_ in range(len(info)):
        for b_ in range(a_+1, len(info)):
            if a_ in used or b_ in used: continue
            e1, u1 = info[a_]; e2, u2 = info[b_]
            if np.hypot(*(e1 - e2)) < S(12): continue
            cosang = abs(u1 @ u2)
            if cosang < 0.4:                                    # a corner
                A = np.array([[u1[0], -u2[0]], [u1[1], -u2[1]]])
                try: t1, t2 = np.linalg.solve(A, e2 - e1)
                except np.linalg.LinAlgError: continue
                if not (0 < t1 < Dmax and 0 < t2 < Dmax): continue
                Pc = e1 + u1*t1
                cv2.line(canvas, tuple(int(round(v)) for v in e1), tuple(int(round(v)) for v in Pc), 1, wid)
                cv2.line(canvas, tuple(int(round(v)) for v in e2), tuple(int(round(v)) for v in Pc), 1, wid)
                used.add(a_); used.add(b_)
            elif cosang > 0.97:                                 # facing each other along one line
                d = e2 - e1; dl = np.hypot(*d)
                if dl > Dmax or (d @ u1) / dl < 0.95 or (-d @ u2) / dl < 0.95: continue
                cv2.line(canvas, tuple(int(round(v)) for v in e1), tuple(int(round(v)) for v in e2), 1, wid)
                used.add(a_); used.add(b_)
    return out | (canvas > 0)

def _outer_ring(reg):
    """ink of `reg` forming the outer outline of the shape, WITHOUT thin things sticking out of it (arrows, ticks):
    the filled silhouette is opened with a kernel wider than any stroke, so protrusions drop out of it."""
    if not reg.any(): return reg
    r8 = reg.astype(np.uint8)
    pad = cv2.copyMakeBorder(r8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    n, lab, _, _ = cv2.connectedComponentsWithStats((pad == 0).astype(np.uint8), connectivity=4)
    filled = ~(lab == lab[0, 0])[1:-1, 1:-1]                      # ink + enclosed cells
    dt = cv2.distanceTransform(r8, cv2.DIST_L2, 3)
    wmax = 2*float(np.percentile(dt[reg], 90)) + 2
    k = int(round(5*wmax)) | 1
    opened = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    if not opened.any(): return np.zeros_like(reg)
    dist = cv2.distanceTransform(opened.astype(np.uint8), cv2.DIST_L2, 3)       # depth inside the silhouette
    return reg & opened & (dist <= wmax)

def _text_sliced(cm, lab, small_free, s):
    """ids of small free line fragments that are squeezed between the strokes of printed text
    (text on opposite sides): dots with >= 20% text around them, dashes with >= 35%."""
    S = lambda v: max(1, int(round(v*s)))
    H, W = cm.shape
    out = set()
    for (i, x, y, w, h, a) in small_free:
        diag = float(np.hypot(w, h))
        if diag >= S(100) and a/diag <= S(16): need = 0.35
        elif w <= S(26) and h <= S(26) and a >= S(20): need = 0.20
        else: continue
        p = S(8)
        x0, y0, x1, y1 = max(x-p, 0), max(y-p, 0), min(x+w+p, W), min(y+h+p, H)
        f = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
        k = 2*S(5) + 1
        ring = (cv2.dilate(f, np.ones((k, k), np.uint8)) > 0) & (f == 0)
        c = cm[y0:y1, x0:x1]
        txt = (c == BLUE) | (c == RED)
        if ring.sum() and (ring & txt).sum() / ring.sum() >= need: out.add(i)
    return out

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

def _join_under_labels(m, cm, s, log):
    """A line (or a plot corner) hidden under a run of labels is longer than one label, so the per-label heal
    cannot close it. Look at the free ends of the kept linework that stop at printed text and
      (a) carry on straight into other ink on the far side of the text, or
      (b) meet a second free end at a right-angled corner under the text,
    and draw the missing stretch."""
    from skimage.morphology import skeletonize
    H, W = m.shape
    txt = ((cm == RED) | (cm == BLUE)).astype(np.uint8)
    S = lambda v: max(1, int(round(v*s)))
    # dash segments of a dimension line are not text: a join must not run along them
    nt, lt, stt, _ = cv2.connectedComponentsWithStats(txt, connectivity=8, ltype=cv2.CV_16U)
    ww, hh = stt[:, 2], stt[:, 3]
    dashy = (np.maximum(ww, hh) > 2.6*np.maximum(1, np.minimum(ww, hh))) & (np.maximum(ww, hh) > S(25))
    dashy[0] = False
    txt[dashy[lt]] = 0
    del lt
    big = cv2.dilate(txt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*S(30)+1, 2*S(30)+1)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(big, connectivity=8, ltype=cv2.CV_16U)
    near = cv2.dilate(m.astype(np.uint8), np.ones((2*S(25)+1, 2*S(25)+1), np.uint8)) > 0
    ids = np.unique(lab[near & (big > 0)]); ids = ids[ids > 0]
    out = np.zeros_like(m)
    Dmax = S(420); drawn = 0
    for i in ids:
        x, y, w, h, a = st[i]
        if max(w, h) > S(700): continue
        pad = Dmax // 2 + S(60)
        x0, y0, x1, y1 = max(x-pad, 0), max(y-pad, 0), min(x+w+pad, W), min(y+h+pad, H)
        ms = m[y0:y1, x0:x1]
        if not ms.any(): continue
        tx = txt[y0:y1, x0:x1] > 0
        tx_d = cv2.dilate(tx.astype(np.uint8), np.ones((2*S(8)+1, 2*S(8)+1), np.uint8)) > 0
        tx_w = cv2.dilate(tx.astype(np.uint8), np.ones((2*S(22)+1, 2*S(22)+1), np.uint8)) > 0
        sk = skeletonize(ms)
        sk8 = sk.astype(np.uint8)
        nb = cv2.filter2D(sk8, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk8
        ends = np.argwhere(sk & (nb == 1))
        if len(ends) == 0: continue
        nl, ll = cv2.connectedComponents(sk8, connectivity=8)
        dtm = cv2.distanceTransform(ms.astype(np.uint8), cv2.DIST_L2, 3)
        wid = max(3, int(round(2*float(np.median(dtm[sk])))))
        info = []
        for (ey, ex) in ends:
            # outward direction from the last stretch of the branch
            yy, xx = np.nonzero(ll == ll[ey, ex])
            P = np.stack([xx, yy], 1).astype(np.float64)
            if len(P) < S(150): continue                  # a short piece (a dash, a stub) is not a boundary running under text
            dd = np.hypot(P[:, 0]-ex, P[:, 1]-ey)
            q = P[dd <= S(70)]
            if len(q) < 8: continue
            cq = q.mean(0); _, sv_, vt_ = np.linalg.svd(q - cq, full_matrices=False); u = vt_[0]
            if sv_[1] > 0.12*sv_[0] or len(q) < 0.6*S(70): continue    # the last stretch must be a straight line, not the curve of a digit
            if (np.array([ex, ey]) - cq) @ u < 0: u = -u
            # the end has to stop at text
            px_, py_ = int(ex), int(ey)
            r_ = S(60)
            if not tx_d[max(py_-r_, 0):py_+r_+1, max(px_-r_, 0):px_+r_+1].any(): continue
            info.append((np.array([ex, ey], np.float64), u, ll[ey, ex]))
        lines = []
        used = set()
        # (a) straight on into other ink through the text
        for k, (e, u, l_) in enumerate(info):
            hit = None
            for d in range(S(14), Dmax, 2):
                q = e + u*d; qx, qy = int(round(q[0])), int(round(q[1]))
                if not (0 <= qy < sk.shape[0] and 0 <= qx < sk.shape[1]): break
                win = sk[max(qy-2, 0):qy+3, max(qx-2, 0):qx+3]
                if win.any():
                    lw = ll[max(qy-2, 0):qy+3, max(qx-2, 0):qx+3][win]
                    if (lw != l_).any(): hit = d; break
                    elif d > S(60): break
            if hit is None: continue
            ds = np.arange(0, hit, 2)
            ptx = np.array([[int(round((e + u*d)[0])), int(round((e + u*d)[1]))] for d in ds])
            ok = np.mean([tx_d[min(max(p[1], 0), tx_d.shape[0]-1), min(max(p[0], 0), tx_d.shape[1]-1)] for p in ptx])
            if ok >= 0.5:
                lines.append((e, e + u*hit)); used.add(k)
        if lines:
            canvas = np.zeros(ms.shape, np.uint8)
            for p, q in lines:
                cv2.line(canvas, (int(round(p[0])), int(round(p[1]))), (int(round(q[0])), int(round(q[1]))), 1, wid)
            out[y0:y1, x0:x1] |= canvas > 0
            drawn += len(lines)
    log(f'  joined {drawn} line stretches hidden under runs of labels')
    return out

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
        # the line being healed must run straight through the label: only close along the direction of the
        # ink that touches the label (a digit between two parallel lines or beside a divider is not a gap in a line)
        ring = cv2.dilate(lm.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        nc_, lc_, sc_, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        touch = np.unique(lc_[ring & (sub > 0)]); touch = touch[touch > 0]
        dirs = []
        for q in touch:
            if sc_[q, 4] < 40*s: continue
            ys_, xs_ = np.nonzero(lc_ == q)
            sel = np.arange(len(xs_))[::max(1, len(xs_)//1500)]
            Pq = np.stack([xs_[sel], ys_[sel]], 1).astype(np.float64)
            near_l = Pq[np.hypot(Pq[:, 0] - (lm.shape[1]/2), Pq[:, 1] - (lm.shape[0]/2)) <= 2.2*max(w, h) + 20*s]
            if len(near_l) < 12: continue
            sv, vt = np.linalg.svd(near_l - near_l.mean(0), full_matrices=False)[1:]
            if sv[1] > 0.45*sv[0]: continue
            dirs.append(np.degrees(np.arctan2(vt[0][1], vt[0][0])) % 180)
        if not dirs: continue
        for k, ang in zip(kernels, angles):
            if min(min(abs(ang - d_), 180 - abs(ang - d_)) for d_ in dirs) > 18: continue
            acc |= (cv2.morphologyEx(sub, cv2.MORPH_CLOSE, k) > 0) & lm & (sub == 0)
        if acc.any():
            out[y0:y1, x0:x1] |= acc; cnt += 1
    log(f'  healed {cnt} labels sitting on lines')
    return out

def _plot_numbers(cm, s, log):
    """Plot names are usually printed larger than the dimension digits, but some are printed in the SAME size
    (survey 61: 2, 5, 4A are large; 1A, 1B, 4B, 7 are dimension-sized). So take both: the larger-than-dimension
    strings, and the same-size strings that are upright, horizontal and have no decimal point."""
    kept = _plot_numbers_pass(cm, s, log, 1.10)
    log('  also looking for plot names printed in the dimension size')
    return kept | _plot_numbers_pass(cm, s, log, 0.85)

def _text_unit(sz, a, s):
    """Height in px of a typical dimension digit on THIS sheet = the most common size among blue glyph-like
    components. Font size differs from sheet to sheet (51 px on survey 3 / 6, 34 px on survey 8) so nothing
    about the text may be hard-coded in pixels."""
    ok = (a >= 0.1*sz*sz) & (sz >= 15*s) & (sz <= 200*s)
    if not ok.any(): return 50.0*s
    hist = np.bincount(sz[ok].astype(int))
    sm = np.convolve(hist, np.ones(3), 'same')
    m = int(np.argmax(sm))
    lo, hi = max(0, m-2), m+3
    v = np.arange(len(hist))[lo:hi]; wgt = hist[lo:hi]
    return float((v*wgt).sum() / max(1, wgt.sum()))

def _plot_numbers_pass(cm, s, log, thr_f):
    blue = (cm == BLUE).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(blue, connectivity=8, ltype=cv2.CV_16U)
    del blue
    x, y, w, h, a = (st[1:, k] for k in range(5))
    sz = np.maximum(w, h)
    H1 = _text_unit(sz, a, s)
    ts = H1 / 36.0                                  # the old constants were tuned at H1 = 51 px  (ts = 1.41 there)
    S = lambda v: max(1, int(round(v*ts)))
    glyph = (a >= S(30)**2 // 6) & (sz >= S(30))
    dots = (sz <= S(20)) & (a >= S(15)) & ~glyph
    gs = np.nonzero(glyph)[0]
    thr = H1 * thr_f
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
        # a decimal point sits on the baseline, beside the digits - not above/below them (that is another string's dot)
        onbase = (dy >= y[gs] + 0.45*h[gs]) & (dy <= y[gs] + 1.3*h[gs])
        rowok = np.abs(cy - dy) <= 0.8*sz[gs]                       # same text row (slanted strings drift a little)
        cand = np.nonzero((dist <= 0.8*H1) & onbase & rowok)[0]
        # ... and it has a digit on BOTH sides (52.0); a dot with a neighbour on one side only belongs to something else
        if not ((cx[cand] < dx).any() and (cx[cand] > dx).any()): continue
        for j in cand:
            kj = gs[j]
            if w[kj] > 1.15*h[kj] and h[kj] >= thr: continue     # plot name fused with a dimension: the dot is the dimension's
            hasdot.add(root[j])
    # parentheses: a paren-like glyph with neighbours on ONE side belongs to a dimension "(217.6)";
    # with neighbours on both sides it is the "/" of a plot name such as 3E/1
    def _parenlike(k):
        ar_ = max(w[k], h[k]) / max(1, min(w[k], h[k])); fl_ = a[k] / (w[k]*h[k])
        return (ar_ > 2.4 and fl_ < 0.5) or (ar_ > 1.85 and fl_ < 0.30)      # slash / bracket, incl. a rotated ')'
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
        if sz[mem].min() < thr or np.minimum(w[mem], h[mem]).max() > 1.4*H1 or sz[mem].max() > 3.2*H1: continue
        ar = np.maximum(w[mem], h[mem]) / np.maximum(1, np.minimum(w[mem], h[mem]))
        fill = a[mem] / (w[mem]*h[mem])
        if (ar > 3.2).any() or (fill < 0.12).any(): continue         # thin dashes
        if ((fill >= 0.58) & (np.minimum(w[mem], h[mem]) > 0.9*H1)).any(): continue    # solid / hatched blob = building symbol, not a letter
        if len(mem) > 8: continue
        # a box outline (building / hut symbol) is mostly enclosed empty space; no letter or digit is
        hole_big = False
        for k in mem:
            x0_, y0_, w_, h_ = int(x[k]), int(y[k]), int(w[k]), int(h[k])
            sub_ = (lab[y0_:y0_+h_, x0_:x0_+w_] == k + 1).astype(np.uint8)
            cn_, hr_ = cv2.findContours(sub_, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            hole_a = sum(cv2.contourArea(cn_[q]) for q in range(len(cn_)) if hr_[0][q][3] >= 0)
            if hole_a >= 0.5*w_*h_ and min(w_, h_) > 0.9*H1: hole_big = True; break
        if hole_big: continue
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
            if abs(np.degrees(np.arctan2(ax[1], ax[0]))) % 180 > 14 and abs(np.degrees(np.arctan2(ax[1], ax[0]))) % 180 < 166: continue
        elif w[mem[0]] > 1.15*h[mem[0]] and h[mem[0]] < thr: continue   # wide blob taller than any dimension digit = plot name touching a dimension
        else:
            # a lone glyph whose looser neighbours (slightly different size because of rotation) run diagonally
            # is part of a slanted dimension such as (145.4), not a plot name
            k0 = mem[0]
            gxn = np.maximum(0, np.maximum(x[gs] - (x[k0] + w[k0]), x[k0] - (x[gs] + w[gs])))
            gyn = np.maximum(0, np.maximum(y[gs] - (y[k0] + h[k0]), y[k0] - (y[gs] + h[gs])))
            nbr = np.nonzero((np.hypot(gxn, gyn) <= 0.45*H1) & (gs != k0) & (sz[gs] >= 0.6*sz[k0]) & (sz[gs] <= 1.6*sz[k0]))[0]
            slant = False
            for jn in nbr:
                ddx = abs(cx[jn] - (x[k0] + w[k0]/2)); ddy = abs(cy[jn] - (y[k0] + h[k0]/2))
                if ddy > 0.5*max(ddx, 1): slant = True
            if slant and not _parenlike(k0): continue
        # square symbol (hatched building): a LONE, nearly square, filled blob - never a letter inside a string like 5A / 4A / 8A
        if len(mem) == 1 and 0.85 <= w[mem[0]]/h[mem[0]] <= 1.25 and fill[0] > 0.45: continue
        # a lone thin curved fragment (end of a rotated "(156.4)" dimension) is not a digit
        if len(mem) == 1 and fill[0] < 0.27: continue
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
        if thr_f < 1.0:
            # loose pass only: a name printed at dimension size stands in clear space. A string inside a dense knot of
            # overlapping dimension text (several numbers on top of each other) is clutter, not a plot name.
            d_ = np.hypot(cx - (x[mem] + w[mem]/2).mean(), cy - (y[mem] + h[mem]/2).mean())
            if int(((d_ <= 2.0*H1) & ~np.isin(gs, mem)).sum()) >= 7: continue
        keep_ids.extend((mem + 1).tolist())
    kept_roots = {root[np.nonzero(gs == (k - 1))[0][0]] for k in keep_ids if (k - 1) in set(gs.tolist())}
    for ii, nb in slashes:
        if any(root[j] in kept_roots for j in nb):
            keep_ids.append(int(gs[ii]) + 1)
            # the rest of the name around the slash (4/D): neighbours of plot-name size belong to it too
            for j in nb:
                if sz[gs[j]] >= thr and sz[gs[j]] <= 2.0*H1 and not _parenlike(gs[j]) and abs(cy[j] - cy[ii]) <= 0.3*sz[gs[ii]] and (gs[j] + 1) not in keep_ids:
                    keep_ids.append(int(gs[j]) + 1)
    log(f'  plot-number glyphs kept: {len(keep_ids)}')
    if not keep_ids: return np.zeros(cm.shape, bool)
    lut = np.zeros(n, bool); lut[keep_ids] = True
    return lut[lab]

if __name__ == '__main__':
    import sys
    r = clean(sys.argv[1])
    Image.fromarray(r).save(sys.argv[2])