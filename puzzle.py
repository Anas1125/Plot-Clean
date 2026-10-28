"""Automatic puzzle for approved survey sheets.
Every approved sheet is a cleaned survey-number sketch.  Neighbouring survey numbers share a
stretch of their thick outer boundary, so the sheets can be glued together by finding where
those boundaries coincide:
 1. extract_sheet()  - read header (survey no, scale 1:N, village), take the thick outline,
                       skeletonise it, make a filled silhouette, remember ground scale.
 2. coarse search    - FFT cross-correlation of outline vs outline for every pair of sheets of a
                       village, at a small range of rotations (sheets are rarely perfectly north-up).
                       Silhouettes that overlap are penalised, so sheets sit *next to* each other.
 3. verify / refine  - ICP on the outline points, then accept only pairs with a long shared border
                       and almost no overlapping area.
 4. assemble         - best-border-first spanning forest, one composite per connected group.
 5. render           - warp the full-resolution cleaned sheets onto one canvas.
Ground scale: all sheets are assumed to be the same paper size, so "ground units per pixel" is
N / (long side in px).  Sheets are therefore brought to a common scale from the printed 1:N.
"""
import io
import math
import os
import pickle
import re
import cv2
import numpy as np
from PIL import Image
from scipy import fft as sfft
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
Image.MAX_IMAGE_PIXELS = None
AN_LONG = 1800                    # analysis resolution (long side, px)
BASE_ANGLES = np.arange(-12.5, 12.6, 2.5)
# Survey sheets can be scanned in portrait or landscape. Keep normal fine-angle search,
# and also search near quarter turns for mixed-orientation pairs.
ANGLES = np.concatenate((BASE_ANGLES, 90.0 + BASE_ANGLES, 270.0 + BASE_ANGLES))
OUT_MAX = 8000                    # longest side of a rendered composite (px)
MIN_OUTLINE_PX = 600   # fewer outline pixels than this -> retry with the sheet's own line widths
FEATURE_VERSION = 14  # angle-balanced peak search and strong-border corner fallback
_OCR_LAST_ERROR = None
# acceptance thresholds for "these two sheets really touch"
MIN_LEN_FRAC = 0.06               # shared border / perimeter of the smaller sheet
MIN_LEN_EXT = 0.10                # shared border / extent of the smaller sheet
MAX_OVERLAP = 0.06                # overlapping silhouette area / smaller silhouette
HYPOTHESES_PER_PAIR = 6
MAX_NORMAL_ANGLE = 12.5
DEEP_LIMIT = 0.25
HULL_OVERLAP_LIMIT = 0.22      # footprint (convex hull) overlap allowed when fitting a sheet to a group
GROW_MODE = False                 # True while fitting a loose sheet to a whole group (slightly looser overlap test)
GROW_MAX_OVERLAP = 0.15
WIDE_ANGLES = np.arange(-45.0, 45.1, 2.5)   # used when fitting a loose sheet against a whole stitched group
ALLOW_QUARTER_TURNS = False       # sheets of one village share an orientation; set True for mixed-rotation data          # candidate placements verified per pair of sheets
ROT_SIGMA = 30.0                  # soft prior: sheets of one village are drawn at similar orientation
TRI_TOL_ROT = 4.0                 # degrees: two routes to the same sheet must agree within this ...
TRI_TOL_POS = 0.08                # ... and within this fraction of the sheet size
# --------------------------------------------------------------------------------------
# 1. per-sheet features
# --------------------------------------------------------------------------------------
def _read_gray(path):
    im = Image.open(path)
    im.load()
    if im.mode != 'L':
        im = im.convert('L')
    return np.array(im)


def _resize_long(g, long_side, interp=None):
    h, w = g.shape
    s = long_side / float(max(h, w))
    if abs(s - 1) < 1e-6:
        return g.copy(), 1.0
    if interp is None:
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(g, (int(round(w * s)), int(round(h * s))), interpolation=interp), s


def _find_zone(ink):
    """Drawing zone: inside the frame, below the header rule, above the footer."""
    H, W = ink.shape
    s = min(H, W) / 9917.0
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    frames = [i for i in range(1, n) if st[i, 2] >= 0.85 * W and st[i, 3] >= 0.85 * H]
    if frames:
        inner = min(frames, key=lambda i: st[i, 2] * st[i, 3])
        x, y, w, h, _ = st[inner]
        crop = (lab[y:y + h, x:x + w] == inner)
        ys = np.nonzero(crop.sum(1) >= 0.9 * w)[0]
        groups = []
        for r in ys:
            if groups and r - groups[-1][1] <= 3:
                groups[-1][1] = r
            else:
                groups.append([r, r])
        top = None
        for g0, g1 in groups:
            if 150 * s < g0 < 0.2 * H:
                top = y + g1 + 6
                break
        if top is None:
            top = int(y + 0.105 * min(H, W))
        bot = y + h - int(700 * s)
        return top, bot, x + 6, x + w - 6
    return int(0.0775 * H), int(0.933 * H), int(0.03 * W), int(0.97 * W)


def _ocr(gray_full, box, psm=6, threshold=False):
    """OCR a header crop and preserve actionable diagnostics if Tesseract is unavailable."""
    global _OCR_LAST_ERROR
    try:
        import pytesseract
    except Exception as exc:
        _OCR_LAST_ERROR = "Python OCR package missing (pytesseract). Install it with: pip install pytesseract."
        print("  Header OCR unavailable:", _OCR_LAST_ERROR, exc)
        return ""
    H, W = gray_full.shape
    x0, y0, x1, y1 = box
    crop = gray_full[max(0, int(y0 * H)):min(H, int(y1 * H)),
                     max(0, int(x0 * W)):min(W, int(x1 * W))]
    if crop.size == 0:
        return ""
    # Enlarge small header characters without downscaling large crops.
    target_h = 900
    factor = min(4.0, max(1.0, target_h / max(1, crop.shape[0])))
    if crop.shape[0] * factor > 1200:
        factor = 1200.0 / crop.shape[0]
    crop = cv2.resize(crop, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA)
    if threshold:
        crop = cv2.GaussianBlur(crop, (3, 3), 0)
        _, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        # PSM 6 works better on the multi-row right header than the old PSM 7 default.
        return pytesseract.image_to_string(crop, config=f'--oem 3 --psm {int(psm)}')
    except Exception as exc:
        _OCR_LAST_ERROR = (
            "Tesseract OCR engine could not run. On macOS install it with `brew install tesseract`, "
            "then install the Python wrapper with `pip install pytesseract`."
        )
        print("  Header OCR unavailable:", _OCR_LAST_ERROR, repr(exc))
        return ""


def _scale_candidate(text):
    """Return a plausible 1:N denominator from OCR text, correcting common OCR digit confusions."""
    cleaned = text.replace("：", ":").replace("；", ";")
    patterns = (
        r'\bsca[l1iI]e\b\s*[:;.,]?\s*1\s*[:;.,\-]\s*([0-9OQDIilLSBZG]{2,6})',
        r'\bsca[l1iI]e\b.{0,24}?\b1\b.{0,8}?([0-9OQDIilLSBZG]{2,6})',
    )
    translate = str.maketrans({
        "O": "0", "o": "0", "Q": "0",
        "I": "1", "i": "1", "l": "1", "|": "1",
        "S": "5", "s": "5",
        "B": "8",
        "Z": "2", "z": "2",
        "G": "6", "g": "9",
    })
    for pattern in patterns:
        for match in re.finditer(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL):
            raw = match.group(1).translate(translate)
            if not raw.isdigit():
                continue
            try:
                value = int(raw)
            except ValueError:
                continue
            if 50 <= value <= 200000:
                return value
    return None


def parse_header(text_right, text_left=''):
    """Parse header fields from one or more OCR crops."""
    meta = {}
    combined = "\n".join((text_right or "", text_left or ""))
    scale = _scale_candidate(combined)
    if scale is not None:
        meta["scale"] = scale
    survey_patterns = (
        r'\bSurvey\s*(?:No|N[o0]?|N0)\s*[:;.]?\s*([0-9]+[A-Za-z0-9/-]*)',
        r'\bSurvey\s*(?:No|N[o0]?|N0)\b[^\d]{0,10}(\d{1,8}[A-Za-z0-9/-]*)',
    )
    for pattern in survey_patterns:
        match = re.search(pattern, combined, flags=re.IGNORECASE)
        if match:
            meta["survey"] = match.group(1).strip(" /-")
            break
    area_match = re.search(r'\bHect?\s*(\d+)\s*Ares?\s*([\d.]+)', combined, flags=re.IGNORECASE)
    if area_match:
        try:
            meta["area_m2"] = float(area_match.group(1)) * 10000 + float(area_match.group(2)) * 100
        except ValueError:
            pass
    for key in ("district", "taluk", "village"):
        match = re.search(
            rf'\b{key}\s*[:;.]?\s*([^\r\n]+)',
            text_left or combined,
            flags=re.IGNORECASE
        )
        if match:
            value = match.group(1).strip()
            # Keep the village's numeric map-sheet code, e.g. "Manoor [110]".
            value = re.sub(r'\s{2,}', ' ', value)
            value = re.sub(r'[|*_]+$', '', value).strip()
            if value:
                meta[key] = value
    return meta


def village_key(meta):
    """'Manoor [110] <a', 'Manoor [110] iS' ... -> 'manoor110': first word + the village code, OCR junk ignored."""
    village = (meta.get("village") or "").strip().lower()
    village = re.sub(r'\b(village|vill)\b\s*[:;.]?', '', village)
    word = re.search(r'[a-z]{3,}', village)
    num = re.search(r'\d{2,}', village)
    return (word.group(0) if word else '') + (num.group(0) if num else '')


def read_header(gray_full):
    """Read survey number, village and 1:N scale from overlapping header crops."""
    global _OCR_LAST_ERROR
    _OCR_LAST_ERROR = None
    # Coordinates are fractions of the full sheet: right side holds survey number/scale,
    # left side holds district/taluk/village. Repeated crops make this robust to page orientation.
    crops = [
        ((0.54, 0.025, 0.995, 0.125), 6, True),   # right header, enhanced
        ((0.50, 0.000, 0.995, 0.150), 6, False), # right header, full multi-row text
        ((0.50, 0.000, 0.995, 0.150), 11, True), # sparse-text fallback
        ((0.000, 0.000, 0.50, 0.150), 6, True),  # district/taluk/village block
        ((0.000, 0.000, 1.000, 0.150), 11, False),
    ]
    parsed_rows = []
    for box, psm, threshold in crops:
        text = _ocr(gray_full, box, psm=psm, threshold=threshold)
        # On the left crop, parse village fields from the same crop; on right crops, the text
        # also contains survey/scale. parse_header will simply leave absent values empty.
        parsed_rows.append(parse_header(text, text))
        got = {k for row in parsed_rows for k, v in row.items() if v not in (None, '')}
        if {'survey', 'village', 'scale'} <= got and any(int(r.get('scale', 0) or 0) >= 100 for r in parsed_rows):
            break                               # header fully read: no need for the slower fallback crops
    result = {}
    # Merge fields with the most frequent OCR value. For scale only, prefer plausible long
    # map denominators (e.g. 808/936) over common truncated 2-digit false reads such as 57.
    for key in ('survey', 'village', 'district', 'taluk', 'area_m2', 'scale'):
        values = [row[key] for row in parsed_rows if row.get(key) not in (None, '')]
        if not values:
            continue
        if key == 'scale':
            longer = [v for v in values if int(v) >= 100]
            if longer:
                values = longer
        counts = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        result[key] = max(counts, key=lambda value: (counts[value], values.index(value)))
    if _OCR_LAST_ERROR:
        result['ocr_error'] = _OCR_LAST_ERROR
    if 'scale' not in result:
        print('  Header OCR did not find Scale 1:N.')
    else:
        print(f"  header detected: survey={result.get('survey', '?')}, "
              f"village={result.get('village', '?')}, scale=1:{result['scale']}")
    return result


def detect_stones(ink):
    """Boundary stones: small hollow squares drawn on the survey boundary.  Neighbouring sheets show
    the same stone, so one coincidence pins the join exactly (a straight border alone cannot)."""
    f = ink.astype(np.float32)
    found = []
    for size in range(7, 15):
        t = 2
        ring = np.zeros((size, size), np.float32)
        ring[:t, :] = 1
        ring[-t:, :] = 1
        ring[:, :t] = 1
        ring[:, -t:] = 1
        inner = np.zeros((size, size), np.float32)
        inner[t + 1:-t - 1, t + 1:-t - 1] = 1
        if inner.sum() == 0:
            continue
        rf = cv2.filter2D(f, -1, ring / ring.sum(), borderType=cv2.BORDER_CONSTANT)
        inf = cv2.filter2D(f, -1, inner / inner.sum(), borderType=cv2.BORDER_CONSTANT)
        ys, xs = np.nonzero((rf >= 0.80) & (inf <= 0.06))
        for y, x in zip(ys, xs):
            found.append((float(rf[y, x]), int(x), int(y)))
    pts = []
    for sc, x, y in sorted(found, reverse=True):
        if all((x - a) ** 2 + (y - b) ** 2 > 100 for a, b in pts):
            pts.append((x, y))
    return pts[:12]


def _thick_threshold(d2):
    """Half-width above which a line counts as the survey boundary.  Boundary lines are the WIDEST
    line class that is long enough to matter (thin parcel lines and text are narrower)."""
    ridge = d2[(d2 >= cv2.dilate(d2, np.ones((3, 3), np.float32)) - 1e-6) & (d2 >= 1.5)]
    if ridge.size < 100:
        return 6.0
    hist = np.bincount(np.minimum((ridge * 2).astype(int), 80)).astype(float)
    tot = hist[3:].sum()
    w_max = None
    for i in range(len(hist) - 2, 3, -1):                 # widest class with at least 1.5% of all line pixels
        if hist[i - 1:i + 2].sum() >= 0.015 * tot:
            w_max = i / 2.0
            break
    med = float(np.median(ridge))
    if w_max is None:
        return max(2.5, 2.0 * med)
    return max(w_max - 1.5, 1.2 * med, 2.5)


def extract_sheet(path, name=None, meta=None):
    """Features of one cleaned sheet (see module docstring).  `meta` overrides OCR values."""
    full = _read_gray(path)
    # OCR the header every time the cache version changes. Explicit metadata from the app
    # takes precedence: previously entered scales may predate the scale_source flag.
    need = ('scale', 'survey', 'village')
    have_meta = bool(meta) and all(meta.get(k) not in (None, '') for k in need)
    ocr_meta = {} if have_meta else read_header(full)
    if meta:
        for key, value in meta.items():
            if value not in (None, '') and key != 'scale_source':
                ocr_meta[key] = value
    # If OCR cannot recover Survey No but the source is conventionally named 2.tif / 3.tif,
    # use that filename as a safe display fallback. Do not guess village or scale from a name.
    if not ocr_meta.get('survey'):
        source_name = os.path.splitext(os.path.basename(name or path))[0]
        match = re.fullmatch(r'(?:survey[ _-]*)?(\d{1,6})(?:[ _-]*[a-z])?', source_name, re.I)
        if match:
            ocr_meta['survey'] = match.group(1)
    g, _ = _resize_long(full, AN_LONG)
    ink_all = (g < 170).astype(np.uint8)
    top, bot, x0, x1 = _find_zone(ink_all)
    H, W = g.shape
    top, bot, x0, x1 = max(0, top), min(H, bot), max(0, x0), min(W, x1)
    zone = ink_all[top:bot, x0:x1].copy()
    sheet = {'version': FEATURE_VERSION, 'name': name or os.path.basename(path), 'meta': ocr_meta,
             'size_full': full.shape, 'zone': (x0, top, x1, bot), 'ok': False}
    ys, xs = np.nonzero(zone)
    if len(xs) < 200:
        sheet['error'] = 'no drawing found'
        return sheet
    # thick outline = pixels whose local half-width is clearly above the thin parcel lines
    d = cv2.distanceTransform(zone, cv2.DIST_L2, 5)
    thick = (d >= 1.45).astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)

    def _outline_from(thick):
        for _ in range(2):
            thick = cv2.dilate(thick, k3) & zone
        # a thick border drawn on a slant is sometimes only 2-3 px wide after downscaling: bridge those
        # short breaks along the line itself so the outline stays one continuous curve
        bridged = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        thick = (bridged & cv2.dilate(zone, k3)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(thick, connectivity=8)
        small = [i for i in range(1, n) if st[i, 4] < 25]
        if small:
            thick[np.isin(lab, small)] = 0
        return thick, skeletonize(thick > 0)

    thick, sk = _outline_from(thick)
    if int(sk.sum()) < MIN_OUTLINE_PX:
        # This sheet's boundary lines are finer than the usual ones (different scan size / pen), so the
        # fixed width test missed them.  Measure the sheet's own line widths at 4x resolution instead.
        g2, _ = _resize_long(full, 4 * AN_LONG)
        d2 = cv2.distanceTransform((g2 < 170).astype(np.uint8), cv2.DIST_L2, 5)
        thr2 = _thick_threshold(d2)
        tf = cv2.resize((d2 >= thr2).astype(np.uint8) * 255, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
        thick2, sk2 = _outline_from(((tf[top:bot, x0:x1] > 60) & (zone > 0)).astype(np.uint8))
        if int(sk2.sum()) > int(sk.sum()):
            thick, sk = thick2, sk2
    sy, sx = np.nonzero(sk)
    if len(sx) < 50:
        sheet['error'] = 'no thick outline found'
        return sheet
    # silhouette (all ink, gaps closed, outside flooded away)
    closed = cv2.morphologyEx(zone, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27)))   # open outlines
    pad = cv2.copyMakeBorder(cv2.dilate(closed, k3, iterations=2), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    ff = pad.copy()
    mask = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 2)
    fill = (ff != 2).astype(np.uint8)[2:-2, 2:-2]
    fill = cv2.morphologyEx(fill, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    fill4 = cv2.resize(fill * 255, (fill.shape[1] // 4, fill.shape[0] // 4), interpolation=cv2.INTER_AREA)
    bx0, by0, bx1, by1 = xs.min() + x0, ys.min() + top, xs.max() + x0, ys.max() + top
    sheet.update(
        ok=True,
        skel=np.stack([sx + x0, sy + top], 1).astype(np.int16),
        fill4=fill4.astype(np.uint8),
        bbox=(int(bx0), int(by0), int(bx1), int(by1)),
        origin=(float(bx0 + bx1) / 2, float(by0 + by1) / 2),
        stones=[(x + x0, y + top) for x, y in detect_stones(zone)],
        hull=cv2.convexHull(np.stack([xs + x0, ys + top], 1).astype(np.int32))[:, 0, :].astype(np.float32),
    )
    ys2, xs2 = np.nonzero(fill)
    sheet['centroid'] = (float(xs2.mean() + x0), float(ys2.mean() + top)) if len(xs2) else sheet['origin']
    paper = ((fill > 0) & (zone == 0)).astype(np.uint8)
    dt = cv2.distanceTransform(paper, cv2.DIST_L2, 5)
    if dt.max() > 0:
        py, px = np.unravel_index(dt.argmax(), dt.shape)
        sheet['tag'] = (float(px + x0), float(py + top))
        sheet['tag_r'] = float(dt.max())
    else:
        sheet['tag'] = sheet['centroid']
        sheet['tag_r'] = 20.0
    scale = sheet['meta'].get('scale')
    sheet['g'] = (float(scale) / AN_LONG) if scale else None
    return sheet


def load_sheet(item, cache_dir=None):
    """item: dict(id, path, name, meta).  Features are cached on disk next to the approved file."""
    key = None
    if cache_dir:
        st = os.stat(item['path'])
        key = os.path.join(cache_dir, f"{item['id']}.puzzle.pkl")
        sig = (FEATURE_VERSION, st.st_mtime_ns, st.st_size, repr(sorted((item.get('meta') or {}).items())))
        if os.path.isfile(key):
            try:
                with open(key, 'rb') as fh:
                    cached = pickle.load(fh)
                if cached.get('sig') == sig:
                    return cached['sheet']
            except Exception:
                pass
    sheet = extract_sheet(item['path'], item.get('name'), item.get('meta'))
    sheet['id'] = item['id']
    sheet['path'] = item['path']
    if key:
        try:
            with open(key, 'wb') as fh:
                pickle.dump({'sig': sig, 'sheet': sheet}, fh)
        except Exception:
            pass
    return sheet
# --------------------------------------------------------------------------------------
# 2. coarse search (FFT)
# --------------------------------------------------------------------------------------
def _rot_matrix(angle, c0):
    return cv2.getRotationMatrix2D((c0, c0), float(angle), 1.0)


class _Coarse:
    """Rasters of every sheet at a common coarse ground resolution, centred on a P x P canvas."""
    def __init__(self, sheets, P_choices=(256, 512)):
        ext = []
        for s in sheets:
            bx0, by0, bx1, by1 = s['bbox']
            ext.append(max(bx1 - bx0, by1 - by0) * s['g'])
        ext = np.array(ext)
        for P in P_choices:
            G = ext.max() / (P / 2 - 6)
            if np.median(ext) / G >= 45 or P == P_choices[-1]:
                break
        self.P, self.G = P, G
        self.c0 = P // 2
        self.ext_px = ext / G
        n = len(sheets)
        self.S = np.zeros((n, P, P), np.float32)
        self.F = np.zeros((n, P, P), np.float32)
        self.L = np.zeros(n)
        k3 = np.ones((3, 3), np.uint8)
        kb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.T = np.zeros((n, P, P), np.float32)
        for i, s in enumerate(sheets):
            ox, oy = s['origin']
            sk = s['skel'].astype(np.float64)
            cx = ((sk[:, 0] - ox) * s['g'] / G + self.c0).round().astype(int)
            cy = ((sk[:, 1] - oy) * s['g'] / G + self.c0).round().astype(int)
            ok = (cx >= 0) & (cx < P) & (cy >= 0) & (cy < P)
            r = np.zeros((P, P), np.uint8)
            r[cy[ok], cx[ok]] = 1
            r = cv2.dilate(r, k3)
            self.S[i] = r
            self.T[i] = cv2.dilate(r, kb)
            self.L[i] = r.sum()
            self.F[i] = self.fill_canvas(s) / 255.0
        self.mask_overlap = 0.25
        self.med_ext = float(np.median(self.ext_px))
        self.lam = 10.0 / max(self.med_ext, 1.0)
        self.kap = 0.12 * self.med_ext
        # corners / junctions / line ends of the outline: a shared border starts and ends at them
        self.C = np.zeros((n, P, P), np.float32)
        self.corners = []
        for i in range(n):
            h = cv2.cornerHarris(self.S[i], 3, 3, 0.04)
            mx = h.max()
            pk = (h > 0.10 * mx) & (h == cv2.dilate(h, np.ones((7, 7), np.float32))) if mx > 0 else np.zeros_like(h, bool)
            ys, xs = np.nonzero(pk)
            self.corners.append(np.stack([xs, ys], 1).astype(np.float64))
            blob = np.zeros((P, P), np.float32)
            blob[ys, xs] = 1
            blob = cv2.GaussianBlur(blob, (0, 0), 1.5)
            if blob.max() > 0:
                blob /= blob.max()
            self.C[i] = blob
        self.FC = sfft.rfft2(self.C, workers=-1).astype(np.complex64)
        self.FT = sfft.rfft2(self.T, workers=-1).astype(np.complex64)
        self.FF = sfft.rfft2(self.F, workers=-1).astype(np.complex64)
    def fill_canvas(self, s):
        P = self.P
        x0, y0, _, _ = s['zone']
        ox, oy = s['origin']
        a = 4.0 * s['g'] / self.G
        M = np.float32([[a, 0, (x0 - ox) * s['g'] / self.G + self.c0],
                        [0, a, (y0 - oy) * s['g'] / self.G + self.c0]])
        return cv2.warpAffine(s['fill4'], M, (P, P), flags=cv2.INTER_LINEAR)
    def spectra_rot(self, j, angle):
        M = _rot_matrix(angle, self.c0)
        Sr = cv2.warpAffine(self.S[j], M, (self.P, self.P), flags=cv2.INTER_LINEAR)
        Fr = cv2.warpAffine(self.F[j], M, (self.P, self.P), flags=cv2.INTER_LINEAR)
        Cr = cv2.warpAffine(self.C[j], M, (self.P, self.P), flags=cv2.INTER_LINEAR)
        return sfft.rfft2(Sr), sfft.rfft2(Fr), sfft.rfft2(Cr)
    def field(self, i_idx, j, angle):
        """score maps of sheet j (rotated) placed relative to the sheets in i_idx"""
        fs, ff, fc = self.spectra_rot(j, angle)
        Z = (self.FT[i_idx] * np.conj(fs).astype(np.complex64)
             - self.lam * self.FF[i_idx] * np.conj(ff).astype(np.complex64)
             + self.kap * self.FC[i_idx] * np.conj(fc).astype(np.complex64))
        f = sfft.irfft2(Z, s=(self.P, self.P), workers=-1)
        if self.mask_overlap:
            # positions where the two silhouettes would lie on top of each other are not neighbours
            ov = sfft.irfft2(self.FF[i_idx] * np.conj(ff).astype(np.complex64), s=(self.P, self.P), workers=-1)
            f[ov > self.mask_overlap * float(self.F[j].sum())] = -1e5
        return f


def _wrap(v, P):
    return v if v < P // 2 else v - P


def _is_landscape(sheet):
    """Whether the source page itself is landscape (width > height)."""
    shape = sheet.get('size_full')
    return bool(shape and len(shape) >= 2 and shape[1] > shape[0])
SPATIAL_PEAKS_PER_ANGLE = 8
PEAK_SUPPRESS_RADIUS = 9


def _top_spatial_peaks(score_map, count=SPATIAL_PEAKS_PER_ANGLE, radius=PEAK_SUPPRESS_RADIUS):
    """Return spatially distinct local maxima from the circular FFT shift map.
    Keeping more than the single strongest translation is important when a straight boundary
    produces several plausible alignments; the geometric verifier can then rank alternatives.
    """
    work = np.asarray(score_map, dtype=np.float32).copy()
    h, w = work.shape
    yy, xx = np.ogrid[:h, :w]
    peaks = []
    for _ in range(int(count)):
        idx = int(np.argmax(work))
        y, x = np.unravel_index(idx, work.shape)
        value = float(work[y, x])
        if not np.isfinite(value) or value <= -1e8:
            break
        peaks.append((value, int(y), int(x)))
        dy = np.abs(yy - y)
        dx = np.abs(xx - x)
        dy = np.minimum(dy, h - dy)
        dx = np.minimum(dx, w - dx)
        work[(dy * dy + dx * dx) <= radius * radius] = -np.inf
    return peaks


def _is_landscape(sheet):
    """Whether the source page itself is landscape (width > height)."""
    shape = sheet.get('size_full')
    return bool(shape and len(shape) >= 2 and shape[1] > shape[0])
PRESCREEN_MIN = 24            # below this many sheets every pair is searched in full
PRESCREEN_PARTNERS = 14       # detailed search only against the best-looking partners of each sheet


def _prescreen(sheets, progress=None):
    """Cheap low-resolution pass over ALL pairs; returns the pairs worth a detailed search."""
    n = len(sheets)
    cs0 = _Coarse(sheets, (128,))
    score = np.full((n, n), -1e9, np.float32)
    angles = BASE_ANGLES[::2]
    for j in range(1, n):
        idx = np.arange(j)
        best = np.full(j, -1e9, np.float32)
        for ang in angles:
            f = cs0.field(idx, j, ang)
            best = np.maximum(best, f.reshape(j, -1).max(1))
        score[:j, j] = best
        if progress:
            progress(j / max(1, n - 1))
    L = np.maximum(1.0, np.minimum.outer(cs0.L, cs0.L))
    norm = np.where(score > -1e8, score / L, -1e9)
    norm = np.maximum(norm, norm.T)
    cand = np.zeros((n, n), bool)
    for i in range(n):
        order = np.argsort(-norm[i])
        order = [k for k in order if k != i and norm[i, k] > -1e8][:PRESCREEN_PARTNERS]
        for k in order:
            cand[min(i, k), max(i, k)] = True
    return cand


def coarse_search(sheets, progress=None, anchor=None, force_pairs=None):
    """Keep several distinct translations for each angle, including quarter-turns."""
    n = len(sheets)
    cand = None
    if anchor is not None:                       # only compare sheet 0 (a stitched group) with the others
        cand = np.zeros((n, n), bool)
        cand[0, 1:] = True
    elif n > PRESCREEN_MIN:
        cand = _prescreen(sheets, (lambda f: progress(0.35 * f)) if progress else None)
        for a_, b_ in (force_pairs or ()):          # survey-number / approval-order neighbours are always searched
            cand[min(a_, b_), max(a_, b_)] = True
    cs = _Coarse(sheets)
    if anchor is not None:
        cs.mask_overlap = 0.12
    na = len(ANGLES)
    nk = SPATIAL_PEAKS_PER_ANGLE
    val = np.full((n, n, na, nk), -1e9, np.float32)
    loc = np.zeros((n, n, na, nk, 2), np.int16)
    landscape = np.array([_is_landscape(s) for s in sheets], dtype=bool)
    for j in range(1, n):
        idx = np.arange(j) if cand is None else np.nonzero(cand[:j, j])[0]
        if len(idx) == 0:
            continue
        mixed_orientation = idx[landscape[idx] != landscape[j]]
        for a, ang in enumerate(ANGLES):
            # Normal-angle matching is used for same-orientation pages. Quarter-turn searches
            # are enabled for portrait/landscape pairs.
            normal_angle = abs(float(ang)) <= MAX_NORMAL_ANGLE + 0.0001
            quarter_turn = ALLOW_QUARTER_TURNS and (abs(float(ang) - 90.0) <= 12.5001 or
                                                    abs(float(ang) - 270.0) <= 12.5001)
            active_idx = idx if normal_angle else (mixed_orientation if quarter_turn else np.array([], dtype=int))
            if len(active_idx) == 0:
                continue
            fields = cs.field(active_idx, j, ang)
            for row, i in enumerate(active_idx):
                for rank, (peak, py, px) in enumerate(_top_spatial_peaks(fields[row], nk)):
                    val[i, j, a, rank] = peak
                    dy = py if py < cs.P // 2 else py - cs.P
                    dx = px if px < cs.P // 2 else px - cs.P
                    loc[i, j, a, rank] = (dy, dx)
        if progress:
            progress((0.35 if cand is not None else 0.0) + (0.65 if cand is not None else 1.0) * j / max(1, n - 1))
    return cs, val, loc


def pair_hypotheses(val, loc, i, j, k=48):
    """Return diverse translations while deliberately including near-quarter-turn seeds.
    For mixed portrait/landscape sheets, the numerically strongest FFT angle can be a false
    sliding-border match. Always test angles near +/-90 degrees as well as raw score peaks.
    """
    pair_values = val[i, j]     # (angle, peak_rank)
    pair_locations = loc[i, j]
    angle_best = pair_values.max(axis=1)
    valid = np.flatnonzero(angle_best > -1e8)
    if not len(valid):
        return []
    score_order = list(valid[np.argsort(-angle_best[valid])])
    norm_angles = {int(idx): _norm_degrees(ANGLES[int(idx)]) for idx in valid}
    quarter_turn = any(55.0 <= abs(norm_angles[int(idx)]) <= 125.0 for idx in valid)
    angle_order = []
    if quarter_turn:
        centered = sorted(
            valid,
            key=lambda idx: (
                min(abs(norm_angles[int(idx)] - 90.0),
                    abs(norm_angles[int(idx)] + 90.0)),
                -float(angle_best[int(idx)])
            )
        )
        # Force useful seeds close to +/-90 degrees into the verifier, even if their FFT
        # score is below the bad alignment along a long straight boundary.
        angle_order.extend(int(idx) for idx in centered[:10])
    angle_order.extend(int(idx) for idx in score_order)
    # Stable de-duplication.
    angle_order = list(dict.fromkeys(angle_order))
    candidates = []
    for peak_rank in range(pair_values.shape[1]):
        for angle_idx in angle_order:
            score = float(pair_values[angle_idx, peak_rank])
            if score <= -1e8:
                continue
            delta = pair_locations[angle_idx, peak_rank].astype(int)
            angle = float(ANGLES[angle_idx])
            if any(abs(_norm_degrees(angle - old_angle)) < 1.0 and
                   np.hypot(*(delta - old_delta)) < 6 for old_angle, old_delta, _ in candidates):
                continue
            candidates.append((angle, delta, score))
            if len(candidates) >= k:
                return candidates
    return candidates
# --------------------------------------------------------------------------------------
# 3. verify / refine
# --------------------------------------------------------------------------------------
def _ground_pts(s, max_pts=6000):
    pts = s['skel'].astype(np.float64)
    if len(pts) > max_pts:
        pts = pts[np.linspace(0, len(pts) - 1, max_pts).astype(int)]
    return (pts - np.array(s['origin'])) * s['g']


def _icp(A, B, R, t, g_bar, iters=14):
    tree = cKDTree(A)
    for it in range(iters):
        tau = g_bar * (10.0 - 7.0 * it / (iters - 1))
        Bc = B @ R.T + t
        d, idx = tree.query(Bc, distance_upper_bound=tau)
        m = np.isfinite(d)
        if m.sum() < 20:
            break
        P, Q = Bc[m], A[idx[m]]
        pc, qc = P.mean(0), Q.mean(0)
        U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
        Rd = Vt.T @ U.T
        if np.linalg.det(Rd) < 0:
            Vt[-1] *= -1
            Rd = Vt.T @ U.T
        R = Rd @ R
        t = Rd @ t + (qc - Rd @ pc)
    return R, t


def _slide_refine(A, tree, B, R, t, g_bar):
    """ICP cannot move a straight border along itself; scan that one direction explicitly."""
    Bs = B if len(B) <= 1500 else B[np.linspace(0, len(B) - 1, 1500).astype(int)]
    Bc = Bs @ R.T + t
    d, _ = tree.query(Bc, distance_upper_bound=3.5 * g_bar)
    m = np.isfinite(d)
    if m.sum() < 15:
        return R, t
    P = Bc[m]
    c = P.mean(0)
    w, v = np.linalg.eigh(np.cov((P - c).T))
    u = v[:, 1]
    span = float(np.ptp((P - c) @ u)) + 1e-9
    steps = np.linspace(-0.6 * span, 0.6 * span, 41)
    counts = []
    for sft in steps:
        dd, _ = tree.query(Bc + sft * u, distance_upper_bound=3.5 * g_bar)
        counts.append(np.isfinite(dd).sum())
    counts = np.array(counts)
    top = np.nonzero(counts >= counts.max() - 1)[0]
    # plateau closest to "no move" first, then its centre
    run = [k for k in top]
    s_best = steps[int(np.mean(run))] if len(run) > 1 and (run[-1] - run[0] == len(run) - 1) else steps[run[len(run) // 2]]
    return R, t + s_best * u


def _norm_degrees(value):
    return (float(value) + 180.0) % 360.0 - 180.0


def _rotation_family(seed_angle):
    """Expected relative rotation family used by the coarse search: 0 or +/- 90 degrees."""
    angle = _norm_degrees(seed_angle)
    if abs(angle) <= 20.0:
        return 0.0
    if 60.0 <= angle <= 120.0:
        return 90.0
    if -120.0 <= angle <= -60.0:
        return -90.0
    return angle


def verify_pair(cs, sheets, i, j, ang, delta):
    si, sj = sheets[i], sheets[j]
    M = _rot_matrix(ang, cs.c0)
    R0 = M[:, :2].copy()
    t0 = np.array([delta[1], delta[0]], float) * cs.G          # (dx, dy)
    A, B = _ground_pts(si), _ground_pts(sj)
    g_bar = 0.5 * (si['g'] + sj['g'])
    # alignment on a thinned copy of the outlines (fast); the final measurements use every point
    As = A[::max(1, len(A) // 1200)]
    Bs = B[::max(1, len(B) // 1200)]
    R, t = _icp(As, Bs, R0, t0, g_bar)
    tree_s = cKDTree(As)
    R, t = _slide_refine(As, tree_s, Bs, R, t, g_bar)
    R, t = _icp(As, Bs, R, t, g_bar * 0.5, iters=6)
    tree_a = cKDTree(A)
    B_aligned = B @ R.T + t
    d_b, _ = tree_a.query(B_aligned)
    inl_b = d_b <= 3.5 * g_bar
    # Also measure A-to-B support. A false alignment can make many points from one outline
    # sit near a long line in the other while providing very little reciprocal support.
    tree_b = cKDTree(B_aligned)
    d_a, _ = tree_b.query(A)
    inl_a = d_a <= 3.5 * g_bar
    perim_i = len(si['skel']) * si['g']
    perim_j = len(sj['skel']) * sj['g']
    matched_b = float(inl_b.sum()) * (len(sj['skel']) / len(B)) * sj['g']
    matched_a = float(inl_a.sum()) * (len(si['skel']) / len(A)) * si['g']
    matched = min(matched_a, matched_b)
    match_balance = matched / max(1e-9, max(matched_a, matched_b))
    ext_i = cs.ext_px[i] * cs.G
    ext_j = cs.ext_px[j] * cs.G
    # silhouette overlap, at coarse resolution
    P, c0, G = cs.P, cs.c0, cs.G
    Fj = cs.F[j]
    Mw = np.float32(np.hstack([R, (c0 - R @ np.array([c0, c0]) + t / G)[:, None]]))
    Fw = cv2.warpAffine(Fj, Mw, (P, P), flags=cv2.INTER_LINEAR)
    ov = float((Fw * cs.F[i]).sum() / max(1.0, min(Fw.sum(), cs.F[i].sum())))
    # uniqueness of the coarse peak (a straight border can slide, a notched one cannot)
    f = cs.field(np.array([i]), j, ang)[0]
    py = int(delta[0]) % P
    px = int(delta[1]) % P
    yy, xx = np.ogrid[:P, :P]
    dy = np.minimum(abs(yy - py), P - abs(yy - py))
    dx = np.minimum(abs(xx - px), P - abs(xx - px))
    far = f[(dy ** 2 + dx ** 2) > (0.10 * min(cs.ext_px[i], cs.ext_px[j])) ** 2 + 16]
    candidate_peak = float(f[py, px])
    ratio = float(far.max() / candidate_peak) if candidate_peak > 0 and far.size else 1.0
    conf = 'high' if ratio < 0.80 else 'medium' if ratio < 0.92 else 'low'
    ci = (cs.corners[i] - cs.c0) * cs.G
    cj = (cs.corners[j] - cs.c0) * cs.G
    support = 0
    support_forward = 0
    support_reverse = 0
    if len(ci) and len(cj):
        transformed_cj = cj @ R.T + t
        dc, _ = cKDTree(ci).query(transformed_cj)
        support_forward = int((dc <= 4.0 * g_bar).sum())
        dc_reverse, _ = cKDTree(transformed_cj).query(ci)
        support_reverse = int((dc_reverse <= 4.0 * g_bar).sum())
        support = min(support_forward, support_reverse)
    refined_angle = math.degrees(math.atan2(R[0, 1], R[0, 0]))
    family_center = _rotation_family(ang)
    family_delta = _norm_degrees(refined_angle - family_center)
    frac = matched / max(1e-9, min(perim_i, perim_j))
    reasons = []
    if abs(family_delta) > 12.75:
        reasons.append(f"refined rotation {refined_angle:.2f}° left the searched orientation range around {family_center:.0f}°")
    if match_balance < 0.40:
        reasons.append(f"border match is not reciprocal (balance {match_balance:.2f})")
    if frac < MIN_LEN_FRAC:
        reasons.append(f"shared border fraction {frac:.3f} below {MIN_LEN_FRAC:.3f}")
    if matched < MIN_LEN_EXT * min(ext_i, ext_j):
        reasons.append("shared border is too short")
    if ov > MAX_OVERLAP:
        reasons.append(f"silhouette overlap {ov:.3f} above {MAX_OVERLAP:.3f}")
    # boundary stones: the same stone drawn on both sheets must land on the same point
    stone_support = 0
    si_st, sj_st = si.get('stones') or [], sj.get('stones') or []
    if si_st and sj_st:
        pi = (np.array(si_st, float) - np.array(si['origin'])) * si['g']
        pj = (np.array(sj_st, float) - np.array(sj['origin'])) * sj['g']
        dstone, _ = cKDTree(pi).query(pj @ R.T + t)
        stone_support = int((dstone <= 6.0 * g_bar).sum())
    # one stone alone can coincide by chance between two different boundary points: it only anchors a join
    # when the borders also genuinely run together (or two stones agree)
    stone_anchored = (stone_support >= 1 and ov <= MAX_OVERLAP and abs(refined_angle) <= 25.0
                      and (frac >= 0.12 or stone_support >= 2))
    if stone_anchored:
        reasons = [r for r in reasons if r.startswith('silhouette overlap')]
        conf = 'high'
    # a boundary stone drawn on one sheet right on the shared border must also be drawn on the other
    stone_missing = 0
    if si_st or sj_st:
        pi_g = (np.array(si_st, float) - np.array(si['origin'])) * si['g'] if si_st else np.zeros((0, 2))
        pj_g = ((np.array(sj_st, float) - np.array(sj['origin'])) * sj['g']) @ R.T + t if sj_st else np.zeros((0, 2))
        for p in pi_g:
            if cKDTree(B_aligned).query(p)[0] <= 3.5 * g_bar and (len(pj_g) == 0 or cKDTree(pj_g).query(p)[0] > 6.0 * g_bar):
                stone_missing += 1
        for p in pj_g:
            if tree_a.query(p)[0] <= 3.5 * g_bar and (len(pi_g) == 0 or cKDTree(pi_g).query(p)[0] > 6.0 * g_bar):
                stone_missing += 1
    relaxed = bool(GROW_MODE and reasons and frac >= 0.18 and ov <= GROW_MAX_OVERLAP and match_balance >= 0.5
                   and abs(refined_angle) <= 20.0 and all(r.startswith('silhouette overlap') for r in reasons))
    if stone_missing and not stone_anchored:
        relaxed = False
        reasons = reasons + ['a boundary stone on one sheet has no counterpart on the other']
    if relaxed:                   # long, reciprocal border; the silhouette of an open outline over-estimates overlap
        reasons = []
        conf = 'low'
    ok = not reasons
    # Corner detections can be sparse on rasterized parcel outlines. Permit a narrow fallback
    # only for a long, reciprocal shared edge with very little silhouette overlap; this preserves
    # strong joins such as the verified S2-S3 pair without blindly accepting shorter S3-S4 slides.
    strong_cornerless_border = (
        frac >= 0.22 and ov <= 0.02 and match_balance >= 0.40
    )
    if ok and support < 2 and not strong_cornerless_border and not stone_anchored and not relaxed:
        ok = False
        reasons.append(
            "ambiguous border alignment (fewer than two reciprocal corner matches; "
            "shared border is not strong enough for the cornerless fallback)"
        )
    return dict(i=i, j=j, R=R, t=t, matched=matched, matched_a=matched_a, matched_b=matched_b,
                match_balance=match_balance, frac=frac, overlap=ov, confidence=conf,
                peak_ratio=ratio, accepted=bool(ok), rejection_reason="; ".join(reasons),
                corners=support, stones=stone_support, stone_missing=stone_missing, corner_support_forward=support_forward, corner_support_reverse=support_reverse,
                rotation_drift_deg=round(abs(family_delta), 3),
                cornerless_fallback=bool(support < 2 and strong_cornerless_border),
                rot_deg=refined_angle)


def _nudge_off_overlap(cs, sheets, e):
    """A straight border lets a sheet slide along its neighbour.  Slide it to where it no longer lies on
    top of other sheets, while keeping (almost) the whole shared border."""
    i, j = e['i'], e['j']
    si, sj = sheets[i], sheets[j]
    g_bar = 0.5 * (si['g'] + sj['g'])
    A = _ground_pts(si)
    B = _ground_pts(sj)
    Bs = B[::max(1, len(B) // 1200)]
    tree = cKDTree(A)
    R, t0 = e['R'], e['t']
    P_, c0, G = cs.P, cs.c0, cs.G

    def evaluate(t):
        d, _ = tree.query(Bs @ R.T + t, distance_upper_bound=3.5 * g_bar)
        inl = float(np.isfinite(d).mean())
        Mw = np.float32(np.hstack([R, (c0 - R @ np.array([c0, c0]) + t / G)[:, None]]))
        Fw = cv2.warpAffine(cs.F[j], Mw, (P_, P_), flags=cv2.INTER_LINEAR)
        ov = float((Fw * cs.F[i]).sum() / max(1.0, min(Fw.sum(), cs.F[i].sum())))
        return inl, ov
    inl0, ov0 = evaluate(t0)
    best = (inl0 - 3.0 * ov0, t0, inl0, ov0)
    step = G
    for dy in range(-12, 13):
        for dx in range(-12, 13):
            t = t0 + np.array([dx, dy]) * step
            inl, ov = evaluate(t)
            if inl >= 0.85 * inl0 and inl - 3.0 * ov > best[0] + 1e-6:
                best = (inl - 3.0 * ov, t, inl, ov)
    if best[3] < ov0 - 0.005:
        e = dict(e)
        e['t'] = best[1]
        e['overlap'] = best[3]
        e['frac'] = e['frac'] * best[2] / max(inl0, 1e-9)
        e['nudged'] = True
    return e


def find_edges(cs, sheets, val, loc, progress=None, per_sheet=6, anchor=None, force_pairs=None):
    n = len(sheets)
    if anchor is not None:
        per_sheet = n
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if anchor is not None and i != anchor:
                continue
            top = float(val[i, j].max())
            if top >= 0.08 * min(cs.L[i], cs.L[j]):
                pairs.append((top, i, j))
    pairs.sort(reverse=True)
    count = np.zeros(n, int)
    chosen = []
    for sc, i, j in pairs:
        if count[i] >= per_sheet and count[j] >= per_sheet:
            continue
        count[i] += 1
        count[j] += 1
        chosen.append((i, j))
    have = set(chosen)
    forced = {(min(a_, b_), max(a_, b_)) for a_, b_ in (force_pairs or ())}
    for pr in sorted(forced):
        if pr not in have:
            chosen.append(pr)
    have = set(chosen)
    for i in range(n):                                  # sheets that share a boundary stone are always compared
        for j in range(i + 1, n):
            if anchor is not None and i != anchor:
                continue
            if (i, j) not in have and sheets[i].get('stones') and sheets[j].get('stones'):
                chosen.append((i, j))

    def stone_hypotheses(i, j):
        out = []
        si, sj = sheets[i], sheets[j]
        for pa in (si.get('stones') or [])[:4]:
            for pb in (sj.get('stones') or [])[:4]:
                p_i = (np.array(pa, float) - np.array(si['origin'])) * si['g']
                p_j = (np.array(pb, float) - np.array(sj['origin'])) * sj['g']
                for ang in (-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5):
                    Rr = _rot_matrix(ang, cs.c0)[:, :2]
                    tt = p_i - Rr @ p_j
                    out.append((ang, np.array([tt[1] / cs.G, tt[0] / cs.G]), 1.0))
        return out

    def work(pair):
        i, j = pair
        hypotheses = pair_hypotheses(val, loc, i, j, k=HYPOTHESES_PER_PAIR) + stone_hypotheses(i, j)
        cands = []
        best_coarse = max((score for _, _, score in hypotheses), default=1.0)
        for ang, delta, score in hypotheses:
            candidate = verify_pair(cs, sheets, i, j, ang, delta)
            candidate['coarse_score'] = float(score)
            candidate['coarse_ratio'] = float(score / best_coarse) if best_coarse > 0 else 0.0
            cands.append(candidate)
        good = [c for c in cands if c['accepted']]
        if GROW_MODE:
            good = [_nudge_off_overlap(cs, sheets, c) if c['overlap'] > 0.02 else c for c in good]
        if good:
            return good                              # the assembly step arbitrates between hypotheses
        if (min(i, j), max(i, j)) in forced:         # keep several rejected tries for the number-neighbour fallback
            top = sorted(cands, key=lambda c: c['matched'], reverse=True)[:3]
            for c in top:
                c['forced'] = True
            return top
        return [max(cands, key=lambda c: c['matched'])] if cands else []

    from concurrent.futures import ThreadPoolExecutor
    workers = max(1, min(8, (os.cpu_count() or 2)))
    edges = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for k, res in enumerate(pool.map(work, chosen)):
            edges.extend(res)
            if progress:
                progress((k + 1) / max(1, len(chosen)))
    return edges


# --------------------------------------------------------------------------------------
# 4. assemble
# --------------------------------------------------------------------------------------
def _mask(s, R, t, mn, res, W, H):
    x0, y0, _, _ = s['zone']
    ox, oy = s['origin']
    a = 4.0 * s['g'] / res
    off = (R @ ((np.array([x0, y0], float) - np.array([ox, oy])) * s['g']) + t - mn) / res
    M = np.float32(np.hstack([a * R, off[:, None]]))
    return cv2.warpAffine((s['fill4'] > 127).astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0


def _frame(sheets, members, place_list):
    pts = []
    for k, (R, t) in zip(members, place_list):
        s = sheets[k]
        c = (_corners(s) - np.array(s['origin'])) * s['g']
        pts.append(c @ R.T + t)
    pts = np.vstack(pts)
    return pts.min(0) - 1, pts.max(0) + 1


def _merge_ok(sheets, pa, pb, tol=None):
    if tol is None:
        tol = 0.08
    """pb is already expressed in pa's frame.  Every sheet of the smaller group must stay off the other group."""
    small, big = (pb, pa) if len(pb) <= len(pa) else (pa, pb)
    ids = list(pa) + list(pb)
    mn, mx = _frame(sheets, ids, [pa[k] if k in pa else pb[k] for k in ids])
    ext = np.median([max(sheets[k]['bbox'][2] - sheets[k]['bbox'][0], sheets[k]['bbox'][3] - sheets[k]['bbox'][1]) * sheets[k]['g'] for k in ids])
    res = max(ext / 70.0, float(max(mx - mn)) / 1500.0)
    W = int((mx[0] - mn[0]) / res) + 2
    H = int((mx[1] - mn[1]) / res) + 2
    union = np.zeros((H, W), bool)
    for k, (R, t) in big.items():
        union |= _mask(sheets[k], R, t, mn, res, W, H)
    if GROW_MODE:
        def hull_mask(k, R, t):
            sh = sheets[k]
            m = np.zeros((H, W), np.uint8)
            for hp_ in (sh.get('hulls') or [sh['hull']]):      # a stitched group: one footprint per member
                p = (((hp_.astype(np.float64) - np.array(sh['origin'])) * sh['g']) @ R.T + t - mn) / res
                cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
            return m > 0
        hull_union = np.zeros((H, W), bool)
        for k, (R, t) in big.items():
            hull_union |= hull_mask(k, R, t)
        for k, (R, t) in small.items():
            hm = hull_mask(k, R, t)
            if hm.sum() and (hm & hull_union).sum() / hm.sum() > HULL_OVERLAP_LIMIT:
                return False        # its whole footprint would lie on top of what is already placed
    deep = cv2.erode(union.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    for k, (R, t) in small.items():
        m = _mask(sheets[k], R, t, mn, res, W, H)
        a = m.sum()
        if a and (m & union).sum() / a > tol:
            return False
        # the sheet's own boundary must not run through the inside of the other sheets
        q = _ground_pts(sheets[k], 1500) @ R.T + t
        cx = np.clip(((q[:, 0] - mn[0]) / res).astype(int), 0, W - 1)
        cy = np.clip(((q[:, 1] - mn[1]) / res).astype(int), 0, H - 1)
        if deep[cy, cx].mean() > DEEP_LIMIT:
            return False
    return True


def _rel_rot(R):
    return _norm_degrees(math.degrees(math.atan2(R[0, 1], R[0, 0])))


def rotation_prior(rot_deg):
    """1.0 for a small tilt, falling off smoothly; a 90 degree turn is almost never right inside one village."""
    if not ROT_SIGMA:
        return 1.0
    r = abs(_norm_degrees(rot_deg))
    return float(math.exp(-(r / ROT_SIGMA) ** 2))


def add_triangle_support(sheets, edges):
    """Loop closure: a join i-j is trusted more when going i->k->j through a third sheet lands in the
    same place.  Straight borders can slide, but then the loop around a third neighbour does not close."""
    acc = [e for e in edges if e['accepted']]
    by_pair = {}
    for e in acc:
        by_pair.setdefault((e['i'], e['j']), []).append(e)
    adj = {}
    for (i, j) in by_pair:
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)

    def routes(a, b):
        """all transforms mapping frame b -> frame a"""
        out = []
        for e in by_pair.get((a, b), []):
            out.append((e['R'], e['t']))
        for e in by_pair.get((b, a), []):
            Rinv = e['R'].T
            out.append((Rinv, -Rinv @ e['t']))
        return out
    ext = [max(s['bbox'][2] - s['bbox'][0], s['bbox'][3] - s['bbox'][1]) * s['g'] for s in sheets]
    for e in acc:
        i, j = e['i'], e['j']
        sup = 0
        tol_t = TRI_TOL_POS * min(ext[i], ext[j])
        for k in adj.get(i, set()) & adj.get(j, set()):
            ok = False
            for (Rik, tik) in routes(i, k):
                for (Rkj, tkj) in routes(k, j):
                    R = Rik @ Rkj
                    t = Rik @ tkj + tik
                    if (abs(_norm_degrees(_rel_rot(R) - _rel_rot(e['R']))) <= TRI_TOL_ROT and
                            np.linalg.norm(t - e['t']) <= tol_t):
                        ok = True
                        break
                if ok:
                    break
            sup += int(ok)
        e['triangle_support'] = sup
    return acc


def assemble(sheets, edges):
    """Greedy, best border first.  A link is only used when merging the two groups leaves no
    silhouettes lying on top of each other - correct neighbours tile, wrong 'slides' collide."""
    n = len(sheets)
    wconf = {'high': 1.0, 'medium': 0.85, 'low': 0.7}
    acc = add_triangle_support(sheets, edges)
    def edge_quality(e):
        balance = max(0.0, min(1.0, float(e.get('match_balance', 1.0))))
        # A secondary peak may have a slightly lower raw correlation but a better geometric fit.
        # Keep peak uniqueness as a modest tie-breaker rather than overwhelming the geometry.
        ambiguity_weight = 1.0 - 0.25 * min(1.0, max(0.0, float(e.get('peak_ratio', 1.0))))
        coarse_weight = 0.9 + 0.1 * max(0.0, min(1.0, float(e.get('coarse_ratio', 1.0))))
        corner_weight = 1.0 + 1.8 * min(e['corners'], 4)
        drift = min(15.0, max(0.0, float(e.get('rotation_drift_deg', 0.0))))
        rotation_weight = 1.0 - 0.025 * drift
        overlap_weight = max(0.25, 1.0 - 2.0 * max(0.0, float(e.get('overlap', 0.0))))
        fallback_weight = 0.88 if e.get('cornerless_fallback') else 1.0
        loop_weight = (1.0 + 1.5 * min(e.get('triangle_support', 0), 3)) * (1.0 + 3.0 * min(e.get('stones', 0), 2))
        return (e['matched'] * balance * wconf[e['confidence']] *
                ambiguity_weight * coarse_weight * corner_weight *
                rotation_weight * overlap_weight * fallback_weight *
                rotation_prior(e['rot_deg']) * loop_weight * (6.0 if e.get('number_edge') else 1.0))
    acc.sort(key=edge_quality, reverse=True)
    grp = {i: i for i in range(n)}
    place = {i: {i: (np.eye(2), np.zeros(2))} for i in range(n)}
    used = []
    for e in acc:
        i, j = e['i'], e['j']
        gi, gj = grp[i], grp[j]
        if gi == gj:
            continue
        Ri, ti = place[gi][i]
        Re, te = e['R'], e['t']
        Rj_new, tj_new = Ri @ Re, Ri @ te + ti                     # sheet j in group gi's frame
        Rj_old, tj_old = place[gj][j]
        D = Rj_new @ Rj_old.T
        dt = tj_new - D @ tj_old
        moved = {k: (D @ R, D @ t + dt) for k, (R, t) in place[gj].items()}
        if not _merge_ok(sheets, place[gi], moved):
            continue
        place[gi].update(moved)
        for k in moved:
            grp[k] = gi
        del place[gj]
        used.append(e)
    return list(place.values()), used
# --------------------------------------------------------------------------------------
# 5. render
# --------------------------------------------------------------------------------------
def _corners(s):
    bx0, by0, bx1, by1 = s['bbox']
    return np.array([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]], float)


def render_group(sheets, place, edges_used, title=''):
    ids = sorted(place)
    allc = []
    for k in ids:
        s = sheets[k]
        R, t = place[k]
        c = (_corners(s) - np.array(s['origin'])) * s['g']
        allc.append(c @ R.T + t)
    allc = np.vstack(allc)
    mn, mx = allc.min(0), allc.max(0)
    span = float(max(mx - mn))
    margin = 0.03 * span
    mn = mn - margin
    mx = mx + margin
    full_long = max(max(sheets[k]['size_full']) for k in ids)
    native = max((max(sheets[k]['size_full']) / AN_LONG) / sheets[k]['g'] for k in ids)
    rho = min(OUT_MAX / float(max(mx - mn)), native)
    W = int(math.ceil((mx[0] - mn[0]) * rho))
    H = int(math.ceil((mx[1] - mn[1]) * rho))
    ink = np.zeros((H, W), np.uint8)
    tags = []
    for k in ids:
        s = sheets[k]
        R, t = place[k]
        full = _read_gray(s['path'])
        rf = max(full.shape) / AN_LONG
        bx0, by0, bx1, by1 = s['bbox']
        pad = 6
        cx0, cy0 = max(0, bx0 - pad), max(0, by0 - pad)
        cx1, cy1 = bx1 + pad, by1 + pad
        crop = 255 - full[int(cy0 * rf):int(cy1 * rf), int(cx0 * rf):int(cx1 * rf)]
        lin = rho * s['g'] / rf
        if lin < 0.6:
            crop = cv2.resize(crop, None, fx=lin, fy=lin, interpolation=cv2.INTER_AREA)
            f_pre = lin
        else:
            f_pre = 1.0
        L = (rho * s['g'] / rf / f_pre) * R
        ox, oy = s['origin']
        # crop pixel (u,v) -> render px = (u/f_pre + cx0*rf) -> analysis px = /rf
        off = rho * (R @ (np.array([cx0, cy0]) - np.array([ox, oy])) * s['g'] + t - mn)
        Mx = np.float32(np.hstack([L, off[:, None]]))
        h, w = crop.shape
        co = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
        tc = co @ L.T + off
        x0, y0 = int(max(0, math.floor(tc[:, 0].min()) - 1)), int(max(0, math.floor(tc[:, 1].min()) - 1))
        x1, y1 = int(min(W, math.ceil(tc[:, 0].max()) + 2)), int(min(H, math.ceil(tc[:, 1].max()) + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        Mr = Mx.copy()
        Mr[:, 2] -= (x0, y0)
        warped = cv2.warpAffine(crop, Mr, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=0)
        ink[y0:y1, x0:x1] = np.maximum(ink[y0:y1, x0:x1], warped)
        cxa, cya = s['tag']
        pc = (R @ ((np.array([cxa, cya]) - np.array([ox, oy])) * s['g']) + t - mn) * rho
        tags.append((s['meta'].get('survey') or s['name'], pc, s['tag_r'] * s['g'] * rho))
    lines = 255 - ink                                  # black lines on white, no tags
    rgb = cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR)
    fs = max(0.7, max(W, H) / 2600.0)
    th = max(2, int(round(fs * 2.2)))
    for label, (px, py), rad in tags:
        text = f'S.{label}'
        f_t = fs * 1.6
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, f_t, th + 1)
        while tw > 1.9 * rad and f_t > 0.35:        # keep the tag inside the empty parcel
            f_t *= 0.9
            (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, f_t, th + 1)
        org = (int(px - tw / 2), int(py + tht / 2))
        cv2.putText(rgb, text, org, cv2.FONT_HERSHEY_SIMPLEX, f_t, (30, 30, 200), th + 1, cv2.LINE_AA)
    if title:
        cv2.putText(rgb, title, (int(0.02 * W), int(0.03 * H) + 10), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.1, (60, 60, 60),
                    th, cv2.LINE_AA)
    # coverage-overlap sanity check (sheets placed on top of each other = bad match somewhere)
    return rgb, lines, dict(width=W, height=H)


def overlap_check(cs_like, sheets, place):
    """fraction of covered silhouette area that is covered more than once."""
    ids = sorted(place)
    if len(ids) < 3:
        return 0.0
    pts = []
    for k in ids:
        s = sheets[k]
        R, t = place[k]
        c = (_corners(s) - np.array(s['origin'])) * s['g']
        pts.append(c @ R.T + t)
    pts = np.vstack(pts)
    mn, mx = pts.min(0) - 1, pts.max(0) + 1
    gres = float(max(mx - mn)) / 700.0
    W = int((mx[0] - mn[0]) / gres) + 2
    H = int((mx[1] - mn[1]) / gres) + 2
    cnt = np.zeros((H, W), np.uint8)
    for k in ids:
        s = sheets[k]
        R, t = place[k]
        x0, y0, _, _ = s['zone']
        ox, oy = s['origin']
        a = 4.0 * s['g'] / gres
        L = a * R
        off = (R @ (np.array([x0, y0]) - np.array([ox, oy])) * s['g'] + t - mn) / gres
        M = np.float32(np.hstack([L, off[:, None]]))
        f = cv2.warpAffine((s['fill4'] > 127).astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST)
        cnt += f
    cov = (cnt > 0).sum()
    return float((cnt > 1).sum() / cov) if cov else 0.0
# --------------------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------------------
def make_super(sheets, place):
    """A stitched group seen as one big sheet (its frame is the group frame), so a loose sheet can be
    matched against everything already placed instead of one neighbour at a time."""
    ids = sorted(place)
    gref = float(np.median([sheets[k]['g'] for k in ids]))
    pts, stones = [], []
    for k in ids:
        s = sheets[k]
        R, t = place[k]
        rel = (s['skel'].astype(np.float64) - np.array(s['origin'])) * s['g']
        pts.append((rel @ R.T + t) / gref)
        if s.get('stones'):
            st = (np.array(s['stones'], float) - np.array(s['origin'])) * s['g']
            stones.extend(((st @ R.T + t) / gref).tolist())
    allp = np.vstack(pts)
    mn, mx = allp.min(0) - 8, allp.max(0) + 8
    W, H = int((mx[0] - mn[0]) / 4) + 2, int((mx[1] - mn[1]) / 4) + 2
    fill = np.zeros((H, W), np.uint8)
    for k in ids:
        s = sheets[k]
        R, t = place[k]
        x0, y0, _, _ = s['zone']
        off = ((R @ ((np.array([x0, y0], float) - np.array(s['origin'])) * s['g']) + t) / gref - mn) / 4.0
        M = np.float32(np.hstack([(s['g'] / gref) * R, off[:, None]]))
        fill = np.maximum(fill, cv2.warpAffine(s['fill4'], M, (W, H), flags=cv2.INTER_LINEAR))
    hp = []
    for k in ids:
        s = sheets[k]
        R, t = place[k]
        hp.append((((s['hull'].astype(np.float64) - np.array(s['origin'])) * s['g']) @ R.T + t) / gref)
    hull_all = cv2.convexHull(np.vstack(hp).astype(np.float32))[:, 0, :]
    hulls = [h.astype(np.float32) for h in hp]
    cx, cy = float(allp[:, 0].mean()), float(allp[:, 1].mean())
    return dict(hull=hull_all, hulls=hulls, ok=True, id='__group__', name='group', meta={}, g=gref, origin=(0.0, 0.0),
                skel=np.round(allp).astype(np.int32), fill4=fill, stones=[tuple(p) for p in stones],
                zone=(float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])),
                bbox=(float(allp[:, 0].min()), float(allp[:, 1].min()), float(allp[:, 0].max()), float(allp[:, 1].max())),
                centroid=(cx, cy), tag=(cx, cy), tag_r=20.0, size_full=(1, 1), path='')


def grow(sheets, groups, used, edges, max_rounds=10):
    """Attach loose sheets - and smaller stitched groups - to the biggest stitched group: match each
    against the whole group, keep what fits without overlapping anything, repeat until nothing attaches."""
    for _ in range(max_rounds):
        progressed = False
        for big in sorted([p for p in groups if len(p) > 1], key=len, reverse=True):
            if not any(big is g for g in groups):
                continue
            others = [g for g in groups if g is not big]
            if not others:
                return groups, used, edges
            sup = make_super(sheets, big)
            sub = [sup] + [sheets[next(iter(g))] if len(g) == 1 else make_super(sheets, g) for g in others]
            global ANGLES, MAX_NORMAL_ANGLE, HYPOTHESES_PER_PAIR, GROW_MODE
            saved = (ANGLES, MAX_NORMAL_ANGLE, HYPOTHESES_PER_PAIR, GROW_MODE)
            ANGLES, MAX_NORMAL_ANGLE, HYPOTHESES_PER_PAIR, GROW_MODE = WIDE_ANGLES, 45.0, 16, True
            try:
                cs2, val2, loc2 = coarse_search(sub, anchor=0)
                edges2 = find_edges(cs2, sub, val2, loc2, anchor=0)
                g2, used2 = assemble(sub, edges2)
            finally:
                ANGLES, MAX_NORMAL_ANGLE, HYPOTHESES_PER_PAIR, GROW_MODE = saved
            for q in g2:
                if 0 not in q or len(q) < 2:
                    continue
                R0, t0 = q[0]
                for kk, (Rk, tk) in q.items():
                    if kk == 0:
                        continue
                    other = others[kk - 1]
                    Rg, tg = R0.T @ Rk, R0.T @ (tk - t0)          # that group's frame -> the big group's frame
                    moved = {m: (Rg @ Rm, Rg @ tm + tg) for m, (Rm, tm) in other.items()}
                    big.update(moved)
                    groups[:] = [g for g in groups if g is not other]
                    progressed = True
                    best = (None, None, -1)             # which two sheets touch most (for the report)
                    for mo in moved:
                        Bg = _ground_pts(sheets[mo]) @ big[mo][0].T + big[mo][1]
                        for mb in big:
                            if mb in moved:
                                continue
                            Am = _ground_pts(sheets[mb]) @ big[mb][0].T + big[mb][1]
                            gb = 0.5 * (sheets[mo]['g'] + sheets[mb]['g'])
                            d, _ = cKDTree(Am).query(Bg, distance_upper_bound=3.5 * gb)
                            cnt = int(np.isfinite(d).sum())
                            if cnt > best[2]:
                                best = (mb, mo, cnt)
                    src = next((e for e in used2 if e['j'] == kk), None)
                    if src is not None and best[0] is not None:
                        ne = dict(src)
                        ne['i'], ne['j'] = best[0], best[1]
                        ne['grown'] = True
                        edges.append(ne)
                        used.append(ne)
            if progressed:
                break                                    # groups changed: start the next round afresh
        if not progressed:
            break
    return groups, used, edges


def _trusted(e):
    """Joins that are safe to make from two sheets alone: a shared boundary stone, or a long unambiguous
    border.  Everything weaker is only accepted when the sheet is compared with the whole group."""
    return bool((e.get('stones', 0) >= 1 and (e.get('frac', 0) >= 0.12 or e.get('stones', 0) >= 2)) or e.get('cornerless_fallback') or
                (e.get('corners', 0) >= 2 and e.get('confidence') == 'high' and e.get('frac', 0) >= 0.10))


def assemble_all(sheets, edges):
    """trusted pair joins -> grow groups -> weaker pair joins between loose sheets -> grow again"""
    acc = [e for e in edges if e['accepted']]
    groups, used = assemble(sheets, [e for e in acc if _trusted(e)])
    groups, used, edges = grow(sheets, groups, used, edges)
    if sum(len(g) > 1 for g in groups) > 1:
        # a small group that cannot attach as a whole may have been joined wrongly inside itself:
        # let its sheets try the main group one by one instead
        main = max(groups, key=len)
        loose = set()
        for g in [g for g in groups if g is not main and len(g) > 1]:
            groups.remove(g)
            loose |= set(g)
            for m in g:
                groups.append({m: (np.eye(2), np.zeros(2))})
        used = [u for u in used if not (u['i'] in loose and u['j'] in loose)]
        groups, used, edges = grow(sheets, groups, used, edges)
    for _ in range(3):
        singles = {next(iter(g)) for g in groups if len(g) == 1}
        weak = [e for e in acc if e['i'] in singles and e['j'] in singles]
        if not weak:
            break
        g2, u2 = assemble(sheets, weak)
        new = [q for q in g2 if len(q) > 1]
        if not new:
            break
        for q in new:
            groups[:] = [g for g in groups if not (len(g) == 1 and next(iter(g)) in q)]
            groups.append(q)
        used.extend(u2)
        groups, used, edges = grow(sheets, groups, used, edges)
    return groups, used, edges



# --------------------------------------------------------------------------------------
# approval-order fallback and manual placement
# --------------------------------------------------------------------------------------
LAST = {}          # state of the last build, so manual adjustments can re-draw without searching again
SIDES = ('right', 'left', 'above', 'below')


def _edge_to(e, s, p):
    """(R, t) of sheet s expressed in the frame of sheet p, from an edge between them (None if unrelated)."""
    Re, te = e['R'], e['t']
    if e['i'] == p and e['j'] == s:
        return Re, te
    if e['i'] == s and e['j'] == p:
        return Re.T, -Re.T @ te
    return None



OVERLAP_LIMIT = 0.03      # real neighbours overlap ~1% (drawing lines touching); a sheet lying on another is 5%+


def sheet_overlaps(sheets, place):
    """{(a, b): overlapping silhouette area / smaller silhouette} for every pair of placed sheets."""
    ids = sorted(place)
    if len(ids) < 2:
        return {}
    mn, mx = _frame(sheets, ids, [place[k] for k in ids])
    res = float(max(mx - mn)) / 1200.0
    W, H = int((mx[0] - mn[0]) / res) + 2, int((mx[1] - mn[1]) / res) + 2
    masks = {k: _mask(sheets[k], place[k][0], place[k][1], mn, res, W, H) for k in ids}
    out = {}
    for ai, a in enumerate(ids):
        for b in ids[ai + 1:]:
            small = max(1, min(int(masks[a].sum()), int(masks[b].sum())))
            out[(a, b)] = float((masks[a] & masks[b]).sum()) / small
    return out


def _support(k, place, used):
    """How well a sheet is tied into its group: 3 high, 2 medium, 1 low, 0 only by the approval-order fallback."""
    best = -1
    for e in used:
        if (e['i'] == k and e['j'] in place) or (e['j'] == k and e['i'] in place):
            v = 0 if e.get('sequence_fallback') else {'high': 3, 'medium': 2, 'low': 1}.get(e.get('confidence'), 1)
            best = max(best, v)
    return best


def prune_overlaps(sheets, groups, used, order):
    """No two sheets of a survey may lie on top of each other.  While some pair overlaps more than
    OVERLAP_LIMIT, the sheet that is tied in more weakly is taken out (and anything that only hung on it),
    so it can be placed by hand instead of sitting on a wrong spot.  Returns reasons for released sheets."""
    reasons = {}
    priors = {}
    changed = True
    while changed:
        changed = False
        for place in [g for g in groups if len(g) > 1]:
            ov = sheet_overlaps(sheets, place)
            if not ov:
                continue
            (a, b), worst = max(ov.items(), key=lambda kv: kv[1])
            if worst <= OVERLAP_LIMIT:
                continue
            sa, sb = _support(a, place, used), _support(b, place, used)
            drop = a if (sa, -order.get(a, a)) < (sb, -order.get(b, b)) else b
            other = b if drop == a else a
            remaining = {k for k in place if k != drop}
            # keep the biggest connected part (by the joins actually used); the rest is released as well
            comp, seen = [], set()
            for s0 in remaining:
                if s0 in seen:
                    continue
                part, stack = {s0}, [s0]
                while stack:
                    x = stack.pop()
                    for e in used:
                        for u, v in ((e['i'], e['j']), (e['j'], e['i'])):
                            if u == x and v in remaining and v not in part:
                                part.add(v)
                                stack.append(v)
                seen |= part
                comp.append(part)
            comp.sort(key=len, reverse=True)
            keep = comp[0] if comp else set()
            groups.remove(place)
            if keep:
                groups.append({k: place[k] for k in keep})
            released = [drop] + [k for k in remaining if k not in keep]
            for k in released:
                priors[k] = (place[k][0], place[k][1], other)
                groups.append({k: (np.eye(2), np.zeros(2))})
                reasons[k] = (f"its automatic position lay on top of {sheets[other]['name']} "
                              f"({100 * worst:.0f}% overlap), so it was not attached")
            used[:] = [e for e in used if not (e['i'] in released or e['j'] in released)]
            changed = True
            break
    return groups, used, reasons, priors



def refine_cluster(sheets, big, cluster, reach=0.14, max_rot=3.0, limit=0.02, step_frac=0.02):
    """Slide/turn a rigid cluster of loose sheets (their positions relative to each other stay fixed) around a
    rough position until it touches the placed sheets without lying on them - the way a person nudges a
    puzzle piece.  Among positions where no member overlaps the placed sheets by more than `limit`, the one
    whose thick outlines touch the placed outlines most wins.  Returns ({k: (R, t)}, contact) or None."""
    ks = list(cluster)
    ids = list(big)
    span = max(max(_sheet_size(sheets[k])) for k in ks)
    mn, mx = _frame(sheets, ids + ks, [big[m] for m in ids] + [cluster[k] for k in ks])
    mn, mx = mn - 0.3 * span, mx + 0.3 * span
    res = float(max(mx - mn)) / 1000.0
    W, H = int((mx[0] - mn[0]) / res) + 2, int((mx[1] - mn[1]) / res) + 2
    union = np.zeros((H, W), bool)
    gpts = []
    for m in ids:
        Rm, tm = big[m]
        union |= _mask(sheets[m], Rm, tm, mn, res, W, H)
        gpts.append(_ground_pts(sheets[m], 2500) @ Rm.T + tm)
    tree = cKDTree(np.vstack(gpts))
    P = {k: _ground_pts(sheets[k], 1500) for k in ks}
    total_pts = float(sum(len(v) for v in P.values()))
    C = np.mean([cluster[k][0] @ ((np.array(sheets[k]['centroid']) - np.array(sheets[k]['origin'])) * sheets[k]['g']) + cluster[k][1]
                 for k in ks], axis=0)

    def build(dx, dy, rot):
        a = math.radians(rot)
        Rz = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        return {k: (Rz @ cluster[k][0], Rz @ (cluster[k][1] - C) + C + np.array([dx, dy])) for k in ks}

    def score(dx, dy, rot):
        pl = build(dx, dy, rot)
        worst, hits = 0.0, 0
        for k in ks:
            R, t = pl[k]
            m = _mask(sheets[k], R, t, mn, res, W, H)
            a = int(m.sum())
            if not a:
                return None
            worst = max(worst, float((m & union).sum()) / a)
            if worst > 0.5:
                return worst, 0.0
            d, _ = tree.query(P[k] @ R.T + t, distance_upper_bound=3.5 * sheets[k]['g'])
            hits += int(np.isfinite(d).sum())
        return worst, hits / total_pts

    base = score(0.0, 0.0, 0.0)
    base_contact = base[1] if base else 0.0
    best = None

    def consider(dx, dy, rot):
        nonlocal best
        sc = score(dx, dy, rot)
        if sc is None or sc[0] > limit:
            return
        key = (sc[1], -math.hypot(dx, dy) / span)
        if best is None or key > best[0]:
            best = (key, dx, dy, rot)

    step = step_frac * span
    n = int(round(reach / step_frac))
    for rot in np.arange(-max_rot, max_rot + 0.01, 1.5):
        for ix in range(-n, n + 1):
            for iy in range(-n, n + 1):
                consider(ix * step, iy * step, float(rot))
    if best is None:
        return None
    _, bx, by, br = best
    for fs, fr in ((0.005 * span, 0.5), (0.002 * span, 0.2)):
        cx, cy, cr = bx, by, br
        for ix in (-2, -1, 0, 1, 2):
            for iy in (-2, -1, 0, 1, 2):
                for dr in (-fr, 0.0, fr):
                    consider(cx + ix * fs, cy + iy * fs, cr + dr)
        _, bx, by, br = best
    contact = best[0][0]
    if contact < max(0.10, 0.6 * base_contact):
        return None
    pl = build(bx, by, br)
    if not _merge_ok(sheets, big, pl):
        return None
    return pl, contact


def refine_placement(sheets, big, k, R0, t0, **kw):
    got = refine_cluster(sheets, big, {k: (R0, t0)}, **kw)
    return None if got is None else (got[0][k][0], got[0][k][1], got[1])


def _loose_partners(sheets, groups, edges, cluster):
    """Loose sheets with a well-verified, overlap-free pair join to a member of the cluster, with their
    position relative to it.  They travel with the cluster so they cannot land on top of each other."""
    singles = {next(iter(g)) for g in groups if len(g) == 1}
    changed = True
    while changed:
        changed = False
        for e in sorted((e for e in edges if e.get('accepted') and not e.get('grown')),
                        key=lambda e: float(e.get('matched', 0.0)), reverse=True):
            for m, j in ((e['i'], e['j']), (e['j'], e['i'])):
                if m in cluster and j in singles and j not in cluster:
                    if float(e.get('overlap', 1.0)) <= 0.03 and float(e.get('frac', 0.0)) >= 0.15:
                        rel = _edge_to(e, j, m)
                        Rm, tm = cluster[m]
                        cluster[j] = (Rm @ rel[0], Rm @ rel[1] + tm)
                        changed = True
    return cluster


def reattach_released(sheets, groups, used, priors, edges, order):
    """Sheets released for overlapping get one more try: the nearest overlap-free position that still touches,
    fitted together with the loose sheets that are verified neighbours of theirs."""
    for k in sorted(priors, key=lambda x: order.get(x, x)):
        if any(k in g and len(g) > 1 for g in groups):
            continue
        big = max((g for g in groups if len(g) > 1), key=len, default=None)
        if big is None:
            break
        R0, t0, anchor = priors[k]
        cluster = _loose_partners(sheets, groups, edges, {k: (R0, t0)})
        got = None
        if len(cluster) > 1:
            got = refine_cluster(sheets, big, cluster)
        if got is None:
            one = refine_cluster(sheets, big, {k: (R0, t0)})
            got = one
        if got is None:
            continue
        for kk, (Rk, tk) in got[0].items():
            big[kk] = (Rk, tk)
            groups[:] = [g for g in groups if not (len(g) == 1 and kk in g)]
            used.append({'i': anchor if kk == k else k, 'j': kk, 'confidence': 'low', 'sequence_fallback': True, 'fitted': True})
    return groups, used

def sequence_fallback(sheets, groups, used, edges, order):
    """Last automatic step.  A loose sheet whose best border candidate was refused only because the border
    is weak/ambiguous (for example a long straight edge it could slide along) is still attached to the
    neighbour it matched best, provided it does not land on top of anything.  Sheets are tried in approval
    order, and every such join is flagged low confidence so the person knows to check it."""
    changed = True
    while changed:
        changed = False
        for k in sorted((next(iter(g)) for g in groups if len(g) == 1), key=lambda x: order.get(x, x)):
            cands = [e for e in edges if e.get('accepted') and not e.get('grown') and (e['i'] == k or e['j'] == k)]
            cands.sort(key=lambda e: float(e.get('matched', 0.0)), reverse=True)
            for e in cands:
                p = e['j'] if e['i'] == k else e['i']
                big = next((g for g in groups if p in g and len(g) > 1), None)
                if big is None:
                    continue
                rel = _edge_to(e, k, p)
                Rp, tp = big[p]
                Rs, ts = Rp @ rel[0], Rp @ rel[1] + tp
                ok = _merge_ok(sheets, big, {k: (Rs, ts)})
                if ok:
                    trial = dict(big)
                    trial[k] = (Rs, ts)
                    ok = max((v for (x, y), v in sheet_overlaps(sheets, trial).items() if k in (x, y)), default=0.0) <= OVERLAP_LIMIT
                if not ok:
                    got = refine_placement(sheets, big, k, Rs, ts)
                    if got is None:
                        continue
                    Rs, ts = got[0], got[1]
                big[k] = (Rs, ts)
                groups[:] = [g for g in groups if not (len(g) == 1 and k in g)]
                ne = dict(e)
                ne['confidence'] = 'low'
                ne['sequence_fallback'] = True
                edges.append(ne)
                used.append(ne)
                changed = True
                break
            if changed:
                break
    return groups, used, edges


def _survey_num(s):
    m = re.match(r'\s*(\d{1,6})', str((s.get('meta') or {}).get('survey') or ''))
    return int(m.group(1)) if m else None


def number_neighbour_pairs(sheets):
    """Pairs of sheets that are next to each other by survey number (6-7, 7-8 ...) or approval order."""
    pairs = set()
    nums = sorted((n, k) for k, s in enumerate(sheets) for n in [_survey_num(s)] if n is not None)
    for (n1, a), (n2, b) in zip(nums, nums[1:]):
        if n2 - n1 <= 2:
            pairs.add((min(a, b), max(a, b)))
    byorder = sorted(range(len(sheets)), key=lambda k: sheets[k].get('order', k))
    for a, b in zip(byorder, byorder[1:]):
        pairs.add((min(a, b), max(a, b)))
    return pairs


def _side_seeds(sheets, p, k):
    """Starting placements of sheet k touching each of the four sides of sheet p, with the direction it may slide."""
    A, B = _ground_pts(sheets[p]), _ground_pts(sheets[k])
    I = np.eye(2)
    ca, cb = (A.min(0) + A.max(0)) / 2, (B.min(0) + B.max(0)) / 2
    return [
        ((I, np.array([ca[0] - cb[0], A.max(0)[1] - B.min(0)[1]])), np.array([1.0, 0.0])),   # below
        ((I, np.array([ca[0] - cb[0], A.min(0)[1] - B.max(0)[1]])), np.array([1.0, 0.0])),   # above
        ((I, np.array([A.max(0)[0] - B.min(0)[0], ca[1] - cb[1]])), np.array([0.0, 1.0])),   # right
        ((I, np.array([A.min(0)[0] - B.max(0)[0], ca[1] - cb[1]])), np.array([0.0, 1.0])),   # left
    ]


def _hull_corners(P, eps=6.0):
    """Corner points of an outline's convex hull (ground units): where two sheets that touch end flush."""
    h = cv2.convexHull(P.astype(np.float32)).reshape(-1, 1, 2)
    return cv2.approxPolyDP(h, eps, True).reshape(-1, 2).astype(float)


CORNER_TOL = 7.0       # metres: two sheet corners closer than this count as the same corner
CORNER_BONUS = 80.0    # in outline points: a flush shared corner is worth this much border


CONTINUE_DEG = 20.0    # an outer boundary may bend this much where it runs on from one sheet into the next
CONTINUE_BONUS = 100.0


def _flush_corners(Ah, Bw, tol=CORNER_TOL):
    """Pairs (i, j): hull corner i of one sheet and hull corner j of the other lie at the same place."""
    out = []
    for j in range(len(Bw)):
        d = np.linalg.norm(Ah - Bw[j], axis=1)
        i = int(d.argmin())
        if d[i] <= tol:
            out.append((i, j))
    return out


def _outline_continues(Ah, Bw, pairs):
    """At a shared corner the outer boundary of one sheet should run on as the boundary of the other: some
    side edge leaving the corner on sheet A continues, within CONTINUE_DEG, as an edge of sheet B."""
    def edges(H, i):
        n = len(H)
        return [H[(i + d) % n] - H[i] for d in (-1, 1) if np.linalg.norm(H[(i + d) % n] - H[i]) >= 15.0]
    for i, j in pairs:
        for ea in edges(Ah, i):
            for eb in edges(Bw, j):
                cosang = float(ea @ (-eb)) / (np.linalg.norm(ea) * np.linalg.norm(eb))
                if math.degrees(math.acos(max(-1.0, min(1.0, cosang)))) <= CONTINUE_DEG:
                    return True
    return False


def _scan_slide(sheets, p, k, seeds, max_rot=8.0):
    """Sheet k next to sheet p.  A straight shared border lets k slide along p and tilt a few degrees, and the
    border matcher then locks onto a wrong slide.  Scan slide x tilt around each seed (seed = ((R, t), axis or
    None)) and score by how much of BOTH outlines lie on each other, with a mild preference for small tilts.
    Returns [(score, R, t)] in the frame of p, best first; score is a fraction of the sheet's outline."""
    from scipy.spatial import cKDTree
    A = _ground_pts(sheets[p])
    B = _ground_pts(sheets[k])
    g = 0.5 * (sheets[p]['g'] + sheets[k]['g'])
    As, Bs = A[::max(1, len(A) // 1500)], B[::max(1, len(B) // 1500)]
    tA = cKDTree(As)
    Ah, Bh = _hull_corners(A), _hull_corners(B)
    tAh = cKDTree(Ah)
    c = Bs.mean(0)
    reach = 0.6 * float(max(np.ptp(B[:, 0]), np.ptp(B[:, 1])))
    merged = []
    for (Rs0, ts0), axis_d in seeds:
        ang0 = math.degrees(math.atan2(Rs0[1, 0], Rs0[0, 0]))
        Bt = Bs @ Rs0.T + ts0
        d, _ = tA.query(Bt)
        pts = Bt[d <= 3.5 * g]
        if len(pts) > 20:
            w, v = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))
            axis = v[:, 1]
        else:
            axis = axis_d if axis_d is not None else np.array([1.0, 0.0])
        normal = np.array([-axis[1], axis[0]])
        out = []
        for ang in np.arange(ang0 - max_rot, ang0 + max_rot + 0.01, 1.0):
            a = math.radians(ang)
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            base = (c @ Rs0.T + ts0) - c @ R.T          # keep the sheet's centre where the seed had it
            for sl in np.arange(-reach, reach + 0.1, 5.0):
                for off in (-8.0, -3.0, 0.0, 3.0, 8.0):
                    t = base + axis * sl + normal * off
                    Bt = Bs @ R.T + t
                    nb = int((tA.query(Bt)[0] <= 3.5 * g).sum())
                    if nb < 20:
                        continue
                    na = int((cKDTree(Bt).query(As)[0] <= 3.5 * g).sum())
                    Bw = Bh @ R.T + t
                    pairs = _flush_corners(Ah, Bw)
                    bonus = CORNER_BONUS * min(len(pairs), 2)
                    if pairs and _outline_continues(Ah, Bw, pairs):
                        bonus += CONTINUE_BONUS
                    out.append(((min(na, nb) - 4.0 * abs(ang) + bonus) / len(Bs), R, t))
        out.sort(key=lambda x: -x[0])
        keep = []
        for sc, R, t in out:                              # distinct placements only
            if all(np.linalg.norm(t - t2) > 8.0 or abs(R[1, 0] - R2[1, 0]) > 0.02 for _, R2, t2 in keep):
                keep.append((sc, R, t))
            if len(keep) >= 6:
                break
        merged.extend(keep)
    merged.sort(key=lambda x: -x[0])

    def _score(R, t, ang):
        Bt = Bs @ R.T + t
        nb = int((tA.query(Bt)[0] <= 3.5 * g).sum())
        if nb < 20:
            return None
        na = int((cKDTree(Bt).query(As)[0] <= 3.5 * g).sum())
        Bw = Bh @ R.T + t
        pairs = _flush_corners(Ah, Bw)
        bonus = CORNER_BONUS * min(len(pairs), 2) + (CONTINUE_BONUS if pairs and _outline_continues(Ah, Bw, pairs) else 0.0)
        return (min(na, nb) - 4.0 * abs(ang) + bonus) / len(Bs)

    refined = []
    for sc, R, t in merged[:4]:                           # fine pass: close any gap and settle the tilt
        ang0 = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        best = (sc, R, t)
        for dang in (-1.0, -0.5, 0.0, 0.5, 1.0):
            a = math.radians(ang0 + dang)
            R2 = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            t_c = (c @ R.T + t) - c @ R2.T
            for dx in np.arange(-4.0, 4.1, 1.0):
                for dy in np.arange(-4.0, 4.1, 1.0):
                    t2 = t_c + np.array([dx, dy])
                    v = _score(R2, t2, ang0 + dang)
                    if v is not None and v > best[0]:
                        best = (v, R2, t2)
        refined.append(best)
    return sorted(refined + merged[4:], key=lambda x: -x[0])


NF_LOG = {}        # sheet id -> why the survey-number fallback could not place it (shown in the report)


def number_fallback(sheets, groups, used, edges, order):
    """Loose sheet whose survey number (or approval order) is next to a placed sheet - S.7 next to S.6.
    The sheet is put beside THAT sheet by scanning slide and tilt along their shared border, taking the
    placement with the most outline on outline that does not land on anything.  Flagged low confidence."""
    def gap(k, p):
        a, b = _survey_num(sheets[k]), _survey_num(sheets[p])
        d = abs(a - b) if a is not None and b is not None else 999
        return (d, abs(order.get(k, k) - order.get(p, p)))
    tried = set()
    changed = True
    while changed:
        changed = False
        for k in sorted((next(iter(g)) for g in groups if len(g) == 1), key=lambda x: order.get(x, x)):
            partners = sorted((p for p in range(len(sheets)) if p != k), key=lambda p: gap(k, p))
            for p in partners:
                if (k, p) in tried or (gap(k, p)[0] > 2 and gap(k, p)[1] > 1):
                    continue                                   # not a number or approval-order neighbour
                tried.add((k, p))
                big = next((g for g in groups if p in g and len(g) > 1), None)
                if big is None:
                    continue
                seeds = []
                for e in sorted((e for e in edges if {e['i'], e['j']} == {k, p}),
                                key=lambda e: float(e.get('matched', 0.0)), reverse=True)[:2]:
                    rel = _edge_to(e, k, p)
                    seeds.append(((rel[0], rel[1]), None))
                Rp, tp = big[p]
                scans = _scan_slide(sheets, p, k, seeds) if seeds else []     # where the matcher saw a border first
                scans = scans + _scan_slide(sheets, p, k, _side_seeds(sheets, p, k))   # then every side of the neighbour
                why = []
                for sc, R, t in scans:
                    if sc < 0.08:
                        why.append('best border-on-border %.0f%% is too little' % (100 * sc))
                        break                                  # too little border on border to trust
                    Rs, ts = Rp @ R, Rp @ t + tp
                    trial = dict(big)
                    trial[k] = (Rs, ts)
                    ov = max(((v, sheets[x if y == k else y].get('name')) for (x, y), v in
                              sheet_overlaps(sheets, trial).items() if k in (x, y)), default=(0.0, None))
                    if ov[0] > OVERLAP_LIMIT:
                        got = refine_placement(sheets, big, k, Rs, ts) if ov[0] <= 0.25 else None   # nudge it off, as by hand
                        if got is None:
                            why.append('placement with %.0f%% border-on-border lands on %s (%.0f%% overlap)' % (100 * sc, ov[1], 100 * ov[0]))
                            continue
                        Rs, ts = got[0], got[1]
                        R, t = Rp.T @ Rs, Rp.T @ (ts - tp)
                        ov = (0.0, None)
                    big[k] = (Rs, ts)
                    NF_LOG.pop(sheets[k].get('id'), None)
                    groups[:] = [g for g in groups if not (len(g) == 1 and k in g)]
                    ne = dict(i=p, j=k, R=R, t=t, matched=float(sc) * len(_ground_pts(sheets[k])) * sheets[k]['g'],
                              frac=float(sc), overlap=float(ov[0]), rot_deg=math.degrees(math.atan2(R[1, 0], R[0, 0])),
                              corners=0, accepted=True)
                    ne['confidence'] = 'low'
                    ne['sequence_fallback'] = True
                    ne['number_fallback'] = True
                    ne['accepted'] = True
                    edges.append(ne)
                    used.append(ne)
                    changed = True
                    break
                if not changed and why:
                    NF_LOG[sheets[k].get('id')] = ('survey-number fallback beside %s failed: %s'
                                                   % (sheets[p].get('name'), '; '.join(why[:3])))
                if changed:
                    break
            if changed:
                break
    return groups, used, edges


def number_edges(sheets, groups, used, edges, order):
    """Survey numbers 1 apart that the matcher could not join directly (a straight shared border can slide):
    scan slide and tilt and, when a placement has a lot of outline on outline AND flush corners, offer it to
    the assembly as a strong join.  The assembly then drops the weaker, inconsistent links around it."""
    joined = {frozenset((e['i'], e['j'])) for e in used}
    new = []
    n = len(sheets)
    for p in range(n):
        for k in range(p + 1, n):
            a, b = _survey_num(sheets[p]), _survey_num(sheets[k])
            if a is None or b is None or abs(a - b) != 1 or frozenset((p, k)) in joined:
                continue
            seeds = _side_seeds(sheets, p, k)
            res = _scan_slide(sheets, p, k, seeds, max_rot=6.0)
            if not res or res[0][0] < NUMBER_EDGE_MIN:
                continue
            sc, R, t = res[0]
            nB = len(_ground_pts(sheets[k]))
            new.append(dict(i=p, j=k, R=R, t=t, frac=float(sc), overlap=0.0,
                            matched=float(sc) * 150.0, matched_a=float(sc) * 150.0, matched_b=float(sc) * 150.0,
                            match_balance=1.0, rot_deg=math.degrees(math.atan2(R[1, 0], R[0, 0])),
                            corners=2, stones=0, confidence='high', accepted=True, number_edge=True,
                            reason='survey-number neighbours: border-on-border and flush corners'))
    return new


NUMBER_EDGE_MIN = 0.20     # border-on-border + flush-corner score a number-neighbour join needs


def group_number_merge(sheets, groups, used, edges, order):
    """Groups that are still apart although they belong together.  Candidate neighbour pairs are the pairs the
    border matcher already found between the two groups (even when refused) and sheets whose survey numbers
    are 1 or 2 apart.  For each pair, a slide/tilt scan; the best placement that lands on nothing wins and the
    entire second group moves with it."""
    def snum(k):
        return _survey_num(sheets[k])
    merged_any = True
    while merged_any and len(groups) > 1:
        merged_any = False
        where = {m: gi for gi, g in enumerate(groups) for m in g}
        pair_best = {}
        for ia, A in enumerate(groups):
            for ib, B in enumerate(groups):
                if ia >= ib:
                    continue
                for p in A:
                    for k in B:
                        a, b = snum(p), snum(k)
                        if a is not None and b is not None and 1 <= abs(a - b) <= 2:
                            pair_best.setdefault((p, k), 0.0)
        for e in edges:                                   # the matcher's own evidence ranks the number-neighbour pairs
            i, j = e['i'], e['j']
            for key in ((i, j), (j, i)):
                if key in pair_best:
                    pair_best[key] = max(pair_best[key], float(e.get('frac', 0.0)))
        ranked = sorted(pair_best.items(), key=lambda kv: (-kv[1], order.get(kv[0][0], 0)))[:4]
        results = []
        for (p, k), _ in ranked:
            ia, ib = where[p], where[k]
            A, B = groups[ia], groups[ib]
            seeds = []
            for e in sorted((e for e in edges if {e['i'], e['j']} == {p, k}),
                            key=lambda e: float(e.get('matched', 0.0)), reverse=True)[:2]:
                rel = _edge_to(e, k, p)
                seeds.append(((rel[0], rel[1]), None))
            seeds += _side_seeds(sheets, p, k)
            Rp, tp = A[p]
            Rk, tk = B[k]
            for sc, R, t in _scan_slide(sheets, p, k, seeds):
                if sc < 0.08:
                    NF_LOG.setdefault(sheets[k].get('id'), 'group join with %s: best border-on-border %.0f%% is too little'
                                      % (sheets[p].get('name'), 100 * sc))
                    break
                Rs, ts = Rp @ R, Rp @ t + tp
                M = Rs @ Rk.T
                c = ts - M @ tk
                trial = dict(A)
                for m, (Rm, tm) in B.items():
                    trial[m] = (M @ Rm, M @ tm + c)
                hits = [(v, x, y) for (x, y), v in sheet_overlaps(sheets, trial).items()
                        if (x in B) != (y in B) and v > OVERLAP_LIMIT]
                if hits:
                    v, x, y = max(hits)
                    NF_LOG[sheets[k].get('id')] = ('group join with %s: placement lands on %s over %s (%.0f%% overlap)'
                                                    % (sheets[p].get('name'), sheets[x].get('name'), sheets[y].get('name'), 100 * v))
                    continue
                results.append((sc, ia, ib, p, k, R, t, trial))
                break
        if results:
            sc, ia, ib, p, k, R, t, trial = max(results, key=lambda r: r[0])
            A, B = groups[ia], groups[ib]
            A.update({m: v for m, v in trial.items() if m in B})
            groups.pop(ib)
            ne = dict(i=p, j=k, R=R, t=t, matched=0.0, frac=float(sc), overlap=0.0,
                      rot_deg=math.degrees(math.atan2(R[1, 0], R[0, 0])), corners=0, accepted=True,
                      confidence='low', sequence_fallback=True, number_fallback=True)
            edges.append(ne)
            used.append(ne)
            for m in B:
                NF_LOG.pop(sheets[m].get('id'), None)
            merged_any = True
    if len(groups) > 1:
        for msg in NF_LOG.values():
            print('[puzzle] ' + msg, flush=True)           # why a group could not be joined (server console)
    return groups, used, edges


def _ground_hull(s, R, t):
    return ((s['hull'].astype(np.float64) - np.array(s['origin'])) * s['g']) @ R.T + t


def _sheet_size(s):
    x0, y0, x1, y1 = s['bbox']
    return (x1 - x0) * s['g'], (y1 - y0) * s['g']


def place_beside(sheets, place, k, target, side, gap=0.0):
    """Put loose sheet k flush against `target` (already in `place`) on the given side, centred on it, with the
    same orientation as the target.  Returns (R, t)."""
    Rt, tt = place[target]
    Rk = Rt.copy()
    a = _ground_hull(sheets[target], Rt, tt)
    h0 = _ground_hull(sheets[k], Rk, np.zeros(2))
    amn, amx, bmn, bmx = a.min(0), a.max(0), h0.min(0), h0.max(0)
    ctr_a, ctr_b = (amn + amx) / 2, (bmn + bmx) / 2
    if side == 'right':
        d = np.array([amx[0] + gap - bmn[0], ctr_a[1] - ctr_b[1]])
    elif side == 'left':
        d = np.array([amn[0] - gap - bmx[0], ctr_a[1] - ctr_b[1]])
    elif side == 'below':
        d = np.array([ctr_a[0] - ctr_b[0], amx[1] + gap - bmn[1]])
    else:
        d = np.array([ctr_a[0] - ctr_b[0], amn[1] - gap - bmx[1]])
    return Rk, d


def apply_manual(sheets, groups, manual, id_index):
    """manual: {sheet_id: {'to': id, 'side': 'right'|..., 'fx': float, 'fy': float, 'rot': degrees}}.
    fx/fy are shifts as a fraction of the sheet's longest side.  Returns the ids that were placed by hand."""
    touched = set()
    # first the loose sheets that were told where to go ('to' + 'side'), repeating so chains resolve
    pending = {sid: m for sid, m in manual.items() if m.get('to') and sid in id_index and m['to'] in id_index}
    for _ in range(len(pending) + 1):
        for sid, m in list(pending.items()):
            k, tg = id_index[sid], id_index[m['to']]
            g_tg = next((g for g in groups if tg in g), None)
            if g_tg is None or (len(g_tg) == 1 and tg not in g_tg) or k == tg:
                continue
            if any(k in g and len(g) > 1 and tg in g and m.get('_done') for g in groups):
                continue
            if any(k in g and tg in g for g in groups) and not m.get('force'):
                # already together with its target (the automatic step found it): leave it where it is
                del pending[sid]
                continue
            R, t = place_beside(sheets, g_tg, k, tg, m.get('side', 'right'), m.get('gap', 0.0))
            for g in list(groups):
                if k in g and g is not g_tg:
                    del g[k]
                    if not g:
                        groups.remove(g)
            g_tg[k] = (R, t)
            touched.add(sid)
            del pending[sid]
    for sid, m in manual.items():
        if sid not in id_index:
            continue
        k = id_index[sid]
        g = next((g for g in groups if k in g), None)
        if g is None or not (m.get('fx') or m.get('fy') or m.get('rot')):
            continue
        R, t = g[k]
        s = sheets[k]
        w, h = _sheet_size(s)
        span = max(w, h)
        c = R @ ((np.array(s['centroid']) - np.array(s['origin'])) * s['g']) + t      # rotate about its own centre
        a = math.radians(m.get('rot') or 0.0)
        Rz = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        R2 = Rz @ R
        t2 = c - R2 @ ((np.array(s['centroid']) - np.array(s['origin'])) * s['g'])
        t2 = t2 + np.array([m.get('fx') or 0.0, m.get('fy') or 0.0]) * span
        g[k] = (R2, t2)
        touched.add(sid)
    return touched


def _write_group(sheets, place, used, vname, gid, out_dir, cs, manual_ids=()):
    surveys = sorted((sheets[k]['meta'].get('survey') or '?') for k in place)
    title = f"{vname} - survey nos. {', '.join(surveys)}" if len(surveys) <= 12 else f'{vname} - {len(surveys)} sheets'
    rgb, lines, info = render_group(sheets, place, used, title)
    cv2.imwrite(os.path.join(out_dir, f'group_{gid}.png'), rgb)
    prev = rgb
    long_side = max(prev.shape[:2])
    if long_side > 2200:
        f = 2200.0 / long_side
        prev = cv2.resize(prev, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out_dir, f'group_{gid}_preview.png'), prev)
    Image.fromarray(lines).point(lambda v: 255 if v > 140 else 0).convert('1').save(
        os.path.join(out_dir, f'group_{gid}.tif'), format='TIFF', compression='group4')
    members = []
    for k, (R, t) in sorted(place.items()):
        members.append({'id': sheets[k]['id'], 'name': sheets[k]['name'],
                        'survey': sheets[k]['meta'].get('survey'),
                        'order': sheets[k].get('order'),
                        'manual': sheets[k]['id'] in manual_ids,
                        'rotation_deg': round(math.degrees(math.atan2(R[0, 1], R[0, 0])), 2)})
    conf = [e['confidence'] for e in used if e['i'] in place and e['j'] in place]
    return {'group': gid, 'village': vname, 'sheets': members, 'width': info['width'], 'height': info['height'],
            'overlap_warning': overlap_check(cs, sheets, place) > 0.05,
            'low_confidence_links': sum(c == 'low' for c in conf),
            'sequence_links': sum(bool(e.get('sequence_fallback')) for e in used if e['i'] in place and e['j'] in place)}


def redraw(manual, out_dir):
    """Re-draw the last build with new manual placement (no border search).  Returns a fresh report."""
    if not LAST:
        raise RuntimeError('No puzzle has been built yet - press Build puzzle first.')
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, name))
    report = {'groups': [], 'unattached': [], 'needs_attention': list(LAST['needs_attention']),
              'edges': LAST['edges'], 'sheet_metadata': LAST['sheet_metadata']}
    gid = 0
    for v in LAST['villages']:
        sheets = v['sheets']
        id_index = {s['id']: i for i, s in enumerate(sheets)}
        groups = [dict(g) for g in v['groups']]
        touched = apply_manual(sheets, groups, manual, id_index)
        for place in groups:
            if len(place) == 1:
                k = next(iter(place))
                report['unattached'].append({'id': sheets[k]['id'], 'name': sheets[k]['name'],
                                             'survey': sheets[k]['meta'].get('survey'),
                                             'reason': v['reasons'].get(k, 'no neighbouring sheet with a matching border')})
                continue
            gid += 1
            report['groups'].append(_write_group(sheets, place, v['used'], v['vname'], gid, out_dir, v['cs'], touched))
    return report


def build_puzzle(items, out_dir, progress=None, cache_dir=None, manual=None):
    """items: list of dict(id, path, name, meta).  Writes group_<k>.png/_preview.png/.tif to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    def prog(stage, frac, msg):
        if progress:
            progress(stage, float(frac), msg)
    sheets_all = []
    for k, it in enumerate(items):
        prog('read', k / max(1, len(items)), f'Reading sheet {k + 1} of {len(items)}…')
        try:
            s = load_sheet(it, cache_dir)
        except Exception as exc:                                  # a broken file must not stop the batch
            s = {'ok': False, 'error': f'{type(exc).__name__}: {exc}', 'name': it.get('name'), 'meta': {}}
        s['id'] = it['id']
        s['path'] = it['path']
        s['name'] = it.get('name') or s.get('name')
        s['order'] = it.get('order', k + 1)
        sheets_all.append(s)
    report = {'groups': [], 'unattached': [], 'needs_attention': [], 'edges': [], 'sheet_metadata': []}
    LAST.clear()
    NF_LOG.clear()
    LAST.update(villages=[], needs_attention=[], edges=[], sheet_metadata=[])
    manual = manual or {}
    usable = []
    for s in sheets_all:
        if not s.get('ok'):
            report['needs_attention'].append({'id': s['id'], 'name': s['name'], 'reason': s.get('error', 'unreadable')})
        elif not s.get('g'):
            report['needs_attention'].append({'id': s['id'], 'name': s['name'],
                                              'reason': ('Header OCR unavailable: ' + s['meta'].get('ocr_error') + ' Scale (1:N) was not read - install/verify Tesseract or enter the printed denominator.' if s['meta'].get('ocr_error') else 'Scale (1:N) not detected. OCR ran but did not recognize the header; verify Tesseract and check the sheet header.')})
        else:
            usable.append(s)
    for sheet in sheets_all:
        smeta = sheet.get('meta') or {}
        report['sheet_metadata'].append({
            'id': sheet.get('id'),
            'name': sheet.get('name'),
            'ok': bool(sheet.get('ok')),
            'scale': smeta.get('scale'),
            'survey': smeta.get('survey'),
            'village': smeta.get('village'),
            'village_key': village_key(smeta),
            'ocr_error': smeta.get('ocr_error'),
            'outline_points': int(len(sheet.get('skel', []))) if sheet.get('ok') else 0,
            'width': int(sheet.get('size_full', (0, 0))[1]) if sheet.get('size_full') else None,
            'height': int(sheet.get('size_full', (0, 0))[0]) if sheet.get('size_full') else None,
            'orientation': ('landscape' if _is_landscape(sheet) else 'portrait') if sheet.get('size_full') else 'unknown',
            'error': sheet.get('error'),
        })
    # sheets of different villages never touch
    usable.sort(key=lambda s: s.get('order', 0))
    byv = {}
    for s in usable:
        byv.setdefault(village_key(s['meta']), []).append(s)
    gid = 0
    vkeys = list(byv)
    for vi, vk in enumerate(vkeys):
        sheets = byv[vk]
        vname = re.sub(r'(\]).*$', r'\1', sheets[0]['meta'].get('village') or 'village').strip()
        if len(sheets) < 2:
            for s in sheets:
                report['unattached'].append({'id': s['id'], 'name': s['name'], 'survey': s['meta'].get('survey'),
                                             'reason': 'only sheet of this village'})
            continue
        base, span = vi / len(vkeys), 1.0 / len(vkeys)
        prog('match', base, f'Searching borders in {vname}…')
        force = number_neighbour_pairs(sheets)
        cs, val, loc = coarse_search(
            sheets, lambda f: prog('match', base + span * 0.6 * f, f'Searching borders in {vname}…'),
            force_pairs=force)
        edges = find_edges(cs, sheets, val, loc,
                           lambda f: prog('verify', base + span * (0.6 + 0.3 * f), 'Checking shared borders…'),
                           force_pairs=force)
        groups, used, edges = assemble_all(sheets, edges)
        order = {i: s.get('order', i) for i, s in enumerate(sheets)}
        extra = number_edges(sheets, groups, used, edges, order)
        if extra:
            edges = edges + extra
            groups, used, edges = assemble_all(sheets, edges)
        groups, used, pruned, priors = prune_overlaps(sheets, groups, used, order)
        groups, used = reattach_released(sheets, groups, used, priors, edges, order)
        groups, used, edges = sequence_fallback(sheets, groups, used, edges, order)
        groups, used, edges = number_fallback(sheets, groups, used, edges, order)
        groups, used, edges = group_number_merge(sheets, groups, used, edges, order)
        auto_groups = [dict(g) for g in groups]
        id_index = {s['id']: i for i, s in enumerate(sheets)}
        touched = apply_manual(sheets, groups, manual, id_index)
        for e in edges:
            report['edges'].append({'a': sheets[e['i']]['id'], 'b': sheets[e['j']]['id'],
                                    'a_survey': sheets[e['i']]['meta'].get('survey'),
                                    'b_survey': sheets[e['j']]['meta'].get('survey'),
                                    'shared_border_pct': round(100 * e['frac'], 1),
                                    'overlap_pct': round(100 * e['overlap'], 2),
                                    'rotation_deg': round(e['rot_deg'], 2),
                                    'matched_border': round(float(e['matched']), 3),
                                    'matched_a': round(float(e.get('matched_a', 0.0)), 3),
                                    'matched_b': round(float(e.get('matched_b', 0.0)), 3),
                                    'match_balance': round(float(e.get('match_balance', 1.0)), 3),
                                    'coarse_ratio': round(float(e.get('coarse_ratio', 1.0)), 3),
                                    'corners': int(e.get('corners', 0)),
                                    'cornerless_fallback': bool(e.get('cornerless_fallback', False)),
                                    'rotation_drift_deg': float(e.get('rotation_drift_deg', 0.0)),
                                    'corner_support_forward': int(e.get('corner_support_forward', 0)),
                                    'corner_support_reverse': int(e.get('corner_support_reverse', 0)),
                                    'rejection_reason': e.get('rejection_reason', ''),
                                    'confidence': e['confidence'], 'accepted': e['accepted'],
                                    'sequence_fallback': bool(e.get('sequence_fallback', False)),
                                    'used': any(e is u for u in used)})
        reasons = {}
        for place in groups:
            if len(place) == 1:
                k = next(iter(place))
                candidates_for_sheet = [e for e in edges if e.get('i') == k or e.get('j') == k]
                best = max(candidates_for_sheet, key=lambda e: float(e.get('matched', 0.0)), default=None)
                reason = 'no neighbouring sheet with a matching border'
                if best is not None:
                    other_idx = best['j'] if best['i'] == k else best['i']
                    other_name = sheets[other_idx].get('name', 'another sheet')
                    if not best.get('accepted'):
                        detail = best.get('rejection_reason') or 'candidate failed border validation'
                        reason += f'; best candidate {other_name} rejected: {detail}'
                    elif not any(best is used_edge for used_edge in used):
                        reason += (f'; best candidate {other_name} was not used in the final assembly '
                                   f'(rotation {best.get("rot_deg", 0.0):.2f}°, '
                                   f'overlap {100 * best.get("overlap", 0.0):.2f}%, '
                                   f'{best.get("confidence", "unknown")} confidence)')
                if sheets[k].get('id') in NF_LOG:
                    reason += '; ' + NF_LOG[sheets[k]['id']]
                if k in pruned:
                    reason = pruned[k]
                reason += ' - use Manual placement to put it next to its neighbour'
                reasons[k] = reason
                report['unattached'].append({'id': sheets[k]['id'], 'name': sheets[k]['name'],
                                             'survey': sheets[k]['meta'].get('survey'),
                                             'reason': reason})
                continue
            gid += 1
            prog('render', base + span * 0.92, f'Drawing group {gid}…')
            report['groups'].append(_write_group(sheets, place, used, vname, gid, out_dir, cs, touched))
        LAST['villages'].append({'sheets': sheets, 'groups': auto_groups, 'used': used, 'cs': cs,
                                 'vname': vname, 'reasons': reasons})
    LAST['needs_attention'] = list(report['needs_attention'])
    LAST['edges'] = report['edges']
    LAST['sheet_metadata'] = report['sheet_metadata']
    prog('done', 1.0, 'Puzzle ready.')
    return report