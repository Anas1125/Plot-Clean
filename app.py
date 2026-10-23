"""Plot Map Cleaner - upload .tif/.png/.jpg survey sheets, get cleaned TIFs back.
Run:  pip install flask pillow numpy opencv-python-headless scikit-image
      python app.py   ->  http://localhost:5000
"""
import io, os, time, zipfile, tempfile
from flask import Flask, request, send_file, render_template_string, abort
from PIL import Image
import numpy as np
from cleaner import clean
Image.MAX_IMAGE_PIXELS = None
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 400 * 1024 * 1024

PAGE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Plot Map Cleaner</title>
<style>body{font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px;color:#222}
.box{border:2px dashed #999;border-radius:12px;padding:28px;text-align:center}
button{background:#1a73e8;color:#fff;border:0;border-radius:8px;padding:10px 22px;font-size:16px;cursor:pointer}
small{color:#666}</style>
<h2>Plot Map Cleaner</h2>
<p>Upload survey-sheet TIFs. You get back the same sheet with dimensions, point numbers, dashed lines,
arrows and building icons removed. Plot numbers and boundaries are kept.</p>
<form class=box method=post action=/clean enctype=multipart/form-data onsubmit="b.disabled=true;b.textContent='Cleaning… (~15 sec per sheet)'">
<input type=file name=files accept=".tif,.tiff,.png,.jpg,.jpeg" multiple required><br><br>
<label><input type=checkbox name=bw checked> save as 1-bit black &amp; white (smaller file)</label><br><br>
<label><input type=checkbox name=dbg> also give a debug image (everything removed shown in red)</label><br><br>
<button id=b>Clean</button></form>
<p><small>One file → cleaned .tif download. Several files (or debug image) → .zip.</small></p>"""

def process(data, name, bw, debug=False):
    im = Image.open(io.BytesIO(data))
    dpi = im.info.get('dpi')
    if debug:
        out, dbg = clean(im, log=print, debug=True)
    else:
        out, dbg = clean(im, log=print), None

    # Always create the output image before applying optional 1-bit conversion.
    o = Image.fromarray(out)
    if bw:
        o = o.point(lambda v: 255 if v > 140 else 0).convert('1')
    buf = io.BytesIO()
    kw = dict(compression='group4' if bw else 'tiff_lzw')
    if dpi and min(dpi) > 1: kw['dpi'] = dpi
    o.save(buf, format='TIFF', **kw)
    base = os.path.splitext(name)[0]
    res = [(base + '_cleaned.tif', buf.getvalue())]
    if dbg is not None:
        pb = io.BytesIO(); Image.fromarray(dbg).save(pb, format='PNG')
        res.append((base + '_removed_in_red.png', pb.getvalue()))
    return res

@app.route('/')
def index(): return render_template_string(PAGE)

@app.route('/clean', methods=['POST'])
def do_clean():
    files = request.files.getlist('files')
    if not files: abort(400)
    bw = 'bw' in request.form
    dbg = 'dbg' in request.form
    results = [r for f in files for r in process(f.read(), f.filename, bw, dbg)]
    if len(results) == 1:
        n, d = results[0]
        return send_file(io.BytesIO(d), mimetype='image/tiff', as_attachment=True, download_name=n)
    z = io.BytesIO()
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        for n, d in results: zf.writestr(n, d)
    z.seek(0)
    return send_file(z, mimetype='application/zip', as_attachment=True, download_name='cleaned_maps.zip')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
