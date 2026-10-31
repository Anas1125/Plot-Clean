FROM python:3.13-slim

# tesseract-ocr is the OCR engine that pytesseract calls (puzzle.py reads the printed scale / area with it)
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# approved sheets, puzzle files and board layout are written here: mount a persistent disk / volume on this path
VOLUME ["/app/approved_images"]

ENV PLOT_WORKERS=1 PYTHONUNBUFFERED=1
# ONE process (jobs and progress live in memory) with several threads; long timeout for very large TIFF uploads
CMD ["sh", "-c", "gunicorn -w 1 --threads 8 --timeout 900 -b 0.0.0.0:${PORT:-5001} app:app"]