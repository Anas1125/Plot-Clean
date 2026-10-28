# 🔍 Welcome to Smart Survey Application

## What is it?

This is a software where we can upload our survey and it cleans the survey and gives us the output.

The software gets rid of these things:

1. Decimal numbers
2. Dashed lines
3. Arrow marks
4. Diagrams

While removing them it keeps the **plot numbers** (1, 2, ..., 11B etc.) and the **inner and outer boundaries** untouched.

After the survey is cleaned we can manually check it. If we are satisfied we can download it or approve it to the puzzle feature.

The puzzle feature is where we add all the approved surveys together using edge snapping. Once everything is joined, we can download the final joined survey.

---

## 1. How to install

### macOS

Make sure these are installed first:

- **Python 3.11 or newer** (`brew install python@3.12`)
- **Tesseract OCR** (`brew install tesseract`). The puzzle needs it to read the survey number and scale from the sheet header.

Then, in the project folder:

1. Create the virtual environment (only the first time):

       python3 -m venv venv

2. Activate it:

       source venv/bin/activate

3. Install the requirements:

       pip install -r requirements.txt

4. Start the server:

       python app.py

5. Open the browser and go to `http://localhost:5001`

Next time you only need steps 2, 4 and 5.

### Windows

1. Install Python 3.11+ (tick **Add python.exe to PATH**) and the Tesseract installer from UB Mannheim. Add `C:\Program Files\Tesseract-OCR` to PATH.
2. In Command Prompt, inside the project folder:

       python -m venv venv
       venv\Scripts\activate
       pip install -r requirements.txt
       python app.py

3. Open `http://localhost:5001`

---

## 2. How to use

1. Click **Choose Files** and add your survey images (TIFF, PNG or JPG). You can add many files and it will process them all at once, depending on how much RAM your computer has.
2. Click **Clean images**. Each file gets a progress bar.
3. Check the colour of each file when it finishes:
   - **Green**: cleaned well
   - **Yellow**: might be wrong, please check it
   - **Red**: looks completely wrong
4. Click **Review** to check the cleaned survey. If something is wrong we can fix it with the tools:
   - **Brush erase**: paints white over unwanted marks
   - **Add line**: draws a line
   - **Add text**: types a plot number where we click
   - **Magic line**: drag a box around a gap and it joins the broken line
   - **Magic erase**: click a number or a small piece of line to remove it
   - **Show original**: opens the original in a small window so we can compare
5. Press **Save edits** (Ctrl/Cmd + S) so the changes are kept.
6. Now we can either **Download final TIFF** or **Approve (add to puzzle)**.

---

## 3. Puzzle feature

### Getting the surveys in

- Open **Approved & Puzzle** from the left side.
- Every approved survey comes automatically into the **tray** on the left (the first time it may take a moment to prepare the pieces). From the tray we can drag it onto the board, or double-click to drop it in the middle.
- **Auto-arrange pieces** is optional. It places the pieces for us, but first we have to open the "Automatic match" box and press **Build puzzle**.

### Moving the pieces

There are 2 modes:

1. **Snap Edge**: the piece automatically joins to its neighbouring place.
2. **Normal Mode**: move it around manually. Untick "Snap edges" to use this, or hold **Alt / Option** to use it temporarily.

The plot can be moved sideways, up and down, and rotated 360 degrees.

### Other features

- **Plot labels**: inside the puzzle every plot number is marked with its survey number, for example "11A" of survey 2 becomes "S2-11A", so we know which survey it is from.
- **Lock / Unlock**: we can lock a single survey or the whole board, so nothing moves by accident.
- **Show Inner Lines**: when this is on, the inner boundaries also show in the puzzle. It also decides whether the download has the inner lines.
- **New Colours**: changes the colour of every survey.
- **Background image**: we can add a background image to help build the puzzle.
- **Auto-size to neighbours**: tries different sizes for the selected plot and keeps the one that matches its neighbouring plots best.
- **Auto-size chain from here**: treats the selected plot as correct, then fits the plots touching it, then the ones touching those, and so on.
- **Pin-fit to background**: puts a plot at a specific area of the background. First select the plot and add a background image, then:
  1. Click a point on the plot (for example a corner)
  2. Click the same place on the background
  3. Click a second point on the plot (far from the first one)
  4. Click the same place on the background

  The plot is moved, rotated and resized to match. `Ctrl/Cmd + Z` undoes it. Press `Esc` to cancel.

### Downloading

The joined puzzle can be downloaded as **PNG, PDF or TIF**.

---

## Good to know

- Approved surveys are saved in the `approved_images` folder, so they stay even after restarting the app.
- Cleaning sessions are only kept for 3 hours and are lost when the app is stopped, so download or approve before closing.
- If the computer is slow while cleaning, start the app with fewer parallel jobs: `PLOT_WORKERS=1 python app.py`