#!/usr/bin/env python3
"""Draw the repo banner. Pure PIL, no generated imagery — RyanAI Lab house style.

Concept: MTP speculative decoding (4 draft tokens) feeding into a vLLM async-scheduling
host/device race that produces a runaway repetition loop, and the one-flag fix that
closes it. Left = wordmark. Right = a request stream that hits an async/MTP race
(dashed loop) and breaks into a runaway repetition, with a single orange arrow for
the --no-async-scheduling fix that returns the stream to normal.

Install: `pip install Pillow` (any 10.x release; the script uses only stable
Image/ImageDraw/ImageFont APIs and pins no exact version). The wordmark and slab
labels use the macOS system fonts HelveticaNeue.ttc and Menlo.ttc; on other
platforms the font() helper silently falls back to ImageFont.load_default(), which
will change metrics and line breaks, so the checked-in banner.png was generated on
macOS — regenerate there for a pixel-identical result.
"""
import pathlib
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640
BG = (10, 10, 10)
WHITE = (245, 245, 245)
GREY = (140, 140, 140)
DIM = (70, 70, 70)
ORANGE = (255, 122, 26)
OUT = pathlib.Path(__file__).resolve().parents[1] / "docs/assets/banner.png"

HN = "/System/Library/Fonts/HelveticaNeue.ttc"
MENLO = "/System/Library/Fonts/Menlo.ttc"

def font(path, size, index=0):
    try:
        return ImageFont.truetype(path, size, index=index)
    except Exception:
        return ImageFont.load_default()

f_title = font(HN, 88, 7)     # Light
f_tag = font(HN, 30, 7)
f_mono = font(MENLO, 17)
f_label = font(MENLO, 15)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

# crosshair corners
for cx, cy in ((40, 40), (W - 40, H - 40)):
    d.line([(cx - 11, cy), (cx + 11, cy)], fill=DIM, width=1)
    d.line([(cx, cy - 11), (cx, cy + 11)], fill=DIM, width=1)

# 5x3 dot lattices
for ox, oy in ((72, 78), (1150, 536)):
    for r in range(3):
        for c in range(5):
            x, y = ox + c * 15, oy + r * 12
            d.ellipse([x, y, x + 1.6, y + 1.6], fill=DIM)

# wordmark
d.text((80, 196), "MTP RUNAWAY", font=f_title, fill=WHITE)
d.text((80, 288), "LOOP FIX", font=f_title, fill=WHITE)
d.text((82, 414), "vLLM async × MTP race on two Dell Pro Max with GB10.", font=f_tag, fill=WHITE)
d.text((82, 462), "vLLM · MTP4 · YaRN ×4 1M · --no-async-scheduling · ONE-FLAG FIX", font=f_mono, fill=GREY)

# right: three tiers as slabs (isometric-ish parallelograms), stacked
SX, SY = 860, 150
tiers = [("G1  REQ", "01"), ("G2  RACE", "02"), ("G3  FIXED", "03")]
slab_w, slab_h, skew, gap = 260, 58, 42, 52
boxes = []
for i, (name, num) in enumerate(tiers):
    y = SY + i * (slab_h + gap)
    poly = [(SX + skew, y), (SX + skew + slab_w, y), (SX + slab_w, y + slab_h), (SX, y + slab_h)]
    d.polygon(poly, outline=(96, 96, 96), width=1)
    d.text((SX + skew + 14, y + 18), name, font=f_label, fill=GREY)
    d.text((SX + skew + slab_w + 16, y + 20), num, font=f_label, fill=DIM)
    boxes.append(y)

# dashed down-flow (offload) on the left edge of the stack
def dashed(p0, p1, dash=6, gap_=6, fill=DIM):
    x0, y0 = p0; x1, y1 = p1
    L = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    n = int(L // (dash + gap_))
    for k in range(n):
        t0 = k * (dash + gap_) / L; t1 = (k * (dash + gap_) + dash) / L
        d.line([(x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0), (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1)], fill=fill, width=1)

lx = SX + 20
dashed((lx, boxes[0] + slab_h), (lx, boxes[1]))
dashed((lx, boxes[1] + slab_h), (lx, boxes[2]))
d.text((lx - 70, boxes[1] - 30), "async race", font=f_label, fill=DIM)

# the single orange fix arrow: from the race straight back to a clean request
rx = SX + skew + slab_w - 30
top, bot = boxes[0] + slab_h + 6, boxes[2] - 6
d.line([(rx, bot), (rx, top)], fill=ORANGE, width=2)
d.polygon([(rx, top - 2), (rx - 6, top + 10), (rx + 6, top + 10)], fill=ORANGE)
d.ellipse([rx - 5, bot - 5, rx + 5, bot + 5], fill=ORANGE)
d.text((rx + 14, (top + bot) // 2 - 8), "NO-ASYNC", font=f_label, fill=ORANGE)

# footer rule + labels
d.line([(80, 560), (W - 80, 560)], fill=(40, 40, 40), width=1)
d.text((80, 576), "RYANAI LAB", font=f_label, fill=GREY)
d.text((W - 80 - 250, 576), "DELL PRO MAX WITH GB10", font=f_label, fill=GREY)

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT)
print("wrote", OUT)
