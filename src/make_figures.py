"""Split a 1x4 four-panel diagnostic PNG into four panels and restack as 2x2.

Usage: python make_2x2.py <input.png> <output.png>
Panels are detected from blank runs; the figure-level suptitle is dropped
(the LaTeX caption states the dataset).
"""
import sys
import numpy as np
from PIL import Image


def split_panels(path):
    im = Image.open(path).convert("RGB")
    a = np.array(im)
    g = a.mean(axis=2)
    ink = g < 245
    h, w = ink.shape

    # --- vertical cuts: near the 1/4, 2/4, 3/4 positions pick the blankest column ---
    col_ink = ink.sum(axis=0)
    xs = [0]
    for k in (1, 2, 3):
        target = int(round(k * w / 4.0))
        lo, hi = max(1, target - 90), min(w - 2, target + 90)
        best = lo + int(np.argmin(col_ink[lo:hi]))
        # widen to the centre of the enclosing blank run
        a, b = best, best
        while a > lo and col_ink[a - 1] == 0:
            a -= 1
        while b < hi and col_ink[b + 1] == 0:
            b += 1
        xs.append((a + b) // 2)
    xs.append(w - 1)

    # --- top cut: first blank row band below the suptitle ---
    row_blank = ~ink.any(axis=1)
    y_top = 0
    i = 0
    while i < h:
        if row_blank[i]:
            j = i
            while j < h and row_blank[j]:
                j += 1
            if j - i >= 8 and i > 5:
                y_top = j
                break
            i = j
        else:
            i += 1

    # --- bottom: last row containing ink ---
    rows_ink = np.where(ink.any(axis=1))[0]
    y_bot = int(rows_ink[-1]) + 1

    panels = [im.crop((xs[k], y_top, xs[k + 1], y_bot)) for k in range(4)]
    return panels


def stack_2x2(panels, pad=24):
    w = max(p.width for p in panels)
    h = max(p.height for p in panels)
    canvas = Image.new("RGB", (2 * w + 3 * pad, 2 * h + 3 * pad), "white")
    pos = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for p, (cx, cy) in zip(panels, pos):
        # center each panel in its cell so axes align across the grid
        dx = (w - p.width) // 2
        dy = (h - p.height) // 2
        canvas.paste(p, (pad + cx * (w + pad) + dx, pad + cy * (h + pad) + dy))
    return canvas


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    ps = split_panels(src)
    print("panel sizes:", [p.size for p in ps])
    out = stack_2x2(ps)
    out.save(dst, dpi=(600, 600))
    print("saved", dst, out.size)
