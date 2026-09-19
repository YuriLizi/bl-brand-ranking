"""Geometry QA for the generated deck.

LibreOffice is not installed on this machine, so slides cannot be rasterised for a visual
pass. This checks the defects that a visual pass would catch and that are decidable from
the layout itself:

  * shapes outside the slide canvas, or inside the 0.5" margin
  * overlapping text boxes
  * text boxes whose content is very unlikely to fit the box

Text fit is an estimate, not a render, so it reports "risk" rather than "failure".
"""
from __future__ import annotations

import sys
from collections import defaultdict

from pptx import Presentation
from pptx.util import Emu

EMU_IN = 914400.0
SLIDE_W, SLIDE_H = 13.333, 7.5
MARGIN = 0.5
# Rough average glyph width as a fraction of font size, for a proportional face.
CHAR_W_RATIO = 0.50
LINE_H_RATIO = 1.22


def inches(v) -> float:
    return (v or 0) / EMU_IN


def shape_box(sh):
    return (inches(sh.left), inches(sh.top), inches(sh.width), inches(sh.height))


def estimate_overflow(sh) -> tuple[bool, float]:
    """Crudely estimate whether the text needs more height than the box has."""
    if not sh.has_text_frame:
        return False, 0.0
    tf = sh.text_frame
    text = tf.text or ""
    if not text.strip():
        return False, 0.0

    x, y, w, h = shape_box(sh)
    if w <= 0 or h <= 0:
        return False, 0.0

    total_lines = 0.0
    max_pt = 0.0
    for para in tf.paragraphs:
        runs = para.runs
        if not runs:
            total_lines += 1
            continue
        pt = max((r.font.size.pt if r.font.size else 18) for r in runs)
        max_pt = max(max_pt, pt)
        chars = sum(len(r.text) for r in runs)
        char_w_in = pt * CHAR_W_RATIO / 72.0
        per_line = max(1.0, w / char_w_in) if char_w_in else 1.0
        total_lines += max(1.0, -(-chars // per_line))  # ceil

    needed = total_lines * (max_pt * LINE_H_RATIO / 72.0)
    return needed > h * 1.06, needed


def overlaps(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ox = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    oy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return ox * oy


def main(path: str) -> int:
    pres = Presentation(path)
    issues = defaultdict(list)

    for idx, slide in enumerate(pres.slides, start=1):
        texts = []
        for sh in slide.shapes:
            if sh.left is None or sh.top is None:
                continue
            x, y, w, h = shape_box(sh)

            if x < -0.01 or y < -0.01 or x + w > SLIDE_W + 0.01 or y + h > SLIDE_H + 0.01:
                issues[idx].append(
                    f"OUT OF BOUNDS: {sh.shape_type}, box=({x:.2f},{y:.2f},{w:.2f},{h:.2f})"
                )
            elif x < MARGIN - 0.11 or y < MARGIN - 0.41 or (x + w) > SLIDE_W - MARGIN + 0.11:
                issues[idx].append(
                    f"tight margin: box=({x:.2f},{y:.2f},{w:.2f},{h:.2f})"
                )

            if sh.has_text_frame and (sh.text_frame.text or "").strip():
                over, needed = estimate_overflow(sh)
                if over:
                    snippet = sh.text_frame.text.strip().replace("\n", " ")[:55]
                    issues[idx].append(
                        f"OVERFLOW RISK: needs ~{needed:.2f}in, box h={h:.2f}in :: {snippet!r}"
                    )
                texts.append((shape_box(sh), sh.text_frame.text.strip()[:32]))

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                area = overlaps(texts[i][0], texts[j][0])
                ai = texts[i][0][2] * texts[i][0][3]
                aj = texts[j][0][2] * texts[j][0][3]
                if area > 0.35 * min(ai, aj) and area > 0.06:
                    issues[idx].append(
                        f"TEXT OVERLAP {area:.2f}in^2: {texts[i][1]!r} / {texts[j][1]!r}"
                    )

    total = sum(len(v) for v in issues.values())
    n_slides = len(pres.slides._sldIdLst)
    print(f"Deck: {path}")
    print(f"Slides: {n_slides}   canvas: {SLIDE_W}x{SLIDE_H}in\n")
    if not total:
        print("Geometry QA: no issues found.")
        return 0
    for idx in sorted(issues):
        print(f"--- slide {idx} ---")
        for msg in issues[idx]:
            print("   ", msg)
    print(f"\n{total} item(s) to review.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "BL_MLOps_Presentation.pptx"))
