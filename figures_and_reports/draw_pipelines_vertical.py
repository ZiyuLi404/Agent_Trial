#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vertical (portrait) pipeline diagrams for Method A and Method B.
Same style as the horizontal versions.
Outputs methodA_pipeline_vertical.png and methodB_pipeline_vertical.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAVY="#122A4A"; BLUE="#2F6FB5"; TEAL="#1F9E8F"; GREY="#555B66"
LBLUE="#EDF3FA"; LTEAL="#E6F4F1"; LGREY="#F1F4F8"; BORDER="#B6C2D2"
WHITE="#FFFFFF"; AMBER="#E0A93B"; LAMBER="#FFF7E8"; DAMBER="#9A6B12"


def render(out, title_lines, subtitle, steps, supports, support_label, readout):
    fig, ax = plt.subplots(figsize=(8.6, 13.6), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 150); ax.axis("off")

    def box(x, y, w, h, title, lines, fc, ec, tc=NAVY, fs_t=13, fs_b=11, lw=2):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.6,rounding_size=2.2",
                     fc=fc, ec=ec, lw=lw, zorder=3))
        cx = x + w/2
        if title:
            ax.text(cx, y+h-3.2, title, ha="center", va="top", fontsize=fs_t,
                    fontweight="bold", color=tc, zorder=4)
        if lines:
            ax.text(cx, y+h-(7.0 if title else 3.0), "\n".join(lines), ha="center",
                    va="top", fontsize=fs_b, color=GREY, zorder=4, linespacing=1.5)

    def arrow(x1, y1, x2, y2, color=NAVY, lw=2.4, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=18, lw=lw, color=color, zorder=2,
                     shrinkA=0, shrinkB=0, linestyle=ls))

    # title + subtitle
    ty = 147
    for tl in title_lines:
        ax.text(4, ty, tl, ha="left", va="top", fontsize=16,
                fontweight="bold", color=NAVY)
        ty -= 5.0
    ax.text(4, ty-0.4, subtitle, ha="left", va="top", fontsize=11, color=GREY)

    # main vertical column
    colx, colw, h = 9, 42, 14
    ys = [116, 96, 76, 56, 36]
    for (t, lines, fc, ec, tc), y in zip(steps, ys):
        box(colx, y, colw, h, t, lines, fc, ec, tc)
    for top_y in ys[:-1]:           # arrows downward
        arrow(colx+colw/2, top_y, colx+colw/2, top_y-6)

    # support boxes to the right of step 2 (box2 spans y 96-110, center 103)
    sx, sw = 58, 34
    sy = [104, 88]
    if support_label:
        ax.text(sx+sw/2, sy[0]+12.6, support_label, ha="center", va="bottom",
                fontsize=10.5, style="italic", color=BLUE)
    for (t, lines), y in zip(supports, sy):
        box(sx, y, sw, 11, t, lines, WHITE, BORDER, tc=BLUE,
            fs_t=11, fs_b=9.8, lw=1.6)
        arrow(sx, y+5.5, colx+colw, 103, color=BLUE, lw=1.6, ls=(0,(4,3)))

    # how-to-read box at the bottom, full width
    rx, ry, rw, rh = 9, 13, 83, 15
    box(rx, ry, rw, rh, readout[0], readout[1:], LAMBER, AMBER, tc=DAMBER,
        fs_t=12.5, fs_b=11)
    arrow(colx+colw/2, ys[-1], colx+colw/2, ry+rh, color=AMBER, lw=1.8)

    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white", pad_inches=0.25)
    print("saved", out)


# ---------------- Method A ----------------
render(
    "methodA_pipeline_vertical.png",
    ["Method A — Categorical (ICD-10)", "Version-Drift Detection"],
    "Free-text diagnoses become standard codes; versions compared as distributions.",
    [
        ("1 · Model Outputs",
         ["Each model's diagnoses", "Many cases × 10 runs  (free text)"], LGREY, GREY, NAVY),
        ("2 · Map to ICD-10",
         ["An LLM turns each free-text", "diagnosis into one ICD-10 code"], LBLUE, BLUE, NAVY),
        ("3 · Code Distribution",
         ["How often each code appears", "→ one distribution per model"], LBLUE, BLUE, NAVY),
        ("4 · Compare Models",
         ["Distance between two distributions", "(KL / JS divergence)"], LTEAL, TEAL, NAVY),
        ("5 · Output",
         ["Similarity score for each pair of models"], LTEAL, TEAL, NAVY),
    ],
    [("ICD-10 Dictionary", ["Suggests candidate", "codes to the LLM"]),
     ("Cache", ["Reuse past mappings"])],
    None,
    ["How to Read It",
     "Same model, two runs  →  baseline noise  (similarity ≈ 0.82)",
     "Different models  →  real change  (similarity ≈ 0.74–0.80)"],
)

# ---------------- Method B ----------------
render(
    "methodB_pipeline_vertical.png",
    ["Method B — Embedding Similarity", "Version-Drift Detection"],
    "Each output becomes a vector; versions compared by how close the vectors are.",
    [
        ("1 · Model Outputs",
         ["Each model's dialogues", "Many cases × 10 runs  (free text)"], LGREY, GREY, NAVY),
        ("2 · Text → Vector",
         ["An embedding model turns each", "text into a vector (Qwen embedding)"], LBLUE, BLUE, NAVY),
        ("3 · Average per Case",
         ["Average the 10 vectors for a", "model & case → one vector"], LBLUE, BLUE, NAVY),
        ("4 · Compare Models",
         ["Closeness of two vectors", "(cosine similarity)"], LTEAL, TEAL, NAVY),
        ("5 · Output",
         ["Similarity score per pair  +  changed cases"], LTEAL, TEAL, NAVY),
    ],
    [("Full Conversation", ["Captures how the", "doctor works"]),
     ("Diagnosis Only", ["Captures the", "final answer"])],
    "Two text views",
    ["How to Read It",
     "Higher similarity  →  more alike   (flash vs pro ≈ 0.98)",
     "Lower similarity  →  bigger change   (qwen vs gpt ≈ 0.89)"],
)
