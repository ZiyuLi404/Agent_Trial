#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline diagram for Method B — embedding-similarity version-drift detection.
Mirrors case_level_embedding_full_dialogue.py.
Outputs methodB_pipeline.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# theme (identical to Method A diagram)
NAVY="#122A4A"; BLUE="#2F6FB5"; TEAL="#1F9E8F"; GREY="#555B66"
LBLUE="#EDF3FA"; LTEAL="#E6F4F1"; LGREY="#F1F4F8"; BORDER="#B6C2D2"
WHITE="#FFFFFF"

fig, ax = plt.subplots(figsize=(15.5, 7.6), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

def box(x, y, w, h, title, lines, fc=WHITE, ec=BLUE, tc=NAVY, fs_t=12.5, fs_b=10.8, lw=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2.2",
                       fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(p)
    cx = x + w/2
    if title:
        ax.text(cx, y+h-3.6, title, ha="center", va="top", fontsize=fs_t,
                fontweight="bold", color=tc, zorder=4)
    if lines:
        ax.text(cx, y+h-(8.0 if title else 3.2), "\n".join(lines), ha="center",
                va="top", fontsize=fs_b, color=GREY, zorder=4, linespacing=1.6)

def arrow(x1, y1, x2, y2, color=NAVY, lw=2.4, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=20, lw=lw, color=color, zorder=2,
                 shrinkA=0, shrinkB=0))

# title
ax.text(2, 58.5, "Method B — Embedding Similarity for Version-Drift Detection",
        ha="left", va="top", fontsize=17, fontweight="bold", color=NAVY)
ax.text(2, 53.8, "Each output is turned into a vector; versions are compared by "
        "how close their vectors are.",
        ha="left", va="top", fontsize=11.5, color=GREY)

# ---- main flow row ----
yb = 30; h = 15
box(1.5, yb, 17, h, "1 · Model Outputs",
    ["Each model's dialogues", "", "Many cases × 10 runs", "(free text)"],
    fc=LGREY, ec=GREY, tc=NAVY)
box(22, yb, 19, h, "2 · Text → Vector",
    ["An embedding model", "turns each text", "into a vector", "(Qwen embedding)"],
    fc=LBLUE, ec=BLUE)
box(44.5, yb, 18, h, "3 · Average per Case",
    ["Average the 10 vectors", "for a model & case", "→ one vector", "per model"],
    fc=LBLUE, ec=BLUE)
box(66, yb, 18, h, "4 · Compare Models",
    ["Closeness of", "two vectors", "", "(cosine similarity)"],
    fc=LTEAL, ec=TEAL, tc=NAVY)
box(87.5, yb, 11.5, h, "5 · Output",
    ["Similarity score", "for each pair", "+ changed cases"],
    fc=LTEAL, ec=TEAL, tc=NAVY)

for x1, x2 in [(18.5,22),(41,44.5),(62.5,66),(84,87.5)]:
    arrow(x1, yb+h/2, x2, yb+h/2)

# ---- support row: two text views feed into step 2 ----
ys = 9; hs = 13
box(20.5, ys, 12.5, hs, "Full Conversation",
    ["Captures how the", "doctor works"],
    fc=WHITE, ec=BORDER, tc=BLUE, fs_t=10.8, fs_b=9.8, lw=1.6)
box(34.5, ys, 11, hs, "Diagnosis Only",
    ["Captures the", "final answer"],
    fc=WHITE, ec=BORDER, tc=BLUE, fs_t=10.8, fs_b=9.8, lw=1.6)
ax.text(33, ys+hs+1.6, "Two text views", ha="center", va="bottom",
        fontsize=10, color=BLUE, style="italic")
for sx in (26.7, 40):
    ax.add_patch(FancyArrowPatch((sx, ys+hs), (sx, yb), arrowstyle="-|>",
                 mutation_scale=14, lw=1.6, color=BLUE, ls=(0,(4,3)), zorder=2))

# ---- readout ----
yr = 9; hr = 13
box(64.5, yr, 34.5, hr, "How to Read It",
    ["Higher similarity  →  more alike   (flash vs pro ≈ 0.98)",
     "Lower similarity  →  bigger change   (qwen vs gpt ≈ 0.89)"],
    fc="#FFF7E8", ec="#E0A93B", tc="#9A6B12", fs_t=11.5, fs_b=10.2, lw=1.8)
arrow(81, yb, 81, yr+hr, color="#E0A93B", lw=1.8, style="-|>")

plt.tight_layout()
fig.savefig("methodB_pipeline.png", bbox_inches="tight", facecolor="white", pad_inches=0.25)
print("saved methodB_pipeline.png")
