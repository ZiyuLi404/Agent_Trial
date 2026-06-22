#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline diagram for Method A — ICD-10 categorical version-drift detection.
Mirrors result_categorize/icd10_categorize_compare.py.
Outputs methodA_pipeline.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# theme
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
    return (x, y, w, h)

def arrow(x1, y1, x2, y2, color=NAVY, lw=2.4, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=20, lw=lw, color=color, zorder=2,
                 shrinkA=0, shrinkB=0))

# title
ax.text(2, 58.5, "Method A — Categorical (ICD-10) Version-Drift Detection",
        ha="left", va="top", fontsize=17, fontweight="bold", color=NAVY)
ax.text(2, 53.8, "Free-text diagnoses are collapsed to standardized ICD-10 codes, "
        "then versions are compared as distributions.",
        ha="left", va="top", fontsize=11.5, color=GREY)

# ---- main flow row (y ~ 30) ----
yb = 30; h = 15
# 1 input
box(1.5, yb, 17, h, "1 · Model Outputs",
    ["Each model's diagnoses", "", "Many cases × 10 runs", "(free text)"],
    fc=LGREY, ec=GREY, tc=NAVY)
# 2 mapping
box(22, yb, 19, h, "2 · Map to ICD-10",
    ["An LLM turns each", "free-text diagnosis", "into one standard", "ICD-10 code"],
    fc=LBLUE, ec=BLUE)
# 3 distribution
box(44.5, yb, 18, h, "3 · Code Distribution",
    ["How often each", "code appears", "→ one distribution", "per model"],
    fc=LBLUE, ec=BLUE)
# 4 divergence
box(66, yb, 18, h, "4 · Compare Models",
    ["Distance between", "two distributions", "", "(KL / JS divergence)"],
    fc=LTEAL, ec=TEAL, tc=NAVY)
# 5 output
box(87.5, yb, 11.5, h, "5 · Output",
    ["Similarity score", "for each pair", "of models"],
    fc=LTEAL, ec=TEAL, tc=NAVY)

# arrows between main boxes
for x1, x2 in [(18.5,22),(41,44.5),(62.5,66),(84,87.5)]:
    arrow(x1, yb+h/2, x2, yb+h/2)

# ---- support row under box 2 (LLM helpers) ----
ys = 9; hs = 13
box(20.5, ys, 12.5, hs, "ICD-10 Dictionary",
    ["Suggests candidate", "codes to the LLM"],
    fc=WHITE, ec=BORDER, tc=BLUE, fs_t=10.8, fs_b=9.8, lw=1.6)
box(34.5, ys, 9.5, hs, "Cache",
    ["Reuse past", "mappings"],
    fc=WHITE, ec=BORDER, tc=BLUE, fs_t=10.8, fs_b=9.8, lw=1.6)
# dashed arrows up into mapping box
for sx in (26.7, 39.2):
    ax.add_patch(FancyArrowPatch((sx, ys+hs), (sx, yb), arrowstyle="-|>",
                 mutation_scale=14, lw=1.6, color=BLUE, ls=(0,(4,3)), zorder=2))

# ---- readout under box 4/5 ----
yr = 9; hr = 13
box(64.5, yr, 34.5, hr, "How to Read It",
    ["Same model, two runs  →  baseline noise  (similarity ≈ 0.82)",
     "Different models  →  real change  (similarity ≈ 0.74–0.80)"],
    fc="#FFF7E8", ec="#E0A93B", tc="#9A6B12", fs_t=11.5, fs_b=10.2, lw=1.8)
arrow(81, yb, 81, yr+hr, color="#E0A93B", lw=1.8, style="-|>")

plt.tight_layout()
fig.savefig("methodA_pipeline.png", bbox_inches="tight", facecolor="white", pad_inches=0.25)
print("saved methodA_pipeline.png")
