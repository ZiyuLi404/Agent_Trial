#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replicate-level (each model × 2 runs) embedding-similarity heatmaps:
full conversation vs diagnosis only. Each model gets a coloured band on the
top/left axes so it's easy to tell models apart; same-model 2×2 blocks are
boxed (within-model noise floor). Shared RdYlGn colour scale for the cells.
Outputs embedding_similarity_heatmaps.png
"""
import zipfile, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

NAVY="#122A4A"; GREY="#555B66"

CANON = {
    "deepseek_flash_1":"Flash 1", "deepseek_flash_2":"Flash 2",
    "deepseek_pro_1":"Pro 1", "deepseek_pro_2":"Pro 2",
    "Qwen_plus_turbo_1":"Qwen 1", "Qwen_plus_turbo_2":"Qwen 2",
    "gpt_5_4_mini_1":"GPT-mini 1", "gpt_5_4_mini_2":"GPT-mini 2",
    "gpt_5_5_1":"GPT-5.5 1", "gpt_5_5_2":"GPT-5.5 2",
}
ORDER = ["Flash 1","Flash 2","Pro 1","Pro 2","Qwen 1","Qwen 2",
         "GPT-mini 1","GPT-mini 2","GPT-5.5 1","GPT-5.5 2"]
N = len(ORDER)
# one colour per model (pairs of adjacent runs)
MODELS = [("DS-Flash","#2F6FB5"), ("DS-Pro","#1F9E8F"), ("Qwen","#E0A93B"),
          ("GPT-mini","#C0504D"), ("GPT-5.5","#7E5FA4")]

def load_matrix(df):
    df.index = [CANON.get(i, i) for i in df.index]
    df.columns = [CANON.get(c, c) for c in df.columns]
    return df.reindex(index=ORDER, columns=ORDER).astype(float)

full = load_matrix(pd.read_csv(
    "embedding_full_dialogue_results_gpt_8b/mean_group_similarity_matrix.csv", index_col=0))
with zipfile.ZipFile("embedding_diagnosis_only_5models_results.zip") as zf:
    raw = zf.read("embedding_data_5models_results/mean_group_similarity_matrix.csv")
diag = load_matrix(pd.read_csv(io.BytesIO(raw), index_col=0))

cmap = "RdYlGn"
_off = np.concatenate([full.values[~np.eye(N, dtype=bool)],
                       diag.values[~np.eye(N, dtype=bool)]])
VMIN, VMAX = float(np.floor(np.nanmin(_off) * 20) / 20), 1.0

fig, axes = plt.subplots(1, 2, figsize=(17.5, 8.8), dpi=200)
fig.suptitle("Embedding similarity between runs  (5 models × 2 runs · 9 cases)",
             fontsize=20, fontweight="bold", color=NAVY, x=0.5, y=0.99)

BW = 0.9   # band width (in cells)

def panel(ax, M, title):
    im = ax.imshow(M.values, cmap=cmap, vmin=VMIN, vmax=VMAX, aspect="equal")
    ax.set_title(title, fontsize=16, fontweight="bold", color=NAVY, pad=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xticks(np.arange(-.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-.5, N, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="both", length=0)
    # cell values + small run number
    for i in range(N):
        for j in range(N):
            v = M.values[i, j]
            ax.text(j, i-0.16, "1.00" if i == j else f"{v:.2f}",
                    ha="center", va="center", fontsize=11,
                    color="#222", fontweight="normal" if i == j else "bold")
    # coloured model bands on top and left (drawn outside the grid)
    for m, (name, col) in enumerate(MODELS):
        x0 = 2*m - 0.5
        # top band
        ax.add_patch(Rectangle((x0, -1.5), 2, BW, facecolor=col,
                     edgecolor="white", lw=1.5, clip_on=False, zorder=6))
        ax.text(x0+1, -1.5+BW/2, name, ha="center", va="center", color="white",
                fontsize=11.5, fontweight="bold", clip_on=False, zorder=7)
        # left band
        ax.add_patch(Rectangle((-1.5, x0), BW, 2, facecolor=col,
                     edgecolor="white", lw=1.5, clip_on=False, zorder=6))
        ax.text(-1.5+BW/2, x0+1, name, ha="center", va="center", color="white",
                fontsize=11.5, fontweight="bold", rotation=90,
                clip_on=False, zorder=7)
        # run numbers inside each cell row/col band edge
    # run index ticks (1/2) just outside cells
    for k in range(N):
        r = "1" if k % 2 == 0 else "2"
        ax.text(k, -0.62, r, ha="center", va="center", fontsize=9, color=GREY)
        ax.text(-0.62, k, r, ha="center", va="center", fontsize=9, color=GREY)
    # box same-model 2x2 blocks -> within-model noise floor
    for k in range(0, N, 2):
        ax.add_patch(Rectangle((k-0.5, k-0.5), 2, 2, fill=False,
                     edgecolor="#15366b", lw=2.8, zorder=5))
    ax.set_xlim(-1.6, N-0.5); ax.set_ylim(N-0.5, -1.6)
    return im

panel(axes[0], full, "Full Conversation")
im = panel(axes[1], diag, "Diagnosis Only")

cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04)
cbar.set_label("cosine similarity", fontsize=13, color=GREY)
cbar.ax.tick_params(labelsize=11, color=GREY)

fig.text(0.5, 0.04,
         "Each model has its own colour band (DS = DeepSeek).  Dark-boxed 2×2 blocks = same model, "
         "two runs → noise floor (≈ 0.97–0.99).  Cells outside the boxes = cross-model similarity.",
         ha="center", fontsize=12, color=GREY, style="italic")

fig.savefig("embedding_similarity_heatmaps.png", bbox_inches="tight",
            facecolor="white", pad_inches=0.35)
print("saved embedding_similarity_heatmaps.png  vmin=", VMIN)
