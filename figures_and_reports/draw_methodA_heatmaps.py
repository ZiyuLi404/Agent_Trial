#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Method A (categorical / ICD) result heatmaps, RdYlGn style matching Method B.
  1) Model axis  : flash vs pro, 4x4 JS similarity, same-model 2x2 blocks boxed.
  2) Prompt axis : 6x6, diagonal = within-prompt self-consistency (noise floor),
                   off-diagonal = cross-prompt similarity.
Outputs methodA_heatmap_model.png and methodA_heatmap_prompt.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

NAVY="#122A4A"; GREY="#555B66"; CMAP="RdYlGn"

def base_panel(ax, M, labels, title, vmin):
    n=len(labels)
    im=ax.imshow(M, cmap=CMAP, vmin=vmin, vmax=1.0, aspect="equal")
    ax.set_title(title, fontsize=14, fontweight="bold", color=NAVY, pad=10)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=10.5, color=GREY)
    ax.set_yticklabels(labels, fontsize=10.5, color=GREY)
    ax.set_xticks(np.arange(-.5,n,1), minor=True)
    ax.set_yticks(np.arange(-.5,n,1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="both", length=0)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    fontsize=10.5, color="#222",
                    fontweight="normal" if i==j else "bold")
    return im

# ----------------------------------------------------------------- model axis
df = pd.read_csv(
    "result_categorize/deepseek_flash_vs_pro/compare/folder_similarity_matrix_js.csv",
    index_col=0)
order = ["deepseek_flash_1","deepseek_flash_2","deepseek_pro_1","deepseek_pro_2"]
df = df.reindex(index=order, columns=order)
labels = ["DS-Flash 1","DS-Flash 2","DS-Pro 1","DS-Pro 2"]
M = df.values.astype(float)
off = M[~np.eye(4, dtype=bool)]
vmin = float(np.floor(off.min()*20)/20)

fig, ax = plt.subplots(figsize=(6.6, 6.2), dpi=200)
fig.suptitle("Method A — Categorical (ICD): model version",
             fontsize=15, fontweight="bold", color=NAVY, y=0.98)
im = base_panel(ax, M, labels, "DeepSeek: Flash vs Pro  ·  JS similarity", vmin)
for k in (0, 2):                       # box same-model 2x2 blocks
    ax.add_patch(Rectangle((k-0.5,k-0.5),2,2, fill=False,
                 edgecolor="#15366b", lw=2.6, zorder=5))
cb=fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=9)
cb.set_label("JS similarity (1 = identical)", fontsize=10, color=GREY)
fig.text(0.5, 0.015,
         "DS = DeepSeek.  Boxes = same model, two runs (noise floor ≈ 0.80–0.82).  "
         "Flash–Pro cross cells ≈ 0.74–0.80 → at the noise floor.",
         ha="center", fontsize=9.5, color=GREY, style="italic")
fig.savefig("methodA_heatmap_model.png", bbox_inches="tight",
            facecolor="white", pad_inches=0.3)
print("saved methodA_heatmap_model.png  vmin=", vmin)

# ---------------------------------------------------------------- prompt axis
d = json.load(open("results/prompts37_Junhan/analysis_kl_js.json"))
nf  = d["within_style_noise_floor_js"]          # divergence per prompt (diagonal)
cpm = d["cross_prompt_js_matrix"]               # pairwise divergence (off-diag)
porder = ["balanced_baseline","aggressive_efficiency_first",
          "conservative_safety_first","distinct_diagnosis","low_cost_testing"]
plabels = ["Balanced","Aggressive","Conservative","Distinct","Low-cost"]
n=len(porder)
P=np.zeros((n,n))
for i,a in enumerate(porder):
    for j,b in enumerate(porder):
        P[i,j] = 1.0 - (nf[a] if i==j else cpm[a][b])   # -> similarity
voff = float(np.floor(P.min()*20)/20)

fig, ax = plt.subplots(figsize=(7.4, 6.8), dpi=200)
fig.suptitle("Method A — Categorical (ICD): doctor prompt",
             fontsize=15, fontweight="bold", color=NAVY, y=0.98)
im = base_panel(ax, P, plabels, "Cross-prompt similarity  (deepseek-v4-pro)", voff)
for k in range(n):                      # outline diagonal = noise floor
    ax.add_patch(Rectangle((k-0.5,k-0.5),1,1, fill=False,
                 edgecolor="#15366b", lw=2.4, zorder=5))
cb=fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=9)
cb.set_label("similarity (1 = identical)", fontsize=10, color=GREY)
fig.text(0.5, 0.015,
         "Boxed diagonal = within-prompt self-consistency (noise floor).  "
         "Diagonal ≈ off-diagonal → the 5 prompts are hard to tell apart.",
         ha="center", fontsize=9.0, color=GREY, style="italic")
fig.savefig("methodA_heatmap_prompt.png", bbox_inches="tight",
            facecolor="white", pad_inches=0.3)
print("saved methodA_heatmap_prompt.png  vmin=", voff)
