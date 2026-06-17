# -*- coding: utf-8 -*-
"""把 full_dialogue embedding 相似度分析渲染成多页 PDF 报告。

数据来源: embedding_full_dialogue_results/  (Qwen3-Embedding-4B, 9 cases × 6 组 × 10 runs)
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager as fm

# ── 中文字体 ──
FONT_PATH = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
fm.fontManager.addfont(FONT_PATH)
CJK = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["font.family"] = ["DejaVu Sans", CJK]
plt.rcParams["axes.unicode_minus"] = False

RES = sys.argv[1] if len(sys.argv) > 1 else "embedding_full_dialogue_results"
EMB_MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen3-Embedding-4B"

# ── 读数据 ──
gm = pd.read_csv(os.path.join(RES, "mean_group_similarity_matrix.csv"), index_col=0)
mm = pd.read_csv(os.path.join(RES, "mean_model_similarity_matrix.csv"), index_col=0)
gsum = pd.read_csv(os.path.join(RES, "summary_group_pairwise_similarities.csv"))
msum = pd.read_csv(os.path.join(RES, "summary_model_similarities.csv"))

GROUPS = list(gm.columns)
SHORT = {"Qwen_plus_turbo_1": "qwen_1", "Qwen_plus_turbo_2": "qwen_2",
         "deepseek_flash_1": "flash_1", "deepseek_flash_2": "flash_2",
         "deepseek_pro_1": "pro_1", "deepseek_pro_2": "pro_2"}


def model_of(g):
    g = g.lower()
    return "qwen" if "qwen" in g else ("flash" if "flash" in g else "pro")


# 噪声地板 = 同模型两次重复; 信号 = 跨模型
nf_pairs, cross_pairs = [], []
for a, b in itertools.combinations(GROUPS, 2):
    v = gm.loc[a, b]
    (nf_pairs if model_of(a) == model_of(b) else cross_pairs).append(v)
NF = float(np.mean(nf_pairs))
CROSS = float(np.mean(cross_pairs))
# 用 1-相似度 看可分性
NF_gap, CROSS_gap = 1 - NF, 1 - CROSS
SNR = CROSS_gap / NF_gap

# 颜色
C_HEAD = "#3b4252"
C_STRIPE = "#f4f5f7"
C_NF = "#dcefe0"     # 绿: 噪声地板
C_BEST = "#dde7f5"


def draw_table(ax, col_labels, cell_text, row_colors=None, col_widths=None,
               fontsize=8.5, scale_y=1.5, cell_colors=None):
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center",
                   loc="upper center", colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, scale_y)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d0d3d8")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(C_HEAD)
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(cell.get_height() * 1.15)
        else:
            fc = None
            if cell_colors is not None:
                fc = cell_colors.get((r - 1, c))
            if fc is None:
                rc = (row_colors or {}).get(r - 1)
                fc = rc if rc else ("white" if (r % 2) else C_STRIPE)
            cell.set_facecolor(fc)
        if c == 0 and r > 0:
            cell.set_text_props(ha="left")
            cell._loc = "left"
    return tbl


pdf_path = os.path.join(RES, "embedding_full_dialogue_report.pdf")
_page = [0]


def _save(pdf, fig):
    _page[0] += 1
    fig.savefig(f"/tmp/embrep_p{_page[0]}.png", dpi=110)
    pdf.savefig(fig)


with PdfPages(pdf_path) as pdf:
    # ───────── 第 1 页：标题 + KPI + 结论 + 6×6 热力图 ─────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("整段对话 Embedding 相似度分析", x=0.06, y=0.965, ha="left",
                 fontsize=20, fontweight="bold")
    fig.text(0.06, 0.928,
             f"来源 {RES}  ·  模型 {EMB_MODEL}  ·  9 cases × 6 组 × 10 runs  ·  "
             "对每组取 10 段 full_dialogue 编码后求平均向量  ·  度量 = 余弦相似度",
             fontsize=8.0, color="#555")

    stats = [
        (f"{NF:.3f}", "噪声地板 (同模型重跑)", "#2c7a3f"),
        (f"{CROSS:.3f}", "跨模型平均相似度 (信号)", "#b06a00"),
        (f"{CROSS_gap:.3f}", "跨模型平均距离 (1-相似度)", "#b06a00"),
        (f"{SNR:.1f}×", "信噪比 (跨模型距离/噪声距离)", "#2c7a3f"),
    ]
    for i, (val, lab, col) in enumerate(stats):
        x = 0.06 + i * 0.235
        ax = fig.add_axes([x, 0.79, 0.21, 0.10]); ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                     facecolor="#f4f5f7", edgecolor="#d0d3d8", lw=0.6))
        ax.text(0.5, 0.62, val, ha="center", va="center", fontsize=18,
                fontweight="bold", color=col)
        ax.text(0.5, 0.2, lab, ha="center", va="center", fontsize=7.0, color="#555")

    axc = fig.add_axes([0.06, 0.63, 0.88, 0.13]); axc.axis("off")
    axc.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axc.transAxes,
                  facecolor="#eef6f0", edgecolor="#9ccaa8", lw=0.8))
    axc.text(0.02, 0.85, "结论：噪声地板高且稳定，跨模型差异清晰可辨", fontsize=11,
             fontweight="bold", color="#2c7a3f", va="top")
    axc.text(0.02, 0.58,
             f"同一模型重跑两次的相似度（噪声地板）≈{NF:.3f}（pro 0.986 / flash 0.985 / qwen 0.980），非常高且稳定；\n"
             f"跨模型相似度 ≈{CROSS:.3f}，明显更低。以“距离=1−相似度”衡量，跨模型距离是噪声地板的约 {SNR:.1f}×，\n"
             "差异真实可辨。还能看出模型家族层级：flash 与 pro 同属 deepseek，彼此最像（0.978）；qwen 离两者都远。",
             fontsize=8.2, color="#333", va="top")

    # 6×6 热力图
    fig.text(0.06, 0.555, "组级 6×6 余弦相似度方阵（对角线块 = 同模型两次重复 = 噪声地板）",
             fontsize=12, fontweight="bold")
    axh = fig.add_axes([0.18, 0.07, 0.64, 0.45])
    M = gm.values.astype(float)
    im = axh.imshow(M, cmap="RdYlGn", vmin=0.90, vmax=1.0, aspect="auto")
    labels = [SHORT[g] for g in GROUPS]
    axh.set_xticks(range(len(GROUPS))); axh.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    axh.set_yticks(range(len(GROUPS))); axh.set_yticklabels(labels, fontsize=9)
    for i in range(len(GROUPS)):
        for j in range(len(GROUPS)):
            axh.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                     fontsize=8.5, color="#222")
    # 给同模型 2×2 块描边
    for k in range(0, 6, 2):
        axh.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 2, 2, fill=False,
                      edgecolor="#1b5e20", lw=2.2))
    cb = fig.colorbar(im, ax=axh, fraction=0.046, pad=0.03)
    cb.set_label("余弦相似度", fontsize=8)
    fig.text(0.5, 0.035, "绿框 = 同模型重跑（噪声地板，最亮）；框外越偏黄/红 = 跨模型差异越大",
             ha="center", fontsize=8, color="#666")
    _save(pdf, fig); plt.close(fig)

    # ───────── 第 2 页：模型层方阵 + 两两汇总表 + 柱状图 ─────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("模型层对比  与  两两相似度明细", fontsize=15, fontweight="bold", y=0.96)

    # 模型 3×3 表
    fig.text(0.06, 0.89, "模型层 3×3 相似度方阵（合并两次重复）", fontsize=12, fontweight="bold")
    axm = fig.add_axes([0.06, 0.66, 0.42, 0.18])
    MODELS = list(mm.columns)
    mheaders = [""] + MODELS
    mrows = [[r] + [f"{mm.loc[r, c]:.3f}" for c in MODELS] for r in MODELS]
    draw_table(axm, mheaders, mrows, fontsize=9.5, scale_y=2.2)

    # 模型层柱状图: 跨模型相似度
    axb = fig.add_axes([0.56, 0.62, 0.38, 0.24])
    mc = msum[msum["comparison"].str.contains("_vs_")].copy()
    axb.bar(range(len(mc)), mc["mean_case_level_similarity"], color="#3a6ea5")
    axb.set_xticks(range(len(mc)))
    axb.set_xticklabels(mc["comparison"].str.replace("_vs_", "\nvs\n"), fontsize=7.5)
    axb.set_ylim(0.90, 1.0); axb.set_ylabel("余弦相似度", fontsize=8)
    axb.set_title("跨模型相似度", fontsize=10)
    axb.grid(axis="y", ls=":", alpha=0.4)
    for i, v in enumerate(mc["mean_case_level_similarity"]):
        axb.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=7.5)

    # 组级两两汇总（按相似度排序，标出噪声地板行）
    fig.text(0.06, 0.55, "组级两两相似度（升序；绿 = 同模型重跑 = 噪声地板）", fontsize=12, fontweight="bold")
    axt = fig.add_axes([0.06, 0.06, 0.88, 0.46])
    gs = gsum.sort_values("mean_case_level_similarity").reset_index(drop=True)
    headers = ["对比", "平均相似度", "标准差", "最小", "最大", "类型"]
    rows, rcolors = [], {}
    for i, r in enumerate(gs.itertuples()):
        a, b = r.comparison.split("_vs_")
        same = model_of(a) == model_of(b)
        nm = f"{SHORT.get(a,a)} vs {SHORT.get(b,b)}"
        rows.append([nm, f"{r.mean_case_level_similarity:.4f}", f"{r.std:.4f}",
                     f"{r.min:.4f}", f"{r.max:.4f}", "噪声地板" if same else "跨模型"])
        if same:
            rcolors[i] = C_NF
    draw_table(axt, headers, rows, row_colors=rcolors,
               col_widths=[0.28, 0.16, 0.13, 0.13, 0.13, 0.14], fontsize=8.5, scale_y=1.6)
    _save(pdf, fig); plt.close(fig)

    # ───────── 第 3 页：方法说明 + 与字符串分桶法对比 ─────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("方法说明  ·  与诊断字符串分桶法的对比", x=0.06, y=0.96, ha="left",
                 fontsize=15, fontweight="bold")

    axm1 = fig.add_axes([0.06, 0.62, 0.88, 0.27]); axm1.axis("off")
    axm1.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axm1.transAxes,
                   facecolor="#f4f5f7", edgecolor="#d0d3d8", lw=0.6))
    axm1.text(0.02, 0.9, "本方法（embedding）流程", fontsize=11, fontweight="bold",
              color="#3b4252", va="top")
    axm1.text(0.02, 0.66,
              "1. 每个 case、每个组取 10 段 full_dialogue（完整医患问诊全过程，约 6000 字符/段）。\n"
              f"2. 用 {EMB_MODEL} 把每段对话编码为向量，组内求平均并归一化，得到“该组在该 case 的代表向量”。\n"
              "3. 计算各组两两余弦相似度，再对 9 个 case 求平均，得到相似度方阵。\n"
              "4. 同模型两次重复(_1/_2)的相似度 = 噪声地板；不同模型之间 = 信号。",
              fontsize=8.5, color="#333", va="top")

    fig.text(0.06, 0.55, "两种方法对比", fontsize=12, fontweight="bold")
    axc2 = fig.add_axes([0.06, 0.26, 0.88, 0.26])
    headers = ["维度", "诊断字符串 + exact 分桶 + JS 散度", "整段对话 + embedding + 余弦(本法)"]
    rows = [
        ["比较对象", "仅最终诊断字符串", "整段问诊对话(信息量大)"],
        ["量化方式", "exact 精确匹配分桶", "语义向量"],
        ["度量", "JS 散度(越大越不同)", "余弦相似度(越大越像)"],
        ["对文字外壳", "敏感(大小写/标点/后缀都算不同桶)", "不敏感(只看语义)"],
        ["主要问题", "default 自由文本→大量碎桶→假性高散度", "规避了文字外壳噪声"],
    ]
    draw_table(axc2, headers, rows, col_widths=[0.16, 0.42, 0.42], fontsize=8.5, scale_y=1.9)

    axc3 = fig.add_axes([0.06, 0.06, 0.88, 0.16]); axc3.axis("off")
    axc3.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axc3.transAxes,
                   facecolor="#eef6f0", edgecolor="#9ccaa8", lw=0.8))
    axc3.text(0.02, 0.85, "要点", fontsize=10.5, fontweight="bold", color="#2c7a3f", va="top")
    axc3.text(0.02, 0.6,
              "embedding 法给出的噪声地板(同模型重跑≈0.985)与跨模型信号(≈0.947)差距清晰稳定，且不受文字外壳噪声干扰；\n"
              "比之前在诊断字符串上做 exact 分桶 + JS 散度更干净，能更可靠地反映模型间的真实分歧，适合作为版本/能力变化的指纹基线。",
              fontsize=8.3, color="#333", va="top")
    _save(pdf, fig); plt.close(fig)

    d = pdf.infodict(); d["Title"] = "整段对话 Embedding 相似度分析报告"

print("WROTE", pdf_path)
