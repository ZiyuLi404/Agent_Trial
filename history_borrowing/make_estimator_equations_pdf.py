#!/usr/bin/env python3
"""
Create a 2-page portrait PDF with only the equations/methods for:
  1. Original similarity-weighted history borrowing
  2. Bayesian/effective-sample-size history borrowing

No empirical results are included.
"""

import os
import tempfile
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch


OUTPUT_PDF = Path("history_borrowing/data/results/history_borrowing_two_estimators_equations.pdf")

BLUE = "#1f4e79"
LIGHT_BLUE = "#eaf2fb"
GRAY = "#5f6673"
TEXT = "#20242c"
BOX_EDGE = "#c7d3e3"
BOX_FILL = "#f7f9fc"


def new_page(title, subtitle):
    fig, ax = plt.subplots(figsize=(8.5, 11), facecolor="white")
    ax.axis("off")
    ax.text(
        0.07,
        0.955,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=21,
        color=BLUE,
        fontweight="bold",
    )
    ax.text(
        0.07,
        0.915,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color=GRAY,
    )
    return fig, ax


def wrapped(ax, x, y, text, width=84, fontsize=10.5, color=TEXT, weight="normal"):
    lines = []
    for para in text.split("\n"):
        if para.strip():
            lines.extend(textwrap.wrap(para, width=width))
        else:
            lines.append("")
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        linespacing=1.35,
    )


def equation_box(ax, x, y, w, h, title, body, fontsize=12):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.014",
        transform=ax.transAxes,
        facecolor=BOX_FILL,
        edgecolor=BOX_EDGE,
        linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.025,
        y + h - 0.04,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.5,
        color=BLUE,
        fontweight="bold",
    )
    ax.text(
        x + 0.025,
        y + h - 0.095,
        body,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color=TEXT,
        linespacing=1.45,
        family="DejaVu Sans Mono",
    )


def footer(fig, page):
    fig.text(
        0.07,
        0.03,
        page,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=GRAY,
    )
    fig.text(
        0.93,
        0.03,
        "History borrowing estimators",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=GRAY,
    )


def page_original(pdf):
    fig, ax = new_page(
        "Estimator 1: Original History Borrowing",
        "Similarity-weighted peer borrowing for estimating full-evaluation performance from a small current bucket.",
    )

    wrapped(
        ax,
        0.08,
        0.86,
        "Goal: estimate the target model's full-evaluation performance using its observed 25-case bucket and the observed bucket performances of behaviorally similar peer models.",
        width=84,
        fontsize=11,
    )

    equation_box(
        ax,
        0.08,
        0.63,
        0.84,
        0.17,
        "Step 1. Convert similarity to distance",
        "d(i,j) = 1 - sim(i,j)\n\n"
        "where sim(i,j) is the similarity between peer model i\n"
        "and target model j.",
    )

    equation_box(
        ax,
        0.08,
        0.41,
        0.84,
        0.18,
        "Step 2. Compute normalized peer weights",
        "w_ij = exp(-lambda * d(i,j))\n"
        "       / sum_{k != j} exp(-lambda * d(k,j))\n\n"
        "Weights sum to 1 across all peer models i != j.",
    )

    equation_box(
        ax,
        0.08,
        0.17,
        0.84,
        0.20,
        "Step 3. Borrowed performance estimate",
        "theta_borrowed_j = alpha * theta_j\n"
        "                 + (1 - alpha) * sum_{i != j} w_ij * theta_i\n\n"
        "theta_j: observed current-bucket performance of target j\n"
        "theta_i: observed current-bucket performance of peer i",
    )

    wrapped(
        ax,
        0.08,
        0.105,
        "Interpretation: alpha controls trust in the target model's own bucket. Larger alpha means less borrowing; smaller alpha means more peer borrowing. lambda controls how sharply borrowing favors close/similar peers.",
        width=86,
        fontsize=10,
        color=GRAY,
    )
    footer(fig, "Page 1 of 2")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_bayesian(pdf):
    fig, ax = new_page(
        "Estimator 2: Bayesian History Borrowing",
        "Sequential update estimator where previous models act as similarity-discounted historical prior evidence.",
    )

    wrapped(
        ax,
        0.08,
        0.86,
        "Goal: evaluate possible update orders model_t0 -> model_t1 -> model_t2 -> model_t3. At each step, previous models provide historical prior information and the current model provides current observed evidence.",
        width=84,
        fontsize=11,
    )

    equation_box(
        ax,
        0.08,
        0.62,
        0.84,
        0.18,
        "Step 1. Similarity-discounted effective prior sample size",
        "m_h = alpha * n_h * exp(-lambda * d(h,j))\n\n"
        "h: historical model from an earlier update step\n"
        "j: current target model\n"
        "n_h: historical sample size for model h",
    )

    equation_box(
        ax,
        0.08,
        0.42,
        0.84,
        0.16,
        "Step 2. Historical prior mean",
        "prior_mean_j = sum_h m_h * theta_h / sum_h m_h\n\n"
        "The prior is an effective-sample-size-weighted\n"
        "average of historical model performances.",
    )

    equation_box(
        ax,
        0.08,
        0.20,
        0.84,
        0.18,
        "Step 3. Posterior mean estimate",
        "posterior_j = (m * prior_mean_j + n_j * current_mean_j)\n"
        "              / (m + n_j)\n\n"
        "where m = sum_h m_h and n_j is current evidence size.",
    )

    wrapped(
        ax,
        0.08,
        0.125,
        "Binary-metric interpretation: for correctness/accuracy, this is equivalent to adding historical pseudo-correct counts m * prior_mean to current correct counts n_j * current_mean_j.",
        width=86,
        fontsize=10,
        color=GRAY,
    )
    wrapped(
        ax,
        0.08,
        0.075,
        "Important distinction: in this Bayesian estimator, larger alpha means stronger historical prior borrowing. This is the opposite direction from alpha in the original estimator.",
        width=86,
        fontsize=10,
        color=GRAY,
        weight="bold",
    )
    footer(fig, "Page 2 of 2")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT_PDF) as pdf:
        page_original(pdf)
        page_bayesian(pdf)
    print(f"Wrote PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
