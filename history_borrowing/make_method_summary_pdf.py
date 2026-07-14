#!/usr/bin/env python3
"""
Build a mentor-facing PDF summary of the history borrowing method.

The report is generated entirely from local repository files and existing
history_borrowing outputs. It uses matplotlib's PDF backend, so it does not
require pandoc or LaTeX.

Usage:
    python history_borrowing/make_method_summary_pdf.py
"""

import csv
import json
import math
import os
import tempfile
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch


BLUE = "#1f4e79"
LIGHT_BLUE = "#eaf2fb"
GREEN = "#2ca02c"
RED = "#d94f45"
GRAY = "#666666"
LIGHT_GRAY = "#f5f6f8"
TEXT = "#222222"

RESULTS_DIR = Path("history_borrowing/data/results")
COMPARE_DIR = RESULTS_DIR / "compare"
OUTPUT_PDF = RESULTS_DIR / "history_borrowing_method_summary.pdf"

ACCURACY_CSV = Path("history_borrowing/data/accuracy_by_25_cases.csv")
PARAMS_JSON = COMPARE_DIR / "borrow_params_similarity_comparison.json"
BAYESIAN_SUMMARY_CSV = (
    RESULTS_DIR
    / "bayesian_update_embedding_diagnosis_full_grid"
    / "mae_comparison_summary.csv"
)

FIGURES = {
    "embedding_dashboard": COMPARE_DIR / "embedding_diagnosis_visualization.png",
    "mae_comparison": (
        RESULTS_DIR
        / "bayesian_update_embedding_diagnosis_full_grid"
        / "mae_comparison.png"
    ),
    "similarity_relationship": (
        COMPARE_DIR
        / "similarity_relationship"
        / "embedding_vs_fingerprint_conversation_similarity.png"
    ),
}


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path):
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pct(x):
    return f"{100.0 * float(x):.1f}%"


def fmt(x, digits=4):
    if x is None or x == "":
        return ""
    try:
        val = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(val):
        return ""
    return f"{val:.{digits}f}"


def add_wrapped_text(ax, x, y, text, width=95, fontsize=10, color=TEXT,
                     weight="normal", line_spacing=1.25):
    wrapped_lines = []
    for para in text.split("\n"):
        if not para.strip():
            wrapped_lines.append("")
        else:
            wrapped_lines.extend(textwrap.wrap(para, width=width))
    ax.text(
        x,
        y,
        "\n".join(wrapped_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        linespacing=line_spacing,
    )


def new_page(title, subtitle=None):
    fig, ax = plt.subplots(figsize=(11, 8.5), facecolor="white")
    ax.axis("off")
    ax.text(
        0.05,
        0.95,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
        color=BLUE,
    )
    if subtitle:
        add_wrapped_text(ax, 0.05, 0.885, subtitle, width=120, fontsize=10.5, color=GRAY)
    return fig, ax


def add_footer(fig, page_label):
    fig.text(
        0.05,
        0.025,
        page_label,
        ha="left",
        va="bottom",
        fontsize=8,
        color=GRAY,
    )
    fig.text(
        0.95,
        0.025,
        "AgentClinic history_borrowing",
        ha="right",
        va="bottom",
        fontsize=8,
        color=GRAY,
    )


def add_box(ax, xy, width, height, title, body, facecolor=LIGHT_BLUE):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor="#c7d3e3",
        linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.02,
        y + height - 0.04,
        title,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        color=BLUE,
        fontweight="bold",
    )
    add_wrapped_text(ax, x + 0.02, y + height - 0.10, body, width=44, fontsize=9.2)


def add_table(ax, rows, col_labels, bbox, fontsize=8.5, header_color=BLUE):
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        bbox=bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dee8")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("white" if row % 2 else "#f8fafc")
    return table


def add_image_page(pdf, image_path, title, subtitle, page_label):
    fig, ax = new_page(title, subtitle)
    if image_path.exists():
        image = mpimg.imread(image_path)
        ax.imshow(image, extent=[0.05, 0.95, 0.08, 0.80], aspect="auto")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        add_wrapped_text(
            ax,
            0.08,
            0.70,
            f"Figure not found: {image_path}",
            width=100,
            fontsize=12,
            color=RED,
        )
    add_footer(fig, page_label)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_accuracy_table():
    rows = read_csv(ACCURACY_CSV)
    out = []
    for row in rows:
        out.append([
            row["model"],
            fmt(row.get("total")),
            fmt(row.get("bucket1")),
            fmt(row.get("bucket2")),
            fmt(row.get("bucket3")),
            fmt(row.get("bucket4")),
        ])
    return out


def build_source_summary(params):
    sources = [
        ("embedding_diagnosis", "Embedding diagnosis"),
        ("fingerprint_conversation", "Fingerprint conversation"),
        (
            "hybrid_0.7_embedding_diagnosis_0.3_fingerprint_conversation",
            "Hybrid 0.7 embedding + 0.3 fingerprint",
        ),
    ]
    rows = []
    for key, label in sources:
        mae_25 = params.get(f"global_mae_25_{key}")
        mae_b = params.get(f"global_mae_borrowed_{key}")
        imp = params.get(f"global_improvement_rel_{key}")
        rows.append([
            label,
            fmt(params.get(f"best_alpha_{key}"), 2),
            fmt(params.get(f"best_lambda_{key}"), 0),
            fmt(mae_25),
            fmt(mae_b),
            pct(imp) if imp is not None else "",
        ])
    return rows


def build_per_model_table(params, source="embedding_diagnosis"):
    per_model = params.get("per_model_mae", {}).get(source, {})
    rows = []
    for model, vals in per_model.items():
        mae_25 = vals.get("mae_25")
        mae_b = vals.get("mae_borrowed")
        rel = (mae_25 - mae_b) / mae_25 if mae_25 else None
        rows.append([
            model,
            fmt(mae_25),
            fmt(mae_b),
            pct(rel) if rel is not None else "",
        ])
    return rows


def build_bayesian_table():
    rows = read_csv(BAYESIAN_SUMMARY_CSV)
    out = []
    for row in rows:
        out.append([
            row.get("method", ""),
            fmt(row.get("global_mae")),
            row.get("source", ""),
            row.get("alpha", ""),
            row.get("lambda", ""),
        ])
    return out


def make_pdf():
    params = read_json(PARAMS_JSON)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(OUTPUT_PDF) as pdf:
        # Page 1: executive summary
        fig, ax = new_page(
            "History Borrowing for Model Performance Estimation",
            "A concise method summary for estimating performance of a model with limited current cases by borrowing information from behaviorally similar models.",
        )
        add_box(
            ax,
            (0.055, 0.58),
            0.42,
            0.22,
            "Core question",
            "When a new or updated clinical LLM has only a small evaluation bucket, can we reduce estimation error by borrowing from other models that behave similarly?",
        )
        add_box(
            ax,
            (0.525, 0.58),
            0.42,
            0.22,
            "Short answer",
            "Yes in the current offline experiment. The 25-case baseline MAE is 0.0800; the best original history borrowing source reduces global MAE to 0.0536.",
            facecolor="#edf7ed",
        )
        add_wrapped_text(
            ax,
            0.06,
            0.50,
            "The method is offline: it does not run new consultations. It reads precomputed per-bucket accuracies and precomputed model similarity matrices, then estimates each model's full-evaluation accuracy from a small bucket plus similarity-weighted peer information.",
            width=112,
            fontsize=11,
        )
        add_table(
            ax,
            build_source_summary(params),
            ["Similarity source", "alpha", "lambda", "25-case MAE", "Borrowed MAE", "Rel. improvement"],
            bbox=[0.06, 0.20, 0.88, 0.22],
            fontsize=8.5,
        )
        add_footer(fig, "Page 1 - Executive summary")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: data and pipeline
        fig, ax = new_page(
            "Data Inputs and Pipeline",
            "The current experiment uses four model families, four 25-case buckets, and multiple similarity sources.",
        )
        add_table(
            ax,
            build_accuracy_table(),
            ["Model", "Full eval", "Bucket 1", "Bucket 2", "Bucket 3", "Bucket 4"],
            bbox=[0.06, 0.58, 0.88, 0.22],
            fontsize=9,
        )
        boxes = [
            ("Ground truth runs", "100-case full-evaluation JSON files under history_borrowing/data/groundtruth."),
            ("Accuracy summary", "accuracy_summary.py converts correctness records into total and bucket accuracies."),
            ("Similarity matrices", "Embedding diagnosis, embedding conversation, fingerprint conversation, and hybrid matrices."),
            ("Borrowing estimate", "history_borrowing.py combines a model's current bucket with weighted peer buckets."),
            ("All orders", "run_all_orders.py evaluates all 24 bucket-to-model assignments."),
            ("Parameter fit", "train_borrow_params.py chooses one global alpha/lambda per similarity source by minimizing MAE."),
        ]
        x_positions = [0.06, 0.37, 0.68, 0.06, 0.37, 0.68]
        y_positions = [0.40, 0.40, 0.40, 0.18, 0.18, 0.18]
        for (title, body), x, y in zip(boxes, x_positions, y_positions):
            add_box(ax, (x, y), 0.25, 0.15, title, body, facecolor=LIGHT_GRAY)
        add_footer(fig, "Page 2 - Inputs and workflow")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: original method equations
        fig, ax = new_page(
            "Original History Borrowing Estimator",
            "This is the estimator implemented in history_borrowing.py and trained in train_borrow_params.py.",
        )
        equation_text = (
            "1. Convert similarity to distance:\n"
            "       d(i,j) = 1 - sim(i,j)\n\n"
            "2. Compute peer weights with a negative-distance softmax:\n"
            "       w_ij = exp(-lambda * d(i,j)) / sum_{k != j} exp(-lambda * d(k,j))\n\n"
            "3. Estimate model j's performance from its 25-case bucket and peer buckets:\n"
            "       theta_borrowed_j = alpha * theta_j + (1 - alpha) * sum_{i != j} w_ij * theta_i\n\n"
            "4. Select alpha and lambda by minimizing global mean absolute error:\n"
            "       MAE = mean_j |theta_borrowed_j - theta_full_j|"
        )
        add_wrapped_text(ax, 0.07, 0.78, equation_text, width=105, fontsize=12)
        add_box(
            ax,
            (0.08, 0.16),
            0.38,
            0.18,
            "Interpretation of alpha",
            "In the original estimator, larger alpha means trust the target model's own 25-case estimate more; smaller alpha means borrow more from peers.",
        )
        add_box(
            ax,
            (0.54, 0.16),
            0.38,
            0.18,
            "Interpretation of lambda",
            "Larger lambda concentrates borrowing on very similar peers. lambda = 0 gives equal peer weights after excluding the target model.",
        )
        add_footer(fig, "Page 3 - Original estimator")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4: results table
        fig, ax = new_page(
            "Main Result: MAE Reduction",
            "The primary metric is MAE to full-evaluation ground truth accuracy. Lower is better.",
        )
        add_table(
            ax,
            build_source_summary(params),
            ["Similarity source", "alpha", "lambda", "25-case MAE", "Borrowed MAE", "Rel. improvement"],
            bbox=[0.06, 0.58, 0.88, 0.22],
            fontsize=8.5,
        )
        add_table(
            ax,
            build_per_model_table(params, "embedding_diagnosis"),
            ["Model", "25-case MAE", "Borrowed MAE", "Rel. improvement"],
            bbox=[0.13, 0.24, 0.74, 0.22],
            fontsize=9,
        )
        add_wrapped_text(
            ax,
            0.08,
            0.15,
            "In this run, embedding diagnosis similarity gives the lowest global borrowed MAE among the compared sources. Fingerprint and hybrid similarities also improve the 25-case baseline, but by a smaller amount.",
            width=105,
            fontsize=10.5,
        )
        add_footer(fig, "Page 4 - Quantitative results")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 5: Bayesian extension
        fig, ax = new_page(
            "Bayesian Borrowing Extension",
            "The newer bayesian_update_all_orders.py script treats previous models as historical prior information in a possible update sequence.",
        )
        bayes_text = (
            "For each order model_t0 -> model_t1 -> model_t2 -> model_t3, previous models supply historical pseudo-sample size:\n\n"
            "       m_h = alpha * n_h * exp(-lambda * d(h,target))\n\n"
            "The historical prior mean is the effective-sample weighted mean:\n\n"
            "       prior_mean = sum_h m_h * theta_h / sum_h m_h\n\n"
            "The posterior estimate combines historical prior evidence and current observed evidence:\n\n"
            "       posterior = (m * prior_mean + n * current_mean) / (m + n)\n\n"
            "Important difference: in the Bayesian extension, larger alpha means stronger historical borrowing. In the original estimator, larger alpha means greater trust in the current model's own bucket."
        )
        add_wrapped_text(ax, 0.07, 0.80, bayes_text, width=105, fontsize=11.3)
        bayes_rows = build_bayesian_table()
        if bayes_rows:
            add_table(
                ax,
                bayes_rows,
                ["Method", "Global MAE", "Source/mode", "alpha", "lambda"],
                bbox=[0.13, 0.13, 0.74, 0.16],
                fontsize=9,
            )
        add_footer(fig, "Page 5 - Bayesian extension")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 6: dashboard figure
        add_image_page(
            pdf,
            FIGURES["embedding_dashboard"],
            "Visualization: Original History Borrowing",
            "Existing dashboard for embedding diagnosis similarity, showing before/after MAE across bucket orders and models.",
            "Page 6 - Original borrowing dashboard",
        )

        # Page 7: Bayesian comparison figure
        add_image_page(
            pdf,
            FIGURES["mae_comparison"],
            "Visualization: Baseline vs Original vs Bayesian Borrowing",
            "Comparison of 25-case only, original history borrowing, and Bayesian borrowing against full-evaluation ground truth.",
            "Page 7 - Method comparison",
        )

        # Page 8: similarity relationship
        add_image_page(
            pdf,
            FIGURES["similarity_relationship"],
            "Similarity Source Sanity Check",
            "Linear relationship between embedding conversation similarity and fingerprint conversation similarity over shared replicate labels.",
            "Page 8 - Similarity relationship",
        )

        # Page 9: caveats and next steps
        fig, ax = new_page(
            "Interpretation, Caveats, and Next Steps",
            "A concise checklist for discussing the method with a mentor.",
        )
        caveats = (
            "Interpretation:\n"
            "- The method estimates full-evaluation accuracy from limited current evidence plus behaviorally similar historical evidence.\n"
            "- The strongest current result is a reduction from 0.0800 MAE to about 0.0536 MAE using embedding diagnosis similarity.\n\n"
            "Caveats:\n"
            "- The analysis is offline and retrospective; it does not prove prospective trial performance.\n"
            "- The similarity matrices are themselves estimated artifacts and may encode model-family effects.\n"
            "- The current data is small: four model families and four 25-case buckets.\n"
            "- The original estimator and Bayesian extension use alpha differently, so alpha values should not be compared directly.\n\n"
            "Next steps:\n"
            "- Validate on more cases and more model versions.\n"
            "- Use held-out orders or held-out model families to test robustness.\n"
            "- Compare diagnosis-only, conversation-level, fingerprint, and hybrid similarity sources prospectively.\n"
            "- Consider uncertainty intervals, not just point-estimate MAE."
        )
        add_wrapped_text(ax, 0.07, 0.80, caveats, width=105, fontsize=11)
        add_footer(fig, "Page 9 - Caveats")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return OUTPUT_PDF


def main():
    output = make_pdf()
    print(f"Wrote PDF: {output}")


if __name__ == "__main__":
    main()
