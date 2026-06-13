"""
Generate all figures for the Phase 3 final report.

Reads result CSVs from ./outputs/ and writes publication-quality figures
to ./figures/.  All figures use shape + colour distinguishability so they
remain readable in greyscale / print.

Usage:
    python src/generate_figures.py

Inputs:
    - ./outputs/bm25_results.csv, bm25_per_query.csv
    - ./outputs/reranker_results.csv, reranker_per_query.csv
    - ./outputs/biencoder_results.csv, biencoder_per_query.csv  (if available)
    - ./outputs/rq2_analysis.csv
Outputs:
    - ./figures/aggregate_comparison.png
    - ./figures/subgroup_comparison.png
    - ./figures/scatter_bm25_vs_biencoder.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

RESULTS_DIR = "./outputs"
FIGURES_DIR = "./figures"

# ── Styling constants ─────────────────────────────────────────────────────────
SYSTEMS = {
    "BM25":         {"colour": "#4dac26", "hatch": "",   "marker": "o", "ls": "-"},
    "CrossEncoder": {"colour": "#d01c8b", "hatch": "//", "marker": "s", "ls": "--"},
    "BiEncoder":    {"colour": "#0571b0", "hatch": "xx", "marker": "^", "ls": "-."},
}
METRICS = ["RR@10", "nDCG@10", "R@100"]
METRIC_LABELS = {"RR@10": "RR@10 (MRR@10)", "nDCG@10": "nDCG@10", "R@100": "R@100"}


def _load_agg(path: str) -> dict:
    """Load aggregate metrics from a results CSV into a dict.

    Args:
        path: Path to results CSV (wide format with metric columns)
    Returns:
        Dict mapping metric name to float value, or empty dict if file missing
    """
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {m: float(df[m].iloc[0]) for m in METRICS if m in df.columns}


def _load_per_query(path: str) -> pd.DataFrame:
    """Load per-query results into wide-format DataFrame.

    Args:
        path: Path to per-query CSV (long or wide format)
    Returns:
        DataFrame indexed by qid with one column per metric, or empty DataFrame
    """
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "measure" in df.columns and "value" in df.columns:
        df = df.pivot_table(index="qid", columns="measure", values="value")
        df.columns.name = None
    else:
        df = df.set_index("qid")
    df.index = df.index.astype(str)
    return df


# ── Figure 1: Aggregate metric comparison ────────────────────────────────────

def fig_aggregate():
    """Bar chart of aggregate RR@10, nDCG@10, R@100 for all available systems."""
    data = {
        "BM25":         _load_agg(f"{RESULTS_DIR}/bm25_results.csv"),
        "CrossEncoder": _load_agg(f"{RESULTS_DIR}/reranker_results.csv"),
        "BiEncoder":    _load_agg(f"{RESULTS_DIR}/biencoder_results.csv"),
    }
    present = {k: v for k, v in data.items() if v}
    if not present:
        print("  No aggregate results found — skipping figure 1")
        return

    n_sys = len(present)
    n_met = len(METRICS)
    x = np.arange(n_met)
    width = 0.8 / n_sys

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (sys_name, metrics) in enumerate(present.items()):
        style = SYSTEMS[sys_name]
        vals = [metrics.get(m, 0) for m in METRICS]
        offset = (i - n_sys / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=sys_name,
            color=style["colour"],
            hatch=style["hatch"],
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_ylabel("Metric score")
    ax.set_title("Aggregate retrieval metrics on TREC-ToT 2025/dev1 (n=142 queries)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, max(v for m in present.values() for v in m.values()) * 1.25)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out = f"{FIGURES_DIR}/aggregate_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Figure 2: Subgroup comparison (RQ2 + biencoder) ──────────────────────────

def fig_subgroup():
    """Grouped bar chart of mean RR@10 per query-length subgroup for all systems."""
    rq2_path = f"{RESULTS_DIR}/rq2_analysis.csv"
    if not os.path.exists(rq2_path):
        print("  rq2_analysis.csv missing — skipping subgroup figure")
        return

    rq2 = pd.read_csv(rq2_path)
    rq2["qid"] = rq2["qid"].astype(str)

    # Merge biencoder if available
    bi_pq_path = f"{RESULTS_DIR}/biencoder_per_query.csv"
    if os.path.exists(bi_pq_path):
        bi_pq = _load_per_query(bi_pq_path)
        if "RR@10" in bi_pq.columns:
            bi_rr = bi_pq["RR@10"].rename("biencoder")
            rq2 = rq2.merge(bi_rr, left_on="qid", right_index=True, how="left")

    subgroups = [
        ("Long\n(> median)", rq2["long_query"] == True),
        ("Short\n(≤ median)", rq2["long_query"] == False),
    ]
    system_cols = {"BM25": "bm25", "CrossEncoder": "reranker", "BiEncoder": "biencoder"}
    present_sys = {k: v for k, v in system_cols.items() if v in rq2.columns}

    x = np.arange(len(subgroups))
    width = 0.8 / len(present_sys)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (sys_name, col) in enumerate(present_sys.items()):
        style = SYSTEMS[sys_name]
        vals = [rq2[mask][col].mean() for _, mask in subgroups]
        offset = (i - len(present_sys) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=sys_name,
            color=style["colour"],
            hatch=style["hatch"],
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in subgroups])
    ax.set_ylabel("Mean RR@10")
    ax.set_title("RQ2: Mean RR@10 by query length subgroup")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out = f"{FIGURES_DIR}/subgroup_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Figure 3: Scatter BM25 vs BiEncoder per-query ───────────────────────────

def fig_scatter_biencoder():
    """Per-query scatter: BM25 RR@10 (x) vs bi-encoder RR@10 (y)."""
    bm25_pq = _load_per_query(f"{RESULTS_DIR}/bm25_per_query.csv")
    bi_pq   = _load_per_query(f"{RESULTS_DIR}/biencoder_per_query.csv")

    if bm25_pq.empty or bi_pq.empty:
        print("  Per-query files missing — skipping scatter figure")
        return
    if "RR@10" not in bm25_pq.columns or "RR@10" not in bi_pq.columns:
        print("  RR@10 column missing — skipping scatter figure")
        return

    common = bm25_pq.index.intersection(bi_pq.index)
    bm25_rr = bm25_pq.loc[common, "RR@10"]
    bi_rr   = bi_pq.loc[common, "RR@10"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(bm25_rr, bi_rr, alpha=0.55, s=30,
               color="#0571b0", edgecolors="black", linewidths=0.3, marker="^")
    lim = max(bm25_rr.max(), bi_rr.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="y = x")
    ax.set_xlim(-0.01, lim)
    ax.set_ylim(-0.01, lim)
    ax.set_xlabel("BM25 RR@10")
    ax.set_ylabel("Bi-encoder RR@10")
    ax.set_title(f"Per-query RR@10: BM25 vs Bi-encoder (n={len(common)})\n"
                 "Points above diagonal = bi-encoder wins")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()

    out = f"{FIGURES_DIR}/scatter_bm25_vs_biencoder.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating report figures...")
    fig_aggregate()
    fig_subgroup()
    fig_scatter_biencoder()
    print("Done.")
