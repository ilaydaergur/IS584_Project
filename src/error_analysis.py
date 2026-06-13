"""
Systematic error analysis comparing BM25, cross-encoder, and bi-encoder.

Identifies four outcome categories per query and characterises failure modes
by query length, retrieval rank, and score patterns. Generates a figure and
a summary CSV for the final report.

Usage:
    python src/error_analysis.py

Inputs:
    - ./outputs/bm25_per_query.csv
    - ./outputs/reranker_per_query.csv
    - ./outputs/biencoder_per_query.csv
    - ./outputs/rq2_analysis.csv
    - ./outputs/bm25_run.csv
    - TREC-ToT 2025/dev1 queries via ir_datasets
Outputs:
    - ./figures/error_analysis.png
    - ./outputs/error_analysis.csv
"""

import os

import ir_datasets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = "./outputs"
FIGURES_DIR = "./figures"
METRIC = "RR@10"
SYSTEMS = ["bm25", "reranker", "biencoder"]
SYSTEM_LABELS = {"bm25": "BM25", "reranker": "Cross-encoder", "biencoder": "Bi-encoder"}


def load_per_query(path: str, metric: str, system_name: str) -> pd.Series:
    """Load a per-query metrics CSV and return one metric column.

    Handles both long format (measure/value columns) and wide format.

    Args:
        path:        Path to per-query CSV file
        metric:      Metric name (e.g. 'RR@10')
        system_name: Label used for error messages
    Returns:
        pd.Series indexed by qid (as str)
    """
    df = pd.read_csv(path)
    if "measure" in df.columns and "value" in df.columns:
        df = df[df["measure"] == metric].set_index("qid")["value"]
    else:
        df = df.set_index("qid")[metric]
    df.index = df.index.astype(str)
    return df


def classify_outcome(row: pd.Series) -> str:
    """Assign an outcome label based on which systems retrieved a relevant document.

    Args:
        row: Series with columns 'bm25', 'reranker', 'biencoder' (RR@10 scores)
    Returns:
        One of: 'All fail', 'BM25 only', 'Neural only', 'All succeed'
    """
    bm25_hit = row.get("bm25", 0) > 0
    rr_hit   = row.get("reranker", 0) > 0
    bi_hit   = row.get("biencoder", 0) > 0
    neural_hit = rr_hit or bi_hit

    if not bm25_hit and not neural_hit:
        return "All fail"
    if bm25_hit and not neural_hit:
        return "BM25 only"
    if not bm25_hit and neural_hit:
        return "Neural only"
    return "All succeed"


def run_error_analysis():
    """Build the per-query outcome table, print failure stats, and save outputs."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load per-query scores ─────────────────────────────────────────────────
    paths = {
        "bm25":      f"{RESULTS_DIR}/bm25_per_query.csv",
        "reranker":  f"{RESULTS_DIR}/reranker_per_query.csv",
        "biencoder": f"{RESULTS_DIR}/biencoder_per_query.csv",
    }
    available = {k: v for k, v in paths.items() if os.path.exists(v)}
    if "bm25" not in available:
        print("bm25_per_query.csv missing — cannot run error analysis")
        return

    series = {}
    for system, path in available.items():
        series[system] = load_per_query(path, METRIC, system)

    df = pd.DataFrame(series)
    df.index.name = "qid"
    df = df.reset_index()

    # ── Load query features ───────────────────────────────────────────────────
    rq2_path = f"{RESULTS_DIR}/rq2_analysis.csv"
    if os.path.exists(rq2_path):
        rq2 = pd.read_csv(rq2_path)[["qid", "query_length", "long_query"]]
        rq2["qid"] = rq2["qid"].astype(str)
        df = df.merge(rq2, on="qid", how="left")

    # ── Load bm25 run for per-query rank of relevant doc ─────────────────────
    bm25_run_path = f"{RESULTS_DIR}/bm25_run.csv"
    if os.path.exists(bm25_run_path):
        bm25_run = pd.read_csv(bm25_run_path)
        bm25_run["qid"] = bm25_run["qid"].astype(str)

    # ── Classify outcomes ─────────────────────────────────────────────────────
    if "biencoder" in df.columns:
        df["outcome"] = df.apply(classify_outcome, axis=1)
    else:
        # Fallback to two-system classification
        def _two_system(row):
            b = row.get("bm25", 0) > 0
            r = row.get("reranker", 0) > 0
            if not b and not r:   return "All fail"
            if b and not r:       return "BM25 only"
            if not b and r:       return "Neural only"
            return "All succeed"
        df["outcome"] = df.apply(_two_system, axis=1)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n=== Error Analysis ({METRIC}) ===")
    outcome_counts = df["outcome"].value_counts()
    print(f"\nOutcome distribution (n={len(df)}):")
    for outcome, count in outcome_counts.items():
        pct = 100 * count / len(df)
        print(f"  {outcome:<20s}: {count:3d}  ({pct:.1f}%)")

    # Show failure modes by query length
    if "long_query" in df.columns:
        print("\nFailure rate by query length:")
        for is_long, label in [(True, "Long queries"), (False, "Short queries")]:
            sub = df[df["long_query"] == is_long]
            fail_rate = (sub["outcome"] == "All fail").mean()
            bm25_rate = (sub["bm25"] > 0).mean()
            print(f"  {label} (n={len(sub)}): "
                  f"all-fail={fail_rate:.1%}  BM25-hit={bm25_rate:.1%}")

    # Failure examples: BM25 succeeds but neural fails
    bm25_only = df[df["outcome"] == "BM25 only"].copy()
    if not bm25_only.empty and "query_length" in bm25_only.columns:
        print(f"\n'BM25 only' cases (n={len(bm25_only)}) — "
              f"median query length: {bm25_only['query_length'].median():.0f} words")

    neural_only = df[df["outcome"] == "Neural only"].copy()
    if not neural_only.empty and "query_length" in neural_only.columns:
        print(f"'Neural only' cases (n={len(neural_only)}) — "
              f"median query length: {neural_only['query_length'].median():.0f} words")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df.to_csv(f"{RESULTS_DIR}/error_analysis.csv", index=False)
    print(f"\nSaved → {RESULTS_DIR}/error_analysis.csv")

    # ── Figure: outcome breakdown + mean score by outcome ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: stacked bar of outcome counts
    ax = axes[0]
    outcome_order = ["All fail", "BM25 only", "Neural only", "All succeed"]
    outcome_colours = ["#d7191c", "#fdae61", "#abd9e9", "#2c7bb6"]
    counts = [outcome_counts.get(o, 0) for o in outcome_order]
    bars = ax.bar(outcome_order, counts, color=outcome_colours,
                  edgecolor="black", linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of queries")
    ax.set_title("Query outcome distribution")
    ax.tick_params(axis="x", labelsize=8)

    # Panel B: mean BM25/reranker/biencoder score per outcome category
    ax = axes[1]
    avail_systems = [s for s in SYSTEMS if s in df.columns]
    means = df.groupby("outcome")[avail_systems].mean().reindex(outcome_order)

    x = np.arange(len(outcome_order))
    width = 0.25
    sys_colours = {"bm25": "#4dac26", "reranker": "#d01c8b", "biencoder": "#f1b6da"}
    sys_markers = {"bm25": "o", "reranker": "s", "biencoder": "^"}

    for i, system in enumerate(avail_systems):
        offset = (i - len(avail_systems) / 2 + 0.5) * width
        vals = means[system].fillna(0).values
        ax.bar(x + offset, vals, width,
               label=SYSTEM_LABELS[system],
               color=sys_colours[system],
               edgecolor="black", linewidth=0.5,
               hatch=["", "//", "xx"][i])

    ax.set_xticks(x)
    ax.set_xticklabels(outcome_order, fontsize=8)
    ax.set_ylabel(f"Mean {METRIC}")
    ax.set_title(f"Mean {METRIC} by outcome category")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = f"{FIGURES_DIR}/error_analysis.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    run_error_analysis()
