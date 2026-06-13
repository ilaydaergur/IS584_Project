"""
Evaluation utilities: statistical testing and RQ2/RQ3 subgroup analysis.

Runs Wilcoxon signed-rank tests (RQ1: CrossEncoder vs BM25,
RQ3: BiEncoder vs BM25) and a stratified comparison by query length (RQ2).

Usage:
    python src/evaluate.py                    # uses RR@10 by default
    python src/evaluate.py --metric "nDCG@10"

Inputs:
    - ./outputs/bm25_per_query.csv
    - ./outputs/reranker_per_query.csv
    - ./outputs/biencoder_per_query.csv  (optional, for RQ3)
    - ./outputs/bm25_run.csv             (for keyword-overlap computation)
    - TREC-ToT 2025/dev1 queries via ir_datasets
Outputs:
    - Wilcoxon test results printed to stdout
    - ./outputs/rq2_analysis.csv         per-query analysis with subgroup flags
"""

import os
import argparse

import numpy as np
import pandas as pd
import ir_datasets
from scipy.stats import wilcoxon

RESULTS_DIR = "./outputs"


# ── Metric loading ─────────────────────────────────────────────────────────────

def load_per_query(path: str, metric: str) -> pd.Series:
    """Read a per-query metrics CSV and return one metric as a Series.

    Handles both wide format (one column per metric) and long format
    (columns: name, qid, measure, value) produced by pt.Experiment(perquery=True).

    Args:
        path:   Path to per-query CSV
        metric: Metric name, e.g. 'RR@10'
    Returns:
        pd.Series indexed by qid
    """
    df = pd.read_csv(path)
    if "measure" in df.columns and "value" in df.columns:
        subset = df[df["measure"] == metric]
        if subset.empty:
            available = df["measure"].unique().tolist()
            raise ValueError(f"Metric '{metric}' not in {path}. Available: {available}")
        return subset.set_index("qid")["value"]
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not in {path}. Available: {list(df.columns)}")
    return df.set_index("qid")[metric]


# ── RQ1: Statistical test ──────────────────────────────────────────────────────

def wilcoxon_test(
    scores_a: pd.Series,
    scores_b: pd.Series,
    label_a: str = "System A",
    label_b: str = "System B",
) -> dict:
    """Two-sided Wilcoxon signed-rank test on paired per-query scores.

    Pairs are formed over the intersection of query IDs. Reports the
    mean per-query delta, test statistic, p-value, and significance.

    Args:
        scores_a: Per-query scores for system A (e.g. reranker)
        scores_b: Per-query scores for system B (e.g. BM25)
        label_a:  Display name for system A
        label_b:  Display name for system B
    Returns:
        dict with keys: n, mean_diff, stat, pvalue, significant
    """
    common = scores_a.index.intersection(scores_b.index)
    a = scores_a.loc[common].values
    b = scores_b.loc[common].values

    # Wilcoxon requires non-zero differences; handle edge case
    if np.all(a == b):
        print("All per-query scores are identical; Wilcoxon test not applicable.")
        return {}

    stat, pvalue = wilcoxon(a, b, alternative="two-sided")
    mean_diff = float(np.mean(a - b))
    significant = pvalue < 0.05

    print(f"\n=== RQ1: Wilcoxon Signed-Rank Test ({label_a} vs {label_b}) ===")
    print(f"  Queries compared : {len(common)}")
    print(f"  Mean Δ           : {mean_diff:+.4f}  ({label_a} − {label_b})")
    print(f"  Statistic W      : {stat:.2f}")
    print(f"  p-value          : {pvalue:.4f}")
    print(f"  Significant (α=0.05): {'YES ✓' if significant else 'NO'}")

    return {
        "n": len(common),
        "mean_diff": mean_diff,
        "stat": stat,
        "pvalue": pvalue,
        "significant": significant,
    }


# ── RQ2: Query-level analysis ──────────────────────────────────────────────────

def jaccard_overlap(query: str, doc_text: str) -> float:
    """Jaccard coefficient between the query word-set and document word-set.

    Args:
        query:    Query string
        doc_text: Document text (or snippet)
    Returns:
        Float in [0, 1]; 0 if either string is empty
    """
    q_words = set(query.lower().split())
    d_words = set(doc_text.lower().split())
    if not q_words or not d_words:
        return 0.0
    return len(q_words & d_words) / len(q_words | d_words)


def build_query_features() -> pd.DataFrame:
    """Compute per-query length and keyword-overlap features.

    Overlap is measured between each query and the top-1 BM25 candidate's
    stored text snippet (as a proxy for retrieval difficulty).

    Returns:
        DataFrame with columns: qid, query_length, overlap,
                                long_query (bool), low_overlap (bool)
    """
    test_ds = ir_datasets.load("trec-tot/2025/dev1")
    queries = {
        q.query_id: getattr(q, "text", "")
        for q in test_ds.queries_iter()
    }

    bm25_run = pd.read_csv(f"{RESULTS_DIR}/bm25_run.csv")
    text_col = "text_snippet" if "text_snippet" in bm25_run.columns else "text"
    top1 = (
        bm25_run.sort_values("rank")
        .groupby("qid")
        .first()
        .reset_index()[["qid", text_col]]
    )
    top1 = top1.rename(columns={text_col: "doc_text"})

    rows = []
    for qid, q_text in queries.items():
        q_len = len(q_text.split())
        match = top1[top1["qid"] == qid]
        doc_text = match["doc_text"].values[0] if len(match) > 0 else ""
        overlap = jaccard_overlap(q_text, doc_text)
        rows.append({"qid": qid, "query_length": q_len, "overlap": overlap})

    df = pd.DataFrame(rows)
    len_median = df["query_length"].median()
    df["long_query"] = df["query_length"] > len_median
    # For ToT queries keyword overlap is near-zero for most queries;
    # split on overlap == 0 (no shared words) vs overlap > 0 (some shared words)
    df["low_overlap"] = df["overlap"] == 0
    return df


def rq2_analysis(metric: str = "RR@10") -> pd.DataFrame:
    """Stratified comparison of BM25 vs reranker for RQ2.

    Splits queries into long/short and low/high keyword-overlap subgroups
    and reports mean metric scores for each system in each subgroup.

    Args:
        metric: Metric column to compare (e.g. 'RR@10', 'nDCG@10')
    Returns:
        DataFrame with per-query features and system scores
    """
    bm25_pq = load_per_query(f"{RESULTS_DIR}/bm25_per_query.csv", metric)
    reranker_pq = load_per_query(f"{RESULTS_DIR}/reranker_per_query.csv", metric)

    features = build_query_features()
    # Align qid types before merging (ir_datasets uses str, pt.Experiment uses int)
    features["qid"] = features["qid"].astype(str)
    bm25_pq.index = bm25_pq.index.astype(str)
    reranker_pq.index = reranker_pq.index.astype(str)

    merged = (
        features
        .join(bm25_pq.rename("bm25"), on="qid")
        .join(reranker_pq.rename("reranker"), on="qid")
    )

    subgroups = [
        ("Long queries   (> median length)", merged["long_query"]),
        ("Short queries  (≤ median length)", ~merged["long_query"]),
        ("Zero overlap   (no shared words)", merged["low_overlap"]),
        ("Non-zero overlap (some shared words)", ~merged["low_overlap"]),
    ]

    print(f"\n=== RQ2: Subgroup Analysis ({metric}) ===")
    print(f"  Median query length : {merged['query_length'].median():.0f} words")
    print(f"  Zero-overlap queries: {merged['low_overlap'].sum()} / {len(merged)}")
    print()

    for label, mask in subgroups:
        sub = merged[mask].dropna(subset=["bm25", "reranker"])
        mean_bm25 = sub["bm25"].mean()
        mean_rr = sub["reranker"].mean()
        delta = mean_rr - mean_bm25
        print(
            f"  {label}  (n={len(sub):3d}): "
            f"BM25={mean_bm25:.4f}  Reranker={mean_rr:.4f}  Δ={delta:+.4f}"
        )

    merged.to_csv(f"{RESULTS_DIR}/rq2_analysis.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR}/rq2_analysis.csv")
    return merged


# ── Entry point ────────────────────────────────────────────────────────────────

def rq2_analysis_extended(metric: str = "RR@10") -> pd.DataFrame:
    """Extend rq2_analysis CSV with bi-encoder per-query scores if available.

    Args:
        metric: Metric column name (e.g. 'RR@10')
    Returns:
        Merged DataFrame with bm25, reranker, and (optionally) biencoder columns
    """
    merged = rq2_analysis(metric)  # writes rq2_analysis.csv with bm25 + reranker

    bi_path = f"{RESULTS_DIR}/biencoder_per_query.csv"
    if not os.path.exists(bi_path):
        return merged

    bi_scores = load_per_query(bi_path, metric)
    bi_scores.index = bi_scores.index.astype(str)
    merged["qid"] = merged["qid"].astype(str)
    merged = merged.merge(
        bi_scores.rename("biencoder"),
        left_on="qid", right_index=True, how="left",
    )

    # Print extended subgroup table
    subgroups = [
        ("Long queries   (> median length)", merged["long_query"]),
        ("Short queries  (≤ median length)", ~merged["long_query"]),
    ]
    print(f"\n=== RQ2 Extended: Subgroup Analysis with BiEncoder ({metric}) ===")
    for label, mask in subgroups:
        sub = merged[mask].dropna(subset=["bm25", "reranker"])
        mean_bm25 = sub["bm25"].mean()
        mean_rr   = sub["reranker"].mean()
        mean_bi   = sub["biencoder"].mean() if "biencoder" in sub else float("nan")
        print(
            f"  {label}  (n={len(sub):3d}): "
            f"BM25={mean_bm25:.4f}  CrossEnc={mean_rr:.4f}  BiEnc={mean_bi:.4f}"
        )

    merged.to_csv(f"{RESULTS_DIR}/rq2_analysis.csv", index=False)
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Statistical evaluation and subgroup analysis for TREC-ToT 2025/dev1"
    )
    parser.add_argument(
        "--metric", type=str, default="RR@10",
        choices=["RR@10", "nDCG@10"],
        help="Per-query metric to use for testing and analysis",
    )
    args = parser.parse_args()

    bm25_path     = f"{RESULTS_DIR}/bm25_per_query.csv"
    reranker_path = f"{RESULTS_DIR}/reranker_per_query.csv"
    bi_path       = f"{RESULTS_DIR}/biencoder_per_query.csv"

    missing = [p for p in [bm25_path, reranker_path] if not os.path.exists(p)]
    if missing:
        print(
            "Missing result files:\n  " + "\n  ".join(missing) +
            "\nRun run_bm25.py and run_reranker.py first."
        )
    else:
        bm25_scores     = load_per_query(bm25_path, args.metric)
        reranker_scores = load_per_query(reranker_path, args.metric)

        # RQ1: CrossEncoder vs BM25
        wilcoxon_test(reranker_scores, bm25_scores, "CrossEncoder", "BM25")

        # RQ3: BiEncoder vs BM25 (if available)
        if os.path.exists(bi_path):
            bi_scores = load_per_query(bi_path, args.metric)
            wilcoxon_test(bi_scores, bm25_scores, "BiEncoder", "BM25")
            wilcoxon_test(bi_scores, reranker_scores, "BiEncoder", "CrossEncoder")

        rq2_analysis_extended(args.metric)
