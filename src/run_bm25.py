"""
BM25 baseline retrieval with Weights & Biases parameter sweep.

Usage:
    python src/run_bm25.py --mode sweep          # grid sweep over k1 / b
    python src/run_bm25.py --mode best --k1 1.2 --b 0.75  # single best run

Inputs:
    - PyTerrier index at ./index/  (build with src/index_corpus.py first)
    - TREC-ToT 2025/dev1 queries and qrels (same 2024 corpus, has qrels)
Outputs:
    - WANDB sweep logs per run
    - ./outputs/bm25_run.csv          retrieved candidates (best params)
    - ./outputs/bm25_results.csv      aggregate metrics
    - ./outputs/bm25_per_query.csv    per-query metrics
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
import pyterrier as pt
import wandb
from ir_measures import MRR, nDCG, R

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

INDEX_DIR = "./index"
RESULTS_DIR = "./outputs"
WANDB_PROJECT = "is584-tot-retrieval"
NUM_CANDIDATES = 100
EVAL_DATASET = "irds:trec-tot/2025/dev1"  # shares 2024 corpus; has qrels

EVAL_METRICS = [MRR @ 10, nDCG @ 10, R @ 100]
METRIC_COLS = ["RR@10", "nDCG@10", "R@100"]  # RR@10 = per-query reciprocal rank (mean = MRR@10)


def _init_pt():
    if not pt.java.started():
        pt.java.init()


def load_topics_and_qrels():
    """Load topics and qrels for the evaluation dataset.

    Returns:
        topics (DataFrame): qid, query columns
        qrels  (DataFrame): qid, docno, label columns
    """
    dataset = pt.get_dataset(EVAL_DATASET)
    topics = dataset.get_topics("text")
    qrels = dataset.get_qrels()
    return topics, qrels


def build_bm25(index, k1: float, b: float) -> pt.terrier.Retriever:
    """Instantiate a BM25 retriever.

    Args:
        index: PyTerrier index object (from pt.terrier.IndexFactory.of)
        k1:    term-frequency saturation parameter
        b:     document-length normalisation parameter
    Returns:
        Configured PyTerrier retriever transformer
    """
    return pt.terrier.Retriever(
        index,
        wmodel="BM25",
        controls={"bm25.k1": str(k1), "bm25.b": str(b)},
        num_results=NUM_CANDIDATES,
        metadata=["docno", "text_snippet", "title"],
    )


def run_experiment(retriever, topics, qrels, perquery=False):
    """Run pt.Experiment and return the results DataFrame.

    Args:
        retriever:  PyTerrier transformer
        topics:     DataFrame with qid, query columns
        qrels:      DataFrame with qid, docno, label columns
        perquery:   If True return per-query results
    Returns:
        pt.Experiment results DataFrame
    """
    return pt.Experiment(
        [retriever],
        topics,
        qrels,
        eval_metrics=EVAL_METRICS,
        names=["BM25"],
        perquery=perquery,
    )


# ── WANDB sweep ──────────────────────────────────────────────────────────────

def _sweep_fn():
    """Single WANDB sweep trial: retrieve with current config and log metrics."""
    with wandb.init() as _:
        config = wandb.config
        _init_pt()
        indexref = pt.terrier.IndexFactory.of(INDEX_DIR)
        topics, qrels = load_topics_and_qrels()
        bm25 = build_bm25(indexref, config.k1, config.b)

        result = run_experiment(bm25, topics, qrels)
        metrics = {col: float(result[col].iloc[0]) for col in METRIC_COLS}
        wandb.log(metrics)
        print(
            f"k1={config.k1}, b={config.b} → "
            + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )


def run_sweep():
    """Launch a WANDB grid sweep over BM25 k1 and b parameters."""
    sweep_config = {
        "method": "grid",
        "name": "bm25-parameter-sweep",
        "metric": {"name": "RR@10", "goal": "maximize"},
        "parameters": {
            "k1": {"values": [0.5, 1.0, 1.2, 1.5, 2.0]},
            "b":  {"values": [0.3, 0.5, 0.75, 1.0]},
        },
    }
    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT)
    wandb.agent(sweep_id, _sweep_fn)


# ── Single best run ──────────────────────────────────────────────────────────

def run_best(k1: float = 1.2, b: float = 0.75):
    """Retrieve with chosen BM25 parameters, evaluate, and save outputs.

    Args:
        k1: BM25 k1 value (default 1.2)
        b:  BM25 b value  (default 0.75)
    """
    _init_pt()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    indexref = pt.terrier.IndexFactory.of(INDEX_DIR)
    topics, qrels = load_topics_and_qrels()
    bm25 = build_bm25(indexref, k1, b)

    print(f"Running BM25 (k1={k1}, b={b})...")
    run_df = bm25.transform(topics)
    run_df.to_csv(f"{RESULTS_DIR}/bm25_run.csv", index=False)

    agg = run_experiment(bm25, topics, qrels, perquery=False)
    pq = run_experiment(bm25, topics, qrels, perquery=True)

    agg.to_csv(f"{RESULTS_DIR}/bm25_results.csv", index=False)
    pq.to_csv(f"{RESULTS_DIR}/bm25_per_query.csv", index=False)

    print("\n=== BM25 Aggregate Results ===")
    for col in METRIC_COLS:
        print(f"  {col}: {float(agg[col].iloc[0]):.4f}")

    metrics = {col: float(agg[col].iloc[0]) for col in METRIC_COLS}
    with wandb.init(
        project=WANDB_PROJECT,
        name=f"bm25-best-k1{k1}-b{b}",
        config={"model": "BM25", "k1": k1, "b": b},
    ):
        wandb.log(metrics)

    print(f"\nSaved run       → {RESULTS_DIR}/bm25_run.csv")
    print(f"Saved aggregate → {RESULTS_DIR}/bm25_results.csv")
    print(f"Saved per-query → {RESULTS_DIR}/bm25_per_query.csv")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM25 baseline for TREC-ToT 2024")
    parser.add_argument(
        "--mode", choices=["sweep", "best"], default="sweep",
        help="sweep: WANDB grid sweep; best: single run with --k1 / --b",
    )
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 k1 parameter")
    parser.add_argument("--b",  type=float, default=0.75, help="BM25 b parameter")
    args = parser.parse_args()

    if args.mode == "sweep":
        run_sweep()
    else:
        run_best(args.k1, args.b)
