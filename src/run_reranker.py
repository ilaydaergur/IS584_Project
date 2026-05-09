"""
Cross-encoder reranking on top of BM25 candidates.

Reads the BM25 candidate set, scores each (query, document) pair with a
cross-encoder, re-ranks, evaluates, and logs results to Weights & Biases.

Usage:
    python src/run_reranker.py
    python src/run_reranker.py --model cross-encoder/ms-marco-MiniLM-L-12-v2
    python src/run_reranker.py --batch_size 16 --rerank_depth 50

Inputs:
    - ./outputs/bm25_run.csv  (produced by src/run_bm25.py --mode best)
    - TREC-ToT 2024 qrels via ir_datasets
Outputs:
    - ./outputs/reranker_run.csv
    - ./outputs/reranker_results.csv
    - ./outputs/reranker_per_query.csv
    - WANDB run with MRR@10, nDCG@10, R@100 logged
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
import pyterrier as pt
import wandb
from ir_measures import MRR, nDCG, R
from sentence_transformers import CrossEncoder
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = "./outputs"
WANDB_PROJECT = "is584-tot-retrieval"
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_DEPTH = 100  # number of BM25 candidates to rerank per query
DEFAULT_BATCH_SIZE = 32


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CrossEncoderReranker(pt.Transformer):
    """PyTerrier transformer that reranks BM25 candidates with a cross-encoder.

    Args:
        model_name:  HuggingFace model identifier (must be a cross-encoder)
        batch_size:  Query-document pairs scored per forward pass
        max_length:  Maximum token length; long queries/docs are truncated
        device:      Torch device string; auto-detected if None
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = 512,
        device: str = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        device = device or _detect_device()
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        print(f"CrossEncoder loaded: {model_name}  |  device: {device}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score all (query, document) pairs and re-rank per query group.

        Args:
            df: DataFrame with at minimum columns qid, query, docno,
                and either 'text_snippet' or 'text'
        Returns:
            Same DataFrame sorted by cross-encoder score descending within qid
        """
        text_col = "text_snippet" if "text_snippet" in df.columns else "text"
        # Truncate queries to 100 words so the document gets enough token budget
        # within the 512-token cross-encoder limit (avg ToT query is 238 words)
        queries_truncated = [" ".join(q.split()[:100]) for q in df["query"].tolist()]
        pairs = list(zip(queries_truncated, df[text_col].tolist()))
        n_queries = df["qid"].nunique()
        print(f"Scoring {len(pairs)} pairs across {n_queries} queries (queries truncated to 100 words)...")
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=True
        )
        out = df.copy()
        out["score"] = scores
        return (
            out.sort_values(["qid", "score"], ascending=[True, False])
            .reset_index(drop=True)
        )


EVAL_METRICS = [MRR @ 10, nDCG @ 10, R @ 100]
METRIC_COLS = ["RR@10", "nDCG@10", "R@100"]  # RR@10 = per-query reciprocal rank (mean = MRR@10)


def _run_experiment(run_df, topics, qrels, perquery=False):
    """Evaluate a pre-computed run using pt.Experiment.

    Args:
        run_df:    DataFrame with columns qid, docno, score
        topics:    DataFrame with columns qid, query
        qrels:     DataFrame with columns qid, docno, label
        perquery:  Return per-query results if True
    Returns:
        pt.Experiment results DataFrame
    """
    return pt.Experiment(
        [run_df],
        topics,
        qrels,
        eval_metrics=EVAL_METRICS,
        names=["CrossEncoder"],
        perquery=perquery,
    )


def run_reranker(
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rerank_depth: int = DEFAULT_RERANK_DEPTH,
):
    """Load BM25 run, rerank, evaluate, save results, and log to WANDB.

    Args:
        model_name:   Cross-encoder HuggingFace model name
        batch_size:   Inference batch size
        rerank_depth: How many top BM25 candidates to rerank per query
    """
    if not pt.java.started():
        pt.java.init()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    bm25_path = f"{RESULTS_DIR}/bm25_run.csv"
    if not os.path.exists(bm25_path):
        raise FileNotFoundError(
            f"BM25 run not found at '{bm25_path}'. "
            "Run: python src/run_bm25.py --mode best"
        )

    print(f"Loading BM25 candidates from {bm25_path}...")
    bm25_run = pd.read_csv(bm25_path)
    # keep only the top-rerank_depth candidates per query
    bm25_run = (
        bm25_run.sort_values(["qid", "rank"])
        .groupby("qid")
        .head(rerank_depth)
        .reset_index(drop=True)
    )
    print(
        f"Reranking {len(bm25_run)} (query, doc) pairs "
        f"({rerank_depth} per query)..."
    )

    dataset = pt.get_dataset("irds:trec-tot/2025/dev1")
    topics = dataset.get_topics("text")
    qrels = dataset.get_qrels()

    reranker = CrossEncoderReranker(model_name=model_name, batch_size=batch_size)
    reranked = reranker.transform(bm25_run)
    reranked.to_csv(f"{RESULTS_DIR}/reranker_run.csv", index=False)

    agg = _run_experiment(reranked, topics, qrels, perquery=False)
    pq = _run_experiment(reranked, topics, qrels, perquery=True)

    agg.to_csv(f"{RESULTS_DIR}/reranker_results.csv", index=False)
    pq.to_csv(f"{RESULTS_DIR}/reranker_per_query.csv", index=False)

    print("\n=== Cross-Encoder Reranker Results ===")
    for col in METRIC_COLS:
        print(f"  {col}: {float(agg[col].iloc[0]):.4f}")

    metrics = {col: float(agg[col].iloc[0]) for col in METRIC_COLS}
    short_name = model_name.split("/")[-1]
    with wandb.init(
        project=WANDB_PROJECT,
        name=f"reranker-{short_name}-depth{rerank_depth}",
        config={
            "model": model_name,
            "rerank_depth": rerank_depth,
            "batch_size": batch_size,
        },
    ):
        wandb.log(metrics)

    print(f"\nSaved reranked run  → {RESULTS_DIR}/reranker_run.csv")
    print(f"Saved agg results   → {RESULTS_DIR}/reranker_results.csv")
    print(f"Saved per-query     → {RESULTS_DIR}/reranker_per_query.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-encoder reranking for TREC-ToT 2024"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="HuggingFace cross-encoder model name",
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Inference batch size (lower if OOM)",
    )
    parser.add_argument(
        "--rerank_depth", type=int, default=DEFAULT_RERANK_DEPTH,
        help="Number of BM25 candidates to rerank per query",
    )
    args = parser.parse_args()
    run_reranker(args.model, args.batch_size, args.rerank_depth)
