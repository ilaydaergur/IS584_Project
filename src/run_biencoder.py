"""
Bi-encoder reranking on top of BM25 candidates.

Encodes queries and document snippets independently with a SentenceTransformer
bi-encoder, computes cosine similarity, and reranks the BM25 top-K candidates.
Includes a WANDB sweep over model variants and rerank depth.

Usage:
    python src/run_biencoder.py --mode sweep
    python src/run_biencoder.py --mode best \
        --model sentence-transformers/msmarco-distilbert-base-v4 \
        --rerank_depth 100

Inputs:
    - ./outputs/bm25_run.csv  (produced by src/run_bm25.py --mode best)
    - TREC-ToT 2025/dev1 qrels via ir_datasets
Outputs:
    - ./outputs/biencoder_run.csv
    - ./outputs/biencoder_results.csv
    - ./outputs/biencoder_per_query.csv
    - WANDB sweep + best-run logs
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
import pyterrier as pt
import wandb
from ir_measures import MRR, nDCG, R
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = "./outputs"
WANDB_PROJECT = "is584-tot-retrieval"
DEFAULT_MODEL = "sentence-transformers/msmarco-distilbert-base-v4"
DEFAULT_RERANK_DEPTH = 100
DEFAULT_BATCH_SIZE = 64

EVAL_METRICS = [MRR @ 10, nDCG @ 10, R @ 100]
METRIC_COLS = ["RR@10", "nDCG@10", "R@100"]

SWEEP_MODELS = [
    "sentence-transformers/msmarco-distilbert-base-v4",
    "sentence-transformers/all-MiniLM-L6-v2",
]
SWEEP_DEPTHS = [50, 100]


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BiEncoderReranker(pt.Transformer):
    """PyTerrier transformer that reranks BM25 candidates with a bi-encoder.

    Encodes queries and document snippets separately and scores pairs by cosine
    similarity. Avoids the 512-token constraint of cross-encoders because queries
    and documents are encoded independently.

    Args:
        model_name:  SentenceTransformer model name
        batch_size:  Encoding batch size
        device:      Torch device string; auto-detected if None
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        device = device or _detect_device()
        self.model = SentenceTransformer(model_name, device=device)
        print(f"BiEncoder loaded: {model_name}  |  device: {device}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode queries and documents independently, score by cosine similarity.

        Args:
            df: DataFrame with columns qid, query, docno, and text_snippet or text
        Returns:
            DataFrame re-ranked by cosine similarity score within each qid
        """
        text_col = "text_snippet" if "text_snippet" in df.columns else "text"

        unique_queries = df[["qid", "query"]].drop_duplicates("qid")
        unique_docs = df[["docno", text_col]].drop_duplicates("docno")

        print(f"Encoding {len(unique_queries)} queries...")
        q_texts = unique_queries["query"].tolist()
        q_embs = self.model.encode(
            q_texts, batch_size=self.batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        q_emb_map = dict(zip(unique_queries["qid"].tolist(), q_embs))

        print(f"Encoding {len(unique_docs)} document snippets...")
        d_texts = unique_docs[text_col].fillna("").tolist()
        d_embs = self.model.encode(
            d_texts, batch_size=self.batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        d_emb_map = dict(zip(unique_docs["docno"].tolist(), d_embs))

        # cosine similarity = dot product (embeddings already L2-normalised)
        scores = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring pairs"):
            q_emb = q_emb_map.get(row["qid"])
            d_emb = d_emb_map.get(row["docno"])
            if q_emb is None or d_emb is None:
                scores.append(0.0)
            else:
                scores.append(float(np.dot(q_emb, d_emb)))

        out = df.copy()
        out["score"] = scores
        return (
            out.sort_values(["qid", "score"], ascending=[True, False])
            .reset_index(drop=True)
        )

    def get_query_embeddings(self, queries: list[str]) -> np.ndarray:
        """Encode a list of query strings and return L2-normalised embeddings.

        Args:
            queries: List of query text strings
        Returns:
            numpy array of shape (n_queries, embedding_dim)
        """
        return self.model.encode(
            queries, batch_size=self.batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )

    def get_doc_embeddings(self, texts: list[str]) -> np.ndarray:
        """Encode a list of document snippet strings and return L2-normalised embeddings.

        Args:
            texts: List of document text strings
        Returns:
            numpy array of shape (n_docs, embedding_dim)
        """
        return self.model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )


def _load_bm25_run(rerank_depth: int) -> pd.DataFrame:
    """Load BM25 run and keep top-rerank_depth candidates per query.

    Args:
        rerank_depth: Maximum number of BM25 candidates to keep per query
    Returns:
        Filtered BM25 run DataFrame
    """
    path = f"{RESULTS_DIR}/bm25_run.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BM25 run not found at '{path}'. "
            "Run: python src/run_bm25.py --mode best"
        )
    bm25_run = pd.read_csv(path)
    return (
        bm25_run.sort_values(["qid", "rank"])
        .groupby("qid")
        .head(rerank_depth)
        .reset_index(drop=True)
    )


def _evaluate(run_df: pd.DataFrame, perquery: bool = False) -> pd.DataFrame:
    """Evaluate a run with PyTerrier Experiment.

    Args:
        run_df:   DataFrame with qid, docno, score columns
        perquery: Return per-query results when True
    Returns:
        Experiment results DataFrame
    """
    if not pt.java.started():
        pt.java.init()
    dataset = pt.get_dataset("irds:trec-tot/2025/dev1")
    topics = dataset.get_topics("text")
    qrels = dataset.get_qrels()
    return pt.Experiment(
        [run_df], topics, qrels,
        eval_metrics=EVAL_METRICS,
        names=["BiEncoder"],
        perquery=perquery,
    )


# ── WANDB sweep ───────────────────────────────────────────────────────────────

def run_sweep():
    """WANDB grid sweep over bi-encoder model and rerank depth."""
    sweep_config = {
        "method": "grid",
        "name": "biencoder-sweep",
        "metric": {"name": "RR@10", "goal": "maximize"},
        "parameters": {
            "model":        {"values": SWEEP_MODELS},
            "rerank_depth": {"values": SWEEP_DEPTHS},
        },
    }
    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT)

    def _sweep_fn():
        with wandb.init() as run:
            cfg = wandb.config
            bm25_run = _load_bm25_run(cfg.rerank_depth)
            reranker = BiEncoderReranker(model_name=cfg.model)
            reranked = reranker.transform(bm25_run)
            agg = _evaluate(reranked, perquery=False)
            metrics = {col: float(agg[col].iloc[0]) for col in METRIC_COLS}
            wandb.log(metrics)
            short = cfg.model.split("/")[-1]
            print(
                f"model={short}  depth={cfg.rerank_depth}  "
                + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            )

    wandb.agent(sweep_id, _sweep_fn)


# ── Single best run ───────────────────────────────────────────────────────────

def run_best(
    model_name: str = DEFAULT_MODEL,
    rerank_depth: int = DEFAULT_RERANK_DEPTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """Run bi-encoder with chosen settings, evaluate, save, and log to WANDB.

    Args:
        model_name:   SentenceTransformer model identifier
        rerank_depth: BM25 candidates to rerank per query
        batch_size:   Encoding batch size
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    bm25_run = _load_bm25_run(rerank_depth)
    reranker = BiEncoderReranker(model_name=model_name, batch_size=batch_size)
    reranked = reranker.transform(bm25_run)
    reranked.to_csv(f"{RESULTS_DIR}/biencoder_run.csv", index=False)

    agg = _evaluate(reranked, perquery=False)
    pq = _evaluate(reranked, perquery=True)

    agg.to_csv(f"{RESULTS_DIR}/biencoder_results.csv", index=False)
    pq.to_csv(f"{RESULTS_DIR}/biencoder_per_query.csv", index=False)

    print("\n=== Bi-Encoder Reranker Results ===")
    for col in METRIC_COLS:
        print(f"  {col}: {float(agg[col].iloc[0]):.4f}")

    metrics = {col: float(agg[col].iloc[0]) for col in METRIC_COLS}
    short = model_name.split("/")[-1]
    with wandb.init(
        project=WANDB_PROJECT,
        name=f"biencoder-{short}-depth{rerank_depth}",
        config={"model": model_name, "rerank_depth": rerank_depth,
                "batch_size": batch_size},
    ):
        wandb.log(metrics)

    print(f"\nSaved reranked run  → {RESULTS_DIR}/biencoder_run.csv")
    print(f"Saved agg results   → {RESULTS_DIR}/biencoder_results.csv")
    print(f"Saved per-query     → {RESULTS_DIR}/biencoder_per_query.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bi-encoder reranking for TREC-ToT 2024"
    )
    parser.add_argument(
        "--mode", choices=["sweep", "best"], default="sweep",
        help="sweep: WANDB grid; best: single run with --model / --rerank_depth",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--rerank_depth", type=int, default=DEFAULT_RERANK_DEPTH,
        help="Number of BM25 candidates to rerank per query",
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Encoding batch size",
    )
    args = parser.parse_args()

    if args.mode == "sweep":
        run_sweep()
    else:
        run_best(args.model, args.rerank_depth, args.batch_size)
