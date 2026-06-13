"""
Interpretability analysis for the neural retrieval pipeline.

Two techniques:
  1. t-SNE projection of bi-encoder query and document embeddings to visualise
     the semantic space and understand where the model succeeds vs fails.
  2. Sentence-level attribution for the cross-encoder: each sentence of a query
     is masked in turn and the score drop reveals which narrative clues the
     cross-encoder relies on most.

Usage:
    python src/interpretability.py

Inputs:
    - ./outputs/biencoder_run.csv  (produced by src/run_biencoder.py --mode best)
    - ./outputs/rq2_analysis.csv   (produced by src/evaluate.py)
    - ./outputs/bm25_run.csv
    - TREC-ToT 2025/dev1 queries via ir_datasets
Outputs:
    - ./figures/tsne_embedding_space.png
    - ./figures/cross_encoder_attribution.png
    - ./outputs/attribution_results.csv
"""

import os
import re

import ir_datasets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.manifold import TSNE
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = "./outputs"
FIGURES_DIR = "./figures"
BIENCODER_MODEL = "sentence-transformers/msmarco-distilbert-base-v4"
CROSSENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
N_ATTRIBUTION_EXAMPLES = 6  # 3 reranker-wins + 3 reranker-losses


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── 1. t-SNE of bi-encoder embedding space ───────────────────────────────────

def run_tsne_analysis():
    """Project query and document embeddings to 2D with t-SNE and save figure.

    Loads the bi-encoder run, encodes queries and their top-1 retrieved
    documents, then projects embeddings to 2D. Points are coloured by type
    (query / retrieved-relevant / retrieved-irrelevant) to reveal clustering
    behaviour and explain why the bi-encoder retrieves what it does.
    """
    print("\n=== t-SNE Embedding Space Analysis ===")

    biencoder_path = f"{RESULTS_DIR}/biencoder_run.csv"
    rq2_path = f"{RESULTS_DIR}/rq2_analysis.csv"
    if not os.path.exists(biencoder_path):
        print(f"  {biencoder_path} not found — skipping t-SNE")
        return
    if not os.path.exists(rq2_path):
        print(f"  {rq2_path} not found — skipping t-SNE")
        return

    bi_run = pd.read_csv(biencoder_path)
    rq2 = pd.read_csv(rq2_path)
    rq2["qid"] = rq2["qid"].astype(str)

    # Load qrels to know which retrieved docs are relevant
    import pyterrier as pt
    if not pt.java.started():
        pt.java.init()
    dataset = pt.get_dataset("irds:trec-tot/2025/dev1")
    qrels = dataset.get_qrels()
    relevant_pairs = set(zip(qrels["qid"].astype(str), qrels["docno"].astype(str)))

    # Top-1 retrieved document per query
    top1 = (
        bi_run.sort_values("score", ascending=False)
        .groupby("qid")
        .first()
        .reset_index()[["qid", "docno", "query", "score"]]
    )
    text_col = "text_snippet" if "text_snippet" in bi_run.columns else "text"
    top1_texts = (
        bi_run.sort_values("score", ascending=False)
        .groupby("qid")
        .first()
        .reset_index()[["qid", text_col]]
    )
    top1 = top1.merge(top1_texts[["qid", text_col]], on="qid")
    top1["qid"] = top1["qid"].astype(str)
    top1["docno"] = top1["docno"].astype(str)
    top1["is_relevant"] = top1.apply(
        lambda r: (r["qid"], r["docno"]) in relevant_pairs, axis=1
    )

    device = _detect_device()
    biencoder = SentenceTransformer(BIENCODER_MODEL, device=device)

    queries = top1["query"].tolist()
    docs = top1[text_col].fillna("").tolist()

    print("  Encoding queries...")
    q_embs = biencoder.encode(queries, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True)
    print("  Encoding top-1 documents...")
    d_embs = biencoder.encode(docs, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True)

    # Stack for joint t-SNE
    all_embs = np.vstack([q_embs, d_embs])
    labels = (["query"] * len(q_embs)
              + ["doc-relevant" if r else "doc-irrelevant"
                 for r in top1["is_relevant"].tolist()])

    print("  Running t-SNE (this may take ~1 minute)...")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(all_embs)

    # Build figure
    colour_map = {
        "query":          ("#2166ac", "s", "Query"),
        "doc-relevant":   ("#1a9641", "o", "Top-1 Doc (relevant)"),
        "doc-irrelevant": ("#d7191c", "^", "Top-1 Doc (irrelevant)"),
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, (colour, marker, legend_label) in colour_map.items():
        mask = np.array([l == label for l in labels])
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=colour, marker=marker, s=40, alpha=0.75, linewidths=0,
            label=legend_label,
        )

    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_title("Bi-encoder embedding space (t-SNE)\nQueries vs. top-1 retrieved documents")
    ax.legend(loc="best", framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.3)

    out_path = f"{FIGURES_DIR}/tsne_embedding_space.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")

    # Summary stats
    n_rel = sum(1 for l in labels if l == "doc-relevant")
    n_irrel = sum(1 for l in labels if l == "doc-irrelevant")
    print(f"  Top-1 relevant: {n_rel}, irrelevant: {n_irrel} (of {len(top1)} queries)")


# ── 2. Cross-encoder sentence-level attribution ───────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split query text into sentences using punctuation heuristics.

    Args:
        text: Query string (typically a long narrative paragraph)
    Returns:
        List of non-empty sentence strings
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if len(p.split()) >= 3]


def _score_pair(model: CrossEncoder, query: str, doc: str) -> float:
    """Score a single (query, doc) pair with the cross-encoder.

    Args:
        model: Loaded CrossEncoder instance
        query: Query string (truncated to 100 words if needed)
        doc:   Document snippet string
    Returns:
        Float relevance score
    """
    q = " ".join(query.split()[:100])
    result = model.predict([(q, doc)])
    return float(result[0]) if hasattr(result, "__len__") else float(result)


def run_attribution_analysis():
    """Sentence-level attribution for the cross-encoder on example queries.

    For each example query, the full query is scored, then each sentence is
    removed in turn and the score drop is computed. Large drops indicate
    sentences the cross-encoder relies on most.
    Saves a bar-chart figure and a CSV of all attribution scores.
    """
    print("\n=== Cross-Encoder Sentence Attribution ===")

    rq2_path = f"{RESULTS_DIR}/rq2_analysis.csv"
    bm25_path = f"{RESULTS_DIR}/bm25_run.csv"
    if not os.path.exists(rq2_path) or not os.path.exists(bm25_path):
        print("  Required files missing — skipping attribution")
        return

    rq2 = pd.read_csv(rq2_path)
    rq2["qid"] = rq2["qid"].astype(str)
    bm25_run = pd.read_csv(bm25_path)
    text_col = "text_snippet" if "text_snippet" in bm25_run.columns else "text"

    # Select examples: 3 where reranker wins, 3 where it loses
    delta = rq2["reranker"] - rq2["bm25"]
    wins = rq2[delta > 0].nlargest(3, "reranker")["qid"].tolist()
    losses = rq2[delta < 0].nsmallest(3, "reranker")["qid"].tolist()
    example_qids = wins + losses
    example_labels = (["Win"] * len(wins)) + (["Loss"] * len(losses))

    if not example_qids:
        print("  No win/loss examples found — skipping attribution")
        return

    device = _detect_device()
    ce_model = CrossEncoder(CROSSENCODER_MODEL, device=device)

    # Load queries and top-1 BM25 doc for each example
    import pyterrier as pt
    if not pt.java.started():
        pt.java.init()
    dataset = pt.get_dataset("irds:trec-tot/2025/dev1")
    topics = dataset.get_topics("text")
    topics["qid"] = topics["qid"].astype(str)
    query_map = dict(zip(topics["qid"], topics["query"]))

    top1_docs = (
        bm25_run.sort_values("rank")
        .groupby("qid")
        .first()
        .reset_index()[["qid", text_col]]
    )
    top1_docs["qid"] = top1_docs["qid"].astype(str)
    doc_map = dict(zip(top1_docs["qid"], top1_docs[text_col]))

    all_rows = []
    panel_data = []  # (qid, label, sentences, scores)

    for qid, label in zip(example_qids, example_labels):
        query = query_map.get(qid, "")
        doc = doc_map.get(qid, "")
        if not query or not doc:
            continue

        sentences = _split_sentences(query)
        if len(sentences) < 2:
            continue

        full_score = _score_pair(ce_model, query, doc)

        drops = []
        for i, sent in enumerate(sentences):
            ablated = " ".join(s for j, s in enumerate(sentences) if j != i)
            ablated_score = _score_pair(ce_model, ablated, doc)
            drop = full_score - ablated_score
            drops.append(drop)
            all_rows.append({
                "qid": qid,
                "label": label,
                "sentence_idx": i,
                "sentence": sent,
                "full_score": full_score,
                "ablated_score": ablated_score,
                "score_drop": drop,
            })

        panel_data.append((qid, label, sentences, drops))
        print(f"  qid={qid} ({label})  full_score={full_score:.3f}  "
              f"sentences={len(sentences)}")

    if not panel_data:
        print("  No valid examples — skipping attribution plot")
        return

    # Save CSV
    attr_df = pd.DataFrame(all_rows)
    attr_df.to_csv(f"{RESULTS_DIR}/attribution_results.csv", index=False)

    # Figure: one panel per example query
    n = len(panel_data)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.5 * n))
    if n == 1:
        axes = [axes]

    win_colour = "#2166ac"
    loss_colour = "#d7191c"

    for ax, (qid, label, sentences, drops) in zip(axes, panel_data):
        colour = win_colour if label == "Win" else loss_colour
        x = range(len(sentences))
        bars = ax.bar(x, drops, color=colour, alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(list(x))
        short_labels = [f"S{i+1}" for i in x]
        ax.set_xticklabels(short_labels, fontsize=8)
        ax.set_ylabel("Score drop")
        ax.set_title(f"qid {qid} ({label})  —  {len(sentences)} sentences", fontsize=9)

    fig.suptitle(
        "Cross-encoder sentence attribution\n"
        "(score drop when each sentence is removed; higher = more important)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()

    out_path = f"{FIGURES_DIR}/cross_encoder_attribution.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    print(f"  Saved CSV → {RESULTS_DIR}/attribution_results.csv")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_tsne_analysis()
    run_attribution_analysis()
