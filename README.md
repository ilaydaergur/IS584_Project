# IS584 Term Project — TREC Tip-of-the-Tongue Retrieval

**Student:** İlayda Ergür  
**Student ID:** 2784247  
**Course:** IS584 — Information Retrieval, METU  
**Dataset:** TREC Tip-of-the-Tongue (ToT) 2024 corpus / 2025 dev1 queries  
**WANDB Project:** [is584-tot-retrieval](https://wandb.ai/ilaydaergur/is584-tot-retrieval)

---

## Research Questions

- **RQ1:** Does cross-encoder reranking significantly improve retrieval effectiveness over BM25 on tip-of-the-tongue queries (Wilcoxon signed-rank test, α = 0.05)?
- **RQ2:** Does the reranker gain differ between long vs. short queries and between zero vs. non-zero keyword-overlap queries?

---

## Project Structure

```
IS584_Project/
├── src/
│   ├── index_corpus.py       # Build PyTerrier index over TREC-ToT 2024 corpus
│   ├── run_bm25.py           # BM25 baseline + WANDB parameter sweep
│   ├── run_reranker.py       # Cross-encoder reranking pipeline
│   └── evaluate.py           # Wilcoxon test (RQ1) + subgroup analysis (RQ2)
├── outputs/
│   ├── bm25_results.csv      # BM25 aggregate metrics
│   ├── bm25_per_query.csv    # BM25 per-query metrics
│   ├── reranker_results.csv  # CrossEncoder aggregate metrics
│   ├── reranker_per_query.csv
│   └── rq2_analysis.csv      # Per-query features + system scores
├── reports/
│   └── IS584_Phase1.pdf
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create and activate the conda environment

```bash
conda create -n ai python=3.11
conda activate ai
conda install -c conda-forge openjdk=21
pip install -r requirements.txt
```

### 2. Login to Weights & Biases

```bash
wandb login
```

---

## Reproducing the Experiments

Run the scripts in order from the project root directory.

### Step 1 — Build the index

```bash
python src/index_corpus.py
```

Downloads and indexes the TREC-ToT 2024 corpus (~3.18 M Wikipedia documents) into `./index/`. Takes approximately 30–60 minutes on first run; subsequent runs are skipped automatically.

### Step 2 — BM25 parameter sweep

```bash
python src/run_bm25.py --mode sweep
```

Runs a WANDB grid sweep over k1 ∈ {0.5, 1.0, 1.2, 1.5, 2.0} and b ∈ {0.3, 0.5, 0.75, 1.0} (20 runs total). Results are logged to the WANDB project.

### Step 3 — BM25 best run

```bash
python src/run_bm25.py --mode best --k1 1.2 --b 0.75
```

Retrieves with the best parameters found in the sweep. Saves `outputs/bm25_run.csv`, `outputs/bm25_results.csv`, and `outputs/bm25_per_query.csv`.

### Step 4 — Cross-encoder reranking

```bash
python src/run_reranker.py
```

Reranks the top-100 BM25 candidates per query using `cross-encoder/ms-marco-MiniLM-L-6-v2`. Queries are truncated to 100 words before scoring to fit the 512-token model limit. Saves `outputs/reranker_run.csv`, `outputs/reranker_results.csv`, and `outputs/reranker_per_query.csv`.

Optional arguments:
```bash
python src/run_reranker.py --model cross-encoder/ms-marco-MiniLM-L-12-v2 \
                           --batch_size 16 \
                           --rerank_depth 50
```

### Step 5 — Statistical evaluation

```bash
python src/evaluate.py --metric RR@10
```

Runs the Wilcoxon signed-rank test (RQ1) and subgroup analysis (RQ2). Saves `outputs/rq2_analysis.csv`.

---

## Results (Phase 2)

Evaluated on **trec-tot/2025/dev1** (142 queries, same 2024 corpus).

| System | RR@10 | nDCG@10 | R@100 |
|---|---|---|---|
| BM25 (k1=1.2, b=0.75) | 0.0751 | 0.0812 | 0.2183 |
| CrossEncoder reranker | 0.0463 | 0.0635 | 0.2183 |

**RQ1 — Wilcoxon test:** W = 54.5, p = 0.1023 (not significant at α = 0.05). The reranker does not significantly outperform BM25.

**RQ2 — Subgroup analysis:**

| Subgroup | n | BM25 RR@10 | Reranker RR@10 | Δ |
|---|---|---|---|---|
| Long queries (> median) | 71 | 0.0829 | 0.0118 | −0.0711 |
| Short queries (≤ median) | 71 | 0.0673 | 0.0807 | +0.0134 |
| Zero keyword overlap | 142 | 0.0751 | 0.0463 | −0.0288 |

All 142 queries have zero Jaccard overlap with their top BM25 result — confirming that ToT queries are purely semantic and keyword matching alone is insufficient.

---

## Dependencies

| Package | Purpose |
|---|---|
| `pyterrier` | Indexing and retrieval pipeline |
| `ir-datasets` | TREC-ToT corpus and queries |
| `ir-measures` | MRR@10, nDCG@10, R@100 metrics |
| `sentence-transformers` | Cross-encoder reranking model |
| `wandb` | Experiment tracking and sweep |
| `scipy` | Wilcoxon signed-rank test |
| `pandas`, `numpy` | Data processing |

Full list: `requirements.txt`
