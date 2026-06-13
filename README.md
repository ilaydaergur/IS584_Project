# IS584 Term Project — TREC Tip-of-the-Tongue Retrieval

**Student:** İlayda Ergür  
**Student ID:** 2784247  
**Course:** IS584 — Information Retrieval, METU  
**Dataset:** TREC Tip-of-the-Tongue (ToT) 2024 corpus / 2025 dev1 queries  
**WANDB Project:** [is584-tot-retrieval](https://wandb.ai/ilaydaergur/is584-tot-retrieval)

---

## Research Questions

- **RQ1:** Does cross-encoder reranking significantly improve retrieval effectiveness over BM25 on tip-of-the-tongue queries (Wilcoxon signed-rank test, α = 0.05)?
- **RQ2:** Does the reranker gain differ between long vs. short queries?
- **RQ3:** Does bi-encoder reranking outperform BM25 and the cross-encoder on ToT queries?

---

## Project Structure

```
IS584_Project/
├── src/
│   ├── index_corpus.py          # Build PyTerrier index over TREC-ToT 2024 corpus
│   ├── run_bm25.py              # BM25 baseline + WANDB parameter sweep
│   ├── run_reranker.py          # Cross-encoder reranking pipeline
│   ├── run_biencoder.py         # Bi-encoder reranking pipeline + WANDB sweep
│   ├── evaluate.py              # Wilcoxon tests (RQ1/RQ3) + subgroup analysis (RQ2)
│   ├── interpretability.py      # t-SNE embedding visualisation + sentence attribution
│   ├── error_analysis.py        # Per-query outcome classification across all systems
│   └── generate_figures.py      # Publication-quality figures for the report
├── outputs/
│   ├── bm25_results.csv         # BM25 aggregate metrics
│   ├── bm25_per_query.csv       # BM25 per-query metrics
│   ├── reranker_results.csv     # CrossEncoder aggregate metrics
│   ├── reranker_per_query.csv
│   ├── biencoder_results.csv    # BiEncoder aggregate metrics (best config)
│   ├── biencoder_per_query.csv
│   ├── biencoder_sweep_results.csv  # All 4 sweep configurations
│   ├── rq2_analysis.csv         # Per-query features + all system scores
│   ├── attribution_results.csv  # Cross-encoder sentence attribution scores
│   └── error_analysis.csv       # Per-query outcome classification
├── figures/
│   ├── aggregate_comparison.png
│   ├── subgroup_comparison.png
│   ├── tsne_embedding_space.png
│   ├── cross_encoder_attribution.png
│   ├── scatter_bm25_vs_biencoder.png
│   └── error_analysis.png
├── reports/
│   ├── IS584_Phase3.tex         # Final IEEE-format report (LaTeX source)
│   └── IS584_Phase3.pdf         # Final report PDF
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

Run all scripts from the **project root directory** (`IS584_Project/`) in the order shown below.

### Step 1 — Build the index

```bash
python src/index_corpus.py
```

Downloads and indexes the TREC-ToT 2024 corpus (~3.18 M Wikipedia documents) into `./index/`. Takes approximately 30–60 minutes on first run; subsequent runs are skipped automatically.

### Step 2 — BM25 hyperparameter sweep

```bash
python src/run_bm25.py --mode sweep
```

Runs a WANDB grid sweep over k1 ∈ {0.5, 1.0, 1.2, 1.5, 2.0} and b ∈ {0.3, 0.5, 0.75, 1.0} (20 runs). Results are logged to the WANDB project.

### Step 3 — BM25 best run

```bash
python src/run_bm25.py --mode best --k1 1.2 --b 0.75
```

Saves `outputs/bm25_run.csv`, `outputs/bm25_results.csv`, and `outputs/bm25_per_query.csv`.

### Step 4 — Cross-encoder reranking

```bash
python src/run_reranker.py
```

Reranks the top-100 BM25 candidates per query using `cross-encoder/ms-marco-MiniLM-L-6-v2`. Saves `outputs/reranker_run.csv`, `outputs/reranker_results.csv`, and `outputs/reranker_per_query.csv`.

Optional arguments:

```bash
python src/run_reranker.py --model cross-encoder/ms-marco-MiniLM-L-12-v2 \
                           --batch_size 16 \
                           --rerank_depth 50
```

### Step 5 — Bi-encoder hyperparameter sweep

```bash
python src/run_biencoder.py --mode sweep
```

Runs a grid sweep over 2 SentenceTransformer models × 2 reranking depths (4 configurations total), logging each as an individual WANDB run. Results are saved to `outputs/biencoder_sweep_results.csv`.

### Step 6 — Bi-encoder best run

```bash
python src/run_biencoder.py --mode best \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --rerank_depth 100
```

Saves `outputs/biencoder_run.csv`, `outputs/biencoder_results.csv`, and `outputs/biencoder_per_query.csv`.

### Step 7 — Statistical evaluation

```bash
python src/evaluate.py --metric RR@10
```

Runs Wilcoxon signed-rank tests (RQ1: CrossEncoder vs BM25; RQ3: BiEncoder vs BM25 and CrossEncoder) and subgroup analysis (RQ2). Saves `outputs/rq2_analysis.csv`.

### Step 8 — Interpretability analysis

```bash
python src/interpretability.py
```

Produces:
- `figures/tsne_embedding_space.png` — t-SNE projection of bi-encoder query and document embeddings
- `figures/cross_encoder_attribution.png` — sentence-level attribution for the cross-encoder
- `outputs/attribution_results.csv`

Requires `outputs/biencoder_run.csv` and `outputs/bm25_run.csv` (Steps 3 and 6).

### Step 9 — Error analysis

```bash
python src/error_analysis.py
```

Classifies each query into one of four outcome categories (All fail / BM25 only / Neural only / All succeed) across all three systems. Saves `figures/error_analysis.png` and `outputs/error_analysis.csv`.

### Step 10 — Generate report figures

```bash
python src/generate_figures.py
```

Produces publication-quality figures used in the final report:
- `figures/aggregate_comparison.png`
- `figures/subgroup_comparison.png`
- `figures/scatter_bm25_vs_biencoder.png`

---

## Results (Phase 3)

Evaluated on **trec-tot/2025/dev1** (142 queries, TREC-ToT 2024 corpus).

### Bi-encoder sweep (RQ3 — model selection)

| Model | Rerank depth | RR@10 | nDCG@10 | R@100 |
|---|---|---|---|---|
| msmarco-distilbert-base-v4 | 50 | 0.0441 | 0.0599 | 0.1761 |
| msmarco-distilbert-base-v4 | 100 | 0.0329 | 0.0420 | 0.2183 |
| **all-MiniLM-L6-v2** | **50** | **0.0470** | **0.0675** | **0.1761** |
| **all-MiniLM-L6-v2** | **100** | **0.0534** | **0.0707** | **0.2183** |

Best configuration: `all-MiniLM-L6-v2`, depth=100.

### Aggregate metrics (all three systems)

| System | RR@10 | nDCG@10 | R@100 |
|---|---|---|---|
| BM25 (k1=1.2, b=0.75) | 0.0751 | 0.0812 | 0.2183 |
| CrossEncoder (ms-marco-MiniLM-L-6-v2) | 0.0463 | 0.0635 | 0.2183 |
| BiEncoder (all-MiniLM-L6-v2, depth=100) | 0.0534 | 0.0707 | 0.2183 |

**BM25 leads on RR@10 and nDCG@10; all three systems achieve the same R@100.**

### Statistical testing (Wilcoxon signed-rank, α = 0.05)

| Comparison | Δ mean RR@10 | p-value | Significant? |
|---|---|---|---|
| CrossEncoder vs BM25 | −0.0289 | 0.1023 | No |
| BiEncoder vs BM25 | −0.0217 | 0.3382 | No |
| BiEncoder vs CrossEncoder | +0.0072 | 0.6811 | No |

### Error analysis

| Outcome | n | % |
|---|---|---|
| All fail | 117 | 82.4% |
| All succeed | 13 | 9.2% |
| Neural only | 11 | 7.7% |
| BM25 only | 1 | 0.7% |

---

## Dependencies

| Package | Purpose |
|---|---|
| `pyterrier` | Indexing and retrieval pipeline |
| `ir-datasets` | TREC-ToT corpus and queries |
| `ir-measures` | MRR@10, nDCG@10, R@100 metrics |
| `sentence-transformers` | Cross-encoder and bi-encoder models |
| `wandb` | Experiment tracking and hyperparameter sweeps |
| `scipy` | Wilcoxon signed-rank test |
| `scikit-learn` | t-SNE embedding projection |
| `matplotlib` | Figures |
| `pandas`, `numpy` | Data processing |

Full list: `requirements.txt`
