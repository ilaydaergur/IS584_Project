import os
import ir_datasets
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"
FIG_DIR = "figures"
DOC_SAMPLE_SIZE = 5000


def word_count(text):
    return len(str(text).split())


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    corpus_ds = ir_datasets.load("trec-tot/2024")
    test_ds = ir_datasets.load("trec-tot/2024/test")

    print("Loading queries...")
    query_rows = []
    for q in test_ds.queries_iter():
        query_rows.append({
            "query_id": q.query_id,
            "query": getattr(q, "query", ""),   # FIXED
        })

    queries_df = pd.DataFrame(query_rows)
    queries_df["query_length"] = queries_df["query"].apply(word_count)

    print("Sampling documents...")
    doc_rows = []
    for i, doc in enumerate(corpus_ds.docs_iter()):
        doc_rows.append({
            "doc_id": doc.doc_id,
            "text_length": word_count(doc.text)
        })
        if i + 1 >= DOC_SAMPLE_SIZE:
            break

    docs_df = pd.DataFrame(doc_rows)

    print("\n=== QUERY STATS ===")
    print(queries_df["query_length"].describe())

    print("\n=== DOCUMENT STATS (SAMPLE) ===")
    print(docs_df["text_length"].describe())

    print("\n=== EMPTY QUERY CHECK ===")
    empty_queries = (queries_df["query"].str.strip() == "").sum()
    print("Empty queries:", empty_queries)

    queries_df.to_csv(f"{OUTPUT_DIR}/queries_summary.csv", index=False)
    docs_df.to_csv(f"{OUTPUT_DIR}/doc_sample_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(queries_df["query_length"], bins=20)
    plt.xlabel("Query length (words)")
    plt.ylabel("Frequency")
    plt.title("Query Length Distribution")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/query_length_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(docs_df["text_length"], bins=20)
    plt.xlabel("Document length (words)")
    plt.ylabel("Frequency")
    plt.title("Document Length Distribution (Sample)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/document_length_distribution_sample.png", dpi=300)
    plt.close()

    print("\nEDA completed.")
    print(f"Saved: {OUTPUT_DIR}/queries_summary.csv")
    print(f"Saved: {OUTPUT_DIR}/doc_sample_summary.csv")
    print(f"Saved: {FIG_DIR}/query_length_distribution.png")
    print(f"Saved: {FIG_DIR}/document_length_distribution_sample.png")


if __name__ == "__main__":
    main()