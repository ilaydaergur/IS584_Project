import ir_datasets


def main():
    corpus_ds = ir_datasets.load("trec-tot/2024")
    test_ds = ir_datasets.load("trec-tot/2024/test")

    print("=== BASIC INFO ===")
    print("Corpus dataset:", corpus_ds.dataset_id())
    print("Test dataset:", test_ds.dataset_id())
    print("Number of corpus docs:", corpus_ds.docs_count())
    print("Number of test queries:", test_ds.queries_count())

    print("\n=== SAMPLE QUERIES ===")
    for i, query in enumerate(test_ds.queries_iter()):
        print(f"\nQuery {i+1}")
        print(query)
        if i == 2:
            break

    print("\n=== SAMPLE DOCUMENTS ===")
    for i, doc in enumerate(corpus_ds.docs_iter()):
        print(f"\nDocument {i+1}")
        print("doc_id:", doc.doc_id)
        print("page_title:", getattr(doc, "page_title", "N/A"))
        print("text snippet:", doc.text[:300])
        if i == 2:
            break


if __name__ == "__main__":
    main()