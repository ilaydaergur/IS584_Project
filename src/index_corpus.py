"""
Build a PyTerrier index over the TREC Tip-of-the-Tongue 2024 corpus.

Inputs:  none (corpus loaded automatically via ir_datasets)
Outputs: Terrier index written to ./index/
"""

import os
import pyterrier as pt
import ir_datasets

INDEX_DIR = "./index"
TEXT_SNIPPET_CHARS = 2000  # characters stored as metadata for reranking


def corpus_generator(corpus_ds):
    """Yield index-ready dicts from the ir_datasets corpus iterator.

    Args:
        corpus_ds: ir_datasets dataset object with a docs_iter() method
    Yields:
        dict with keys: docno, text (for BM25 field), title, text_snippet
    """
    for doc in corpus_ds.docs_iter():
        yield {
            "docno": doc.doc_id,
            "text": doc.text,
            "title": getattr(doc, "page_title", ""),
            "text_snippet": doc.text[:TEXT_SNIPPET_CHARS],
        }


def build_index():
    """Index the TREC-ToT 2024 corpus with PyTerrier.

    Skips if the index already exists at INDEX_DIR.
    Indexes 'text' and 'title' fields for BM25; stores a text snippet
    in metadata so the reranker can retrieve it without a second corpus pass.
    """
    if os.path.exists(os.path.join(INDEX_DIR, "data.properties")):
        print(f"Index already exists at '{INDEX_DIR}'. Delete the folder to rebuild.")
        return

    print("Loading corpus via ir_datasets (trec-tot/2024)...")
    corpus_ds = ir_datasets.load("trec-tot/2024")
    print(f"Corpus size: {corpus_ds.docs_count()} documents")

    os.makedirs(INDEX_DIR, exist_ok=True)
    indexer = pt.IterDictIndexer(
        INDEX_DIR,
        fields=["text", "title"],
        meta={"docno": 26, "text_snippet": TEXT_SNIPPET_CHARS + 10, "title": 512},
        overwrite=True,
    )
    print("Indexing... (this may take 30–60 minutes on the full corpus)")
    indexref = indexer.index(corpus_generator(corpus_ds))
    index = pt.terrier.IndexFactory.of(indexref)
    print(f"Done. Documents indexed: {index.getCollectionStatistics().getNumberOfDocuments()}")
    print(f"Index location: {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
