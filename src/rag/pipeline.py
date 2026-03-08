from typing import Any
from dataclasses import dataclass
from rag.chunker import chunk_text
from rag.embedder import embed_query, embed_texts, load_bge_m3_embedder
from rag.parser import build_llama_parser, merge_documents, parse_documents
from rag.retriever import hybrid_search
from storage.vector_store import MilvusVectorStore

@dataclass
class RAGContext:
    store: MilvusVectorStore
    embedder: Any
    col_name: str
    dense_dim: int

_PROBE_DIM = 1024  # BGE-M3 dense dim; used only to open existing collections


def ingest(data_dir="data", uri="./milvus.db", col_name="hybrid"):
    # --- Fast path: collection already populated, skip parse/embed/insert ---
    probe = MilvusVectorStore(uri=uri, col_name=col_name, dense_dim=_PROBE_DIM)
    if probe.already_exists:
        embedder = load_bge_m3_embedder()
        actual_dim = probe.col.schema.fields[-1].params["dim"]
        print(f"[ingest] {col_name}: reusing existing collection ({probe.col.num_entities} chunks).")
        return RAGContext(store=probe, embedder=embedder, col_name=col_name, dense_dim=actual_dim)

    # --- Slow path: parse → chunk → embed → insert ---
    parser = build_llama_parser()
    docs = parse_documents(data_dir, parser)
    text = merge_documents(docs)

    chunks = chunk_text(text, max_chunk_size=300, hard_max_length=512)
    if not chunks:
        raise ValueError("No chunks generated from input documents")

    embedder = load_bge_m3_embedder()
    emb = embed_texts(chunks, embedder)
    dense_dim = emb["dense"][0].shape[0]

    # Re-init with correct dim (probe used placeholder dim and found no data)
    store = MilvusVectorStore(uri=uri, col_name=col_name, dense_dim=dense_dim)
    store.insert(chunks, emb)
    return RAGContext(store=store, embedder=embedder, col_name=col_name, dense_dim=dense_dim)

def _get_sparse_row(sparse_matrix, idx: int):
    """Safely extract one row from a scipy sparse matrix."""
    if hasattr(sparse_matrix, "getrow"):
        return sparse_matrix.getrow(idx)
    if hasattr(sparse_matrix, "_getrow"):
        return sparse_matrix._getrow(idx)
    return sparse_matrix[idx]


def retrieve(query, ctx, limit=5, sparse_weight=1.0, dense_weight=0.7, score_threshold=0.0):
    query_emb = embed_query(query, ctx.embedder)
    dense = query_emb["dense"][0]
    sparse = _get_sparse_row(query_emb["sparse"], 0)
    results = hybrid_search(
        ctx.store.col,
        dense,
        sparse,
        sparse_weight=sparse_weight,
        dense_weight=dense_weight,
        limit=limit,
    )
    if score_threshold > 0:
        results = [(t, s) for t, s in results if s >= score_threshold]
    return [{"text": t, "score": round(s, 4), "rank": i + 1} for i, (t, s) in enumerate(results)]


def format_citations(items):
    return "".join(f"[{x['rank']}] {x['text']}\n" for x in items)
