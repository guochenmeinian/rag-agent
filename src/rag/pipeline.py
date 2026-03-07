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

def ingest(data_dir="data", uri="./milvus.db", col_name="hybrid"):
    parser = build_llama_parser()
    docs = parse_documents(data_dir, parser)
    text = merge_documents(docs)

    chunks = chunk_text(text, max_chunk_size=300, hard_max_length=512)
    if not chunks:
        raise ValueError("No chunks generated from input documents")
    
    embedder = load_bge_m3_embedder()
    emb = embed_texts(chunks, embedder)
    dense_dim = emb["dense"][0].shape[0]

    store = MilvusVectorStore(uri=uri, col_name=col_name, dense_dim=dense_dim)
    store.insert(chunks, emb)
    return RAGContext(store=store, embedder=embedder, col_name=col_name, dense_dim=dense_dim)

def retrieve(query, ctx, limit=10, sparse_weight=0.7, dense_weight=1.0):
    query_emb = embed_query(query, ctx.embedder)
    dense = query_emb["dense"][0]
    sparse = query_emb["sparse"]._getrow(0)
    texts = hybrid_search(
        ctx.store.col,
        dense,
        sparse,
        sparse_weight=sparse_weight,
        dense_weight=dense_weight,
        limit=limit,
    )
    return [{"text": t, "rank": i + 1} for i, t in enumerate(texts)]

def format_citations(items):
    return "".join(f"[{x['rank']}] {x['text']}\n" for x in items)
