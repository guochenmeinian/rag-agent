import os
from typing import Any
from dataclasses import dataclass

import config
from rag.chunker import chunk_text
from rag.embedder import embed_query, embed_texts, load_bge_m3_embedder
from rag.parser import build_llama_parser, parse_single_file
from rag.retriever import hybrid_search
from storage.vector_store import MilvusVectorStore
from storage.ingest_manager import IngestManager
from storage.grep_index import GrepIndex

@dataclass
class RAGContext:
    store: MilvusVectorStore
    embedder: Any
    col_name: str
    dense_dim: int

_PROBE_DIM = 1024  # BGE-M3 dense dim; used only to open existing collections

# Global IngestManager instance (shared across all ingest calls)
_ingest_manager: IngestManager | None = None

def _backfill_grep_if_missing(col_name: str, pdf_files: list, manager: IngestManager):
    """Skip 路径：若 grep 无该 col 数据，从 parse 缓存回填。"""
    grep_path = getattr(config, "GREP_INDEX_PATH", None)
    if not grep_path:
        return
    try:
        gidx = GrepIndex(grep_path)
        if gidx.has_chunks(col_name):
            return
        parser = build_llama_parser()
        all_text = []
        for fp in sorted(pdf_files):
            t = manager.get_or_parse(fp, lambda f: parse_single_file(f, parser), verbose=False)
            all_text.append(t)
        merged = "".join(all_text)
        chunks = chunk_text(merged, max_chunk_size=600, hard_max_length=900)
        if chunks:
            gidx.insert_chunks(col_name, chunks)
            print(f"[ingest] {col_name}: grep index backfilled from cache ({len(chunks)} chunks)")
    except Exception as e:
        print(f"[ingest] {col_name}: grep backfill skipped - {e}")


def get_ingest_manager() -> IngestManager:
    """获取全局 IngestManager 实例"""
    global _ingest_manager
    if _ingest_manager is None:
        _ingest_manager = IngestManager()
    return _ingest_manager


def ingest(data_dir="data", uri="./milvus.db", col_name="hybrid", force=False, file_filter=None, grep_path=None):
    """
    Ingest PDF 文档到向量数据库

    Args:
        data_dir: 数据目录
        uri: Milvus 连接 URI
        col_name: Collection 名称
        force: 是否强制重新 ingest（忽略缓存）
        file_filter: 只处理匹配的文件名（如 "EC6.pdf"），None 表示处理所有 PDF

    Returns:
        RAGContext 包含 store, embedder 等信息

    优化：
    1. 文件指纹检测 - 检测文件是否变化，没变则跳过
    2. Parse 缓存 - 即使需要 ingest，也优先使用缓存的 parse 结果
    """
    manager = get_ingest_manager()

    # 收集要处理的 PDF 文件
    pdf_files = []
    if os.path.isdir(data_dir):
        for f in os.listdir(data_dir):
            if f.lower().endswith(".pdf"):
                if file_filter is None or f == file_filter:
                    pdf_files.append(os.path.join(data_dir, f))

    if not pdf_files:
        raise ValueError(f"No PDF files found in {data_dir}" +
                         (f" matching '{file_filter}'" if file_filter else ""))

    # 检查 collection 状态
    probe = MilvusVectorStore(uri=uri, col_name=col_name, dense_dim=_PROBE_DIM)
    collection_exists = probe.already_exists
    collection_count = probe.col.num_entities if collection_exists else 0

    # 检查是否需要 ingest
    if not force:
        status = manager.check_ingest_status(
            pdf_files=pdf_files,
            col_name=col_name,
            collection_exists=collection_exists,
            collection_count=collection_count,
        )

        if status.skip:
            embedder = load_bge_m3_embedder()
            actual_dim = probe.col.schema.fields[-1].params["dim"]
            print(f"[ingest] {col_name}: skipping - {status.reason}")
            # 若 grep 索引缺失，从 parse 缓存回填
            _backfill_grep_if_missing(col_name, pdf_files, manager)
            return RAGContext(store=probe, embedder=embedder, col_name=col_name, dense_dim=actual_dim)


        print(f"[ingest] {col_name}: need to ingest - {status.reason}")
    else:
        print(f"[ingest] {col_name}: force re-ingest")

    # Slow path: 需要 ingest
    # 1. 初始化 parser
    parser = build_llama_parser()

    # 2. Parse 每个文件（使用缓存）
    all_text = []
    for filepath in sorted(pdf_files):
        parsed_text = manager.get_or_parse(
            filepath,
            lambda fp: parse_single_file(fp, parser),
            verbose=True,
        )
        all_text.append(parsed_text)

    # 3. Chunk
    merged_text = "".join(all_text)
    chunks = chunk_text(merged_text, max_chunk_size=600, hard_max_length=900)
    if not chunks:
        raise ValueError("No chunks generated from input documents")

    print(f"[ingest] {col_name}: generated {len(chunks)} chunks")

    # 4. Embed
    embedder = load_bge_m3_embedder()
    emb = embed_texts(chunks, embedder)
    dense_dim = emb["dense"][0].shape[0]

    # 5. 如果 collection 已存在但需要重建，先删除
    if collection_exists:
        print(f"[ingest] {col_name}: dropping existing collection")
        probe.col.drop()

    # 6. 重新创建并插入
    store = MilvusVectorStore(uri=uri, col_name=col_name, dense_dim=dense_dim)
    store.insert(chunks, emb)

    print(f"[ingest] {col_name}: inserted {len(chunks)} chunks")

    # 6b. 写入 grep 全文索引（Claude 风格 keyword 检索）
    grep_path = grep_path or getattr(config, "GREP_INDEX_PATH", None)
    if grep_path:
        gidx = GrepIndex(grep_path)
        gidx.insert_chunks(col_name, chunks)
        print(f"[ingest] {col_name}: grep index updated")

    # 7. 更新 manifest（只记录实际处理的文件）
    manager.mark_ingested_files(col_name, pdf_files)

    return RAGContext(store=store, embedder=embedder, col_name=col_name, dense_dim=dense_dim)


def _get_sparse_row(sparse_matrix, idx: int):
    """Safely extract one row from a scipy sparse matrix."""
    if hasattr(sparse_matrix, "getrow"):
        return sparse_matrix.getrow(idx)
    if hasattr(sparse_matrix, "_getrow"):
        return sparse_matrix._getrow(idx)
    return sparse_matrix[idx]


def retrieve(query, ctx, limit=5, search_limit=15, sparse_weight=0.5, dense_weight=1.0, score_threshold=0.35):
    """Hybrid retrieval with two-stage candidate selection.

    Args:
        limit:        Max chunks returned to caller (after filtering).
        search_limit: Candidates fetched from Milvus before threshold filtering.
                      Larger → higher recall at the cost of more computation.
        sparse_weight: Weight for BM25 keyword matching (lower → less keyword bias).
        dense_weight:  Weight for semantic dense embedding (higher → better concept recall).
        score_threshold: Minimum score to keep a chunk. Lower → more permissive.
    """
    query_emb = embed_query(query, ctx.embedder)
    dense = query_emb["dense"][0]
    sparse = _get_sparse_row(query_emb["sparse"], 0)
    raw = hybrid_search(
        ctx.store.col,
        dense,
        sparse,
        sparse_weight=sparse_weight,
        dense_weight=dense_weight,
        limit=search_limit,  # fetch more candidates before filtering
    )
    total_before_filter = len(raw)
    if score_threshold > 0:
        raw = [(t, s) for t, s in raw if s >= score_threshold]
    # Keep only top-`limit` after filtering
    raw = raw[:limit]
    items = [{"text": t, "score": round(s, 4), "rank": i + 1} for i, (t, s) in enumerate(raw)]
    for item in items:
        item["_total_before_filter"] = total_before_filter
        item["_score_threshold"] = score_threshold
    return items


def format_citations(items):
    return "".join(f"[{x['rank']}] {x['text']}\n" for x in items)