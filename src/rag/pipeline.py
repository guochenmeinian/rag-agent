import hashlib
import json
import os
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import config
from rag.chunker import chunk_text
from rag.embedder import embed_query, embed_texts, load_bge_m3_embedder
from rag.embedder import load_bge_reranker
from rag.parser import build_llama_parser, parse_single_file
from rag.retriever import hybrid_search
from rag.retriever import rerank_candidates
from storage.vector_store import MilvusVectorStore
from storage.ingest_manager import IngestManager
from storage.grep_index import GrepIndex

@dataclass
class RAGContext:
    store: MilvusVectorStore
    embedder: Any
    col_name: str
    dense_dim: int
    reranker: Any = None

_PROBE_DIM = 1024  # BGE-M3 dense dim; used only to open existing collections

# IngestManager cache keyed by URI hash — supports multiple URIs in the same process
_ingest_managers: dict[str, IngestManager] = {}

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
        chunk_triples = chunk_text(merged, max_chunk_size=600, hard_max_length=1200)
        parent_chunks = [p for _, p, _ in chunk_triples]
        if parent_chunks:
            gidx.insert_chunks(col_name, parent_chunks)
            print(f"[ingest] {col_name}: grep index backfilled from cache ({len(parent_chunks)} chunks)")
    except Exception as e:
        print(f"[ingest] {col_name}: grep backfill skipped - {e}")


def _generate_chunk_contexts(
    chunks: list[str],
    doc_text: str,
    col_name: str,
    cache_dir: str,
    pipeline_flags: dict | None = None,
) -> list[str]:
    """Prepend LLM-generated context summaries to each chunk (Contextual Retrieval).

    For each chunk, calls Qwen with the first ~1500 chars of the document plus the chunk
    to generate a 1-2 sentence description. The description is prepended so that the
    embedding captures both the chunk's position in the document and its content.

    Results are cached by content hash + pipeline config — re-runs with unchanged
    chunks and config are free.

    Returns a list of strings: "<context description>\\n<original chunk>"
    """
    # Cache key: MD5 of chunk content + pipeline config that affects chunk shape
    flags = pipeline_flags or {}
    config_tag = f"v{flags.get('chunk_version', 1)}_hmax{flags.get('hard_max_length', 900)}"
    content_hash = hashlib.md5("\n".join(chunks).encode()).hexdigest()[:12]
    cache_path = os.path.join(cache_dir, f"{col_name}_ctx_{config_tag}_{content_hash}.json")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if len(cached) == len(chunks):
                print(f"[ingest] {col_name}: loaded {len(chunks)} chunk contexts from cache")
                return cached
        except (json.JSONDecodeError, IOError):
            pass

    from agent.qwen_client import get_qwen_client
    client = get_qwen_client()
    model = getattr(config, "CONTEXTUAL_RETRIEVAL_MODEL", config.QWEN_MODEL)

    doc_prefix = doc_text[:1500]
    contextualized: list[str] = []
    total = len(chunks)
    print(f"[ingest] {col_name}: generating contexts for {total} chunks (model={model})...")

    for i, chunk in enumerate(chunks):
        prompt = (
            f"<document_excerpt>\n{doc_prefix}\n</document_excerpt>\n\n"
            f"以下是该文档中的一个片段：\n<chunk>\n{chunk}\n</chunk>\n\n"
            "请用1-2句话描述这个片段的主要内容，包括涉及的车型和参数类别。直接输出描述，不加任何前缀。"
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0,
                timeout=30,
                extra_body={"enable_thinking": False},    
            )
            ctx_text = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"\n  [ctx] chunk {i}: failed ({e}), skipping context")
            ctx_text = ""

        contextualized.append(f"{ctx_text}\n{chunk}" if ctx_text else chunk)

        # 进度条
        done = i + 1
        bar_len = 30
        filled = int(bar_len * done / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {done}/{total}", end="", flush=True)

    print()  # 换行

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(contextualized, f, ensure_ascii=False, indent=2)
    print(f"[ingest] {col_name}: context generation done, cached to {cache_path}")

    return contextualized


def get_ingest_manager() -> IngestManager:
    """获取 IngestManager 实例，按 MILVUS_URI 缓存，URI 变化时自动新建"""
    uri = config.MILVUS_URI
    uri_tag = hashlib.md5(uri.encode()).hexdigest()[:8]
    if uri_tag not in _ingest_managers:
        project_root = Path(__file__).resolve().parents[2]
        manifest_path = str(project_root / f".ingest_manifest_{uri_tag}.json")
        _ingest_managers[uri_tag] = IngestManager(manifest_path=manifest_path)
    return _ingest_managers[uri_tag]


def ingest(data_dir="data", uri="http://localhost:19530", col_name="hybrid", force=False, file_filter=None, grep_path=None):
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

    # 检查 collection 状态（probe 仅用于检测，skip 路径外立即 release 释放内存）
    probe = MilvusVectorStore(uri=uri, col_name=col_name, dense_dim=_PROBE_DIM)
    collection_exists = probe.already_exists
    collection_count = probe.col.num_entities if collection_exists else 0

    # 检查是否需要 ingest
    pipeline_flags = {
        "contextual_retrieval": getattr(config, "CONTEXTUAL_RETRIEVAL", True),
        "chunk_version": 3,        # v3 = table-to-text + heading injection + parent chunk
        "hard_max_length": 1200,
    }
    if not force:
        status = manager.check_ingest_status(
            pdf_files=pdf_files,
            col_name=col_name,
            collection_exists=collection_exists,
            collection_count=collection_count,
            pipeline_flags=pipeline_flags,
        )

        if status.skip:
            embedder = load_bge_m3_embedder()
            reranker = load_bge_reranker()
            actual_dim = probe.col.schema.fields[-1].params["dim"]
            print(f"[ingest] {col_name}: skipping - {status.reason}")
            # 若 grep 索引缺失，从 parse 缓存回填
            _backfill_grep_if_missing(col_name, pdf_files, manager)
            return RAGContext(store=probe, embedder=embedder, col_name=col_name, dense_dim=actual_dim, reranker=reranker)


        print(f"[ingest] {col_name}: need to ingest - {status.reason}")
    else:
        print(f"[ingest] {col_name}: force re-ingest")

    # Slow path: 需要 ingest
    # 1. 初始化 parser
    parser = build_llama_parser()

    # 2. Parse 每个文件（使用缓存）并按文件分别 chunk，追踪 source_file
    all_texts: dict[str, str] = {}
    for filepath in sorted(pdf_files):
        parsed_text = manager.get_or_parse(
            filepath,
            lambda fp: parse_single_file(fp, parser),
            verbose=True,
        )
        all_texts[filepath] = parsed_text

    # 3. Chunk per-file — track source_file and section per chunk
    small_chunks: list[str] = []
    parent_chunks: list[str] = []
    chunk_source_files: list[str] = []
    chunk_sections: list[str] = []

    for filepath, text in all_texts.items():
        fname = os.path.basename(filepath)
        triples = chunk_text(text, max_chunk_size=600, hard_max_length=1200)
        for s, p, sec in triples:
            small_chunks.append(s)
            parent_chunks.append(p)
            chunk_source_files.append(fname)
            chunk_sections.append(sec)

    if not small_chunks:
        raise ValueError("No chunks generated from input documents")

    # merged_text is used only as document prefix for contextual retrieval
    merged_text = "".join(all_texts.values())

    print(f"[ingest] {col_name}: generated {len(small_chunks)} chunks from {len(all_texts)} file(s)")

    # 3b. Contextual Retrieval: prepend LLM-generated context to small chunks only
    if getattr(config, "CONTEXTUAL_RETRIEVAL", True):
        ctx_cache_dir = str(manager.cache_dir / "contexts")
        small_chunks = _generate_chunk_contexts(small_chunks, merged_text, col_name, ctx_cache_dir, pipeline_flags)

    # 4. Embed small chunks (contextualised or plain)
    embedder = load_bge_m3_embedder()
    reranker = load_bge_reranker()
    emb = embed_texts(small_chunks, embedder)
    dense_dim = emb["dense"][0].shape[0]

    # 5. 如果 collection 已存在但需要重建，先释放再删除
    if collection_exists:
        print(f"[ingest] {col_name}: dropping existing collection")
        probe.release()
        probe.col.drop()
    else:
        probe.release()

    # 6. 重新创建并插入（同时存 small chunk 和 parent chunk）
    store = MilvusVectorStore(uri=uri, col_name=col_name, dense_dim=dense_dim)
    store.insert(small_chunks, emb, parent_chunks=parent_chunks,
                 source_files=chunk_source_files, sections=chunk_sections)

    print(f"[ingest] {col_name}: inserted {len(small_chunks)} chunks")

    # 6b. 写入 grep 全文索引（用 parent chunks，内容更完整）
    grep_path = grep_path or getattr(config, "GREP_INDEX_PATH", None)
    if grep_path:
        gidx = GrepIndex(grep_path)
        gidx.insert_chunks(col_name, parent_chunks)
        print(f"[ingest] {col_name}: grep index updated")

    # 7. 更新 manifest（只记录实际处理的文件）
    manager.mark_ingested_files(col_name, pdf_files, pipeline_flags=pipeline_flags)

    return RAGContext(store=store, embedder=embedder, col_name=col_name, dense_dim=dense_dim, reranker=reranker)


def _get_sparse_row(sparse_matrix, idx: int):
    """Safely extract one row from a scipy sparse matrix."""
    if hasattr(sparse_matrix, "getrow"):
        return sparse_matrix.getrow(idx)
    if hasattr(sparse_matrix, "_getrow"):
        return sparse_matrix._getrow(idx)
    return sparse_matrix[idx]


def retrieve(query, ctx, limit=5, search_limit=10, sparse_weight=0.5, dense_weight=1.0, score_threshold=0.35):
    """Hybrid retrieval (dense + sparse BM25) with score threshold filtering.

    Args:
        limit:           Max chunks returned to caller.
        search_limit:    Candidates fetched from Milvus before threshold filtering.
        sparse_weight:   Weight for BM25 keyword matching.
        dense_weight:    Weight for semantic dense embedding.
        score_threshold: Minimum hybrid score to keep a chunk.
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
        limit=search_limit,
    )
    total_before_filter = len(raw)
    if ctx.reranker is not None:
        raw = rerank_candidates(query, raw, ctx.reranker, top_k=limit)
    else:
        if score_threshold > 0:
            raw = [(t, p, s, sf, sec) for t, p, s, sf, sec in raw if s >= score_threshold]
        raw = raw[:limit]

    # text = parent chunk (full context for LLM); chunk = small chunk used for retrieval
    items = [
        {
            "text": p,
            "chunk": t,
            "score": round(s, 4),
            "rank": i + 1,
            "source_file": sf,
            "section": sec,
        }
        for i, (t, p, s, sf, sec) in enumerate(raw)
    ]
    for item in items:
        item["_total_before_filter"] = total_before_filter
        item["_score_threshold"] = score_threshold
    return items


def format_citations(items):
    parts = []
    for x in items:
        source_info = ""
        if x.get("source_file"):
            source_info = f" ({x['source_file']}"
            if x.get("section"):
                source_info += f" · {x['section']}"
            source_info += ")"
        parts.append(f"[{x['rank']}]{source_info} {x['text']}\n")
    return "".join(parts)