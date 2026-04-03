from __future__ import annotations

from pymilvus import (
    AnnSearchRequest,
    WeightedRanker,
)

def _output_fields(col) -> list[str]:
    """Return output fields to request based on what the collection schema has."""
    field_names = {f.name for f in col.schema.fields}
    fields = ["text"]
    if "parent_text" in field_names:
        fields.append("parent_text")
    if "source_file" in field_names:
        fields.append("source_file")
    if "section" in field_names:
        fields.append("section")
    return fields


def _hit_to_tuple(hit, fields: list[str]) -> tuple[str, str, float, str, str]:
    """Return (text, parent_text, score, source_file, section)."""
    text = hit.get("text") or ""
    parent = hit.get("parent_text") if "parent_text" in fields else None
    source_file = hit.get("source_file") or "" if "source_file" in fields else ""
    section = hit.get("section") or "" if "section" in fields else ""
    return (text, parent or text, hit.score, source_file, section)


def dense_search(col, query_dense_embedding, limit=5):
    fields = _output_fields(col)
    search_params = {"metric_type": "IP", "params": {"ef": 128}}
    res = col.search(
        [query_dense_embedding],
        anns_field="dense_vector",
        limit=limit,
        output_fields=fields,
        param=search_params,
    )[0]
    return [_hit_to_tuple(hit, fields) for hit in res]


def sparse_search(col, query_sparse_embedding, limit=5):
    fields = _output_fields(col)
    search_params = {"metric_type": "IP", "params": {"ef": 128}}
    res = col.search(
        [query_sparse_embedding],
        anns_field="sparse_vector",
        limit=limit,
        output_fields=fields,
        param=search_params,
    )[0]
    return [_hit_to_tuple(hit, fields) for hit in res]


def hybrid_search(col, query_dense_embedding, query_sparse_embedding, sparse_weight=1.0, dense_weight=0.7, limit=5):
    fields = _output_fields(col)
    dense_search_params = {"metric_type": "IP", "params": {"ef": 128}}
    dense_req = AnnSearchRequest(
        [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
    )
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest(
        [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search(
        [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=fields
    )[0]
    return [_hit_to_tuple(hit, fields) for hit in res]


def rerank_candidates(
    query: str,
    candidates: list[tuple[str, str, float, str, str]],
    reranker,
    top_k: int = 5,
) -> list[tuple[str, str, float, str, str]]:
    """Re-score candidates with a cross-encoder and return top_k results.

    Args:
        query:      The user query string.
        candidates: List of (text, parent_text, score, source_file, section).
        reranker:   A FlagReranker instance (cross-encoder).
        top_k:      How many to return after reranking.
    """
    if not candidates:
        return []
    # Rerank on small chunks (text) for precision; other fields are passed through
    pairs = [[query, text] for text, parent, _, sf, sec in candidates]
    scores = reranker.compute_score(pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [scores]
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [
        (text, parent, float(score), sf, sec)
        for (text, parent, _, sf, sec), score in ranked[:top_k]
    ]