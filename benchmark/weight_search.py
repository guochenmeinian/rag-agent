"""Hybrid search weight experiment.

Tests a grid of dense_ratio values (dense_ratio ∈ [0.1, 0.9]) where:
    dense_weight  = dense_ratio
    sparse_weight = 1 - dense_ratio

Milvus WeightedRanker requires both weights in [0, 1], so we always normalise
to (dense_ratio, 1-dense_ratio).  Only the *ratio* matters anyway — multiplying
both by any constant gives identical results.

Metric: facts_coverage — fraction of expected key_facts that appear as
substrings in the combined text of the top-k retrieved chunks.

Key optimisation: each query is embedded ONCE (embedding is independent of
weights), then hybrid_search is called with different weight combos — so the
total number of model calls == number of test cases, not (cases × combos).

Reranker is intentionally disabled so we isolate the effect of the weights
themselves.  If you want end-to-end numbers, pass --rerank.

Usage
─────
cd /path/to/rag-agent
python benchmark/weight_search.py                   # default grid (9 ratios)
python benchmark/weight_search.py --top 3           # show top-3 combos
python benchmark/weight_search.py --rerank          # include reranker
python benchmark/weight_search.py --out ws.json     # save full results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

import config as src_config
from rag.pipeline import ingest, _get_sparse_row
from rag.embedder import embed_query
from rag.retriever import hybrid_search, rerank_candidates

# ─────────────────────────────────────────────────────────────
# Grid definition
# ─────────────────────────────────────────────────────────────
# dense_ratio = dense_weight / (dense_weight + sparse_weight)
# We test ratios from 0.1 (very sparse-heavy) to 0.9 (very dense-heavy).
# Production default: dense=1.0, sparse=0.5 → ratio = 1.0/1.5 ≈ 0.667

DEFAULT_DENSE_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Car models in query order — ET5T must come before ET5 to avoid partial match
_MODEL_SCAN_ORDER = ["EC6", "EC7", "ES6", "ES8", "ET5T", "ET5", "ET7", "ET9"]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def detect_models(query: str) -> list[str]:
    """Return all NIO model codes found in query (ET5T before ET5)."""
    found = []
    for m in _MODEL_SCAN_ORDER:
        if m in query and m not in found:
            found.append(m)
    return found


def facts_coverage(chunks: list[dict], key_facts: list[str]) -> dict:
    """Check what fraction of key_facts appear in the retrieved chunks.

    Each fact is matched as a substring (case-insensitive) against the
    combined text of all retrieved chunks.

    Returns:
        {"coverage": float, "found": [...], "missing": [...]}
    """
    if not key_facts:
        return {"coverage": 1.0, "found": [], "missing": []}
    combined = " ".join(c.get("text", "") + " " + c.get("chunk", "") for c in chunks).lower()
    found   = [f for f in key_facts if f.lower() in combined]
    missing = [f for f in key_facts if f.lower() not in combined]
    return {
        "coverage": round(len(found) / len(key_facts), 3),
        "found":    found,
        "missing":  missing,
    }


# ─────────────────────────────────────────────────────────────
# Load RAG contexts
# ─────────────────────────────────────────────────────────────

def load_contexts() -> dict[str, object]:
    """Load one RAGContext per car model (skips ingest if already indexed)."""
    contexts: dict[str, object] = {}
    for model in _MODEL_SCAN_ORDER:
        col_name = f"nio_{model.lower()}"
        flat_pdf = os.path.join(src_config.DATA_ROOT, f"{model}.pdf")
        if not os.path.isfile(flat_pdf):
            continue
        print(f"  Loading {model} ({col_name})…", end=" ", flush=True)
        try:
            ctx = ingest(
                data_dir=src_config.DATA_ROOT,
                uri=src_config.MILVUS_URI,
                col_name=col_name,
                file_filter=f"{model}.pdf",
            )
            contexts[model] = ctx
            print("ok")
        except Exception as e:
            print(f"FAILED: {e}")
    return contexts


# ─────────────────────────────────────────────────────────────
# Pre-embed queries
# ─────────────────────────────────────────────────────────────

def preembed_cases(cases: list[dict], contexts: dict) -> list[dict]:
    """Embed each case query once; attach dense/sparse vectors."""
    # Use any context's embedder — they're all the same BGE-M3 model
    any_ctx = next(iter(contexts.values()))
    enriched = []
    total = len(cases)
    for i, case in enumerate(cases):
        query = case["input"]
        emb   = embed_query(query, any_ctx.embedder)
        enriched.append({
            **case,
            "_dense":  emb["dense"][0],
            "_sparse": _get_sparse_row(emb["sparse"], 0),
        })
        print(f"\r  Embedding queries: {i+1}/{total}", end="", flush=True)
    print()
    return enriched


# ─────────────────────────────────────────────────────────────
# Run one weight combo
# ─────────────────────────────────────────────────────────────

def run_combo(
    cases: list[dict],
    contexts: dict,
    dense_ratio: float,
    limit: int,
    search_limit: int,
    use_rerank: bool,
) -> dict:
    """Run all cases with one dense_ratio and return aggregated stats.

    dense_weight  = dense_ratio        (always in [0, 1])
    sparse_weight = 1 - dense_ratio    (always in [0, 1])
    """
    dense_w  = round(dense_ratio, 4)
    sparse_w = round(1.0 - dense_ratio, 4)
    case_results = []

    for case in cases:
        models = detect_models(case["input"])
        if not models:
            # Fallback: search all collections
            models = list(contexts.keys())

        # Collect raw candidates from relevant collections
        raw_all: list[tuple] = []
        for m in models:
            ctx = contexts.get(m)
            if ctx is None:
                continue
            hits = hybrid_search(
                ctx.store.col,
                case["_dense"],
                case["_sparse"],
                sparse_weight=sparse_w,
                dense_weight=dense_w,
                limit=search_limit,
            )
            raw_all.extend(hits)

        if not raw_all:
            case_results.append({
                "id": case["id"],
                "input": case["input"],
                "coverage": 0.0,
                "found": [],
                "missing": case.get("answer_gt", {}).get("key_facts", []),
            })
            continue

        # Sort by score, take top search_limit candidates
        raw_all.sort(key=lambda x: x[2], reverse=True)
        raw_all = raw_all[:search_limit]

        if use_rerank:
            # Find any loaded context to get the shared reranker instance
            ctx0 = next((contexts[m] for m in models if m in contexts), None) \
                   or next(iter(contexts.values()))
            if ctx0.reranker:
                raw_all = rerank_candidates(case["input"], raw_all, ctx0.reranker, top_k=limit)

        # Format as chunks list (text = parent chunk for richer context)
        chunks = [
            {"id": f"{i}", "text": p, "chunk": t, "score": s}
            for i, (t, p, s, sf, sec) in enumerate(raw_all[:limit])
        ]

        key_facts = case.get("answer_gt", {}).get("key_facts", [])
        fc = facts_coverage(chunks, key_facts)

        case_results.append({
            "id":       case["id"],
            "input":    case["input"][:60],
            "coverage": fc["coverage"],
            "found":    fc["found"],
            "missing":  fc["missing"],
        })

    # Aggregate
    coverages = [r["coverage"] for r in case_results]
    avg_cov   = round(sum(coverages) / len(coverages), 3) if coverages else 0.0
    full_hits = sum(1 for c in coverages if c == 1.0)
    zero_hits = sum(1 for c in coverages if c == 0.0)

    return {
        "dense_ratio":   dense_ratio,
        "dense_weight":  dense_w,
        "sparse_weight": sparse_w,
        "avg_coverage":  avg_cov,
        "full_coverage_rate": round(full_hits / len(case_results), 3),
        "zero_coverage_rate": round(zero_hits / len(case_results), 3),
        "n_cases":       len(case_results),
        "cases":         case_results,
    }


# ─────────────────────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────────────────────

def print_results_table(grid_results: list[dict]):
    """Print results as a ranked table."""
    ranked = sorted(
        grid_results,
        key=lambda r: (r["avg_coverage"], r["full_coverage_rate"]),
        reverse=True,
    )
    print(f"\n{'dense_ratio':<14} {'dense_w':<10} {'sparse_w':<10} {'avg_cov':<12} {'full_cov%':<12} {'zero_cov%'}")
    print("-" * 70)
    for r in ranked:
        marker = " ← 当前默认" if abs(r["dense_ratio"] - 0.667) < 0.02 else ""
        full_str = f"{r['full_coverage_rate']*100:.1f}%"
        zero_str = f"{r['zero_coverage_rate']*100:.1f}%"
        print(
            f"{r['dense_ratio']:<14.2f} {r['dense_weight']:<10.4f} {r['sparse_weight']:<10.4f} "
            f"{r['avg_coverage']:<12.3f} {full_str:<12} {zero_str}{marker}"
        )


def print_top_combos(grid_results: list[dict], top_n: int):
    """Print the top-N combinations ranked by avg_coverage then full_coverage_rate."""
    ranked = sorted(
        grid_results,
        key=lambda r: (r["avg_coverage"], r["full_coverage_rate"]),
        reverse=True,
    )
    print(f"\nTop {top_n} weight combinations:")
    print(f"{'Rank':<5} {'dense_ratio':<14} {'avg_cov':<12} {'full_cov%':<12} {'zero_cov%'}")
    print("-" * 55)
    for i, r in enumerate(ranked[:top_n], 1):
        full_str = f"{r['full_coverage_rate']*100:.1f}%"
        zero_str = f"{r['zero_coverage_rate']*100:.1f}%"
        print(
            f"{i:<5} {r['dense_ratio']:<14.2f} "
            f"{r['avg_coverage']:<12.3f} {full_str:<12} {zero_str}"
        )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hybrid search weight grid search")
    parser.add_argument("--dataset",      default="benchmark/data/cases_rag_v2.json")
    parser.add_argument("--limit",        type=int,   default=5,    help="Final chunks returned")
    parser.add_argument("--search-limit", type=int,   default=10,   help="Candidates before rerank")
    parser.add_argument("--top",          type=int,   default=5,    help="Top-N combos to display")
    parser.add_argument("--rerank",       action="store_true",      help="Include reranker")
    parser.add_argument("--out",          default=None,             help="Save full results JSON")
    parser.add_argument("--dense-ratios", nargs="+", type=float,    default=DEFAULT_DENSE_RATIOS,
                        help="dense_ratio values to test (all must be in (0,1))")
    args = parser.parse_args()

    # Validate ratios
    for r in args.dense_ratios:
        if not (0.0 < r < 1.0):
            print(f"ERROR: dense_ratio={r} must be strictly in (0, 1)")
            sys.exit(1)

    print("=" * 60)
    print("Hybrid Search Weight Experiment")
    print("=" * 60)

    # Load test cases
    with open(args.dataset, encoding="utf-8") as f:
        all_cases = json.load(f)["cases"]
    cases = [c for c in all_cases if c.get("answer_gt", {}).get("key_facts")]
    print(f"\n数据集: {args.dataset}")
    print(f"带 key_facts 的测试用例: {len(cases)} 条")
    # Current production default: dense=1.0, sparse=0.5 → ratio=0.667
    print(f"测试的 dense_ratio 值: {args.dense_ratios}")
    print(f"当前生产默认值: dense=1.0, sparse=0.5 → dense_ratio≈0.667")
    print(f"limit={args.limit}, search_limit={args.search_limit}, rerank={args.rerank}")

    # Load RAG contexts
    print("\n[1/3] 加载 RAG contexts…")
    contexts = load_contexts()
    print(f"已加载 {len(contexts)} 个 collection: {list(contexts.keys())}")
    missing = [m for m in _MODEL_SCAN_ORDER if m not in contexts]
    if missing:
        print(f"  ⚠ 以下 collection 加载失败，涉及的 case 将返回空结果: {missing}")
        print(f"  （原因通常是 milvus.db 被 API server 占用，关闭 server 后重跑可修复）")

    # Pre-embed all queries
    print("\n[2/3] 预嵌入查询（每条查询只 embed 一次）…")
    cases_emb = preembed_cases(cases, contexts)

    # Grid search
    n_combos = len(args.dense_ratios)
    print(f"\n[3/3] 测试 {n_combos} 个 dense_ratio × {len(cases_emb)} 条用例…")

    grid_results = []
    t0 = time.time()

    for idx, ratio in enumerate(args.dense_ratios, 1):
        result = run_combo(
            cases_emb, contexts,
            dense_ratio=ratio,
            limit=args.limit,
            search_limit=args.search_limit,
            use_rerank=args.rerank,
        )
        grid_results.append(result)
        elapsed = time.time() - t0
        avg_per = elapsed / idx
        remaining = avg_per * (n_combos - idx)
        print(
            f"  [{idx:>2}/{n_combos}] ratio={ratio:.1f} "
            f"(dense={result['dense_weight']:.2f}, sparse={result['sparse_weight']:.2f}) → "
            f"avg_coverage={result['avg_coverage']:.3f}  "
            f"full%={result['full_coverage_rate']*100:.1f}%  "
            f"(ETA {remaining:.0f}s)"
        )

    print("\n" + "=" * 60)
    print("实验结果")
    print("=" * 60)

    print_results_table(grid_results)
    print_top_combos(grid_results, args.top)

    # Compare with production default
    prod_ratio = 1.0 / 1.5  # dense=1.0, sparse=0.5
    closest = min(grid_results, key=lambda r: abs(r["dense_ratio"] - prod_ratio))
    print(f"\n最接近当前生产默认 (dense_ratio≈{prod_ratio:.3f}) 的结果:")
    print(f"  dense_ratio={closest['dense_ratio']:.2f}  avg_coverage={closest['avg_coverage']:.3f}  "
          f"full%={closest['full_coverage_rate']*100:.1f}%")

    # Save results
    out_path = args.out or f"benchmark/results/weight_search_{int(time.time())}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "n_cases": len(cases_emb),
            "grid": grid_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved → {out_path}")


if __name__ == "__main__":
    main()
