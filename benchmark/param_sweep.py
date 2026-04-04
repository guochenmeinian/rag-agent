"""Parameter sensitivity sweep — test different retrieval configurations.

Runs retrieval-only evaluation with varying parameters to find optimal settings.
Does NOT call LLMs (uses run_benchmark_isolated's retrieval path).

Usage
─────
# Sweep chunk retrieval top-k
python benchmark/param_sweep.py --sweep top_k --dataset benchmark/data/cases_rag_critical.json

# Sweep score threshold
python benchmark/param_sweep.py --sweep threshold

# Sweep hybrid search weights
python benchmark/param_sweep.py --sweep weights

# All sweeps
python benchmark/param_sweep.py --sweep all

# Custom output
python benchmark/param_sweep.py --sweep top_k --out benchmark/results/sweep_topk.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

import config as src_config
import benchmark.config as bm_config
from rag.pipeline import retrieve, RAGContext, ingest
from rag.retriever import hybrid_search, rerank_candidates
from rag.embedder import embed_query, load_bge_m3_embedder, load_bge_reranker
from benchmark.eval.retrieval_eval import eval_retrieval_case


# ─────────────────────────────────────────────────────────────
# Sweep definitions
# ─────────────────────────────────────────────────────────────

SWEEPS = {
    "top_k": {
        "description": "Vary top-k retrieval limit (how many chunks returned to LLM)",
        "param": "limit",
        "values": [3, 5, 10, 15, 20],
    },
    "threshold": {
        "description": "Vary score threshold (minimum hybrid score to keep a chunk)",
        "param": "score_threshold",
        "values": [0.0, 0.2, 0.35, 0.5, 0.6, 0.7],
    },
    "weights": {
        "description": "Vary sparse/dense weight ratio for hybrid search",
        "param": "weights",
        "values": [
            (0.0, 1.0),   # dense only
            (0.3, 1.0),   # slight sparse
            (0.5, 1.0),   # balanced
            (0.7, 1.0),   # moderate sparse
            (1.0, 1.0),   # equal
            (1.0, 0.7),   # sparse favored (current default)
            (1.0, 0.5),   # strong sparse
            (1.0, 0.3),   # very sparse
        ],
    },
    "search_limit": {
        "description": "Vary candidates fetched from Milvus before reranking",
        "param": "search_limit",
        "values": [5, 10, 15, 20, 30, 50],
    },
}


# ─────────────────────────────────────────────────────────────
# Context loading
# ─────────────────────────────────────────────────────────────

def load_rag_contexts() -> dict[str, RAGContext]:
    contexts = {}
    for model in src_config.NIO_CAR_MODELS:
        subdir = os.path.join(src_config.DATA_ROOT, model)
        flat_pdf = os.path.join(src_config.DATA_ROOT, f"{model}.pdf")
        col_name = f"nio_{model.lower()}"
        try:
            if os.path.isdir(subdir):
                ctx = ingest(data_dir=subdir, uri=src_config.MILVUS_URI, col_name=col_name)
            elif os.path.isfile(flat_pdf):
                ctx = ingest(data_dir=src_config.DATA_ROOT, uri=src_config.MILVUS_URI,
                             col_name=col_name, file_filter=f"{model}.pdf")
            else:
                continue
            contexts[model] = ctx
        except Exception as exc:
            print(f"[warn] {model}: {exc}")
    return contexts


# ─────────────────────────────────────────────────────────────
# Single retrieval run with custom params
# ─────────────────────────────────────────────────────────────

def retrieve_with_params(
    query: str,
    ctx: RAGContext,
    limit: int = 5,
    search_limit: int = 20,
    sparse_weight: float = 1.0,
    dense_weight: float = 0.7,
    score_threshold: float = 0.35,
) -> list[dict]:
    """Wrapper around retrieve() with explicit parameter control."""
    return retrieve(
        query, ctx,
        limit=limit,
        search_limit=search_limit,
        sparse_weight=sparse_weight,
        dense_weight=dense_weight,
        score_threshold=score_threshold,
    )


# ─────────────────────────────────────────────────────────────
# Sweep execution
# ─────────────────────────────────────────────────────────────

def run_sweep(
    sweep_name: str,
    cases: list[dict],
    rag_contexts: dict[str, RAGContext],
) -> dict:
    """Run one parameter sweep across all cases."""
    sweep_cfg = SWEEPS[sweep_name]
    param = sweep_cfg["param"]
    values = sweep_cfg["values"]

    print(f"\n  Sweep: {sweep_name} — {sweep_cfg['description']}")
    print(f"  Values: {values}\n")

    sweep_results: dict[str, dict] = {}

    for val in values:
        val_key = str(val)
        print(f"  --- {param}={val} ---")

        case_results = []
        for case in cases:
            gt = case.get("retrieval_gt", {})
            # Determine car model from case
            car_model = None
            for tool_gt in case.get("router_gt", {}).get("expected_tools", []):
                if tool_gt.get("car_model"):
                    car_model = tool_gt["car_model"]
                    break
            if not car_model:
                # Try to infer from case input
                for m in src_config.NIO_CAR_MODELS:
                    if m in case["input"]:
                        car_model = m
                        break

            if not car_model or car_model not in rag_contexts:
                continue

            ctx = rag_contexts[car_model]
            query = case["input"]

            # Build kwargs based on sweep type
            kwargs: dict = {"limit": 5, "search_limit": 20, "sparse_weight": 1.0,
                           "dense_weight": 0.7, "score_threshold": 0.35}

            if param == "weights":
                kwargs["sparse_weight"] = val[0]
                kwargs["dense_weight"] = val[1]
            elif param in kwargs:
                kwargs[param] = val

            t0 = time.monotonic()
            try:
                results = retrieve_with_params(query, ctx, **kwargs)
            except Exception as e:
                print(f"    ERR {case['id']}: {e}")
                continue
            latency = round(time.monotonic() - t0, 3)

            # Build ranked_chunks for evaluation
            ranked_chunks = [
                {"id": f"rag_search:{car_model}", "content": r.get("text", r.get("chunk", ""))}
                for r in results
            ]

            eval_result = eval_retrieval_case(case, ranked_chunks)
            eval_result["latency_s"] = latency
            eval_result["n_results"] = len(results)
            if results:
                eval_result["scores"] = [r.get("score", 0) for r in results]
            case_results.append(eval_result)

        # Aggregate
        valid = [r for r in case_results if not r.get("skip")]
        if valid:
            metrics: dict = {"n": len(valid)}
            # Collect all numeric metrics
            for r in valid:
                for k, v in r.get("metrics", {}).items():
                    if isinstance(v, (int, float)):
                        metrics.setdefault(k, []).append(v)

            avg_metrics = {}
            for k, vals in metrics.items():
                if isinstance(vals, list):
                    avg_metrics[k] = round(sum(vals) / len(vals), 4)
                else:
                    avg_metrics[k] = vals

            latencies = [r["latency_s"] for r in valid if "latency_s" in r]
            if latencies:
                avg_metrics["latency_avg"] = round(sum(latencies) / len(latencies), 3)

            print(f"    n={len(valid)}  " + "  ".join(f"{k}={v}" for k, v in avg_metrics.items() if k != "n"))
        else:
            avg_metrics = {"n": 0}
            print(f"    (no valid results)")

        sweep_results[val_key] = {
            "param_value": val,
            "metrics": avg_metrics,
            "case_results": case_results,
        }

    return {
        "sweep": sweep_name,
        "param": param,
        "description": sweep_cfg["description"],
        "results": sweep_results,
    }


def print_sweep_table(sweep_data: dict):
    """Print a comparison table for one sweep."""
    print(f"\n{'='*70}")
    print(f"  SWEEP: {sweep_data['sweep']} ({sweep_data['description']})")
    print(f"{'='*70}")

    results = sweep_data["results"]
    if not results:
        print("  No results.")
        return

    # Collect all metric keys
    all_keys: set[str] = set()
    for v in results.values():
        all_keys.update(k for k, val in v["metrics"].items() if isinstance(val, (int, float)))

    metric_keys = sorted(all_keys - {"n"})
    param_vals = list(results.keys())

    # Header
    col_w = 12
    header = f"  {sweep_data['param']:<15}" + "".join(f"{mk:>{col_w}}" for mk in metric_keys)
    print(header)
    print("  " + "-" * (15 + col_w * len(metric_keys)))

    for pv in param_vals:
        m = results[pv]["metrics"]
        row = f"  {pv:<15}"
        for mk in metric_keys:
            val = m.get(mk)
            if val is None:
                row += f"{'—':>{col_w}}"
            elif isinstance(val, float):
                row += f"{val:>{col_w}.4f}"
            else:
                row += f"{val:>{col_w}}"
        print(row)

    print()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parameter sensitivity sweep")
    p.add_argument("--sweep", nargs="+", default=["top_k"],
                   choices=list(SWEEPS.keys()) + ["all"],
                   help="Which sweep(s) to run")
    p.add_argument("--dataset", default=str(bm_config.DATA_DIR / "cases_rag_critical.json"))
    p.add_argument("--out", default="", help="Output JSON path")
    p.add_argument("--ids", default="", help="Filter by case IDs")
    return p.parse_args()


def main():
    args = parse_args()

    sweeps = list(SWEEPS.keys()) if "all" in args.sweep else args.sweep

    with open(args.dataset, encoding="utf-8") as f:
        all_cases = json.load(f)["cases"]

    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",")}
        all_cases = [c for c in all_cases if c["id"] in wanted]

    # Only keep cases with retrieval ground truth
    cases = [c for c in all_cases if c.get("retrieval_gt")]
    if not cases:
        print("No cases with retrieval_gt found.")
        return

    print(f"Parameter sweep: {len(cases)} retrieval cases × {len(sweeps)} sweeps")

    rag_contexts = load_rag_contexts()
    if not rag_contexts:
        print("No RAG contexts loaded. Ensure Milvus is running and data is ingested.")
        return

    all_sweeps: dict[str, dict] = {}
    for sweep_name in sweeps:
        result = run_sweep(sweep_name, cases, rag_contexts)
        all_sweeps[sweep_name] = result
        print_sweep_table(result)

    out_path = args.out or str(bm_config.RESULTS_DIR / f"sweep_{int(time.time())}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_sweeps, f, ensure_ascii=False, indent=2, default=str)
    print(f"Results saved -> {out_path}")


if __name__ == "__main__":
    main()
