"""Third-party RAG comparison benchmark.

Runs the same answer-layer test cases (cases_smoke.json) against RAGFlow and/or
QAnything, then scores each answer with the same LLM judge used for our own pipeline.

Usage
─────
# Test both systems (both services must be running):
  python benchmark/third_party/run_comparison.py

# Test only RAGFlow:
  python benchmark/third_party/run_comparison.py --systems ragflow

# Test only QAnything:
  python benchmark/third_party/run_comparison.py --systems qanything

# Custom dataset file:
  python benchmark/third_party/run_comparison.py --dataset benchmark/data/cases_smoke.json

# Custom service URLs:
  python benchmark/third_party/run_comparison.py \\
      --ragflow-url http://localhost:80 \\
      --qanything-url http://localhost:8777

# Skip setup (KB already indexed, jump straight to Q&A):
  python benchmark/third_party/run_comparison.py --skip-setup

# Save output to specific file:
  python benchmark/third_party/run_comparison.py --out benchmark/results/comparison_ragflow.json

Environment variables
─────────────────────
  RAGFLOW_API_KEY      — API key from RAGFlow web UI (Settings → API Key)
  OPENAI_API_BASE      — OpenAI-compatible endpoint for QAnything LLM calls
  OPENAI_API_KEY       — API key for the above endpoint
  OPENAI_MODEL         — Model name (default: gpt-4o-mini)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

import config as src_config
from benchmark.eval.answer_eval import eval_answer_case, aggregate_answer

# ─────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────

DEFAULT_DATASET = _REPO / "benchmark" / "data" / "cases_smoke.json"


def load_answer_cases(path: Path) -> list[dict]:
    """Load only answer-layer cases (layer_targets contains 'answer')."""
    with open(path, encoding="utf-8") as f:
        all_cases = json.load(f)["cases"]
    cases = [c for c in all_cases if "answer" in c.get("layer_targets", [])]
    print(f"Loaded {len(cases)} answer cases from {path.name}")
    return cases


def collect_pdfs(data_dir: str) -> list[str]:
    """Return all PDF files from data_dir (flat) or from model subdirs."""
    root = Path(data_dir)
    pdfs = list(root.glob("*.pdf"))
    if not pdfs:
        pdfs = list(root.glob("*/*.pdf"))
    paths = sorted(str(p) for p in pdfs)
    print(f"Found {len(paths)} PDFs in {data_dir}")
    return paths


# ─────────────────────────────────────────────────────────────
# Per-system runner
# ─────────────────────────────────────────────────────────────

def run_system(
    adapter,
    cases: list[dict],
    pdf_files: list[str],
    skip_setup: bool = False,
    delay: float = 2.0,
) -> list[dict]:
    """Run all cases against one adapter; return list of eval_answer_case results."""
    print(f"\n{'='*60}")
    print(f"  System: {adapter.name.upper()}")
    print(f"{'='*60}")

    if not skip_setup:
        print(f"\n[setup] Ingesting {len(pdf_files)} PDFs…")
        adapter.setup(pdf_files)
    else:
        print(f"\n[setup] Skipped (--skip-setup)")
        # Still need kb_id for QAnythingAdapter even when skipping
        if hasattr(adapter, "_find_existing_kb") and adapter._kb_id is None:
            adapter._kb_id = adapter._find_existing_kb()
            if adapter._kb_id:
                print(f"  [qanything] found existing KB → kb_id={adapter._kb_id}")

    results: list[dict] = []
    print(f"\n[eval] Running {len(cases)} cases…\n")

    for i, case in enumerate(cases):
        if i > 0 and delay > 0:
            time.sleep(delay)

        cid   = case["id"]
        query = case["input"]
        print(f"  → {cid}: {query[:55]}")

        t0 = time.monotonic()
        try:
            answer = adapter.ask(query)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            results.append({
                "id":      cid,
                "input":   query,
                "answer":  "",
                "error":   str(exc),
                "metrics": {"match": 0, "hallucination": 0, "clarification": 0},
                "detail":  {},
            })
            continue
        latency = round(time.monotonic() - t0, 2)

        r = eval_answer_case(case, answer)
        r["latency_s"] = latency

        m = r["metrics"]
        print(
            f"     match={m['match']}  hall={m['hallucination']}  "
            f"clar={m['clarification']}  ({latency}s)"
        )
        results.append(r)

    adapter.teardown()
    return results


# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────

def print_comparison(all_results: dict[str, list[dict]], cases: list[dict]):
    """Print a side-by-side comparison table."""
    systems = list(all_results.keys())

    print("\n" + "=" * 75)
    print("  COMPARISON SUMMARY")
    print("=" * 75)

    # Aggregate per system
    for sys_name, results in all_results.items():
        valid = [r for r in results if not r.get("error")]
        agg   = aggregate_answer(valid)
        print(f"\n  {sys_name.upper()}  (n={agg.get('n', 0)})")
        print(f"    match_avg             {agg.get('match_avg', 'N/A'):.3f}")
        print(f"    match_full            {agg.get('match_full', 'N/A'):.3f}")
        print(f"    hallucination_clean   {agg.get('hallucination_clean', 'N/A'):.3f}")
        print(f"    clarification_acc     {agg.get('clarification_acc', 'N/A'):.3f}")
        print(f"    key_facts_coverage    {agg.get('key_facts_coverage_avg', 'N/A')}")

    # Per-case breakdown
    print("\n" + "-" * 75)
    print(f"  {'CASE':<32} " + "  ".join(f"{s.upper()[:10]:<10}" for s in systems))
    print("-" * 75)

    # Build lookup: system → case_id → metrics string
    lookup: dict[str, dict[str, str]] = {}
    for sys_name, results in all_results.items():
        lookup[sys_name] = {}
        for r in results:
            if r.get("error"):
                cell = "ERROR"
            else:
                m = r["metrics"]
                cell = f"m={m['match']} h={m['hallucination']}"
            lookup[sys_name][r["id"]] = cell

    for case in cases:
        cid = case["id"]
        row = f"  {cid:<32} "
        for sys_name in systems:
            cell = lookup.get(sys_name, {}).get(cid, "—")
            row += f"{cell:<12}"
        print(row)

    print("=" * 75 + "\n")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare third-party RAG systems against our benchmark")
    p.add_argument(
        "--systems", nargs="+", default=["ragflow", "qanything"],
        choices=["ragflow", "qanything"],
        help="Which systems to test (default: both)",
    )
    p.add_argument("--dataset",       default=str(DEFAULT_DATASET), help="Path to cases JSON")
    p.add_argument("--data-dir",      default=src_config.DATA_ROOT,  help="Directory with PDFs")
    p.add_argument("--ragflow-url",   default="http://localhost:80")
    p.add_argument("--qanything-url", default="http://localhost:8777")
    p.add_argument("--ragflow-key",   default="", help="RAGFlow API key (or set RAGFLOW_API_KEY)")
    p.add_argument("--skip-setup",    action="store_true",
                   help="Skip PDF upload/indexing (assumes KB already indexed)")
    p.add_argument("--chunk-method",  default="naive",
                   choices=["naive", "deepdoc"],
                   help="RAGFlow chunk method (naive=CPU-only, deepdoc=requires GPU models)")
    p.add_argument("--out",           default="", help="Save results JSON to this path")
    p.add_argument("--delay",         type=float, default=2.0,
                   help="Seconds between cases (avoids LLM rate limits)")
    return p.parse_args()


def main():
    args = parse_args()

    cases    = load_answer_cases(Path(args.dataset))
    pdf_files = collect_pdfs(args.data_dir)

    if not pdf_files:
        print(f"ERROR: No PDFs found in {args.data_dir}")
        sys.exit(1)

    # Build adapters
    adapters = []
    if "ragflow" in args.systems:
        from benchmark.third_party.ragflow_adapter import RAGFlowAdapter
        adapters.append(RAGFlowAdapter(
            base_url=args.ragflow_url,
            api_key=args.ragflow_key or None,
            chunk_method=args.chunk_method,
        ))
    if "qanything" in args.systems:
        from benchmark.third_party.qanything_adapter import QAnythingAdapter
        adapters.append(QAnythingAdapter(base_url=args.qanything_url))

    all_results: dict[str, list[dict]] = {}
    aggregates:  dict[str, dict]       = {}

    for adapter in adapters:
        results = run_system(
            adapter, cases, pdf_files,
            skip_setup=args.skip_setup,
            delay=args.delay,
        )
        all_results[adapter.name] = results
        valid = [r for r in results if not r.get("error")]
        aggregates[adapter.name] = aggregate_answer(valid)

    print_comparison(all_results, cases)

    # Save JSON
    out_path = args.out or str(
        _REPO / "benchmark" / "results" / f"comparison_{int(time.time())}.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "systems":    list(all_results.keys()),
                "aggregates": aggregates,
                "results":    all_results,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"Results → {out_path}")


if __name__ == "__main__":
    main()
