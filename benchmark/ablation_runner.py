"""Systematic ablation study runner.

Runs the same test cases under multiple pipeline configurations to measure
the contribution of each component (rewriter, reranker, hybrid search, etc.).

Usage
─────
# Full ablation matrix on smoke tests
python benchmark/ablation_runner.py --dataset benchmark/data/cases_smoke.json

# Quick: only test rewriter on/off
python benchmark/ablation_runner.py --dataset benchmark/data/cases_smoke.json --configs baseline full +rewriter

# List available configs
python benchmark/ablation_runner.py --list-configs

# Custom delay between cases
python benchmark/ablation_runner.py --dataset benchmark/data/cases_smoke.json --delay 1.0
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

import benchmark.config as bm_config
from benchmark.run_benchmark import (
    ALL_LAYERS,
    load_dataset,
    load_rag_contexts,
    run_case,
    aggregate_all,
)

# ─────────────────────────────────────────────────────────────
# Ablation configurations
# ─────────────────────────────────────────────────────────────

ABLATION_CONFIGS: dict[str, dict] = {
    # Progressive build-up (measures each component's marginal contribution)
    "baseline": {
        "description": "Dense search only, no rewriter/memory",
        "disabled": {"rewriter", "web", "memory"},
        "env_overrides": {},
    },
    "+rewriter": {
        "description": "Baseline + query rewriter",
        "disabled": {"web", "memory"},
        "env_overrides": {},
    },
    "+memory": {
        "description": "Baseline + rewriter + conversation memory",
        "disabled": {"web"},
        "env_overrides": {},
    },
    "full": {
        "description": "Full pipeline (all components enabled)",
        "disabled": set(),
        "env_overrides": {},
    },
    "no_rewriter": {
        "description": "Full pipeline minus rewriter",
        "disabled": {"rewriter"},
        "env_overrides": {},
    },
    "no_web": {
        "description": "Full pipeline minus web search",
        "disabled": {"web"},
        "env_overrides": {},
    },
    "no_memory": {
        "description": "Full pipeline minus memory",
        "disabled": {"memory"},
        "env_overrides": {},
    },
}

DEFAULT_CONFIGS = ["baseline", "+rewriter", "+memory", "full"]


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────

def run_ablation(
    cases: list[dict],
    configs: list[str],
    layers: list[str],
    delay: float = 3.0,
) -> dict:
    """Run cases under each config and return structured results."""
    rag_contexts = load_rag_contexts()

    all_results: dict[str, dict] = {}

    for config_name in configs:
        cfg = ABLATION_CONFIGS[config_name]
        disabled = cfg["disabled"]

        # Apply env overrides
        original_env = {}
        for k, v in cfg.get("env_overrides", {}).items():
            original_env[k] = os.environ.get(k)
            os.environ[k] = v

        print(f"\n{'='*60}")
        print(f"  CONFIG: {config_name}")
        print(f"  {cfg['description']}")
        print(f"  Disabled: {sorted(disabled) if disabled else 'none'}")
        print(f"{'='*60}\n")

        results = []
        for i, case in enumerate(cases):
            if i > 0 and delay > 0:
                time.sleep(delay)
            print(f"  [{config_name}] {case['id']}: {case['input'][:50]}")
            r = run_case(
                case,
                layers=layers,
                rag_contexts=rag_contexts,
                disabled=disabled,
            )
            results.append(r)
            # Print quick status
            if r.get("error"):
                print(f"    ERR: {r['error'][:60]}")
            else:
                metrics_str = "  ".join(
                    f"{k.split('/')[-1][:6]}={v}"
                    for k, v in list(r.get("metrics", {}).items())[:3]
                )
                print(f"    OK  {metrics_str}  {r.get('latency_s', '?')}s")

        summary = aggregate_all(results, layers)

        # Restore env
        for k, v in original_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

        all_results[config_name] = {
            "config": {
                "name": config_name,
                "description": cfg["description"],
                "disabled": sorted(cfg["disabled"]),
            },
            "summary": summary,
            "results": results,
        }

    return all_results


# ─────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────

def print_comparison(all_results: dict):
    """Print a comparison table across all ablation configs."""
    print(f"\n{'='*80}")
    print("  ABLATION COMPARISON")
    print(f"{'='*80}\n")

    configs = list(all_results.keys())

    # Collect all metric keys
    metric_keys: list[str] = []
    for cfg_data in all_results.values():
        s = cfg_data["summary"]
        for layer in ["rewriter", "router", "retrieval", "answer"]:
            if layer in s:
                for k, v in s[layer].items():
                    full_key = f"{layer}/{k}"
                    if full_key not in metric_keys and isinstance(v, (int, float)):
                        metric_keys.append(full_key)

    # Add latency and cost
    metric_keys.extend(["latency/p50", "latency/p95", "cost/total_tokens"])

    # Header
    col_w = max(14, *(len(c) for c in configs))
    header = f"  {'Metric':<35}" + "".join(f"{c:>{col_w}}" for c in configs)
    print(header)
    print("  " + "-" * (35 + col_w * len(configs)))

    # Rows
    for mk in metric_keys:
        parts = mk.split("/")
        row = f"  {mk:<35}"
        for cfg_name in configs:
            s = all_results[cfg_name]["summary"]
            val = None
            if len(parts) == 2:
                layer, key = parts
                if layer == "latency":
                    val = s.get("latency", {}).get(key)
                elif layer == "cost":
                    val = s.get("cost", {}).get(key)
                elif layer in s:
                    val = s[layer].get(key)

            if val is None:
                row += f"{'—':>{col_w}}"
            elif isinstance(val, float):
                row += f"{val:>{col_w}.3f}"
            else:
                row += f"{val:>{col_w}}"
        print(row)

    # Summary row
    print("  " + "-" * (35 + col_w * len(configs)))
    row = f"  {'cases_run':<35}"
    for cfg_name in configs:
        n = all_results[cfg_name]["summary"].get("n_run", 0)
        row += f"{n:>{col_w}}"
    print(row)

    row = f"  {'errors':<35}"
    for cfg_name in configs:
        n = all_results[cfg_name]["summary"].get("n_error", 0)
        row += f"{n:>{col_w}}"
    print(row)

    print(f"\n{'='*80}\n")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Systematic ablation study runner")
    p.add_argument("--dataset", default=str(bm_config.DATA_DIR / "cases_smoke.json"))
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                   help=f"Configs to run. Available: {list(ABLATION_CONFIGS.keys())}")
    p.add_argument("--layers", nargs="+", default=ALL_LAYERS, choices=ALL_LAYERS)
    p.add_argument("--out", default="", help="Output JSON path")
    p.add_argument("--delay", type=float, default=3.0)
    p.add_argument("--list-configs", action="store_true", help="List available configs and exit")
    p.add_argument("--ids", default="", help="Comma-separated case IDs to filter")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_configs:
        print("\nAvailable ablation configurations:\n")
        for name, cfg in ABLATION_CONFIGS.items():
            disabled = sorted(cfg["disabled"]) if cfg["disabled"] else "none"
            marker = " *" if name in DEFAULT_CONFIGS else ""
            print(f"  {name:<20} disabled={disabled}{marker}")
        print(f"\n  * = included in default run\n")
        return

    # Validate configs
    for c in args.configs:
        if c not in ABLATION_CONFIGS:
            print(f"Unknown config: {c}. Use --list-configs to see options.")
            sys.exit(1)

    cases = load_dataset(args.dataset)
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",")}
        cases = [c for c in cases if c["id"] in wanted]

    if not cases:
        print("No cases matched the filter.")
        return

    print(f"\nAblation study: {len(cases)} cases × {len(args.configs)} configs")
    print(f"Configs: {args.configs}")
    print(f"Layers: {args.layers}")

    all_results = run_ablation(cases, args.configs, args.layers, delay=args.delay)

    print_comparison(all_results)

    out_path = args.out or str(bm_config.RESULTS_DIR / f"ablation_{int(time.time())}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
