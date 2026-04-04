"""Regression checker — compare two benchmark runs and flag degradations.

Usage
─────
# Compare latest run against a baseline
python benchmark/regression_checker.py baseline.json latest.json

# With custom threshold (flag if metric drops > 10%)
python benchmark/regression_checker.py baseline.json latest.json --threshold 0.10

# Compare ablation results
python benchmark/regression_checker.py results/ablation_old.json results/ablation_new.json --ablation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Metric extraction
# ─────────────────────────────────────────────────────────────

def extract_metrics(data: dict) -> dict[str, float]:
    """Flatten summary metrics into a flat dict for comparison."""
    flat: dict[str, float] = {}

    summary = data.get("summary", data)

    for layer in ["rewriter", "router", "retrieval", "answer"]:
        if layer in summary:
            for k, v in summary[layer].items():
                if isinstance(v, (int, float)) and v is not None:
                    flat[f"{layer}/{k}"] = float(v)

    if "latency" in summary:
        for k, v in summary["latency"].items():
            flat[f"latency/{k}"] = float(v)

    if "cost" in summary:
        for k, v in summary["cost"].items():
            flat[f"cost/{k}"] = float(v)

    flat["n_run"] = float(summary.get("n_run", 0))
    flat["n_error"] = float(summary.get("n_error", 0))

    return flat


def extract_ablation_metrics(data: dict) -> dict[str, dict[str, float]]:
    """Extract metrics per ablation config."""
    result = {}
    for config_name, config_data in data.items():
        if isinstance(config_data, dict) and "summary" in config_data:
            result[config_name] = extract_metrics(config_data)
    return result


# ─────────────────────────────────────────────────────────────
# Comparison
# ─────────────────────────────────────────────────────────────

# Metrics where higher is better (regressions = value decreased)
HIGHER_IS_BETTER = {
    "rewriter/standalone", "rewriter/entity_extraction_accuracy", "rewriter/clarify_detection",
    "rewriter/coref_resolution_rate", "rewriter/ellipsis_fill_rate",
    "router/tool_classification_accuracy", "router/parameter", "router/multi_query",
    "retrieval/hit@1", "retrieval/hit@3", "retrieval/hit@5", "retrieval/mrr",
    "retrieval/relevance@5", "retrieval/facts_coverage_avg", "retrieval/no_hit_ok",
    "retrieval/ndcg@5", "retrieval/recall@5",
    "answer/match_avg", "answer/match_full", "answer/hallucination_clean",
    "answer/clarification_acc", "answer/key_facts_coverage_avg",
}

# Metrics where lower is better (regressions = value increased)
LOWER_IS_BETTER = {
    "latency/p50", "latency/p95",
    "cost/total_tokens", "cost/total_prompt_tokens", "cost/total_completion_tokens",
    "n_error", "router/avg_duplicate_calls",
}


def compare_metrics(
    baseline: dict[str, float],
    current: dict[str, float],
    threshold: float = 0.05,
) -> list[dict]:
    """Compare two metric dicts and return list of findings.

    Returns list of:
        {"metric": str, "baseline": float, "current": float, "delta": float,
         "delta_pct": float, "severity": "regression"|"improvement"|"neutral",
         "direction": "higher_better"|"lower_better"|"unknown"}
    """
    findings = []
    all_keys = sorted(set(baseline.keys()) | set(current.keys()))

    for key in all_keys:
        b_val = baseline.get(key)
        c_val = current.get(key)

        if b_val is None or c_val is None:
            continue

        delta = c_val - b_val
        delta_pct = delta / abs(b_val) if b_val != 0 else (1.0 if delta != 0 else 0.0)

        if key in HIGHER_IS_BETTER:
            direction = "higher_better"
            is_regression = delta_pct < -threshold
            is_improvement = delta_pct > threshold
        elif key in LOWER_IS_BETTER:
            direction = "lower_better"
            is_regression = delta_pct > threshold
            is_improvement = delta_pct < -threshold
        else:
            direction = "unknown"
            is_regression = False
            is_improvement = abs(delta_pct) > threshold

        severity = "regression" if is_regression else ("improvement" if is_improvement else "neutral")

        findings.append({
            "metric": key,
            "baseline": b_val,
            "current": c_val,
            "delta": round(delta, 4),
            "delta_pct": round(delta_pct * 100, 1),
            "severity": severity,
            "direction": direction,
        })

    return findings


# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────

def print_report(findings: list[dict], baseline_path: str, current_path: str):
    """Print human-readable comparison report."""
    regressions = [f for f in findings if f["severity"] == "regression"]
    improvements = [f for f in findings if f["severity"] == "improvement"]
    neutral = [f for f in findings if f["severity"] == "neutral"]

    print(f"\n{'='*70}")
    print("  REGRESSION CHECK REPORT")
    print(f"{'='*70}")
    print(f"  Baseline: {baseline_path}")
    print(f"  Current:  {current_path}")
    print(f"  Metrics compared: {len(findings)}")
    print(f"  Regressions: {len(regressions)}  |  Improvements: {len(improvements)}  |  Neutral: {len(neutral)}")

    if regressions:
        print(f"\n  REGRESSIONS ({len(regressions)}):")
        for f in sorted(regressions, key=lambda x: abs(x["delta_pct"]), reverse=True):
            print(f"    {f['metric']:<40} {f['baseline']:.3f} -> {f['current']:.3f}  ({f['delta_pct']:+.1f}%)")

    if improvements:
        print(f"\n  IMPROVEMENTS ({len(improvements)}):")
        for f in sorted(improvements, key=lambda x: abs(x["delta_pct"]), reverse=True):
            print(f"    {f['metric']:<40} {f['baseline']:.3f} -> {f['current']:.3f}  ({f['delta_pct']:+.1f}%)")

    verdict = "PASS" if not regressions else "FAIL"
    print(f"\n  VERDICT: {verdict}")
    print(f"{'='*70}\n")

    return len(regressions) == 0


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Compare benchmark runs and detect regressions")
    p.add_argument("baseline", help="Path to baseline results JSON")
    p.add_argument("current", help="Path to current results JSON")
    p.add_argument("--threshold", type=float, default=0.05,
                   help="Regression threshold (default: 0.05 = 5%%)")
    p.add_argument("--ablation", action="store_true",
                   help="Compare ablation results (per-config comparison)")
    p.add_argument("--json", action="store_true", help="Output findings as JSON")
    args = p.parse_args()

    with open(args.baseline, encoding="utf-8") as f:
        baseline_data = json.load(f)
    with open(args.current, encoding="utf-8") as f:
        current_data = json.load(f)

    if args.ablation:
        baseline_cfgs = extract_ablation_metrics(baseline_data)
        current_cfgs = extract_ablation_metrics(current_data)
        all_pass = True
        for cfg_name in sorted(set(baseline_cfgs) & set(current_cfgs)):
            print(f"\n  --- Config: {cfg_name} ---")
            findings = compare_metrics(baseline_cfgs[cfg_name], current_cfgs[cfg_name], args.threshold)
            passed = print_report(findings, f"{args.baseline}[{cfg_name}]", f"{args.current}[{cfg_name}]")
            if not passed:
                all_pass = False
        sys.exit(0 if all_pass else 1)
    else:
        baseline_metrics = extract_metrics(baseline_data)
        current_metrics = extract_metrics(current_data)
        findings = compare_metrics(baseline_metrics, current_metrics, args.threshold)

        if args.json:
            print(json.dumps(findings, indent=2, ensure_ascii=False))
        else:
            passed = print_report(findings, args.baseline, args.current)
            sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
