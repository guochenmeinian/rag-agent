"""Centralized path and config for the benchmark module."""
from pathlib import Path

# ── Roots ─────────────────────────────────────────────────────
BENCHMARK_ROOT = Path(__file__).resolve().parent
REPO_ROOT      = BENCHMARK_ROOT.parent

# ── Data ──────────────────────────────────────────────────────
DATA_DIR     = BENCHMARK_ROOT / "data"
CASES_FILE   = DATA_DIR / "cases4.json"
RESULTS_DIR  = BENCHMARK_ROOT / "results"

# ── Synthesis ─────────────────────────────────────────────────
SYNTHESIS_DIR = BENCHMARK_ROOT / "synthesis"

# ── Ensure dirs exist at import time ──────────────────────────
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
