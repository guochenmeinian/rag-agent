"""RAGAS integration — wraps the official ragas library for faithfulness and context_recall.

Metrics used
────────────
faithfulness    (0–1): Are answer claims supported by retrieved chunks?
                       1.0 = fully grounded, 0.0 = completely ungrounded
                       Note: different from hallucination — this measures grounding
                       in *actually retrieved* chunks, not ground truth.

context_recall  (0–1): Do retrieved chunks cover all ground truth sentences?
                       1.0 = retrieval found everything needed, 0.0 = missed it all
                       Direct signal for retrieval gaps.

Usage
─────
    from benchmark.eval.ragas_eval import evaluate_ragas

    result = evaluate_ragas(
        question="ET9有几个座位？",
        answer="ET9是4座行政轿车。",
        contexts=["ET9采用4座设定...", "轴距3250mm..."],
        ground_truth="ET9为4座设定的行政旗舰轿车。",
    )
    # result = {"faithfulness": 1.0, "context_recall": 1.0, "skipped": False}
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

import config as _cfg

# ─────────────────────────────────────────────────────────────
# Lazy-initialised RAGAS objects (avoid slow import at module load)
# ─────────────────────────────────────────────────────────────

_ragas_ready: bool = False
_faithfulness = None
_context_recall = None


def _init_ragas() -> bool:
    """Initialise RAGAS metrics with the project's OpenAI client. Idempotent."""
    global _ragas_ready, _faithfulness, _context_recall

    if _ragas_ready:
        return True

    try:
        from openai import OpenAI
        from ragas.llms import llm_factory
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.metrics import faithfulness as _f, context_recall as _cr

        client = OpenAI(
            api_key=_cfg.EXECUTOR_API_KEY,
            **({"base_url": _cfg.EXECUTOR_BASE_URL} if _cfg.EXECUTOR_BASE_URL else {}),
        )
        llm = llm_factory(_cfg.EXECUTOR_MODEL, client=client)
        _f.llm = llm
        _cr.llm = llm

        _faithfulness = _f
        _context_recall = _cr
        _ragas_ready = True
        return True

    except Exception as exc:
        print(f"[ragas_eval] RAGAS init failed: {exc}", file=sys.stderr)
        return False


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def evaluate_ragas(
    question: str,
    answer: str,
    contexts: list[str] | str,
    ground_truth: str,
) -> dict:
    """Run RAGAS faithfulness + context_recall on a single sample.

    Args:
        question:     user question string
        answer:       system-generated answer
        contexts:     list of retrieved chunk strings (plain text, not dicts)
        ground_truth: expected correct answer

    Returns:
        {
          "faithfulness":    float 0–1  (or None if skipped),
          "context_recall":  float 0–1  (or None if skipped),
          "skipped":         bool,
          "skip_reason":     str | None,
        }
    """
    # Normalise: contexts must be a list of strings
    if isinstance(contexts, str):
        contexts = [contexts] if contexts.strip() else []

    if not contexts or not answer.strip():
        return {
            "faithfulness":   None,
            "context_recall": None,
            "skipped":        True,
            "skip_reason":    "no retrieved contexts" if not contexts else "empty answer",
        }

    if not _init_ragas():
        return {
            "faithfulness":   None,
            "context_recall": None,
            "skipped":        True,
            "skip_reason":    "ragas init failed",
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas import EvaluationDataset, SingleTurnSample, evaluate

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth or "",
        )
        dataset = EvaluationDataset(samples=[sample])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = evaluate(dataset, metrics=[_faithfulness, _context_recall])

        import math
        df = result.to_pandas()

        def _safe(col: str):
            if col not in df.columns:
                return None
            v = float(df[col].iloc[0])
            return None if math.isnan(v) else round(v, 3)

        faith  = _safe("faithfulness")
        recall = _safe("context_recall")

        return {
            "faithfulness":   faith,
            "context_recall": recall,
            "skipped":        False,
            # NaN means RAGAS couldn't decompose claims (empty/vague answer)
            "skip_reason":    "faithfulness=NaN (no extractable claims)" if faith is None else None,
        }

    except Exception as exc:
        return {
            "faithfulness":   None,
            "context_recall": None,
            "skipped":        True,
            "skip_reason":    str(exc)[:200],
        }
