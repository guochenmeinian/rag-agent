# Benchmark Refactor Notes

## What Changed

- Unified the rewriter and executor response contracts in `src/agent/contracts.py`.
- Hardened `QueryRewriter` parsing so invalid JSON now falls back through a typed contract instead of ad hoc dict checks.
- Fixed the benchmark runner to build `RagSearchTool` with real `RAGContext` objects, matching the actual tool constructor.
- Added phase-1 draft validation for synthesis output in `benchmark/synthesis/models.py`.
- Added heuristic quality gates in `benchmark/synthesis/quality.py` to reject trivially short or structurally weak synthetic cases.

## Why

The repo had three recurring problems:

1. Runtime contracts were implicit.
   The rewriter returned raw dicts and the benchmark runner assumed tool/runtime shapes without a shared typed layer.

2. The benchmark runner was not fully wired to the real retrieval stack.
   `benchmark/run_benchmark.py` instantiated `RagSearchTool()` without the required `contexts` argument.

3. Synthetic benchmark data was validated for schema only, not for difficulty.
   This made it too easy for short, template-like cases to pass into the dataset.

## Current Expectations

### Rewriter output

The rewriter is expected to emit exactly:

```json
{"type": "rewrite", "content": "..."}
```

or

```json
{"type": "clarify", "content": "..."}
```

No extra fields should be relied on downstream.

### Phase-1 synthesis output

Single-turn targets:

```json
{"input": "..."}
```

Conversation targets:

```json
{
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "...", "is_test_turn": true}
  ]
}
```

The final message must be the only `is_test_turn`.

## Remaining Gaps

- `benchmark/data/cases.json` is still only a small rewriter-only sample, not a full benchmark corpus.
- Retrieval labels still depend on stable chunk identifiers. If chunking changes, `retrieval_gt` can drift.
- The LLM judge remains same-family with the executor by default; this is fine for iteration but weaker for final evaluation.

## Recommended Next Steps

1. Split benchmark data into `synthetic_train_like` and `heldout_hard`.
2. Add explicit `difficulty` and `failure_mode` tags to benchmark cases.
3. Build a small manually reviewed hard set for router, retrieval, and answer layers.
4. Add offline quality reports for synthesis rejection reasons and target coverage.
