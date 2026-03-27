"""Benchmark case schema — organized around evaluation metrics, not field names.

Each case covers one or more layers. Ground truth fields map directly to the
metric they are used to compute.

Layer metric overview
─────────────────────
Rewriter   : standalone | entity_extraction_accuracy | clarification
Router     : tool_classification_accuracy | parameter | multi_query
Retrieval  : hit@k | MRR
Answer     : match | hallucination | clarification
"""
from __future__ import annotations
from typing import Literal, Optional, TypedDict


# ─────────────────────────────────────────────────────────────
# Shared primitives
# ─────────────────────────────────────────────────────────────

class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class Context(TypedDict):
    """Pre-loaded session state injected before the eval turn."""
    user_profile: str                     # static background (budget, family …)
    memory_facts: list[str]               # extracted persistent facts
    conversation_history: list[Message]   # recent turns to simulate mid-session


# ─────────────────────────────────────────────────────────────
# Bucketing dimensions
# ─────────────────────────────────────────────────────────────

class Dimensions(TypedDict):
    """Orthogonal slicing axes for disaggregated results."""
    is_multi_turn: bool       # requires prior-turn context to resolve
    history_length: int       # number of prior turns provided
    has_coref: bool           # contains pronoun / implicit reference
    depends_on_memory: bool   # needs user memory facts to answer correctly
    is_ambiguous: bool        # intentionally underspecified input
    should_clarify: bool      # correct system behavior is to ask the user


# ─────────────────────────────────────────────────────────────
# Layer 2 — Rewriter ground truth
# ─────────────────────────────────────────────────────────────

class RewriterGT(TypedDict):
    """Ground truth fields, grouped by the metric they compute.

    standalone (0/1)
        coref_map         — {surface_expr: resolved_entity}  (hard match)
        ellipsis_slots    — slot names that must be filled from context (hard)
        reference_rewrite — an acceptable rewrite for LLM judge comparison

    entity_extraction_accuracy (0/1)
        required_entities — strings that MUST appear in the rewritten query
        forbidden_entities — strings that must NOT appear (hallucinated entities)

    clarification (0/1)
        should_clarify      — True: correct output is to ask user, not rewrite
        is_prompt_injection — True: model should refuse / sanitize input
    """
    # standalone
    coref_map: dict[str, str]
    ellipsis_slots: list[str]
    reference_rewrite: str

    # entity_extraction_accuracy
    required_entities: list[str]
    forbidden_entities: list[str]

    # clarification
    should_clarify: bool
    is_prompt_injection: bool


# ─────────────────────────────────────────────────────────────
# Layer 3 — Router ground truth
# ─────────────────────────────────────────────────────────────

class ToolParams(TypedDict, total=False):
    """Expected parameter values for a single tool call."""
    car_model: str            # exact model name, e.g. "ET5"
    query_keywords: list[str] # at least one keyword should appear in query arg
    dense_weight: float       # expected dense coefficient (optional)


class RouterGT(TypedDict):
    """Ground truth fields grouped by the metric they compute.

    tool_classification_accuracy (0/1)
        expected_tools   — tools that SHOULD be called
        forbidden_tools  — tools that must NOT be called
        no_tool_needed   — True: direct answer expected, no tool calls

    parameter (0/1 × 3)
        tool_params      — {tool_name: ToolParams}; checks correct tool,
                           correct format, correct content

    multi_query (0/1 × 2)
        min_calls        — model must issue at least this many calls
        max_calls        — model must not exceed this many calls
        must_be_parallel — if True, expected_tools must be called simultaneously
    """
    # tool_classification_accuracy
    expected_tools: list[str]
    forbidden_tools: list[str]
    no_tool_needed: bool

    # parameter
    tool_params: dict[str, ToolParams]

    # multi_query
    min_calls: int
    max_calls: int
    must_be_parallel: bool


# ─────────────────────────────────────────────────────────────
# Layer 4 — Retrieval ground truth
# ─────────────────────────────────────────────────────────────

class RetrievalGT(TypedDict, total=False):
    """Ground truth for retrieval evaluation. Two modes:

    Mode A — chunk-ID based (offline annotation, precise):
        relevant_chunk_ids — chunk IDs that count as a "hit"
        eval_at_k          — k values to evaluate (default [1, 3, 5])

    Mode B — LLM-judge based (no chunk IDs needed, annotation-free):
        query_intent    — describes what a relevant chunk should contain
        expected_facts  — key strings that should appear somewhere in top-k chunks
        eval_at_k       — k value to judge (default [5])

    Shared:
        expect_no_hit   — True for out-of-domain queries (retriever should return nothing)
    """
    # Mode A
    relevant_chunk_ids: list[str]

    # Mode B
    query_intent: str
    expected_facts: list[str]

    # Shared
    eval_at_k: list[int]
    expect_no_hit: bool


# ─────────────────────────────────────────────────────────────
# Layer 5 — Answer ground truth
# ─────────────────────────────────────────────────────────────

class AnswerGT(TypedDict):
    """Ground truth grouped by the metric they compute.

    match (0/1/2)
        ground_truth    — reference answer for LLM judge similarity scoring
        key_facts       — SHORT atomic facts (numbers, codes, names) for hard
                          exact/normalized substring match; missing one caps
                          the score at 1.
        semantic_claims — LONGER descriptive assertions (behaviour, logic,
                          compound conditions) verified by LLM judge only;
                          never used in hard substring check.

    hallucination (0/1)
        forbidden_content — strings / facts that must NOT appear in the answer

    clarification (0/1)
        should_clarify — True: model should express uncertainty or ask user
        is_safe        — False: model should refuse (safety / prompt injection)
    """
    # match
    ground_truth: str
    key_facts: list[str]
    semantic_claims: list[str]

    # hallucination
    forbidden_content: list[str]

    # clarification
    should_clarify: bool
    is_safe: bool


# ─────────────────────────────────────────────────────────────
# Unified case
# ─────────────────────────────────────────────────────────────

LayerName = Literal["rewriter", "router", "retrieval", "answer"]


class BenchmarkCase(TypedDict, total=False):
    id: str
    layer_targets: list[LayerName]   # which layers this case exercises
    category: str                    # human-readable bucket label (e.g. "multi_car_compare")
    notes: str

    # Bucketing
    dimensions: Dimensions

    # Pre-loaded session state
    context: Context

    # Raw user utterance (the eval input)
    input: str

    # Per-layer ground truth — omit entirely if layer is not evaluated
    rewriter_gt: RewriterGT
    router_gt: RouterGT
    retrieval_gt: RetrievalGT
    answer_gt: AnswerGT


class BenchmarkDataset(TypedDict):
    version: str
    description: str
    cases: list[BenchmarkCase]
