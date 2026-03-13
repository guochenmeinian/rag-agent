import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmark.synthesis.models import ConversationDraft
from benchmark.synthesis.quality import check_case_quality


def test_conversation_draft_requires_final_test_turn():
    draft = ConversationDraft.model_validate(
        {
            "conversation": [
                {"role": "user", "content": "先聊 ET5"},
                {"role": "assistant", "content": "好的"},
                {"role": "user", "content": "那它快充呢", "is_test_turn": True},
            ]
        }
    )
    assert draft.test_input == "那它快充呢"
    assert len(draft.history) == 2


def test_quality_flags_short_single_case():
    case = {
        "input": "ET5?",
        "dimensions": {
            "has_coref": False,
            "depends_on_memory": False,
            "should_clarify": False,
        },
        "context": {"conversation_history": []},
    }
    target = {"id": "rt_parameter", "mode": "single"}
    errors = check_case_quality(case, target)
    assert errors


def test_quality_flags_missing_coref_surface():
    case = {
        "input": "ET7 快充怎么样",
        "dimensions": {
            "has_coref": True,
            "depends_on_memory": False,
            "should_clarify": False,
        },
        "context": {"conversation_history": [{"role": "user", "content": "先看 ET7"}]},
    }
    target = {"id": "rw_standalone_coref", "mode": "short"}
    errors = check_case_quality(case, target)
    assert any("coreference" in err for err in errors)
