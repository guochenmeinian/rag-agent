"""蔚来AI助手 - Streamlit frontend.

Run:
    cd src && streamlit run frontend.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: F401 — loads .env before anything else

import shutil
import tempfile
import streamlit as st
from agent.workflow import AgentWorkflow
from agent.memory import ConversationMemory
from tools.registry import ToolRegistry
from tools.web_search import WebSearchTool
from tools.rag_search import RagSearchTool
from rag.pipeline import ingest, RAGContext

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="蔚来AI助手",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Tool call badge */
.tool-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    margin: 2px 3px;
}
.badge-rag  { background: #dbeafe; color: #1d4ed8; }
.badge-web  { background: #dcfce7; color: #15803d; }

/* Source footer */
.source-line {
    font-size: 12px;
    color: #6b7280;
    margin-top: 8px;
    border-top: 1px solid #e5e7eb;
    padding-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Initialization helpers (cached across re-runs)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="加载蔚来知识库...")
def load_rag_contexts() -> dict[str, RAGContext]:
    """Supports two data layouts:
      1. data/<model>/<files>  (subdirectory per model)
      2. data/<model>.pdf      (flat PDFs, auto-wrapped in temp dir)
    """
    contexts = {}
    data_root = config.DATA_ROOT
    db_uri = config.MILVUS_URI

    for model in config.NIO_CAR_MODELS:
        subdir = os.path.join(data_root, model)
        flat_pdf = os.path.join(data_root, f"{model}.pdf")
        col_name = f"nio_{model.lower()}"

        try:
            if os.path.isdir(subdir):
                # Layout 1: data/EC6/
                ctx = ingest(data_dir=subdir, uri=db_uri, col_name=col_name)
            elif os.path.isfile(flat_pdf):
                # Layout 2: data/EC6.pdf — copy into a temp dir for ingest
                with tempfile.TemporaryDirectory() as tmpdir:
                    shutil.copy(flat_pdf, os.path.join(tmpdir, f"{model}.pdf"))
                    ctx = ingest(data_dir=tmpdir, uri=db_uri, col_name=col_name)
            else:
                continue
            contexts[model] = ctx
        except Exception as e:
            st.warning(f"车型 {model} 知识库加载失败: {e}")

    return contexts


@st.cache_resource(show_spinner=False)
def build_registry() -> ToolRegistry:
    rag_contexts = load_rag_contexts()
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    if rag_contexts:
        registry.register(RagSearchTool(contexts=rag_contexts))
    return registry


def get_workflow(user_profile: str) -> AgentWorkflow:
    """Per-session workflow stored in session_state."""
    if "workflow" not in st.session_state or st.session_state.get("profile") != user_profile:
        registry = build_registry()
        st.session_state.workflow = AgentWorkflow(registry=registry, user_profile=user_profile)
        st.session_state.profile = user_profile
    return st.session_state.workflow


# ---------------------------------------------------------------------------
# Helper: render a single assistant turn
# ---------------------------------------------------------------------------

def render_tool_calls(calls: list[dict]):
    badges = []
    for c in calls:
        name = c.get("name", "")
        query = c.get("query", "")
        car = c.get("car_model", "")
        if name == "rag_search":
            label = f"📚 知识库 · {car} · {query}"
            badges.append(f'<span class="tool-badge badge-rag">{label}</span>')
        else:
            label = f"🌐 网络搜索 · {query}"
            badges.append(f'<span class="tool-badge badge-web">{label}</span>')
    st.markdown(" ".join(badges), unsafe_allow_html=True)


def render_message(msg: dict, dev_mode: bool = False):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                with st.expander("🔍 工具调用", expanded=False):
                    render_tool_calls(tool_calls)
            st.markdown(msg["content"])
            if dev_mode and msg.get("dev_logs"):
                with st.expander("🔧 开发者日志", expanded=False):
                    for log in msg["dev_logs"]:
                        st.write(log)
        else:
            st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🚗 蔚来AI助手")
    st.caption("基于RAG + Claude的智能问答系统")
    st.divider()

    user_profile = st.text_area(
        "用户偏好 (可选)",
        placeholder="例如：我在意续航和充电速度，预算50万以内，家用为主",
        height=100,
        key="user_profile_input",
    )

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        # Reset workflow memory
        if "workflow" in st.session_state:
            del st.session_state["workflow"]
        st.rerun()

    st.divider()

    rag_contexts = load_rag_contexts()
    if rag_contexts:
        st.markdown("**📖 已加载知识库**")
        for model in rag_contexts:
            st.markdown(f"  - {model}")
    else:
        st.info("未找到车型数据目录，仅使用网络搜索。\n\n`data/<车型>/` 放入文档后重启。")

    st.divider()
    dev_mode = st.toggle("🔧 开发者模式", value=False)

    st.divider()
    st.caption(f"主模型: `{config.OPENAI_MODEL}`")
    st.caption(f"改写/反思: `{config.QWEN_MODEL}`")

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.header("蔚来汽车智能助手", divider="blue")

# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    render_message(msg, dev_mode=dev_mode)

# Chat input
if prompt := st.chat_input("问我关于蔚来汽车的任何问题..."):
    # Show user message immediately
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get or create workflow
    workflow = get_workflow(user_profile)

    # Stream response
    with st.chat_message("assistant"):
        tool_calls_this_turn: list[dict] = []
        dev_logs: list = []
        answer = ""

        with st.status("思考中...", expanded=True) as status_box:
            for event in workflow.run_stream(prompt):
                etype = event["type"]

                if etype == "rewriting":
                    st.write("✏️ 理解问题...")
                    dev_logs.append("✏️ 理解问题...")

                elif etype == "refined":
                    refined = event["query"]
                    if dev_mode:
                        st.write(f"💡 改写后问题: `{refined}`")
                    elif refined != prompt:
                        st.write(f"💡 精准问题: *{refined}*")
                    dev_logs.append(f"💡 改写后问题: {refined}")

                elif etype == "tool_calling":
                    calls = event["calls"]
                    tool_calls_this_turn.extend(calls)
                    for c in calls:
                        name = c.get("name", "")
                        query = c.get("query", "")
                        car = c.get("car_model", "")
                        if name == "rag_search":
                            st.write(f"📚 查询知识库: **{car}** — {query}")
                            dev_logs.append(f"📚 rag_search | car={car} | query={query}")
                        else:
                            st.write(f"🌐 网络搜索: {query}")
                            dev_logs.append(f"🌐 web_search | query={query}")

                elif etype == "tool_done":
                    results = event["results"]
                    if dev_mode:
                        for r in results:
                            st.write(f"✅ `{r['name']}` 返回:")
                            st.code(r["result"][:1000], language="text")
                    else:
                        st.write(f"✅ 获取到 {len(results)} 条结果")
                    for r in results:
                        dev_logs.append(f"✅ {r['name']} 结果:\n{r['result'][:500]}")

                elif etype == "reflecting":
                    st.write("🔎 验证答案质量...")
                    dev_logs.append("🔎 反思中...")

                elif etype == "retry":
                    feedback = event["feedback"]
                    if dev_mode:
                        st.write(f"⚠️ 反思未通过: {feedback}")
                    else:
                        st.write(f"⚠️ 重新生成 (反思反馈: {feedback[:60]}...)")
                    dev_logs.append(f"⚠️ 反思未通过: {feedback}")

                elif etype == "done":
                    answer = event["answer"]
                    status_box.update(label="完成", state="complete", expanded=dev_mode)

        # Render tool call summary if any
        if tool_calls_this_turn:
            with st.expander("🔍 工具调用详情", expanded=False):
                render_tool_calls(tool_calls_this_turn)

        # Render final answer
        st.markdown(answer)

    # Persist to session history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "tool_calls": tool_calls_this_turn,
        "dev_logs": dev_logs,
    })
