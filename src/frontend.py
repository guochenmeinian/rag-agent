"""蔚来 AI 助手 — Notion 风格双视图前端

视图：
  💬 对话    — 用户端，简洁聊天界面
  🔧 开发者  — 执行轨迹 / 记忆状态 / 系统信息

Run:
    cd src && streamlit run frontend.py
"""
import sys, os, time, shutil, tempfile, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # loads .env first
import streamlit as st
from agent.workflow import AgentWorkflow
from tools.registry import ToolRegistry
from tools.web_search import WebSearchTool
from tools.rag_search import RagSearchTool
from rag.pipeline import ingest, RAGContext

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="蔚来 AI 助手",
    page_icon="○",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Notion-style CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
body, .stApp { background:#FFFFFF; }
.main .block-container { padding-top:0.75rem; max-width:100%; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background:#F7F6F3 !important; border-right:1px solid #E9E9E6 !important; }
section[data-testid="stSidebar"] hr { border-color:#E9E9E6 !important; margin:10px 0 !important; }
section[data-testid="stSidebar"] .stMarkdown p { font-size:13px; color:#6B7280; line-height:1.5; }

/* ── Typography ── */
h1,h2,h3,h4 { color:#191919 !important; letter-spacing:-0.01em !important; }
.notion-title { font-size:1.4rem; font-weight:700; color:#191919; letter-spacing:-0.02em; margin-bottom:0; }
.notion-sub   { font-size:13px; color:#9B9B98; margin-top:2px; }
.section-label { font-size:11px; font-weight:600; color:#9B9B98; letter-spacing:0.07em; text-transform:uppercase; margin:12px 0 4px; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] { border-bottom:1px solid #E9E9E6; gap:0; }
[data-testid="stTabs"] button[role="tab"] {
    font-size:13px; font-weight:500; color:#6B7280;
    padding:6px 14px; border-radius:0; border-bottom:2px solid transparent;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color:#191919; border-bottom:2px solid #191919; background:transparent;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] { background:transparent !important; border:none !important; padding:4px 0 !important; }
[data-testid="stChatMessageContent"] { font-size:14.5px; line-height:1.65; color:#37352F; }

/* ── Notion inline tags ── */
.ntag {
    display:inline-flex; align-items:center; gap:4px;
    padding:1px 8px; border-radius:4px;
    font-size:11.5px; font-weight:500;
    border:1px solid #E9E9E6; background:#F7F6F3; color:#6B7280;
    margin:1px 3px;
}
.ntag-rag  { border-color:#C5D7EF; background:#EEF3FA; color:#2B5C9A; }
.ntag-web  { border-color:#C4D9C4; background:#EEF5EE; color:#276327; }
.ntag-pass { border-color:#C4D9C4; background:#EEF5EE; color:#276327; }
.ntag-fail { border-color:#ECC8C8; background:#FAEEEE; color:#992D2D; }
.ntag-info { border-color:#DDDDDD; background:#F5F5F3; color:#555551; }

/* ── Dev cards ── */
.dev-card {
    border:1px solid #E9E9E6; border-radius:6px;
    padding:10px 14px; margin:3px 0;
    background:#FAFAF9; font-size:13px; line-height:1.55; color:#37352F;
}
.dev-card-title { font-weight:600; font-size:12px; color:#9B9B98; margin-bottom:6px; letter-spacing:0.04em; text-transform:uppercase; }

/* ── Timeline ── */
.tl-item { display:flex; align-items:flex-start; gap:10px; padding:4px 0; font-size:13px; color:#37352F; }
.tl-dot  { width:6px; height:6px; border-radius:50%; margin-top:5px; flex-shrink:0; background:#D1D5DB; }
.tl-dot-done { background:#6EAF6E; }
.tl-dot-tool { background:#6BA3D6; }
.tl-dot-fail { background:#E07878; }
.tl-dot-info { background:#C4B5FD; }
.tl-time { font-size:11px; color:#AEAEAA; font-family:ui-monospace,monospace; white-space:nowrap; padding-top:2px; }

/* ── Memory slots ── */
.mem-grid { display:grid; grid-template-columns:80px 1fr; row-gap:1px; }
.mem-key  { font-size:11px; font-weight:600; color:#9B9B98; letter-spacing:0.04em; text-transform:uppercase; padding:5px 0; border-bottom:1px solid #F0EFE9; }
.mem-val  { font-size:13px; color:#37352F; padding:5px 0 5px 8px; border-bottom:1px solid #F0EFE9; white-space:pre-wrap; }

/* ── Stat rows ── */
.stat-row { display:flex; justify-content:space-between; align-items:center; padding:5px 0; font-size:13px; border-bottom:1px solid #F5F5F3; }
.stat-lbl  { color:#6B7280; }
.stat-val  { color:#191919; font-weight:500; font-family:ui-monospace,monospace; font-size:12px; }
.dot-ok    { display:inline-block; width:6px; height:6px; border-radius:50%; background:#6EAF6E; margin-right:5px; }
.dot-no    { display:inline-block; width:6px; height:6px; border-radius:50%; background:#CCCCCC; margin-right:5px; }

/* ── Dialog ── */
[data-testid="stModal"] { background:#FFFFFF !important; }

/* ── Buttons — flatten Streamlit defaults ── */
.stButton button {
    border:1px solid #E9E9E6 !important; border-radius:4px !important;
    background:#FFFFFF !important; color:#37352F !important;
    font-size:12px !important; padding:3px 10px !important;
}
.stButton button:hover { background:#F7F6F3 !important; }

/* ── Expander ── */
[data-testid="stExpander"] { border:1px solid #E9E9E6 !important; border-radius:6px !important; }
summary { font-size:13px !important; color:#37352F !important; }

/* ── Scrollable dev panel ── */
.dev-scroll { max-height:72vh; overflow-y:auto; padding-right:4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Provider catalog
# ─────────────────────────────────────────────────────────────
PROVIDER_CATALOG = {
    "OpenAI": {
        "base_url":      None,
        "default_model": "gpt-4o",
        "key_hint":      "OPENAI_API_KEY",
    },
    "Moonshot (Kimi)": {
        "base_url":      "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-8k",
        "key_hint":      "MOONSHOT_API_KEY",
    },
    "DeepSeek": {
        "base_url":      "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "key_hint":      "DEEPSEEK_API_KEY",
    },
    "DashScope (Qwen)": {
        "base_url":      "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-max",
        "key_hint":      "DASHSCOPE_API_KEY",
    },
    "智谱 (GLM)": {
        "base_url":      "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4",
        "key_hint":      "ZHIPU_API_KEY",
    },
    "自定义": {
        "base_url":      "",
        "default_model": "",
        "key_hint":      "API_KEY",
    },
}

_PROVIDER_NAMES = list(PROVIDER_CATALOG)


def _provider_defaults(role: str) -> tuple[str, str]:
    """Return (provider_name, model) defaults from config for a given role."""
    if role == "executor":
        return "OpenAI", config.OPENAI_MODEL
    return "DashScope (Qwen)", config.QWEN_MODEL


def _build_model_cfg(role: str) -> dict:
    """Read session_state model config for a role and return kwargs dict."""
    provider_name = st.session_state.get(f"{role}_provider", _provider_defaults(role)[0])
    provider      = PROVIDER_CATALOG.get(provider_name, PROVIDER_CATALOG["OpenAI"])
    model         = st.session_state.get(f"{role}_model", "") or provider["default_model"]
    api_key       = st.session_state.get(f"{role}_api_key", "")
    base_url      = st.session_state.get(f"{role}_base_url", "") or provider["base_url"]

    cfg: dict = {"model": model}
    if api_key:
        cfg["api_key"] = api_key
    if base_url:
        cfg["base_url"] = base_url
    return cfg


# ─────────────────────────────────────────────────────────────
# Resource caching
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="初始化知识库…")
def load_rag_contexts() -> dict[str, RAGContext]:
    contexts: dict[str, RAGContext] = {}
    for model in config.NIO_CAR_MODELS:
        col_name = f"nio_{model.lower()}"
        subdir   = os.path.join(config.DATA_ROOT, model)
        flat_pdf = os.path.join(config.DATA_ROOT, f"{model}.pdf")
        try:
            if os.path.isdir(subdir):
                ctx = ingest(data_dir=subdir, uri=config.MILVUS_URI, col_name=col_name)
            elif os.path.isfile(flat_pdf):
                with tempfile.TemporaryDirectory() as tmp:
                    shutil.copy(flat_pdf, os.path.join(tmp, f"{model}.pdf"))
                    ctx = ingest(data_dir=tmp, uri=config.MILVUS_URI, col_name=col_name)
            else:
                continue
            contexts[model] = ctx
        except Exception as e:
            st.warning(f"{model} 知识库加载失败: {e}")
    return contexts


@st.cache_resource(show_spinner=False)
def build_registry() -> ToolRegistry:
    ctx = load_rag_contexts()
    reg = ToolRegistry()
    reg.register(WebSearchTool())
    if ctx:
        reg.register(RagSearchTool(contexts=ctx))
    return reg


def get_workflow() -> AgentWorkflow:
    profile      = st.session_state.get("user_profile", "")
    session_id   = st.session_state.get("session_id") or None
    exec_cfg     = _build_model_cfg("executor")
    qwen_cfg     = _build_model_cfg("qwen")
    key = (
        f"workflow__{session_id}__{profile}"
        f"__{exec_cfg.get('model')}__{exec_cfg.get('base_url')}"
        f"__{qwen_cfg.get('model')}__{qwen_cfg.get('base_url')}"
    )
    if st.session_state.get("_wf_key") != key:
        st.session_state.workflow = AgentWorkflow(
            registry=build_registry(),
            user_profile=profile,
            session_id=session_id,
            executor_cfg=exec_cfg,
            qwen_cfg=qwen_cfg,
        )
        st.session_state._wf_key = key
    return st.session_state.workflow

# ─────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []      # {role, content, tool_calls, trace}
if "stats" not in st.session_state:
    st.session_state.stats = {"turns": 0, "tool_calls": 0, "retries": 0}
if "_loaded_session" not in st.session_state:
    st.session_state._loaded_session = None

# Restore chat history when the user sets (or changes) their session_id
_current_sid = st.session_state.get("session_id", "").strip()
if _current_sid and st.session_state._loaded_session != _current_sid:
    # Always reset state when switching sessions
    st.session_state.messages = []
    st.session_state.stats = {"turns": 0, "tool_calls": 0, "retries": 0}
    for _k in ("workflow", "_wf_key"):
        st.session_state.pop(_k, None)
    # Load saved history if it exists
    _path = os.path.join(config.MEMORY_DIR, f"{_current_sid}.json")
    if os.path.exists(_path):
        try:
            import json as _json
            with open(_path, encoding="utf-8") as _f:
                _saved = _json.load(_f)
            st.session_state.messages = _saved.get("ui_messages", [])
        except Exception:
            pass
    st.session_state._loaded_session = _current_sid

# ─────────────────────────────────────────────────────────────
# Dialog windows
# ─────────────────────────────────────────────────────────────

@st.dialog("工具返回详情", width="large")
def dialog_tool_result():
    r = st.session_state.get("_dlg_tool")
    if not r:
        return
    c1, c2 = st.columns([1, 3])
    c1.markdown(f"**工具**")
    c2.markdown(f"`{r['name']}`")
    c1.markdown(f"**查询**")
    c2.markdown(r.get("query", "—"))
    if r.get("car_model"):
        c1.markdown(f"**车型**")
        c2.markdown(r.get("car_model"))
    st.divider()
    raw = r.get("result")
    if isinstance(raw, dict):
        content, meta = raw.get("content", ""), raw.get("metadata", {})
    else:
        content = raw.content if hasattr(raw, "content") else str(raw)
        meta    = raw.metadata if hasattr(raw, "metadata") else {}
    st.code(content, language="text")
    if meta:
        st.json(meta, expanded=False)


@st.dialog("记忆状态详情", width="large")
def dialog_memory():
    wf = st.session_state.get("workflow")
    if not wf:
        st.info("尚未创建工作流。")
        return
    mem = wf.memory
    st.markdown("#### 结构化摘要")
    st.code(mem.context_summary, language="text")
    st.markdown("#### 最近消息")
    if mem.recent_messages:
        for m in mem.recent_messages:
            with st.container():
                role_label = "用户" if m["role"] == "user" else "助手"
                st.markdown(f"**{role_label}**　{m['content'][:300]}")
    else:
        st.caption("暂无最近消息")
    if mem.user_profile:
        st.markdown("#### 用户偏好")
        st.markdown(mem.user_profile)


@st.dialog("完整执行轨迹", width="large")
def dialog_trace():
    trace = st.session_state.get("_dlg_trace")
    if not trace:
        return
    st.caption(f"原始问题：{trace.get('original_query', '')}")
    st.caption(f"改写后：{trace.get('refined_query', '')}")
    st.caption(f"耗时：{trace.get('elapsed', 0):.2f}s")
    st.divider()
    for ev in trace.get("events", []):
        _render_trace_event_full(ev)


def _render_trace_event_full(ev: dict):
    etype = ev["type"]
    if etype == "rewriting":
        st.markdown("**✏️ 查询改写**")
    elif etype == "refined":
        st.code(ev["query"], language="text")
    elif etype == "tool_calling":
        st.markdown("**🔧 工具调用**")
        for c in ev.get("calls", []):
            st.json(c, expanded=False)
    elif etype == "tool_done":
        st.markdown("**📦 工具返回**")
        for r in ev.get("results", []):
            with st.expander(f"`{r['name']}` — {r.get('query','')[:50]}"):
                raw = r.get("result")
                if isinstance(raw, dict):
                    content, meta = raw.get("content", ""), raw.get("metadata", {})
                else:
                    content = raw.content if hasattr(raw, "content") else str(raw)
                    meta    = raw.metadata if hasattr(raw, "metadata") else {}
                st.code(content, language="text")
                if meta:
                    st.json(meta, expanded=False)
    elif etype == "reflecting":
        st.markdown("**🔎 反思校验**")
    elif etype == "retry":
        st.markdown("**↩️ 重新生成**")
        st.warning(ev.get("feedback", ""))
    elif etype == "done":
        st.markdown("**✅ 完成**")
        st.markdown(ev.get("answer", "")[:500])

# ─────────────────────────────────────────────────────────────
# Helpers — render
# ─────────────────────────────────────────────────────────────

def _serialize_tool_result(raw) -> dict:
    """Convert a single ToolResult object to a JSON-serializable dict."""
    if isinstance(raw, dict):
        return raw
    return {
        "content":    getattr(raw, "content",    str(raw)),
        "success":    getattr(raw, "success",    True),
        "metadata":   getattr(raw, "metadata",   {}),
        "latency_ms": getattr(raw, "latency_ms", 0),
    }


def _serialize_trace(trace: dict) -> dict:
    """Convert ToolResult objects inside a trace to JSON-serializable dicts."""
    serialized_events = []
    for ev in trace.get("events", []):
        ev_type = ev.get("type")
        if ev_type == "tool_done":
            results_copy = [
                {**r, "result": _serialize_tool_result(r["result"])}
                if "result" in r else r
                for r in ev.get("results", [])
            ]
            serialized_events.append({**ev, "results": results_copy})
        elif ev_type == "done" and ev.get("tool_results"):
            tool_results_copy = [
                {**r, "result": _serialize_tool_result(r["result"])}
                if "result" in r else r
                for r in ev["tool_results"]
            ]
            serialized_events.append({**ev, "tool_results": tool_results_copy})
        else:
            serialized_events.append(ev)
    return {**trace, "events": serialized_events}


def _tool_tags_html(calls: list[dict]) -> str:
    parts = []
    for c in calls:
        name = c.get("name", "")
        if name == "rag_search":
            car = c.get("car_model", "")
            q   = c.get("query", "")[:20]
            parts.append(f'<span class="ntag ntag-rag">知识库 · {car} · {q}</span>')
        else:
            q = c.get("query", "")[:28]
            parts.append(f'<span class="ntag ntag-web">网络搜索 · {q}</span>')
    return " ".join(parts)


def _render_user_message(msg: dict):
    with st.chat_message("user"):
        st.markdown(msg["content"])


def _render_assistant_message_user(msg: dict):
    """User-view assistant bubble: answer + subtle tool pills."""
    with st.chat_message("assistant"):
        calls = msg.get("tool_calls", [])
        if calls:
            st.markdown(_tool_tags_html(calls), unsafe_allow_html=True)
        st.markdown(msg["content"])


def _render_assistant_message_dev(msg: dict, turn_idx: int):
    """Dev-view assistant bubble: answer + trace button."""
    with st.chat_message("assistant"):
        calls = msg.get("tool_calls", [])
        if calls:
            st.markdown(_tool_tags_html(calls), unsafe_allow_html=True)
        st.markdown(msg["content"])
        trace = msg.get("trace")
        if trace:
            if st.button("查看轨迹", key=f"trace_btn_{turn_idx}", help="打开完整执行轨迹"):
                st.session_state._dlg_trace = trace
                dialog_trace()


def _parse_memory_slots(summary: str) -> dict:
    """Parse the 4-slot structured summary into a dict."""
    slots = {"关注车型": "", "用户需求": "", "已确认数据": "", "对话脉络": ""}
    for line in summary.splitlines():
        for key in slots:
            prefix = f"[{key}]"
            if line.strip().startswith(prefix):
                slots[key] = line.strip()[len(prefix):].strip()
    return slots


def _render_dev_panel():
    """Right-column developer panel with 3 tabs."""
    tab_trace, tab_mem, tab_sys = st.tabs(["📋 执行轨迹", "🧠 记忆状态", "⚙️ 系统信息"])

    # ── Tab 1: Execution trace ──────────────────────────────
    with tab_trace:
        st.markdown('<div class="dev-scroll">', unsafe_allow_html=True)
        # Latest turn first
        assistant_msgs = [
            (i, m) for i, m in enumerate(st.session_state.messages)
            if m["role"] == "assistant" and m.get("trace")
        ]
        if not assistant_msgs:
            st.markdown('<p style="font-size:13px;color:#9B9B98;margin-top:12px;">尚无执行记录。发送一条消息后这里会显示 Agent 的完整推理过程。</p>', unsafe_allow_html=True)
        else:
            for turn_num, (msg_idx, msg) in enumerate(reversed(assistant_msgs)):
                trace = msg["trace"]
                elapsed = trace.get("elapsed", 0)
                with st.expander(
                    f"**Turn {len(assistant_msgs) - turn_num}** · {trace.get('original_query','')[:40]}… · {elapsed:.1f}s",
                    expanded=(turn_num == 0),
                ):
                    _render_trace_timeline(trace, msg_idx)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Tab 2: Memory state ──────────────────────────────────
    with tab_mem:
        wf = st.session_state.get("workflow")
        if not wf:
            st.markdown('<p style="font-size:13px;color:#9B9B98;margin-top:12px;">发送第一条消息后记忆状态将显示在这里。</p>', unsafe_allow_html=True)
        else:
            mem = wf.memory
            slots = _parse_memory_slots(mem.context_summary)

            st.markdown('<div class="section-label">结构化摘要</div>', unsafe_allow_html=True)
            html = '<div class="dev-card"><div class="mem-grid">'
            labels = {"关注车型": "CAR", "用户需求": "NEEDS", "已确认数据": "DATA", "对话脉络": "THREAD"}
            for key, abbr in labels.items():
                val = slots.get(key, "无") or "无"
                html += f'<div class="mem-key">{abbr}</div><div class="mem-val">{val}</div>'
            html += "</div></div>"
            st.markdown(html, unsafe_allow_html=True)

            if st.button("🔍 查看完整记忆", key="mem_detail_btn"):
                dialog_memory()

            if mem.recent_messages:
                st.markdown('<div class="section-label" style="margin-top:14px;">最近消息</div>', unsafe_allow_html=True)
                for m in reversed(list(mem.recent_messages)):
                    role_cn = "用户" if m["role"] == "user" else "助手"
                    preview = m["content"][:80].replace("\n", " ")
                    tag_cls = "ntag-info" if m["role"] == "user" else ""
                    st.markdown(
                        f'<div class="dev-card" style="margin-bottom:2px;">'
                        f'<span class="ntag {tag_cls}" style="margin-right:6px;">{role_cn}</span>'
                        f'{preview}{"…" if len(m["content"])>80 else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── Tab 3: System info ───────────────────────────────────
    with tab_sys:
        ctx = load_rag_contexts()
        stats = st.session_state.stats

        def stat(label, val):
            st.markdown(
                f'<div class="stat-row"><span class="stat-lbl">{label}</span>'
                f'<span class="stat-val">{val}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-label">模型配置</div>', unsafe_allow_html=True)
        exec_cfg = _build_model_cfg("executor")
        qwen_cfg = _build_model_cfg("qwen")
        exec_provider = st.session_state.get("executor_provider", "OpenAI")
        qwen_provider = st.session_state.get("qwen_provider", "DashScope (Qwen)")
        st.markdown('<div class="dev-card">', unsafe_allow_html=True)
        stat("推理引擎", f"{exec_cfg.get('model','—')} · {exec_provider}")
        stat("改写 & 反思", f"{qwen_cfg.get('model','—')} · {qwen_provider}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:10px;">API 密钥</div>', unsafe_allow_html=True)
        st.markdown('<div class="dev-card">', unsafe_allow_html=True)
        for label, key, hint in [
            ("OpenAI",    config.OPENAI_API_KEY,    exec_cfg.get("model", config.OPENAI_MODEL)),
            ("DashScope", config.DASHSCOPE_API_KEY, qwen_cfg.get("model", config.QWEN_MODEL)),
            ("Serper",    config.SERPER_API_KEY,    "web search"),
        ]:
            dot  = '<span class="dot-ok"></span>' if key else '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;border:1.5px solid #d1d5db;margin-right:6px;"></span>'
            hint_html = f'<span class="stat-val">{hint}</span>'
            st.markdown(
                f'<div class="stat-row"><span class="stat-lbl">{dot}{label}</span>{hint_html}</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:10px;">知识库</div>', unsafe_allow_html=True)
        if ctx:
            st.markdown('<div class="dev-card">', unsafe_allow_html=True)
            for model, rctx in ctx.items():
                try:
                    n = rctx.store.col.num_entities
                except Exception:
                    n = "?"
                stat(model, f"{n} chunks")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.caption("未找到任何知识库")

        st.markdown('<div class="section-label" style="margin-top:10px;">本次会话</div>', unsafe_allow_html=True)
        sid = st.session_state.get("session_id") or "—"
        st.markdown('<div class="dev-card">', unsafe_allow_html=True)
        stat("Session ID", sid)
        stat("总轮次",     stats["turns"])
        stat("工具调用",   stats["tool_calls"])
        stat("反思重试",   stats["retries"])
        st.markdown('</div>', unsafe_allow_html=True)


def _render_trace_timeline(trace: dict, msg_idx: int):
    """Compact timeline inside an expander."""
    st.markdown(
        f'<div style="font-size:12px;color:#9B9B98;margin-bottom:6px;">'
        f'改写 → <em>{trace.get("refined_query","")[:60]}…</em></div>',
        unsafe_allow_html=True,
    )

    for ev in trace.get("events", []):
        etype = ev["type"]
        ts    = ev.get("ts", 0)
        time_str = f"{ts:.2f}s"

        if etype == "rewriting":
            _tl_row("tl-dot-info", "✏️ 查询改写", time_str)
        elif etype == "refined":
            pass  # already shown above
        elif etype == "tool_calling":
            calls = ev.get("calls", [])
            label = f"🔧 工具调用 ({len(calls)}个并行)" if len(calls) > 1 else "🔧 工具调用"
            _tl_row("tl-dot-tool", label, time_str)
            for c in calls:
                name = c.get("name", "")
                if name == "rag_search":
                    car = c.get("car_model", "")
                    q   = c.get("query", "")
                    badge = f'<span class="ntag ntag-rag">知识库 · {car} · {q[:20]}</span>'
                else:
                    q = c.get("query", "")[:28]
                    badge = f'<span class="ntag ntag-web">网络 · {q}</span>'
                st.markdown(
                    f'<div style="padding-left:16px;margin:2px 0;">{badge}</div>',
                    unsafe_allow_html=True,
                )
        elif etype == "tool_done":
            results = ev.get("results", [])
            _tl_row("tl-dot-tool", f"📦 返回结果 ({len(results)}条)", time_str)
            for i, r in enumerate(results):
                col1, col2 = st.columns([5, 1])
                col1.markdown(
                    f'<div style="padding-left:16px;font-size:12px;color:#6B7280;">'
                    f'{r.get("query","")[:40]} — {len((r.get("result") or "").content if hasattr(r.get("result"), "content") else str(r.get("result","")))} chars'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if col2.button("展开", key=f"tool_result_{msg_idx}_{i}"):
                    st.session_state._dlg_tool = {**r, "car_model": r.get("car_model", "")}
                    dialog_tool_result()

        elif etype == "reflecting":
            _tl_row("tl-dot-info", "🔎 质量校验", time_str)
        elif etype == "retry":
            _tl_row("tl-dot-fail", f"↩️ 重试 — {ev.get('feedback','')[:50]}", time_str)
        elif etype == "done":
            _tl_row("tl-dot-done", "✅ 回答完成", time_str)


def _tl_row(dot_cls: str, text: str, time_str: str):
    st.markdown(
        f'<div class="tl-item">'
        f'<div class="tl-dot {dot_cls}"></div>'
        f'<div style="flex:1">{text}</div>'
        f'<div class="tl-time">{time_str}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="notion-title">蔚来 AI 助手</div>', unsafe_allow_html=True)
    st.markdown('<div class="notion-sub">RAG + Agent 智能问答</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="section-label">会话</div>', unsafe_allow_html=True)
    st.text_input(
        "Session ID",
        placeholder="留空则不持久化",
        label_visibility="collapsed",
        key="session_id",
    )
    st.text_area(
        "用户偏好",
        placeholder="例：预算50万，家用，注重续航和空间",
        height=80,
        label_visibility="collapsed",
        key="user_profile",
    )

    if st.button("清空对话", use_container_width=True):
        sid = st.session_state.get("session_id", "").strip()
        if sid:
            _del_path = os.path.join(config.MEMORY_DIR, f"{sid}.json")
            if os.path.exists(_del_path):
                os.remove(_del_path)
        st.session_state.messages = []
        st.session_state.stats    = {"turns": 0, "tool_calls": 0, "retries": 0}
        st.session_state._loaded_session = None
        for k in ("workflow", "_wf_key"):
            st.session_state.pop(k, None)
        st.rerun()

    st.divider()

    # ── Model configuration ───────────────────────────────────
    st.markdown('<div class="section-label">模型配置</div>', unsafe_allow_html=True)

    for role, label, default_provider in [
        ("executor", "推理引擎", "OpenAI"),
        ("qwen",     "改写 & 反思", "DashScope (Qwen)"),
    ]:
        with st.expander(label, expanded=False):
            prev_provider = st.session_state.get(f"{role}_provider", default_provider)
            provider_name = st.selectbox(
                "服务商",
                _PROVIDER_NAMES,
                index=_PROVIDER_NAMES.index(prev_provider) if prev_provider in _PROVIDER_NAMES else 0,
                key=f"{role}_provider",
                label_visibility="collapsed",
            )
            provider = PROVIDER_CATALOG[provider_name]

            # Reset model text when provider changes
            if st.session_state.get(f"{role}_provider_last") != provider_name:
                st.session_state[f"{role}_model"]    = provider["default_model"]
                st.session_state[f"{role}_base_url"] = provider["base_url"] or ""
                st.session_state[f"{role}_provider_last"] = provider_name

            st.text_input(
                "模型名称",
                key=f"{role}_model",
                placeholder=provider["default_model"] or "model-name",
            )
            st.text_input(
                f"API Key ({provider['key_hint']})",
                type="password",
                key=f"{role}_api_key",
                placeholder="留空则使用 .env 中的值",
            )
            if provider_name == "自定义":
                st.text_input(
                    "Base URL",
                    key=f"{role}_base_url",
                    placeholder="https://your-api.com/v1",
                )

    st.divider()
    ctx = load_rag_contexts()
    st.markdown('<div class="section-label">知识库</div>', unsafe_allow_html=True)
    if ctx:
        for model in ctx:
            st.markdown(
                f'<span class="ntag ntag-rag" style="margin:2px 0;display:inline-block;">{model}</span>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<p style="font-size:12px;color:#9B9B98;">未找到数据目录<br/>data/&lt;车型&gt;/ 放入文档后重启</p>',
            unsafe_allow_html=True,
        )

    st.divider()
    web_ok = bool(config.SERPER_API_KEY)
    st.markdown(
        f'<p style="font-size:12px;color:#9B9B98;">'
        f'<span class="{"dot-ok" if web_ok else "dot-no"}"></span>'
        f'网络搜索 {"已启用" if web_ok else "未配置"}</p>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# Main — tab layout
# ─────────────────────────────────────────────────────────────
tab_user, tab_dev = st.tabs(["💬 对话", "🔧 开发者"])

# ── User view ────────────────────────────────────────────────
with tab_user:
    # Render history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            _render_user_message(msg)
        else:
            _render_assistant_message_user(msg)

# ── Developer view ───────────────────────────────────────────
with tab_dev:
    col_chat, col_debug = st.columns([55, 45], gap="medium")

    with col_chat:
        st.markdown('<div class="section-label">对话记录</div>', unsafe_allow_html=True)
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                _render_user_message(msg)
            else:
                _render_assistant_message_dev(msg, i)

    with col_debug:
        _render_dev_panel()

# ─────────────────────────────────────────────────────────────
# Chat input — at page bottom, feeds both views
# ─────────────────────────────────────────────────────────────
if prompt := st.chat_input("问我关于蔚来汽车的任何问题…"):
    # Append user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})

    workflow = get_workflow()

    # Collect streaming events with timestamps
    t0 = time.time()
    events: list[dict] = []
    tool_calls_this_turn: list[dict] = []
    answer   = ""
    refined  = prompt

    # Render live status in user tab chat bubble
    with st.chat_message("assistant"):
        with st.status("思考中…", expanded=True) as status_box:
            for event in workflow.run_stream(prompt):
                etype = event["type"]
                ev_ts = time.time() - t0
                events.append({**event, "ts": ev_ts})

                if etype == "rewriting":
                    st.write("✏️ 理解问题…")

                elif etype == "refined":
                    refined = event["query"]
                    if refined != prompt:
                        st.write(f"💡 *{refined}*")

                elif etype == "tool_calling":
                    calls = event["calls"]
                    tool_calls_this_turn.extend(calls)
                    st.session_state.stats["tool_calls"] += len(calls)
                    if len(calls) > 1:
                        st.write(f"🔧 并行查询 {len(calls)} 个工具…")
                    for c in calls:
                        if c.get("name") == "rag_search":
                            st.write(f"　📚 {c.get('car_model','')} — {c.get('query','')}")
                        else:
                            st.write(f"　🌐 {c.get('query','')}")

                elif etype == "tool_done":
                    st.write(f"✅ 获取到 {len(event['results'])} 条结果")

                elif etype == "reflecting":
                    st.write("🔎 校验答案质量…")

                elif etype == "retry":
                    st.session_state.stats["retries"] += 1
                    st.write(f"↩️ 重新生成…")

                elif etype == "done":
                    answer = event["answer"]
                    status_box.update(label=f"完成 · {ev_ts:.1f}s", state="complete", expanded=False)

        # Tool pills + final answer
        if tool_calls_this_turn:
            st.markdown(_tool_tags_html(tool_calls_this_turn), unsafe_allow_html=True)
        st.markdown(answer)

    # Build trace for dev panel
    trace = {
        "original_query": prompt,
        "refined_query":  refined,
        "elapsed":        time.time() - t0,
        "events":         events,
    }

    st.session_state.messages.append({
        "role":       "assistant",
        "content":    answer,
        "tool_calls": tool_calls_this_turn,
        "trace":      trace,
    })
    st.session_state.stats["turns"] += 1

    # Persist UI history to session JSON so it survives page refresh
    _wf = st.session_state.get("workflow")
    if _wf and _wf.session_path:
        _wf.memory.ui_messages = [
            {"role": m["role"], "content": m["content"],
             "tool_calls": m.get("tool_calls") or [],
             "trace": _serialize_trace(m["trace"]) if m.get("trace") else None}
            for m in st.session_state.messages
        ]
        _wf.memory.save(_wf.session_path)

    st.rerun()
