"""FastAPI backend for the NIO car assistant.

Run:
    cd src && uvicorn api:app --reload --port 8000
    # or from project root:
    PYTHONPATH=src uvicorn src.api:app --reload --port 8000
"""
import sys, os, json, threading, asyncio, dataclasses, time
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # loads .env
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.workflow import AgentWorkflow
from agent.memory import ConversationMemory
from tools.registry import ToolRegistry
from tools.web_search import WebSearchTool
from tools.rag_search import RagSearchTool
from tools.grep_search import GrepSearchTool
from rag.pipeline import ingest, RAGContext

class _Encoder(json.JSONEncoder):
    """Serialize dataclasses (e.g. ToolResult) that json.dumps can't handle natively."""
    def default(self, o):
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        return super().default(o)


# ── Global singletons ─────────────────────────────────────────────────────────

_rag_contexts: dict[str, RAGContext] = {}
_registry: ToolRegistry | None = None
_workflows: dict[str, AgentWorkflow] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_contexts, _registry
    _rag_contexts = _load_rag_contexts()
    _registry = _build_registry(_rag_contexts)
    yield


app = FastAPI(title="NIO AI Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_rag_contexts() -> dict[str, RAGContext]:
    contexts: dict[str, RAGContext] = {}
    for model in config.NIO_CAR_MODELS:
        col_name = f"nio_{model.lower()}"
        subdir   = os.path.join(config.DATA_ROOT, model)
        flat_pdf = os.path.join(config.DATA_ROOT, f"{model}.pdf")
        try:
            if os.path.isdir(subdir):
                ctx = ingest(data_dir=subdir, uri=config.MILVUS_URI, col_name=col_name)
            elif os.path.isfile(flat_pdf):
                ctx = ingest(data_dir=config.DATA_ROOT, uri=config.MILVUS_URI,
                             col_name=col_name, file_filter=f"{model}.pdf")
            else:
                continue
            contexts[model] = ctx
            print(f"[startup] {model} ready ({ctx.store.col.num_entities} chunks)")
        except Exception as e:
            print(f"[startup] {model} failed: {e}")
    return contexts


def _build_registry(rag_contexts: dict) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(WebSearchTool())
    if rag_contexts:
        reg.register(RagSearchTool(contexts=rag_contexts))
    reg.register(GrepSearchTool())
    return reg


def _get_or_create_workflow(session_id: str, user_profile: str) -> AgentWorkflow:
    exec_cfg = config.get_executor_cfg()
    qwen_cfg = config.get_qwen_cfg()
    if not session_id:
        return AgentWorkflow(
            registry=_registry,
            user_profile=user_profile,
            executor_cfg=exec_cfg,
            qwen_cfg=qwen_cfg,
        )
    if session_id not in _workflows:
        _workflows[session_id] = AgentWorkflow(
            registry=_registry,
            user_profile=user_profile,
            session_id=session_id,
            executor_cfg=exec_cfg,
            qwen_cfg=qwen_cfg,
        )
    elif user_profile:
        _workflows[session_id].memory.user_profile = user_profile
        _workflows[session_id].memory.global_user_info.raw = user_profile
    return _workflows[session_id]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/chat/stream")
async def chat_stream(
    q:            str = Query(...),
    session_id:   str = Query(default=""),
    user_profile: str = Query(default=""),
):
    """SSE endpoint — yields workflow events as `data: <json>\n\n` frames."""
    workflow = _get_or_create_workflow(session_id.strip(), user_profile.strip())
    event_q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _run_sync():
        collected: list[dict] = []
        t0 = time.time()
        try:
            for ev in workflow.run_stream(q):
                collected.append({**ev, "ts": time.time() - t0})
                asyncio.run_coroutine_threadsafe(event_q.put(ev), loop).result()
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                event_q.put({"type": "error", "message": str(e)}), loop
            ).result()
        finally:
            done_ev = next((e for e in collected if e["type"] == "done"), None)
            if done_ev:
                refined_ev = next((e for e in collected if e["type"] == "refined"), None)
                trace = {
                    "original_query": q,
                    "refined_query": refined_ev["query"] if refined_ev else q,
                    "elapsed": time.time() - t0,
                    "events": json.loads(json.dumps(collected, cls=_Encoder)),
                }
                workflow.memory.attach_trace_to_last_assistant(trace)
                workflow._persist()
            asyncio.run_coroutine_threadsafe(event_q.put(None), loop).result()

    threading.Thread(target=_run_sync, daemon=True).start()

    async def _sse():
        while True:
            ev = await event_q.get()
            if ev is None:
                break
            yield f"data: {json.dumps(ev, cls=_Encoder, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ClearRequest(BaseModel):
    session_id: str = ""


@app.post("/api/session/clear")
async def clear_session(req: ClearRequest):
    key = req.session_id.strip() or "__anon__"
    _workflows.pop(key, None)
    if req.session_id.strip():
        path = os.path.join(config.MEMORY_DIR, f"{req.session_id.strip()}.json")
        if os.path.exists(path):
            os.remove(path)
    return {"ok": True}


@app.get("/api/session/memory")
async def get_memory(session_id: str = Query(default="")):
    sid = session_id.strip()
    mem: ConversationMemory | None = None

    if sid:
        wf = _workflows.get(sid)
        if wf:
            mem = wf.memory
        else:
            path = os.path.join(config.MEMORY_DIR, f"{sid}.json")
            if os.path.exists(path):
                mem = ConversationMemory.load(path)

    if mem is None:
        return {"facts": [], "global_user_info": {}, "recent_messages": []}

    return {
        "facts": mem.facts,
        "global_user_info": mem.global_user_info.to_dict(),
        "recent_messages": list(mem.recent_messages),
    }


@app.get("/api/sessions")
async def list_sessions():
    """List all saved sessions, sorted by last-modified descending."""
    import glob
    sessions = []
    pattern = os.path.join(config.MEMORY_DIR, "*.json")
    for path in sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
            session_id = os.path.basename(path).replace(".json", "")
            messages = data.get("recent_messages", [])
            preview = next((m["content"] for m in messages if m["role"] == "user"), "")
            sessions.append({
                "session_id": session_id,
                "preview": preview[:80],
                "message_count": len(messages),
                "last_modified": os.path.getmtime(path),
            })
        except Exception:
            pass
    return sessions


@app.get("/api/session/messages")
async def get_session_messages(session_id: str = Query(default="")):
    """Return ui_messages (with trace) for a session (from disk or in-memory workflow)."""
    sid = session_id.strip()
    if not sid:
        return {"ui_messages": [], "recent_messages": []}
    # Prefer in-memory workflow (has latest state)
    wf = _workflows.get(sid)
    if wf:
        return {
            "ui_messages": wf.memory.ui_messages,
            "recent_messages": list(wf.memory.recent_messages),
        }
    # Fall back to disk
    path = os.path.join(config.MEMORY_DIR, f"{sid}.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            return {
                "ui_messages": data.get("ui_messages", []),
                "recent_messages": data.get("recent_messages", []),
            }
        except Exception:
            pass
    return {"ui_messages": [], "recent_messages": []}


@app.get("/api/status")
async def status():
    rag_models = {}
    for model, ctx in _rag_contexts.items():
        try:
            rag_models[model] = ctx.store.col.num_entities
        except Exception:
            rag_models[model] = "?"
    return {
        "rag": {"models": rag_models},
        "models": {
            "executor": config.EXECUTOR_MODEL,
            "qwen": config.QWEN_MODEL,
        },
        "api_keys": {
            "Executor":  bool(config.EXECUTOR_API_KEY),
            "Qwen":     bool(config.QWEN_API_KEY),
            "Serper":   bool(config.SERPER_API_KEY),
        },
    }


# ── Static files (must be registered last) ───────────────────────────────────

_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
