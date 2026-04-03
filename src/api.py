"""FastAPI backend for the NIO car assistant.

Run:
    cd src && uvicorn api:app --reload --port 8000
    # or from project root:
    PYTHONPATH=src uvicorn src.api:app --reload --port 8000
"""
import math
import sys, os, json, threading, asyncio, dataclasses, time
import re
from collections import OrderedDict
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # loads .env
from fastapi import FastAPI, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.workflow import AgentWorkflow
from agent.memory import ConversationMemory
from tools.registry import ToolRegistry
from tools.web_search import WebSearchTool
from tools.rag_search import RagSearchTool
from tools.grep_search import GrepSearchTool
from rag.pipeline import ingest, RAGContext


# ── Session ID validation ────────────────────────────────────────────────────

_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _validate_session_id(sid: str) -> str | None:
    """Return sanitized session_id or None if invalid."""
    sid = sid.strip()
    if not sid or not _SESSION_ID_RE.match(sid):
        return None
    # Double-check resolved path stays within MEMORY_DIR
    resolved = os.path.realpath(os.path.join(config.MEMORY_DIR, f"{sid}.json"))
    if not resolved.startswith(os.path.realpath(config.MEMORY_DIR)):
        return None
    return sid


def _sanitize_for_json(obj):
    """Recursively replace NaN/Infinity floats with None for JSON safety."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


class _Encoder(json.JSONEncoder):
    """Serialize dataclasses + sanitize NaN/Infinity."""
    def default(self, o):
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return _sanitize_for_json(dataclasses.asdict(o))
        return super().default(o)


# ── Global singletons ─────────────────────────────────────────────────────────

_rag_contexts: dict[str, RAGContext] = {}
_registry: ToolRegistry | None = None

_MAX_WORKFLOWS = 50
_workflows: OrderedDict[str, AgentWorkflow] = OrderedDict()
_workflow_locks: dict[str, threading.Lock] = {}
_workflow_meta_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_contexts, _registry
    _rag_contexts = _load_rag_contexts()
    _registry = _build_registry(_rag_contexts)
    yield
    # Shutdown: release all loaded collections to free memory
    for model, ctx in _rag_contexts.items():
        try:
            ctx.store.release()
            print(f"[shutdown] {model} collection released")
        except Exception:
            pass
    from pymilvus import connections
    from storage.vector_store import _connected_uris
    try:
        connections.disconnect("default")
        _connected_uris.clear()
    except Exception:
        pass


app = FastAPI(title="NIO AI Assistant", lifespan=lifespan)

# CORS: configurable via CORS_ORIGINS env var; defaults to localhost for security
_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:*,http://127.0.0.1:*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
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


def _get_session_lock(session_id: str) -> threading.Lock:
    """Get or create a per-session lock (thread-safe)."""
    with _workflow_meta_lock:
        if session_id not in _workflow_locks:
            _workflow_locks[session_id] = threading.Lock()
        return _workflow_locks[session_id]


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
    with _workflow_meta_lock:
        if session_id not in _workflows:
            # Evict oldest if at capacity
            while len(_workflows) >= _MAX_WORKFLOWS:
                _workflows.popitem(last=False)
            _workflows[session_id] = AgentWorkflow(
                registry=_registry,
                user_profile=user_profile,
                session_id=session_id,
                executor_cfg=exec_cfg,
                qwen_cfg=qwen_cfg,
            )
        else:
            # Move to end (most recently used)
            _workflows.move_to_end(session_id)
            if user_profile:
                _workflows[session_id].memory.user_profile = user_profile
                _workflows[session_id].memory.global_user_info.raw = user_profile
        return _workflows[session_id]


# ── Endpoints ─────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="", max_length=64)
    user_profile: str = Field(default="", max_length=500)


# Keep GET for backward compat but prefer POST
@app.post("/api/chat/stream")
@app.get("/api/chat/stream")
async def chat_stream(
    request: Request,
    q:            str = Query(default=""),
    session_id:   str = Query(default=""),
    user_profile: str = Query(default=""),
):
    """SSE endpoint — yields workflow events as `data: <json>\\n\\n` frames."""
    # Support both POST JSON body and GET query params
    if request.method == "POST":
        try:
            body = await request.json()
            req = ChatRequest(**body)
            q, session_id, user_profile = req.q, req.session_id, req.user_profile
        except Exception:
            pass  # Fall through to query params

    if not q or not q.strip():
        return StreamingResponse(
            iter(["data: {\"type\":\"error\",\"message\":\"q is required\"}\n\n"]),
            media_type="text/event-stream",
        )

    sid = _validate_session_id(session_id) or ""
    workflow = _get_or_create_workflow(sid, user_profile.strip()[:500])

    # Acquire session lock at the request entry point (not in background thread)
    session_lock = _get_session_lock(sid) if sid else None
    if session_lock and not session_lock.acquire(blocking=False):
        return StreamingResponse(
            iter(["data: {\"type\":\"error\",\"message\":\"session busy\"}\n\n", "data: [DONE]\n\n"]),
            media_type="text/event-stream",
        )

    # Unbounded queue — bounded queue + sync put can deadlock the producer thread
    event_q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    cancelled = threading.Event()

    def _run_sync():
        collected: list[dict] = []
        t0 = time.time()
        try:
            for ev in workflow.run_stream(q):
                if cancelled.is_set():
                    break
                collected.append({**ev, "ts": time.time() - t0})
                asyncio.run_coroutine_threadsafe(event_q.put(ev), loop).result()
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                event_q.put({"type": "error", "message": str(e)}), loop
            ).result()
        finally:
            if session_lock:
                session_lock.release()
            done_ev = next((e for e in collected if e["type"] == "done"), None)
            if done_ev:
                refined_ev = next((e for e in collected if e["type"] == "refined"), None)
                trace = {
                    "original_query": q,
                    "refined_query": refined_ev["query"] if refined_ev else q,
                    "elapsed": time.time() - t0,
                    "events": json.loads(json.dumps(
                        _sanitize_for_json([e for e in collected if e["type"] != "text_delta"]),
                        cls=_Encoder,
                    )),
                }
                workflow.memory.attach_trace_to_last_assistant(trace)
                workflow._persist()
            asyncio.run_coroutine_threadsafe(event_q.put(None), loop).result()

    threading.Thread(target=_run_sync, daemon=True).start()

    async def _sse():
        try:
            while True:
                if await request.is_disconnected():
                    cancelled.set()
                    break
                try:
                    ev = await asyncio.wait_for(event_q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if ev is None:
                    break
                yield f"data: {json.dumps(_sanitize_for_json(ev), cls=_Encoder, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            cancelled.set()
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
    sid = _validate_session_id(req.session_id)
    if not sid:
        return {"ok": False, "error": "invalid session_id"}
    lock = _get_session_lock(sid)
    with lock:
        with _workflow_meta_lock:
            _workflows.pop(sid, None)
            _workflow_locks.pop(sid, None)
        path = os.path.join(config.MEMORY_DIR, f"{sid}.json")
        if os.path.exists(path):
            os.remove(path)
    return {"ok": True}


@app.get("/api/session/memory")
async def get_memory(session_id: str = Query(default="")):
    sid = _validate_session_id(session_id)
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
    sid = _validate_session_id(session_id)
    if not sid:
        return {"ui_messages": [], "recent_messages": []}
    # Prefer in-memory workflow (has latest state)
    wf = _workflows.get(sid)
    if wf:
        return _sanitize_for_json({
            "ui_messages": wf.memory.ui_messages,
            "recent_messages": list(wf.memory.recent_messages),
        })
    # Fall back to disk
    path = os.path.join(config.MEMORY_DIR, f"{sid}.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            return _sanitize_for_json({
                "ui_messages": data.get("ui_messages", []),
                "recent_messages": data.get("recent_messages", []),
            })
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
