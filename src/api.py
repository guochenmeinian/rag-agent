"""FastAPI backend for the NIO car assistant.

Run:
    cd src && uvicorn api:app --reload --port 8000
    # or from project root:
    PYTHONPATH=src uvicorn src.api:app --reload --port 8000
"""
import sys, os, json, threading, asyncio, shutil, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # loads .env
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.workflow import AgentWorkflow
from tools.registry import ToolRegistry
from tools.web_search import WebSearchTool
from tools.rag_search import RagSearchTool
from rag.pipeline import ingest, RAGContext

app = FastAPI(title="NIO AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global singletons ─────────────────────────────────────────────────────────

_rag_contexts: dict[str, RAGContext] = {}
_registry: ToolRegistry | None = None
_workflows: dict[str, AgentWorkflow] = {}


@app.on_event("startup")
async def _startup():
    global _rag_contexts, _registry
    _rag_contexts = _load_rag_contexts()
    _registry = _build_registry(_rag_contexts)


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
                with tempfile.TemporaryDirectory() as tmp:
                    shutil.copy(flat_pdf, os.path.join(tmp, f"{model}.pdf"))
                    ctx = ingest(data_dir=tmp, uri=config.MILVUS_URI, col_name=col_name)
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
    return reg


def _get_or_create_workflow(session_id: str, user_profile: str) -> AgentWorkflow:
    key = session_id or "__anon__"
    if key not in _workflows:
        _workflows[key] = AgentWorkflow(
            registry=_registry,
            user_profile=user_profile,
            session_id=session_id or None,
        )
    elif user_profile:
        _workflows[key].memory.user_profile = user_profile
    return _workflows[key]


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
        try:
            for ev in workflow.run_stream(q):
                asyncio.run_coroutine_threadsafe(event_q.put(ev), loop).result()
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                event_q.put({"type": "error", "message": str(e)}), loop
            ).result()
        finally:
            asyncio.run_coroutine_threadsafe(event_q.put(None), loop).result()

    threading.Thread(target=_run_sync, daemon=True).start()

    async def _sse():
        while True:
            ev = await event_q.get()
            if ev is None:
                break
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
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
        "api_keys": {
            "OpenAI":    bool(config.OPENAI_API_KEY),
            "DashScope": bool(config.DASHSCOPE_API_KEY),
            "Serper":    bool(config.SERPER_API_KEY),
        },
    }


# ── Static files (must be registered last) ───────────────────────────────────

_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
