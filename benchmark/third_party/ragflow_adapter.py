"""RAGFlow adapter — wraps ragflow_sdk to ingest PDFs and answer questions.

Setup checklist (one-time)
──────────────────────────
1. Start RAGFlow via Docker:
       cd third-party/ragflow
       docker compose -f docker/docker-compose.yml up -d
   Web UI → http://localhost:80

2. Configure your LLM inside the Web UI (use your existing project keys):
       Settings → Model providers → Add
         Provider:  OpenAI (or whichever your EXECUTOR_BASE_URL points to)
         API Key:   <your EXECUTOR_API_KEY>
         Base URL:  <your EXECUTOR_BASE_URL>  (leave empty for OpenAI default)
       Settings → Model providers → Set as default LLM
   After this step RAGFlow knows how to generate answers using your key.

3. Create a RAGFlow SDK API key (this is for authenticating the Python SDK,
   NOT the same as your OpenAI key):
       Settings → API Key → Create
       Copy the key (format: ragflow-xxxxxxxxxxxxxxxx)

4. Export it:
       export RAGFLOW_API_KEY=ragflow-xxxxxxxxxxxxxxxx

   This is the only key the adapter needs at runtime — RAGFlow handles the
   LLM key internally after step 2.

Why is a RAGFlow SDK key required?
───────────────────────────────────
Every HTTP request to the RAGFlow API uses  Authorization: Bearer <key>.
There is no unauthenticated mode, even for self-hosted instances.
The SDK key identifies which RAGFlow user account owns the datasets.

Chunk method
────────────
  "naive"   — CPU-only, token-based chunking (default, works everywhere)
  "deepdoc" — RAGFlow's vision-model parser (requires GPU + downloaded models,
               best for tables / complex layouts, the whole reason we're testing it)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from .base_adapter import BaseAdapter

_DATASET_NAME = "nio_benchmark"
_CHAT_NAME    = "nio_benchmark_chat"


def _load_project_config() -> dict:
    """Try to load EXECUTOR_* values from src/config.py for reference in error messages."""
    try:
        _repo = Path(__file__).resolve().parents[2]
        if str(_repo / "src") not in sys.path:
            sys.path.insert(0, str(_repo / "src"))
        import config as _cfg
        return {
            "executor_model":    getattr(_cfg, "EXECUTOR_MODEL", "gpt-4o"),
            "executor_base_url": getattr(_cfg, "EXECUTOR_BASE_URL", ""),
        }
    except Exception:
        return {}


class RAGFlowAdapter(BaseAdapter):
    name = "ragflow"

    def __init__(
        self,
        base_url: str = "http://localhost:80",
        api_key: str | None = None,
        chunk_method: str = "naive",
        poll_interval: int = 10,
        poll_timeout: int = 600,
    ):
        self.base_url      = base_url.rstrip("/")
        self.chunk_method  = chunk_method
        self.poll_interval = poll_interval
        self.poll_timeout  = poll_timeout

        # RAGFlow SDK key — must come from env var RAGFLOW_API_KEY or explicit param.
        # This is NOT your OpenAI/executor key.  See module docstring for how to get it.
        self.api_key = api_key or os.environ.get("RAGFLOW_API_KEY", "")

        self._rag     = None
        self._dataset = None
        self._chat    = None

    # ─────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────

    def _client(self):
        if self._rag is not None:
            return self._rag

        if not self.api_key:
            cfg = _load_project_config()
            model = cfg.get("executor_model", "gpt-4o")
            base  = cfg.get("executor_base_url") or "https://api.openai.com/v1"
            raise RuntimeError(
                "RAGFlow SDK API key not set.\n\n"
                "This is NOT your OpenAI key — it's a RAGFlow-specific auth token.\n\n"
                "Steps to get it:\n"
                "  1. Open RAGFlow web UI → http://localhost:80\n"
                "  2. Settings → Model providers → add your LLM:\n"
                f"       API Key   = EXECUTOR_API_KEY from your .env\n"
                f"       Base URL  = {base}\n"
                f"       Model     = {model}\n"
                "     Set it as default LLM.\n"
                "  3. Settings → API Key → Create  (copy the ragflow-xxx... key)\n"
                "  4. export RAGFLOW_API_KEY=ragflow-xxxxxxxxxxxxxxxx\n"
            )

        try:
            from ragflow_sdk import RAGFlow
        except ImportError:
            raise ImportError(
                "ragflow_sdk not installed. Run:\n"
                "  pip install third-party/ragflow/sdk/python/"
            )

        self._rag = RAGFlow(api_key=self.api_key, base_url=self.base_url)
        return self._rag

    def _get_or_create_dataset(self):
        rag = self._client()
        existing = rag.list_datasets(name=_DATASET_NAME)
        if existing:
            print(f"  [ragflow] reusing dataset '{_DATASET_NAME}'")
            return existing[0]
        print(f"  [ragflow] creating dataset '{_DATASET_NAME}' (chunk_method={self.chunk_method})")
        return rag.create_dataset(name=_DATASET_NAME, chunk_method=self.chunk_method)

    def _upload_missing_pdfs(self, dataset, pdf_files: list[str]):
        existing_names = {doc.name for doc in dataset.list_documents()}
        to_upload = [p for p in pdf_files if Path(p).name not in existing_names]
        if not to_upload:
            print(f"  [ragflow] all {len(pdf_files)} PDFs already present, skipping upload")
            return

        print(f"  [ragflow] uploading {len(to_upload)} PDF(s): {[Path(p).name for p in to_upload]}")
        doc_list = [
            {"display_name": Path(p).name, "blob": open(p, "rb").read()}
            for p in to_upload
        ]
        dataset.upload_documents(doc_list)

    def _wait_for_indexing(self, dataset):
        deadline = time.time() + self.poll_timeout
        print(f"  [ragflow] waiting for indexing (timeout={self.poll_timeout}s) …")

        # Kick off async parsing for any un-parsed docs
        docs = dataset.list_documents()
        ids_to_parse = [
            d.id for d in docs
            if str(getattr(d, "run", "")).upper() not in ("DONE", "1")
        ]
        if ids_to_parse:
            dataset.async_parse_documents(ids_to_parse)

        while time.time() < deadline:
            docs     = dataset.list_documents()
            done_n   = sum(1 for d in docs if str(getattr(d, "run", "")).upper() in ("DONE", "1"))
            total_n  = len(docs)
            statuses = {str(getattr(d, "run", "?")).upper() for d in docs}
            print(f"    {done_n}/{total_n} done  (statuses: {statuses})")
            if done_n == total_n > 0:
                print("  [ragflow] indexing complete")
                return
            time.sleep(self.poll_interval)

        raise TimeoutError(
            f"RAGFlow indexing did not finish within {self.poll_timeout}s.\n"
            "Check the RAGFlow web UI → Datasets for errors."
        )

    def _get_or_create_chat(self, dataset):
        rag = self._client()
        existing = rag.list_chats(name=_CHAT_NAME)
        if existing:
            print(f"  [ragflow] reusing chat assistant '{_CHAT_NAME}'")
            return existing[0]
        print(f"  [ragflow] creating chat assistant '{_CHAT_NAME}'")
        # model_name=None → use the system's default LLM configured in Web UI
        return rag.create_chat(name=_CHAT_NAME, dataset_ids=[dataset.id])

    # ─────────────────────────────────────────────────────────
    # BaseAdapter interface
    # ─────────────────────────────────────────────────────────

    def setup(self, pdf_files: list[str]) -> None:
        self._dataset = self._get_or_create_dataset()
        self._upload_missing_pdfs(self._dataset, pdf_files)
        self._wait_for_indexing(self._dataset)
        self._chat = self._get_or_create_chat(self._dataset)
        print("  [ragflow] setup complete")

    def ask(self, question: str) -> str:
        if self._chat is None:
            raise RuntimeError("Call setup() before ask()")
        # Fresh session per question — no conversation bleed-through between test cases
        session = self._chat.create_session()
        parts: list[str] = []
        for msg in session.ask(question, stream=False):
            parts.append(msg.content)
        return " ".join(parts).strip()

    def teardown(self) -> None:
        # Dataset and chat are preserved for reuse on the next run
        print("  [ragflow] teardown (dataset + chat preserved for reuse)")
