"""QAnything adapter — wraps the QAnything REST API to ingest PDFs and answer questions.

Prerequisites
─────────────
1. QAnything running via Docker:
       cd third-party/QAnything
       # Linux/Mac:
       docker compose -f docker-compose-linux.yaml up -d
       # Mac M-series (ARM):
       docker compose -f docker-compose-mac.yaml up -d
   Default API port: 8777

2. No extra Python packages needed — uses only `requests` (already in env).

3. QAnything v2 does NOT bundle a local LLM.  It needs any OpenAI-compatible
   endpoint for the answer-generation step.  The embedding & reranking are
   handled locally by its own models (no key needed for those).

   Recommended: reuse this project's existing executor config from src/config.py.
   The adapter reads EXECUTOR_BASE_URL / EXECUTOR_API_KEY / EXECUTOR_MODEL
   automatically, so no extra env vars are needed if your .env is already set up.

   Alternatives:
     • Ollama (local, free):
         api_base = "http://host.docker.internal:11434/v1"
         api_key  = "ollama"
         model    = "qwen2:7b"   # or any model you have pulled
         WARNING: ollama defaults to 2048-token context — answers may be truncated.
     • DashScope (Qwen / 通义千问):
         api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
         api_key  = "<your DashScope key>"
         model    = "qwen-max"

Notes
─────
- QAnything v2 uses a fixed user_id="zzp" by default (matches the web UI).
- Knowledge base named "nio_benchmark" is created and reused across runs.
- hybrid_search=True enables both dense + sparse retrieval (matches our pipeline).
- rerank=True enables QAnything's built-in cross-encoder reranker.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from .base_adapter import BaseAdapter

_USER_ID = "zzp"
_KB_NAME = "nio_benchmark"
_GREEN   = "green"


class QAnythingAdapter(BaseAdapter):
    name = "qanything"

    def __init__(
        self,
        base_url: str = "http://localhost:8777",
        openai_api_base: str | None = None,
        openai_api_key: str | None = None,
        openai_model: str | None = None,
        max_token: int = 1024,
        hybrid_search: bool = True,
        rerank: bool = True,
        poll_interval: int = 10,
        poll_timeout: int = 600,
    ):
        # Priority: explicit param → env var → project src/config.py (EXECUTOR_*)
        # This means no extra env vars are needed if your .env is already configured.
        _repo = Path(__file__).resolve().parents[2]
        try:
            if str(_repo / "src") not in sys.path:
                sys.path.insert(0, str(_repo / "src"))
            import config as _cfg
            _default_base  = getattr(_cfg, "EXECUTOR_BASE_URL", "") or "https://api.openai.com/v1"
            _default_key   = getattr(_cfg, "EXECUTOR_API_KEY", "")
            _default_model = getattr(_cfg, "EXECUTOR_MODEL", "gpt-4o")
        except Exception:
            _default_base  = "https://api.openai.com/v1"
            _default_key   = ""
            _default_model = "gpt-4o"

        self.base_url      = base_url.rstrip("/")
        self.api_base      = openai_api_base or os.environ.get("OPENAI_API_BASE", _default_base)
        self.api_key       = openai_api_key  or os.environ.get("OPENAI_API_KEY",  _default_key)
        self.model         = openai_model    or os.environ.get("OPENAI_MODEL",     _default_model)
        self.max_token      = max_token
        self.hybrid_search  = hybrid_search
        self.rerank         = rerank
        self.poll_interval  = poll_interval
        self.poll_timeout   = poll_timeout

        self._kb_id: str | None = None

    # ─────────────────────────────────────────────────────────
    # Low-level HTTP helpers
    # ─────────────────────────────────────────────────────────

    def _post(self, path: str, payload: dict, timeout: int = 60) -> dict:
        url = f"{self.base_url}{path}"
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _post_files(self, path: str, data: dict, files: list, timeout: int = 120) -> dict:
        url = f"{self.base_url}{path}"
        resp = requests.post(url, data=data, files=files, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    # ─────────────────────────────────────────────────────────
    # KB management
    # ─────────────────────────────────────────────────────────

    def _find_existing_kb(self) -> str | None:
        result = self._post("/api/local_doc_qa/list_knowledge_base", {"user_id": _USER_ID})
        for kb in result.get("data", []):
            if kb.get("kb_name") == _KB_NAME:
                return kb["kb_id"]
        return None

    def _create_kb(self) -> str:
        result = self._post(
            "/api/local_doc_qa/new_knowledge_base",
            {"user_id": _USER_ID, "kb_name": _KB_NAME},
        )
        kb_id = result["data"]["kb_id"]
        print(f"  [qanything] created knowledge base '{_KB_NAME}' → kb_id={kb_id}")
        return kb_id

    def _list_files(self) -> list[dict]:
        result = self._post(
            "/api/local_doc_qa/list_files",
            {"user_id": _USER_ID, "kb_id": self._kb_id, "page_offset": 1, "page_limit": 100},
        )
        return result.get("data", {}).get("details", [])

    def _upload_pdfs(self, pdf_files: list[str]) -> None:
        existing_names = {f["file_name"] for f in self._list_files()}
        to_upload = [p for p in pdf_files if Path(p).name not in existing_names]
        if not to_upload:
            print(f"  [qanything] all {len(pdf_files)} PDFs already uploaded, skipping")
            return

        print(f"  [qanything] uploading {len(to_upload)} PDF(s)…")
        for pdf_path in to_upload:
            fname = Path(pdf_path).name
            with open(pdf_path, "rb") as f:
                result = self._post_files(
                    "/api/local_doc_qa/upload_files",
                    data={"user_id": _USER_ID, "kb_id": self._kb_id, "mode": "soft"},
                    files=[("files", (fname, f, "application/pdf"))],
                )
            if result.get("code") != 200:
                print(f"  [qanything] WARNING: upload failed for {fname}: {result.get('msg')}")
            else:
                print(f"  [qanything] uploaded {fname}")

    def _wait_for_indexing(self) -> None:
        """Poll until all files reach 'green' status."""
        deadline = time.time() + self.poll_timeout
        print(f"  [qanything] waiting for indexing (timeout={self.poll_timeout}s)…")
        while time.time() < deadline:
            files = self._list_files()
            if not files:
                time.sleep(self.poll_interval)
                continue
            green_n = sum(1 for f in files if f.get("status") == _GREEN)
            total_n = len(files)
            statuses = {f.get("status") for f in files}
            print(f"    {green_n}/{total_n} green  (statuses: {statuses})")
            if green_n == total_n:
                print("  [qanything] indexing complete")
                return
            if any(f.get("status") == "red" for f in files):
                failed = [f["file_name"] for f in files if f.get("status") == "red"]
                print(f"  [qanything] WARNING: some files failed to index: {failed}")
                # Continue polling for remaining files
            time.sleep(self.poll_interval)

        raise TimeoutError(
            f"QAnything indexing did not finish within {self.poll_timeout}s. "
            "Check the QAnything web UI for errors."
        )

    # ─────────────────────────────────────────────────────────
    # BaseAdapter interface
    # ─────────────────────────────────────────────────────────

    def setup(self, pdf_files: list[str]) -> None:
        self._kb_id = self._find_existing_kb()
        if self._kb_id:
            print(f"  [qanything] reusing existing KB '{_KB_NAME}' (kb_id={self._kb_id})")
        else:
            self._kb_id = self._create_kb()

        self._upload_pdfs(pdf_files)
        self._wait_for_indexing()
        print(f"  [qanything] setup complete — kb_id={self._kb_id}")

    def ask(self, question: str) -> str:
        if self._kb_id is None:
            raise RuntimeError("Call setup() before ask()")
        payload = {
            "user_id":           _USER_ID,
            "kb_ids":            [self._kb_id],
            "question":          question,
            "history":           [],
            "streaming":         False,
            "networking":        False,
            "rerank":            self.rerank,
            "hybrid_search":     self.hybrid_search,
            "only_need_search_results": False,
            "max_token":         self.max_token,
            "api_base":          self.api_base,
            "api_key":           self.api_key,
            "model":             self.model,
            "api_context_length": 16000,
            "temperature":       0.1,
            "top_p":             0.99,
        }
        result = self._post("/api/local_doc_qa/local_doc_chat", payload, timeout=120)
        return result.get("response", "").strip()

    def teardown(self) -> None:
        print("  [qanything] teardown (KB preserved for reuse)")
