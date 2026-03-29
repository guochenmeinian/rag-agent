"""Abstract base class for third-party RAG system adapters."""
from __future__ import annotations
from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    """
    Common interface for RAGFlow / QAnything / any external RAG system.

    Lifecycle:
        adapter = MyAdapter(base_url="http://localhost:XXXX", **cfg)
        adapter.setup(pdf_files=["/data/ET7.pdf", ...])   # upload + ingest
        answer = adapter.ask("ET7的150kWh续航是多少？")
        adapter.teardown()                                  # optional cleanup
    """

    name: str = "unknown"

    @abstractmethod
    def setup(self, pdf_files: list[str]) -> None:
        """Upload PDFs and wait until they are fully indexed.

        This method should be idempotent — if the knowledge base already exists
        and is fully indexed, it should not re-upload.
        """

    @abstractmethod
    def ask(self, question: str) -> str:
        """Send a question and return the plain-text answer string."""

    def teardown(self) -> None:
        """Optional: delete temporary KB / session created during setup."""
