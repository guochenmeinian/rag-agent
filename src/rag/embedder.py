# download bge model
import os
import platform
from pathlib import Path
from modelscope import snapshot_download
from pymilvus import model
from FlagEmbedding import FlagReranker

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def _get_device() -> str:
    """Return the best available device: mps > cpu."""
    if platform.system() == "Darwin":
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
    return "cpu"

def _get_model_cache_dir() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    default_cache_dir = project_root / ".cache" / "modelscope"
    cache_dir = Path(os.getenv("RAG_MODEL_CACHE", str(default_cache_dir)))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_bge_m3_embedder():
    model_cache_dir = _get_model_cache_dir()
    local_model_dir = model_cache_dir / "BAAI" / "bge-m3"

    if local_model_dir.exists() and (local_model_dir / "config.json").exists():
        model_dir = str(local_model_dir)
    else:
        model_dir = snapshot_download('BAAI/bge-m3',
        cache_dir=str(model_cache_dir), revision='master')

    # BGEM3EmbeddingFunction: https://milvus.io/api-reference/pymilvus/v2.4.x/EmbeddingModels/BGEM3EmbeddingFunction/BGEM3EmbeddingFunction.md
    return model.hybrid.BGEM3EmbeddingFunction(
        model_dir,
        use_fp16=False,
        device=_get_device()
    )


_reranker_instance: FlagReranker | None = None


def load_bge_reranker() -> FlagReranker:
    """Load BGE-Reranker-v2-m3 cross-encoder (singleton — loaded once, shared across all RAGContexts)."""
    global _reranker_instance
    if _reranker_instance is not None:
        return _reranker_instance

    model_cache_dir = _get_model_cache_dir()
    local_model_dir = model_cache_dir / "BAAI" / "bge-reranker-v2-m3"

    if local_model_dir.exists() and (local_model_dir / "config.json").exists():
        model_dir = str(local_model_dir)
    else:
        model_dir = snapshot_download(
            'BAAI/bge-reranker-v2-m3',
            cache_dir=str(model_cache_dir),
            revision='master',
        )

    _reranker_instance = FlagReranker(model_dir, use_fp16=False, device=_get_device())
    return _reranker_instance

def embed_texts(chunks, embedder):
    if not chunks:
        raise ValueError("Chunks list is empty.")
    
    return embedder(chunks)

def embed_query(query, embedder):
    if not query:
        raise ValueError("Query is empty.")
    
    return embedder([query])
