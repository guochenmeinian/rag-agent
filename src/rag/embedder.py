# download bge model 
import os
from pathlib import Path
from modelscope import snapshot_download
from pymilvus import model

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def load_bge_m3_embedder():
    project_root = Path(__file__).resolve().parents[2]
    default_cache_dir = project_root / ".cache" / "modelscope"
    model_cache_dir = Path(os.getenv("RAG_MODEL_CACHE", str(default_cache_dir)))
    model_cache_dir.mkdir(parents=True, exist_ok=True)
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
        device="cpu"
    )

def embed_texts(chunks, embedder):
    if not chunks:
        raise ValueError("Chunks list is empty.")
    
    return embedder(chunks)

def embed_query(query, embedder):
    if not query:
        raise ValueError("Query is empty.")
    
    return embedder([query])
