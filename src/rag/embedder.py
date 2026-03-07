# download bge model 
import os
from modelscope import snapshot_download
from pymilvus import model

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def load_bge_m3_embedder():

    model_dir = snapshot_download('BAAI/bge-m3',
    cache_dir='./', revision='master')

    # BGEM3EmbeddingFunction: https://milvus.io/api-reference/pymilvus/v2.4.x/EmbeddingModels/BGEM3EmbeddingFunction/BGEM3EmbeddingFunction.md
    return model.hybrid.BGEM3EmbeddingFunction(
        model_dir,
        use_fp16=False, 
        device="cpu"
    )

def embed_texts(chunks, embedder):
    if not chunks:
        return ValueError("Chunks list is empty.")
    
    return embedder(chunks)

def embed_query(query, embedder):
    if not query:
        return ValueError("Query is empty.")
    return embedder([query])