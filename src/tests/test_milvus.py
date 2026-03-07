from storage.vector_store import MilvusVectorStore
from rag.embedder import load_bge_m3_embedder
from rag.chunker import chunk_text
from rag.parser import build_llama_parser, parse_documents, merge_documents




query = '什么是以人为本的座舱'
embedder = load_bge_m3_embedder()
query_embedding = embedder.embed_query(query)

parser = build_llama_parser()
documents = parse_documents("data", parser)
all_doc = merge_documents(documents)
chunks = chunk_text(all_doc, max_chunk_size=300, hard_max_length=512)
chunks_embeddings = embedder.embed_texts(chunks)
store = MilvusVectorStore(collection_name="test_collection", dim=1024)
store.insert(chunks, chunks_embeddings)

query_dense = query_embedding["dense"][0]
query_sparse = query_embedding["sparse"]._getrow(0)

dense_results = store.dense_search(query_dense)
sparse_results = store.sparse_search(query_sparse)
hybrid_results = store.hybrid_search(store.col, query_dense, query_sparse,
                                     sparse_weight=0.7, dense_weight=1.0)

print(dense_results[:3])
print(sparse_results[:3])
print(hybrid_results[:3])