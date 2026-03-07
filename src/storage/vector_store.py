from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)


class MilvusVectorStore():

    def __init__(self, dense_dim, uri="./milvus.db", col_name="hybrid_demo"):
        self.dense_dim = dense_dim
        self.uri = uri
        self.col_name = col_name

        self._connect(self.uri)
        self.col = self._init_collection()
    
    def _connect(self, uri):
        connections.connect(uri=uri)
    
    def _build_schema(self):
        fields = [
            FieldSchema(
                name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
            ),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dense_dim),
        ]
        return CollectionSchema(fields)
    
    def _build_index(self):
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        self.col.create_index("sparse_vector", sparse_index)
        dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
        self.col.create_index("dense_vector", dense_index)
    
    def _init_collection(self):
        if utility.has_collection(self.col_name):
            Collection(self.col_name).drop()
        
        col = Collection(self.col_name, self._build_schema(), consistency_level="Strong")
        self._build_index()
        col.load()
        return col
    

    def insert(self, chunks, chunk_embeddings):
        if not chunks or not chunk_embeddings:
            raise ValueError("Chunks and chunk_embeddings cannot be empty.")
        
        if len(chunks) != len(chunk_embeddings["dense"]):
            raise ValueError("Length of chunks and chunk_embeddings must match.")
        
        for i in range(0, len(chunks), 50):
            self.col.insert(
                data = [
                    chunks[i : i+50],
                    chunk_embeddings["sparse"][i : i+50],
                    chunk_embeddings["dense"][i : i+50],
                ],
                fields=["text", "sparse_vector", "dense_vector"]
            )
    
        self.col.flush()
    
    def dense_search(self, query_dense_embedding, limit=10):
        search_params = {"metric_type": "IP", "params": {}}
        res = self.col.search(
            [query_dense_embedding],
            anns_field="dense_vector",
            limit=limit,
            output_fields=["text"],
            param=search_params,
        )[0]
        return [hit.get("text") for hit in res]

    def sparse_search(self, query_sparse_embedding, limit=10):
        search_params = {
            "metric_type": "IP",
            "params": {},
        }
        res = self.col.search(
            [query_sparse_embedding],
            anns_field="sparse_vector",
            limit=limit,
            output_fields=["text"],
            param=search_params,
        )[0]
        return [hit.get("text") for hit in res]

    def hybrid_search(self, query_dense_embedding, query_sparse_embedding, sparse_weight=1.0, dense_weight=1.0, limit=10):
        dense_search_params = {"metric_type": "IP", "params": {}}
        dense_req = AnnSearchRequest(
            [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
        )
        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_req = AnnSearchRequest(
            [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
        )
        rerank = WeightedRanker(sparse_weight, dense_weight)
        res = self.col.hybrid_search(
            [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["text"]
        )[0]
        return [hit.get("text") for hit in res]