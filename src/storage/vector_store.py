from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
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
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dense_dim),
        ]
        return CollectionSchema(fields)
    
    def _build_index(self, col):
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        col.create_index("sparse_vector", sparse_index)
        dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
        col.create_index("dense_vector", dense_index)
    
    def _init_collection(self):
        if utility.has_collection(self.col_name):
            col = Collection(self.col_name)
            col.load()
            if col.num_entities > 0:
                self.already_exists = True
                return col
            col.drop()

        self.already_exists = False
        col = Collection(self.col_name, self._build_schema(), consistency_level="Strong")
        self._build_index(col)
        col.load()
        return col
    

    def insert(self, chunks, chunk_embeddings):
        if not chunks or not chunk_embeddings:
            raise ValueError("Chunks and chunk_embeddings cannot be empty.")
        
        if len(chunks) != len(chunk_embeddings["dense"]):
            raise ValueError("Length of chunks and chunk_embeddings must match.")

        sparse_embeddings = chunk_embeddings["sparse"]

        for i in range(0, len(chunks), 50):
            end = min(i + 50, len(chunks))
            # pymilvus has edge cases with scipy sparse array batches (e.g. coo_array),
            # so convert to one sparse row object per entity.
            sparse_batch = [self._sparse_row(sparse_embeddings, row_idx) for row_idx in range(i, end)]
            self.col.insert(
                data = [
                    chunks[i:end],
                    sparse_batch,
                    chunk_embeddings["dense"][i:end],
                ],
                fields=["text", "sparse_vector", "dense_vector"]
            )
    
        self.col.flush()

    @staticmethod
    def _sparse_row(sparse_matrix, idx):
        if hasattr(sparse_matrix, "getrow"):
            return sparse_matrix.getrow(idx)
        if hasattr(sparse_matrix, "_getrow"):
            return sparse_matrix._getrow(idx)
        return sparse_matrix[idx]
