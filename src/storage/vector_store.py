from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# 全局连接追踪：避免对同一 URI 重复创建 gRPC channel
_connected_uris: set[str] = set()


class MilvusVectorStore():

    def __init__(self, dense_dim, uri="./milvus.db", col_name="hybrid_demo"):
        self.dense_dim = dense_dim
        self.uri = uri
        self.col_name = col_name

        self._connect(self.uri)
        self.col = self._init_collection()

    def _connect(self, uri):
        if uri not in _connected_uris:
            connections.connect(uri=uri)
            _connected_uris.add(uri)

    def release(self):
        """Release the collection from memory."""
        try:
            self.col.release()
        except Exception:
            pass
    
    def _build_schema(self):
        fields = [
            FieldSchema(
                name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
            ),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="parent_text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=512),
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
            # 先检查 entity 数量再决定是否 load，避免空 collection 白白占内存
            if col.num_entities > 0:
                col.load()
                self.already_exists = True
                return col
            col.drop()

        self.already_exists = False
        col = Collection(self.col_name, self._build_schema(), consistency_level="Strong")
        self._build_index(col)
        col.load()
        return col
    

    def insert(self, chunks, chunk_embeddings, parent_chunks=None, source_files=None, sections=None):
        if not chunks or not chunk_embeddings:
            raise ValueError("Chunks and chunk_embeddings cannot be empty.")

        if len(chunks) != len(chunk_embeddings["dense"]):
            raise ValueError("Length of chunks and chunk_embeddings must match.")

        # parent_chunks falls back to chunks when not provided (old schema compat)
        if parent_chunks is None:
            parent_chunks = chunks
        if source_files is None:
            source_files = [""] * len(chunks)
        if sections is None:
            sections = [""] * len(chunks)

        sparse_embeddings = chunk_embeddings["sparse"]
        schema_field_names = {f.name for f in self.col.schema.fields}
        has_metadata = "source_file" in schema_field_names

        for i in range(0, len(chunks), 50):
            end = min(i + 50, len(chunks))
            sparse_batch = [self._sparse_row(sparse_embeddings, row_idx) for row_idx in range(i, end)]
            if has_metadata:
                self.col.insert(
                    data=[
                        chunks[i:end],
                        parent_chunks[i:end],
                        source_files[i:end],
                        sections[i:end],
                        sparse_batch,
                        chunk_embeddings["dense"][i:end],
                    ],
                    fields=["text", "parent_text", "source_file", "section", "sparse_vector", "dense_vector"]
                )
            else:
                self.col.insert(
                    data=[
                        chunks[i:end],
                        parent_chunks[i:end],
                        sparse_batch,
                        chunk_embeddings["dense"][i:end],
                    ],
                    fields=["text", "parent_text", "sparse_vector", "dense_vector"]
                )

        self.col.flush()

    @staticmethod
    def _sparse_row(sparse_matrix, idx):
        if hasattr(sparse_matrix, "getrow"):
            return sparse_matrix.getrow(idx)
        if hasattr(sparse_matrix, "_getrow"):
            return sparse_matrix._getrow(idx)
        return sparse_matrix[idx]
