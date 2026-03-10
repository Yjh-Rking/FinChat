import logging
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from qdrant_client.conversions import common_types
from .types import Chunk

logger = logging.getLogger(__name__)


class QdrantStore:
    def __init__(self, collection, url=None, path=None):
        if path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(url=url)
        self.collection = collection

    def init_collection(self, vector_size: int = 1024, distance=models.Distance.COSINE):
        if not self.client.collection_exists(collection_name=self.collection):
            try:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=distance,
                    ),
                )
                logger.debug(
                    f"Qdrant collection: {self.collection} created successfully"
                )
            except Exception as e:
                logger.error(
                    f"Qdrant collection: {self.collection} creation failed: {str(e)}"
                )
                raise
        else:
            logger.debug(f"Qdrant collection: {self.collection} already exists")

    def upsert_chunks(self, chunks: list[Chunk], q_embedding: list[list[float]]):
        points = []
        for chunk, emb in zip(chunks, q_embedding):
            point = PointStruct(
                id=chunk.chunk_id,
                vector=emb,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                },
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection,
            points=points,
        )

    def get_points_by_embedding(
        self, q_embedding: list[float], topk: int = 10
    ) -> common_types.QueryResponse:

        result = self.client.query_points(
            collection_name=self.collection,
            query=q_embedding,
            search_params=models.SearchParams(hnsw_ef=128, exact=False),
            limit=topk,
            with_payload=True,  # 建议加上，方便拿业务字段
            with_vectors=False,  # 通常不返回向量本体，省带宽
        )
        return result
