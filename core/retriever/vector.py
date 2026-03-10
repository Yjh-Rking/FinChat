import logging
from core.models import RetrievalDoc, SQLiteStore, QdrantStore

logger = logging.getLogger(__name__)


class VectorRetriever:
    def __init__(self, sqlite_store: SQLiteStore):
        self.sqlite = sqlite_store

    def qdrant_retrieve(
        self, qdrant: QdrantStore, q_embedding: list[float], topk: int = 10
    ) -> list[RetrievalDoc]:
        points = qdrant.get_points_by_embedding(q_embedding=q_embedding, topk=topk)
        retrieval_docs = []

        for point in points.points:
            payload = point.payload if point.payload else {}
            chunk_id = payload["chunk_id"]
            chunk = self.sqlite.get_chunk_by_id(chunk_id)
            if chunk:
                retrieval_docs.append(
                    RetrievalDoc(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        text=chunk.text,
                        score=point.score,
                        source="qdrant_vector",
                        metadata=chunk.metadata,
                    )
                )
        return retrieval_docs
