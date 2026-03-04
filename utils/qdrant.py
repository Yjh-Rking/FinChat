import logging
from qdrant_client import models

logger = logging.getLogger(__name__)


def create_collection(client, collection_name="text_collection", vector_size=1024):
    try:
        result = client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,  # 向量维度 根据使用的模型确定
                distance=models.Distance.COSINE,  # 距离度量方式: 余弦相似度
            ),
        )
        logger.info(f"集合 {collection_name} 创建成功")
        return result
    except Exception as e:
        logger.error(f"创建集合 {collection_name} 失败: {str(e)}")
        raise


def insert_single_point(client, collection_name, point_id, vector, payload):
    try:
        point = models.PointStruct(id=point_id, vector=vector, payload=payload)
        result = client.upsert(collection_name=collection_name, points=[point])
        logger.info(f"点数据 {point_id} 插入成功")
        return result
    except Exception as e:
        logger.error(f"插入点数据失败: {str(e)}")
        raise
