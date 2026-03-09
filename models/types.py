from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Document:
    doc_id: str
    title: str
    source: str
    metadata: Dict[str, Any]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any]


@dataclass
class RetrievalDoc:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    source: str
    metadata: Dict[str, Any]
