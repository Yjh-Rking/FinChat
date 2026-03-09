from pathlib import Path
from typing import List, Tuple
from models.types import Document


def load_markdown(folder_path: str) -> List[Tuple[Document, str]]:
    folder = Path(folder_path)
    results = []

    for md_file in folder.rglob("*.md"):  # rglob 递归查找；用 glob 则不递归
        text = md_file.read_text(encoding="utf-8")
        doc_id = md_file.stem
        doc = Document(
            doc_id=doc_id,
            title=md_file.name,
            source=str(md_file.resolve()),  # 绝对路径
            metadata={},
        )
        results.append((doc, text))
    return results
