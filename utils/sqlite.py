import sqlite3
import logging
from utils.config import config

logger = logging.getLogger(__name__)


def init_table():
    with sqlite3.connect(config.sqlite.path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS chunks(
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            text TEXT
        )""")
        conn.commit()
