import os
import sqlite3
from datetime import datetime

DB_PATH = "data/document_registry.db"

def init_registry():
    """Initialize the SQLite document registry table."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            source_file TEXT,
            ticker TEXT,
            doc_type TEXT,
            fiscal_period TEXT,
            file_hash TEXT,
            chunk_count INTEGER,
            embedding_model TEXT,
            ingested_at TEXT
        )
    """)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_registry()
    print("Document registry initialized successfully.")