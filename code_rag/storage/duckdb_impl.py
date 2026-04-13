import json
import logging
import duckdb
import numpy as np
from typing import List, Optional
from ..core.interfaces import IStorage
from ..core.models import KnowledgeUnit, Relation, UnitKind
from ..intelligence.embedder import Embedder

logger = logging.getLogger(__name__)

class DuckDBStorage(IStorage):
    """
    DuckDB implementation of knowledge storage with VSS support.
    """
    
    def __init__(self, db_path: str, embedder: Optional[Embedder] = None):
        self.conn = duckdb.connect(db_path)
        self.embedder = embedder
        self._init_db()

    def _init_db(self):
        """Initializes tables and VSS extension."""
        try:
            self.conn.execute("INSTALL vss; LOAD vss;")
        except Exception as e:
            logger.warning(f"Failed to load VSS extension: {e}. Vector search will be limited.")

        # Table for units
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS units (
                id VARCHAR PRIMARY KEY,
                kind VARCHAR,
                name VARCHAR,
                path VARCHAR,
                signature VARCHAR,
                summary TEXT,
                code_hash VARCHAR,
                tags VARCHAR[],
                metadata JSON
            )
        """)

        # Table for relations
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                from_id VARCHAR,
                to_id VARCHAR,
                type VARCHAR,
                PRIMARY KEY (from_id, to_id, type)
            )
        """)

        # Table for embeddings (VSS)
        # Note: 384 is the dimension for MiniLM models
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS unit_embeddings (
                id VARCHAR PRIMARY KEY,
                vec FLOAT[384]
            )
        """)

    async def upsert_unit(self, unit: KnowledgeUnit):
        """Inserts or updates a knowledge unit and its embedding."""
        # 1. Upsert metadata
        self.conn.execute("""
            INSERT OR REPLACE INTO units (id, kind, name, path, signature, summary, code_hash, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            unit.id, unit.kind.value, unit.name, unit.path, 
            unit.signature, unit.summary, unit.code_hash, 
            unit.tags, json.dumps(unit.metadata)
        ])

        # 2. Update embedding
        if self.embedder:
            # Fallback to name/signature if summary is missing
            text_to_embed = unit.summary or f"{unit.kind.value} {unit.name} {unit.signature or ''}"
            vec = self.embedder.embed([text_to_embed])[0]
            self.conn.execute("""
                INSERT OR REPLACE INTO unit_embeddings (id, vec)
                VALUES (?, ?)
            """, [unit.id, vec.tolist()])

    async def get_unit(self, unit_id: str) -> Optional[KnowledgeUnit]:
        """Retrieves a unit by its unique ID."""
        res = self.conn.execute("SELECT * FROM units WHERE id = ?", [unit_id]).fetchone()
        if not res:
            return None
        return self._map_row_to_unit(res)

    async def search_units(self, query: str, limit: int = 5) -> List[KnowledgeUnit]:
        """Hybrid search using VSS and FTS."""
        if not self.embedder:
            # Fallback to basic text search if no embedder
            res = self.conn.execute("""
                SELECT * FROM units 
                WHERE name ILIKE ? OR summary ILIKE ? 
                LIMIT ?
            """, [f"%{query}%", f"%{query}%", limit]).fetchall()
        else:
            # Vector similarity search
            query_vec = self.embedder.embed([query])[0].tolist()
            res = self.conn.execute("""
                SELECT u.*, 
                       array_distance(ue.vec, ?::FLOAT[384]) as distance
                FROM units u
                JOIN unit_embeddings ue ON u.id = ue.id
                ORDER BY distance ASC
                LIMIT ?
            """, [query_vec, limit]).fetchall()

        return [self._map_row_to_unit(row) for row in res]

    def _map_row_to_unit(self, row) -> KnowledgeUnit:
        return KnowledgeUnit(
            id=row[0],
            kind=UnitKind(row[1]),
            name=row[2],
            path=row[3],
            signature=row[4],
            summary=row[5],
            code_hash=row[6],
            tags=row[7] if row[7] else [],
            metadata=json.loads(row[8]) if row[8] else {}
        )
