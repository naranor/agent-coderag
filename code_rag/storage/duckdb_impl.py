import json
import logging
import duckdb
from typing import List, Optional
from ..core.interfaces import IStorage
from ..core.models import KnowledgeUnit, UnitKind

logger = logging.getLogger(__name__)

class DuckDBStorage(IStorage):
    """
    DuckDB-based storage with Vector Similarity Search (VSS) capabilities.
    """
    
    def __init__(self, db_path: str, embedder=None):
        self.db_path = db_path
        self.embedder = embedder
        self.conn = duckdb.connect(self.db_path)
        self._setup_db()

    def _setup_db(self):
        """Initializes tables and extensions."""
        self.conn.execute("INSTALL vss;")
        self.conn.execute("LOAD vss;")
        
        # Metadata table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS units (
                id VARCHAR PRIMARY KEY,
                kind VARCHAR,
                name VARCHAR,
                path VARCHAR,
                signature VARCHAR,
                summary VARCHAR,
                code_hash VARCHAR,
                tags VARCHAR[],
                metadata JSON
            )
        """)
        
        # Vector table (MiniLM dimension is 384)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS unit_embeddings (
                id VARCHAR PRIMARY KEY,
                vec FLOAT[384]
            )
        """)
        logger.info("Storage initialized at %s", self.db_path)

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
            query_vec = self.embedder.embed([query])[0]
            res = self.conn.execute("""
                SELECT u.*, array_distance(e.vec, ?::FLOAT[384]) as dist
                FROM units u
                JOIN unit_embeddings e ON u.id = e.id
                ORDER BY dist ASC
                LIMIT ?
            """, [query_vec.tolist(), limit]).fetchall()
            
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
