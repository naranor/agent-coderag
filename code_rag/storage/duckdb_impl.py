import json
import logging
import asyncio
import duckdb
from typing import List, Optional, Dict
from ..core.interfaces import IStorage
from ..core.models import KnowledgeUnit, UnitKind, Relation, RelationType
from ..core.constants import EMBEDDING_DIM

logger = logging.getLogger(__name__)

# Canonical list of columns for units table to ensure robust mapping
UNIT_COLUMNS = [
    "id",
    "kind",
    "name",
    "path",
    "signature",
    "docstring",
    "summary",
    "code_hash",
    "tags",
    "metadata",
]


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
        cols_definition = ", ".join(
            [
                "id VARCHAR PRIMARY KEY",
                "kind VARCHAR",
                "name VARCHAR",
                "path VARCHAR",
                "signature VARCHAR",
                "docstring VARCHAR",
                "summary VARCHAR",
                "code_hash VARCHAR",
                "tags VARCHAR[]",
                "metadata JSON",
            ]
        )
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS units ({cols_definition})")

        # Vector table (MiniLM dimension is EMBEDDING_DIM)
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS unit_embeddings (
                id VARCHAR PRIMARY KEY,
                vec FLOAT[{EMBEDDING_DIM}]
            )
        """
        )

        # Relations table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS relations (
                from_id VARCHAR,
                to_id VARCHAR,
                type VARCHAR,
                PRIMARY KEY (from_id, to_id, type)
            )
        """
        )

        # Dependencies table for API Discovery
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dependencies (
                name VARCHAR PRIMARY KEY,
                path VARCHAR
            )
        """
        )
        logger.info("Storage initialized at %s", self.db_path)

    async def set_dependency_path(self, lib_name: str, path: str) -> None:
        """Caches the absolute path to a library's JAR/binary."""
        await asyncio.to_thread(
            self.conn.execute,
            "INSERT OR REPLACE INTO dependencies (name, path) VALUES (?, ?)",
            [lib_name, path],
        )

    async def get_dependency_path(self, lib_name: str) -> Optional[str]:
        """Retrieves the cached path for a library."""
        res = await asyncio.to_thread(
            lambda: self.conn.execute(
                "SELECT path FROM dependencies WHERE name = ?", [lib_name]
            ).fetchone()
        )
        return res[0] if res else None

    async def upsert_unit(self, unit: KnowledgeUnit):
        """Inserts or updates a knowledge unit and its embedding."""
        # 1. Upsert metadata
        placeholders = ", ".join(["?"] * len(UNIT_COLUMNS))
        cols = ", ".join(UNIT_COLUMNS)
        await asyncio.to_thread(
            self.conn.execute,
            f"""
            INSERT OR REPLACE INTO units ({cols})
            VALUES ({placeholders})
        """,
            [
                unit.id,
                unit.kind.value,
                unit.name,
                unit.path,
                unit.signature,
                unit.docstring,
                unit.summary,
                unit.code_hash,
                unit.tags,
                json.dumps(unit.metadata),
            ],
        )

        # 2. Update embedding
        if self.embedder:
            # Fallback to name/signature if summary is missing
            text_to_embed = (
                unit.summary
                or f"{unit.kind.value} {unit.name} {unit.signature or ''} {unit.docstring or ''}"
            )
            vec = self.embedder.embed([text_to_embed])[0]
            await asyncio.to_thread(
                self.conn.execute,
                """
                INSERT OR REPLACE INTO unit_embeddings (id, vec)
                VALUES (?, ?)
            """,
                [unit.id, vec.tolist()],
            )

        # 3. Upsert relations
        for rel in unit.relations:
            await self.upsert_relation(rel)

    async def get_unit(self, unit_id: str) -> Optional[KnowledgeUnit]:
        """Retrieves a unit by its unique ID."""
        cols = ", ".join(UNIT_COLUMNS)
        res = await asyncio.to_thread(
            lambda: self.conn.execute(
                f"SELECT {cols} FROM units WHERE id = ?",  # nosec
                [unit_id],
            ).fetchone()
        )
        if not res:
            return None
        unit = self._map_row_to_unit(res)
        unit.relations = await self.get_relations(unit_id, direction="out")
        return unit

    async def search_units(self, query: str, limit: int = 5) -> List[KnowledgeUnit]:
        """Hybrid search using VSS and FTS."""
        cols = ", ".join([f"u.{c}" for c in UNIT_COLUMNS])
        if not self.embedder:
            # Fallback to basic text search if no embedder
            res = await asyncio.to_thread(
                lambda: self.conn.execute(
                    f"""
                    SELECT {cols} FROM units u
                    WHERE u.name ILIKE ? OR u.summary ILIKE ? OR u.docstring ILIKE ?
                    LIMIT ?
                """,  # nosec
                    [f"%{query}%", f"%{query}%", f"%{query}%", limit],
                ).fetchall()
            )
        else:
            query_vec = self.embedder.embed([query])[0]
            res = await asyncio.to_thread(
                lambda: self.conn.execute(
                    f"""
                    SELECT {cols}, array_distance(e.vec, ?::FLOAT[{EMBEDDING_DIM}]) as dist
                    FROM units u
                    JOIN unit_embeddings e ON u.id = e.id
                    ORDER BY dist ASC
                    LIMIT ?
                """,  # nosec
                    [query_vec.tolist(), limit],
                ).fetchall()
            )

        units = [self._map_row_to_unit(row) for row in res]
        if not units:
            return []

        # Optimization: Fix N+1 problem by batch-fetching all relations
        unit_ids = [u.id for u in units]
        # DuckDB handles list parameters natively for IN clause
        rels_res = await asyncio.to_thread(
            lambda: self.conn.execute(
                "SELECT from_id, to_id, type FROM relations WHERE from_id IN (?)",
                [unit_ids],
            ).fetchall()
        )

        # Map relations to units
        rels_by_id: Dict[str, List[Relation]] = {}
        for r in rels_res:
            from_id, to_id, r_type = r
            if from_id not in rels_by_id:
                rels_by_id[from_id] = []
            rels_by_id[from_id].append(
                Relation(from_id=from_id, to_id=to_id, type=RelationType(r_type))
            )

        for unit in units:
            unit.relations = rels_by_id.get(unit.id, [])

        return units

    async def upsert_relation(self, relation: Relation):
        """Inserts or updates a relation between units."""
        await asyncio.to_thread(
            self.conn.execute,
            """
            INSERT OR REPLACE INTO relations (from_id, to_id, type)
            VALUES (?, ?, ?)
        """,
            [relation.from_id, relation.to_id, relation.type.value],
        )

    async def get_relations(self, unit_id: str, direction: str = "out") -> List[Relation]:
        """Retrieves relations for a unit."""
        if direction == "out":
            res = await asyncio.to_thread(
                lambda: self.conn.execute(
                    "SELECT from_id, to_id, type FROM relations WHERE from_id = ?",
                    [unit_id],
                ).fetchall()
            )
        else:
            res = await asyncio.to_thread(
                lambda: self.conn.execute(
                    "SELECT from_id, to_id, type FROM relations WHERE to_id = ?", [unit_id]
                ).fetchall()
            )

        return [Relation(from_id=r[0], to_id=r[1], type=RelationType(r[2])) for r in res]

    async def delete_stale_units(self, file_path: str, current_unit_ids: List[str]) -> None:
        """Removes units that are no longer present in the given file."""
        # 1. Delete embeddings
        await asyncio.to_thread(
            lambda: self.conn.execute(
                "DELETE FROM unit_embeddings WHERE id IN (SELECT id FROM units WHERE path = ? AND id NOT IN (SELECT unnest(?)))",
                [file_path, current_unit_ids],
            )
        )
        # 2. Delete relations (cascading cleanup usually handled by logic if not DB constraints)
        await asyncio.to_thread(
            lambda: self.conn.execute(
                "DELETE FROM relations WHERE from_id IN (SELECT id FROM units WHERE path = ? AND id NOT IN (SELECT unnest(?)))",
                [file_path, current_unit_ids],
            )
        )
        # 3. Delete units
        await asyncio.to_thread(
            lambda: self.conn.execute(
                "DELETE FROM units WHERE path = ? AND id NOT IN (SELECT unnest(?))",
                [file_path, current_unit_ids],
            )
        )
        logger.debug("Cleaned up stale units for %s", file_path)

    async def close(self) -> None:
        """Closes the DuckDB connection."""
        if self.conn:
            await asyncio.to_thread(self.conn.close)
            logger.info("Storage connection closed.")

    def _map_row_to_unit(self, row) -> KnowledgeUnit:
        # Map by index based on our explicit column list
        data = dict(zip(UNIT_COLUMNS, row))
        return KnowledgeUnit(
            id=data["id"],
            kind=UnitKind(data["kind"]),
            name=data["name"],
            path=data["path"],
            signature=data["signature"],
            docstring=data["docstring"],
            summary=data["summary"],
            code_hash=data["code_hash"],
            tags=data["tags"] if data["tags"] else [],
            metadata=json.loads(data["metadata"]) if data["metadata"] else {},
        )
