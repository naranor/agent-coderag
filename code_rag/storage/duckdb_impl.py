import json
import logging
import asyncio
import re
import sys
import duckdb
from typing import List, Optional, Dict
from ..core.interfaces import IStorage, IEmbedder
from ..core.models import KnowledgeUnit, UnitKind, Relation, RelationType
from ..core.constants import (
    EMBEDDING_DIM,
    EMBEDDING_PROBE_TEXT,
    LOCAL_EMBEDDING_MODEL_ID,
)
from ..core.exceptions import StorageError, IntelligenceError

logger = logging.getLogger(__name__)

META_DIM_KEY = "embedding_dim"
META_MODEL_KEY = "embedding_model"

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


def float_vec_sql_type(dim: int) -> str:
    if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
        raise StorageError("embedding dimension must be a positive int")
    return f"FLOAT[{dim}]"


def _parse_positive_dim(raw: str) -> int:
    if raw is None or not str(raw).strip().isdigit():
        raise StorageError("Corrupt embedding metadata. Run rebuild.")
    value = int(str(raw).strip())
    if value < 1:
        raise StorageError("Corrupt embedding metadata. Run rebuild.")
    return value


def _table_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM information_schema.tables WHERE lower(table_name) = lower(?)",
        [name],
    ).fetchone()
    return row is not None


def _schema_vec_width(conn) -> int | None:
    rows = conn.execute("DESCRIBE unit_embeddings").fetchall()
    for row in rows:
        col_name, col_type = row[0], row[1]
        if col_name == "vec":
            match = re.search(r"FLOAT\[(\d+)\]", str(col_type), re.I)
            if match:
                return int(match.group(1))
    return None


def _meta_get(conn, key: str) -> str | None:
    if not _table_exists(conn, "index_meta"):
        return None
    row = conn.execute("SELECT value FROM index_meta WHERE key = ?", [key]).fetchone()
    return row[0] if row else None


def _meta_set(conn, key: str, value: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
        [key, value],
    )


async def _probe_or_bind_dimension(embedder: IEmbedder) -> int:
    try:
        already = embedder.dimension
        embedder.bind_dimension(already)
        return already
    except Exception as exc:
        rows = await embedder.aembed([EMBEDDING_PROBE_TEXT])
        if not rows or not rows[0]:
            raise IntelligenceError("Embedding count mismatch") from exc
        n = len(rows[0])
        embedder.bind_dimension(n)
        return n


def _bind_embedder_to_target(embedder: IEmbedder, target: int) -> int:
    try:
        already = embedder.dimension
    except IntelligenceError:
        already = None
    if already is not None and already != target:
        raise StorageError(
            f"Embedding dimension mismatch (index={target}, embedder={already}). Run rebuild."
        )
    embedder.bind_dimension(target)
    return target


async def bind_embedder_dimension(conn, embedder: IEmbedder, *, wiped: bool) -> int:
    table_exists = (not wiped) and _table_exists(conn, "unit_embeddings")
    meta_dim_raw = _meta_get(conn, META_DIM_KEY) if table_exists else None
    meta_model = _meta_get(conn, META_MODEL_KEY) if table_exists else None
    schema_n = _schema_vec_width(conn) if table_exists else None

    if table_exists and meta_dim_raw is not None and meta_model is not None:
        meta_n = _parse_positive_dim(meta_dim_raw)
        if schema_n is not None and schema_n != meta_n:
            raise StorageError("Corrupt embedding metadata. Run rebuild.")
        if meta_model == embedder.model_id:
            target = schema_n if schema_n is not None else meta_n
            return _bind_embedder_to_target(embedder, target)
        return await _probe_or_bind_dimension(embedder)

    if table_exists and (meta_dim_raw is None or meta_model is None):
        inferred = schema_n if schema_n is not None else EMBEDDING_DIM
        _meta_set(conn, META_DIM_KEY, str(inferred))
        _meta_set(conn, META_MODEL_KEY, LOCAL_EMBEDDING_MODEL_ID)
        if embedder.model_id == LOCAL_EMBEDDING_MODEL_ID:
            return _bind_embedder_to_target(embedder, inferred)
        return await _probe_or_bind_dimension(embedder)

    return await _probe_or_bind_dimension(embedder)


def _ensure_base_tables(conn) -> None:
    conn.execute("INSTALL vss;")
    conn.execute("LOAD vss;")
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
    conn.execute(f"CREATE TABLE IF NOT EXISTS units ({cols_definition})")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS relations (
            from_id VARCHAR,
            to_id VARCHAR,
            type VARCHAR,
            PRIMARY KEY (from_id, to_id, type)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dependencies (
            name VARCHAR PRIMARY KEY,
            path VARCHAR
        )
        """
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS index_meta (key VARCHAR PRIMARY KEY, value VARCHAR)"
    )


class DuckDBStorage(IStorage):
    """
    DuckDB-based storage with Vector Similarity Search (VSS) capabilities.
    """

    def __init__(self, conn, embedder: IEmbedder, *, db_path: str):
        self.db_path = db_path
        self._embedder = embedder
        self.conn = conn
        self._embedding_model_dirty = False
        self._embeddings_bound = False
        self._pending_wipe = False
        self._bind_lock = asyncio.Lock()

    @property
    def embedder(self) -> IEmbedder:
        return self._embedder

    @property
    def embedding_model_dirty(self) -> bool:
        return self._embedding_model_dirty

    def _refresh_dirty_from_meta(self) -> None:
        if self._pending_wipe or not _table_exists(self.conn, "unit_embeddings"):
            self._embedding_model_dirty = False
            return
        meta_model = _meta_get(self.conn, META_MODEL_KEY)
        if meta_model is None:
            self._embedding_model_dirty = (
                self.embedder.model_id != LOCAL_EMBEDDING_MODEL_ID
            )
            return
        self._embedding_model_dirty = meta_model != self.embedder.model_id

    async def ensure_embeddings_bound(self) -> None:
        if self._embeddings_bound:
            return
        async with self._bind_lock:
            if self._embeddings_bound:
                return
            await self._bind_embeddings()
            self._embeddings_bound = True
            self._pending_wipe = False

    async def _bind_embeddings(self) -> None:
        dim = await bind_embedder_dimension(
            self.conn, self._embedder, wiped=self._pending_wipe
        )
        schema_n = (
            _schema_vec_width(self.conn)
            if _table_exists(self.conn, "unit_embeddings")
            else None
        )
        meta_model = _meta_get(self.conn, META_MODEL_KEY)
        if schema_n is not None and schema_n != dim:
            raise StorageError(
                f"Embedding dimension mismatch (index={schema_n}, embedder={dim}). Run rebuild."
            )
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS unit_embeddings (
                id VARCHAR PRIMARY KEY,
                vec {float_vec_sql_type(dim)}
            )
            """
        )
        _meta_set(self.conn, META_DIM_KEY, str(dim))
        if meta_model is None:
            _meta_set(self.conn, META_MODEL_KEY, self._embedder.model_id)

    @classmethod
    async def open(cls, path: str, embedder: IEmbedder, *, wipe: bool = False):
        if embedder is None:
            raise StorageError("embedder is required")
        conn = duckdb.connect(path)
        try:
            _ensure_base_tables(conn)
            if wipe:
                conn.execute("DROP TABLE IF EXISTS unit_embeddings")
                conn.execute(
                    "DELETE FROM index_meta WHERE key IN (?, ?)",
                    [META_DIM_KEY, META_MODEL_KEY],
                )
            storage = cls(conn, embedder, db_path=path)
            storage._pending_wipe = wipe
            storage._refresh_dirty_from_meta()
            return storage
        except Exception:
            try:
                conn.close()
            except Exception:  # nosec B110
                pass
            raise

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

    async def upsert_unit(
        self, unit: KnowledgeUnit, vector: Optional[list[float]] = None
    ):
        if vector is not None:
            await self.ensure_embeddings_bound()
        placeholders = ", ".join(["?"] * len(UNIT_COLUMNS))
        cols = ", ".join(UNIT_COLUMNS)
        await asyncio.to_thread(
            self.conn.execute,
            f"INSERT OR REPLACE INTO units ({cols}) VALUES ({placeholders})",
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
        if vector is not None:
            await asyncio.to_thread(
                self.conn.execute,
                "INSERT OR REPLACE INTO unit_embeddings (id, vec) VALUES (?, ?)",
                [unit.id, vector],
            )
        for rel in unit.relations:
            await self.upsert_relation(rel)

    async def has_embedding(self, unit_id: str) -> bool:
        await self.ensure_embeddings_bound()
        row = await asyncio.to_thread(
            lambda: self.conn.execute(
                "SELECT 1 FROM unit_embeddings WHERE id = ?", [unit_id]
            ).fetchone()
        )
        return row is not None

    async def list_units(self) -> List[KnowledgeUnit]:
        cols = ", ".join(UNIT_COLUMNS)
        rows = await asyncio.to_thread(
            lambda: self.conn.execute(f"SELECT {cols} FROM units").fetchall()  # nosec
        )
        return [self._map_row_to_unit(row) for row in rows]

    async def mark_embedding_model_synced(self) -> None:
        _meta_set(self.conn, META_MODEL_KEY, self.embedder.model_id)
        self._embedding_model_dirty = False

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

    async def search_units(  # pylint: disable=too-many-locals
        self, query: str, limit: int = 5
    ) -> List[KnowledgeUnit]:
        await self.ensure_embeddings_bound()
        cols = ", ".join([f"u.{c}" for c in UNIT_COLUMNS])
        if self.embedding_model_dirty:
            print(
                "Warning: embedding model id changed; search quality may be degraded. Run rebuild or sync --all.",
                file=sys.stderr,
            )
        dim = self.embedder.dimension
        query_vec = (await self.embedder.aembed([query]))[0]
        res = await asyncio.to_thread(
            lambda: self.conn.execute(
                f"""
                SELECT {cols}, array_distance(e.vec, ?::{float_vec_sql_type(dim)}) as dist
                FROM units u
                JOIN unit_embeddings e ON u.id = e.id
                ORDER BY dist ASC
                LIMIT ?
                """,  # nosec
                [query_vec, limit],
            ).fetchall()
        )
        units = [self._map_row_to_unit(row) for row in res]
        if not units:
            return []

        unit_ids = [u.id for u in units]
        rels_res = await asyncio.to_thread(
            lambda: self.conn.execute(
                "SELECT from_id, to_id, type FROM relations WHERE from_id IN (SELECT unnest(?))",
                [unit_ids],
            ).fetchall()
        )
        rels_by_id: Dict[str, List[Relation]] = {}
        for r in rels_res:
            from_id, to_id, r_type = r
            rels_by_id.setdefault(from_id, []).append(
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

    async def get_relations(
        self, unit_id: str, direction: str = "out"
    ) -> List[Relation]:
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
                    "SELECT from_id, to_id, type FROM relations WHERE to_id = ?",
                    [unit_id],
                ).fetchall()
            )

        return [
            Relation(from_id=r[0], to_id=r[1], type=RelationType(r[2])) for r in res
        ]

    async def delete_stale_units(
        self, file_path: str, current_unit_ids: List[str]
    ) -> None:
        """Removes units that are no longer present in the given file."""
        await self.ensure_embeddings_bound()
        await asyncio.to_thread(
            lambda: self.conn.execute(
                "DELETE FROM unit_embeddings WHERE id IN (SELECT id FROM units WHERE path = ? AND id NOT IN (SELECT unnest(?)))",
                [file_path, current_unit_ids],
            )
        )
        await asyncio.to_thread(
            lambda: self.conn.execute(
                "DELETE FROM relations WHERE from_id IN (SELECT id FROM units WHERE path = ? AND id NOT IN (SELECT unnest(?)))",
                [file_path, current_unit_ids],
            )
        )
        await asyncio.to_thread(
            lambda: self.conn.execute(
                "DELETE FROM units WHERE path = ? AND id NOT IN (SELECT unnest(?))",
                [file_path, current_unit_ids],
            )
        )
        logger.debug("Cleaned up stale units for %s", file_path)

    async def close(self) -> None:
        if self.embedder is not None:
            await self.embedder.close()
        if self.conn:
            await asyncio.to_thread(self.conn.close)
            logger.info("Storage connection closed.")

    def _map_row_to_unit(self, row) -> KnowledgeUnit:
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
