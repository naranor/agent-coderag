import json
import logging
import asyncio
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import List, Optional, Dict
from ..core.interfaces import IStorage, IEmbedder
from ..core.models import KnowledgeUnit, UnitKind, Relation, RelationType
from ..core.constants import (
    EMBEDDING_DIM,
    EMBEDDING_PROBE_TEXT,
    LOCAL_EMBEDDING_MODEL_ID,
)
from ..core.error_codes import ErrorCode
from ..core.exceptions import StorageError, IntelligenceError

logger = logging.getLogger(__name__)

EMBEDDINGS_MISSING_MSG = (
    "Embeddings table is missing; run sync (CodeRAG.sync) before search."
)


class AccessMode(str, Enum):
    """Whether a connection may write. Not part of the public package API."""

    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


META_DIM_KEY = "embedding_dim"
META_MODEL_KEY = "embedding_model"
PATHS_MIGRATED_KEY = "paths_migrated"

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
        raise StorageError(
            "Corrupt embedding metadata. Run rebuild.", code=ErrorCode.STORAGE_CORRUPT
        )
    value = int(str(raw).strip())
    if value < 1:
        raise StorageError(
            "Corrupt embedding metadata. Run rebuild.", code=ErrorCode.STORAGE_CORRUPT
        )
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


def _rewrite_path_prefix(conn, old: str, new: str, *, has_embeddings: bool) -> None:
    prefix = old + ":"
    start = len(old) + 1  # 1-based index of the colon; remainder includes ':'
    conn.execute(
        """
        UPDATE units
        SET path = ?,
            id = CASE
                WHEN starts_with(id, ?) THEN ? || substring(id, ?)
                ELSE id
            END
        WHERE path = ?
        """,
        [new, prefix, new, start, old],
    )
    if has_embeddings:
        conn.execute(
            """
            UPDATE unit_embeddings
            SET id = ? || substring(id, ?)
            WHERE starts_with(id, ?)
            """,
            [new, start, prefix],
        )
    conn.execute(
        """
        UPDATE relations
        SET from_id = CASE
                WHEN starts_with(from_id, ?) THEN ? || substring(from_id, ?)
                ELSE from_id
            END,
            to_id = CASE
                WHEN starts_with(to_id, ?) THEN ? || substring(to_id, ?)
                ELSE to_id
            END
        WHERE starts_with(from_id, ?) OR starts_with(to_id, ?)
        """,
        [prefix, new, start, prefix, new, start, prefix, prefix],
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
            f"Embedding dimension mismatch (index={target}, embedder={already}). Run rebuild.",
            code=ErrorCode.EMBEDDING_MISMATCH,
        )
    embedder.bind_dimension(target)
    return target


async def _offload(fn, executor: Optional[ThreadPoolExecutor]):
    """Run ``fn`` on ``executor`` if given, else inline (for bare-conn callers)."""
    if executor is None:
        return fn()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, fn)


async def bind_embedder_dimension(
    conn,
    embedder: IEmbedder,
    *,
    wiped: bool,
    executor: Optional[ThreadPoolExecutor] = None,
    read_only: bool = False,
) -> int:
    """Determine/bind the embedder's vector dimension against index metadata.

    All ``conn`` reads/writes are offloaded to ``executor`` when provided
    (the connection's dedicated thread); only ``embedder.aembed`` awaits
    (via ``_probe_or_bind_dimension``) run on the event loop.
    Incomplete-meta inference writes ``index_meta`` only when not ``read_only``.
    """

    def _read_state():
        table_exists = (not wiped) and _table_exists(conn, "unit_embeddings")
        meta_dim_raw = _meta_get(conn, META_DIM_KEY) if table_exists else None
        meta_model = _meta_get(conn, META_MODEL_KEY) if table_exists else None
        schema_n = _schema_vec_width(conn) if table_exists else None
        return table_exists, meta_dim_raw, meta_model, schema_n

    table_exists, meta_dim_raw, meta_model, schema_n = await _offload(
        _read_state, executor
    )

    if table_exists and meta_dim_raw is not None and meta_model is not None:
        meta_n = _parse_positive_dim(meta_dim_raw)
        if schema_n is not None and schema_n != meta_n:
            raise StorageError(
                "Corrupt embedding metadata. Run rebuild.",
                code=ErrorCode.STORAGE_CORRUPT,
            )
        if meta_model == embedder.model_id:
            target = schema_n if schema_n is not None else meta_n
            return _bind_embedder_to_target(embedder, target)
        return await _probe_or_bind_dimension(embedder)

    if table_exists and (meta_dim_raw is None or meta_model is None):
        inferred = schema_n if schema_n is not None else EMBEDDING_DIM

        def _write_inferred():
            _meta_set(conn, META_DIM_KEY, str(inferred))
            _meta_set(conn, META_MODEL_KEY, LOCAL_EMBEDDING_MODEL_ID)

        if not read_only:
            await _offload(_write_inferred, executor)
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


def apply_rw_schema(conn, *, wipe: bool) -> None:
    """Create base tables and optionally wipe embeddings. Read-write only."""
    _ensure_base_tables(conn)
    if wipe:
        conn.execute("DROP TABLE IF EXISTS unit_embeddings")
        conn.execute(
            "DELETE FROM index_meta WHERE key IN (?, ?)",
            [META_DIM_KEY, META_MODEL_KEY],
        )


class DuckDBStorage(IStorage):
    """
    DuckDB-based storage with Vector Similarity Search (VSS) capabilities.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        conn,
        embedder: Optional[IEmbedder],
        *,
        db_path: str,
        mode: AccessMode = AccessMode.READ_WRITE,
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        self.db_path = db_path
        self._embedder = embedder
        self.conn = conn
        self.mode = mode
        self._executor = (
            executor if executor is not None else ThreadPoolExecutor(max_workers=1)
        )
        self._embedding_model_dirty = False
        self._embeddings_bound = False
        self._pending_wipe = False
        self._closed = False
        self._conn_lock = asyncio.Lock()

    @property
    def embedder(self) -> IEmbedder:
        if self._embedder is None:
            raise StorageError("embedder is required for this operation")
        return self._embedder

    @property
    def embedding_model_dirty(self) -> bool:
        return self._embedding_model_dirty

    def _refresh_dirty_from_meta(self) -> None:
        if self._embedder is None:
            self._embedding_model_dirty = False
            return
        if self._pending_wipe or not _table_exists(self.conn, "unit_embeddings"):
            self._embedding_model_dirty = False
            return
        meta_model = _meta_get(self.conn, META_MODEL_KEY)
        if meta_model is None:
            self._embedding_model_dirty = (
                self._embedder.model_id != LOCAL_EMBEDDING_MODEL_ID
            )
            return
        self._embedding_model_dirty = meta_model != self._embedder.model_id

    async def _run_on_executor(self, fn):
        """Run DuckDB work on this connection's dedicated thread.

        Does not acquire ``_conn_lock`` — for callers that already hold it.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn)

    async def _with_conn(self, fn):
        """Run DuckDB work serially, on this connection's dedicated thread."""
        async with self._conn_lock:
            return await self._run_on_executor(fn)

    async def ensure_embeddings_bound(self) -> None:
        if self._embeddings_bound:
            return
        async with self._conn_lock:
            if self._embeddings_bound:
                return
            await self._bind_embeddings()
            self._embeddings_bound = True
            self._pending_wipe = False

    async def _bind_embeddings(self) -> None:
        if self.mode is AccessMode.READ_ONLY:
            exists = await self._run_on_executor(
                lambda: _table_exists(self.conn, "unit_embeddings")
            )
            if not exists:
                raise StorageError(
                    EMBEDDINGS_MISSING_MSG,
                    code=ErrorCode.EMBEDDINGS_MISSING,
                )

        dim = await bind_embedder_dimension(
            self.conn,
            self.embedder,
            wiped=self._pending_wipe,
            executor=self._executor,
            read_only=self.mode is AccessMode.READ_ONLY,
        )
        embedder_model_id = self.embedder.model_id

        def _create_table_and_sync_meta():
            schema_n = (
                _schema_vec_width(self.conn)
                if _table_exists(self.conn, "unit_embeddings")
                else None
            )
            meta_model = _meta_get(self.conn, META_MODEL_KEY)
            if schema_n is not None and schema_n != dim:
                raise StorageError(
                    f"Embedding dimension mismatch (index={schema_n}, embedder={dim}). Run rebuild.",
                    code=ErrorCode.EMBEDDING_MISMATCH,
                )
            if self.mode is AccessMode.READ_ONLY:
                return
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
                _meta_set(self.conn, META_MODEL_KEY, embedder_model_id)

        await self._run_on_executor(_create_table_and_sync_meta)

    async def set_dependency_path(self, lib_name: str, path: str) -> None:
        """Caches the absolute path to a library's JAR/binary."""
        await self._with_conn(
            lambda: self.conn.execute(
                "INSERT OR REPLACE INTO dependencies (name, path) VALUES (?, ?)",
                [lib_name, path],
            )
        )

    async def get_dependency_path(self, lib_name: str) -> Optional[str]:
        """Retrieves the cached path for a library."""
        res = await self._with_conn(
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
        payload = [
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
        ]
        vec = vector
        unit_id = unit.id

        def _write():
            self.conn.execute(
                f"INSERT OR REPLACE INTO units ({cols}) VALUES ({placeholders})",
                payload,
            )
            if vec is not None:
                self.conn.execute(
                    "INSERT OR REPLACE INTO unit_embeddings (id, vec) VALUES (?, ?)",
                    [unit_id, vec],
                )

        await self._with_conn(_write)
        for rel in unit.relations:
            await self.upsert_relation(rel)

    async def has_embedding(self, unit_id: str) -> bool:
        await self.ensure_embeddings_bound()
        row = await self._with_conn(
            lambda: self.conn.execute(
                "SELECT 1 FROM unit_embeddings WHERE id = ?", [unit_id]
            ).fetchone()
        )
        return row is not None

    async def list_units(self) -> List[KnowledgeUnit]:
        cols = ", ".join(UNIT_COLUMNS)
        rows = await self._with_conn(
            lambda: self.conn.execute(f"SELECT {cols} FROM units").fetchall()  # nosec
        )
        return [self._map_row_to_unit(row) for row in rows]

    async def mark_embedding_model_synced(self) -> None:
        model_id = self.embedder.model_id
        await self._with_conn(lambda: _meta_set(self.conn, META_MODEL_KEY, model_id))
        self._embedding_model_dirty = False

    async def get_unit(self, unit_id: str) -> Optional[KnowledgeUnit]:
        """Retrieves a unit by its unique ID."""
        cols = ", ".join(UNIT_COLUMNS)
        res = await self._with_conn(
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
        sql = f"""
                SELECT {cols}, array_distance(e.vec, ?::{float_vec_sql_type(dim)}) as dist
                FROM units u
                JOIN unit_embeddings e ON u.id = e.id
                ORDER BY dist ASC
                LIMIT ?
                """  # nosec
        res = await self._with_conn(
            lambda: self.conn.execute(sql, [query_vec, limit]).fetchall()
        )
        units = [self._map_row_to_unit(row) for row in res]
        if not units:
            return []

        unit_ids = [u.id for u in units]
        rels_res = await self._with_conn(
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
        await self._with_conn(
            lambda: self.conn.execute(
                """
            INSERT OR REPLACE INTO relations (from_id, to_id, type)
            VALUES (?, ?, ?)
        """,
                [relation.from_id, relation.to_id, relation.type.value],
            )
        )

    async def get_relations(
        self, unit_id: str, direction: str = "out"
    ) -> List[Relation]:
        """Retrieves relations for a unit."""
        if direction == "out":
            res = await self._with_conn(
                lambda: self.conn.execute(
                    "SELECT from_id, to_id, type FROM relations WHERE from_id = ?",
                    [unit_id],
                ).fetchall()
            )
        else:
            res = await self._with_conn(
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

        def _delete():
            self.conn.execute(
                "DELETE FROM unit_embeddings WHERE id IN (SELECT id FROM units WHERE path = ? AND id NOT IN (SELECT unnest(?)))",
                [file_path, current_unit_ids],
            )
            self.conn.execute(
                "DELETE FROM relations WHERE from_id IN (SELECT id FROM units WHERE path = ? AND id NOT IN (SELECT unnest(?)))",
                [file_path, current_unit_ids],
            )
            self.conn.execute(
                "DELETE FROM units WHERE path = ? AND id NOT IN (SELECT unnest(?))",
                [file_path, current_unit_ids],
            )

        await self._with_conn(_delete)
        logger.debug("Cleaned up stale units for %s", file_path)

    async def paths_migration_done(self) -> bool:
        value = await self._with_conn(lambda: _meta_get(self.conn, PATHS_MIGRATED_KEY))
        return value == "1"

    async def commit_path_migration(self, mapping: dict[str, str]) -> None:
        """Rewrite absolute paths, then set the once-only mark, in one transaction."""

        def _commit():
            self.conn.execute("BEGIN TRANSACTION")
            try:
                has_embeddings = _table_exists(self.conn, "unit_embeddings")
                for old, new in mapping.items():
                    _rewrite_path_prefix(
                        self.conn, old, new, has_embeddings=has_embeddings
                    )
                _meta_set(self.conn, PATHS_MIGRATED_KEY, "1")
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

        await self._with_conn(_commit)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.conn:
            await self._with_conn(self.conn.close)
            logger.info("Storage connection closed.")
        await asyncio.to_thread(self._executor.shutdown, wait=True)

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


async def finalize_rw_open(storage: DuckDBStorage, *, wipe: bool) -> None:
    """Record the pending-wipe flag and refresh dirty state after open."""
    storage._pending_wipe = wipe  # pylint: disable=protected-access
    # pylint: disable-next=protected-access
    await storage._run_on_executor(storage._refresh_dirty_from_meta)
