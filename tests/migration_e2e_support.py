import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

from code_rag.core.interfaces import IEmbedder
from code_rag.parsers.multi_parser import MultiParser
from code_rag.storage.db_connection import AccessMode, open_db_connection
from code_rag.storage.duckdb_impl import META_DIM_KEY, META_MODEL_KEY, _meta_set

VECTOR = [0.1] * 8
MODEL_ID = "fake-embed"


@contextmanager
def isolated_home_at(home: Path) -> Iterator[Path]:
    """Point get_global_dir at home for this process. Restore afterward."""
    key = "LOCALAPPDATA" if os.name == "nt" else "XDG_CACHE_HOME"
    previous = os.environ.get(key)
    os.environ[key] = str(home)
    try:
        yield home
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def write_embedding_config(home: Path, base_url: str) -> None:
    config = home / "agent-coderag" / "config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps(
            {
                "embedding_base": base_url,
                "embedding_model": MODEL_ID,
                "embedding_key": "test-key",
                "embedding_provider": "openai",
            }
        ),
        encoding="utf-8",
    )


class EmbeddingServer:
    def __init__(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/v1/embeddings":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                inputs = body["input"]
                payload = {
                    "object": "list",
                    "model": body.get("model", MODEL_ID),
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    "data": [
                        {
                            "object": "embedding",
                            "index": index,
                            "embedding": list(VECTOR),
                        }
                        for index, _text in enumerate(inputs)
                    ],
                }
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, fmt: str, *args: object) -> None:
                return

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._thread.join(timeout=5)


async def seed_parsed_units(
    db: Path,
    embedder: IEmbedder,
    rows: list[tuple[Path, str, str]],
) -> None:
    """Store parser units under stored_absolute_path with a non-empty summary."""
    parser = MultiParser()
    storage = await open_db_connection(
        db,
        embedder,
        mode=AccessMode.READ_WRITE,
        connect_timeout_seconds=0,
    )
    try:
        for source, stored, summary in rows:
            units = await parser.distill_file(str(source))
            if not units:
                raise AssertionError(f"parser returned no units for {source}")
            for unit in units:
                unit.path = stored
                unit.id = f"{stored}:{unit.name}"
                unit.summary = summary
                await storage.upsert_unit(unit, vector=list(VECTOR))
        _meta_set(storage.conn, META_MODEL_KEY, embedder.model_id)
        _meta_set(storage.conn, META_DIM_KEY, str(embedder.dimension))
    finally:
        await storage.close()


def run_cli(
    args: list[str], *, cwd: Path, home: Path
) -> subprocess.CompletedProcess[str]:
    key = "LOCALAPPDATA" if os.name == "nt" else "XDG_CACHE_HOME"
    env = os.environ.copy()
    env[key] = str(home)
    name = "agent-coderag.exe" if os.name == "nt" else "agent-coderag"
    script = str(Path(sys.executable).resolve().parent / name)
    return subprocess.run(
        [script, *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
