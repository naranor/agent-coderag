import os
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from code_rag.api.models import SetupResult
from code_rag.intelligence.embedder import get_global_dir

MODEL_FILES = {
    "model.onnx": "https://huggingface.co/naranor/all-MiniLM-L6-v2-onnx/resolve/main/model.onnx",
    "tokenizer.json": "https://huggingface.co/naranor/all-MiniLM-L6-v2-onnx/resolve/main/tokenizer.json",
}

HttpGet = Callable[..., Any]

# Progress callback signature: (event, name, error) where event is one of
# "downloading", "skipped", "error" and error is set only for "error".
ProgressCallback = Callable[[str, str, Optional[BaseException]], None]


def _download_file(get: HttpGet, url: str, target_path: Path) -> None:
    tmp_path = target_path.with_suffix(".tmp")
    try:
        response = get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(tmp_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                handle.write(chunk)
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


async def run_setup(
    *,
    force: bool = False,
    global_dir: Optional[Path] = None,
    on_progress: Optional[ProgressCallback] = None,
    http_get: Optional[HttpGet] = None,
) -> SetupResult:
    """Downloads missing local embedding model files into the global dir."""
    target_dir = global_dir if global_dir is not None else get_global_dir()
    model_dir = target_dir / "models" / "mini-lm"
    model_dir.mkdir(parents=True, exist_ok=True)
    get: HttpGet = http_get or requests.get

    downloaded: list[str] = []
    skipped: list[str] = []

    for name, url in MODEL_FILES.items():
        target_path = model_dir / name
        if target_path.exists() and not force:
            skipped.append(name)
            if on_progress:
                on_progress("skipped", name, None)
            continue

        if on_progress:
            on_progress("downloading", name, None)

        try:
            _download_file(get, url, target_path)
            downloaded.append(name)
        except Exception as exc:
            if on_progress:
                on_progress("error", name, exc)
            raise

    return SetupResult(model_dir=str(model_dir), downloaded=downloaded, skipped=skipped)
