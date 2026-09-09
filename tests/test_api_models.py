from code_rag.api.models import SyncResult, ApiReport, SetupResult


def test_sync_result_shape():
    result = SyncResult(status="success", indexed_files=3)
    assert result.model_dump() == {"status": "success", "indexed_files": 3}


def test_api_report_shape():
    report = ApiReport(library="pydantic", language="python", report="ok")
    assert report.model_dump() == {
        "library": "pydantic",
        "language": "python",
        "report": "ok",
    }


def test_setup_result_shape():
    setup = SetupResult(
        model_dir="/tmp/models",
        downloaded=["model.onnx"],
        skipped=["tokenizer.json"],
    )
    assert setup.downloaded == ["model.onnx"]
    assert setup.skipped == ["tokenizer.json"]
