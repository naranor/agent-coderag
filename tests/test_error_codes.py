from code_rag.core.error_codes import ErrorCode
from code_rag.core.exceptions import CodeRAGError, StorageBusyError, StorageError


def test_storage_busy_has_code():
    err = StorageBusyError("busy")
    assert err.code is ErrorCode.STORAGE_BUSY
    assert isinstance(err, StorageError)
    assert isinstance(err, CodeRAGError)


def test_storage_error_accepts_code():
    err = StorageError("corrupt", code=ErrorCode.STORAGE_CORRUPT)
    assert err.code is ErrorCode.STORAGE_CORRUPT


def test_public_exports():
    import code_rag

    assert hasattr(code_rag, "ErrorCode")
    assert hasattr(code_rag, "StorageBusyError")
