import pytest
from code_rag.discovery.dependency import extract_library_api

@pytest.mark.asyncio
async def test_extract_library_api_json():
    """
    Verifies that extract_library_api correctly finds a library (json in this case).
    """
    # 'json' is a built-in library, should be findable
    output = await extract_library_api("json")
    assert "Public API for json" in output
    # JSON library has Encoder/Decoder classes
    assert "JSONEncoder" in output or "JSONDecoder" in output

@pytest.mark.asyncio
async def test_extract_library_api_not_found():
    """
    Should return a graceful message for missing libraries.
    """
    output = await extract_library_api("non_existent_library_xyz_123")
    assert "not found" in output
