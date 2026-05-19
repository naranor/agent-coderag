import pytest
from code_rag.discovery.dependency import extract_library_api


@pytest.mark.asyncio
async def test_extract_library_api_success():
    """
    Test extraction of public API for a standard library (math).
    """
    result = await extract_library_api("math", language="python")

    assert "Public API for Python Library 'math'" in result
    assert "Function: cos" in result
    assert "Function: sin" in result

    result_json = await extract_library_api("json", language="python")
    assert "Public API for Python Library 'json'" in result_json
    assert "Function: dumps" in result_json
    assert "Class: JSONEncoder" in result_json


@pytest.mark.asyncio
async def test_extract_library_api_failure():
    """
    Test extraction of public API for a non-existent library.
    """
    result = await extract_library_api(
        "non_existent_library_name_12345", language="python"
    )
    # Python runtime fallback still returns a descriptive error string
    assert "Failed to extract API" in result


@pytest.mark.asyncio
async def test_extract_library_api_limit():
    """
    Verify that the output is limited to a reasonable size.
    """
    result = await extract_library_api("os", language="python")
    lines = result.splitlines()
    assert len(lines) <= 101  # header + limit
