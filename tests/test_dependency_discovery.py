import pytest
from code_rag.discovery.dependency import extract_library_api

@pytest.mark.asyncio
async def test_extract_library_api_success():
    """
    Test extraction of public API for a standard library (math).
    """
    result = await extract_library_api("math")

    assert "Public API for 'math':" in result
    assert "Function: cos" in result
    assert "Function: sin" in result
    # 'math' might not have classes in all Python versions, let's also check 'json'

    result_json = await extract_library_api("json")
    assert "Public API for 'json':" in result_json
    assert "Function: dumps" in result_json
    assert "Class: JSONEncoder" in result_json

@pytest.mark.asyncio
async def test_extract_library_api_failure():
    """
    Test extraction of public API for a non-existent library.
    """
    result = await extract_library_api("non_existent_library_name_12345")
    assert "Failed to extract API for 'non_existent_library_name_12345':" in result

@pytest.mark.asyncio
async def test_extract_library_api_limit():
    """
    Verify that the output is limited to 100 lines.
    """
    # math has many functions, it might reach the limit if the library was very large
    # but the current implementation limits to 100 lines/entries.
    # Let's check a library with many members like 'os'
    result = await extract_library_api("os")
    lines = result.splitlines()
    assert len(lines) <= 100
