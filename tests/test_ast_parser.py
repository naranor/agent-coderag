import pytest
from code_rag.parsers.ast_index import AstIndexParser
from code_rag.core.models import UnitKind


@pytest.fixture
def test_py_file(tmp_path):
    content = """
class MyClass:
    def method_one(self, a: int):
        return a * 2

def top_level_func(b: str) -> str:
    return f"Hello {b}"

# Simple variable
x = 42
"""
    f_path = tmp_path / "sample.py"
    f_path.write_text(content)
    return str(f_path)


@pytest.mark.asyncio
async def test_ast_parser_real_file(test_py_file):
    """
    Verifies that AstIndexParser correctly extracts symbols from a real file.
    """
    parser = AstIndexParser()
    units = await parser.distill_file(test_py_file)

    names = [u.name for u in units]

    # We expect top-level symbols
    assert "MyClass" in names
    assert "top_level_func" in names

    # Verify kinds mapping
    my_class_unit = next(u for u in units if u.name == "MyClass")
    assert my_class_unit.kind == UnitKind.CLASS

    func_unit = next(u for u in units if u.name == "top_level_func")
    assert func_unit.kind == UnitKind.FUNCTION

    # Check that metadata contains raw code
    assert "def top_level_func" in func_unit.metadata["raw_code"]


@pytest.mark.asyncio
async def test_ast_parser_empty_or_no_symbols(tmp_path):
    """
    Verifies fallback to module-level indexing if no symbols are found.
    """
    f_path = tmp_path / "empty.py"
    f_path.write_text("# Just comments\n# Nothing here")

    parser = AstIndexParser()
    units = await parser.distill_file(str(f_path))

    assert len(units) == 1
    assert units[0].kind == UnitKind.MODULE
    assert units[0].name == "empty.py"
    assert "Just comments" in units[0].metadata["raw_code"]


@pytest.mark.asyncio
async def test_ast_parser_non_existent_file():
    """
    Should return empty list gracefully for missing files.
    """
    parser = AstIndexParser()
    units = await parser.distill_file("non_existent_file.py")
    assert units == []
