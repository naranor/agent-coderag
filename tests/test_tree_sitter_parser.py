import pytest
import os
import tempfile
from code_rag.parsers.tree_sitter import TreeSitterParser, GrammarNotFoundError


def normalize_ws(s: str) -> str:
    return " ".join(s.split())


@pytest.mark.asyncio
async def test_python_distillation():
    parser = TreeSitterParser()
    content = """
class MyClass:
    def method(self):
        pass

def top_level_fn(a, b):
    return a + b
"""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(content.encode("utf-8"))
        temp_path = f.name

    try:
        units = await parser.distill_file(temp_path)
        print(f"\nPython units: {[u.name for u in units]}")

        assert len(units) == 3

        # Class
        cls_unit = next(u for u in units if u.name == "MyClass")
        assert cls_unit.kind == "class"
        assert f"{temp_path}:MyClass" == cls_unit.id
        assert "class MyClass: ..." == normalize_ws(cls_unit.signature)
        assert "class MyClass:" in cls_unit.metadata["raw_code"]
        assert "def method(self):" in cls_unit.metadata["raw_code"]

        # Method
        method_unit = next(u for u in units if u.name == "method")
        assert method_unit.kind == "function"
        assert f"{temp_path}:MyClass.method" == method_unit.id
        assert "def method(self): ..." == normalize_ws(method_unit.signature)
        assert "def method(self):" in method_unit.metadata["raw_code"]

        # Function
        fn_unit = next(u for u in units if u.name == "top_level_fn")
        assert fn_unit.kind == "function"
        assert f"{temp_path}:top_level_fn" == fn_unit.id
        assert "def top_level_fn(a, b): ..." == normalize_ws(fn_unit.signature)
        assert "return a + b" in fn_unit.metadata["raw_code"]

    finally:
        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_javascript_distillation():
    parser = TreeSitterParser()
    content = """
class UserService {
    async getUser(id) {
        return await db.users.find(id);
    }
}

function healthCheck() {
    return "OK";
}

const arrow = (x) => {
    return x * 2;
};
"""
    with tempfile.NamedTemporaryFile(suffix=".js", delete=False) as f:
        f.write(content.encode("utf-8"))
        temp_path = f.name

    try:
        units = await parser.distill_file(temp_path)

        assert len(units) == 4

        # Class
        cls_unit = next(u for u in units if u.name == "UserService")
        assert cls_unit.kind == "class"
        assert f"{temp_path}:UserService" == cls_unit.id
        assert "class UserService { ... }" == normalize_ws(cls_unit.signature)
        assert "async getUser(id)" in cls_unit.metadata["raw_code"]

        # Method
        method_unit = next(u for u in units if u.name == "getUser")
        assert method_unit.kind == "method"
        assert f"{temp_path}:UserService.getUser" == method_unit.id
        assert "async getUser(id) { ... }" == normalize_ws(method_unit.signature)
        assert "return await db.users.find(id)" in method_unit.metadata["raw_code"]

        # Function
        fn_unit = next(u for u in units if u.name == "healthCheck")
        assert fn_unit.kind == "function"
        assert f"{temp_path}:healthCheck" == fn_unit.id
        assert "function healthCheck() { ... }" == normalize_ws(fn_unit.signature)

        # Arrow Function
        arrow_unit = next(
            u for u in units if u.kind == "function" and u.name == "arrow"
        )
        assert f"{temp_path}:arrow" == arrow_unit.id
        assert "=> { ... }" in normalize_ws(arrow_unit.signature)
        assert "return x * 2" in arrow_unit.metadata["raw_code"]

    finally:
        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_missing_grammar():
    parser = TreeSitterParser()
    # .rb is ruby, we didn't install tree-sitter-ruby
    with tempfile.NamedTemporaryFile(suffix=".rb", delete=False) as f:
        f.write(b"def hello; end")
        temp_path = f.name

    try:
        with pytest.raises(GrammarNotFoundError) as excinfo:
            await parser.distill_file(temp_path)
        assert "[MISSING DEPENDENCY]" in str(excinfo.value)
        assert "tree_sitter_ruby" in str(excinfo.value)
    finally:
        os.unlink(temp_path)
