import pytest
from code_rag.parsers.java_parser import JavaParser
from code_rag.core.models import UnitKind


@pytest.fixture
def test_java_file(tmp_path):
    content = """
import java.util.List;

public class MyClass {
    public void methodOne(String arg) {
        System.out.println(arg);
    }

    private int methodTwo() {
        return 42;
    }
}

interface MyInterface {
    void doSomething();
}
"""
    f_path = tmp_path / "Test.java"
    f_path.write_text(content)
    return str(f_path)


@pytest.mark.asyncio
async def test_java_parser_extraction(test_java_file):
    parser = JavaParser()
    units = await parser.distill_file(test_java_file)

    # 1. Check Module
    module = next(u for u in units if u.kind == UnitKind.MODULE)
    assert module.name == "Test.java"
    assert len(module.relations) == 1
    assert module.relations[0].to_id == "java.util.List"

    # 2. Check Class
    clazz = next(u for u in units if u.name == "MyClass" and u.kind == UnitKind.CLASS)
    assert clazz.metadata.get("is_interface") is False
    assert "public class MyClass" in clazz.metadata.get("raw_code")

    # 3. Check Method
    method = next(
        u for u in units if u.name == "methodOne" and u.kind == UnitKind.METHOD
    )
    assert method.signature == "(arg)"
    assert "System.out.println(arg)" in method.metadata.get("raw_code")
    assert method.id.endswith("MyClass.methodOne")

    # 4. Check Interface
    iface = next(
        u for u in units if u.name == "MyInterface" and u.kind == UnitKind.CLASS
    )
    assert iface.metadata.get("is_interface") is True


@pytest.mark.asyncio
async def test_java_parser_non_existent_file():
    parser = JavaParser()
    units = await parser.distill_file("non_existent.java")
    assert units == []
