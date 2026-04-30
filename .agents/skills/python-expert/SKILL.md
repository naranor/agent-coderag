---
name: python-expert
description: Expert guidance for modern Python (2024-2025). Use when writing, refactoring, or optimizing Python code to ensure high performance, type safety, and adherence to modern standards like PEP 622, asyncio TaskGroups, and Pydantic v2.
---

# Python Expert Guide (2024-2025)

## 1. Modern Type Hinting (PEP 585, 604)
- Use native generics: `list[str]` instead of `List[str]`.
- Use the pipe operator: `int | str` instead of `Union[int, str]`.
- Use `typing.Self` for methods returning an instance of their class.

## 2. Advanced Asyncio (Python 3.11+)
- **TaskGroup:** Always use `asyncio.TaskGroup()` for managing multiple concurrent tasks safely.
  ```python
  async with asyncio.TaskGroup() as tg:
      tg.create_task(func1())
      tg.create_task(func2())
  ```
- Avoid `asyncio.gather()` in favor of `TaskGroup` for better error handling.

## 3. Structural Integrity & Patterns
- **Guard Clauses:** Return early to keep logic flat.
- **Pathlib:** Always use `pathlib.Path` for filesystem operations instead of `os.path`.
- **Protocols:** Use `typing.Protocol` for structural subtyping (static duck typing).
- **Dataclasses/Pydantic:** Prefer structured models over dictionaries for data transfer.

## 4. Common Pitfalls (Anti-patterns)
- **NO:** `def func(a=[])` -> **YES:** `def func(a: list | None = None)`.
- **NO:** `except: pass` -> **YES:** Catch specific exceptions.
- **NO:** String concatenation for paths -> **YES:** `path / "subdir" / "file.txt"`.

## 5. Performance Tips
- Use generators (`yield`) for large datasets to save memory.
- Use `slots=True` in dataclasses for reduced memory footprint and faster attribute access.
- Use `itertools` and `collections` for optimized data manipulation.
