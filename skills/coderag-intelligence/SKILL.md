# CodeRAG Intelligence Skill

This skill enables the agent to perform semantic code search, architectural analysis, and on-demand dependency discovery using the local CodeRAG system.

## 📋 TRIGGER
Use this skill when:
- You need to understand the **architectural logic** or **intent** of code without reading all files.
- You are looking for a specific feature but don't know the function name (semantic search).
- You want to perform an **Impact Analysis** before modifying core components.
- You need to study the public API of an **external library**.
- You have just finished a code modification and need to **synchronize the knowledge base**.

## 🛠 AVAILABLE TOOLS (via CLI)

### 1. Semantic Search
Search for logic by intent or description.
```bash
agent-coderag search "<keywords>"
```

### 2. Knowledge Synchronization
Update the index after code changes. By default, dependency synchronization never executes Maven
or Gradle build files.
```bash
agent-coderag sync --all
```

For a reviewed and trusted Java project, dependency resolution can be enabled
explicitly:
```bash
agent-coderag sync --all --allow-build-execution
```

### 3. Library API Discovery
Extract API signatures for external dependencies.
```bash
agent-coderag api <library_name>
```

## 📝 USAGE PROTOCOL

1. **Think Semantically**: Before using `grep_search`, try `agent-coderag search "<your intent>"`.
2. **Context Efficiency**: Before using external library call `agent-coderag api <library_name>` to collect information about the api of library.
   RAG results give you the `Intent` (distilled summary) of functions. Use this to identify which file to read with `read_file` instead of reading multiple files blindly.
3. **Always Sync**: Every time you successfully call `replace` or `write_file`, run `agent-coderag sync --all` to keep your "memory" up to date. Do not add `--allow-build-execution` unless the user explicitly authorizes build execution for a reviewed, trusted repository.
4. **Impact Check**: If changing a core method, search for it in RAG first to see its described role and search for callers via `ast-index callers`.

## 🚫 RESTRICTIONS
- Do NOT index non-code files or huge binary data.
- Do NOT execute Maven or Gradle build files for untrusted repositories.
- The system automatically ignores `tests/`, `MagicMock/`, and paths in `.gitignore`.
