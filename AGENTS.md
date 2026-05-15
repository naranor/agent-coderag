# Instructions for AI Agents

You are an AI Coding Agent. Use **CodeRAG** to explore the codebase efficiently without blowing your context window.

## Core Strategy
1.  **Search First**: Before reading full files, use `agent-coderag --json search "topic"` to find relevant code units (functions, classes, modules).
2.  **Use Intent**: Pay attention to the `summary` (Intent) field in the JSON output. It explains *what* the code does, saving you from reading the implementation details prematurely.
3.  **Verify APIs**: If you are unsure about a library's method signature (e.g., Pydantic, FastAPI), run `agent-coderag api <library_name>`.

## Usage Examples

### Semantic Search (JSON)
```bash
agent-coderag --json search "logic for data persistence" --limit 3
```

### API Discovery
```bash
# Recommended: specify language
agent-coderag api litellm --lang python
```

## Integration Tips

### For Cursor (.cursorrules)
Add the following to your `.cursorrules`:
> "Always use `agent-coderag --json search` to locate logic before reading files. If you encounter a library API mismatch, run `agent-coderag api <lib>` to check live signatures."

### For Gemini CLI (Policies)
Ensure your tool policy allows execution of `agent-coderag`. Use it to "compress" project knowledge into your context.

## Output Schema
The `--json` flag returns a list of objects:
- `id`: Unique identifier (path:qname).
- `name`: Entity name.
- `signature`: Function/Method arguments and return type.
- `summary`: High-level technical intent.
- `path`: Relative path to file.
