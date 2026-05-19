# Instructions for AI Agents

You are an AI Coding Agent. Use **CodeRAG** to explore the codebase efficiently without blowing your context window.

## Core Strategy
1.  **Search First**: Before reading full files, use `agent-coderag --json search "topic"` to find relevant code units (functions, classes, modules).
2.  **Use Intent**: Pay attention to the `summary` (Intent) field in the JSON output. It explains *what* the code does, saving you from reading the implementation details prematurely.
3.  **Verify APIs**: If you are unsure about a library's method signature (e.g., Pydantic, FastAPI), run `agent-coderag api <library_name>`.
4.  **Verified Delivery Protocol (VDP)**: Never commit or push without shadowing CI. Run exact commands from `.github/workflows/ci.yml` locally. Use of `--no-verify` is strictly forbidden.

## CI Shadowing Commands
Before commit, you MUST pass:
```bash
# Linting
prospector code_rag --profile .prospector.yaml --with-tool mypy --with-tool bandit
vulture code_rag --min-confidence 80 --exclude code_rag/core/models.py

# Testing (with coverage check)
pytest --cov --cov-report=term-missing --cov-fail-under=90
```

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

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---
