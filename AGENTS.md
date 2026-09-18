# Instructions for AI Agents

You are an AI Coding Agent. Use **CodeRAG** to explore the codebase efficiently without blowing your context window.

## Core Strategy
1.  **Search First**: Before reading full files, use `agent-coderag --json search "topic"` to find relevant code units (functions, classes, modules).
2.  **Use Intent**: Pay attention to the `summary` (Intent) field in the JSON output. It explains *what* the code does, saving you from reading the implementation details prematurely.
3.  **Verify APIs**: If you are unsure about a library's method signature (e.g., Pydantic, FastAPI), run `agent-coderag api <library_name>`.
4.  **Verified Delivery Protocol (VDP)**: Never commit or push without shadowing CI. Run exact commands from `.github/workflows/ci.yml` locally. Use of `--no-verify` is strictly forbidden.

## CI Shadowing Commands
**Always use the project `venv/`** (never system Python). Windows: `venv\Scripts\...`; Unix: `venv/bin/...`.

Before commit, you MUST pass (via venv):
```bash
# Linting
prospector code_rag --profile .prospector.yaml --with-tool mypy --with-tool bandit
vulture code_rag --min-confidence 80 --exclude code_rag/core/models.py

# Testing (with coverage check)
python -m pytest --cov --cov-report=term-missing --cov-fail-under=90
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

## 5. Hermetic E2E and Hang Discipline

**E2E must not depend on the developer machine. Hangs are failures, not "noise".**

### Environment
- Isolate `LOCALAPPDATA` **and** `XDG_CACHE_HOME` (same temp cache root) for package and live e2e. Do not rely on the developer's global `config.json`.
- Prefer an explicit `--onnx` (or MiniLM seeded into the isolated cache). Optional override: `CODERAG_E2E_ONNX` → `model.onnx` or its directory.
- If no ONNX is available after seed: **`pytest.skip`** with a clear message (run `agent-coderag setup` or set `CODERAG_E2E_ONNX`) — do not soft-fail mid-suite.
- At the start of path/resolve live checks: `chdir` into a clean temp tree. Never assert resolve behavior from the repo root if `code_rag.db` / `.coderag.db` may exist in cwd.

### Subprocesses and timeouts
- Every test `subprocess.run` / CLI helper must set `timeout=`.
- Prefer `pytest-timeout` (or equivalent) for e2e so a hang fails the job instead of requiring a manual kill.
- If a process was killed to unblock the agent, **do not** treat the suite as green until it passes without kills.

### Hang playbook (Windows)
1. Reproduce with a short timeout (don't attach a debugger first).
2. Bisect: 1 file vs 2+ files in `sync --all` (concurrency).
3. Re-run under isolated appdata (config/distill).
4. Only then inspect locks / PIDs — remember `venv\Scripts\python.exe` may be a stub over the real interpreter (parent/child with the same argv is often normal).

## 6. Concurrency and Queue Safety

**Worker pools need negative tests with deadlines.**

- Do not use `queue.empty()` followed by blocking `queue.get()` in asyncio workers. Prefer `get_nowait()` / sentinels / join patterns.
- Any change to `sync_project` (or similar pools): add a test with **more workers than items** and wrap in `asyncio.wait_for(..., timeout=…)`.
- E2E sync coverage must use **≥ 2 source files** so worker-pool races can surface.
- When reviewing thread-affine DuckDB / single-thread executors: check that concurrent callers cannot deadlock on locks + executor shutdown.

## 7. Live Compat Scripts vs Product Bugs

**A red live script is not automatically a product bug.**

- Prefer canonical tests under `e2e_tests/` or `tests/` over one-off TEMP scripts. If a live script is needed, copy public API usage from README / `test_readme_*`.
- Before blaming resolve/storage: read the real signature (`resolve_db_path(db, *, root=…)`) and existing unit tests (`test_db_path.py`, incomplete-meta storage tests).
- Import meta keys from the module that owns them (e.g. storage), or use the string values those tests already use — do not invent `constants.META_*`.
- Separate failures: wrong test harness vs product regression. Fix the harness first when the unit suite already covers the behavior.

## 8. Scope Blind Spots After Redesigns

**A redesign does not excuse untested neighbors.**

- After storage/lifetime/API changes, still run concurrency e2e (`sync --all` multi-file) and legacy-DB compat — even if the redesign did not touch that code.
- Manual kill of "orphan CLI" during e2e is a smell: file a hang repro (1 vs 2 files, isolated env) before continuing the roadmap.

---
