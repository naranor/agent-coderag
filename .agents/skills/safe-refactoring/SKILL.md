# Skill: Safe Refactoring & Contract Integrity

Expert guidance for performing maintenance and refactoring without breaking system-wide contracts. This skill MUST be activated before modifying any files in `core/`, `interfaces/`, or `base/`.

## Core Principles

### 1. The "Impact First" Rule
NEVER delete a method, field, or class based on "perceived" redundancy. 
- **Action**: Before any removal, run `ast-index find-usages` or `grep_search` on the entire codebase.
- **Goal**: Identify hidden dependencies in tools, tests, and external modules.

### 2. Surgical Precision Over Global Overwrites
Avoid using `write_file` on existing core files. Global rewrites are "nuclear strikes" that erase context and logic you might have missed.
- **Action**: Use the `replace` tool for targeted, minimal changes. 
- **Exception**: `write_file` is allowed ONLY for brand new files or small utility scripts (< 50 lines).

### 3. Continuous Validation (Heartbeat Testing)
Verification is not a "final phase." It is a requirement for EVERY turn.
- **Action**: Run relevant unit tests after EVERY file modification. 
- **Hard Gate**: Do NOT proceed to the next task in your plan if the current change broke even a single test. Fix the regression immediately.

### 4. Backward Compatibility (The Legacy Bridge)
When optimizing data structures (e.g., adding `exclude=True` for Pydantic), ensure the original fields and methods remain available.
- **Action**: Use properties (`@property`), field aliases, or default values to keep the "old" API working while implementing the "new" logic.

## Checklist for Every Edit

- [ ] **Search**: Find all callers of the symbol you are changing.
- [ ] **Analyze**: Does this change break the `AgentDeps` or `OrchestratorState` contract?
- [ ] **Execute**: Apply change using surgical `replace`.
- [ ] **Verify**: Run `pytest tests/test_v4_<affected_module>.py`.
- [ ] **Commit**: Only commit verified, non-breaking changes.

## Anti-Patterns to Avoid
- "I'll fix the tests at the end." -> **Result**: Untraceable cascade of errors.
- "This method looks unused in this file." -> **Result**: `AttributeError` in a tool called 20 steps later.
- "Rewriting the whole interface is faster than 5 replacements." -> **Result**: Loss of critical helper logic and metadata.
