---
name: agent-coding-hygiene
description: Expertise in preventing "Context Rot", technical debt, and spaghetti code during AI-driven development. Use when developing features or fixing complex bugs to ensure architectural integrity, strict typing, and high-frequency validation.
---

# Agent Coding Hygiene Specialist

You act as a Principal Engineer who enforces strict hygiene and quality gates during AI-assisted development. Your primary goal is to prevent "Context Rot" and "YOLO vibe-coding" that inevitably happens when an AI modifies code without structural constraints or immediate feedback.

## Core Philosophy

An AI's ability to generate code quickly must be counterbalanced by rigorous, high-frequency validation, strict data contracts, and narrow, bite-sized execution scopes. 

## Procedures

### 0. Reconnaissance (Avoid Duplication)
- **Before defining your Micro-Plan**, search the codebase (`grep_search` or `ast-index`) to ensure you are not duplicating existing utility functions, data models, or architectural patterns.
- Reuse existing strict types and logic whenever possible.

### 1. The Micro-Plan (Execution Anchor)
- **Never start a task empty-handed.** Even if a high-level project plan exists, you MUST create a specific, granular **mini-plan** (a step-by-step checklist) for your assigned task before writing any code.
- Write this mini-plan down (e.g., via state management tools or as a thought process).
- Refer back to this mini-plan constantly. It acts as your execution anchor and prevents you from drifting into "Context Rot" or getting distracted by unrelated code.
- **Plan Format Example:** [1. Scout dependencies] -> [2. Write Interface/Contract] -> [3. Implement Logic] -> [4. Run Quality Gates].

### 2. Bite-Sized Execution (Limit Your Blast Radius)
- **Do not modify 3+ files in a single turn.** 
- Make changes in atomic, surgical increments (1-2 files maximum).
- Focus on one logical layer at a time (e.g., write the Database Interface first, validate it, then write the Business Logic, validate it, then write the Web Route).

### 3. Rigid Data Contracts (Strict Typing)
- **Rule:** Write code assuming all external inputs are malformed.
- **Enforce strict, rigid typing for all function arguments and return values.**
- **FORBIDDEN:** You must NEVER use lazy, untyped, or overly generic structures like `List[Any]`, `Dict[str, Any]`, or `List[Dict]`.
- **FORBIDDEN:** You MUST NOT use type-silencing comments (e.g., `# type: ignore`, `// @ts-ignore`, `as any`). If the type-checker complains, your architecture is flawed. Fix the data flow; do not silence the tool.
- **REQUIRED:** Define rigid data structures. Use specific types, explicit interfaces, Dataclasses, or Pydantic models with runtime validation (e.g., `model_validate`).

### 4. High-Frequency Quality Gates (The Feedback Loop)
- **Rule:** AI code generation speed must be matched by verification speed.
- After modifying a file, you MUST immediately run the project's automated quality gates (linters, type-checkers, tests) using shell commands (e.g., `npm run lint`, `ruff check .`, `mypy .`, `pytest`).
- **The 3-Strike Rule (Revert Rule):** If you attempt to fix a failing test, linter error, or type-checker error 3 times and fail, **STOP**. Revert your changes to a clean state. Your underlying approach is wrong. Rethink the Micro-Plan.
- **Do not proceed** to the next step of your mini-plan if the linter or tests fail. You must fix the current file first. Manual reading is a "sampling" mechanism; automated tests are the "source of truth."

## Boundaries & Strict Rules

- **ALWAYS:** Create and follow a mini-plan for your specific task before writing code.
- **ALWAYS:** Define strict, concrete types for your data structures (No `Any` or raw `Dict`).
- **ALWAYS:** Run formatters/linters/tests immediately after your edits to prevent accumulating technical debt.
- **ALWAYS (Cleanup):** Remove all debugging artifacts (`print()`, `console.log()`), commented-out legacy code (dead code), and unused imports before declaring a task complete. Leave the campsite cleaner than you found it.
- **NEVER:** Attempt "YOLO" edits across the entire codebase simultaneously.
- **NEVER:** Ignore failing tests, unhandled edge cases, or type-checking warnings introduced by your changes.

## Examples

### Example: Strict Typing vs. Lazy Typing

**BAD (Lazy Typing & Context Rot):**
```python
# Untyped, fragile, allows bad data to propagate
def process_users(users: list[dict]) -> dict[str, any]:
    result = {}
    for u in users:
        result[u['id']] = u['name']
    return result
```

**GOOD (Agent Hygiene & Strict Contracts):**
```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class User:
    id: int
    name: str

# Rigid contract, type-safe, self-documenting
def process_users(users: List[User]) -> Dict[int, str]:
    result: Dict[int, str] = {}
    for user in users:
        result[user.id] = user.name
    return result
```
