---
name: pydantic-ai-expert
description: Expert guidance for PydanticAI (2024-2025). Use when designing, implementing, or refactoring AI agents using the PydanticAI framework to ensure type safety, structured outputs, and robust dependency injection.
---

# PydanticAI Expert Guide

## Overview
PydanticAI is a model-agnostic agent framework that leverages Pydantic for validation, type safety, and structured data handling.

## Core Patterns

### 1. Agent Initialization
Define your agent with explicit dependency and result types.
```python
from pydantic_ai import Agent
from .models import MyResult, MyDeps

agent = Agent(
    'openai:gpt-4o', # Or 'openai:auto' for AI Revolver
    deps_type=MyDeps,
    result_type=MyResult,
    system_prompt="Your role-specific instructions..."
)
```

### 2. Dependency Injection (RunContext)
Use `RunContext` to access session-specific data (like `SessionManager`) without global variables.
```python
from pydantic_ai import RunContext

@agent.tool
async def my_tool(ctx: RunContext[MyDeps], arg1: str) -> str:
    # Use ctx.deps to access shared state
    workspace = ctx.deps.session_manager.get_workspace_path()
    ...
```

### 3. Structured Result Validation
Force the agent to return a valid Pydantic model.
```python
result = await agent.run("Task description", deps=my_deps_instance)
# result.data is now a validated Pydantic model instance
print(result.data.field_name)
```

### 4. Dynamic System Prompts
Use `@agent.system_prompt` to inject dynamic information into the system prompt at runtime.
```python
@agent.system_prompt
def get_system_prompt(ctx: RunContext[MyDeps]) -> str:
    return f"Context from session {ctx.deps.session_id}: ..."
```

## Integration with Agentic OS
- **Isolation:** Always pass the `SessionManager` instance through `deps`.
- **Reporting:** Use `result_type` to ensure every phase produces a structured report.
- **Error Handling:** PydanticAI handles tool validation errors internally by asking the LLM to retry with the error message.

## Best Practices
- **Prefer `@agent.tool`** for async tools that need context.
- **Use `BaseModel`** for complex arguments to get automatic validation.
- **Enable Logfire** for tracing if available.
- **Explicit Provider:** When using AI Revolver, pass `custom_llm_provider="openai"` to `litellm` calls under the hood if needed.
