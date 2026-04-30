---
name: python-code-auditor
description: Performs a comprehensive audit of Python codebases for quality, SOLID compliance, and best practices. Use when you need to analyze project structure, find technical debt, and prioritize improvements.
---

# Python Code Auditor

This skill provides a systematic workflow for auditing Python codebases.

## Audit Workflow

### 1. Automated Analysis
Execute the audit script to get a high-level overview of static analysis issues.

```bash
python3 .gemini/skills/python-code-auditor/scripts/run_audit.py
```

### 2. Best Practices Review
Refer to [references/solid_best_practices.md](references/solid_best_practices.md) to evaluate the code against:
- SOLID Principles
- PEP 8 Standards
- Documentation and Typing quality
- Common Anti-patterns

### 3. Structural Audit
Analyze the project's architecture:
- **Dependency Map**: Use `tree` or `grep` to find circular dependencies or high coupling.
- **Missing Components**: Identify lack of tests, documentation, or proper configuration (e.g., missing `README.md`, `requirements.txt`, or CI/CD configs).
- **Security Audit**: Use `grep` or specialized tools to find secrets, insecure `eval()` calls, or outdated dependencies.

### 4. Reporting & Prioritization
Generate a Quality Control Report with the following sections:
1. **Executive Summary**: Overall health of the codebase.
2. **Key Findings**: Top 3-5 critical issues (e.g., architectural flaws, security risks).
3. **SOLID Compliance**: Assessment of each principle.
4. **Actionable Roadmap**: Prioritized list of tasks:
   - **High Priority**: Immediate fixes (security, crashes, major debt).
   - **Medium Priority**: Refactorings, documentation improvements.
   - **Low Priority**: Style consistency, minor optimizations.

## Tool Integration
- **Ruff**: Primary static analyzer.
- **Pytest**: Use to check if existing tests cover the critical paths.
- **Grep**: Use for pattern-based architectural searches.
