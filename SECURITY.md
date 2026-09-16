# Security Policy

## Reporting a Vulnerability

We take the security of CodeRAG seriously. If you find a security vulnerability, please do NOT open a public issue. Instead, report it by sending an email to `naranor@gmail.com`.

## Safety Warnings

### LLM API Keys
*   **Never commit your API keys** to version control.
*   CodeRAG uses `litellm` which can read keys from environment variables. Prefer using `.env` files (make sure they are in `.gitignore`).
*   CodeRAG saves its configuration in a global cache folder (`~/.cache/agent-coderag/config.json`). This file is readable by your user. Ensure your system is secure if you store sensitive keys there.

### Local Database
*   The DuckDB index file (`code_rag.db` or `.coderag.db`) contains embeddings and metadata of your code. When `db` is omitted, CodeRAG resolves the path in order: `./code_rag.db` (cwd) if present, else `{project_root}/code_rag.db`, else `{project_root}/.coderag.db` (created on first write). Legacy projects may still use `code_rag.db`; new projects should use `.coderag.db`.
*   DuckDB may create sidecar files (for example `.wal`) next to the database during indexing.
*   **Avoid sharing the index or its sidecars** if your source code is private, as embeddings can sometimes be used to reconstruct parts of the original text.
