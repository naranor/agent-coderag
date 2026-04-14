# Security Policy

## Reporting a Vulnerability

We take the security of CodeRAG seriously. If you find a security vulnerability, please do NOT open a public issue. Instead, report it by sending an email to `naranor@gmail.com`.

## Safety Warnings

### LLM API Keys
*   **Never commit your API keys** to version control.
*   CodeRAG uses `litellm` which can read keys from environment variables. Prefer using `.env` files (make sure they are in `.gitignore`).
*   CodeRAG saves its configuration in a global cache folder (`~/.cache/agent-coderag/config.json`). This file is readable by your user. Ensure your system is secure if you store sensitive keys there.

### Local Database
*   The `.code_rag.db` file contains embeddings and metadata of your code. By default, it is stored in your project directory. 
*   **Avoid sharing this file** if your source code is private, as embeddings can sometimes be used to reconstruct parts of the original text.
