# Agent-CodeRAG: Semantic Intelligence for AI Coding Agents

> **Fast. Local. Agent-First. Token-Efficient.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![No PyTorch](https://img.shields.io/badge/Footprint-No_PyTorch-green.svg)](#-key-technologies)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## 📖 Table of Contents
- [🧠 The Problem: The API Knowledge Gap](#-the-problem-the-api-knowledge-gap)
- [🚀 The Solution: Real-Time Contextual Truth](#-the-solution-real-time-contextual-truth)
- [🛠 How it Works](#-how-it-works)
- [📡 API Discovery](#-api-discovery)
- [🏃 Quick Start](#-quick-start)
- [🤖 For AI Agents](#-for-ai-agents)
- [🔧 Development](#-development)
- [📄 License](#-license)

---

## 🧠 The Problem: The API Knowledge Gap

AI coding agents often hallucinate when calling library APIs because their training data is static. This leads to a **"Fail-Fix-Fail" cycle**:

1.  **Broken Code**: Agents use deprecated parameters or non-existent methods from outdated versions.
2.  **Token Waste**: You provide the error, the agent tries to fix it using more outdated data, consuming thousands of tokens in a loop.
3.  **Environment Mismatch**: The agent knows the API for version 1.0, but your environment has 2.0.

### Real-world Example (The Pydantic Gap)
*   **Agent's Knowledge**: Knows Pydantic v1 (`model.dict()`).
*   **Your Environment**: Uses Pydantic v2 (`model.model_dump()`).
*   **The Result**: The agent writes `dict()`, the code fails, and it wastes **5000+ tokens** trying to "fix" a problem it doesn't understand.

## 🚀 The Solution: Real-Time Contextual Truth

Agent-CodeRAG acts as a lightweight semantic bridge between your local environment and the LLM. 

*   **API Discovery**: Extracts *actual* signatures from your installed libraries.
*   **Semantic Retrieval**: Provides the LLM with the exact **Intent** of your code units, indexed locally via ONNX.
*   **Token Efficiency**: Instead of sending whole files, Agent-CodeRAG distills code into compact semantic summaries, **saving up to 80% of context window tokens**.

---

## 🛠 How it Works

```mermaid
graph TD
    A[Local Python Code] --> B[AST Parser]
    B --> C{Delta-Sync}
    C -- Changed/New --> D[LLM Distiller]
    C -- Unchanged --> E[Local Cache]
    D --> F[Semantic Summary]
    E --> F
    F --> G[ONNX Embedder]
    G --> H[(DuckDB VSS)]
    H --> I[Semantic Search / JSON API]
```

### ✨ Key Features
*   **⚡ No PyTorch**: Uses `onnxruntime` and `tokenizers` (Rust) for a tiny footprint and instant startup.
*   **💾 DuckDB VSS**: High-performance vector similarity search stored in a single local file.
*   **🔄 Delta-Sync**: Uses SHA-256 hashing to only re-distill changed code, saving your API budget.
*   **🔌 Hybrid Intelligence**: Works offline using name-based embeddings; adds AI-distilled reasoning when an LLM is connected.

---

## 📡 API Discovery
To help your agent understand a specific library version installed in your environment:
```bash
code-rag api pydantic
```
Returns the *live* public API, methods, and signatures.

---

## 🏃 Quick Start

### 1. Install
```bash
pip install agent-coderag
```

### 2. Setup AI Models
Download the lightweight `paraphrase-multilingual-MiniLM` ONNX model to your global cache:
```bash
code-rag setup
```

### 3. Configure your LLM (For Distillation)
```bash
code-rag config --url "http://your-proxy:8383/v1" --model "gpt-4o"
```

### 4. Index your Project
```bash
code-rag sync --all
```

### 5. Search
*   **Human Mode (Compact)**: `code-rag search "how to handle errors"`
*   **Agent Mode (JSON)**: `code-rag --json search "data storage" --limit 1`

### 🐳 Docker (Alternative)
```bash
docker build -t agent-coderag .
docker run -v ~/.cache/code-rag:/root/.cache/code-rag agent-coderag setup
```

---

## 🤖 For AI Agents

Agent-CodeRAG is built specifically for programmatic consumption.

### Agent Strategy
1.  **Search First**: Use `code-rag --json search "topic"` to find relevant code units before reading files.
2.  **Use Intent**: The `summary` field provides technical intent, allowing you to skip reading complex implementation details.

---

## 🔧 Development

### Running Tests
```bash
pytest tests/
pytest e2e_tests/
```

### Pre-commit Hooks
We use `pre-commit` to maintain high code standards:
```bash
pip install pre-commit
pre-commit install
```

---

## 📄 License
MIT © 2026 Igor Boloban

## 🙏 Acknowledgments
This project stands on the shoulders of giants. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for a full list of open-source libraries used in Agent-CodeRAG.
