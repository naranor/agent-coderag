# Contributing to CodeRAG

Thank you for your interest in contributing to CodeRAG!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/naranor/agent-coderag
cd agent-coderag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install in editable mode
pip install -e .
```

## Running Tests

```bash
# Run unit tests
pytest

# Run E2E tests
pytest e2e_tests/
```

## Code Style

- Use type hints for all new functions
- Follow PEP 8
- Run linting: `ruff check .`

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and add tests
4. Commit with a clear message
5. Push to your fork and submit a PR

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `chore:` for maintenance
- `test:` for adding or fixing tests

## Creating a Pull Request

- Provide a clear title and description
- Reference any related issues (e.g., `Closes #123`)
- Ensure all CI checks are green before requesting a review
- Keep PRs focused on a single change

## License

By contributing to CodeRAG, you agree that your contributions will be licensed under the MIT License.