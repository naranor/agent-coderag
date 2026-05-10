.PHONY: install test lint format clean sync search api setup rebuild

PYTHON = .venv/Scripts/python.exe
PYTEST = .venv/Scripts/pytest.exe
RUFF = .venv/Scripts/ruff.exe
MYPY = .venv/Scripts/mypy.exe

install:
	$(PYTHON) -m pip install -e ".[dev]"
	.venv/Scripts/pre-commit.exe install

test:
	$(PYTHON) -m pytest --verbose --cov --cov-branch --cov-report=xml

lint:
	$(PYTHON) -m prospector code_rag --profile .prospector.yaml --with-tool mypy --with-tool bandit

format:
	$(RUFF) format .
	$(RUFF) check --fix .

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage .ruff_cache .mypy_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +

sync:
	$(PYTHON) -m code_rag.entry.cli sync --all

search:
	$(PYTHON) -m code_rag.entry.cli search "$(query)"

api:
	$(PYTHON) -m code_rag.entry.cli api "$(lib)"

setup:
	$(PYTHON) -m code_rag.entry.cli setup

rebuild:
	$(PYTHON) -m code_rag.entry.cli rebuild
