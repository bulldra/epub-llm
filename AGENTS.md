# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Core app (`app.py`), EPUB/RAG utilities (`epub_util.py`, `mlx_embedding_service.py`, `simple_epub_service.py`, `rag_util.py`, `history_util.py`), MCP tools (`mcp_server.py`), FAISS router (`mlx_faiss_integration.py`).
- `templates/`, `static/`: Web UI assets.  `config/`: `app_config.yaml`.
- `epub/`: Put `.epub` files.  `cache/`, `log/`: Generated indices/history/logs.
- `test/`: Pytest suite.  Helpers: `run.sh`, `run_server.py`.

## Build, Test, and Development Commands
- Setup venv: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt -r requirements-dev.txt`
- Run server: `uvicorn src.app:app --reload` or `./run.sh`
- Tests: `pytest -q` (coverage: `pytest --cov=src`)
- Lint/format: `black src && ruff format . && ruff check . && flake8 src && pylint src`
- Types: `mypy src`

## Coding Style & Naming Conventions
- Python 3.12, PEP 8, 4-space indents, type hints for new/changed code.
- 88-character line length (Black/Ruff). Prefer `snake_case`, `PascalCase`, `UPPER_CASE`.
- Keep modules focused; avoid one-letter names. Use `logging` over `print`.

## Testing Guidelines
- Framework: Pytest (`test/test_*.py`). Use t-wada TDD: red→green→refactor.
- Add/adjust tests alongside changes; avoid network; prefer unit over integration.
- Use `tmp_path`/`tempfile`; mock external I/O.

## Commit & Pull Request Guidelines
- Commit: short, imperative summary (JP/EN). Example: `epub_util: improve TOC filtering`.
- Separate functional vs. formatting changes.
- PRs: clear description, rationale, linked issues, screenshots for UI, and test notes.

## Security & Configuration Tips
- Configure via env or `config/app_config.yaml` overrides; never commit secrets.
- Set `mlx.embedding_model` and LM Studio settings before chat/RAG.
- Do not hand-edit data in `cache/`/`log/`.

## Architecture Overview
- FastAPI app exposes bookshelf/upload/search/chat; EPUB→Markdown cached in `cache/*.md`.
- FAISS index in `cache/` powers per-book/cross-book search; RAG builds system prompts from top‑k snippets.

## Copilot/Agent Instructions Alignment
- Follow `.github/copilot-instructions.md`: Python 3.12, type hints, 88-char lines.
- Prefer Japanese for user-facing text; code/comments minimal and purposeful.
- TDD (pytest), and run before PR: `mypy`, `black`, `ruff check`, `flake8`, `pylint`.
- Avoid `import-outside-toplevel` and `pylint: disable` unless strictly necessary and justified.
