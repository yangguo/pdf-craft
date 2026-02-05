# Repository Guidelines

## Project Structure & Module Organization
- `pdf_craft/`: core library modules (PDF handling, OCR, EPUB/Markdown transforms).
- `tests/`: unit tests; `tests/assets/` holds sample PDFs used by tests.
- `docs/`: contributor and user documentation (installation, development, release).
- `scripts/`: helper utilities (e.g., generate Markdown/EPUB, sync deps).
- Root config: `pyproject.toml`, `.pylintrc`, `.editorconfig`, `test.py`.

## Build, Test, and Development Commands
- `poetry install --with dev`: install runtime + dev dependencies.
- `poetry run pyright pdf_craft tests`: type checking (CI runs this).
- `poetry run pylint pdf_craft tests`: linting (CI runs this).
- `poetry run python test.py`: run unit tests via unittest discovery.
- `poetry run python test.py test_parser`: run a single test file.
- `poetry build`: build sdist/wheel.

Note: Poppler and PyTorch are required for real OCR runs; see `docs/DEVELOPMENT.md` for CUDA/CPU setup details.

## Coding Style & Naming Conventions
- Follow `.editorconfig`: 4-space indents, LF endings, trim trailing whitespace.
- Prefer Python lines ≤ 100 chars (editor rule); pylint allows up to 120.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Keep type hints accurate; CI enforces `pyright`.

## Testing Guidelines
- Tests use `unittest`; files are named `test_*.py`.
- Add focused unit tests for new behavior and fixtures under `tests/assets/` when PDFs are needed.
- Run locally with `poetry run python test.py` before opening a PR.

## Commit & Pull Request Guidelines
- Commit messages follow Conventional Commits: `feat: add xyz (#123)`, `fix: ...`, `docs: ...`, `chore: ...`.
- PRs should include a clear summary, test/lint status, and any doc updates for behavior changes.
- Keep PRs scoped; call out GPU/Poppler-related impacts when relevant.
