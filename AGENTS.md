# Repository Guidelines

## Project Structure
- `paddle_pipeline/`: core package — PaddleOCR/MinerU PDF-to-EPUB conversion pipeline.
- `tests/`: unit tests; `tests/assets/` holds sample PDFs used by tests.
- `skills/`: Claude Code skill definitions (hybrid text+visual EPUB pipeline).
- `docs/`: documentation and design plans.
- Root config: `pyproject.toml`, `.pylintrc`, `.editorconfig`.

## Build, Test, and Development Commands
- `pip install pymupdf ebooklib python-dotenv requests`: install runtime dependencies.
- `python3 -m pytest tests/ -v`: run all tests.
- `python3 -m pytest tests/test_*.py -v`: run a single test file.
- `python3 -m pyright paddle_pipeline/`: type checking.

## Coding Style & Naming Conventions
- Follow `.editorconfig`: 4-space indents, LF endings, trim trailing whitespace.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Internal functions are prefixed `_`; public API re-exported in `__init__.py`.

## Testing Guidelines
- Tests use `pytest`; files are named `test_*.py`.
- Mock targets must use `mod.submodule.function` paths matching the package structure.

## Commit Guidelines
- Commit messages follow Conventional Commits: `feat:`, `fix:`, `docs:`, `chore:`.
