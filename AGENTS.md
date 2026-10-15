# AGENTS.md

Instructions for AI coding agents (and a quick reference for people) working in `qccodec`. CONTRIBUTING.md explains the same workflow in more detail.

## This package

`qccodec` holds everything program-specific: one `Codec` per program in `src/qccodec/programs/<program>.py` that encodes a `ProgramInput` into a `SubprocessInput` or `PythonInput` and decodes native output into `qcdata` results. It never executes programs; that is `qccompute`'s job.

- Keywords pass through to the program in its native form, so each program's own documentation applies. The exception is atom indices: they are 0-based in `ProgramInput` (matching `structure`), and each codec converts them for programs that count from 1. List those fields in the codec module's docstring.
- Parsers are small functions registered with `@codec.parser(...)`; keep them independent of each other and of execution.
- Every parser is tested through fixture cases in `tests/data/<program>/<case>/` (`input.json`, `expected.json`, outputs, optional `case.json` and `encoded/`), run by `qccodec.testing`. Add a case for every new parser, program version, or bug fix. See CONTRIBUTING.md.

## The qc* suite

Dependencies point one way: qcref -> qcdata -> {qcinf, qccodec} -> qccompute. Never import a package that sits later in that order, and keep changes that span packages in separate pull requests, released in that order.

## Commands

```sh
uv sync --all-extras --all-groups     # install everything
uv run pre-commit install             # once per clone: run the checks on every commit
uv run pytest                         # tests
uv run ruff check --fix . && uv run ruff format .
uv run mypy
uv run pre-commit run --all-files     # exactly what CI checks
```

Use `uv add` / `uv remove` for dependencies; never edit `uv.lock` by hand.

## Code conventions

- Use type annotations for all function and method parameters (except `self` and `cls`) and return values, including `-> None`. mypy enforces this for library code.
- Annotate class and instance attributes, including dataclass and pydantic fields. Annotate local variables only when inference is insufficient.
- Prefer precise types; avoid `Any` unless genuinely necessary.
- Write Google-style docstrings for all public modules, classes, functions, and methods (ruff enforces this). Start with a one-line summary ending in a period; add Args, Returns, Yields, and Raises sections where applicable; describe meaning and units without repeating types already in annotations.
- Use `X | None`, not `Optional[X]`. Target Python 3.10.
- Keep changes focused. Do not reformat or refactor unrelated code.

## Before you finish

- `uv run pre-commit run --all-files` and `uv run pytest` pass.
- New behavior has tests; bug fixes have a test that failed before the fix.
- User-visible changes are listed under `## [unreleased]` in CHANGELOG.md.
- `pyproject.toml` has no `[tool.uv.sources]` path or git entries for qc* packages (CI rejects them).
- Do not commit, push, tag, or release unless asked to.
