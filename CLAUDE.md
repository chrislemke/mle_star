# CLAUDE.md - Operational Guide

This file contains operational knowledge for the AI agent. Update this as you learn about the codebase.

---

## Commands

### Build
```bash
uv sync
```

### Test & Coverage
```bash
uv run pytest
```
> Includes coverage (90% minimum, term-missing report) via addopts in pyproject.toml.
> Use hypothesis for property-based testing.

### Typecheck
```bash
uv run mypy --config-file=pyproject.toml src tests
```

### Lint
```bash
uv run ruff format
uv run ruff check . --fix
```

### Check
```bash
uv run xenon --max-average=B --max-modules=B --max-absolute=B src
 ```

```bash
uv run bandit -c pyproject.toml -r .
```

### Mutation Testing
```bash
uv run mutmut run
```
> Validates test quality by injecting small bugs and checking if tests catch them. Use `uv run mutmut browse` to inspect surviving mutants.

### Security Audit
```bash
uv run pip-audit
```

### Run
```bash
uv run mle_star
```

---

## Project Structure

[Add project structure here]

---

## Operational Notes

<!-- Add learnings here as you work on the codebase -->

-

---

## Codebase Patterns

<!-- Document patterns and conventions you discover -->

---

## Known Issues

<!-- Track issues that affect development -->

-

---

## Dependencies

<!-- Note important dependencies and their purposes -->

-
