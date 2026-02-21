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

```
src/mle_star/
  __init__.py          # Package init
  cli.py               # CLI entry point (uv run mle_star)
  models.py            # Pydantic data models (enums, configs, schemas)
  scoring.py           # Score parsing, comparison functions, ScoreFunction protocol (Task 07)
  prompts/             # YAML prompt templates for 14 agents
    __init__.py        # PromptRegistry class (Task 08)
    *.yaml
tests/
  __init__.py
  test_models_core.py          # Tests for core config models (Task 03)
  test_models_results.py       # Tests for evaluation & phase result models (Task 06)
  test_models_scoring.py       # Tests for score interface & comparisons (Task 07)
  test_models_agent_config.py  # Tests for AgentConfig & build_default_agent_configs (Task 09)
  test_prompt_system.py        # Tests for PromptTemplate & PromptRegistry (Task 08)
```

---

## Operational Notes

<!-- Add learnings here as you work on the codebase -->

-

---

## Codebase Patterns

- All Pydantic models use `ConfigDict(frozen=True)` for immutability (except `SolutionScript`)
- StrEnum for all string enums (TaskType, DataModality, MetricDirection, etc.)
- Google-style docstrings on all public classes and functions
- `field_validator` for simple field constraints, `model_validator` for cross-field validation
- Tests use `@pytest.mark.unit` markers and hypothesis for property-based testing
- YAML prompt templates have two formats: single-template (top-level keys) and multi-template (`templates:` list with `variant` key)
- `PromptRegistry` loads from `src/mle_star/prompts/*.yaml` using `pathlib.Path(__file__).parent`

---

## Known Issues

<!-- Track issues that affect development -->

-

---

## Dependencies

<!-- Note important dependencies and their purposes -->

-
