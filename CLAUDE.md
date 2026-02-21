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
  execution.py         # Execution harness: env setup, working dir, GPU, async script exec, output parsing, evaluation pipeline, subsampling utilities, submission verification, batch evaluation, solution ranking (Tasks 11-17)
  safety.py            # Safety modules: debugger agent, leakage agent, data agent, code block extraction (Tasks 19, 20, 21)
  phase2_inner.py      # Phase 2 inner loop: coder and planner agent invocations (Task 23)
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
  test_execution_env.py        # Tests for working directory & environment setup (Task 11)
  test_execution_script_write.py # Tests for script writing & validation (Task 12)
  test_execution_async.py        # Tests for async script execution (Task 13)
  test_execution_output.py       # Tests for output parsing & evaluation result (Task 14)
  test_execution_eval.py         # Tests for evaluation pipeline, retry, score comparison (Task 15)
  test_execution_subsample.py    # Tests for subsampling utilities (Task 16)
  test_execution_submission.py   # Tests for submission verification, batch eval, ranking (Task 17)
  test_safety_debugger.py        # Tests for debugger safety agent (Task 19)
  test_safety_data.py            # Tests for data usage verification agent (Task 21)
  test_safety_leakage.py         # Tests for leakage detection/correction agent (Task 20)
  test_phase2_inner_agents.py    # Tests for coder and planner agents (Task 23)
```

---

## Operational Notes

<!-- Add learnings here as you work on the codebase -->

- `asyncio.shield(communicate_task)` inside `asyncio.wait_for` preserves partial stdout/stderr on timeout — without shield, the communicate task is cancelled and buffered data is lost
- `start_new_session=True` on subprocess creates a new process group, enabling `os.killpg` to clean up orphan child processes on timeout
- Agent invocation pattern: `PromptRegistry().get(AgentType.X)` → `template.render(**vars)` → `await client.send_message(agent_type=str(AgentType.X), message=prompt)` → parse response
- A_coder uses `extract_code_block()` for response parsing; A_planner returns raw stripped text (no extraction)

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
