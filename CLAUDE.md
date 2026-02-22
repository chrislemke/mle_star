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
  phase1.py            # Phase 1 agents + orchestration: retrieve_models, generate_candidate, merge_solutions, run_phase1 (Tasks 27, 28)
  phase2_inner.py      # Phase 2 inner loop: coder/planner agents + run_phase2_inner_loop orchestration with safety integration (Tasks 23, 24, 25)
  phase2_outer.py      # Phase 2 outer loop: ablation agent, summarize agent, extractor agent, code block validation, run_phase2_outer_loop orchestration (Tasks 31, 32, 33)
  phase3.py            # Phase 3 agents: invoke_ens_planner, invoke_ensembler + formatting helpers (Task 35)
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
  test_phase1_agents.py          # Tests for Phase 1 agents: retriever, init, merger (Task 27)
  test_phase1_orchestration.py   # Tests for run_phase1 orchestration (Task 28)
  test_phase1_safety.py          # Tests for Phase 1 post-merge safety checks (Task 29)
  test_phase2_inner_agents.py    # Tests for coder and planner agents (Task 23)
  test_phase2_inner_loop.py      # Tests for run_phase2_inner_loop orchestration (Task 24)
  test_phase2_inner_safety.py    # Tests for inner loop safety integration (Task 25)
  test_phase2_outer_ablation.py  # Tests for ablation agent invocation and execution (Task 31)
  test_phase2_outer_agents.py    # Tests for summarize and extractor agents (Task 32)
  test_phase2_outer_loop.py      # Tests for run_phase2_outer_loop orchestration (Task 33)
  test_phase3_agents.py          # Tests for Phase 3 ensemble planner and ensembler agents (Task 35)
```

---

## Operational Notes

<!-- Add learnings here as you work on the codebase -->

- `asyncio.shield(communicate_task)` inside `asyncio.wait_for` preserves partial stdout/stderr on timeout — without shield, the communicate task is cancelled and buffered data is lost
- `start_new_session=True` on subprocess creates a new process group, enabling `os.killpg` to clean up orphan child processes on timeout
- Agent invocation pattern: `PromptRegistry().get(AgentType.X)` → `template.render(**vars)` → `await client.send_message(agent_type=str(AgentType.X), message=prompt)` → parse response
- A_coder uses `extract_code_block()` for response parsing; A_planner returns raw stripped text (no extraction)
- Inner loop passes `list(accumulated_plans)` (copies) to `invoke_planner` to provide a snapshot at invocation time — passing the mutable list directly would let later mutations leak into captured references
- Safety integration pattern in inner loop: `check_and_fix_leakage(candidate, task, client)` → `make_debug_callback(task, config, client)` → `evaluate_with_retry(candidate, task, config, debug_callback)` — leakage check runs before EVERY evaluation, debug retry handles execution errors
- Phase 1 agent pattern: `retrieve_models` parses structured JSON via `RetrieverOutput.model_validate_json()`; `generate_candidate` and `merge_solutions` extract code via `extract_code_block()` and return `SolutionScript | None` (None on empty extraction). Empty-check uses `.strip()` for whitespace-only responses
- Phase 1 orchestration (`run_phase1`): decomposed into `_generate_and_evaluate_candidates` + `_run_merge_loop` + `_apply_post_merge_safety` to stay under xenon complexity threshold B. `_CandidateResults` class accumulates candidate loop state. Merge loop uses `is_improvement_or_equal` (>= semantics) and breaks on first failure/non-improvement
- Post-merge safety pattern: `_apply_safety_check()` generic helper uses identity check (`is not`) to detect modification, re-evaluates if modified, and falls back to pre-check version on failure. Applied twice in sequence: `check_data_usage` (exactly once, REQ-P1-030) then `check_and_fix_leakage` (REQ-P1-031). Phase1Result.initial_score always reflects the final post-safety score
- Ablation script execution uses a custom retry loop (NOT `evaluate_with_retry`) because ablation scripts are informational only — no score parsing, custom timeout, and different error recovery semantics. Timeout formula: `min(time_limit // (outer_steps * 2), 600)` using integer division
- `_format_previous_ablations()` returns empty string for first iteration (omits section from prompt), and numbered markdown for subsequent iterations with header "# Previous Ablation Study Results"
- A_summarize uses full text response (no code block extraction); fallback on empty/whitespace: `"[Auto-summary from raw output] " + raw_output[-2000:]` (REQ-P2O-036)
- A_extractor uses structured output (`ExtractorOutput.model_validate_json`) with one retry on parse failure (REQ-P2O-034); returns `ExtractorOutput | None`
- `validate_code_block(code_block, solution)` is a simple `code_block in solution.content` substring check (REQ-P2O-017)
- `_format_previous_blocks()` mirrors `_format_previous_ablations()` pattern: empty list → empty string, non-empty → numbered markdown with header "# Previously Improved Code Blocks"
- Outer loop orchestration (`run_phase2_outer_loop`): decomposed into `_run_outer_step` helper + `_make_skipped_step` to stay under xenon complexity B. Step helper returns a dict with internal `_new_h_best` and `_new_best_solution` keys (popped by caller). Uses `is_improvement_or_equal` (>= semantics) for best-score update — NOT `InnerLoopResult.improved` (which uses strict >). Skipped iterations (extractor None or validation failure) produce `was_skipped=True` records. Accumulates T_abl and C lists; each `CodeBlock` has `outer_step=t` set. `initial_score: float` is explicit parameter because `SolutionScript.score` is `float | None`
- Phase 3 agent pattern: `invoke_ens_planner` returns raw stripped text (like A_planner), `invoke_ensembler` uses `extract_code_block()` (like A_coder/A_init/A_merger). Both validate `len(solutions) >= 2`. `_format_solutions()` numbers solutions as "# {n}th Python Solution" with fenced code blocks. `_format_ensemble_history()` mirrors `_format_plan_history()` from phase2_inner.py: `## Plan:` / `## Score:` labels, None → "N/A (evaluation failed)", empty list → empty string

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
