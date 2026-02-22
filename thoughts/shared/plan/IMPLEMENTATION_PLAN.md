# Implementation Plan

> This file tracks all tasks for the project. Update it as tasks are completed.

---

## Summary

| Priority | Pending | In Progress | Done |
|----------|---------|-------------|------|
| P1       | 5       | 0           | 39   |
| P2       | 8       | 0           | 0    |
| P3       | 0       | 0           | 0    |

---

## Gap Analysis

**Current state:** The codebase contains only a skeleton CLI (`src/mle_star/cli.py`) and an empty `__init__.py`. No production dependencies, no tests, and no implementation code exist. The entire MLE-STAR pipeline (436 requirements across 9 SRS documents) must be built from scratch.

**Dependency order (bottom-up):**

```
Layer 0: Project Setup (dependencies, prompt templates)
Layer 1: Spec 01 — Data Models (foundation for all)
Layer 2: Spec 02 — Execution Harness (depends on 01)
Layer 3: Spec 03 — Safety Modules (depends on 01 + 02)
Layer 4: Phase Implementations (depend on 01 + 02 + 03)
  ├─ Spec 06 — Phase 2 Inner Loop
  ├─ Spec 04 — Phase 1
  ├─ Spec 05 — Phase 2 Outer Loop (depends on Spec 06)
  ├─ Spec 07 — Phase 3 Ensemble
  └─ Spec 08 — Finalization
Layer 5: Spec 09 — Orchestrator (depends on all above)
```

**Key cross-cutting constraints (apply to all tasks):**
- REQ-SF-022: `check_and_fix_leakage()` must run before EVERY evaluation across ALL phases
- REQ-EX-011: `parse_score()` returns the LAST match, not the first
- REQ-P2I-021/022/023: Inner loop always uses ORIGINAL code block and ORIGINAL solution for replacement
- REQ-OR-008: Tool assignments from orchestrator spec override individual spec tool lists (e.g., REQ-SF-001 says `["Read", "Bash"]` for A_debugger, but REQ-OR-008 mandates `["Bash", "Edit", "Write", "Read"]`)
- REQ-DM-039: All Pydantic models frozen except SolutionScript
- All phase functions receive a `client` parameter (the shared `ClaudeSDKClient` instance)
- REQ-SF-006/008 layering: `debug_solution()` returns the final attempted pair (may still be broken); the **calling code** (evaluate_with_retry / phase orchestration) maintains the reference to the last known working version and performs fallback
- Score comparison semantics vary by context: `is_improvement_or_equal()` (>=) for best-score tracking within loops AND outer loop `s_final` update (REQ-P2O-027); `is_improvement()` (strict >) ONLY for `InnerLoopResult.improved` flag

**SDK client threading:** Several spec function signatures omit the `client` parameter despite requiring SDK agent invocation (REQ-SF-006, REQ-P2I-005, REQ-P2I-013, REQ-P2I-016, REQ-FN-009, REQ-FN-019, REQ-FN-034). All functions that invoke agents shall receive `client: ClaudeSDKClient` as a parameter. This applies to: phase-level entry points (`run_phase1`, `run_phase2_outer_loop`, `run_phase2_inner_loop`, `run_phase3`, `run_finalization`) and agent invocation functions (`invoke_coder`, `invoke_planner`, `debug_solution`, `make_debug_callback`, `remove_subsampling`, `generate_test_submission`, `check_contamination`). The plan adds `client` to these signatures as an implementation necessity; specs that omit it are treated as under-specified.

**Score mutation pattern:** The execution harness does NOT mutate `SolutionScript` (REQ-EX-016). Phase orchestration code is responsible for updating `solution.score = result.score` after evaluation. `SolutionScript` is non-frozen (REQ-DM-039) specifically to allow this.

**Session ID strategy:** Phase 1: `"phase-1"`, Phase 2 paths: `"path-{i}"` (REQ-OR-021), Phase 3: `"phase-3"`, Finalization: `"finalization"`. Phase 2 paths may fork from Phase 1 session via `fork_session=True`.

**Fallback chain priority:** Phase 3 best > Phase 2 best > Phase 1 best. Specifically: Phase 1 fails → `PipelineError` (REQ-OR-042); Phase 2 path fails → substitute Phase 1 solution (REQ-OR-040); Phase 3 fails → best Phase 2 solution (REQ-OR-041); Finalization fails → `submission_path=""` (REQ-OR-043).

**Time budget redistribution:** After Phase 1 completes, remaining time redistributed proportionally among Phase 2/3/Finalization based on their proportions (65/15/10 → normalized to 72.2/16.7/11.1%). Phase 2 per-path budget = `phase2_budget / L` (REQ-OR-026).

**A_test multi-mode agent:** `AgentType.test` has 4 operational modes sharing the same agent type but with variant-specific prompt templates and config overrides: (1) test submission (default config: `tools=["Read"]`, `output_schema=None`), (2) subsampling extraction (variant `"subsampling_extract"`), (3) subsampling removal (variant `"subsampling_remove"`), (4) contamination check (variant `"contamination_check"`, override: `tools=None`, `output_schema=DataContaminationResult`). See REQ-FN-048.

**Project target:** Python 3.13+ (`requires-python = ">=3.13"` in pyproject.toml). Specs reference 3.10+ as a floor, but the project already targets 3.13+. Use modern syntax (type unions `X | Y`, etc.).

**Pre-commit hooks (apply to all tasks):** Every commit must pass the full pre-commit hook chain configured in `.pre-commit-config.yaml`. This imposes cross-cutting requirements beyond what the specs mandate:
- **Ruff lint** with D rules enabled (Google-style docstrings required for all public functions, classes, and methods — `pydocstyle convention = "google"` in pyproject.toml). D100 (module) and D104 (package) are ignored, but D101 (class), D102 (method), D103 (function) are enforced. Every task producing public APIs must include Google-style docstrings.
- **Ruff format** auto-formats on commit (line-length=88, double quotes, space indent).
- **Mypy strict mode** (`strict = true`, `disallow_untyped_defs = true`) — all function signatures must have complete type annotations.
- **Bandit security scan** — no security anti-patterns (subprocess with `shell=True`, hardcoded passwords, etc.). B404/B603 are pre-skipped for safe subprocess usage.
- **Xenon code complexity** — max average B, max modules B, max absolute B. Functions must stay below cyclomatic complexity threshold B (≤5). Complex orchestration functions may need decomposition.
- **pip-audit** — all production dependencies must be free of known vulnerabilities at commit time.
- Each task's acceptance criteria ("Tests pass with ≥90% coverage; mypy clean") implicitly includes "all pre-commit hooks pass" as a gate.

---

## Tasks

### Layer 0 — Project Setup

---

## [P1] 01. Add production dependencies
**Status:** done
**Priority:** P1

### Description
Add required production dependencies to `pyproject.toml`. The project currently has zero production dependencies — only dev dependencies are configured. The MLE-STAR pipeline requires Pydantic v2 for data models and JSON schema generation, PyYAML for prompt template loading, and the `claude-agent-sdk` for agent invocation. All data model schemas, structured outputs, and SDK integrations depend on these packages being available.

### Acceptance Criteria
- [x] `pydantic>=2.0.0` added to `[project] dependencies`
- [x] `pyyaml>=6.0` added to `[project] dependencies`
- [x] `claude-agent-sdk>=0.1.39` added to `[project] dependencies`
- [x] `uv sync` succeeds without errors
- [x] `uv run python -c "import pydantic; import yaml; print('ok')"` succeeds

---

## [P1] 02. Create prompt template YAML files for all 14 agents
**Status:** done
**Priority:** P1

### Description
Create YAML prompt template files containing the prompt text for each of the 14 MLE-STAR agents, based on the paper's Figures 9–28. Templates use `{variable}` placeholders for runtime substitution. Store in `src/mle_star/prompts/` directory. This task provides the prompt content consumed by `PromptRegistry` (REQ-DM-032 to REQ-DM-035). Includes dual templates for leakage (detection + correction) and subsampling (extract + remove).

**Depends on:** None (content authoring only)

### Acceptance Criteria
- [x] YAML files exist for all 14 agent types: retriever, init, merger, ablation, summarize, extractor, coder, planner, ens_planner, ensembler, debugger, leakage, data, test
- [x] Leakage agent has two template variants: `detection` and `correction` (Figures 20, 21)
- [x] Test agent has three additional variants: `subsampling_extract` (Figure 26), `subsampling_remove` (Figure 27), and `contamination_check` (Figure 28)
- [x] Each template specifies: `agent_type`, `figure_ref`, `template`, `variables`
- [x] All `{variable}` placeholders in templates match the `variables` list
- [x] Total of 18 prompt templates across the YAML files (11 base + 2 leakage variants + 1 test base + 3 test variants + 1 data = 18)
- [x] `src/mle_star/prompts/__init__.py` exists to make prompts a proper Python package (see also Task 52)

---

### Layer 1 — Data Models (Spec 01: `mle_star/models.py`)

---

## [P1] 03. Core configuration models
**Status:** done
**Priority:** P1

### Description
Implement the foundational configuration Pydantic models in `src/mle_star/models.py`. Covers `PipelineConfig` with paper-specified defaults (M=4, T=4, K=4, L=2, R=5), validation (all ints > 0), and JSON serialization. Also covers `TaskType`, `DataModality`, and `MetricDirection` string enums, and the `TaskDescription` model with all required fields. `PipelineConfig` must also include orchestrator-level fields from Spec 09: `max_budget_usd` (REQ-OR-028), `permission_mode` (REQ-OR-009), `model` (REQ-OR-044), and `log_level` (REQ-OR-047).

**Spec:** SRS 01 | **Reqs:** REQ-DM-001 to REQ-DM-007, REQ-OR-009, REQ-OR-025, REQ-OR-028, REQ-OR-044

### Acceptance Criteria
- [x] `PipelineConfig()` with no args produces paper defaults (M=4, T=4, K=4, L=2, R=5, time_limit=86400, subsample_limit=30000, max_debug_attempts=3)
- [x] `PipelineConfig` includes `max_budget_usd: float | None = None` (REQ-OR-028)
- [x] `PipelineConfig` includes `permission_mode: str = "bypassPermissions"` (REQ-OR-009)
- [x] `PipelineConfig` includes `model: str = "sonnet"` (REQ-OR-044)
- [x] `PipelineConfig` includes `log_level: str = "INFO"` (REQ-OR-047)
- [x] `PipelineConfig` includes `log_file: str | None = None` for optional file handler (REQ-OR-047)
- [x] `PhaseTimeBudget` model defined in `models.py` with fields: `phase1_pct: float = 10.0`, `phase2_pct: float = 65.0`, `phase3_pct: float = 15.0`, `finalization_pct: float = 10.0`; validator ensures sum equals 100.0; frozen model
- [x] `PipelineConfig` includes `phase_time_budget: PhaseTimeBudget | None = None` (REQ-OR-025)
- [x] `PipelineConfig(num_retrieved_models=0)` raises `ValidationError`
- [x] Round-trip JSON serialization preserves all `PipelineConfig` field values
- [x] `TaskType` enum has 8 values; `DataModality` has 5; `MetricDirection` has 2
- [x] `TaskDescription` has fields: competition_id, task_type, data_modality, evaluation_metric, metric_direction, description, data_dir (default `"./input"`), output_dir (default `"./final"`)
- [x] `TaskDescription` missing required field raises `ValidationError`
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 04. Solution and code block models
**Status:** done
**Priority:** P1

### Description
Implement `SolutionPhase` enum, `SolutionScript` model (with `content`, `phase`, `score`, `is_executable`, `source_model`, `created_at` fields and `replace_block()` method), `CodeBlockCategory` enum, and `CodeBlock` model. `SolutionScript.replace_block()` returns a new instance with the first occurrence of `old` replaced by `new`, raising `ValueError` if not found. `SolutionScript` is the only non-frozen model (REQ-DM-039).

**Spec:** SRS 01 | **Reqs:** REQ-DM-008 to REQ-DM-012

### Acceptance Criteria
- [x] `SolutionPhase` has 5 values: init, merged, refined, ensemble, final
- [x] `SolutionScript` fields: content (str), phase (SolutionPhase), score (float|None), is_executable (bool, default True), source_model (str|None), created_at (datetime, auto)
- [x] `SolutionScript` uses `frozen=False` (mutable for score updates)
- [x] `replace_block(old, new)` returns new `SolutionScript` with first occurrence substitution
- [x] `replace_block()` raises `ValueError` when `old` not in content
- [x] `CodeBlockCategory` has 8 values; `CodeBlock` has content (str), category (CodeBlockCategory|None), outer_step (int|None)
- [x] `CodeBlock` uses `frozen=True`
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 05. Agent type enum and structured output schemas
**Status:** done
**Priority:** P1

### Description
Implement the `AgentType` enum with exactly 14 values and all structured output schemas used by agents requiring JSON output. Includes: `RetrievedModel` (model_name, example_code), `RetrieverOutput` (models list with validator `len >= 1`), `RefinePlan` (code_block, plan), `ExtractorOutput` (plans list with validator `len >= 1`), `LeakageAnswer` (leakage_status Literal, code_block), `LeakageDetectionOutput` (answers list with validator `len >= 1`), and `DataContaminationResult` (verdict Literal).

**Spec:** SRS 01 | **Reqs:** REQ-DM-013 to REQ-DM-020

### Acceptance Criteria
- [x] `len(AgentType)` equals 14; values match paper agent names
- [x] `RetrievedModel` has fields: `model_name: str`, `example_code: str` (REQ-DM-014)
- [x] `RetrieverOutput` has `models: list[RetrievedModel]` with `len >= 1` validator (REQ-DM-015)
- [x] `RefinePlan` has fields: `code_block: str`, `plan: str` (REQ-DM-016)
- [x] `ExtractorOutput` has `plans: list[RefinePlan]` with `len >= 1` validator (REQ-DM-017)
- [x] `LeakageAnswer` has `leakage_status: Literal["Yes Data Leakage", "No Data Leakage"]` AND `code_block: str` (REQ-DM-018)
- [x] `LeakageDetectionOutput` has `answers: list[LeakageAnswer]` with `len >= 1` validator (REQ-DM-019)
- [x] `DataContaminationResult` has `verdict: Literal["Novel", "Same"]` (REQ-DM-020)
- [x] All schemas produce valid JSON via `model_json_schema()` (REQ-DM-041)
- [x] Validators reject empty lists (`RetrieverOutput(models=[])` raises error)
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 06. Evaluation and phase result models
**Status:** done
**Priority:** P1

### Description
Implement `EvaluationResult` model (score, stdout, stderr, exit_code, duration_seconds, is_error, error_traceback) and all four phase result models: `Phase1Result`, `Phase2Result`, `Phase3Result`, and `FinalResult`. These are the primary data containers flowing between pipeline phases.

**Spec:** SRS 01 | **Reqs:** REQ-DM-021 to REQ-DM-025

### Acceptance Criteria
- [x] `EvaluationResult` has all 7 fields with correct types (REQ-DM-021)
- [x] `Phase1Result` has: retrieved_models, candidate_solutions, candidate_scores, initial_solution, initial_score (REQ-DM-022)
- [x] `Phase2Result` has: ablation_summaries, refined_blocks, best_solution, best_score, step_history (REQ-DM-023)
- [x] `Phase3Result` has: input_solutions, ensemble_plans, ensemble_scores, best_ensemble, best_ensemble_score (REQ-DM-024)
- [x] `FinalResult` has: task, config, phase1, phase2_results (list), phase3 (optional None), final_solution, submission_path, total_duration_seconds, total_cost_usd (REQ-DM-025)
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 07. Score interface and comparison functions
**Status:** done
**Priority:** P1

### Description
Implement the `ScoreFunction` protocol class, the default score parsing function using regex `r"Final Validation Performance:\s*([\d.eE+-]+)"`, and two comparison functions: `is_improvement()` (strict) and `is_improvement_or_equal()` (non-strict). Both respect `MetricDirection` — maximize means higher is better, minimize means lower is better. **IMPORTANT**: Score parsing must return the LAST match when multiple exist (REQ-EX-011 takes precedence over REQ-DM-027).

**Spec:** SRS 01 | **Reqs:** REQ-DM-026 to REQ-DM-029

### Acceptance Criteria
- [x] `ScoreFunction` is a `Protocol` with `__call__(solution, task) -> EvaluationResult`
- [x] Score parsing of `"Final Validation Performance: 0.8196"` returns `0.8196`
- [x] Score parsing returns `None` when pattern not found
- [x] Score parsing of `"...Performance: 0.5\n...Performance: 0.8196"` returns `0.8196` (LAST match per REQ-EX-011)
- [x] `is_improvement(0.9, 0.8, "maximize")` returns `True`
- [x] `is_improvement(0.9, 0.8, "minimize")` returns `False`
- [x] `is_improvement_or_equal(0.8, 0.8, "maximize")` returns `True`
- [x] Tests include property-based testing with hypothesis
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 08. Prompt template system
**Status:** done
**Priority:** P1

### Description
Implement `PromptTemplate` Pydantic model (agent_type, figure_ref, template, variables) with `render(**kwargs)` method, and `PromptRegistry` class with `get(agent_type, variant=None)` method. Registry loads templates from YAML files and supports dual templates for leakage (detection/correction variants) and subsampling (extract/remove variants). `render()` raises `KeyError` for missing variables.

**Spec:** SRS 01 | **Reqs:** REQ-DM-030 to REQ-DM-035 | **Depends on:** Task 02

### Acceptance Criteria
- [x] `PromptTemplate.render(M=4, task_description="classify images")` substitutes placeholders
- [x] `PromptTemplate.render()` raises `KeyError` for missing required variables
- [x] `PromptRegistry.get(AgentType.retriever)` returns retriever template
- [x] `PromptRegistry.get(AgentType.leakage, variant="detection")` returns detection template
- [x] `PromptRegistry.get(AgentType.leakage, variant="correction")` returns correction template
- [x] `PromptRegistry.get(AgentType.test, variant="subsampling_extract")` works
- [x] `len(registry)` covers all 14 agent types (18 total template variants)
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 09. Agent config and SDK integration types
**Status:** done
**Priority:** P1

### Description
Implement `AgentConfig` Pydantic model mapping MLE-STAR agents to SDK configuration (agent_type, description, system_prompt, tools, model, output_schema, max_turns). Provide `to_agent_definition()` returning a dict for `ClaudeAgentOptions.agents`, `to_output_format()` returning `{"type": "json_schema", "schema": ...}` when output_schema is set, and `build_default_agent_configs()` factory returning pre-configured configs for all 14 agent types. **Tool assignments must match REQ-OR-008** (orchestrator spec), which gives execution-capable agents `["Bash", "Edit", "Write", "Read"]`.

**Spec:** SRS 01 | **Reqs:** REQ-DM-036 to REQ-DM-040, REQ-OR-008

### Acceptance Criteria
- [x] `AgentConfig` has all 7 fields with correct types
- [x] `to_agent_definition()` returns dict with keys: description, prompt, tools, model
- [x] `to_output_format()` returns JSON schema dict when output_schema is set
- [x] `to_output_format()` returns `None` when output_schema is `None`
- [x] `len(build_default_agent_configs())` equals 14
- [x] Retriever config has `tools=["WebSearch", "WebFetch"]` and output_schema=RetrieverOutput
- [x] A_init, A_merger, A_abl, A_coder, A_ensembler, A_debugger, A_test have `tools=["Bash", "Edit", "Write", "Read"]` (REQ-OR-008)
- [x] A_summarize, A_extractor, A_planner, A_ens_planner, A_leakage, A_data have `tools=["Read"]` (REQ-OR-008)
- [x] A_extractor has output_schema=ExtractorOutput; A_leakage has output_schema=LeakageDetectionOutput; A_data has output_schema=None (REQ-SF-025; `DataContaminationResult` is used by contamination check variant of A_test per REQ-FN-026/048)
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 10. Iteration records and model constraints
**Status:** done
**Priority:** P1

### Description
Implement `RefinementAttempt` (plan, score, code_block, was_improvement), `EnsembleAttempt` (plan, score, solution), and `InnerLoopResult` (best_solution, best_score, attempts, improved) models. Apply cross-cutting model constraints: JSON schema generation compatibility for all structured output models (REQ-DM-041), `ConfigDict(frozen=True)` on all models except SolutionScript (REQ-DM-039), and verify all models reside in single module `mle_star/models.py` (REQ-DM-045). Re-export public types from `__init__.py` (REQ-DM-046).

**Spec:** SRS 01 | **Reqs:** REQ-DM-039, REQ-DM-041 to REQ-DM-050, REQ-P2I-036

### Acceptance Criteria
- [x] `RefinementAttempt` has 4 fields: plan (str), score (float|None), code_block (str), was_improvement (bool) (REQ-DM-042)
- [x] `EnsembleAttempt` has 3 fields: plan (str), score (float|None), solution (SolutionScript) (REQ-DM-043)
- [x] `InnerLoopResult` has 4 fields: best_solution (SolutionScript), best_score (float), attempts (list[RefinementAttempt]), improved (bool) (REQ-P2I-036)
- [x] `InnerLoopResult.improved` is True iff best_score is strictly better than input (uses `is_improvement`, not `is_improvement_or_equal`)
- [x] All structured output schemas produce valid JSON via `model_json_schema()`
- [x] Frozen models raise error on attribute assignment
- [x] `SolutionScript` is mutable (not frozen)
- [x] All models in single `mle_star/models.py` module
- [x] Public types re-exported from `mle_star/__init__.py`
- [x] Tests pass with ≥90% coverage; mypy clean

---

### Layer 2 — Execution Harness (Spec 02: `mle_star/execution.py`)

---

## [P1] 11. Working directory and environment setup
**Status:** done
**Priority:** P1

### Description
Implement environment setup functions in `src/mle_star/execution.py`: `setup_working_directory(base_path)` creates the working directory structure, `clean_output_directory(base_path)` removes previous outputs, `detect_gpu_info()` detects available GPUs via CUDA subprocess probe, and `build_execution_env(gpu_indices)` constructs environment variables (PYTHONUNBUFFERED=1, PYTHONHASHSEED=0, CUDA_VISIBLE_DEVICES).

**Spec:** SRS 02 | **Reqs:** REQ-EX-001 to REQ-EX-004 | **Depends on:** Spec 01

### Acceptance Criteria
- [x] `setup_working_directory()` creates required directory structure
- [x] `clean_output_directory()` removes previous output files
- [x] `detect_gpu_info()` returns dict with GPU information (or empty if none); graceful fallback on no CUDA
- [x] `build_execution_env()` returns dict with PYTHONUNBUFFERED=1, PYTHONHASHSEED=0, CUDA_VISIBLE_DEVICES
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 12. Script writing and validation
**Status:** done
**Priority:** P1

### Description
Implement `write_script(solution, working_dir, filename)` that writes a `SolutionScript` to disk with validation. Pre-write validation checks: (1) content not empty after stripping, (2) no `exit()` or `sys.exit()` calls (matched via regex `r"\bexit\s*\("` and `r"\bsys\.exit\s*\("`). Raises `ValueError` on validation failure. Returns the absolute path to the written file.

**Spec:** SRS 02 | **Reqs:** REQ-EX-005 to REQ-EX-006 | **Depends on:** Task 11

### Acceptance Criteria
- [x] `write_script()` writes solution content to `working_dir/filename` with UTF-8 encoding
- [x] Returns absolute path to written file
- [x] Raises `ValueError` for empty content (whitespace-only)
- [x] Raises `ValueError` when content contains `exit()`, `sys.exit()`, `os._exit()`, or `quit()` (REQ-EX-006, REQ-EX-044 extended list)
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 13. Async script execution
**Status:** done
**Priority:** P1

### Description
Implement `execute_script(script_path, working_dir, timeout_seconds, env)` as an async function using `asyncio.create_subprocess_exec`. Define `ExecutionRawResult` dataclass for raw output (stdout, stderr, exit_code, duration_seconds, timed_out). Enforce timeout via SIGTERM then SIGKILL after 5s grace period. Ensure resource isolation through subprocess model.

**Spec:** SRS 02 | **Reqs:** REQ-EX-007 to REQ-EX-010 | **Depends on:** Task 12

### Acceptance Criteria
- [x] `execute_script()` is async and runs scripts via subprocess
- [x] `ExecutionRawResult` captures stdout, stderr, exit_code, duration_seconds, timed_out
- [x] Script exceeding timeout: SIGTERM, then SIGKILL after 5s grace (REQ-EX-009)
- [x] Timed-out result has `timed_out=True` and `exit_code=-1`
- [x] Partial stdout/stderr preserved on timeout
- [x] Subprocess inherits only provided environment variables
- [x] Non-zero exit codes do not raise exceptions (captured in result)
- [x] Orphan child process cleanup via `os.killpg` on process group after timeout (REQ-EX-037)
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 14. Output parsing and evaluation result construction
**Status:** done
**Priority:** P1

### Description
Implement output parsing functions: `parse_score(stdout)` extracts float from "Final Validation Performance" pattern (returns LAST match when multiple exist per REQ-EX-011), `extract_traceback(stderr)` extracts the LAST Python traceback, `detect_error(raw)` checks for execution errors (non-zero exit, timeout, or traceback in stderr), and `build_evaluation_result(raw)` constructs an `EvaluationResult` from raw execution output by composing all parsers.

**Spec:** SRS 02 | **Reqs:** REQ-EX-011 to REQ-EX-014 | **Depends on:** Task 13

### Acceptance Criteria
- [x] `parse_score("Final Validation Performance: 0.8196\n")` returns `0.8196`
- [x] `parse_score("...Performance: 0.5\n...Performance: 0.8196\n")` returns `0.8196` (LAST match, REQ-EX-011)
- [x] `parse_score("Training complete.\n")` returns `None`
- [x] `parse_score()` returns `None` on float conversion failure
- [x] `extract_traceback()` extracts LAST traceback from stderr (REQ-EX-012)
- [x] `detect_error()` returns `True` for: non-zero exit code, timed_out, or traceback in stderr (REQ-EX-013)
- [x] `build_evaluation_result()` combines parsers into `EvaluationResult`; `error_traceback` set only when `is_error=True`
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 15. Solution evaluation pipeline
**Status:** done
**Priority:** P1

### Description
Implement the end-to-end evaluation pipeline: `evaluate_solution(solution, task, config, timeout_override)` (async) that writes, executes, and parses a solution script. Also implement `evaluate_with_retry(solution, task, config, debug_callback, max_retries)` (async) that retries evaluation with debug callback on failure, and `is_better_solution(new_result, old_score, direction)` for score comparison delegation. The harness does NOT mutate the input SolutionScript (REQ-EX-016).

**Spec:** SRS 02 | **Reqs:** REQ-EX-015 to REQ-EX-023 | **Depends on:** Task 14

### Acceptance Criteria
- [x] `evaluate_solution()` orchestrates: setup_working_directory → clean_output → write_script → build_env → execute → build_evaluation_result
- [x] `evaluate_solution()` does NOT mutate the input SolutionScript (REQ-EX-016)
- [x] `evaluate_with_retry()` calls debug_callback on failure and retries
- [x] `evaluate_with_retry()` respects max_retries limit (defaults to config.max_debug_attempts)
- [x] `is_better_solution()` delegates to Spec 01 score comparison functions
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 16. Subsampling utilities
**Status:** done
**Priority:** P1

### Description
Implement subsampling-related utilities: `SUBSAMPLE_INSTRUCTION` constant, `get_subsample_instruction(config)` returning the subsampling instruction text with the configured limit, `request_subsample_removal(solution)` and `request_subsample_extraction(solution)` returning prompt text for agent requests. These are used by Phase 1 (init agent) and Finalization (subsampling removal).

**Spec:** SRS 02 | **Reqs:** REQ-EX-017 to REQ-EX-020 | **Depends on:** Spec 01

### Acceptance Criteria
- [x] `SUBSAMPLE_INSTRUCTION` is a constant string with subsampling guidance
- [x] `get_subsample_instruction(config)` includes the config's `subsample_limit` value
- [x] `request_subsample_removal()` returns prompt text for removing subsampling
- [x] `request_subsample_extraction()` returns prompt text for extracting subsampling code
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 17. Submission verification and batch evaluation
**Status:** done
**Priority:** P1

### Description
Implement `verify_submission(working_dir, expected_filename)` to check that `submission.csv` exists and is valid, `get_submission_info(working_dir, expected_filename)` to return metadata about the submission file, `evaluate_batch(solutions, task, config)` (async) for evaluating multiple solutions **sequentially** (not in parallel, per REQ-EX-020), and `rank_solutions(solutions, results, direction)` for sorting by score (None scores at end, is_error after None-scored).

**Spec:** SRS 02 | **Reqs:** REQ-EX-024 to REQ-EX-027 | **Depends on:** Task 15

### Acceptance Criteria
- [x] `verify_submission()` returns `True` when valid submission.csv exists and is non-empty
- [x] `verify_submission()` returns `False` when file missing or invalid
- [x] `get_submission_info()` returns dict with row count, column names, file size
- [x] `evaluate_batch()` evaluates multiple solutions **sequentially** (REQ-EX-026), not concurrently
- [x] `rank_solutions()` returns solutions sorted by score per metric direction; None scores at end
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P2] 18. Execution harness constraints and interface compliance
**Status:** pending
**Priority:** P2

### Description
Implement remaining execution harness requirements: `detect_error_masking(content)` advisory detection (REQ-EX-045), `ExecutorStrategy` enum for selecting execution backends (REQ-EX-034), `execute_script_via_sdk(script_path, working_dir, timeout_ms)` alternative executor using SDK Bash tool (REQ-EX-033), SDK Bash timeout cap at 600,000ms with automatic fallback to subprocess executor (REQ-EX-047), interface compliance verification (SolutionScript/EvaluationResult/TaskDescription/PipelineConfig type contracts, REQ-EX-028 to REQ-EX-032), performance constraints (execution overhead < 2s per invocation excluding script runtime REQ-EX-035, score parsing < 10ms REQ-EX-036), reliability (graceful timeout with orphan cleanup REQ-EX-037, large output handling up to 100MB with truncation REQ-EX-038, UTF-8 `errors="replace"` REQ-EX-043), structured logging (REQ-EX-039), default timeout derivation from config (REQ-EX-046), single module organization (REQ-EX-040), and subprocess-only execution constraint (REQ-EX-041).

**Spec:** SRS 02 | **Reqs:** REQ-EX-028 to REQ-EX-047 | **Depends on:** Tasks 11–17

### Acceptance Criteria
- [ ] `detect_error_masking(content)` identifies broad try/except patterns that suppress errors (REQ-EX-045)
- [ ] `ExecutorStrategy` enum with values `"subprocess"` and `"sdk_bash"` (REQ-EX-034)
- [ ] `execute_script_via_sdk(script_path, working_dir, timeout_ms) -> ExecutionRawResult` using SDK Bash tool interface `{"command": str, "timeout": int}` (REQ-EX-033)
- [ ] SDK Bash executor timeout capped at 600,000ms; falls back to subprocess when timeout exceeds cap (REQ-EX-047)
- [ ] `evaluate_solution()` accepts optional `strategy: ExecutorStrategy` parameter to select backend (REQ-EX-034)
- [ ] All public functions accept/return correct Spec 01 types (REQ-EX-028 to REQ-EX-032)
- [ ] `evaluate_solution()` satisfies `ScoreFunction` protocol when wrapped (REQ-EX-032)
- [ ] Execution overhead < 2 seconds per invocation excluding script runtime (REQ-EX-035)
- [ ] Score parsing executes in < 10ms for stdout up to 1MB (REQ-EX-036)
- [ ] UTF-8 decoding with `errors="replace"` for subprocess output (REQ-EX-043)
- [ ] Large output handling: up to 100MB stdout/stderr with truncation warning (REQ-EX-038)
- [ ] Default timeout derived from `config.time_limit_seconds` when no override provided (REQ-EX-046)
- [ ] Structured logging covers all execution events (REQ-EX-039)
- [ ] No persistent state between executions (REQ-EX-042)
- [ ] Tests pass with ≥90% coverage; mypy clean

---

### Layer 3 — Safety Modules (Spec 03: `mle_star/safety.py`)

---

## [P1] 19. Debugger agent
**Status:** done
**Priority:** P1

### Description
Implement the A_debugger agent in `src/mle_star/safety.py`: agent definition with prompt template from registry (Figure 19), `extract_code_block(response)` utility to extract the longest fenced code block (or full response if no fences), `debug_solution(solution, traceback, task, config, client)` (async) retry loop that invokes A_debugger up to max_debug_attempts times, `make_debug_callback(task, config, client)` factory returning an async callback compatible with `evaluate_with_retry`. Includes subsampling preservation in debug prompt, and appending "Final Validation Performance" print if missing from debugged code (REQ-SF-010).

**Important layering note:** `debug_solution()` returns the final `(SolutionScript, EvaluationResult)` pair per REQ-SF-006 — this may still be a broken solution if all retries failed. The **calling code** (phase orchestration) is responsible for maintaining a reference to the last known working version and performing fallback per REQ-SF-008. `make_debug_callback()` wraps a single debugger invocation (no retry loop) for use with `evaluate_with_retry()`.

**Spec:** SRS 03 | **Reqs:** REQ-SF-001 to REQ-SF-010 | **Depends on:** Spec 01, Spec 02

### Acceptance Criteria
- [x] `extract_code_block()` extracts longest Python code from markdown fences; full response if no fences
- [x] `debug_solution()` invokes A_debugger agent with traceback context
- [x] `debug_solution()` retries up to `config.max_debug_attempts` times
- [x] `debug_solution()` returns the final `(SolutionScript, EvaluationResult)` pair (REQ-SF-006) — may still be broken
- [x] Calling code maintains last known working reference for fallback (REQ-SF-008); if no previous working version exists, returns the failed solution with `is_executable=False`
- [x] If debugged code lacks "Final Validation Performance" pattern, it is appended (REQ-SF-010)
- [x] `make_debug_callback(task, config, client)` returns async callable wrapping a SINGLE debugger invocation (no retry loop) for `evaluate_with_retry`
- [x] Debug prompt includes subsampling preservation instruction ("Do not remove subsampling if exists")
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 20. Leakage detection and correction agent
**Status:** done
**Priority:** P1

### Description
Implement the A_leakage agent with two-step orchestration: detection (using structured output `LeakageDetectionOutput`) and correction (freeform response). Implement `check_and_fix_leakage(solution, task, client)` (async) that first invokes detection. For each `LeakageAnswer` with status "Yes Data Leakage": invoke correction agent, extract code block, and apply fix via `SolutionScript.replace_block()`. If `replace_block` raises `ValueError` (block not found), log warning and skip that replacement (REQ-SF-021). This function runs before EVERY evaluation per REQ-SF-022.

**Spec:** SRS 03 | **Reqs:** REQ-SF-011 to REQ-SF-023 | **Depends on:** Spec 01, Spec 02

### Acceptance Criteria
- [x] Detection agent uses `LeakageDetectionOutput` structured output schema
- [x] Detection prompt loaded from registry with `variant="detection"`
- [x] Correction prompt loaded from registry with `variant="correction"`
- [x] `check_and_fix_leakage()` returns original solution when no leakage found
- [x] `check_and_fix_leakage()` returns corrected solution when leakage found
- [x] Correction uses `SolutionScript.replace_block()` for targeted fix
- [x] `replace_block` ValueError is caught, logged as warning, and skipped (REQ-SF-021)
- [x] Agent failure returns original solution (graceful degradation)
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 21. Data contamination agent
**Status:** done
**Priority:** P1

### Description
Implement the A_data agent: agent definition (Figure 22), `check_data_usage(solution, task, client)` (async) integration function, and `parse_data_agent_response(response, original_solution)`. Parse logic: if response contains "All the provided information is used." (case-insensitive) → return original unchanged; else extract code block and return new SolutionScript. The data agent ensures proper data usage (no try/except masking, correct data paths). **Runs exactly once** — after Phase 1 merge (REQ-SF-030).

**Spec:** SRS 03 | **Reqs:** REQ-SF-024 to REQ-SF-031 | **Depends on:** Spec 01, Spec 02

### Acceptance Criteria
- [x] `check_data_usage()` invokes A_data agent with solution and task context
- [x] `parse_data_agent_response()` returns original when response contains "All the provided information is used."
- [x] `parse_data_agent_response()` extracts corrected solution via `extract_code_block()` otherwise
- [x] Data agent prompt includes "no try/except" instruction
- [x] Returns original solution on agent failure (graceful degradation)
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P2] 22. Safety module constraints
**Status:** pending
**Priority:** P2

### Description
Implement remaining safety module requirements: agent default configs inclusion in `build_default_agent_configs()`, prompt templates loaded from PromptRegistry, type contract enforcement (all safety agents operate on/produce SolutionScript), performance constraints, graceful degradation for all three agents, structured logging, SDK invocation patterns, single module organization, and safety invariants (debugger doesn't add functionality, leakage preserves non-preprocessing code, data agent doesn't suppress errors).

**Spec:** SRS 03 | **Reqs:** REQ-SF-032 to REQ-SF-046 | **Depends on:** Tasks 19–21

### Acceptance Criteria
- [ ] Safety agent configs included in `build_default_agent_configs()`
- [ ] All safety agents load prompts from `PromptRegistry`
- [ ] Graceful degradation: agent failure returns original solution (never raises)
- [ ] Structured logging for all safety agent invocations
- [ ] Tests pass with ≥90% coverage; mypy clean

---

### Layer 4a — Phase 2 Inner Loop (Spec 06: `mle_star/phase2_inner.py`)

---

## [P1] 23. Coder and planner agents
**Status:** done
**Priority:** P1

### Description
Implement `invoke_coder(code_block, plan, client)` and `invoke_planner(code_block, plans, scores, client)` functions in `src/mle_star/phase2_inner.py`. A_coder (REQ-P2I-001 to REQ-P2I-007) receives a code block + plan and returns modified code. A_planner (REQ-P2I-008 to REQ-P2I-015) receives a code block + previous plan/score history and returns a new refinement plan (3-5 sentences). History format uses Plan:/Score: labels; failed scores displayed as "N/A (evaluation failed)". Both use prompt templates from PromptRegistry and extract code/text from agent responses.

**Spec:** SRS 06 | **Reqs:** REQ-P2I-001 to REQ-P2I-015 | **Depends on:** Spec 01, Spec 02, Spec 03

### Acceptance Criteria
- [x] `invoke_coder(code_block, plan, client)` sends code_block + plan to A_coder via SDK client and returns modified code string
- [x] `invoke_coder()` returns `None` on agent failure (unparseable response)
- [x] `invoke_planner(code_block, plans, scores, client)` sends code_block + history to A_planner via SDK client and returns plan string
- [x] `invoke_planner()` formats history with Plan:/Score: labels; None scores as "N/A (evaluation failed)"
- [x] `invoke_planner()` returns `None` on agent failure (empty response)
- [x] Both functions load prompts from `PromptRegistry`
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 24. Inner loop orchestration
**Status:** done
**Priority:** P1

### Description
Implement `run_phase2_inner_loop(client, solution, code_block, initial_plan, best_score, task, config)` that executes exactly K inner iterations per Algorithm 2. First iteration (k=0) uses the initial plan from A_extractor — NO planner call (REQ-P2I-018); subsequent iterations (k=1..K-1) invoke A_planner with full history of all previous plans and scores (REQ-P2I-020). Each attempt: A_coder always receives ORIGINAL c_t (REQ-P2I-021), `replace_block` against ORIGINAL solution s_t (REQ-P2I-022/023), evaluate, track score. Uses `is_improvement_or_equal()` for best-score update. `InnerLoopResult.improved` uses strict `is_improvement()`.

**Spec:** SRS 06 | **Reqs:** REQ-P2I-016 to REQ-P2I-029 | **Depends on:** Task 23

### Acceptance Criteria
- [x] `run_phase2_inner_loop(client, solution, code_block, initial_plan, best_score, task, config)` executes exactly K iterations (failed ones still count, REQ-P2I-029)
- [x] k=0: uses `initial_plan` directly, NO A_planner call (REQ-P2I-018)
- [x] k≥1: A_planner receives full history of ALL previous plans/scores (REQ-P2I-020)
- [x] A_coder always receives original `code_block.content` (REQ-P2I-021), never a previous attempt's code
- [x] `replace_block` called against original `solution`, not any intermediate (REQ-P2I-022/023)
- [x] Best updated when `is_improvement_or_equal()` returns True (REQ-P2I-026, >= semantics)
- [x] `local_best_score` initialized from `best_score` parameter (REQ-P2I-024)
- [x] `RefinementAttempt` records created for EVERY iteration including failures (REQ-P2I-028)
- [x] `attempts` list has exactly K entries in step order (REQ-P2I-029)
- [x] Returns `InnerLoopResult`; `improved` is True iff strict `is_improvement()` from input best_score
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 25. Inner loop safety integration and error handling
**Status:** done
**Priority:** P1

### Description
Integrate safety checks into the inner loop: call `check_and_fix_leakage()` before each evaluation (REQ-SF-022), use `evaluate_with_retry()` with debug callback for error recovery. Implement error handling per REQ-P2I-032/033/034: coder failure records attempt with `code_block=""` and skips evaluation; replace_block ValueError records attempt with `code_block=c_t_k` and skips evaluation; planner failure records attempt with `plan="[planner failed]"` and skips. Failed attempts included in history for A_planner context (REQ-P2I-035). None scores never trigger best-score update (REQ-P2I-027).

**Spec:** SRS 06 | **Reqs:** REQ-P2I-030 to REQ-P2I-038 | **Depends on:** Task 24

### Acceptance Criteria
- [x] `check_and_fix_leakage()` called before every evaluation (REQ-SF-022)
- [x] `evaluate_with_retry()` used with `make_debug_callback()` for retries
- [x] Coder failure: records RefinementAttempt(score=None, code_block="", was_improvement=False), skips eval
- [x] Replace_block ValueError: records RefinementAttempt(score=None, code_block=c_t_k, was_improvement=False)
- [x] Planner failure: records RefinementAttempt(plan="[planner failed]", score=None, code_block="", was_improvement=False)
- [x] Failed attempts still included in accumulated_plans/scores for A_planner (REQ-P2I-035)
- [x] None score never triggers best-score update (REQ-P2I-027)
- [x] `InnerLoopResult` preserves input solution when no improvement found (REQ-P2I-038)
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P2] 26. Phase 2 inner loop constraints
**Status:** pending
**Priority:** P2

### Description
Implement remaining inner loop requirements: performance constraints (agent invocation overhead), reliability (graceful degradation on agent timeout), observability (structured logging per iteration with plan, score, improvement status), technology constraints (SDK-only invocation, single module), algorithmic constraints (sequential iterations, monotonic best score, immutable input solution, code block provenance tracking).

**Spec:** SRS 06 | **Reqs:** REQ-P2I-039 to REQ-P2I-050 | **Depends on:** Tasks 23–25

### Acceptance Criteria
- [ ] Structured logging for each inner iteration
- [ ] Sequential iteration execution verified
- [ ] Best score is monotonically non-decreasing (maximize) or non-increasing (minimize)
- [ ] Input solution is never mutated
- [ ] Tests pass with ≥90% coverage; mypy clean

---

### Layer 4b — Phase 1 (Spec 04: `mle_star/phase1.py`)

---

## [P1] 27. Phase 1 agents (retriever, init, merger)
**Status:** done
**Priority:** P1

### Description
Implement three Phase 1 agent invocation functions in `src/mle_star/phase1.py`: `retrieve_models(task, config, client)` invokes A_retriever with structured output `RetrieverOutput` to get M candidate models; `generate_candidate(task, model, config, client)` invokes A_init to produce a `SolutionScript` for a given model; `merge_solutions(base, reference, config, client)` invokes A_merger to merge two solutions. All use PromptRegistry for prompt templates. A_retriever returns < M models → log warning and proceed; 0 models → raise ValueError (REQ-P1-005).

**Spec:** SRS 04 | **Reqs:** REQ-P1-001 to REQ-P1-017 | **Depends on:** Spec 01, Spec 02, Spec 03

### Acceptance Criteria
- [x] `retrieve_models()` returns `list[RetrievedModel]` with ≥1 models
- [x] `retrieve_models()` uses structured output `RetrieverOutput` schema
- [x] `retrieve_models()` with < M results logs warning; 0 results raises ValueError (REQ-P1-005)
- [x] Models with empty `example_code` excluded from candidate generation (REQ-P1-006)
- [x] `generate_candidate()` returns `SolutionScript` with `phase="init"` and `source_model` set to `RetrievedModel.model_name`
- [x] `merge_solutions()` returns `SolutionScript` with `phase="merged"`
- [x] All functions load prompts from `PromptRegistry`
- [x] Agent failures return appropriate error/fallback values
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 28. Phase 1 orchestration (run_phase1, Algorithm 1)
**Status:** done
**Priority:** P1

### Description
Implement `run_phase1(client, task, config)` as the Phase 1 entry point implementing Algorithm 1. Steps: (1) retrieve M models, (2-5) generate and evaluate M candidates with leakage check on each, (6) sort by score (permutation pi), (7) initialize best as top candidate, (8-17) merge loop — iteratively merge best with next candidate, evaluate with leakage check, update best if improved (`is_improvement_or_equal`, >=), break on first non-improvement (REQ-P1-028). Handle edge cases: all candidates failed (raise error), single candidate (skip merge).

**Spec:** SRS 04 | **Reqs:** REQ-P1-018 to REQ-P1-029 | **Depends on:** Task 27

### Acceptance Criteria
- [x] `run_phase1()` retrieves M models and generates M candidates
- [x] Leakage check called on each candidate after evaluation
- [x] Candidates sorted by score before merge loop
- [x] Merge loop uses `is_improvement_or_equal()` (>=) for comparison per Algorithm 1
- [x] Break-on-first-failure: merge loop stops on first non-improvement OR execution failure (REQ-P1-028)
- [x] Single candidate (M=1 or only 1 success) skips merge loop
- [x] All-candidates-failed raises `RuntimeError("Phase 1 failed: all {M} candidates produced execution errors")` (REQ-P1-022)
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 29. Phase 1 safety checks and Phase1Result
**Status:** done
**Priority:** P1

### Description
Add post-merge safety checks to `run_phase1`: invoke `check_data_usage()` EXACTLY ONCE on the merged solution (REQ-SF-030), re-evaluate if modified with fallback on failure (REQ-P1-030), then `check_and_fix_leakage()` final check. Construct `Phase1Result` with all fields (retrieved_models, candidate_solutions, candidate_scores, initial_solution, initial_score). Ensure score consistency — `initial_score` matches the score of `initial_solution`.

**Spec:** SRS 04 | **Reqs:** REQ-P1-030 to REQ-P1-033 | **Depends on:** Task 28

### Acceptance Criteria
- [x] `check_data_usage()` called exactly once on merged solution (REQ-SF-030)
- [x] If A_data modifies solution, re-evaluate; fallback to pre-modification if re-eval fails (REQ-P1-030)
- [x] `check_and_fix_leakage()` called after data check
- [x] `Phase1Result` correctly constructed with all fields
- [x] `initial_score` equals `initial_solution.score`
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P2] 30. Phase 1 constraints
**Status:** pending
**Priority:** P2

### Description
Implement remaining Phase 1 requirements: candidate independence (parallel generation if possible), evaluation independence, sequential merge loop, orchestration overhead budget, structured logging, SDK-only invocation, single module organization, Algorithm 1 fidelity verification, leakage integration points (on each candidate, on each merge result, on final s_0 after A_data), and prompt fidelity for retriever/init/merger agents.

**Spec:** SRS 04 | **Reqs:** REQ-P1-034 to REQ-P1-045 | **Depends on:** Tasks 27–29

### Acceptance Criteria
- [ ] Structured logging for each Phase 1 step
- [ ] Merge loop executes sequentially (not parallel)
- [ ] Leakage check at 3 integration points: each candidate, each merge result, final s_0
- [ ] Algorithm 1 steps match paper specification
- [ ] Tests pass with ≥90% coverage; mypy clean

---

### Layer 4c — Phase 2 Outer Loop (Spec 05: `mle_star/phase2_outer.py`)

---

## [P1] 31. Ablation agent
**Status:** done
**Priority:** P1

### Description
Implement A_abl agent invocation in `src/mle_star/phase2_outer.py`. The ablation agent receives a solution and produces an ablation study script (2-3 components, avoid previously ablated, validation-only — no test data). Includes: agent definition (Figure 12), prompt template from registry, ablation script execution with timeout = `min(time_limit / (outer_steps * 2), 600)` seconds (REQ-P2O-035), debug retry on execution error. Ablation output feeds into A_summarize.

**Spec:** SRS 05 | **Reqs:** REQ-P2O-001 to REQ-P2O-007 | **Depends on:** Spec 01, Spec 02, Spec 03

### Acceptance Criteria
- [x] Ablation agent invocation returns ablation script content
- [x] Ablation script is executed with timeout = `min(time_limit / (outer_steps * 2), 600)` (REQ-P2O-035)
- [x] On execution error: A_debugger retry up to max_debug_attempts; all fail → set T_abl^t="" and continue
- [x] Previously ablated components tracked and excluded
- [x] Evaluation is validation-only (no test set)
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 32. Summarize and extractor agents
**Status:** done
**Priority:** P1

### Description
Implement A_summarize and A_extractor agents. A_summarize (REQ-P2O-008 to REQ-P2O-011) receives ablation results and produces a text summary (T_abl); uses full response (no extraction needed). A_extractor (REQ-P2O-012 to REQ-P2O-018) receives the solution + ablation summary and produces structured output `ExtractorOutput` containing `list[RefinePlan]` (each with `code_block` + `plan`). Includes `validate_code_block(code_block, solution)` for exact substring verification with recovery: (1) whitespace-normalized match, (2) re-invoke A_extractor up to 2 more times, (3) try other plans in list, (4) skip if all fail (REQ-P2O-017/018).

**Spec:** SRS 05 | **Reqs:** REQ-P2O-008 to REQ-P2O-018 | **Depends on:** Task 31

### Acceptance Criteria
- [x] A_summarize returns text summary from ablation results (uses full response)
- [x] A_summarize fallback on empty/unparseable response: truncate raw ablation output to last 2000 chars with prefix `"[Auto-summary from raw output] "` (REQ-P2O-036)
- [x] A_extractor uses `ExtractorOutput` structured output schema
- [x] A_extractor retry on malformed JSON: retry once before entering validation recovery (REQ-P2O-034)
- [x] A_extractor returns `list[RefinePlan]` with code_block + plan pairs
- [x] `validate_code_block(code_block, solution)` returns `True` iff code_block is exact substring of `solution.content`
- [x] Validation recovery level 1: strip trailing whitespace from each line of both code_block and solution, check substring match; if match found, use the matched substring from solution source as c_t
- [x] Validation recovery level 2: re-invoke A_extractor with error feedback up to 2 additional times ("The previously extracted code block was not found in the solution. Please extract the code block exactly as it appears in the script.")
- [x] Validation recovery level 3: iterate through `ExtractorOutput.plans` list and select the first `RefinePlan` whose `code_block` passes validation (REQ-P2O-018 step 3)
- [x] Validation recovery level 4: if no plan passes validation, skip this outer loop iteration entirely and proceed to the next
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 33. Outer loop orchestration (run_phase2_outer_loop)
**Status:** done
**Priority:** P1

### Description
Implement `run_phase2_outer_loop(client, task, config, initial_solution, initial_score, session_id)` executing T outer iterations per Algorithm 2. `initial_score: float` is passed explicitly (REQ-P2O-019) because `SolutionScript.score` is `float | None` but `h_best` must be a guaranteed `float`. Each iteration: (1) run ablation, (2) summarize → accumulate T_abl, (3) extract code blocks (FIRST plan's code_block = c_t, FIRST plan = p_0), (4) call `run_phase2_inner_loop()`, (5) update best solution when inner loop's best score passes `is_improvement_or_equal()` (>= semantics, REQ-P2O-027). Track state: T_abl (accumulated ablation summaries), C (refined code blocks). Construct `Phase2Result` with all fields.

**Spec:** SRS 05 | **Reqs:** REQ-P2O-019 to REQ-P2O-030 | **Depends on:** Tasks 31, 32, Layer 4a (Phase 2 Inner)

### Acceptance Criteria
- [x] Signature: `run_phase2_outer_loop(client, task, config, initial_solution, initial_score, session_id)` — `initial_score: float` required per REQ-P2O-019
- [x] `h_best` initialized from `initial_score` parameter (not from `initial_solution.score`) to avoid `None` check
- [x] Executes exactly T outer iterations
- [x] Each iteration: ablation → summarize → extract → inner loop
- [x] Uses FIRST plan from ExtractorOutput: c_t = plans[0].code_block, p_0 = plans[0].plan
- [x] Ablation summaries accumulated across iterations in T_abl list
- [x] Code blocks accumulated for provenance tracking in C list; each `CodeBlock` has `outer_step=t` set (REQ-P2O-044)
- [x] Best solution updated when inner loop's best score passes `is_improvement_or_equal()` (>=) against current `h_best` (REQ-P2O-027) — do NOT use `InnerLoopResult.improved` flag (which uses strict >)
- [x] Inner loop receives: solution=s_t, code_block=c_t, initial_plan=p_0, best_score=h_best
- [x] Skipped iterations (validation recovery level 4 failure) recorded in step history with `was_skipped=True` (REQ-P2O-030)
- [x] `Phase2Result.best_score` never worse than `initial_score` — score guarantee (REQ-P2O-029)
- [x] `Phase2Result` correctly constructed with all fields
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P2] 34. Phase 2 outer loop constraints
**Status:** pending
**Priority:** P2

### Description
Implement remaining outer loop requirements: performance constraints (A_abl response overhead, A_extractor validation overhead, total duration budget), reliability (graceful degradation for extractor/summarize, ablation timeout), observability (structured logging per outer step), technology constraints (SDK invocation, single module), algorithmic constraints (sequential outer loop, monotonic best score, ablation script self-containment, immutable input, code block provenance).

**Spec:** SRS 05 | **Reqs:** REQ-P2O-031 to REQ-P2O-044 | **Depends on:** Tasks 31–33

### Acceptance Criteria
- [ ] Structured logging for each outer iteration step
- [ ] Sequential outer loop execution verified
- [ ] Best score monotonically improving (never overwritten with worse)
- [ ] Input solution never mutated
- [ ] Ablation scripts are self-contained (do not import s_t separately)
- [ ] Tests pass with ≥90% coverage; mypy clean

---

### Layer 4d — Phase 3 Ensemble (Spec 07: `mle_star/phase3.py`)

---

## [P1] 35. Ensemble planner and ensembler agents
**Status:** done
**Priority:** P1

### Description
Implement two Phase 3 agent functions in `src/mle_star/phase3.py`: `invoke_ens_planner(solutions, plans, scores, client)` invokes A_ens_planner (Figure 17) to generate an ensemble strategy plan, and `invoke_ensembler(plan, solutions, client)` invokes A_ensembler (Figure 18) to produce a FULL standalone ensemble solution script (not a code block replacement). First invocation (r=0) has no history; subsequent invocations include all prior plans and scores. Planner told to "differ from previous plans" and "concentrate how to merge." Ensembler produces a script with "Do not subsample" instruction.

**Spec:** SRS 07 | **Reqs:** REQ-P3-001 to REQ-P3-016 | **Depends on:** Spec 01, Spec 02, Spec 03

### Acceptance Criteria
- [x] `invoke_ens_planner()` returns ensemble plan string
- [x] First call (empty history) uses initial prompt variant
- [x] Subsequent calls include full plan+score history formatted with `Plan {r}:` / `Score {r}:` labels (consistent with inner loop planner format from Task 23); None scores displayed as "N/A (evaluation failed)"
- [x] `invoke_ensembler()` returns `SolutionScript` with `phase="ensemble"` — FULL program, not code block
- [x] Ensembler prompt includes "Do not subsample" instruction
- [x] Both return `None` on agent failure
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 36. Phase 3 orchestration (run_phase3, Algorithm 3)
**Status:** done
**Priority:** P1

### Description
Implement `run_phase3(client, task, config, solutions)` executing R ensemble rounds per Algorithm 3. If `len(solutions) == 1`, skip ensemble entirely and return a `Phase3Result` with that solution as `best_ensemble` (REQ-P3-018). Raises ValueError if `len(solutions) == 0` (empty input). Round r=0: plan → implement → evaluate. Rounds r=1..R-1: plan (with history) → implement → evaluate. Safety integration: `check_and_fix_leakage()` before every evaluation (REQ-SF-022), `evaluate_with_retry()` with debug callback. Track scores; tie-breaking: LAST occurrence wins (consistent with >= semantics, REQ-P3-025). Exactly R `EnsembleAttempt` records created (REQ-P3-044). All-rounds-failed: fallback to best input solution without raising (REQ-P3-026). Construct `Phase3Result`.

**Spec:** SRS 07 | **Reqs:** REQ-P3-017 to REQ-P3-035 | **Depends on:** Task 35

### Acceptance Criteria
- [x] Handles `len(solutions) == 1` by returning immediately with single solution as `best_ensemble` (REQ-P3-018)
- [x] Raises ValueError if `len(solutions) == 0` (empty input)
- [x] Executes R rounds with plan → implement → evaluate cycle
- [x] History passed to planner grows each round
- [x] Best ensemble selected by score comparison; ties won by LAST occurrence (REQ-P3-025)
- [x] `check_and_fix_leakage()` called before each evaluation (REQ-SF-022)
- [x] Exactly R `EnsembleAttempt` records created regardless of success/failure (REQ-P3-044)
- [x] Failed ensembler: record `EnsembleAttempt` with `SolutionScript(content="", ...)` (REQ-P3-030)
- [x] Failed ens_planner: record plan as `"[ens_planner failed]"` in accumulated history (REQ-P3-031)
- [x] All-rounds-failed: fallback returns best input solution, no exception raised (REQ-P3-026)
- [x] `Phase3Result` correctly constructed
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P2] 37. Phase 3 constraints
**Status:** pending
**Priority:** P2

### Description
Implement remaining Phase 3 requirements: performance (orchestration overhead), reliability (never raises on round failure), observability (structured logging per round), technology constraints (SDK-only, single module), algorithm fidelity (Algorithm 3 step correspondence, sequential rounds, iteration count = R), prompt fidelity (ens_planner Figure 17, ensembler Figure 18), ensemble scripts as full programs, "Final Validation Performance" output pattern.

**Spec:** SRS 07 | **Reqs:** REQ-P3-036 to REQ-P3-049 | **Depends on:** Tasks 35–36

### Acceptance Criteria
- [ ] Round failure does not propagate — `run_phase3` never raises (graceful degradation)
- [ ] Structured logging for each ensemble round
- [ ] Sequential rounds verified
- [ ] Exactly R rounds attempted (unless time budget exceeded)
- [ ] Tests pass with ≥90% coverage; mypy clean

---

### Layer 4e — Finalization (Spec 08: `mle_star/finalization.py`)

---

## [P1] 38. Subsampling removal
**Status:** done
**Priority:** P1

### Description
Implement subsampling removal pipeline in `src/mle_star/finalization.py`. Uses two A_test agent calls with different variants: (1) variant "subsampling_extract" (Figure 26) identifies the subsampling code block, (2) variant "subsampling_remove" (Figure 27) generates replacement code without subsampling. `remove_subsampling(client, solution, task)` (async) orchestrates both steps, applies fix via `SolutionScript.replace_block()`. No-subsampling passthrough: if no subsampling detected (empty block or not found in content), returns original solution.

**Spec:** SRS 08 | **Reqs:** REQ-FN-001 to REQ-FN-009 | **Depends on:** Spec 01, Spec 02, Spec 03

### Acceptance Criteria
- [x] Extraction uses A_test with `variant="subsampling_extract"` (Figure 26)
- [x] Removal uses A_test with `variant="subsampling_remove"` (Figure 27)
- [x] `remove_subsampling(client, solution, task)` applies fix via `replace_block()`
- [x] No-subsampling detected (empty block or not substring) → returns original solution unchanged
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 39. Test submission agent
**Status:** done
**Priority:** P1

### Description
Implement A_test agent (Figure 25) and `generate_test_submission(client, task, solution)` (async). The test agent takes a validated solution and minimally modifies it to produce test predictions: uses full training set (no subsampling), reads test data from correct location, writes `./final/submission.csv` covering all test samples. Execute with FULL timeout (not reduced, REQ-FN-021), verify submission file. Retry with debug on failure (up to max_debug_attempts). Fallback to best validation solution if all retries fail (REQ-FN-025).

**Spec:** SRS 08 | **Reqs:** REQ-FN-010 to REQ-FN-025 | **Depends on:** Task 38

### Acceptance Criteria
- [x] `generate_test_submission(client, task, solution)` invokes A_test agent via SDK client with minimal-modification instruction
- [x] `clean_output_directory()` called before test script execution to clear `./final/` (REQ-FN-020)
- [x] Test script executed with FULL timeout (REQ-FN-021)
- [x] `verify_submission()` then `get_submission_info()` used to validate output
- [x] `evaluate_with_retry()` used for error recovery with leakage check (REQ-SF-022)
- [x] Fallback to validation solution on exhausted retries (REQ-FN-025)
- [x] Submission uses full training data (no subsampling)
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 40. Contamination check and run_finalization
**Status:** done
**Priority:** P1

### Description
Implement `check_contamination(client, solution, reference_discussions)` (async) using a contamination check variant of A_test agent (`AgentType.test`, not A_data) with `DataContaminationResult` structured output (Figure 28, REQ-FN-026). **Contamination check variant uses `tools=None` and `output_schema=DataContaminationResult`**, differing from the default A_test config (`tools=["Read"]`, `output_schema=None`) — override at invocation time per REQ-FN-048. Multiple references: ANY "Same" verdict → overall "Same". Implement `run_finalization(client, solution, task, config, phase1_result, phase2_results, phase3_result, reference_discussions=None)` entry point per REQ-FN-034 (with `client` added per cross-cutting SDK threading convention). Pipeline: (1) remove subsampling, (2) generate test submission, (3) leakage check on test script (REQ-SF-022), (4) evaluate with full timeout, (5) verify submission, (6) fallback handling if execution failed (REQ-FN-025), (7) check contamination (optional, skipped if no references), (8) construct FinalResult with all phase results (REQ-FN-036).

**Spec:** SRS 08 | **Reqs:** REQ-FN-026 to REQ-FN-036 | **Depends on:** Tasks 38, 39

### Acceptance Criteria
- [x] `check_contamination()` returns `DataContaminationResult` or `None` (when no references)
- [x] Multiple references: ANY "Same" → overall "Same" verdict
- [x] `run_finalization(client, solution, task, config, phase1_result, phase2_results, phase3_result, reference_discussions=None)` signature matches REQ-FN-034 (with `client` added)
- [x] `run_finalization()` executes full pipeline: remove subsampling → test submission → leakage check → execute with retry → verify → fallback → contamination → FinalResult
- [x] `FinalResult` correctly assembled with all phase results per REQ-FN-036
- [x] Contamination check skipped when no reference discussions provided
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P2] 41. Finalization constraints
**Status:** pending
**Priority:** P2

### Description
Implement remaining finalization requirements: performance (overhead < 5s excluding script runtime, subsampling removal latency), reliability (graceful degradation for extraction, A_test, contamination check), observability (structured logging), technology constraints (SDK invocation, single module), submission invariants (`./final/submission.csv` path, no `exit()`, no error masking), agent configs in `build_default_agent_configs()`, and AgentType.test multiple operational modes (test submission, subsampling extract/remove, contamination check — REQ-FN-048).

**Spec:** SRS 08 | **Reqs:** REQ-FN-037 to REQ-FN-048 | **Depends on:** Tasks 38–40

### Acceptance Criteria
- [ ] Graceful degradation: any finalization step failure doesn't crash pipeline
- [ ] Structured logging for finalization steps
- [ ] Submission file written to `./final/submission.csv`
- [ ] Tests pass with ≥90% coverage; mypy clean

---

### Layer 5 — Orchestrator (Spec 09: `mle_star/orchestrator.py`)

---

## [P1] 42. Pipeline entry point and SDK client setup
**Status:** done
**Priority:** P1

### Description
Implement `run_pipeline(task, config)` (async) as the main entry point in `src/mle_star/orchestrator.py`, and `run_pipeline_sync(task, config)` synchronous wrapper via `asyncio.run()` (REQ-OR-053). Includes: input validation (TaskDescription, PipelineConfig, data_dir exists with at least one file — REQ-OR-002), `setup_working_directory()` delegation, `detect_gpu_info()`, SDK client initialization (`ClaudeSDKClient` creation with model, permission_mode, max_budget_usd, agents, hooks), agent registration (14 `AgentDefinition` entries via `build_default_agent_configs()`), system prompt configuration (Kaggle grandmaster persona + task context + GPU info — REQ-OR-007), MCP server registration (REQ-OR-010, optional with warning on failure), and `client.disconnect()` cleanup via try/finally (REQ-OR-011). Define `PipelineError` (with `diagnostics` attribute, REQ-OR-042) and `PipelineTimeoutError` (REQ-OR-030) exception classes.

**Spec:** SRS 09 | **Reqs:** REQ-OR-001 to REQ-OR-011, REQ-OR-042, REQ-OR-053 | **Depends on:** All Layer 0–4 tasks

### Acceptance Criteria
- [x] `run_pipeline()` is async, accepts `TaskDescription` and optional `PipelineConfig`
- [x] `run_pipeline_sync()` synchronous wrapper via `asyncio.run()` (REQ-OR-053)
- [x] Input validation: raises ValueError for invalid task/config, missing data_dir, empty data_dir (REQ-OR-002)
- [x] SDK client created with model, permission_mode, max_budget_usd, agents (14), hooks
- [x] System prompt includes Kaggle grandmaster persona + task description + metric + GPU info (REQ-OR-007)
- [x] MCP server registration attempted; failure logged as warning (REQ-OR-010)
- [x] Client disconnected on completion even on error (try/finally) (REQ-OR-011)
- [x] `PipelineError` defined with `diagnostics` attribute (REQ-OR-042)
- [x] `PipelineTimeoutError` defined for Phase 1 incomplete timeouts (REQ-OR-030)
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 43. Phase dispatch and sequencing
**Status:** done
**Priority:** P1

### Description
Implement sequential phase dispatch within `run_pipeline`: Phase 1 → Phase 2 (L parallel) → Phase 3 → Finalization. Each dispatch records start time and duration. Phase 3 skip condition when `config.num_parallel_solutions == 1` (REQ-OR-015). Finalization receives the best solution from Phase 3 (or Phase 2 if skipped/failed).

**Spec:** SRS 09 | **Reqs:** REQ-OR-012 to REQ-OR-017 | **Depends on:** Task 42

### Acceptance Criteria
- [x] Phases execute in strict order: P1 → P2 → P3 → Finalization (REQ-OR-017)
- [x] Phase 2 does not begin until Phase 1 completes
- [x] Phase 3 skipped when L=1; `FinalResult.phase3` is `None` (REQ-OR-015)
- [x] Finalization receives best available solution
- [x] Start time and duration recorded per phase
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 44. Asyncio parallelism for Phase 2 paths
**Status:** done
**Priority:** P1

### Description
Implement L parallel Phase 2 paths using `asyncio.gather(*tasks, return_exceptions=True)`. Each path receives a deep copy of the Phase 1 solution (REQ-OR-020), operates in its own working subdirectory (`./work/path-{i}/`), and uses a unique session ID (`"path-{i}"`) (REQ-OR-021). Sessions may fork from Phase 1 via `ClaudeAgentOptions(resume=phase1_session_id, fork_session=True)`. Failed paths logged; successful results collected. If all paths fail, fall back to Phase 1 solution (REQ-OR-022). Wait bounded by Phase 2 time allocation; overtime paths cancelled via `asyncio.Task.cancel()` (REQ-OR-023).

**Spec:** SRS 09 | **Reqs:** REQ-OR-018 to REQ-OR-023 | **Depends on:** Task 43

### Acceptance Criteria
- [x] L paths run concurrently via `asyncio.gather(return_exceptions=True)`
- [x] Each path receives a deep copy of Phase 1 solution (REQ-OR-020) and `Phase1Result.initial_score` as the `initial_score` parameter
- [x] Each path uses separate working directory `./work/path-{i}/` and session ID `"path-{i}"` (REQ-OR-020/021)
- [x] Session forking from Phase 1 session supported (REQ-OR-021)
- [x] Failed path does not affect other paths (REQ-OR-022)
- [x] All-paths-failed falls back to Phase 1 solution (REQ-OR-022)
- [x] Overtime paths cancelled gracefully via `asyncio.Task.cancel()` (REQ-OR-023)
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 45. Time and cost control
**Status:** done
**Priority:** P1

### Description
Implement time and cost budgeting: overall `time_limit_seconds` enforcement with `time.monotonic()` deadline (REQ-OR-024), proportional phase time allocation (Phase 1: 10%, Phase 2: 65%, Phase 3: 15%, Finalization: 10%) configurable via `PhaseTimeBudget` (REQ-OR-025), per-path Phase 2 time budget (`phase2_budget / L`) (REQ-OR-026). Cost accumulation from `ResultMessage.total_cost_usd` (REQ-OR-027), `max_budget_usd` enforcement with 80% warning (REQ-OR-029). Graceful shutdown on timeout or budget exceeded: cancel in-progress tasks, use best solution so far, skip to finalization if Phase 1 completed, else raise PipelineTimeoutError (REQ-OR-030).

**Spec:** SRS 09 | **Reqs:** REQ-OR-024 to REQ-OR-030 | **Depends on:** Task 44

### Acceptance Criteria
- [x] Deadline computed at pipeline start via `time.monotonic()`
- [x] Deadline checked before each phase and agent call
- [x] Phase time budgets follow proportional allocation (configurable via PhaseTimeBudget)
- [x] Per-path Phase 2 budget = `phase2_budget / L`
- [x] Cost accumulated and checked after each agent call
- [x] 80% budget warning logged (REQ-OR-029)
- [x] Graceful shutdown: cancel tasks → best solution so far → skip to finalization (REQ-OR-030)
- [x] Phase 1 incomplete at timeout → raise PipelineTimeoutError (REQ-OR-030)
- [x] Pipeline with `time_limit_seconds=60` terminates within 90s
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 46. Hook system
**Status:** done
**Priority:** P1

### Description
Implement the 5 SDK hooks: (1) `PostToolUse` progress hook — structured JSON logging of agent activity (REQ-OR-031), (2) cost accumulation hook on `Stop`/`SubagentStop` — thread-safe cost tracker with budget check (REQ-OR-032), (3) `PreToolUse` safety hook — blocks dangerous bash commands (rm -rf /, mkfs, dd if=, fork bomb, out-of-directory modifications) returning `BlockToolUse` result (REQ-OR-033), (4) `PostToolUse` timeout hook — monitors elapsed time and sets "finalize now" flag when remaining < 10% of total budget or < 5 minutes (whichever is larger) (REQ-OR-034), (5) `PostToolUse` error logging hook — captures failures with consecutive failure counting per agent type (REQ-OR-035).

**Spec:** SRS 09 | **Reqs:** REQ-OR-031 to REQ-OR-035 | **Depends on:** Task 45

### Acceptance Criteria
- [x] Progress hook logs: timestamp, agent type, tool name, session ID, elapsed time, success/failure (JSON format)
- [x] Cost hook accumulates per-agent costs thread-safely (safe for concurrent Phase 2 paths)
- [x] Safety hook blocks known dangerous bash patterns; returns `BlockToolUse` result
- [x] Safety hook has configurable blocked-command list
- [x] Timeout hook sets finalize flag when remaining < 10% budget OR < 5 min (whichever is larger)
- [x] Error hook tracks consecutive failures per agent type for circuit-breaker logic
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 47. Result assembly and error recovery
**Status:** done
**Priority:** P1

### Description
Implement `FinalResult` construction from all phase outputs: field assembly, per-phase cost breakdown (REQ-OR-037), per-phase duration breakdown (REQ-OR-038), solution lineage tracing (REQ-OR-039). Error recovery: Phase 2 failure substitutes Phase 1 solution for that path's ensemble contribution (REQ-OR-040), Phase 3 failure selects best Phase 2 solution for finalization (REQ-OR-041), complete failure (Phase 1 fails) raises PipelineError with diagnostics (REQ-OR-042), best-effort FinalResult returned even on partial failure (REQ-OR-043) with finalization failure setting submission_path="" and logging.

**Spec:** SRS 09 | **Reqs:** REQ-OR-036 to REQ-OR-043 | **Depends on:** Task 46

### Acceptance Criteria
- [x] `FinalResult` assembled with all phase results
- [x] Cost summary includes per-phase breakdown (REQ-OR-037)
- [x] Duration summary includes per-phase breakdown (REQ-OR-038)
- [x] Phase 2 failure: Phase 1 solution substituted for failed path's contribution; `Phase2Result.step_history` includes `failed=True` flag (REQ-OR-040)
- [x] Phase 3 failure: best Phase 2 solution used for finalization (REQ-OR-041)
- [x] All phases fail (Phase 1 fails): raises PipelineError with diagnostics (REQ-OR-042)
- [x] Partial failure produces best-effort FinalResult; finalization failure → submission_path="" (REQ-OR-043)
- [x] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 48. Configuration and environment
**Status:** pending
**Priority:** P1

### Description
Implement configuration management: sensible defaults for all hyperparameters (REQ-OR-044), per-run override via `PipelineConfig` (REQ-OR-045), environment variable support (REQ-OR-046) with precedence: env vars override PipelineConfig defaults but are overridden by explicit constructor arguments. Environment vars: `ANTHROPIC_API_KEY` (required, raise EnvironmentError if missing), `MLE_STAR_MODEL`, `MLE_STAR_LOG_LEVEL`, `MLE_STAR_MAX_BUDGET`, `MLE_STAR_TIME_LIMIT`. Python `logging` module configuration with logger name "mle_star", configurable level, console handler, optional file handler, phase boundary markers (REQ-OR-047). `PipelineState` class for runtime introspection (current_phase, elapsed_seconds, accumulated_cost_usd, phase2_path_statuses, best_score_so_far, agent_call_count — REQ-OR-050).

**Spec:** SRS 09 | **Reqs:** REQ-OR-044 to REQ-OR-050 | **Depends on:** Task 47

### Acceptance Criteria
- [ ] All hyperparameters have sensible defaults from paper
- [ ] `PipelineConfig` overrides applied correctly
- [ ] `ANTHROPIC_API_KEY` read from environment; raise EnvironmentError if missing
- [ ] `MLE_STAR_*` env vars override defaults but NOT explicit constructor args (REQ-OR-046)
- [ ] Logging configured: logger "mle_star", configurable level, structured output, optional file handler
- [ ] Phase boundary markers logged (e.g., "=== Phase 1: Initial Solution Generation ===")
- [ ] `PipelineState` tracks: current_phase, elapsed_seconds, accumulated_cost_usd, phase2_path_statuses, best_score_so_far, agent_call_count
- [ ] Tests pass with ≥90% coverage; mypy clean

---

## [P2] 49. Orchestrator constraints
**Status:** pending
**Priority:** P2

### Description
Implement remaining orchestrator requirements: performance (overhead < 1% of total runtime, memory efficiency — don't retain all intermediate solutions, REQ-OR-049), observability (`PipelineState` introspection — REQ-OR-050), reliability (idempotent retry safety — fresh state each run, REQ-OR-051; SDK reconnection with 3 retries and exponential backoff 1s/2s/4s using `resume=session_id`, REQ-OR-052), technology constraints (Python 3.10+/asyncio — REQ-OR-053, SDK version >= 0.1.39 with ImportError on incompatible — REQ-OR-054, single module — REQ-OR-055), SDK compatibility (concurrent session limit handling with serialization fallback — REQ-OR-056, agent name uniqueness matching AgentType enum — REQ-OR-057).

**Spec:** SRS 09 | **Reqs:** REQ-OR-048 to REQ-OR-057 | **Depends on:** Tasks 42–48

### Acceptance Criteria
- [ ] Orchestrator overhead < 1% of total pipeline time (REQ-OR-048)
- [ ] Memory: only current best + under-evaluation solutions retained per path (REQ-OR-049)
- [ ] Each `run_pipeline()` call creates fresh state (REQ-OR-051)
- [ ] SDK reconnection: 3 retries, exponential backoff (1s, 2s, 4s), using `resume=session_id` (REQ-OR-052)
- [ ] Concurrent session limits respected; excess paths serialized with warning (REQ-OR-056)
- [ ] Agent names unique, matching `AgentType` enum values (REQ-OR-057)
- [ ] Tests pass with ≥90% coverage; mypy clean

---

### Layer 6 — Integration and CLI

---

## [P1] 50. CLI entry point integration
**Status:** pending
**Priority:** P1

### Description
Update `src/mle_star/cli.py` to wire the CLI entry point to `run_pipeline_sync()`. The existing skeleton prints "Hello from mle_star!" but does not invoke any pipeline logic. The CLI should: parse a task description (from a YAML/JSON file path or inline arguments), construct a `TaskDescription` and optional `PipelineConfig`, call `run_pipeline_sync()`, and print/save the `FinalResult`. Use `argparse` for argument parsing. This is the user-facing interface to the pipeline and is required for `uv run mle_star` to function as documented.

**Depends on:** Task 42 (orchestrator entry point), Task 03 (config models)

### Acceptance Criteria
- [ ] `uv run mle_star --task <path_to_task.yaml>` loads task description and runs pipeline
- [ ] `--config <path_to_config.yaml>` optional flag for custom PipelineConfig
- [ ] `--help` prints usage information
- [ ] Exit code 0 on success, non-zero on PipelineError
- [ ] Errors printed to stderr with useful diagnostics
- [ ] Tests pass with ≥90% coverage; mypy clean

---

## [P1] 51. Shared test infrastructure
**Status:** done
**Priority:** P1

### Description
Create shared test fixtures, factories, and mock objects in `tests/conftest.py` for use by all test modules. Includes: mock `ClaudeSDKClient` (returns configurable agent responses), factory functions for `SolutionScript`, `TaskDescription`, `PipelineConfig`, `EvaluationResult` (with sensible defaults), mock `PromptRegistry` that returns template stubs, temporary directory fixtures for working directory tests, and async test helpers. Every task from Layer 2 onward needs to mock the SDK client; centralizing this prevents duplication and ensures consistency.

**Depends on:** Task 03 (config models), Task 04 (solution models), Task 05 (agent types)

### Acceptance Criteria
- [x] `conftest.py` provides `mock_client` fixture returning a mock `ClaudeSDKClient` with configurable responses
- [x] Factory functions: `make_solution()`, `make_task()`, `make_config()`, `make_eval_result()` with sensible defaults
- [x] `mock_registry` fixture returning a stub `PromptRegistry`
- [x] `tmp_working_dir` fixture providing a temporary directory with standard layout
- [x] All fixtures are session-scoped or function-scoped as appropriate
- [x] mypy clean

---

## [P1] 52. Prompts package initialization
**Status:** done
**Priority:** P1

### Description
Create `src/mle_star/prompts/__init__.py` to make the prompts directory a proper Python package. This is required for `importlib.resources` or `pkgutil` to locate the YAML prompt template files at runtime. Without it, `PromptRegistry` (Task 08) cannot reliably discover and load template files from the installed package.

**Depends on:** None (package structure)

### Acceptance Criteria
- [x] `src/mle_star/prompts/__init__.py` exists (can be empty or contain package docstring)
- [x] `importlib.resources.files("mle_star.prompts")` resolves correctly
- [x] mypy clean

---

## Requirement Coverage

| Spec | Module | Requirements | P1 Tasks | P2 Tasks | Total |
|------|--------|-------------|----------|----------|-------|
| 01 — Data Models | `models.py` | REQ-DM-001 to 050 (50) | 03–10 (8) | — | 8 |
| 02 — Execution Harness | `execution.py` | REQ-EX-001 to 047 (47) | 11–17 (7) | 18 (1) | 8 |
| 03 — Safety Modules | `safety.py` | REQ-SF-001 to 046 (46) | 19–21 (3) | 22 (1) | 4 |
| 04 — Phase 1 | `phase1.py` | REQ-P1-001 to 045 (45) | 27–29 (3) | 30 (1) | 4 |
| 05 — Phase 2 Outer | `phase2_outer.py` | REQ-P2O-001 to 044 (44) | 31–33 (3) | 34 (1) | 4 |
| 06 — Phase 2 Inner | `phase2_inner.py` | REQ-P2I-001 to 050 (50) | 23–25 (3) | 26 (1) | 4 |
| 07 — Phase 3 Ensemble | `phase3.py` | REQ-P3-001 to 049 (49) | 35–36 (2) | 37 (1) | 3 |
| 08 — Finalization | `finalization.py` | REQ-FN-001 to 048 (48) | 38–40 (3) | 41 (1) | 4 |
| 09 — Orchestrator | `orchestrator.py` | REQ-OR-001 to 057 (57) | 42–48 (7) | 49 (1) | 8 |
| — — Project Setup | `pyproject.toml`, `prompts/` | — | 01–02, 52 (3) | — | 3 |
| — — Test Infrastructure | `tests/conftest.py` | — | 51 (1) | — | 1 |
| — — CLI Integration | `cli.py` | — | 50 (1) | — | 1 |
| **Total** | | **436** | **44** | **8** | **52** |

---

## Changelog

### 2026-02-21 (v14)

Verification pass against all 36 spec files, full implementation plan (v13), codebase, pyproject.toml, and .pre-commit-config.yaml. Codebase still empty (skeleton CLI only) — all 52 tasks remain pending.

**Result:** No gaps found. Plan is complete and accurate.

**Verified:**
- All 436 requirements across 9 SRS documents (01–09) covered by 52 tasks
- Summary table correct: 44 P1 + 8 P2 = 52 total
- Requirement Coverage table row totals match summary
- Dependency ordering (Layer 0 → 5) is correct and complete
- All cross-cutting constraints (REQ-SF-022 leakage before every eval, REQ-EX-011 last match, REQ-P2I-021/022/023 original code block, REQ-OR-008 tool overrides, REQ-DM-039 frozen models, score comparison semantics) remain accurate
- Pre-commit hook constraints (ruff D-rules, mypy strict, bandit, xenon B, pip-audit) still apply
- Task sizing remains appropriate (no individual task covers more than ~20 requirements; P2 constraint tasks batch non-functional/observability requirements)
- No requirements were added, removed, or modified in the spec files since v13

---

### 2026-02-21 (v13)

Re-analysis of all 36 spec files, full implementation plan (v12), and codebase against requirements. Codebase still empty (skeleton CLI only) — all 52 tasks remain pending.

**Bugs fixed:**
1. **Summary table count error**: P1 was 43, should be 44. Total was 51, should be 52. The v12 changelog added 3 tasks (50, 51, 52) to the previous 49, yielding 52 total. Of those, 8 are P2 (tasks 18, 22, 26, 30, 34, 37, 41, 49), leaving 44 P1. The off-by-one propagated through the summary table and the Requirement Coverage total row. Both corrected.
2. **Requirement Coverage table total**: Bottom row said "43 P1, 8 P2, 51 total" but individual rows summed to 44 P1, 8 P2, 52 total. Corrected to match.

**Improvements:**
1. **Task 18 (Execution harness constraints) — expanded acceptance criteria**: Added explicit acceptance criteria for SDK Bash executor requirements that were within the task's scope (REQ-EX-028 to REQ-EX-047) but only partially reflected in the acceptance criteria:
   - `execute_script_via_sdk()` function using SDK Bash tool (REQ-EX-033)
   - `ExecutorStrategy` enum with subprocess/sdk_bash values (REQ-EX-034)
   - SDK Bash 600,000ms timeout cap with automatic subprocess fallback (REQ-EX-047)
   - `evaluate_solution()` accepting optional `strategy` parameter (REQ-EX-034)
   - `ScoreFunction` protocol compliance (REQ-EX-032)
   - Expanded performance criteria with specific thresholds from spec (REQ-EX-035/036)
   - Large output handling (REQ-EX-038), default timeout derivation (REQ-EX-046), no persistent state (REQ-EX-042)
   - Description expanded with full REQ-ID cross-references for traceability

**Verified correct (no change needed):**
- All 436 requirements from specs 01-09 still covered by 52 tasks
- Task priorities and dependencies unchanged
- All cross-cutting constraints remain accurate
- No new requirements discovered from spec re-read

---

### 2026-02-21 (v12)

Re-analysis of all 36 spec files, full implementation plan (v11), and codebase against requirements. Codebase still empty (skeleton CLI only) — all tasks remain pending.

**Summary table fix:**
1. **P1 task count was wrong**: Summary table said 40 P1 tasks but actual count was 41 (tasks 01-49 = 49 tasks, of which 41 are P1 and 8 are P2). The Requirement Coverage table also said "Total: 48" but should have been 49. Both corrected.

**New tasks added (3):**
1. **Task 50 — CLI entry point integration (P1)**: The existing `src/mle_star/cli.py` skeleton prints "Hello from mle_star!" but no task wired it to `run_pipeline_sync()`. Without this, `uv run mle_star` does nothing useful. Task 50 adds argparse-based CLI with task YAML loading, optional config override, and FinalResult output. Depends on Task 42 (orchestrator) and Task 03 (config models).
2. **Task 51 — Shared test infrastructure (P1)**: Every task from Layer 2 onward needs to mock the SDK client, create test `SolutionScript`/`TaskDescription`/`PipelineConfig` instances, and set up temporary working directories. Without shared fixtures, each test module would independently reimplement these mocks. Task 51 centralizes this in `tests/conftest.py`. Depends on Tasks 03-05 (model layer).
3. **Task 52 — Prompts package initialization (P1)**: Task 02 creates YAML files in `src/mle_star/prompts/` but the directory needs an `__init__.py` to be a proper Python package. Without it, `importlib.resources.files("mle_star.prompts")` fails at runtime and `PromptRegistry` (Task 08) cannot discover template files. Also added a cross-reference in Task 02's acceptance criteria.

**Updated counts:** 44 P1 + 8 P2 = 52 total tasks covering 436 requirements. *(Note: v12 originally stated 43 P1 / 51 total due to a counting error; corrected in v13.)*

**Verified correct (no change needed):**
- All 436 requirements from specs 01-09 still covered
- Task priorities and dependencies for tasks 01-49 unchanged
- All previously documented cross-cutting constraints remain accurate
- Pre-commit hook constraints from v11 remain applicable

---

### 2026-02-21 (v11)

Re-analysis of all 36 spec files and project tooling configuration against v10 plan. Codebase still empty (skeleton CLI only) — all 48 tasks remain pending.

**Gaps fixed:**
1. **Pre-commit hook constraints (cross-cutting)**: Added comprehensive cross-cutting note documenting pre-commit hook requirements from `.pre-commit-config.yaml`. The hook chain enforces: ruff lint with Google-style docstrings (D rules), ruff format, mypy strict, bandit security, xenon complexity (max B), and pip-audit vulnerability checks. These gates apply to every task and were not previously documented in the plan. Particularly impactful: (a) all public functions/classes/methods need Google-style docstrings to pass ruff D101/D102/D103, (b) xenon complexity cap at B may require decomposing complex orchestration functions (e.g., `run_phase1`, `run_phase2_outer_loop`, `run_phase3`), (c) bandit scans will flag subprocess usage patterns that aren't pre-skipped.

**Verified correct (no change needed):**
- All 436 requirements still covered across 48 tasks
- Requirement Coverage table unchanged
- Task priorities and dependencies unchanged
- All previously documented cross-cutting constraints remain accurate
- `requires-python = ">=3.13"` confirmed in pyproject.toml; dev venv uses Python 3.14 (compatible)
- pytest-asyncio configured with `asyncio_mode = "auto"` (no manual `@pytest.mark.asyncio` needed)
- Coverage threshold (90%) in pyproject.toml matches task acceptance criteria

---

### 2026-02-21 (v10)

Re-analysis of all 36 spec files against v9 plan. Codebase still empty (skeleton CLI only) — all 48 tasks remain pending.

**Bugs fixed:**
1. **Task 33 (Outer loop orchestration) — description/signature mismatch**: The description's step 5 still said "update best solution only if inner loop `improved` (strict >)" despite the v6 correction to the acceptance criteria. This created a contradiction within the same task — the description implied strict `>` while the acceptance criteria correctly specified `is_improvement_or_equal()` (>=). Fixed description to match the acceptance criteria and REQ-P2O-027.
2. **Task 33 (Outer loop orchestration) — missing `initial_score` parameter**: The function signature `run_phase2_outer_loop(client, task, config, initial_solution, session_id)` was missing the `initial_score: float` parameter that REQ-P2O-019 explicitly includes. This parameter matters because `SolutionScript.score` is `float | None` but `h_best` must be a guaranteed `float` for the outer loop's comparison logic. Fixed signature to `run_phase2_outer_loop(client, task, config, initial_solution, initial_score, session_id)`. Also updated Task 44 to note that `Phase1Result.initial_score` is passed to each Phase 2 path.

**Verified correct (no change needed):**
- All 436 requirements still covered across 48 tasks
- Requirement Coverage table unchanged
- Task priorities and dependencies unchanged
- All previously documented cross-cutting constraints remain accurate
- REQ-P2I-036 (`InnerLoopResult`) correctly mapped to Task 10 (models layer)
- REQ-SF-034 type contract signatures correctly augmented with `client` and `task` per cross-cutting convention
- REQ-DM-050 (no external deps beyond Pydantic for models module) naturally satisfied by design

---

### 2026-02-21 (v9)

Re-analysis of all 36 spec files against v8 plan. Codebase still empty (skeleton CLI only) — all 48 tasks remain pending.

**Gaps fixed:**
1. **SDK client parameter threading**: Added cross-cutting note documenting that all functions invoking SDK agents must receive `client: ClaudeSDKClient` as a parameter. Several spec function signatures omit `client` (REQ-SF-006, REQ-P2I-005, REQ-P2I-013, REQ-P2I-016, REQ-FN-009, REQ-FN-019, REQ-FN-034) despite requiring agent invocation — these are treated as under-specified and the plan adds `client` to their signatures.
2. **Task 03 (PhaseTimeBudget)**: Added explicit field definitions for `PhaseTimeBudget` model (`phase1_pct`, `phase2_pct`, `phase3_pct`, `finalization_pct` with sum=100 validator). Previously, Task 03 referenced `PhaseTimeBudget` as a type on `PipelineConfig` without defining its fields, which would force implementers to infer field names.
3. **Task 19 (Debugger)**: Added `client` to `debug_solution(solution, traceback, task, config, client)` and `make_debug_callback(task, config, client)` signatures. These functions invoke A_debugger via the SDK but the spec signatures (REQ-SF-006/007) omitted `client`.
4. **Task 23 (Coder/planner agents)**: Replaced `...` with explicit `client` parameter in `invoke_coder(code_block, plan, client)` and `invoke_planner(code_block, plans, scores, client)` signatures. Spec signatures (REQ-P2I-005/013) omitted `client`.
5. **Task 24 (Inner loop orchestration)**: Added `client` as first parameter to `run_phase2_inner_loop(client, solution, code_block, initial_plan, best_score, task, config)`. The spec signature (REQ-P2I-016) omitted `client` but the function calls `invoke_coder`, `invoke_planner`, `check_and_fix_leakage`, and `make_debug_callback`, all of which require the SDK client.
6. **Task 38 (Subsampling removal)**: Normalized `remove_subsampling` signature to `remove_subsampling(client, solution, task)` with `client` as first parameter, consistent with the cross-cutting convention. Spec signature (REQ-FN-009) omitted `client`.
7. **Task 39 (Test submission)**: Removed `config` from `generate_test_submission(client, task, solution)` signature to match REQ-FN-019 (which specifies only task + solution inputs). Added `client` per cross-cutting convention.
8. **Task 40 (Contamination/finalization)**: Added `client` as first parameter to both `check_contamination(client, solution, reference_discussions)` and `run_finalization(client, solution, task, config, ...)`. Spec signature for `run_finalization` (REQ-FN-034) omitted `client` but the function invokes `remove_subsampling`, `generate_test_submission`, `check_and_fix_leakage`, `make_debug_callback`, and `check_contamination`, all requiring the SDK client.

**Verified correct (no change needed):**
- All 436 requirements still covered across 48 tasks
- Requirement Coverage table unchanged
- Task priorities and dependencies unchanged
- All previously documented cross-cutting constraints remain accurate
- Phase-level entry points `run_phase1(client, ...)`, `run_phase2_outer_loop(client, ...)`, and `run_phase3(client, ...)` already had `client` — no change needed

---

### 2026-02-21 (v8)

Re-analysis of all 36 spec files against v7 plan. Codebase still empty (skeleton CLI only) — all 48 tasks remain pending.

**Gaps fixed:**
1. **Task 03 (Core configuration models)**: Added `log_file: str | None = None` field to PipelineConfig acceptance criteria per REQ-OR-047 ("if PipelineConfig.log_file is set, append logs to the specified file"). This optional field was missing from PipelineConfig despite being specified in the orchestrator spec.
2. **Task 17 (Batch evaluation)**: Corrected cross-reference typo — `evaluate_batch()` sequential requirement references REQ-EX-026, not REQ-EX-020 (which is `request_subsample_extraction`). Comment-only fix; no behavioral change.
3. **Task 39 (Test submission)**: Added explicit `clean_output_directory()` step before test script execution per REQ-FN-020 ("clean ./final/ first"). The clean step was implied by the orchestration sequence in Task 40 but not listed in Task 39's own acceptance criteria.

**Verified correct (no change needed):**
- All 436 requirements still covered across 48 tasks
- Requirement Coverage table unchanged
- Task priorities and dependencies unchanged
- All previously documented cross-cutting constraints remain accurate
- REQ-OR-006 vs REQ-SF-024 spec discrepancy (A_data output_schema) correctly resolved in favor of REQ-SF-024 (output_schema=None)

---

### 2026-02-21 (v7)

Re-analysis of all 36 spec files (including constraint sections 02d, 03d, 05d, 08d) against v6 plan. Codebase still empty (skeleton CLI only) — all 48 tasks remain pending.

**Missing details added:**
1. **Task 12 (Script validation)**: Added `os._exit()` and `quit()` to rejection patterns per REQ-EX-044 extended list (was only `exit()` and `sys.exit()`).
2. **Task 13 (Async execution)**: Added orphan child process cleanup via `os.killpg` on process group after timeout (REQ-EX-037).
3. **Task 27 (Phase 1 agents)**: Added exclusion of models with empty `example_code` from candidate generation (REQ-P1-006). Added `source_model` field set to `RetrievedModel.model_name` in `generate_candidate()`.
4. **Task 28 (Phase 1 orchestration)**: Specified exact error message format: `RuntimeError("Phase 1 failed: all {M} candidates produced execution errors")` per REQ-P1-022.
5. **Task 32 (Summarize/extractor agents)**: Added A_summarize fallback behavior — truncate raw output to last 2000 chars with prefix `"[Auto-summary from raw output] "` per REQ-P2O-036. Added A_extractor malformed JSON retry (once before validation recovery) per REQ-P2O-034.
6. **Task 33 (Outer loop orchestration)**: Added `was_skipped=True` field for skipped iterations in step history (REQ-P2O-030). Added `CodeBlock.outer_step=t` provenance tracking (REQ-P2O-044). Added `Phase2Result.best_score` guarantee — never worse than `initial_score` (REQ-P2O-029).
7. **Task 36 (Phase 3 orchestration)**: Added failed ensembler recording with empty content `SolutionScript` (REQ-P3-030). Added failed ens_planner history placeholder `"[ens_planner failed]"` (REQ-P3-031).

**Verified correct (no change needed):**
- All 436 requirements still covered across 48 tasks
- Requirement Coverage table unchanged
- Task priorities and dependencies unchanged
- All previously documented cross-cutting constraints remain accurate
- Tool assignments per REQ-OR-008 correctly reflected in Task 09

---

### 2026-02-21 (v6)

Re-analysis of all 36 spec files (including constraint/traceability sections 04c, 04d, 05d, 06c, 06d, 07c, 07d, 08d, 09d) against v5 plan. Codebase still empty (skeleton CLI only) — all 48 tasks remain pending.

**Bugs fixed:**
1. **Task 33 (Outer loop orchestration)**: Corrected outer loop best-score update semantics from strict `>` (via `InnerLoopResult.improved` flag) to `>=` (via `is_improvement_or_equal()`) per REQ-P2O-027. The outer loop must independently compare `inner_result.best_score` against `h_best` using `is_improvement_or_equal()`, NOT rely on the `improved` flag (which uses strict `>`). Example: if inner loop starts with h_best=0.85 and returns best_score=0.85, `improved=False` but outer loop should still adopt the solution.
2. **Gap Analysis cross-cutting note**: Corrected score comparison semantics — `is_improvement_or_equal()` (>=) applies to BOTH inner loop best-score tracking AND outer loop `s_final` update (REQ-P2O-027). `is_improvement()` (strict >) is ONLY for `InnerLoopResult.improved` flag.

**Missing details added:**
1. **Task 03 (Core configuration models)**: Added `phase_time_budget: PhaseTimeBudget | None = None` field to `PipelineConfig` acceptance criteria per REQ-OR-025 (proportional time allocation is an optional PipelineConfig field). Added REQ-OR-025 to task's Reqs reference.

**Verified correct (no change needed):**
- All 436 requirements still covered across 48 tasks
- Requirement Coverage table unchanged
- Task priorities and dependencies unchanged
- All other cross-cutting constraints remain accurate
- Non-functional requirements from specs 04c, 06c, 06d, 07c, 08d correctly mapped to P2 constraint tasks (30, 26, 37, 41, 49)
- Traceability sections (04d, 05d, 07d, 09d) reference same requirement IDs already covered by functional/orchestration tasks

---

### 2026-02-20 (v5)

Re-analysis of all 36 spec files against v4 plan. Codebase still empty (skeleton CLI only) — all 48 tasks remain pending.

**Gaps fixed:**
1. **Task 02 (Prompt templates)**: Added `contamination_check` as explicit 3rd A_test variant (Figure 28, REQ-FN-027). The total of 18 templates was already correct but the breakdown only listed 2 test variants (subsampling_extract, subsampling_remove); now lists all 3 additional variants. Clarified count: 11 base + 2 leakage variants + 1 test base + 3 test variants + 1 data = 18.
2. **Task 40 (Contamination check)**: Added explicit note that contamination check variant uses `tools=None` and `output_schema=DataContaminationResult`, differing from default A_test config (`tools=["Read"]`, `output_schema=None`), per REQ-FN-048.
3. **Task 47 (Result assembly)**: Added `failed=True` flag requirement for failed Phase 2 path `step_history` per REQ-OR-040.
4. **Gap Analysis section**: Added 5 new cross-cutting clarifications:
   - Score mutation pattern: harness does NOT mutate SolutionScript; phase code updates `solution.score`
   - Session ID strategy: `"phase-1"`, `"path-{i}"`, `"phase-3"`, `"finalization"`
   - Fallback chain priority: Phase3 > Phase2 > Phase1, with specific failure behaviors
   - Time budget redistribution algorithm after Phase 1 completion
   - A_test multi-mode agent summary (4 operational modes with variant-specific config overrides)

**Verified correct (no change needed):**
- All 436 requirements still covered across 48 tasks
- Requirement Coverage table unchanged
- Task priorities and dependencies unchanged
- All previously documented cross-cutting constraints remain accurate

---

### 2026-02-20 (v4)

Full re-analysis of all 36 spec files (including 07c, 07d, 08b, 08c, 08d, 09c, 09d) against the v3 plan. Codebase still empty (skeleton CLI only) — all 48 tasks remain pending.

**Bugs fixed:**
1. **Task 36 (Phase 3 orchestration)**: Corrected `run_phase3` to handle `len(solutions) == 1` gracefully per REQ-P3-018 (returns immediately with single solution as `best_ensemble`), instead of incorrectly raising `ValueError`. Only raises `ValueError` for empty input (`len(solutions) == 0`). REQ-P3-003 applies to `invoke_ens_planner` (requires >= 2), not to `run_phase3` itself.
2. **Task 40 (Contamination check)**: Corrected agent type from "A_data agent" to "contamination check variant of A_test agent (`AgentType.test`)" per REQ-FN-026 which explicitly states `agent_type = AgentType.test`.
3. **Task 40 (run_finalization signature)**: Corrected signature to match REQ-FN-034 — includes `phase1_result`, `phase2_results`, `phase3_result` parameters needed for `FinalResult` construction (REQ-FN-036). Added explicit fallback handling step (REQ-FN-025).
4. **Task 09 (A_data output_schema)**: Corrected from `output_schema=DataContaminationResult` to `output_schema=None` per REQ-SF-025. The `DataContaminationResult` schema is used by the contamination check variant of A_test (REQ-FN-026/048), not by A_data.

**Verified correct (no change needed):**
- All 436 requirements still covered across 48 tasks
- Requirement Coverage table unchanged
- Task priorities and dependencies unchanged
- All cross-cutting constraints in Gap Analysis section remain accurate

---

### 2026-02-20 (v3)

Full re-analysis of all 36 spec files against the existing v2 plan and codebase. Codebase confirmed empty (skeleton CLI only) — all 48 tasks remain pending.

**Clarifications and refinements:**
1. **Gap Analysis section**: Added cross-cutting constraint notes for REQ-SF-006/008 layering (debug_solution returns final pair; calling code handles fallback) and score comparison semantics (>= for tracking, > for improved flag). Added Python 3.13+ target note.
2. **Task 19 (Debugger)**: Clarified REQ-SF-006 vs REQ-SF-008 layering — `debug_solution()` returns the final attempted pair (may still be broken); calling code maintains last-known-working reference for fallback. Clarified `make_debug_callback()` wraps a single invocation (no retry loop).
3. **Task 32 (Extractor validation)**: Expanded single "validation failure recovery" criterion into 4 explicit levels per REQ-P2O-018: (1) whitespace-normalized match, (2) re-invoke with error feedback up to 2 times, (3) try alternate plans from ExtractorOutput list, (4) skip iteration.
4. **Task 35 (Ensemble planner)**: Added explicit history formatting requirement — `Plan {r}:` / `Score {r}:` labels consistent with inner loop planner format.
5. **REQ-OR-008 note**: Added explicit example of tool assignment override (A_debugger: spec says `["Read", "Bash"]`, orchestrator mandates `["Bash", "Edit", "Write", "Read"]`).

**Verified correct (no change needed):**
- Task 24 already correctly specifies `InnerLoopResult.improved` uses strict `is_improvement()` (line 553)
- Task 25 already correctly specifies failed planner attempts included in history for A_planner (line 573)
- Task 23 already correctly specifies history format with Plan:/Score: labels (line 527)
- All 436 requirements covered across 48 tasks

---

### 2026-02-20 (v2)

Gap analysis against all 40 spec files identified and corrected the following:

**Bugs fixed:**
1. **Task 05**: `RefinePlan` fields corrected to `code_block: str, plan: str` (was `plan, expected_score`). `LeakageAnswer` now includes `code_block: str` field (was missing). All schema fields now match REQ-DM-014 through REQ-DM-020 exactly.
2. **Task 09**: Tool assignments corrected per REQ-OR-008 — execution-capable agents get `["Bash", "Edit", "Write", "Read"]` (was `["Bash"]`). Read-only agents get `["Read"]`.
3. **Task 14**: Added LAST match requirement for `parse_score()` per REQ-EX-011 (was ambiguous).
4. **Task 17**: `evaluate_batch()` corrected to sequential execution per REQ-EX-020 (was "concurrently").

**Missing details added:**
1. **Task 03**: Added `max_budget_usd` (REQ-OR-028), `permission_mode` (REQ-OR-009), `model` (REQ-OR-044), `log_level` (REQ-OR-047) to PipelineConfig. Added TaskDescription fields.
2. **Task 10**: Added `InnerLoopResult` model from REQ-P2I-036 with `improved` flag semantics (strict `is_improvement`).
3. **Task 19**: Added REQ-SF-010 (append "Final Validation Performance" if missing from debugged code). Clarified `extract_code_block` returns longest fenced block.
4. **Task 20**: Added REQ-SF-021 (replace_block ValueError → log warning and skip). Added graceful degradation criterion.
5. **Task 21**: Added "All the provided information is used." passthrough detection. Added REQ-SF-030 (runs exactly once) note.
6. **Task 24**: Comprehensive algorithmic detail — k=0 no planner call, A_coder receives original code block, replace_block against original solution, >= semantics, exactly K attempts.
7. **Task 25**: Detailed error handling per REQ-P2I-032/033/034/035 with specific RefinementAttempt field values.
8. **Task 31**: Added ablation timeout formula: `min(time_limit / (outer_steps * 2), 600)` (REQ-P2O-035).
9. **Task 33**: Clarified FIRST plan selection, strict > for outer loop update, inner loop handoff parameters.
10. **Task 36**: Added ValueError for < 2 solutions (REQ-P3-003), tie-breaking rule (REQ-P3-025), exactly R records (REQ-P3-044).
11. **Task 42**: Added `run_pipeline_sync()` wrapper (REQ-OR-053), `PipelineError` (REQ-OR-042), `PipelineTimeoutError` (REQ-OR-030), MCP server registration (REQ-OR-010).
12. **Task 44**: Added session forking from Phase 1 (REQ-OR-021).
13. **Task 48**: Added env var precedence logic (REQ-OR-046), PipelineState fields (REQ-OR-050), EnvironmentError for missing API key.
14. **Task 49**: Added specific constraint details — reconnection parameters, memory policy, serialization fallback.
15. **Cross-cutting**: Added key constraints summary in Gap Analysis section.
