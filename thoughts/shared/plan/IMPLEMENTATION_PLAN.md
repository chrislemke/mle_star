# Implementation Plan

> This file tracks all tasks for the project. Update it as tasks are completed.

---

## Summary

| Priority | Pending | In Progress | Done |
|----------|---------|-------------|------|
| P1       | 49      | 0           | 0    |
| P2       | 4       | 0           | 0    |
| P3       | 3       | 0           | 0    |

---

## Tasks

<!-- ============================================================ -->
<!-- SECTION 0: PROJECT SETUP                                      -->
<!-- ============================================================ -->

## [P1] 0.1 — Add runtime dependencies to pyproject.toml
**Status:** pending
**Priority:** P1

### Description
Add `pydantic>=2.0.0` and `claude-agent-sdk>=0.1.39` as runtime dependencies in `pyproject.toml`. These are required by every spec in the project. REQ-DM-048 mandates Pydantic v2; REQ-OR-054 mandates claude-agent-sdk v0.1.39+.

### Acceptance Criteria
- [ ] `pydantic>=2.0.0` listed under `[project] dependencies`
- [ ] `claude-agent-sdk>=0.1.39` listed under `[project] dependencies`
- [ ] `uv sync` succeeds without errors
- [ ] `uv run python -c "import pydantic; import claude_agent_sdk"` succeeds

---

## [P1] 0.2 — Create module skeleton files
**Status:** pending
**Priority:** P1

### Description
Create the module files specified by each spec's single-module constraint (REQ-DM-045, REQ-EX-046, etc.): `models.py`, `execution.py`, `safety.py`, `phase1.py`, `phase2_outer.py`, `phase2_inner.py`, `phase3.py`, `finalization.py`, `orchestrator.py`, and `prompts.py`. Each file starts with docstring and necessary imports only.

### Acceptance Criteria
- [ ] All 10 module files exist under `src/mle_star/`
- [ ] Each file has a module docstring matching its spec's purpose
- [ ] `uv run mypy --config-file=pyproject.toml src` passes (no import errors)
- [ ] `uv run ruff check .` passes

---

<!-- ============================================================ -->
<!-- SECTION 1: DATA MODELS AND INTERFACES (Spec 01)               -->
<!-- ============================================================ -->

## [P1] 1.1 — Implement enums (TaskType, DataModality, MetricDirection, SolutionPhase, CodeBlockCategory, AgentType)
**Status:** pending
**Priority:** P1

### Description
Implement all six string enums in `mle_star/models.py` as specified in REQ-DM-004 through REQ-DM-008, REQ-DM-011, and REQ-DM-013. `AgentType` must have exactly 14 values.

### Acceptance Criteria
- [ ] `TaskType` has 8 values per REQ-DM-004
- [ ] `DataModality` has 5 values per REQ-DM-005
- [ ] `MetricDirection` has 2 values per REQ-DM-006
- [ ] `SolutionPhase` has 5 values per REQ-DM-008
- [ ] `CodeBlockCategory` has 8 values per REQ-DM-011
- [ ] `AgentType` has exactly 14 values per REQ-DM-013
- [ ] All enums are string enums (`str, Enum`)
- [ ] Tests verify enum membership and `len()` assertions
- [ ] Mypy passes

---

## [P1] 1.2 — Implement PipelineConfig model
**Status:** pending
**Priority:** P1

### Description
Implement the `PipelineConfig` Pydantic model in `mle_star/models.py` with all fields from REQ-DM-001, positive-int validation per REQ-DM-002, JSON round-trip per REQ-DM-003, and the `max_budget_usd` field per REQ-OR-028. Use `ConfigDict(frozen=True)` per REQ-DM-039.

### Acceptance Criteria
- [ ] All 8+ fields with paper-specified defaults (M=4, T=4, K=4, L=2, R=5, etc.)
- [ ] `PipelineConfig()` instantiates with defaults matching the paper
- [ ] `PipelineConfig(num_retrieved_models=0)` raises `ValidationError`
- [ ] JSON round-trip via `model_dump_json()` / `model_validate_json()` preserves values
- [ ] Model is frozen (assignment raises error)
- [ ] Tests cover all validations with hypothesis for property-based checking

---

## [P1] 1.3 — Implement TaskDescription and SolutionScript models
**Status:** pending
**Priority:** P1

### Description
Implement `TaskDescription` (REQ-DM-007) and `SolutionScript` (REQ-DM-009, REQ-DM-010) Pydantic models. `SolutionScript` must include `replace_block()` method and use `frozen=False` per REQ-DM-039. `TaskDescription` uses `frozen=True`.

### Acceptance Criteria
- [ ] `TaskDescription` has all required fields per REQ-DM-007
- [ ] Missing required fields raise `ValidationError`
- [ ] `SolutionScript` has all fields per REQ-DM-009 with `created_at` auto-set
- [ ] `replace_block(old, new)` returns new `SolutionScript` with substitution
- [ ] `replace_block()` raises `ValueError` when `old` not found
- [ ] Tests cover `replace_block` edge cases (multiple occurrences: only first replaced)

---

## [P1] 1.4 — Implement CodeBlock and structured output schemas
**Status:** pending
**Priority:** P1

### Description
Implement `CodeBlock` (REQ-DM-012), `RetrievedModel` (REQ-DM-014), `RetrieverOutput` (REQ-DM-015), `RefinePlan` (REQ-DM-016), `ExtractorOutput` (REQ-DM-017), `LeakageAnswer` (REQ-DM-018), `LeakageDetectionOutput` (REQ-DM-019), and `DataContaminationResult` (REQ-DM-020). Each with `model_json_schema()` compatibility per REQ-DM-041.

### Acceptance Criteria
- [ ] All 8 models defined with correct fields and types
- [ ] `RetrieverOutput` validator ensures `len(models) >= 1`
- [ ] `ExtractorOutput` validator ensures `len(plans) >= 1`
- [ ] `LeakageDetectionOutput` validator ensures `len(answers) >= 1`
- [ ] `LeakageAnswer.leakage_status` uses `Literal["Yes Data Leakage", "No Data Leakage"]`
- [ ] `DataContaminationResult.verdict` uses `Literal["Novel", "Same"]`
- [ ] `.model_json_schema()` produces valid JSON schema for all output models
- [ ] All models use `frozen=True`

---

## [P1] 1.5 — Implement EvaluationResult and phase result models
**Status:** pending
**Priority:** P1

### Description
Implement `EvaluationResult` (REQ-DM-021), `Phase1Result` (REQ-DM-022), `Phase2Result` (REQ-DM-023), `Phase3Result` (REQ-DM-024), `FinalResult` (REQ-DM-025), `RefinementAttempt` (REQ-DM-042), and `EnsembleAttempt` (REQ-DM-043) models.

### Acceptance Criteria
- [ ] `EvaluationResult` has all 7 fields per REQ-DM-021
- [ ] `Phase1Result` through `FinalResult` have all fields per REQ-DM-022–025
- [ ] `RefinementAttempt` has 4 fields per REQ-DM-042
- [ ] `EnsembleAttempt` has 3 fields per REQ-DM-043
- [ ] `FinalResult.phase3` is `Phase3Result | None`
- [ ] Tests verify construction and field access

---

## [P1] 1.6 — Implement score function interface and comparison utilities
**Status:** pending
**Priority:** P1

### Description
Implement the `ScoreFunction` protocol (REQ-DM-026), default score parsing with regex per REQ-DM-027, `is_improvement()` per REQ-DM-028, and `is_improvement_or_equal()` per REQ-DM-029.

### Acceptance Criteria
- [ ] `ScoreFunction` is a `Protocol` with `__call__(self, solution, task) -> EvaluationResult`
- [ ] `parse_score()` matches regex `r"Final Validation Performance:\s*([\d.eE+-]+)"`
- [ ] `parse_score("Final Validation Performance: 0.8196")` returns `0.8196`
- [ ] Returns `None` when no match found
- [ ] `is_improvement(0.9, 0.8, "maximize")` returns `True`
- [ ] `is_improvement(0.7, 0.8, "minimize")` returns `True`
- [ ] `is_improvement_or_equal` handles equality case per REQ-DM-029
- [ ] Tests cover edge cases (scientific notation, negative scores, equal scores)

---

## [P1] 1.7 — Implement PromptTemplate, PromptRegistry, and AgentConfig
**Status:** pending
**Priority:** P1

### Description
Implement `PromptTemplate` (REQ-DM-030) with `render()` method (REQ-DM-031), `PromptRegistry` (REQ-DM-032) with `get()` supporting optional `variant` parameter for leakage (REQ-DM-034) and subsampling (REQ-DM-035), and `AgentConfig` (REQ-DM-036) with `to_agent_definition()` (REQ-DM-037) and `to_output_format()` (REQ-DM-038).

### Acceptance Criteria
- [ ] `PromptTemplate.render(**kwargs)` substitutes placeholders; raises `KeyError` on missing vars
- [ ] `PromptRegistry.get(agent_type)` returns template; raises `KeyError` if unregistered
- [ ] `PromptRegistry.get(AgentType.leakage, variant="detection")` works per REQ-DM-034
- [ ] `PromptRegistry.get(AgentType.test, variant="subsampling_extract")` works per REQ-DM-035
- [ ] `AgentConfig.to_agent_definition()` returns correct dict shape
- [ ] `AgentConfig.to_output_format()` returns `{"type": "json_schema", "schema": ...}` when `output_schema` set
- [ ] Tests cover all methods and error cases

---

## [P1] 1.8 — Implement build_default_agent_configs and __init__.py re-exports
**Status:** pending
**Priority:** P1

### Description
Implement `build_default_agent_configs()` factory function per REQ-DM-040 returning configs for all 14 agents with appropriate tools and output schemas. Update `__init__.py` to re-export all public models per REQ-DM-046.

### Acceptance Criteria
- [ ] `build_default_agent_configs()` returns `dict[AgentType, AgentConfig]` with 14 entries
- [ ] Each agent has correct `tools` list per REQ-OR-008
- [ ] Agents requiring structured output have `output_schema` set (retriever, extractor, leakage, data)
- [ ] `from mle_star import PipelineConfig, TaskDescription, SolutionScript` works
- [ ] Tests verify `len(build_default_agent_configs()) == 14`

---

<!-- ============================================================ -->
<!-- SECTION 2: EXECUTION HARNESS (Spec 02)                        -->
<!-- ============================================================ -->

## [P1] 2.1 — Implement working directory setup and GPU detection
**Status:** pending
**Priority:** P1

### Description
Implement `setup_working_directory(task)` per REQ-EX-001, `clean_output_directory()` per REQ-EX-002, and `detect_gpu_info()` per REQ-EX-003 in `mle_star/execution.py`. Also implement `build_execution_env()` per REQ-EX-004.

### Acceptance Criteria
- [ ] `setup_working_directory()` creates/verifies `./input/` and `./final/` directories
- [ ] `clean_output_directory()` empties the `./final/` directory
- [ ] `detect_gpu_info()` returns GPU count and type (or indicates no GPU)
- [ ] `build_execution_env()` returns env dict with GPU and data path info
- [ ] Tests verify directory creation and GPU detection on CPU-only machines

---

## [P1] 2.2 — Implement write_script and execute_script
**Status:** pending
**Priority:** P1

### Description
Implement `write_script()` per REQ-EX-005/REQ-EX-006 (write solution to temp file with validation including `exit()`/`sys.exit()`/`os._exit()`/`quit()` detection per REQ-EX-044) and `execute_script()` per REQ-EX-007 (async subprocess execution with stdout/stderr capture, timeout enforcement per REQ-EX-009, orphan process cleanup via `os.killpg` per REQ-EX-037, large output truncation per REQ-EX-038). Return `ExecutionRawResult` per REQ-EX-008. Include executor strategy selection: prefer SDK Bash tool when available, fall back to direct subprocess when SDK Bash timeout cap (600s) is exceeded per REQ-EX-047.

### Acceptance Criteria
- [ ] `write_script()` writes to temp file and validates content is non-empty Python
- [ ] `write_script()` detects `exit()`, `sys.exit()`, `os._exit()`, `quit()` calls per REQ-EX-044
- [ ] Advisory `try/except` detection (warn, not block) per REQ-EX-045
- [ ] `execute_script()` runs script as subprocess, captures stdout/stderr
- [ ] Timeout enforcement terminates scripts exceeding limit via `os.killpg` (no orphan processes) per REQ-EX-037
- [ ] Large output (>100MB) truncated with warning appended per REQ-EX-038
- [ ] Executor strategy: SDK Bash when timeout ≤ 600s, direct subprocess otherwise per REQ-EX-047
- [ ] `ExecutionRawResult` has exit_code, stdout, stderr, duration_seconds per REQ-EX-008
- [ ] Tests use a simple script that prints a known value
- [ ] Tests verify timeout kills long-running scripts without orphan processes

---

## [P1] 2.3 — Implement score parsing, traceback extraction, and error detection
**Status:** pending
**Priority:** P1

### Description
Implement `parse_score()` per REQ-EX-011 (regex matching `"Final Validation Performance: {score}"`), `extract_traceback()` per REQ-EX-012, and `detect_error()` per REQ-EX-013 in `mle_star/execution.py`.

### Acceptance Criteria
- [ ] `parse_score(stdout)` extracts float from the regex pattern
- [ ] Returns `None` when pattern not found
- [ ] `extract_traceback(stderr)` extracts Python traceback block
- [ ] `detect_error(exit_code, stderr)` returns `(is_error, error_traceback)`
- [ ] Tests cover: valid score, no score, scientific notation, tracebacks of various lengths

---

## [P1] 2.4 — Implement evaluate_solution and evaluate_with_retry
**Status:** pending
**Priority:** P1

### Description
Implement `build_evaluation_result()` per REQ-EX-014, `evaluate_solution()` per REQ-EX-015 (write + execute + parse + construct EvaluationResult), and `evaluate_with_retry()` per REQ-EX-021 (retry with A_debugger on failure). Also implement `is_better_solution()` per REQ-EX-023. Critical: the harness shall NOT mutate the input `SolutionScript` per REQ-EX-016; the caller must handle score updates.

### Acceptance Criteria
- [ ] `evaluate_solution()` returns `EvaluationResult` from a `SolutionScript`
- [ ] `evaluate_solution()` does NOT mutate the input `SolutionScript` per REQ-EX-016
- [ ] Sets `score=None` and `is_error=True` when script fails
- [ ] `evaluate_with_retry()` calls A_debugger callback on error, retries up to `max_debug_attempts`
- [ ] Falls back to original solution after max retries exceeded
- [ ] `is_better_solution()` delegates to `is_improvement_or_equal()` from models
- [ ] Tests verify successful evaluation, error detection, retry flow
- [ ] Tests verify input SolutionScript is not mutated after evaluation

---

## [P1] 2.5 — Implement subsampling utilities
**Status:** pending
**Priority:** P1

### Description
Implement subsampling instruction generation per REQ-EX-017 (adding subsampling instruction to prompts) and subsampling removal utilities per REQ-EX-018. These are used during Phase 2 refinement and finalization respectively.

### Acceptance Criteria
- [ ] Subsampling instruction text references `config.subsample_limit` (default 30000)
- [ ] Instruction text follows spec wording for agent prompts
- [ ] Tests verify instruction contains the limit value

---

## [P1] 2.6 — Implement submission verification and batch evaluation
**Status:** pending
**Priority:** P1

### Description
Implement `verify_submission()` per REQ-EX-021 (check `./final/submission.csv` exists, is non-empty CSV), `get_submission_info()` per REQ-EX-022, `evaluate_batch()` per REQ-EX-023, and `rank_solutions()` per REQ-EX-024.

### Acceptance Criteria
- [ ] `verify_submission()` returns bool; checks file exists, is valid CSV with rows
- [ ] `get_submission_info()` returns row count and column names
- [ ] `evaluate_batch()` evaluates multiple solutions, returns list of `EvaluationResult`
- [ ] `rank_solutions()` sorts solutions by score respecting metric direction
- [ ] Tests cover valid/invalid/missing submission files

---

<!-- ============================================================ -->
<!-- SECTION 3: SAFETY MODULES (Spec 03)                           -->
<!-- ============================================================ -->

## [P1] 3.1 — Implement A_debugger (debug_solution, make_debug_callback)
**Status:** pending
**Priority:** P1

### Description
Implement `debug_solution()` per REQ-SF-001 through REQ-SF-010 and `make_debug_callback()` per REQ-SF-007 in `mle_star/safety.py`. The debugger receives code + traceback, invokes the SDK `debugger` agent, and returns fixed code. Includes retry logic, fallback to original solution per REQ-SF-008, and auto-append of score print line if missing per REQ-SF-010.

### Acceptance Criteria
- [ ] `debug_solution(client, solution, traceback, task)` invokes `debugger` agent via SDK
- [ ] Extracts fixed code from agent response using `extract_code_block()`
- [ ] Returns new `SolutionScript` with fixed content
- [ ] Auto-appends `print("Final Validation Performance: ...")` line if missing from debugged code per REQ-SF-010
- [ ] Auto-append respects `if __name__` blocks (injects inside, not after)
- [ ] Falls back to original solution if agent returns unparseable output per REQ-SF-008/REQ-SF-040
- [ ] `make_debug_callback()` returns a callable compatible with `evaluate_with_retry()`
- [ ] Tests verify agent invocation, code extraction, score line injection, and fallback behavior (mocked SDK)

---

## [P1] 3.2 — Implement A_leakage (detect + correct pipeline)
**Status:** pending
**Priority:** P1

### Description
Implement the two-step leakage pipeline per REQ-SF-011 through REQ-SF-023 in `mle_star/safety.py`: detection via `LeakageDetectionOutput` structured output, correction via code block replacement, and the combined `check_and_fix_leakage()` function. **Cross-cutting concern (REQ-SF-022):** leakage check must run before EVERY evaluation across all phases (Phase 1 post-merge, Phase 2 inner loop, Phase 3 ensemble). Callers in Tasks 4.5, 6.3, and 7.2 must invoke this function.

### Acceptance Criteria
- [ ] Detection invokes `leakage` agent with `variant="detection"` prompt per REQ-SF-012
- [ ] Parses `LeakageDetectionOutput` structured response per REQ-SF-014/REQ-SF-015
- [ ] If leakage detected, invokes `leakage` agent with `variant="correction"` prompt per REQ-SF-017
- [ ] Correction extracts fixed preprocessing code block and replaces in solution per REQ-SF-019
- [ ] Graceful handling when replacement fails (whitespace differences): log and skip per REQ-SF-021
- [ ] `check_and_fix_leakage()` combines both steps; returns (fixed_solution, was_fixed) per REQ-SF-020
- [ ] No-op when no leakage detected
- [ ] Graceful degradation on malformed JSON: return original solution per REQ-SF-038
- [ ] Tests cover: no leakage, leakage detected and corrected, unparseable response, replacement failure

---

## [P1] 3.3 — Implement A_data (check_data_usage)
**Status:** pending
**Priority:** P1

### Description
Implement `check_data_usage()` per REQ-SF-031 through REQ-SF-040 and `parse_data_agent_response()` in `mle_star/safety.py`. The data agent verifies all provided data files are used in the solution.

### Acceptance Criteria
- [ ] `check_data_usage(client, solution, task)` invokes `data` agent via SDK
- [ ] Agent receives list of data files from `task.data_dir` and solution content
- [ ] Returns data usage verification result
- [ ] Tests verify agent invocation with mocked SDK

---

## [P1] 3.4 — Implement extract_code_block utility
**Status:** pending
**Priority:** P1

### Description
Implement `extract_code_block()` per REQ-SF-041 through REQ-SF-046 in `mle_star/safety.py`. This utility extracts Python code from agent text responses (fenced code blocks, raw code).

### Acceptance Criteria
- [ ] Extracts code from ` ```python ... ``` ` fenced blocks
- [ ] Extracts code from ` ``` ... ``` ` unfenced blocks
- [ ] Falls back to full response if no code fence found
- [ ] Strips leading/trailing whitespace
- [ ] Returns `None` if response is empty or whitespace-only
- [ ] Tests cover all extraction patterns and edge cases

---

<!-- ============================================================ -->
<!-- SECTION 4: PHASE 1 — INITIAL SOLUTION (Spec 04)               -->
<!-- ============================================================ -->

## [P1] 4.1 — Implement retrieve_models (A_retriever invocation)
**Status:** pending
**Priority:** P1

### Description
Implement `retrieve_models()` per REQ-P1-001 through REQ-P1-010 in `mle_star/phase1.py`. Invokes the `retriever` agent with web search tools, receives `RetrieverOutput` structured response containing M models.

### Acceptance Criteria
- [ ] `retrieve_models(client, task, config)` invokes `retriever` agent
- [ ] Uses prompt template with `{M}` and `{task_description}` variables
- [ ] Parses `RetrieverOutput` from structured response
- [ ] Returns `list[RetrievedModel]` with `len >= 1`
- [ ] Truncates to M models if more returned
- [ ] Tests verify with mocked SDK client

---

## [P1] 4.2 — Implement generate_candidate (A_init invocation)
**Status:** pending
**Priority:** P1

### Description
Implement `generate_candidate()` per REQ-P1-011 through REQ-P1-020 in `mle_star/phase1.py`. Invokes the `init` agent for each retrieved model to produce a candidate solution script.

### Acceptance Criteria
- [ ] `generate_candidate(client, task, model, config)` invokes `init` agent
- [ ] Uses prompt template with task description, model name, example code, subsampling instruction
- [ ] Extracts code from agent response using `extract_code_block()`
- [ ] Returns `SolutionScript` with `phase="init"` and `source_model` set
- [ ] Tests verify code extraction and SolutionScript construction

---

## [P1] 4.3 — Implement merge_solutions (A_merger invocation)
**Status:** pending
**Priority:** P1

### Description
Implement `merge_solutions()` per REQ-P1-021 through REQ-P1-030 in `mle_star/phase1.py`. Invokes the `merger` agent to integrate a reference solution into a base solution.

### Acceptance Criteria
- [ ] `merge_solutions(client, base, reference, task)` invokes `merger` agent
- [ ] Uses prompt with base solution code, reference solution code, task description
- [ ] Extracts merged code from agent response
- [ ] Returns `SolutionScript` with `phase="merged"`
- [ ] Tests verify merge invocation and output construction

---

## [P1] 4.4 — Implement run_phase1 (Algorithm 1 orchestration)
**Status:** pending
**Priority:** P1

### Description
Implement `run_phase1()` per REQ-P1-031 through REQ-P1-040 in `mle_star/phase1.py`. Orchestrates Algorithm 1: retrieve M models → generate M candidates → evaluate and sort by score → merge with break-on-first-failure loop.

### Acceptance Criteria
- [ ] `run_phase1(client, task, config)` returns `Phase1Result`
- [ ] Retrieves M models, generates M candidates, evaluates each
- [ ] Sorts candidates by score (best first per metric direction)
- [ ] Merges sorted candidates into base with break-on-first-failure
- [ ] Uses `evaluate_with_retry()` for each candidate and merged result
- [ ] `Phase1Result` contains retrieved_models, candidate_solutions, initial_solution, initial_score
- [ ] Tests verify orchestration flow with mocked components

---

## [P1] 4.5 — Implement Phase 1 post-merge safety checks
**Status:** pending
**Priority:** P1

### Description
Implement post-merge safety checks per REQ-P1-041 through REQ-P1-045 in `mle_star/phase1.py`. After merging, invoke `check_data_usage()` and `check_and_fix_leakage()` on the merged solution.

### Acceptance Criteria
- [ ] `run_phase1()` calls `check_data_usage()` after merge
- [ ] `run_phase1()` calls `check_and_fix_leakage()` after merge
- [ ] If leakage fixed, re-evaluates the corrected solution
- [ ] Tests verify safety check invocation order

---

<!-- ============================================================ -->
<!-- SECTION 5: PHASE 2 OUTER LOOP (Spec 05)                       -->
<!-- ============================================================ -->

## [P1] 5.1 — Implement A_abl invocation and ablation script execution
**Status:** pending
**Priority:** P1

### Description
Implement ablation agent invocation per REQ-P2O-001 through REQ-P2O-012 in `mle_star/phase2_outer.py`. A_abl generates ablation study code that is executed via the execution harness. Results are captured for summarization.

### Acceptance Criteria
- [ ] Invokes `ablation` agent with solution code, task description, and previous ablation summaries
- [ ] Extracts ablation script from agent response
- [ ] Executes ablation script via `evaluate_solution()` (or directly via `execute_script()`)
- [ ] Handles execution errors via A_debugger
- [ ] Returns ablation stdout/stderr for summarization
- [ ] Tests verify agent invocation and execution flow

---

## [P1] 5.2 — Implement A_summarize and A_extractor invocations
**Status:** pending
**Priority:** P1

### Description
Implement `A_summarize` per REQ-P2O-013 through REQ-P2O-020 and `A_extractor` per REQ-P2O-021 through REQ-P2O-030 in `mle_star/phase2_outer.py`. A_summarize condenses ablation output; A_extractor identifies target code block and proposes initial plan.

### Acceptance Criteria
- [ ] A_summarize receives raw ablation output, returns concise text summary T_abl^t
- [ ] A_extractor receives solution, ablation summaries, and previously refined blocks C
- [ ] A_extractor returns `ExtractorOutput` with `plans: list[RefinePlan]`
- [ ] Each `RefinePlan` has `code_block` (exact substring of solution) and `plan` text
- [ ] `validate_code_block()` verifies extracted code_block exists in solution content
- [ ] Tests verify structured output parsing and code block validation

---

## [P1] 5.3 — Implement run_phase2_outer_loop
**Status:** pending
**Priority:** P1

### Description
Implement `run_phase2_outer_loop()` per REQ-P2O-031 through REQ-P2O-044 in `mle_star/phase2_outer.py`. Manages T outer iterations: ablation → summarize → extract → inner loop → state update. Accumulates T_abl and C across steps.

### Acceptance Criteria
- [ ] `run_phase2_outer_loop(client, task, config, initial_solution, session_id)` returns `Phase2Result`
- [ ] Iterates T times (from `config.outer_loop_steps`)
- [ ] Each iteration: invoke A_abl, A_summarize, A_extractor, then hand off to inner loop
- [ ] Accumulates ablation summaries T_abl across steps
- [ ] Accumulates refined code blocks C across steps (passed to A_extractor to avoid re-targeting)
- [ ] After inner loop returns, updates best solution if improved
- [ ] Constructs `Phase2Result` with ablation_summaries, refined_blocks, best_solution, best_score, step_history
- [ ] Tests verify state accumulation and iteration control

---

<!-- ============================================================ -->
<!-- SECTION 6: PHASE 2 INNER LOOP (Spec 06)                       -->
<!-- ============================================================ -->

## [P1] 6.1 — Implement invoke_coder (A_coder invocation)
**Status:** pending
**Priority:** P1

### Description
Implement `invoke_coder()` per REQ-P2I-001 through REQ-P2I-015 in `mle_star/phase2_inner.py`. A_coder receives the current solution, target code block, and refinement plan, then produces a modified code block.

### Acceptance Criteria
- [ ] `invoke_coder(client, solution, code_block, plan, task)` invokes `coder` agent
- [ ] Uses prompt template with solution code, target code block, plan, task description
- [ ] Extracts modified code block from agent response via `extract_code_block()`
- [ ] Returns the new code block string
- [ ] Tests verify agent invocation and code extraction

---

## [P1] 6.2 — Implement invoke_planner (A_planner invocation)
**Status:** pending
**Priority:** P1

### Description
Implement `invoke_planner()` per REQ-P2I-016 through REQ-P2I-030 in `mle_star/phase2_inner.py`. A_planner proposes the next refinement strategy given history of previous attempts `{(p_j, h(s_t^j))}`.

### Acceptance Criteria
- [ ] `invoke_planner(client, solution, code_block, history, task)` invokes `planner` agent
- [ ] History formatted as list of `(plan_text, score)` tuples
- [ ] Returns new plan text string (3-5 sentences)
- [ ] Tests verify history formatting and agent invocation

---

## [P1] 6.3 — Implement run_phase2_inner_loop
**Status:** pending
**Priority:** P1

### Description
Implement `run_phase2_inner_loop()` per REQ-P2I-031 through REQ-P2I-050 in `mle_star/phase2_inner.py`. Orchestrates K inner iterations: k=0 uses initial plan from extractor, k=1..K-1 uses A_planner. Each iteration: A_coder → replace_block → A_leakage → evaluate → A_debugger if error → track best.

### Acceptance Criteria
- [ ] `run_phase2_inner_loop(client, task, config, solution, code_block, initial_plan, session_id)` returns `InnerLoopResult`
- [ ] Iterates K times (from `config.inner_loop_steps`)
- [ ] k=0: uses `initial_plan`; k>0: invokes A_planner with history
- [ ] Each iteration: invoke A_coder → `solution.replace_block(old, new)` → check_and_fix_leakage → evaluate_with_retry
- [ ] Tracks best solution/score across iterations using `is_improvement_or_equal()`
- [ ] Accumulates `RefinementAttempt` history
- [ ] Code block replacement is against original solution `s_t` (not accumulated modifications)
- [ ] Tests verify iteration control, best tracking, and history accumulation

---

<!-- ============================================================ -->
<!-- SECTION 7: PHASE 3 — ENSEMBLE (Spec 07)                       -->
<!-- ============================================================ -->

## [P1] 7.1 — Implement invoke_ens_planner and invoke_ensembler
**Status:** pending
**Priority:** P1

### Description
Implement `invoke_ens_planner()` per REQ-P3-001 through REQ-P3-015 and `invoke_ensembler()` per REQ-P3-016 through REQ-P3-030 in `mle_star/phase3.py`. A_ens_planner proposes an ensemble strategy; A_ensembler implements it.

### Acceptance Criteria
- [ ] `invoke_ens_planner(client, solutions, history, task)` invokes `ens_planner` agent
- [ ] Receives L solution scripts and history of previous attempts
- [ ] Returns ensemble plan text string
- [ ] `invoke_ensembler(client, solutions, plan, task)` invokes `ensembler` agent
- [ ] Receives L solution scripts and the ensemble plan
- [ ] Extracts ensemble script from response via `extract_code_block()`
- [ ] Returns `SolutionScript` with `phase="ensemble"`
- [ ] Tests verify agent invocation and output construction

---

## [P1] 7.2 — Implement run_phase3 (Algorithm 3 orchestration)
**Status:** pending
**Priority:** P1

### Description
Implement `run_phase3()` per REQ-P3-031 through REQ-P3-049 in `mle_star/phase3.py`. Orchestrates Algorithm 3: R rounds of (A_ens_planner → A_ensembler → leakage check → evaluate → debugger if error → track best). Includes skip condition when L=1.

### Acceptance Criteria
- [ ] `run_phase3(client, task, config, solutions)` returns `Phase3Result`
- [ ] Skips entirely and returns `None` when `len(solutions) == 1`
- [ ] R rounds of ensemble planning + implementation + evaluation
- [ ] r=0: initial plan; r>0: A_ens_planner with history of `{(e_j, h(s_ens^j))}`
- [ ] Each round: leakage check → evaluate_with_retry → track best
- [ ] Best selection uses argmax/argmin based on metric direction
- [ ] Accumulates `EnsembleAttempt` history
- [ ] `Phase3Result` has input_solutions, ensemble_plans, ensemble_scores, best_ensemble, best_ensemble_score
- [ ] Tests verify skip condition, iteration, and best tracking

---

<!-- ============================================================ -->
<!-- SECTION 8: SUBMISSION AND FINALIZATION (Spec 08)               -->
<!-- ============================================================ -->

## [P1] 8.1 — Implement subsampling removal pipeline
**Status:** pending
**Priority:** P1

### Description
Implement `remove_subsampling()` per REQ-FN-001 through REQ-FN-015 in `mle_star/finalization.py`. Uses two agent calls (subsampling extraction via Figure 26, subsampling removal via Figure 27) to remove the subsampling code from the final solution so it trains on full data.

### Acceptance Criteria
- [ ] Invokes `test` agent with `variant="subsampling_extract"` to identify subsampling code block
- [ ] Invokes `test` agent with `variant="subsampling_remove"` to produce de-subsampled code
- [ ] Replaces subsampling block in solution via `replace_block()`
- [ ] Returns modified `SolutionScript` with subsampling removed
- [ ] No-op if no subsampling detected
- [ ] Tests verify extraction and removal flow

---

## [P1] 8.2 — Implement A_test (generate_test_submission)
**Status:** pending
**Priority:** P1

### Description
Implement `generate_test_submission()` per REQ-FN-016 through REQ-FN-025 in `mle_star/finalization.py`. A_test transforms a validation solution into a test submission script that generates `./final/submission.csv`.

### Acceptance Criteria
- [ ] `generate_test_submission(client, solution, task)` invokes `test` agent
- [ ] Uses prompt template (Figure 25) with solution code, task description, output path
- [ ] Extracts test submission script from response
- [ ] Returns `SolutionScript` with `phase="final"`
- [ ] Tests verify agent invocation and output construction

---

## [P1] 8.3 — Implement test execution, verification, and error handling
**Status:** pending
**Priority:** P1

### Description
Implement test script execution per REQ-FN-026 through REQ-FN-035. Execute the test submission script, verify `./final/submission.csv` exists and is valid, handle errors with A_debugger retry.

### Acceptance Criteria
- [ ] Executes test submission script via `execute_script()` (no subsampling, full data)
- [ ] Verifies `./final/submission.csv` via `verify_submission()`
- [ ] On error, retries with A_debugger up to `max_debug_attempts`
- [ ] Returns final `EvaluationResult` from test execution
- [ ] Tests verify execution and verification flow

---

## [P2] 8.4 — Implement data contamination check
**Status:** pending
**Priority:** P2

### Description
Implement `check_contamination()` per REQ-FN-036 through REQ-FN-042 in `mle_star/finalization.py`. Optional check using `data` agent with `DataContaminationResult` structured output (Figure 28).

### Acceptance Criteria
- [ ] `check_contamination(client, solution, task)` invokes `data` agent
- [ ] Parses `DataContaminationResult` with `verdict: "Novel" | "Same"`
- [ ] Returns contamination result; logs warning if `"Same"`
- [ ] Invocation is optional (configurable)
- [ ] Tests verify structured output parsing

---

## [P1] 8.5 — Implement run_finalization orchestration
**Status:** pending
**Priority:** P1

### Description
Implement `run_finalization()` per REQ-FN-043 through REQ-FN-048 in `mle_star/finalization.py`. Orchestrates: subsampling removal → A_test → execute → verify → optional contamination check → construct FinalResult.

### Acceptance Criteria
- [ ] `run_finalization(client, task, config, best_solution)` returns finalized `SolutionScript` and `submission_path`
- [ ] Calls `remove_subsampling()` first
- [ ] Calls `generate_test_submission()` to produce test script
- [ ] Executes and verifies submission
- [ ] Optionally calls `check_contamination()`
- [ ] Returns the final solution and path to `./final/submission.csv`
- [ ] Tests verify orchestration order

---

<!-- ============================================================ -->
<!-- SECTION 9: TOP-LEVEL ORCHESTRATOR (Spec 09)                   -->
<!-- ============================================================ -->

## [P1] 9.1 — Implement run_pipeline entry point and input validation
**Status:** pending
**Priority:** P1

### Description
Implement `run_pipeline()` async function per REQ-OR-001 through REQ-OR-004 in `mle_star/orchestrator.py`. Validates inputs, sets up working directory, detects GPU.

### Acceptance Criteria
- [ ] `async def run_pipeline(task: TaskDescription, config: PipelineConfig | None = None) -> FinalResult`
- [ ] Defaults `config` to `PipelineConfig()` when `None`
- [ ] Validates `task.data_dir` exists and contains files; raises `ValueError` otherwise
- [ ] Calls `setup_working_directory(task)` and `detect_gpu_info()`
- [ ] Tests verify input validation and error messages

---

## [P1] 9.2 — Implement SDK client lifecycle management
**Status:** pending
**Priority:** P1

### Description
Implement SDK client initialization per REQ-OR-005 through REQ-OR-011 in `mle_star/orchestrator.py`. Create `ClaudeSDKClient` with agent registrations, system prompt, permission mode, and cleanup in `try/finally`. Include SDK reconnection on transient failure per REQ-OR-052.

### Acceptance Criteria
- [ ] Client created with model, permission_mode, agents dict, hooks
- [ ] System prompt includes Kaggle grandmaster persona per REQ-OR-007
- [ ] System prompt includes task description, metric, direction, GPU info
- [ ] All 14 agents registered via `build_default_agent_configs()` → `to_agent_definition()`
- [ ] `client.disconnect()` called in `finally` block per REQ-OR-011
- [ ] Permission mode configurable per REQ-OR-009
- [ ] SDK reconnection with exponential backoff (3 retries) on transient failure per REQ-OR-052
- [ ] Agent name uniqueness enforced per REQ-OR-057
- [ ] Tests verify client setup, cleanup, and reconnection behavior (mocked SDK)

---

## [P1] 9.3 — Implement phase dispatch and ordering
**Status:** pending
**Priority:** P1

### Description
Implement sequential phase dispatch per REQ-OR-012 through REQ-OR-017 in `mle_star/orchestrator.py`. Phase 1 → Phase 2 (L parallel) → Phase 3 → Finalization. Phase 3 skipped when L=1 per REQ-OR-015.

### Acceptance Criteria
- [ ] Calls `run_phase1()` first, records duration
- [ ] Calls `run_phase2_outer_loop()` L times (via parallelism), records duration
- [ ] Calls `run_phase3()` with L solutions (skipped when L=1)
- [ ] Calls `run_finalization()` with best solution
- [ ] Phase ordering is strictly sequential per REQ-OR-017
- [ ] When L=1, `FinalResult.phase3` is `None`
- [ ] Tests verify phase ordering and L=1 skip

---

## [P1] 9.4 — Implement L-parallel Phase 2 paths
**Status:** pending
**Priority:** P1

### Description
Implement asyncio-based parallel Phase 2 paths per REQ-OR-018 through REQ-OR-023 in `mle_star/orchestrator.py`. Deep copy initial solution, unique session IDs, per-path working subdirectories per REQ-OR-020, error isolation via `return_exceptions=True`, and cancellation on timeout per REQ-OR-023.

### Acceptance Criteria
- [ ] Deep copies Phase 1 solution for each of L paths
- [ ] Each path gets unique `session_id` ("path-0", "path-1", etc.) per REQ-OR-021
- [ ] Each path gets its own working subdirectory (e.g., `./work/path-0/`) per REQ-OR-020
- [ ] Uses `asyncio.gather(*tasks, return_exceptions=True)` per REQ-OR-022
- [ ] Failed paths logged, successful results collected
- [ ] If all paths fail, falls back to Phase 1 solution per REQ-OR-022
- [ ] Cancellation of outstanding paths when timeout reached per REQ-OR-023
- [ ] Per-path time budget = total Phase 2 budget / L per REQ-OR-026
- [ ] Tests verify concurrent execution, error isolation, directory isolation, and fallback

---

## [P1] 9.5 — Implement time and cost budget enforcement
**Status:** pending
**Priority:** P1

### Description
Implement time limit enforcement per REQ-OR-024 through REQ-OR-026, cost tracking per REQ-OR-027 through REQ-OR-029, and graceful shutdown per REQ-OR-030 in `mle_star/orchestrator.py`.

### Acceptance Criteria
- [ ] Computes deadline at pipeline start from `config.time_limit_seconds`
- [ ] Checks deadline before each phase
- [ ] Proportional time allocation across phases (10%/65%/15%/10%)
- [ ] Accumulates `total_cost_usd` from each agent call
- [ ] Warns at 80% budget consumption
- [ ] Graceful shutdown: cancels tasks, uses best-known solution, skips to finalization
- [ ] `PipelineTimeoutError` raised if Phase 1 hasn't completed
- [ ] Tests verify timeout and budget enforcement

---

## [P1] 9.6 — Implement hooks (progress, cost, safety, timeout, error)
**Status:** pending
**Priority:** P1

### Description
Implement all hooks per REQ-OR-031 through REQ-OR-035 in `mle_star/orchestrator.py`. PostToolUse progress logging, cost accumulation, PreToolUse safety (block dangerous bash), timeout monitoring, error logging.

### Acceptance Criteria
- [ ] Progress hook logs: timestamp, agent_type, tool_name, session_id, elapsed_time, success/failure
- [ ] Cost hook accumulates per-agent costs, triggers budget check
- [ ] Safety hook blocks dangerous commands (rm -rf /, mkfs, dd, fork bomb, out-of-directory writes)
- [ ] Timeout hook sets "finalize now" flag when <10% time remaining
- [ ] Error hook logs tool failures with traceback and maintains consecutive failure count
- [ ] Tests verify hook registration and trigger behavior

---

## [P1] 9.7 — Implement FinalResult assembly and error handling
**Status:** pending
**Priority:** P1

### Description
Implement `FinalResult` construction per REQ-OR-036 through REQ-OR-039, and error handling per REQ-OR-040 through REQ-OR-043. Include Phase 2 failure fallback, Phase 3 failure fallback, best-effort result return, per-phase cost/duration breakdowns per REQ-OR-037/REQ-OR-038, and solution lineage tracing per REQ-OR-039.

### Acceptance Criteria
- [ ] `FinalResult` assembled with all phase outputs, duration, cost
- [ ] Per-phase cost breakdown included per REQ-OR-037
- [ ] Per-phase duration breakdown included per REQ-OR-038
- [ ] Solution lineage tracing: provenance chain logged across all phases per REQ-OR-039
- [ ] Phase 2 failure: substitutes Phase 1 solution per REQ-OR-040
- [ ] Phase 3 failure: selects best Phase 2 solution per REQ-OR-041
- [ ] Complete failure: raises `PipelineError` with diagnostics per REQ-OR-042
- [ ] Best-effort: always returns `FinalResult` with best-known solution per REQ-OR-043
- [ ] Tests verify all fallback scenarios and lineage output

---

## [P1] 9.8 — Implement PipelineState, configuration, and environment variables
**Status:** pending
**Priority:** P1

### Description
Implement `PipelineState` per REQ-OR-050, env var support per REQ-OR-046, logging per REQ-OR-047, `run_pipeline_sync()` wrapper per REQ-OR-053, and sensible defaults per REQ-OR-044.

### Acceptance Criteria
- [ ] `PipelineState` tracks current_phase, elapsed_seconds, cost, path_statuses, best_score, agent_call_count
- [ ] Reads `ANTHROPIC_API_KEY` (required), `MLE_STAR_MODEL`, `MLE_STAR_LOG_LEVEL`, `MLE_STAR_MAX_BUDGET`, `MLE_STAR_TIME_LIMIT`
- [ ] Env vars override defaults but not explicit `PipelineConfig` args
- [ ] Missing `ANTHROPIC_API_KEY` raises `EnvironmentError`
- [ ] Logger named `"mle_star"` with configurable level and structured output
- [ ] `run_pipeline_sync()` wraps async call with `asyncio.run()`
- [ ] Tests verify env var precedence and logging setup

---

<!-- ============================================================ -->
<!-- SECTION 10: PROMPT TEMPLATES (Spec 01 REQ-DM-033)             -->
<!-- ============================================================ -->

## [P1] 10.1 — Populate prompt templates for Phase 1 agents (A_retriever, A_init, A_merger)
**Status:** pending
**Priority:** P1

### Description
Create prompt templates for A_retriever (Figure 9), A_init (Figure 10), and A_merger (Figure 11) in `mle_star/prompts.py` and register them in the `PromptRegistry`. Templates must match paper figures with appropriate `{variable}` placeholders.

### Acceptance Criteria
- [ ] Retriever template references Figure 9 with variables: M, task_description
- [ ] Init template references Figure 10 with variables: task_description, model_name, example_code, subsampling_instruction
- [ ] Merger template references Figure 11 with variables: base_solution, reference_solution, task_description
- [ ] All templates registered in `PromptRegistry`
- [ ] `render()` produces well-formed prompts
- [ ] Tests verify template rendering

---

## [P1] 10.2 — Populate prompt templates for Phase 2 agents (A_abl, A_summarize, A_extractor, A_planner, A_coder)
**Status:** pending
**Priority:** P1

### Description
Create prompt templates for A_abl (Figure 12), A_summarize (Figure 13), A_extractor (Figure 14), A_planner (Figure 15), and A_coder (Figure 18/19) in `mle_star/prompts.py`.

### Acceptance Criteria
- [ ] All 5 templates created with correct variables and figure references
- [ ] Extractor template produces JSON schema format instruction
- [ ] Planner template includes history format `{(plan, score)}` pairs
- [ ] Coder template includes target code block and plan
- [ ] All registered in `PromptRegistry`
- [ ] Tests verify rendering

---

## [P1] 10.3 — Populate prompt templates for Phase 3 and safety agents
**Status:** pending
**Priority:** P1

### Description
Create prompt templates for A_ens_planner (Figure 22), A_ensembler (Figure 23), A_debugger (Figure 16), A_leakage detection (Figure 20), A_leakage correction (Figure 21), A_data (Figure 17), A_test (Figure 25), and subsampling templates (Figures 26, 27) in `mle_star/prompts.py`.

### Acceptance Criteria
- [ ] All 9 templates created with correct variables and figure references
- [ ] Leakage has two variants: detection and correction
- [ ] Subsampling has two variants: extract and remove
- [ ] A_ens_planner includes history format `{(plan, score)}` pairs
- [ ] All registered in `PromptRegistry`
- [ ] `len(registry)` covers all 14 agent types per REQ-DM-033
- [ ] Tests verify `PromptRegistry` completeness

---

<!-- ============================================================ -->
<!-- SECTION 11: CLI INTEGRATION                                   -->
<!-- ============================================================ -->

## [P2] 11.1 — Wire run_pipeline to CLI entry point
**Status:** pending
**Priority:** P2

### Description
Update `mle_star/cli.py` to accept a task configuration file (YAML or JSON), parse it into `TaskDescription` and optional `PipelineConfig`, and call `run_pipeline_sync()`. Print result summary on completion.

### Acceptance Criteria
- [ ] CLI accepts `--task` argument pointing to a task config file
- [ ] Parses task config into `TaskDescription`
- [ ] Optional `--config` for `PipelineConfig` overrides
- [ ] Calls `run_pipeline_sync()` and prints summary (score, duration, submission path)
- [ ] Exit code 0 on success, 1 on failure
- [ ] Tests verify CLI argument parsing

---

<!-- ============================================================ -->
<!-- SECTION 12: TESTING AND QUALITY                               -->
<!-- ============================================================ -->

## [P1] 12.1 — Create test infrastructure and fixtures
**Status:** pending
**Priority:** P1

### Description
Set up `tests/` directory with conftest.py, fixtures for common test data (sample TaskDescription, PipelineConfig, SolutionScript, mock SDK client), and test helpers for mocking agent responses.

### Acceptance Criteria
- [ ] `tests/conftest.py` with shared fixtures
- [ ] Mock SDK client fixture that simulates agent calls
- [ ] Sample task description fixture
- [ ] Sample solution script fixture
- [ ] `uv run pytest` executes without errors (even if no test files yet)

---

## [P2] 12.2 — Create integration test for full pipeline (mocked SDK)
**Status:** pending
**Priority:** P2

### Description
Create an integration test that runs the full pipeline with a mocked SDK client and verifies end-to-end flow: Phase 1 → Phase 2 → Phase 3 → Finalization → FinalResult.

### Acceptance Criteria
- [ ] Test exercises `run_pipeline()` with mocked SDK
- [ ] Verifies all phases execute in order
- [ ] Verifies `FinalResult` contains all expected fields
- [ ] Verifies Phase 3 skip when L=1
- [ ] Coverage report shows all module entry points exercised

---

## [P3] 12.3 — Achieve 90% test coverage across all modules
**Status:** pending
**Priority:** P3

### Description
Write unit tests to reach the 90% coverage minimum configured in pyproject.toml. Focus on edge cases, error paths, and boundary conditions.

### Acceptance Criteria
- [ ] `uv run pytest` shows >= 90% coverage
- [ ] All error paths tested (ValueError, KeyError, timeout, fallback)
- [ ] Hypothesis property-based tests for model validation
- [ ] No surviving mutants on critical logic (score comparison, phase ordering)

---

## [P3] 12.4 — Pass all quality gates (mypy, ruff, xenon, bandit)
**Status:** pending
**Priority:** P3

### Description
Ensure all code passes strict mypy type checking, ruff formatting/linting, xenon complexity checks (max B), and bandit security scanning.

### Acceptance Criteria
- [ ] `uv run mypy --config-file=pyproject.toml src tests` passes with 0 errors
- [ ] `uv run ruff check .` passes with 0 errors
- [ ] `uv run ruff format --check .` passes
- [ ] `uv run xenon --max-average=B --max-modules=B --max-absolute=B src` passes
- [ ] `uv run bandit -c pyproject.toml -r .` passes with 0 issues

---

## [P3] 12.5 — Run mutation testing on critical modules
**Status:** pending
**Priority:** P3

### Description
Run `mutmut` on critical modules (models.py, execution.py, orchestrator.py) and fix tests to kill surviving mutants in score comparison, phase ordering, and timeout logic.

### Acceptance Criteria
- [ ] `uv run mutmut run --paths-to-mutate=src/mle_star/models.py` kills > 90% mutants
- [ ] Score comparison functions (`is_improvement`, `is_improvement_or_equal`) have 100% mutant kill rate
- [ ] `uv run mutmut browse` shows no surviving mutants in critical logic paths

---

## [P2] 9.9 — Implement MCP server registration for score parsing and file listing
**Status:** pending
**Priority:** P2

### Description
Implement MCP server registration per REQ-OR-010 in `mle_star/orchestrator.py`. Register two MCP servers with the SDK client: one for score parsing (exposing `parse_score()` as a tool) and one for file listing (exposing data directory contents). This is a "Should" priority requirement that enhances agent capabilities.

### Acceptance Criteria
- [ ] Score-parsing MCP server registered with SDK client
- [ ] File-listing MCP server registered with SDK client
- [ ] Agents can invoke these tools during execution
- [ ] Graceful degradation if MCP registration fails (log warning, continue without)
- [ ] Tests verify MCP server registration (mocked SDK)

---

<!-- ============================================================ -->
<!-- END OF TASKS                                                  -->
<!-- ============================================================ -->
