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
  orchestrator.py      # Pipeline entry point: run_pipeline, SDK client setup, phase dispatch, parallelism, time/cost control, CostTracker, PipelineError/PipelineTimeoutError, result assembly, error recovery, hooks (Tasks 42-47)
  models.py            # Pydantic data models (enums, configs, schemas)
  scoring.py           # Score parsing, comparison functions, ScoreFunction protocol (Task 07)
  execution.py         # Execution harness: env setup, working dir, GPU, async script exec, output parsing, evaluation pipeline, subsampling utilities, submission verification, batch evaluation, solution ranking, error masking detection, ExecutorStrategy, SDK Bash executor, output truncation, structured logging (Tasks 11-17, 18)
  safety.py            # Safety modules: debugger agent, leakage agent, data agent, code block extraction, structured logging (Tasks 19, 20, 21, 22)
  phase1.py            # Phase 1 agents + orchestration: retrieve_models, generate_candidate, merge_solutions, run_phase1 (Tasks 27, 28)
  phase2_inner.py      # Phase 2 inner loop: coder/planner agents + run_phase2_inner_loop orchestration with safety integration, structured logging (Tasks 23, 24, 25, 26)
  phase2_outer.py      # Phase 2 outer loop: ablation agent, summarize agent, extractor agent, code block validation, run_phase2_outer_loop orchestration (Tasks 31, 32, 33)
  phase3.py            # Phase 3 agents + orchestration: invoke_ens_planner, invoke_ensembler, run_phase3 (Tasks 35, 36)
  finalization.py      # Finalization: remove_subsampling, generate_test_submission, check_contamination, run_finalization (Tasks 38, 39, 40)
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
  test_execution_constraints.py  # Tests for error masking, ExecutorStrategy, SDK executor, logging, truncation, performance (Task 18)
  test_safety_debugger.py        # Tests for debugger safety agent (Task 19)
  test_safety_data.py            # Tests for data usage verification agent (Task 21)
  test_safety_leakage.py         # Tests for leakage detection/correction agent (Task 20)
  test_safety_constraints.py     # Tests for safety module constraints, logging, graceful degradation (Task 22)
  test_phase1_agents.py          # Tests for Phase 1 agents: retriever, init, merger (Task 27)
  test_phase1_orchestration.py   # Tests for run_phase1 orchestration (Task 28)
  test_phase1_safety.py          # Tests for Phase 1 post-merge safety checks (Task 29)
  test_phase2_inner_agents.py    # Tests for coder and planner agents (Task 23)
  test_phase2_inner_loop.py      # Tests for run_phase2_inner_loop orchestration (Task 24)
  test_phase2_inner_safety.py    # Tests for inner loop safety integration (Task 25)
  test_phase2_inner_constraints.py # Tests for inner loop constraints, logging, immutability, monotonic score (Task 26)
  test_phase2_outer_ablation.py  # Tests for ablation agent invocation and execution (Task 31)
  test_phase2_outer_agents.py    # Tests for summarize and extractor agents (Task 32)
  test_phase2_outer_loop.py      # Tests for run_phase2_outer_loop orchestration (Task 33)
  test_phase2_outer_constraints.py # Tests for outer loop constraints, logging, performance, immutability, monotonic score (Task 34)
  test_phase3_agents.py          # Tests for Phase 3 ensemble planner and ensembler agents (Task 35)
  test_phase3_orchestration.py   # Tests for run_phase3 orchestration (Task 36)
  test_finalization_subsampling.py # Tests for subsampling removal (Task 38)
  test_finalization_test_submission.py # Tests for test submission agent (Task 39)
  test_orchestrator_entry.py     # Tests for pipeline entry point and SDK client setup (Task 42)
  test_orchestrator_dispatch.py  # Tests for phase dispatch and sequencing (Task 43)
  test_orchestrator_parallelism.py # Tests for asyncio parallelism, deep copy, working dirs, cancellation (Task 44)
  test_orchestrator_time_cost.py # Tests for time budgets, cost tracking, deadline enforcement, graceful shutdown (Task 45)
  test_orchestrator_results.py   # Tests for result assembly, error recovery, finalization fallback, phase summaries, lineage (Task 47)
  test_orchestrator_hooks.py     # Tests for SDK hooks: progress, cost, safety, timeout, error, agent tracking (Task 46)
  test_orchestrator_config.py  # Tests for configuration, env vars, logging, PipelineState (Task 48)
  test_finalization_contamination.py # Tests for contamination check and run_finalization (Task 40)
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
- Safety structured logging (REQ-SF-041): `_invoke_debugger_agent` logs INFO at start ("Debug invocation start for phase=...") and result ("Debug invocation result length=..."); `_check_and_fix_leakage_impl` logs INFO at detection start/result ("Leakage detection start", "Leakage detection result: N answers"); `_check_data_usage_impl` logs INFO at start ("Data usage check start") and result ("Data usage check result: confirmed/modified"). Parse failures and `replace_block` skips log WARNING. Graceful degradation (outer try/except) logs via `logger.exception`
- Phase 1 agent pattern: `retrieve_models` parses structured JSON via `RetrieverOutput.model_validate_json()`; `generate_candidate` and `merge_solutions` extract code via `extract_code_block()` and return `SolutionScript | None` (None on empty extraction). Empty-check uses `.strip()` for whitespace-only responses
- Phase 1 orchestration (`run_phase1`): decomposed into `_generate_and_evaluate_candidates` + `_run_merge_loop` + `_apply_post_merge_safety` to stay under xenon complexity threshold B. `_CandidateResults` class accumulates candidate loop state. Merge loop uses `is_improvement_or_equal` (>= semantics) and breaks on first failure/non-improvement
- Post-merge safety pattern: `_apply_safety_check()` generic helper uses identity check (`is not`) to detect modification, re-evaluates if modified, and falls back to pre-check version on failure. Applied twice in sequence: `check_data_usage` (exactly once, REQ-P1-030) then `check_and_fix_leakage` (REQ-P1-031). Phase1Result.initial_score always reflects the final post-safety score
- Phase 1 structured logging (REQ-P1-038): `run_phase1` logs INFO at start ("Phase 1 start: competition_id=..., M=..."), retrieval complete ("Retrieval complete: count=..., models=..."), sorted candidates ("Candidates sorted: ranked_order=..."), phase1 complete ("Phase 1 complete: final_score=..., duration=..., merges=..."); ERROR when all candidates fail. `_generate_and_evaluate_candidates` logs INFO for each candidate gen start/complete and eval result. `_run_merge_loop` logs INFO for merge start/result and returns 3-tuple `(solution, score, merge_count)`. `_apply_safety_check` logs INFO at start/result with modified/unchanged status. Duration tracked via `time.monotonic()`
- Ablation script execution uses a custom retry loop (NOT `evaluate_with_retry`) because ablation scripts are informational only — no score parsing, custom timeout, and different error recovery semantics. Timeout formula: `min(time_limit // (outer_steps * 2), 600)` using integer division
- `_format_previous_ablations()` returns empty string for first iteration (omits section from prompt), and numbered markdown for subsequent iterations with header "# Previous Ablation Study Results"
- A_summarize uses full text response (no code block extraction); fallback on empty/whitespace: `"[Auto-summary from raw output] " + raw_output[-2000:]` (REQ-P2O-036)
- A_extractor uses structured output (`ExtractorOutput.model_validate_json`) with one retry on parse failure (REQ-P2O-034); returns `ExtractorOutput | None`
- `validate_code_block(code_block, solution)` is a simple `code_block in solution.content` substring check (REQ-P2O-017)
- `_format_previous_blocks()` mirrors `_format_previous_ablations()` pattern: empty list → empty string, non-empty → numbered markdown with header "# Previously Improved Code Blocks"
- Outer loop orchestration (`run_phase2_outer_loop`): decomposed into `_run_outer_step` helper + `_make_skipped_step` to stay under xenon complexity B. Step helper returns a dict with internal `_new_h_best` and `_new_best_solution` keys (popped by caller). Uses `is_improvement_or_equal` (>= semantics) for best-score update — NOT `InnerLoopResult.improved` (which uses strict >). Skipped iterations (extractor None or validation failure) produce `was_skipped=True` records. Accumulates T_abl and C lists; each `CodeBlock` has `outer_step=t` set. `initial_score: float` is explicit parameter because `SolutionScript.score` is `float | None`
- Phase 3 agent pattern: `invoke_ens_planner` returns raw stripped text (like A_planner), `invoke_ensembler` uses `extract_code_block()` (like A_coder/A_init/A_merger). Both validate `len(solutions) >= 2`. `_format_solutions()` numbers solutions as "# {n}th Python Solution" with fenced code blocks. `_format_ensemble_history()` mirrors `_format_plan_history()` from phase2_inner.py: `## Plan:` / `## Score:` labels, None → "N/A (evaluation failed)", empty list → empty string
- Phase 3 orchestration (`run_phase3`): decomposed into `_run_ensemble_round` helper + `_select_best_input` fallback to stay under xenon complexity B. Single-solution skip returns immediately (REQ-P3-018). `make_debug_callback` called once at the start. Each round: plan → implement → `check_and_fix_leakage` → `evaluate_with_retry`. Passes `list(accumulated_plans)` (copies) to planner for snapshot semantics. Best selection uses `is_improvement_or_equal` (>= semantics, LAST tie wins per REQ-P3-025). Fallback `_select_best_input` uses direct max/min (NOT `is_improvement_or_equal`) for correct behavior when mocked. Failed planner: `"[ens_planner failed]"` plan, None score, empty solution. Failed ensembler: plan preserved, None score, empty solution, no leakage/eval call
- Finalization subsampling removal pattern: `remove_subsampling(client, solution, task)` uses two A_test agent calls with different variants — `subsampling_extract` then `subsampling_remove`. Both use `AgentType.TEST` with `PromptRegistry` variant selection. Extraction parsed via `extract_code_block()`; result verified as non-empty substring of solution. Graceful degradation: outer try/except returns original on any failure. `replace_block` ValueError caught separately with warning log
- Test submission agent pattern: `generate_test_submission(client, task, solution)` uses A_test agent with default variant (no variant specified). Renders template with `task_description` and `final_solution` variables. Response parsed via `extract_code_block()`. Returns new `SolutionScript(phase=FINAL, is_executable=True)`. Empty extraction logs warning and returns empty-content SolutionScript (triggering debug retry in `run_finalization`). Exceptions propagate to caller (unlike `remove_subsampling` which catches them) — `run_finalization` handles fallback (REQ-FN-025)
- Contamination check pattern: `check_contamination(client, solution, reference_discussions)` uses A_test with `variant="contamination_check"` and `output_format={"type": "json_schema", "schema": DataContaminationResult.model_json_schema()}`. Skips when `reference_discussions` is None/empty (returns None). For each ref: renders template with `reference_discussion` + `final_solution`, sends with `output_format`, parses via `DataContaminationResult.model_validate_json()`. Aggregation: ANY "Same" → overall "Same"; ALL "Novel" → overall "Novel". Graceful degradation: outer try/except returns None on any failure (REQ-FN-041)
- Finalization orchestration (`run_finalization`): decomposed into main function + `_apply_fallback` helper to stay under xenon complexity B. Pipeline: `remove_subsampling` → `generate_test_submission` → `check_and_fix_leakage` → `evaluate_with_retry(make_debug_callback)` → `verify_submission` / `get_submission_info` → `_apply_fallback` → `check_contamination` → `FinalResult`. Fallback triggers on `eval_result.is_error or not submission_verified` — returns original `solution` param with `submission_path=""`. Duration tracked with `time.monotonic()`. `total_cost_usd` always `None` (tracked by orchestrator, not finalization)
- Orchestrator entry pattern: `_validate_inputs()` runs before SDK client creation (no client on validation failure). `ClaudeSDKClient(ClaudeAgentOptions(...))` with `connect()`/`disconnect()` in try/finally. `_build_agents_dict()` converts all 14 `AgentConfig` via `to_agent_definition()` keyed by `str(AgentType)`. `_build_system_prompt()` assembles Kaggle grandmaster persona + task context + GPU info. `_register_mcp_servers()` wrapped in try/except with warning log on failure. `run_phase1` signature is `(task, config, client)` — task first, not client first. `PipelineError(message, *, diagnostics=dict)` and `PipelineTimeoutError` subclass it
- Phase dispatch pattern: P1 → P2 (L paths via `asyncio.gather(return_exceptions=True)`) → P3 (skip when L=1) → Finalization. `_dispatch_phase2()` creates L coroutines and gathers them. `_collect_phase2_results()` separates successes from exceptions, substituting Phase 1 solution for failed paths (REQ-OR-040). Phase 3 receives `phase2_solutions` list (one per path). Best solution for finalization: from Phase 3 `best_ensemble` (L>1) or Phase 2 `best_solution` (L=1). Each phase logs start/duration via `time.monotonic()`. Phase boundary markers: `"=== Phase N: ... ==="`
- Phase 2 parallelism pattern: `_dispatch_phase2()` uses `copy.deepcopy()` for each path's initial solution (REQ-OR-020) so mutations are isolated. `_create_path_work_directories()` creates `./work/path-{i}/` relative to `task.data_dir` parent (REQ-OR-020). When `phase2_timeout` is set, uses `asyncio.wait(tasks, timeout=...)` + `task.cancel()` for overtime paths (REQ-OR-023); when None, uses `asyncio.gather(return_exceptions=True)`. Cancelled paths appear as `asyncio.CancelledError` in results and are handled by `_collect_phase2_results()` as failures (Phase 1 fallback). Results are collected in original order (preserving path indices)
- Time/cost control pattern: `run_pipeline()` computes `deadline = pipeline_start + time_limit_seconds` at the start (REQ-OR-024). `_execute_phase1_with_deadline()` wraps `run_phase1` in `asyncio.wait_for()` with remaining time; timeout raises `PipelineTimeoutError`. After Phase 1, `_compute_phase_budgets()` distributes remaining time proportionally (Phase 2: 65%, Phase 3: 15%, Finalization: 10% — normalized to 100% after Phase 1, REQ-OR-025). Per-path budget = `phase2_budget / L` (REQ-OR-026). `_execute_post_phase1()` checks deadline before Phase 2; if expired, skips directly to finalization with Phase 1 solution (REQ-OR-030). `_execute_phase3_or_skip()` enforces Phase 3 budget via `asyncio.wait_for()` with fallback on timeout. `CostTracker` is thread-safe (threading.Lock), accumulates cost, logs 80% warning once (REQ-OR-029), and exposes `exceeded` property. Helpers `_create_sdk_client()` and `_try_register_mcp()` extracted from `run_pipeline()` to stay under xenon complexity B
- Hook system pattern: 6 factory functions (`create_progress_hook`, `create_cost_hook`, `create_safety_hook`, `create_timeout_hook`, `create_error_hook`, `create_agent_tracking_hook`) create async closures over shared state. `build_hooks()` assembles them into `dict[str, list[HookMatcher]]` for `ClaudeAgentOptions.hooks`. Inner hooks use `Any` for input/context types instead of SDK TypedDicts to avoid mypy union narrowing issues (each hook only fires for its registered event type). `_DEFAULT_BLOCKED_PATTERNS` is module-level list of regex strings. Safety hook uses pre-compiled patterns and `_check_blocked_command()` / `_make_deny_result()` helpers to stay under xenon complexity B. `create_agent_tracking_hook` on SubagentStart populates `session_agent_map: dict[str, str]` so progress hook can resolve session_id to agent_type. Timeout threshold: `max(10% * time_limit, 300.0)`
- Result assembly and error recovery pattern: `_make_failed_phase2_result(phase1_result)` creates synthetic Phase2Result with `step_history=[{"step": 0, "failed": True}]` for failed paths (REQ-OR-040). `_collect_phase2_results` now includes synthetic Phase2Results for failed paths — both output lists always have `len(raw_results)` entries. `_execute_phase3_or_skip` catches both `TimeoutError` and general `Exception` (REQ-OR-041). `_finalize_with_recovery` wraps `run_finalization` in try/except: success → `model_copy(update={...})` to set pipeline-level `total_duration_seconds` and `total_cost_usd`; failure → constructs FinalResult with `submission_path=""` (REQ-OR-043). `_log_phase_summary` logs structured JSON with per-phase costs and durations (REQ-OR-037/038). `_log_solution_lineage` traces solution evolution through phases (REQ-OR-039). `_execute_post_phase1` accepts keyword-only `pipeline_start` and `cost_tracker` args

- Configuration and environment pattern: `validate_api_key()` checks `ANTHROPIC_API_KEY` (raises `OSError` if missing/empty/whitespace). `apply_env_overrides(config)` reads `MLE_STAR_MODEL`, `MLE_STAR_LOG_LEVEL`, `MLE_STAR_MAX_BUDGET`, `MLE_STAR_TIME_LIMIT` and applies them to config — but only overrides **default** values (if field value == PipelineConfig default, env var wins; if field was explicitly set differently, explicit wins). Uses `model_copy(update=...)` since PipelineConfig is frozen. Invalid numeric env vars silently ignored; time_limit < 1 silently ignored. `configure_logging(config)` sets up logger "mle_star" with console handler + optional file handler; idempotent (checks for existing handlers before adding). `PipelineState` is the only **non-frozen** Pydantic model besides `SolutionScript` — used for runtime introspection. `conftest.py` has an `autouse` fixture `_set_api_key_env` that sets a dummy `ANTHROPIC_API_KEY` for all tests
- Inner loop constraints pattern (Task 26): `run_phase2_inner_loop` decomposed into `_run_inner_step` (plan determination + delegation) + `_execute_coder_step` (coder → replace → leakage → eval) to stay under xenon complexity B. Structured logging (REQ-P2I-043): inner loop start/complete at INFO, planner/coder start/complete at INFO, coder failure and planner failure at WARNING, replacement success at DEBUG, replacement failure at WARNING, leakage start/complete at INFO (uses `is not` identity check for content_changed), eval start/complete at INFO (score or "failed"), best score update at INFO (old/new). `_execute_coder_step` returns dict with internal `_candidate`, `_eval_result`, `_successful` keys (consumed by caller). Plan text truncated via `%.200s` format. `successful_evals` counter tracks non-None scores for completion log
- Outer loop constraints pattern (Task 34): Structured logging (REQ-P2O-037) added to all agent invocation functions and orchestration: `invoke_ablation` logs INFO at start (solution_length, previous_summaries count) and complete (script_length); `execute_ablation_with_retry` logs INFO at start (script_path, timeout) and complete (exit_code, output_length, duration); `invoke_summarize` logs INFO at start (ablation_code_length, raw_output_length) and complete (summary_length); `invoke_extractor` logs INFO at start (summary_length, solution_length, previous_blocks count) and complete (plans count, code_block_length). `_run_outer_step` logs INFO for validation pass, inner loop handoff (block_length, plan truncated to 200 chars via `%.200s`), inner loop return (best_score, improvement yes/no), and step complete (t, h_best, duration). `run_phase2_outer_loop` logs INFO for outer loop complete (steps, final_h_best, total_duration). Duration tracking uses `time.monotonic()`. The `improved` flag is computed once via `is_improvement_or_equal` and reused for both logging and the best-score update condition
- Phase 3 constraints pattern (Task 37): Structured logging (REQ-P3-039) added to `_run_ensemble_round` and `run_phase3`. `_run_ensemble_round` accepts keyword-only `round_index` parameter for log context. Logs: round start (r, previous_plans count), ens_planner start/complete (r, history_size, plan truncated via `%.200s`), ens_planner empty WARNING (r), ensembler start/complete (r, plan truncated via `%.200s`, script_length), ensembler extraction failure WARNING (r), leakage start/complete (r, solution_content_length, content_changed via `is not` identity), evaluation start/complete (r, score or "failed", is_error, duration), round failed WARNING (r, error_summary, plan_summary). `run_phase3` logs: Phase 3 start (L, R, competition_id), Phase 3 skipped single solution (score, competition_id), best selection (best_round, best_score, successful_rounds), all rounds failed WARNING (R), Phase 3 complete (best_score, best_round, duration, rounds_attempted). Decomposed into `_execute_rounds` helper to stay under xenon complexity B. Duration tracked via `time.monotonic()`
- Finalization constraints pattern (Task 41): Structured logging (REQ-FN-042) added to `_remove_subsampling_impl` and `run_finalization`. `_remove_subsampling_impl` logs INFO at extraction start (solution_content_length), extraction result (found=True/False, block_length), removal result (original_block_length, replacement_block_length), replacement result (success=True/False, content_length_change). `run_finalization` logs INFO at finalization start (solution_phase, content_length, competition_id), submission verification result (file_exists, size_bytes, row_count), and finalization complete (solution_phase, submission_path, duration). `_apply_fallback` logs WARNING with reason, fallback solution phase and score. `AgentType.TEST` config changed from `_EXECUTION_TOOLS` to `_READ_ONLY_TOOLS` per REQ-FN-048 — A_test only needs Read access (script execution handled by evaluation harness)
- Execution constraints pattern (Task 18): `detect_error_masking(content)` uses two compiled regexes — `_BARE_EXCEPT_PATTERN` for bare `except:` and `_BROAD_EXCEPT_PASS_PATTERN` for `except (Exception|BaseException): pass` — returns list of warning strings (advisory only, never raises). `ExecutorStrategy` StrEnum with SUBPROCESS/SDK_BASH values. `execute_script_via_sdk(script_path, working_dir, timeout_ms, *, client)` uses SDK Bash tool interface; timeout capped at `_SDK_BASH_TIMEOUT_CAP_MS` (600,000ms). `evaluate_solution()` accepts keyword-only `strategy` and `client` params — SDK_BASH with timeout > cap falls back to subprocess. `_truncate_output()` truncates strings exceeding `_MAX_OUTPUT_BYTES` (100MB) with warning. Structured logging: write_script at DEBUG, execute_script start/complete at INFO, timeout at WARNING, error at WARNING, retry at INFO

---

## Codebase Patterns

- All Pydantic models use `ConfigDict(frozen=True)` for immutability (except `SolutionScript` and `PipelineState`)
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
