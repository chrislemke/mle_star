# SRS 09 — Orchestrator: Results, Error Handling, Configuration, and NFRs

## 9. Result Assembly Requirements

### 9.1 FinalResult Construction

> **REQ-OR-036**: *FinalResult Assembly* -- Upon pipeline completion, the orchestrator shall construct a `FinalResult` (REQ-DM-025) by aggregating outputs from all completed phases:
>
> | FinalResult Field | Source |
> |-------------------|--------|
> | `task` | Input `TaskDescription` |
> | `config` | Input or defaulted `PipelineConfig` |
> | `phase1` | `Phase1Result` from `run_phase1()` |
> | `phase2_results` | List of L `Phase2Result` from `run_phase2_outer_loop()` calls |
> | `phase3` | `Phase3Result` from `run_phase3()` (or `None` if skipped) |
> | `final_solution` | `SolutionScript` from `run_finalization()` |
> | `submission_path` | Path to `./final/submission.csv` |
> | `total_duration_seconds` | `time.monotonic()` delta from start to end |
> | `total_cost_usd` | Accumulated cost from REQ-OR-027 |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Every field of `FinalResult` shall be populated (except `phase3` when L=1 or Phase 3 skipped).

### 9.2 Cost Summary

> **REQ-OR-037**: *Per-Phase Cost Breakdown* -- The orchestrator shall compute and log a per-phase cost breakdown:
>
> | Metric | Description |
> |--------|-------------|
> | `phase1_cost_usd` | Total cost of all Phase 1 agent calls |
> | `phase2_cost_usd` | Total cost of all Phase 2 agent calls (across all L paths) |
> | `phase2_per_path_cost_usd` | List of per-path costs |
> | `phase3_cost_usd` | Total cost of all Phase 3 agent calls |
> | `finalization_cost_usd` | Total cost of finalization agent calls |
> | `total_cost_usd` | Sum of all above |
>
> - The cost breakdown shall be included in the pipeline log output.
> - Priority: Should | Verify: Inspection | Release: MVP

### 9.3 Duration Summary

> **REQ-OR-038**: *Per-Phase Duration Breakdown* -- The orchestrator shall compute and log a per-phase duration breakdown:
>
> | Metric | Description |
> |--------|-------------|
> | `phase1_duration_seconds` | Wall-clock time for Phase 1 |
> | `phase2_duration_seconds` | Wall-clock time for Phase 2 (from first path start to last path end) |
> | `phase3_duration_seconds` | Wall-clock time for Phase 3 |
> | `finalization_duration_seconds` | Wall-clock time for finalization |
> | `total_duration_seconds` | Total pipeline wall-clock time |
>
> - The duration breakdown shall be included in the pipeline log output and in `FinalResult.total_duration_seconds`.
> - Priority: Should | Verify: Inspection | Release: MVP

### 9.4 Solution Lineage

> **REQ-OR-039**: *Solution Lineage Tracing* -- The orchestrator shall maintain a solution lineage that traces the final submitted solution back through each pipeline phase:
>
> 1. Phase 1: which retrieved models were used, which candidate was selected, merged solution score.
> 2. Phase 2: which path produced the best solution, which outer/inner steps produced improvements.
> 3. Phase 3: which ensemble round produced the best ensemble, which input solutions were combined.
> 4. Finalization: subsampling removal applied, test script modifications.
>
> - The lineage shall be logged at pipeline completion for debugging and reproducibility.
> - Priority: Should | Verify: Inspection | Release: MVP
> - Source: REF-01 Section 3 (pipeline data flow)

---

## 10. Error Handling Requirements

### 10.1 Phase Failure Recovery

> **REQ-OR-040**: *Phase 2 Failure Fallback* -- If Phase 2 fails for one or more paths (but Phase 1 succeeded), the orchestrator shall substitute the Phase 1 initial solution `s_0` for each failed path's contribution to the ensemble.
>
> - The substituted solution shall retain the Phase 1 score as its score.
> - The `Phase2Result` for a failed path shall contain the Phase 1 solution as `best_solution` and a flag `failed=True` in `step_history`.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: If both Phase 2 paths fail, Phase 3 shall receive two copies of the Phase 1 solution.

> **REQ-OR-041**: *Phase 3 Failure Fallback* -- If Phase 3 fails (but Phase 2 produced results), the orchestrator shall select the best Phase 2 solution (by score) and pass it directly to finalization.
>
> - The best solution shall be selected using `is_improvement_or_equal()` (REQ-DM-029) to compare across paths.
> - `FinalResult.phase3` shall be `None` when Phase 3 is skipped due to failure.
> - Priority: Must | Verify: Test | Release: MVP

> **REQ-OR-042**: *Complete Failure Handling* -- If all phases fail (including Phase 1), the orchestrator shall raise a `PipelineError` with:
>
> - The original exception from Phase 1.
> - Diagnostic information: elapsed time, cost consumed, last successful operation.
> - Any partial results collected before the failure.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `PipelineError` shall include a `diagnostics` attribute with structured failure information.

### 10.2 Partial Results

> **REQ-OR-043**: *Best-Effort Result Return* -- The orchestrator shall always attempt to return a `FinalResult` with the best solution found, even when later phases fail:
>
> | Failure Point | Best Available Solution | Phases Completed |
> |---------------|----------------------|------------------|
> | Phase 2 fails | Phase 1 initial solution | Phase 1 only |
> | Phase 3 fails | Best Phase 2 solution | Phase 1 + Phase 2 |
> | Finalization fails | Best pre-finalization solution | Phase 1 + Phase 2 + Phase 3 |
> | Timeout during Phase 2 | Best intermediate solution from any path | Partial |
>
> - The `FinalResult` shall indicate which phases completed successfully via the presence/absence of phase result fields.
> - If finalization itself fails, the orchestrator shall return a `FinalResult` without a `submission_path` (set to `""`) and log the failure.
> - Priority: Must | Verify: Test | Release: MVP

---

## 11. Configuration Requirements

### 11.1 Default Configurations

> **REQ-OR-044**: *Sensible Defaults for All Hyperparameters* -- The orchestrator shall provide sensible defaults via `PipelineConfig()` (REQ-DM-001) so that `run_pipeline(task)` works without any configuration:
>
> | Parameter | Default | Source |
> |-----------|---------|--------|
> | M (num_retrieved_models) | 4 | REF-01 Section 4 |
> | T (outer_loop_steps) | 4 | REF-01 Section 4 |
> | K (inner_loop_steps) | 4 | REF-01 Section 4 |
> | L (num_parallel_solutions) | 2 | REF-01 Section 4 |
> | R (ensemble_rounds) | 5 | REF-01 Section 4 |
> | time_limit_seconds | 86400 | REF-01 Section 4 (24h) |
> | max_budget_usd | None (unlimited) | -- |
> | max_debug_attempts | 3 | REF-01 Section 3.4 |
> | permission_mode | `"bypassPermissions"` | REF-06 Section 4 |
> | model | `"sonnet"` | -- |
> | log_level | `"INFO"` | -- |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `await run_pipeline(task)` with default `PipelineConfig` shall execute without errors on a valid task.

### 11.2 Configuration Override

> **REQ-OR-045**: *Per-Run Configuration Override* -- The `run_pipeline()` function shall accept a `PipelineConfig` instance that overrides any or all default hyperparameters:
>
> ```python
> config = PipelineConfig(
>     outer_loop_steps=2,          # T=2 instead of 4
>     inner_loop_steps=2,          # K=2 instead of 4
>     num_parallel_solutions=1,    # L=1, skip ensemble
>     time_limit_seconds=3600,     # 1 hour limit
>     max_budget_usd=10.0,         # $10 budget cap
> )
> result = await run_pipeline(task, config=config)
> ```
>
> - All hyperparameters in `PipelineConfig` shall be independently overridable.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `PipelineConfig(outer_loop_steps=1)` shall cause Phase 2 to run exactly 1 outer loop iteration.

### 11.3 Environment Variables

> **REQ-OR-046**: *Environment Variable Support* -- The orchestrator shall read the following environment variables during initialization:
>
> | Variable | Purpose | Required |
> |----------|---------|----------|
> | `ANTHROPIC_API_KEY` | SDK authentication | Yes |
> | `MLE_STAR_MODEL` | Override default model (e.g., `"opus"`) | No |
> | `MLE_STAR_LOG_LEVEL` | Override log level (e.g., `"DEBUG"`) | No |
> | `MLE_STAR_MAX_BUDGET` | Override max budget in USD | No |
> | `MLE_STAR_TIME_LIMIT` | Override time limit in seconds | No |
>
> - Environment variables shall take precedence over `PipelineConfig` defaults but be overridden by explicit `PipelineConfig` constructor arguments.
> - If `ANTHROPIC_API_KEY` is not set, `run_pipeline()` shall raise `EnvironmentError` with a clear message.
> - Priority: Must | Verify: Test | Release: MVP

### 11.4 Logging Configuration

> **REQ-OR-047**: *Configurable Logging* -- The orchestrator shall configure Python's `logging` module with:
>
> - Logger name: `"mle_star"`
> - Default level: `INFO` (overridable via `PipelineConfig.log_level` or `MLE_STAR_LOG_LEVEL`)
> - Console handler: structured log output with timestamp, level, logger name, and message
> - File handler (optional): if `PipelineConfig.log_file` is set, append logs to the specified file
> - Phase markers: log entries at phase boundaries (e.g., `"=== Phase 1: Initial Solution Generation ==="`)
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Acceptance: Setting `log_level="DEBUG"` shall produce debug-level output including agent prompts and responses.

---

## 12. Non-Functional Requirements

### 12.1 Performance

> **REQ-OR-048**: *Orchestrator Overhead* -- The orchestrator's own overhead (excluding agent calls and script execution) shall be less than 1% of total pipeline wall-clock time.
>
> - Orchestration logic (phase dispatch, result collection, hook execution) shall complete in under 100 milliseconds per phase transition.
> - Priority: Should | Verify: Test | Release: MVP

> **REQ-OR-049**: *Memory Efficiency for Parallel Paths* -- The orchestrator shall not hold all intermediate solutions in memory simultaneously. Each Phase 2 path shall retain only its current best solution and the solution under evaluation.
>
> - Historical solutions shall be summarized (score + metadata) rather than retained in full.
> - Priority: Should | Verify: Inspection | Release: MVP
> - Rationale: With L=2 paths, each generating up to T*K solutions (each potentially 50 KB+), unbounded retention could consume significant memory.

### 12.2 Observability

> **REQ-OR-050**: *Pipeline State Introspection* -- The orchestrator shall maintain a `PipelineState` object that is queryable at any time during execution:
>
> | Field | Type | Description |
> |-------|------|-------------|
> | `current_phase` | `str` | `"phase1"`, `"phase2"`, `"phase3"`, `"finalization"`, or `"complete"` |
> | `elapsed_seconds` | `float` | Wall-clock time since pipeline start |
> | `accumulated_cost_usd` | `float` | Total cost so far |
> | `phase2_path_statuses` | `list[str]` | Per-path status: `"running"`, `"completed"`, `"failed"`, `"cancelled"` |
> | `best_score_so_far` | `float | None` | Best score achieved across all phases |
> | `agent_call_count` | `int` | Total number of agent calls made |
>
> - Priority: Should | Verify: Inspection | Release: MVP

### 12.3 Reliability

> **REQ-OR-051**: *Idempotent Retry Safety* -- If `run_pipeline()` is called again after a previous failure, it shall not be affected by leftover state from the previous run.
>
> - Each `run_pipeline()` call shall create fresh state (new client, new sessions, new accumulators).
> - Working directory contents from a previous run shall be overwritten, not appended to.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Calling `run_pipeline(task)` twice in succession shall produce independent results.

> **REQ-OR-052**: *SDK Reconnection on Transient Failure* -- If the SDK client connection drops during pipeline execution, the orchestrator shall attempt to reconnect up to 3 times with exponential backoff (1s, 2s, 4s) before failing.
>
> - The reconnection shall resume the current session using `ClaudeAgentOptions(resume=session_id)`.
> - If all reconnection attempts fail, the orchestrator shall invoke graceful shutdown (REQ-OR-030).
> - Priority: Should | Verify: Test | Release: MVP
> - Source: REF-06 Section 8 (resume sessions)

---

## 13. Constraints

### 13.1 Technology Constraints

> **REQ-OR-053**: *Python 3.10+ and asyncio* -- The orchestrator shall be implemented as async Python code compatible with Python 3.10, 3.11, 3.12, and 3.13, using `asyncio` for concurrency.
>
> - The `run_pipeline()` function shall be an `async def` coroutine.
> - A synchronous wrapper `run_pipeline_sync()` shall be provided for non-async callers: `asyncio.run(run_pipeline(task, config))`.
> - Priority: Must | Verify: Test | Release: MVP

> **REQ-OR-054**: *SDK Version Dependency* -- The orchestrator shall require `claude-agent-sdk` v0.1.39 or later and shall not use deprecated SDK APIs.
>
> - If an incompatible SDK version is detected at import time, a clear `ImportError` shall be raised.
> - Priority: Must | Verify: Inspection | Release: MVP

> **REQ-OR-055**: *Single Module Implementation* -- The orchestrator shall be implemented in a single Python module (e.g., `mle_star/orchestrator.py`) to centralize pipeline control flow.
>
> - Helper classes (`PipelineState`, `PhaseTimeBudget`) may be defined in the same module.
> - Priority: Should | Verify: Inspection | Release: MVP

### 13.2 SDK Compatibility Constraints

> **REQ-OR-056**: *Concurrent Session Limit* -- The orchestrator shall respect the SDK client's maximum concurrent session count. If the SDK limits concurrent sessions to fewer than L, the orchestrator shall serialize excess paths (run them sequentially after earlier paths complete).
>
> - A warning shall be logged if serialization is required.
> - Priority: Must | Verify: Test | Release: MVP

> **REQ-OR-057**: *Agent Name Uniqueness* -- All 14 agent names registered with the SDK client shall be unique strings matching the `AgentType` enum values (REQ-DM-013).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: No two agents shall share the same name in the `agents` dictionary.

---

## 14. Traceability Matrix

### 14.1 Requirements to Paper Sections

| Req ID | Paper Section | Paper Element | SDK Construct |
|--------|--------------|---------------|---------------|
| REQ-OR-001 | Section 3 | Full pipeline | `ClaudeSDKClient` |
| REQ-OR-002 | Section 3 | Input validation | Pydantic validator |
| REQ-OR-003 | Section 4 | Working directory | -- |
| REQ-OR-004 | Section 4 | GPU detection | -- |
| REQ-OR-005 | -- | Client init | `ClaudeSDKClient()` |
| REQ-OR-006 | Section 6 | 14 agents | `AgentDefinition` |
| REQ-OR-007 | Section 3 | Kaggle persona | System prompt |
| REQ-OR-008 | Section 3 | Agent tools | `AgentDefinition.tools` |
| REQ-OR-009 | -- | Permissions | `permission_mode` |
| REQ-OR-010 | -- | Custom tools | MCP servers |
| REQ-OR-011 | -- | Cleanup | `client.disconnect()` |
| REQ-OR-012 | Algorithm 1 | Phase 1 dispatch | `run_phase1()` |
| REQ-OR-013 | Section 3.2 | L parallel Phase 2 | `run_phase2_outer_loop()` |
| REQ-OR-014 | Algorithm 3 | Phase 3 dispatch | `run_phase3()` |
| REQ-OR-015 | Algorithm 3 | L=1 skip condition | -- |
| REQ-OR-016 | Section 3.5 | Finalization | `run_finalization()` |
| REQ-OR-017 | Section 3 | Phase ordering | Sequential execution |
| REQ-OR-018 | Section 3.3 | L=2 paths | -- |
| REQ-OR-019 | Section 4 | Parallel execution | `asyncio.gather()` |
| REQ-OR-020 | Section 3.3 | Path independence | Deep copy |
| REQ-OR-021 | -- | Session isolation | `session_id` |
| REQ-OR-022 | -- | Error isolation | `return_exceptions=True` |
| REQ-OR-023 | -- | Result collection | `asyncio.gather()` |
| REQ-OR-024 | Section 4 | 24h time limit | -- |
| REQ-OR-025 | Section 4 | Time allocation | -- |
| REQ-OR-026 | Section 4 | Per-path budget | -- |
| REQ-OR-027 | -- | Cost tracking | `ResultMessage.total_cost_usd` |
| REQ-OR-028 | -- | Budget config | `max_budget_usd` |
| REQ-OR-029 | -- | Budget enforcement | -- |
| REQ-OR-030 | Section 4 | Graceful shutdown | -- |
| REQ-OR-031 | -- | Progress logging | `PostToolUse` hook |
| REQ-OR-032 | -- | Cost hook | `Stop` hook |
| REQ-OR-033 | -- | Safety hook | `PreToolUse` hook |
| REQ-OR-034 | Section 4 | Timeout hook | `PostToolUse` hook |
| REQ-OR-035 | -- | Error logging | `PostToolUse` hook |
| REQ-OR-036 | -- | Result assembly | `FinalResult` |
| REQ-OR-037 | -- | Cost summary | -- |
| REQ-OR-038 | -- | Duration summary | -- |
| REQ-OR-039 | Section 3 | Solution lineage | -- |
| REQ-OR-040 | -- | Phase 2 fallback | -- |
| REQ-OR-041 | -- | Phase 3 fallback | -- |
| REQ-OR-042 | -- | Complete failure | `PipelineError` |
| REQ-OR-043 | -- | Partial results | `FinalResult` |
| REQ-OR-044 | Section 4 | Default configs | `PipelineConfig()` |
| REQ-OR-045 | -- | Config override | `PipelineConfig(...)` |
| REQ-OR-046 | -- | Env vars | `os.environ` |
| REQ-OR-047 | -- | Logging | `logging` module |
| REQ-OR-048 | -- | Overhead | -- |
| REQ-OR-049 | -- | Memory efficiency | -- |
| REQ-OR-050 | -- | Observability | `PipelineState` |
| REQ-OR-051 | -- | Idempotency | -- |
| REQ-OR-052 | -- | Reconnection | `resume` session |
| REQ-OR-053 | -- | Python + asyncio | `async def` |
| REQ-OR-054 | -- | SDK version | `claude-agent-sdk` |
| REQ-OR-055 | -- | Single module | `orchestrator.py` |
| REQ-OR-056 | -- | Session limit | -- |
| REQ-OR-057 | -- | Agent uniqueness | `AgentType` enum |

### 14.2 Cross-References to Other Specs

| Req ID | References Spec | Referenced Requirement |
|--------|----------------|----------------------|
| REQ-OR-001 | SPEC-01 | REQ-DM-007 (TaskDescription), REQ-DM-001 (PipelineConfig), REQ-DM-025 (FinalResult) |
| REQ-OR-002 | SPEC-01 | REQ-DM-002 (PipelineConfig validation), REQ-DM-007 (TaskDescription) |
| REQ-OR-003 | SPEC-02 | REQ-EX-001 (setup_working_directory) |
| REQ-OR-004 | SPEC-02 | REQ-EX-003 (detect_gpu) |
| REQ-OR-005 | SPEC-01 | REQ-DM-036 (AgentConfig), REQ-DM-040 (build_default_agent_configs) |
| REQ-OR-006 | SPEC-01 | REQ-DM-013 (AgentType enum), REQ-DM-037 (to_agent_definition) |
| REQ-OR-008 | SPEC-01 | REQ-DM-036 (AgentConfig.tools) |
| REQ-OR-010 | SPEC-01 | REQ-DM-026 (ScoreFunction) |
| REQ-OR-012 | SPEC-04 | REQ-P1-* (Phase 1 entry point) |
| REQ-OR-013 | SPEC-05 | REQ-P2O-* (Phase 2 outer loop entry point) |
| REQ-OR-014 | SPEC-07 | REQ-P3-* (Phase 3 entry point) |
| REQ-OR-016 | SPEC-08 | REQ-FN-* (Finalization entry point) |
| REQ-OR-020 | SPEC-01 | REQ-DM-009 (SolutionScript) |
| REQ-OR-027 | SPEC-01 | REQ-DM-025 (FinalResult.total_cost_usd) |
| REQ-OR-033 | SPEC-02, SPEC-03 | REQ-EX-* (execution safety), REQ-SF-* (safety agents) |
| REQ-OR-036 | SPEC-01 | REQ-DM-022 (Phase1Result), REQ-DM-023 (Phase2Result), REQ-DM-024 (Phase3Result), REQ-DM-025 (FinalResult) |
| REQ-OR-040 | SPEC-04 | Phase 1 initial solution as fallback |
| REQ-OR-041 | SPEC-01 | REQ-DM-029 (is_improvement_or_equal) |

### 14.3 Requirements Referenced By Other Specs

This is the final integration spec (09 of 09). No other spec references requirements defined here, as this spec consumes all other specs' entry points.

---

## 15. Change Control

### 15.1 Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2026-02-20 | MLE-STAR Team | Initial draft -- all 57 requirements |

### 15.2 Baselining Policy

This SRS is baselined at version 1.0. After baselining, changes require impact analysis across all 9 specs, as this spec integrates all other specs and changes here may propagate to interface contracts. Changes to phase entry point signatures (REQ-OR-012 through REQ-OR-016) require coordinated updates with the referenced specs.
