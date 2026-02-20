# SRS 09 — Orchestrator: Cost/Time Control and Hooks

## 7. Cost and Time Control Requirements

### 7.1 Time Limit Enforcement

> **REQ-OR-024**: *Overall Time Limit* -- The `run_pipeline()` function shall enforce `config.time_limit_seconds` (default: 86400, i.e., 24 hours) as the maximum wall-clock time for the entire pipeline.
>
> - A `datetime`-based deadline shall be computed at pipeline start: `deadline = time.monotonic() + config.time_limit_seconds`.
> - The deadline shall be checked before each phase begins and at the start of each agent call.
> - If the deadline is exceeded, the pipeline shall invoke graceful shutdown (REQ-OR-030).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: A pipeline configured with `time_limit_seconds=60` that would otherwise run longer shall terminate within 90 seconds (allowing 30 seconds for shutdown).
> - Source: REF-01 Section 4 ("24-hour time limit per competition")

### 7.2 Per-Phase Time Budgets

> **REQ-OR-025**: *Proportional Time Allocation* -- The orchestrator shall allocate the total time budget across phases according to the following default proportions:
>
> | Phase | Proportion | Rationale |
> |-------|-----------|-----------|
> | Phase 1 | 10% | ~10 agent calls |
> | Phase 2 | 65% | ~44 agent calls per path, L paths |
> | Phase 3 | 15% | ~10 agent calls |
> | Finalization | 10% | Script execution + verification |
>
> - These proportions shall be configurable via a `PhaseTimeBudget` configuration (optional field on `PipelineConfig`).
> - The remaining time (after Phase 1 completes) shall be redistributed proportionally among remaining phases.
> - Priority: Should | Verify: Test | Release: MVP
> - Source: REF-01 Section 4 (average 14.1 hours per solution)

> **REQ-OR-026**: *Phase 2 Time Enforcement* -- Each parallel Phase 2 path shall receive `phase2_time_budget / L` seconds as its maximum execution time to ensure all paths get a fair share.
>
> - If a path exceeds its time budget, it shall be cancelled and its best intermediate result shall be used.
> - The Phase 2 time budget shall be calculated as: `remaining_time * 0.65 / 0.90` (to account for Phase 3 and finalization).
> - Priority: Should | Verify: Test | Release: MVP

### 7.3 Cost Tracking

> **REQ-OR-027**: *Cost Accumulation* -- The orchestrator shall accumulate `total_cost_usd` across all agent calls by reading `ResultMessage.total_cost_usd` from each SDK response.
>
> - Cost shall be tracked per-phase for the cost summary (REQ-OR-037).
> - Cost shall be tracked per-path for parallel Phase 2 paths.
> - The running total shall be accessible to hooks (REQ-OR-032).
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-06 Section 2 (ResultMessage.total_cost_usd)

### 7.4 Budget Enforcement

> **REQ-OR-028**: *PipelineConfig Budget Field* -- The `PipelineConfig` model (REQ-DM-001) shall include an optional field `max_budget_usd: float | None` (default: `None`, meaning unlimited).
>
> - When set, this field shall be passed to the SDK client as `max_budget_usd` during initialization.
> - Priority: Must | Verify: Test | Release: MVP

> **REQ-OR-029**: *Budget Exceeded Handling* -- When the accumulated cost reaches `config.max_budget_usd`, the orchestrator shall invoke graceful shutdown (REQ-OR-030).
>
> - The budget check shall occur after each agent call completes.
> - A warning shall be logged when 80% of the budget is consumed.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: A pipeline configured with `max_budget_usd=1.00` that accumulates $1.01 in cost shall trigger graceful shutdown.

### 7.5 Graceful Shutdown

> **REQ-OR-030**: *Graceful Shutdown on Timeout or Budget* -- When time or budget limits are exceeded, the orchestrator shall:
>
> 1. Cancel any in-progress agent calls and asyncio tasks.
> 2. Collect the best solution found so far (from whichever phase completed last).
> 3. If at least Phase 1 has completed, skip remaining phases and proceed directly to finalization with the best available solution.
> 4. If Phase 1 has not completed, raise `PipelineTimeoutError` with diagnostic information.
> 5. The `FinalResult` shall indicate which phases were completed and which were skipped.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: A pipeline that times out during Phase 2 shall still produce a `FinalResult` using the Phase 1 solution.

---

## 8. Hook Requirements

### 8.1 Progress Tracking Hook

> **REQ-OR-031**: *PostToolUse Progress Hook* -- The orchestrator shall register a `PostToolUse` hook that logs agent activity for observability:
>
> - Log entry shall include: timestamp, agent type, tool name, session ID, elapsed time, and a success/failure indicator.
> - Log format shall be structured (JSON) to support automated analysis.
> - The hook shall not modify tool results or block execution.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-06 Section 7 (Hooks)

### 8.2 Cost Tracking Hook

> **REQ-OR-032**: *Cost Accumulation Hook* -- The orchestrator shall register a hook (on `Stop` or `SubagentStop` events) that accumulates per-agent-call costs:
>
> - The hook shall read `ResultMessage.total_cost_usd` from each completed agent turn.
> - The hook shall update a shared cost accumulator (thread-safe for concurrent paths).
> - The hook shall trigger budget checks (REQ-OR-029) after each update.
> - Priority: Must | Verify: Test | Release: MVP

### 8.3 Safety Hook

> **REQ-OR-033**: *PreToolUse Safety Hook* -- The orchestrator shall register a `PreToolUse` hook that blocks dangerous bash commands:
>
> - Blocked patterns shall include: `rm -rf /`, `mkfs`, `dd if=`, `:(){ :|:& };:` (fork bomb), and any command that modifies files outside the working directory.
> - The hook shall return a `BlockToolUse` result with an explanation when a dangerous command is detected.
> - The blocked-command list shall be configurable.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: SPEC-02 (execution safety), REF-06 Section 7 (PreToolUse hook)

### 8.4 Timeout Hook

> **REQ-OR-034**: *Elapsed Time Monitoring Hook* -- The orchestrator shall register a hook that monitors elapsed wall-clock time and triggers graceful shutdown when the deadline is approaching:
>
> - The hook shall fire on every `PostToolUse` event.
> - When remaining time is less than 10% of the total budget (or less than 5 minutes, whichever is larger), the hook shall set a "finalize now" flag.
> - The "finalize now" flag shall cause the current phase to complete its current iteration and then skip remaining iterations.
> - Priority: Must | Verify: Test | Release: MVP

### 8.5 Error Logging Hook

> **REQ-OR-035**: *PostToolUse Error Logging Hook* -- The orchestrator shall register a hook that captures and logs all tool execution failures:
>
> - The hook shall fire on tool use results that indicate failure (non-zero exit codes, error messages).
> - Each failure shall be logged with: timestamp, agent type, tool name, error message, and full traceback if available.
> - The hook shall maintain a count of consecutive failures per agent type to support circuit-breaker logic.
> - Priority: Should | Verify: Inspection | Release: MVP
