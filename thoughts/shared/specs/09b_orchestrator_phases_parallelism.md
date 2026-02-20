# SRS 09 — Orchestrator: Phase Orchestration and Parallelism

## 5. Phase Orchestration Requirements

### 5.1 Phase 1 Dispatch

> **REQ-OR-012**: *Phase 1 Invocation* -- The orchestrator shall call `run_phase1(client, task, config)` (SPEC-04) as the first pipeline phase.
>
> - Input: the initialized `ClaudeSDKClient`, `TaskDescription`, and `PipelineConfig`.
> - Output: `Phase1Result` (REQ-DM-022) containing the merged initial solution `s_0` and its score `h_best`.
> - The orchestrator shall record the start time and duration of Phase 1.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 1, SPEC-04

### 5.2 Phase 2 Dispatch

> **REQ-OR-013**: *Phase 2 Parallel Path Dispatch* -- After Phase 1 completes, the orchestrator shall dispatch L parallel Phase 2 refinement paths (where L = `config.num_parallel_solutions`).
>
> - Each path shall receive a copy of the Phase 1 initial solution `s_0` as its starting point.
> - Each path shall independently call `run_phase2_outer_loop(client, task, config, initial_solution, session_id)` (SPEC-05).
> - Output: L `Phase2Result` instances (REQ-DM-023), one per path.
> - The orchestrator shall record the start time and aggregate duration of Phase 2.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Section 3.2 (L parallel solutions), SPEC-05

### 5.3 Phase 3 Dispatch

> **REQ-OR-014**: *Phase 3 Invocation* -- After all Phase 2 paths complete (or timeout), the orchestrator shall collect the best solution from each path and call `run_phase3(client, task, config, solutions)` (SPEC-07).
>
> - Input: a list of L `SolutionScript` instances (the `best_solution` from each `Phase2Result`).
> - If any Phase 2 path failed, the orchestrator shall use the Phase 1 initial solution as a substitute for that path's contribution (REQ-OR-040).
> - Output: `Phase3Result` (REQ-DM-024).
> - The orchestrator shall record the start time and duration of Phase 3.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 3, SPEC-07

### 5.4 Phase 3 Skip Condition

> **REQ-OR-015**: *Phase 3 Skip When L=1* -- When `config.num_parallel_solutions == 1`, the orchestrator shall skip Phase 3 entirely and pass the single Phase 2 result directly to finalization.
>
> - The `FinalResult.phase3` field shall be `None` in this case (per REQ-DM-025).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: With `PipelineConfig(num_parallel_solutions=1)`, Phase 3 shall not be invoked and `FinalResult.phase3` shall be `None`.

### 5.5 Finalization Dispatch

> **REQ-OR-016**: *Finalization Invocation* -- After Phase 3 completes (or is skipped), the orchestrator shall call `run_finalization(client, task, config, best_solution)` (SPEC-08).
>
> - Input: the best solution from Phase 3 (or Phase 2 if Phase 3 was skipped or failed).
> - Output: `FinalResult` (REQ-DM-025) including the `submission_path` to the generated `submission.csv`.
> - The orchestrator shall record the start time and duration of finalization.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: SPEC-08

### 5.6 Phase Ordering

> **REQ-OR-017**: *Strictly Sequential Phase Execution* -- The pipeline phases shall execute in strict sequential order: Phase 1 -> Phase 2 (L parallel) -> Phase 3 -> Finalization.
>
> - Phase 2 shall not begin until Phase 1 completes.
> - Phase 3 shall not begin until all Phase 2 paths complete (or timeout).
> - Finalization shall not begin until Phase 3 completes (or is skipped/failed).
> - Within Phase 2, the L paths run concurrently (see Section 6), but Phase 2 as a whole is a single sequential stage.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Timestamps shall confirm that no phase starts before its predecessor completes.
> - Source: REF-01 Section 3 (pipeline flow: Algorithm 1 -> Algorithm 2 -> Algorithm 3 -> submission)

---

## 6. Parallelism Requirements

### 6.1 L Parallel Paths

> **REQ-OR-018**: *L Independent Phase 2 Paths* -- The orchestrator shall create L independent Phase 2 refinement paths, where L = `config.num_parallel_solutions` (default: 2).
>
> - Phase 1 runs once and produces a single initial solution `s_0`.
> - The initial solution is copied (deep copy) to each of the L paths.
> - Each path independently refines its copy through T outer x K inner iterations.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Section 3.3 ("We run L=2 parallel solutions")

### 6.2 Concurrent Execution

> **REQ-OR-019**: *Asyncio-Based Concurrent Paths* -- The L Phase 2 paths shall run concurrently using `asyncio.gather()` (or equivalent asyncio concurrency primitive).
>
> ```python
> phase2_tasks = [
>     run_phase2_outer_loop(client, task, config, solution_copy, session_id=f"path-{i}")
>     for i in range(config.num_parallel_solutions)
> ]
> phase2_results = await asyncio.gather(*phase2_tasks, return_exceptions=True)
> ```
>
> - The orchestrator shall not serialize the L paths unless the SDK client cannot support concurrent sessions.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Section 4 (parallel solution paths)

### 6.3 Path Independence

> **REQ-OR-020**: *Forked Solution Copies* -- Each Phase 2 path shall operate on a deep copy of the Phase 1 initial solution, so that modifications in one path do not affect any other path.
>
> - The copy shall include the full `SolutionScript` content and score.
> - Each path shall have its own working subdirectory (e.g., `./work/path-0/`, `./work/path-1/`) to avoid file system conflicts.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Modifying the solution in path 0 shall have no effect on the solution in path 1.

### 6.4 Session Isolation

> **REQ-OR-021**: *Per-Path Session IDs* -- Each parallel Phase 2 path shall use a unique SDK `session_id` to maintain conversation context isolation.
>
> - Session IDs shall follow the pattern `"path-{i}"` where `i` is the zero-based path index.
> - All agent calls within a path shall use that path's session ID.
> - Sessions may be forked from the Phase 1 session using `ClaudeAgentOptions(resume=phase1_session_id, fork_session=True)` to carry forward context.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-06 Section 8 (Session management, fork_session)

### 6.5 Error Isolation

> **REQ-OR-022**: *Path Failure Isolation* -- A failure in one Phase 2 path shall not cause other paths to fail.
>
> - `asyncio.gather()` shall be called with `return_exceptions=True` to capture per-path exceptions.
> - Failed paths shall be logged with full exception details.
> - The orchestrator shall proceed with results from successful paths only.
> - If all L paths fail, the orchestrator shall fall back to the Phase 1 initial solution for ensemble (REQ-OR-040).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: If path 0 raises an exception, path 1 shall still complete normally and its result shall be used.

### 6.6 Path Result Collection

> **REQ-OR-023**: *Wait for All Paths* -- The orchestrator shall wait for all L paths to complete (or reach their per-path time budget) before proceeding to Phase 3.
>
> - The wait shall be bounded by the Phase 2 time allocation (REQ-OR-026).
> - If the time allocation is exceeded, still-running paths shall be cancelled via `asyncio.Task.cancel()`.
> - Cancelled paths shall be treated as failed paths (REQ-OR-022).
> - The orchestrator shall collect results only from paths that completed successfully.
> - Priority: Must | Verify: Test | Release: MVP
