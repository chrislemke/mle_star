# SRS 07 — Phase 3: Algorithm 3 Orchestration

---

## 5. Algorithm 3 Orchestration Requirements

### 5.1 Phase 3 Entry Point

> **REQ-P3-017**: *Phase 3 Entry Point* -- The system shall define an async function `run_phase3(solutions: list[SolutionScript], task: TaskDescription, config: PipelineConfig) -> Phase3Result` that implements Algorithm 3 from REF-01 Appendix B. This function orchestrates the full Phase 3 pipeline: ensemble planning, ensemble implementation, evaluation, and best selection across R rounds.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `run_phase3(solutions, task, config)` shall return a `Phase3Result` (REQ-DM-024) with a non-None `best_ensemble` and `best_ensemble_score`.
> - Source: REF-01 Algorithm 3

### 5.2 Skip Condition

> **REQ-P3-018**: *Single Solution Skip Condition* -- If `len(solutions) == 1` (i.e., L=1, only one parallel refinement path was configured or produced a result), the `run_phase3` function shall skip the ensemble process entirely and return a `Phase3Result` with:
>
> | Field | Value |
> |-------|-------|
> | `input_solutions` | `[solutions[0]]` |
> | `ensemble_plans` | `[]` (empty) |
> | `ensemble_scores` | `[]` (empty) |
> | `best_ensemble` | `solutions[0]` (the single solution, unchanged) |
> | `best_ensemble_score` | `solutions[0].score` |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `run_phase3([single_solution], task, config)` shall return immediately with the single solution as `best_ensemble`, without invoking A_ens_planner or A_ensembler.
> - Source: REF-01 Section 3.3 -- ensembling requires multiple solutions; REQ-DM-001 (L default=2 but L=1 is valid)

### 5.3 Initial Round (r=0)

> **REQ-P3-019**: *Algorithm 3 Step 1 -- Initial Ensemble Plan* -- The `run_phase3` function shall begin the ensemble loop by invoking `invoke_ens_planner(solutions, [], [])` (REQ-P3-009) with no history to obtain the initial ensemble plan e_0.
>
> - This corresponds to Algorithm 3 line 1: `e_0 = A_ens_planner({s_final^l}_{l=1}^L)`.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 3 line 1

> **REQ-P3-020**: *Algorithm 3 Step 2 -- Initial Ensemble Implementation* -- After obtaining e_0, the function shall invoke `invoke_ensembler(e_0, solutions)` (REQ-P3-016) to produce the initial ensemble solution s_ens^0.
>
> - This corresponds to Algorithm 3 line 2: `s_ens^0 = A_ensembler(e_0, {s_final^l}_{l=1}^L)`.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 3 line 2

> **REQ-P3-021**: *Algorithm 3 Step 3 -- Initial Ensemble Evaluation* -- After obtaining s_ens^0, the function shall:
>
> 1. Invoke `check_and_fix_leakage(s_ens^0)` (REQ-SF-020, REQ-SF-022) before evaluation.
> 2. Evaluate h(s_ens^0) using `evaluate_with_retry` (REQ-EX-021) with the debug callback (REQ-SF-007).
> 3. Record the score h(s_ens^0).
>
> - This corresponds to Algorithm 3 line 3: "Evaluate h(s_ens^0) using D".
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 3 line 3

### 5.4 Subsequent Rounds (r=1..R-1)

> **REQ-P3-022**: *Algorithm 3 Steps 4-8 -- Iteration Loop* -- For each ensemble round r from 1 to R-1 (where R = `config.ensemble_rounds`), the `run_phase3` function shall:
>
> 1. Invoke `invoke_ens_planner(solutions, accumulated_plans, accumulated_scores)` to obtain e_r, providing the full history of all previous plans and their scores.
> 2. If A_ens_planner returns `None` (empty response), skip this round: record `score=None` and a placeholder plan, and proceed to r+1.
> 3. Invoke `invoke_ensembler(e_r, solutions)` (REQ-P3-016) to produce s_ens^r.
> 4. If A_ensembler returns `None` (extraction failed), record `score=None` and proceed to r+1.
> 5. Invoke `check_and_fix_leakage(s_ens^r)` (REQ-SF-020, REQ-SF-022) before evaluation.
> 6. Evaluate h(s_ens^r) using `evaluate_with_retry` (REQ-EX-021) with the debug callback (REQ-SF-007).
> 7. Record the score h(s_ens^r).
> 8. Append e_r to `accumulated_plans` and h(s_ens^r) to `accumulated_scores`.
>
> - This implements Algorithm 3 lines 4-8.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given `config.ensemble_rounds = 5`, the loop shall attempt rounds r=1, r=2, r=3, r=4 (4 iterations after the initial round at r=0).
> - Source: REF-01 Algorithm 3 lines 4-8

> **REQ-P3-023**: *A_ens_planner Receives Full History* -- At ensemble round r, the A_ens_planner invocation shall receive the complete history of all previous attempts (j=0 through j=r-1), including:
>
> - All plan texts: `[e_0, e_1, ..., e_{r-1}]`
> - All scores: `[h(s_ens^0), h(s_ens^1), ..., h(s_ens^{r-1})]`
>
> - Scores for failed attempts (execution error after debugging, or ensembler failure) shall be represented as `None` in the scores list.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: At r=3, A_ens_planner shall receive plans and scores for rounds r=0, r=1, and r=2.
> - Source: REF-01 Algorithm 3 line 5 -- `e_r = A_ens_planner({s_final^l}_{l=1}^L, {(e_j, h(s_ens^j))}_{j=0}^{r-1})`

### 5.5 Score Tracking and Best Selection

> **REQ-P3-024**: *Score Tracking* -- The `run_phase3` function shall maintain a list of all ensemble scores `[h(s_ens^0), h(s_ens^1), ..., h(s_ens^{R-1})]` and a parallel list of all ensemble solutions `[s_ens^0, s_ens^1, ..., s_ens^{R-1}]`, recording `None` for scores of failed attempts.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After R=5 rounds, the score list shall have exactly 5 entries.
> - Source: REF-01 Algorithm 3 lines 3, 7 -- evaluation at each round

> **REQ-P3-025**: *Best Ensemble Selection* -- After all R rounds complete, the function shall select the best ensemble solution s_ens* as the solution with the best score across all R attempts:
>
> - For `metric_direction == "maximize"`: `r* = argmax_{r in {0,...,R-1}} h(s_ens^r)` (considering only non-None scores).
> - For `metric_direction == "minimize"`: `r* = argmin_{r in {0,...,R-1}} h(s_ens^r)` (considering only non-None scores).
> - If multiple rounds share the best score, the last such round shall be selected (consistent with `>=` semantics in the rest of the pipeline).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given scores `[0.85, 0.88, None, 0.87, 0.88]` with maximize direction, r*=4 (last occurrence of 0.88) and s_ens* = s_ens^4.
> - Source: REF-01 Algorithm 3 line 9 -- `r* = arg max_{r} h(s_ens^r)`

> **REQ-P3-026**: *All Rounds Failed* -- If all R ensemble attempts fail (all scores are `None`), the `run_phase3` function shall fall back to the single best input solution. Specifically:
>
> 1. Select the input solution with the best score from the L input solutions.
> 2. Log a warning: "Phase 3 ensemble: all {R} attempts failed; falling back to best input solution".
> 3. Return a Phase3Result with `best_ensemble` set to the best input solution and `best_ensemble_score` set to that solution's score.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given R=5 where all ensemble scripts fail, the returned `best_ensemble` shall be the input solution with the highest (or lowest, per direction) score.

### 5.6 Safety Integration

> **REQ-P3-027**: *Leakage Check Before Ensemble Evaluation* -- At each ensemble round r, after A_ensembler produces s_ens^r and before evaluation, the system shall invoke `check_and_fix_leakage(s_ens^r)` (REQ-SF-020, REQ-SF-022). The leakage-checked (and potentially corrected) solution shall be the one submitted for evaluation.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Acceptance: Every code path from `invoke_ensembler` to `evaluate_solution` shall include a `check_and_fix_leakage` call.
> - Source: REF-01 Section 3.4 -- "Every generated solution before evaluation (all phases)"; REQ-SF-022

> **REQ-P3-028**: *Debug on Ensemble Execution Error* -- When evaluating s_ens^r produces an execution error, the system shall use the debug retry mechanism by invoking `evaluate_with_retry` (REQ-EX-021) with the debug callback from `make_debug_callback(task, config)` (REQ-SF-007).
>
> - The debug retry shall attempt up to `config.max_debug_attempts` fixes before declaring the attempt failed.
> - If debug succeeds, the fixed solution and its score shall be used for this round's result.
> - If debug exhausts all retries and the solution still errors, the round shall be recorded with `score=None`.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given an ensemble script that errors once and is fixed by A_debugger, the round shall record the score from the fixed solution.
> - Source: REF-01 Section 3.4 -- A_debugger applies to all generated code; REQ-SF-006, REQ-SF-007

### 5.7 Error Handling

> **REQ-P3-029**: *Ensemble Script Failure Handling* -- If an ensemble script s_ens^r fails to execute after all debug retries are exhausted, the system shall:
>
> 1. Log a warning indicating that ensemble round r failed: plan summary (first 200 characters), error summary (first line of traceback).
> 2. Record `score=None` for this round in both the score tracking list and the `EnsembleAttempt` record.
> 3. Continue to the next round (r+1). Do not abort Phase 3.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given R=5 where rounds 0 and 2 fail, the function shall complete all 5 rounds and select the best from the 3 successful rounds.

> **REQ-P3-030**: *Ensembler Extraction Failure Handling* -- If `invoke_ensembler` returns `None` (the agent's response did not contain an extractable code block), the system shall:
>
> 1. Log a warning indicating the ensembler response could not be parsed at round r.
> 2. Record an `EnsembleAttempt` with `score=None` and a `SolutionScript` with empty `content`.
> 3. Continue to the next round (r+1).
>
> - Priority: Must | Verify: Test | Release: MVP

> **REQ-P3-031**: *Ens_planner Failure Handling* -- If `invoke_ens_planner` returns `None` (empty response) at round r, the system shall:
>
> 1. Log a warning indicating the ens_planner returned an empty response at round r.
> 2. Record an `EnsembleAttempt` with `plan="[ens_planner failed]"`, `score=None`, and a `SolutionScript` with empty `content`.
> 3. Append the placeholder plan and `None` score to the accumulated history so that subsequent rounds have accurate context.
> 4. Continue to the next round (r+1).
>
> - Priority: Must | Verify: Test | Release: MVP

### 5.8 EnsembleAttempt History

> **REQ-P3-032**: *EnsembleAttempt Record Construction* -- At each ensemble round r, the system shall construct an `EnsembleAttempt` (REQ-DM-043) with the following field values:
>
> | Field | Value |
> |-------|-------|
> | `plan` | The ensemble plan text e_r used for this round |
> | `score` | h(s_ens^r) if evaluation succeeded, `None` if evaluation failed |
> | `solution` | The ensemble solution s_ens^r (or an empty-content SolutionScript if ensembler failed) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After R=5 rounds, the history list shall contain exactly 5 `EnsembleAttempt` records.
> - Source: REF-01 Algorithm 3 -- tracks `{(e_j, h(s_ens^j))}` history; REQ-DM-043

> **REQ-P3-033**: *EnsembleAttempt History Ordering* -- The list of `EnsembleAttempt` records shall be ordered by round index (r=0 first, r=1 second, etc.). The list shall always have length R (= `config.ensemble_rounds`), including records for failed rounds.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `attempts[0].plan` shall correspond to e_0 (the initial ensemble plan).

### 5.9 Phase3Result Construction

> **REQ-P3-034**: *Phase3Result Construction* -- Upon completion of all R ensemble rounds and best selection, the `run_phase3` function shall construct a `Phase3Result` (REQ-DM-024) with:
>
> | Field | Value |
> |-------|-------|
> | `input_solutions` | The list of L `SolutionScript` instances received as input |
> | `ensemble_plans` | The list of ensemble plan texts `[e_0, e_1, ..., e_{R-1}]` |
> | `ensemble_scores` | The list of scores `[h(s_ens^0), h(s_ens^1), ..., h(s_ens^{R-1})]` (including `None` for failed rounds) |
> | `best_ensemble` | The best ensemble solution s_ens* (REQ-P3-025) |
> | `best_ensemble_score` | The score of s_ens* |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `len(Phase3Result.ensemble_plans) == len(Phase3Result.ensemble_scores) == config.ensemble_rounds`.
> - Source: REQ-DM-024

> **REQ-P3-035**: *Phase3Result Score Consistency* -- The `Phase3Result.best_ensemble_score` shall equal the score of `Phase3Result.best_ensemble`. The `best_ensemble` shall be the `SolutionScript` instance corresponding to the best-scoring ensemble round (REQ-P3-025) or the best input solution if all rounds failed (REQ-P3-026).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `Phase3Result.best_ensemble_score` shall equal the score recorded for the round selected by the best-selection logic.
