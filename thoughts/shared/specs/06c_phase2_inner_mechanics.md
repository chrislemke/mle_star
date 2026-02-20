# SRS 06 — Phase 2 Inner Loop: Mechanics

### 5.4 Code Block Replacement

> **REQ-P2I-022**: *Code Block Replacement Mechanism* -- At each inner step k, the system shall construct the candidate solution s_t^k by calling `solution.replace_block(code_block.content, c_t_k)` (REQ-DM-010), where:
>
> - `solution` is the original solution s_t (not the result of a previous inner step).
> - `code_block.content` is the original code block c_t.
> - `c_t_k` is the refined code block produced by A_coder at step k.
>
> - The replacement produces a new `SolutionScript` instance; the original `solution` is not mutated.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `solution.replace_block(c_t, c_t_k).content` shall equal `solution.content` with the first occurrence of `c_t` replaced by `c_t_k`.
> - Source: REF-01 Algorithm 2 lines 10, 19 -- `s_t^k = s_t.replace(c_t, c_t^k)`

> **REQ-P2I-023**: *Replacement Uses Original Solution* -- Each code block replacement shall be performed against the original solution s_t (the solution passed to `run_phase2_inner_loop`), not against any previously modified s_t^j. This ensures that each inner step is independent and that the replacement target c_t is always present.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Rationale: Since c_t is replaced by c_t^k, the code block c_t may no longer exist in s_t^k. Replacing against the original guarantees c_t is found.
> - Source: REF-01 Algorithm 2 lines 10, 19 -- `s_t^k = s_t.replace(c_t, c_t^k)` (s_t is the fixed base)

### 5.5 Score Tracking

> **REQ-P2I-024**: *Best Score Initialization* -- The inner loop shall initialize its best-tracking state from the `best_score` parameter:
>
> - `local_best_score = best_score` (the h_best from the outer loop)
> - `local_best_solution = solution` (the s_t from the outer loop)
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 2 lines 1-2 (outer loop initialization, carried into inner loop)

> **REQ-P2I-025**: *Best Score Update* -- After each successful evaluation at inner step k, the system shall compare h(s_t^k) against `local_best_score` using `is_improvement_or_equal(new_score=h(s_t^k), old_score=local_best_score, direction=task.metric_direction)` (REQ-DM-029):
>
> - If the comparison returns `True`: update `local_best_solution = s_t^k` and `local_best_score = h(s_t^k)`.
> - If the comparison returns `False`: retain the current best.
> - Mark the corresponding `RefinementAttempt.was_improvement` accordingly.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a maximize metric, if h(s_t^0) = 0.82 and h(s_t^1) = 0.85, the best shall be updated to s_t^1 with score 0.85 after step 1.
> - Acceptance: Given a maximize metric, if h(s_t^0) = 0.85 and h(s_t^1) = 0.82, the best shall remain at s_t^0 with score 0.85.
> - Source: REF-01 Algorithm 2 lines 12-14, 21-24 -- `if h(s_t^k) >= h_best then ...`

> **REQ-P2I-026**: *Score Comparison Semantics (>=)* -- The inner loop shall use `is_improvement_or_equal` (REQ-DM-029) for the best-update check, implementing >= semantics as specified in Algorithm 2. This means that a score equal to h_best will also trigger an update, keeping the most recent solution with the best score.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given h_best = 0.82 and h(s_t^k) = 0.82 (maximize), the update condition shall be `True` and the best solution shall be updated to s_t^k.
> - Source: REF-01 Algorithm 2 lines 12, 21 -- `if h(s_t^k) >= h_best`

> **REQ-P2I-027**: *Failed Evaluation Score Handling* -- When an evaluation fails (execution error after all debug retries are exhausted, or evaluation returns `score=None`), the score for that attempt shall be recorded as `None` in both the `RefinementAttempt.score` field and the `accumulated_scores` list. A `None` score shall never trigger a best-score update.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: An attempt with `score=None` shall have `RefinementAttempt.was_improvement = False`.
> - Acceptance: A `None` score shall not be passed to `is_improvement_or_equal`.

### 5.6 History Accumulation

> **REQ-P2I-028**: *RefinementAttempt Record Construction* -- At each inner step k, the system shall construct a `RefinementAttempt` (REQ-DM-042) with the following field values:
>
> | Field | Value |
> |-------|-------|
> | `plan` | The plan text p_k used for this attempt |
> | `score` | h(s_t^k) if evaluation succeeded, `None` if evaluation failed |
> | `code_block` | The refined code block c_t^k if A_coder succeeded, empty string `""` if A_coder failed |
> | `was_improvement` | `True` if this attempt triggered a best-score update, `False` otherwise |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After K=4 inner steps, the history list shall contain exactly 4 `RefinementAttempt` records.
> - Source: REF-01 Algorithm 2 -- tracks `{(p_j, h(s_t^j))}` history

> **REQ-P2I-029**: *History List Ordering* -- The list of `RefinementAttempt` records shall be ordered by inner step index (k=0 first, k=1 second, etc.). The list shall always have length K (= `config.inner_loop_steps`), including records for skipped or failed attempts.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `attempts[0].plan == initial_plan` (the p_0 from A_extractor).

### 5.7 Safety Integration

> **REQ-P2I-030**: *Leakage Check Before Evaluation* -- At each inner step k, after constructing s_t^k via `replace_block`, and before evaluation, the system shall invoke `check_and_fix_leakage(s_t^k)` (REQ-SF-020, REQ-SF-022). The leakage-checked (and potentially corrected) solution shall be the one submitted for evaluation.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Acceptance: Every code path from `replace_block` to `evaluate_solution` shall include a `check_and_fix_leakage` call.
> - Source: REF-01 Section 3.4 -- "Every generated solution before evaluation (all phases)"; REQ-SF-022

> **REQ-P2I-031**: *Debug on Execution Error* -- When evaluating s_t^k produces an execution error, the system shall use the debug retry mechanism by invoking `evaluate_with_retry` (REQ-EX-021) with the debug callback from `make_debug_callback(task, config)` (REQ-SF-007).
>
> - The debug retry shall attempt up to `config.max_debug_attempts` fixes before declaring the attempt failed.
> - If debug succeeds, the fixed solution and its score shall be used for this inner step's result.
> - If debug exhausts all retries and the solution still errors, the attempt shall be recorded with `score=None` and `was_improvement=False`.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a refined solution that errors once and is fixed by A_debugger, the inner step shall record the score from the fixed solution.
> - Source: REF-01 Section 3.4 -- A_debugger applies to all generated code

### 5.8 Error Handling

> **REQ-P2I-032**: *Unparseable Coder Output* -- If `invoke_coder` returns `None` (the agent's response did not contain an extractable code block), the inner loop shall:
>
> 1. Log a warning indicating the coder response could not be parsed at inner step k.
> 2. Record a `RefinementAttempt` with `score=None`, `code_block=""`, and `was_improvement=False`.
> 3. Proceed to the next inner step (k+1) without performing replacement or evaluation.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given an unparseable coder response at k=1, the inner loop shall continue to k=2 and the attempt at k=1 shall have `code_block=""`.

> **REQ-P2I-033**: *Replacement Failure* -- If `solution.replace_block(code_block.content, c_t_k)` raises `ValueError` (the original code block c_t is not found in the solution), the inner loop shall:
>
> 1. Log a warning indicating the code block replacement failed at inner step k.
> 2. Record a `RefinementAttempt` with `score=None`, `code_block=c_t_k` (the coder output), and `was_improvement=False`.
> 3. Proceed to the next inner step (k+1) without performing evaluation.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a replacement failure at k=0, the inner loop shall proceed to k=1 and the attempt at k=0 shall have `score=None`.
> - Rationale: Although A-01 assumes c_t is always present, edge cases (e.g., leakage correction modifying the target area) may cause replacement to fail. Graceful handling prevents the entire inner loop from aborting.

> **REQ-P2I-034**: *Planner Failure* -- If `invoke_planner` returns `None` (empty response) at inner step k (k >= 1), the inner loop shall:
>
> 1. Log a warning indicating the planner returned an empty response at inner step k.
> 2. Record a `RefinementAttempt` with `plan="[planner failed]"`, `score=None`, `code_block=""`, and `was_improvement=False`.
> 3. Proceed to the next inner step (k+1).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a planner failure at k=1, the inner loop shall continue to k=2.

> **REQ-P2I-035**: *Accumulated Plans Include Failed Attempts* -- When accumulating history for A_planner, failed attempts (where A_coder failed, replacement failed, or evaluation failed) shall still have their plan text included in the `accumulated_plans` list, and their score included as `None` in the `accumulated_scores` list. This ensures A_planner has full context about what was tried and what failed.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: If attempt k=1 failed evaluation, A_planner at k=2 shall receive plans `[p_0, p_1]` and scores `[h(s_t^0), None]`.
> - Rationale: Even failed plans provide useful context; A_planner should avoid repeating strategies that caused errors.

### 5.9 Result Construction

> **REQ-P2I-036**: *InnerLoopResult Model* -- The system shall define a Pydantic model (or dataclass) `InnerLoopResult` with the following fields:
>
> | Field | Type | Description |
> |-------|------|-------------|
> | `best_solution` | `SolutionScript` | Best solution found across all K attempts (or the original if none improved) |
> | `best_score` | `float` | Score of the best solution |
> | `attempts` | `list[RefinementAttempt]` | Ordered list of all K attempt records |
> | `improved` | `bool` | Whether any attempt improved upon the input `best_score` |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `InnerLoopResult.improved` shall be `True` if and only if `InnerLoopResult.best_score` is strictly better than the input `best_score` (using `is_improvement`, REQ-DM-028).
> - Source: REF-01 Algorithm 2 lines 9-25 (inner loop output)

> **REQ-P2I-037**: *InnerLoopResult Construction* -- Upon completion of all K inner steps, the `run_phase2_inner_loop` function shall construct and return an `InnerLoopResult` with:
>
> - `best_solution`: the `local_best_solution` tracked throughout the loop (REQ-P2I-024, REQ-P2I-025).
> - `best_score`: the `local_best_score` tracked throughout the loop.
> - `attempts`: the full list of `RefinementAttempt` records in step order (REQ-P2I-029).
> - `improved`: `True` if `local_best_score` differs from the input `best_score` (checked via `is_improvement`, REQ-DM-028).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: The returned `InnerLoopResult.attempts` shall have length exactly `config.inner_loop_steps`.
> - Source: REQ-P2O-026, REQ-P2O-027 (outer loop expects this return type)

> **REQ-P2I-038**: *InnerLoopResult Preserves Input on No Improvement* -- If no inner step produces an improvement, the `InnerLoopResult.best_solution` shall be the original `solution` (s_t) and `InnerLoopResult.best_score` shall be the original `best_score` (h_best). The system shall never return a result with a worse score than the input.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given K=4 inner steps where all evaluations score below h_best, the returned `best_solution.content` shall equal `solution.content`.
> - Source: REF-01 Algorithm 2 -- conditional update only on `>=`
