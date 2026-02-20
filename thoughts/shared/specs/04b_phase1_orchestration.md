# SRS 04 — Phase 1: Orchestration

## 6. Algorithm 1 Orchestration Requirements

### 6.1 End-to-End Phase 1 Function

> **REQ-P1-018**: *Phase 1 Entry Point* -- The system shall define an async function `run_phase1(task: TaskDescription, config: PipelineConfig) -> Phase1Result` that implements Algorithm 1 from REF-01 Appendix B. This function orchestrates the full Phase 1 pipeline: retrieval, candidate generation, evaluation, sorting, and merging.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `run_phase1(task, config)` shall return a `Phase1Result` (REQ-DM-022) with a non-None `initial_solution` and `initial_score`.
> - Source: REF-01 Algorithm 1

### 6.2 Step 1: Model Retrieval

> **REQ-P1-019**: *Algorithm 1 Step 1 -- Model Retrieval* -- The `run_phase1` function shall begin by invoking `retrieve_models(task, config)` (REQ-P1-007) to obtain a list of M retrieved models.
>
> - This corresponds to Algorithm 1 line 1: `{T_model^i, T_code^i}_{i=1}^M = A_retriever(T_task)`.
> - The retrieved models shall be stored for inclusion in the `Phase1Result`.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 1 line 1

### 6.3 Steps 2-5: Candidate Generation and Evaluation

> **REQ-P1-020**: *Algorithm 1 Steps 2-5 -- Candidate Generation* -- For each retrieved model (i = 1 to M), the `run_phase1` function shall:
>
> 1. Invoke `generate_candidate(task, model_i, config)` (REQ-P1-012) to produce `s_init^i`.
> 2. Run the leakage checker on the candidate: `s_init^i = check_and_fix_leakage(s_init^i)` (REQ-SF-022).
> 3. Evaluate the candidate using `evaluate_with_retry(s_init^i, task, config, debug_callback)` (REQ-EX-021) to obtain `(s_init^i, result_i)`.
> 4. Record `s_init^i` and the resulting score `result_i.score`.
>
> - This corresponds to Algorithm 1 lines 2-5.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 1 lines 2-5

> **REQ-P1-021**: *Candidate Evaluation Failure Handling* -- If a candidate `s_init^i` fails to execute (i.e., `result_i.is_error == True` after all debug retries are exhausted):
>
> 1. Log a warning indicating the candidate failed: model name, error summary (first line of traceback).
> 2. Record the candidate's score as `None`.
> 3. Continue to the next candidate. Do not abort Phase 1.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given M=4 candidates where 2 fail to execute, the system shall proceed with the 2 successful candidates for sorting and merging.
> - Source: REF-01 Algorithm 1 -- implicit; the algorithm evaluates all candidates

> **REQ-P1-022**: *All Candidates Failed* -- If all M candidates fail to execute (all scores are `None`), the `run_phase1` function shall raise a `RuntimeError("Phase 1 failed: all {M} candidates produced execution errors")`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given M=4 candidates where all 4 fail, the function shall raise `RuntimeError`.

### 6.4 Step 6: Sort Candidates by Score

> **REQ-P1-023**: *Algorithm 1 Step 6 -- Sort by Score* -- After evaluating all candidates, the `run_phase1` function shall sort the successful candidates (those with non-None scores) by score in the best-first order using `rank_solutions()` (REQ-EX-027) with the task's `metric_direction`.
>
> - This produces the permutation pi where pi(1) is the best-scoring candidate.
> - Candidates with `None` scores shall be placed at the end and excluded from the merge loop.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: For `direction="maximize"` with candidate scores `[0.7, 0.9, 0.8, None]`, the sorted order shall be `[0.9, 0.8, 0.7]` (excluding None).
> - Source: REF-01 Algorithm 1 line 6 -- `s_0 <- s_init^{pi(1)}`

> **REQ-P1-024**: *Algorithm 1 Step 6-7 -- Initialize Best Solution* -- The `run_phase1` function shall set the initial best solution and score:
>
> - `s_0 = s_init^{pi(1)}` (the best-scoring candidate)
> - `h_best = h(s_0)` (the score of the best candidate)
>
> - This corresponds to Algorithm 1 lines 6-7.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 1 lines 6-7

### 6.5 Steps 8-17: Merge Loop

> **REQ-P1-025**: *Algorithm 1 Steps 8-17 -- Merge Loop* -- The `run_phase1` function shall iterate over the remaining sorted candidates (i = 2 to M, sorted by score descending) and for each:
>
> 1. Invoke `merge_solutions(s_0, s_init^{pi(i)}, config)` (REQ-P1-017) to produce `s_candidate`.
> 2. Run the leakage checker: `s_candidate = check_and_fix_leakage(s_candidate)` (REQ-SF-022).
> 3. Evaluate `s_candidate` using `evaluate_with_retry(s_candidate, task, config, debug_callback)` (REQ-EX-021).
> 4. Compare the score using `is_improvement_or_equal(h(s_candidate), h_best, direction)` (REQ-DM-029).
> 5. If the merged candidate improves or equals the best score:
>    - Update `s_0 = s_candidate` and `h_best = h(s_candidate)`.
>    - Continue to the next candidate.
> 6. If the merged candidate does not improve:
>    - **Break** out of the merge loop (do not attempt further merges).
>
> - This implements Algorithm 1 lines 8-17 with the break-on-first-failure semantics of line 15.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given sorted candidates with scores `[0.9, 0.85, 0.8]` where merging s_0 with the second produces score 0.91 and merging with the third produces score 0.89, the function shall stop after the third merge (since 0.89 < 0.91) and return `s_0` with `h_best = 0.91`.
> - Source: REF-01 Algorithm 1 lines 8-17

> **REQ-P1-026**: *Merge Loop Score Comparison Semantics* -- The merge loop score comparison shall use `is_improvement_or_equal()` (REQ-DM-029), not strict improvement (`is_improvement`, REQ-DM-028). This means a merge that achieves the same score as the current best is accepted and merging continues.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: If `h_best = 0.85` and the merged candidate scores `0.85`, the merge shall be accepted (s_0 updated) and the loop shall continue.
> - Source: REF-01 Algorithm 1 line 11 -- `if h(s_candidate) >= h_best then`

> **REQ-P1-027**: *Merge Loop Break-on-First-Failure* -- The merge loop shall terminate immediately upon the first merge that fails to improve or equal the best score. No further merge attempts shall be made after this point.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given 3 remaining candidates to merge, if the first merge succeeds but the second fails, only 2 merges shall be attempted (not 3).
> - Source: REF-01 Algorithm 1 lines 14-16 -- `else: break`

> **REQ-P1-028**: *Merge Candidate Execution Failure* -- If a merged candidate `s_candidate` fails to execute (after debug retries), the system shall treat this as a failed merge (score did not improve) and break out of the merge loop, consistent with break-on-first-failure semantics.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: If the merge produces a script that cannot execute even after debugging, the merge loop shall terminate and `s_0` shall remain unchanged.

### 6.6 Single Candidate Case

> **REQ-P1-029**: *Single Candidate -- No Merge Required* -- If only one candidate has a non-None score after evaluation (either because M=1 or because all other candidates failed), the merge loop shall be skipped entirely. The single successful candidate becomes `s_0` directly.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given M=4 where only one candidate succeeds, `s_0` shall be that candidate's solution with no merge attempts.

---

## 7. Post-Merge Safety Requirements

### 7.1 A_data Invocation

> **REQ-P1-030**: *Post-Merge Data Check* -- After the merge loop completes and `s_0` is finalized, the `run_phase1` function shall invoke `check_data_usage(s_0, task)` (REQ-SF-030) to ensure all provided data sources are utilized.
>
> - The returned solution shall replace `s_0`: `s_0 = check_data_usage(s_0, task)`.
> - If the data agent modifies the solution, the modified solution shall be re-evaluated using `evaluate_with_retry()` to obtain an updated score. If the modified solution fails to execute after debug retries, the system shall fall back to the pre-A_data version of `s_0` (REQ-SF-008 fallback semantics).
> - This corresponds to the paper's `s_0 <- A_data(s_0, T_task)` step that occurs after Algorithm 1.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After merging, `check_data_usage` shall be called exactly once with the merged `s_0` and the task description.
> - Source: REF-01 Section 3.4, REQ-SF-030 -- "A_data runs once after initial solution generation (Phase 1 only)"

### 7.2 A_leakage Invocation

> **REQ-P1-031**: *Post-Data-Check Leakage Check* -- After the A_data check (REQ-P1-030), the `run_phase1` function shall invoke `check_and_fix_leakage(s_0)` (REQ-SF-020) on the final solution.
>
> - The returned solution shall replace `s_0`: `s_0 = check_and_fix_leakage(s_0)`.
> - If the leakage agent modifies the solution, the modified solution shall be re-evaluated using `evaluate_with_retry()` to obtain an updated score.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After the A_data check, `check_and_fix_leakage` shall be called on the resulting solution.
> - Source: REF-01 Section 3.4, REQ-SF-022 -- "A_leakage runs before every evaluation"

### 7.3 Phase1Result Construction

> **REQ-P1-032**: *Phase1Result Construction* -- After all post-merge safety checks complete, the `run_phase1` function shall construct a `Phase1Result` (REQ-DM-022) with:
>
> | Field | Value |
> |-------|-------|
> | `retrieved_models` | The list of `RetrievedModel` instances returned by A_retriever |
> | `candidate_solutions` | The list of `SolutionScript` instances produced by A_init (all M, including failed ones) |
> | `candidate_scores` | The list of scores (as `float | None`) for each candidate, in the same order as `candidate_solutions` |
> | `initial_solution` | The final `s_0` after merging and safety checks |
> | `initial_score` | The score of the final `s_0` (`h_best` after all post-merge steps) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: The returned `Phase1Result` shall have `len(candidate_solutions) == len(candidate_scores) == len(retrieved_models)` (or fewer if retrieval returned fewer than M).
> - Source: REQ-DM-022

> **REQ-P1-033**: *Phase1Result Score Consistency* -- The `Phase1Result.initial_score` shall reflect the score of `Phase1Result.initial_solution` after all post-merge safety checks. If post-merge safety checks altered the solution and re-evaluation produced a different score, `initial_score` shall reflect the final re-evaluated score.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `Phase1Result.initial_score` shall equal the score obtained from evaluating `Phase1Result.initial_solution`.
