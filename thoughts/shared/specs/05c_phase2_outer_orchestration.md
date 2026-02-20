# SRS 05 — Phase 2 Outer Loop: Orchestration and Control Flow

---

## 6. Outer Loop Control Flow Requirements

### 6.1 Outer Loop Function

> **REQ-P2O-019**: *Phase 2 Outer Loop Function* -- The system shall define an async function `run_phase2_outer_loop(initial_solution: SolutionScript, initial_score: float, task: TaskDescription, config: PipelineConfig) -> Phase2Result` that implements Algorithm 2 (outer loop portion):
>
> ```
> Input: initial_solution s_0, initial_score h_best, task, config
> 1. s_final <- s_0
> 2. h_best <- initial_score
> 3. T_abl <- []          (accumulated ablation summaries)
> 4. C <- []              (accumulated refined code blocks)
> 5. step_history <- []
> 6. for t = 0 to config.outer_loop_steps - 1:
> 7.     a_t = invoke A_abl(s_t, T_abl)
> 8.     r_t = execute a_t (with debug retry on error)
> 9.     T_abl_t = invoke A_summarize(a_t, r_t)
> 10.    c_t, p_0 = invoke A_extractor(T_abl_t, s_t, C)
> 11.    validate c_t is exact substring of s_t
> 12.    [hand off to inner loop: Spec 06]
> 13.    [receive back: best_solution_from_inner, best_score_from_inner, inner_history]
> 14.    if best_score_from_inner >= h_best:
> 15.        s_final <- best_solution_from_inner
> 16.        h_best <- best_score_from_inner
> 17.        s_t <- best_solution_from_inner
> 18.    T_abl.append(T_abl_t)
> 19.    C.append(c_t)
> 20.    step_history.append({outer_step, inner_history})
> 21. end for
> 22. return Phase2Result(T_abl, C, s_final, h_best, step_history)
> ```
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given `config.outer_loop_steps = 2`, the function shall execute exactly 2 iterations of the outer loop.
> - Source: REF-01 Algorithm 2 lines 1-28

### 6.2 Ablation Script Execution

> **REQ-P2O-020**: *Ablation Script Execution* -- The outer loop shall execute the ablation study script a_t through the execution harness by:
>
> 1. Writing the ablation script content to a temporary file using `write_script()` (REQ-EX-005).
> 2. Executing it using `execute_script()` (REQ-EX-007) with the working directory set to the task's data directory.
> 3. Capturing full stdout and stderr from the execution.
>
> - The ablation script is not a solution script and shall not be evaluated for a validation score. It produces informational output about component impact.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 2 line 6 -- `r_t = exec(a_t)`

> **REQ-P2O-021**: *Ablation Script Error Handling* -- If the ablation script a_t produces an execution error (non-zero exit code or traceback in stderr), the system shall:
>
> 1. Invoke A_debugger (REQ-SF-006) to fix the ablation script, passing the ablation `SolutionScript` and the extracted traceback.
> 2. Re-execute the fixed script.
> 3. Repeat up to `config.max_debug_attempts` times (REQ-DM-001).
> 4. If all debug attempts are exhausted and the ablation script still fails, skip the ablation for this outer step: set T_abl^t to an empty string indicating "Ablation study failed for this step", and proceed to A_extractor with the empty summary. A_extractor shall rely on the accumulated previous summaries and the solution itself.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given an ablation script that errors once and succeeds after one debug attempt, the system shall produce a valid raw output r_t from the fixed script.
> - Acceptance: Given an ablation script that errors on all debug attempts, the system shall proceed to A_extractor with an empty summary and not raise an exception.
> - Source: REF-01 Section 3.4 -- A_debugger applies to all generated code

### 6.3 Outer Loop State Management

> **REQ-P2O-022**: *Ablation Summary Accumulation* -- The outer loop shall maintain a list `T_abl: list[str]` that accumulates ablation summaries across iterations. After each outer step t, the summary T_abl^t shall be appended to this list.
>
> - At outer step t, the input to A_abl is `T_abl[0:t]` (all summaries from prior steps).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After 3 outer steps, `len(T_abl)` shall equal 3.
> - Source: REF-01 Algorithm 2 line 26 -- `T_abl <- T_abl + T_abl^t`

> **REQ-P2O-023**: *Refined Code Block Accumulation* -- The outer loop shall maintain a list `C: list[CodeBlock]` that accumulates the code blocks targeted for refinement across iterations. After each outer step t, the code block c_t (wrapped in a `CodeBlock` model, REQ-DM-012) shall be appended to this list.
>
> - At outer step t, the input to A_extractor is `[cb.content for cb in C[0:t]]` (content strings of all previously refined blocks).
> - The `CodeBlock.outer_step` field shall be set to `t` for the block refined at step t.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After 3 outer steps, `len(C)` shall equal 3 and `C[2].outer_step` shall equal 2.
> - Source: REF-01 Algorithm 2 line 27 -- `C <- C + c_t`

> **REQ-P2O-024**: *Current Solution Tracking* -- The outer loop shall maintain a reference `s_t: SolutionScript` to the current best solution. At the start of the outer loop, `s_t` shall be set to `initial_solution` (s_0). After each inner loop completes, `s_t` shall be updated to the best solution returned by the inner loop if it improves upon `h_best`.
>
> - The comparison shall use `is_improvement_or_equal()` (REQ-DM-029) with the task's `metric_direction`.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: If the inner loop produces a solution with score 0.85 and `h_best` is 0.82 (maximize), `s_t` shall be updated to the new solution.
> - Source: REF-01 Algorithm 2 lines 12-14 (within inner loop), extended to outer loop tracking

### 6.4 Outer Loop Iteration

> **REQ-P2O-025**: *Outer Loop Iteration Count* -- The outer loop shall execute exactly `config.outer_loop_steps` (T) iterations (REQ-DM-001, default: 4).
>
> - If an iteration encounters an unrecoverable error (e.g., ablation fails and extractor also fails), the iteration shall be skipped but the loop shall continue to the next iteration.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given `config.outer_loop_steps = 4`, the outer loop shall attempt exactly 4 iterations.
> - Source: REF-01 Algorithm 2 line 4 -- `for t = 0 to T-1 do`

### 6.5 Handoff to Inner Loop

> **REQ-P2O-026**: *Inner Loop Handoff* -- At each outer step t, after obtaining c_t and p_0 from A_extractor, the outer loop shall invoke the inner loop (defined in Spec 06) by calling an async function with the following signature:
>
> ```python
> async def run_phase2_inner_loop(
>     solution: SolutionScript,        # current best solution s_t
>     code_block: CodeBlock,           # target code block c_t
>     initial_plan: str,               # initial refinement plan p_0
>     best_score: float,               # current h_best
>     task: TaskDescription,
>     config: PipelineConfig,
> ) -> InnerLoopResult
> ```
>
> - The inner loop function is defined in Spec 06; this spec defines only the call site and the data passed.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Algorithm 2 lines 9-25 (inner loop)

> **REQ-P2O-027**: *Post-Inner-Loop Update* -- After the inner loop returns, the outer loop shall:
>
> 1. Read the best solution and best score from the inner loop result.
> 2. If the inner loop's best score is an improvement or equal to the current `h_best` (using `is_improvement_or_equal()`, REQ-DM-029):
>    a. Update `s_final` to the inner loop's best solution.
>    b. Update `h_best` to the inner loop's best score.
>    c. Update `s_t` to the inner loop's best solution (for use in the next outer step).
> 3. Record the inner loop's step history (list of `RefinementAttempt`, REQ-DM-042) in the outer step record.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: If the inner loop returns a solution with score 0.90 and `h_best` is 0.85 (maximize), `s_final` and `h_best` shall be updated.
> - Acceptance: If the inner loop returns a solution with score 0.80 and `h_best` is 0.85 (maximize), `s_final` and `h_best` shall remain unchanged.
> - Source: REF-01 Algorithm 2 lines 12-14 (best tracking) and lines 26-27 (accumulation)

### 6.6 Phase2Result Construction

> **REQ-P2O-028**: *Phase2Result Construction* -- Upon completion of the outer loop, the system shall construct a `Phase2Result` (REQ-DM-023) with the following field mappings:
>
> | Phase2Result Field | Source |
> |-------------------|--------|
> | `ablation_summaries` | `T_abl` (accumulated list of summary strings) |
> | `refined_blocks` | `C` (accumulated list of `CodeBlock` objects) |
> | `best_solution` | `s_final` (best solution found across all iterations) |
> | `best_score` | `h_best` (score of the best solution) |
> | `step_history` | List of per-step records, each containing: outer step index, ablation summary, code block extracted, inner loop history (list of `RefinementAttempt`), best score after this step |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: The returned `Phase2Result.best_score` shall be >= `initial_score` (for maximize direction) or <= `initial_score` (for minimize direction).
> - Acceptance: `len(Phase2Result.ablation_summaries)` shall equal `config.outer_loop_steps` (or fewer if iterations were skipped).
> - Source: REF-01 Algorithm 2 line 29 -- `Output: final solution s_final`

> **REQ-P2O-029**: *Phase2Result Preserves Initial Score on No Improvement* -- If no outer loop iteration produces an improvement, the `Phase2Result.best_solution` shall be the initial solution s_0 and `Phase2Result.best_score` shall be the initial score h_best. The system shall never return a Phase2Result with a worse score than the initial.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given 4 outer loop iterations where all inner loops fail to improve, `Phase2Result.best_solution.content` shall equal `initial_solution.content`.
> - Source: REF-01 Algorithm 2 lines 1-2 -- `s_final <- s_0`, `h_best <- h(s_0)`

### 6.7 Step History Recording

> **REQ-P2O-030**: *Outer Step History Record* -- Each outer step shall produce a history record (stored in `step_history`) containing:
>
> | Field | Type | Description |
> |-------|------|-------------|
> | `outer_step` | `int` | Outer loop step index (0-based) |
> | `ablation_summary` | `str` | T_abl^t summary text |
> | `code_block` | `str` | Extracted code block content (c_t) |
> | `plan` | `str` | Initial refinement plan (p_0) |
> | `inner_loop_attempts` | `list[RefinementAttempt]` | Inner loop refinement history (REQ-DM-042) |
> | `best_score_after_step` | `float` | Value of h_best after this step completed |
> | `was_skipped` | `bool` | Whether the step was skipped due to errors |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After the outer loop completes, `len(step_history)` shall equal `config.outer_loop_steps`.
> - Source: REF-01 Algorithm 2 (full loop structure)
