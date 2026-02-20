# SRS 06 — Phase 2 Inner Loop: Orchestration

## 5. Inner Loop Orchestration Requirements

### 5.1 Inner Loop Function Signature

> **REQ-P2I-016**: *Inner Loop Function* -- The system shall define an async function `run_phase2_inner_loop` with the following signature and behavior:
>
> ```python
> async def run_phase2_inner_loop(
>     solution: SolutionScript,        # current best solution s_t
>     code_block: CodeBlock,           # target code block c_t
>     initial_plan: str,               # initial refinement plan p_0
>     best_score: float,               # current h_best from outer loop
>     task: TaskDescription,
>     config: PipelineConfig,
> ) -> InnerLoopResult
> ```
>
> - The function shall implement Algorithm 2 lines 9-25 (the inner loop).
> - The function shall return an `InnerLoopResult` (REQ-P2I-037) containing the best solution, best score, and full history of attempts.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given valid inputs with `config.inner_loop_steps = 4`, the function shall attempt up to 4 refinement strategies and return the best result.
> - Source: REF-01 Algorithm 2 lines 9-25, REQ-P2O-026 (outer loop handoff signature)

### 5.2 Initial Attempt (k=0)

> **REQ-P2I-017**: *Initial Attempt Execution* -- The inner loop shall execute the initial attempt (k=0) as follows:
>
> 1. Invoke `invoke_coder(code_block.content, initial_plan)` to obtain c_t^0.
> 2. If A_coder returns `None` (unparseable output), record a failed attempt and proceed to k=1.
> 3. Construct s_t^0 by calling `solution.replace_block(code_block.content, c_t^0)` (REQ-DM-010).
> 4. If `replace_block` raises `ValueError` (code block not found), record a failed attempt and proceed to k=1.
> 5. Invoke `check_and_fix_leakage(s_t^0)` (REQ-SF-020, REQ-SF-022) before evaluation.
> 6. Evaluate h(s_t^0) using `evaluate_with_retry` (REQ-EX-021) with the debug callback (REQ-SF-007).
> 7. If h(s_t^0) >= h_best (using `is_improvement_or_equal`, REQ-DM-029), update s_final and h_best.
> 8. Record the attempt as a `RefinementAttempt` (REQ-DM-042).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a valid code block and plan, the initial attempt shall produce an evaluated solution and a RefinementAttempt record.
> - Source: REF-01 Algorithm 2 lines 9-15

> **REQ-P2I-018**: *Initial Attempt Uses Extractor Plan* -- The initial attempt (k=0) shall use the `initial_plan` parameter (p_0) directly, without invoking A_planner. A_planner is only invoked for subsequent attempts (k >= 1).
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Rationale: Algorithm 2 specifies `c_t^0 = A_coder(c_t, p_0)` at line 9, where p_0 comes from A_extractor (outer loop), not A_planner.
> - Source: REF-01 Algorithm 2 lines 9 vs. 17

### 5.3 Subsequent Attempts (k=1..K-1)

> **REQ-P2I-019**: *Subsequent Attempt Iteration* -- For each inner step k from 1 to K-1 (where K = `config.inner_loop_steps`), the inner loop shall:
>
> 1. Invoke `invoke_planner(code_block.content, accumulated_plans, accumulated_scores)` to obtain p_k.
> 2. If A_planner returns `None` (empty response), skip this attempt and proceed to k+1.
> 3. Invoke `invoke_coder(code_block.content, p_k)` to obtain c_t^k.
> 4. If A_coder returns `None` (unparseable output), record a failed attempt and proceed to k+1.
> 5. Construct s_t^k by calling `solution.replace_block(code_block.content, c_t^k)` (REQ-DM-010).
> 6. If `replace_block` raises `ValueError`, record a failed attempt and proceed to k+1.
> 7. Invoke `check_and_fix_leakage(s_t^k)` (REQ-SF-020, REQ-SF-022) before evaluation.
> 8. Evaluate h(s_t^k) using `evaluate_with_retry` (REQ-EX-021) with the debug callback (REQ-SF-007).
> 9. If h(s_t^k) >= h_best (using `is_improvement_or_equal`, REQ-DM-029), update s_final and h_best.
> 10. Record the attempt as a `RefinementAttempt` (REQ-DM-042).
> 11. Append p_k to `accumulated_plans` and h(s_t^k) to `accumulated_scores`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given `config.inner_loop_steps = 4`, the loop shall attempt steps k=1, k=2, k=3 (3 iterations after the initial attempt at k=0), invoking A_planner before each.
> - Source: REF-01 Algorithm 2 lines 16-25

> **REQ-P2I-020**: *A_planner Receives Full History* -- At inner step k, the A_planner invocation shall receive the complete history of all previous attempts (j=0 through j=k-1), including:
>
> - All plan texts: `[p_0, p_1, ..., p_{k-1}]`
> - All scores: `[h(s_t^0), h(s_t^1), ..., h(s_t^{k-1})]`
>
> - Scores for failed attempts (execution error, replacement failure, or unparseable coder output) shall be represented as `None` in the scores list.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: At k=2, A_planner shall receive plans and scores for attempts k=0 and k=1.
> - Source: REF-01 Algorithm 2 line 17 -- `p_k = A_planner(c_t, {(p_j, h(s_t^j))}_{j=0}^{k-1})`

> **REQ-P2I-021**: *A_coder Always Receives Original Code Block* -- At every inner step k, A_coder shall receive the original code block c_t (not any previously refined version c_t^j). Each refinement attempt starts from the original code block and applies a new plan.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: At k=2, the `code_block` argument to `invoke_coder` shall be `code_block.content` (the original c_t), not c_t^0 or c_t^1.
> - Source: REF-01 Algorithm 2 line 18 -- `c_t^k = A_coder(c_t, p_k)` (c_t is the original, not c_t^{k-1})
