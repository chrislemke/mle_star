# SRS 07 — Phase 3: Non-Functional Requirements & Constraints

---

## 6. Non-Functional Requirements

### 6.1 Performance

> **REQ-P3-036**: *Phase 3 Overhead Budget* -- The Phase 3 orchestration overhead (prompt rendering, output parsing, score comparison, Phase3Result construction) excluding agent LLM calls and script execution time shall not exceed 5 seconds total across all R rounds.
>
> - Priority: Should | Verify: Test | Release: MVP

> **REQ-P3-037**: *Phase 3 Total Duration* -- A single Phase 3 execution (R=5 rounds) shall complete within 60 minutes under normal conditions, excluding script execution time. The dominant cost is 2R LLM calls (R ens_planner calls + R ensembler calls, plus leakage and debug calls).
>
> - Priority: Should | Verify: Demonstration | Release: MVP
> - Source: REF-01 Section 4 -- 24-hour total budget; Phase 3 is one of 3 phases

### 6.2 Reliability

> **REQ-P3-038**: *Phase 3 Never Raises on Round Failure* -- The `run_phase3` function shall not raise exceptions due to individual round failures (ens_planner empty, ensembler unparseable, evaluation error). Each such failure shall be handled gracefully per REQ-P3-029 through REQ-P3-031, and the function shall always return a valid `Phase3Result`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given R=5 where all 5 rounds fail, the function shall return a Phase3Result with the best input solution as fallback (REQ-P3-026).

### 6.3 Observability

> **REQ-P3-039**: *Phase 3 Logging* -- The Phase 3 orchestration shall log the following events using Python's `logging` module at the specified levels:
>
> | Event | Level | Content |
> |-------|-------|---------|
> | Phase 3 start | `INFO` | L value, R value, competition ID |
> | Phase 3 skipped (L=1) | `INFO` | Single solution score, competition ID |
> | Ensemble round start | `INFO` | Round index r, number of previous plans in history |
> | A_ens_planner invocation start | `INFO` | Round r, history size |
> | A_ens_planner invocation complete | `INFO` | Round r, plan text (first 200 chars) |
> | A_ens_planner empty response | `WARNING` | Round r |
> | A_ensembler invocation start | `INFO` | Round r, plan text (first 200 chars) |
> | A_ensembler invocation complete | `INFO` | Round r, script length (or "failed to parse") |
> | A_ensembler extraction failure | `WARNING` | Round r, response summary (first 200 chars) |
> | Leakage check start | `INFO` | Round r, solution content length |
> | Leakage check complete | `INFO` | Round r, leakage found (yes/no), content changed (yes/no) |
> | Evaluation start | `INFO` | Round r, solution content length |
> | Evaluation complete | `INFO` | Round r, score (or "failed"), is_error, duration |
> | Round failed (execution error) | `WARNING` | Round r, error summary, plan summary |
> | Best selection | `INFO` | Best round r*, best score, total successful rounds |
> | All rounds failed (fallback) | `WARNING` | R value, fallback solution score |
> | Phase 3 complete | `INFO` | Best score, best round, total duration, rounds attempted |
>
> - Priority: Must | Verify: Inspection | Release: MVP

---

## 7. Constraints

### 7.1 Technology Constraints

> **REQ-P3-040**: *SDK Agent Invocation* -- Both Phase 3 agents (A_ens_planner and A_ensembler) shall be invoked via the Claude Agent SDK agent mechanism. They shall not use direct API calls, raw HTTP requests, or any non-SDK LLM invocation method.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-02 -- all agent interactions through the SDK

> **REQ-P3-041**: *Single Module Organization* -- All Phase 3 functions defined in this spec shall reside in a single Python module (e.g., `mle_star/phase3.py`).
>
> - Priority: Should | Verify: Inspection | Release: MVP

### 7.2 Algorithm Fidelity Constraints

> **REQ-P3-042**: *Algorithm 3 Fidelity* -- The Phase 3 implementation shall faithfully reproduce Algorithm 3 from REF-01 Appendix B. Specifically:
>
> 1. The initial ensemble round (r=0) shall invoke A_ens_planner with no history (line 1).
> 2. Each round shall invoke A_ensembler with the plan and all L solutions (lines 2, 6).
> 3. Each ensemble solution shall be evaluated (lines 3, 7).
> 4. Subsequent rounds (r=1..R-1) shall pass full history to A_ens_planner (line 5).
> 5. The best ensemble shall be selected as `argmax` (or `argmin` per direction) over all R rounds (line 9).
> 6. The output shall be the single best ensemble solution s_ens* (line 10).
>
> - Deviations from Algorithm 3 are permitted only for error handling (e.g., skipping failed rounds, debugging), which the paper does not address explicitly.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Appendix B, Algorithm 3

> **REQ-P3-043**: *Sequential Ensemble Rounds* -- The ensemble rounds shall execute sequentially, not concurrently. Each round depends on the accumulated history from all prior rounds (plans and scores).
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Rationale: A_ens_planner at round r requires the scores from rounds 0 through r-1; parallel execution would prevent this.
> - Source: REF-01 Algorithm 3 lines 4-8 -- sequential loop with history accumulation

> **REQ-P3-044**: *Ensemble Iteration Count* -- The ensemble loop shall attempt exactly `config.ensemble_rounds` (R) rounds (REQ-DM-001, default: 5). Failed rounds count as iterations; the loop does not add extra rounds to compensate for failures.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given `config.ensemble_rounds = 5`, the function shall produce exactly 5 `EnsembleAttempt` records regardless of success or failure.
> - Source: REF-01 Algorithm 3 -- `for r = 1 to R-1 do` (R-1 iterations after the initial round, R total)

> **REQ-P3-045**: *Leakage Check Integration Points* -- Within Phase 3, the leakage checker `check_and_fix_leakage()` (REQ-SF-022) shall be invoked on every ensemble solution s_ens^r before evaluation. This ensures every solution that enters evaluation has been checked for data leakage.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Section 3.4, REQ-SF-022

### 7.3 Prompt Fidelity Constraints

> **REQ-P3-046**: *A_ens_planner Prompt Fidelity* -- The A_ens_planner prompt (REQ-P3-002) shall preserve the semantic intent of Figure 17 from the paper. The prompt shall include:
>
> 1. The Kaggle grandmaster persona introduction.
> 2. All L solution scripts presented in full.
> 3. The history of previous plans and scores (when available).
> 4. The instruction to concentrate on merging strategy, not hyperparameters.
> 5. The instruction that the plan must be easy to implement, novel, and effective.
> 6. The instruction that the plan should differ from previous plans.
> 7. The instruction that the plan should not modify original solutions too much.
> 8. The response format instruction (natural language outline, no additional headings).
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 17

> **REQ-P3-047**: *A_ensembler Prompt Fidelity* -- The A_ensembler prompt (REQ-P3-011) shall preserve the semantic intent of Figure 18 from the paper. The prompt shall include:
>
> 1. The Kaggle grandmaster persona introduction.
> 2. All L solution scripts presented in full.
> 3. The ensemble plan presented in full.
> 4. The instruction to implement the ensemble plan.
> 5. The instruction not to modify original solutions too much (unless the plan says to).
> 6. The `./input/` data directory instruction and no-unzip instruction.
> 7. The no-load-submissions instruction.
> 8. The no-subsample and no-dummy-variables instruction.
> 9. The `./final/submission.csv` file requirement.
> 10. The "Final Validation Performance" output requirement.
> 11. The single code block response format.
> 12. The self-contained single-file constraint.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 18

### 7.4 Ensemble Script Constraints

> **REQ-P3-048**: *Ensemble Scripts Are Full Programs* -- Ensemble scripts produced by A_ensembler are complete, self-contained, single-file Python programs (not code block replacements). They include all imports, data loading, model training, prediction, validation metric computation, and submission file generation. This distinguishes them from Phase 2 refinement outputs (which are code block replacements within an existing solution).
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Rationale: Ensemble scripts combine L solutions into a new program, rather than modifying one solution's code block.
> - Source: REF-01 Figure 18 -- "The code should be a single-file Python program that is self-contained and can be executed as-is."

> **REQ-P3-049**: *Validation Performance Output Requirement* -- Every ensemble script produced by A_ensembler shall print the evaluation metric in the format `"Final Validation Performance: {score}"` so that the execution harness (REQ-DM-027) can parse the score from stdout.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Acceptance: The prompt shall include the instruction to print "Final Validation Performance: {final_validation_score}".
> - Source: REF-01 Figure 18; REQ-DM-027 (score parsing regex)
