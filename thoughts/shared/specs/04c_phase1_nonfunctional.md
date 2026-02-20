# SRS 04 — Phase 1: Non-Functional Requirements and Constraints

## 8. Non-Functional Requirements

### 8.1 Parallelization

> **REQ-P1-034**: *Candidate Generation Independence* -- The M candidate generations (A_init invocations in Algorithm 1 lines 2-5) are independent of each other. The system should document this independence to enable future parallel execution.
>
> - The current implementation shall execute candidate generations sequentially (consistent with REQ-EX-026 sequential batch evaluation).
> - A future optimization may execute them concurrently, as no candidate depends on another's output.
> - Priority: Should | Verify: Inspection | Release: MVP
> - Source: REF-01 Algorithm 1 lines 2-5 -- each iteration is independent

> **REQ-P1-035**: *Candidate Evaluation Independence* -- The M candidate evaluations (Algorithm 1 lines 4) are independent of each other and may be executed concurrently in a future optimization.
>
> - The current implementation shall use `evaluate_batch()` (REQ-EX-026) which evaluates sequentially.
> - Priority: Should | Verify: Inspection | Release: MVP

> **REQ-P1-036**: *Merge Loop Sequential Requirement* -- The merge loop (Algorithm 1 lines 8-17) is inherently sequential: each merge depends on the current `s_0` which may have been updated by the previous merge. The merge loop shall not be parallelized.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Algorithm 1 lines 8-17 -- `s_0` is updated in each iteration

### 8.2 Performance

> **REQ-P1-037**: *Phase 1 Overhead Budget* -- The Phase 1 orchestration overhead (prompt rendering, output parsing, score comparison, Phase1Result construction) excluding agent LLM calls and script execution time shall not exceed 5 seconds total.
>
> - Priority: Should | Verify: Test | Release: MVP

### 8.3 Observability

> **REQ-P1-038**: *Phase 1 Logging* -- The Phase 1 orchestration shall log the following events using Python's `logging` module at the specified levels:
>
> | Event | Level | Content |
> |-------|-------|---------|
> | Phase 1 start | `INFO` | Competition ID, M value |
> | Retrieval complete | `INFO` | Number of models retrieved, model names |
> | Candidate generation start | `INFO` | Model index (i/M), model name |
> | Candidate generation complete | `INFO` | Model name, code length |
> | Candidate evaluation result | `INFO` | Model name, score (or "failed"), duration |
> | Candidate skipped (execution failure) | `WARNING` | Model name, error summary |
> | All candidates failed | `ERROR` | M value, all model names |
> | Candidates sorted | `INFO` | Sorted order (model names and scores) |
> | Merge attempt start | `INFO` | Merge index, base score, reference model name |
> | Merge attempt result | `INFO` | Merged score, accepted or rejected |
> | Merge loop break | `INFO` | Reason (score did not improve), merge index |
> | Post-merge A_data start | `INFO` | Solution content length |
> | Post-merge A_data result | `INFO` | Whether solution was modified |
> | Post-merge A_leakage start | `INFO` | Solution content length |
> | Post-merge A_leakage result | `INFO` | Whether leakage was detected and corrected |
> | Phase 1 complete | `INFO` | Final score, total duration, number of merges performed |
>
> - Priority: Must | Verify: Inspection | Release: MVP

---

## 9. Constraints

### 9.1 Technology Constraints

> **REQ-P1-039**: *SDK Agent Invocation* -- All three Phase 1 agents (A_retriever, A_init, A_merger) shall be invoked via the Claude Agent SDK agent mechanism. They shall not use direct API calls, raw HTTP requests, or any non-SDK LLM invocation method.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-02 -- all agent interactions through the SDK

> **REQ-P1-040**: *Single Module Organization* -- All Phase 1 functions defined in this spec shall reside in a single Python module (e.g., `mle_star/phase1.py`).
>
> - Priority: Should | Verify: Inspection | Release: MVP

### 9.2 Algorithm Fidelity Constraints

> **REQ-P1-041**: *Algorithm 1 Fidelity* -- The Phase 1 implementation shall faithfully reproduce Algorithm 1 from REF-01 Appendix B. Specifically:
>
> 1. Retrieval shall occur exactly once (line 1).
> 2. All M candidates shall be generated and evaluated (lines 2-5).
> 3. Candidates shall be sorted by score, best first (line 6).
> 4. The best candidate shall be selected as the initial `s_0` (lines 6-7).
> 5. Merging shall iterate over remaining sorted candidates in descending score order (lines 8-17).
> 6. The merge loop shall use `>=` comparison (line 11), not strict `>`.
> 7. The merge loop shall break on first failure (line 15).
>
> - Deviations from Algorithm 1 are permitted only for error handling (e.g., skipping failed candidates, debugging), which the paper does not address explicitly.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Appendix B, Algorithm 1

> **REQ-P1-042**: *Leakage Check Integration Points* -- Within Phase 1, the leakage checker `check_and_fix_leakage()` (REQ-SF-022) shall be invoked:
>
> 1. On each candidate `s_init^i` before evaluation (REQ-P1-020 step 2).
> 2. On each merged candidate `s_candidate` before evaluation (REQ-P1-025 step 2).
> 3. On the final `s_0` during post-merge safety checks (REQ-P1-031).
>
> - This ensures every solution that enters evaluation has been checked for data leakage.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Section 3.4, REQ-SF-022

### 9.3 Prompt Fidelity Constraints

> **REQ-P1-043**: *A_retriever Prompt Fidelity* -- The A_retriever prompt (REQ-P1-002) shall preserve the semantic intent of Figure 9 from the paper. The prompt shall include:
>
> 1. The competition task description (verbatim from `TaskDescription.description`).
> 2. The instruction to list M recent effective models.
> 3. The requirement that example code be concise and simple.
> 4. The requirement that example code must be provided (no GitHub/paper references only).
> 5. The JSON schema for the response format.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 9

> **REQ-P1-044**: *A_init Prompt Fidelity* -- The A_init prompt (REQ-P1-009) shall preserve the semantic intent of Figure 10 from the paper. The prompt shall include:
>
> 1. The Kaggle grandmaster persona introduction.
> 2. The task description.
> 3. The model name and example code.
> 4. Instructions to implement using the specified model.
> 5. Instruction for simple solution (no ensembling, no hyperparameter optimization).
> 6. The `./input/` data directory instruction.
> 7. The PyTorch-over-TensorFlow preference.
> 8. The 30,000 subsample limit instruction.
> 9. The "Final Validation Performance" output requirement.
> 10. The single code block response format.
> 11. The no-`exit()` constraint.
> 12. The no-try/except constraint.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 10

> **REQ-P1-045**: *A_merger Prompt Fidelity* -- The A_merger prompt (REQ-P1-014) shall preserve the semantic intent of Figure 11 from the paper. The prompt shall include:
>
> 1. The Kaggle grandmaster persona introduction.
> 2. The base solution code.
> 3. The reference solution code.
> 4. Instruction to integrate reference into base.
> 5. Instruction that code base should be the base solution.
> 6. Instruction to train additional model from reference.
> 7. Instruction to keep similar functionality together.
> 8. Instruction to ensemble the models.
> 9. Instruction for simple design.
> 10. The "Final Validation Performance" output requirement.
> 11. The single code block response format.
> 12. The `./input/` data directory instruction.
> 13. The 30,000 subsample limit instruction.
> 14. The no-`exit()` constraint.
> 15. The no-try/except constraint.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 11
