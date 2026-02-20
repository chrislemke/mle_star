# SRS 06 — Phase 2 Inner Loop: Constraints

---

## 6. Non-Functional Requirements

### 6.1 Performance

> **REQ-P2I-039**: *Agent Response Overhead* -- The system overhead for processing each agent response (prompt rendering, SDK invocation setup, code block extraction, and `SolutionScript` construction) shall not exceed 500 milliseconds per invocation, excluding LLM call latency and script execution time.
>
> - Priority: Should | Verify: Test | Release: MVP

> **REQ-P2I-040**: *Inner Loop Total Duration* -- A single inner loop execution (K=4 attempts) shall complete within 30 minutes under normal conditions, excluding script execution time. The dominant cost is K LLM calls (K coder calls + up to K-1 planner calls = up to 2K-1 LLM calls total, plus leakage and debug calls).
>
> - Priority: Should | Verify: Demonstration | Release: MVP
> - Source: REF-01 Section 4 -- 24-hour total budget; Phase 2 is one of 3 phases across T outer steps

### 6.2 Reliability

> **REQ-P2I-041**: *Inner Loop Never Raises on Agent Failure* -- The `run_phase2_inner_loop` function shall not raise exceptions due to individual agent failures (A_coder unparseable, A_planner empty, replacement failure, or evaluation error). Each such failure shall be handled gracefully per REQ-P2I-032 through REQ-P2I-034, and the inner loop shall always return a valid `InnerLoopResult`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given K=4 where all 4 attempts fail (coder returns garbage), the function shall return an `InnerLoopResult` with `improved=False` and `attempts` of length 4.

> **REQ-P2I-042**: *Progressive Improvement Expectation* -- While not guaranteed, the inner loop is designed such that successive A_planner iterations produce progressively better plans, as evidenced by the paper's experimental results showing monotonically improving error reduction across inner steps (Figure 8: Step 0: 0%, Step 1: ~12.6%, Step 2: ~17.7%, Step 3: ~20.8%, Step 4: ~22.3% error reduction).
>
> This requirement documents the empirical expectation; no automated enforcement is needed.
>
> - Priority: Informational | Verify: N/A | Release: N/A
> - Source: REF-01 Figure 8

### 6.3 Observability

> **REQ-P2I-043**: *Inner Loop Logging* -- Each inner loop execution shall log the following events using Python's `logging` module at the specified levels:
>
> | Event | Level | Content |
> |-------|-------|---------|
> | Inner loop start | `INFO` | Code block length, initial plan (first 200 chars), input h_best, K value |
> | A_coder invocation start | `INFO` | Inner step k, plan text (first 200 chars) |
> | A_coder invocation complete | `INFO` | Inner step k, output code block length (or "failed to parse") |
> | A_coder unparseable response | `WARNING` | Inner step k, response summary (first 200 chars) |
> | A_planner invocation start | `INFO` | Inner step k, number of previous attempts in history |
> | A_planner invocation complete | `INFO` | Inner step k, plan text (first 200 chars) |
> | A_planner empty response | `WARNING` | Inner step k |
> | Code block replacement success | `DEBUG` | Inner step k, original block length, new block length |
> | Code block replacement failure | `WARNING` | Inner step k, error message |
> | Leakage check start | `INFO` | Inner step k, solution content length |
> | Leakage check complete | `INFO` | Inner step k, leakage found (yes/no), content changed (yes/no) |
> | Evaluation start | `INFO` | Inner step k, solution content length |
> | Evaluation complete | `INFO` | Inner step k, score (or "failed"), is_error, duration |
> | Best score updated | `INFO` | Inner step k, old best score, new best score |
> | Attempt skipped | `WARNING` | Inner step k, reason (coder failed / replacement failed / planner failed) |
> | Inner loop complete | `INFO` | Total attempts, successful evaluations, best score, improved (yes/no) |
>
> - Priority: Must | Verify: Inspection | Release: MVP

---

## 7. Constraints

### 7.1 Technology Constraints

> **REQ-P2I-044**: *SDK Agent Invocation* -- Both agents (A_coder and A_planner) shall be invoked via the Claude Agent SDK agent mechanism. They shall not use direct API calls, raw HTTP requests, or any non-SDK LLM invocation method.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-02 -- all agent interactions through the SDK

> **REQ-P2I-045**: *Single Module Organization* -- All inner loop functions and agent invocation logic defined in this spec shall reside in a single Python module (e.g., `mle_star/phase2_inner.py`).
>
> - Priority: Should | Verify: Inspection | Release: MVP

### 7.2 Algorithmic Constraints

> **REQ-P2I-046**: *Sequential Inner Loop Execution* -- The inner loop iterations shall execute sequentially, not concurrently. Each iteration depends on the accumulated history from all prior iterations (plans and scores).
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Rationale: A_planner at step k requires the scores from steps 0 through k-1; parallel execution would prevent this.
> - Source: REF-01 Algorithm 2 lines 16-25 -- sequential loop with history accumulation

> **REQ-P2I-047**: *Monotonic Best Score* -- The `local_best_score` tracked within the inner loop shall be monotonically non-decreasing (for maximize) or non-increasing (for minimize). The system shall never overwrite `local_best_solution` with a solution that has a worse score than the current `local_best_score`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After every inner step, `local_best_score` shall be >= (maximize) or <= (minimize) the value at the start of that step.
> - Source: REF-01 Algorithm 2 lines 12-14, 21-24 -- conditional update on `>=`

> **REQ-P2I-048**: *Inner Loop Iteration Count* -- The inner loop shall attempt exactly `config.inner_loop_steps` (K) iterations (REQ-DM-001, default: 4). Failed attempts count as iterations; the loop does not add extra iterations to compensate for failures.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given `config.inner_loop_steps = 4`, the inner loop shall produce exactly 4 `RefinementAttempt` records regardless of success or failure.
> - Source: REF-01 Algorithm 2 -- `for k = 1 to K-1 do` (K-1 iterations after the initial attempt, K total)

### 7.3 Data Integrity Constraints

> **REQ-P2I-049**: *Immutable Input Solution* -- The `run_phase2_inner_loop` function shall not mutate the `solution` parameter. All modifications to the solution shall produce new `SolutionScript` instances via `replace_block`. The original `solution` shall remain available as the replacement base for all K attempts.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Rationale: Each inner step replaces c_t in the original s_t, not in the result of a previous step.

> **REQ-P2I-050**: *Immutable Code Block* -- The `code_block.content` parameter shall not be modified during the inner loop. Every invocation of A_coder and every `replace_block` call shall use the original `code_block.content` as provided by the outer loop.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Rationale: Algorithm 2 uses c_t (the original extraction) consistently across all K inner steps.

---

## 8. Traceability Matrix

### 8.1 Requirements to Paper Sections

| Req ID | Paper Section | Paper Element | SDK Construct |
|--------|--------------|---------------|---------------|
| REQ-P2I-001 | Section 3.2 | A_coder agent | `AgentDefinition` |
| REQ-P2I-002 | Figure 15 | Coder prompt template | `prompt` parameter |
| REQ-P2I-003 | Alg 2 lines 9, 18 | `c_t^k = A_coder(c_t, p_k)` input | -- |
| REQ-P2I-004 | Figure 15 | `c_t^k = A_coder(c_t, p_k)` output | -- |
| REQ-P2I-005 | Figure 15 | Coder invocation function | -- |
| REQ-P2I-006 | Figure 15 | "Do not remove subsampling" | -- |
| REQ-P2I-007 | Figure 15 | "Do not introduce dummy variables" | -- |
| REQ-P2I-008 | Section 3.2 | A_planner agent | `AgentDefinition` |
| REQ-P2I-009 | Figure 16 | Planner prompt template | `prompt` parameter |
| REQ-P2I-010 | Figure 16 | History formatting | -- |
| REQ-P2I-011 | Alg 2 line 17 | `p_k = A_planner(c_t, history)` input | -- |
| REQ-P2I-012 | Figure 16 | `p_k = A_planner(c_t, history)` output | -- |
| REQ-P2I-013 | Figure 16 | Planner invocation function | -- |
| REQ-P2I-014 | Figure 16 | "novel and effective" / "differ from previous" | -- |
| REQ-P2I-015 | Figure 16 | "avoid plans which can make running time too long" | -- |
| REQ-P2I-016 | Alg 2 lines 9-25 | Inner loop function signature | -- |
| REQ-P2I-017 | Alg 2 lines 9-15 | Initial attempt (k=0) | -- |
| REQ-P2I-018 | Alg 2 line 9 | p_0 from A_extractor, not A_planner | -- |
| REQ-P2I-019 | Alg 2 lines 16-25 | Subsequent attempts (k=1..K-1) | -- |
| REQ-P2I-020 | Alg 2 line 17 | Full history to A_planner | -- |
| REQ-P2I-021 | Alg 2 line 18 | A_coder always receives original c_t | -- |
| REQ-P2I-022 | Alg 2 lines 10, 19 | `s_t^k = s_t.replace(c_t, c_t^k)` | -- |
| REQ-P2I-023 | Alg 2 lines 10, 19 | Replace against original s_t | -- |
| REQ-P2I-024 | Alg 2 lines 1-2 | Best score initialization | -- |
| REQ-P2I-025 | Alg 2 lines 12-14, 21-24 | Best score update | -- |
| REQ-P2I-026 | Alg 2 lines 12, 21 | `>=` comparison semantics | -- |
| REQ-P2I-027 | -- | Failed evaluation score handling | -- |
| REQ-P2I-028 | Alg 2 | RefinementAttempt record | REQ-DM-042 |
| REQ-P2I-029 | Alg 2 | History ordering | -- |
| REQ-P2I-030 | Section 3.4 | Leakage check before evaluation | REQ-SF-022 |
| REQ-P2I-031 | Section 3.4 | A_debugger on execution error | REQ-SF-006, REQ-SF-007 |
| REQ-P2I-032 | -- | Unparseable coder output handling | -- |
| REQ-P2I-033 | -- | Replacement failure handling | -- |
| REQ-P2I-034 | -- | Planner failure handling | -- |
| REQ-P2I-035 | Alg 2 line 17 | Failed attempts in history | -- |
| REQ-P2I-036 | Alg 2 lines 9-25 | InnerLoopResult model | -- |
| REQ-P2I-037 | Alg 2 lines 9-25 | InnerLoopResult construction | -- |
| REQ-P2I-038 | Alg 2 lines 1-2 | Preserve input on no improvement | -- |
| REQ-P2I-039 | -- | Agent response overhead | -- |
| REQ-P2I-040 | Section 4 | Inner loop duration | -- |
| REQ-P2I-041 | -- | Never raises on agent failure | -- |
| REQ-P2I-042 | Figure 8 | Progressive improvement evidence | -- |
| REQ-P2I-043 | -- | Logging | Python `logging` |
| REQ-P2I-044 | -- | SDK-only invocation | `claude-agent-sdk` |
| REQ-P2I-045 | -- | Module organization | -- |
| REQ-P2I-046 | Alg 2 lines 16-25 | Sequential execution | -- |
| REQ-P2I-047 | Alg 2 lines 12-14, 21-24 | Monotonic best score | -- |
| REQ-P2I-048 | Alg 2 lines 9, 16 | Iteration count = K | -- |
| REQ-P2I-049 | -- | Immutable input solution | -- |
| REQ-P2I-050 | -- | Immutable code block | -- |

### 8.2 Cross-References to Other Specs

| Req ID | Referenced By |
|--------|--------------|
| REQ-P2I-001 (A_coder agent def) | Spec 09 (orchestrator configures agents) |
| REQ-P2I-008 (A_planner agent def) | Spec 09 (orchestrator configures agents) |
| REQ-P2I-016 (inner loop function) | Spec 05 (outer loop invokes inner loop, REQ-P2O-026) |
| REQ-P2I-036 (InnerLoopResult) | Spec 05 (outer loop receives inner loop result, REQ-P2O-027) |
| REQ-P2I-037 (InnerLoopResult construction) | Spec 05 (outer loop post-inner-loop update, REQ-P2O-027) |

### 8.3 Spec 01 Dependencies (Inbound)

| Spec 01 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-DM-001 (PipelineConfig) | REQ-P2I-016, 048 | `inner_loop_steps` (K), `max_debug_attempts` |
| REQ-DM-006 (MetricDirection) | REQ-P2I-025, 026 | Score comparison direction |
| REQ-DM-007 (TaskDescription) | REQ-P2I-016, 025 | Task context, metric direction |
| REQ-DM-009 (SolutionScript) | REQ-P2I-016, 022, 023, 036, 049 | Solution type, input/output |
| REQ-DM-010 (replace_block) | REQ-P2I-017, 019, 022, 023, 033 | Code block replacement in solution |
| REQ-DM-012 (CodeBlock) | REQ-P2I-016, 050 | Code block input type |
| REQ-DM-013 (AgentType) | REQ-P2I-001, 008 | Agent identity enum values |
| REQ-DM-028 (is_improvement) | REQ-P2I-036, 037 | Strict improvement check for `improved` field |
| REQ-DM-029 (is_improvement_or_equal) | REQ-P2I-025, 026 | `>=` score comparison for best update |
| REQ-DM-032 (PromptRegistry) | REQ-P2I-002, 009 | Template retrieval for coder and planner |
| REQ-DM-036 (AgentConfig) | REQ-P2I-001, 008 | Agent-to-SDK mapping |
| REQ-DM-042 (RefinementAttempt) | REQ-P2I-028, 029, 036 | Attempt history records |

### 8.4 Spec 02 Dependencies (Inbound)

| Spec 02 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-EX-015 (evaluate_solution) | REQ-P2I-017, 019 | Evaluate refined solutions |
| REQ-EX-021 (evaluate_with_retry) | REQ-P2I-017, 019, 031 | Evaluate with debug retry |

### 8.5 Spec 03 Dependencies (Inbound)

| Spec 03 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-SF-005 (extract_code_block) | REQ-P2I-004, 005 | Extract code block from A_coder response |
| REQ-SF-006 (debug_solution) | REQ-P2I-031 | Debug execution errors in refined solutions |
| REQ-SF-007 (make_debug_callback) | REQ-P2I-017, 019, 031 | Debug callback for evaluate_with_retry |
| REQ-SF-020 (check_and_fix_leakage) | REQ-P2I-017, 019, 030 | Leakage detection and correction |
| REQ-SF-022 (leakage integration point) | REQ-P2I-030 | Leakage check before every evaluation |

### 8.6 Spec 05 Dependencies (Inbound)

| Spec 05 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-P2O-026 (inner loop handoff) | REQ-P2I-016 | Defines the call site and input data for the inner loop |
| REQ-P2O-027 (post-inner-loop update) | REQ-P2I-036, 037 | Defines what the outer loop expects back from the inner loop |

---

## 9. Change Control

### 9.1 Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2026-02-20 | MLE-STAR Team | Initial draft -- all 50 requirements |

### 9.2 Baselining Policy

This SRS is baselined at version 1.0. After baselining, changes require impact analysis against Spec 05 (outer loop depends on inner loop return type and function signature), Spec 09 (orchestrator invokes Phase 2 via outer loop), Spec 01 (upstream data model dependencies), Spec 02 (upstream execution harness dependencies), and Spec 03 (upstream safety agent dependencies).
