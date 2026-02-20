# SRS 05 — Phase 2 Outer Loop: Constraints and Traceability

---

## 7. Non-Functional Requirements

### 7.1 Performance

> **REQ-P2O-031**: *A_abl Response Overhead* -- The system overhead for processing an A_abl agent response (code block extraction, SolutionScript wrapping) shall not exceed 500 milliseconds, excluding LLM call latency and script execution time.
>
> - Priority: Should | Verify: Test | Release: MVP

> **REQ-P2O-032**: *A_extractor Validation Overhead* -- The code block validation function `validate_code_block()` (REQ-P2O-017) shall execute in under 50 milliseconds for solution scripts up to 50 KB.
>
> - Priority: Should | Verify: Test | Release: MVP

> **REQ-P2O-033**: *Outer Loop Total Duration* -- A single outer loop iteration (A_abl invocation + ablation execution + A_summarize invocation + A_extractor invocation, excluding inner loop time) shall complete within 15 minutes under normal conditions. If ablation script execution exceeds this budget, the execution timeout for ablation scripts shall be set per REQ-P2O-035.
>
> - Priority: Should | Verify: Demonstration | Release: MVP

### 7.2 Reliability

> **REQ-P2O-034**: *A_extractor Graceful Degradation* -- If the extractor agent returns a response that cannot be parsed as `ExtractorOutput` (e.g., malformed JSON, schema validation failure):
>
> 1. Log a warning with the raw response content.
> 2. Re-invoke A_extractor once with the same inputs.
> 3. If the re-invocation also fails parsing, skip this outer loop iteration: do not invoke the inner loop, append an empty summary and a placeholder code block to the accumulators, and proceed to the next outer step.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a malformed extractor response on the first attempt and a valid response on the retry, the system shall successfully extract c_t and p_0 from the retry.
> - Acceptance: Given two consecutive malformed responses, the outer loop shall skip the iteration without raising an exception.

> **REQ-P2O-035**: *Ablation Script Execution Timeout* -- Ablation scripts shall be executed with a timeout equal to `min(config.time_limit_seconds / (config.outer_loop_steps * 2), 600)` seconds, capping at 10 minutes per ablation execution. This prevents a single ablation study from consuming an excessive portion of the total time budget.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given `config.time_limit_seconds = 86400` and `config.outer_loop_steps = 4`, the ablation timeout shall be `min(86400/8, 600) = 600` seconds.

> **REQ-P2O-036**: *A_summarize Graceful Degradation* -- If the summarization agent returns an empty or unparseable response:
>
> 1. Log a warning with the raw response content.
> 2. Use a fallback summary constructed from the raw ablation output: truncate `raw_output` to the last 2000 characters and prefix with `"[Auto-summary from raw output] "`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given an empty summarization response, the system shall produce a fallback summary and not raise an exception.

### 7.3 Observability

> **REQ-P2O-037**: *Outer Loop Logging* -- Each outer loop step shall log the following events using Python's `logging` module at the specified levels:
>
> | Event | Level | Content |
> |-------|-------|---------|
> | Outer step start | `INFO` | Step index t, current h_best, number of accumulated summaries |
> | A_abl invocation start | `INFO` | Solution content length, number of previous summaries |
> | A_abl invocation complete | `INFO` | Ablation script length (characters) |
> | Ablation execution start | `INFO` | Script path, timeout |
> | Ablation execution complete | `INFO` | Exit code, output length, duration |
> | Ablation execution error | `WARNING` | Exit code, traceback summary (first line) |
> | A_summarize invocation start | `INFO` | Ablation code length, raw output length |
> | A_summarize invocation complete | `INFO` | Summary length |
> | A_extractor invocation start | `INFO` | Summary length, solution length, number of previous blocks |
> | A_extractor invocation complete | `INFO` | Number of plans returned, code block length |
> | Code block validation result | `INFO` | Pass/fail, match method (exact / whitespace-normalized) |
> | Code block validation failure | `WARNING` | Code block first 100 chars, re-invocation attempt number |
> | Inner loop handoff | `INFO` | Code block length, plan text (first 200 chars) |
> | Inner loop return | `INFO` | Best score from inner loop, improvement (yes/no) |
> | Outer step complete | `INFO` | Step index t, updated h_best, duration |
> | Outer step skipped | `WARNING` | Step index t, reason for skipping |
> | Outer loop complete | `INFO` | Total steps completed, final h_best, total duration |
>
> - Priority: Must | Verify: Inspection | Release: MVP

---

## 8. Constraints

### 8.1 Technology Constraints

> **REQ-P2O-038**: *SDK Agent Invocation* -- All three agents (A_abl, A_summarize, A_extractor) shall be invoked via the Claude Agent SDK agent mechanism. They shall not use direct API calls, raw HTTP requests, or any non-SDK LLM invocation method.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-02 -- all agent interactions through the SDK

> **REQ-P2O-039**: *Single Module Organization* -- All outer loop functions and agent invocation logic defined in this spec shall reside in a single Python module (e.g., `mle_star/phase2_outer.py`).
>
> - Priority: Should | Verify: Inspection | Release: MVP

### 8.2 Algorithmic Constraints

> **REQ-P2O-040**: *Sequential Outer Loop Execution* -- The outer loop iterations shall execute sequentially, not concurrently. Each iteration depends on the accumulated state (T_abl, C, s_t) from all prior iterations.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Rationale: Each iteration's ablation study depends on previous summaries, and the solution may change between iterations.
> - Source: REF-01 Algorithm 2 -- sequential loop with state accumulation

> **REQ-P2O-041**: *Monotonic Best Score* -- The `h_best` value tracked across the outer loop shall be monotonically non-decreasing (for maximize) or non-increasing (for minimize). The system shall never overwrite `s_final` with a solution that has a worse score than the current `h_best`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After every outer step, `h_best` shall be >= (maximize) or <= (minimize) the value at the start of that step.
> - Source: REF-01 Algorithm 2 lines 12-14 -- conditional update on `>=`

> **REQ-P2O-042**: *Ablation Script Independence* -- The ablation study script a_t generated by A_abl shall be a self-contained Python script independent of the solution script s_t. It may read and modify s_t's logic internally, but it shall not import s_t or depend on s_t being present as a separate file.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 12 -- "generate a simple Python code that performs an ablation study on the train.py script"

### 8.3 Data Integrity Constraints

> **REQ-P2O-043**: *Immutable Input Solution* -- The `run_phase2_outer_loop` function shall not mutate the `initial_solution` parameter. All modifications to the solution shall produce new `SolutionScript` instances. The original `initial_solution` shall remain available for fallback (REQ-P2O-029).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Rationale: Preserves the Phase 1 baseline for rollback if Phase 2 fails to improve.

> **REQ-P2O-044**: *Code Block Provenance* -- Each `CodeBlock` stored in C shall have its `outer_step` field set to the iteration index at which it was extracted. This enables A_extractor in subsequent iterations to identify which blocks have already been refined.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `C[t].outer_step == t` for all t in range `[0, len(C))`.

---

## 9. Traceability Matrix

### 9.1 Requirements to Paper Sections

| Req ID | Paper Section | Paper Element | SDK Construct |
|--------|--------------|---------------|---------------|
| REQ-P2O-001 | Section 3.2 | A_abl agent | `AgentDefinition` |
| REQ-P2O-002 | Figure 12 | Ablation prompt template | `prompt` parameter |
| REQ-P2O-003 | Alg 2 line 5 | `a_t = A_abl(s_t, T_abl)` input | -- |
| REQ-P2O-004 | Alg 2 line 5 | `a_t = A_abl(s_t, T_abl)` output | -- |
| REQ-P2O-005 | Figure 12 | "2-3 parts" ablation scope | -- |
| REQ-P2O-006 | Figure 12 | "not previously considered" | -- |
| REQ-P2O-007 | Figure 12 | "not load test data" | -- |
| REQ-P2O-008 | Section 3.2 | A_summarize agent | `AgentDefinition` |
| REQ-P2O-009 | Figure 13 | Summarization prompt template | `prompt` parameter |
| REQ-P2O-010 | Alg 2 line 7 | `T_abl^t = A_summarize(a_t, r_t)` input | -- |
| REQ-P2O-011 | Alg 2 line 7 | `T_abl^t = A_summarize(a_t, r_t)` output | -- |
| REQ-P2O-012 | Section 3.2 | A_extractor agent | `AgentDefinition` |
| REQ-P2O-013 | Figure 14 | Extractor prompt template | `prompt` parameter |
| REQ-P2O-014 | Figure 14 | Structured output schema usage | `output_format` |
| REQ-P2O-015 | Alg 2 line 8 | `c_t, p_0 = A_extractor(...)` output | `output_format` |
| REQ-P2O-016 | Alg 2 line 8 | `A_extractor(T_abl^t, s_t, C)` input | -- |
| REQ-P2O-017 | Figure 14 | "exactly extracted from the Python script" | -- |
| REQ-P2O-018 | Figure 14 | Code block validation failure handling | -- |
| REQ-P2O-019 | Alg 2 lines 1-28 | Outer loop function | -- |
| REQ-P2O-020 | Alg 2 line 6 | `r_t = exec(a_t)` | -- |
| REQ-P2O-021 | Section 3.4 | A_debugger for ablation errors | -- |
| REQ-P2O-022 | Alg 2 line 26 | `T_abl <- T_abl + T_abl^t` | -- |
| REQ-P2O-023 | Alg 2 line 27 | `C <- C + c_t` | -- |
| REQ-P2O-024 | Alg 2 lines 1-2 | `s_final <- s_0`, `h_best <- h(s_0)` tracking | -- |
| REQ-P2O-025 | Alg 2 line 4 | `for t = 0 to T-1` | -- |
| REQ-P2O-026 | Alg 2 lines 9-25 | Inner loop handoff | -- |
| REQ-P2O-027 | Alg 2 lines 12-14, 26-27 | Post-inner-loop update | -- |
| REQ-P2O-028 | Alg 2 line 29 | `Output: s_final` result construction | -- |
| REQ-P2O-029 | Alg 2 lines 1-2 | Preserve initial score on no improvement | -- |
| REQ-P2O-030 | Alg 2 (full) | Per-step history record | -- |
| REQ-P2O-031 | -- | Processing overhead | -- |
| REQ-P2O-032 | -- | Validation overhead | -- |
| REQ-P2O-033 | -- | Outer step duration | -- |
| REQ-P2O-034 | -- | Extractor graceful degradation | -- |
| REQ-P2O-035 | Section 4 | Ablation timeout | -- |
| REQ-P2O-036 | -- | Summarize graceful degradation | -- |
| REQ-P2O-037 | -- | Logging | Python `logging` |
| REQ-P2O-038 | -- | SDK-only invocation | `claude-agent-sdk` |
| REQ-P2O-039 | -- | Module organization | -- |
| REQ-P2O-040 | Alg 2 | Sequential execution | -- |
| REQ-P2O-041 | Alg 2 lines 12-14 | Monotonic best score | -- |
| REQ-P2O-042 | Figure 12 | Self-contained ablation script | -- |
| REQ-P2O-043 | -- | Immutable input solution | -- |
| REQ-P2O-044 | -- | Code block provenance | -- |

### 9.2 Cross-References to Other Specs

| Req ID | Referenced By |
|--------|--------------|
| REQ-P2O-001 (A_abl agent def) | Spec 09 (orchestrator configures agents) |
| REQ-P2O-008 (A_summarize agent def) | Spec 09 (orchestrator configures agents) |
| REQ-P2O-012 (A_extractor agent def) | Spec 09 (orchestrator configures agents) |
| REQ-P2O-015 (extractor output) | Spec 06 (inner loop receives c_t, p_0) |
| REQ-P2O-017 (code block validation) | Spec 06 (inner loop uses validated code block for replacement) |
| REQ-P2O-019 (outer loop function) | Spec 09 (orchestrator invokes Phase 2) |
| REQ-P2O-026 (inner loop handoff) | Spec 06 (defines inner loop function signature) |
| REQ-P2O-027 (post-inner-loop update) | Spec 06 (inner loop return type) |
| REQ-P2O-028 (Phase2Result) | Spec 07 (Phase 3 receives Phase2Result.best_solution), Spec 09 (orchestrator collects results) |

### 9.3 Spec 01 Dependencies (Inbound)

| Spec 01 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-DM-001 (PipelineConfig) | REQ-P2O-019, 025, 035 | `outer_loop_steps`, `max_debug_attempts`, `time_limit_seconds` |
| REQ-DM-007 (TaskDescription) | REQ-P2O-019, 024 | Task context, metric direction |
| REQ-DM-009 (SolutionScript) | REQ-P2O-003, 004, 016, 019, 024, 043 | Solution input/output type |
| REQ-DM-010 (replace_block) | REQ-P2O-017, 018 | Code block replacement (used by inner loop, validated here) |
| REQ-DM-012 (CodeBlock) | REQ-P2O-023, 028, 044 | Code block model for accumulation in C |
| REQ-DM-013 (AgentType) | REQ-P2O-001, 008, 012 | Agent identity enum values |
| REQ-DM-016 (RefinePlan) | REQ-P2O-015 | Structured output schema element |
| REQ-DM-017 (ExtractorOutput) | REQ-P2O-012, 014, 015 | Structured output model for extractor |
| REQ-DM-023 (Phase2Result) | REQ-P2O-028, 029 | Result model for outer loop output |
| REQ-DM-029 (is_improvement_or_equal) | REQ-P2O-024, 027 | Score comparison for best tracking |
| REQ-DM-032 (PromptRegistry) | REQ-P2O-002, 009, 013 | Template retrieval for all agents |
| REQ-DM-036 (AgentConfig) | REQ-P2O-001, 008, 012 | Agent-to-SDK mapping |
| REQ-DM-042 (RefinementAttempt) | REQ-P2O-027, 030 | Inner loop history records |

### 9.4 Spec 02 Dependencies (Inbound)

| Spec 02 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-EX-005 (write_script) | REQ-P2O-020 | Write ablation script to file |
| REQ-EX-007 (execute_script) | REQ-P2O-020 | Execute ablation script |
| REQ-EX-011 (parse_score) | REQ-P2O-020 | Parse ablation output (informational only) |
| REQ-EX-012 (extract_traceback) | REQ-P2O-021 | Extract traceback for A_debugger |
| REQ-EX-015 (evaluate_solution) | REQ-P2O-020 | Evaluate ablation scripts |

### 9.5 Spec 03 Dependencies (Inbound)

| Spec 03 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-SF-005 (extract_code_block) | REQ-P2O-004 | Extract ablation code from A_abl response |
| REQ-SF-006 (debug_solution) | REQ-P2O-021 | Fix ablation script errors |
| REQ-SF-022 (check_and_fix_leakage) | REQ-P2O-027 | Leakage check before evaluation (delegated to inner loop via Spec 06) |

### 9.6 Spec 04 Dependencies (Inbound)

| Spec 04 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-DM-022 (Phase1Result) | REQ-P2O-019 | Initial solution s_0 and initial score h_best |

---

## 10. Change Control

### 10.1 Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2026-02-20 | MLE-STAR Team | Initial draft -- all 44 requirements |

### 10.2 Baselining Policy

This SRS is baselined at version 1.0. After baselining, changes require impact analysis against Spec 06 (inner loop depends on outer loop handoff), Spec 09 (orchestrator invokes outer loop), Spec 01 (upstream data model dependencies), Spec 02 (upstream execution harness dependencies), and Spec 03 (upstream safety agent dependencies).
