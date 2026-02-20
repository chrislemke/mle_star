# SRS 03 — Safety Modules: Constraints & Traceability

## 7. Non-Functional Requirements

### 7.1 Performance

> **REQ-SF-036**: *A_debugger Latency* -- A single debugger agent invocation (prompt rendering + LLM call + response parsing) shall complete within the LLM's response time. The system shall not add more than 500 milliseconds of overhead beyond the LLM call itself (for prompt construction, code extraction, and `SolutionScript` construction).
>
> - Priority: Should | Verify: Test | Release: MVP

> **REQ-SF-037**: *A_leakage Two-Step Latency* -- The full leakage detection and correction pipeline (`check_and_fix_leakage`) shall complete within the time of two sequential LLM calls (detection + correction). When no leakage is detected, it shall complete within the time of one LLM call (detection only).
>
> - Priority: Should | Verify: Test | Release: MVP

### 7.2 Reliability

> **REQ-SF-038**: *A_leakage Detection Graceful Degradation* -- If the leakage detection agent returns a response that cannot be parsed as `LeakageDetectionOutput` (e.g., malformed JSON, schema validation failure):
>
> 1. Log a warning with the raw response content.
> 2. Return the original solution unchanged (assume no leakage).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a malformed detection response, `check_and_fix_leakage` shall return the original solution and not raise an exception.
> - Rationale: A failure in the safety agent should not crash the pipeline; conservative behavior (assume no leakage) is preferred over halting.

> **REQ-SF-039**: *A_data Graceful Degradation* -- If the data usage agent returns a response that cannot be parsed as either a confirmation or a valid code block:
>
> 1. Log a warning with the raw response content.
> 2. Return the original solution unchanged.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given an unparseable response, `check_data_usage` shall return the original solution and not raise an exception.

> **REQ-SF-040**: *A_debugger Graceful Degradation* -- If the debugger agent returns a response that does not contain extractable code:
>
> 1. Log a warning with the raw response content.
> 2. Return the original (broken) solution unchanged, allowing the retry loop to exhaust its attempts or the fallback to activate.
>
> - Priority: Must | Verify: Test | Release: MVP

### 7.3 Observability

> **REQ-SF-041**: *Safety Agent Logging* -- Each safety agent invocation shall log the following events using Python's `logging` module at the specified levels:
>
> | Event | Level | Content |
> |-------|-------|---------|
> | Debug invocation start | `INFO` | Solution phase, traceback summary (first line), attempt number |
> | Debug invocation result | `INFO` | Whether fix was successful, code length change |
> | Debug fallback triggered | `WARNING` | Number of attempts exhausted, falling back to last working version |
> | Leakage detection start | `INFO` | Solution phase, content length |
> | Leakage detection result | `INFO` | Number of answers, leakage found (yes/no) |
> | Leakage correction start | `INFO` | Code block length being corrected |
> | Leakage correction result | `INFO` | Corrected code block length, replacement success |
> | Leakage replacement skipped | `WARNING` | Original code block not found in solution |
> | Data check start | `INFO` | Solution phase, task competition_id |
> | Data check result | `INFO` | Whether solution was modified or confirmed as complete |
> | Parse failure | `WARNING` | Agent type, raw response summary (first 200 chars) |
>
> - Priority: Must | Verify: Inspection | Release: MVP

---

## 8. Constraints

### 8.1 Technology Constraints

> **REQ-SF-042**: *SDK Agent Invocation* -- All three safety agents shall be invoked via the Claude Agent SDK agent mechanism. They shall not use direct API calls, raw HTTP requests, or any non-SDK LLM invocation method.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-02 -- all agent interactions through the SDK

> **REQ-SF-043**: *Single Module Organization* -- All safety agent functions defined in this spec shall reside in a single Python module (e.g., `mle_star/safety.py`), with the shared `extract_code_block` utility either in this module or in a shared utilities module.
>
> - Priority: Should | Verify: Inspection | Release: MVP

### 8.2 Safety Invariants

> **REQ-SF-044**: *A_debugger Shall Not Introduce New Functionality* -- The debugger agent is intended to fix execution errors only. The prompt (REQ-SF-002) instructs the agent to "revise the code to fix the error" without adding new features, models, or data processing steps. This invariant is enforced via prompt instruction, not automated verification.
>
> - Priority: Should | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 19 -- "Please revise the code to fix the error"

> **REQ-SF-045**: *A_leakage Shall Preserve Non-Preprocessing Code* -- The leakage correction agent shall modify only the preprocessing code block identified during detection. The correction prompt (REQ-SF-017) states "Just modify it with the above code," instructing the agent to return only the corrected block rather than rewriting the entire solution.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 21 -- "Note that all the variables are defined earlier. Just modify it with the above code."

> **REQ-SF-046**: *A_data Shall Not Suppress Errors* -- The data agent prompt (REQ-SF-025, REQ-SF-029) explicitly prohibits the use of try/except blocks. This ensures that errors from incorporating new data sources are propagated to the debugger agent for proper resolution.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 22 -- "DO NOT USE TRY AND EXCEPT; just occur error so we can debug it!"

---

## 9. Traceability Matrix

### 9.1 Requirements to Paper Sections

| Req ID | Paper Section | Paper Element | SDK Construct |
|--------|--------------|---------------|---------------|
| REQ-SF-001 | Section 3.4 | A_debugger agent | `AgentDefinition` |
| REQ-SF-002 | Figure 19 | Debugger prompt template | `prompt` parameter |
| REQ-SF-003 | Eq. 10 | `s <- A_debugger(s, T_bug)` input | -- |
| REQ-SF-004 | Eq. 10 | `s <- A_debugger(s, T_bug)` output | -- |
| REQ-SF-005 | Figures 19, 21, 22 | Code block extraction from response | -- |
| REQ-SF-006 | Section 3.4 | Debug retry loop | -- |
| REQ-SF-007 | Section 3.4 | Debug callback for evaluate_with_retry | `debug_callback` parameter |
| REQ-SF-008 | Section 3.4 | Fallback to last executable version | -- |
| REQ-SF-009 | Figure 19 | "Do not remove subsampling" | -- |
| REQ-SF-010 | Figure 19 | "Final Validation Performance" enforcement | -- |
| REQ-SF-011 | Section 3.4 | A_leakage detection agent | `AgentDefinition` |
| REQ-SF-012 | Figure 20 | Leakage detection prompt template | `prompt` parameter |
| REQ-SF-013 | Section 3.4 | Detection input | -- |
| REQ-SF-014 | Figure 20 | Detection structured output | `output_format` |
| REQ-SF-015 | Figure 20 | Structured output schema usage | `output_format` |
| REQ-SF-016 | Section 3.4 | A_leakage correction agent | `AgentDefinition` |
| REQ-SF-017 | Figure 21 | Leakage correction prompt template | `prompt` parameter |
| REQ-SF-018 | Section 3.4 | Correction input | -- |
| REQ-SF-019 | Section 3.4 | Correction output | -- |
| REQ-SF-020 | Section 3.4 | Two-step detect-then-correct pipeline | -- |
| REQ-SF-021 | Section 3.4 | `s.replace(c_data, c_data*)` | -- |
| REQ-SF-022 | Section 3.4 | Runs before every evaluation | -- |
| REQ-SF-023 | Table 5 | Ablation evidence (leakage) | -- |
| REQ-SF-024 | Section 3.4 | A_data agent | `AgentDefinition` |
| REQ-SF-025 | Figure 22 | Data usage prompt template | `prompt` parameter |
| REQ-SF-026 | Eq. 11 | `s_0 <- A_data(s_0, T_task)` input | -- |
| REQ-SF-027 | Eq. 11 | `s_0 <- A_data(s_0, T_task)` output | -- |
| REQ-SF-028 | Figure 22 | Two response format options | -- |
| REQ-SF-029 | Figure 22 | "DO NOT USE TRY AND EXCEPT" | -- |
| REQ-SF-030 | Section 3.4 | Runs once after Phase 1 | -- |
| REQ-SF-031 | Table 6 | Ablation evidence (data usage) | -- |
| REQ-SF-032 | Section 6 | Default agent configs | `AgentDefinition` |
| REQ-SF-033 | Figures 19-22 | Prompt template registry usage | `PromptRegistry` |
| REQ-SF-034 | -- | Uniform SolutionScript type contract | -- |
| REQ-SF-035 | -- | Immutable data flow | -- |
| REQ-SF-036 | -- | Debugger latency | -- |
| REQ-SF-037 | -- | Leakage pipeline latency | -- |
| REQ-SF-038 | -- | Leakage graceful degradation | -- |
| REQ-SF-039 | -- | Data agent graceful degradation | -- |
| REQ-SF-040 | -- | Debugger graceful degradation | -- |
| REQ-SF-041 | -- | Logging | Python `logging` |
| REQ-SF-042 | -- | SDK-only invocation | `claude-agent-sdk` |
| REQ-SF-043 | -- | Module organization | -- |
| REQ-SF-044 | Figure 19 | Debugger scope constraint | -- |
| REQ-SF-045 | Figure 21 | Correction scope constraint | -- |
| REQ-SF-046 | Figure 22 | No try/except constraint | -- |

### 9.2 Cross-References to Other Specs

| Req ID | Referenced By |
|--------|--------------|
| REQ-SF-005 (extract_code_block) | Specs 04-07 (phases that parse agent code responses) |
| REQ-SF-006 (debug_solution) | Specs 04-07 (all phases may debug failing scripts) |
| REQ-SF-007 (make_debug_callback) | Specs 04-07 (passed to evaluate_with_retry) |
| REQ-SF-008 (fallback behavior) | Spec 09 (orchestrator maintains solution history) |
| REQ-SF-020 (check_and_fix_leakage) | Specs 04-07 (called before every evaluation) |
| REQ-SF-022 (leakage integration point) | Specs 04-07 (define invocation points) |
| REQ-SF-028 (parse_data_agent_response) | Spec 04 (Phase 1 invokes A_data) |
| REQ-SF-030 (check_data_usage) | Spec 04 (Phase 1 invokes A_data after initial solution) |

### 9.3 Spec 01 Dependencies (Inbound)

| Spec 01 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-DM-001 (PipelineConfig) | REQ-SF-006 | `max_debug_attempts` for retry loop |
| REQ-DM-007 (TaskDescription) | REQ-SF-006, REQ-SF-026, REQ-SF-030 | Task context for debugger and data agent |
| REQ-DM-009 (SolutionScript) | REQ-SF-003, REQ-SF-004, REQ-SF-013, REQ-SF-018, REQ-SF-026, REQ-SF-027, REQ-SF-034, REQ-SF-035 | Input/output type for all safety agents |
| REQ-DM-010 (replace_block) | REQ-SF-021 | Code block replacement in leakage correction |
| REQ-DM-013 (AgentType) | REQ-SF-001, REQ-SF-011, REQ-SF-016, REQ-SF-024, REQ-SF-032, REQ-SF-033 | Agent identity enum values |
| REQ-DM-018 (LeakageAnswer) | REQ-SF-014, REQ-SF-020 | Structured output schema for detection |
| REQ-DM-019 (LeakageDetectionOutput) | REQ-SF-011, REQ-SF-014, REQ-SF-015, REQ-SF-032 | Structured output model for detection agent |
| REQ-DM-021 (EvaluationResult) | REQ-SF-006, REQ-SF-034 | Evaluation result from debug retry |
| REQ-DM-032 (PromptRegistry) | REQ-SF-002, REQ-SF-012, REQ-SF-017, REQ-SF-025, REQ-SF-033 | Template retrieval for all agents |
| REQ-DM-034 (Leakage dual templates) | REQ-SF-012, REQ-SF-017, REQ-SF-033 | Detection/correction template variants |
| REQ-DM-036 (AgentConfig) | REQ-SF-001, REQ-SF-011, REQ-SF-016, REQ-SF-024, REQ-SF-032 | Agent-to-SDK mapping |
| REQ-DM-040 (build_default_agent_configs) | REQ-SF-032 | Default safety agent configurations |

### 9.4 Spec 02 Dependencies (Inbound)

| Spec 02 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-EX-012 (extract_traceback) | REQ-SF-002, REQ-SF-003 | Traceback extraction for debugger input |
| REQ-EX-013 (detect_error) | REQ-SF-006 | Error detection in debug retry loop |
| REQ-EX-015 (evaluate_solution) | REQ-SF-006 | Evaluation of fixed solutions |
| REQ-EX-021 (evaluate_with_retry) | REQ-SF-007 | Retry-after-debug integration |

---

## 10. Change Control

### 10.1 Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2026-02-20 | MLE-STAR Team | Initial draft -- all 46 requirements |

### 10.2 Baselining Policy

This SRS is baselined at version 1.0. After baselining, changes require impact analysis against Specs 04-09 (all downstream consumers of the safety agents), Spec 01 (upstream data model dependencies), and Spec 02 (upstream execution harness dependencies).
