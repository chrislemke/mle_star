# SRS 08 — Finalization: Constraints and Traceability

---

## 8. Non-Functional Requirements

### 8.1 Performance

> **REQ-FN-037**: *Finalization Overhead* -- The finalization process overhead (excluding LLM calls and script execution time) shall not exceed 5 seconds. This includes prompt rendering, response parsing, code block extraction, replacement, submission file verification, and `FinalResult` construction.
>
> - Priority: Should | Verify: Test | Release: MVP

> **REQ-FN-038**: *Subsampling Removal Latency* -- The subsampling removal step (REQ-FN-009) shall complete within the time of two sequential LLM calls (extraction + removal). When no subsampling is found, it shall complete within the time of one LLM call (extraction only).
>
> - Priority: Should | Verify: Test | Release: MVP

### 8.2 Reliability

> **REQ-FN-039**: *Subsampling Extraction Graceful Degradation* -- If the subsampling extraction agent returns a response that cannot be parsed as a valid code block or returns a code block that is not found in the solution:
>
> 1. Log a warning with the raw response content (first 200 characters).
> 2. Return the original solution unchanged (assume no subsampling).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a malformed extraction response, `remove_subsampling()` shall return the original solution and not raise an exception.

> **REQ-FN-040**: *A_test Graceful Degradation* -- If the A_test agent returns a response that does not contain extractable code:
>
> 1. Log a warning with the raw response content (first 200 characters).
> 2. Proceed to the debug retry flow, treating the empty code as an execution error.
>
> - If the debug retry also fails after all attempts, the fallback behavior (REQ-FN-025) shall activate.
> - Priority: Must | Verify: Test | Release: MVP

> **REQ-FN-041**: *Contamination Check Graceful Degradation* -- If the contamination check agent returns a response that cannot be parsed as `DataContaminationResult`:
>
> 1. Log a warning with the raw response content (first 200 characters).
> 2. Treat the check as inconclusive and record `None` for the contamination result.
> 3. Do not block finalization; proceed to `FinalResult` construction.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a malformed contamination response, `check_contamination()` shall return `None` and not raise an exception.

### 8.3 Observability

> **REQ-FN-042**: *Finalization Logging* -- The finalization process shall log the following events using Python's `logging` module at the specified levels:
>
> | Event | Level | Content |
> |-------|-------|---------|
> | Finalization start | `INFO` | Solution phase, content length, competition_id |
> | Subsampling extraction start | `INFO` | Solution content length |
> | Subsampling extraction result | `INFO` | Whether subsampling was found, extracted block length |
> | Subsampling removal result | `INFO` | Original block length, replacement block length |
> | Subsampling replacement result | `INFO` | Whether replacement succeeded, solution content length change |
> | A_test invocation start | `INFO` | Task competition_id, solution content length |
> | A_test invocation result | `INFO` | Generated script content length |
> | Test script execution start | `INFO` | Script content length, timeout |
> | Test script execution result | `INFO` | Exit code, duration, whether submission.csv produced |
> | Submission verification result | `INFO` | File exists, size bytes, row count |
> | Debug retry triggered | `WARNING` | Attempt number, error summary |
> | Fallback activated | `WARNING` | Reason for fallback, fallback solution phase and score |
> | Contamination check start | `INFO` | Number of reference discussions |
> | Contamination check result | `INFO` | Per-reference verdicts, overall verdict |
> | Contamination check skipped | `INFO` | Reason (no references provided) |
> | FinalResult construction | `INFO` | Final solution phase, submission path, total duration |
>
> - Priority: Must | Verify: Inspection | Release: MVP

---

## 9. Constraints

### 9.1 Technology Constraints

> **REQ-FN-043**: *SDK Agent Invocation* -- All agents defined in this spec (A_test, subsampling extraction, subsampling removal, contamination check) shall be invoked via the Claude Agent SDK agent mechanism. They shall not use direct API calls, raw HTTP requests, or any non-SDK LLM invocation method.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-02 -- all agent interactions through the SDK

> **REQ-FN-044**: *Single Module Organization* -- All finalization functions defined in this spec shall reside in a single Python module (e.g., `mle_star/finalization.py`).
>
> - Priority: Should | Verify: Inspection | Release: MVP

### 9.2 Submission Invariants

> **REQ-FN-045**: *Submission File Path Convention* -- The test submission script shall always write the submission file to `./final/submission.csv`. This path is hardcoded in the A_test prompt (REQ-FN-011) and verified by `verify_submission()` (REQ-EX-024).
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figures 10, 11, 25

> **REQ-FN-046**: *No exit() in Test Script* -- The test submission script shall not contain calls to `exit()`, `sys.exit()`, `os._exit()`, or `quit()`. This is enforced by the A_test prompt (REQ-FN-011) instruction "Do not use exit() function" and validated by `write_script()` (REQ-EX-006) before execution.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 25

> **REQ-FN-047**: *No Error Masking in Test Script* -- The test submission script shall not use `try:/except:` or `if/else` to ignore unintended behavior. This is enforced via the A_test prompt instruction (REQ-FN-011). The advisory detection in REQ-EX-045 also applies.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 25

### 9.3 Agent Configuration Constraints

> **REQ-FN-048**: *Finalization Agent Default Configs* -- The `build_default_agent_configs()` function (REQ-DM-040) shall include `AgentConfig` entries for the A_test agent type. The subsampling extraction and removal operations shall use templates registered under `AgentType.test` with variants (REQ-DM-035).
>
> | AgentType | Tools | Output Schema | Model |
> |-----------|-------|---------------|-------|
> | `test` | `["Read"]` | `None` (A_test) / `DataContaminationResult` (contamination variant) | `None` |
>
> - The A_test agent has multiple operational modes (test submission, subsampling extraction, subsampling removal, contamination check) sharing `AgentType.test`. The default `AgentConfig` shall be for the test submission variant. Other variants are selected by swapping the prompt template and, for contamination, setting `output_schema=DataContaminationResult`.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REQ-DM-035, REQ-DM-040

---

## 10. Traceability Matrix

### 10.1 Requirements to Paper Sections

| Req ID | Paper Section | Paper Element | SDK Construct |
|--------|--------------|---------------|---------------|
| REQ-FN-001 | Figure 26 | Subsampling extraction prompt | `prompt` parameter |
| REQ-FN-002 | Figure 26 | Extraction input (final solution) | -- |
| REQ-FN-003 | Figure 26 | Extraction output (code block) | -- |
| REQ-FN-004 | Figure 27 | Subsampling removal prompt | `prompt` parameter |
| REQ-FN-005 | Figure 27 | Removal input (code block) | -- |
| REQ-FN-006 | Figure 27 | Removal output (code block without subsampling) | -- |
| REQ-FN-007 | Figures 26-27 | Code block replacement | `SolutionScript.replace_block()` |
| REQ-FN-008 | -- | No-subsampling passthrough | -- |
| REQ-FN-009 | Figures 26-27 | Subsampling removal orchestration | -- |
| REQ-FN-010 | Section 4 | A_test agent definition | `AgentDefinition` |
| REQ-FN-011 | Figure 25 | A_test prompt template | `prompt` parameter |
| REQ-FN-012 | Figure 25 | A_test input (task + solution) | -- |
| REQ-FN-013 | Figure 25 | A_test output (test submission script) | -- |
| REQ-FN-014 | Figure 25 | Minimal modification constraint | -- |
| REQ-FN-015 | Figure 25 | Full training set usage | -- |
| REQ-FN-016 | Figure 25 | Test data from ./input/ | -- |
| REQ-FN-017 | Figure 25 | ./final/submission.csv output | -- |
| REQ-FN-018 | Figure 25 | All test samples predicted | -- |
| REQ-FN-019 | Figure 25 | generate_test_submission() function | -- |
| REQ-FN-020 | Section 4 | Execute test submission script | `evaluate_solution()` |
| REQ-FN-021 | Figure 25 | Verify submission.csv exists | `verify_submission()` |
| REQ-FN-022 | -- | Verify submission.csv content | `get_submission_info()` |
| REQ-FN-023 | Section 3.4 | A_debugger for test script errors | `evaluate_with_retry()` |
| REQ-FN-024 | Section 3.4 | Retry limit from PipelineConfig | REQ-DM-001 |
| REQ-FN-025 | -- | Fallback to validation solution | -- |
| REQ-FN-026 | Figure 28 | Contamination check agent definition | `AgentDefinition` |
| REQ-FN-027 | Figure 28 | Contamination check prompt | `prompt` parameter |
| REQ-FN-028 | Figure 28 | Contamination check input | -- |
| REQ-FN-029 | Figure 28 | Contamination check structured output | `output_format` |
| REQ-FN-030 | Appendix H | Optional contamination check | -- |
| REQ-FN-031 | Appendix H | Multiple reference discussions | -- |
| REQ-FN-032 | -- | Contamination result logging | Python `logging` |
| REQ-FN-033 | Figure 28 | check_contamination() function | -- |
| REQ-FN-034 | Section 4 | run_finalization() signature | -- |
| REQ-FN-035 | Section 4 | Finalization orchestration steps | -- |
| REQ-FN-036 | Section 4 | FinalResult construction | REQ-DM-025 |
| REQ-FN-037 | -- | Finalization overhead | -- |
| REQ-FN-038 | -- | Subsampling removal latency | -- |
| REQ-FN-039 | -- | Subsampling extraction graceful degradation | -- |
| REQ-FN-040 | -- | A_test graceful degradation | -- |
| REQ-FN-041 | -- | Contamination check graceful degradation | -- |
| REQ-FN-042 | -- | Logging | Python `logging` |
| REQ-FN-043 | -- | SDK-only invocation | `claude-agent-sdk` |
| REQ-FN-044 | -- | Module organization | -- |
| REQ-FN-045 | Figures 10, 11, 25 | ./final/submission.csv path | -- |
| REQ-FN-046 | Figure 25 | No exit() | REQ-EX-006 |
| REQ-FN-047 | Figure 25 | No error masking | REQ-EX-045 |
| REQ-FN-048 | Section 6 | Default agent configs | `AgentDefinition` |

### 10.2 Cross-References to Other Specs

| Req ID | Referenced By |
|--------|--------------|
| REQ-FN-009 (remove_subsampling) | Spec 09 (orchestrator calls finalization) |
| REQ-FN-019 (generate_test_submission) | Spec 09 (orchestrator calls finalization) |
| REQ-FN-033 (check_contamination) | Spec 09 (orchestrator calls finalization) |
| REQ-FN-034 (run_finalization) | Spec 09 (orchestrator calls run_finalization as final pipeline step) |
| REQ-FN-036 (FinalResult construction) | Spec 09 (orchestrator receives FinalResult) |

### 10.3 Spec 01 Dependencies (Inbound)

| Spec 01 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-DM-001 (PipelineConfig) | REQ-FN-020, REQ-FN-023, REQ-FN-024, REQ-FN-034 | Timeout, max debug attempts |
| REQ-DM-007 (TaskDescription) | REQ-FN-012, REQ-FN-019, REQ-FN-034, REQ-FN-035 | Task context for A_test and finalization |
| REQ-DM-008 (SolutionPhase) | REQ-FN-013 | SolutionPhase.final for test submission script |
| REQ-DM-009 (SolutionScript) | REQ-FN-002, REQ-FN-007, REQ-FN-009, REQ-FN-012, REQ-FN-013, REQ-FN-019, REQ-FN-028 | Input/output type throughout finalization |
| REQ-DM-010 (replace_block) | REQ-FN-007 | Code block replacement for subsampling removal |
| REQ-DM-013 (AgentType) | REQ-FN-010, REQ-FN-026, REQ-FN-048 | Agent identity enum (AgentType.test) |
| REQ-DM-020 (DataContaminationResult) | REQ-FN-026, REQ-FN-029, REQ-FN-033 | Structured output for contamination check |
| REQ-DM-022 (Phase1Result) | REQ-FN-034, REQ-FN-036 | FinalResult construction |
| REQ-DM-023 (Phase2Result) | REQ-FN-025, REQ-FN-034, REQ-FN-036 | Fallback solution, FinalResult construction |
| REQ-DM-024 (Phase3Result) | REQ-FN-025, REQ-FN-034, REQ-FN-036 | Fallback solution, FinalResult construction |
| REQ-DM-025 (FinalResult) | REQ-FN-034, REQ-FN-036 | Return type of run_finalization() |
| REQ-DM-032 (PromptRegistry) | REQ-FN-001, REQ-FN-004, REQ-FN-011, REQ-FN-027 | Template retrieval for all agents |
| REQ-DM-035 (Subsampling templates) | REQ-FN-001, REQ-FN-004 | Subsampling extraction/removal templates |
| REQ-DM-036 (AgentConfig) | REQ-FN-010, REQ-FN-026, REQ-FN-048 | Agent-to-SDK mapping |
| REQ-DM-040 (build_default_agent_configs) | REQ-FN-048 | Default finalization agent configs |

### 10.4 Spec 02 Dependencies (Inbound)

| Spec 02 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-EX-002 (clean_output_directory) | REQ-FN-020 | Clear ./final/ before test execution |
| REQ-EX-006 (write_script validation) | REQ-FN-046 | No exit() enforcement before execution |
| REQ-EX-015 (evaluate_solution) | REQ-FN-020 | Execute test submission script |
| REQ-EX-019 (request_subsample_removal) | REQ-FN-004 | Subsampling removal prompt construction |
| REQ-EX-020 (request_subsample_extraction) | REQ-FN-001 | Subsampling extraction prompt construction |
| REQ-EX-021 (evaluate_with_retry) | REQ-FN-023 | Debug retry for test script |
| REQ-EX-024 (verify_submission) | REQ-FN-021 | Submission file existence check |
| REQ-EX-025 (get_submission_info) | REQ-FN-022 | Submission file content check |
| REQ-EX-045 (detect_error_masking) | REQ-FN-047 | Advisory try/except detection |

### 10.5 Spec 03 Dependencies (Inbound)

| Spec 03 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-SF-005 (extract_code_block) | REQ-FN-003, REQ-FN-006, REQ-FN-013 | Parse code blocks from agent responses |
| REQ-SF-006 (debug_solution) | REQ-FN-023 | Debug failing test scripts |
| REQ-SF-007 (make_debug_callback) | REQ-FN-023, REQ-FN-035 | Debug callback for evaluate_with_retry |
| REQ-SF-020 (check_and_fix_leakage) | REQ-FN-035 | Leakage check on test submission script |
| REQ-SF-022 (leakage before evaluation) | REQ-FN-035 | Leakage check integration point |

---

## 11. Change Control

### 11.1 Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2026-02-20 | MLE-STAR Team | Initial draft -- all 48 requirements |

### 11.2 Baselining Policy

This SRS is baselined at version 1.0. After baselining, changes require impact analysis against Spec 09 (the orchestrator that calls `run_finalization()`), Spec 01 (upstream data model dependencies), Spec 02 (upstream execution harness dependencies), and Spec 03 (upstream safety agent dependencies).
