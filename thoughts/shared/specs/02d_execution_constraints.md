# SRS 02 — Execution Harness: Constraints and Traceability

## 6. Non-Functional Requirements

### 6.1 Performance

> **REQ-EX-035**: *Execution Overhead* -- The execution harness overhead (file write, subprocess launch, output parsing) excluding the actual script runtime shall not exceed 2 seconds per invocation under normal conditions.
>
> - Measurement: `duration_seconds` from `ExecutionRawResult` minus actual script compute time.
> - Priority: Should | Verify: Test | Release: MVP

> **REQ-EX-036**: *Score Parsing Speed* -- The `parse_score` function shall execute in under 10 milliseconds for stdout strings up to 1 MB.
>
> - Priority: Should | Verify: Test | Release: MVP

### 6.2 Reliability

> **REQ-EX-037**: *Graceful Timeout Handling* -- When a subprocess is killed due to timeout, the harness shall not leave orphan child processes. All child processes of the killed subprocess shall also be terminated.
>
> - Implementation: Use process group termination (`os.killpg`) or equivalent.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After timeout of a script that spawns child processes, no child processes from that execution shall remain running.

> **REQ-EX-038**: *Large Output Handling* -- The harness shall handle stdout and stderr outputs up to 100 MB without crashing. Outputs exceeding 100 MB should be truncated with a warning appended to the truncated output.
>
> - Priority: Should | Verify: Test | Release: MVP
> - Rationale: ML training scripts can produce verbose logging.

### 6.3 Observability

> **REQ-EX-039**: *Execution Logging* -- The harness shall log the following events using Python's `logging` module at the specified levels:
>
> | Event | Level | Content |
> |-------|-------|---------|
> | Script write | `DEBUG` | Script path, content length |
> | Execution start | `INFO` | Script path, working directory, timeout |
> | Execution complete | `INFO` | Exit code, duration, score (if parsed) |
> | Timeout triggered | `WARNING` | Script path, timeout value |
> | Error detected | `WARNING` | Exit code, traceback summary (first line) |
> | Retry attempt | `INFO` | Retry number, max retries |
>
> - Priority: Must | Verify: Inspection | Release: MVP

### 6.4 Maintainability

> **REQ-EX-040**: *Module Organization* -- All execution harness functions defined in this spec shall reside in a single Python module (e.g., `mle_star/execution.py`).
>
> - Priority: Should | Verify: Inspection | Release: MVP

---

## 7. Constraints

### 7.1 Technology Constraints

> **REQ-EX-041**: *Python Subprocess Only* -- Script execution shall use Python's `subprocess` module (or the SDK Bash tool) as the execution mechanism. The harness shall not use `exec()`, `eval()`, or `importlib` to run solution scripts.
>
> - Rationale: Subprocess execution provides process isolation, separate stdout/stderr streams, timeout enforcement, and exit code capture.
> - Priority: Must | Verify: Inspection | Release: MVP

> **REQ-EX-042**: *No Persistent State Between Executions* -- The harness shall not maintain in-process state between script executions. Each call to `execute_script` shall be independent.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Rationale: Prevents state leakage between solution evaluations.

> **REQ-EX-043**: *UTF-8 Encoding* -- All script files shall be written and read using UTF-8 encoding. Stdout and stderr shall be decoded as UTF-8 with `errors="replace"` to handle non-UTF-8 output gracefully.
>
> - Priority: Must | Verify: Test | Release: MVP

### 7.2 Script Content Constraints

> **REQ-EX-044**: *No exit() Enforcement* -- The `write_script` function (REQ-EX-006) shall reject scripts containing calls to `exit()`, `sys.exit()`, `os._exit()`, or `quit()`.
>
> - Rationale: These calls terminate the subprocess without producing the expected output pattern.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Figures 10, 19, 25 -- "Do not use exit()"

> **REQ-EX-045**: *No Broad try/except Detection* -- The system should define a function `detect_error_masking(content: str) -> list[str]` that scans solution script content for patterns that mask errors:
>
> 1. Bare `except:` clauses (without specifying an exception type).
> 2. `except Exception:` or `except BaseException:` clauses that contain only `pass` in the handler body.
>
> - The function shall return a list of warning strings describing each detected pattern.
> - This is advisory only; it shall not prevent script execution.
> - Priority: Should | Verify: Test | Release: MVP
> - Source: REF-01 Figures 10, 22 -- "Do not mask errors with try/except"

### 7.3 Timeout Constraints

> **REQ-EX-046**: *Default Timeout Derivation* -- When no `timeout_override` is provided to `evaluate_solution`, the timeout shall be derived from `config.time_limit_seconds` (REQ-DM-001). The harness shall not apply its own arbitrary default timeout.
>
> - Priority: Must | Verify: Inspection | Release: MVP

> **REQ-EX-047**: *SDK Bash Timeout Limit* -- When using the SDK Bash tool executor (REQ-EX-033), the timeout shall be capped at 600,000 milliseconds (10 minutes) per the SDK specification. If the required timeout exceeds this limit, the harness shall fall back to the subprocess executor (REQ-EX-007).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-02 -- SDK Bash tool has a 600,000ms maximum timeout

---

## 8. Traceability Matrix

### 8.1 Requirements to Paper Sections

| Req ID | Paper Section | Paper Element | Spec 01 Dependency |
|--------|--------------|---------------|---------------------|
| REQ-EX-001 | Figures 10, 11 | `./input/`, `./final/` directories | REQ-DM-007 (TaskDescription.data_dir, output_dir) |
| REQ-EX-002 | -- | Output cleanup | -- |
| REQ-EX-003 | Appendix F | 8 NVIDIA V100 GPUs | -- |
| REQ-EX-004 | Appendix F | CUDA environment | -- |
| REQ-EX-005 | Figures 10, 11 | Write script to file | REQ-DM-009 (SolutionScript) |
| REQ-EX-006 | Figures 10, 19, 25 | No exit() validation | REQ-DM-009 (SolutionScript) |
| REQ-EX-007 | Section 3 | Subprocess execution | -- |
| REQ-EX-008 | -- | Raw result model | -- |
| REQ-EX-009 | Section 4 | 24h time limit | REQ-DM-001 (PipelineConfig.time_limit_seconds) |
| REQ-EX-010 | -- | Process isolation | -- |
| REQ-EX-011 | Figures 10-19 | Score parsing pattern | REQ-DM-027 (score regex) |
| REQ-EX-012 | Figure 19 | Traceback for A_debugger | REQ-DM-021 (EvaluationResult.error_traceback) |
| REQ-EX-013 | -- | Error detection | REQ-DM-021 (EvaluationResult.is_error) |
| REQ-EX-014 | Section 3 | Build EvaluationResult | REQ-DM-021 (EvaluationResult) |
| REQ-EX-015 | Section 3 | End-to-end h(s) | REQ-DM-026 (ScoreFunction) |
| REQ-EX-016 | -- | Immutability of input | REQ-DM-009 (SolutionScript) |
| REQ-EX-017 | Figure 10 | Subsampling instruction | REQ-DM-001 (PipelineConfig.subsample_limit) |
| REQ-EX-018 | Figure 10 | 30K subsample cap | REQ-DM-001 (PipelineConfig.subsample_limit) |
| REQ-EX-019 | Figure 27 | Remove subsampling code | REQ-DM-035 (subsampling removal template) |
| REQ-EX-020 | Figure 26 | Extract subsampling code | REQ-DM-035 (subsampling extraction template) |
| REQ-EX-021 | Figure 19 | Debug retry loop | REQ-DM-001 (PipelineConfig.max_debug_attempts) |
| REQ-EX-022 | Alg 2 lines 12, 21 | Score comparison delegation | REQ-DM-028, REQ-DM-029 |
| REQ-EX-023 | Alg 2 lines 12, 21 | Convenience comparator | REQ-DM-028 (is_improvement) |
| REQ-EX-024 | Figures 10, 11 | Submission file check | -- |
| REQ-EX-025 | -- | Submission file details | -- |
| REQ-EX-026 | -- | Batch evaluation | REQ-DM-009, REQ-DM-021 |
| REQ-EX-027 | -- | Rank solutions | REQ-DM-006 (MetricDirection) |
| REQ-EX-028 | -- | Type alignment | REQ-DM-009 (SolutionScript) |
| REQ-EX-029 | -- | Type alignment | REQ-DM-021 (EvaluationResult) |
| REQ-EX-030 | -- | Type alignment | REQ-DM-007 (TaskDescription) |
| REQ-EX-031 | -- | Type alignment | REQ-DM-001 (PipelineConfig) |
| REQ-EX-032 | Section 3 | ScoreFunction protocol | REQ-DM-026 (ScoreFunction) |
| REQ-EX-033 | -- | SDK Bash tool | -- |
| REQ-EX-034 | -- | Executor strategy | -- |
| REQ-EX-035 | -- | Execution overhead | -- |
| REQ-EX-036 | -- | Parsing speed | -- |
| REQ-EX-037 | -- | Orphan process cleanup | -- |
| REQ-EX-038 | -- | Large output handling | -- |
| REQ-EX-039 | -- | Logging | -- |
| REQ-EX-040 | -- | Module organization | -- |
| REQ-EX-041 | -- | Subprocess only | -- |
| REQ-EX-042 | -- | No persistent state | -- |
| REQ-EX-043 | -- | UTF-8 encoding | -- |
| REQ-EX-044 | Figures 10, 19, 25 | No exit() | -- |
| REQ-EX-045 | Figures 10, 22 | No error masking | -- |
| REQ-EX-046 | Section 4 | Timeout from config | REQ-DM-001 (PipelineConfig.time_limit_seconds) |
| REQ-EX-047 | -- | SDK timeout cap | -- |

### 8.2 Cross-References to Other Specs

| Req ID | Referenced By |
|--------|--------------|
| REQ-EX-007 (execute_script) | Specs 03-09 (all phases execute scripts) |
| REQ-EX-011 (parse_score) | Specs 04-08 (all phases parse scores) |
| REQ-EX-012 (extract_traceback) | Spec 06 (inner loop debug), Spec 04 (Phase 1 debug) |
| REQ-EX-015 (evaluate_solution) | Specs 04-08 (primary evaluation entry point) |
| REQ-EX-017-018 (subsampling) | Specs 04-06 (refinement phases include subsampling) |
| REQ-EX-019-020 (subsampling removal) | Spec 08 (submission removes subsampling) |
| REQ-EX-021 (evaluate_with_retry) | Specs 04-07 (all phases may debug and retry) |
| REQ-EX-024 (verify_submission) | Spec 08 (submission verification) |
| REQ-EX-026-027 (batch/rank) | Spec 04 (rank Phase 1 candidates), Spec 07 (rank ensembles) |

### 8.3 Spec 01 Dependencies (Inbound)

| Spec 01 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-DM-001 (PipelineConfig) | REQ-EX-015, 018, 021, 031, 046 | Timeout, subsample limit, max debug attempts |
| REQ-DM-006 (MetricDirection) | REQ-EX-023, 027 | Score comparison direction |
| REQ-DM-007 (TaskDescription) | REQ-EX-001, 015, 030 | Working directory paths |
| REQ-DM-009 (SolutionScript) | REQ-EX-005, 006, 015, 016, 019, 020, 021, 026, 027, 028 | Script content for execution |
| REQ-DM-021 (EvaluationResult) | REQ-EX-014, 015, 023, 026, 027, 029 | Evaluation result construction |
| REQ-DM-026 (ScoreFunction) | REQ-EX-032 | Protocol compliance |
| REQ-DM-027 (Score regex) | REQ-EX-011 | Score parsing pattern |
| REQ-DM-028 (is_improvement) | REQ-EX-022, 023 | Score comparison |
| REQ-DM-029 (is_improvement_or_equal) | REQ-EX-022 | Score comparison |
| REQ-DM-035 (Subsampling templates) | REQ-EX-019, 020 | Subsampling agent prompts |

---

## 9. Change Control

### 9.1 Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2026-02-20 | MLE-STAR Team | Initial draft -- all 47 requirements |

### 9.2 Baselining Policy

This SRS is baselined at version 1.0. After baselining, changes require impact analysis against Specs 03-09 (all downstream consumers of the execution harness) and Spec 01 (upstream data model dependencies).
