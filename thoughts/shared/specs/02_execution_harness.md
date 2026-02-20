# Software Requirements Specification: MLE-STAR Script Execution and Evaluation Harness

| Field | Value |
|-------|-------|
| Version | 0.1.0 |
| Date | 2026-02-20 |
| Status | Draft |
| Spec ID | 02 of 09 |
| Requirement Prefix | REQ-EX- |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Product Perspective](#2-product-perspective)
3. [Execution Environment Requirements](#3-execution-environment-requirements)
4. [Functional Requirements](#4-functional-requirements)
5. [Interface Requirements](#5-interface-requirements)
6. [Non-Functional Requirements](#6-non-functional-requirements)
7. [Constraints](#7-constraints)
8. [Traceability Matrix](#8-traceability-matrix)
9. [Change Control](#9-change-control)

---

## 1. Introduction

### 1.1 Purpose

This SRS defines the Python execution engine that runs agent-generated solution scripts, captures output, parses scores, manages working directories, enforces timeouts, and implements subsampling logic. It is the runtime layer between the agent-generated code and the score function defined in Spec 01.

Intended audience: developers implementing the MLE-STAR system using the Claude Agent SDK for Python.

### 1.2 Scope

**Product name**: MLE-STAR (Machine Learning Engineering agent via Search and Targeted Refinement)

**What this spec covers**:
- Writing solution scripts to temporary files and executing them as subprocesses
- Capturing stdout, stderr, and exit codes from script execution
- Parsing validation scores from stdout
- Extracting Python tracebacks from stderr for debugger input
- Constructing `EvaluationResult` objects from execution output
- Working directory setup and verification (`./input/`, `./final/`)
- Subsampling enforcement during refinement and removal before final submission
- Timeout enforcement and resource isolation
- GPU availability detection and environment variable setup
- Retry-after-debug execution pattern
- Score comparison delegation to Spec 01 utilities
- Submission file verification
- Batch evaluation of multiple solutions

**Out of scope**:
- Data model definitions (covered by Spec 01)
- Agent behavior logic (covered by Specs 03-08)
- Orchestration control flow (covered by Spec 09)
- Safety and leakage detection logic (covered by Spec 03)

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|------|-----------|
| SRS | Software Requirements Specification |
| MLE-STAR | ML Engineering agent with web Search and TArgeted code block Refinement |
| h(s) | Score function -- maps a solution script to a real-valued performance score |
| A_debugger | Debugger agent -- receives traceback and produces a fixed solution |
| Subsampling | Capping training data to 30,000 samples during refinement for faster iteration |
| Working directory | The directory containing `./input/` (data) and `./final/` (output) |
| Traceback | Python stack trace produced on unhandled exceptions |
| Exit code | Integer returned by a process; 0 indicates success, non-zero indicates failure |

### 1.4 References

| ID | Title | Version | Source |
|----|-------|---------|--------|
| REF-01 | MLE-STAR paper | v3 | arXiv:2506.15692v3 |
| REF-02 | Claude Agent SDK reference | v0.1.39 | `claude-agent-sdk` PyPI |
| REF-03 | MLE-STAR architecture notes | -- | `thoughts/notes/mle_star_architecture.md` |
| REF-04 | MLE-STAR paper extraction | -- | `thoughts/notes/mle_star_paper.md` |
| REF-05 | Spec 01 -- Data Models and Interfaces | 0.1.0 | `thoughts/specs/01_data_models_and_interfaces.md` |

### 1.5 Document Overview

- Section 3: Execution environment requirements (working directory, GPU, environment variables)
- Section 4: Functional requirements (script writing, execution, parsing, subsampling, retry)
- Section 5: Interface requirements (integration with Spec 01 types, SDK Bash tool)
- Section 6: Non-functional requirements (timeout, isolation, performance)
- Section 7: Constraints (technology, subprocess model)
- Section 8: Traceability matrix

---

## 2. Product Perspective

### 2.1 System Context

This spec defines the execution harness that sits between agent-generated code and the score function. Every phase that evaluates a solution script passes through this harness:

```
Agent-generated SolutionScript
        |
        v
  [Spec 02: Execution Harness]
   1. Write script to file
   2. Set up environment
   3. Execute as subprocess
   4. Capture stdout/stderr/exit_code
   5. Parse score from stdout
   6. Extract traceback (on error)
   7. Build EvaluationResult
        |
        v
  EvaluationResult -> consumed by calling phase
```

### 2.2 Dependency Diagram

```
Spec 01 (Data Models)
  |
  |-- SolutionScript (REQ-DM-009)
  |-- EvaluationResult (REQ-DM-021)
  |-- TaskDescription (REQ-DM-007)
  |-- PipelineConfig (REQ-DM-001)
  |-- MetricDirection (REQ-DM-006)
  |-- Score parsing pattern (REQ-DM-027)
  |-- is_improvement() (REQ-DM-028)
  |-- is_improvement_or_equal() (REQ-DM-029)
  |
  v
Spec 02 (this) ──> Spec 03 (Safety uses execution)
               ──> Spec 04 (Phase 1 evaluates candidates)
               ──> Spec 05 (Phase 2 outer evaluates refined solutions)
               ──> Spec 06 (Phase 2 inner evaluates code block changes)
               ──> Spec 07 (Phase 3 evaluates ensembles)
               ──> Spec 08 (Submission runs final evaluation)
               ──> Spec 09 (Orchestrator coordinates execution)
```

### 2.3 Product Functions Summary

1. Write solution scripts to temporary files and execute them as Python subprocesses
2. Capture and parse execution output to build `EvaluationResult` objects
3. Manage working directory structure (`./input/`, `./final/`)
4. Enforce subsampling during refinement and remove subsampling before final submission
5. Enforce timeout limits and provide resource isolation
6. Support retry-after-debug execution patterns
7. Verify submission files after final execution

### 2.4 Operating Environment

- **Runtime**: Python 3.10+
- **Execution model**: `subprocess.run()` or equivalent for script isolation
- **Hardware target**: 96 vCPUs, 360 GB RAM, 8 NVIDIA V100 GPUs (per REF-01 Appendix F)
- **SDK alternative**: Execution may use the SDK `Bash` tool (`{"command": str, "timeout": int}` -> `{"output": str, "exitCode": int}`)

### 2.5 Assumptions and Dependencies

| ID | Assumption | Impact if Invalid |
|----|-----------|-------------------|
| A-01 | Solution scripts are self-contained single-file Python programs | Multi-file execution would require packaging logic |
| A-02 | Python 3.10+ is available in the execution environment | Scripts may use language features unavailable on older versions |
| A-03 | The `./input/` directory contains pre-downloaded competition data | Scripts will fail at data loading |
| A-04 | NVIDIA drivers and CUDA toolkit are installed if GPU execution is needed | GPU-dependent scripts will fail |
| A-05 | Scripts print `"Final Validation Performance: {score}"` to stdout on success | Score parsing will return None |

| ID | Dependency | Owner | Risk if Unavailable |
|----|-----------|-------|---------------------|
| D-01 | Spec 01 types (SolutionScript, EvaluationResult, etc.) | Spec 01 | Cannot construct inputs or outputs |
| D-02 | Python subprocess module | Python stdlib | Cannot execute scripts |
| D-03 | `claude-agent-sdk` v0.1.39+ (if using SDK Bash tool) | Anthropic | Must fall back to direct subprocess |

---

## 3. Execution Environment Requirements

### 3.1 Working Directory Setup

> **REQ-EX-001**: *Working Directory Structure* -- The system shall define a function `setup_working_directory(base_path: str) -> str` that creates or verifies the following directory structure:
>
> ```
> {base_path}/
>   input/      # Competition dataset (read-only from script perspective)
>   final/      # Script output directory (submission.csv written here)
> ```
>
> - The function shall create `input/` and `final/` if they do not exist.
> - The function shall return the absolute path to `base_path`.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After calling `setup_working_directory("/tmp/comp1")`, both `/tmp/comp1/input/` and `/tmp/comp1/final/` shall exist.
> - Source: REF-01 Figures 10, 11, 15, 18, 19, 25 -- data in `./input/`, output to `./final/`

> **REQ-EX-002**: *Working Directory Cleanup* -- The system shall define a function `clean_output_directory(base_path: str) -> None` that removes all files in `{base_path}/final/` without deleting the directory itself.
>
> - Rationale: Each script execution should start with a clean output directory to avoid stale submission files.
> - Priority: Should | Verify: Test | Release: MVP
> - Acceptance: After calling `clean_output_directory`, the `final/` directory shall exist but contain no files.

### 3.2 GPU and Hardware Detection

> **REQ-EX-003**: *GPU Availability Check* -- The system shall define a function `detect_gpu_info() -> dict` that returns a dictionary with at minimum the following keys:
>
> | Key | Type | Description |
> |-----|------|-------------|
> | `cuda_available` | `bool` | Whether CUDA is available |
> | `gpu_count` | `int` | Number of visible GPUs (0 if CUDA unavailable) |
> | `gpu_names` | `list[str]` | Names of detected GPUs |
>
> - The function shall not raise exceptions; it shall return `{"cuda_available": False, "gpu_count": 0, "gpu_names": []}` if detection fails.
> - Priority: Should | Verify: Demonstration | Release: MVP
> - Source: REF-01 Appendix F -- 8 NVIDIA V100 GPUs

### 3.3 Environment Variables

> **REQ-EX-004**: *Execution Environment Variables* -- The system shall define a function `build_execution_env(gpu_indices: list[int] | None = None) -> dict[str, str]` that returns a copy of the current environment with the following additions or overrides:
>
> | Variable | Value | Description |
> |----------|-------|-------------|
> | `PYTHONUNBUFFERED` | `"1"` | Ensure real-time stdout/stderr capture |
> | `PYTHONHASHSEED` | `"0"` | Deterministic hash behavior for reproducibility |
> | `CUDA_VISIBLE_DEVICES` | Comma-separated `gpu_indices` or all available | Control GPU visibility |
>
> - If `gpu_indices` is `None`, the `CUDA_VISIBLE_DEVICES` variable shall not be set (inherit from parent).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `build_execution_env(gpu_indices=[0, 1])` shall return an env dict where `CUDA_VISIBLE_DEVICES == "0,1"`.

---

## 4. Functional Requirements

### 4.1 Script Writer

> **REQ-EX-005**: *Write Script to File* -- The system shall define a function `write_script(solution: SolutionScript, working_dir: str, filename: str = "solution.py") -> str` that writes `solution.content` to `{working_dir}/{filename}` and returns the absolute path to the written file.
>
> - The function shall create the file with UTF-8 encoding.
> - The function shall overwrite any existing file at the target path.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After `write_script(s, "/tmp/work")`, the file `/tmp/work/solution.py` shall contain exactly `s.content`.
> - Source: REF-01 Figures 10, 11 -- scripts are written then executed

> **REQ-EX-006**: *Script Validation Before Write* -- Before writing, the `write_script` function shall perform the following static checks on `solution.content`:
>
> 1. Content is not empty (length > 0 after stripping whitespace).
> 2. Content does not contain calls to `exit()` or `sys.exit()` (matched via regex `r"\bexit\s*\("` and `r"\bsys\.exit\s*\("`).
>
> - If validation fails, the function shall raise `ValueError` with a descriptive message identifying the violation.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `write_script(SolutionScript(content="exit()", ...), ...)` shall raise `ValueError`.
> - Source: REF-01 Figures 10, 19, 25 -- "Do not use exit()"

### 4.2 Script Executor

> **REQ-EX-007**: *Execute Script as Subprocess* -- The system shall define an async function `execute_script(script_path: str, working_dir: str, timeout_seconds: int, env: dict[str, str] | None = None) -> ExecutionRawResult` that:
>
> 1. Runs `python {script_path}` as a subprocess with `cwd` set to `working_dir`.
> 2. Uses the provided `env` dictionary (or inherits the current environment if `None`).
> 3. Captures stdout and stderr separately.
> 4. Records the wall-clock duration in seconds.
> 5. Returns an `ExecutionRawResult` (see REQ-EX-008).
>
> - The function shall not raise exceptions on non-zero exit codes; errors are captured in the result.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 -- all execution follows write-then-run pattern

> **REQ-EX-008**: *ExecutionRawResult Model* -- The system shall define a dataclass or Pydantic model `ExecutionRawResult` with the following fields:
>
> | Field | Type | Description |
> |-------|------|-------------|
> | `stdout` | `str` | Full standard output from the subprocess |
> | `stderr` | `str` | Full standard error from the subprocess |
> | `exit_code` | `int` | Process exit code (0 = success) |
> | `duration_seconds` | `float` | Wall-clock execution time |
> | `timed_out` | `bool` | Whether execution was killed due to timeout |
>
> - Priority: Must | Verify: Inspection | Release: MVP

> **REQ-EX-009**: *Timeout Enforcement* -- The `execute_script` function shall enforce the `timeout_seconds` parameter by killing the subprocess (SIGTERM, then SIGKILL after 5 seconds grace) if it exceeds the allotted time.
>
> - When timeout occurs, `ExecutionRawResult.timed_out` shall be `True` and `exit_code` shall be `-1`.
> - Partial stdout/stderr captured before timeout shall be preserved.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: A script containing `time.sleep(600)` executed with `timeout_seconds=5` shall return a result with `timed_out=True` within 15 seconds.
> - Source: REF-01 Section 4 -- 24-hour maximum per competition

> **REQ-EX-010**: *Resource Isolation* -- Each invocation of `execute_script` shall run the solution in a new subprocess. The subprocess shall not share in-memory state with the calling process or with other concurrent script executions.
>
> - Priority: Must | Verify: Analysis | Release: MVP
> - Rationale: Prevents one failed script from corrupting the state of subsequent executions.

### 4.3 Output Parsing

> **REQ-EX-011**: *Score Parsing* -- The system shall define a function `parse_score(stdout: str) -> float | None` that extracts the validation score from stdout using the regex pattern defined in REQ-DM-027: `r"Final Validation Performance:\s*([\d.eE+-]+)"`.
>
> - If multiple matches exist, the function shall return the **last** match (the final score printed).
> - If no match exists, the function shall return `None`.
> - The matched string shall be converted to `float`. If conversion fails, the function shall return `None`.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `parse_score("Final Validation Performance: 0.8196\n")` shall return `0.8196`.
> - Acceptance: `parse_score("Training complete.\n")` shall return `None`.
> - Acceptance: `parse_score("...Performance: 0.5\n...Performance: 0.8196\n")` shall return `0.8196`.
> - Source: REF-01 Figures 10, 11, 15, 18, 19

> **REQ-EX-012**: *Traceback Extraction* -- The system shall define a function `extract_traceback(stderr: str) -> str | None` that extracts the Python traceback from stderr.
>
> - The function shall match the standard Python traceback pattern, starting from the line containing `"Traceback (most recent call last):"` through the final exception line.
> - If multiple tracebacks exist, the function shall return the **last** one.
> - If no traceback is found, the function shall return `None`.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given stderr containing a full Python traceback, the extracted string shall begin with `"Traceback (most recent call last):"` and end with the exception message line.
> - Source: REF-01 Figure 19 -- A_debugger receives `{bug}` (the error traceback)

> **REQ-EX-013**: *Error Detection* -- The system shall define a function `detect_error(raw: ExecutionRawResult) -> bool` that returns `True` if any of the following conditions hold:
>
> 1. `raw.exit_code != 0`
> 2. `raw.timed_out is True`
> 3. `raw.stderr` contains the string `"Traceback (most recent call last):"`
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: A result with `exit_code=1` shall be detected as an error. A result with `exit_code=0` and no traceback shall not.

### 4.4 EvaluationResult Construction

> **REQ-EX-014**: *Build EvaluationResult* -- The system shall define a function `build_evaluation_result(raw: ExecutionRawResult) -> EvaluationResult` that constructs an `EvaluationResult` (REQ-DM-021) from an `ExecutionRawResult` by:
>
> 1. Calling `parse_score(raw.stdout)` to obtain the score.
> 2. Calling `detect_error(raw)` to set `is_error`.
> 3. Calling `extract_traceback(raw.stderr)` to obtain `error_traceback` (only when `is_error` is `True`).
> 4. Mapping all other fields directly (`stdout`, `stderr`, `exit_code`, `duration_seconds`).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a successful execution with stdout containing `"Final Validation Performance: 0.82"`, the resulting `EvaluationResult` shall have `score=0.82`, `is_error=False`, `error_traceback=None`.
> - Source: REF-01 Section 3 -- h(s) returns a score; Section 3.4 -- A_debugger receives traceback

### 4.5 End-to-End Evaluation

> **REQ-EX-015**: *Evaluate Solution* -- The system shall define an async function `evaluate_solution(solution: SolutionScript, task: TaskDescription, config: PipelineConfig, timeout_override: int | None = None) -> EvaluationResult` that performs the full evaluation pipeline:
>
> 1. Call `setup_working_directory(task.data_dir)` (REQ-EX-001).
> 2. Call `clean_output_directory` to clear `./final/` (REQ-EX-002).
> 3. Call `write_script(solution, working_dir)` (REQ-EX-005).
> 4. Call `build_execution_env()` (REQ-EX-004).
> 5. Call `execute_script(script_path, working_dir, timeout, env)` (REQ-EX-007).
> 6. Call `build_evaluation_result(raw)` (REQ-EX-014).
> 7. Return the `EvaluationResult`.
>
> - `timeout_override` shall take precedence over `config.time_limit_seconds` when provided.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a valid solution that prints the score pattern, `evaluate_solution` shall return an `EvaluationResult` with a non-None score.

> **REQ-EX-016**: *Update SolutionScript After Evaluation* -- After a successful evaluation (score is not `None`), the `evaluate_solution` function shall return the `EvaluationResult` and the caller shall be responsible for updating `solution.score` and `solution.is_executable`. The harness itself shall not mutate the input `SolutionScript`.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Rationale: Separation of concerns -- the harness evaluates, the caller updates state.

### 4.6 Subsampling

> **REQ-EX-017**: *Subsampling Instruction Text* -- The system shall define a constant `SUBSAMPLE_INSTRUCTION` containing the subsampling instruction text: `"If there are more than {limit} training samples, you must subsample to {limit} for a faster run."` where `{limit}` is parameterized.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 10

> **REQ-EX-018**: *Subsampling Enforcement* -- The system shall define a function `get_subsample_instruction(config: PipelineConfig) -> str` that returns the `SUBSAMPLE_INSTRUCTION` with `{limit}` replaced by `config.subsample_limit` (default: 30,000).
>
> - This instruction shall be included in prompts for A_init, A_coder, A_ensembler, and A_debugger during refinement phases.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `get_subsample_instruction(PipelineConfig())` shall return a string containing `"30000"`.
> - Source: REF-01 Figure 10 -- "If there are more than 30,000 training samples, you must subsample to 30,000 for a faster run"

> **REQ-EX-019**: *Subsampling Removal Interface* -- The system shall define a function `request_subsample_removal(solution: SolutionScript) -> str` that returns a prompt string instructing an agent to:
>
> 1. Identify all subsampling code in the solution script.
> 2. Remove the subsampling code while preserving all other functionality.
> 3. Return the full modified script.
>
> - The returned prompt shall include the full solution script content.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: The returned prompt shall contain `solution.content` and instruction text referencing subsampling removal.
> - Source: REF-01 Figures 26-27 -- subsampling extraction and removal before final submission

> **REQ-EX-020**: *Subsampling Extraction Interface* -- The system shall define a function `request_subsample_extraction(solution: SolutionScript) -> str` that returns a prompt string instructing an agent to identify and extract the subsampling code block from the solution script.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Figure 26 -- extract subsampling code

### 4.7 Retry-After-Debug Pattern

> **REQ-EX-021**: *Execution with Debug Retry* -- The system shall define an async function `evaluate_with_retry(solution: SolutionScript, task: TaskDescription, config: PipelineConfig, debug_callback: Callable[[SolutionScript, str], Awaitable[SolutionScript]], max_retries: int | None = None) -> tuple[SolutionScript, EvaluationResult]` that:
>
> 1. Calls `evaluate_solution(solution, task, config)` (REQ-EX-015).
> 2. If `result.is_error` is `True` and retries remain:
>    a. Calls `debug_callback(solution, result.error_traceback)` to obtain a fixed `SolutionScript`.
>    b. Recursively evaluates the fixed solution.
>    c. Decrements retry count.
> 3. Returns the final `(SolutionScript, EvaluationResult)` pair.
>
> - `max_retries` shall default to `config.max_debug_attempts` (REQ-DM-001) when `None`.
> - If all retries are exhausted and the script still errors, the function shall return the last `(SolutionScript, EvaluationResult)` with `is_error=True`.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a script that fails once and succeeds after debugging, `evaluate_with_retry` shall return the fixed solution with a successful result after exactly one retry.
> - Source: REF-01 Figure 19 -- A_debugger receives traceback, produces fixed script; Section 3.4

### 4.8 Score Comparison Delegation

> **REQ-EX-022**: *Score Comparison via Spec 01* -- The execution harness shall not implement its own score comparison logic. All score comparisons shall delegate to `is_improvement()` (REQ-DM-028) and `is_improvement_or_equal()` (REQ-DM-029) from Spec 01.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Rationale: Single source of truth for score direction logic; avoids inconsistent maximize/minimize handling.

> **REQ-EX-023**: *Convenience Comparison Function* -- The system shall define a function `is_better_solution(new_result: EvaluationResult, old_score: float, direction: MetricDirection) -> bool` that:
>
> 1. Returns `False` if `new_result.score` is `None`.
> 2. Returns `False` if `new_result.is_error` is `True`.
> 3. Otherwise delegates to `is_improvement(new_result.score, old_score, direction)` (REQ-DM-028).
>
> - Priority: Should | Verify: Test | Release: MVP
> - Acceptance: `is_better_solution(EvaluationResult(score=0.9, is_error=False, ...), 0.8, MetricDirection.maximize)` shall return `True`.

### 4.9 Submission File Verification

> **REQ-EX-024**: *Verify Submission File* -- The system shall define a function `verify_submission(working_dir: str, expected_filename: str = "submission.csv") -> bool` that returns `True` if the file `{working_dir}/final/{expected_filename}` exists and has size > 0 bytes.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After a successful script execution that writes `./final/submission.csv`, `verify_submission` shall return `True`.
> - Source: REF-01 Figures 10, 11 -- scripts must produce `submission.csv`

> **REQ-EX-025**: *Submission File Details* -- The system shall define a function `get_submission_info(working_dir: str, expected_filename: str = "submission.csv") -> dict` that returns:
>
> | Key | Type | Description |
> |-----|------|-------------|
> | `exists` | `bool` | Whether the submission file exists |
> | `path` | `str` | Absolute path to the submission file |
> | `size_bytes` | `int` | File size in bytes (0 if not exists) |
> | `row_count` | `int \| None` | Number of rows (lines minus header), None if not exists |
>
> - Priority: Should | Verify: Test | Release: MVP

### 4.10 Batch Evaluation

> **REQ-EX-026**: *Evaluate Multiple Solutions* -- The system shall define an async function `evaluate_batch(solutions: list[SolutionScript], task: TaskDescription, config: PipelineConfig) -> list[EvaluationResult]` that evaluates each solution sequentially by calling `evaluate_solution` (REQ-EX-015) for each.
>
> - Solutions shall be evaluated sequentially, not concurrently, to avoid resource contention.
> - The returned list shall be in the same order as the input list.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `evaluate_batch([s1, s2, s3], task, config)` shall return a list of 3 `EvaluationResult` objects in order.

> **REQ-EX-027**: *Sort Solutions by Score* -- The system shall define a function `rank_solutions(solutions: list[SolutionScript], results: list[EvaluationResult], direction: MetricDirection) -> list[tuple[SolutionScript, EvaluationResult]]` that returns a list of `(solution, result)` tuples sorted by score (best first according to `direction`).
>
> - Solutions with `None` scores shall be placed at the end of the list.
> - Solutions with `is_error=True` shall be placed after `None`-scored solutions.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: For `direction="maximize"`, a solution with score `0.9` shall appear before one with score `0.8`.

---

## 5. Interface Requirements

### 5.1 Integration with Spec 01 Types

> **REQ-EX-028**: *SolutionScript Input Type* -- All functions that accept a solution script shall use the `SolutionScript` type defined in REQ-DM-009.
>
> - Priority: Must | Verify: Inspection | Release: MVP

> **REQ-EX-029**: *EvaluationResult Output Type* -- All functions that return evaluation results shall use the `EvaluationResult` type defined in REQ-DM-021.
>
> - Priority: Must | Verify: Inspection | Release: MVP

> **REQ-EX-030**: *TaskDescription Input Type* -- All functions that require task context shall use the `TaskDescription` type defined in REQ-DM-007.
>
> - Priority: Must | Verify: Inspection | Release: MVP

> **REQ-EX-031**: *PipelineConfig Input Type* -- All functions that require configuration shall use the `PipelineConfig` type defined in REQ-DM-001.
>
> - Priority: Must | Verify: Inspection | Release: MVP

### 5.2 ScoreFunction Protocol Compliance

> **REQ-EX-032**: *ScoreFunction Implementation* -- The `evaluate_solution` function (REQ-EX-015) shall satisfy the `ScoreFunction` protocol defined in REQ-DM-026 when wrapped: i.e., calling `evaluate_solution(solution, task, config)` shall produce an `EvaluationResult` consistent with the `h: S -> R` interface.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Rationale: The execution harness is the concrete implementation of the abstract score function.

### 5.3 SDK Bash Tool Interface

> **REQ-EX-033**: *SDK Bash Tool Compatibility* -- The system should define an alternative executor `execute_script_via_sdk(script_path: str, working_dir: str, timeout_ms: int) -> ExecutionRawResult` that uses the Claude Agent SDK's `Bash` tool with the interface `{"command": str, "timeout": int}` -> `{"output": str, "exitCode": int}`.
>
> - The `command` shall be `f"cd {working_dir} && python {script_path}"`.
> - The `timeout` shall be `timeout_ms` (milliseconds; max 600,000 per SDK limit).
> - `stdout` and `stderr` in the returned `ExecutionRawResult` shall be derived from the combined `output` field.
> - Priority: Should | Verify: Demonstration | Release: MVP
> - Source: REF-02 -- SDK Bash tool specification

> **REQ-EX-034**: *Executor Strategy Selection* -- The system shall define an enum `ExecutorStrategy` with values `"subprocess"` and `"sdk_bash"`, and the `evaluate_solution` function shall accept an optional `strategy: ExecutorStrategy` parameter to select the execution backend.
>
> - Default: `"subprocess"` (direct subprocess execution for full control over stdout/stderr separation).
> - Priority: Should | Verify: Test | Release: MVP

---

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
