# SRS 02 — Execution Harness: Script Operations

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
