# SRS 02 — Execution Harness: Advanced Operations and Interfaces

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
