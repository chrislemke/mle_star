# SRS 08 — Finalization: Test Submission

---

## 4. A_test Requirements

### 4.1 Agent Definition

> **REQ-FN-010**: *A_test Agent Definition* -- The system shall define an `AgentDefinition`-compatible configuration for the test submission agent with the following properties:
>
> | Property | Value |
> |----------|-------|
> | `agent_type` | `AgentType.test` |
> | `description` | Agent that transforms a validation solution into a test submission script |
> | `prompt` | Rendered from the A_test template (Figure 25, REQ-DM-032) |
> | `tools` | `["Read"]` |
> | `output_schema` | `None` (free-form code block response) |
> | `model` | `None` (inherit from orchestrator) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `AgentConfig(agent_type=AgentType.test).to_agent_definition()` shall produce a valid dictionary for `ClaudeAgentOptions.agents`.
> - Source: REF-01 Section 4, Figure 25

> **REQ-FN-011**: *A_test Prompt Template* -- The A_test agent prompt shall be constructed by rendering the Figure 25 template from the `PromptRegistry` (REQ-DM-032) with the following variables:
>
> | Variable | Type | Description |
> |----------|------|-------------|
> | `task description` | `str` | Full task description text (`TaskDescription.description`) |
> | `final solution` | `str` | Source code of the solution with subsampling removed (`SolutionScript.content`) |
>
> - The rendered prompt shall include all instructions from Figure 25:
>   - Load test samples and create a submission file
>   - All data in `./input/` directory, no need to unzip
>   - Save test predictions as `submission.csv` in `./final/` directory
>   - Do not drop any test samples; predict for all test samples
>   - Replace validation samples with test samples; use full training set
>   - Do not modify the solution code too much; integrate with minimal changes
>   - No additional headings or text; single-file self-contained Python program
>   - Response as a single code block
>   - Do not forget `./final/submission.csv`
>   - Do not use `exit()`
>   - Do not use `try:/except:` or `if/else` to ignore unintended behavior
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 25

### 4.2 Input/Output Contract

> **REQ-FN-012**: *A_test Input Contract* -- The A_test agent shall accept two inputs:
>
> 1. `task: TaskDescription` -- the task description providing context about the competition.
> 2. `solution: SolutionScript` -- the final solution with subsampling removed (output of REQ-FN-009).
>
> - Precondition: `solution.content` is non-empty and `task.description` is non-empty.
> - Precondition: The solution should have subsampling already removed (or have never contained subsampling).
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Figure 25 -- "We will now provide a task description and a Python solution"

> **REQ-FN-013**: *A_test Output Contract* -- The A_test agent shall return a `SolutionScript` containing a single-file, self-contained test submission Python script extracted from the agent's response.
>
> - The response shall be parsed using `extract_code_block()` (REQ-SF-005) to obtain the script code.
> - The returned `SolutionScript` shall have `phase` set to `SolutionPhase.final` and `is_executable` set to `True` (optimistic; will be verified by execution).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given an agent response containing a fenced code block, the returned `SolutionScript.content` shall contain only the code within the fences.

### 4.3 Behavioral Constraints

> **REQ-FN-014**: *A_test Minimal Modification* -- The A_test agent shall not heavily alter the original solution. The prompt (REQ-FN-011) instructs "Do not modify the given Python solution code too much. Try to integrate test submission with minimal changes." This constraint is enforced via prompt instruction.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 25

> **REQ-FN-015**: *A_test Full Training Set Usage* -- The test submission script produced by A_test shall use the full training set (not a subsample). The prompt (REQ-FN-011) instructs "you can even use the full training set." Since subsampling has already been removed (REQ-FN-009), the input solution should already train on full data.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 25

> **REQ-FN-016**: *A_test Test Data Loading* -- The test submission script shall load test data from the `./input/` directory. The prompt (REQ-FN-011) instructs "Test data is available in the `./input/` directory."
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 25

> **REQ-FN-017**: *A_test Submission File Generation* -- The test submission script shall generate a `submission.csv` file in the `./final/` directory. The prompt (REQ-FN-011) instructs "Save the test predictions in a `submission.csv` file. Put the `submission.csv` into `./final` directory."
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 25

> **REQ-FN-018**: *A_test All Test Samples* -- The test submission script shall predict for ALL test samples without dropping any. The prompt (REQ-FN-011) instructs "You should not drop any test samples. Predict the target value for all test samples."
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 25

### 4.4 A_test Invocation Function

> **REQ-FN-019**: *generate_test_submission() Function* -- The system shall define an async function `generate_test_submission(task: TaskDescription, solution: SolutionScript) -> SolutionScript` that:
>
> 1. Renders the A_test prompt template (REQ-FN-011) with `task.description` and `solution.content`.
> 2. Invokes the A_test agent via the Claude Agent SDK.
> 3. Parses the agent response using `extract_code_block()` (REQ-SF-005).
> 4. Constructs and returns a new `SolutionScript` with `phase=SolutionPhase.final`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `generate_test_submission(task, solution)` shall return a `SolutionScript` with `phase == SolutionPhase.final` and non-empty `content`.

---

## 5. Execution and Verification Requirements

### 5.1 Test Script Execution

> **REQ-FN-020**: *Execute Test Submission Script* -- The system shall execute the test submission script (output of REQ-FN-019) using `evaluate_solution()` (REQ-EX-015) from the execution harness (Spec 02).
>
> - The execution shall use the full `config.time_limit_seconds` timeout (not a reduced timeout), since the test script trains on the full dataset.
> - The `clean_output_directory` step (REQ-EX-002) shall clear `./final/` before execution to ensure no stale `submission.csv` exists.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Section 4 -- final submission is executed to produce submission.csv

### 5.2 Submission File Verification

> **REQ-FN-021**: *Verify Submission File Exists* -- After executing the test submission script, the system shall call `verify_submission()` (REQ-EX-024) to confirm that `./final/submission.csv` exists and has size > 0 bytes.
>
> - If verification fails (file does not exist or is empty), the system shall treat this as an execution error and trigger the debug retry flow (REQ-FN-023).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After a successful test script execution, `verify_submission(working_dir)` shall return `True`.
> - Source: REF-01 Figure 25 -- "Do not forget the ./final/submission.csv file"

> **REQ-FN-022**: *Verify Submission File Content* -- After confirming the submission file exists, the system shall call `get_submission_info()` (REQ-EX-025) to obtain the row count and verify that:
>
> 1. The file is parseable (not corrupted or malformed).
> 2. The file contains at least one data row (row_count >= 1).
>
> - If the file is not parseable or contains zero data rows, the system shall log a warning but not automatically trigger a retry (the file format may be intentionally non-CSV for certain competitions).
> - Priority: Should | Verify: Test | Release: MVP
> - Acceptance: `get_submission_info(working_dir)` shall return a dict with `exists=True`, `row_count >= 1` for a valid submission.

### 5.3 Error Handling and Retry

> **REQ-FN-023**: *A_debugger for Test Script Errors* -- If the test submission script fails (i.e., `EvaluationResult.is_error` is `True` or `verify_submission()` returns `False`), the system shall invoke the debug retry flow using `evaluate_with_retry()` (REQ-EX-021) with `make_debug_callback()` (REQ-SF-007).
>
> - The maximum number of debug attempts shall be `config.max_debug_attempts` (REQ-DM-001, default: 3).
> - Each retry cycle: A_debugger fixes the script -> re-execute -> re-verify submission file.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a test script that fails once and is fixed by A_debugger, the system shall produce a valid `submission.csv` after one retry.
> - Source: REF-01 Section 3.4 -- A_debugger applies to all phases including finalization

> **REQ-FN-024**: *Retry Limit for Test Script* -- The maximum number of debug retry attempts for the test submission script shall be `config.max_debug_attempts` (REQ-DM-001). This limit is the same as used in other pipeline phases.
>
> - Priority: Must | Verify: Inspection | Release: MVP

> **REQ-FN-025**: *Fallback on Test Script Failure* -- If the test submission script fails after all debug retries are exhausted, the system shall fall back to the best validation solution from Phase 2/3 as the final solution.
>
> - The fallback solution shall be the `best_solution` from `Phase2Result` (REQ-DM-023) or `best_ensemble` from `Phase3Result` (REQ-DM-024), whichever was the input to finalization.
> - The system shall log a warning indicating that test submission generation failed and the validation solution is being used as a fallback.
> - The `FinalResult.submission_path` shall be set to an empty string or `None` to indicate no valid submission file was produced.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After exhausting all retries, `run_finalization()` shall return a `FinalResult` with `final_solution` set to the best validation solution and a logged warning.
