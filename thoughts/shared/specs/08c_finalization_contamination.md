# SRS 08 — Finalization: Contamination Check and Orchestration

---

## 6. Data Contamination Check Requirements

### 6.1 Contamination Check Agent Definition

> **REQ-FN-026**: *Data Contamination Check Agent Definition* -- The system shall define an `AgentDefinition`-compatible configuration for the data contamination check agent with the following properties:
>
> | Property | Value |
> |----------|-------|
> | `agent_type` | `AgentType.test` (shares type with A_test) |
> | `description` | Agent that checks whether a solution is copied from a Kaggle discussion |
> | `prompt` | Rendered from the contamination check template (Figure 28) |
> | `tools` | `None` (no tools needed; pure text comparison) |
> | `output_schema` | `DataContaminationResult` (REQ-DM-020) |
> | `model` | `None` (inherit from orchestrator) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Figure 28, Appendix H

> **REQ-FN-027**: *Data Contamination Check Prompt Template* -- The contamination check agent prompt shall be constructed by rendering the Figure 28 template with the following variables:
>
> | Variable | Type | Description |
> |----------|------|-------------|
> | `reference discussion` | `str` | Text of a reference Kaggle discussion post |
> | `final solution` | `str` | Source code of the final solution (`SolutionScript.content`) |
>
> - The rendered prompt shall include all instructions from Figure 28:
>   - Check whether the Python solution just copies the reference discussion
>   - If sufficiently novel and different, answer "Novel"
>   - If too similar, answer "Same"
>   - Answer should be only one of "Novel" or "Same"
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 28

### 6.2 Contamination Check Input/Output Contract

> **REQ-FN-028**: *Data Contamination Check Input Contract* -- The data contamination check agent shall accept two inputs:
>
> 1. `reference_discussion: str` -- the text of a reference Kaggle discussion post.
> 2. `solution: SolutionScript` -- the final solution to check for contamination.
>
> - Precondition: Both `reference_discussion` and `solution.content` are non-empty strings.
> - Priority: Must | Verify: Test | Release: MVP

> **REQ-FN-029**: *Data Contamination Check Output Contract* -- The data contamination check agent shall return a `DataContaminationResult` (REQ-DM-020) parsed from the agent's structured JSON response.
>
> - The output contains a single field `verdict` with value `"Novel"` or `"Same"`.
> - The agent shall use the Claude Agent SDK's `output_format` parameter set to `{"type": "json_schema", "schema": DataContaminationResult.model_json_schema()}` to ensure the response conforms to the schema.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: The agent response shall be parseable via `DataContaminationResult.model_validate_json(response)` without errors.
> - Source: REF-01 Figure 28

### 6.3 Optional Invocation

> **REQ-FN-030**: *Contamination Check Is Optional* -- The data contamination check shall only be executed when reference discussion texts are available. If no reference discussions are provided (empty list or `None`), the check shall be skipped entirely.
>
> - The system shall log an informational message indicating whether the contamination check was run or skipped.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `run_finalization(..., reference_discussions=None)` shall skip the contamination check without error.
> - Source: REF-01 Appendix H -- contamination check depends on available reference discussions

> **REQ-FN-031**: *Contamination Check Against Multiple References* -- When multiple reference discussions are provided, the system shall run the contamination check against each reference independently and collect the results.
>
> - If **any** reference produces a `verdict == "Same"`, the overall contamination status shall be `"Same"`.
> - If **all** references produce `verdict == "Novel"`, the overall contamination status shall be `"Novel"`.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given references `["ref1", "ref2"]` where `ref1` yields `"Novel"` and `ref2` yields `"Same"`, the overall result shall be `"Same"`.

> **REQ-FN-032**: *Contamination Check Result Logging* -- The system shall log the contamination check result at `INFO` level, including:
>
> - The number of reference discussions checked
> - The per-reference verdict
> - The overall verdict
>
> - Priority: Must | Verify: Inspection | Release: MVP

### 6.4 Contamination Check Function

> **REQ-FN-033**: *check_contamination() Function* -- The system shall define an async function `check_contamination(solution: SolutionScript, reference_discussions: list[str]) -> DataContaminationResult | None` that:
>
> 1. If `reference_discussions` is empty or `None`, return `None` (REQ-FN-030).
> 2. For each reference discussion, invoke the contamination check agent (REQ-FN-026) with the reference and solution.
> 3. Parse each response as `DataContaminationResult` (REQ-FN-029).
> 4. Determine the overall verdict per REQ-FN-031.
> 5. Log the result (REQ-FN-032).
> 6. Return the overall `DataContaminationResult`.
>
> - Priority: Must | Verify: Test | Release: MVP

---

## 7. Finalization Orchestration Requirements

### 7.1 run_finalization() Function

> **REQ-FN-034**: *run_finalization() Function Signature* -- The system shall define an async function with the following signature:
>
> ```python
> async def run_finalization(
>     solution: SolutionScript,
>     task: TaskDescription,
>     config: PipelineConfig,
>     phase1_result: Phase1Result,
>     phase2_results: list[Phase2Result],
>     phase3_result: Phase3Result | None,
>     reference_discussions: list[str] | None = None,
> ) -> FinalResult
> ```
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Section 4 -- finalization after main pipeline

> **REQ-FN-035**: *run_finalization() Orchestration Steps* -- The `run_finalization()` function shall execute the following steps in order:
>
> 1. **Subsampling removal**: Call `remove_subsampling(solution)` (REQ-FN-009) to obtain a solution without subsampling code.
> 2. **Test submission generation**: Call `generate_test_submission(task, solution_no_subsample)` (REQ-FN-019) to obtain the test submission script.
> 3. **Leakage check**: Call `check_and_fix_leakage(test_script)` (REQ-SF-020, REQ-SF-022) to verify the test script does not contain data leakage.
> 4. **Execution with retry**: Call `evaluate_with_retry(test_script, task, config, make_debug_callback(task, config))` (REQ-EX-021) to execute the test script with debug retry on failure.
> 5. **Submission verification**: Call `verify_submission(working_dir)` (REQ-EX-024) and `get_submission_info(working_dir)` (REQ-EX-025) to verify the output file.
> 6. **Fallback handling**: If execution failed after all retries or verification failed, apply fallback behavior (REQ-FN-025).
> 7. **Contamination check**: If `reference_discussions` is provided, call `check_contamination(final_solution, reference_discussions)` (REQ-FN-033).
> 8. **FinalResult construction**: Build and return the `FinalResult` (REQ-FN-036).
>
> - Each step shall be logged at `INFO` level before starting.
> - Priority: Must | Verify: Test | Release: MVP

### 7.2 FinalResult Construction

> **REQ-FN-036**: *FinalResult Construction* -- The `run_finalization()` function shall construct a `FinalResult` (REQ-DM-025) with the following field mappings:
>
> | FinalResult Field | Source |
> |-------------------|--------|
> | `task` | `task` parameter |
> | `config` | `config` parameter |
> | `phase1` | `phase1_result` parameter |
> | `phase2_results` | `phase2_results` parameter |
> | `phase3` | `phase3_result` parameter (may be `None` if L=1) |
> | `final_solution` | The successfully executed test submission script, or the fallback validation solution (REQ-FN-025) |
> | `submission_path` | Absolute path to `./final/submission.csv` if produced, otherwise empty string |
> | `total_duration_seconds` | Wall-clock time from pipeline start to finalization completion |
> | `total_cost_usd` | Total API cost if tracked, otherwise `None` |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: The returned `FinalResult` shall have all required fields populated. `final_solution.phase` shall be `SolutionPhase.final` (or the original phase if fallback was used).
> - Source: REQ-DM-025
