# Software Requirements Specification: MLE-STAR Submission and Finalization

| Field | Value |
|-------|-------|
| Version | 0.1.0 |
| Date | 2026-02-20 |
| Status | Draft |
| Spec ID | 08 of 09 |
| Requirement Prefix | REQ-FN- |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Product Perspective](#2-product-perspective)
3. [Subsampling Removal Requirements](#3-subsampling-removal-requirements)
4. [A_test Requirements](#4-a_test-requirements)
5. [Execution and Verification Requirements](#5-execution-and-verification-requirements)
6. [Data Contamination Check Requirements](#6-data-contamination-check-requirements)
7. [Finalization Orchestration Requirements](#7-finalization-orchestration-requirements)
8. [Non-Functional Requirements](#8-non-functional-requirements)
9. [Constraints](#9-constraints)
10. [Traceability Matrix](#10-traceability-matrix)
11. [Change Control](#11-change-control)

---

## 1. Introduction

### 1.1 Purpose

This SRS defines the finalization process that runs after the main pipeline phases (Phase 1, Phase 2, Phase 3) complete. It covers subsampling removal so the model trains on full data, test submission script generation via A_test, final script execution and submission file verification, and an optional data contamination check. Together these steps produce the final `submission.csv` and the `FinalResult` record.

Intended audience: developers implementing the MLE-STAR system using the Claude Agent SDK for Python.

### 1.2 Scope

**Product name**: MLE-STAR (Machine Learning Engineering agent via Search and Targeted Refinement)

**What this spec covers**:
- Subsampling extraction from the final solution script (Figure 26)
- Subsampling removal to restore full training data usage (Figure 27)
- Code block replacement to integrate the de-subsampled code into the solution
- A_test agent definition, input/output contract, and prompt (Figure 25)
- Execution of the test submission script and verification of `./final/submission.csv`
- Error handling and debug retry for the test submission script
- Optional data contamination check against reference Kaggle discussions (Figure 28)
- `run_finalization()` orchestration function
- `FinalResult` construction

**Out of scope**:
- Data model definitions (covered by Spec 01)
- Script execution and subprocess management (covered by Spec 02)
- A_debugger behavior logic (covered by Spec 03; referenced here for error handling)
- Phase 1, 2, 3 logic that produces the final solution input (covered by Specs 04-07)
- Top-level pipeline orchestration (covered by Spec 09)

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|------|-----------|
| SRS | Software Requirements Specification |
| MLE-STAR | ML Engineering agent with web Search and TArgeted code block Refinement |
| A_test | Test submission agent -- transforms a validation solution into a test submission script |
| Subsampling | Capping training data to a configured limit during refinement for faster iteration |
| Subsampling removal | Removing the subsampling code so the model trains on the full dataset before final submission |
| Data contamination | A solution being copied or heavily derived from publicly available Kaggle discussion posts |
| FinalResult | The comprehensive output record of the entire pipeline run |

### 1.4 References

| ID | Title | Version | Source |
|----|-------|---------|--------|
| REF-01 | MLE-STAR paper | v3 | arXiv:2506.15692v3 |
| REF-02 | Claude Agent SDK reference | v0.1.39 | `claude-agent-sdk` PyPI |
| REF-03 | MLE-STAR architecture notes | -- | `thoughts/notes/mle_star_architecture.md` |
| REF-04 | MLE-STAR paper extraction | -- | `thoughts/notes/mle_star_paper.md` |
| REF-05 | Spec 01 -- Data Models and Interfaces | 0.1.0 | `thoughts/specs/01_data_models_and_interfaces.md` |
| REF-06 | Spec 02 -- Execution Harness | 0.1.0 | `thoughts/specs/02_execution_harness.md` |
| REF-07 | Spec 03 -- Safety Agents | 0.1.0 | `thoughts/specs/03_safety_modules.md` |

### 1.5 Document Overview

- Section 3: Subsampling removal requirements (extraction, removal, code block replacement)
- Section 4: A_test requirements (agent definition, input/output contract, prompt template)
- Section 5: Execution and verification requirements (script execution, submission file checks, error handling)
- Section 6: Data contamination check requirements (agent definition, structured output, optional invocation)
- Section 7: Finalization orchestration requirements (`run_finalization()` function, FinalResult construction)
- Section 8: Non-functional requirements (performance, reliability, observability)
- Section 9: Constraints (technology, submission invariants)
- Section 10: Traceability matrix

---

## 2. Product Perspective

### 2.1 System Context

This spec defines the final stage of the MLE-STAR pipeline. It receives the best solution from Phase 2 (or Phase 3 if ensembling was used), removes subsampling, generates a test submission script, executes it, verifies the submission file, and optionally checks for data contamination.

```
Spec 01 (Data Models)
  |-- SolutionScript (REQ-DM-009)
  |-- SolutionScript.replace_block() (REQ-DM-010)
  |-- TaskDescription (REQ-DM-007)
  |-- PipelineConfig (REQ-DM-001)
  |-- DataContaminationResult (REQ-DM-020)
  |-- FinalResult (REQ-DM-025)
  |-- AgentType (REQ-DM-013)
  |-- AgentConfig (REQ-DM-036)
  |-- PromptRegistry (REQ-DM-032)
  |
Spec 02 (Execution Harness)
  |-- evaluate_solution (REQ-EX-015)
  |-- evaluate_with_retry (REQ-EX-021)
  |-- verify_submission (REQ-EX-024)
  |-- get_submission_info (REQ-EX-025)
  |-- request_subsample_extraction (REQ-EX-020)
  |-- request_subsample_removal (REQ-EX-019)
  |
Spec 03 (Safety Agents)
  |-- debug_solution / make_debug_callback (REQ-SF-006, REQ-SF-007)
  |-- check_and_fix_leakage (REQ-SF-020, REQ-SF-022)
  |-- extract_code_block (REQ-SF-005)
  |
  v
Spec 08 (this) -- Submission and Finalization
  |-- Subsampling removal: extract -> remove -> replace
  |-- A_test: generate test submission script
  |-- Execute and verify submission
  |-- Optional: data contamination check
  |-- Construct FinalResult
  |
  v
Used by: Spec 09 (Orchestrator calls run_finalization)
```

### 2.2 Finalization Flow

```
Final solution (from Phase 2/3)
    |
    v
Subsampling extraction (Figure 26) -> subsampling code block
    |
    v
Subsampling removal (Figure 27) -> code block without subsampling
    |
    v
Replace in solution (SolutionScript.replace_block)
    |
    v
A_test(task_description, solution_without_subsampling) -> test submission script
    |
    v
Execute test submission script -> verify ./final/submission.csv
    |
    v
Optional: A_leakage check (REQ-SF-022) before evaluation
    |
    v
Optional: Data contamination check (Figure 28)
    |
    v
FinalResult construction
```

### 2.3 Product Functions Summary

1. Remove subsampling code from the final solution so the model trains on the full dataset
2. Generate a test submission script that loads test data and produces `./final/submission.csv`
3. Execute the test submission script and verify the output file
4. Handle execution errors via A_debugger with retry
5. Optionally check for data contamination against reference Kaggle discussions
6. Construct the `FinalResult` record capturing the entire pipeline run

### 2.4 Operating Environment

- **Runtime**: Python 3.10+
- **SDK**: `claude-agent-sdk` v0.1.39+
- **Validation library**: Pydantic v2 (for `DataContaminationResult` structured output)
- **Execution**: Test submission script executes via Spec 02 harness; output to `./final/submission.csv`

### 2.5 Assumptions and Dependencies

| ID | Assumption | Impact if Invalid |
|----|-----------|-------------------|
| A-01 | The final solution from Phase 2/3 is a valid, executable Python script | Finalization will fail at the execution step; A_debugger retry applies |
| A-02 | Subsampling code, if present, is identifiable as a contiguous code block | Extraction agent may fail to locate subsampling code; passthrough applies |
| A-03 | Test data is available in the `./input/` directory | Test submission script will fail at data loading |
| A-04 | The LLM can reliably transform a validation solution into a test submission script | A_test may produce a script that fails execution; debug retry applies |
| A-05 | Reference Kaggle discussions are available as text for contamination checking | Contamination check is skipped when no references are provided |

| ID | Dependency | Owner | Risk if Unavailable |
|----|-----------|-------|---------------------|
| D-01 | Spec 01 types (SolutionScript, FinalResult, DataContaminationResult, etc.) | Spec 01 | Cannot construct inputs or outputs |
| D-02 | Spec 02 execution harness (evaluate_solution, verify_submission, etc.) | Spec 02 | Cannot execute or verify scripts |
| D-03 | Spec 03 safety agents (A_debugger, A_leakage, extract_code_block) | Spec 03 | Cannot debug errors or check leakage |
| D-04 | `claude-agent-sdk` v0.1.39+ | Anthropic | Cannot define or invoke agents |
| D-05 | PromptRegistry with templates for A_test, subsampling, contamination | Spec 01 | Cannot construct agent prompts |

---

## 3. Subsampling Removal Requirements

### 3.1 Subsampling Extraction Agent

> **REQ-FN-001**: *Subsampling Extraction Agent Prompt* -- The system shall define an agent invocation that uses the subsampling extraction template (Figure 26, REQ-DM-035 variant `"subsampling_extract"`) from the `PromptRegistry` with the following variable:
>
> | Variable | Type | Description |
> |----------|------|-------------|
> | `final solution` | `str` | Full source code of the final solution (`SolutionScript.content`) |
>
> - The rendered prompt shall instruct the agent to: extract the code block where subsampling of training samples is used, return exactly the extracted code block from the Python script, and format the response as a single markdown code block.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 26

> **REQ-FN-002**: *Subsampling Extraction Input Contract* -- The subsampling extraction agent shall accept a single input:
>
> 1. `solution: SolutionScript` -- the final solution from which to extract subsampling code.
>
> - Precondition: `solution.content` is non-empty.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Figure 26

> **REQ-FN-003**: *Subsampling Extraction Output Contract* -- The subsampling extraction agent shall return a string containing the extracted subsampling code block.
>
> - The response shall be parsed using `extract_code_block()` (REQ-SF-005) to obtain the code block text.
> - The extracted code block shall be an exact substring of `solution.content`.
> - If the agent cannot find subsampling code (the extracted block is not a substring of the solution), the system shall treat this as a no-subsampling case (REQ-FN-008).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a solution containing `df_train = df_train.sample(n=30000, random_state=42)`, the extracted code block shall include that line and its immediate context.

### 3.2 Subsampling Removal Agent

> **REQ-FN-004**: *Subsampling Removal Agent Prompt* -- The system shall define an agent invocation that uses the subsampling removal template (Figure 27, REQ-DM-035 variant `"subsampling_remove"`) from the `PromptRegistry` with the following variable:
>
> | Variable | Type | Description |
> |----------|------|-------------|
> | `code block with subsampling` | `str` | The code block extracted by the subsampling extraction step (REQ-FN-003) |
>
> - The rendered prompt shall instruct the agent to: remove the subsampling and use full training samples, not introduce dummy variables (since actual data variables are defined earlier), and format the response as a single markdown code block.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 27

> **REQ-FN-005**: *Subsampling Removal Input Contract* -- The subsampling removal agent shall accept a single input:
>
> 1. `subsampling_block: str` -- the code block containing subsampling logic, as extracted by REQ-FN-003.
>
> - Precondition: `subsampling_block` is a non-empty string containing identifiable subsampling logic.
> - Priority: Must | Verify: Test | Release: MVP

> **REQ-FN-006**: *Subsampling Removal Output Contract* -- The subsampling removal agent shall return a string containing the modified code block with subsampling removed.
>
> - The response shall be parsed using `extract_code_block()` (REQ-SF-005) to obtain the replacement code block.
> - The returned code block shall not contain subsampling logic (e.g., `.sample()`, `.head()`, array slicing for size reduction) that was present in the input.
> - The returned code block shall not introduce new variable definitions that did not exist in the surrounding context.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given input `"df_train = df_train.sample(n=30000, random_state=42)\nmodel.fit(df_train)"`, the output shall be equivalent to `"model.fit(df_train)"` (or the block with only the subsampling line removed).

### 3.3 Code Block Replacement

> **REQ-FN-007**: *Subsampling Code Block Replacement* -- After obtaining the original subsampling block (REQ-FN-003) and the replacement block without subsampling (REQ-FN-006), the system shall replace the subsampling code in the solution using `SolutionScript.replace_block(old, new)` (REQ-DM-010), where `old` is the extracted subsampling block and `new` is the de-subsampled replacement.
>
> - If `replace_block` raises `ValueError` (the original code block is not found in the solution), the system shall log a warning and return the original solution unchanged.
> - The returned `SolutionScript` shall have the same `phase` and metadata as the input, but with updated `content`.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given solution content `"A\ndf = df.sample(30000)\nB"` and replacement `"B"` for block `"df = df.sample(30000)\nB"`, the result content shall be `"A\nB"`.
> - Source: REF-01 Figures 26-27 -- extract, remove, replace

### 3.4 No-Subsampling Passthrough

> **REQ-FN-008**: *No-Subsampling Passthrough* -- If the subsampling extraction agent determines that the solution contains no subsampling code, the system shall pass the solution through unchanged to the A_test step.
>
> - Detection: The extraction agent returns an empty code block, a block that is not a substring of the solution, or an explicit indication that no subsampling was found.
> - The system shall log an informational message indicating no subsampling was detected.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a solution that never subsamples training data, `remove_subsampling(solution)` shall return the original solution unchanged.

### 3.5 Subsampling Removal Function

> **REQ-FN-009**: *remove_subsampling() Function* -- The system shall define an async function `remove_subsampling(solution: SolutionScript) -> SolutionScript` that orchestrates the full subsampling removal process:
>
> 1. Invoke the subsampling extraction agent (REQ-FN-001) with the solution to obtain the subsampling code block.
> 2. Parse the extraction response using `extract_code_block()` (REQ-SF-005).
> 3. Verify the extracted block is a non-empty substring of `solution.content`. If not, return the solution unchanged (REQ-FN-008).
> 4. Invoke the subsampling removal agent (REQ-FN-004) with the extracted block to obtain the replacement.
> 5. Parse the removal response using `extract_code_block()` (REQ-SF-005).
> 6. Replace the original block using `SolutionScript.replace_block()` (REQ-FN-007).
> 7. Return the updated solution.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a solution with subsampling, `remove_subsampling(solution)` shall return a solution whose `content` no longer contains the subsampling logic.
> - Source: REF-01 Section 4, Figures 26-27

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
