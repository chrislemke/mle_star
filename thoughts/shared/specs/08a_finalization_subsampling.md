# SRS 08 — Finalization: Subsampling Removal

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
