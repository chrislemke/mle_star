# Software Requirements Specification: MLE-STAR Safety Agents

| Field | Value |
|-------|-------|
| Version | 0.1.0 |
| Date | 2026-02-20 |
| Status | Draft |
| Spec ID | 03 of 09 |
| Requirement Prefix | REQ-SF- |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Product Perspective](#2-product-perspective)
3. [A_debugger Requirements](#3-a_debugger-requirements)
4. [A_leakage Requirements](#4-a_leakage-requirements)
5. [A_data Requirements](#5-a_data-requirements)
6. [Cross-Cutting Requirements](#6-cross-cutting-requirements)
7. [Non-Functional Requirements](#7-non-functional-requirements)
8. [Constraints](#8-constraints)
9. [Traceability Matrix](#9-traceability-matrix)
10. [Change Control](#10-change-control)

---

## 1. Introduction

### 1.1 Purpose

This SRS defines three safety agents that guard solution quality across all pipeline phases: A_debugger (fixes execution errors), A_leakage (detects and corrects data leakage), and A_data (ensures all data sources are used). These agents are cross-cutting -- they are invoked by Phases 1, 2, and 3 rather than belonging to a single phase.

Intended audience: developers implementing the MLE-STAR system using the Claude Agent SDK for Python.

### 1.2 Scope

**Product name**: MLE-STAR (Machine Learning Engineering agent via Search and Targeted Refinement)

**What this spec covers**:
- A_debugger agent definition, input/output contract, retry logic, and fallback behavior
- A_leakage two-step detection and correction pipeline, structured output handling, and code block replacement
- A_data agent definition, response parsing, and integration point in Phase 1
- Agent configurations, prompt template usage, and SDK mapping for all three safety agents

**Out of scope**:
- Data model definitions (covered by Spec 01)
- Script execution and subprocess management (covered by Spec 02)
- Phase-specific orchestration logic that invokes these agents (covered by Specs 04-09)
- The agents' internal LLM reasoning (emergent from prompt + model)

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|------|-----------|
| SRS | Software Requirements Specification |
| MLE-STAR | ML Engineering agent with web Search and TArgeted code block Refinement |
| A_debugger | Debugger agent -- receives code + traceback, produces fixed code |
| A_leakage | Leakage checker agent -- detects and corrects data leakage in solutions |
| A_data | Data usage agent -- verifies all provided data sources are utilized |
| T_bug | Error traceback -- Python stack trace extracted from failed execution |
| c_data | Preprocessing code block -- the section of a solution that handles data preparation |
| Data leakage | Information from validation or test sets influencing the training process |
| Subsampling | Capping training data to a configured limit during refinement for faster iteration |

### 1.4 References

| ID | Title | Version | Source |
|----|-------|---------|--------|
| REF-01 | MLE-STAR paper | v3 | arXiv:2506.15692v3 |
| REF-02 | Claude Agent SDK reference | v0.1.39 | `claude-agent-sdk` PyPI |
| REF-03 | MLE-STAR architecture notes | -- | `thoughts/notes/mle_star_architecture.md` |
| REF-04 | MLE-STAR paper extraction | -- | `thoughts/notes/mle_star_paper.md` |
| REF-05 | Spec 01 -- Data Models and Interfaces | 0.1.0 | `thoughts/specs/01_data_models_and_interfaces.md` |
| REF-06 | Spec 02 -- Execution Harness | 0.1.0 | `thoughts/specs/02_execution_harness.md` |

### 1.5 Document Overview

- Section 3: A_debugger requirements (agent definition, retry loop, fallback, output validation)
- Section 4: A_leakage requirements (detection agent, correction agent, two-step orchestration, integration)
- Section 5: A_data requirements (agent definition, response parsing, integration point)
- Section 6: Cross-cutting requirements (agent configs, prompt templates, type contracts)
- Section 7: Non-functional requirements (performance, reliability, observability)
- Section 8: Constraints (technology, safety invariants)
- Section 9: Traceability matrix

---

## 2. Product Perspective

### 2.1 System Context

The three safety agents defined in this spec are cross-cutting concerns invoked at specific points throughout the pipeline. They do not constitute a pipeline phase themselves; instead, they are called by phases 1-3 to enforce correctness.

```
Spec 01 (Data Models)
  |-- SolutionScript (REQ-DM-009)
  |-- EvaluationResult (REQ-DM-021)
  |-- TaskDescription (REQ-DM-007)
  |-- LeakageAnswer (REQ-DM-018)
  |-- LeakageDetectionOutput (REQ-DM-019)
  |-- AgentType (REQ-DM-013)
  |-- AgentConfig (REQ-DM-036)
  |-- PromptRegistry (REQ-DM-032)
  |
Spec 02 (Execution Harness)
  |-- evaluate_solution (REQ-EX-015)
  |-- evaluate_with_retry (REQ-EX-021)
  |-- extract_traceback (REQ-EX-012)
  |-- detect_error (REQ-EX-013)
  |
  v
Spec 03 (this) -- Safety Agents
  |-- A_debugger: s <- A_debugger(s, T_bug)
  |-- A_leakage: c_data* <- A_leakage(c_data)
  |-- A_data:    s_0 <- A_data(s_0, T_task)
  |
  v
Used by: Spec 04 (Phase 1), Spec 05 (Phase 2 Outer),
         Spec 06 (Phase 2 Inner), Spec 07 (Phase 3),
         Spec 08 (Submission), Spec 09 (Orchestrator)
```

### 2.2 Product Functions Summary

1. Fix execution errors in agent-generated solutions via A_debugger
2. Detect data leakage in preprocessing code via A_leakage (structured detection)
3. Correct data leakage by rewriting preprocessing code via A_leakage (correction)
4. Verify that all provided data sources are used via A_data
5. Provide Claude Agent SDK `AgentDefinition`-compatible configurations for all three agents

### 2.3 Operating Environment

- **Runtime**: Python 3.10+
- **SDK**: `claude-agent-sdk` v0.1.39+
- **Validation library**: Pydantic v2 (for LeakageDetectionOutput structured output)
- **Execution**: Agents produce text (code blocks) or structured JSON; execution of fixed scripts delegates to Spec 02

### 2.4 Assumptions and Dependencies

| ID | Assumption | Impact if Invalid |
|----|-----------|-------------------|
| A-01 | LLM can reliably identify and fix Python errors given code + traceback | Debugging may loop to max attempts without resolving the error |
| A-02 | Data leakage manifests as preprocessing code that combines train+test statistics | Novel leakage patterns may evade detection |
| A-03 | Unused data sources are identifiable from the task description text | Ambiguous task descriptions may cause false positives |
| A-04 | A single code block replacement is sufficient to correct leakage | Multi-site leakage would require multiple correction passes |

| ID | Dependency | Owner | Risk if Unavailable |
|----|-----------|-------|---------------------|
| D-01 | Spec 01 types (SolutionScript, LeakageDetectionOutput, AgentConfig, etc.) | Spec 01 | Cannot construct agent inputs or parse outputs |
| D-02 | Spec 02 execution harness (evaluate_with_retry, extract_traceback) | Spec 02 | Cannot execute fixed scripts or extract tracebacks |
| D-03 | `claude-agent-sdk` v0.1.39+ | Anthropic | Cannot define or invoke agents |
| D-04 | PromptRegistry with templates for debugger, leakage, data agents | Spec 01 | Cannot construct agent prompts |

---

## 3. A_debugger Requirements

### 3.1 Agent Definition

> **REQ-SF-001**: *A_debugger Agent Definition* -- The system shall define an `AgentDefinition`-compatible configuration for the debugger agent with the following properties:
>
> | Property | Value |
> |----------|-------|
> | `agent_type` | `AgentType.debugger` |
> | `description` | Agent that fixes Python execution errors in solution scripts |
> | `prompt` | Rendered from the debugger template (Figure 19, REQ-DM-032) |
> | `tools` | `["Read", "Bash"]` |
> | `output_schema` | `None` (free-form code block response) |
> | `model` | `None` (inherit from orchestrator) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `AgentConfig(agent_type=AgentType.debugger).to_agent_definition()` shall produce a valid dictionary for `ClaudeAgentOptions.agents`.
> - Source: REF-01 Section 3.4, Figure 19

> **REQ-SF-002**: *A_debugger Prompt Template* -- The debugger agent prompt shall be constructed by rendering the Figure 19 template from the `PromptRegistry` (REQ-DM-032) with the following variables:
>
> | Variable | Type | Description |
> |----------|------|-------------|
> | `code` | `str` | Full source code of the failing solution (`SolutionScript.content`) |
> | `bug` | `str` | Error traceback extracted by `extract_traceback()` (REQ-EX-012) |
>
> - The rendered prompt shall include all instructions from Figure 19: do not remove subsampling, provide self-contained script, no additional headings, data in `./input/`, print `Final Validation Performance`, single code block response, no `exit()`.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 19

### 3.2 Input/Output Contract

> **REQ-SF-003**: *A_debugger Input Contract* -- The debugger agent shall accept two inputs:
>
> 1. `solution: SolutionScript` -- the solution that produced an execution error.
> 2. `traceback: str` -- the Python traceback string extracted from stderr.
>
> - Precondition: `solution.content` is non-empty and `traceback` is a non-empty string starting with `"Traceback (most recent call last):"`.
> - Error: If `traceback` is empty or `None`, the function shall raise `ValueError("No traceback provided for debugging")`.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Equation 10 -- `s <- A_debugger(s, T_bug)`

> **REQ-SF-004**: *A_debugger Output Contract* -- The debugger agent shall return a `SolutionScript` containing the fixed Python code extracted from the agent's response.
>
> - The response shall be parsed to extract a single code block (text between triple backticks or the entire response if no backticks are present).
> - The returned `SolutionScript` shall have `phase` set to the same phase as the input solution and `is_executable` set to `True` (optimistic; will be verified by re-execution).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given an agent response containing a fenced code block, the returned `SolutionScript.content` shall contain only the code within the fences (excluding the fence markers and any language identifier).

> **REQ-SF-005**: *A_debugger Code Block Extraction* -- The system shall define a function `extract_code_block(response: str) -> str` that extracts a Python code block from an agent response using the following rules:
>
> 1. If the response contains a fenced code block (`` ```python ... ``` `` or `` ``` ... ``` ``), extract the content between the outermost fence pair.
> 2. If multiple fenced code blocks exist, extract the **longest** one (by character count).
> 3. If no fenced code block exists, return the entire response stripped of leading/trailing whitespace.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `extract_code_block("```python\nprint('hello')\n```")` shall return `"print('hello')"`.
> - Rationale: All three safety agents return code blocks; this shared utility is used by A_debugger, A_leakage correction, and A_data.

### 3.3 Retry Loop

> **REQ-SF-006**: *A_debugger Retry Loop* -- The system shall define an async function `debug_solution(solution: SolutionScript, traceback: str, task: TaskDescription, config: PipelineConfig) -> tuple[SolutionScript, EvaluationResult]` that implements the debug retry loop:
>
> 1. Invoke the debugger agent with the solution and traceback to obtain a fixed `SolutionScript`.
> 2. Evaluate the fixed solution using `evaluate_solution()` (REQ-EX-015).
> 3. If evaluation produces an error and attempts remain (< `config.max_debug_attempts`):
>    a. Extract the new traceback from the evaluation result.
>    b. Invoke the debugger agent again with the new solution and new traceback.
>    c. Repeat from step 2.
> 4. Return the final `(SolutionScript, EvaluationResult)` pair.
>
> - The maximum number of debug attempts shall be `config.max_debug_attempts` (REQ-DM-001, default: 3).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a solution that fails once and is successfully fixed on the first debug attempt, the function shall return a successful `EvaluationResult` after exactly 1 debug invocation and 2 total evaluations.
> - Source: REF-01 Section 3.4 -- repeated until script executes successfully or max rounds reached

> **REQ-SF-007**: *A_debugger Integration with evaluate_with_retry* -- The `debug_solution` function (REQ-SF-006) shall be compatible with the `debug_callback` parameter of `evaluate_with_retry()` (REQ-EX-021). Specifically, the system shall provide a factory function `make_debug_callback(task: TaskDescription, config: PipelineConfig) -> Callable[[SolutionScript, str], Awaitable[SolutionScript]]` that wraps the debugger agent invocation (without the retry loop) so it can be passed directly to `evaluate_with_retry`.
>
> - The callback shall invoke the debugger agent exactly once per call and return the fixed `SolutionScript`.
> - The retry loop itself is managed by `evaluate_with_retry` (REQ-EX-021), not by the callback.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `evaluate_with_retry(solution, task, config, debug_callback=make_debug_callback(task, config))` shall invoke the debugger agent on each error and retry up to `config.max_debug_attempts` times.

### 3.4 Fallback Behavior

> **REQ-SF-008**: *A_debugger Fallback to Last Known Working Version* -- When all debug attempts are exhausted and the solution still fails (i.e., `EvaluationResult.is_error` is `True` after `config.max_debug_attempts` retries), the system shall fall back to the last known executable version of the solution.
>
> - The calling code shall maintain a reference to the last `SolutionScript` where `is_executable == True` and `score is not None`.
> - If no previous executable version exists (e.g., first-ever execution), the system shall return the failed solution with `is_executable=False` and `score=None`.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a solution history `[s_working, s_broken]` where debugging `s_broken` fails after all retries, the system shall return `s_working` as the active solution.
> - Source: REF-01 Section 3.4 -- "falls back to last known executable version if unresolved"

### 3.5 Output Validation

> **REQ-SF-009**: *A_debugger Subsampling Preservation* -- The debugger agent prompt (REQ-SF-002) shall include the instruction "Do not remove subsampling if exists." The system shall not perform automated verification of subsampling preservation; enforcement is via prompt instruction.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 19 -- "Do not remove subsampling if exists."

> **REQ-SF-010**: *A_debugger Output Score Line Validation* -- After extracting the fixed code from the debugger response, the system shall verify that the code contains the string `"Final Validation Performance"`. If this string is absent:
>
> 1. Log a warning indicating the score output line is missing from the debugged code.
> 2. Append the line `print(f"Final Validation Performance: {final_validation_score}")` to the end of the extracted code (before `if __name__` block if present, otherwise at the very end).
>
> - Priority: Should | Verify: Test | Release: MVP
> - Acceptance: Given debugged code that omits the print line, the returned `SolutionScript.content` shall contain `"Final Validation Performance"`.
> - Source: REF-01 Figure 19 -- "Remember to print a line with 'Final Validation Performance'"

---

## 4. A_leakage Requirements

### 4.1 Detection Agent Definition

> **REQ-SF-011**: *A_leakage Detection Agent Definition* -- The system shall define an `AgentDefinition`-compatible configuration for the leakage detection agent with the following properties:
>
> | Property | Value |
> |----------|-------|
> | `agent_type` | `AgentType.leakage` |
> | `description` | Agent that detects data leakage in solution preprocessing code |
> | `prompt` | Rendered from the leakage detection template (Figure 20, REQ-DM-034 variant `"detection"`) |
> | `tools` | `["Read"]` |
> | `output_schema` | `LeakageDetectionOutput` (REQ-DM-019) |
> | `model` | `None` (inherit from orchestrator) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Section 3.4, Figure 20

> **REQ-SF-012**: *A_leakage Detection Prompt Template* -- The leakage detection agent prompt shall be constructed by rendering the Figure 20 template from the `PromptRegistry` (REQ-DM-034, variant `"detection"`) with the following variable:
>
> | Variable | Type | Description |
> |----------|------|-------------|
> | `code` | `str` | Full source code of the solution (`SolutionScript.content`) |
>
> - The rendered prompt shall include all instructions from Figure 20: extract preprocessing code block, check model trains on training samples only, check validation samples are not used for training before printing score, detect data leakage, return JSON schema with `leakage_status` and `code_block`.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 20

### 4.2 Detection Input/Output Contract

> **REQ-SF-013**: *A_leakage Detection Input Contract* -- The leakage detection agent shall accept a single input:
>
> 1. `solution: SolutionScript` -- the solution to check for data leakage.
>
> - Precondition: `solution.content` is non-empty.
> - Priority: Must | Verify: Test | Release: MVP

> **REQ-SF-014**: *A_leakage Detection Output Contract* -- The leakage detection agent shall return a `LeakageDetectionOutput` (REQ-DM-019) parsed from the agent's structured JSON response.
>
> - The output contains a list of `LeakageAnswer` objects (REQ-DM-018), each with:
>   - `leakage_status`: one of `"Yes Data Leakage"` or `"No Data Leakage"`
>   - `code_block`: the exact preprocessing code block extracted from the solution
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a solution with leakage, the response shall contain at least one `LeakageAnswer` with `leakage_status == "Yes Data Leakage"` and `code_block` matching an exact substring of `solution.content`.
> - Source: REF-01 Figure 20 -- `Answer = {'leakage_status': str, 'code_block': str}`

> **REQ-SF-015**: *A_leakage Detection Structured Output Usage* -- The leakage detection agent shall use the Claude Agent SDK's `output_format` parameter set to `{"type": "json_schema", "schema": LeakageDetectionOutput.model_json_schema()}` to ensure the response conforms to the `LeakageDetectionOutput` schema (REQ-DM-019).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: The agent response shall be parseable via `LeakageDetectionOutput.model_validate_json(response)` without errors.
> - Source: REF-01 Figure 20, REF-02 Section 12 (Structured Outputs)

### 4.3 Correction Agent Definition

> **REQ-SF-016**: *A_leakage Correction Agent Definition* -- The system shall define an `AgentDefinition`-compatible configuration for the leakage correction agent with the following properties:
>
> | Property | Value |
> |----------|-------|
> | `agent_type` | `AgentType.leakage` (shared with detection) |
> | `description` | Agent that corrects data leakage in solution preprocessing code |
> | `prompt` | Rendered from the leakage correction template (Figure 21, REQ-DM-034 variant `"correction"`) |
> | `tools` | `["Read"]` |
> | `output_schema` | `None` (free-form code block response) |
> | `model` | `None` (inherit from orchestrator) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Section 3.4, Figure 21

> **REQ-SF-017**: *A_leakage Correction Prompt Template* -- The leakage correction agent prompt shall be constructed by rendering the Figure 21 template from the `PromptRegistry` (REQ-DM-034, variant `"correction"`) with the following variable:
>
> | Variable | Type | Description |
> |----------|------|-------------|
> | `code` | `str` | Full source code of the solution (`SolutionScript.content`) |
>
> - The rendered prompt shall include all instructions from Figure 21: ensure model trains on training samples only, ensure validation samples are not used before printing score, refine code to prevent leakage, return single code block, note that variables are defined earlier.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 21

### 4.4 Correction Input/Output Contract

> **REQ-SF-018**: *A_leakage Correction Input Contract* -- The leakage correction agent shall accept a single input:
>
> 1. `solution: SolutionScript` -- the solution containing detected data leakage.
>
> - Precondition: At least one `LeakageAnswer` with `leakage_status == "Yes Data Leakage"` was returned by the detection step.
> - Priority: Must | Verify: Test | Release: MVP

> **REQ-SF-019**: *A_leakage Correction Output Contract* -- The leakage correction agent shall return a corrected code block extracted from the agent's response using `extract_code_block()` (REQ-SF-005).
>
> - The returned code block shall be a corrected version of the preprocessing code, suitable for replacing the original leaky code block in the solution.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: The returned code block shall be a non-empty string that does not match the original leaky `code_block` from the detection step.

### 4.5 Two-Step Orchestration

> **REQ-SF-020**: *A_leakage Two-Step Pipeline* -- The system shall define an async function `check_and_fix_leakage(solution: SolutionScript) -> SolutionScript` that orchestrates the two-step leakage detection and correction process:
>
> 1. Invoke the leakage detection agent with `solution` to obtain a `LeakageDetectionOutput`.
> 2. For each `LeakageAnswer` in the output where `leakage_status == "Yes Data Leakage"`:
>    a. Invoke the leakage correction agent with the current `solution`.
>    b. Extract the corrected code block from the correction response.
>    c. Replace the leaky code block in the solution using `SolutionScript.replace_block(answer.code_block, corrected_block)` (REQ-DM-010).
>    d. Update `solution` to the result of the replacement.
> 3. If no leakage was detected (all answers have `leakage_status == "No Data Leakage"`), return the original solution unchanged.
> 4. Return the final corrected solution.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a solution with leakage, the function shall return a `SolutionScript` where the leaky code block has been replaced with corrected code.
> - Source: REF-01 Section 3.4 -- `c_data* = A_leakage(c_data)`, then `s <- s.replace(c_data, c_data*)`

> **REQ-SF-021**: *A_leakage Code Block Replacement* -- When replacing a leaky code block, the `check_and_fix_leakage` function shall use `SolutionScript.replace_block(old, new)` (REQ-DM-010) where `old` is the `code_block` from the `LeakageAnswer` and `new` is the corrected code block from the correction agent.
>
> - If `replace_block` raises `ValueError` (the original code block is not found in the solution content), the system shall log a warning and skip the replacement for that answer, returning the solution unchanged for that particular leakage finding.
> - Priority: Must | Verify: Test | Release: MVP
> - Rationale: The detection agent may extract a code block that has minor whitespace differences from the actual solution content; graceful handling prevents crashes.

### 4.6 Integration Points

> **REQ-SF-022**: *A_leakage Runs Before Every Evaluation* -- The leakage checker shall be invoked on every generated or modified `SolutionScript` before it is evaluated, across all pipeline phases (Phase 1, Phase 2, Phase 3).
>
> - The calling phase is responsible for invoking `check_and_fix_leakage(solution)` and using the returned (potentially corrected) solution for evaluation.
> - This spec defines the function; Specs 04-07 define the invocation points.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Acceptance: Every path through the pipeline that leads to `evaluate_solution()` (REQ-EX-015) shall have a preceding call to `check_and_fix_leakage()`.
> - Source: REF-01 Section 3.4 -- "Every generated solution before evaluation (all phases)"

> **REQ-SF-023**: *A_leakage Evidence of Necessity* -- The leakage checker is critical for preventing overfitting to validation data. Without the leakage checker:
>
> - Validation score may be inflated (e.g., +5.0% on spaceship-titanic)
> - Test score may degrade severely (e.g., -8.9% on spaceship-titanic)
>
> This requirement documents the empirical justification; no implementation action is needed.
>
> - Priority: Informational | Verify: N/A | Release: N/A
> - Source: REF-01 Table 5 -- ablation study results

---

## 5. A_data Requirements

### 5.1 Agent Definition

> **REQ-SF-024**: *A_data Agent Definition* -- The system shall define an `AgentDefinition`-compatible configuration for the data usage agent with the following properties:
>
> | Property | Value |
> |----------|-------|
> | `agent_type` | `AgentType.data` |
> | `description` | Agent that verifies all provided data sources are used in the solution |
> | `prompt` | Rendered from the data template (Figure 22, REQ-DM-032) |
> | `tools` | `["Read"]` |
> | `output_schema` | `None` (free-form response: either code block or confirmation text) |
> | `model` | `None` (inherit from orchestrator) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Section 3.4, Figure 22

> **REQ-SF-025**: *A_data Prompt Template* -- The data usage agent prompt shall be constructed by rendering the Figure 22 template from the `PromptRegistry` (REQ-DM-032) with the following variables:
>
> | Variable | Type | Description |
> |----------|------|-------------|
> | `initial solution` | `str` | Full source code of the initial solution (`SolutionScript.content`) |
> | `task description` | `str` | Task description text (`TaskDescription.description`) |
>
> - The rendered prompt shall include all instructions from Figure 22: incorporate unused information, do not bypass with try-except, examine task description for extraction guidance, preserve `Final Validation Performance` print line, and the two response format options.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 22

### 5.2 Input/Output Contract

> **REQ-SF-026**: *A_data Input Contract* -- The data usage agent shall accept two inputs:
>
> 1. `solution: SolutionScript` -- the initial solution to check for data utilization.
> 2. `task: TaskDescription` -- the task description providing context about available data sources.
>
> - Precondition: `solution.content` is non-empty and `task.description` is non-empty.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Equation 11 -- `s_0 <- A_data(s_0, T_task)`

> **REQ-SF-027**: *A_data Output Contract* -- The data usage agent shall return either:
>
> 1. A revised `SolutionScript` with updated code that incorporates previously unused data sources, **or**
> 2. The original `SolutionScript` unchanged, if all data sources are already used.
>
> - The system shall determine which case applies based on response parsing (REQ-SF-028).
> - Priority: Must | Verify: Test | Release: MVP

### 5.3 Response Parsing

> **REQ-SF-028**: *A_data Response Parsing* -- The system shall define a function `parse_data_agent_response(response: str, original_solution: SolutionScript) -> SolutionScript` that distinguishes between the two response formats:
>
> 1. **Confirmation**: If the response contains the exact phrase `"All the provided information is used."` (case-insensitive match), return the `original_solution` unchanged.
> 2. **Revised code**: Otherwise, extract the code block using `extract_code_block()` (REQ-SF-005) and return a new `SolutionScript` with the extracted code as `content`, preserving the original solution's `phase` and other metadata.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `parse_data_agent_response("All the provided information is used.", s)` shall return `s` unchanged.
> - Acceptance: `parse_data_agent_response("```python\nimport pandas as pd\n...\n```", s)` shall return a new `SolutionScript` with the extracted code.
> - Source: REF-01 Figure 22 -- two response format options

### 5.4 No try/except Enforcement

> **REQ-SF-029**: *A_data No try/except Instruction* -- The data agent prompt (REQ-SF-025) shall include the explicit instructions "DO NOT USE TRY AND EXCEPT; just occur error so we can debug it!" This ensures that errors from incorporating new data sources are surfaced to the debugger agent rather than silently caught.
>
> - This is enforced via prompt instruction. The system shall not perform automated try/except detection for A_data specifically; the general advisory detection in REQ-EX-045 applies.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 22

### 5.5 Integration Point

> **REQ-SF-030**: *A_data Runs Once After Initial Solution Generation* -- The data usage agent shall be invoked exactly once per pipeline execution, after the initial solution `s_0` is generated in Phase 1 (after merging, before Phase 2 refinement begins).
>
> - The calling code in Phase 1 (Spec 04) is responsible for the invocation.
> - This spec defines the function `check_data_usage(solution: SolutionScript, task: TaskDescription) -> SolutionScript`; Spec 04 defines the invocation point.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Section 3.4 -- "After initial solution generation (Phase 1 only)"

> **REQ-SF-031**: *A_data Evidence of Necessity* -- The data usage agent is critical for tasks with auxiliary data files. Without the data usage agent:
>
> - Performance on competitions with non-CSV data sources (e.g., `.xyz` geometry files in nomad2018) may be significantly degraded.
>
> This requirement documents the empirical justification; no implementation action is needed.
>
> - Priority: Informational | Verify: N/A | Release: N/A
> - Source: REF-01 Table 6 -- ablation study results

---

## 6. Cross-Cutting Requirements

### 6.1 Agent Configurations

> **REQ-SF-032**: *Safety Agent Default Configs* -- The `build_default_agent_configs()` function (REQ-DM-040) shall include `AgentConfig` entries for all three safety agents:
>
> | AgentType | Tools | Output Schema | Model |
> |-----------|-------|---------------|-------|
> | `debugger` | `["Read", "Bash"]` | `None` | `None` |
> | `leakage` | `["Read"]` | `LeakageDetectionOutput` (detection variant) / `None` (correction variant) | `None` |
> | `data` | `["Read"]` | `None` | `None` |
>
> - The leakage agent has two operational modes (detection and correction) sharing the same `AgentType.leakage`. The `AgentConfig` stored in the defaults shall be for the detection variant (with `output_schema`). The correction variant config shall be constructable by setting `output_schema=None` on a copy.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `build_default_agent_configs()[AgentType.debugger]` shall return an `AgentConfig` with `tools=["Read", "Bash"]`.
> - Source: REQ-DM-040

### 6.2 Prompt Template Usage

> **REQ-SF-033**: *Safety Agent Prompt Templates from Registry* -- All three safety agents shall obtain their prompt templates from the `PromptRegistry` (REQ-DM-032):
>
> | Agent | Registry Key | Variant | Figure |
> |-------|-------------|---------|--------|
> | A_debugger | `AgentType.debugger` | (default) | Figure 19 |
> | A_leakage detection | `AgentType.leakage` | `"detection"` | Figure 20 |
> | A_leakage correction | `AgentType.leakage` | `"correction"` | Figure 21 |
> | A_data | `AgentType.data` | (default) | Figure 22 |
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REQ-DM-032, REQ-DM-034

### 6.3 Type Contracts

> **REQ-SF-034**: *Safety Agents Operate on SolutionScript* -- All three safety agent functions shall accept `SolutionScript` (REQ-DM-009) as input and produce `SolutionScript` as output. This ensures uniform integration with the pipeline:
>
> | Function | Signature |
> |----------|-----------|
> | `debug_solution` | `(SolutionScript, str, TaskDescription, PipelineConfig) -> tuple[SolutionScript, EvaluationResult]` |
> | `check_and_fix_leakage` | `(SolutionScript) -> SolutionScript` |
> | `check_data_usage` | `(SolutionScript, TaskDescription) -> SolutionScript` |
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Rationale: Uniform type contracts allow safety agents to be composed and chained in any order.

> **REQ-SF-035**: *Safety Agents Produce New SolutionScript Instances* -- Safety agent functions shall return new `SolutionScript` instances rather than mutating the input. The original `SolutionScript` shall remain unchanged after a safety agent call.
>
> - Exception: If no change is needed (e.g., no leakage detected, all data used), the function may return the original instance.
> - Priority: Must | Verify: Test | Release: MVP
> - Rationale: Immutable data flow supports fallback to previous versions (REQ-SF-008).

---

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
