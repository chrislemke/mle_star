# SRS 03 — Safety Modules: Debugger Agent

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
