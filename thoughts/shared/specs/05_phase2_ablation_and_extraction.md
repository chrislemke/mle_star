# Software Requirements Specification: MLE-STAR Phase 2 -- Ablation Study and Code Block Extraction (Outer Loop)

| Field | Value |
|-------|-------|
| Version | 0.1.0 |
| Date | 2026-02-20 |
| Status | Draft |
| Spec ID | 05 of 09 |
| Requirement Prefix | REQ-P2O- |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Product Perspective](#2-product-perspective)
3. [A_abl Requirements](#3-a_abl-requirements)
4. [A_summarize Requirements](#4-a_summarize-requirements)
5. [A_extractor Requirements](#5-a_extractor-requirements)
6. [Outer Loop Control Flow Requirements](#6-outer-loop-control-flow-requirements)
7. [Non-Functional Requirements](#7-non-functional-requirements)
8. [Constraints](#8-constraints)
9. [Traceability Matrix](#9-traceability-matrix)
10. [Change Control](#10-change-control)

---

## 1. Introduction

### 1.1 Purpose

This SRS defines the outer loop of Phase 2 refinement in the MLE-STAR pipeline: performing ablation studies to identify the most impactful code components, summarizing ablation results, and extracting the target code block for refinement. It covers agents A_abl, A_summarize, and A_extractor, as well as the outer loop control flow from Algorithm 2 (lines 4-8 and 26-28).

Intended audience: developers implementing the MLE-STAR system using the Claude Agent SDK for Python.

### 1.2 Scope

**Product name**: MLE-STAR (Machine Learning Engineering agent via Search and Targeted Refinement)

**What this spec covers**:
- A_abl agent definition, prompt template, input/output contract, and ablation scope requirements
- A_summarize agent definition, prompt template, and input/output contract
- A_extractor agent definition, prompt template, structured output, and code block validation
- Ablation script execution and error handling (delegation to Spec 02 and Spec 03)
- Outer loop state management (accumulation of ablation summaries T_abl and refined code blocks C)
- Outer loop iteration control (T steps from PipelineConfig)
- Handoff to inner loop (Spec 06) and post-inner-loop state update
- Phase2Result construction at the end of the outer loop

**Out of scope**:
- Data model definitions (covered by Spec 01)
- Script execution and subprocess management (covered by Spec 02)
- Safety agent behavior (A_debugger, A_leakage) (covered by Spec 03)
- Phase 1 initial solution generation (covered by Spec 04)
- Inner loop refinement (A_planner, A_coder) (covered by Spec 06)
- Ensemble (covered by Spec 07)
- Orchestration coordination across phases (covered by Spec 09)

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|------|-----------|
| SRS | Software Requirements Specification |
| MLE-STAR | ML Engineering agent with web Search and TArgeted code block Refinement |
| A_abl | Ablation agent -- generates ablation study code to test component impact |
| A_summarize | Summarization agent -- condenses raw ablation output into a concise textual summary |
| A_extractor | Extractor agent -- identifies the most impactful code block and proposes a refinement plan |
| T_abl | Ablation summary -- accumulated text summaries of ablation study results across outer steps |
| T_abl^t | Ablation summary for outer step t |
| c_t | Code block -- the exact code substring extracted for refinement at outer step t |
| p_0 | Initial plan -- the natural language refinement plan proposed by A_extractor |
| C | Set of previously refined code blocks accumulated across outer steps |
| s_t | Current best solution at outer step t |
| s_0 | Initial solution from Phase 1 |
| T | Number of outer loop steps (from PipelineConfig.outer_loop_steps, default: 4) |
| K | Number of inner loop steps per code block (from PipelineConfig.inner_loop_steps) |

### 1.4 References

| ID | Title | Version | Source |
|----|-------|---------|--------|
| REF-01 | MLE-STAR paper | v3 | arXiv:2506.15692v3 |
| REF-02 | Claude Agent SDK reference | v0.1.39 | `claude-agent-sdk` PyPI |
| REF-03 | MLE-STAR architecture notes | -- | `thoughts/notes/mle_star_architecture.md` |
| REF-04 | MLE-STAR paper extraction | -- | `thoughts/notes/mle_star_paper.md` |
| REF-05 | Spec 01 -- Data Models and Interfaces | 0.1.0 | `thoughts/specs/01_data_models_and_interfaces.md` |
| REF-06 | Spec 02 -- Execution Harness | 0.1.0 | `thoughts/specs/02_execution_harness.md` |
| REF-07 | Spec 03 -- Safety Modules | 0.1.0 | `thoughts/specs/03_safety_modules.md` |

### 1.5 Document Overview

- Section 3: A_abl requirements (agent definition, prompt, input/output, ablation scope)
- Section 4: A_summarize requirements (agent definition, prompt, input/output)
- Section 5: A_extractor requirements (agent definition, prompt, structured output, code block validation)
- Section 6: Outer loop control flow requirements (state management, iteration, handoff, result construction)
- Section 7: Non-functional requirements (performance, reliability, observability)
- Section 8: Constraints (technology, algorithmic invariants)
- Section 9: Traceability matrix

---

## 2. Product Perspective

### 2.1 System Context

This spec defines the outer loop of Phase 2 -- the targeting mechanism that identifies which code block to refine in each iteration. The outer loop runs T times, and each iteration produces a code block and plan that are handed to the inner loop (Spec 06) for actual refinement.

```
Spec 04 (Phase 1)
  |-- initial solution s_0, initial score h_best
  |
  v
Spec 05 (this) -- Phase 2 Outer Loop
  for t = 0 to T-1:
    |
    |-- A_abl(s_t, T_abl)         -> ablation study code a_t
    |-- exec(a_t)                  -> raw output r_t
    |-- A_summarize(a_t, r_t)     -> summary T_abl^t
    |-- A_extractor(T_abl^t, s_t, C) -> code block c_t, plan p_0
    |
    |   +----> Spec 06 (Phase 2 Inner Loop)
    |   |        refine c_t with K strategies
    |   |        return best refined solution
    |   <----+
    |
    |-- update s_t, T_abl, C
  end for
  |
  v
Phase2Result -> Spec 07 (Phase 3 Ensemble) or Spec 08 (Submission)
```

### 2.2 Dependency Diagram

```
Spec 01 (Data Models)
  |-- SolutionScript (REQ-DM-009)
  |-- CodeBlock (REQ-DM-012)
  |-- RefinePlan (REQ-DM-016)
  |-- ExtractorOutput (REQ-DM-017)
  |-- Phase2Result (REQ-DM-023)
  |-- RefinementAttempt (REQ-DM-042)
  |-- AgentType (REQ-DM-013)
  |-- AgentConfig (REQ-DM-036)
  |-- PromptRegistry (REQ-DM-032)
  |-- PipelineConfig (REQ-DM-001)
  |
Spec 02 (Execution Harness)
  |-- execute_script (REQ-EX-007)
  |-- parse_score (REQ-EX-011)
  |-- evaluate_solution (REQ-EX-015)
  |-- evaluate_with_retry (REQ-EX-021)
  |
Spec 03 (Safety Modules)
  |-- debug_solution / make_debug_callback (REQ-SF-006, REQ-SF-007)
  |-- check_and_fix_leakage (REQ-SF-020, REQ-SF-022)
  |-- extract_code_block (REQ-SF-005)
  |
Spec 04 (Phase 1)
  |-- Phase1Result provides s_0, h_best (REQ-DM-022)
  |
  v
Spec 05 (this) -- Phase 2 Outer Loop
  |
  v
Referenced by:
  Spec 06 (inner loop receives c_t, p_0)
  Spec 09 (orchestrator invokes Phase 2)
```

### 2.3 Product Functions Summary

1. Generate ablation study scripts that test 2-3 code components of the current solution via A_abl
2. Execute ablation scripts and capture raw output via the execution harness
3. Summarize ablation results into concise text via A_summarize
4. Extract the most impactful code block and propose a refinement plan via A_extractor
5. Validate that extracted code blocks are exact substrings of the current solution
6. Manage outer loop state: accumulate ablation summaries and refined code blocks
7. Orchestrate handoff to and return from the inner loop (Spec 06) for each outer step
8. Construct Phase2Result upon outer loop completion

### 2.4 Operating Environment

- **Runtime**: Python 3.10+
- **SDK**: `claude-agent-sdk` v0.1.39+
- **Validation library**: Pydantic v2 (for ExtractorOutput structured output)
- **Execution**: Ablation scripts execute via Spec 02 harness; agents produce text or structured JSON

### 2.5 Assumptions and Dependencies

| ID | Assumption | Impact if Invalid |
|----|-----------|-------------------|
| A-01 | The initial solution s_0 from Phase 1 is executable and has a valid score | Ablation studies would fail on a broken baseline |
| A-02 | Ablation study code can be generated as a self-contained single-file Python script | Multi-file ablation setups would require new execution model |
| A-03 | 2-3 components per ablation study is sufficient granularity | Larger solutions may need more components per study |
| A-04 | LLM can reliably extract exact code substrings from a solution script | Approximate matches would require fuzzy matching logic |
| A-05 | Each outer loop iteration produces at least one viable code block for refinement | Empty extractor output would stall the outer loop |

| ID | Dependency | Owner | Risk if Unavailable |
|----|-----------|-------|---------------------|
| D-01 | Spec 01 types (SolutionScript, CodeBlock, ExtractorOutput, Phase2Result, etc.) | Spec 01 | Cannot construct agent inputs or parse outputs |
| D-02 | Spec 02 execution harness (execute_script, evaluate_solution, evaluate_with_retry) | Spec 02 | Cannot execute ablation scripts or evaluate solutions |
| D-03 | Spec 03 safety agents (debug_solution, check_and_fix_leakage) | Spec 03 | Cannot fix ablation script errors or detect leakage |
| D-04 | Spec 04 Phase1Result (s_0, h_best) | Spec 04 | No initial solution to refine |
| D-05 | Spec 06 inner loop (refine code block) | Spec 06 | Cannot refine extracted code blocks |
| D-06 | `claude-agent-sdk` v0.1.39+ | Anthropic | Cannot define or invoke agents |
| D-07 | PromptRegistry with templates for ablation, summarize, extractor agents | Spec 01 | Cannot construct agent prompts |

---

## 3. A_abl Requirements

### 3.1 Agent Definition

> **REQ-P2O-001**: *A_abl Agent Definition* -- The system shall define an `AgentDefinition`-compatible configuration for the ablation study agent with the following properties:
>
> | Property | Value |
> |----------|-------|
> | `agent_type` | `AgentType.ablation` |
> | `description` | Agent that generates ablation study code to identify impactful solution components |
> | `prompt` | Rendered from the ablation template (Figure 12, REQ-DM-032) |
> | `tools` | `["Read"]` |
> | `output_schema` | `None` (free-form response containing a single Python code block) |
> | `model` | `None` (inherit from orchestrator) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `AgentConfig(agent_type=AgentType.ablation).to_agent_definition()` shall produce a valid dictionary for `ClaudeAgentOptions.agents`.
> - Source: REF-01 Section 3.2, Figure 12

> **REQ-P2O-002**: *A_abl Prompt Template* -- The ablation study agent prompt shall be constructed by rendering the Figure 12 template from the `PromptRegistry` (REQ-DM-032) with the following variables:
>
> | Variable | Type | Description |
> |----------|------|-------------|
> | `solution_script` | `str` | Full source code of the current best solution (`SolutionScript.content`) |
> | `previous_ablations` | `list[str]` | List of previous ablation summaries (T_abl^0 through T_abl^{t-1}) |
>
> - The rendered prompt shall include all instructions from Figure 12:
>   1. Kaggle grandmaster persona introduction
>   2. Current Python solution presented in full
>   3. All previous ablation study summaries (numbered)
>   4. Instructions to generate a Python ablation study script that creates variations by modifying or disabling 2-3 parts
>   5. Instruction to concentrate on parts not previously considered
>   6. Instruction to print performance of each ablation and identify the most impactful component
>   7. Response format constraint: no additional headings or text, no test data loading, include printing statements
> - When `previous_ablations` is empty (first iteration), the section for previous ablation results shall be omitted.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 12

### 3.2 Input/Output Contract

> **REQ-P2O-003**: *A_abl Input Contract* -- The ablation study agent shall accept two inputs:
>
> 1. `solution: SolutionScript` -- the current best solution at outer step t.
> 2. `previous_summaries: list[str]` -- list of ablation summaries from all prior outer steps (T_abl^0 through T_abl^{t-1}).
>
> - Precondition: `solution.content` is non-empty and `solution.is_executable` is `True`.
> - `previous_summaries` may be empty (first outer loop iteration).
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 2 line 5 -- `a_t = A_abl(s_t, T_abl)`

> **REQ-P2O-004**: *A_abl Output Contract* -- The ablation study agent shall return an executable Python script extracted from the agent's response.
>
> - The response shall be parsed using `extract_code_block()` (REQ-SF-005) to obtain the ablation study script.
> - The extracted script shall be wrapped in a `SolutionScript` with `phase` set to `"refined"` and `is_executable` set to `True` (optimistic; will be verified by execution).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given an agent response containing a fenced Python code block, the returned script shall contain only the code within the fences.
> - Source: REF-01 Algorithm 2 line 5 -- `a_t` is executable ablation code

### 3.3 Ablation Scope

> **REQ-P2O-005**: *A_abl Ablation Component Count* -- The ablation study agent prompt (REQ-P2O-002) shall instruct the agent to test exactly 2-3 code components per ablation study. This is enforced via prompt instruction, not automated verification.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 12 -- "create variations by modifying or disabling parts (2-3 parts)"

> **REQ-P2O-006**: *A_abl Avoid Previously Ablated Components* -- The ablation study agent prompt (REQ-P2O-002) shall instruct the agent to concentrate on components that have not been previously ablated, using the provided `previous_summaries` as context. This is enforced via prompt instruction, not automated verification.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 12 -- "Your ablation study should concentrate on the other parts that have not been previously considered."

> **REQ-P2O-007**: *A_abl Validation-Only Evaluation* -- The ablation study agent prompt (REQ-P2O-002) shall instruct the agent to generate code that evaluates on the validation set only. The generated ablation script shall not load or process test data.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 12 -- "The Python code for the ablation study should not load test data. It should only focus on training and evaluating the model on the validation set."

---

## 4. A_summarize Requirements

### 4.1 Agent Definition

> **REQ-P2O-008**: *A_summarize Agent Definition* -- The system shall define an `AgentDefinition`-compatible configuration for the ablation summarization agent with the following properties:
>
> | Property | Value |
> |----------|-------|
> | `agent_type` | `AgentType.summarize` |
> | `description` | Agent that summarizes raw ablation study output into a concise textual summary |
> | `prompt` | Rendered from the summarization template (Figure 13, REQ-DM-032) |
> | `tools` | `None` (no tools needed) |
> | `output_schema` | `None` (free-form text summary) |
> | `model` | `None` (inherit from orchestrator) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `AgentConfig(agent_type=AgentType.summarize).to_agent_definition()` shall produce a valid dictionary for `ClaudeAgentOptions.agents`.
> - Source: REF-01 Section 3.2, Figure 13

> **REQ-P2O-009**: *A_summarize Prompt Template* -- The ablation summarization agent prompt shall be constructed by rendering the Figure 13 template from the `PromptRegistry` (REQ-DM-032) with the following variables:
>
> | Variable | Type | Description |
> |----------|------|-------------|
> | `ablation_code` | `str` | Full source code of the ablation study script (a_t) |
> | `raw_result` | `str` | Raw execution output (stdout) from running the ablation script (r_t) |
>
> - The rendered prompt shall include:
>   1. The ablation study code that was executed
>   2. The raw printed output from execution
>   3. Instruction to summarize the result of the ablation study based on the code and printed output
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 13

### 4.2 Input/Output Contract

> **REQ-P2O-010**: *A_summarize Input Contract* -- The ablation summarization agent shall accept two inputs:
>
> 1. `ablation_code: str` -- the source code of the ablation study script that was executed.
> 2. `raw_output: str` -- the raw stdout captured from executing the ablation script.
>
> - Precondition: `ablation_code` is non-empty. `raw_output` may be empty if the script produced no output (though this is an abnormal condition).
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 2 line 7 -- `T_abl^t = A_summarize(a_t, r_t)`

> **REQ-P2O-011**: *A_summarize Output Contract* -- The ablation summarization agent shall return a plain text summary string (T_abl^t).
>
> - The summary shall be the complete text content of the agent's response (no code block extraction, no structured parsing).
> - The summary shall identify which code components had the most and least impact on model performance.
> - The summary text shall be stored as-is for accumulation in T_abl and for input to A_extractor.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given an ablation output showing "Baseline: 0.8196, No StandardScaler: 0.8102, No OneHotEncoder: 0.7886, No Imputation: 0.8196", the summary shall identify OneHotEncoder as the most impactful component.
> - Source: REF-01 Algorithm 2 line 7, Appendix C Figures 23-24

---

## 5. A_extractor Requirements

### 5.1 Agent Definition

> **REQ-P2O-012**: *A_extractor Agent Definition* -- The system shall define an `AgentDefinition`-compatible configuration for the code block extractor agent with the following properties:
>
> | Property | Value |
> |----------|-------|
> | `agent_type` | `AgentType.extractor` |
> | `description` | Agent that identifies the most impactful code block and proposes a refinement plan |
> | `prompt` | Rendered from the extractor template (Figure 14, REQ-DM-032) |
> | `tools` | `["Read"]` |
> | `output_schema` | `ExtractorOutput` (REQ-DM-017) |
> | `model` | `None` (inherit from orchestrator) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `AgentConfig(agent_type=AgentType.extractor).to_agent_definition()` shall produce a valid dictionary for `ClaudeAgentOptions.agents`.
> - Source: REF-01 Section 3.2, Figure 14

> **REQ-P2O-013**: *A_extractor Prompt Template* -- The code block extractor agent prompt shall be constructed by rendering the Figure 14 template from the `PromptRegistry` (REQ-DM-032) with the following variables:
>
> | Variable | Type | Description |
> |----------|------|-------------|
> | `solution_script` | `str` | Full source code of the current best solution (`SolutionScript.content`) |
> | `ablation_summary` | `str` | Summary of the current ablation study (T_abl^t) |
> | `previous_code_blocks` | `list[str]` | List of previously refined code blocks (C) |
>
> - The rendered prompt shall include all instructions from Figure 14:
>   1. Kaggle grandmaster persona introduction
>   2. Goal: extract a code block and improve it for better performance
>   3. Current Python solution presented in full
>   4. Ablation study results summary
>   5. Previously improved code blocks (numbered)
>   6. Instruction to suggest an effective plan (3-5 sentences)
>   7. Instruction to avoid plans with excessive runtime (e.g., large hyperparameter search)
>   8. Instruction to improve parts not previously considered
>   9. Instruction to extract a code block exactly from the Python script
>   10. Response format: JSON schema `Refine_Plan = {'code_block': str, 'plan': str}` as `list[Refine_Plan]`
> - When `previous_code_blocks` is empty (first iteration), the section for previous code blocks shall be omitted.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 14

### 5.2 Structured Output

> **REQ-P2O-014**: *A_extractor Structured Output Usage* -- The extractor agent shall use the Claude Agent SDK's `output_format` parameter set to `{"type": "json_schema", "schema": ExtractorOutput.model_json_schema()}` to ensure the response conforms to the `ExtractorOutput` schema (REQ-DM-017).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: The agent response shall be parseable via `ExtractorOutput.model_validate_json(response)` without errors.
> - Source: REF-01 Figure 14, REF-02 Section 12 (Structured Outputs)

> **REQ-P2O-015**: *A_extractor Output Contract* -- The extractor agent shall return an `ExtractorOutput` (REQ-DM-017) parsed from the agent's structured JSON response, containing one or more `RefinePlan` objects (REQ-DM-016), each with:
>
> - `code_block: str` -- an exact code substring extracted from the current solution
> - `plan: str` -- a natural language refinement plan (3-5 sentences)
>
> - The system shall use the **first** `RefinePlan` in the returned list as the primary target for the current outer loop iteration (c_t, p_0).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `ExtractorOutput.plans[0].code_block` shall be an exact substring of `solution.content`.
> - Source: REF-01 Algorithm 2 line 8 -- `c_t, p_0 = A_extractor(T_abl^t, s_t, C)`

### 5.3 Input Contract

> **REQ-P2O-016**: *A_extractor Input Contract* -- The code block extractor agent shall accept three inputs:
>
> 1. `summary: str` -- the ablation summary for the current outer step (T_abl^t).
> 2. `solution: SolutionScript` -- the current best solution at outer step t.
> 3. `previous_blocks: list[str]` -- list of code block content strings from all previously refined code blocks (C).
>
> - Precondition: `summary` is non-empty and `solution.content` is non-empty.
> - `previous_blocks` may be empty (first outer loop iteration).
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 2 line 8 -- `c_t, p_0 = A_extractor(T_abl^t, s_t, C)`

### 5.4 Code Block Validation

> **REQ-P2O-017**: *Code Block Exact Substring Validation* -- The system shall define a function `validate_code_block(code_block: str, solution: SolutionScript) -> bool` that returns `True` if and only if `code_block` is found as an exact substring within `solution.content`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a solution containing `"scaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)"` and a code block `"scaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)"`, validation shall return `True`.
> - Acceptance: A code block with any character difference (including whitespace) from the solution content shall cause validation to return `False`.
> - Source: REF-01 Figure 14 -- "The code block can be long but should be exactly extracted from the Python script provided above."

> **REQ-P2O-018**: *Code Block Validation Failure Handling* -- If `validate_code_block()` returns `False` for the extracted code block, the system shall:
>
> 1. Attempt a whitespace-normalized match: strip trailing whitespace from each line of both the code block and the solution, then check for substring match. If this succeeds, use the whitespace-normalized code block (matched from the solution source) as c_t.
> 2. If the whitespace-normalized match also fails, log a warning and re-invoke A_extractor with the same inputs (up to 2 re-invocations). Include in the re-invocation prompt an additional instruction: "The previously extracted code block was not found in the solution. Please extract the code block exactly as it appears in the script."
> 3. If all re-invocations fail validation, select the first `RefinePlan` whose `code_block` passes validation from the `ExtractorOutput.plans` list. If none pass, skip this outer loop iteration and proceed to the next.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a code block with trailing whitespace differences, the whitespace-normalized match shall succeed and return the correct substring from the solution.
> - Source: REF-01 Equation 7 -- `s_t.replace(c_t, c_t^k)` requires c_t to be an exact substring

---

## 6. Outer Loop Control Flow Requirements

### 6.1 Outer Loop Function

> **REQ-P2O-019**: *Phase 2 Outer Loop Function* -- The system shall define an async function `run_phase2_outer_loop(initial_solution: SolutionScript, initial_score: float, task: TaskDescription, config: PipelineConfig) -> Phase2Result` that implements Algorithm 2 (outer loop portion):
>
> ```
> Input: initial_solution s_0, initial_score h_best, task, config
> 1. s_final <- s_0
> 2. h_best <- initial_score
> 3. T_abl <- []          (accumulated ablation summaries)
> 4. C <- []              (accumulated refined code blocks)
> 5. step_history <- []
> 6. for t = 0 to config.outer_loop_steps - 1:
> 7.     a_t = invoke A_abl(s_t, T_abl)
> 8.     r_t = execute a_t (with debug retry on error)
> 9.     T_abl_t = invoke A_summarize(a_t, r_t)
> 10.    c_t, p_0 = invoke A_extractor(T_abl_t, s_t, C)
> 11.    validate c_t is exact substring of s_t
> 12.    [hand off to inner loop: Spec 06]
> 13.    [receive back: best_solution_from_inner, best_score_from_inner, inner_history]
> 14.    if best_score_from_inner >= h_best:
> 15.        s_final <- best_solution_from_inner
> 16.        h_best <- best_score_from_inner
> 17.        s_t <- best_solution_from_inner
> 18.    T_abl.append(T_abl_t)
> 19.    C.append(c_t)
> 20.    step_history.append({outer_step, inner_history})
> 21. end for
> 22. return Phase2Result(T_abl, C, s_final, h_best, step_history)
> ```
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given `config.outer_loop_steps = 2`, the function shall execute exactly 2 iterations of the outer loop.
> - Source: REF-01 Algorithm 2 lines 1-28

### 6.2 Ablation Script Execution

> **REQ-P2O-020**: *Ablation Script Execution* -- The outer loop shall execute the ablation study script a_t through the execution harness by:
>
> 1. Writing the ablation script content to a temporary file using `write_script()` (REQ-EX-005).
> 2. Executing it using `execute_script()` (REQ-EX-007) with the working directory set to the task's data directory.
> 3. Capturing full stdout and stderr from the execution.
>
> - The ablation script is not a solution script and shall not be evaluated for a validation score. It produces informational output about component impact.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 2 line 6 -- `r_t = exec(a_t)`

> **REQ-P2O-021**: *Ablation Script Error Handling* -- If the ablation script a_t produces an execution error (non-zero exit code or traceback in stderr), the system shall:
>
> 1. Invoke A_debugger (REQ-SF-006) to fix the ablation script, passing the ablation `SolutionScript` and the extracted traceback.
> 2. Re-execute the fixed script.
> 3. Repeat up to `config.max_debug_attempts` times (REQ-DM-001).
> 4. If all debug attempts are exhausted and the ablation script still fails, skip the ablation for this outer step: set T_abl^t to an empty string indicating "Ablation study failed for this step", and proceed to A_extractor with the empty summary. A_extractor shall rely on the accumulated previous summaries and the solution itself.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given an ablation script that errors once and succeeds after one debug attempt, the system shall produce a valid raw output r_t from the fixed script.
> - Acceptance: Given an ablation script that errors on all debug attempts, the system shall proceed to A_extractor with an empty summary and not raise an exception.
> - Source: REF-01 Section 3.4 -- A_debugger applies to all generated code

### 6.3 Outer Loop State Management

> **REQ-P2O-022**: *Ablation Summary Accumulation* -- The outer loop shall maintain a list `T_abl: list[str]` that accumulates ablation summaries across iterations. After each outer step t, the summary T_abl^t shall be appended to this list.
>
> - At outer step t, the input to A_abl is `T_abl[0:t]` (all summaries from prior steps).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After 3 outer steps, `len(T_abl)` shall equal 3.
> - Source: REF-01 Algorithm 2 line 26 -- `T_abl <- T_abl + T_abl^t`

> **REQ-P2O-023**: *Refined Code Block Accumulation* -- The outer loop shall maintain a list `C: list[CodeBlock]` that accumulates the code blocks targeted for refinement across iterations. After each outer step t, the code block c_t (wrapped in a `CodeBlock` model, REQ-DM-012) shall be appended to this list.
>
> - At outer step t, the input to A_extractor is `[cb.content for cb in C[0:t]]` (content strings of all previously refined blocks).
> - The `CodeBlock.outer_step` field shall be set to `t` for the block refined at step t.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After 3 outer steps, `len(C)` shall equal 3 and `C[2].outer_step` shall equal 2.
> - Source: REF-01 Algorithm 2 line 27 -- `C <- C + c_t`

> **REQ-P2O-024**: *Current Solution Tracking* -- The outer loop shall maintain a reference `s_t: SolutionScript` to the current best solution. At the start of the outer loop, `s_t` shall be set to `initial_solution` (s_0). After each inner loop completes, `s_t` shall be updated to the best solution returned by the inner loop if it improves upon `h_best`.
>
> - The comparison shall use `is_improvement_or_equal()` (REQ-DM-029) with the task's `metric_direction`.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: If the inner loop produces a solution with score 0.85 and `h_best` is 0.82 (maximize), `s_t` shall be updated to the new solution.
> - Source: REF-01 Algorithm 2 lines 12-14 (within inner loop), extended to outer loop tracking

### 6.4 Outer Loop Iteration

> **REQ-P2O-025**: *Outer Loop Iteration Count* -- The outer loop shall execute exactly `config.outer_loop_steps` (T) iterations (REQ-DM-001, default: 4).
>
> - If an iteration encounters an unrecoverable error (e.g., ablation fails and extractor also fails), the iteration shall be skipped but the loop shall continue to the next iteration.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given `config.outer_loop_steps = 4`, the outer loop shall attempt exactly 4 iterations.
> - Source: REF-01 Algorithm 2 line 4 -- `for t = 0 to T-1 do`

### 6.5 Handoff to Inner Loop

> **REQ-P2O-026**: *Inner Loop Handoff* -- At each outer step t, after obtaining c_t and p_0 from A_extractor, the outer loop shall invoke the inner loop (defined in Spec 06) by calling an async function with the following signature:
>
> ```python
> async def run_phase2_inner_loop(
>     solution: SolutionScript,        # current best solution s_t
>     code_block: CodeBlock,           # target code block c_t
>     initial_plan: str,               # initial refinement plan p_0
>     best_score: float,               # current h_best
>     task: TaskDescription,
>     config: PipelineConfig,
> ) -> InnerLoopResult
> ```
>
> - The inner loop function is defined in Spec 06; this spec defines only the call site and the data passed.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Algorithm 2 lines 9-25 (inner loop)

> **REQ-P2O-027**: *Post-Inner-Loop Update* -- After the inner loop returns, the outer loop shall:
>
> 1. Read the best solution and best score from the inner loop result.
> 2. If the inner loop's best score is an improvement or equal to the current `h_best` (using `is_improvement_or_equal()`, REQ-DM-029):
>    a. Update `s_final` to the inner loop's best solution.
>    b. Update `h_best` to the inner loop's best score.
>    c. Update `s_t` to the inner loop's best solution (for use in the next outer step).
> 3. Record the inner loop's step history (list of `RefinementAttempt`, REQ-DM-042) in the outer step record.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: If the inner loop returns a solution with score 0.90 and `h_best` is 0.85 (maximize), `s_final` and `h_best` shall be updated.
> - Acceptance: If the inner loop returns a solution with score 0.80 and `h_best` is 0.85 (maximize), `s_final` and `h_best` shall remain unchanged.
> - Source: REF-01 Algorithm 2 lines 12-14 (best tracking) and lines 26-27 (accumulation)

### 6.6 Phase2Result Construction

> **REQ-P2O-028**: *Phase2Result Construction* -- Upon completion of the outer loop, the system shall construct a `Phase2Result` (REQ-DM-023) with the following field mappings:
>
> | Phase2Result Field | Source |
> |-------------------|--------|
> | `ablation_summaries` | `T_abl` (accumulated list of summary strings) |
> | `refined_blocks` | `C` (accumulated list of `CodeBlock` objects) |
> | `best_solution` | `s_final` (best solution found across all iterations) |
> | `best_score` | `h_best` (score of the best solution) |
> | `step_history` | List of per-step records, each containing: outer step index, ablation summary, code block extracted, inner loop history (list of `RefinementAttempt`), best score after this step |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: The returned `Phase2Result.best_score` shall be >= `initial_score` (for maximize direction) or <= `initial_score` (for minimize direction).
> - Acceptance: `len(Phase2Result.ablation_summaries)` shall equal `config.outer_loop_steps` (or fewer if iterations were skipped).
> - Source: REF-01 Algorithm 2 line 29 -- `Output: final solution s_final`

> **REQ-P2O-029**: *Phase2Result Preserves Initial Score on No Improvement* -- If no outer loop iteration produces an improvement, the `Phase2Result.best_solution` shall be the initial solution s_0 and `Phase2Result.best_score` shall be the initial score h_best. The system shall never return a Phase2Result with a worse score than the initial.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given 4 outer loop iterations where all inner loops fail to improve, `Phase2Result.best_solution.content` shall equal `initial_solution.content`.
> - Source: REF-01 Algorithm 2 lines 1-2 -- `s_final <- s_0`, `h_best <- h(s_0)`

### 6.7 Step History Recording

> **REQ-P2O-030**: *Outer Step History Record* -- Each outer step shall produce a history record (stored in `step_history`) containing:
>
> | Field | Type | Description |
> |-------|------|-------------|
> | `outer_step` | `int` | Outer loop step index (0-based) |
> | `ablation_summary` | `str` | T_abl^t summary text |
> | `code_block` | `str` | Extracted code block content (c_t) |
> | `plan` | `str` | Initial refinement plan (p_0) |
> | `inner_loop_attempts` | `list[RefinementAttempt]` | Inner loop refinement history (REQ-DM-042) |
> | `best_score_after_step` | `float` | Value of h_best after this step completed |
> | `was_skipped` | `bool` | Whether the step was skipped due to errors |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After the outer loop completes, `len(step_history)` shall equal `config.outer_loop_steps`.
> - Source: REF-01 Algorithm 2 (full loop structure)

---

## 7. Non-Functional Requirements

### 7.1 Performance

> **REQ-P2O-031**: *A_abl Response Overhead* -- The system overhead for processing an A_abl agent response (code block extraction, SolutionScript wrapping) shall not exceed 500 milliseconds, excluding LLM call latency and script execution time.
>
> - Priority: Should | Verify: Test | Release: MVP

> **REQ-P2O-032**: *A_extractor Validation Overhead* -- The code block validation function `validate_code_block()` (REQ-P2O-017) shall execute in under 50 milliseconds for solution scripts up to 50 KB.
>
> - Priority: Should | Verify: Test | Release: MVP

> **REQ-P2O-033**: *Outer Loop Total Duration* -- A single outer loop iteration (A_abl invocation + ablation execution + A_summarize invocation + A_extractor invocation, excluding inner loop time) shall complete within 15 minutes under normal conditions. If ablation script execution exceeds this budget, the execution timeout for ablation scripts shall be set per REQ-P2O-035.
>
> - Priority: Should | Verify: Demonstration | Release: MVP

### 7.2 Reliability

> **REQ-P2O-034**: *A_extractor Graceful Degradation* -- If the extractor agent returns a response that cannot be parsed as `ExtractorOutput` (e.g., malformed JSON, schema validation failure):
>
> 1. Log a warning with the raw response content.
> 2. Re-invoke A_extractor once with the same inputs.
> 3. If the re-invocation also fails parsing, skip this outer loop iteration: do not invoke the inner loop, append an empty summary and a placeholder code block to the accumulators, and proceed to the next outer step.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a malformed extractor response on the first attempt and a valid response on the retry, the system shall successfully extract c_t and p_0 from the retry.
> - Acceptance: Given two consecutive malformed responses, the outer loop shall skip the iteration without raising an exception.

> **REQ-P2O-035**: *Ablation Script Execution Timeout* -- Ablation scripts shall be executed with a timeout equal to `min(config.time_limit_seconds / (config.outer_loop_steps * 2), 600)` seconds, capping at 10 minutes per ablation execution. This prevents a single ablation study from consuming an excessive portion of the total time budget.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given `config.time_limit_seconds = 86400` and `config.outer_loop_steps = 4`, the ablation timeout shall be `min(86400/8, 600) = 600` seconds.

> **REQ-P2O-036**: *A_summarize Graceful Degradation* -- If the summarization agent returns an empty or unparseable response:
>
> 1. Log a warning with the raw response content.
> 2. Use a fallback summary constructed from the raw ablation output: truncate `raw_output` to the last 2000 characters and prefix with `"[Auto-summary from raw output] "`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given an empty summarization response, the system shall produce a fallback summary and not raise an exception.

### 7.3 Observability

> **REQ-P2O-037**: *Outer Loop Logging* -- Each outer loop step shall log the following events using Python's `logging` module at the specified levels:
>
> | Event | Level | Content |
> |-------|-------|---------|
> | Outer step start | `INFO` | Step index t, current h_best, number of accumulated summaries |
> | A_abl invocation start | `INFO` | Solution content length, number of previous summaries |
> | A_abl invocation complete | `INFO` | Ablation script length (characters) |
> | Ablation execution start | `INFO` | Script path, timeout |
> | Ablation execution complete | `INFO` | Exit code, output length, duration |
> | Ablation execution error | `WARNING` | Exit code, traceback summary (first line) |
> | A_summarize invocation start | `INFO` | Ablation code length, raw output length |
> | A_summarize invocation complete | `INFO` | Summary length |
> | A_extractor invocation start | `INFO` | Summary length, solution length, number of previous blocks |
> | A_extractor invocation complete | `INFO` | Number of plans returned, code block length |
> | Code block validation result | `INFO` | Pass/fail, match method (exact / whitespace-normalized) |
> | Code block validation failure | `WARNING` | Code block first 100 chars, re-invocation attempt number |
> | Inner loop handoff | `INFO` | Code block length, plan text (first 200 chars) |
> | Inner loop return | `INFO` | Best score from inner loop, improvement (yes/no) |
> | Outer step complete | `INFO` | Step index t, updated h_best, duration |
> | Outer step skipped | `WARNING` | Step index t, reason for skipping |
> | Outer loop complete | `INFO` | Total steps completed, final h_best, total duration |
>
> - Priority: Must | Verify: Inspection | Release: MVP

---

## 8. Constraints

### 8.1 Technology Constraints

> **REQ-P2O-038**: *SDK Agent Invocation* -- All three agents (A_abl, A_summarize, A_extractor) shall be invoked via the Claude Agent SDK agent mechanism. They shall not use direct API calls, raw HTTP requests, or any non-SDK LLM invocation method.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-02 -- all agent interactions through the SDK

> **REQ-P2O-039**: *Single Module Organization* -- All outer loop functions and agent invocation logic defined in this spec shall reside in a single Python module (e.g., `mle_star/phase2_outer.py`).
>
> - Priority: Should | Verify: Inspection | Release: MVP

### 8.2 Algorithmic Constraints

> **REQ-P2O-040**: *Sequential Outer Loop Execution* -- The outer loop iterations shall execute sequentially, not concurrently. Each iteration depends on the accumulated state (T_abl, C, s_t) from all prior iterations.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Rationale: Each iteration's ablation study depends on previous summaries, and the solution may change between iterations.
> - Source: REF-01 Algorithm 2 -- sequential loop with state accumulation

> **REQ-P2O-041**: *Monotonic Best Score* -- The `h_best` value tracked across the outer loop shall be monotonically non-decreasing (for maximize) or non-increasing (for minimize). The system shall never overwrite `s_final` with a solution that has a worse score than the current `h_best`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After every outer step, `h_best` shall be >= (maximize) or <= (minimize) the value at the start of that step.
> - Source: REF-01 Algorithm 2 lines 12-14 -- conditional update on `>=`

> **REQ-P2O-042**: *Ablation Script Independence* -- The ablation study script a_t generated by A_abl shall be a self-contained Python script independent of the solution script s_t. It may read and modify s_t's logic internally, but it shall not import s_t or depend on s_t being present as a separate file.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 12 -- "generate a simple Python code that performs an ablation study on the train.py script"

### 8.3 Data Integrity Constraints

> **REQ-P2O-043**: *Immutable Input Solution* -- The `run_phase2_outer_loop` function shall not mutate the `initial_solution` parameter. All modifications to the solution shall produce new `SolutionScript` instances. The original `initial_solution` shall remain available for fallback (REQ-P2O-029).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Rationale: Preserves the Phase 1 baseline for rollback if Phase 2 fails to improve.

> **REQ-P2O-044**: *Code Block Provenance* -- Each `CodeBlock` stored in C shall have its `outer_step` field set to the iteration index at which it was extracted. This enables A_extractor in subsequent iterations to identify which blocks have already been refined.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `C[t].outer_step == t` for all t in range `[0, len(C))`.

---

## 9. Traceability Matrix

### 9.1 Requirements to Paper Sections

| Req ID | Paper Section | Paper Element | SDK Construct |
|--------|--------------|---------------|---------------|
| REQ-P2O-001 | Section 3.2 | A_abl agent | `AgentDefinition` |
| REQ-P2O-002 | Figure 12 | Ablation prompt template | `prompt` parameter |
| REQ-P2O-003 | Alg 2 line 5 | `a_t = A_abl(s_t, T_abl)` input | -- |
| REQ-P2O-004 | Alg 2 line 5 | `a_t = A_abl(s_t, T_abl)` output | -- |
| REQ-P2O-005 | Figure 12 | "2-3 parts" ablation scope | -- |
| REQ-P2O-006 | Figure 12 | "not previously considered" | -- |
| REQ-P2O-007 | Figure 12 | "not load test data" | -- |
| REQ-P2O-008 | Section 3.2 | A_summarize agent | `AgentDefinition` |
| REQ-P2O-009 | Figure 13 | Summarization prompt template | `prompt` parameter |
| REQ-P2O-010 | Alg 2 line 7 | `T_abl^t = A_summarize(a_t, r_t)` input | -- |
| REQ-P2O-011 | Alg 2 line 7 | `T_abl^t = A_summarize(a_t, r_t)` output | -- |
| REQ-P2O-012 | Section 3.2 | A_extractor agent | `AgentDefinition` |
| REQ-P2O-013 | Figure 14 | Extractor prompt template | `prompt` parameter |
| REQ-P2O-014 | Figure 14 | Structured output schema usage | `output_format` |
| REQ-P2O-015 | Alg 2 line 8 | `c_t, p_0 = A_extractor(...)` output | `output_format` |
| REQ-P2O-016 | Alg 2 line 8 | `A_extractor(T_abl^t, s_t, C)` input | -- |
| REQ-P2O-017 | Figure 14 | "exactly extracted from the Python script" | -- |
| REQ-P2O-018 | Figure 14 | Code block validation failure handling | -- |
| REQ-P2O-019 | Alg 2 lines 1-28 | Outer loop function | -- |
| REQ-P2O-020 | Alg 2 line 6 | `r_t = exec(a_t)` | -- |
| REQ-P2O-021 | Section 3.4 | A_debugger for ablation errors | -- |
| REQ-P2O-022 | Alg 2 line 26 | `T_abl <- T_abl + T_abl^t` | -- |
| REQ-P2O-023 | Alg 2 line 27 | `C <- C + c_t` | -- |
| REQ-P2O-024 | Alg 2 lines 1-2 | `s_final <- s_0`, `h_best <- h(s_0)` tracking | -- |
| REQ-P2O-025 | Alg 2 line 4 | `for t = 0 to T-1` | -- |
| REQ-P2O-026 | Alg 2 lines 9-25 | Inner loop handoff | -- |
| REQ-P2O-027 | Alg 2 lines 12-14, 26-27 | Post-inner-loop update | -- |
| REQ-P2O-028 | Alg 2 line 29 | `Output: s_final` result construction | -- |
| REQ-P2O-029 | Alg 2 lines 1-2 | Preserve initial score on no improvement | -- |
| REQ-P2O-030 | Alg 2 (full) | Per-step history record | -- |
| REQ-P2O-031 | -- | Processing overhead | -- |
| REQ-P2O-032 | -- | Validation overhead | -- |
| REQ-P2O-033 | -- | Outer step duration | -- |
| REQ-P2O-034 | -- | Extractor graceful degradation | -- |
| REQ-P2O-035 | Section 4 | Ablation timeout | -- |
| REQ-P2O-036 | -- | Summarize graceful degradation | -- |
| REQ-P2O-037 | -- | Logging | Python `logging` |
| REQ-P2O-038 | -- | SDK-only invocation | `claude-agent-sdk` |
| REQ-P2O-039 | -- | Module organization | -- |
| REQ-P2O-040 | Alg 2 | Sequential execution | -- |
| REQ-P2O-041 | Alg 2 lines 12-14 | Monotonic best score | -- |
| REQ-P2O-042 | Figure 12 | Self-contained ablation script | -- |
| REQ-P2O-043 | -- | Immutable input solution | -- |
| REQ-P2O-044 | -- | Code block provenance | -- |

### 9.2 Cross-References to Other Specs

| Req ID | Referenced By |
|--------|--------------|
| REQ-P2O-001 (A_abl agent def) | Spec 09 (orchestrator configures agents) |
| REQ-P2O-008 (A_summarize agent def) | Spec 09 (orchestrator configures agents) |
| REQ-P2O-012 (A_extractor agent def) | Spec 09 (orchestrator configures agents) |
| REQ-P2O-015 (extractor output) | Spec 06 (inner loop receives c_t, p_0) |
| REQ-P2O-017 (code block validation) | Spec 06 (inner loop uses validated code block for replacement) |
| REQ-P2O-019 (outer loop function) | Spec 09 (orchestrator invokes Phase 2) |
| REQ-P2O-026 (inner loop handoff) | Spec 06 (defines inner loop function signature) |
| REQ-P2O-027 (post-inner-loop update) | Spec 06 (inner loop return type) |
| REQ-P2O-028 (Phase2Result) | Spec 07 (Phase 3 receives Phase2Result.best_solution), Spec 09 (orchestrator collects results) |

### 9.3 Spec 01 Dependencies (Inbound)

| Spec 01 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-DM-001 (PipelineConfig) | REQ-P2O-019, 025, 035 | `outer_loop_steps`, `max_debug_attempts`, `time_limit_seconds` |
| REQ-DM-007 (TaskDescription) | REQ-P2O-019, 024 | Task context, metric direction |
| REQ-DM-009 (SolutionScript) | REQ-P2O-003, 004, 016, 019, 024, 043 | Solution input/output type |
| REQ-DM-010 (replace_block) | REQ-P2O-017, 018 | Code block replacement (used by inner loop, validated here) |
| REQ-DM-012 (CodeBlock) | REQ-P2O-023, 028, 044 | Code block model for accumulation in C |
| REQ-DM-013 (AgentType) | REQ-P2O-001, 008, 012 | Agent identity enum values |
| REQ-DM-016 (RefinePlan) | REQ-P2O-015 | Structured output schema element |
| REQ-DM-017 (ExtractorOutput) | REQ-P2O-012, 014, 015 | Structured output model for extractor |
| REQ-DM-023 (Phase2Result) | REQ-P2O-028, 029 | Result model for outer loop output |
| REQ-DM-029 (is_improvement_or_equal) | REQ-P2O-024, 027 | Score comparison for best tracking |
| REQ-DM-032 (PromptRegistry) | REQ-P2O-002, 009, 013 | Template retrieval for all agents |
| REQ-DM-036 (AgentConfig) | REQ-P2O-001, 008, 012 | Agent-to-SDK mapping |
| REQ-DM-042 (RefinementAttempt) | REQ-P2O-027, 030 | Inner loop history records |

### 9.4 Spec 02 Dependencies (Inbound)

| Spec 02 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-EX-005 (write_script) | REQ-P2O-020 | Write ablation script to file |
| REQ-EX-007 (execute_script) | REQ-P2O-020 | Execute ablation script |
| REQ-EX-011 (parse_score) | REQ-P2O-020 | Parse ablation output (informational only) |
| REQ-EX-012 (extract_traceback) | REQ-P2O-021 | Extract traceback for A_debugger |
| REQ-EX-015 (evaluate_solution) | REQ-P2O-020 | Evaluate ablation scripts |

### 9.5 Spec 03 Dependencies (Inbound)

| Spec 03 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-SF-005 (extract_code_block) | REQ-P2O-004 | Extract ablation code from A_abl response |
| REQ-SF-006 (debug_solution) | REQ-P2O-021 | Fix ablation script errors |
| REQ-SF-022 (check_and_fix_leakage) | REQ-P2O-027 | Leakage check before evaluation (delegated to inner loop via Spec 06) |

### 9.6 Spec 04 Dependencies (Inbound)

| Spec 04 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-DM-022 (Phase1Result) | REQ-P2O-019 | Initial solution s_0 and initial score h_best |

---

## 10. Change Control

### 10.1 Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2026-02-20 | MLE-STAR Team | Initial draft -- all 44 requirements |

### 10.2 Baselining Policy

This SRS is baselined at version 1.0. After baselining, changes require impact analysis against Spec 06 (inner loop depends on outer loop handoff), Spec 09 (orchestrator invokes outer loop), Spec 01 (upstream data model dependencies), Spec 02 (upstream execution harness dependencies), and Spec 03 (upstream safety agent dependencies).
