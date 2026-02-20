# SRS 05 — Phase 2 Outer Loop: Ablation Agent

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
