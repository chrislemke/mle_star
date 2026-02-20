# SRS 06 — Phase 2 Inner Loop: Agents

| Field | Value |
|-------|-------|
| Version | 0.1.0 |
| Date | 2026-02-20 |
| Status | Draft |
| Spec ID | 06 of 09 |
| Requirement Prefix | REQ-P2I- |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Product Perspective](#2-product-perspective)
3. [A_coder Requirements](#3-a_coder-requirements)
4. [A_planner Requirements](#4-a_planner-requirements)
5. [Inner Loop Orchestration Requirements](#5-inner-loop-orchestration-requirements)
6. [Non-Functional Requirements](#6-non-functional-requirements)
7. [Constraints](#7-constraints)
8. [Traceability Matrix](#8-traceability-matrix)
9. [Change Control](#9-change-control)

---

## 1. Introduction

### 1.1 Purpose

This SRS defines the inner loop of Phase 2 refinement in the MLE-STAR pipeline: implementing refinement plans on a target code block, evaluating results, and planning successive improvement strategies. It covers agents A_coder and A_planner, as well as the inner loop control flow from Algorithm 2 (lines 9-25).

Intended audience: developers implementing the MLE-STAR system using the Claude Agent SDK for Python.

### 1.2 Scope

**Product name**: MLE-STAR (Machine Learning Engineering agent via Search and Targeted Refinement)

**What this spec covers**:
- A_coder agent definition, prompt template, input/output contract, and output constraints
- A_planner agent definition, prompt template, input/output contract, and output constraints
- Inner loop orchestration function: initial attempt (k=0), subsequent attempts (k=1..K-1)
- Code block replacement using SolutionScript.replace_block
- Safety integration: A_leakage before evaluation, A_debugger on execution errors
- Score tracking and best-solution selection across all K attempts
- History accumulation as RefinementAttempt records
- Error handling for unparseable agent output and failed replacements
- InnerLoopResult return type construction

**Out of scope**:
- Data model definitions (covered by Spec 01)
- Script execution and subprocess management (covered by Spec 02)
- Safety agent internals (covered by Spec 03; this spec defines invocation points only)
- Outer loop targeting (A_abl, A_summarize, A_extractor) (covered by Spec 05)
- Phase 1 initial solution generation (covered by Spec 04)
- Phase 3 ensemble (covered by Spec 07)
- Orchestration coordination across phases (covered by Spec 09)

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|------|-----------|
| SRS | Software Requirements Specification |
| MLE-STAR | ML Engineering agent with web Search and TArgeted code block Refinement |
| A_coder | Coder agent -- implements a refinement plan on the target code block |
| A_planner | Planner agent -- proposes the next refinement strategy given previous attempt history |
| c_t | Code block -- the exact code substring extracted for refinement at outer step t |
| c_t^k | Refined code block -- the improved version of c_t produced by A_coder at inner step k |
| p_k | Refinement plan -- natural language description of the improvement strategy at inner step k |
| p_0 | Initial plan -- the refinement plan proposed by A_extractor (provided by the outer loop) |
| s_t | Current best solution at outer step t |
| s_t^k | Candidate solution -- s_t with c_t replaced by c_t^k |
| h(s) | Score function -- maps a solution script to a real-valued performance score |
| h_best | Best score -- the highest score achieved so far |
| K | Number of inner loop iterations per code block (from PipelineConfig.inner_loop_steps, default: 4) |

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
| REF-08 | Spec 05 -- Phase 2 Outer Loop | 0.1.0 | `thoughts/specs/05_phase2_ablation_and_extraction.md` |

### 1.5 Document Overview

- Section 3: A_coder requirements (agent definition, prompt template, input/output contract, output constraints)
- Section 4: A_planner requirements (agent definition, prompt template, input/output contract, output constraints)
- Section 5: Inner loop orchestration requirements (function signature, initial attempt, iteration loop, score tracking, safety integration, error handling, result construction)
- Section 6: Non-functional requirements (performance, reliability, observability)
- Section 7: Constraints (technology, algorithmic invariants, data integrity)
- Section 8: Traceability matrix

---

## 2. Product Perspective

### 2.1 System Context

This spec defines the inner loop of Phase 2 -- the refinement mechanism that iteratively improves a target code block using K strategies. The inner loop is invoked once per outer loop iteration (Spec 05), receiving a code block c_t and an initial plan p_0, and returning the best refined solution found across K attempts.

```
Spec 05 (Phase 2 Outer Loop)
  |-- c_t, p_0 from A_extractor
  |
  v
Spec 06 (this) -- Phase 2 Inner Loop
  k=0: A_coder(c_t, p_0) -> c_t^0 -> replace -> evaluate
  for k=1 to K-1:
    |
    |-- A_planner(c_t, history) -> p_k
    |-- A_coder(c_t, p_k)      -> c_t^k
    |-- s_t^k = s_t.replace(c_t, c_t^k)
    |-- A_leakage(s_t^k) [before evaluation]
    |-- evaluate h(s_t^k) [with A_debugger on error]
    |-- track best: if h(s_t^k) >= h_best -> update
  end for
  |
  v
InnerLoopResult -> Spec 05 (outer loop post-inner-loop update)
```

### 2.2 Dependency Diagram

```
Spec 01 (Data Models)
  |-- SolutionScript (REQ-DM-009)
  |-- SolutionScript.replace_block (REQ-DM-010)
  |-- CodeBlock (REQ-DM-012)
  |-- PipelineConfig (REQ-DM-001)
  |-- RefinementAttempt (REQ-DM-042)
  |-- MetricDirection (REQ-DM-006)
  |-- AgentType (REQ-DM-013)
  |-- AgentConfig (REQ-DM-036)
  |-- PromptRegistry (REQ-DM-032)
  |-- TaskDescription (REQ-DM-007)
  |-- is_improvement_or_equal (REQ-DM-029)
  |
Spec 02 (Execution Harness)
  |-- evaluate_solution (REQ-EX-015)
  |-- evaluate_with_retry (REQ-EX-021)
  |
Spec 03 (Safety Modules)
  |-- check_and_fix_leakage (REQ-SF-020, REQ-SF-022)
  |-- debug_solution / make_debug_callback (REQ-SF-006, REQ-SF-007)
  |-- extract_code_block (REQ-SF-005)
  |
Spec 05 (Phase 2 Outer Loop)
  |-- Provides c_t, p_0, s_t, h_best (REQ-P2O-026)
  |
  v
Spec 06 (this) -- Phase 2 Inner Loop
  |
  v
Referenced by:
  Spec 05 (outer loop receives inner loop result, REQ-P2O-027)
  Spec 09 (orchestrator, indirectly via Spec 05)
```

### 2.3 Product Functions Summary

1. Implement a refinement plan on a target code block via A_coder
2. Propose successive refinement strategies using feedback from previous attempts via A_planner
3. Replace the target code block in the solution with the refined version
4. Evaluate refined solutions with safety checks (leakage detection, debug-on-error)
5. Track the best score and best solution across all K inner loop iterations
6. Accumulate a history of RefinementAttempt records for observability and A_planner context
7. Handle errors gracefully (unparseable output, failed replacement, execution errors)
8. Return the best solution, best score, and full attempt history to the outer loop

### 2.4 Operating Environment

- **Runtime**: Python 3.10+
- **SDK**: `claude-agent-sdk` v0.1.39+
- **Execution**: A_coder and A_planner are text-only agents (no tools, no structured output); evaluation delegates to Spec 02 harness

### 2.5 Assumptions and Dependencies

| ID | Assumption | Impact if Invalid |
|----|-----------|-------------------|
| A-01 | The code block c_t provided by the outer loop is an exact substring of s_t.content | `SolutionScript.replace_block` will raise `ValueError` |
| A-02 | A_coder can reliably produce a code block in its response (fenced or unfenced) | Unparseable responses will be skipped, reducing effective K |
| A-03 | A_planner can generate novel strategies that differ from previous plans | Repeated strategies may produce diminishing returns |
| A-04 | K=4 inner loop steps provide sufficient exploration for each code block | Higher K may improve results but increases latency and cost |
| A-05 | A single code block replacement is sufficient per inner step | Multi-block changes would require a different replacement model |

| ID | Dependency | Owner | Risk if Unavailable |
|----|-----------|-------|---------------------|
| D-01 | Spec 01 types (SolutionScript, CodeBlock, RefinementAttempt, PipelineConfig, etc.) | Spec 01 | Cannot construct agent inputs or results |
| D-02 | Spec 02 execution harness (evaluate_solution, evaluate_with_retry) | Spec 02 | Cannot evaluate refined solutions |
| D-03 | Spec 03 safety agents (check_and_fix_leakage, debug_solution, extract_code_block) | Spec 03 | Cannot ensure leakage-free evaluation or debug failures |
| D-04 | Spec 05 outer loop handoff (c_t, p_0, s_t, h_best) | Spec 05 | No target code block or initial plan to refine |
| D-05 | `claude-agent-sdk` v0.1.39+ | Anthropic | Cannot define or invoke agents |
| D-06 | PromptRegistry with templates for coder and planner agents | Spec 01 | Cannot construct agent prompts |

---

## 3. A_coder Requirements

### 3.1 Agent Definition

> **REQ-P2I-001**: *A_coder Agent Definition* -- The system shall define an `AgentDefinition`-compatible configuration for the coder agent with the following properties:
>
> | Property | Value |
> |----------|-------|
> | `agent_type` | `AgentType.coder` |
> | `description` | Agent that implements a refinement plan on a target code block |
> | `prompt` | Rendered from the coder template (Figure 15, REQ-DM-032) |
> | `tools` | `None` (no tools; operates purely on provided code block) |
> | `output_schema` | `None` (free-form response containing a single code block) |
> | `model` | `None` (inherit from orchestrator) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `AgentConfig(agent_type=AgentType.coder).to_agent_definition()` shall produce a valid dictionary for `ClaudeAgentOptions.agents`.
> - Source: REF-01 Section 3.2, Figure 15

> **REQ-P2I-002**: *A_coder Prompt Template* -- The coder agent prompt shall be constructed by rendering the Figure 15 template from the `PromptRegistry` (REQ-DM-032) with the following variables:
>
> | Variable | Type | Description |
> |----------|------|-------------|
> | `code_block` | `str` | The target code block content (`CodeBlock.content` or `c_t`) |
> | `plan` | `str` | The refinement plan to implement (`p_k`) |
>
> - The rendered prompt shall include all instructions from Figure 15:
>   1. Kaggle grandmaster persona introduction
>   2. Goal: refine the code block for better performance based on the improvement plan
>   3. The code block presented in full
>   4. The improvement plan presented in full
>   5. Instruction to implement the improvement plan on the code block
>   6. Instruction to not remove subsampling if it exists
>   7. Instruction to not introduce dummy variables (since variables including actual data are defined earlier)
>   8. Response format: a single markdown code block (wrapped in ```) which is the improved code block, with no additional headings or text
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 15

### 3.2 Input/Output Contract

> **REQ-P2I-003**: *A_coder Input Contract* -- The coder agent shall accept two inputs:
>
> 1. `code_block: str` -- the exact code block content to be improved (c_t).
> 2. `plan: str` -- the natural language refinement plan to implement (p_k).
>
> - Precondition: `code_block` is non-empty and `plan` is non-empty.
> - Error: If either input is empty, the invocation function shall raise `ValueError` with a descriptive message.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 2 line 9 -- `c_t^0 = A_coder(c_t, p_0)` and line 18 -- `c_t^k = A_coder(c_t, p_k)`

> **REQ-P2I-004**: *A_coder Output Contract* -- The coder agent shall return an improved code block string (c_t^k) extracted from the agent's response.
>
> - The response shall be parsed using `extract_code_block()` (REQ-SF-005) to obtain the refined code block.
> - The extracted code block shall be a non-empty string suitable for replacing c_t in the solution script via `SolutionScript.replace_block()` (REQ-DM-010).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given an agent response containing a fenced code block, the returned string shall contain only the code within the fences (excluding fence markers and language identifier).
> - Source: REF-01 Figure 15 -- "Your response should be a single markdown code block"

> **REQ-P2I-005**: *A_coder Invocation Function* -- The system shall define an async function `invoke_coder(code_block: str, plan: str) -> str | None` that:
>
> 1. Renders the A_coder prompt template with the provided `code_block` and `plan`.
> 2. Invokes the coder agent via the Claude Agent SDK.
> 3. Extracts the code block from the response using `extract_code_block()` (REQ-SF-005).
> 4. Returns the extracted code block string, or `None` if extraction fails.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a valid code block and plan, the function shall return a non-None string containing the improved code block.

### 3.3 Output Constraints

> **REQ-P2I-006**: *A_coder Subsampling Preservation* -- The coder agent prompt (REQ-P2I-002) shall include the instruction "Do not remove subsampling if exists." The system shall not perform automated verification of subsampling preservation; enforcement is via prompt instruction.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 15 -- "But do not remove subsampling if exists."

> **REQ-P2I-007**: *A_coder No Dummy Variables* -- The coder agent prompt (REQ-P2I-002) shall include the instruction that all variables including actual data are defined earlier (since the agent is seeing a code block, not the full script), and therefore the agent must not introduce dummy variables.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 15 -- "Note that all the variable including actual data is defined earlier (since you are just seeing a code block), therefore do not introduce dummy variables."

---

## 4. A_planner Requirements

### 4.1 Agent Definition

> **REQ-P2I-008**: *A_planner Agent Definition* -- The system shall define an `AgentDefinition`-compatible configuration for the planner agent with the following properties:
>
> | Property | Value |
> |----------|-------|
> | `agent_type` | `AgentType.planner` |
> | `description` | Agent that proposes the next refinement strategy using feedback from previous attempts |
> | `prompt` | Rendered from the planner template (Figure 16, REQ-DM-032) |
> | `tools` | `None` (no tools; operates purely on provided context) |
> | `output_schema` | `None` (free-form natural language response) |
> | `model` | `None` (inherit from orchestrator) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `AgentConfig(agent_type=AgentType.planner).to_agent_definition()` shall produce a valid dictionary for `ClaudeAgentOptions.agents`.
> - Source: REF-01 Section 3.2, Figure 16

> **REQ-P2I-009**: *A_planner Prompt Template* -- The planner agent prompt shall be constructed by rendering the Figure 16 template from the `PromptRegistry` (REQ-DM-032) with the following variables:
>
> | Variable | Type | Description |
> |----------|------|-------------|
> | `code_block` | `str` | The target code block content (`c_t`) |
> | `plans` | `list[str]` | List of previous refinement plan texts (p_0 through p_{k-1}) |
> | `scores` | `list[float \| None]` | List of scores achieved by each previous plan (h(s_t^0) through h(s_t^{k-1})) |
>
> - The rendered prompt shall include all instructions from Figure 16:
>   1. Kaggle grandmaster persona introduction
>   2. Goal: improve the code block for better performance
>   3. The code block presented in full
>   4. Previous improvement plans with their scores, formatted as numbered entries (each with "Plan:" and "Score:" labels)
>   5. Instruction to suggest a better plan that is novel and effective
>   6. Instruction to avoid plans that cause excessive runtime (e.g., searching hyperparameters in a very large search space)
>   7. Instruction that the suggested plan should differ from previous plans and should receive a higher score
>   8. Response format: a brief outline/sketch of the proposed solution in natural language (3-5 sentences), with no additional headings or text
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 16

> **REQ-P2I-010**: *A_planner History Formatting* -- The prompt template rendering for A_planner shall format the history of previous attempts as follows:
>
> ```
> # Improvement plans you have tried
>
> ## Plan: {plans[0]}
> ## Score: {scores[0]}
>
> ## Plan: {plans[1]}
> ## Score: {scores[1]}
> ...
> ```
>
> - Each plan-score pair shall be presented in order (k=0 first, then k=1, etc.).
> - Scores that are `None` (e.g., from failed evaluations) shall be rendered as `"N/A (evaluation failed)"`.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given plans `["plan A", "plan B"]` and scores `[0.82, None]`, the formatted section shall contain `"## Plan: plan A\n## Score: 0.82"` followed by `"## Plan: plan B\n## Score: N/A (evaluation failed)"`.
> - Source: REF-01 Figure 16 -- history format

### 4.2 Input/Output Contract

> **REQ-P2I-011**: *A_planner Input Contract* -- The planner agent shall accept three inputs:
>
> 1. `code_block: str` -- the target code block content (c_t, the original, not any refined version).
> 2. `plans: list[str]` -- list of previous refinement plan texts from inner steps 0 through k-1.
> 3. `scores: list[float | None]` -- list of scores achieved by each previous plan.
>
> - Precondition: `code_block` is non-empty, `len(plans) >= 1`, `len(plans) == len(scores)`.
> - Error: If `plans` is empty, the invocation function shall raise `ValueError("At least one previous plan is required for A_planner")`.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 2 line 17 -- `p_k = A_planner(c_t, {(p_j, h(s_t^j))}_{j=0}^{k-1})`

> **REQ-P2I-012**: *A_planner Output Contract* -- The planner agent shall return a natural language refinement plan string (p_k).
>
> - The plan shall be the complete text content of the agent's response (no code block extraction, no structured parsing), stripped of leading and trailing whitespace.
> - The plan shall be a brief outline in 3-5 sentences describing the proposed improvement strategy.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a valid history of previous attempts, the returned plan shall be a non-empty string.
> - Source: REF-01 Figure 16 -- "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences)."

> **REQ-P2I-013**: *A_planner Invocation Function* -- The system shall define an async function `invoke_planner(code_block: str, plans: list[str], scores: list[float | None]) -> str | None` that:
>
> 1. Validates that `len(plans) >= 1` and `len(plans) == len(scores)`.
> 2. Renders the A_planner prompt template with the provided arguments.
> 3. Invokes the planner agent via the Claude Agent SDK.
> 4. Returns the agent's response text stripped of whitespace, or `None` if the response is empty.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a valid history, the function shall return a non-None, non-empty string.

### 4.3 Output Constraints

> **REQ-P2I-014**: *A_planner Novelty Constraint* -- The planner agent prompt (REQ-P2I-009) shall instruct the agent that the suggested plan must be novel and differ from all previously tried plans. This is enforced via prompt instruction, not automated verification.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 16 -- "The suggested plan must be novel and effective." and "The suggested plan should be differ from the previous plans you have tried."

> **REQ-P2I-015**: *A_planner Runtime Constraint* -- The planner agent prompt (REQ-P2I-009) shall instruct the agent to avoid plans that would cause excessive runtime (e.g., searching hyperparameters in a very large search space). This is enforced via prompt instruction, not automated verification.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 16 -- "Please avoid plans which can make the solution's running time too long (e.g., searching hyperparameters in a very large search space)."
