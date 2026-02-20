# Software Requirements Specification: MLE-STAR Phase 2 -- Inner Loop Refinement

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

---

## 5. Inner Loop Orchestration Requirements

### 5.1 Inner Loop Function Signature

> **REQ-P2I-016**: *Inner Loop Function* -- The system shall define an async function `run_phase2_inner_loop` with the following signature and behavior:
>
> ```python
> async def run_phase2_inner_loop(
>     solution: SolutionScript,        # current best solution s_t
>     code_block: CodeBlock,           # target code block c_t
>     initial_plan: str,               # initial refinement plan p_0
>     best_score: float,               # current h_best from outer loop
>     task: TaskDescription,
>     config: PipelineConfig,
> ) -> InnerLoopResult
> ```
>
> - The function shall implement Algorithm 2 lines 9-25 (the inner loop).
> - The function shall return an `InnerLoopResult` (REQ-P2I-037) containing the best solution, best score, and full history of attempts.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given valid inputs with `config.inner_loop_steps = 4`, the function shall attempt up to 4 refinement strategies and return the best result.
> - Source: REF-01 Algorithm 2 lines 9-25, REQ-P2O-026 (outer loop handoff signature)

### 5.2 Initial Attempt (k=0)

> **REQ-P2I-017**: *Initial Attempt Execution* -- The inner loop shall execute the initial attempt (k=0) as follows:
>
> 1. Invoke `invoke_coder(code_block.content, initial_plan)` to obtain c_t^0.
> 2. If A_coder returns `None` (unparseable output), record a failed attempt and proceed to k=1.
> 3. Construct s_t^0 by calling `solution.replace_block(code_block.content, c_t^0)` (REQ-DM-010).
> 4. If `replace_block` raises `ValueError` (code block not found), record a failed attempt and proceed to k=1.
> 5. Invoke `check_and_fix_leakage(s_t^0)` (REQ-SF-020, REQ-SF-022) before evaluation.
> 6. Evaluate h(s_t^0) using `evaluate_with_retry` (REQ-EX-021) with the debug callback (REQ-SF-007).
> 7. If h(s_t^0) >= h_best (using `is_improvement_or_equal`, REQ-DM-029), update s_final and h_best.
> 8. Record the attempt as a `RefinementAttempt` (REQ-DM-042).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a valid code block and plan, the initial attempt shall produce an evaluated solution and a RefinementAttempt record.
> - Source: REF-01 Algorithm 2 lines 9-15

> **REQ-P2I-018**: *Initial Attempt Uses Extractor Plan* -- The initial attempt (k=0) shall use the `initial_plan` parameter (p_0) directly, without invoking A_planner. A_planner is only invoked for subsequent attempts (k >= 1).
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Rationale: Algorithm 2 specifies `c_t^0 = A_coder(c_t, p_0)` at line 9, where p_0 comes from A_extractor (outer loop), not A_planner.
> - Source: REF-01 Algorithm 2 lines 9 vs. 17

### 5.3 Subsequent Attempts (k=1..K-1)

> **REQ-P2I-019**: *Subsequent Attempt Iteration* -- For each inner step k from 1 to K-1 (where K = `config.inner_loop_steps`), the inner loop shall:
>
> 1. Invoke `invoke_planner(code_block.content, accumulated_plans, accumulated_scores)` to obtain p_k.
> 2. If A_planner returns `None` (empty response), skip this attempt and proceed to k+1.
> 3. Invoke `invoke_coder(code_block.content, p_k)` to obtain c_t^k.
> 4. If A_coder returns `None` (unparseable output), record a failed attempt and proceed to k+1.
> 5. Construct s_t^k by calling `solution.replace_block(code_block.content, c_t^k)` (REQ-DM-010).
> 6. If `replace_block` raises `ValueError`, record a failed attempt and proceed to k+1.
> 7. Invoke `check_and_fix_leakage(s_t^k)` (REQ-SF-020, REQ-SF-022) before evaluation.
> 8. Evaluate h(s_t^k) using `evaluate_with_retry` (REQ-EX-021) with the debug callback (REQ-SF-007).
> 9. If h(s_t^k) >= h_best (using `is_improvement_or_equal`, REQ-DM-029), update s_final and h_best.
> 10. Record the attempt as a `RefinementAttempt` (REQ-DM-042).
> 11. Append p_k to `accumulated_plans` and h(s_t^k) to `accumulated_scores`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given `config.inner_loop_steps = 4`, the loop shall attempt steps k=1, k=2, k=3 (3 iterations after the initial attempt at k=0), invoking A_planner before each.
> - Source: REF-01 Algorithm 2 lines 16-25

> **REQ-P2I-020**: *A_planner Receives Full History* -- At inner step k, the A_planner invocation shall receive the complete history of all previous attempts (j=0 through j=k-1), including:
>
> - All plan texts: `[p_0, p_1, ..., p_{k-1}]`
> - All scores: `[h(s_t^0), h(s_t^1), ..., h(s_t^{k-1})]`
>
> - Scores for failed attempts (execution error, replacement failure, or unparseable coder output) shall be represented as `None` in the scores list.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: At k=2, A_planner shall receive plans and scores for attempts k=0 and k=1.
> - Source: REF-01 Algorithm 2 line 17 -- `p_k = A_planner(c_t, {(p_j, h(s_t^j))}_{j=0}^{k-1})`

> **REQ-P2I-021**: *A_coder Always Receives Original Code Block* -- At every inner step k, A_coder shall receive the original code block c_t (not any previously refined version c_t^j). Each refinement attempt starts from the original code block and applies a new plan.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: At k=2, the `code_block` argument to `invoke_coder` shall be `code_block.content` (the original c_t), not c_t^0 or c_t^1.
> - Source: REF-01 Algorithm 2 line 18 -- `c_t^k = A_coder(c_t, p_k)` (c_t is the original, not c_t^{k-1})

### 5.4 Code Block Replacement

> **REQ-P2I-022**: *Code Block Replacement Mechanism* -- At each inner step k, the system shall construct the candidate solution s_t^k by calling `solution.replace_block(code_block.content, c_t_k)` (REQ-DM-010), where:
>
> - `solution` is the original solution s_t (not the result of a previous inner step).
> - `code_block.content` is the original code block c_t.
> - `c_t_k` is the refined code block produced by A_coder at step k.
>
> - The replacement produces a new `SolutionScript` instance; the original `solution` is not mutated.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `solution.replace_block(c_t, c_t_k).content` shall equal `solution.content` with the first occurrence of `c_t` replaced by `c_t_k`.
> - Source: REF-01 Algorithm 2 lines 10, 19 -- `s_t^k = s_t.replace(c_t, c_t^k)`

> **REQ-P2I-023**: *Replacement Uses Original Solution* -- Each code block replacement shall be performed against the original solution s_t (the solution passed to `run_phase2_inner_loop`), not against any previously modified s_t^j. This ensures that each inner step is independent and that the replacement target c_t is always present.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Rationale: Since c_t is replaced by c_t^k, the code block c_t may no longer exist in s_t^k. Replacing against the original guarantees c_t is found.
> - Source: REF-01 Algorithm 2 lines 10, 19 -- `s_t^k = s_t.replace(c_t, c_t^k)` (s_t is the fixed base)

### 5.5 Score Tracking

> **REQ-P2I-024**: *Best Score Initialization* -- The inner loop shall initialize its best-tracking state from the `best_score` parameter:
>
> - `local_best_score = best_score` (the h_best from the outer loop)
> - `local_best_solution = solution` (the s_t from the outer loop)
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 2 lines 1-2 (outer loop initialization, carried into inner loop)

> **REQ-P2I-025**: *Best Score Update* -- After each successful evaluation at inner step k, the system shall compare h(s_t^k) against `local_best_score` using `is_improvement_or_equal(new_score=h(s_t^k), old_score=local_best_score, direction=task.metric_direction)` (REQ-DM-029):
>
> - If the comparison returns `True`: update `local_best_solution = s_t^k` and `local_best_score = h(s_t^k)`.
> - If the comparison returns `False`: retain the current best.
> - Mark the corresponding `RefinementAttempt.was_improvement` accordingly.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a maximize metric, if h(s_t^0) = 0.82 and h(s_t^1) = 0.85, the best shall be updated to s_t^1 with score 0.85 after step 1.
> - Acceptance: Given a maximize metric, if h(s_t^0) = 0.85 and h(s_t^1) = 0.82, the best shall remain at s_t^0 with score 0.85.
> - Source: REF-01 Algorithm 2 lines 12-14, 21-24 -- `if h(s_t^k) >= h_best then ...`

> **REQ-P2I-026**: *Score Comparison Semantics (>=)* -- The inner loop shall use `is_improvement_or_equal` (REQ-DM-029) for the best-update check, implementing >= semantics as specified in Algorithm 2. This means that a score equal to h_best will also trigger an update, keeping the most recent solution with the best score.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given h_best = 0.82 and h(s_t^k) = 0.82 (maximize), the update condition shall be `True` and the best solution shall be updated to s_t^k.
> - Source: REF-01 Algorithm 2 lines 12, 21 -- `if h(s_t^k) >= h_best`

> **REQ-P2I-027**: *Failed Evaluation Score Handling* -- When an evaluation fails (execution error after all debug retries are exhausted, or evaluation returns `score=None`), the score for that attempt shall be recorded as `None` in both the `RefinementAttempt.score` field and the `accumulated_scores` list. A `None` score shall never trigger a best-score update.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: An attempt with `score=None` shall have `RefinementAttempt.was_improvement = False`.
> - Acceptance: A `None` score shall not be passed to `is_improvement_or_equal`.

### 5.6 History Accumulation

> **REQ-P2I-028**: *RefinementAttempt Record Construction* -- At each inner step k, the system shall construct a `RefinementAttempt` (REQ-DM-042) with the following field values:
>
> | Field | Value |
> |-------|-------|
> | `plan` | The plan text p_k used for this attempt |
> | `score` | h(s_t^k) if evaluation succeeded, `None` if evaluation failed |
> | `code_block` | The refined code block c_t^k if A_coder succeeded, empty string `""` if A_coder failed |
> | `was_improvement` | `True` if this attempt triggered a best-score update, `False` otherwise |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After K=4 inner steps, the history list shall contain exactly 4 `RefinementAttempt` records.
> - Source: REF-01 Algorithm 2 -- tracks `{(p_j, h(s_t^j))}` history

> **REQ-P2I-029**: *History List Ordering* -- The list of `RefinementAttempt` records shall be ordered by inner step index (k=0 first, k=1 second, etc.). The list shall always have length K (= `config.inner_loop_steps`), including records for skipped or failed attempts.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `attempts[0].plan == initial_plan` (the p_0 from A_extractor).

### 5.7 Safety Integration

> **REQ-P2I-030**: *Leakage Check Before Evaluation* -- At each inner step k, after constructing s_t^k via `replace_block`, and before evaluation, the system shall invoke `check_and_fix_leakage(s_t^k)` (REQ-SF-020, REQ-SF-022). The leakage-checked (and potentially corrected) solution shall be the one submitted for evaluation.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Acceptance: Every code path from `replace_block` to `evaluate_solution` shall include a `check_and_fix_leakage` call.
> - Source: REF-01 Section 3.4 -- "Every generated solution before evaluation (all phases)"; REQ-SF-022

> **REQ-P2I-031**: *Debug on Execution Error* -- When evaluating s_t^k produces an execution error, the system shall use the debug retry mechanism by invoking `evaluate_with_retry` (REQ-EX-021) with the debug callback from `make_debug_callback(task, config)` (REQ-SF-007).
>
> - The debug retry shall attempt up to `config.max_debug_attempts` fixes before declaring the attempt failed.
> - If debug succeeds, the fixed solution and its score shall be used for this inner step's result.
> - If debug exhausts all retries and the solution still errors, the attempt shall be recorded with `score=None` and `was_improvement=False`.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a refined solution that errors once and is fixed by A_debugger, the inner step shall record the score from the fixed solution.
> - Source: REF-01 Section 3.4 -- A_debugger applies to all generated code

### 5.8 Error Handling

> **REQ-P2I-032**: *Unparseable Coder Output* -- If `invoke_coder` returns `None` (the agent's response did not contain an extractable code block), the inner loop shall:
>
> 1. Log a warning indicating the coder response could not be parsed at inner step k.
> 2. Record a `RefinementAttempt` with `score=None`, `code_block=""`, and `was_improvement=False`.
> 3. Proceed to the next inner step (k+1) without performing replacement or evaluation.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given an unparseable coder response at k=1, the inner loop shall continue to k=2 and the attempt at k=1 shall have `code_block=""`.

> **REQ-P2I-033**: *Replacement Failure* -- If `solution.replace_block(code_block.content, c_t_k)` raises `ValueError` (the original code block c_t is not found in the solution), the inner loop shall:
>
> 1. Log a warning indicating the code block replacement failed at inner step k.
> 2. Record a `RefinementAttempt` with `score=None`, `code_block=c_t_k` (the coder output), and `was_improvement=False`.
> 3. Proceed to the next inner step (k+1) without performing evaluation.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a replacement failure at k=0, the inner loop shall proceed to k=1 and the attempt at k=0 shall have `score=None`.
> - Rationale: Although A-01 assumes c_t is always present, edge cases (e.g., leakage correction modifying the target area) may cause replacement to fail. Graceful handling prevents the entire inner loop from aborting.

> **REQ-P2I-034**: *Planner Failure* -- If `invoke_planner` returns `None` (empty response) at inner step k (k >= 1), the inner loop shall:
>
> 1. Log a warning indicating the planner returned an empty response at inner step k.
> 2. Record a `RefinementAttempt` with `plan="[planner failed]"`, `score=None`, `code_block=""`, and `was_improvement=False`.
> 3. Proceed to the next inner step (k+1).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a planner failure at k=1, the inner loop shall continue to k=2.

> **REQ-P2I-035**: *Accumulated Plans Include Failed Attempts* -- When accumulating history for A_planner, failed attempts (where A_coder failed, replacement failed, or evaluation failed) shall still have their plan text included in the `accumulated_plans` list, and their score included as `None` in the `accumulated_scores` list. This ensures A_planner has full context about what was tried and what failed.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: If attempt k=1 failed evaluation, A_planner at k=2 shall receive plans `[p_0, p_1]` and scores `[h(s_t^0), None]`.
> - Rationale: Even failed plans provide useful context; A_planner should avoid repeating strategies that caused errors.

### 5.9 Result Construction

> **REQ-P2I-036**: *InnerLoopResult Model* -- The system shall define a Pydantic model (or dataclass) `InnerLoopResult` with the following fields:
>
> | Field | Type | Description |
> |-------|------|-------------|
> | `best_solution` | `SolutionScript` | Best solution found across all K attempts (or the original if none improved) |
> | `best_score` | `float` | Score of the best solution |
> | `attempts` | `list[RefinementAttempt]` | Ordered list of all K attempt records |
> | `improved` | `bool` | Whether any attempt improved upon the input `best_score` |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `InnerLoopResult.improved` shall be `True` if and only if `InnerLoopResult.best_score` is strictly better than the input `best_score` (using `is_improvement`, REQ-DM-028).
> - Source: REF-01 Algorithm 2 lines 9-25 (inner loop output)

> **REQ-P2I-037**: *InnerLoopResult Construction* -- Upon completion of all K inner steps, the `run_phase2_inner_loop` function shall construct and return an `InnerLoopResult` with:
>
> - `best_solution`: the `local_best_solution` tracked throughout the loop (REQ-P2I-024, REQ-P2I-025).
> - `best_score`: the `local_best_score` tracked throughout the loop.
> - `attempts`: the full list of `RefinementAttempt` records in step order (REQ-P2I-029).
> - `improved`: `True` if `local_best_score` differs from the input `best_score` (checked via `is_improvement`, REQ-DM-028).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: The returned `InnerLoopResult.attempts` shall have length exactly `config.inner_loop_steps`.
> - Source: REQ-P2O-026, REQ-P2O-027 (outer loop expects this return type)

> **REQ-P2I-038**: *InnerLoopResult Preserves Input on No Improvement* -- If no inner step produces an improvement, the `InnerLoopResult.best_solution` shall be the original `solution` (s_t) and `InnerLoopResult.best_score` shall be the original `best_score` (h_best). The system shall never return a result with a worse score than the input.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given K=4 inner steps where all evaluations score below h_best, the returned `best_solution.content` shall equal `solution.content`.
> - Source: REF-01 Algorithm 2 -- conditional update only on `>=`

---

## 6. Non-Functional Requirements

### 6.1 Performance

> **REQ-P2I-039**: *Agent Response Overhead* -- The system overhead for processing each agent response (prompt rendering, SDK invocation setup, code block extraction, and `SolutionScript` construction) shall not exceed 500 milliseconds per invocation, excluding LLM call latency and script execution time.
>
> - Priority: Should | Verify: Test | Release: MVP

> **REQ-P2I-040**: *Inner Loop Total Duration* -- A single inner loop execution (K=4 attempts) shall complete within 30 minutes under normal conditions, excluding script execution time. The dominant cost is K LLM calls (K coder calls + up to K-1 planner calls = up to 2K-1 LLM calls total, plus leakage and debug calls).
>
> - Priority: Should | Verify: Demonstration | Release: MVP
> - Source: REF-01 Section 4 -- 24-hour total budget; Phase 2 is one of 3 phases across T outer steps

### 6.2 Reliability

> **REQ-P2I-041**: *Inner Loop Never Raises on Agent Failure* -- The `run_phase2_inner_loop` function shall not raise exceptions due to individual agent failures (A_coder unparseable, A_planner empty, replacement failure, or evaluation error). Each such failure shall be handled gracefully per REQ-P2I-032 through REQ-P2I-034, and the inner loop shall always return a valid `InnerLoopResult`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given K=4 where all 4 attempts fail (coder returns garbage), the function shall return an `InnerLoopResult` with `improved=False` and `attempts` of length 4.

> **REQ-P2I-042**: *Progressive Improvement Expectation* -- While not guaranteed, the inner loop is designed such that successive A_planner iterations produce progressively better plans, as evidenced by the paper's experimental results showing monotonically improving error reduction across inner steps (Figure 8: Step 0: 0%, Step 1: ~12.6%, Step 2: ~17.7%, Step 3: ~20.8%, Step 4: ~22.3% error reduction).
>
> This requirement documents the empirical expectation; no automated enforcement is needed.
>
> - Priority: Informational | Verify: N/A | Release: N/A
> - Source: REF-01 Figure 8

### 6.3 Observability

> **REQ-P2I-043**: *Inner Loop Logging* -- Each inner loop execution shall log the following events using Python's `logging` module at the specified levels:
>
> | Event | Level | Content |
> |-------|-------|---------|
> | Inner loop start | `INFO` | Code block length, initial plan (first 200 chars), input h_best, K value |
> | A_coder invocation start | `INFO` | Inner step k, plan text (first 200 chars) |
> | A_coder invocation complete | `INFO` | Inner step k, output code block length (or "failed to parse") |
> | A_coder unparseable response | `WARNING` | Inner step k, response summary (first 200 chars) |
> | A_planner invocation start | `INFO` | Inner step k, number of previous attempts in history |
> | A_planner invocation complete | `INFO` | Inner step k, plan text (first 200 chars) |
> | A_planner empty response | `WARNING` | Inner step k |
> | Code block replacement success | `DEBUG` | Inner step k, original block length, new block length |
> | Code block replacement failure | `WARNING` | Inner step k, error message |
> | Leakage check start | `INFO` | Inner step k, solution content length |
> | Leakage check complete | `INFO` | Inner step k, leakage found (yes/no), content changed (yes/no) |
> | Evaluation start | `INFO` | Inner step k, solution content length |
> | Evaluation complete | `INFO` | Inner step k, score (or "failed"), is_error, duration |
> | Best score updated | `INFO` | Inner step k, old best score, new best score |
> | Attempt skipped | `WARNING` | Inner step k, reason (coder failed / replacement failed / planner failed) |
> | Inner loop complete | `INFO` | Total attempts, successful evaluations, best score, improved (yes/no) |
>
> - Priority: Must | Verify: Inspection | Release: MVP

---

## 7. Constraints

### 7.1 Technology Constraints

> **REQ-P2I-044**: *SDK Agent Invocation* -- Both agents (A_coder and A_planner) shall be invoked via the Claude Agent SDK agent mechanism. They shall not use direct API calls, raw HTTP requests, or any non-SDK LLM invocation method.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-02 -- all agent interactions through the SDK

> **REQ-P2I-045**: *Single Module Organization* -- All inner loop functions and agent invocation logic defined in this spec shall reside in a single Python module (e.g., `mle_star/phase2_inner.py`).
>
> - Priority: Should | Verify: Inspection | Release: MVP

### 7.2 Algorithmic Constraints

> **REQ-P2I-046**: *Sequential Inner Loop Execution* -- The inner loop iterations shall execute sequentially, not concurrently. Each iteration depends on the accumulated history from all prior iterations (plans and scores).
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Rationale: A_planner at step k requires the scores from steps 0 through k-1; parallel execution would prevent this.
> - Source: REF-01 Algorithm 2 lines 16-25 -- sequential loop with history accumulation

> **REQ-P2I-047**: *Monotonic Best Score* -- The `local_best_score` tracked within the inner loop shall be monotonically non-decreasing (for maximize) or non-increasing (for minimize). The system shall never overwrite `local_best_solution` with a solution that has a worse score than the current `local_best_score`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After every inner step, `local_best_score` shall be >= (maximize) or <= (minimize) the value at the start of that step.
> - Source: REF-01 Algorithm 2 lines 12-14, 21-24 -- conditional update on `>=`

> **REQ-P2I-048**: *Inner Loop Iteration Count* -- The inner loop shall attempt exactly `config.inner_loop_steps` (K) iterations (REQ-DM-001, default: 4). Failed attempts count as iterations; the loop does not add extra iterations to compensate for failures.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given `config.inner_loop_steps = 4`, the inner loop shall produce exactly 4 `RefinementAttempt` records regardless of success or failure.
> - Source: REF-01 Algorithm 2 -- `for k = 1 to K-1 do` (K-1 iterations after the initial attempt, K total)

### 7.3 Data Integrity Constraints

> **REQ-P2I-049**: *Immutable Input Solution* -- The `run_phase2_inner_loop` function shall not mutate the `solution` parameter. All modifications to the solution shall produce new `SolutionScript` instances via `replace_block`. The original `solution` shall remain available as the replacement base for all K attempts.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Rationale: Each inner step replaces c_t in the original s_t, not in the result of a previous step.

> **REQ-P2I-050**: *Immutable Code Block* -- The `code_block.content` parameter shall not be modified during the inner loop. Every invocation of A_coder and every `replace_block` call shall use the original `code_block.content` as provided by the outer loop.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Rationale: Algorithm 2 uses c_t (the original extraction) consistently across all K inner steps.

---

## 8. Traceability Matrix

### 8.1 Requirements to Paper Sections

| Req ID | Paper Section | Paper Element | SDK Construct |
|--------|--------------|---------------|---------------|
| REQ-P2I-001 | Section 3.2 | A_coder agent | `AgentDefinition` |
| REQ-P2I-002 | Figure 15 | Coder prompt template | `prompt` parameter |
| REQ-P2I-003 | Alg 2 lines 9, 18 | `c_t^k = A_coder(c_t, p_k)` input | -- |
| REQ-P2I-004 | Figure 15 | `c_t^k = A_coder(c_t, p_k)` output | -- |
| REQ-P2I-005 | Figure 15 | Coder invocation function | -- |
| REQ-P2I-006 | Figure 15 | "Do not remove subsampling" | -- |
| REQ-P2I-007 | Figure 15 | "Do not introduce dummy variables" | -- |
| REQ-P2I-008 | Section 3.2 | A_planner agent | `AgentDefinition` |
| REQ-P2I-009 | Figure 16 | Planner prompt template | `prompt` parameter |
| REQ-P2I-010 | Figure 16 | History formatting | -- |
| REQ-P2I-011 | Alg 2 line 17 | `p_k = A_planner(c_t, history)` input | -- |
| REQ-P2I-012 | Figure 16 | `p_k = A_planner(c_t, history)` output | -- |
| REQ-P2I-013 | Figure 16 | Planner invocation function | -- |
| REQ-P2I-014 | Figure 16 | "novel and effective" / "differ from previous" | -- |
| REQ-P2I-015 | Figure 16 | "avoid plans which can make running time too long" | -- |
| REQ-P2I-016 | Alg 2 lines 9-25 | Inner loop function signature | -- |
| REQ-P2I-017 | Alg 2 lines 9-15 | Initial attempt (k=0) | -- |
| REQ-P2I-018 | Alg 2 line 9 | p_0 from A_extractor, not A_planner | -- |
| REQ-P2I-019 | Alg 2 lines 16-25 | Subsequent attempts (k=1..K-1) | -- |
| REQ-P2I-020 | Alg 2 line 17 | Full history to A_planner | -- |
| REQ-P2I-021 | Alg 2 line 18 | A_coder always receives original c_t | -- |
| REQ-P2I-022 | Alg 2 lines 10, 19 | `s_t^k = s_t.replace(c_t, c_t^k)` | -- |
| REQ-P2I-023 | Alg 2 lines 10, 19 | Replace against original s_t | -- |
| REQ-P2I-024 | Alg 2 lines 1-2 | Best score initialization | -- |
| REQ-P2I-025 | Alg 2 lines 12-14, 21-24 | Best score update | -- |
| REQ-P2I-026 | Alg 2 lines 12, 21 | `>=` comparison semantics | -- |
| REQ-P2I-027 | -- | Failed evaluation score handling | -- |
| REQ-P2I-028 | Alg 2 | RefinementAttempt record | REQ-DM-042 |
| REQ-P2I-029 | Alg 2 | History ordering | -- |
| REQ-P2I-030 | Section 3.4 | Leakage check before evaluation | REQ-SF-022 |
| REQ-P2I-031 | Section 3.4 | A_debugger on execution error | REQ-SF-006, REQ-SF-007 |
| REQ-P2I-032 | -- | Unparseable coder output handling | -- |
| REQ-P2I-033 | -- | Replacement failure handling | -- |
| REQ-P2I-034 | -- | Planner failure handling | -- |
| REQ-P2I-035 | Alg 2 line 17 | Failed attempts in history | -- |
| REQ-P2I-036 | Alg 2 lines 9-25 | InnerLoopResult model | -- |
| REQ-P2I-037 | Alg 2 lines 9-25 | InnerLoopResult construction | -- |
| REQ-P2I-038 | Alg 2 lines 1-2 | Preserve input on no improvement | -- |
| REQ-P2I-039 | -- | Agent response overhead | -- |
| REQ-P2I-040 | Section 4 | Inner loop duration | -- |
| REQ-P2I-041 | -- | Never raises on agent failure | -- |
| REQ-P2I-042 | Figure 8 | Progressive improvement evidence | -- |
| REQ-P2I-043 | -- | Logging | Python `logging` |
| REQ-P2I-044 | -- | SDK-only invocation | `claude-agent-sdk` |
| REQ-P2I-045 | -- | Module organization | -- |
| REQ-P2I-046 | Alg 2 lines 16-25 | Sequential execution | -- |
| REQ-P2I-047 | Alg 2 lines 12-14, 21-24 | Monotonic best score | -- |
| REQ-P2I-048 | Alg 2 lines 9, 16 | Iteration count = K | -- |
| REQ-P2I-049 | -- | Immutable input solution | -- |
| REQ-P2I-050 | -- | Immutable code block | -- |

### 8.2 Cross-References to Other Specs

| Req ID | Referenced By |
|--------|--------------|
| REQ-P2I-001 (A_coder agent def) | Spec 09 (orchestrator configures agents) |
| REQ-P2I-008 (A_planner agent def) | Spec 09 (orchestrator configures agents) |
| REQ-P2I-016 (inner loop function) | Spec 05 (outer loop invokes inner loop, REQ-P2O-026) |
| REQ-P2I-036 (InnerLoopResult) | Spec 05 (outer loop receives inner loop result, REQ-P2O-027) |
| REQ-P2I-037 (InnerLoopResult construction) | Spec 05 (outer loop post-inner-loop update, REQ-P2O-027) |

### 8.3 Spec 01 Dependencies (Inbound)

| Spec 01 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-DM-001 (PipelineConfig) | REQ-P2I-016, 048 | `inner_loop_steps` (K), `max_debug_attempts` |
| REQ-DM-006 (MetricDirection) | REQ-P2I-025, 026 | Score comparison direction |
| REQ-DM-007 (TaskDescription) | REQ-P2I-016, 025 | Task context, metric direction |
| REQ-DM-009 (SolutionScript) | REQ-P2I-016, 022, 023, 036, 049 | Solution type, input/output |
| REQ-DM-010 (replace_block) | REQ-P2I-017, 019, 022, 023, 033 | Code block replacement in solution |
| REQ-DM-012 (CodeBlock) | REQ-P2I-016, 050 | Code block input type |
| REQ-DM-013 (AgentType) | REQ-P2I-001, 008 | Agent identity enum values |
| REQ-DM-028 (is_improvement) | REQ-P2I-036, 037 | Strict improvement check for `improved` field |
| REQ-DM-029 (is_improvement_or_equal) | REQ-P2I-025, 026 | `>=` score comparison for best update |
| REQ-DM-032 (PromptRegistry) | REQ-P2I-002, 009 | Template retrieval for coder and planner |
| REQ-DM-036 (AgentConfig) | REQ-P2I-001, 008 | Agent-to-SDK mapping |
| REQ-DM-042 (RefinementAttempt) | REQ-P2I-028, 029, 036 | Attempt history records |

### 8.4 Spec 02 Dependencies (Inbound)

| Spec 02 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-EX-015 (evaluate_solution) | REQ-P2I-017, 019 | Evaluate refined solutions |
| REQ-EX-021 (evaluate_with_retry) | REQ-P2I-017, 019, 031 | Evaluate with debug retry |

### 8.5 Spec 03 Dependencies (Inbound)

| Spec 03 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-SF-005 (extract_code_block) | REQ-P2I-004, 005 | Extract code block from A_coder response |
| REQ-SF-006 (debug_solution) | REQ-P2I-031 | Debug execution errors in refined solutions |
| REQ-SF-007 (make_debug_callback) | REQ-P2I-017, 019, 031 | Debug callback for evaluate_with_retry |
| REQ-SF-020 (check_and_fix_leakage) | REQ-P2I-017, 019, 030 | Leakage detection and correction |
| REQ-SF-022 (leakage integration point) | REQ-P2I-030 | Leakage check before every evaluation |

### 8.6 Spec 05 Dependencies (Inbound)

| Spec 05 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-P2O-026 (inner loop handoff) | REQ-P2I-016 | Defines the call site and input data for the inner loop |
| REQ-P2O-027 (post-inner-loop update) | REQ-P2I-036, 037 | Defines what the outer loop expects back from the inner loop |

---

## 9. Change Control

### 9.1 Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2026-02-20 | MLE-STAR Team | Initial draft -- all 50 requirements |

### 9.2 Baselining Policy

This SRS is baselined at version 1.0. After baselining, changes require impact analysis against Spec 05 (outer loop depends on inner loop return type and function signature), Spec 09 (orchestrator invokes Phase 2 via outer loop), Spec 01 (upstream data model dependencies), Spec 02 (upstream execution harness dependencies), and Spec 03 (upstream safety agent dependencies).
