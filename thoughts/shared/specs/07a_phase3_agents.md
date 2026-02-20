# SRS 07 — Phase 3: Agents

| Field | Value |
|-------|-------|
| Version | 0.1.0 |
| Date | 2026-02-20 |
| Status | Draft |
| Spec ID | 07 of 09 |
| Requirement Prefix | REQ-P3- |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Product Perspective](#2-product-perspective)
3. [A_ens_planner Requirements](#3-a_ens_planner-requirements)
4. [A_ensembler Requirements](#4-a_ensembler-requirements)
5. [Algorithm 3 Orchestration Requirements](#5-algorithm-3-orchestration-requirements)
6. [Non-Functional Requirements](#6-non-functional-requirements)
7. [Constraints](#7-constraints)
8. [Traceability Matrix](#8-traceability-matrix)
9. [Change Control](#9-change-control)

---

## 1. Introduction

### 1.1 Purpose

This SRS defines Phase 3 of the MLE-STAR pipeline: ensembling multiple final solutions from parallel Phase 2 refinement runs into a single best-performing ensemble solution. It specifies two agents (A_ens_planner and A_ensembler), their orchestration according to Algorithm 3 from the paper, and the safety integration points that produce a validated ensemble solution ready for final submission.

Intended audience: developers implementing the MLE-STAR system using the Claude Agent SDK for Python.

### 1.2 Scope

**Product name**: MLE-STAR (Machine Learning Engineering agent via Search and Targeted Refinement)

**What this spec covers**:
- A_ens_planner agent definition, prompt template, input/output contract, and history formatting
- A_ensembler agent definition, prompt template, tools, input/output contract, and output constraints
- Algorithm 3 orchestration: initial ensemble round (r=0), iterative ensemble rounds (r=1..R-1), best selection
- Skip condition when only a single solution is available (L=1)
- Safety integration: A_leakage before evaluation, A_debugger on execution errors
- EnsembleAttempt history accumulation
- Phase3Result construction

**Out of scope**:
- Data model definitions (covered by Spec 01)
- Script execution and subprocess management (covered by Spec 02)
- Safety agent internals (covered by Spec 03; this spec only defines invocation points)
- Phase 1 initial solution generation (covered by Spec 04)
- Phase 2 refinement (covered by Specs 05-06)
- Submission finalization (covered by Spec 08)
- Orchestrator control flow (covered by Spec 09)

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|------|-----------|
| SRS | Software Requirements Specification |
| MLE-STAR | ML Engineering agent with web Search and TArgeted code block Refinement |
| A_ens_planner | Ensemble planner agent -- proposes ensemble strategies using history of previous attempts |
| A_ensembler | Ensembler agent -- implements the ensemble plan on L solution scripts into a single program |
| s_final^l | Final solution from the l-th parallel Phase 2 refinement path |
| e_r | Ensemble plan at round r -- natural language description of the ensemble strategy |
| s_ens^r | Ensemble solution at round r -- single-file Python script implementing e_r |
| h(s) | Score function -- maps a solution script to a real-valued performance score |
| s_ens* | Best ensemble solution -- the ensemble with the highest (or lowest) score across all rounds |
| r* | Index of the best ensemble round |
| L | Number of parallel solutions for ensembling (from PipelineConfig, default: 2) |
| R | Ensemble strategy exploration rounds (from PipelineConfig, default: 5) |
| Algorithm 3 | "Ensembling Final Solutions" algorithm from REF-01 Appendix B |

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

- Section 3: A_ens_planner requirements (agent definition, prompt template, input/output contract, history formatting)
- Section 4: A_ensembler requirements (agent definition, prompt template, tools, input/output contract, output constraints)
- Section 5: Algorithm 3 orchestration (entry point, skip condition, initial round, iteration loop, best selection, safety integration, error handling, result construction)
- Section 6: Non-functional requirements (performance, reliability, observability)
- Section 7: Constraints (technology, algorithm fidelity, prompt fidelity)
- Section 8: Traceability matrix

---

## 2. Product Perspective

### 2.1 System Context

Phase 3 is the final computational phase of the MLE-STAR pipeline. It receives L final solutions (one from each parallel Phase 2 refinement path) and produces a single best ensemble solution s_ens* that becomes the input to Phase 4 (submission finalization, Spec 08). Phase 3 implements Algorithm 3 from the paper.

```
Spec 01 (Data Models)
  |-- PipelineConfig (REQ-DM-001) -- L, R parameters
  |-- SolutionScript (REQ-DM-009)
  |-- Phase3Result (REQ-DM-024)
  |-- EnsembleAttempt (REQ-DM-043)
  |-- AgentType (REQ-DM-013)
  |-- AgentConfig (REQ-DM-036)
  |-- PromptRegistry (REQ-DM-032)
  |-- TaskDescription (REQ-DM-007)
  |-- MetricDirection (REQ-DM-006)
  |
Spec 02 (Execution Harness)
  |-- evaluate_solution (REQ-EX-015)
  |-- evaluate_with_retry (REQ-EX-021)
  |-- verify_submission (REQ-EX-024)
  |
Spec 03 (Safety Agents)
  |-- extract_code_block (REQ-SF-005)
  |-- check_and_fix_leakage (REQ-SF-020, REQ-SF-022)
  |-- debug_solution / make_debug_callback (REQ-SF-006, REQ-SF-007)
  |
Spec 05 (Phase 2 Outer Loop)
  |-- Phase2Result.best_solution (provides the L input solutions)
  |
  v
Spec 07 (this) -- Phase 3: Ensemble Optimization
  r=0: A_ens_planner(solutions)              -> e_0
       A_ensembler(e_0, solutions)            -> s_ens^0
       evaluate h(s_ens^0)
  for r=1 to R-1:
       A_ens_planner(solutions, history)      -> e_r
       A_ensembler(e_r, solutions)            -> s_ens^r
       evaluate h(s_ens^r)
  end for
  s_ens* = argmax/argmin over r
  |
  v
Used by: Spec 08 (finalization takes ensemble output)
         Spec 09 (orchestrator invokes Phase 3)
```

### 2.2 Product Functions Summary

1. Propose an ensemble strategy from L solution scripts using A_ens_planner
2. Implement the ensemble strategy into a single-file Python script using A_ensembler
3. Evaluate each ensemble solution with safety checks (leakage detection, debug-on-error)
4. Iterate R rounds with feedback from previous plans and scores
5. Select the best-scoring ensemble solution across all R rounds
6. Skip ensemble entirely when only one solution is available (L=1)
7. Construct and return a Phase3Result

### 2.3 Operating Environment

- **Runtime**: Python 3.10+
- **SDK**: `claude-agent-sdk` v0.1.39+
- **Execution**: Ensemble script evaluation delegates to Spec 02 harness
- **Tools**: A_ens_planner requires no tools; A_ensembler requires `["Read"]` (to inspect data files)

### 2.4 Assumptions and Dependencies

| ID | Assumption | Impact if Invalid |
|----|-----------|-------------------|
| A-01 | L >= 2 solutions are available for ensembling in the standard case | If L=1, ensemble is skipped (the single solution is returned directly) |
| A-02 | All L input solutions are executable and have non-None scores from Phase 2 | Non-executable solutions provide poor material for ensembling |
| A-03 | LLM can generate a self-contained single-file ensemble script from L solution scripts and a plan | A_ensembler may produce non-executable scripts, requiring debugger intervention |
| A-04 | R=5 ensemble rounds provide sufficient exploration of ensemble strategies | Higher R may improve results but increases latency and cost |
| A-05 | Agent-driven ensembling outperforms naive averaging on complex tasks | Paper Table 4 shows equivalent overall medals but higher gold rate for agent-driven approach |

| ID | Dependency | Owner | Risk if Unavailable |
|----|-----------|-------|---------------------|
| D-01 | Spec 01 types (PipelineConfig, SolutionScript, Phase3Result, EnsembleAttempt, AgentType, AgentConfig, PromptRegistry, TaskDescription, MetricDirection) | Spec 01 | Cannot construct inputs, outputs, or compare scores |
| D-02 | Spec 02 execution harness (evaluate_solution, evaluate_with_retry, verify_submission) | Spec 02 | Cannot evaluate ensemble scripts |
| D-03 | Spec 03 safety agents (extract_code_block, check_and_fix_leakage, make_debug_callback) | Spec 03 | Cannot extract code from responses, cannot run safety checks |
| D-04 | `claude-agent-sdk` v0.1.39+ with Read tool | Anthropic | A_ensembler cannot inspect data files |
| D-05 | PromptRegistry with templates for ens_planner (Figure 17) and ensembler (Figure 18) | Spec 01 | Cannot construct agent prompts |
| D-06 | L Phase 2 final solutions from Spec 05 | Spec 05 | No input solutions to ensemble |

---

## 3. A_ens_planner Requirements

### 3.1 Agent Definition

> **REQ-P3-001**: *A_ens_planner Agent Definition* -- The system shall define an `AgentDefinition`-compatible configuration for the ensemble planner agent with the following properties:
>
> | Property | Value |
> |----------|-------|
> | `agent_type` | `AgentType.ens_planner` |
> | `description` | Agent that proposes ensemble strategies for combining L solution scripts |
> | `prompt` | Rendered from the ens_planner template (Figure 17, REQ-DM-032) |
> | `tools` | `None` (no tools; operates purely on provided solution scripts and history) |
> | `output_schema` | `None` (free-form natural language response) |
> | `model` | `None` (inherit from orchestrator) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `AgentConfig(agent_type=AgentType.ens_planner).to_agent_definition()` shall produce a valid dictionary for `ClaudeAgentOptions.agents`.
> - Source: REF-01 Section 3.3, Figure 17

> **REQ-P3-002**: *A_ens_planner Prompt Template* -- The ensemble planner agent prompt shall be constructed by rendering the Figure 17 template from the `PromptRegistry` (REQ-DM-032) with the following variables:
>
> | Variable | Type | Description |
> |----------|------|-------------|
> | `L` | `int` | Number of solution scripts being ensembled |
> | `solutions` | `list[str]` | Full source code of each of the L solution scripts |
> | `plans` | `list[str]` | List of previous ensemble plan texts (empty on first invocation) |
> | `scores` | `list[float \| None]` | List of scores from previous ensemble attempts (empty on first invocation) |
>
> - The rendered prompt shall include all instructions from Figure 17:
>   1. Kaggle grandmaster persona introduction
>   2. Goal: ensemble L Python Solutions for better performance
>   3. All L solution scripts presented in full, each labeled as "# {n}th Python Solution"
>   4. Previous ensemble plans and their scores (when history is non-empty)
>   5. Instruction to suggest a better plan that concentrates on merging, not other parts like hyperparameters
>   6. Instruction that the plan must be easy to implement, novel, and effective
>   7. Instruction that the plan should differ from previous plans and should receive a higher (or lower) score
>   8. Response format: an outline/sketch of the proposed solution in natural language, with no additional headings or text
>   9. Instruction that the plan should not modify the original solutions too much since execution error can occur
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 17

### 3.2 Input Contract

> **REQ-P3-003**: *A_ens_planner Input -- L Solutions* -- The ensemble planner agent shall receive all L solution scripts as part of its prompt. Each solution shall be presented in full (the complete `SolutionScript.content` for each of the L input solutions).
>
> - Precondition: `len(solutions) >= 2` (ensemble requires at least 2 solutions; the single-solution case is handled by the skip condition, REQ-P3-016).
> - Error: If `len(solutions) < 2`, the invocation function shall raise `ValueError("A_ens_planner requires at least 2 solutions for ensembling")`.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 3 input -- `s_final^1, ..., s_final^L`

> **REQ-P3-004**: *A_ens_planner Input -- History (First Invocation)* -- On the first invocation (r=0), the ensemble planner shall receive no history. The "Ensemble plans you have tried" section of the prompt shall either be omitted or state that no previous plans have been tried.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: At r=0, the rendered prompt shall not contain any "Plan:" / "Score:" history entries.
> - Source: REF-01 Algorithm 3 line 1 -- `e_0 = A_ens_planner({s_final^l}_{l=1}^L)` (no history argument)

> **REQ-P3-005**: *A_ens_planner Input -- History (Subsequent Invocations)* -- On subsequent invocations (r >= 1), the ensemble planner shall receive the complete history of all previous ensemble plans and their scores as `{(e_j, h(s_ens^j))}_{j=0}^{r-1}`.
>
> - The history shall be formatted as follows:
>   ```
>   # Ensemble plans you have tried
>
>   ## Plan: {plans[0]}
>   ## Score: {scores[0]}
>
>   ## Plan: {plans[1]}
>   ## Score: {scores[1]}
>   ...
>   ```
> - Scores that are `None` (from failed evaluations) shall be rendered as `"N/A (evaluation failed)"`.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: At r=2, the prompt shall include history entries for plans e_0 and e_1 with their respective scores.
> - Source: REF-01 Algorithm 3 line 5 -- `e_r = A_ens_planner({s_final^l}_{l=1}^L, {(e_j, h(s_ens^j))}_{j=0}^{r-1})`

### 3.3 Output Contract

> **REQ-P3-006**: *A_ens_planner Output Contract* -- The ensemble planner agent shall return a natural language ensemble plan string (e_r).
>
> - The plan shall be the complete text content of the agent's response, stripped of leading and trailing whitespace.
> - The plan shall be an outline/sketch describing the ensemble strategy (e.g., averaging probabilities, stacking with a meta-learner, weighted averaging).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given L=2 solution scripts, the returned plan shall be a non-empty string.
> - Source: REF-01 Figure 17 -- "Your response should be an outline/sketch of your proposed solution in natural language."

> **REQ-P3-007**: *A_ens_planner Novelty Constraint* -- The ensemble planner agent prompt (REQ-P3-002) shall instruct the agent that the suggested plan must differ from all previously tried plans. This is enforced via prompt instruction, not automated verification.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 17 -- "The suggested plan should be differ from the previous plans you have tried"

> **REQ-P3-008**: *A_ens_planner Focus Constraint* -- The ensemble planner agent prompt (REQ-P3-002) shall instruct the agent to concentrate on how to merge the solutions, not on other aspects like hyperparameters. This is enforced via prompt instruction, not automated verification.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 17 -- "You should concentrate how to merge, not the other parts like hyperparameters."

### 3.4 Invocation

> **REQ-P3-009**: *A_ens_planner Invocation Function* -- The system shall define an async function `invoke_ens_planner(solutions: list[SolutionScript], plans: list[str], scores: list[float | None]) -> str | None` that:
>
> 1. Validates that `len(solutions) >= 2` and `len(plans) == len(scores)`.
> 2. Retrieves the ens_planner prompt template from the `PromptRegistry` (REQ-DM-032).
> 3. Renders the template with `L=len(solutions)`, `solutions=[s.content for s in solutions]`, `plans=plans`, and `scores=scores`.
> 4. On the first invocation (when `plans` is empty), renders the template without the history section (REQ-P3-004).
> 5. On subsequent invocations (when `plans` is non-empty), renders the template with the full history section (REQ-P3-005).
> 6. Invokes the ens_planner agent via the SDK with the rendered prompt and no tools.
> 7. Returns the agent's response text stripped of whitespace, or `None` if the response is empty.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `invoke_ens_planner(solutions, [], [])` shall return a non-None, non-empty string (first invocation). `invoke_ens_planner(solutions, ["plan A"], [0.85])` shall return a non-None, non-empty string (subsequent invocation with history).

---

## 4. A_ensembler Requirements

### 4.1 Agent Definition

> **REQ-P3-010**: *A_ensembler Agent Definition* -- The system shall define an `AgentDefinition`-compatible configuration for the ensembler agent with the following properties:
>
> | Property | Value |
> |----------|-------|
> | `agent_type` | `AgentType.ensembler` |
> | `description` | Agent that implements an ensemble plan on L solution scripts into a single Python program |
> | `prompt` | Rendered from the ensembler template (Figure 18, REQ-DM-032) |
> | `tools` | `["Read"]` (to inspect data files in the `./input/` directory) |
> | `output_schema` | `None` (free-form response containing a single code block) |
> | `model` | `None` (inherit from orchestrator) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `AgentConfig(agent_type=AgentType.ensembler).to_agent_definition()` shall produce a valid dictionary for `ClaudeAgentOptions.agents`.
> - Source: REF-01 Section 3.3, Figure 18

> **REQ-P3-011**: *A_ensembler Prompt Template* -- The ensembler agent prompt shall be constructed by rendering the Figure 18 template from the `PromptRegistry` (REQ-DM-032) with the following variables:
>
> | Variable | Type | Description |
> |----------|------|-------------|
> | `L` | `int` | Number of solution scripts being ensembled |
> | `solutions` | `list[str]` | Full source code of each of the L solution scripts |
> | `plan` | `str` | The ensemble plan to implement (e_r) |
>
> - The rendered prompt shall include all instructions from Figure 18:
>   1. Kaggle grandmaster persona introduction
>   2. Goal: ensemble L Python Solutions for better performance based on the ensemble plan
>   3. All L solution scripts presented in full, each labeled as "# {n}th Python Solution"
>   4. The ensemble plan presented in full
>   5. Instruction to implement the ensemble plan with the provided solutions
>   6. Instruction that unless mentioned in the ensemble plan, do not modify the original Python Solutions too much
>   7. All provided data is already prepared and available in the `./input/` directory; there is no need to unzip any files
>   8. Instruction not to load previous submissions (do not load submissions)
>   9. Instruction that the code should implement the proposed solution and print the evaluation metric on a hold-out validation set
>   10. Response format: a single markdown code block (wrapped in ```) which is the ensemble of L Python Solutions, with no additional headings or text
>   11. Instruction: do not subsample or introduce dummy variables; must provide full new Python Solution using the L provided solutions
>   12. Instruction: do not forget the `./final/submission.csv` file
>   13. Instruction: print or return "Final Validation Performance: {final_validation_score}"
>   14. Instruction: the code should be a single-file Python program that is self-contained and can be executed as-is
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 18

### 4.2 Input Contract

> **REQ-P3-012**: *A_ensembler Input -- Plan and Solutions* -- The ensembler agent shall receive the ensemble plan (e_r) and all L solution scripts as part of its prompt.
>
> - Precondition: `plan` is a non-empty string and `len(solutions) >= 2`.
> - Error: If `plan` is empty, the invocation function shall raise `ValueError("A_ensembler requires a non-empty ensemble plan")`. If `len(solutions) < 2`, the invocation function shall raise `ValueError("A_ensembler requires at least 2 solutions for ensembling")`.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 3 lines 2, 6 -- `s_ens^r = A_ensembler(e_r, {s_final^l}_{l=1}^L)`

### 4.3 Output Contract

> **REQ-P3-013**: *A_ensembler Output Contract -- Single-File Ensemble Script* -- The ensembler agent shall return a `SolutionScript` containing the ensembled Python code extracted from the agent's response.
>
> - The response shall be parsed using `extract_code_block()` (REQ-SF-005) to extract a single code block.
> - The returned `SolutionScript` shall have:
>   - `content`: the extracted code block (the complete ensemble program)
>   - `phase`: `SolutionPhase.ensemble`
>   - `score`: `None` (not yet evaluated)
>   - `is_executable`: `True` (optimistic; will be verified by evaluation)
>   - `source_model`: `None` (ensemble solutions have no single source model)
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 3 lines 2, 6 -- `s_ens^r = A_ensembler(e_r, {s_final^l}_{l=1}^L)`

> **REQ-P3-014**: *A_ensembler No Subsampling Constraint* -- The ensembler agent prompt (REQ-P3-011) shall instruct the agent not to subsample or introduce dummy variables. The ensemble script must provide a full solution using the L provided solutions. This is enforced via prompt instruction, not automated verification.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 18 -- "Do not subsample or introduce dummy variables."

> **REQ-P3-015**: *A_ensembler Submission File Requirement* -- The ensembler agent prompt (REQ-P3-011) shall instruct the agent that the ensemble script must generate a `./final/submission.csv` file. Submission file existence shall be verified after evaluation using `verify_submission()` (REQ-EX-024).
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Acceptance: After successful evaluation of an ensemble script, `verify_submission(".")` shall return `True`.
> - Source: REF-01 Figure 18 -- "Do not forget the `./final/submission.csv` file."

### 4.4 Invocation

> **REQ-P3-016**: *A_ensembler Invocation Function* -- The system shall define an async function `invoke_ensembler(plan: str, solutions: list[SolutionScript]) -> SolutionScript | None` that:
>
> 1. Validates that `plan` is non-empty and `len(solutions) >= 2`.
> 2. Retrieves the ensembler prompt template from the `PromptRegistry` (REQ-DM-032).
> 3. Renders the template with `L=len(solutions)`, `solutions=[s.content for s in solutions]`, and `plan=plan`.
> 4. Invokes the ensembler agent via the SDK with the rendered prompt and tools `["Read"]`.
> 5. Extracts the code block from the response using `extract_code_block()` (REQ-SF-005).
> 6. If extraction fails, returns `None`.
> 7. Constructs and returns a `SolutionScript` (REQ-P3-013).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a valid plan and L=2 solutions, the function shall return a `SolutionScript` with `phase == SolutionPhase.ensemble` and non-empty `content`.
