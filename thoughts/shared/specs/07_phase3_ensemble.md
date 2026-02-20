# Software Requirements Specification: MLE-STAR Phase 3 -- Ensemble Optimization

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

---

## 5. Algorithm 3 Orchestration Requirements

### 5.1 Phase 3 Entry Point

> **REQ-P3-017**: *Phase 3 Entry Point* -- The system shall define an async function `run_phase3(solutions: list[SolutionScript], task: TaskDescription, config: PipelineConfig) -> Phase3Result` that implements Algorithm 3 from REF-01 Appendix B. This function orchestrates the full Phase 3 pipeline: ensemble planning, ensemble implementation, evaluation, and best selection across R rounds.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `run_phase3(solutions, task, config)` shall return a `Phase3Result` (REQ-DM-024) with a non-None `best_ensemble` and `best_ensemble_score`.
> - Source: REF-01 Algorithm 3

### 5.2 Skip Condition

> **REQ-P3-018**: *Single Solution Skip Condition* -- If `len(solutions) == 1` (i.e., L=1, only one parallel refinement path was configured or produced a result), the `run_phase3` function shall skip the ensemble process entirely and return a `Phase3Result` with:
>
> | Field | Value |
> |-------|-------|
> | `input_solutions` | `[solutions[0]]` |
> | `ensemble_plans` | `[]` (empty) |
> | `ensemble_scores` | `[]` (empty) |
> | `best_ensemble` | `solutions[0]` (the single solution, unchanged) |
> | `best_ensemble_score` | `solutions[0].score` |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `run_phase3([single_solution], task, config)` shall return immediately with the single solution as `best_ensemble`, without invoking A_ens_planner or A_ensembler.
> - Source: REF-01 Section 3.3 -- ensembling requires multiple solutions; REQ-DM-001 (L default=2 but L=1 is valid)

### 5.3 Initial Round (r=0)

> **REQ-P3-019**: *Algorithm 3 Step 1 -- Initial Ensemble Plan* -- The `run_phase3` function shall begin the ensemble loop by invoking `invoke_ens_planner(solutions, [], [])` (REQ-P3-009) with no history to obtain the initial ensemble plan e_0.
>
> - This corresponds to Algorithm 3 line 1: `e_0 = A_ens_planner({s_final^l}_{l=1}^L)`.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 3 line 1

> **REQ-P3-020**: *Algorithm 3 Step 2 -- Initial Ensemble Implementation* -- After obtaining e_0, the function shall invoke `invoke_ensembler(e_0, solutions)` (REQ-P3-016) to produce the initial ensemble solution s_ens^0.
>
> - This corresponds to Algorithm 3 line 2: `s_ens^0 = A_ensembler(e_0, {s_final^l}_{l=1}^L)`.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 3 line 2

> **REQ-P3-021**: *Algorithm 3 Step 3 -- Initial Ensemble Evaluation* -- After obtaining s_ens^0, the function shall:
>
> 1. Invoke `check_and_fix_leakage(s_ens^0)` (REQ-SF-020, REQ-SF-022) before evaluation.
> 2. Evaluate h(s_ens^0) using `evaluate_with_retry` (REQ-EX-021) with the debug callback (REQ-SF-007).
> 3. Record the score h(s_ens^0).
>
> - This corresponds to Algorithm 3 line 3: "Evaluate h(s_ens^0) using D".
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 3 line 3

### 5.4 Subsequent Rounds (r=1..R-1)

> **REQ-P3-022**: *Algorithm 3 Steps 4-8 -- Iteration Loop* -- For each ensemble round r from 1 to R-1 (where R = `config.ensemble_rounds`), the `run_phase3` function shall:
>
> 1. Invoke `invoke_ens_planner(solutions, accumulated_plans, accumulated_scores)` to obtain e_r, providing the full history of all previous plans and their scores.
> 2. If A_ens_planner returns `None` (empty response), skip this round: record `score=None` and a placeholder plan, and proceed to r+1.
> 3. Invoke `invoke_ensembler(e_r, solutions)` (REQ-P3-016) to produce s_ens^r.
> 4. If A_ensembler returns `None` (extraction failed), record `score=None` and proceed to r+1.
> 5. Invoke `check_and_fix_leakage(s_ens^r)` (REQ-SF-020, REQ-SF-022) before evaluation.
> 6. Evaluate h(s_ens^r) using `evaluate_with_retry` (REQ-EX-021) with the debug callback (REQ-SF-007).
> 7. Record the score h(s_ens^r).
> 8. Append e_r to `accumulated_plans` and h(s_ens^r) to `accumulated_scores`.
>
> - This implements Algorithm 3 lines 4-8.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given `config.ensemble_rounds = 5`, the loop shall attempt rounds r=1, r=2, r=3, r=4 (4 iterations after the initial round at r=0).
> - Source: REF-01 Algorithm 3 lines 4-8

> **REQ-P3-023**: *A_ens_planner Receives Full History* -- At ensemble round r, the A_ens_planner invocation shall receive the complete history of all previous attempts (j=0 through j=r-1), including:
>
> - All plan texts: `[e_0, e_1, ..., e_{r-1}]`
> - All scores: `[h(s_ens^0), h(s_ens^1), ..., h(s_ens^{r-1})]`
>
> - Scores for failed attempts (execution error after debugging, or ensembler failure) shall be represented as `None` in the scores list.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: At r=3, A_ens_planner shall receive plans and scores for rounds r=0, r=1, and r=2.
> - Source: REF-01 Algorithm 3 line 5 -- `e_r = A_ens_planner({s_final^l}_{l=1}^L, {(e_j, h(s_ens^j))}_{j=0}^{r-1})`

### 5.5 Score Tracking and Best Selection

> **REQ-P3-024**: *Score Tracking* -- The `run_phase3` function shall maintain a list of all ensemble scores `[h(s_ens^0), h(s_ens^1), ..., h(s_ens^{R-1})]` and a parallel list of all ensemble solutions `[s_ens^0, s_ens^1, ..., s_ens^{R-1}]`, recording `None` for scores of failed attempts.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After R=5 rounds, the score list shall have exactly 5 entries.
> - Source: REF-01 Algorithm 3 lines 3, 7 -- evaluation at each round

> **REQ-P3-025**: *Best Ensemble Selection* -- After all R rounds complete, the function shall select the best ensemble solution s_ens* as the solution with the best score across all R attempts:
>
> - For `metric_direction == "maximize"`: `r* = argmax_{r in {0,...,R-1}} h(s_ens^r)` (considering only non-None scores).
> - For `metric_direction == "minimize"`: `r* = argmin_{r in {0,...,R-1}} h(s_ens^r)` (considering only non-None scores).
> - If multiple rounds share the best score, the last such round shall be selected (consistent with `>=` semantics in the rest of the pipeline).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given scores `[0.85, 0.88, None, 0.87, 0.88]` with maximize direction, r*=4 (last occurrence of 0.88) and s_ens* = s_ens^4.
> - Source: REF-01 Algorithm 3 line 9 -- `r* = arg max_{r} h(s_ens^r)`

> **REQ-P3-026**: *All Rounds Failed* -- If all R ensemble attempts fail (all scores are `None`), the `run_phase3` function shall fall back to the single best input solution. Specifically:
>
> 1. Select the input solution with the best score from the L input solutions.
> 2. Log a warning: "Phase 3 ensemble: all {R} attempts failed; falling back to best input solution".
> 3. Return a Phase3Result with `best_ensemble` set to the best input solution and `best_ensemble_score` set to that solution's score.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given R=5 where all ensemble scripts fail, the returned `best_ensemble` shall be the input solution with the highest (or lowest, per direction) score.

### 5.6 Safety Integration

> **REQ-P3-027**: *Leakage Check Before Ensemble Evaluation* -- At each ensemble round r, after A_ensembler produces s_ens^r and before evaluation, the system shall invoke `check_and_fix_leakage(s_ens^r)` (REQ-SF-020, REQ-SF-022). The leakage-checked (and potentially corrected) solution shall be the one submitted for evaluation.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Acceptance: Every code path from `invoke_ensembler` to `evaluate_solution` shall include a `check_and_fix_leakage` call.
> - Source: REF-01 Section 3.4 -- "Every generated solution before evaluation (all phases)"; REQ-SF-022

> **REQ-P3-028**: *Debug on Ensemble Execution Error* -- When evaluating s_ens^r produces an execution error, the system shall use the debug retry mechanism by invoking `evaluate_with_retry` (REQ-EX-021) with the debug callback from `make_debug_callback(task, config)` (REQ-SF-007).
>
> - The debug retry shall attempt up to `config.max_debug_attempts` fixes before declaring the attempt failed.
> - If debug succeeds, the fixed solution and its score shall be used for this round's result.
> - If debug exhausts all retries and the solution still errors, the round shall be recorded with `score=None`.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given an ensemble script that errors once and is fixed by A_debugger, the round shall record the score from the fixed solution.
> - Source: REF-01 Section 3.4 -- A_debugger applies to all generated code; REQ-SF-006, REQ-SF-007

### 5.7 Error Handling

> **REQ-P3-029**: *Ensemble Script Failure Handling* -- If an ensemble script s_ens^r fails to execute after all debug retries are exhausted, the system shall:
>
> 1. Log a warning indicating that ensemble round r failed: plan summary (first 200 characters), error summary (first line of traceback).
> 2. Record `score=None` for this round in both the score tracking list and the `EnsembleAttempt` record.
> 3. Continue to the next round (r+1). Do not abort Phase 3.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given R=5 where rounds 0 and 2 fail, the function shall complete all 5 rounds and select the best from the 3 successful rounds.

> **REQ-P3-030**: *Ensembler Extraction Failure Handling* -- If `invoke_ensembler` returns `None` (the agent's response did not contain an extractable code block), the system shall:
>
> 1. Log a warning indicating the ensembler response could not be parsed at round r.
> 2. Record an `EnsembleAttempt` with `score=None` and a `SolutionScript` with empty `content`.
> 3. Continue to the next round (r+1).
>
> - Priority: Must | Verify: Test | Release: MVP

> **REQ-P3-031**: *Ens_planner Failure Handling* -- If `invoke_ens_planner` returns `None` (empty response) at round r, the system shall:
>
> 1. Log a warning indicating the ens_planner returned an empty response at round r.
> 2. Record an `EnsembleAttempt` with `plan="[ens_planner failed]"`, `score=None`, and a `SolutionScript` with empty `content`.
> 3. Append the placeholder plan and `None` score to the accumulated history so that subsequent rounds have accurate context.
> 4. Continue to the next round (r+1).
>
> - Priority: Must | Verify: Test | Release: MVP

### 5.8 EnsembleAttempt History

> **REQ-P3-032**: *EnsembleAttempt Record Construction* -- At each ensemble round r, the system shall construct an `EnsembleAttempt` (REQ-DM-043) with the following field values:
>
> | Field | Value |
> |-------|-------|
> | `plan` | The ensemble plan text e_r used for this round |
> | `score` | h(s_ens^r) if evaluation succeeded, `None` if evaluation failed |
> | `solution` | The ensemble solution s_ens^r (or an empty-content SolutionScript if ensembler failed) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After R=5 rounds, the history list shall contain exactly 5 `EnsembleAttempt` records.
> - Source: REF-01 Algorithm 3 -- tracks `{(e_j, h(s_ens^j))}` history; REQ-DM-043

> **REQ-P3-033**: *EnsembleAttempt History Ordering* -- The list of `EnsembleAttempt` records shall be ordered by round index (r=0 first, r=1 second, etc.). The list shall always have length R (= `config.ensemble_rounds`), including records for failed rounds.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `attempts[0].plan` shall correspond to e_0 (the initial ensemble plan).

### 5.9 Phase3Result Construction

> **REQ-P3-034**: *Phase3Result Construction* -- Upon completion of all R ensemble rounds and best selection, the `run_phase3` function shall construct a `Phase3Result` (REQ-DM-024) with:
>
> | Field | Value |
> |-------|-------|
> | `input_solutions` | The list of L `SolutionScript` instances received as input |
> | `ensemble_plans` | The list of ensemble plan texts `[e_0, e_1, ..., e_{R-1}]` |
> | `ensemble_scores` | The list of scores `[h(s_ens^0), h(s_ens^1), ..., h(s_ens^{R-1})]` (including `None` for failed rounds) |
> | `best_ensemble` | The best ensemble solution s_ens* (REQ-P3-025) |
> | `best_ensemble_score` | The score of s_ens* |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `len(Phase3Result.ensemble_plans) == len(Phase3Result.ensemble_scores) == config.ensemble_rounds`.
> - Source: REQ-DM-024

> **REQ-P3-035**: *Phase3Result Score Consistency* -- The `Phase3Result.best_ensemble_score` shall equal the score of `Phase3Result.best_ensemble`. The `best_ensemble` shall be the `SolutionScript` instance corresponding to the best-scoring ensemble round (REQ-P3-025) or the best input solution if all rounds failed (REQ-P3-026).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `Phase3Result.best_ensemble_score` shall equal the score recorded for the round selected by the best-selection logic.

---

## 6. Non-Functional Requirements

### 6.1 Performance

> **REQ-P3-036**: *Phase 3 Overhead Budget* -- The Phase 3 orchestration overhead (prompt rendering, output parsing, score comparison, Phase3Result construction) excluding agent LLM calls and script execution time shall not exceed 5 seconds total across all R rounds.
>
> - Priority: Should | Verify: Test | Release: MVP

> **REQ-P3-037**: *Phase 3 Total Duration* -- A single Phase 3 execution (R=5 rounds) shall complete within 60 minutes under normal conditions, excluding script execution time. The dominant cost is 2R LLM calls (R ens_planner calls + R ensembler calls, plus leakage and debug calls).
>
> - Priority: Should | Verify: Demonstration | Release: MVP
> - Source: REF-01 Section 4 -- 24-hour total budget; Phase 3 is one of 3 phases

### 6.2 Reliability

> **REQ-P3-038**: *Phase 3 Never Raises on Round Failure* -- The `run_phase3` function shall not raise exceptions due to individual round failures (ens_planner empty, ensembler unparseable, evaluation error). Each such failure shall be handled gracefully per REQ-P3-029 through REQ-P3-031, and the function shall always return a valid `Phase3Result`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given R=5 where all 5 rounds fail, the function shall return a Phase3Result with the best input solution as fallback (REQ-P3-026).

### 6.3 Observability

> **REQ-P3-039**: *Phase 3 Logging* -- The Phase 3 orchestration shall log the following events using Python's `logging` module at the specified levels:
>
> | Event | Level | Content |
> |-------|-------|---------|
> | Phase 3 start | `INFO` | L value, R value, competition ID |
> | Phase 3 skipped (L=1) | `INFO` | Single solution score, competition ID |
> | Ensemble round start | `INFO` | Round index r, number of previous plans in history |
> | A_ens_planner invocation start | `INFO` | Round r, history size |
> | A_ens_planner invocation complete | `INFO` | Round r, plan text (first 200 chars) |
> | A_ens_planner empty response | `WARNING` | Round r |
> | A_ensembler invocation start | `INFO` | Round r, plan text (first 200 chars) |
> | A_ensembler invocation complete | `INFO` | Round r, script length (or "failed to parse") |
> | A_ensembler extraction failure | `WARNING` | Round r, response summary (first 200 chars) |
> | Leakage check start | `INFO` | Round r, solution content length |
> | Leakage check complete | `INFO` | Round r, leakage found (yes/no), content changed (yes/no) |
> | Evaluation start | `INFO` | Round r, solution content length |
> | Evaluation complete | `INFO` | Round r, score (or "failed"), is_error, duration |
> | Round failed (execution error) | `WARNING` | Round r, error summary, plan summary |
> | Best selection | `INFO` | Best round r*, best score, total successful rounds |
> | All rounds failed (fallback) | `WARNING` | R value, fallback solution score |
> | Phase 3 complete | `INFO` | Best score, best round, total duration, rounds attempted |
>
> - Priority: Must | Verify: Inspection | Release: MVP

---

## 7. Constraints

### 7.1 Technology Constraints

> **REQ-P3-040**: *SDK Agent Invocation* -- Both Phase 3 agents (A_ens_planner and A_ensembler) shall be invoked via the Claude Agent SDK agent mechanism. They shall not use direct API calls, raw HTTP requests, or any non-SDK LLM invocation method.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-02 -- all agent interactions through the SDK

> **REQ-P3-041**: *Single Module Organization* -- All Phase 3 functions defined in this spec shall reside in a single Python module (e.g., `mle_star/phase3.py`).
>
> - Priority: Should | Verify: Inspection | Release: MVP

### 7.2 Algorithm Fidelity Constraints

> **REQ-P3-042**: *Algorithm 3 Fidelity* -- The Phase 3 implementation shall faithfully reproduce Algorithm 3 from REF-01 Appendix B. Specifically:
>
> 1. The initial ensemble round (r=0) shall invoke A_ens_planner with no history (line 1).
> 2. Each round shall invoke A_ensembler with the plan and all L solutions (lines 2, 6).
> 3. Each ensemble solution shall be evaluated (lines 3, 7).
> 4. Subsequent rounds (r=1..R-1) shall pass full history to A_ens_planner (line 5).
> 5. The best ensemble shall be selected as `argmax` (or `argmin` per direction) over all R rounds (line 9).
> 6. The output shall be the single best ensemble solution s_ens* (line 10).
>
> - Deviations from Algorithm 3 are permitted only for error handling (e.g., skipping failed rounds, debugging), which the paper does not address explicitly.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Appendix B, Algorithm 3

> **REQ-P3-043**: *Sequential Ensemble Rounds* -- The ensemble rounds shall execute sequentially, not concurrently. Each round depends on the accumulated history from all prior rounds (plans and scores).
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Rationale: A_ens_planner at round r requires the scores from rounds 0 through r-1; parallel execution would prevent this.
> - Source: REF-01 Algorithm 3 lines 4-8 -- sequential loop with history accumulation

> **REQ-P3-044**: *Ensemble Iteration Count* -- The ensemble loop shall attempt exactly `config.ensemble_rounds` (R) rounds (REQ-DM-001, default: 5). Failed rounds count as iterations; the loop does not add extra rounds to compensate for failures.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given `config.ensemble_rounds = 5`, the function shall produce exactly 5 `EnsembleAttempt` records regardless of success or failure.
> - Source: REF-01 Algorithm 3 -- `for r = 1 to R-1 do` (R-1 iterations after the initial round, R total)

> **REQ-P3-045**: *Leakage Check Integration Points* -- Within Phase 3, the leakage checker `check_and_fix_leakage()` (REQ-SF-022) shall be invoked on every ensemble solution s_ens^r before evaluation. This ensures every solution that enters evaluation has been checked for data leakage.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Section 3.4, REQ-SF-022

### 7.3 Prompt Fidelity Constraints

> **REQ-P3-046**: *A_ens_planner Prompt Fidelity* -- The A_ens_planner prompt (REQ-P3-002) shall preserve the semantic intent of Figure 17 from the paper. The prompt shall include:
>
> 1. The Kaggle grandmaster persona introduction.
> 2. All L solution scripts presented in full.
> 3. The history of previous plans and scores (when available).
> 4. The instruction to concentrate on merging strategy, not hyperparameters.
> 5. The instruction that the plan must be easy to implement, novel, and effective.
> 6. The instruction that the plan should differ from previous plans.
> 7. The instruction that the plan should not modify original solutions too much.
> 8. The response format instruction (natural language outline, no additional headings).
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 17

> **REQ-P3-047**: *A_ensembler Prompt Fidelity* -- The A_ensembler prompt (REQ-P3-011) shall preserve the semantic intent of Figure 18 from the paper. The prompt shall include:
>
> 1. The Kaggle grandmaster persona introduction.
> 2. All L solution scripts presented in full.
> 3. The ensemble plan presented in full.
> 4. The instruction to implement the ensemble plan.
> 5. The instruction not to modify original solutions too much (unless the plan says to).
> 6. The `./input/` data directory instruction and no-unzip instruction.
> 7. The no-load-submissions instruction.
> 8. The no-subsample and no-dummy-variables instruction.
> 9. The `./final/submission.csv` file requirement.
> 10. The "Final Validation Performance" output requirement.
> 11. The single code block response format.
> 12. The self-contained single-file constraint.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 18

### 7.4 Ensemble Script Constraints

> **REQ-P3-048**: *Ensemble Scripts Are Full Programs* -- Ensemble scripts produced by A_ensembler are complete, self-contained, single-file Python programs (not code block replacements). They include all imports, data loading, model training, prediction, validation metric computation, and submission file generation. This distinguishes them from Phase 2 refinement outputs (which are code block replacements within an existing solution).
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Rationale: Ensemble scripts combine L solutions into a new program, rather than modifying one solution's code block.
> - Source: REF-01 Figure 18 -- "The code should be a single-file Python program that is self-contained and can be executed as-is."

> **REQ-P3-049**: *Validation Performance Output Requirement* -- Every ensemble script produced by A_ensembler shall print the evaluation metric in the format `"Final Validation Performance: {score}"` so that the execution harness (REQ-DM-027) can parse the score from stdout.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Acceptance: The prompt shall include the instruction to print "Final Validation Performance: {final_validation_score}".
> - Source: REF-01 Figure 18; REQ-DM-027 (score parsing regex)

---

## 8. Traceability Matrix

### 8.1 Requirements to Paper Sections

| Req ID | Paper Section | Paper Element | SDK Construct |
|--------|--------------|---------------|---------------|
| REQ-P3-001 | Section 3.3 | A_ens_planner agent | `AgentDefinition` |
| REQ-P3-002 | Figure 17 | Ens_planner prompt template | `prompt` parameter |
| REQ-P3-003 | Algorithm 3 input | L solutions input | -- |
| REQ-P3-004 | Algorithm 3 line 1 | First invocation (no history) | -- |
| REQ-P3-005 | Algorithm 3 line 5 | Subsequent invocations (with history) | -- |
| REQ-P3-006 | Figure 17 | Ensemble plan output (natural language) | -- |
| REQ-P3-007 | Figure 17 | "differ from the previous plans" | -- |
| REQ-P3-008 | Figure 17 | "concentrate how to merge" | -- |
| REQ-P3-009 | Algorithm 3 lines 1, 5 | `A_ens_planner(...)` invocation | SDK agent call |
| REQ-P3-010 | Section 3.3 | A_ensembler agent | `AgentDefinition` |
| REQ-P3-011 | Figure 18 | Ensembler prompt template | `prompt` parameter |
| REQ-P3-012 | Algorithm 3 lines 2, 6 | `A_ensembler(e_r, solutions)` input | -- |
| REQ-P3-013 | Algorithm 3 lines 2, 6 | `s_ens^r = A_ensembler(...)` output | -- |
| REQ-P3-014 | Figure 18 | "Do not subsample" | -- |
| REQ-P3-015 | Figure 18 | `./final/submission.csv` | REQ-EX-024 |
| REQ-P3-016 | Algorithm 3 lines 2, 6 | Ensembler invocation function | SDK agent call |
| REQ-P3-017 | Algorithm 3 | Full Phase 3 entry point | -- |
| REQ-P3-018 | Section 3.3 | L=1 skip condition | -- |
| REQ-P3-019 | Algorithm 3 line 1 | Initial ensemble plan (r=0) | -- |
| REQ-P3-020 | Algorithm 3 line 2 | Initial ensemble implementation (r=0) | -- |
| REQ-P3-021 | Algorithm 3 line 3 | Initial ensemble evaluation (r=0) | -- |
| REQ-P3-022 | Algorithm 3 lines 4-8 | Subsequent rounds iteration loop | -- |
| REQ-P3-023 | Algorithm 3 line 5 | Full history to A_ens_planner | -- |
| REQ-P3-024 | Algorithm 3 lines 3, 7 | Score tracking | -- |
| REQ-P3-025 | Algorithm 3 line 9 | `r* = argmax h(s_ens^r)` best selection | -- |
| REQ-P3-026 | -- | All rounds failed fallback | -- |
| REQ-P3-027 | Section 3.4 | Leakage check before evaluation | REQ-SF-022 |
| REQ-P3-028 | Section 3.4 | A_debugger on execution error | REQ-SF-006, REQ-SF-007 |
| REQ-P3-029 | -- | Ensemble script failure handling | -- |
| REQ-P3-030 | -- | Ensembler extraction failure handling | -- |
| REQ-P3-031 | -- | Ens_planner failure handling | -- |
| REQ-P3-032 | Algorithm 3 | EnsembleAttempt record | REQ-DM-043 |
| REQ-P3-033 | Algorithm 3 | History ordering | -- |
| REQ-P3-034 | Algorithm 3 | Phase3Result construction | REQ-DM-024 |
| REQ-P3-035 | Algorithm 3 | Score consistency | -- |
| REQ-P3-036 | -- | Orchestration overhead | -- |
| REQ-P3-037 | Section 4 | Phase 3 duration budget | -- |
| REQ-P3-038 | -- | Never raises on round failure | -- |
| REQ-P3-039 | -- | Logging | Python `logging` |
| REQ-P3-040 | -- | SDK-only invocation | `claude-agent-sdk` |
| REQ-P3-041 | -- | Module organization | -- |
| REQ-P3-042 | Appendix B | Algorithm 3 fidelity | -- |
| REQ-P3-043 | Algorithm 3 lines 4-8 | Sequential rounds | -- |
| REQ-P3-044 | Algorithm 3 | Iteration count = R | -- |
| REQ-P3-045 | Section 3.4 | Leakage integration points | REQ-SF-022 |
| REQ-P3-046 | Figure 17 | Ens_planner prompt fidelity | -- |
| REQ-P3-047 | Figure 18 | Ensembler prompt fidelity | -- |
| REQ-P3-048 | Figure 18 | Ensemble scripts are full programs | -- |
| REQ-P3-049 | Figure 18, Figures 10-19 | "Final Validation Performance" output | REQ-DM-027 |

### 8.2 Cross-References to Other Specs

| Req ID | Referenced By |
|--------|--------------|
| REQ-P3-017 (run_phase3) | Spec 09 (Orchestrator invokes Phase 3) |
| REQ-P3-034 (Phase3Result) | Spec 08 (Finalization takes Phase3Result.best_ensemble as input) |
| REQ-P3-034 (Phase3Result) | Spec 09 (Orchestrator stores Phase3Result in FinalResult) |

### 8.3 Spec 01 Dependencies (Inbound)

| Spec 01 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-DM-001 (PipelineConfig) | REQ-P3-002, REQ-P3-017, REQ-P3-022, REQ-P3-044 | L and R parameters, max_debug_attempts |
| REQ-DM-006 (MetricDirection) | REQ-P3-025 | Score comparison direction for best selection |
| REQ-DM-007 (TaskDescription) | REQ-P3-017, REQ-P3-021, REQ-P3-022, REQ-P3-028 | Task context for evaluation and metric direction |
| REQ-DM-009 (SolutionScript) | REQ-P3-003, REQ-P3-013, REQ-P3-016, REQ-P3-017, REQ-P3-018 | Solution type for input/output |
| REQ-DM-013 (AgentType) | REQ-P3-001, REQ-P3-010 | Agent identity enum values (ens_planner, ensembler) |
| REQ-DM-024 (Phase3Result) | REQ-P3-017, REQ-P3-018, REQ-P3-034, REQ-P3-035 | Return type of run_phase3 |
| REQ-DM-027 (Score Parsing) | REQ-P3-049 | "Final Validation Performance" regex pattern |
| REQ-DM-032 (PromptRegistry) | REQ-P3-002, REQ-P3-009, REQ-P3-011, REQ-P3-016 | Template retrieval for both agents |
| REQ-DM-036 (AgentConfig) | REQ-P3-001, REQ-P3-010 | Agent-to-SDK mapping |
| REQ-DM-043 (EnsembleAttempt) | REQ-P3-032, REQ-P3-033 | Ensemble history records |

### 8.4 Spec 02 Dependencies (Inbound)

| Spec 02 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-EX-015 (evaluate_solution) | REQ-P3-021, REQ-P3-022 | Evaluate ensemble solutions |
| REQ-EX-021 (evaluate_with_retry) | REQ-P3-021, REQ-P3-022, REQ-P3-028 | Evaluate with debug retry support |
| REQ-EX-024 (verify_submission) | REQ-P3-015 | Verify submission.csv after evaluation |

### 8.5 Spec 03 Dependencies (Inbound)

| Spec 03 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-SF-005 (extract_code_block) | REQ-P3-013, REQ-P3-016 | Extract code block from A_ensembler response |
| REQ-SF-006 (debug_solution) | REQ-P3-028 | Debug execution errors in ensemble scripts |
| REQ-SF-007 (make_debug_callback) | REQ-P3-021, REQ-P3-022, REQ-P3-028 | Debug callback for evaluate_with_retry |
| REQ-SF-020 (check_and_fix_leakage) | REQ-P3-021, REQ-P3-022, REQ-P3-027 | Leakage detection and correction |
| REQ-SF-022 (leakage integration point) | REQ-P3-027, REQ-P3-045 | Leakage check before every evaluation |

---

## 9. Change Control

### 9.1 Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2026-02-20 | MLE-STAR Team | Initial draft -- all 49 requirements |

### 9.2 Baselining Policy

This SRS is baselined at version 1.0. After baselining, changes require impact analysis against Spec 08 (finalization depends on Phase3Result.best_ensemble), Spec 09 (orchestrator invokes Phase 3), Spec 01 (upstream data model dependencies), Spec 02 (upstream execution harness dependencies), and Spec 03 (upstream safety agent dependencies).
