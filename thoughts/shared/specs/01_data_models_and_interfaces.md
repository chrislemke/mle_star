# Software Requirements Specification: MLE-STAR Core Data Models and Interfaces

| Field | Value |
|-------|-------|
| Version | 0.1.0 |
| Date | 2026-02-20 |
| Status | Draft |
| Spec ID | 01 of 09 |
| Requirement Prefix | REQ-DM- |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Product Perspective](#2-product-perspective)
3. [Data Model Requirements](#3-data-model-requirements)
4. [Interface Requirements](#4-interface-requirements)
5. [Functional Requirements](#5-functional-requirements)
6. [Non-Functional Requirements](#6-non-functional-requirements)
7. [Constraints](#7-constraints)
8. [Traceability Matrix](#8-traceability-matrix)
9. [Change Control](#9-change-control)

---

## 1. Introduction

### 1.1 Purpose

This SRS defines every shared data model, configuration type, interface, and Pydantic schema used across the MLE-STAR multi-agent pipeline. It is the foundational spec referenced by all other specs (02-09).

Intended audience: developers implementing the MLE-STAR system using the Claude Agent SDK for Python.

### 1.2 Scope

**Product name**: MLE-STAR (Machine Learning Engineering agent via Search and Targeted Refinement)

**What this spec covers**:
- Pipeline configuration model (hyperparameters M, T, K, L, R, time_limit, subsample_limit)
- Task description model
- Solution script wrapper model
- Code block model
- All structured output schemas (retriever, extractor, leakage detection)
- Score function interface
- Agent identity enumeration
- Evaluation result model
- Phase result models
- Prompt template registry interface
- SDK integration type aliases

**Out of scope**:
- Agent behavior logic (covered by specs 03-08)
- Orchestration control flow (covered by spec 09)
- Execution harness internals (covered by spec 02)

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|------|-----------|
| SRS | Software Requirements Specification |
| MLE-STAR | ML Engineering agent with web Search and TArgeted code block Refinement |
| T_task | Task description — string describing the Kaggle competition |
| D | Dataset — collection of files in `./input/` directory |
| s | Solution script — single-file self-contained Python program |
| c | Code block — exact substring of a solution script |
| p | Plan — natural language refinement plan (3-5 sentences) |
| T_abl | Ablation summary — structured summary of ablation study results |
| e | Ensemble plan — natural language ensemble strategy description |
| h(s) | Score function — maps a solution script to a real-valued performance score |
| M | Number of candidate models retrieved via web search (default: 4) |
| T | Outer loop iterations for code block targeting (default: 4) |
| K | Inner loop iterations for refinement strategies (default: 4) |
| L | Number of parallel solutions for ensembling (default: 2) |
| R | Ensemble strategy exploration rounds (default: 5) |
| SDK | Claude Agent SDK for Python (`claude-agent-sdk`) |

### 1.4 References

| ID | Title | Version | Source |
|----|-------|---------|--------|
| REF-01 | MLE-STAR paper | v3 | arXiv:2506.15692v3 |
| REF-02 | Claude Agent SDK reference | v0.1.39 | `claude-agent-sdk` PyPI |
| REF-03 | MLE-STAR architecture notes | — | `thoughts/notes/mle_star_architecture.md` |
| REF-04 | MLE-STAR paper extraction | — | `thoughts/notes/mle_star_paper.md` |
| REF-05 | Claude Agent SDK examples | — | `thoughts/notes/claude_agent_sdk_examples.md` |
| REF-06 | Claude Agent SDK API reference | — | `thoughts/notes/claude_agent_sdk_reference.md` |

### 1.5 Document Overview

- Section 3: Data model requirements (Pydantic schemas, enums, configuration)
- Section 4: Interface requirements (score function, prompt templates, SDK types)
- Section 5: Functional requirements (validation, serialization, construction)
- Section 6: Non-functional requirements (performance, portability)
- Section 7: Constraints (technology, compatibility)
- Section 8: Traceability matrix

---

## 2. Product Perspective

### 2.1 System Context

This spec defines the shared type system for the MLE-STAR pipeline. All 9 specs share these data models:

```
Spec 01 (this) ──> Spec 02 (Execution Harness)
                ──> Spec 03 (Safety Modules)
                ──> Spec 04 (Phase 1)
                ──> Spec 05 (Phase 2 Outer)
                ──> Spec 06 (Phase 2 Inner)
                ──> Spec 07 (Phase 3 Ensemble)
                ──> Spec 08 (Submission)
                ──> Spec 09 (Orchestrator)
```

### 2.2 Product Functions Summary

1. Define all shared Pydantic models for pipeline data flow
2. Define configuration model with paper-specified defaults
3. Define structured output schemas for agents requiring JSON output
4. Define score function interface for solution evaluation
5. Define prompt template registry for agent prompt formatting

### 2.3 Operating Environment

- **Runtime**: Python 3.10+
- **Validation library**: Pydantic v2
- **SDK**: `claude-agent-sdk` v0.1.39+
- **Serialization**: JSON (for SDK `output_format` integration)

### 2.4 Assumptions and Dependencies

| ID | Assumption | Impact if Invalid |
|----|-----------|-------------------|
| A-01 | Pydantic v2 is used for all model definitions | Schema generation for `output_format` would need alternative |
| A-02 | All solution scripts are single-file Python programs | Multi-file solutions would require new model |
| A-03 | Score is always parseable as a float from stdout | Non-numeric metrics would require extended parsing |

| ID | Dependency | Owner | Risk if Unavailable |
|----|-----------|-------|---------------------|
| D-01 | `claude-agent-sdk` v0.1.39+ | Anthropic | Agent execution unavailable |
| D-02 | Pydantic v2 | Pydantic team | Schema validation unavailable |

---

## 3. Data Model Requirements

### 3.1 Pipeline Configuration

> **REQ-DM-001**: *PipelineConfig Model* — The system shall define a Pydantic model `PipelineConfig` with the following fields and defaults:
>
> | Field | Type | Default | Description |
> |-------|------|---------|-------------|
> | `num_retrieved_models` (M) | `int` | `4` | Number of candidate models to retrieve |
> | `outer_loop_steps` (T) | `int` | `4` | Outer loop iterations |
> | `inner_loop_steps` (K) | `int` | `4` | Inner loop iterations per code block |
> | `num_parallel_solutions` (L) | `int` | `2` | Parallel solution paths for ensemble |
> | `ensemble_rounds` (R) | `int` | `5` | Ensemble strategy exploration rounds |
> | `time_limit_seconds` | `int` | `86400` | Maximum runtime per competition (24h) |
> | `subsample_limit` | `int` | `30000` | Max training samples during refinement |
> | `max_debug_attempts` | `int` | `3` | Max debugging retries before fallback |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Instantiating `PipelineConfig()` with no arguments shall produce an object with all paper-specified defaults.
> - Source: REF-01 Section 4 (Experimental Setup), REF-03 Section 3

> **REQ-DM-002**: *PipelineConfig Validation* — The `PipelineConfig` model shall validate that all integer fields are strictly positive (>= 1).
>
> - Error: `ValidationError` with field name and constraint description.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `PipelineConfig(num_retrieved_models=0)` shall raise `ValidationError`.

> **REQ-DM-003**: *PipelineConfig Serialization* — The `PipelineConfig` model shall support JSON serialization via `model_dump_json()` and deserialization via `model_validate_json()`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Round-trip serialization shall preserve all field values.

### 3.2 Task Description

> **REQ-DM-004**: *TaskType Enum* — The system shall define a `TaskType` string enum with the following values: `"classification"`, `"regression"`, `"image_classification"`, `"image_to_image"`, `"text_classification"`, `"audio_classification"`, `"sequence_to_sequence"`, `"tabular"`.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Table 7 (competition categories)

> **REQ-DM-005**: *DataModality Enum* — The system shall define a `DataModality` string enum with the following values: `"tabular"`, `"image"`, `"text"`, `"audio"`, `"mixed"`.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Section 3 (Problem Setup)

> **REQ-DM-006**: *MetricDirection Enum* — The system shall define a `MetricDirection` string enum with two values: `"maximize"` (higher is better) and `"minimize"` (lower is better).
>
> - Priority: Must | Verify: Inspection | Release: MVP

> **REQ-DM-007**: *TaskDescription Model* — The system shall define a Pydantic model `TaskDescription` with the following fields:
>
> | Field | Type | Required | Description |
> |-------|------|----------|-------------|
> | `competition_id` | `str` | Yes | Unique competition identifier (e.g., `"spaceship-titanic"`) |
> | `task_type` | `TaskType` | Yes | Type of ML task |
> | `data_modality` | `DataModality` | Yes | Primary data modality |
> | `evaluation_metric` | `str` | Yes | Name of the evaluation metric (e.g., `"accuracy"`, `"RMSE"`) |
> | `metric_direction` | `MetricDirection` | Yes | Whether to maximize or minimize the metric |
> | `description` | `str` | Yes | Full task description text (T_task) |
> | `data_dir` | `str` | `"./input"` | Path to dataset directory |
> | `output_dir` | `str` | `"./final"` | Path for submission output |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: All required fields must be present; missing fields shall raise `ValidationError`.
> - Source: REF-01 Section 3 (Problem Setup), Figure 10 prompt

### 3.3 Solution Script

> **REQ-DM-008**: *SolutionPhase Enum* — The system shall define a `SolutionPhase` string enum with values: `"init"`, `"merged"`, `"refined"`, `"ensemble"`, `"final"`.
>
> - Priority: Must | Verify: Inspection | Release: MVP

> **REQ-DM-009**: *SolutionScript Model* — The system shall define a Pydantic model `SolutionScript` with the following fields:
>
> | Field | Type | Required | Description |
> |-------|------|----------|-------------|
> | `content` | `str` | Yes | Full Python source code |
> | `phase` | `SolutionPhase` | Yes | Which pipeline phase produced this script |
> | `score` | `float \| None` | No | Evaluation score (None if not yet evaluated) |
> | `is_executable` | `bool` | `True` | Whether the script ran without errors |
> | `source_model` | `str \| None` | No | Name of the model used (if from Phase 1) |
> | `created_at` | `datetime` | auto | Timestamp of creation |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Equations 2, 3, 7, 9

> **REQ-DM-010**: *SolutionScript.replace_block()* — The `SolutionScript` model shall provide a method `replace_block(old: str, new: str) -> SolutionScript` that returns a new `SolutionScript` with the first occurrence of `old` in `content` replaced by `new`.
>
> - Error: Shall raise `ValueError` if `old` is not found in `content`.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `s.replace_block(c, c_new).content == s.content.replace(c, c_new, 1)` for the first occurrence.
> - Source: REF-01 Equation 7 — `s_t^k = s_t.replace(c_t, c_t^k)`

### 3.4 Code Block

> **REQ-DM-011**: *CodeBlockCategory Enum* — The system shall define a `CodeBlockCategory` string enum with values: `"preprocessing"`, `"feature_engineering"`, `"model_selection"`, `"training"`, `"hyperparameter_tuning"`, `"ensemble"`, `"postprocessing"`, `"other"`.
>
> - Priority: Should | Verify: Inspection | Release: MVP

> **REQ-DM-012**: *CodeBlock Model* — The system shall define a Pydantic model `CodeBlock` with the following fields:
>
> | Field | Type | Required | Description |
> |-------|------|----------|-------------|
> | `content` | `str` | Yes | Exact code block text extracted from solution script |
> | `category` | `CodeBlockCategory \| None` | No | Semantic category of the code block |
> | `outer_step` | `int \| None` | No | Outer loop step t at which this block was extracted |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Equation 6 — `c_t, p_0 = A_extractor(...)`

### 3.5 Agent Identity

> **REQ-DM-013**: *AgentType Enum* — The system shall define an `AgentType` string enum with exactly these 14 values: `"retriever"`, `"init"`, `"merger"`, `"ablation"`, `"summarize"`, `"extractor"`, `"coder"`, `"planner"`, `"ens_planner"`, `"ensembler"`, `"debugger"`, `"leakage"`, `"data"`, `"test"`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `len(AgentType)` shall equal 14; each value shall correspond to one agent from the paper (A_retriever through A_test).
> - Source: REF-01 Section 6 (Agent Types table), REF-04 Section 6

### 3.6 Structured Output Schemas

> **REQ-DM-014**: *RetrievedModel Schema* — The system shall define a Pydantic model `RetrievedModel` with fields:
>
> | Field | Type | Description |
> |-------|------|-------------|
> | `model_name` | `str` | Name of the retrieved ML model |
> | `example_code` | `str` | Concise example code for the model |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Figure 9 — `Model = {'model_name': str, 'example_code': str}`

> **REQ-DM-015**: *RetrieverOutput Schema* — The system shall define a Pydantic model `RetrieverOutput` with a single field `models: list[RetrievedModel]` and a `@field_validator` ensuring `len(models) >= 1`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `RetrieverOutput.model_json_schema()` shall produce a valid JSON schema compatible with `ClaudeAgentOptions.output_format`.

> **REQ-DM-016**: *RefinePlan Schema* — The system shall define a Pydantic model `RefinePlan` with fields:
>
> | Field | Type | Description |
> |-------|------|-------------|
> | `code_block` | `str` | Exact code block extracted from the solution script |
> | `plan` | `str` | Natural language refinement plan (3-5 sentences) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Figure 14 — `Refine_Plan = {'code_block': str, 'plan': str}`

> **REQ-DM-017**: *ExtractorOutput Schema* — The system shall define a Pydantic model `ExtractorOutput` with a single field `plans: list[RefinePlan]` and a `@field_validator` ensuring `len(plans) >= 1`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `ExtractorOutput.model_json_schema()` shall produce a valid JSON schema compatible with `ClaudeAgentOptions.output_format`.

> **REQ-DM-018**: *LeakageAnswer Schema* — The system shall define a Pydantic model `LeakageAnswer` with fields:
>
> | Field | Type | Description |
> |-------|------|-------------|
> | `leakage_status` | `Literal["Yes Data Leakage", "No Data Leakage"]` | Whether data leakage was detected |
> | `code_block` | `str` | The preprocessing code block extracted from the solution |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Figure 20 — `Answer = {'leakage_status': str, 'code_block': str}`

> **REQ-DM-019**: *LeakageDetectionOutput Schema* — The system shall define a Pydantic model `LeakageDetectionOutput` with a single field `answers: list[LeakageAnswer]` and a `@field_validator` ensuring `len(answers) >= 1`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `LeakageDetectionOutput.model_json_schema()` shall produce a valid JSON schema compatible with `ClaudeAgentOptions.output_format`.

> **REQ-DM-020**: *DataContaminationResult Schema* — The system shall define a Pydantic model `DataContaminationResult` with a single field `verdict: Literal["Novel", "Same"]`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Figure 28

### 3.7 Evaluation Result

> **REQ-DM-021**: *EvaluationResult Model* — The system shall define a Pydantic model `EvaluationResult` with the following fields:
>
> | Field | Type | Required | Description |
> |-------|------|----------|-------------|
> | `score` | `float \| None` | No | Parsed validation score (None if parsing failed) |
> | `stdout` | `str` | Yes | Full standard output from script execution |
> | `stderr` | `str` | Yes | Full standard error from script execution |
> | `exit_code` | `int` | Yes | Process exit code (0 = success) |
> | `duration_seconds` | `float` | Yes | Wall-clock execution time in seconds |
> | `is_error` | `bool` | Yes | Whether execution produced an error |
> | `error_traceback` | `str \| None` | No | Python traceback if error occurred |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Section 3.4 (A_debugger — receives T_bug traceback)

### 3.8 Phase Result Models

> **REQ-DM-022**: *Phase1Result Model* — The system shall define a Pydantic model `Phase1Result` with the following fields:
>
> | Field | Type | Description |
> |-------|------|-------------|
> | `retrieved_models` | `list[RetrievedModel]` | Models retrieved by A_retriever |
> | `candidate_solutions` | `list[SolutionScript]` | Scripts produced by A_init for each model |
> | `candidate_scores` | `list[float \| None]` | Scores for each candidate |
> | `initial_solution` | `SolutionScript` | Final merged solution s_0 |
> | `initial_score` | `float` | Best score after merging (h_best) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 1

> **REQ-DM-023**: *Phase2Result Model* — The system shall define a Pydantic model `Phase2Result` with the following fields:
>
> | Field | Type | Description |
> |-------|------|-------------|
> | `ablation_summaries` | `list[str]` | T_abl summaries collected across outer steps |
> | `refined_blocks` | `list[CodeBlock]` | Code blocks c_t refined in each outer step |
> | `best_solution` | `SolutionScript` | Best solution found during refinement (s_final) |
> | `best_score` | `float` | Score of best solution (h_best) |
> | `step_history` | `list[dict]` | Per-step records: plans tried, scores achieved |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 2

> **REQ-DM-024**: *Phase3Result Model* — The system shall define a Pydantic model `Phase3Result` with the following fields:
>
> | Field | Type | Description |
> |-------|------|-------------|
> | `input_solutions` | `list[SolutionScript]` | L solutions fed into ensemble |
> | `ensemble_plans` | `list[str]` | Ensemble plans e_r proposed across rounds |
> | `ensemble_scores` | `list[float \| None]` | Scores for each ensemble attempt |
> | `best_ensemble` | `SolutionScript` | Best ensemble solution (s_ens*) |
> | `best_ensemble_score` | `float` | Score of best ensemble |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 3

> **REQ-DM-025**: *FinalResult Model* — The system shall define a Pydantic model `FinalResult` with the following fields:
>
> | Field | Type | Description |
> |-------|------|-------------|
> | `task` | `TaskDescription` | The task that was solved |
> | `config` | `PipelineConfig` | Configuration used |
> | `phase1` | `Phase1Result` | Phase 1 output |
> | `phase2_results` | `list[Phase2Result]` | One Phase2Result per parallel solution path |
> | `phase3` | `Phase3Result \| None` | Phase 3 output (None if L=1, no ensemble) |
> | `final_solution` | `SolutionScript` | The solution submitted for test evaluation |
> | `submission_path` | `str` | Path to `./final/submission.csv` |
> | `total_duration_seconds` | `float` | Total pipeline wall-clock time |
> | `total_cost_usd` | `float \| None` | Total API cost if tracked |
>
> - Priority: Must | Verify: Test | Release: MVP

---

## 4. Interface Requirements

### 4.1 Score Function Interface

> **REQ-DM-026**: *ScoreFunction Protocol* — The system shall define a Python `Protocol` class `ScoreFunction` with a single method `__call__(self, solution: SolutionScript, task: TaskDescription) -> EvaluationResult`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 — h: S -> R

> **REQ-DM-027**: *Score Parsing Pattern* — The default `ScoreFunction` implementation shall parse the score from stdout by matching the regex pattern `r"Final Validation Performance:\s*([\d.eE+-]+)"` and converting the first captured group to `float`.
>
> - Error: If no match is found, `EvaluationResult.score` shall be `None`.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given stdout containing `"Final Validation Performance: 0.8196"`, the parsed score shall be `0.8196`.
> - Source: REF-01 Figures 10, 11, 15, 18, 19 (all require this output pattern)

> **REQ-DM-028**: *Score Comparison* — The system shall define a function `is_improvement(new_score: float, old_score: float, direction: MetricDirection) -> bool` that returns `True` when `new_score` is strictly better than `old_score` per the metric direction.
>
> - For `"maximize"`: `new_score > old_score`
> - For `"minimize"`: `new_score < old_score`
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 2 lines 12, 21 — `if h(s_t^k) >= h_best`

> **REQ-DM-029**: *Score Comparison (Equal-or-Better)* — The system shall define a function `is_improvement_or_equal(new_score: float, old_score: float, direction: MetricDirection) -> bool` that returns `True` when `new_score` is better than or equal to `old_score` per the metric direction.
>
> - For `"maximize"`: `new_score >= old_score`
> - For `"minimize"`: `new_score <= old_score`
> - Priority: Must | Verify: Test | Release: MVP
> - Rationale: The paper uses `>=` (not strict `>`) in Algorithm 1 line 11 and Algorithm 2 lines 12, 21 for the improvement check.

### 4.2 Prompt Template Registry

> **REQ-DM-030**: *PromptTemplate Model* — The system shall define a Pydantic model `PromptTemplate` with fields:
>
> | Field | Type | Description |
> |-------|------|-------------|
> | `agent_type` | `AgentType` | Which agent this template belongs to |
> | `figure_ref` | `str` | Paper figure reference (e.g., `"Figure 9"`) |
> | `template` | `str` | Template string with `{variable}` placeholders |
> | `variables` | `list[str]` | Required variable names for this template |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Figures 9-28

> **REQ-DM-031**: *PromptTemplate.render()* — The `PromptTemplate` model shall provide a method `render(**kwargs) -> str` that substitutes all `{variable}` placeholders with provided keyword arguments.
>
> - Error: Shall raise `KeyError` if a required variable is not provided.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given template `"List {M} models for {task_description}"` and `render(M=4, task_description="classify images")`, output shall be `"List 4 models for classify images"`.

> **REQ-DM-032**: *PromptRegistry Class* — The system shall define a `PromptRegistry` class that stores `PromptTemplate` instances keyed by `AgentType` and provides a `get(agent_type: AgentType) -> PromptTemplate` method.
>
> - Error: Shall raise `KeyError` if no template is registered for the given agent type.
> - Priority: Must | Verify: Test | Release: MVP

> **REQ-DM-033**: *PromptRegistry Coverage* — The `PromptRegistry` shall contain templates for all 14 agent types upon initialization.
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Acceptance: `len(registry)` shall equal 14 and include templates referencing Figures 9-22, 25-28 from the paper.
> - Source: REF-01 Appendix A (Figures 9-28)

> **REQ-DM-034**: *Leakage Agent Dual Templates* — The `PromptRegistry` shall store two templates for the `"leakage"` agent type: one for detection (Figure 20) and one for correction (Figure 21). The `get()` method shall accept an optional `variant: str` parameter to select between `"detection"` and `"correction"`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Figures 20, 21

> **REQ-DM-035**: *Subsampling Agent Dual Templates* — The `PromptRegistry` shall store two additional templates for subsampling operations: extraction (Figure 26) and removal (Figure 27). These shall be keyed under `AgentType.test` with variants `"subsampling_extract"` and `"subsampling_remove"`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Figures 26, 27

### 4.3 SDK Integration Types

> **REQ-DM-036**: *AgentConfig Model* — The system shall define a Pydantic model `AgentConfig` that maps each MLE-STAR agent to its SDK configuration:
>
> | Field | Type | Description |
> |-------|------|-------------|
> | `agent_type` | `AgentType` | MLE-STAR agent identity |
> | `description` | `str` | Value for `AgentDefinition.description` |
> | `system_prompt` | `str \| None` | Custom system prompt (None = use template) |
> | `tools` | `list[str] \| None` | Allowed SDK tools |
> | `model` | `Literal["sonnet", "opus", "haiku", "inherit"] \| None` | SDK model selection |
> | `output_schema` | `type \| None` | Pydantic model for structured output (if applicable) |
> | `max_turns` | `int \| None` | Maximum agent turns |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-06 Section 6 — `AgentDefinition` dataclass

> **REQ-DM-037**: *AgentConfig.to_agent_definition()* — The `AgentConfig` model shall provide a method `to_agent_definition() -> dict` that returns a dictionary suitable for use as a value in `ClaudeAgentOptions.agents`.
>
> - Output: `{"description": str, "prompt": str, "tools": list[str] | None, "model": str | None}`
> - Priority: Must | Verify: Test | Release: MVP

> **REQ-DM-038**: *AgentConfig.to_output_format()* — When `output_schema` is not `None`, the `AgentConfig` model shall provide a method `to_output_format() -> dict` that returns `{"type": "json_schema", "schema": self.output_schema.model_json_schema()}`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-06 Section 12 — Structured Outputs

---

## 5. Functional Requirements

### 5.1 Model Construction and Defaults

> **REQ-DM-039**: *Immutable Models* — All Pydantic models defined in this spec should use `model_config = ConfigDict(frozen=True)` to prevent mutation after construction, except for `SolutionScript` which shall use `frozen=False` to allow score updates.
>
> - Priority: Should | Verify: Inspection | Release: MVP
> - Rationale: Immutability prevents accidental state corruption in the multi-agent pipeline.

> **REQ-DM-040**: *Default Agent Configs* — The system shall provide a factory function `build_default_agent_configs() -> dict[AgentType, AgentConfig]` that returns pre-configured `AgentConfig` instances for all 14 agent types with:
> - `tools` appropriate to each agent's function (e.g., `["WebSearch", "WebFetch"]` for retriever; `["Bash"]` for execution-related agents)
> - `output_schema` set for agents that require structured output (retriever, extractor, leakage)
> - `model` set to `None` (inherit from orchestrator) by default
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `len(build_default_agent_configs())` shall equal 14.

### 5.2 Structured Output Compatibility

> **REQ-DM-041**: *JSON Schema Generation* — Every Pydantic model that serves as a structured output schema (RetrievedModel, RetrieverOutput, RefinePlan, ExtractorOutput, LeakageAnswer, LeakageDetectionOutput, DataContaminationResult) shall produce a valid JSON Schema via `.model_json_schema()` that is accepted by `ClaudeAgentOptions.output_format`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `{"type": "json_schema", "schema": Model.model_json_schema()}` shall be a valid `output_format` value.

### 5.3 Solution History Tracking

> **REQ-DM-042**: *RefinementAttempt Model* — The system shall define a Pydantic model `RefinementAttempt` with fields:
>
> | Field | Type | Description |
> |-------|------|-------------|
> | `plan` | `str` | The refinement plan text (p_k) |
> | `score` | `float \| None` | Score achieved by this attempt |
> | `code_block` | `str` | The modified code block (c_t^k) |
> | `was_improvement` | `bool` | Whether this improved upon h_best |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 2 — tracks `{(p_j, h(s_t^j))}` history

> **REQ-DM-043**: *EnsembleAttempt Model* — The system shall define a Pydantic model `EnsembleAttempt` with fields:
>
> | Field | Type | Description |
> |-------|------|-------------|
> | `plan` | `str` | The ensemble plan text (e_r) |
> | `score` | `float \| None` | Score achieved by this ensemble |
> | `solution` | `SolutionScript` | The ensemble solution (s_ens^r) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 3 — tracks `{(e_j, h(s_ens^j))}` history

---

## 6. Non-Functional Requirements

### 6.1 Performance

> **REQ-DM-044**: *Model Instantiation Speed* — All Pydantic models defined in this spec shall instantiate in under 1 millisecond for typical payloads (solution scripts up to 50 KB, plan texts up to 2 KB).
>
> - Measurement: `timeit` benchmark of model construction
> - Priority: Should | Verify: Test | Release: MVP

### 6.2 Maintainability

> **REQ-DM-045**: *Single Module* — All data models defined in this spec shall reside in a single Python module (e.g., `mle_star/models.py`) to centralize type definitions.
>
> - Priority: Should | Verify: Inspection | Release: MVP

> **REQ-DM-046**: *Re-export Convenience* — The package `__init__.py` shall re-export all public models and enums from the data models module.
>
> - Priority: Should | Verify: Inspection | Release: MVP

### 6.3 Portability

> **REQ-DM-047**: *Python Version Compatibility* — All data models shall be compatible with Python 3.10, 3.11, 3.12, and 3.13.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: All models shall pass type checking and instantiation tests on each listed Python version.

---

## 7. Constraints

### 7.1 Technology Constraints

> **REQ-DM-048**: *Pydantic v2 Dependency* — All data models shall use Pydantic v2 (>=2.0.0) BaseModel as their base class.
>
> - Rationale: Pydantic v2 provides `.model_json_schema()` needed for SDK `output_format` integration.
> - Priority: Must | Verify: Inspection | Release: MVP

> **REQ-DM-049**: *SDK Structured Output Compatibility* — All structured output schemas shall conform to the `{"type": "json_schema", "schema": ...}` format expected by `claude-agent-sdk` `ClaudeAgentOptions.output_format`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-06 Section 12

> **REQ-DM-050**: *No External Dependencies Beyond Pydantic* — The data models module shall not depend on any packages other than `pydantic`, Python standard library modules, and type stubs.
>
> - Rationale: Keep the foundation module lightweight; SDK types are referenced by string names, not by import.
> - Priority: Must | Verify: Inspection | Release: MVP

---

## 8. Traceability Matrix

### 8.1 Requirements to Paper Sections

| Req ID | Paper Section | Paper Element | SDK Construct |
|--------|--------------|---------------|---------------|
| REQ-DM-001 | Section 4 | Hyperparameters M,T,K,L,R | — |
| REQ-DM-002 | Section 4 | Validation | Pydantic validator |
| REQ-DM-003 | — | Serialization | — |
| REQ-DM-004 | Table 7 | Competition categories | — |
| REQ-DM-005 | Section 3 | Data modalities | — |
| REQ-DM-006 | Section 3 | Score direction | — |
| REQ-DM-007 | Section 3 | T_task | — |
| REQ-DM-008 | Algorithms 1-3 | Pipeline phases | — |
| REQ-DM-009 | Eq. 2,3,7,9 | Solution scripts | — |
| REQ-DM-010 | Eq. 7 | `s_t.replace(c_t, c_t^k)` | — |
| REQ-DM-011 | Section 3.2 | Code block categories | — |
| REQ-DM-012 | Eq. 6 | Code block c_t | — |
| REQ-DM-013 | Section 6 | All 14 agents | `AgentDefinition` |
| REQ-DM-014 | Figure 9 | Retriever JSON schema | `output_format` |
| REQ-DM-015 | Figure 9 | Retriever output | `output_format` |
| REQ-DM-016 | Figure 14 | Extractor JSON schema | `output_format` |
| REQ-DM-017 | Figure 14 | Extractor output | `output_format` |
| REQ-DM-018 | Figure 20 | Leakage JSON schema | `output_format` |
| REQ-DM-019 | Figure 20 | Leakage output | `output_format` |
| REQ-DM-020 | Figure 28 | Contamination check | `output_format` |
| REQ-DM-021 | Section 3.4 | Execution result + traceback | — |
| REQ-DM-022 | Algorithm 1 | Phase 1 outputs | — |
| REQ-DM-023 | Algorithm 2 | Phase 2 outputs | — |
| REQ-DM-024 | Algorithm 3 | Phase 3 outputs | — |
| REQ-DM-025 | Full pipeline | Final result | `ResultMessage` |
| REQ-DM-026 | Section 3 | h: S -> R | — |
| REQ-DM-027 | Figures 10-19 | "Final Validation Performance" | — |
| REQ-DM-028 | Alg 2 lines 12,21 | Score comparison | — |
| REQ-DM-029 | Alg 1 line 11 | Score comparison (>=) | — |
| REQ-DM-030 | Figures 9-28 | Prompt templates | — |
| REQ-DM-031 | Figures 9-28 | Template rendering | — |
| REQ-DM-032 | Figures 9-28 | Template registry | — |
| REQ-DM-033 | Appendix A | 14 agent templates | — |
| REQ-DM-034 | Figures 20,21 | Leakage dual templates | — |
| REQ-DM-035 | Figures 26,27 | Subsampling templates | — |
| REQ-DM-036 | — | Agent-to-SDK mapping | `AgentDefinition` |
| REQ-DM-037 | — | SDK dict output | `ClaudeAgentOptions.agents` |
| REQ-DM-038 | — | Structured output config | `ClaudeAgentOptions.output_format` |
| REQ-DM-039 | — | Immutability | `ConfigDict(frozen=True)` |
| REQ-DM-040 | Section 6 | All 14 agents configured | `AgentDefinition` |
| REQ-DM-041 | — | JSON schema validity | `output_format` |
| REQ-DM-042 | Algorithm 2 | Refinement history | — |
| REQ-DM-043 | Algorithm 3 | Ensemble history | — |
| REQ-DM-044 | — | Performance | — |
| REQ-DM-045 | — | Module organization | — |
| REQ-DM-046 | — | Re-exports | — |
| REQ-DM-047 | — | Python compat | — |
| REQ-DM-048 | — | Pydantic v2 | — |
| REQ-DM-049 | — | SDK compat | `output_format` |
| REQ-DM-050 | — | Minimal deps | — |

### 8.2 Cross-References to Other Specs

| Req ID | Referenced By |
|--------|--------------|
| REQ-DM-001 (PipelineConfig) | Specs 02-09 |
| REQ-DM-007 (TaskDescription) | Specs 02, 04, 08, 09 |
| REQ-DM-009 (SolutionScript) | Specs 02-09 |
| REQ-DM-010 (replace_block) | Specs 05, 06 |
| REQ-DM-012 (CodeBlock) | Specs 05, 06 |
| REQ-DM-013 (AgentType) | Specs 03-09 |
| REQ-DM-014-019 (Structured Schemas) | Specs 03, 04, 05 |
| REQ-DM-021 (EvaluationResult) | Spec 02 |
| REQ-DM-026-029 (Score Functions) | Spec 02 |
| REQ-DM-030-035 (Prompt Registry) | Specs 03-08 |
| REQ-DM-036-038 (AgentConfig) | Spec 09 |
| REQ-DM-042-043 (History Models) | Specs 05, 06, 07 |

---

## 9. Change Control

### 9.1 Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2026-02-20 | MLE-STAR Team | Initial draft — all 50 requirements |

### 9.2 Baselining Policy

This SRS is baselined at version 1.0. After baselining, changes require impact analysis across all 9 specs due to this document's foundational role.
