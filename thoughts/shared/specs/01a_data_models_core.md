# SRS 01 — Data Models: Core Types

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
