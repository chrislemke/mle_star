# SRS 09 — Orchestrator: Entry Point and SDK

| Field | Value |
|-------|-------|
| Version | 0.1.0 |
| Date | 2026-02-20 |
| Status | Draft |
| Spec ID | 09 of 09 |
| Requirement Prefix | REQ-OR- |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Product Perspective](#2-product-perspective)
3. [Pipeline Entry Point Requirements](#3-pipeline-entry-point-requirements)
4. [SDK Client Lifecycle Requirements](#4-sdk-client-lifecycle-requirements)
5. [Phase Orchestration Requirements](#5-phase-orchestration-requirements)
6. [Parallelism Requirements](#6-parallelism-requirements)
7. [Cost and Time Control Requirements](#7-cost-and-time-control-requirements)
8. [Hook Requirements](#8-hook-requirements)
9. [Result Assembly Requirements](#9-result-assembly-requirements)
10. [Error Handling Requirements](#10-error-handling-requirements)
11. [Configuration Requirements](#11-configuration-requirements)
12. [Non-Functional Requirements](#12-non-functional-requirements)
13. [Constraints](#13-constraints)
14. [Traceability Matrix](#14-traceability-matrix)
15. [Change Control](#15-change-control)

---

## 1. Introduction

### 1.1 Purpose

This SRS defines the top-level orchestrator for the MLE-STAR pipeline. It is the final integration spec (09 of 09) and specifies the single entry point that coordinates all pipeline phases: Phase 1 (initial solution generation), Phase 2 (nested-loop refinement with L parallel paths), Phase 3 (ensemble optimization), and finalization (submission generation). It also specifies SDK client lifecycle management, parallelism control, cost and time budgeting, hook-based observability, and error recovery strategies.

Intended audience: developers implementing the MLE-STAR system using the Claude Agent SDK for Python.

### 1.2 Scope

**Product name**: MLE-STAR (Machine Learning Engineering agent via Search and Targeted Refinement)

**What this spec covers**:
- `run_pipeline()` entry point: input validation, phase dispatch, result assembly
- SDK `ClaudeSDKClient` lifecycle: initialization, agent registration, permission configuration, cleanup
- Phase orchestration: sequential dispatch of Phase 1, Phase 2 (L parallel paths), Phase 3, finalization
- Parallelism: asyncio-based concurrent Phase 2 paths with session isolation
- Cost and time budgeting: per-phase time allocation, budget enforcement, graceful shutdown
- Hook registration: progress tracking, cost accumulation, safety enforcement, timeout monitoring
- Result assembly: `FinalResult` construction with cost and duration summaries
- Error handling: phase failure recovery, partial result preservation, diagnostic reporting
- Configuration: defaults, overrides, environment variable support, logging setup

**Out of scope**:
- Data model definitions (covered by Spec 01)
- Script execution and subprocess management (covered by Spec 02)
- Safety agent behavior logic (covered by Spec 03)
- Phase 1 internal logic (covered by Spec 04)
- Phase 2 outer loop internal logic (covered by Spec 05)
- Phase 2 inner loop internal logic (covered by Spec 06)
- Phase 3 ensemble internal logic (covered by Spec 07)
- Finalization internal logic (covered by Spec 08)

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|------|-----------|
| SRS | Software Requirements Specification |
| MLE-STAR | ML Engineering agent with web Search and TArgeted code block Refinement |
| Orchestrator | Top-level controller that sequences pipeline phases and manages global state |
| L | Number of parallel solution paths (default: 2) |
| T | Outer loop iterations for code block targeting (default: 4) |
| K | Inner loop iterations for refinement strategies (default: 4) |
| R | Ensemble strategy exploration rounds (default: 5) |
| SDK | Claude Agent SDK for Python (`claude-agent-sdk`) |
| Session | An SDK conversation context identified by a `session_id` string |
| Hook | An SDK callback invoked at specific lifecycle points (PreToolUse, PostToolUse, Stop) |
| Path | One independent Phase 2 refinement run (L paths total) |
| Budget | Maximum cost in USD that the pipeline may spend on API calls |

### 1.4 References

| ID | Title | Version | Source |
|----|-------|---------|--------|
| REF-01 | MLE-STAR paper | v3 | arXiv:2506.15692v3 |
| REF-02 | Claude Agent SDK reference | v0.1.39 | `claude-agent-sdk` PyPI |
| REF-03 | MLE-STAR architecture notes | -- | `thoughts/notes/mle_star_architecture.md` |
| REF-04 | MLE-STAR paper extraction | -- | `thoughts/notes/mle_star_paper.md` |
| REF-05 | Claude Agent SDK examples | -- | `thoughts/notes/claude_agent_sdk_examples.md` |
| REF-06 | Claude Agent SDK API reference | -- | `thoughts/notes/claude_agent_sdk_reference.md` |
| SPEC-01 | Data Models and Interfaces SRS | 0.1.0 | `thoughts/specs/01_data_models_and_interfaces.md` |
| SPEC-02 | Execution Harness SRS | 0.1.0 | `thoughts/specs/02_execution_harness.md` |
| SPEC-03 | Safety Modules SRS | 0.1.0 | `thoughts/specs/03_safety_modules.md` |
| SPEC-04 | Phase 1 SRS | 0.1.0 | `thoughts/specs/04_phase1_initial_solution.md` |
| SPEC-05 | Phase 2 Outer Loop SRS | 0.1.0 | `thoughts/specs/05_phase2_ablation_and_extraction.md` |
| SPEC-06 | Phase 2 Inner Loop SRS | 0.1.0 | `thoughts/specs/06_phase2_inner_loop_refinement.md` |
| SPEC-07 | Phase 3 Ensemble SRS | 0.1.0 | `thoughts/specs/07_phase3_ensemble.md` |
| SPEC-08 | Finalization SRS | 0.1.0 | `thoughts/specs/08_submission_and_finalization.md` |

### 1.5 Document Overview

- Section 3: Pipeline entry point (input validation, working directory, GPU detection)
- Section 4: SDK client lifecycle (initialization, agent registration, permissions, cleanup)
- Section 5: Phase orchestration (sequential dispatch, phase ordering)
- Section 6: Parallelism (L concurrent paths, session isolation, error isolation)
- Section 7: Cost and time control (budgeting, time allocation, graceful shutdown)
- Section 8: Hooks (progress, cost, safety, timeout, error logging)
- Section 9: Result assembly (FinalResult, cost summary, duration summary, lineage)
- Section 10: Error handling (phase failure recovery, partial results)
- Section 11: Configuration (defaults, overrides, environment variables, logging)
- Section 12: Non-functional requirements (performance, observability, reliability)
- Section 13: Constraints (technology, SDK compatibility)
- Section 14: Traceability matrix

---

## 2. Product Perspective

### 2.1 System Context

This spec is the integration layer that ties together all eight preceding specs. Every other spec defines a self-contained subsystem; this spec defines how they are composed into a complete pipeline.

```
Spec 09 (this) ──calls──> Spec 02 (setup_working_directory, detect_gpu)
               ──calls──> Spec 04 (run_phase1)
               ──calls──> Spec 05 (run_phase2_outer_loop)  x L parallel paths
               ──calls──> Spec 07 (run_phase3)
               ──calls──> Spec 08 (run_finalization)
               ──uses───> Spec 01 (all data models)
               ──uses───> Spec 03 (safety agents, invoked transitively via Specs 04-07)
               ──uses───> Spec 06 (inner loop, invoked transitively via Spec 05)
```

### 2.2 Product Functions Summary

1. Validate inputs and prepare the execution environment
2. Initialize the SDK client with all 14 agent definitions and hook registrations
3. Execute Phase 1 to produce an initial merged solution
4. Fork L parallel Phase 2 paths to independently refine the initial solution
5. Collect L refined solutions and execute Phase 3 ensemble optimization
6. Execute finalization to produce the test submission
7. Assemble and return the complete `FinalResult`
8. Enforce cost and time budgets throughout all phases
9. Handle errors gracefully with fallback to best-known solution

### 2.3 Operating Environment

- **Runtime**: Python 3.10+ with `asyncio` event loop
- **SDK**: `claude-agent-sdk` v0.1.39+
- **Validation**: Pydantic v2
- **Concurrency**: `asyncio.gather()` for L parallel paths
- **Hardware target**: 96 vCPUs, 360 GB RAM, 8 NVIDIA V100 GPUs (paper reference environment)
- **Time budget**: up to 24 hours per competition (default)

### 2.4 Assumptions and Dependencies

| ID | Assumption | Impact if Invalid |
|----|-----------|-------------------|
| A-01 | All eight preceding specs are implemented and their entry points are importable | Pipeline cannot execute |
| A-02 | The SDK client supports multiple concurrent sessions via `session_id` parameter | Parallel paths must be serialized |
| A-03 | The host machine has internet access for SDK API calls | Pipeline cannot execute |
| A-04 | `ANTHROPIC_API_KEY` environment variable is set | Client initialization fails |
| A-05 | Sufficient disk space for L parallel working directories | Phase 2 parallelism fails |
| A-06 | The asyncio event loop is available (not already running in conflicting context) | Concurrency unavailable |

| ID | Dependency | Owner | Risk if Unavailable |
|----|-----------|-------|---------------------|
| D-01 | `claude-agent-sdk` v0.1.39+ | Anthropic | Agent execution unavailable |
| D-02 | Pydantic v2 | Pydantic team | Model validation unavailable |
| D-03 | Spec 01 models module | MLE-STAR team | No shared type definitions |
| D-04 | Spec 02 execution harness | MLE-STAR team | No script execution capability |
| D-05 | Spec 04 `run_phase1()` | MLE-STAR team | No initial solution generation |
| D-06 | Spec 05 `run_phase2_outer_loop()` | MLE-STAR team | No refinement capability |
| D-07 | Spec 07 `run_phase3()` | MLE-STAR team | No ensemble capability |
| D-08 | Spec 08 `run_finalization()` | MLE-STAR team | No submission generation |

---

## 3. Pipeline Entry Point Requirements

### 3.1 Main Entry Point

> **REQ-OR-001**: *run_pipeline() Signature* -- The system shall define an async function `run_pipeline(task: TaskDescription, config: PipelineConfig | None = None) -> FinalResult` as the single top-level entry point for the MLE-STAR pipeline.
>
> - The function shall accept a `TaskDescription` (REQ-DM-007) and an optional `PipelineConfig` (REQ-DM-001).
> - When `config` is `None`, the function shall use `PipelineConfig()` with all paper-specified defaults.
> - The function shall return a fully populated `FinalResult` (REQ-DM-025) on success.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Calling `await run_pipeline(task)` with a valid `TaskDescription` shall return a `FinalResult` instance.
> - Source: REF-01 Section 3 (full pipeline), REF-03 (architecture overview)

### 3.2 Input Validation

> **REQ-OR-002**: *Input Validation* -- The `run_pipeline()` function shall validate its inputs before any phase execution begins:
>
> 1. `task` shall be validated as a well-formed `TaskDescription` instance (all required fields present, `competition_id` non-empty, `data_dir` points to an existing directory).
> 2. `config` shall be validated as a well-formed `PipelineConfig` instance (all fields positive per REQ-DM-002).
> 3. The `task.data_dir` directory shall exist and contain at least one file.
>
> - Error: Shall raise `ValueError` with a descriptive message if any validation fails.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `run_pipeline(task)` where `task.data_dir` does not exist shall raise `ValueError` before any SDK calls are made.

### 3.3 Working Directory Setup

> **REQ-OR-003**: *Working Directory Delegation* -- The `run_pipeline()` function shall delegate working directory setup to the execution harness (SPEC-02, REQ-EX-001) by calling `setup_working_directory(task)` before any phase begins.
>
> - This shall create or verify the `./input/` and `./final/` directory structure.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After `run_pipeline()` returns, both `./input/` and `./final/` directories shall exist.

### 3.4 GPU Detection

> **REQ-OR-004**: *GPU Detection Delegation* -- The `run_pipeline()` function shall delegate GPU detection to the execution harness (SPEC-02, REQ-EX-003) by calling `detect_gpu()` during initialization.
>
> - The detected GPU information shall be stored in pipeline state and made available to agent system prompts for hardware-aware code generation.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: On a machine with GPUs, the pipeline state shall contain GPU count and type; on a CPU-only machine, the state shall indicate no GPUs.

---

## 4. SDK Client Lifecycle Requirements

### 4.1 Client Initialization

> **REQ-OR-005**: *ClaudeSDKClient Creation* -- The orchestrator shall create a single `ClaudeSDKClient` instance during pipeline initialization with the following configuration:
>
> | Parameter | Value | Source |
> |-----------|-------|--------|
> | `model` | Configurable, default `"sonnet"` | PipelineConfig or env var |
> | `permission_mode` | Configurable, default `"bypassPermissions"` | REQ-OR-009 |
> | `max_budget_usd` | From `PipelineConfig.max_budget_usd` if set | REQ-OR-029 |
> | `agents` | All 14 agent definitions | REQ-OR-008 |
> | `hooks` | Registered hook instances | Section 8 |
>
> - The client shall be created inside an `async with` block or equivalent to ensure cleanup.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-06 Section 2 (ClaudeSDKClient initialization)

### 4.2 Agent Registration

> **REQ-OR-006**: *Agent Definition Registration* -- The orchestrator shall register all 14 MLE-STAR agents as `AgentDefinition` instances in the SDK client's `agents` dictionary. The 14 agents are:
>
> | Agent | AgentType | Primary Spec | Structured Output |
> |-------|-----------|-------------|-------------------|
> | A_retriever | `"retriever"` | SPEC-04 | `RetrieverOutput` |
> | A_init | `"init"` | SPEC-04 | None |
> | A_merger | `"merger"` | SPEC-04 | None |
> | A_abl | `"ablation"` | SPEC-05 | None |
> | A_summarize | `"summarize"` | SPEC-05 | None |
> | A_extractor | `"extractor"` | SPEC-05 | `ExtractorOutput` |
> | A_planner | `"planner"` | SPEC-06 | None |
> | A_coder | `"coder"` | SPEC-06 | None |
> | A_ens_planner | `"ens_planner"` | SPEC-07 | None |
> | A_ensembler | `"ensembler"` | SPEC-07 | None |
> | A_debugger | `"debugger"` | SPEC-03 | None |
> | A_leakage | `"leakage"` | SPEC-03 | `LeakageDetectionOutput` |
> | A_data | `"data"` | SPEC-03 | `DataContaminationResult` |
> | A_test | `"test"` | SPEC-08 | None |
>
> - Each agent shall be constructed using `build_default_agent_configs()` (REQ-DM-040), then converted via `AgentConfig.to_agent_definition()` (REQ-DM-037).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: The client's `agents` dictionary shall contain exactly 14 entries, keyed by `AgentType` string values.
> - Source: REF-06 Section 6 (AgentDefinition), SPEC-01 REQ-DM-013 (14 agents)

### 4.3 System Prompt Configuration

> **REQ-OR-007**: *Kaggle Grandmaster Persona* -- The SDK client's system prompt shall establish the Kaggle grandmaster persona used throughout the pipeline:
>
> ```
> You are a Kaggle grandmaster with expert-level skills in machine learning,
> data science, and competitive data analysis. You approach every task
> methodically, writing clean, efficient, and well-tested code. You always
> validate your solutions against the training data before submission.
> ```
>
> - The system prompt shall also include the task description (`task.description`), the evaluation metric (`task.evaluation_metric`), and the metric direction (`task.metric_direction`).
> - The system prompt shall include GPU availability information from REQ-OR-004.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Section 3 (agent persona), Figures 9-28 (prompt preambles)

### 4.4 Agent Tool Configuration

> **REQ-OR-008**: *Per-Agent Tool Assignment* -- Each agent definition shall specify only the SDK tools required for its function:
>
> | Agent | Tools |
> |-------|-------|
> | A_retriever | `["WebSearch", "WebFetch"]` |
> | A_init | `["Bash", "Edit", "Write", "Read"]` |
> | A_merger | `["Bash", "Edit", "Write", "Read"]` |
> | A_abl | `["Bash", "Edit", "Write", "Read"]` |
> | A_summarize | `["Read"]` |
> | A_extractor | `["Read"]` |
> | A_planner | `["Read"]` |
> | A_coder | `["Bash", "Edit", "Write", "Read"]` |
> | A_ens_planner | `["Read"]` |
> | A_ensembler | `["Bash", "Edit", "Write", "Read"]` |
> | A_debugger | `["Bash", "Edit", "Write", "Read"]` |
> | A_leakage | `["Read"]` |
> | A_data | `["Read"]` |
> | A_test | `["Bash", "Edit", "Write", "Read"]` |
>
> - Agents that do not need tool access (planning/analysis only) shall receive read-only tools.
> - Agents that produce code or execute scripts shall receive `Bash`, `Edit`, `Write`, and `Read`.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Section 3 (agent capabilities), REF-06 Section 3 (tools)

### 4.5 Permission Mode

> **REQ-OR-009**: *Configurable Permission Mode* -- The orchestrator shall support configurable permission modes for the SDK client:
>
> | Mode | Description | Use Case |
> |------|-------------|----------|
> | `"bypassPermissions"` | All tool uses auto-approved | Fully automated pipeline (default) |
> | `"acceptEdits"` | File edits require approval | Semi-automated with human review |
> | Custom callback | Fine-grained `can_use_tool` | Advanced control scenarios |
>
> - Default: `"bypassPermissions"` for fully automated execution.
> - The mode shall be configurable via `PipelineConfig` or constructor argument.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-06 Section 4 (Permission modes)

### 4.6 MCP Server Setup

> **REQ-OR-010**: *MCP Server Registration* -- The orchestrator shall register any required MCP (Model Context Protocol) servers for custom tool capabilities:
>
> 1. A score-parsing MCP tool that invokes the `ScoreFunction` (REQ-DM-026) and returns the parsed score.
> 2. A file-listing MCP tool that enumerates files in the data directory for agent context.
>
> - MCP servers shall be started before the first agent call and stopped during cleanup.
> - If MCP registration fails, the orchestrator shall log a warning and continue without custom tools.
> - Priority: Should | Verify: Test | Release: MVP
> - Source: REF-06 Section 10 (MCP Configuration)

### 4.7 Client Cleanup

> **REQ-OR-011**: *Client Disconnect on Completion or Error* -- The orchestrator shall guarantee that `client.disconnect()` is called when the pipeline completes, whether by normal completion, timeout, budget exhaustion, or unhandled exception.
>
> - Implementation shall use a `try/finally` block or `async with` context manager.
> - Any errors during disconnect shall be logged but not propagated (shall not mask the original error).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After `run_pipeline()` returns (or raises), the SDK client shall be disconnected and all sessions closed.
> - Source: REF-06 Section 2 (ClaudeSDKClient lifecycle)
