# Software Requirements Specification: MLE-STAR Top-Level Orchestrator

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

---

## 5. Phase Orchestration Requirements

### 5.1 Phase 1 Dispatch

> **REQ-OR-012**: *Phase 1 Invocation* -- The orchestrator shall call `run_phase1(client, task, config)` (SPEC-04) as the first pipeline phase.
>
> - Input: the initialized `ClaudeSDKClient`, `TaskDescription`, and `PipelineConfig`.
> - Output: `Phase1Result` (REQ-DM-022) containing the merged initial solution `s_0` and its score `h_best`.
> - The orchestrator shall record the start time and duration of Phase 1.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 1, SPEC-04

### 5.2 Phase 2 Dispatch

> **REQ-OR-013**: *Phase 2 Parallel Path Dispatch* -- After Phase 1 completes, the orchestrator shall dispatch L parallel Phase 2 refinement paths (where L = `config.num_parallel_solutions`).
>
> - Each path shall receive a copy of the Phase 1 initial solution `s_0` as its starting point.
> - Each path shall independently call `run_phase2_outer_loop(client, task, config, initial_solution, session_id)` (SPEC-05).
> - Output: L `Phase2Result` instances (REQ-DM-023), one per path.
> - The orchestrator shall record the start time and aggregate duration of Phase 2.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Section 3.2 (L parallel solutions), SPEC-05

### 5.3 Phase 3 Dispatch

> **REQ-OR-014**: *Phase 3 Invocation* -- After all Phase 2 paths complete (or timeout), the orchestrator shall collect the best solution from each path and call `run_phase3(client, task, config, solutions)` (SPEC-07).
>
> - Input: a list of L `SolutionScript` instances (the `best_solution` from each `Phase2Result`).
> - If any Phase 2 path failed, the orchestrator shall use the Phase 1 initial solution as a substitute for that path's contribution (REQ-OR-040).
> - Output: `Phase3Result` (REQ-DM-024).
> - The orchestrator shall record the start time and duration of Phase 3.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 3, SPEC-07

### 5.4 Phase 3 Skip Condition

> **REQ-OR-015**: *Phase 3 Skip When L=1* -- When `config.num_parallel_solutions == 1`, the orchestrator shall skip Phase 3 entirely and pass the single Phase 2 result directly to finalization.
>
> - The `FinalResult.phase3` field shall be `None` in this case (per REQ-DM-025).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: With `PipelineConfig(num_parallel_solutions=1)`, Phase 3 shall not be invoked and `FinalResult.phase3` shall be `None`.

### 5.5 Finalization Dispatch

> **REQ-OR-016**: *Finalization Invocation* -- After Phase 3 completes (or is skipped), the orchestrator shall call `run_finalization(client, task, config, best_solution)` (SPEC-08).
>
> - Input: the best solution from Phase 3 (or Phase 2 if Phase 3 was skipped or failed).
> - Output: `FinalResult` (REQ-DM-025) including the `submission_path` to the generated `submission.csv`.
> - The orchestrator shall record the start time and duration of finalization.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: SPEC-08

### 5.6 Phase Ordering

> **REQ-OR-017**: *Strictly Sequential Phase Execution* -- The pipeline phases shall execute in strict sequential order: Phase 1 -> Phase 2 (L parallel) -> Phase 3 -> Finalization.
>
> - Phase 2 shall not begin until Phase 1 completes.
> - Phase 3 shall not begin until all Phase 2 paths complete (or timeout).
> - Finalization shall not begin until Phase 3 completes (or is skipped/failed).
> - Within Phase 2, the L paths run concurrently (see Section 6), but Phase 2 as a whole is a single sequential stage.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Timestamps shall confirm that no phase starts before its predecessor completes.
> - Source: REF-01 Section 3 (pipeline flow: Algorithm 1 -> Algorithm 2 -> Algorithm 3 -> submission)

---

## 6. Parallelism Requirements

### 6.1 L Parallel Paths

> **REQ-OR-018**: *L Independent Phase 2 Paths* -- The orchestrator shall create L independent Phase 2 refinement paths, where L = `config.num_parallel_solutions` (default: 2).
>
> - Phase 1 runs once and produces a single initial solution `s_0`.
> - The initial solution is copied (deep copy) to each of the L paths.
> - Each path independently refines its copy through T outer x K inner iterations.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Section 3.3 ("We run L=2 parallel solutions")

### 6.2 Concurrent Execution

> **REQ-OR-019**: *Asyncio-Based Concurrent Paths* -- The L Phase 2 paths shall run concurrently using `asyncio.gather()` (or equivalent asyncio concurrency primitive).
>
> ```python
> phase2_tasks = [
>     run_phase2_outer_loop(client, task, config, solution_copy, session_id=f"path-{i}")
>     for i in range(config.num_parallel_solutions)
> ]
> phase2_results = await asyncio.gather(*phase2_tasks, return_exceptions=True)
> ```
>
> - The orchestrator shall not serialize the L paths unless the SDK client cannot support concurrent sessions.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Section 4 (parallel solution paths)

### 6.3 Path Independence

> **REQ-OR-020**: *Forked Solution Copies* -- Each Phase 2 path shall operate on a deep copy of the Phase 1 initial solution, so that modifications in one path do not affect any other path.
>
> - The copy shall include the full `SolutionScript` content and score.
> - Each path shall have its own working subdirectory (e.g., `./work/path-0/`, `./work/path-1/`) to avoid file system conflicts.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Modifying the solution in path 0 shall have no effect on the solution in path 1.

### 6.4 Session Isolation

> **REQ-OR-021**: *Per-Path Session IDs* -- Each parallel Phase 2 path shall use a unique SDK `session_id` to maintain conversation context isolation.
>
> - Session IDs shall follow the pattern `"path-{i}"` where `i` is the zero-based path index.
> - All agent calls within a path shall use that path's session ID.
> - Sessions may be forked from the Phase 1 session using `ClaudeAgentOptions(resume=phase1_session_id, fork_session=True)` to carry forward context.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-06 Section 8 (Session management, fork_session)

### 6.5 Error Isolation

> **REQ-OR-022**: *Path Failure Isolation* -- A failure in one Phase 2 path shall not cause other paths to fail.
>
> - `asyncio.gather()` shall be called with `return_exceptions=True` to capture per-path exceptions.
> - Failed paths shall be logged with full exception details.
> - The orchestrator shall proceed with results from successful paths only.
> - If all L paths fail, the orchestrator shall fall back to the Phase 1 initial solution for ensemble (REQ-OR-040).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: If path 0 raises an exception, path 1 shall still complete normally and its result shall be used.

### 6.6 Path Result Collection

> **REQ-OR-023**: *Wait for All Paths* -- The orchestrator shall wait for all L paths to complete (or reach their per-path time budget) before proceeding to Phase 3.
>
> - The wait shall be bounded by the Phase 2 time allocation (REQ-OR-026).
> - If the time allocation is exceeded, still-running paths shall be cancelled via `asyncio.Task.cancel()`.
> - Cancelled paths shall be treated as failed paths (REQ-OR-022).
> - The orchestrator shall collect results only from paths that completed successfully.
> - Priority: Must | Verify: Test | Release: MVP

---

## 7. Cost and Time Control Requirements

### 7.1 Time Limit Enforcement

> **REQ-OR-024**: *Overall Time Limit* -- The `run_pipeline()` function shall enforce `config.time_limit_seconds` (default: 86400, i.e., 24 hours) as the maximum wall-clock time for the entire pipeline.
>
> - A `datetime`-based deadline shall be computed at pipeline start: `deadline = time.monotonic() + config.time_limit_seconds`.
> - The deadline shall be checked before each phase begins and at the start of each agent call.
> - If the deadline is exceeded, the pipeline shall invoke graceful shutdown (REQ-OR-030).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: A pipeline configured with `time_limit_seconds=60` that would otherwise run longer shall terminate within 90 seconds (allowing 30 seconds for shutdown).
> - Source: REF-01 Section 4 ("24-hour time limit per competition")

### 7.2 Per-Phase Time Budgets

> **REQ-OR-025**: *Proportional Time Allocation* -- The orchestrator shall allocate the total time budget across phases according to the following default proportions:
>
> | Phase | Proportion | Rationale |
> |-------|-----------|-----------|
> | Phase 1 | 10% | ~10 agent calls |
> | Phase 2 | 65% | ~44 agent calls per path, L paths |
> | Phase 3 | 15% | ~10 agent calls |
> | Finalization | 10% | Script execution + verification |
>
> - These proportions shall be configurable via a `PhaseTimeBudget` configuration (optional field on `PipelineConfig`).
> - The remaining time (after Phase 1 completes) shall be redistributed proportionally among remaining phases.
> - Priority: Should | Verify: Test | Release: MVP
> - Source: REF-01 Section 4 (average 14.1 hours per solution)

> **REQ-OR-026**: *Phase 2 Time Enforcement* -- Each parallel Phase 2 path shall receive `phase2_time_budget / L` seconds as its maximum execution time to ensure all paths get a fair share.
>
> - If a path exceeds its time budget, it shall be cancelled and its best intermediate result shall be used.
> - The Phase 2 time budget shall be calculated as: `remaining_time * 0.65 / 0.90` (to account for Phase 3 and finalization).
> - Priority: Should | Verify: Test | Release: MVP

### 7.3 Cost Tracking

> **REQ-OR-027**: *Cost Accumulation* -- The orchestrator shall accumulate `total_cost_usd` across all agent calls by reading `ResultMessage.total_cost_usd` from each SDK response.
>
> - Cost shall be tracked per-phase for the cost summary (REQ-OR-037).
> - Cost shall be tracked per-path for parallel Phase 2 paths.
> - The running total shall be accessible to hooks (REQ-OR-032).
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-06 Section 2 (ResultMessage.total_cost_usd)

### 7.4 Budget Enforcement

> **REQ-OR-028**: *PipelineConfig Budget Field* -- The `PipelineConfig` model (REQ-DM-001) shall include an optional field `max_budget_usd: float | None` (default: `None`, meaning unlimited).
>
> - When set, this field shall be passed to the SDK client as `max_budget_usd` during initialization.
> - Priority: Must | Verify: Test | Release: MVP

> **REQ-OR-029**: *Budget Exceeded Handling* -- When the accumulated cost reaches `config.max_budget_usd`, the orchestrator shall invoke graceful shutdown (REQ-OR-030).
>
> - The budget check shall occur after each agent call completes.
> - A warning shall be logged when 80% of the budget is consumed.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: A pipeline configured with `max_budget_usd=1.00` that accumulates $1.01 in cost shall trigger graceful shutdown.

### 7.5 Graceful Shutdown

> **REQ-OR-030**: *Graceful Shutdown on Timeout or Budget* -- When time or budget limits are exceeded, the orchestrator shall:
>
> 1. Cancel any in-progress agent calls and asyncio tasks.
> 2. Collect the best solution found so far (from whichever phase completed last).
> 3. If at least Phase 1 has completed, skip remaining phases and proceed directly to finalization with the best available solution.
> 4. If Phase 1 has not completed, raise `PipelineTimeoutError` with diagnostic information.
> 5. The `FinalResult` shall indicate which phases were completed and which were skipped.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: A pipeline that times out during Phase 2 shall still produce a `FinalResult` using the Phase 1 solution.

---

## 8. Hook Requirements

### 8.1 Progress Tracking Hook

> **REQ-OR-031**: *PostToolUse Progress Hook* -- The orchestrator shall register a `PostToolUse` hook that logs agent activity for observability:
>
> - Log entry shall include: timestamp, agent type, tool name, session ID, elapsed time, and a success/failure indicator.
> - Log format shall be structured (JSON) to support automated analysis.
> - The hook shall not modify tool results or block execution.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-06 Section 7 (Hooks)

### 8.2 Cost Tracking Hook

> **REQ-OR-032**: *Cost Accumulation Hook* -- The orchestrator shall register a hook (on `Stop` or `SubagentStop` events) that accumulates per-agent-call costs:
>
> - The hook shall read `ResultMessage.total_cost_usd` from each completed agent turn.
> - The hook shall update a shared cost accumulator (thread-safe for concurrent paths).
> - The hook shall trigger budget checks (REQ-OR-029) after each update.
> - Priority: Must | Verify: Test | Release: MVP

### 8.3 Safety Hook

> **REQ-OR-033**: *PreToolUse Safety Hook* -- The orchestrator shall register a `PreToolUse` hook that blocks dangerous bash commands:
>
> - Blocked patterns shall include: `rm -rf /`, `mkfs`, `dd if=`, `:(){ :|:& };:` (fork bomb), and any command that modifies files outside the working directory.
> - The hook shall return a `BlockToolUse` result with an explanation when a dangerous command is detected.
> - The blocked-command list shall be configurable.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: SPEC-02 (execution safety), REF-06 Section 7 (PreToolUse hook)

### 8.4 Timeout Hook

> **REQ-OR-034**: *Elapsed Time Monitoring Hook* -- The orchestrator shall register a hook that monitors elapsed wall-clock time and triggers graceful shutdown when the deadline is approaching:
>
> - The hook shall fire on every `PostToolUse` event.
> - When remaining time is less than 10% of the total budget (or less than 5 minutes, whichever is larger), the hook shall set a "finalize now" flag.
> - The "finalize now" flag shall cause the current phase to complete its current iteration and then skip remaining iterations.
> - Priority: Must | Verify: Test | Release: MVP

### 8.5 Error Logging Hook

> **REQ-OR-035**: *PostToolUse Error Logging Hook* -- The orchestrator shall register a hook that captures and logs all tool execution failures:
>
> - The hook shall fire on tool use results that indicate failure (non-zero exit codes, error messages).
> - Each failure shall be logged with: timestamp, agent type, tool name, error message, and full traceback if available.
> - The hook shall maintain a count of consecutive failures per agent type to support circuit-breaker logic.
> - Priority: Should | Verify: Inspection | Release: MVP

---

## 9. Result Assembly Requirements

### 9.1 FinalResult Construction

> **REQ-OR-036**: *FinalResult Assembly* -- Upon pipeline completion, the orchestrator shall construct a `FinalResult` (REQ-DM-025) by aggregating outputs from all completed phases:
>
> | FinalResult Field | Source |
> |-------------------|--------|
> | `task` | Input `TaskDescription` |
> | `config` | Input or defaulted `PipelineConfig` |
> | `phase1` | `Phase1Result` from `run_phase1()` |
> | `phase2_results` | List of L `Phase2Result` from `run_phase2_outer_loop()` calls |
> | `phase3` | `Phase3Result` from `run_phase3()` (or `None` if skipped) |
> | `final_solution` | `SolutionScript` from `run_finalization()` |
> | `submission_path` | Path to `./final/submission.csv` |
> | `total_duration_seconds` | `time.monotonic()` delta from start to end |
> | `total_cost_usd` | Accumulated cost from REQ-OR-027 |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Every field of `FinalResult` shall be populated (except `phase3` when L=1 or Phase 3 skipped).

### 9.2 Cost Summary

> **REQ-OR-037**: *Per-Phase Cost Breakdown* -- The orchestrator shall compute and log a per-phase cost breakdown:
>
> | Metric | Description |
> |--------|-------------|
> | `phase1_cost_usd` | Total cost of all Phase 1 agent calls |
> | `phase2_cost_usd` | Total cost of all Phase 2 agent calls (across all L paths) |
> | `phase2_per_path_cost_usd` | List of per-path costs |
> | `phase3_cost_usd` | Total cost of all Phase 3 agent calls |
> | `finalization_cost_usd` | Total cost of finalization agent calls |
> | `total_cost_usd` | Sum of all above |
>
> - The cost breakdown shall be included in the pipeline log output.
> - Priority: Should | Verify: Inspection | Release: MVP

### 9.3 Duration Summary

> **REQ-OR-038**: *Per-Phase Duration Breakdown* -- The orchestrator shall compute and log a per-phase duration breakdown:
>
> | Metric | Description |
> |--------|-------------|
> | `phase1_duration_seconds` | Wall-clock time for Phase 1 |
> | `phase2_duration_seconds` | Wall-clock time for Phase 2 (from first path start to last path end) |
> | `phase3_duration_seconds` | Wall-clock time for Phase 3 |
> | `finalization_duration_seconds` | Wall-clock time for finalization |
> | `total_duration_seconds` | Total pipeline wall-clock time |
>
> - The duration breakdown shall be included in the pipeline log output and in `FinalResult.total_duration_seconds`.
> - Priority: Should | Verify: Inspection | Release: MVP

### 9.4 Solution Lineage

> **REQ-OR-039**: *Solution Lineage Tracing* -- The orchestrator shall maintain a solution lineage that traces the final submitted solution back through each pipeline phase:
>
> 1. Phase 1: which retrieved models were used, which candidate was selected, merged solution score.
> 2. Phase 2: which path produced the best solution, which outer/inner steps produced improvements.
> 3. Phase 3: which ensemble round produced the best ensemble, which input solutions were combined.
> 4. Finalization: subsampling removal applied, test script modifications.
>
> - The lineage shall be logged at pipeline completion for debugging and reproducibility.
> - Priority: Should | Verify: Inspection | Release: MVP
> - Source: REF-01 Section 3 (pipeline data flow)

---

## 10. Error Handling Requirements

### 10.1 Phase Failure Recovery

> **REQ-OR-040**: *Phase 2 Failure Fallback* -- If Phase 2 fails for one or more paths (but Phase 1 succeeded), the orchestrator shall substitute the Phase 1 initial solution `s_0` for each failed path's contribution to the ensemble.
>
> - The substituted solution shall retain the Phase 1 score as its score.
> - The `Phase2Result` for a failed path shall contain the Phase 1 solution as `best_solution` and a flag `failed=True` in `step_history`.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: If both Phase 2 paths fail, Phase 3 shall receive two copies of the Phase 1 solution.

> **REQ-OR-041**: *Phase 3 Failure Fallback* -- If Phase 3 fails (but Phase 2 produced results), the orchestrator shall select the best Phase 2 solution (by score) and pass it directly to finalization.
>
> - The best solution shall be selected using `is_improvement_or_equal()` (REQ-DM-029) to compare across paths.
> - `FinalResult.phase3` shall be `None` when Phase 3 is skipped due to failure.
> - Priority: Must | Verify: Test | Release: MVP

> **REQ-OR-042**: *Complete Failure Handling* -- If all phases fail (including Phase 1), the orchestrator shall raise a `PipelineError` with:
>
> - The original exception from Phase 1.
> - Diagnostic information: elapsed time, cost consumed, last successful operation.
> - Any partial results collected before the failure.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `PipelineError` shall include a `diagnostics` attribute with structured failure information.

### 10.2 Partial Results

> **REQ-OR-043**: *Best-Effort Result Return* -- The orchestrator shall always attempt to return a `FinalResult` with the best solution found, even when later phases fail:
>
> | Failure Point | Best Available Solution | Phases Completed |
> |---------------|----------------------|------------------|
> | Phase 2 fails | Phase 1 initial solution | Phase 1 only |
> | Phase 3 fails | Best Phase 2 solution | Phase 1 + Phase 2 |
> | Finalization fails | Best pre-finalization solution | Phase 1 + Phase 2 + Phase 3 |
> | Timeout during Phase 2 | Best intermediate solution from any path | Partial |
>
> - The `FinalResult` shall indicate which phases completed successfully via the presence/absence of phase result fields.
> - If finalization itself fails, the orchestrator shall return a `FinalResult` without a `submission_path` (set to `""`) and log the failure.
> - Priority: Must | Verify: Test | Release: MVP

---

## 11. Configuration Requirements

### 11.1 Default Configurations

> **REQ-OR-044**: *Sensible Defaults for All Hyperparameters* -- The orchestrator shall provide sensible defaults via `PipelineConfig()` (REQ-DM-001) so that `run_pipeline(task)` works without any configuration:
>
> | Parameter | Default | Source |
> |-----------|---------|--------|
> | M (num_retrieved_models) | 4 | REF-01 Section 4 |
> | T (outer_loop_steps) | 4 | REF-01 Section 4 |
> | K (inner_loop_steps) | 4 | REF-01 Section 4 |
> | L (num_parallel_solutions) | 2 | REF-01 Section 4 |
> | R (ensemble_rounds) | 5 | REF-01 Section 4 |
> | time_limit_seconds | 86400 | REF-01 Section 4 (24h) |
> | max_budget_usd | None (unlimited) | -- |
> | max_debug_attempts | 3 | REF-01 Section 3.4 |
> | permission_mode | `"bypassPermissions"` | REF-06 Section 4 |
> | model | `"sonnet"` | -- |
> | log_level | `"INFO"` | -- |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `await run_pipeline(task)` with default `PipelineConfig` shall execute without errors on a valid task.

### 11.2 Configuration Override

> **REQ-OR-045**: *Per-Run Configuration Override* -- The `run_pipeline()` function shall accept a `PipelineConfig` instance that overrides any or all default hyperparameters:
>
> ```python
> config = PipelineConfig(
>     outer_loop_steps=2,          # T=2 instead of 4
>     inner_loop_steps=2,          # K=2 instead of 4
>     num_parallel_solutions=1,    # L=1, skip ensemble
>     time_limit_seconds=3600,     # 1 hour limit
>     max_budget_usd=10.0,         # $10 budget cap
> )
> result = await run_pipeline(task, config=config)
> ```
>
> - All hyperparameters in `PipelineConfig` shall be independently overridable.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `PipelineConfig(outer_loop_steps=1)` shall cause Phase 2 to run exactly 1 outer loop iteration.

### 11.3 Environment Variables

> **REQ-OR-046**: *Environment Variable Support* -- The orchestrator shall read the following environment variables during initialization:
>
> | Variable | Purpose | Required |
> |----------|---------|----------|
> | `ANTHROPIC_API_KEY` | SDK authentication | Yes |
> | `MLE_STAR_MODEL` | Override default model (e.g., `"opus"`) | No |
> | `MLE_STAR_LOG_LEVEL` | Override log level (e.g., `"DEBUG"`) | No |
> | `MLE_STAR_MAX_BUDGET` | Override max budget in USD | No |
> | `MLE_STAR_TIME_LIMIT` | Override time limit in seconds | No |
>
> - Environment variables shall take precedence over `PipelineConfig` defaults but be overridden by explicit `PipelineConfig` constructor arguments.
> - If `ANTHROPIC_API_KEY` is not set, `run_pipeline()` shall raise `EnvironmentError` with a clear message.
> - Priority: Must | Verify: Test | Release: MVP

### 11.4 Logging Configuration

> **REQ-OR-047**: *Configurable Logging* -- The orchestrator shall configure Python's `logging` module with:
>
> - Logger name: `"mle_star"`
> - Default level: `INFO` (overridable via `PipelineConfig.log_level` or `MLE_STAR_LOG_LEVEL`)
> - Console handler: structured log output with timestamp, level, logger name, and message
> - File handler (optional): if `PipelineConfig.log_file` is set, append logs to the specified file
> - Phase markers: log entries at phase boundaries (e.g., `"=== Phase 1: Initial Solution Generation ==="`)
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Acceptance: Setting `log_level="DEBUG"` shall produce debug-level output including agent prompts and responses.

---

## 12. Non-Functional Requirements

### 12.1 Performance

> **REQ-OR-048**: *Orchestrator Overhead* -- The orchestrator's own overhead (excluding agent calls and script execution) shall be less than 1% of total pipeline wall-clock time.
>
> - Orchestration logic (phase dispatch, result collection, hook execution) shall complete in under 100 milliseconds per phase transition.
> - Priority: Should | Verify: Test | Release: MVP

> **REQ-OR-049**: *Memory Efficiency for Parallel Paths* -- The orchestrator shall not hold all intermediate solutions in memory simultaneously. Each Phase 2 path shall retain only its current best solution and the solution under evaluation.
>
> - Historical solutions shall be summarized (score + metadata) rather than retained in full.
> - Priority: Should | Verify: Inspection | Release: MVP
> - Rationale: With L=2 paths, each generating up to T*K solutions (each potentially 50 KB+), unbounded retention could consume significant memory.

### 12.2 Observability

> **REQ-OR-050**: *Pipeline State Introspection* -- The orchestrator shall maintain a `PipelineState` object that is queryable at any time during execution:
>
> | Field | Type | Description |
> |-------|------|-------------|
> | `current_phase` | `str` | `"phase1"`, `"phase2"`, `"phase3"`, `"finalization"`, or `"complete"` |
> | `elapsed_seconds` | `float` | Wall-clock time since pipeline start |
> | `accumulated_cost_usd` | `float` | Total cost so far |
> | `phase2_path_statuses` | `list[str]` | Per-path status: `"running"`, `"completed"`, `"failed"`, `"cancelled"` |
> | `best_score_so_far` | `float | None` | Best score achieved across all phases |
> | `agent_call_count` | `int` | Total number of agent calls made |
>
> - Priority: Should | Verify: Inspection | Release: MVP

### 12.3 Reliability

> **REQ-OR-051**: *Idempotent Retry Safety* -- If `run_pipeline()` is called again after a previous failure, it shall not be affected by leftover state from the previous run.
>
> - Each `run_pipeline()` call shall create fresh state (new client, new sessions, new accumulators).
> - Working directory contents from a previous run shall be overwritten, not appended to.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Calling `run_pipeline(task)` twice in succession shall produce independent results.

> **REQ-OR-052**: *SDK Reconnection on Transient Failure* -- If the SDK client connection drops during pipeline execution, the orchestrator shall attempt to reconnect up to 3 times with exponential backoff (1s, 2s, 4s) before failing.
>
> - The reconnection shall resume the current session using `ClaudeAgentOptions(resume=session_id)`.
> - If all reconnection attempts fail, the orchestrator shall invoke graceful shutdown (REQ-OR-030).
> - Priority: Should | Verify: Test | Release: MVP
> - Source: REF-06 Section 8 (resume sessions)

---

## 13. Constraints

### 13.1 Technology Constraints

> **REQ-OR-053**: *Python 3.10+ and asyncio* -- The orchestrator shall be implemented as async Python code compatible with Python 3.10, 3.11, 3.12, and 3.13, using `asyncio` for concurrency.
>
> - The `run_pipeline()` function shall be an `async def` coroutine.
> - A synchronous wrapper `run_pipeline_sync()` shall be provided for non-async callers: `asyncio.run(run_pipeline(task, config))`.
> - Priority: Must | Verify: Test | Release: MVP

> **REQ-OR-054**: *SDK Version Dependency* -- The orchestrator shall require `claude-agent-sdk` v0.1.39 or later and shall not use deprecated SDK APIs.
>
> - If an incompatible SDK version is detected at import time, a clear `ImportError` shall be raised.
> - Priority: Must | Verify: Inspection | Release: MVP

> **REQ-OR-055**: *Single Module Implementation* -- The orchestrator shall be implemented in a single Python module (e.g., `mle_star/orchestrator.py`) to centralize pipeline control flow.
>
> - Helper classes (`PipelineState`, `PhaseTimeBudget`) may be defined in the same module.
> - Priority: Should | Verify: Inspection | Release: MVP

### 13.2 SDK Compatibility Constraints

> **REQ-OR-056**: *Concurrent Session Limit* -- The orchestrator shall respect the SDK client's maximum concurrent session count. If the SDK limits concurrent sessions to fewer than L, the orchestrator shall serialize excess paths (run them sequentially after earlier paths complete).
>
> - A warning shall be logged if serialization is required.
> - Priority: Must | Verify: Test | Release: MVP

> **REQ-OR-057**: *Agent Name Uniqueness* -- All 14 agent names registered with the SDK client shall be unique strings matching the `AgentType` enum values (REQ-DM-013).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: No two agents shall share the same name in the `agents` dictionary.

---

## 14. Traceability Matrix

### 14.1 Requirements to Paper Sections

| Req ID | Paper Section | Paper Element | SDK Construct |
|--------|--------------|---------------|---------------|
| REQ-OR-001 | Section 3 | Full pipeline | `ClaudeSDKClient` |
| REQ-OR-002 | Section 3 | Input validation | Pydantic validator |
| REQ-OR-003 | Section 4 | Working directory | -- |
| REQ-OR-004 | Section 4 | GPU detection | -- |
| REQ-OR-005 | -- | Client init | `ClaudeSDKClient()` |
| REQ-OR-006 | Section 6 | 14 agents | `AgentDefinition` |
| REQ-OR-007 | Section 3 | Kaggle persona | System prompt |
| REQ-OR-008 | Section 3 | Agent tools | `AgentDefinition.tools` |
| REQ-OR-009 | -- | Permissions | `permission_mode` |
| REQ-OR-010 | -- | Custom tools | MCP servers |
| REQ-OR-011 | -- | Cleanup | `client.disconnect()` |
| REQ-OR-012 | Algorithm 1 | Phase 1 dispatch | `run_phase1()` |
| REQ-OR-013 | Section 3.2 | L parallel Phase 2 | `run_phase2_outer_loop()` |
| REQ-OR-014 | Algorithm 3 | Phase 3 dispatch | `run_phase3()` |
| REQ-OR-015 | Algorithm 3 | L=1 skip condition | -- |
| REQ-OR-016 | Section 3.5 | Finalization | `run_finalization()` |
| REQ-OR-017 | Section 3 | Phase ordering | Sequential execution |
| REQ-OR-018 | Section 3.3 | L=2 paths | -- |
| REQ-OR-019 | Section 4 | Parallel execution | `asyncio.gather()` |
| REQ-OR-020 | Section 3.3 | Path independence | Deep copy |
| REQ-OR-021 | -- | Session isolation | `session_id` |
| REQ-OR-022 | -- | Error isolation | `return_exceptions=True` |
| REQ-OR-023 | -- | Result collection | `asyncio.gather()` |
| REQ-OR-024 | Section 4 | 24h time limit | -- |
| REQ-OR-025 | Section 4 | Time allocation | -- |
| REQ-OR-026 | Section 4 | Per-path budget | -- |
| REQ-OR-027 | -- | Cost tracking | `ResultMessage.total_cost_usd` |
| REQ-OR-028 | -- | Budget config | `max_budget_usd` |
| REQ-OR-029 | -- | Budget enforcement | -- |
| REQ-OR-030 | Section 4 | Graceful shutdown | -- |
| REQ-OR-031 | -- | Progress logging | `PostToolUse` hook |
| REQ-OR-032 | -- | Cost hook | `Stop` hook |
| REQ-OR-033 | -- | Safety hook | `PreToolUse` hook |
| REQ-OR-034 | Section 4 | Timeout hook | `PostToolUse` hook |
| REQ-OR-035 | -- | Error logging | `PostToolUse` hook |
| REQ-OR-036 | -- | Result assembly | `FinalResult` |
| REQ-OR-037 | -- | Cost summary | -- |
| REQ-OR-038 | -- | Duration summary | -- |
| REQ-OR-039 | Section 3 | Solution lineage | -- |
| REQ-OR-040 | -- | Phase 2 fallback | -- |
| REQ-OR-041 | -- | Phase 3 fallback | -- |
| REQ-OR-042 | -- | Complete failure | `PipelineError` |
| REQ-OR-043 | -- | Partial results | `FinalResult` |
| REQ-OR-044 | Section 4 | Default configs | `PipelineConfig()` |
| REQ-OR-045 | -- | Config override | `PipelineConfig(...)` |
| REQ-OR-046 | -- | Env vars | `os.environ` |
| REQ-OR-047 | -- | Logging | `logging` module |
| REQ-OR-048 | -- | Overhead | -- |
| REQ-OR-049 | -- | Memory efficiency | -- |
| REQ-OR-050 | -- | Observability | `PipelineState` |
| REQ-OR-051 | -- | Idempotency | -- |
| REQ-OR-052 | -- | Reconnection | `resume` session |
| REQ-OR-053 | -- | Python + asyncio | `async def` |
| REQ-OR-054 | -- | SDK version | `claude-agent-sdk` |
| REQ-OR-055 | -- | Single module | `orchestrator.py` |
| REQ-OR-056 | -- | Session limit | -- |
| REQ-OR-057 | -- | Agent uniqueness | `AgentType` enum |

### 14.2 Cross-References to Other Specs

| Req ID | References Spec | Referenced Requirement |
|--------|----------------|----------------------|
| REQ-OR-001 | SPEC-01 | REQ-DM-007 (TaskDescription), REQ-DM-001 (PipelineConfig), REQ-DM-025 (FinalResult) |
| REQ-OR-002 | SPEC-01 | REQ-DM-002 (PipelineConfig validation), REQ-DM-007 (TaskDescription) |
| REQ-OR-003 | SPEC-02 | REQ-EX-001 (setup_working_directory) |
| REQ-OR-004 | SPEC-02 | REQ-EX-003 (detect_gpu) |
| REQ-OR-005 | SPEC-01 | REQ-DM-036 (AgentConfig), REQ-DM-040 (build_default_agent_configs) |
| REQ-OR-006 | SPEC-01 | REQ-DM-013 (AgentType enum), REQ-DM-037 (to_agent_definition) |
| REQ-OR-008 | SPEC-01 | REQ-DM-036 (AgentConfig.tools) |
| REQ-OR-010 | SPEC-01 | REQ-DM-026 (ScoreFunction) |
| REQ-OR-012 | SPEC-04 | REQ-P1-* (Phase 1 entry point) |
| REQ-OR-013 | SPEC-05 | REQ-P2O-* (Phase 2 outer loop entry point) |
| REQ-OR-014 | SPEC-07 | REQ-P3-* (Phase 3 entry point) |
| REQ-OR-016 | SPEC-08 | REQ-FN-* (Finalization entry point) |
| REQ-OR-020 | SPEC-01 | REQ-DM-009 (SolutionScript) |
| REQ-OR-027 | SPEC-01 | REQ-DM-025 (FinalResult.total_cost_usd) |
| REQ-OR-033 | SPEC-02, SPEC-03 | REQ-EX-* (execution safety), REQ-SF-* (safety agents) |
| REQ-OR-036 | SPEC-01 | REQ-DM-022 (Phase1Result), REQ-DM-023 (Phase2Result), REQ-DM-024 (Phase3Result), REQ-DM-025 (FinalResult) |
| REQ-OR-040 | SPEC-04 | Phase 1 initial solution as fallback |
| REQ-OR-041 | SPEC-01 | REQ-DM-029 (is_improvement_or_equal) |

### 14.3 Requirements Referenced By Other Specs

This is the final integration spec (09 of 09). No other spec references requirements defined here, as this spec consumes all other specs' entry points.

---

## 15. Change Control

### 15.1 Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2026-02-20 | MLE-STAR Team | Initial draft -- all 57 requirements |

### 15.2 Baselining Policy

This SRS is baselined at version 1.0. After baselining, changes require impact analysis across all 9 specs, as this spec integrates all other specs and changes here may propagate to interface contracts. Changes to phase entry point signatures (REQ-OR-012 through REQ-OR-016) require coordinated updates with the referenced specs.
