# SRS 02 — Execution Harness: Environment Setup

| Field | Value |
|-------|-------|
| Version | 0.1.0 |
| Date | 2026-02-20 |
| Status | Draft |
| Spec ID | 02 of 09 |
| Requirement Prefix | REQ-EX- |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Product Perspective](#2-product-perspective)
3. [Execution Environment Requirements](#3-execution-environment-requirements)
4. [Functional Requirements](#4-functional-requirements)
5. [Interface Requirements](#5-interface-requirements)
6. [Non-Functional Requirements](#6-non-functional-requirements)
7. [Constraints](#7-constraints)
8. [Traceability Matrix](#8-traceability-matrix)
9. [Change Control](#9-change-control)

---

## 1. Introduction

### 1.1 Purpose

This SRS defines the Python execution engine that runs agent-generated solution scripts, captures output, parses scores, manages working directories, enforces timeouts, and implements subsampling logic. It is the runtime layer between the agent-generated code and the score function defined in Spec 01.

Intended audience: developers implementing the MLE-STAR system using the Claude Agent SDK for Python.

### 1.2 Scope

**Product name**: MLE-STAR (Machine Learning Engineering agent via Search and Targeted Refinement)

**What this spec covers**:
- Writing solution scripts to temporary files and executing them as subprocesses
- Capturing stdout, stderr, and exit codes from script execution
- Parsing validation scores from stdout
- Extracting Python tracebacks from stderr for debugger input
- Constructing `EvaluationResult` objects from execution output
- Working directory setup and verification (`./input/`, `./final/`)
- Subsampling enforcement during refinement and removal before final submission
- Timeout enforcement and resource isolation
- GPU availability detection and environment variable setup
- Retry-after-debug execution pattern
- Score comparison delegation to Spec 01 utilities
- Submission file verification
- Batch evaluation of multiple solutions

**Out of scope**:
- Data model definitions (covered by Spec 01)
- Agent behavior logic (covered by Specs 03-08)
- Orchestration control flow (covered by Spec 09)
- Safety and leakage detection logic (covered by Spec 03)

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|------|-----------|
| SRS | Software Requirements Specification |
| MLE-STAR | ML Engineering agent with web Search and TArgeted code block Refinement |
| h(s) | Score function -- maps a solution script to a real-valued performance score |
| A_debugger | Debugger agent -- receives traceback and produces a fixed solution |
| Subsampling | Capping training data to 30,000 samples during refinement for faster iteration |
| Working directory | The directory containing `./input/` (data) and `./final/` (output) |
| Traceback | Python stack trace produced on unhandled exceptions |
| Exit code | Integer returned by a process; 0 indicates success, non-zero indicates failure |

### 1.4 References

| ID | Title | Version | Source |
|----|-------|---------|--------|
| REF-01 | MLE-STAR paper | v3 | arXiv:2506.15692v3 |
| REF-02 | Claude Agent SDK reference | v0.1.39 | `claude-agent-sdk` PyPI |
| REF-03 | MLE-STAR architecture notes | -- | `thoughts/notes/mle_star_architecture.md` |
| REF-04 | MLE-STAR paper extraction | -- | `thoughts/notes/mle_star_paper.md` |
| REF-05 | Spec 01 -- Data Models and Interfaces | 0.1.0 | `thoughts/specs/01_data_models_and_interfaces.md` |

### 1.5 Document Overview

- Section 3: Execution environment requirements (working directory, GPU, environment variables)
- Section 4: Functional requirements (script writing, execution, parsing, subsampling, retry)
- Section 5: Interface requirements (integration with Spec 01 types, SDK Bash tool)
- Section 6: Non-functional requirements (timeout, isolation, performance)
- Section 7: Constraints (technology, subprocess model)
- Section 8: Traceability matrix

---

## 2. Product Perspective

### 2.1 System Context

This spec defines the execution harness that sits between agent-generated code and the score function. Every phase that evaluates a solution script passes through this harness:

```
Agent-generated SolutionScript
        |
        v
  [Spec 02: Execution Harness]
   1. Write script to file
   2. Set up environment
   3. Execute as subprocess
   4. Capture stdout/stderr/exit_code
   5. Parse score from stdout
   6. Extract traceback (on error)
   7. Build EvaluationResult
        |
        v
  EvaluationResult -> consumed by calling phase
```

### 2.2 Dependency Diagram

```
Spec 01 (Data Models)
  |
  |-- SolutionScript (REQ-DM-009)
  |-- EvaluationResult (REQ-DM-021)
  |-- TaskDescription (REQ-DM-007)
  |-- PipelineConfig (REQ-DM-001)
  |-- MetricDirection (REQ-DM-006)
  |-- Score parsing pattern (REQ-DM-027)
  |-- is_improvement() (REQ-DM-028)
  |-- is_improvement_or_equal() (REQ-DM-029)
  |
  v
Spec 02 (this) ──> Spec 03 (Safety uses execution)
               ──> Spec 04 (Phase 1 evaluates candidates)
               ──> Spec 05 (Phase 2 outer evaluates refined solutions)
               ──> Spec 06 (Phase 2 inner evaluates code block changes)
               ──> Spec 07 (Phase 3 evaluates ensembles)
               ──> Spec 08 (Submission runs final evaluation)
               ──> Spec 09 (Orchestrator coordinates execution)
```

### 2.3 Product Functions Summary

1. Write solution scripts to temporary files and execute them as Python subprocesses
2. Capture and parse execution output to build `EvaluationResult` objects
3. Manage working directory structure (`./input/`, `./final/`)
4. Enforce subsampling during refinement and remove subsampling before final submission
5. Enforce timeout limits and provide resource isolation
6. Support retry-after-debug execution patterns
7. Verify submission files after final execution

### 2.4 Operating Environment

- **Runtime**: Python 3.10+
- **Execution model**: `subprocess.run()` or equivalent for script isolation
- **Hardware target**: 96 vCPUs, 360 GB RAM, 8 NVIDIA V100 GPUs (per REF-01 Appendix F)
- **SDK alternative**: Execution may use the SDK `Bash` tool (`{"command": str, "timeout": int}` -> `{"output": str, "exitCode": int}`)

### 2.5 Assumptions and Dependencies

| ID | Assumption | Impact if Invalid |
|----|-----------|-------------------|
| A-01 | Solution scripts are self-contained single-file Python programs | Multi-file execution would require packaging logic |
| A-02 | Python 3.10+ is available in the execution environment | Scripts may use language features unavailable on older versions |
| A-03 | The `./input/` directory contains pre-downloaded competition data | Scripts will fail at data loading |
| A-04 | NVIDIA drivers and CUDA toolkit are installed if GPU execution is needed | GPU-dependent scripts will fail |
| A-05 | Scripts print `"Final Validation Performance: {score}"` to stdout on success | Score parsing will return None |

| ID | Dependency | Owner | Risk if Unavailable |
|----|-----------|-------|---------------------|
| D-01 | Spec 01 types (SolutionScript, EvaluationResult, etc.) | Spec 01 | Cannot construct inputs or outputs |
| D-02 | Python subprocess module | Python stdlib | Cannot execute scripts |
| D-03 | `claude-agent-sdk` v0.1.39+ (if using SDK Bash tool) | Anthropic | Must fall back to direct subprocess |

---

## 3. Execution Environment Requirements

### 3.1 Working Directory Setup

> **REQ-EX-001**: *Working Directory Structure* -- The system shall define a function `setup_working_directory(base_path: str) -> str` that creates or verifies the following directory structure:
>
> ```
> {base_path}/
>   input/      # Competition dataset (read-only from script perspective)
>   final/      # Script output directory (submission.csv written here)
> ```
>
> - The function shall create `input/` and `final/` if they do not exist.
> - The function shall return the absolute path to `base_path`.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: After calling `setup_working_directory("/tmp/comp1")`, both `/tmp/comp1/input/` and `/tmp/comp1/final/` shall exist.
> - Source: REF-01 Figures 10, 11, 15, 18, 19, 25 -- data in `./input/`, output to `./final/`

> **REQ-EX-002**: *Working Directory Cleanup* -- The system shall define a function `clean_output_directory(base_path: str) -> None` that removes all files in `{base_path}/final/` without deleting the directory itself.
>
> - Rationale: Each script execution should start with a clean output directory to avoid stale submission files.
> - Priority: Should | Verify: Test | Release: MVP
> - Acceptance: After calling `clean_output_directory`, the `final/` directory shall exist but contain no files.

### 3.2 GPU and Hardware Detection

> **REQ-EX-003**: *GPU Availability Check* -- The system shall define a function `detect_gpu_info() -> dict` that returns a dictionary with at minimum the following keys:
>
> | Key | Type | Description |
> |-----|------|-------------|
> | `cuda_available` | `bool` | Whether CUDA is available |
> | `gpu_count` | `int` | Number of visible GPUs (0 if CUDA unavailable) |
> | `gpu_names` | `list[str]` | Names of detected GPUs |
>
> - The function shall not raise exceptions; it shall return `{"cuda_available": False, "gpu_count": 0, "gpu_names": []}` if detection fails.
> - Priority: Should | Verify: Demonstration | Release: MVP
> - Source: REF-01 Appendix F -- 8 NVIDIA V100 GPUs

### 3.3 Environment Variables

> **REQ-EX-004**: *Execution Environment Variables* -- The system shall define a function `build_execution_env(gpu_indices: list[int] | None = None) -> dict[str, str]` that returns a copy of the current environment with the following additions or overrides:
>
> | Variable | Value | Description |
> |----------|-------|-------------|
> | `PYTHONUNBUFFERED` | `"1"` | Ensure real-time stdout/stderr capture |
> | `PYTHONHASHSEED` | `"0"` | Deterministic hash behavior for reproducibility |
> | `CUDA_VISIBLE_DEVICES` | Comma-separated `gpu_indices` or all available | Control GPU visibility |
>
> - If `gpu_indices` is `None`, the `CUDA_VISIBLE_DEVICES` variable shall not be set (inherit from parent).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `build_execution_env(gpu_indices=[0, 1])` shall return an env dict where `CUDA_VISIBLE_DEVICES == "0,1"`.
