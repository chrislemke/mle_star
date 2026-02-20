# SRS 03 — Safety Modules: Data Agent & Cross-Cutting

## 5. A_data Requirements

### 5.1 Agent Definition

> **REQ-SF-024**: *A_data Agent Definition* -- The system shall define an `AgentDefinition`-compatible configuration for the data usage agent with the following properties:
>
> | Property | Value |
> |----------|-------|
> | `agent_type` | `AgentType.data` |
> | `description` | Agent that verifies all provided data sources are used in the solution |
> | `prompt` | Rendered from the data template (Figure 22, REQ-DM-032) |
> | `tools` | `["Read"]` |
> | `output_schema` | `None` (free-form response: either code block or confirmation text) |
> | `model` | `None` (inherit from orchestrator) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Section 3.4, Figure 22

> **REQ-SF-025**: *A_data Prompt Template* -- The data usage agent prompt shall be constructed by rendering the Figure 22 template from the `PromptRegistry` (REQ-DM-032) with the following variables:
>
> | Variable | Type | Description |
> |----------|------|-------------|
> | `initial solution` | `str` | Full source code of the initial solution (`SolutionScript.content`) |
> | `task description` | `str` | Task description text (`TaskDescription.description`) |
>
> - The rendered prompt shall include all instructions from Figure 22: incorporate unused information, do not bypass with try-except, examine task description for extraction guidance, preserve `Final Validation Performance` print line, and the two response format options.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 22

### 5.2 Input/Output Contract

> **REQ-SF-026**: *A_data Input Contract* -- The data usage agent shall accept two inputs:
>
> 1. `solution: SolutionScript` -- the initial solution to check for data utilization.
> 2. `task: TaskDescription` -- the task description providing context about available data sources.
>
> - Precondition: `solution.content` is non-empty and `task.description` is non-empty.
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Equation 11 -- `s_0 <- A_data(s_0, T_task)`

> **REQ-SF-027**: *A_data Output Contract* -- The data usage agent shall return either:
>
> 1. A revised `SolutionScript` with updated code that incorporates previously unused data sources, **or**
> 2. The original `SolutionScript` unchanged, if all data sources are already used.
>
> - The system shall determine which case applies based on response parsing (REQ-SF-028).
> - Priority: Must | Verify: Test | Release: MVP

### 5.3 Response Parsing

> **REQ-SF-028**: *A_data Response Parsing* -- The system shall define a function `parse_data_agent_response(response: str, original_solution: SolutionScript) -> SolutionScript` that distinguishes between the two response formats:
>
> 1. **Confirmation**: If the response contains the exact phrase `"All the provided information is used."` (case-insensitive match), return the `original_solution` unchanged.
> 2. **Revised code**: Otherwise, extract the code block using `extract_code_block()` (REQ-SF-005) and return a new `SolutionScript` with the extracted code as `content`, preserving the original solution's `phase` and other metadata.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `parse_data_agent_response("All the provided information is used.", s)` shall return `s` unchanged.
> - Acceptance: `parse_data_agent_response("```python\nimport pandas as pd\n...\n```", s)` shall return a new `SolutionScript` with the extracted code.
> - Source: REF-01 Figure 22 -- two response format options

### 5.4 No try/except Enforcement

> **REQ-SF-029**: *A_data No try/except Instruction* -- The data agent prompt (REQ-SF-025) shall include the explicit instructions "DO NOT USE TRY AND EXCEPT; just occur error so we can debug it!" This ensures that errors from incorporating new data sources are surfaced to the debugger agent rather than silently caught.
>
> - This is enforced via prompt instruction. The system shall not perform automated try/except detection for A_data specifically; the general advisory detection in REQ-EX-045 applies.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 22

### 5.5 Integration Point

> **REQ-SF-030**: *A_data Runs Once After Initial Solution Generation* -- The data usage agent shall be invoked exactly once per pipeline execution, after the initial solution `s_0` is generated in Phase 1 (after merging, before Phase 2 refinement begins).
>
> - The calling code in Phase 1 (Spec 04) is responsible for the invocation.
> - This spec defines the function `check_data_usage(solution: SolutionScript, task: TaskDescription) -> SolutionScript`; Spec 04 defines the invocation point.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Section 3.4 -- "After initial solution generation (Phase 1 only)"

> **REQ-SF-031**: *A_data Evidence of Necessity* -- The data usage agent is critical for tasks with auxiliary data files. Without the data usage agent:
>
> - Performance on competitions with non-CSV data sources (e.g., `.xyz` geometry files in nomad2018) may be significantly degraded.
>
> This requirement documents the empirical justification; no implementation action is needed.
>
> - Priority: Informational | Verify: N/A | Release: N/A
> - Source: REF-01 Table 6 -- ablation study results

---

## 6. Cross-Cutting Requirements

### 6.1 Agent Configurations

> **REQ-SF-032**: *Safety Agent Default Configs* -- The `build_default_agent_configs()` function (REQ-DM-040) shall include `AgentConfig` entries for all three safety agents:
>
> | AgentType | Tools | Output Schema | Model |
> |-----------|-------|---------------|-------|
> | `debugger` | `["Read", "Bash"]` | `None` | `None` |
> | `leakage` | `["Read"]` | `LeakageDetectionOutput` (detection variant) / `None` (correction variant) | `None` |
> | `data` | `["Read"]` | `None` | `None` |
>
> - The leakage agent has two operational modes (detection and correction) sharing the same `AgentType.leakage`. The `AgentConfig` stored in the defaults shall be for the detection variant (with `output_schema`). The correction variant config shall be constructable by setting `output_schema=None` on a copy.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `build_default_agent_configs()[AgentType.debugger]` shall return an `AgentConfig` with `tools=["Read", "Bash"]`.
> - Source: REQ-DM-040

### 6.2 Prompt Template Usage

> **REQ-SF-033**: *Safety Agent Prompt Templates from Registry* -- All three safety agents shall obtain their prompt templates from the `PromptRegistry` (REQ-DM-032):
>
> | Agent | Registry Key | Variant | Figure |
> |-------|-------------|---------|--------|
> | A_debugger | `AgentType.debugger` | (default) | Figure 19 |
> | A_leakage detection | `AgentType.leakage` | `"detection"` | Figure 20 |
> | A_leakage correction | `AgentType.leakage` | `"correction"` | Figure 21 |
> | A_data | `AgentType.data` | (default) | Figure 22 |
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REQ-DM-032, REQ-DM-034

### 6.3 Type Contracts

> **REQ-SF-034**: *Safety Agents Operate on SolutionScript* -- All three safety agent functions shall accept `SolutionScript` (REQ-DM-009) as input and produce `SolutionScript` as output. This ensures uniform integration with the pipeline:
>
> | Function | Signature |
> |----------|-----------|
> | `debug_solution` | `(SolutionScript, str, TaskDescription, PipelineConfig) -> tuple[SolutionScript, EvaluationResult]` |
> | `check_and_fix_leakage` | `(SolutionScript) -> SolutionScript` |
> | `check_data_usage` | `(SolutionScript, TaskDescription) -> SolutionScript` |
>
> - Priority: Must | Verify: Inspection | Release: MVP
> - Rationale: Uniform type contracts allow safety agents to be composed and chained in any order.

> **REQ-SF-035**: *Safety Agents Produce New SolutionScript Instances* -- Safety agent functions shall return new `SolutionScript` instances rather than mutating the input. The original `SolutionScript` shall remain unchanged after a safety agent call.
>
> - Exception: If no change is needed (e.g., no leakage detected, all data used), the function may return the original instance.
> - Priority: Must | Verify: Test | Release: MVP
> - Rationale: Immutable data flow supports fallback to previous versions (REQ-SF-008).
