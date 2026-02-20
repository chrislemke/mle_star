# SRS 03 — Safety Modules: Leakage Agent

## 4. A_leakage Requirements

### 4.1 Detection Agent Definition

> **REQ-SF-011**: *A_leakage Detection Agent Definition* -- The system shall define an `AgentDefinition`-compatible configuration for the leakage detection agent with the following properties:
>
> | Property | Value |
> |----------|-------|
> | `agent_type` | `AgentType.leakage` |
> | `description` | Agent that detects data leakage in solution preprocessing code |
> | `prompt` | Rendered from the leakage detection template (Figure 20, REQ-DM-034 variant `"detection"`) |
> | `tools` | `["Read"]` |
> | `output_schema` | `LeakageDetectionOutput` (REQ-DM-019) |
> | `model` | `None` (inherit from orchestrator) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Section 3.4, Figure 20

> **REQ-SF-012**: *A_leakage Detection Prompt Template* -- The leakage detection agent prompt shall be constructed by rendering the Figure 20 template from the `PromptRegistry` (REQ-DM-034, variant `"detection"`) with the following variable:
>
> | Variable | Type | Description |
> |----------|------|-------------|
> | `code` | `str` | Full source code of the solution (`SolutionScript.content`) |
>
> - The rendered prompt shall include all instructions from Figure 20: extract preprocessing code block, check model trains on training samples only, check validation samples are not used for training before printing score, detect data leakage, return JSON schema with `leakage_status` and `code_block`.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 20

### 4.2 Detection Input/Output Contract

> **REQ-SF-013**: *A_leakage Detection Input Contract* -- The leakage detection agent shall accept a single input:
>
> 1. `solution: SolutionScript` -- the solution to check for data leakage.
>
> - Precondition: `solution.content` is non-empty.
> - Priority: Must | Verify: Test | Release: MVP

> **REQ-SF-014**: *A_leakage Detection Output Contract* -- The leakage detection agent shall return a `LeakageDetectionOutput` (REQ-DM-019) parsed from the agent's structured JSON response.
>
> - The output contains a list of `LeakageAnswer` objects (REQ-DM-018), each with:
>   - `leakage_status`: one of `"Yes Data Leakage"` or `"No Data Leakage"`
>   - `code_block`: the exact preprocessing code block extracted from the solution
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a solution with leakage, the response shall contain at least one `LeakageAnswer` with `leakage_status == "Yes Data Leakage"` and `code_block` matching an exact substring of `solution.content`.
> - Source: REF-01 Figure 20 -- `Answer = {'leakage_status': str, 'code_block': str}`

> **REQ-SF-015**: *A_leakage Detection Structured Output Usage* -- The leakage detection agent shall use the Claude Agent SDK's `output_format` parameter set to `{"type": "json_schema", "schema": LeakageDetectionOutput.model_json_schema()}` to ensure the response conforms to the `LeakageDetectionOutput` schema (REQ-DM-019).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: The agent response shall be parseable via `LeakageDetectionOutput.model_validate_json(response)` without errors.
> - Source: REF-01 Figure 20, REF-02 Section 12 (Structured Outputs)

### 4.3 Correction Agent Definition

> **REQ-SF-016**: *A_leakage Correction Agent Definition* -- The system shall define an `AgentDefinition`-compatible configuration for the leakage correction agent with the following properties:
>
> | Property | Value |
> |----------|-------|
> | `agent_type` | `AgentType.leakage` (shared with detection) |
> | `description` | Agent that corrects data leakage in solution preprocessing code |
> | `prompt` | Rendered from the leakage correction template (Figure 21, REQ-DM-034 variant `"correction"`) |
> | `tools` | `["Read"]` |
> | `output_schema` | `None` (free-form code block response) |
> | `model` | `None` (inherit from orchestrator) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Section 3.4, Figure 21

> **REQ-SF-017**: *A_leakage Correction Prompt Template* -- The leakage correction agent prompt shall be constructed by rendering the Figure 21 template from the `PromptRegistry` (REQ-DM-034, variant `"correction"`) with the following variable:
>
> | Variable | Type | Description |
> |----------|------|-------------|
> | `code` | `str` | Full source code of the solution (`SolutionScript.content`) |
>
> - The rendered prompt shall include all instructions from Figure 21: ensure model trains on training samples only, ensure validation samples are not used before printing score, refine code to prevent leakage, return single code block, note that variables are defined earlier.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 21

### 4.4 Correction Input/Output Contract

> **REQ-SF-018**: *A_leakage Correction Input Contract* -- The leakage correction agent shall accept a single input:
>
> 1. `solution: SolutionScript` -- the solution containing detected data leakage.
>
> - Precondition: At least one `LeakageAnswer` with `leakage_status == "Yes Data Leakage"` was returned by the detection step.
> - Priority: Must | Verify: Test | Release: MVP

> **REQ-SF-019**: *A_leakage Correction Output Contract* -- The leakage correction agent shall return a corrected code block extracted from the agent's response using `extract_code_block()` (REQ-SF-005).
>
> - The returned code block shall be a corrected version of the preprocessing code, suitable for replacing the original leaky code block in the solution.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: The returned code block shall be a non-empty string that does not match the original leaky `code_block` from the detection step.

### 4.5 Two-Step Orchestration

> **REQ-SF-020**: *A_leakage Two-Step Pipeline* -- The system shall define an async function `check_and_fix_leakage(solution: SolutionScript) -> SolutionScript` that orchestrates the two-step leakage detection and correction process:
>
> 1. Invoke the leakage detection agent with `solution` to obtain a `LeakageDetectionOutput`.
> 2. For each `LeakageAnswer` in the output where `leakage_status == "Yes Data Leakage"`:
>    a. Invoke the leakage correction agent with the current `solution`.
>    b. Extract the corrected code block from the correction response.
>    c. Replace the leaky code block in the solution using `SolutionScript.replace_block(answer.code_block, corrected_block)` (REQ-DM-010).
>    d. Update `solution` to the result of the replacement.
> 3. If no leakage was detected (all answers have `leakage_status == "No Data Leakage"`), return the original solution unchanged.
> 4. Return the final corrected solution.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a solution with leakage, the function shall return a `SolutionScript` where the leaky code block has been replaced with corrected code.
> - Source: REF-01 Section 3.4 -- `c_data* = A_leakage(c_data)`, then `s <- s.replace(c_data, c_data*)`

> **REQ-SF-021**: *A_leakage Code Block Replacement* -- When replacing a leaky code block, the `check_and_fix_leakage` function shall use `SolutionScript.replace_block(old, new)` (REQ-DM-010) where `old` is the `code_block` from the `LeakageAnswer` and `new` is the corrected code block from the correction agent.
>
> - If `replace_block` raises `ValueError` (the original code block is not found in the solution content), the system shall log a warning and skip the replacement for that answer, returning the solution unchanged for that particular leakage finding.
> - Priority: Must | Verify: Test | Release: MVP
> - Rationale: The detection agent may extract a code block that has minor whitespace differences from the actual solution content; graceful handling prevents crashes.

### 4.6 Integration Points

> **REQ-SF-022**: *A_leakage Runs Before Every Evaluation* -- The leakage checker shall be invoked on every generated or modified `SolutionScript` before it is evaluated, across all pipeline phases (Phase 1, Phase 2, Phase 3).
>
> - The calling phase is responsible for invoking `check_and_fix_leakage(solution)` and using the returned (potentially corrected) solution for evaluation.
> - This spec defines the function; Specs 04-07 define the invocation points.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Acceptance: Every path through the pipeline that leads to `evaluate_solution()` (REQ-EX-015) shall have a preceding call to `check_and_fix_leakage()`.
> - Source: REF-01 Section 3.4 -- "Every generated solution before evaluation (all phases)"

> **REQ-SF-023**: *A_leakage Evidence of Necessity* -- The leakage checker is critical for preventing overfitting to validation data. Without the leakage checker:
>
> - Validation score may be inflated (e.g., +5.0% on spaceship-titanic)
> - Test score may degrade severely (e.g., -8.9% on spaceship-titanic)
>
> This requirement documents the empirical justification; no implementation action is needed.
>
> - Priority: Informational | Verify: N/A | Release: N/A
> - Source: REF-01 Table 5 -- ablation study results
