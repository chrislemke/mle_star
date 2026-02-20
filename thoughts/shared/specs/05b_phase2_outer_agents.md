# SRS 05 — Phase 2 Outer Loop: Summarize and Extractor Agents

---

## 4. A_summarize Requirements

### 4.1 Agent Definition

> **REQ-P2O-008**: *A_summarize Agent Definition* -- The system shall define an `AgentDefinition`-compatible configuration for the ablation summarization agent with the following properties:
>
> | Property | Value |
> |----------|-------|
> | `agent_type` | `AgentType.summarize` |
> | `description` | Agent that summarizes raw ablation study output into a concise textual summary |
> | `prompt` | Rendered from the summarization template (Figure 13, REQ-DM-032) |
> | `tools` | `None` (no tools needed) |
> | `output_schema` | `None` (free-form text summary) |
> | `model` | `None` (inherit from orchestrator) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `AgentConfig(agent_type=AgentType.summarize).to_agent_definition()` shall produce a valid dictionary for `ClaudeAgentOptions.agents`.
> - Source: REF-01 Section 3.2, Figure 13

> **REQ-P2O-009**: *A_summarize Prompt Template* -- The ablation summarization agent prompt shall be constructed by rendering the Figure 13 template from the `PromptRegistry` (REQ-DM-032) with the following variables:
>
> | Variable | Type | Description |
> |----------|------|-------------|
> | `ablation_code` | `str` | Full source code of the ablation study script (a_t) |
> | `raw_result` | `str` | Raw execution output (stdout) from running the ablation script (r_t) |
>
> - The rendered prompt shall include:
>   1. The ablation study code that was executed
>   2. The raw printed output from execution
>   3. Instruction to summarize the result of the ablation study based on the code and printed output
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 13

### 4.2 Input/Output Contract

> **REQ-P2O-010**: *A_summarize Input Contract* -- The ablation summarization agent shall accept two inputs:
>
> 1. `ablation_code: str` -- the source code of the ablation study script that was executed.
> 2. `raw_output: str` -- the raw stdout captured from executing the ablation script.
>
> - Precondition: `ablation_code` is non-empty. `raw_output` may be empty if the script produced no output (though this is an abnormal condition).
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 2 line 7 -- `T_abl^t = A_summarize(a_t, r_t)`

> **REQ-P2O-011**: *A_summarize Output Contract* -- The ablation summarization agent shall return a plain text summary string (T_abl^t).
>
> - The summary shall be the complete text content of the agent's response (no code block extraction, no structured parsing).
> - The summary shall identify which code components had the most and least impact on model performance.
> - The summary text shall be stored as-is for accumulation in T_abl and for input to A_extractor.
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given an ablation output showing "Baseline: 0.8196, No StandardScaler: 0.8102, No OneHotEncoder: 0.7886, No Imputation: 0.8196", the summary shall identify OneHotEncoder as the most impactful component.
> - Source: REF-01 Algorithm 2 line 7, Appendix C Figures 23-24

---

## 5. A_extractor Requirements

### 5.1 Agent Definition

> **REQ-P2O-012**: *A_extractor Agent Definition* -- The system shall define an `AgentDefinition`-compatible configuration for the code block extractor agent with the following properties:
>
> | Property | Value |
> |----------|-------|
> | `agent_type` | `AgentType.extractor` |
> | `description` | Agent that identifies the most impactful code block and proposes a refinement plan |
> | `prompt` | Rendered from the extractor template (Figure 14, REQ-DM-032) |
> | `tools` | `["Read"]` |
> | `output_schema` | `ExtractorOutput` (REQ-DM-017) |
> | `model` | `None` (inherit from orchestrator) |
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `AgentConfig(agent_type=AgentType.extractor).to_agent_definition()` shall produce a valid dictionary for `ClaudeAgentOptions.agents`.
> - Source: REF-01 Section 3.2, Figure 14

> **REQ-P2O-013**: *A_extractor Prompt Template* -- The code block extractor agent prompt shall be constructed by rendering the Figure 14 template from the `PromptRegistry` (REQ-DM-032) with the following variables:
>
> | Variable | Type | Description |
> |----------|------|-------------|
> | `solution_script` | `str` | Full source code of the current best solution (`SolutionScript.content`) |
> | `ablation_summary` | `str` | Summary of the current ablation study (T_abl^t) |
> | `previous_code_blocks` | `list[str]` | List of previously refined code blocks (C) |
>
> - The rendered prompt shall include all instructions from Figure 14:
>   1. Kaggle grandmaster persona introduction
>   2. Goal: extract a code block and improve it for better performance
>   3. Current Python solution presented in full
>   4. Ablation study results summary
>   5. Previously improved code blocks (numbered)
>   6. Instruction to suggest an effective plan (3-5 sentences)
>   7. Instruction to avoid plans with excessive runtime (e.g., large hyperparameter search)
>   8. Instruction to improve parts not previously considered
>   9. Instruction to extract a code block exactly from the Python script
>   10. Response format: JSON schema `Refine_Plan = {'code_block': str, 'plan': str}` as `list[Refine_Plan]`
> - When `previous_code_blocks` is empty (first iteration), the section for previous code blocks shall be omitted.
> - Priority: Must | Verify: Inspection | Release: MVP
> - Source: REF-01 Figure 14

### 5.2 Structured Output

> **REQ-P2O-014**: *A_extractor Structured Output Usage* -- The extractor agent shall use the Claude Agent SDK's `output_format` parameter set to `{"type": "json_schema", "schema": ExtractorOutput.model_json_schema()}` to ensure the response conforms to the `ExtractorOutput` schema (REQ-DM-017).
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: The agent response shall be parseable via `ExtractorOutput.model_validate_json(response)` without errors.
> - Source: REF-01 Figure 14, REF-02 Section 12 (Structured Outputs)

> **REQ-P2O-015**: *A_extractor Output Contract* -- The extractor agent shall return an `ExtractorOutput` (REQ-DM-017) parsed from the agent's structured JSON response, containing one or more `RefinePlan` objects (REQ-DM-016), each with:
>
> - `code_block: str` -- an exact code substring extracted from the current solution
> - `plan: str` -- a natural language refinement plan (3-5 sentences)
>
> - The system shall use the **first** `RefinePlan` in the returned list as the primary target for the current outer loop iteration (c_t, p_0).
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: `ExtractorOutput.plans[0].code_block` shall be an exact substring of `solution.content`.
> - Source: REF-01 Algorithm 2 line 8 -- `c_t, p_0 = A_extractor(T_abl^t, s_t, C)`

### 5.3 Input Contract

> **REQ-P2O-016**: *A_extractor Input Contract* -- The code block extractor agent shall accept three inputs:
>
> 1. `summary: str` -- the ablation summary for the current outer step (T_abl^t).
> 2. `solution: SolutionScript` -- the current best solution at outer step t.
> 3. `previous_blocks: list[str]` -- list of code block content strings from all previously refined code blocks (C).
>
> - Precondition: `summary` is non-empty and `solution.content` is non-empty.
> - `previous_blocks` may be empty (first outer loop iteration).
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-01 Algorithm 2 line 8 -- `c_t, p_0 = A_extractor(T_abl^t, s_t, C)`

### 5.4 Code Block Validation

> **REQ-P2O-017**: *Code Block Exact Substring Validation* -- The system shall define a function `validate_code_block(code_block: str, solution: SolutionScript) -> bool` that returns `True` if and only if `code_block` is found as an exact substring within `solution.content`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a solution containing `"scaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)"` and a code block `"scaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)"`, validation shall return `True`.
> - Acceptance: A code block with any character difference (including whitespace) from the solution content shall cause validation to return `False`.
> - Source: REF-01 Figure 14 -- "The code block can be long but should be exactly extracted from the Python script provided above."

> **REQ-P2O-018**: *Code Block Validation Failure Handling* -- If `validate_code_block()` returns `False` for the extracted code block, the system shall:
>
> 1. Attempt a whitespace-normalized match: strip trailing whitespace from each line of both the code block and the solution, then check for substring match. If this succeeds, use the whitespace-normalized code block (matched from the solution source) as c_t.
> 2. If the whitespace-normalized match also fails, log a warning and re-invoke A_extractor with the same inputs (up to 2 re-invocations). Include in the re-invocation prompt an additional instruction: "The previously extracted code block was not found in the solution. Please extract the code block exactly as it appears in the script."
> 3. If all re-invocations fail validation, select the first `RefinePlan` whose `code_block` passes validation from the `ExtractorOutput.plans` list. If none pass, skip this outer loop iteration and proceed to the next.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: Given a code block with trailing whitespace differences, the whitespace-normalized match shall succeed and return the correct substring from the solution.
> - Source: REF-01 Equation 7 -- `s_t.replace(c_t, c_t^k)` requires c_t to be an exact substring
