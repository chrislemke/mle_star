# SRS 01 — Data Models: Agents and Output Schemas

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
