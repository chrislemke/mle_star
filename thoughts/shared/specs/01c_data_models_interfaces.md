# SRS 01 — Data Models: Interfaces

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
