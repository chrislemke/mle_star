"""Core data models for the MLE-STAR pipeline.

Defines shared Pydantic models, enums, and configuration types used across
all pipeline phases. This module is the foundational type system referenced
by all other modules (execution, safety, phases, orchestrator).

Refs:
    SRS 01a — Data Models Core (REQ-DM-001 through REQ-DM-012).
    SRS 01b — Data Models Agents and Output Schemas (REQ-DM-013 through REQ-DM-020).
    SRS 01b — Evaluation & Phase Results (REQ-DM-021 through REQ-DM-025).
    SRS 01d — Iteration Records (REQ-DM-042, REQ-DM-043, REQ-P2I-036).
    IMPLEMENTATION_PLAN.md Tasks 03, 04, 05, 06, 10.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class TaskType(StrEnum):
    """Type of ML task for a Kaggle competition (REQ-DM-004).

    Eight categories covering the competition types supported by MLE-STAR.
    """

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_TO_IMAGE = "image_to_image"
    TEXT_CLASSIFICATION = "text_classification"
    AUDIO_CLASSIFICATION = "audio_classification"
    SEQUENCE_TO_SEQUENCE = "sequence_to_sequence"
    TABULAR = "tabular"


class DataModality(StrEnum):
    """Primary data modality for a competition dataset (REQ-DM-005)."""

    TABULAR = "tabular"
    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"
    MIXED = "mixed"


class MetricDirection(StrEnum):
    """Whether the evaluation metric should be maximized or minimized (REQ-DM-006)."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class PhaseTimeBudget(BaseModel):
    """Proportional time budget allocation across pipeline phases.

    Each field is a percentage of the total time budget. The four percentages
    must sum to exactly 100.0.

    Attributes:
        phase1_pct: Percentage allocated to Phase 1 (model retrieval + init).
        phase2_pct: Percentage allocated to Phase 2 (refinement loops).
        phase3_pct: Percentage allocated to Phase 3 (ensemble).
        finalization_pct: Percentage allocated to finalization.
    """

    model_config = ConfigDict(frozen=True)

    phase1_pct: float = 10.0
    phase2_pct: float = 65.0
    phase3_pct: float = 15.0
    finalization_pct: float = 10.0

    @model_validator(mode="after")
    def _check_sum_equals_100(self) -> PhaseTimeBudget:
        """Validate that all percentages sum to 100.0."""
        total = (
            self.phase1_pct + self.phase2_pct + self.phase3_pct + self.finalization_pct
        )
        if abs(total - 100.0) > 1e-6:
            msg = f"Phase percentages must sum to 100.0, got {total}"
            raise ValueError(msg)
        return self


class PipelineConfig(BaseModel):
    """Pipeline hyperparameters and orchestrator configuration (REQ-DM-001).

    All integer hyperparameters default to the paper-specified values:
    M=4, T=4, K=4, L=2, R=5, time_limit=86400s, subsample_limit=30000,
    max_debug_attempts=3.

    Attributes:
        num_retrieved_models: Number of candidate models to retrieve (M).
        outer_loop_steps: Outer loop iterations (T).
        inner_loop_steps: Inner loop iterations per code block (K).
        num_parallel_solutions: Parallel solution paths for ensemble (L).
        ensemble_rounds: Ensemble strategy exploration rounds (R).
        time_limit_seconds: Maximum runtime per competition in seconds.
        subsample_limit: Max training samples during refinement.
        max_debug_attempts: Max debugging retries before fallback.
        permission_mode: SDK permission mode (REQ-OR-009).
        model: Claude model identifier (REQ-OR-044).
        log_level: Logging level string (REQ-OR-047).
        log_file: Optional log file path (REQ-OR-047).
        phase_time_budget: Optional proportional time budget (REQ-OR-025).
    """

    model_config = ConfigDict(frozen=True)

    # Paper hyperparameters
    num_retrieved_models: int = 4
    outer_loop_steps: int = 4
    inner_loop_steps: int = 4
    num_parallel_solutions: int = 2
    ensemble_rounds: int = 5
    time_limit_seconds: int = 86400
    subsample_limit: int = 30000
    max_debug_attempts: int = 3

    # Orchestrator fields
    permission_mode: str = "dangerously-skip-permissions"
    model: str = "opus"
    log_level: str = "INFO"
    log_file: str | None = None
    phase_time_budget: PhaseTimeBudget | None = None
    ablation_timeout_cap: int = 600

    # Validation configuration
    validation_seed_runs: int = 3
    validation_score_tolerance: float = 0.1
    validation_overfit_threshold: float = 0.15

    @field_validator(
        "num_retrieved_models",
        "outer_loop_steps",
        "inner_loop_steps",
        "num_parallel_solutions",
        "ensemble_rounds",
        "time_limit_seconds",
        "subsample_limit",
        "max_debug_attempts",
    )
    @classmethod
    def _must_be_positive(cls, v: int) -> int:
        """Validate that all integer hyperparameters are >= 1 (REQ-DM-002)."""
        if v < 1:
            msg = "Value must be >= 1"
            raise ValueError(msg)
        return v


class TaskDescription(BaseModel):
    """Description of a Kaggle competition task (REQ-DM-007).

    Captures all metadata needed to define a competition: task type,
    data modality, evaluation metric, and dataset/output paths.

    Attributes:
        competition_id: Unique competition identifier.
        task_type: Type of ML task.
        data_modality: Primary data modality.
        evaluation_metric: Name of the evaluation metric.
        metric_direction: Whether to maximize or minimize the metric.
        description: Full task description text (T_task).
        target_column: Name of the column to predict (optional).
        baseline_value: External baseline score that models must beat (optional).
        data_dir: Path to dataset directory.
        output_dir: Path for submission output.
    """

    model_config = ConfigDict(frozen=True)

    competition_id: str
    task_type: TaskType
    data_modality: DataModality
    evaluation_metric: str
    metric_direction: MetricDirection
    description: str
    target_column: str | None = None
    baseline_value: float | None = None
    data_dir: str = "./input"
    output_dir: str = "./final"


# ---------------------------------------------------------------------------
# Solution & Code Block Models (REQ-DM-008 through REQ-DM-012)
# ---------------------------------------------------------------------------


class SolutionPhase(StrEnum):
    """Pipeline phase that produced a solution script (REQ-DM-008).

    Tracks the provenance of each solution through the five pipeline stages.
    """

    INIT = "init"
    MERGED = "merged"
    REFINED = "refined"
    ENSEMBLE = "ensemble"
    FINAL = "final"


class SolutionScript(BaseModel):
    """Wrapper around a single-file Python solution script (REQ-DM-009).

    The only non-frozen model — mutable so that phase orchestration code
    can update ``score`` after evaluation (REQ-DM-039).

    Attributes:
        content: Full Python source code of the solution.
        phase: Which pipeline phase produced this script.
        score: Evaluation score (None if not yet evaluated).
        is_executable: Whether the script ran without errors.
        source_model: Name of the model used (if from Phase 1).
        created_at: Timestamp of creation (auto-set to UTC now).
    """

    model_config = ConfigDict(frozen=False)

    content: str
    phase: SolutionPhase
    score: float | None = None
    is_executable: bool = True
    source_model: str | None = None
    created_at: datetime = None  # type: ignore[assignment]

    @model_validator(mode="after")
    def _set_created_at_default(self) -> SolutionScript:
        """Auto-set created_at to UTC now when not provided."""
        if self.created_at is None:
            self.created_at = datetime.now(UTC)
        return self

    def replace_block(self, old: str, new: str) -> SolutionScript:
        """Return a new SolutionScript with *old* replaced by *new*.

        If *old* appears exactly once, replaces it. If *old* appears
        multiple times, replaces the last occurrence (most likely to be
        the target block in a solution script where imports/helpers
        appear early and model code appears later).

        Args:
            old: Substring to find in content.
            new: Replacement text.

        Returns:
            A new SolutionScript with the substitution applied.

        Raises:
            ValueError: If *old* is not found in content.
        """
        if old != "" and old not in self.content:
            msg = f"Block not found in solution content: {old!r}"
            raise ValueError(msg)
        # Replace the last occurrence to target the main code block
        # rather than early imports/helpers that might share patterns.
        idx = self.content.rfind(old)
        if idx == -1 or old == "":
            replaced = self.content.replace(old, new, 1)
        else:
            replaced = self.content[:idx] + new + self.content[idx + len(old) :]
        return SolutionScript(
            content=replaced,
            phase=self.phase,
            score=self.score,
            is_executable=self.is_executable,
            source_model=self.source_model,
        )


class CodeBlockCategory(StrEnum):
    """Semantic category of a code block extracted from a solution (REQ-DM-011).

    Eight categories matching the types of code regions identified by
    the extractor agent during targeted refinement.
    """

    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_SELECTION = "model_selection"
    TRAINING = "training"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    ENSEMBLE = "ensemble"
    POSTPROCESSING = "postprocessing"
    OTHER = "other"


class CodeBlock(BaseModel):
    """An exact code block substring extracted from a solution script (REQ-DM-012).

    Represents one targeted region of code identified by the extractor agent
    for refinement during Phase 2.

    Attributes:
        content: Exact code block text extracted from the solution.
        category: Semantic category of the code block.
        outer_step: Outer loop step t at which this block was extracted.
    """

    model_config = ConfigDict(frozen=True)

    content: str
    category: CodeBlockCategory | None = None
    outer_step: int | None = None


# ---------------------------------------------------------------------------
# Agent Identity (REQ-DM-013)
# ---------------------------------------------------------------------------


class AgentType(StrEnum):
    """Identifier for each MLE-STAR agent (REQ-DM-013).

    One value per agent from A_retriever through A_test plus A_validator,
    matching the paper's agent naming convention (Section 6).
    """

    BASELINE = "baseline"
    RESEARCHER = "researcher"
    RETRIEVER = "retriever"
    INIT = "init"
    MERGER = "merger"
    ABLATION = "ablation"
    SUMMARIZE = "summarize"
    EXTRACTOR = "extractor"
    CODER = "coder"
    PLANNER = "planner"
    ENS_PLANNER = "ens_planner"
    ENSEMBLER = "ensembler"
    DEBUGGER = "debugger"
    LEAKAGE = "leakage"
    DATA = "data"
    TEST = "test"
    VALIDATOR = "validator"


# ---------------------------------------------------------------------------
# Structured Output Schemas (REQ-DM-014 through REQ-DM-020)
# ---------------------------------------------------------------------------


class RetrievedModel(BaseModel):
    """A single ML model retrieved by the retriever agent (REQ-DM-014).

    Attributes:
        model_name: Name of the retrieved ML model.
        example_code: Concise example code for the model.
    """

    model_config = ConfigDict(frozen=True)

    model_name: str
    example_code: str


class RetrieverOutput(BaseModel):
    """Structured output from the retriever agent (REQ-DM-015).

    Contains at least one retrieved model with its example code.

    Attributes:
        models: Non-empty list of retrieved ML models.
    """

    model_config = ConfigDict(frozen=True)

    models: list[RetrievedModel]

    @field_validator("models")
    @classmethod
    def _models_must_be_nonempty(
        cls,
        v: list[RetrievedModel],
    ) -> list[RetrievedModel]:
        """Validate that models list contains at least one entry."""
        if len(v) < 1:
            msg = "models list must contain at least 1 entry"
            raise ValueError(msg)
        return v


class RefinePlan(BaseModel):
    """A targeted refinement plan for a code block (REQ-DM-016).

    Attributes:
        code_block: Exact code block extracted from the solution script.
        plan: Natural language refinement plan (3-5 sentences).
    """

    model_config = ConfigDict(frozen=True)

    code_block: str
    plan: str


class ExtractorOutput(BaseModel):
    """Structured output from the extractor agent (REQ-DM-017).

    Contains at least one refinement plan targeting a code block.

    Attributes:
        plans: Non-empty list of refinement plans.
    """

    model_config = ConfigDict(frozen=True)

    plans: list[RefinePlan]

    @field_validator("plans")
    @classmethod
    def _plans_must_be_nonempty(cls, v: list[RefinePlan]) -> list[RefinePlan]:
        """Validate that plans list contains at least one entry."""
        if len(v) < 1:
            msg = "plans list must contain at least 1 entry"
            raise ValueError(msg)
        return v


class LeakageAnswer(BaseModel):
    """Detection result for a single preprocessing code block (REQ-DM-018).

    Attributes:
        leakage_status: Whether data leakage was detected in the code block.
        code_block: The preprocessing code block extracted from the solution.
    """

    model_config = ConfigDict(frozen=True)

    leakage_status: Literal["Yes Data Leakage", "No Data Leakage"]
    code_block: str


class LeakageDetectionOutput(BaseModel):
    """Structured output from the leakage detection agent (REQ-DM-019).

    Contains at least one leakage analysis answer for a preprocessing block.

    Attributes:
        answers: Non-empty list of leakage detection answers.
    """

    model_config = ConfigDict(frozen=True)

    answers: list[LeakageAnswer]

    @field_validator("answers")
    @classmethod
    def _answers_must_be_nonempty(
        cls,
        v: list[LeakageAnswer],
    ) -> list[LeakageAnswer]:
        """Validate that answers list contains at least one entry."""
        if len(v) < 1:
            msg = "answers list must contain at least 1 entry"
            raise ValueError(msg)
        return v


class ResearchFindings(BaseModel):
    """Structured output from the internet research agent.

    Contains model recommendations, feature engineering ideas,
    preprocessing techniques, and other domain-specific insights
    gathered from web research.

    Attributes:
        model_recommendations: Recommended model architectures.
        feature_engineering_ideas: Ideas for feature engineering.
        preprocessing_ideas: Preprocessing pipeline suggestions.
        other_insights: Other domain-specific insights.
        raw_summary: Raw summary text of all findings.
    """

    model_config = ConfigDict(frozen=True)

    model_recommendations: list[str]
    feature_engineering_ideas: list[str]
    preprocessing_ideas: list[str]
    other_insights: list[str]
    raw_summary: str


class DataContaminationResult(BaseModel):
    """Verdict from the data contamination check agent (REQ-DM-020).

    Attributes:
        verdict: Whether the competition data is novel or previously seen.
    """

    model_config = ConfigDict(frozen=True)

    verdict: Literal["Novel", "Same"]


# ---------------------------------------------------------------------------
# Iteration Records (REQ-DM-042, REQ-DM-043, REQ-P2I-036)
# ---------------------------------------------------------------------------


class RefinementAttempt(BaseModel):
    """Record of a single inner-loop refinement attempt (REQ-DM-042).

    Captures the plan, resulting score, code block used, and whether
    the attempt produced a strict improvement over the previous best.

    Attributes:
        plan: Natural language refinement plan that was attempted.
        score: Evaluation score after applying the plan (None if failed).
        code_block: The code block that was refined.
        was_improvement: Whether this attempt strictly improved the score.
    """

    model_config = ConfigDict(frozen=True)

    plan: str
    score: float | None = None
    code_block: str
    was_improvement: bool


class EnsembleAttempt(BaseModel):
    """Record of a single ensemble round attempt (REQ-DM-043).

    Captures the ensemble plan, resulting score, and the produced
    ensemble solution script.

    Attributes:
        plan: Natural language ensemble strategy that was attempted.
        score: Evaluation score of the ensemble solution (None if failed).
        solution: The ensemble solution script produced.
    """

    model_config = ConfigDict(frozen=True)

    plan: str
    score: float | None = None
    solution: SolutionScript


class InnerLoopResult(BaseModel):
    """Result of running the Phase 2 inner loop (REQ-P2I-036).

    Aggregates all refinement attempts for a single code block across
    K iterations. The ``improved`` flag is True iff ``best_score`` is
    strictly better than the input score (uses ``is_improvement``,
    not ``is_improvement_or_equal``).

    Attributes:
        best_solution: The best solution found during the inner loop.
        best_score: Score of the best solution.
        attempts: List of all refinement attempts made.
        improved: True iff best_score is strictly better than input.
    """

    model_config = ConfigDict(frozen=True)

    best_solution: SolutionScript
    best_score: float
    attempts: list[RefinementAttempt]
    improved: bool


# ---------------------------------------------------------------------------
# Prompt Template Model (REQ-DM-030, REQ-DM-031)
# ---------------------------------------------------------------------------


class PromptTemplate(BaseModel):
    """A prompt template for an MLE-STAR agent (REQ-DM-030).

    Stores the template string with ``{variable}`` placeholders and the
    metadata associating it with a specific agent type and paper figure.

    Attributes:
        agent_type: Which agent this template belongs to.
        figure_ref: Paper figure reference (e.g., ``"Figure 9"``).
        template: Template string with ``{variable}`` placeholders.
        variables: Required variable names for this template.
    """

    model_config = ConfigDict(frozen=True)

    agent_type: AgentType
    figure_ref: str
    template: str
    variables: list[str]

    def render(self, **kwargs: object) -> str:
        """Substitute all ``{variable}`` placeholders with provided values (REQ-DM-031).

        Args:
            **kwargs: Keyword arguments mapping variable names to values.

        Returns:
            The rendered prompt string with all placeholders substituted.

        Raises:
            KeyError: If a required variable is not provided.
        """
        return self.template.format(**kwargs)


# ---------------------------------------------------------------------------
# Evaluation & Phase Result Models (REQ-DM-021 through REQ-DM-025)
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    """Result of executing and scoring a solution script (REQ-DM-021).

    Captures everything produced by running a solution: parsed score,
    stdout/stderr, exit code, timing, and any error traceback.

    Attributes:
        score: Parsed validation score (None if parsing failed).
        stdout: Full standard output from script execution.
        stderr: Full standard error from script execution.
        exit_code: Process exit code (0 = success).
        duration_seconds: Wall-clock execution time in seconds.
        is_error: Whether execution produced an error.
        error_traceback: Python traceback if error occurred.
    """

    model_config = ConfigDict(frozen=True)

    score: float | None = None
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    is_error: bool
    error_traceback: str | None = None


# ---------------------------------------------------------------------------
# Validation Models
# ---------------------------------------------------------------------------


class ValidationStatus(StrEnum):
    """Outcome status for a single validation check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ValidationCheck(BaseModel):
    """Result of a single validation check (e.g., reproducibility, sanity).

    Attributes:
        name: Check identifier (e.g., ``"reproducibility"``).
        status: Whether the check passed, failed, or was skipped.
        details: Human-readable explanation of the outcome.
        scores: For reproducibility checks, the list of scores across runs.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    status: ValidationStatus
    details: str
    scores: list[float] | None = None


class ValidationResult(BaseModel):
    """Aggregated result of all validation checks on a solution.

    Attributes:
        solution: The solution that was validated.
        checks: Individual check results.
        passed: True if ALL checks passed.
        baseline_beaten: True if the solution score beats the external baseline.
    """

    model_config = ConfigDict(frozen=True)

    solution: SolutionScript
    checks: list[ValidationCheck]
    passed: bool
    baseline_beaten: bool


class Phase1Result(BaseModel):
    """Output of Phase 1: model retrieval and initial solution (REQ-DM-022).

    Contains retrieved models, candidate solutions produced by A_init,
    their scores, and the final merged solution s_0.

    Attributes:
        retrieved_models: Models retrieved by A_retriever.
        candidate_solutions: Scripts produced by A_init for each model.
        candidate_scores: Scores for each candidate.
        initial_solution: Final merged solution s_0.
        initial_score: Best score after merging (h_best).
    """

    model_config = ConfigDict(frozen=True)

    retrieved_models: list[RetrievedModel]
    candidate_solutions: list[SolutionScript]
    candidate_scores: list[float | None]
    initial_solution: SolutionScript
    initial_score: float
    baseline_score: float | None = None
    baseline_solution: SolutionScript | None = None
    research_findings: ResearchFindings | None = None


class Phase2Result(BaseModel):
    """Output of Phase 2: targeted refinement loop (REQ-DM-023).

    Contains ablation summaries, refined code blocks, the best solution
    found during refinement, and per-step history.

    Attributes:
        ablation_summaries: T_abl summaries collected across outer steps.
        refined_blocks: Code blocks c_t refined in each outer step.
        best_solution: Best solution found during refinement (s_final).
        best_score: Score of best solution (h_best).
        step_history: Per-step records: plans tried, scores achieved.
    """

    model_config = ConfigDict(frozen=True)

    ablation_summaries: list[str]
    refined_blocks: list[CodeBlock]
    best_solution: SolutionScript
    best_score: float
    step_history: list[dict[str, Any]]


class Phase3Result(BaseModel):
    """Output of Phase 3: ensemble construction (REQ-DM-024).

    Contains input solutions, ensemble plans proposed across rounds,
    their scores, and the best ensemble solution.

    Attributes:
        input_solutions: L solutions fed into ensemble.
        ensemble_plans: Ensemble plans e_r proposed across rounds.
        ensemble_scores: Scores for each ensemble attempt.
        best_ensemble: Best ensemble solution (s_ens*).
        best_ensemble_score: Score of best ensemble.
    """

    model_config = ConfigDict(frozen=True)

    input_solutions: list[SolutionScript]
    ensemble_plans: list[str]
    ensemble_scores: list[float | None]
    best_ensemble: SolutionScript
    best_ensemble_score: float


class FinalResult(BaseModel):
    """Complete pipeline result aggregating all phases (REQ-DM-025).

    Top-level container returned by the orchestrator after all phases
    complete. Links task, config, phase outputs, and the final submission.

    Attributes:
        task: The task that was solved.
        config: Configuration used.
        phase1: Phase 1 output.
        phase2_results: One Phase2Result per parallel solution path.
        phase3: Phase 3 output (None if L=1, no ensemble).
        final_solution: The solution submitted for test evaluation.
        submission_path: Path to submission file.
        total_duration_seconds: Total pipeline wall-clock time.
        validation_results: Validation results collected across all phases.
    """

    model_config = ConfigDict(frozen=True)

    task: TaskDescription
    config: PipelineConfig
    phase1: Phase1Result
    phase2_results: list[Phase2Result]
    phase3: Phase3Result | None = None
    final_solution: SolutionScript
    submission_path: str
    total_duration_seconds: float
    validation_results: list[ValidationResult] = []


# ---------------------------------------------------------------------------
# Agent Config & SDK Integration Types (REQ-DM-036 through REQ-DM-040)
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    """SDK configuration for a single MLE-STAR agent (REQ-DM-036).

    Maps an ``AgentType`` to the parameters needed to register it with the
    Claude Agent SDK as an ``AgentDefinition``.

    Attributes:
        agent_type: MLE-STAR agent identity.
        description: Value for ``AgentDefinition.description``.
        system_prompt: Custom system prompt (None = use template).
        tools: Allowed SDK tools for this agent.
        model: SDK model selection (``"sonnet"``, ``"opus"``, etc.).
        output_schema: Pydantic model class for structured output.
        max_turns: Maximum agent turns.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    agent_type: AgentType
    description: str
    system_prompt: str | None = None
    tools: list[str] | None = None
    model: str | None = None
    output_schema: type[BaseModel] | None = None
    max_turns: int | None = None

    def to_cli_flags(self) -> list[str]:
        """Return CLI flags for this agent's configuration.

        Produces flags for ``--model``, ``--allowedTools``, and
        ``--max-turns`` based on the agent's configuration.

        Returns:
            List of CLI flag strings.
        """
        flags: list[str] = []
        if self.model:
            flags.extend(["--model", self.model])
        if self.tools:
            flags.extend(["--allowedTools", ",".join(self.tools)])
        if self.max_turns is not None:
            flags.extend(["--max-turns", str(self.max_turns)])
        return flags

    def to_output_flags(self) -> list[str]:
        """Return CLI flags for structured JSON output.

        Produces ``["--output-format", "json", "--json-schema", ...]``
        when ``output_schema`` is set, otherwise returns an empty list.

        Returns:
            List of CLI flag strings for structured output, or empty list.
        """
        if self.output_schema is None:
            return []
        import json

        schema = self.output_schema.model_json_schema()
        return ["--output-format", "json", "--json-schema", json.dumps(schema)]

    def to_agent_definition(self) -> dict[str, Any]:
        """Return a dict suitable for ``ClaudeAgentOptions.agents`` (REQ-DM-037).

        .. deprecated::
            Use ``to_cli_flags()`` and ``to_output_flags()`` instead.

        Returns:
            Dict with keys ``description``, ``prompt``, ``tools``, ``model``.
        """
        return {
            "description": self.description,
            "prompt": self.system_prompt or "",
            "tools": self.tools,
            "model": self.model,
        }

    def to_output_format(self) -> dict[str, Any] | None:
        """Return SDK ``output_format`` dict or None (REQ-DM-038).

        .. deprecated::
            Use ``to_output_flags()`` instead.

        Returns:
            ``{"type": "json_schema", "schema": ...}`` when ``output_schema``
            is set, otherwise ``None``.
        """
        if self.output_schema is None:
            return None
        return {
            "type": "json_schema",
            "schema": self.output_schema.model_json_schema(),
        }


# Tools shared by execution-capable agents (REQ-OR-008).
_EXECUTION_TOOLS: list[str] = ["Bash", "Edit", "Write", "Read"]
# Tools for read-only / analysis-only agents (REQ-OR-008).
_READ_ONLY_TOOLS: list[str] = ["Read"]
# Tools for research/planning agents that read existing notes and write new ones.
_READ_WRITE_TOOLS: list[str] = ["Read", "Write"]


def build_default_agent_configs() -> dict[AgentType, AgentConfig]:
    """Return pre-configured ``AgentConfig`` instances for all 14 agents (REQ-DM-040).

    Tool assignments follow REQ-OR-008. Output schemas are set per
    REQ-OR-006 and REQ-SF-025.

    All agents use the pipeline default model (opus) unless explicitly
    overridden per-agent. No per-agent model overrides are set here.

    Returns:
        A fresh dict mapping every ``AgentType`` to its default config.
    """
    return {
        AgentType.BASELINE: AgentConfig(
            agent_type=AgentType.BASELINE,
            description="Generates a simple baseline solution with default settings.",
            tools=_EXECUTION_TOOLS,
        ),
        AgentType.RESEARCHER: AgentConfig(
            agent_type=AgentType.RESEARCHER,
            description="Conducts internet research for model architectures and techniques.",
            tools=["WebSearch", "WebFetch", "Write"],
            output_schema=ResearchFindings,
        ),
        AgentType.RETRIEVER: AgentConfig(
            agent_type=AgentType.RETRIEVER,
            description="Retrieves ML models and example code from the web.",
            tools=["WebSearch", "WebFetch", "Write"],
        ),
        AgentType.INIT: AgentConfig(
            agent_type=AgentType.INIT,
            description="Generates initial solution scripts from retrieved models.",
            tools=_EXECUTION_TOOLS,
        ),
        AgentType.MERGER: AgentConfig(
            agent_type=AgentType.MERGER,
            description="Merges candidate solutions into a single best script.",
            tools=_EXECUTION_TOOLS,
        ),
        AgentType.ABLATION: AgentConfig(
            agent_type=AgentType.ABLATION,
            description="Performs ablation analysis on solution components.",
            tools=_EXECUTION_TOOLS,
        ),
        AgentType.SUMMARIZE: AgentConfig(
            agent_type=AgentType.SUMMARIZE,
            description="Summarizes ablation results into concise analysis.",
            tools=_READ_WRITE_TOOLS,
        ),
        AgentType.EXTRACTOR: AgentConfig(
            agent_type=AgentType.EXTRACTOR,
            description="Extracts targeted code blocks with refinement plans.",
            tools=_READ_WRITE_TOOLS,
            output_schema=ExtractorOutput,
        ),
        AgentType.PLANNER: AgentConfig(
            agent_type=AgentType.PLANNER,
            description="Creates refinement plans for targeted code blocks.",
            tools=_READ_WRITE_TOOLS,
        ),
        AgentType.CODER: AgentConfig(
            agent_type=AgentType.CODER,
            description="Implements code refinements based on plans.",
            tools=_EXECUTION_TOOLS,
        ),
        AgentType.ENS_PLANNER: AgentConfig(
            agent_type=AgentType.ENS_PLANNER,
            description="Plans ensemble strategies from multiple solutions.",
            tools=_READ_WRITE_TOOLS,
        ),
        AgentType.ENSEMBLER: AgentConfig(
            agent_type=AgentType.ENSEMBLER,
            description="Implements ensemble strategies combining solutions.",
            tools=_EXECUTION_TOOLS,
        ),
        AgentType.DEBUGGER: AgentConfig(
            agent_type=AgentType.DEBUGGER,
            description="Debugs failing solution scripts using error tracebacks.",
            tools=_EXECUTION_TOOLS,
        ),
        AgentType.LEAKAGE: AgentConfig(
            agent_type=AgentType.LEAKAGE,
            description="Detects and corrects data leakage in solutions.",
            tools=_READ_ONLY_TOOLS,
            output_schema=LeakageDetectionOutput,
        ),
        AgentType.DATA: AgentConfig(
            agent_type=AgentType.DATA,
            description="Analyzes dataset characteristics and contamination.",
            tools=_READ_ONLY_TOOLS,
        ),
        AgentType.TEST: AgentConfig(
            agent_type=AgentType.TEST,
            description="Generates test submissions and validates outputs.",
            tools=_READ_ONLY_TOOLS,
        ),
        AgentType.VALIDATOR: AgentConfig(
            agent_type=AgentType.VALIDATOR,
            description="Validates solutions for reproducibility, sanity, leakage, and overfitting.",
            tools=_EXECUTION_TOOLS,
        ),
    }
