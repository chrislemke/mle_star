"""Core data models for the MLE-STAR pipeline.

Defines shared Pydantic models, enums, and configuration types used across
all pipeline phases. This module is the foundational type system referenced
by all other modules (execution, safety, phases, orchestrator).

Refs:
    SRS 01a — Data Models Core (REQ-DM-001 through REQ-DM-012).
    IMPLEMENTATION_PLAN.md Tasks 03, 04.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

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
        max_budget_usd: Optional cost cap in USD (REQ-OR-028).
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
    max_budget_usd: float | None = None
    permission_mode: str = "bypassPermissions"
    model: str = "sonnet"
    log_level: str = "INFO"
    log_file: str | None = None
    phase_time_budget: PhaseTimeBudget | None = None

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
        """Return a new SolutionScript with the first occurrence of *old* replaced by *new*.

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
        replaced = self.content.replace(old, new, 1)
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
