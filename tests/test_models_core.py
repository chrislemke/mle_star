"""Tests for MLE-STAR core configuration data models (Task 03).

Validates TaskType, DataModality, MetricDirection enums, PhaseTimeBudget,
PipelineConfig, and TaskDescription Pydantic models defined in
``src/mle_star/models.py``.  These tests are written TDD-first -- the
implementation does not yet exist.  They serve as the executable
specification for REQ-DM-001 through REQ-DM-007 (plus orchestrator fields
from REQ-OR-009, REQ-OR-025, REQ-OR-028, REQ-OR-044, REQ-OR-047).

Refs:
    SRS 01a (Data Models Core), IMPLEMENTATION_PLAN.md Task 03.
"""

from __future__ import annotations

from enum import StrEnum
import json
from typing import Any

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    DataModality,
    MetricDirection,
    PhaseTimeBudget,
    PipelineConfig,
    TaskDescription,
    TaskType,
)
from pydantic import ValidationError
import pytest

# ---------------------------------------------------------------------------
# Constants -- canonical values from the spec
# ---------------------------------------------------------------------------

TASK_TYPE_VALUES: list[str] = [
    "classification",
    "regression",
    "image_classification",
    "image_to_image",
    "text_classification",
    "audio_classification",
    "sequence_to_sequence",
    "tabular",
]

DATA_MODALITY_VALUES: list[str] = [
    "tabular",
    "image",
    "text",
    "audio",
    "mixed",
]

METRIC_DIRECTION_VALUES: list[str] = [
    "maximize",
    "minimize",
]

PIPELINE_CONFIG_DEFAULTS: dict[str, Any] = {
    "num_retrieved_models": 4,
    "outer_loop_steps": 4,
    "inner_loop_steps": 4,
    "num_parallel_solutions": 2,
    "ensemble_rounds": 5,
    "time_limit_seconds": 86400,
    "subsample_limit": 30000,
    "max_debug_attempts": 3,
}

# Integer fields on PipelineConfig that must be >= 1
PIPELINE_CONFIG_INT_FIELDS: list[str] = [
    "num_retrieved_models",
    "outer_loop_steps",
    "inner_loop_steps",
    "num_parallel_solutions",
    "ensemble_rounds",
    "time_limit_seconds",
    "subsample_limit",
    "max_debug_attempts",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task_description(**overrides: Any) -> TaskDescription:
    """Build a valid TaskDescription with sensible defaults.

    All required fields are populated; any keyword argument overrides
    the corresponding default.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed TaskDescription instance.
    """
    defaults: dict[str, Any] = {
        "competition_id": "spaceship-titanic",
        "task_type": TaskType.CLASSIFICATION,
        "data_modality": DataModality.TABULAR,
        "evaluation_metric": "accuracy",
        "metric_direction": MetricDirection.MAXIMIZE,
        "description": "Predict which passengers were transported.",
    }
    defaults.update(overrides)
    return TaskDescription(**defaults)


# ===========================================================================
# REQ-DM-004: TaskType Enum
# ===========================================================================


@pytest.mark.unit
class TestTaskType:
    """TaskType must be a StrEnum with exactly 8 values (REQ-DM-004)."""

    def test_is_string_enum(self) -> None:
        """TaskType inherits from StrEnum."""
        assert issubclass(TaskType, StrEnum)

    def test_member_count_is_eight(self) -> None:
        """Enum has exactly 8 members."""
        assert len(TaskType) == 8

    @pytest.mark.parametrize("value", TASK_TYPE_VALUES)
    def test_contains_expected_value(self, value: str) -> None:
        """Each spec-mandated value is present in the enum."""
        member = TaskType(value)
        assert member.value == value

    def test_values_match_spec_exactly(self) -> None:
        """The full set of values matches the spec with no extras."""
        actual = sorted(m.value for m in TaskType)
        expected = sorted(TASK_TYPE_VALUES)
        assert actual == expected

    @pytest.mark.parametrize("value", TASK_TYPE_VALUES)
    def test_string_equality(self, value: str) -> None:
        """StrEnum members compare equal to their plain string value."""
        assert TaskType(value) == value

    def test_invalid_value_raises(self) -> None:
        """Constructing with a non-existent value raises ValueError."""
        with pytest.raises(ValueError, match="not a valid"):
            TaskType("nonexistent_task_type")


# ===========================================================================
# REQ-DM-005: DataModality Enum
# ===========================================================================


@pytest.mark.unit
class TestDataModality:
    """DataModality must be a StrEnum with exactly 5 values (REQ-DM-005)."""

    def test_is_string_enum(self) -> None:
        """DataModality inherits from StrEnum."""
        assert issubclass(DataModality, StrEnum)

    def test_member_count_is_five(self) -> None:
        """Enum has exactly 5 members."""
        assert len(DataModality) == 5

    @pytest.mark.parametrize("value", DATA_MODALITY_VALUES)
    def test_contains_expected_value(self, value: str) -> None:
        """Each spec-mandated value is present in the enum."""
        member = DataModality(value)
        assert member.value == value

    def test_values_match_spec_exactly(self) -> None:
        """The full set of values matches the spec with no extras."""
        actual = sorted(m.value for m in DataModality)
        expected = sorted(DATA_MODALITY_VALUES)
        assert actual == expected

    @pytest.mark.parametrize("value", DATA_MODALITY_VALUES)
    def test_string_equality(self, value: str) -> None:
        """StrEnum members compare equal to their plain string value."""
        assert DataModality(value) == value

    def test_invalid_value_raises(self) -> None:
        """Constructing with a non-existent value raises ValueError."""
        with pytest.raises(ValueError, match="not a valid"):
            DataModality("nonexistent_modality")


# ===========================================================================
# REQ-DM-006: MetricDirection Enum
# ===========================================================================


@pytest.mark.unit
class TestMetricDirection:
    """MetricDirection must be a StrEnum with exactly 2 values (REQ-DM-006)."""

    def test_is_string_enum(self) -> None:
        """MetricDirection inherits from StrEnum."""
        assert issubclass(MetricDirection, StrEnum)

    def test_member_count_is_two(self) -> None:
        """Enum has exactly 2 members."""
        assert len(MetricDirection) == 2

    @pytest.mark.parametrize("value", METRIC_DIRECTION_VALUES)
    def test_contains_expected_value(self, value: str) -> None:
        """Each spec-mandated value is present in the enum."""
        member = MetricDirection(value)
        assert member.value == value

    def test_values_match_spec_exactly(self) -> None:
        """The full set of values matches the spec with no extras."""
        actual = sorted(m.value for m in MetricDirection)
        expected = sorted(METRIC_DIRECTION_VALUES)
        assert actual == expected

    @pytest.mark.parametrize("value", METRIC_DIRECTION_VALUES)
    def test_string_equality(self, value: str) -> None:
        """StrEnum members compare equal to their plain string value."""
        assert MetricDirection(value) == value

    def test_invalid_value_raises(self) -> None:
        """Constructing with a non-existent value raises ValueError."""
        with pytest.raises(ValueError, match="not a valid"):
            MetricDirection("neutral")


# ===========================================================================
# PhaseTimeBudget Model
# ===========================================================================


@pytest.mark.unit
class TestPhaseTimeBudget:
    """PhaseTimeBudget is a frozen Pydantic model whose percentages sum to 100."""

    # -- Default construction --

    def test_default_construction_succeeds(self) -> None:
        """Creating PhaseTimeBudget() with no args uses paper defaults."""
        budget = PhaseTimeBudget()
        assert budget.phase1_pct == 10.0
        assert budget.phase2_pct == 65.0
        assert budget.phase3_pct == 15.0
        assert budget.finalization_pct == 10.0

    def test_default_percentages_sum_to_100(self) -> None:
        """Default field values sum to exactly 100.0."""
        budget = PhaseTimeBudget()
        total = (
            budget.phase1_pct
            + budget.phase2_pct
            + budget.phase3_pct
            + budget.finalization_pct
        )
        assert total == pytest.approx(100.0)

    # -- Custom valid values --

    def test_custom_values_summing_to_100_accepted(self) -> None:
        """Custom percentages that sum to 100.0 are accepted."""
        budget = PhaseTimeBudget(
            phase1_pct=25.0,
            phase2_pct=25.0,
            phase3_pct=25.0,
            finalization_pct=25.0,
        )
        assert budget.phase1_pct == 25.0
        assert budget.phase2_pct == 25.0
        assert budget.phase3_pct == 25.0
        assert budget.finalization_pct == 25.0

    def test_asymmetric_values_summing_to_100(self) -> None:
        """Asymmetric custom percentages totalling 100.0 are valid."""
        budget = PhaseTimeBudget(
            phase1_pct=5.0,
            phase2_pct=80.0,
            phase3_pct=10.0,
            finalization_pct=5.0,
        )
        total = (
            budget.phase1_pct
            + budget.phase2_pct
            + budget.phase3_pct
            + budget.finalization_pct
        )
        assert total == pytest.approx(100.0)

    # -- Validation: sum != 100 --

    def test_sum_over_100_raises_validation_error(self) -> None:
        """Percentages summing to more than 100 raise ValidationError."""
        with pytest.raises(ValidationError):
            PhaseTimeBudget(
                phase1_pct=30.0,
                phase2_pct=30.0,
                phase3_pct=30.0,
                finalization_pct=30.0,
            )

    def test_sum_under_100_raises_validation_error(self) -> None:
        """Percentages summing to less than 100 raise ValidationError."""
        with pytest.raises(ValidationError):
            PhaseTimeBudget(
                phase1_pct=10.0,
                phase2_pct=10.0,
                phase3_pct=10.0,
                finalization_pct=10.0,
            )

    def test_sum_slightly_off_raises_validation_error(self) -> None:
        """Even a small deviation from 100.0 is rejected."""
        with pytest.raises(ValidationError):
            PhaseTimeBudget(
                phase1_pct=10.0,
                phase2_pct=65.0,
                phase3_pct=15.0,
                finalization_pct=10.1,
            )

    # -- Frozen (immutable) --

    def test_frozen_prevents_field_mutation(self) -> None:
        """PhaseTimeBudget is frozen; assignment raises an error."""
        budget = PhaseTimeBudget()
        with pytest.raises(ValidationError):
            budget.phase1_pct = 50.0  # type: ignore[misc]

    # -- Property-based: any four floats summing to 100 should be accepted --

    @given(
        a=st.floats(min_value=0.1, max_value=99.0, allow_nan=False),
        b=st.floats(min_value=0.1, max_value=99.0, allow_nan=False),
        c=st.floats(min_value=0.1, max_value=99.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_any_four_floats_summing_to_100_accepted(
        self, a: float, b: float, c: float
    ) -> None:
        """Property: if four positive floats sum to 100.0, construction succeeds."""
        d = 100.0 - a - b - c
        if d <= 0.0:
            return  # skip invalid combos where d is non-positive
        budget = PhaseTimeBudget(
            phase1_pct=a,
            phase2_pct=b,
            phase3_pct=c,
            finalization_pct=d,
        )
        total = (
            budget.phase1_pct
            + budget.phase2_pct
            + budget.phase3_pct
            + budget.finalization_pct
        )
        assert total == pytest.approx(100.0, abs=1e-6)


# ===========================================================================
# REQ-DM-001, REQ-DM-002, REQ-DM-003: PipelineConfig Model
# ===========================================================================


@pytest.mark.unit
class TestPipelineConfigDefaults:
    """PipelineConfig() with no args produces paper defaults (REQ-DM-001)."""

    def test_default_num_retrieved_models(self) -> None:
        """M defaults to 4."""
        cfg = PipelineConfig()
        assert cfg.num_retrieved_models == 4

    def test_default_outer_loop_steps(self) -> None:
        """T defaults to 4."""
        cfg = PipelineConfig()
        assert cfg.outer_loop_steps == 4

    def test_default_inner_loop_steps(self) -> None:
        """K defaults to 4."""
        cfg = PipelineConfig()
        assert cfg.inner_loop_steps == 4

    def test_default_num_parallel_solutions(self) -> None:
        """L defaults to 2."""
        cfg = PipelineConfig()
        assert cfg.num_parallel_solutions == 2

    def test_default_ensemble_rounds(self) -> None:
        """R defaults to 5."""
        cfg = PipelineConfig()
        assert cfg.ensemble_rounds == 5

    def test_default_time_limit_seconds(self) -> None:
        """Time limit defaults to 86400 (24 hours)."""
        cfg = PipelineConfig()
        assert cfg.time_limit_seconds == 86400

    def test_default_subsample_limit(self) -> None:
        """Subsample limit defaults to 30000."""
        cfg = PipelineConfig()
        assert cfg.subsample_limit == 30000

    def test_default_max_debug_attempts(self) -> None:
        """Max debug attempts defaults to 3."""
        cfg = PipelineConfig()
        assert cfg.max_debug_attempts == 3

    @pytest.mark.parametrize(
        "field,expected",
        list(PIPELINE_CONFIG_DEFAULTS.items()),
        ids=list(PIPELINE_CONFIG_DEFAULTS.keys()),
    )
    def test_all_defaults_via_parametrize(self, field: str, expected: int) -> None:
        """Parametrized check of every paper-default field."""
        cfg = PipelineConfig()
        assert getattr(cfg, field) == expected


@pytest.mark.unit
class TestPipelineConfigOrchestratorFields:
    """PipelineConfig includes orchestrator-level fields from Spec 09."""

    def test_permission_mode_default(self) -> None:
        """permission_mode defaults to 'dangerously-skip-permissions' (REQ-OR-009)."""
        cfg = PipelineConfig()
        assert cfg.permission_mode == "dangerously-skip-permissions"

    def test_model_default(self) -> None:
        """Model defaults to 'opus' (REQ-OR-044)."""
        cfg = PipelineConfig()
        assert cfg.model == "opus"

    def test_log_level_default(self) -> None:
        """log_level defaults to 'INFO' (REQ-OR-047)."""
        cfg = PipelineConfig()
        assert cfg.log_level == "INFO"

    def test_log_file_default_is_none(self) -> None:
        """log_file defaults to None (REQ-OR-047)."""
        cfg = PipelineConfig()
        assert cfg.log_file is None

    def test_log_file_accepts_string(self) -> None:
        """log_file accepts a string path."""
        cfg = PipelineConfig(log_file="/tmp/mle_star.log")
        assert cfg.log_file == "/tmp/mle_star.log"

    def test_phase_time_budget_default_is_none(self) -> None:
        """phase_time_budget defaults to None (REQ-OR-025)."""
        cfg = PipelineConfig()
        assert cfg.phase_time_budget is None

    def test_phase_time_budget_accepts_instance(self) -> None:
        """phase_time_budget accepts a PhaseTimeBudget instance."""
        budget = PhaseTimeBudget()
        cfg = PipelineConfig(phase_time_budget=budget)
        assert cfg.phase_time_budget is not None
        assert cfg.phase_time_budget.phase1_pct == 10.0


@pytest.mark.unit
class TestPipelineConfigValidation:
    """PipelineConfig validates all integer fields >= 1 (REQ-DM-002)."""

    @pytest.mark.parametrize("field", PIPELINE_CONFIG_INT_FIELDS)
    def test_zero_raises_validation_error(self, field: str) -> None:
        """Setting any integer hyperparameter to 0 raises ValidationError."""
        with pytest.raises(ValidationError):
            PipelineConfig(**{field: 0})  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", PIPELINE_CONFIG_INT_FIELDS)
    def test_negative_raises_validation_error(self, field: str) -> None:
        """Setting any integer hyperparameter to a negative value raises ValidationError."""
        with pytest.raises(ValidationError):
            PipelineConfig(**{field: -1})  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", PIPELINE_CONFIG_INT_FIELDS)
    def test_one_is_accepted(self, field: str) -> None:
        """Minimum valid value (1) is accepted for every integer field."""
        cfg = PipelineConfig(**{field: 1})  # type: ignore[arg-type]
        assert getattr(cfg, field) == 1

    def test_num_retrieved_models_zero_raises(self) -> None:
        """Explicit acceptance criterion: PipelineConfig(num_retrieved_models=0) raises."""
        with pytest.raises(ValidationError):
            PipelineConfig(num_retrieved_models=0)

    # -- Property-based: any positive int should be accepted --

    @given(value=st.integers(min_value=1, max_value=100_000))
    @settings(max_examples=50)
    def test_any_positive_int_accepted_for_num_retrieved_models(
        self, value: int
    ) -> None:
        """Property: any positive integer is valid for num_retrieved_models."""
        cfg = PipelineConfig(num_retrieved_models=value)
        assert cfg.num_retrieved_models == value

    @given(value=st.integers(min_value=1, max_value=100_000))
    @settings(max_examples=50)
    def test_any_positive_int_accepted_for_outer_loop_steps(self, value: int) -> None:
        """Property: any positive integer is valid for outer_loop_steps."""
        cfg = PipelineConfig(outer_loop_steps=value)
        assert cfg.outer_loop_steps == value


@pytest.mark.unit
class TestPipelineConfigSerialization:
    """PipelineConfig supports round-trip JSON serialization (REQ-DM-003)."""

    def test_round_trip_default_config(self) -> None:
        """Serialize and deserialize default config; all fields preserved."""
        original = PipelineConfig()
        json_str = original.model_dump_json()
        restored = PipelineConfig.model_validate_json(json_str)

        for field_name in PIPELINE_CONFIG_DEFAULTS:
            assert getattr(restored, field_name) == getattr(original, field_name)

    def test_round_trip_custom_config(self) -> None:
        """Serialize and deserialize a fully customized config."""
        original = PipelineConfig(
            num_retrieved_models=10,
            outer_loop_steps=8,
            inner_loop_steps=6,
            num_parallel_solutions=4,
            ensemble_rounds=12,
            time_limit_seconds=3600,
            subsample_limit=5000,
            max_debug_attempts=5,
            permission_mode="askUser",
            model="opus",
            log_level="DEBUG",
            log_file="/var/log/mle.log",
            phase_time_budget=PhaseTimeBudget(
                phase1_pct=20.0,
                phase2_pct=50.0,
                phase3_pct=20.0,
                finalization_pct=10.0,
            ),
        )
        json_str = original.model_dump_json()
        restored = PipelineConfig.model_validate_json(json_str)

        assert restored.num_retrieved_models == 10
        assert restored.outer_loop_steps == 8
        assert restored.inner_loop_steps == 6
        assert restored.num_parallel_solutions == 4
        assert restored.ensemble_rounds == 12
        assert restored.time_limit_seconds == 3600
        assert restored.subsample_limit == 5000
        assert restored.max_debug_attempts == 5
        assert restored.permission_mode == "askUser"
        assert restored.model == "opus"
        assert restored.log_level == "DEBUG"
        assert restored.log_file == "/var/log/mle.log"
        assert restored.phase_time_budget is not None
        assert restored.phase_time_budget.phase1_pct == 20.0

    def test_round_trip_with_none_optionals(self) -> None:
        """Round-trip preserves None for optional fields."""
        original = PipelineConfig()
        json_str = original.model_dump_json()
        restored = PipelineConfig.model_validate_json(json_str)

        assert restored.log_file is None
        assert restored.phase_time_budget is None

    def test_model_dump_json_returns_valid_json(self) -> None:
        """model_dump_json() produces parseable JSON."""
        cfg = PipelineConfig()
        parsed = json.loads(cfg.model_dump_json())
        assert isinstance(parsed, dict)
        assert "num_retrieved_models" in parsed

    def test_round_trip_equality(self) -> None:
        """Serialized then deserialized object equals the original."""
        original = PipelineConfig()
        restored = PipelineConfig.model_validate_json(original.model_dump_json())
        assert original == restored


@pytest.mark.unit
class TestPipelineConfigFrozen:
    """PipelineConfig is frozen (immutable) per REQ-DM-039."""

    def test_cannot_mutate_integer_field(self) -> None:
        """Assignment to an integer field raises an error."""
        cfg = PipelineConfig()
        with pytest.raises(ValidationError):
            cfg.num_retrieved_models = 99  # type: ignore[misc]

    def test_cannot_mutate_string_field(self) -> None:
        """Assignment to a string field raises an error."""
        cfg = PipelineConfig()
        with pytest.raises(ValidationError):
            cfg.model = "haiku"  # type: ignore[misc]

    def test_cannot_mutate_optional_field(self) -> None:
        """Assignment to an optional field raises an error."""
        cfg = PipelineConfig()
        with pytest.raises(ValidationError):
            cfg.log_file = "/tmp/x"  # type: ignore[misc]


# ===========================================================================
# REQ-DM-007: TaskDescription Model
# ===========================================================================


@pytest.mark.unit
class TestTaskDescriptionRequiredFields:
    """TaskDescription has required fields that raise ValidationError if missing."""

    def test_valid_construction(self) -> None:
        """Constructing with all required fields succeeds."""
        td = _make_task_description()
        assert td.competition_id == "spaceship-titanic"
        assert td.task_type == TaskType.CLASSIFICATION
        assert td.data_modality == DataModality.TABULAR
        assert td.evaluation_metric == "accuracy"
        assert td.metric_direction == MetricDirection.MAXIMIZE
        assert td.description == "Predict which passengers were transported."

    @pytest.mark.parametrize(
        "missing_field",
        [
            "competition_id",
            "task_type",
            "data_modality",
            "evaluation_metric",
            "metric_direction",
            "description",
        ],
    )
    def test_missing_required_field_raises(self, missing_field: str) -> None:
        """Omitting any required field raises ValidationError."""
        all_fields: dict[str, Any] = {
            "competition_id": "test-comp",
            "task_type": "classification",
            "data_modality": "tabular",
            "evaluation_metric": "accuracy",
            "metric_direction": "maximize",
            "description": "A description.",
        }
        del all_fields[missing_field]
        with pytest.raises(ValidationError):
            TaskDescription(**all_fields)


@pytest.mark.unit
class TestTaskDescriptionDefaults:
    """TaskDescription optional fields have correct defaults."""

    def test_data_dir_default(self) -> None:
        """data_dir defaults to './input'."""
        td = _make_task_description()
        assert td.data_dir == "./input"

    def test_output_dir_default(self) -> None:
        """output_dir defaults to './final'."""
        td = _make_task_description()
        assert td.output_dir == "./final"

    def test_target_column_default_is_none(self) -> None:
        """target_column defaults to None."""
        td = _make_task_description()
        assert td.target_column is None

    def test_data_dir_override(self) -> None:
        """data_dir can be overridden."""
        td = _make_task_description(data_dir="/custom/data")
        assert td.data_dir == "/custom/data"

    def test_output_dir_override(self) -> None:
        """output_dir can be overridden."""
        td = _make_task_description(output_dir="/custom/output")
        assert td.output_dir == "/custom/output"


@pytest.mark.unit
class TestTaskDescriptionEnumFields:
    """TaskDescription correctly validates enum-typed fields."""

    @pytest.mark.parametrize("task_type", TASK_TYPE_VALUES)
    def test_accepts_all_task_types(self, task_type: str) -> None:
        """Every TaskType value is accepted."""
        td = _make_task_description(task_type=task_type)
        assert td.task_type == task_type

    @pytest.mark.parametrize("modality", DATA_MODALITY_VALUES)
    def test_accepts_all_data_modalities(self, modality: str) -> None:
        """Every DataModality value is accepted."""
        td = _make_task_description(data_modality=modality)
        assert td.data_modality == modality

    @pytest.mark.parametrize("direction", METRIC_DIRECTION_VALUES)
    def test_accepts_all_metric_directions(self, direction: str) -> None:
        """Every MetricDirection value is accepted."""
        td = _make_task_description(metric_direction=direction)
        assert td.metric_direction == direction

    def test_invalid_task_type_raises(self) -> None:
        """Invalid task_type string raises ValidationError."""
        with pytest.raises(ValidationError):
            _make_task_description(task_type="bogus_task")

    def test_invalid_data_modality_raises(self) -> None:
        """Invalid data_modality string raises ValidationError."""
        with pytest.raises(ValidationError):
            _make_task_description(data_modality="bogus_modality")

    def test_invalid_metric_direction_raises(self) -> None:
        """Invalid metric_direction string raises ValidationError."""
        with pytest.raises(ValidationError):
            _make_task_description(metric_direction="bogus_direction")


@pytest.mark.unit
class TestTaskDescriptionTargetColumn:
    """TaskDescription.target_column is an optional string field."""

    def test_default_is_none(self) -> None:
        """target_column defaults to None when not provided."""
        td = _make_task_description()
        assert td.target_column is None

    def test_accepts_string(self) -> None:
        """target_column accepts an explicit string value."""
        td = _make_task_description(target_column="Survived")
        assert td.target_column == "Survived"

    def test_frozen_prevents_mutation(self) -> None:
        """target_column cannot be mutated on a frozen model."""
        td = _make_task_description(target_column="Survived")
        with pytest.raises(ValidationError):
            td.target_column = "Other"  # type: ignore[misc]

    def test_json_round_trip_with_value(self) -> None:
        """JSON round-trip preserves target_column string value."""
        original = _make_task_description(target_column="Transported")
        restored = TaskDescription.model_validate_json(original.model_dump_json())
        assert restored.target_column == "Transported"

    def test_json_round_trip_with_none(self) -> None:
        """JSON round-trip preserves target_column=None."""
        original = _make_task_description()
        restored = TaskDescription.model_validate_json(original.model_dump_json())
        assert restored.target_column is None


@pytest.mark.unit
class TestTaskDescriptionFrozen:
    """TaskDescription is frozen (immutable) per REQ-DM-039."""

    def test_cannot_mutate_competition_id(self) -> None:
        """Assignment to competition_id raises an error."""
        td = _make_task_description()
        with pytest.raises(ValidationError):
            td.competition_id = "other-comp"  # type: ignore[misc]

    def test_cannot_mutate_task_type(self) -> None:
        """Assignment to task_type raises an error."""
        td = _make_task_description()
        with pytest.raises(ValidationError):
            td.task_type = TaskType.REGRESSION  # type: ignore[misc]

    def test_cannot_mutate_description(self) -> None:
        """Assignment to description raises an error."""
        td = _make_task_description()
        with pytest.raises(ValidationError):
            td.description = "Changed."  # type: ignore[misc]

    def test_cannot_mutate_data_dir(self) -> None:
        """Assignment to data_dir raises an error."""
        td = _make_task_description()
        with pytest.raises(ValidationError):
            td.data_dir = "/elsewhere"  # type: ignore[misc]


@pytest.mark.unit
class TestTaskDescriptionSerialization:
    """TaskDescription supports JSON round-trip serialization."""

    def test_round_trip_preserves_all_fields(self) -> None:
        """Serialize and deserialize; all fields preserved."""
        original = _make_task_description(
            data_dir="/custom/data", output_dir="/custom/out"
        )
        json_str = original.model_dump_json()
        restored = TaskDescription.model_validate_json(json_str)

        assert restored.competition_id == original.competition_id
        assert restored.task_type == original.task_type
        assert restored.data_modality == original.data_modality
        assert restored.evaluation_metric == original.evaluation_metric
        assert restored.metric_direction == original.metric_direction
        assert restored.description == original.description
        assert restored.data_dir == original.data_dir
        assert restored.output_dir == original.output_dir

    def test_round_trip_equality(self) -> None:
        """Serialized then deserialized object equals the original."""
        original = _make_task_description()
        restored = TaskDescription.model_validate_json(original.model_dump_json())
        assert original == restored


# ===========================================================================
# Property-based: PipelineConfig with Hypothesis
# ===========================================================================


@pytest.mark.unit
class TestPipelineConfigPropertyBased:
    """Property-based tests for PipelineConfig using Hypothesis."""

    @given(
        m=st.integers(min_value=1, max_value=1000),
        t=st.integers(min_value=1, max_value=1000),
        k=st.integers(min_value=1, max_value=1000),
        l_val=st.integers(min_value=1, max_value=1000),
        r=st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=30)
    def test_any_positive_ints_produce_valid_config(
        self, m: int, t: int, k: int, l_val: int, r: int
    ) -> None:
        """Property: any combination of positive integers creates a valid config."""
        cfg = PipelineConfig(
            num_retrieved_models=m,
            outer_loop_steps=t,
            inner_loop_steps=k,
            num_parallel_solutions=l_val,
            ensemble_rounds=r,
        )
        assert cfg.num_retrieved_models == m
        assert cfg.outer_loop_steps == t
        assert cfg.inner_loop_steps == k
        assert cfg.num_parallel_solutions == l_val
        assert cfg.ensemble_rounds == r

    @given(value=st.integers(max_value=0))
    @settings(max_examples=30)
    def test_non_positive_int_always_rejected(self, value: int) -> None:
        """Property: zero or negative integers are always rejected."""
        with pytest.raises(ValidationError):
            PipelineConfig(num_retrieved_models=value)

    @given(
        m=st.integers(min_value=1, max_value=100),
        t=st.integers(min_value=1, max_value=100),
        k=st.integers(min_value=1, max_value=100),
        l_val=st.integers(min_value=1, max_value=100),
        r=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=30)
    def test_round_trip_preserves_all_positive_ints(
        self, m: int, t: int, k: int, l_val: int, r: int
    ) -> None:
        """Property: JSON round-trip preserves any valid configuration."""
        original = PipelineConfig(
            num_retrieved_models=m,
            outer_loop_steps=t,
            inner_loop_steps=k,
            num_parallel_solutions=l_val,
            ensemble_rounds=r,
        )
        restored = PipelineConfig.model_validate_json(original.model_dump_json())
        assert restored == original
