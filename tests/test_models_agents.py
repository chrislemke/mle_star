"""Tests for MLE-STAR agent type enum and structured output schemas (Task 05).

Validates AgentType enum, RetrievedModel, RetrieverOutput, RefinePlan,
ExtractorOutput, LeakageAnswer, LeakageDetectionOutput, and
DataContaminationResult Pydantic models defined in
``src/mle_star/models.py``.  These tests are written TDD-first -- the
implementation does not yet exist.  They serve as the executable
specification for REQ-DM-013 through REQ-DM-020.

Refs:
    SRS 01a (Data Models Agents), IMPLEMENTATION_PLAN.md Task 05.
"""

from __future__ import annotations

from enum import StrEnum
import json
from typing import Any

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    AgentType,
    DataContaminationResult,
    ExtractorOutput,
    LeakageAnswer,
    LeakageDetectionOutput,
    RefinePlan,
    RetrievedModel,
    RetrieverOutput,
)
from pydantic import ValidationError
import pytest

# ---------------------------------------------------------------------------
# Constants -- canonical values from the spec
# ---------------------------------------------------------------------------

AGENT_TYPE_VALUES: list[str] = [
    "baseline",
    "researcher",
    "retriever",
    "init",
    "merger",
    "ablation",
    "summarize",
    "extractor",
    "coder",
    "planner",
    "ens_planner",
    "ensembler",
    "debugger",
    "leakage",
    "data",
    "test",
    "validator",
]

LEAKAGE_STATUS_VALUES: list[str] = [
    "Yes Data Leakage",
    "No Data Leakage",
]

CONTAMINATION_VERDICT_VALUES: list[str] = [
    "Novel",
    "Same",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_retrieved_model(**overrides: Any) -> RetrievedModel:
    """Build a valid RetrievedModel with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed RetrievedModel instance.
    """
    defaults: dict[str, Any] = {
        "model_name": "RandomForestClassifier",
        "example_code": "from sklearn.ensemble import RandomForestClassifier\nclf = RandomForestClassifier()\n",
    }
    defaults.update(overrides)
    return RetrievedModel(**defaults)


def _make_refine_plan(**overrides: Any) -> RefinePlan:
    """Build a valid RefinePlan with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed RefinePlan instance.
    """
    defaults: dict[str, Any] = {
        "code_block": "clf = RandomForestClassifier(n_estimators=100)",
        "plan": "Increase n_estimators to 500 and add class_weight='balanced'.",
    }
    defaults.update(overrides)
    return RefinePlan(**defaults)


def _make_leakage_answer(**overrides: Any) -> LeakageAnswer:
    """Build a valid LeakageAnswer with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed LeakageAnswer instance.
    """
    defaults: dict[str, Any] = {
        "leakage_status": "No Data Leakage",
        "code_block": "X_train, X_test = train_test_split(X, y, test_size=0.2)",
    }
    defaults.update(overrides)
    return LeakageAnswer(**defaults)


# ===========================================================================
# REQ-DM-013: AgentType Enum
# ===========================================================================


@pytest.mark.unit
class TestAgentType:
    """AgentType must be a StrEnum with exactly 14 values (REQ-DM-013)."""

    def test_is_string_enum(self) -> None:
        """AgentType inherits from StrEnum."""
        assert issubclass(AgentType, StrEnum)

    def test_member_count(self) -> None:
        """Enum has exactly 17 members (14 original + baseline + researcher + validator)."""
        assert len(AgentType) == 17

    @pytest.mark.parametrize("value", AGENT_TYPE_VALUES)
    def test_contains_expected_value(self, value: str) -> None:
        """Each spec-mandated value is present in the enum."""
        member = AgentType(value)
        assert member.value == value

    def test_values_match_spec_exactly(self) -> None:
        """The full set of values matches the spec with no extras."""
        actual = sorted(m.value for m in AgentType)
        expected = sorted(AGENT_TYPE_VALUES)
        assert actual == expected

    @pytest.mark.parametrize("value", AGENT_TYPE_VALUES)
    def test_string_equality(self, value: str) -> None:
        """StrEnum members compare equal to their plain string value."""
        assert AgentType(value) == value

    def test_invalid_value_raises(self) -> None:
        """Constructing with a non-existent value raises ValueError."""
        with pytest.raises(ValueError, match="not a valid"):
            AgentType("nonexistent_agent_type")


# ===========================================================================
# REQ-DM-014: RetrievedModel
# ===========================================================================


@pytest.mark.unit
class TestRetrievedModelConstruction:
    """RetrievedModel has required fields model_name and example_code (REQ-DM-014)."""

    def test_valid_construction(self) -> None:
        """Constructing with both required fields succeeds."""
        model = _make_retrieved_model()
        assert model.model_name == "RandomForestClassifier"
        assert "RandomForestClassifier" in model.example_code

    def test_model_name_is_string(self) -> None:
        """model_name field holds a string value."""
        model = _make_retrieved_model(model_name="XGBClassifier")
        assert isinstance(model.model_name, str)
        assert model.model_name == "XGBClassifier"

    def test_example_code_is_string(self) -> None:
        """example_code field holds a string value."""
        model = _make_retrieved_model(example_code="import xgboost")
        assert isinstance(model.example_code, str)
        assert model.example_code == "import xgboost"

    def test_missing_model_name_raises(self) -> None:
        """Omitting model_name raises ValidationError."""
        with pytest.raises(ValidationError):
            RetrievedModel(example_code="code")  # type: ignore[call-arg]

    def test_missing_example_code_raises(self) -> None:
        """Omitting example_code raises ValidationError."""
        with pytest.raises(ValidationError):
            RetrievedModel(model_name="Model")  # type: ignore[call-arg]

    def test_missing_both_fields_raises(self) -> None:
        """Omitting both required fields raises ValidationError."""
        with pytest.raises(ValidationError):
            RetrievedModel()  # type: ignore[call-arg]


@pytest.mark.unit
class TestRetrievedModelFrozen:
    """RetrievedModel is frozen (immutable) per spec."""

    def test_cannot_mutate_model_name(self) -> None:
        """Assignment to model_name raises an error."""
        model = _make_retrieved_model()
        with pytest.raises(ValidationError):
            model.model_name = "OtherModel"  # type: ignore[misc]

    def test_cannot_mutate_example_code(self) -> None:
        """Assignment to example_code raises an error."""
        model = _make_retrieved_model()
        with pytest.raises(ValidationError):
            model.example_code = "new code"  # type: ignore[misc]


@pytest.mark.unit
class TestRetrievedModelSerialization:
    """RetrievedModel supports JSON round-trip serialization."""

    def test_round_trip_preserves_all_fields(self) -> None:
        """Serialize and deserialize; all fields preserved."""
        original = _make_retrieved_model()
        json_str = original.model_dump_json()
        restored = RetrievedModel.model_validate_json(json_str)
        assert restored.model_name == original.model_name
        assert restored.example_code == original.example_code

    def test_round_trip_equality(self) -> None:
        """Serialized then deserialized object equals the original."""
        original = _make_retrieved_model()
        restored = RetrievedModel.model_validate_json(original.model_dump_json())
        assert original == restored

    def test_model_dump_json_returns_valid_json(self) -> None:
        """model_dump_json() produces parseable JSON."""
        model = _make_retrieved_model()
        parsed = json.loads(model.model_dump_json())
        assert isinstance(parsed, dict)
        assert "model_name" in parsed
        assert "example_code" in parsed


# ===========================================================================
# REQ-DM-015: RetrieverOutput
# ===========================================================================


@pytest.mark.unit
class TestRetrieverOutputConstruction:
    """RetrieverOutput holds a non-empty list of RetrievedModel (REQ-DM-015)."""

    def test_valid_construction_single_model(self) -> None:
        """Constructing with a single-element list succeeds."""
        model = _make_retrieved_model()
        output = RetrieverOutput(models=[model])
        assert len(output.models) == 1
        assert output.models[0].model_name == "RandomForestClassifier"

    def test_valid_construction_multiple_models(self) -> None:
        """Constructing with multiple models succeeds."""
        models = [
            _make_retrieved_model(model_name="RF"),
            _make_retrieved_model(model_name="XGB"),
            _make_retrieved_model(model_name="LGB"),
        ]
        output = RetrieverOutput(models=models)
        assert len(output.models) == 3
        assert output.models[0].model_name == "RF"
        assert output.models[1].model_name == "XGB"
        assert output.models[2].model_name == "LGB"

    def test_empty_list_raises_validation_error(self) -> None:
        """Empty models list raises ValidationError."""
        with pytest.raises(ValidationError):
            RetrieverOutput(models=[])

    def test_missing_models_field_raises(self) -> None:
        """Omitting models field raises ValidationError."""
        with pytest.raises(ValidationError):
            RetrieverOutput()  # type: ignore[call-arg]


@pytest.mark.unit
class TestRetrieverOutputJsonSchema:
    """RetrieverOutput produces a valid JSON schema (REQ-DM-015)."""

    def test_json_schema_is_valid_dict(self) -> None:
        """model_json_schema() produces a valid dict."""
        schema = RetrieverOutput.model_json_schema()
        assert isinstance(schema, dict)

    def test_json_schema_has_properties(self) -> None:
        """JSON schema contains a 'properties' key."""
        schema = RetrieverOutput.model_json_schema()
        assert "properties" in schema

    def test_json_schema_includes_models_field(self) -> None:
        """JSON schema includes the 'models' field."""
        schema = RetrieverOutput.model_json_schema()
        assert "models" in schema["properties"]

    def test_json_schema_serializable(self) -> None:
        """JSON schema can be serialized to a JSON string."""
        schema = RetrieverOutput.model_json_schema()
        json_str = json.dumps(schema)
        parsed = json.loads(json_str)
        assert parsed == schema


@pytest.mark.unit
class TestRetrieverOutputSerialization:
    """RetrieverOutput supports JSON round-trip serialization."""

    def test_round_trip_preserves_models(self) -> None:
        """Serialize and deserialize; models list preserved."""
        original = RetrieverOutput(models=[_make_retrieved_model()])
        json_str = original.model_dump_json()
        restored = RetrieverOutput.model_validate_json(json_str)
        assert len(restored.models) == 1
        assert restored.models[0].model_name == original.models[0].model_name
        assert restored.models[0].example_code == original.models[0].example_code

    def test_round_trip_equality(self) -> None:
        """Serialized then deserialized object equals the original."""
        original = RetrieverOutput(
            models=[
                _make_retrieved_model(model_name="RF"),
                _make_retrieved_model(model_name="XGB"),
            ]
        )
        restored = RetrieverOutput.model_validate_json(original.model_dump_json())
        assert original == restored

    def test_model_dump_json_returns_valid_json(self) -> None:
        """model_dump_json() produces parseable JSON."""
        output = RetrieverOutput(models=[_make_retrieved_model()])
        parsed = json.loads(output.model_dump_json())
        assert isinstance(parsed, dict)
        assert "models" in parsed
        assert isinstance(parsed["models"], list)


@pytest.mark.unit
class TestRetrieverOutputFrozen:
    """RetrieverOutput is frozen (immutable) per spec."""

    def test_cannot_mutate_models(self) -> None:
        """Assignment to models field raises an error."""
        output = RetrieverOutput(models=[_make_retrieved_model()])
        with pytest.raises(ValidationError):
            output.models = [_make_retrieved_model(model_name="Other")]  # type: ignore[misc]


# ===========================================================================
# REQ-DM-016: RefinePlan
# ===========================================================================


@pytest.mark.unit
class TestRefinePlanConstruction:
    """RefinePlan has required fields code_block and plan (REQ-DM-016)."""

    def test_valid_construction(self) -> None:
        """Constructing with both required fields succeeds."""
        plan = _make_refine_plan()
        assert "RandomForestClassifier" in plan.code_block
        assert "n_estimators" in plan.plan

    def test_code_block_is_string(self) -> None:
        """code_block field holds a string value."""
        plan = _make_refine_plan(code_block="model.fit(X, y)")
        assert isinstance(plan.code_block, str)
        assert plan.code_block == "model.fit(X, y)"

    def test_plan_is_string(self) -> None:
        """Plan field holds a string value."""
        plan = _make_refine_plan(plan="Try gradient boosting instead.")
        assert isinstance(plan.plan, str)
        assert plan.plan == "Try gradient boosting instead."

    def test_missing_code_block_raises(self) -> None:
        """Omitting code_block raises ValidationError."""
        with pytest.raises(ValidationError):
            RefinePlan(plan="some plan")  # type: ignore[call-arg]

    def test_missing_plan_raises(self) -> None:
        """Omitting plan raises ValidationError."""
        with pytest.raises(ValidationError):
            RefinePlan(code_block="some code")  # type: ignore[call-arg]

    def test_missing_both_fields_raises(self) -> None:
        """Omitting both required fields raises ValidationError."""
        with pytest.raises(ValidationError):
            RefinePlan()  # type: ignore[call-arg]


@pytest.mark.unit
class TestRefinePlanFrozen:
    """RefinePlan is frozen (immutable) per spec."""

    def test_cannot_mutate_code_block(self) -> None:
        """Assignment to code_block raises an error."""
        plan = _make_refine_plan()
        with pytest.raises(ValidationError):
            plan.code_block = "new code"  # type: ignore[misc]

    def test_cannot_mutate_plan(self) -> None:
        """Assignment to plan raises an error."""
        plan = _make_refine_plan()
        with pytest.raises(ValidationError):
            plan.plan = "new plan"  # type: ignore[misc]


@pytest.mark.unit
class TestRefinePlanSerialization:
    """RefinePlan supports JSON round-trip serialization."""

    def test_round_trip_preserves_all_fields(self) -> None:
        """Serialize and deserialize; all fields preserved."""
        original = _make_refine_plan()
        json_str = original.model_dump_json()
        restored = RefinePlan.model_validate_json(json_str)
        assert restored.code_block == original.code_block
        assert restored.plan == original.plan

    def test_round_trip_equality(self) -> None:
        """Serialized then deserialized object equals the original."""
        original = _make_refine_plan()
        restored = RefinePlan.model_validate_json(original.model_dump_json())
        assert original == restored

    def test_model_dump_json_returns_valid_json(self) -> None:
        """model_dump_json() produces parseable JSON."""
        plan = _make_refine_plan()
        parsed = json.loads(plan.model_dump_json())
        assert isinstance(parsed, dict)
        assert "code_block" in parsed
        assert "plan" in parsed


# ===========================================================================
# REQ-DM-017: ExtractorOutput
# ===========================================================================


@pytest.mark.unit
class TestExtractorOutputConstruction:
    """ExtractorOutput holds a non-empty list of RefinePlan (REQ-DM-017)."""

    def test_valid_construction_single_plan(self) -> None:
        """Constructing with a single-element list succeeds."""
        plan = _make_refine_plan()
        output = ExtractorOutput(plans=[plan])
        assert len(output.plans) == 1
        assert "RandomForestClassifier" in output.plans[0].code_block

    def test_valid_construction_multiple_plans(self) -> None:
        """Constructing with multiple plans succeeds."""
        plans = [
            _make_refine_plan(code_block="block1", plan="plan1"),
            _make_refine_plan(code_block="block2", plan="plan2"),
        ]
        output = ExtractorOutput(plans=plans)
        assert len(output.plans) == 2
        assert output.plans[0].code_block == "block1"
        assert output.plans[1].code_block == "block2"

    def test_empty_list_raises_validation_error(self) -> None:
        """Empty plans list raises ValidationError."""
        with pytest.raises(ValidationError):
            ExtractorOutput(plans=[])

    def test_missing_plans_field_raises(self) -> None:
        """Omitting plans field raises ValidationError."""
        with pytest.raises(ValidationError):
            ExtractorOutput()  # type: ignore[call-arg]


@pytest.mark.unit
class TestExtractorOutputJsonSchema:
    """ExtractorOutput produces a valid JSON schema (REQ-DM-017)."""

    def test_json_schema_is_valid_dict(self) -> None:
        """model_json_schema() produces a valid dict."""
        schema = ExtractorOutput.model_json_schema()
        assert isinstance(schema, dict)

    def test_json_schema_has_properties(self) -> None:
        """JSON schema contains a 'properties' key."""
        schema = ExtractorOutput.model_json_schema()
        assert "properties" in schema

    def test_json_schema_includes_plans_field(self) -> None:
        """JSON schema includes the 'plans' field."""
        schema = ExtractorOutput.model_json_schema()
        assert "plans" in schema["properties"]

    def test_json_schema_serializable(self) -> None:
        """JSON schema can be serialized to a JSON string."""
        schema = ExtractorOutput.model_json_schema()
        json_str = json.dumps(schema)
        parsed = json.loads(json_str)
        assert parsed == schema


@pytest.mark.unit
class TestExtractorOutputSerialization:
    """ExtractorOutput supports JSON round-trip serialization."""

    def test_round_trip_preserves_plans(self) -> None:
        """Serialize and deserialize; plans list preserved."""
        original = ExtractorOutput(plans=[_make_refine_plan()])
        json_str = original.model_dump_json()
        restored = ExtractorOutput.model_validate_json(json_str)
        assert len(restored.plans) == 1
        assert restored.plans[0].code_block == original.plans[0].code_block
        assert restored.plans[0].plan == original.plans[0].plan

    def test_round_trip_equality(self) -> None:
        """Serialized then deserialized object equals the original."""
        original = ExtractorOutput(
            plans=[
                _make_refine_plan(code_block="b1", plan="p1"),
                _make_refine_plan(code_block="b2", plan="p2"),
            ]
        )
        restored = ExtractorOutput.model_validate_json(original.model_dump_json())
        assert original == restored

    def test_model_dump_json_returns_valid_json(self) -> None:
        """model_dump_json() produces parseable JSON."""
        output = ExtractorOutput(plans=[_make_refine_plan()])
        parsed = json.loads(output.model_dump_json())
        assert isinstance(parsed, dict)
        assert "plans" in parsed
        assert isinstance(parsed["plans"], list)


@pytest.mark.unit
class TestExtractorOutputFrozen:
    """ExtractorOutput is frozen (immutable) per spec."""

    def test_cannot_mutate_plans(self) -> None:
        """Assignment to plans field raises an error."""
        output = ExtractorOutput(plans=[_make_refine_plan()])
        with pytest.raises(ValidationError):
            output.plans = [_make_refine_plan(code_block="new")]  # type: ignore[misc]


# ===========================================================================
# REQ-DM-018: LeakageAnswer
# ===========================================================================


@pytest.mark.unit
class TestLeakageAnswerConstruction:
    """LeakageAnswer has leakage_status (Literal) and code_block (REQ-DM-018)."""

    def test_valid_construction_no_leakage(self) -> None:
        """Constructing with 'No Data Leakage' status succeeds."""
        answer = _make_leakage_answer(leakage_status="No Data Leakage")
        assert answer.leakage_status == "No Data Leakage"

    def test_valid_construction_yes_leakage(self) -> None:
        """Constructing with 'Yes Data Leakage' status succeeds."""
        answer = _make_leakage_answer(leakage_status="Yes Data Leakage")
        assert answer.leakage_status == "Yes Data Leakage"

    @pytest.mark.parametrize("status", LEAKAGE_STATUS_VALUES)
    def test_accepts_all_valid_statuses(self, status: str) -> None:
        """All spec-mandated leakage_status values are accepted."""
        answer = _make_leakage_answer(leakage_status=status)
        assert answer.leakage_status == status

    def test_code_block_is_string(self) -> None:
        """code_block field holds a string value."""
        answer = _make_leakage_answer(code_block="train_test_split(X, y)")
        assert isinstance(answer.code_block, str)
        assert answer.code_block == "train_test_split(X, y)"

    def test_invalid_leakage_status_raises(self) -> None:
        """Invalid leakage_status string raises ValidationError."""
        with pytest.raises(ValidationError):
            _make_leakage_answer(leakage_status="Maybe Leakage")

    def test_missing_leakage_status_raises(self) -> None:
        """Omitting leakage_status raises ValidationError."""
        with pytest.raises(ValidationError):
            LeakageAnswer(code_block="some code")  # type: ignore[call-arg]

    def test_missing_code_block_raises(self) -> None:
        """Omitting code_block raises ValidationError."""
        with pytest.raises(ValidationError):
            LeakageAnswer(leakage_status="No Data Leakage")  # type: ignore[call-arg]

    def test_missing_both_fields_raises(self) -> None:
        """Omitting both required fields raises ValidationError."""
        with pytest.raises(ValidationError):
            LeakageAnswer()  # type: ignore[call-arg]


@pytest.mark.unit
class TestLeakageAnswerFrozen:
    """LeakageAnswer is frozen (immutable) per spec."""

    def test_cannot_mutate_leakage_status(self) -> None:
        """Assignment to leakage_status raises an error."""
        answer = _make_leakage_answer()
        with pytest.raises(ValidationError):
            answer.leakage_status = "Yes Data Leakage"  # type: ignore[misc]

    def test_cannot_mutate_code_block(self) -> None:
        """Assignment to code_block raises an error."""
        answer = _make_leakage_answer()
        with pytest.raises(ValidationError):
            answer.code_block = "new code"  # type: ignore[misc]


@pytest.mark.unit
class TestLeakageAnswerSerialization:
    """LeakageAnswer supports JSON round-trip serialization."""

    def test_round_trip_preserves_all_fields(self) -> None:
        """Serialize and deserialize; all fields preserved."""
        original = _make_leakage_answer()
        json_str = original.model_dump_json()
        restored = LeakageAnswer.model_validate_json(json_str)
        assert restored.leakage_status == original.leakage_status
        assert restored.code_block == original.code_block

    def test_round_trip_equality(self) -> None:
        """Serialized then deserialized object equals the original."""
        original = _make_leakage_answer()
        restored = LeakageAnswer.model_validate_json(original.model_dump_json())
        assert original == restored

    def test_model_dump_json_returns_valid_json(self) -> None:
        """model_dump_json() produces parseable JSON."""
        answer = _make_leakage_answer()
        parsed = json.loads(answer.model_dump_json())
        assert isinstance(parsed, dict)
        assert "leakage_status" in parsed
        assert "code_block" in parsed


# ===========================================================================
# REQ-DM-019: LeakageDetectionOutput
# ===========================================================================


@pytest.mark.unit
class TestLeakageDetectionOutputConstruction:
    """LeakageDetectionOutput holds a non-empty list of LeakageAnswer (REQ-DM-019)."""

    def test_valid_construction_single_answer(self) -> None:
        """Constructing with a single-element list succeeds."""
        answer = _make_leakage_answer()
        output = LeakageDetectionOutput(answers=[answer])
        assert len(output.answers) == 1
        assert output.answers[0].leakage_status == "No Data Leakage"

    def test_valid_construction_multiple_answers(self) -> None:
        """Constructing with multiple answers succeeds."""
        answers = [
            _make_leakage_answer(leakage_status="No Data Leakage", code_block="block1"),
            _make_leakage_answer(
                leakage_status="Yes Data Leakage", code_block="block2"
            ),
        ]
        output = LeakageDetectionOutput(answers=answers)
        assert len(output.answers) == 2
        assert output.answers[0].leakage_status == "No Data Leakage"
        assert output.answers[1].leakage_status == "Yes Data Leakage"

    def test_empty_list_raises_validation_error(self) -> None:
        """Empty answers list raises ValidationError."""
        with pytest.raises(ValidationError):
            LeakageDetectionOutput(answers=[])

    def test_missing_answers_field_raises(self) -> None:
        """Omitting answers field raises ValidationError."""
        with pytest.raises(ValidationError):
            LeakageDetectionOutput()  # type: ignore[call-arg]


@pytest.mark.unit
class TestLeakageDetectionOutputJsonSchema:
    """LeakageDetectionOutput produces a valid JSON schema (REQ-DM-019)."""

    def test_json_schema_is_valid_dict(self) -> None:
        """model_json_schema() produces a valid dict."""
        schema = LeakageDetectionOutput.model_json_schema()
        assert isinstance(schema, dict)

    def test_json_schema_has_properties(self) -> None:
        """JSON schema contains a 'properties' key."""
        schema = LeakageDetectionOutput.model_json_schema()
        assert "properties" in schema

    def test_json_schema_includes_answers_field(self) -> None:
        """JSON schema includes the 'answers' field."""
        schema = LeakageDetectionOutput.model_json_schema()
        assert "answers" in schema["properties"]

    def test_json_schema_serializable(self) -> None:
        """JSON schema can be serialized to a JSON string."""
        schema = LeakageDetectionOutput.model_json_schema()
        json_str = json.dumps(schema)
        parsed = json.loads(json_str)
        assert parsed == schema


@pytest.mark.unit
class TestLeakageDetectionOutputSerialization:
    """LeakageDetectionOutput supports JSON round-trip serialization."""

    def test_round_trip_preserves_answers(self) -> None:
        """Serialize and deserialize; answers list preserved."""
        original = LeakageDetectionOutput(answers=[_make_leakage_answer()])
        json_str = original.model_dump_json()
        restored = LeakageDetectionOutput.model_validate_json(json_str)
        assert len(restored.answers) == 1
        assert restored.answers[0].leakage_status == original.answers[0].leakage_status
        assert restored.answers[0].code_block == original.answers[0].code_block

    def test_round_trip_equality(self) -> None:
        """Serialized then deserialized object equals the original."""
        original = LeakageDetectionOutput(
            answers=[
                _make_leakage_answer(code_block="a1"),
                _make_leakage_answer(
                    leakage_status="Yes Data Leakage", code_block="a2"
                ),
            ]
        )
        restored = LeakageDetectionOutput.model_validate_json(
            original.model_dump_json()
        )
        assert original == restored

    def test_model_dump_json_returns_valid_json(self) -> None:
        """model_dump_json() produces parseable JSON."""
        output = LeakageDetectionOutput(answers=[_make_leakage_answer()])
        parsed = json.loads(output.model_dump_json())
        assert isinstance(parsed, dict)
        assert "answers" in parsed
        assert isinstance(parsed["answers"], list)


@pytest.mark.unit
class TestLeakageDetectionOutputFrozen:
    """LeakageDetectionOutput is frozen (immutable) per spec."""

    def test_cannot_mutate_answers(self) -> None:
        """Assignment to answers field raises an error."""
        output = LeakageDetectionOutput(answers=[_make_leakage_answer()])
        with pytest.raises(ValidationError):
            output.answers = [_make_leakage_answer(code_block="new")]  # type: ignore[misc]


# ===========================================================================
# REQ-DM-020: DataContaminationResult
# ===========================================================================


@pytest.mark.unit
class TestDataContaminationResultConstruction:
    """DataContaminationResult has verdict Literal field (REQ-DM-020)."""

    def test_valid_construction_novel(self) -> None:
        """Constructing with 'Novel' verdict succeeds."""
        result = DataContaminationResult(verdict="Novel")
        assert result.verdict == "Novel"

    def test_valid_construction_same(self) -> None:
        """Constructing with 'Same' verdict succeeds."""
        result = DataContaminationResult(verdict="Same")
        assert result.verdict == "Same"

    @pytest.mark.parametrize("verdict", CONTAMINATION_VERDICT_VALUES)
    def test_accepts_all_valid_verdicts(self, verdict: str) -> None:
        """All spec-mandated verdict values are accepted."""
        result = DataContaminationResult(verdict=verdict)  # type: ignore[arg-type]
        assert result.verdict == verdict

    def test_invalid_verdict_raises(self) -> None:
        """Invalid verdict string raises ValidationError."""
        with pytest.raises(ValidationError):
            DataContaminationResult(verdict="Unknown")  # type: ignore[arg-type]

    def test_missing_verdict_raises(self) -> None:
        """Omitting verdict raises ValidationError."""
        with pytest.raises(ValidationError):
            DataContaminationResult()  # type: ignore[call-arg]


@pytest.mark.unit
class TestDataContaminationResultFrozen:
    """DataContaminationResult is frozen (immutable) per spec."""

    def test_cannot_mutate_verdict(self) -> None:
        """Assignment to verdict raises an error."""
        result = DataContaminationResult(verdict="Novel")
        with pytest.raises(ValidationError):
            result.verdict = "Same"  # type: ignore[misc]


@pytest.mark.unit
class TestDataContaminationResultSerialization:
    """DataContaminationResult supports JSON round-trip serialization."""

    def test_round_trip_preserves_verdict(self) -> None:
        """Serialize and deserialize; verdict preserved."""
        original = DataContaminationResult(verdict="Novel")
        json_str = original.model_dump_json()
        restored = DataContaminationResult.model_validate_json(json_str)
        assert restored.verdict == original.verdict

    def test_round_trip_equality(self) -> None:
        """Serialized then deserialized object equals the original."""
        original = DataContaminationResult(verdict="Same")
        restored = DataContaminationResult.model_validate_json(
            original.model_dump_json()
        )
        assert original == restored

    def test_model_dump_json_returns_valid_json(self) -> None:
        """model_dump_json() produces parseable JSON."""
        result = DataContaminationResult(verdict="Novel")
        parsed = json.loads(result.model_dump_json())
        assert isinstance(parsed, dict)
        assert "verdict" in parsed
        assert parsed["verdict"] == "Novel"


# ===========================================================================
# Property-based tests: Agent output models with Hypothesis
# ===========================================================================


@pytest.mark.unit
class TestRetrievedModelPropertyBased:
    """Property-based tests for RetrievedModel using Hypothesis."""

    @given(
        model_name=st.text(min_size=1, max_size=100),
        example_code=st.text(min_size=1, max_size=500),
    )
    @settings(max_examples=50)
    def test_any_valid_strings_produce_valid_model(
        self, model_name: str, example_code: str
    ) -> None:
        """Property: any non-empty strings create a valid RetrievedModel."""
        model = RetrievedModel(model_name=model_name, example_code=example_code)
        assert model.model_name == model_name
        assert model.example_code == example_code

    @given(
        model_name=st.text(min_size=1, max_size=50),
        example_code=st.text(min_size=1, max_size=200),
    )
    @settings(max_examples=30)
    def test_round_trip_preserves_all_fields(
        self, model_name: str, example_code: str
    ) -> None:
        """Property: JSON round-trip preserves all RetrievedModel fields."""
        original = RetrievedModel(model_name=model_name, example_code=example_code)
        restored = RetrievedModel.model_validate_json(original.model_dump_json())
        assert restored == original


@pytest.mark.unit
class TestRetrieverOutputPropertyBased:
    """Property-based tests for RetrieverOutput using Hypothesis."""

    @given(
        model_names=st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10),
    )
    @settings(max_examples=30)
    def test_any_nonempty_list_produces_valid_output(
        self, model_names: list[str]
    ) -> None:
        """Property: any non-empty list of RetrievedModels creates valid output."""
        models = [
            RetrievedModel(model_name=name, example_code=f"import {name}")
            for name in model_names
        ]
        output = RetrieverOutput(models=models)
        assert len(output.models) == len(model_names)
        assert len(output.models) >= 1

    @given(
        model_names=st.lists(st.text(min_size=1, max_size=30), min_size=1, max_size=5),
    )
    @settings(max_examples=30)
    def test_round_trip_preserves_model_count(self, model_names: list[str]) -> None:
        """Property: JSON round-trip preserves number of models."""
        models = [
            RetrievedModel(model_name=name, example_code=f"code_{name}")
            for name in model_names
        ]
        original = RetrieverOutput(models=models)
        restored = RetrieverOutput.model_validate_json(original.model_dump_json())
        assert len(restored.models) == len(original.models)


@pytest.mark.unit
class TestRefinePlanPropertyBased:
    """Property-based tests for RefinePlan using Hypothesis."""

    @given(
        code_block=st.text(min_size=1, max_size=200),
        plan=st.text(min_size=1, max_size=200),
    )
    @settings(max_examples=50)
    def test_any_valid_strings_produce_valid_plan(
        self, code_block: str, plan: str
    ) -> None:
        """Property: any non-empty strings create a valid RefinePlan."""
        refine_plan = RefinePlan(code_block=code_block, plan=plan)
        assert refine_plan.code_block == code_block
        assert refine_plan.plan == plan

    @given(
        code_block=st.text(min_size=1, max_size=100),
        plan=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=30)
    def test_round_trip_preserves_all_fields(self, code_block: str, plan: str) -> None:
        """Property: JSON round-trip preserves all RefinePlan fields."""
        original = RefinePlan(code_block=code_block, plan=plan)
        restored = RefinePlan.model_validate_json(original.model_dump_json())
        assert restored == original


@pytest.mark.unit
class TestExtractorOutputPropertyBased:
    """Property-based tests for ExtractorOutput using Hypothesis."""

    @given(
        plan_count=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30)
    def test_any_nonempty_plan_list_is_valid(self, plan_count: int) -> None:
        """Property: any non-empty list of RefinePlans creates valid output."""
        plans = [
            RefinePlan(code_block=f"block_{i}", plan=f"plan_{i}")
            for i in range(plan_count)
        ]
        output = ExtractorOutput(plans=plans)
        assert len(output.plans) == plan_count
        assert len(output.plans) >= 1


@pytest.mark.unit
class TestLeakageAnswerPropertyBased:
    """Property-based tests for LeakageAnswer using Hypothesis."""

    @given(
        status=st.sampled_from(LEAKAGE_STATUS_VALUES),
        code_block=st.text(min_size=1, max_size=200),
    )
    @settings(max_examples=50)
    def test_any_valid_status_and_code_create_valid_answer(
        self, status: str, code_block: str
    ) -> None:
        """Property: any valid status with any code block creates valid answer."""
        answer = LeakageAnswer(leakage_status=status, code_block=code_block)  # type: ignore[arg-type]
        assert answer.leakage_status == status
        assert answer.code_block == code_block

    @given(
        status=st.sampled_from(LEAKAGE_STATUS_VALUES),
        code_block=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=30)
    def test_round_trip_preserves_all_fields(
        self, status: str, code_block: str
    ) -> None:
        """Property: JSON round-trip preserves all LeakageAnswer fields."""
        original = LeakageAnswer(leakage_status=status, code_block=code_block)  # type: ignore[arg-type]
        restored = LeakageAnswer.model_validate_json(original.model_dump_json())
        assert restored == original


@pytest.mark.unit
class TestLeakageDetectionOutputPropertyBased:
    """Property-based tests for LeakageDetectionOutput using Hypothesis."""

    @given(
        answer_count=st.integers(min_value=1, max_value=10),
        status=st.sampled_from(LEAKAGE_STATUS_VALUES),
    )
    @settings(max_examples=30)
    def test_any_nonempty_answer_list_is_valid(
        self, answer_count: int, status: str
    ) -> None:
        """Property: any non-empty list of LeakageAnswers creates valid output."""
        answers = [
            LeakageAnswer(leakage_status=status, code_block=f"block_{i}")  # type: ignore[arg-type]
            for i in range(answer_count)
        ]
        output = LeakageDetectionOutput(answers=answers)
        assert len(output.answers) == answer_count
        assert len(output.answers) >= 1


@pytest.mark.unit
class TestDataContaminationResultPropertyBased:
    """Property-based tests for DataContaminationResult using Hypothesis."""

    @given(verdict=st.sampled_from(CONTAMINATION_VERDICT_VALUES))
    @settings(max_examples=20)
    def test_any_valid_verdict_creates_valid_result(self, verdict: str) -> None:
        """Property: any valid verdict creates a valid DataContaminationResult."""
        result = DataContaminationResult(verdict=verdict)  # type: ignore[arg-type]
        assert result.verdict == verdict

    @given(verdict=st.sampled_from(CONTAMINATION_VERDICT_VALUES))
    @settings(max_examples=20)
    def test_round_trip_preserves_verdict(self, verdict: str) -> None:
        """Property: JSON round-trip preserves verdict."""
        original = DataContaminationResult(verdict=verdict)  # type: ignore[arg-type]
        restored = DataContaminationResult.model_validate_json(
            original.model_dump_json()
        )
        assert restored == original

    @given(
        bad_verdict=st.text(min_size=1, max_size=50).filter(
            lambda s: s not in CONTAMINATION_VERDICT_VALUES
        )
    )
    @settings(max_examples=30)
    def test_invalid_verdict_always_rejected(self, bad_verdict: str) -> None:
        """Property: any non-valid verdict string is always rejected."""
        with pytest.raises(ValidationError):
            DataContaminationResult(verdict=bad_verdict)  # type: ignore[arg-type]
