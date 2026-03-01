"""Tests for MLE-STAR solution and code block data models (Task 04).

Validates SolutionPhase, SolutionScript, CodeBlockCategory, and CodeBlock
Pydantic models defined in ``src/mle_star/models.py``.  These tests are
written TDD-first -- the implementation does not yet exist.  They serve as
the executable specification for REQ-DM-008 through REQ-DM-012.

Refs:
    SRS 01a (Data Models Solution), IMPLEMENTATION_PLAN.md Task 04.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
import json
from typing import Any

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    CodeBlock,
    CodeBlockCategory,
    SolutionPhase,
    SolutionScript,
)
from pydantic import ValidationError
import pytest

# ---------------------------------------------------------------------------
# Constants -- canonical values from the spec
# ---------------------------------------------------------------------------

SOLUTION_PHASE_VALUES: list[str] = [
    "init",
    "merged",
    "refined",
    "ensemble",
    "final",
]

CODE_BLOCK_CATEGORY_VALUES: list[str] = [
    "preprocessing",
    "feature_engineering",
    "model_selection",
    "training",
    "hyperparameter_tuning",
    "ensemble",
    "postprocessing",
    "other",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_solution_script(**overrides: Any) -> SolutionScript:
    """Build a valid SolutionScript with sensible defaults.

    All required fields are populated; any keyword argument overrides
    the corresponding default.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed SolutionScript instance.
    """
    defaults: dict[str, Any] = {
        "content": "import pandas as pd\ndf = pd.read_csv('train.csv')\n",
        "phase": SolutionPhase.INIT,
    }
    defaults.update(overrides)
    return SolutionScript(**defaults)


# ===========================================================================
# REQ-DM-008: SolutionPhase Enum
# ===========================================================================


@pytest.mark.unit
class TestSolutionPhase:
    """SolutionPhase must be a StrEnum with exactly 5 values (REQ-DM-008)."""

    def test_is_string_enum(self) -> None:
        """SolutionPhase inherits from StrEnum."""
        assert issubclass(SolutionPhase, StrEnum)

    def test_member_count_is_five(self) -> None:
        """Enum has exactly 5 members."""
        assert len(SolutionPhase) == 5

    @pytest.mark.parametrize("value", SOLUTION_PHASE_VALUES)
    def test_contains_expected_value(self, value: str) -> None:
        """Each spec-mandated value is present in the enum."""
        member = SolutionPhase(value)
        assert member.value == value

    def test_values_match_spec_exactly(self) -> None:
        """The full set of values matches the spec with no extras."""
        actual = sorted(m.value for m in SolutionPhase)
        expected = sorted(SOLUTION_PHASE_VALUES)
        assert actual == expected

    @pytest.mark.parametrize("value", SOLUTION_PHASE_VALUES)
    def test_string_equality(self, value: str) -> None:
        """StrEnum members compare equal to their plain string value."""
        assert SolutionPhase(value) == value

    def test_invalid_value_raises(self) -> None:
        """Constructing with a non-existent value raises ValueError."""
        with pytest.raises(ValueError, match="not a valid"):
            SolutionPhase("nonexistent_phase")


# ===========================================================================
# REQ-DM-009, REQ-DM-010: SolutionScript Model -- Construction & Fields
# ===========================================================================


@pytest.mark.unit
class TestSolutionScriptConstruction:
    """SolutionScript has correct required fields and defaults (REQ-DM-009)."""

    def test_valid_construction_with_required_fields_only(self) -> None:
        """Constructing with only required fields (content, phase) succeeds."""
        script = _make_solution_script()
        assert "import pandas" in script.content
        assert script.phase == SolutionPhase.INIT

    def test_valid_construction_with_all_fields(self) -> None:
        """Constructing with all fields succeeds."""
        now = datetime.now(UTC)
        script = SolutionScript(
            content="print('hello')",
            phase=SolutionPhase.REFINED,
            score=0.95,
            is_executable=False,
            source_model="claude-sonnet",
            created_at=now,
        )
        assert script.content == "print('hello')"
        assert script.phase == SolutionPhase.REFINED
        assert script.score == 0.95
        assert script.is_executable is False
        assert script.source_model == "claude-sonnet"
        assert script.created_at == now

    def test_score_defaults_to_none(self) -> None:
        """Score defaults to None when not provided."""
        script = _make_solution_script()
        assert script.score is None

    def test_is_executable_defaults_to_true(self) -> None:
        """is_executable defaults to True when not provided."""
        script = _make_solution_script()
        assert script.is_executable is True

    def test_source_model_defaults_to_none(self) -> None:
        """source_model defaults to None when not provided."""
        script = _make_solution_script()
        assert script.source_model is None

    def test_created_at_auto_set(self) -> None:
        """created_at is automatically set to approximately now."""
        before = datetime.now(UTC)
        script = _make_solution_script()
        after = datetime.now(UTC)
        assert before <= script.created_at <= after

    def test_created_at_is_datetime_type(self) -> None:
        """created_at is a datetime instance."""
        script = _make_solution_script()
        assert isinstance(script.created_at, datetime)

    def test_created_at_is_utc_aware(self) -> None:
        """created_at is timezone-aware (UTC)."""
        script = _make_solution_script()
        assert script.created_at.tzinfo is not None

    @pytest.mark.parametrize("phase_value", SOLUTION_PHASE_VALUES)
    def test_accepts_all_phase_values(self, phase_value: str) -> None:
        """SolutionScript accepts all valid SolutionPhase values."""
        script = _make_solution_script(phase=phase_value)
        assert script.phase == phase_value

    def test_invalid_phase_raises_validation_error(self) -> None:
        """Invalid phase string raises ValidationError."""
        with pytest.raises(ValidationError):
            _make_solution_script(phase="nonexistent_phase")


@pytest.mark.unit
class TestSolutionScriptRequiredFields:
    """SolutionScript raises ValidationError when required fields are missing."""

    def test_missing_content_raises(self) -> None:
        """Omitting content raises ValidationError."""
        with pytest.raises(ValidationError):
            SolutionScript(phase=SolutionPhase.INIT)  # type: ignore[call-arg]

    def test_missing_phase_raises(self) -> None:
        """Omitting phase raises ValidationError."""
        with pytest.raises(ValidationError):
            SolutionScript(content="print('hello')")  # type: ignore[call-arg]

    def test_missing_both_required_fields_raises(self) -> None:
        """Omitting both content and phase raises ValidationError."""
        with pytest.raises(ValidationError):
            SolutionScript()  # type: ignore[call-arg]


# ===========================================================================
# REQ-DM-009: SolutionScript Mutability (frozen=False)
# ===========================================================================


@pytest.mark.unit
class TestSolutionScriptMutability:
    """SolutionScript uses frozen=False; mutable for score updates (REQ-DM-009)."""

    def test_score_can_be_updated(self) -> None:
        """Score field can be mutated after construction."""
        script = _make_solution_script()
        assert script.score is None
        script.score = 0.85
        assert script.score == 0.85

    def test_score_can_be_updated_multiple_times(self) -> None:
        """Score field can be mutated multiple times."""
        script = _make_solution_script(score=0.5)
        script.score = 0.75
        assert script.score == 0.75
        script.score = 0.9
        assert script.score == 0.9

    def test_is_executable_can_be_updated(self) -> None:
        """is_executable field can be mutated after construction."""
        script = _make_solution_script()
        assert script.is_executable is True
        script.is_executable = False
        assert script.is_executable is False

    def test_content_can_be_updated(self) -> None:
        """Content field can be mutated (frozen=False means all fields mutable)."""
        script = _make_solution_script()
        script.content = "new content"
        assert script.content == "new content"

    def test_source_model_can_be_updated(self) -> None:
        """source_model field can be mutated after construction."""
        script = _make_solution_script()
        script.source_model = "claude-opus"
        assert script.source_model == "claude-opus"


# ===========================================================================
# REQ-DM-010: SolutionScript.replace_block()
# ===========================================================================


@pytest.mark.unit
class TestSolutionScriptReplaceBlock:
    """replace_block(old, new) returns a new SolutionScript (REQ-DM-010)."""

    def test_replace_block_happy_path(self) -> None:
        """replace_block replaces old substring with new in content."""
        script = _make_solution_script(content="alpha beta gamma")
        result = script.replace_block("beta", "BETA")
        assert result.content == "alpha BETA gamma"

    def test_replace_block_returns_new_instance(self) -> None:
        """replace_block returns a new SolutionScript, not the same object."""
        script = _make_solution_script(content="hello world")
        result = script.replace_block("hello", "hi")
        assert result is not script

    def test_replace_block_does_not_mutate_original(self) -> None:
        """replace_block does not alter the original SolutionScript content."""
        original_content = "hello world"
        script = _make_solution_script(content=original_content)
        script.replace_block("hello", "hi")
        assert script.content == original_content

    def test_replace_block_preserves_phase(self) -> None:
        """replace_block copies the phase from the original."""
        script = _make_solution_script(content="x = 1", phase=SolutionPhase.ENSEMBLE)
        result = script.replace_block("x = 1", "x = 2")
        assert result.phase == SolutionPhase.ENSEMBLE

    def test_replace_block_preserves_score(self) -> None:
        """replace_block copies the score from the original."""
        script = _make_solution_script(content="x = 1", score=0.92)
        result = script.replace_block("x = 1", "x = 2")
        assert result.score == 0.92

    def test_replace_block_preserves_is_executable(self) -> None:
        """replace_block copies is_executable from the original."""
        script = _make_solution_script(content="x = 1", is_executable=False)
        result = script.replace_block("x = 1", "x = 2")
        assert result.is_executable is False

    def test_replace_block_preserves_source_model(self) -> None:
        """replace_block copies source_model from the original."""
        script = _make_solution_script(content="x = 1", source_model="claude-sonnet")
        result = script.replace_block("x = 1", "x = 2")
        assert result.source_model == "claude-sonnet"

    def test_replace_block_replaces_last_occurrence(self) -> None:
        """replace_block replaces the last occurrence of old."""
        script = _make_solution_script(content="AAA BBB AAA CCC AAA")
        result = script.replace_block("AAA", "ZZZ")
        assert result.content == "AAA BBB AAA CCC ZZZ"

    def test_replace_block_old_not_found_raises_value_error(self) -> None:
        """replace_block raises ValueError when old is not in content."""
        script = _make_solution_script(content="hello world")
        with pytest.raises(ValueError):
            script.replace_block("missing_text", "replacement")

    def test_replace_block_empty_old_does_not_raise(self) -> None:
        """replace_block with empty old string inserts at the beginning.

        Python's str.replace('', 'x', 1) inserts at position 0, so this
        should not raise ValueError since '' is always 'in' a string.
        """
        script = _make_solution_script(content="hello")
        result = script.replace_block("", "prefix_")
        assert result.content.startswith("prefix_")

    def test_replace_block_with_multiline_content(self) -> None:
        """replace_block works with multiline content blocks."""
        old_block = "def train():\n    pass\n"
        new_block = "def train():\n    model.fit(X, y)\n"
        script = _make_solution_script(
            content=f"import sklearn\n{old_block}print('done')"
        )
        result = script.replace_block(old_block, new_block)
        assert new_block in result.content
        assert old_block not in result.content

    def test_replace_block_result_type_is_solution_script(self) -> None:
        """replace_block returns a SolutionScript instance."""
        script = _make_solution_script(content="a b c")
        result = script.replace_block("b", "B")
        assert isinstance(result, SolutionScript)


# ===========================================================================
# REQ-DM-010: SolutionScript.replace_block() -- Property-Based Tests
# ===========================================================================


@pytest.mark.unit
class TestSolutionScriptReplaceBlockPropertyBased:
    """Property-based tests for replace_block using Hypothesis."""

    @given(
        prefix=st.text(min_size=1, max_size=20),
        old=st.text(min_size=1, max_size=20),
        suffix=st.text(min_size=1, max_size=20),
        new=st.text(min_size=0, max_size=20),
    )
    @settings(max_examples=50)
    def test_replace_block_always_contains_new_text(
        self, prefix: str, old: str, suffix: str, new: str
    ) -> None:
        """Property: after replace_block, the result always contains new text."""
        content = prefix + old + suffix
        script = _make_solution_script(content=content)
        result = script.replace_block(old, new)
        assert new in result.content

    @given(
        content=st.text(min_size=5, max_size=100),
        needle=st.text(min_size=1, max_size=10),
    )
    @settings(max_examples=50)
    def test_replace_block_raises_when_needle_absent(
        self, content: str, needle: str
    ) -> None:
        """Property: replace_block raises ValueError when old not in content."""
        if needle in content:
            return  # skip when needle happens to be in content
        script = _make_solution_script(content=content)
        with pytest.raises(ValueError):
            script.replace_block(needle, "replacement")

    @given(
        prefix=st.text(min_size=0, max_size=20),
        marker=st.text(min_size=1, max_size=10),
        suffix=st.text(min_size=0, max_size=20),
    )
    @settings(max_examples=50)
    def test_replace_block_is_not_in_place(
        self, prefix: str, marker: str, suffix: str
    ) -> None:
        """Property: replace_block never mutates the original."""
        content = prefix + marker + suffix
        script = _make_solution_script(content=content)
        original_content = script.content
        script.replace_block(marker, "REPLACED")
        assert script.content == original_content


# ===========================================================================
# SolutionScript Serialization
# ===========================================================================


@pytest.mark.unit
class TestSolutionScriptSerialization:
    """SolutionScript supports JSON round-trip serialization."""

    def test_round_trip_required_fields_only(self) -> None:
        """Serialize and deserialize with required fields; all preserved."""
        original = _make_solution_script()
        json_str = original.model_dump_json()
        restored = SolutionScript.model_validate_json(json_str)

        assert restored.content == original.content
        assert restored.phase == original.phase
        assert restored.is_executable == original.is_executable
        assert restored.score == original.score
        assert restored.source_model == original.source_model

    def test_round_trip_all_fields(self) -> None:
        """Serialize and deserialize with all fields set; all preserved."""
        original = SolutionScript(
            content="print('hi')",
            phase=SolutionPhase.FINAL,
            score=0.99,
            is_executable=False,
            source_model="claude-opus",
        )
        json_str = original.model_dump_json()
        restored = SolutionScript.model_validate_json(json_str)

        assert restored.content == original.content
        assert restored.phase == original.phase
        assert restored.score == original.score
        assert restored.is_executable == original.is_executable
        assert restored.source_model == original.source_model

    def test_model_dump_json_returns_valid_json(self) -> None:
        """model_dump_json() produces parseable JSON."""
        script = _make_solution_script()
        parsed = json.loads(script.model_dump_json())
        assert isinstance(parsed, dict)
        assert "content" in parsed
        assert "phase" in parsed

    def test_created_at_survives_round_trip(self) -> None:
        """created_at datetime is preserved through serialization."""
        original = _make_solution_script()
        json_str = original.model_dump_json()
        restored = SolutionScript.model_validate_json(json_str)
        # Allow up to 1 second drift for serialization rounding
        delta = abs((restored.created_at - original.created_at).total_seconds())
        assert delta < 1.0


# ===========================================================================
# SolutionScript created_at edge cases
# ===========================================================================


@pytest.mark.unit
class TestSolutionScriptCreatedAt:
    """Tests for created_at auto-generation and behaviour."""

    def test_two_scripts_have_non_decreasing_created_at(self) -> None:
        """A script created later has created_at >= earlier script."""
        first = _make_solution_script()
        second = _make_solution_script()
        assert second.created_at >= first.created_at

    def test_explicit_created_at_is_respected(self) -> None:
        """Providing explicit created_at overrides auto-generation."""
        explicit_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        script = _make_solution_script(created_at=explicit_time)
        assert script.created_at == explicit_time

    def test_created_at_within_reasonable_range(self) -> None:
        """Auto-generated created_at is within last minute of now."""
        script = _make_solution_script()
        now = datetime.now(UTC)
        delta = now - script.created_at
        assert delta < timedelta(minutes=1)


# ===========================================================================
# REQ-DM-011: CodeBlockCategory Enum
# ===========================================================================


@pytest.mark.unit
class TestCodeBlockCategory:
    """CodeBlockCategory must be a StrEnum with exactly 8 values (REQ-DM-011)."""

    def test_is_string_enum(self) -> None:
        """CodeBlockCategory inherits from StrEnum."""
        assert issubclass(CodeBlockCategory, StrEnum)

    def test_member_count_is_eight(self) -> None:
        """Enum has exactly 8 members."""
        assert len(CodeBlockCategory) == 8

    @pytest.mark.parametrize("value", CODE_BLOCK_CATEGORY_VALUES)
    def test_contains_expected_value(self, value: str) -> None:
        """Each spec-mandated value is present in the enum."""
        member = CodeBlockCategory(value)
        assert member.value == value

    def test_values_match_spec_exactly(self) -> None:
        """The full set of values matches the spec with no extras."""
        actual = sorted(m.value for m in CodeBlockCategory)
        expected = sorted(CODE_BLOCK_CATEGORY_VALUES)
        assert actual == expected

    @pytest.mark.parametrize("value", CODE_BLOCK_CATEGORY_VALUES)
    def test_string_equality(self, value: str) -> None:
        """StrEnum members compare equal to their plain string value."""
        assert CodeBlockCategory(value) == value

    def test_invalid_value_raises(self) -> None:
        """Constructing with a non-existent value raises ValueError."""
        with pytest.raises(ValueError, match="not a valid"):
            CodeBlockCategory("nonexistent_category")


# ===========================================================================
# REQ-DM-012: CodeBlock Model
# ===========================================================================


@pytest.mark.unit
class TestCodeBlockConstruction:
    """CodeBlock has correct required and optional fields (REQ-DM-012)."""

    def test_valid_construction_with_content_only(self) -> None:
        """Constructing with only content succeeds."""
        block = CodeBlock(content="x = 1")
        assert block.content == "x = 1"

    def test_valid_construction_with_all_fields(self) -> None:
        """Constructing with all fields succeeds."""
        block = CodeBlock(
            content="from sklearn import RandomForest",
            category=CodeBlockCategory.MODEL_SELECTION,
            outer_step=3,
        )
        assert block.content == "from sklearn import RandomForest"
        assert block.category == CodeBlockCategory.MODEL_SELECTION
        assert block.outer_step == 3

    def test_category_defaults_to_none(self) -> None:
        """Category defaults to None when not provided."""
        block = CodeBlock(content="x = 1")
        assert block.category is None

    def test_outer_step_defaults_to_none(self) -> None:
        """outer_step defaults to None when not provided."""
        block = CodeBlock(content="x = 1")
        assert block.outer_step is None

    @pytest.mark.parametrize("category_value", CODE_BLOCK_CATEGORY_VALUES)
    def test_accepts_all_category_values(self, category_value: str) -> None:
        """CodeBlock accepts all valid CodeBlockCategory values."""
        block = CodeBlock(content="pass", category=CodeBlockCategory(category_value))
        assert block.category == category_value

    def test_invalid_category_raises_validation_error(self) -> None:
        """Invalid category string raises ValidationError."""
        with pytest.raises(ValidationError):
            CodeBlock(content="pass", category="bogus_category")  # type: ignore[arg-type]


@pytest.mark.unit
class TestCodeBlockRequiredFields:
    """CodeBlock raises ValidationError when required fields are missing."""

    def test_missing_content_raises(self) -> None:
        """Omitting content raises ValidationError."""
        with pytest.raises(ValidationError):
            CodeBlock()  # type: ignore[call-arg]

    def test_missing_content_with_optional_fields_raises(self) -> None:
        """Providing optional fields without content still raises."""
        with pytest.raises(ValidationError):
            CodeBlock(  # type: ignore[call-arg]
                category=CodeBlockCategory.TRAINING,
                outer_step=1,
            )


@pytest.mark.unit
class TestCodeBlockFrozen:
    """CodeBlock is frozen (immutable) per spec."""

    def test_cannot_mutate_content(self) -> None:
        """Assignment to content raises an error."""
        block = CodeBlock(content="x = 1")
        with pytest.raises(ValidationError):
            block.content = "x = 2"  # type: ignore[misc]

    def test_cannot_mutate_category(self) -> None:
        """Assignment to category raises an error."""
        block = CodeBlock(content="x = 1", category=CodeBlockCategory.TRAINING)
        with pytest.raises(ValidationError):
            block.category = CodeBlockCategory.ENSEMBLE  # type: ignore[misc]

    def test_cannot_mutate_outer_step(self) -> None:
        """Assignment to outer_step raises an error."""
        block = CodeBlock(content="x = 1", outer_step=1)
        with pytest.raises(ValidationError):
            block.outer_step = 2  # type: ignore[misc]


@pytest.mark.unit
class TestCodeBlockSerialization:
    """CodeBlock supports JSON round-trip serialization."""

    def test_round_trip_content_only(self) -> None:
        """Round-trip with content only; all fields preserved."""
        original = CodeBlock(content="x = 1")
        json_str = original.model_dump_json()
        restored = CodeBlock.model_validate_json(json_str)
        assert restored.content == original.content
        assert restored.category is None
        assert restored.outer_step is None

    def test_round_trip_all_fields(self) -> None:
        """Round-trip with all fields set; all fields preserved."""
        original = CodeBlock(
            content="model.fit(X, y)",
            category=CodeBlockCategory.TRAINING,
            outer_step=2,
        )
        json_str = original.model_dump_json()
        restored = CodeBlock.model_validate_json(json_str)
        assert restored.content == original.content
        assert restored.category == original.category
        assert restored.outer_step == original.outer_step

    def test_round_trip_equality(self) -> None:
        """Serialized then deserialized object equals the original."""
        original = CodeBlock(
            content="pass",
            category=CodeBlockCategory.OTHER,
            outer_step=5,
        )
        restored = CodeBlock.model_validate_json(original.model_dump_json())
        assert original == restored

    def test_model_dump_json_returns_valid_json(self) -> None:
        """model_dump_json() produces parseable JSON."""
        block = CodeBlock(content="x = 1")
        parsed = json.loads(block.model_dump_json())
        assert isinstance(parsed, dict)
        assert "content" in parsed


# ===========================================================================
# Property-based: SolutionScript and CodeBlock with Hypothesis
# ===========================================================================


@pytest.mark.unit
class TestSolutionScriptPropertyBased:
    """Property-based tests for SolutionScript using Hypothesis."""

    @given(
        content=st.text(min_size=1, max_size=200),
        phase=st.sampled_from(list(SolutionPhase)),
        score=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
        is_executable=st.booleans(),
        source_model=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
    )
    @settings(max_examples=50)
    def test_any_valid_inputs_produce_valid_script(
        self,
        content: str,
        phase: SolutionPhase,
        score: float | None,
        is_executable: bool,
        source_model: str | None,
    ) -> None:
        """Property: any combination of valid typed inputs creates a valid script."""
        script = SolutionScript(
            content=content,
            phase=phase,
            score=score,
            is_executable=is_executable,
            source_model=source_model,
        )
        assert script.content == content
        assert script.phase == phase
        assert script.score == score
        assert script.is_executable == is_executable
        assert script.source_model == source_model

    @given(
        content=st.text(min_size=1, max_size=100),
        phase=st.sampled_from(list(SolutionPhase)),
    )
    @settings(max_examples=30)
    def test_created_at_always_populated(
        self, content: str, phase: SolutionPhase
    ) -> None:
        """Property: created_at is always populated as a datetime."""
        script = SolutionScript(content=content, phase=phase)
        assert isinstance(script.created_at, datetime)

    @given(
        content=st.text(min_size=1, max_size=100),
        phase=st.sampled_from(list(SolutionPhase)),
        score=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
    )
    @settings(max_examples=30)
    def test_round_trip_preserves_content_and_phase(
        self, content: str, phase: SolutionPhase, score: float | None
    ) -> None:
        """Property: JSON round-trip preserves content, phase, and score."""
        original = SolutionScript(content=content, phase=phase, score=score)
        restored = SolutionScript.model_validate_json(original.model_dump_json())
        assert restored.content == original.content
        assert restored.phase == original.phase
        assert restored.score == original.score


@pytest.mark.unit
class TestCodeBlockPropertyBased:
    """Property-based tests for CodeBlock using Hypothesis."""

    @given(
        content=st.text(min_size=1, max_size=200),
        category=st.one_of(st.none(), st.sampled_from(list(CodeBlockCategory))),
        outer_step=st.one_of(st.none(), st.integers()),
    )
    @settings(max_examples=50)
    def test_any_valid_inputs_produce_valid_block(
        self,
        content: str,
        category: CodeBlockCategory | None,
        outer_step: int | None,
    ) -> None:
        """Property: any combination of valid typed inputs creates a valid block."""
        block = CodeBlock(content=content, category=category, outer_step=outer_step)
        assert block.content == content
        assert block.category == category
        assert block.outer_step == outer_step

    @given(
        content=st.text(min_size=1, max_size=100),
        category=st.one_of(st.none(), st.sampled_from(list(CodeBlockCategory))),
        outer_step=st.one_of(st.none(), st.integers(min_value=0, max_value=1000)),
    )
    @settings(max_examples=30)
    def test_round_trip_preserves_all_fields(
        self,
        content: str,
        category: CodeBlockCategory | None,
        outer_step: int | None,
    ) -> None:
        """Property: JSON round-trip preserves all CodeBlock fields."""
        original = CodeBlock(content=content, category=category, outer_step=outer_step)
        restored = CodeBlock.model_validate_json(original.model_dump_json())
        assert restored == original
