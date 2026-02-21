"""Tests for subsampling utilities in execution module (Task 16).

Validates ``SUBSAMPLE_INSTRUCTION``, ``get_subsample_instruction``,
``request_subsample_removal``, and ``request_subsample_extraction``
functions defined in ``src/mle_star/execution.py``.  Tests are written
TDD-first -- the implementation does not yet exist.  They serve as the
executable specification for REQ-EX-017 through REQ-EX-020.

Refs:
    SRS 02c (Subsampling Utilities), IMPLEMENTATION_PLAN.md Task 16.
"""

from __future__ import annotations

import re

from hypothesis import given, settings, strategies as st
from mle_star.execution import (
    SUBSAMPLE_INSTRUCTION,
    get_subsample_instruction,
    request_subsample_extraction,
    request_subsample_removal,
)
from mle_star.models import PipelineConfig, SolutionPhase, SolutionScript
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_solution(
    content: str, phase: SolutionPhase = SolutionPhase.INIT
) -> SolutionScript:
    """Create a SolutionScript with the given content for testing."""
    return SolutionScript(content=content, phase=phase)


# ===========================================================================
# REQ-EX-017: SUBSAMPLE_INSTRUCTION Constant
# ===========================================================================


@pytest.mark.unit
class TestSubsampleInstructionConstant:
    """SUBSAMPLE_INSTRUCTION is a string template with {limit} placeholder (REQ-EX-017)."""

    def test_is_a_string(self) -> None:
        """SUBSAMPLE_INSTRUCTION is a string type."""
        assert isinstance(SUBSAMPLE_INSTRUCTION, str)

    def test_contains_limit_placeholder(self) -> None:
        """SUBSAMPLE_INSTRUCTION contains the ``{limit}`` placeholder."""
        assert "{limit}" in SUBSAMPLE_INSTRUCTION

    def test_contains_training_samples_text(self) -> None:
        """SUBSAMPLE_INSTRUCTION contains 'training samples' in the message."""
        assert "training samples" in SUBSAMPLE_INSTRUCTION

    def test_contains_subsample_text(self) -> None:
        """SUBSAMPLE_INSTRUCTION contains 'subsample' in the message."""
        assert "subsample" in SUBSAMPLE_INSTRUCTION

    def test_contains_faster_run_text(self) -> None:
        """SUBSAMPLE_INSTRUCTION contains 'faster run' in the message."""
        assert "faster run" in SUBSAMPLE_INSTRUCTION

    def test_contains_more_than_text(self) -> None:
        """SUBSAMPLE_INSTRUCTION contains 'more than' phrase."""
        assert "more than" in SUBSAMPLE_INSTRUCTION

    def test_is_nonempty(self) -> None:
        """SUBSAMPLE_INSTRUCTION is not an empty string."""
        assert len(SUBSAMPLE_INSTRUCTION) > 0

    def test_can_be_formatted_with_limit(self) -> None:
        """SUBSAMPLE_INSTRUCTION can be formatted with a 'limit' keyword argument."""
        result = SUBSAMPLE_INSTRUCTION.format(limit=30000)
        assert isinstance(result, str)
        assert "{limit}" not in result

    def test_format_with_limit_produces_expected_text(self) -> None:
        """Formatting with limit=30000 produces the required text from the spec."""
        result = SUBSAMPLE_INSTRUCTION.format(limit=30000)
        expected_fragment = (
            "If there are more than 30000 training samples, "
            "you must subsample to 30000 for a faster run."
        )
        assert expected_fragment in result

    def test_spec_compliant_template_text(self) -> None:
        """SUBSAMPLE_INSTRUCTION contains the exact template phrase from the spec."""
        expected_template_fragment = (
            "If there are more than {limit} training samples, "
            "you must subsample to {limit} for a faster run."
        )
        assert expected_template_fragment in SUBSAMPLE_INSTRUCTION

    def test_limit_appears_exactly_twice(self) -> None:
        """The ``{limit}`` placeholder appears exactly twice in the template."""
        count = SUBSAMPLE_INSTRUCTION.count("{limit}")
        assert count == 2


# ===========================================================================
# REQ-EX-018: get_subsample_instruction -- Default Config
# ===========================================================================


@pytest.mark.unit
class TestGetSubsampleInstructionDefaultConfig:
    """get_subsample_instruction with default PipelineConfig returns '30000' (REQ-EX-018)."""

    def test_returns_string(self) -> None:
        """Return value is a string."""
        config = PipelineConfig()
        result = get_subsample_instruction(config)
        assert isinstance(result, str)

    def test_contains_30000(self) -> None:
        """Acceptance criterion: result contains '30000' for default config."""
        config = PipelineConfig()
        result = get_subsample_instruction(config)
        assert "30000" in result

    def test_does_not_contain_limit_placeholder(self) -> None:
        """Result does not contain the raw ``{limit}`` placeholder."""
        config = PipelineConfig()
        result = get_subsample_instruction(config)
        assert "{limit}" not in result

    def test_contains_training_samples_text(self) -> None:
        """Result contains 'training samples' in the rendered instruction."""
        config = PipelineConfig()
        result = get_subsample_instruction(config)
        assert "training samples" in result

    def test_contains_subsample_text(self) -> None:
        """Result contains 'subsample' in the rendered instruction."""
        config = PipelineConfig()
        result = get_subsample_instruction(config)
        assert "subsample" in result

    def test_contains_faster_run_text(self) -> None:
        """Result contains 'faster run' in the rendered instruction."""
        config = PipelineConfig()
        result = get_subsample_instruction(config)
        assert "faster run" in result

    def test_acceptance_criterion_full_text(self) -> None:
        """Result contains the full rendered specification text."""
        config = PipelineConfig()
        result = get_subsample_instruction(config)
        expected = (
            "If there are more than 30000 training samples, "
            "you must subsample to 30000 for a faster run."
        )
        assert expected in result


# ===========================================================================
# REQ-EX-018: get_subsample_instruction -- Custom subsample_limit
# ===========================================================================


@pytest.mark.unit
class TestGetSubsampleInstructionCustomLimit:
    """get_subsample_instruction with custom subsample_limit returns correct value (REQ-EX-018)."""

    def test_custom_limit_10000(self) -> None:
        """Config with subsample_limit=10000 returns string containing '10000'."""
        config = PipelineConfig(subsample_limit=10000)
        result = get_subsample_instruction(config)
        assert "10000" in result

    def test_custom_limit_50000(self) -> None:
        """Config with subsample_limit=50000 returns string containing '50000'."""
        config = PipelineConfig(subsample_limit=50000)
        result = get_subsample_instruction(config)
        assert "50000" in result

    def test_custom_limit_1(self) -> None:
        """Config with subsample_limit=1 (minimum) returns string containing '1'."""
        config = PipelineConfig(subsample_limit=1)
        result = get_subsample_instruction(config)
        assert "1" in result

    def test_custom_limit_100000(self) -> None:
        """Config with subsample_limit=100000 returns string containing '100000'."""
        config = PipelineConfig(subsample_limit=100000)
        result = get_subsample_instruction(config)
        assert "100000" in result

    def test_custom_limit_replaces_both_occurrences(self) -> None:
        """Both {limit} placeholders are replaced with the custom value."""
        config = PipelineConfig(subsample_limit=5000)
        result = get_subsample_instruction(config)
        # The limit should appear at least twice in the result (once for each placeholder)
        occurrences = len(re.findall(r"5000", result))
        assert occurrences >= 2

    def test_no_placeholder_remains(self) -> None:
        """No ``{limit}`` placeholder remains in the result."""
        config = PipelineConfig(subsample_limit=7777)
        result = get_subsample_instruction(config)
        assert "{limit}" not in result

    @pytest.mark.parametrize(
        "limit",
        [1, 100, 1000, 10000, 30000, 50000, 100000],
        ids=["1", "100", "1k", "10k", "30k", "50k", "100k"],
    )
    def test_various_limits_produce_correct_value(self, limit: int) -> None:
        """Various subsample_limit values all appear correctly in the result."""
        config = PipelineConfig(subsample_limit=limit)
        result = get_subsample_instruction(config)
        assert str(limit) in result


# ===========================================================================
# REQ-EX-018: get_subsample_instruction -- Property-Based Tests
# ===========================================================================


@pytest.mark.unit
class TestGetSubsampleInstructionPropertyBased:
    """Property-based tests for get_subsample_instruction using Hypothesis."""

    @given(
        limit=st.integers(min_value=1, max_value=10_000_000),
    )
    @settings(max_examples=50)
    def test_result_always_contains_limit_value(self, limit: int) -> None:
        """Property: result always contains the string representation of the limit."""
        config = PipelineConfig(subsample_limit=limit)
        result = get_subsample_instruction(config)
        assert str(limit) in result

    @given(
        limit=st.integers(min_value=1, max_value=10_000_000),
    )
    @settings(max_examples=50)
    def test_result_never_contains_placeholder(self, limit: int) -> None:
        """Property: result never contains the raw ``{limit}`` placeholder."""
        config = PipelineConfig(subsample_limit=limit)
        result = get_subsample_instruction(config)
        assert "{limit}" not in result

    @given(
        limit=st.integers(min_value=1, max_value=10_000_000),
    )
    @settings(max_examples=50)
    def test_result_always_is_string(self, limit: int) -> None:
        """Property: return value is always a string."""
        config = PipelineConfig(subsample_limit=limit)
        result = get_subsample_instruction(config)
        assert isinstance(result, str)

    @given(
        limit=st.integers(min_value=1, max_value=10_000_000),
    )
    @settings(max_examples=50)
    def test_result_always_nonempty(self, limit: int) -> None:
        """Property: result is always a non-empty string."""
        config = PipelineConfig(subsample_limit=limit)
        result = get_subsample_instruction(config)
        assert len(result) > 0

    @given(
        limit=st.integers(min_value=1, max_value=10_000_000),
    )
    @settings(max_examples=30)
    def test_limit_appears_at_least_twice(self, limit: int) -> None:
        """Property: the limit value appears at least twice (matching the two placeholders)."""
        config = PipelineConfig(subsample_limit=limit)
        result = get_subsample_instruction(config)
        count = result.count(str(limit))
        assert count >= 2


# ===========================================================================
# REQ-EX-019: request_subsample_removal -- Happy Path
# ===========================================================================


@pytest.mark.unit
class TestRequestSubsampleRemovalHappyPath:
    """request_subsample_removal returns a prompt for removing subsampling code (REQ-EX-019)."""

    def test_returns_string(self) -> None:
        """Return value is a string."""
        solution = _make_solution(
            "import pandas as pd\ndf = df.sample(1000)\nprint(df)"
        )
        result = request_subsample_removal(solution)
        assert isinstance(result, str)

    def test_result_is_nonempty(self) -> None:
        """Return value is not an empty string."""
        solution = _make_solution("df = df.sample(1000)")
        result = request_subsample_removal(solution)
        assert len(result) > 0

    def test_includes_solution_content(self) -> None:
        """Result includes the full solution script content."""
        content = "import pandas as pd\ndf = pd.read_csv('train.csv')\ndf = df.sample(1000)\nprint(df.shape)"
        solution = _make_solution(content)
        result = request_subsample_removal(solution)
        assert content in result

    def test_includes_removal_instruction(self) -> None:
        """Result includes instruction to remove subsampling code."""
        solution = _make_solution("df = df.head(100)")
        result = request_subsample_removal(solution)
        lower_result = result.lower()
        assert "remove" in lower_result or "remov" in lower_result

    def test_includes_subsampling_reference(self) -> None:
        """Result references subsampling in the prompt."""
        solution = _make_solution("df = df.sample(500)")
        result = request_subsample_removal(solution)
        lower_result = result.lower()
        assert (
            "subsample" in lower_result
            or "subsampling" in lower_result
            or "sample" in lower_result
        )

    def test_includes_preserve_functionality_instruction(self) -> None:
        """Result instructs to preserve other functionality when removing subsampling."""
        solution = _make_solution("df = df.sample(500)\nmodel.fit(df)")
        result = request_subsample_removal(solution)
        lower_result = result.lower()
        assert (
            "preserv" in lower_result
            or "other" in lower_result
            or "functionality" in lower_result
        )

    def test_includes_full_script_instruction(self) -> None:
        """Result instructs to return the full modified script."""
        solution = _make_solution("x = 1\ndf = df.sample(100)\ny = 2")
        result = request_subsample_removal(solution)
        lower_result = result.lower()
        assert (
            "full" in lower_result
            or "script" in lower_result
            or "modified" in lower_result
        )


# ===========================================================================
# REQ-EX-019: request_subsample_removal -- Content Inclusion
# ===========================================================================


@pytest.mark.unit
class TestRequestSubsampleRemovalContentInclusion:
    """request_subsample_removal includes the full solution content in the prompt (REQ-EX-019)."""

    def test_simple_content_included(self) -> None:
        """A simple single-line script is included in the prompt."""
        content = "print('hello world')"
        solution = _make_solution(content)
        result = request_subsample_removal(solution)
        assert content in result

    def test_multiline_content_included(self) -> None:
        """A multi-line script is fully included in the prompt."""
        content = (
            "import pandas as pd\n"
            "import numpy as np\n"
            "df = pd.read_csv('train.csv')\n"
            "df = df.sample(frac=0.1)\n"
            "model = RandomForestClassifier()\n"
            "model.fit(df.drop('target', axis=1), df['target'])"
        )
        solution = _make_solution(content)
        result = request_subsample_removal(solution)
        assert content in result

    def test_content_with_special_characters_included(self) -> None:
        """Content with special characters is included verbatim."""
        content = (
            "# Comment with special chars: @#$%^&*()\nresult = df[df['col'] > 0.5]"
        )
        solution = _make_solution(content)
        result = request_subsample_removal(solution)
        assert content in result


# ===========================================================================
# REQ-EX-019: request_subsample_removal -- Phase Variations
# ===========================================================================


@pytest.mark.unit
class TestRequestSubsampleRemovalPhaseVariations:
    """request_subsample_removal works with solutions from any phase (REQ-EX-019)."""

    @pytest.mark.parametrize(
        "phase",
        list(SolutionPhase),
        ids=[p.value for p in SolutionPhase],
    )
    def test_all_phases_produce_valid_prompt(self, phase: SolutionPhase) -> None:
        """SolutionScript from any phase produces a valid removal prompt."""
        content = f"# phase: {phase.value}\ndf = df.sample(100)"
        solution = _make_solution(content, phase=phase)
        result = request_subsample_removal(solution)
        assert isinstance(result, str)
        assert content in result
        assert len(result) > len(content)


# ===========================================================================
# REQ-EX-019: request_subsample_removal -- Property-Based Tests
# ===========================================================================


@pytest.mark.unit
class TestRequestSubsampleRemovalPropertyBased:
    """Property-based tests for request_subsample_removal using Hypothesis."""

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                whitelist_characters="_=+\n #()',.",
            ),
            min_size=1,
            max_size=500,
        ).filter(lambda s: s.strip()),
    )
    @settings(max_examples=50)
    def test_result_always_contains_solution_content(self, content: str) -> None:
        """Property: the result always contains the full solution content."""
        solution = _make_solution(content)
        result = request_subsample_removal(solution)
        assert content in result

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters="_=\n ",
            ),
            min_size=1,
            max_size=200,
        ).filter(lambda s: s.strip()),
    )
    @settings(max_examples=50)
    def test_result_is_always_longer_than_content(self, content: str) -> None:
        """Property: the prompt is always longer than just the content (adds instructions)."""
        solution = _make_solution(content)
        result = request_subsample_removal(solution)
        assert len(result) > len(content)

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters="_=\n ",
            ),
            min_size=1,
            max_size=200,
        ).filter(lambda s: s.strip()),
    )
    @settings(max_examples=50)
    def test_result_is_always_a_string(self, content: str) -> None:
        """Property: return value is always a string."""
        solution = _make_solution(content)
        result = request_subsample_removal(solution)
        assert isinstance(result, str)


# ===========================================================================
# REQ-EX-020: request_subsample_extraction -- Happy Path
# ===========================================================================


@pytest.mark.unit
class TestRequestSubsampleExtractionHappyPath:
    """request_subsample_extraction returns a prompt for extracting subsampling code (REQ-EX-020)."""

    def test_returns_string(self) -> None:
        """Return value is a string."""
        solution = _make_solution("df = df.sample(1000)")
        result = request_subsample_extraction(solution)
        assert isinstance(result, str)

    def test_result_is_nonempty(self) -> None:
        """Return value is not an empty string."""
        solution = _make_solution("df = df.sample(1000)")
        result = request_subsample_extraction(solution)
        assert len(result) > 0

    def test_includes_solution_content(self) -> None:
        """Result includes the full solution script content."""
        content = "import pandas as pd\ndf = pd.read_csv('train.csv')\ndf = df.sample(1000)\nprint(df.shape)"
        solution = _make_solution(content)
        result = request_subsample_extraction(solution)
        assert content in result

    def test_includes_extraction_instruction(self) -> None:
        """Result includes instruction to extract or identify subsampling code."""
        solution = _make_solution("df = df.head(100)")
        result = request_subsample_extraction(solution)
        lower_result = result.lower()
        assert "extract" in lower_result or "identify" in lower_result

    def test_includes_subsampling_reference(self) -> None:
        """Result references subsampling in the prompt."""
        solution = _make_solution("df = df.sample(500)")
        result = request_subsample_extraction(solution)
        lower_result = result.lower()
        assert (
            "subsample" in lower_result
            or "subsampling" in lower_result
            or "sample" in lower_result
        )


# ===========================================================================
# REQ-EX-020: request_subsample_extraction -- Content Inclusion
# ===========================================================================


@pytest.mark.unit
class TestRequestSubsampleExtractionContentInclusion:
    """request_subsample_extraction includes the full solution content (REQ-EX-020)."""

    def test_simple_content_included(self) -> None:
        """A simple single-line script is included in the prompt."""
        content = "print('hello world')"
        solution = _make_solution(content)
        result = request_subsample_extraction(solution)
        assert content in result

    def test_multiline_content_included(self) -> None:
        """A multi-line script is fully included in the prompt."""
        content = (
            "import pandas as pd\n"
            "import numpy as np\n"
            "df = pd.read_csv('train.csv')\n"
            "df = df.sample(frac=0.1)\n"
            "model = RandomForestClassifier()\n"
            "model.fit(df.drop('target', axis=1), df['target'])"
        )
        solution = _make_solution(content)
        result = request_subsample_extraction(solution)
        assert content in result

    def test_content_with_special_characters_included(self) -> None:
        """Content with special characters is included verbatim."""
        content = (
            "# Comment with special chars: @#$%^&*()\nresult = df[df['col'] > 0.5]"
        )
        solution = _make_solution(content)
        result = request_subsample_extraction(solution)
        assert content in result


# ===========================================================================
# REQ-EX-020: request_subsample_extraction -- Phase Variations
# ===========================================================================


@pytest.mark.unit
class TestRequestSubsampleExtractionPhaseVariations:
    """request_subsample_extraction works with solutions from any phase (REQ-EX-020)."""

    @pytest.mark.parametrize(
        "phase",
        list(SolutionPhase),
        ids=[p.value for p in SolutionPhase],
    )
    def test_all_phases_produce_valid_prompt(self, phase: SolutionPhase) -> None:
        """SolutionScript from any phase produces a valid extraction prompt."""
        content = f"# phase: {phase.value}\ndf = df.sample(100)"
        solution = _make_solution(content, phase=phase)
        result = request_subsample_extraction(solution)
        assert isinstance(result, str)
        assert content in result
        assert len(result) > len(content)


# ===========================================================================
# REQ-EX-020: request_subsample_extraction -- Property-Based Tests
# ===========================================================================


@pytest.mark.unit
class TestRequestSubsampleExtractionPropertyBased:
    """Property-based tests for request_subsample_extraction using Hypothesis."""

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                whitelist_characters="_=+\n #()',.",
            ),
            min_size=1,
            max_size=500,
        ).filter(lambda s: s.strip()),
    )
    @settings(max_examples=50)
    def test_result_always_contains_solution_content(self, content: str) -> None:
        """Property: the result always contains the full solution content."""
        solution = _make_solution(content)
        result = request_subsample_extraction(solution)
        assert content in result

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters="_=\n ",
            ),
            min_size=1,
            max_size=200,
        ).filter(lambda s: s.strip()),
    )
    @settings(max_examples=50)
    def test_result_is_always_longer_than_content(self, content: str) -> None:
        """Property: the prompt is always longer than just the content (adds instructions)."""
        solution = _make_solution(content)
        result = request_subsample_extraction(solution)
        assert len(result) > len(content)

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters="_=\n ",
            ),
            min_size=1,
            max_size=200,
        ).filter(lambda s: s.strip()),
    )
    @settings(max_examples=50)
    def test_result_is_always_a_string(self, content: str) -> None:
        """Property: return value is always a string."""
        solution = _make_solution(content)
        result = request_subsample_extraction(solution)
        assert isinstance(result, str)


# ===========================================================================
# Cross-Function: Removal vs Extraction Produce Different Prompts
# ===========================================================================


@pytest.mark.unit
class TestRemovalVsExtractionDiffer:
    """request_subsample_removal and request_subsample_extraction return different prompts."""

    def test_different_results_for_same_input(self) -> None:
        """Removal and extraction prompts differ for the same solution."""
        solution = _make_solution("df = df.sample(1000)\nmodel.fit(df)")
        removal_result = request_subsample_removal(solution)
        extraction_result = request_subsample_extraction(solution)
        assert removal_result != extraction_result

    def test_both_include_solution_content(self) -> None:
        """Both prompts include the same solution content."""
        content = "df = df.sample(500)\nprint(df.shape)"
        solution = _make_solution(content)
        removal_result = request_subsample_removal(solution)
        extraction_result = request_subsample_extraction(solution)
        assert content in removal_result
        assert content in extraction_result

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters="_=\n ",
            ),
            min_size=5,
            max_size=200,
        ).filter(lambda s: s.strip()),
    )
    @settings(max_examples=30)
    def test_property_removal_and_extraction_always_differ(self, content: str) -> None:
        """Property: removal and extraction prompts always differ for any content."""
        solution = _make_solution(content)
        removal_result = request_subsample_removal(solution)
        extraction_result = request_subsample_extraction(solution)
        assert removal_result != extraction_result
