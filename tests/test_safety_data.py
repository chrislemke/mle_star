"""Tests for the data contamination agent (Task 21).

Validates ``parse_data_agent_response`` and ``check_data_usage`` which
implement the A_data agent for verifying that a solution uses all provided
data sources and incorporating unused information.

Tests are written TDD-first and serve as the executable specification for
REQ-SF-024 through REQ-SF-031.

Refs:
    SRS 03c (Safety Data), IMPLEMENTATION_PLAN.md Task 21.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    AgentType,
    DataModality,
    MetricDirection,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
    TaskType,
)
import pytest

# ---------------------------------------------------------------------------
# Helpers -- factory functions for building valid model instances
# ---------------------------------------------------------------------------


def _make_solution(**overrides: Any) -> SolutionScript:
    """Build a valid SolutionScript with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed SolutionScript instance.
    """
    defaults: dict[str, Any] = {
        "content": (
            "import pandas as pd\n"
            "df_train = pd.read_csv('train.csv')\n"
            "df_test = pd.read_csv('test.csv')\n"
            'print(f"Final Validation Performance: {0.85}")\n'
        ),
        "phase": SolutionPhase.MERGED,
    }
    defaults.update(overrides)
    return SolutionScript(**defaults)


def _make_task(**overrides: Any) -> TaskDescription:
    """Build a valid TaskDescription with sensible defaults.

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
        "data_dir": "/tmp/test_data",
        "output_dir": "./final",
    }
    defaults.update(overrides)
    return TaskDescription(**defaults)


_SAFETY = "mle_star.safety"

_ALL_USED_RESPONSE = "All the provided information is used."

_IMPROVED_CODE = (
    "import pandas as pd\n"
    "df_train = pd.read_csv('train.csv')\n"
    "df_test = pd.read_csv('test.csv')\n"
    "df_extra = pd.read_csv('extra_data.csv')\n"
    'print(f"Final Validation Performance: {0.90}")'
)

_IMPROVED_CODE_RESPONSE = f"```python\n{_IMPROVED_CODE}\n```"


# ===========================================================================
# REQ-SF-024: parse_data_agent_response -- All Info Used
# ===========================================================================


@pytest.mark.unit
class TestParseDataAgentResponseAllUsed:
    """parse_data_agent_response returns original when all info is used (REQ-SF-024)."""

    def test_exact_match_returns_original_unchanged(self) -> None:
        """Response 'All the provided information is used.' returns original solution."""
        from mle_star.safety import parse_data_agent_response

        solution = _make_solution()
        result = parse_data_agent_response(_ALL_USED_RESPONSE, solution)

        assert result is solution

    def test_case_insensitive_match(self) -> None:
        """Case-insensitive match: 'all the provided information is used.' works."""
        from mle_star.safety import parse_data_agent_response

        solution = _make_solution()
        result = parse_data_agent_response(
            "all the provided information is used.", solution
        )

        assert result is solution

    def test_mixed_case_match(self) -> None:
        """Mixed case variant: 'ALL THE PROVIDED INFORMATION IS USED.' works."""
        from mle_star.safety import parse_data_agent_response

        solution = _make_solution()
        result = parse_data_agent_response(
            "ALL THE PROVIDED INFORMATION IS USED.", solution
        )

        assert result is solution

    def test_phrase_embedded_in_longer_response(self) -> None:
        """Phrase embedded in a longer response still matches."""
        from mle_star.safety import parse_data_agent_response

        solution = _make_solution()
        response = (
            "After reviewing the code, I can confirm that "
            "All the provided information is used. "
            "No changes needed."
        )
        result = parse_data_agent_response(response, solution)

        assert result is solution


# ===========================================================================
# REQ-SF-025: parse_data_agent_response -- Code Block Extraction
# ===========================================================================


@pytest.mark.unit
class TestParseDataAgentResponseCodeBlock:
    """parse_data_agent_response extracts code blocks when info is not all used (REQ-SF-025)."""

    def test_response_with_code_block_extracts_code(self) -> None:
        """Response with ```python fence extracts code and returns new SolutionScript."""
        from mle_star.safety import parse_data_agent_response

        solution = _make_solution()
        result = parse_data_agent_response(_IMPROVED_CODE_RESPONSE, solution)

        assert result is not solution
        assert result.content == _IMPROVED_CODE

    def test_extracted_code_preserves_original_phase(self) -> None:
        """Extracted code preserves the original solution's phase field."""
        from mle_star.safety import parse_data_agent_response

        for phase in SolutionPhase:
            solution = _make_solution(phase=phase)
            result = parse_data_agent_response(_IMPROVED_CODE_RESPONSE, solution)

            assert result.phase == phase

    def test_response_without_code_fence_uses_stripped_text(self) -> None:
        """Response without code fences returns stripped response as code content."""
        from mle_star.safety import parse_data_agent_response

        raw_code = "import os\nprint('hello')"
        solution = _make_solution()
        result = parse_data_agent_response(raw_code, solution)

        assert result is not solution
        assert result.content == raw_code

    def test_new_solution_is_different_instance(self) -> None:
        """When code is extracted, a new SolutionScript instance is returned."""
        from mle_star.safety import parse_data_agent_response

        solution = _make_solution()
        result = parse_data_agent_response(_IMPROVED_CODE_RESPONSE, solution)

        assert result is not solution
        assert isinstance(result, SolutionScript)

    def test_generic_code_fence_without_language_tag(self) -> None:
        """Response with ``` fence (no language tag) also extracts code."""
        from mle_star.safety import parse_data_agent_response

        code = "import numpy as np\nprint(np.array([1,2,3]))"
        response = f"```\n{code}\n```"
        solution = _make_solution()
        result = parse_data_agent_response(response, solution)

        assert result.content == code


# ===========================================================================
# REQ-SF-026: parse_data_agent_response -- Metadata Preservation
# ===========================================================================


@pytest.mark.unit
class TestParseDataAgentResponseMetadata:
    """parse_data_agent_response preserves original solution metadata (REQ-SF-026)."""

    def test_preserves_phase_init(self) -> None:
        """Phase INIT is preserved in the returned solution."""
        from mle_star.safety import parse_data_agent_response

        solution = _make_solution(phase=SolutionPhase.INIT)
        result = parse_data_agent_response(_IMPROVED_CODE_RESPONSE, solution)

        assert result.phase == SolutionPhase.INIT

    def test_preserves_phase_merged(self) -> None:
        """Phase MERGED is preserved in the returned solution."""
        from mle_star.safety import parse_data_agent_response

        solution = _make_solution(phase=SolutionPhase.MERGED)
        result = parse_data_agent_response(_IMPROVED_CODE_RESPONSE, solution)

        assert result.phase == SolutionPhase.MERGED

    def test_preserves_phase_refined(self) -> None:
        """Phase REFINED is preserved in the returned solution."""
        from mle_star.safety import parse_data_agent_response

        solution = _make_solution(phase=SolutionPhase.REFINED)
        result = parse_data_agent_response(_IMPROVED_CODE_RESPONSE, solution)

        assert result.phase == SolutionPhase.REFINED

    def test_preserves_phase_ensemble(self) -> None:
        """Phase ENSEMBLE is preserved in the returned solution."""
        from mle_star.safety import parse_data_agent_response

        solution = _make_solution(phase=SolutionPhase.ENSEMBLE)
        result = parse_data_agent_response(_IMPROVED_CODE_RESPONSE, solution)

        assert result.phase == SolutionPhase.ENSEMBLE


# ===========================================================================
# REQ-SF-027: check_data_usage -- Is Async
# ===========================================================================


@pytest.mark.unit
class TestCheckDataUsageIsAsync:
    """check_data_usage is an async function (REQ-SF-027)."""

    def test_is_coroutine_function(self) -> None:
        """check_data_usage is defined as an async function."""
        from mle_star.safety import check_data_usage

        assert asyncio.iscoroutinefunction(check_data_usage)


# ===========================================================================
# REQ-SF-028: check_data_usage -- All Info Used Returns Original
# ===========================================================================


@pytest.mark.unit
class TestCheckDataUsageAllInfoUsed:
    """check_data_usage returns original when agent says all info used (REQ-SF-028)."""

    async def test_returns_original_when_all_used(self) -> None:
        """When agent response says all info is used, returns original solution."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_ALL_USED_RESPONSE)

        solution = _make_solution(content="my_original_code")

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "data prompt"
            mock_registry.get.return_value = mock_template

            result = await check_data_usage(solution, _make_task(), client)

        assert result is solution
        assert result.content == "my_original_code"

    async def test_only_one_agent_call_when_all_used(self) -> None:
        """When all info is used, only one call to the agent is made."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_ALL_USED_RESPONSE)

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            await check_data_usage(_make_solution(), _make_task(), client)

        assert client.send_message.call_count == 1


# ===========================================================================
# REQ-SF-028: check_data_usage -- Corrected Code Returned
# ===========================================================================


@pytest.mark.unit
class TestCheckDataUsageCorrectedCode:
    """check_data_usage returns new SolutionScript with corrected code (REQ-SF-028)."""

    async def test_returns_new_solution_with_corrected_code(self) -> None:
        """When agent returns code block, returns new SolutionScript with that code."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_IMPROVED_CODE_RESPONSE)

        solution = _make_solution()

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_data_usage(solution, _make_task(), client)

        assert result is not solution
        assert result.content == _IMPROVED_CODE

    async def test_corrected_solution_preserves_phase(self) -> None:
        """Corrected solution preserves the original solution's phase."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_IMPROVED_CODE_RESPONSE)

        for phase in (SolutionPhase.MERGED, SolutionPhase.INIT, SolutionPhase.REFINED):
            with patch(
                f"{_SAFETY}.PromptRegistry",
            ) as mock_registry_cls:
                mock_registry = mock_registry_cls.return_value
                mock_template = AsyncMock()
                mock_template.render = lambda **kwargs: "prompt"
                mock_registry.get.return_value = mock_template

                result = await check_data_usage(
                    _make_solution(phase=phase), _make_task(), client
                )

            assert result.phase == phase


# ===========================================================================
# REQ-SF-029: check_data_usage -- Graceful Degradation
# ===========================================================================


@pytest.mark.unit
class TestCheckDataUsageGracefulDegradation:
    """check_data_usage returns original on agent failure (REQ-SF-029)."""

    async def test_runtime_error_returns_original(self) -> None:
        """When agent raises RuntimeError, original solution is returned."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=RuntimeError("API down"))

        solution = _make_solution(content="original_safe_code")

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_data_usage(solution, _make_task(), client)

        assert result.content == "original_safe_code"

    async def test_timeout_error_returns_original(self) -> None:
        """When agent raises TimeoutError, original solution is returned."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=TimeoutError("Timed out"))

        solution = _make_solution(content="timeout_safe_code")

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_data_usage(solution, _make_task(), client)

        assert result.content == "timeout_safe_code"

    async def test_generic_exception_returns_original(self) -> None:
        """When agent raises a generic Exception, original solution is returned."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(
            side_effect=Exception("Unexpected internal error")
        )

        solution = _make_solution(content="exception_safe_code")

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_data_usage(solution, _make_task(), client)

        assert result.content == "exception_safe_code"

    async def test_exception_does_not_propagate(self) -> None:
        """Exceptions from the client do not propagate to the caller."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=Exception("Catastrophic failure"))

        solution = _make_solution()

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            # Should NOT raise
            result = await check_data_usage(solution, _make_task(), client)

        assert isinstance(result, SolutionScript)

    async def test_keyboard_interrupt_not_caught(self) -> None:
        """KeyboardInterrupt is not caught and propagates normally."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=KeyboardInterrupt)

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            with pytest.raises(KeyboardInterrupt):
                await check_data_usage(_make_solution(), _make_task(), client)

    async def test_registry_key_error_returns_original(self) -> None:
        """When PromptRegistry raises KeyError, original solution is returned."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_registry.get.side_effect = KeyError("No template for data")

            solution = _make_solution(content="registry_safe_code")
            result = await check_data_usage(solution, _make_task(), client)

        assert result.content == "registry_safe_code"


# ===========================================================================
# REQ-SF-030: check_data_usage -- Prompt Registry Integration
# ===========================================================================


@pytest.mark.unit
class TestCheckDataUsagePromptRegistry:
    """check_data_usage loads data template from PromptRegistry (REQ-SF-030)."""

    async def test_registry_get_called_with_agent_type_data(self) -> None:
        """PromptRegistry.get is called with AgentType.DATA."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_ALL_USED_RESPONSE)

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            await check_data_usage(_make_solution(), _make_task(), client)

        mock_registry.get.assert_called_once_with(AgentType.DATA)

    async def test_prompt_rendered_with_initial_solution(self) -> None:
        """Prompt template is rendered with solution.content as initial_solution."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_ALL_USED_RESPONSE)

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            solution = _make_solution(content="unique_solution_marker_42")
            await check_data_usage(solution, _make_task(), client)

        assert len(render_kwargs_captured) == 1
        assert (
            render_kwargs_captured[0].get("initial_solution")
            == "unique_solution_marker_42"
        )

    async def test_prompt_rendered_with_task_description(self) -> None:
        """Prompt template is rendered with task.description as task_description."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_ALL_USED_RESPONSE)

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            task = _make_task(description="Unique task description marker 99")
            await check_data_usage(_make_solution(), task, client)

        assert len(render_kwargs_captured) == 1
        assert (
            render_kwargs_captured[0].get("task_description")
            == "Unique task description marker 99"
        )

    async def test_rendered_prompt_sent_to_client(self) -> None:
        """The rendered prompt is sent via client.send_message."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_ALL_USED_RESPONSE)

        expected_prompt = "rendered data agent prompt xyz"

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: expected_prompt
            mock_registry.get.return_value = mock_template

            await check_data_usage(_make_solution(), _make_task(), client)

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("message") == expected_prompt

    async def test_client_invoked_with_data_agent_type(self) -> None:
        """Client.send_message is invoked with agent_type=str(AgentType.DATA)."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_ALL_USED_RESPONSE)

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            await check_data_usage(_make_solution(), _make_task(), client)

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("agent_type") == str(AgentType.DATA)


# ===========================================================================
# REQ-SF-031: check_data_usage -- Return Type
# ===========================================================================


@pytest.mark.unit
class TestCheckDataUsageReturnType:
    """check_data_usage always returns a SolutionScript (REQ-SF-031)."""

    async def test_returns_solution_script_on_all_used(self) -> None:
        """Returns a SolutionScript instance when all info is used."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_ALL_USED_RESPONSE)

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_data_usage(_make_solution(), _make_task(), client)

        assert isinstance(result, SolutionScript)

    async def test_returns_solution_script_on_corrected_code(self) -> None:
        """Returns a SolutionScript instance when corrected code is returned."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_IMPROVED_CODE_RESPONSE)

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_data_usage(_make_solution(), _make_task(), client)

        assert isinstance(result, SolutionScript)

    async def test_returns_solution_script_on_failure(self) -> None:
        """Returns a SolutionScript instance even on agent failure."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=RuntimeError("boom"))

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_data_usage(_make_solution(), _make_task(), client)

        assert isinstance(result, SolutionScript)


# ===========================================================================
# Prompt template integration tests
# ===========================================================================


@pytest.mark.unit
class TestDataPromptTemplate:
    """Validate that the data prompt template exists and renders correctly."""

    def test_data_template_exists_in_registry(self) -> None:
        """PromptRegistry contains a data template (no variant)."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.DATA)
        assert template.agent_type == AgentType.DATA

    def test_data_template_has_initial_solution_variable(self) -> None:
        """Data template declares 'initial_solution' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.DATA)
        assert "initial_solution" in template.variables

    def test_data_template_has_task_description_variable(self) -> None:
        """Data template declares 'task_description' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.DATA)
        assert "task_description" in template.variables

    def test_data_template_renders_with_variables(self) -> None:
        """Data template renders successfully with both required variables."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.DATA)
        rendered = template.render(
            initial_solution="print('hello')",
            task_description="Classify images",
            target_column="Not specified",
        )
        assert "print('hello')" in rendered
        assert "Classify images" in rendered

    def test_data_template_mentions_do_not_use_try_except(self) -> None:
        """Data template contains instruction about not using try-except (REQ-SF-029)."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.DATA)
        rendered = template.render(
            initial_solution="code",
            task_description="task",
            target_column="Not specified",
        )
        assert "TRY AND EXCEPT" in rendered or "try-except" in rendered.lower()

    def test_data_template_mentions_all_provided_information(self) -> None:
        """Data template mentions the key phrase for the 'all used' response."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.DATA)
        rendered = template.render(
            initial_solution="code",
            task_description="task",
            target_column="Not specified",
        )
        assert "All the provided information is used" in rendered


# ===========================================================================
# Edge cases
# ===========================================================================


@pytest.mark.unit
class TestParseDataAgentResponseEdgeCases:
    """Edge case tests for parse_data_agent_response."""

    def test_empty_response_treated_as_code(self) -> None:
        """Empty response (no 'all used' phrase) returns new solution with empty content."""
        from mle_star.safety import parse_data_agent_response

        solution = _make_solution()
        result = parse_data_agent_response("", solution)

        # Empty string doesn't contain the "all used" phrase, so
        # extract_code_block("") returns "" and a new solution is created
        assert result is not solution
        assert result.content == ""

    def test_whitespace_only_response_treated_as_code(self) -> None:
        """Whitespace-only response returns new solution with empty content."""
        from mle_star.safety import parse_data_agent_response

        solution = _make_solution()
        result = parse_data_agent_response("   \n\n  ", solution)

        assert result is not solution
        assert result.content == ""

    def test_response_with_multiple_code_blocks_returns_longest(self) -> None:
        """Response with multiple code blocks returns the longest one."""
        from mle_star.safety import parse_data_agent_response

        short_code = "x = 1"
        long_code = (
            "import pandas as pd\ndf = pd.read_csv('train.csv')\nprint(df.shape)"
        )
        response = (
            f"```python\n{short_code}\n```\nSome text\n```python\n{long_code}\n```"
        )

        solution = _make_solution()
        result = parse_data_agent_response(response, solution)

        assert result.content == long_code

    def test_solution_with_score_field_preserved_on_all_used(self) -> None:
        """Solution score is preserved when all info is used (identity return)."""
        from mle_star.safety import parse_data_agent_response

        solution = _make_solution(score=0.95)
        result = parse_data_agent_response(_ALL_USED_RESPONSE, solution)

        assert result is solution
        assert result.score == pytest.approx(0.95)


@pytest.mark.unit
class TestCheckDataUsageEdgeCases:
    """Edge case tests for check_data_usage."""

    async def test_empty_solution_content(self) -> None:
        """Works correctly with an empty solution content string."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_ALL_USED_RESPONSE)

        solution = _make_solution(content="")

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_data_usage(solution, _make_task(), client)

        assert isinstance(result, SolutionScript)

    async def test_solution_with_score_preserved_on_all_used(self) -> None:
        """Solution score attribute is preserved when all info is used."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_ALL_USED_RESPONSE)

        solution = _make_solution(score=0.88)

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_data_usage(solution, _make_task(), client)

        assert result.score == pytest.approx(0.88)

    async def test_runs_exactly_once(self) -> None:
        """The data agent invokes the client exactly once per call."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_IMPROVED_CODE_RESPONSE)

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            await check_data_usage(_make_solution(), _make_task(), client)

        assert client.send_message.call_count == 1


# ===========================================================================
# Property-based tests
# ===========================================================================


@pytest.mark.unit
class TestParseDataAgentResponsePropertyBased:
    """Property-based tests for parse_data_agent_response invariants."""

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                whitelist_characters="_= \n",
            ),
            min_size=5,
            max_size=200,
        ),
    )
    @settings(max_examples=30)
    def test_all_used_response_always_returns_same_instance(self, content: str) -> None:
        """When response says all info is used, the exact same instance is returned."""
        from mle_star.safety import parse_data_agent_response

        solution = _make_solution(content=content)
        result = parse_data_agent_response(_ALL_USED_RESPONSE, solution)

        assert result is solution

    @given(
        code=st.text(min_size=1, max_size=200).filter(
            lambda s: (
                "```" not in s
                and "all the provided information is used" not in s.lower()
            )
        ),
    )
    @settings(max_examples=30)
    def test_code_response_returns_different_instance(self, code: str) -> None:
        """When response contains code (not the 'all used' phrase), returns a new instance."""
        from mle_star.safety import parse_data_agent_response

        response = f"```python\n{code}\n```"
        solution = _make_solution()
        result = parse_data_agent_response(response, solution)

        assert result is not solution
        assert result.content == code.strip()

    @given(phase=st.sampled_from(list(SolutionPhase)))
    @settings(max_examples=10)
    def test_phase_preserved_across_all_phases(self, phase: SolutionPhase) -> None:
        """Phase is always preserved regardless of which phase value is used."""
        from mle_star.safety import parse_data_agent_response

        solution = _make_solution(phase=phase)
        result = parse_data_agent_response(_IMPROVED_CODE_RESPONSE, solution)

        assert result.phase == phase


@pytest.mark.unit
class TestCheckDataUsagePropertyBased:
    """Property-based tests for check_data_usage invariants."""

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                whitelist_characters="_= \n",
            ),
            min_size=5,
            max_size=200,
        ),
    )
    @settings(max_examples=30)
    async def test_exception_always_returns_solution_script(self, content: str) -> None:
        """On any exception, a SolutionScript is always returned (graceful degradation)."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=RuntimeError("Random failure"))

        solution = _make_solution(content=content)

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_data_usage(solution, _make_task(), client)

        assert isinstance(result, SolutionScript)
        assert result.content == content

    @given(phase=st.sampled_from(list(SolutionPhase)))
    @settings(max_examples=10)
    async def test_phase_preserved_on_all_used(self, phase: SolutionPhase) -> None:
        """Solution phase is preserved when all info is used, for any phase."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_ALL_USED_RESPONSE)

        solution = _make_solution(phase=phase)

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_data_usage(solution, _make_task(), client)

        assert result.phase == phase

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                whitelist_characters="_= \n",
            ),
            min_size=5,
            max_size=200,
        ),
    )
    @settings(max_examples=30)
    async def test_all_used_returns_identical_content(self, content: str) -> None:
        """When all info is used, returned content equals original (identity)."""
        from mle_star.safety import check_data_usage

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_ALL_USED_RESPONSE)

        solution = _make_solution(content=content)

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_data_usage(solution, _make_task(), client)

        assert result.content == content
