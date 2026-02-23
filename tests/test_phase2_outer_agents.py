"""Tests for the Phase 2 outer loop summarize and extractor agents (Task 32).

Validates ``invoke_summarize``, ``validate_code_block``,
``invoke_extractor``, and ``_format_previous_blocks`` which implement
A_summarize and A_extractor for the Phase 2 outer loop.

Tests are written TDD-first and serve as the executable specification for
REQ-P2O-008 through REQ-P2O-018, REQ-P2O-034, and REQ-P2O-036.

Refs:
    SRS 05a (Phase 2 Outer Loop), IMPLEMENTATION_PLAN.md Task 32.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    AgentType,
    DataModality,
    ExtractorOutput,
    MetricDirection,
    PipelineConfig,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
    TaskType,
)
import pytest

# ---------------------------------------------------------------------------
# Module path constant for patching
# ---------------------------------------------------------------------------

_MODULE = "mle_star.phase2_outer"


# ---------------------------------------------------------------------------
# Reusable test helpers
# ---------------------------------------------------------------------------


def _make_task(
    direction: MetricDirection = MetricDirection.MAXIMIZE,
) -> TaskDescription:
    """Create a minimal TaskDescription for testing."""
    return TaskDescription(
        competition_id="test-comp",
        task_type=TaskType.CLASSIFICATION,
        data_modality=DataModality.TABULAR,
        evaluation_metric="accuracy",
        metric_direction=direction,
        description="Predict the target variable from tabular features.",
    )


def _make_config(**kwargs: Any) -> PipelineConfig:
    """Create a PipelineConfig for testing with optional overrides."""
    return PipelineConfig(**kwargs)


def _make_solution(
    content: str = "print('hello')",
    phase: SolutionPhase = SolutionPhase.INIT,
) -> SolutionScript:
    """Create a SolutionScript for testing."""
    return SolutionScript(content=content, phase=phase)


def _make_extractor_json(plans: list[dict[str, str]]) -> str:
    """Build a JSON string conforming to ExtractorOutput schema."""
    return json.dumps({"plans": plans})


# ===========================================================================
# REQ-P2O-008: invoke_summarize -- Is Async
# ===========================================================================


@pytest.mark.unit
class TestInvokeSummarizeIsAsync:
    """invoke_summarize is an async function (REQ-P2O-008)."""

    def test_is_coroutine_function(self) -> None:
        """invoke_summarize is defined as an async function."""
        from mle_star.phase2_outer import invoke_summarize

        assert asyncio.iscoroutinefunction(invoke_summarize)


# ===========================================================================
# REQ-P2O-009: invoke_summarize -- Prompt Loading from Registry
# ===========================================================================


@pytest.mark.unit
class TestInvokeSummarizePromptRegistry:
    """invoke_summarize loads A_summarize prompt from PromptRegistry (REQ-P2O-009)."""

    async def test_registry_get_called_with_summarize_agent_type(self) -> None:
        """PromptRegistry.get is called with AgentType.SUMMARIZE."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="Ablation revealed X is key")

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "rendered summarize prompt"
            mock_registry.get.return_value = mock_template

            await invoke_summarize(
                ablation_code="print('ablation')",
                raw_output="feature importance: 0.85",
                client=client,
            )

        mock_registry.get.assert_called_once_with(AgentType.SUMMARIZE)

    async def test_template_rendered_with_ablation_code_and_raw_result(self) -> None:
        """Summarize template is rendered with ablation_code and raw_result."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="Summary text")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_summarize(
                ablation_code="ablation_code_marker",
                raw_output="raw_output_marker",
                client=client,
            )

        assert len(render_kwargs_captured) == 1
        assert render_kwargs_captured[0]["ablation_code"] == "ablation_code_marker"
        assert render_kwargs_captured[0]["raw_result"] == "raw_output_marker"

    async def test_rendered_prompt_sent_to_client(self) -> None:
        """The rendered prompt is sent via client.send_message."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="Summary output")
        expected_prompt = "rendered summarize prompt xyz"

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = expected_prompt
            mock_registry.get.return_value = mock_template

            await invoke_summarize(
                ablation_code="code",
                raw_output="output",
                client=client,
            )

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("message") == expected_prompt


# ===========================================================================
# REQ-P2O-010: invoke_summarize -- Agent Invocation
# ===========================================================================


@pytest.mark.unit
class TestInvokeSummarizeAgentInvocation:
    """invoke_summarize invokes A_summarize via client.send_message (REQ-P2O-010)."""

    async def test_client_invoked_with_summarize_agent_type(self) -> None:
        """Client.send_message is invoked with agent_type=str(AgentType.SUMMARIZE)."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="Summary result")

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await invoke_summarize(
                ablation_code="code",
                raw_output="output",
                client=client,
            )

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("agent_type") == str(AgentType.SUMMARIZE)

    async def test_client_invoked_exactly_once(self) -> None:
        """Client.send_message is called exactly once per invocation."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="Summary")

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await invoke_summarize(
                ablation_code="code",
                raw_output="output",
                client=client,
            )

        assert client.send_message.call_count == 1


# ===========================================================================
# REQ-P2O-011: invoke_summarize -- Returns Full Text Response
# ===========================================================================


@pytest.mark.unit
class TestInvokeSummarizeReturnsFullText:
    """invoke_summarize returns the full text response (REQ-P2O-011)."""

    async def test_returns_full_agent_response(self) -> None:
        """Returns the full text response without any extraction."""
        from mle_star.phase2_outer import invoke_summarize

        expected_summary = (
            "The ablation study shows that feature engineering contributes "
            "most to model performance, followed by hyperparameter tuning."
        )
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=expected_summary)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_summarize(
                ablation_code="code",
                raw_output="output",
                client=client,
            )

        assert result == expected_summary

    async def test_returns_string_type(self) -> None:
        """Return type is always str."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="A summary")

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_summarize(
                ablation_code="code",
                raw_output="output",
                client=client,
            )

        assert isinstance(result, str)

    async def test_does_not_extract_code_block(self) -> None:
        """Response containing code blocks is NOT extracted -- returns as-is."""
        from mle_star.phase2_outer import invoke_summarize

        response_with_code = (
            "Summary: ```python\nprint('hello')\n``` was the key finding."
        )
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=response_with_code)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_summarize(
                ablation_code="code",
                raw_output="output",
                client=client,
            )

        assert result == response_with_code

    async def test_preserves_newlines_and_whitespace(self) -> None:
        """Return value preserves all formatting from the agent response."""
        from mle_star.phase2_outer import invoke_summarize

        formatted_summary = "Line 1\n\n  Line 2\n\tLine 3"
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=formatted_summary)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_summarize(
                ablation_code="code",
                raw_output="output",
                client=client,
            )

        assert result == formatted_summary


# ===========================================================================
# REQ-P2O-036: invoke_summarize -- Fallback on empty/unparseable response
# ===========================================================================


@pytest.mark.unit
class TestInvokeSummarizeFallback:
    """invoke_summarize falls back to truncated raw_output on empty response (REQ-P2O-036)."""

    async def test_empty_response_triggers_fallback(self) -> None:
        """Empty agent response triggers auto-summary fallback."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_summarize(
                ablation_code="code",
                raw_output="some raw output here",
                client=client,
            )

        assert result.startswith("[Auto-summary from raw output] ")
        assert "some raw output here" in result

    async def test_whitespace_only_response_triggers_fallback(self) -> None:
        """Whitespace-only agent response triggers auto-summary fallback."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="   \n\n  ")

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_summarize(
                ablation_code="code",
                raw_output="fallback output",
                client=client,
            )

        assert result.startswith("[Auto-summary from raw output] ")

    async def test_fallback_prefix_is_exact(self) -> None:
        """Fallback prefix is exactly '[Auto-summary from raw output] '."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")
        raw = "some output"

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_summarize(
                ablation_code="code",
                raw_output=raw,
                client=client,
            )

        assert result == "[Auto-summary from raw output] some output"

    async def test_fallback_truncates_raw_output_to_last_2000_chars(self) -> None:
        """Fallback truncates raw_output to the last 2000 characters."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")
        # Create output longer than 2000 chars
        long_output = "A" * 1000 + "B" * 2000

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_summarize(
                ablation_code="code",
                raw_output=long_output,
                client=client,
            )

        prefix = "[Auto-summary from raw output] "
        assert result.startswith(prefix)
        content_after_prefix = result[len(prefix) :]
        assert len(content_after_prefix) == 2000
        assert content_after_prefix == "B" * 2000

    async def test_fallback_with_exactly_2000_char_output(self) -> None:
        """When raw_output is exactly 2000 chars, no truncation occurs."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")
        raw = "X" * 2000

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_summarize(
                ablation_code="code",
                raw_output=raw,
                client=client,
            )

        prefix = "[Auto-summary from raw output] "
        content_after_prefix = result[len(prefix) :]
        assert content_after_prefix == "X" * 2000

    async def test_fallback_with_short_raw_output(self) -> None:
        """When raw_output is shorter than 2000 chars, entire content is used."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")
        raw = "short output"

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_summarize(
                ablation_code="code",
                raw_output=raw,
                client=client,
            )

        assert result == "[Auto-summary from raw output] short output"

    @pytest.mark.parametrize(
        "empty_response",
        ["", "   ", "\n\n", "  \n  \n  ", "\t\t"],
        ids=["empty", "spaces", "newlines", "mixed-whitespace", "tabs"],
    )
    async def test_various_empty_responses_trigger_fallback(
        self, empty_response: str
    ) -> None:
        """Various empty/whitespace responses all trigger the fallback."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=empty_response)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_summarize(
                ablation_code="code",
                raw_output="fallback content",
                client=client,
            )

        assert result.startswith("[Auto-summary from raw output] ")

    async def test_fallback_with_empty_raw_output(self) -> None:
        """When both response and raw_output are empty, fallback has empty content."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_summarize(
                ablation_code="code",
                raw_output="",
                client=client,
            )

        assert result == "[Auto-summary from raw output] "


# ===========================================================================
# REQ-P2O-017: validate_code_block -- Exact substring matching
# ===========================================================================


@pytest.mark.unit
class TestValidateCodeBlock:
    """validate_code_block checks exact substring containment (REQ-P2O-017)."""

    def test_block_present_returns_true(self) -> None:
        """Returns True when code_block is an exact substring of solution content."""
        from mle_star.phase2_outer import validate_code_block

        solution = _make_solution(
            content="import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())"
        )
        assert validate_code_block("df = pd.read_csv('data.csv')", solution) is True

    def test_block_not_present_returns_false(self) -> None:
        """Returns False when code_block is not found in solution content."""
        from mle_star.phase2_outer import validate_code_block

        solution = _make_solution(content="import numpy as np\nx = np.array([1])")
        assert validate_code_block("import pandas", solution) is False

    def test_entire_content_is_valid_block(self) -> None:
        """The entire solution content is a valid code block."""
        from mle_star.phase2_outer import validate_code_block

        content = "x = 1\ny = 2"
        solution = _make_solution(content=content)
        assert validate_code_block(content, solution) is True

    def test_empty_string_block_returns_true(self) -> None:
        """Empty string is a substring of any string."""
        from mle_star.phase2_outer import validate_code_block

        solution = _make_solution(content="some code")
        assert validate_code_block("", solution) is True

    def test_case_sensitive_matching(self) -> None:
        """Matching is case-sensitive: 'Print' != 'print'."""
        from mle_star.phase2_outer import validate_code_block

        solution = _make_solution(content="print('hello')")
        assert validate_code_block("Print('hello')", solution) is False

    def test_whitespace_sensitive_matching(self) -> None:
        """Matching is whitespace-sensitive: indentation matters."""
        from mle_star.phase2_outer import validate_code_block

        solution = _make_solution(content="  x = 1")
        assert validate_code_block("x = 1", solution) is True
        assert validate_code_block("  x = 1", solution) is True

    def test_whitespace_only_content(self) -> None:
        """Empty code_block in whitespace-only solution returns True."""
        from mle_star.phase2_outer import validate_code_block

        solution = _make_solution(content="   ")
        assert validate_code_block("", solution) is True
        assert validate_code_block(" ", solution) is True

    def test_multiline_block_exact_match(self) -> None:
        """Multiline code blocks must match exactly including newlines."""
        from mle_star.phase2_outer import validate_code_block

        content = "def foo():\n    return 42\n\ndef bar():\n    return 0"
        solution = _make_solution(content=content)
        assert validate_code_block("def foo():\n    return 42", solution) is True
        assert validate_code_block("def foo():\n  return 42", solution) is False

    def test_returns_bool_type(self) -> None:
        """Return type is bool (not truthy int)."""
        from mle_star.phase2_outer import validate_code_block

        solution = _make_solution(content="x = 1")
        result = validate_code_block("x = 1", solution)
        assert isinstance(result, bool)

    def test_partial_line_match(self) -> None:
        """Partial line matches count as valid (substring semantics)."""
        from mle_star.phase2_outer import validate_code_block

        solution = _make_solution(content="x = some_function(arg1, arg2)")
        assert validate_code_block("some_function", solution) is True


# ===========================================================================
# REQ-P2O-012: invoke_extractor -- Is Async
# ===========================================================================


@pytest.mark.unit
class TestInvokeExtractorIsAsync:
    """invoke_extractor is an async function (REQ-P2O-012)."""

    def test_is_coroutine_function(self) -> None:
        """invoke_extractor is defined as an async function."""
        from mle_star.phase2_outer import invoke_extractor

        assert asyncio.iscoroutinefunction(invoke_extractor)


# ===========================================================================
# REQ-P2O-013: invoke_extractor -- Prompt Loading from Registry
# ===========================================================================


@pytest.mark.unit
class TestInvokeExtractorPromptRegistry:
    """invoke_extractor loads A_extractor prompt from PromptRegistry (REQ-P2O-013)."""

    async def test_registry_get_called_with_extractor_agent_type(self) -> None:
        """PromptRegistry.get is called with AgentType.EXTRACTOR."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        valid_json = _make_extractor_json(
            [{"code_block": "x = 1", "plan": "Optimize X"}]
        )
        client.send_message = AsyncMock(return_value=valid_json)
        solution = _make_solution(content="x = 1\ny = 2")

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "rendered extractor prompt"
            mock_registry.get.return_value = mock_template

            await invoke_extractor(
                summary="Feature engineering is key",
                solution=solution,
                previous_blocks=[],
                client=client,
            )

        mock_registry.get.assert_called_once_with(AgentType.EXTRACTOR)

    async def test_template_rendered_with_correct_variables(self) -> None:
        """Extractor template is rendered with solution_script, ablation_summary, previous_code_blocks."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        valid_json = _make_extractor_json([{"code_block": "x = 1", "plan": "Improve"}])
        client.send_message = AsyncMock(return_value=valid_json)
        solution = _make_solution(content="solution_content_marker")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_extractor(
                summary="ablation_summary_marker",
                solution=solution,
                previous_blocks=[],
                client=client,
            )

        assert len(render_kwargs_captured) == 1
        assert render_kwargs_captured[0]["solution_script"] == "solution_content_marker"
        assert (
            render_kwargs_captured[0]["ablation_summary"] == "ablation_summary_marker"
        )
        assert "previous_code_blocks" in render_kwargs_captured[0]

    async def test_rendered_prompt_sent_to_client(self) -> None:
        """The rendered prompt is sent via client.send_message."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        valid_json = _make_extractor_json([{"code_block": "x = 1", "plan": "Improve"}])
        client.send_message = AsyncMock(return_value=valid_json)
        expected_prompt = "rendered extractor prompt content"

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = expected_prompt
            mock_registry.get.return_value = mock_template

            await invoke_extractor(
                summary="summary",
                solution=_make_solution(),
                previous_blocks=[],
                client=client,
            )

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("message") == expected_prompt


# ===========================================================================
# REQ-P2O-014: invoke_extractor -- Agent Invocation
# ===========================================================================


@pytest.mark.unit
class TestInvokeExtractorAgentInvocation:
    """invoke_extractor invokes A_extractor via client.send_message (REQ-P2O-014)."""

    async def test_client_invoked_with_extractor_agent_type(self) -> None:
        """Client.send_message is invoked with agent_type=str(AgentType.EXTRACTOR)."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        valid_json = _make_extractor_json([{"code_block": "x = 1", "plan": "Improve"}])
        client.send_message = AsyncMock(return_value=valid_json)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await invoke_extractor(
                summary="summary",
                solution=_make_solution(),
                previous_blocks=[],
                client=client,
            )

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("agent_type") == str(AgentType.EXTRACTOR)


# ===========================================================================
# REQ-P2O-015: invoke_extractor -- Structured Output Parsing
# ===========================================================================


@pytest.mark.unit
class TestInvokeExtractorStructuredOutput:
    """invoke_extractor parses response via ExtractorOutput.model_validate_json (REQ-P2O-015)."""

    async def test_returns_extractor_output_on_valid_json(self) -> None:
        """Returns ExtractorOutput when response is valid JSON."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        valid_json = _make_extractor_json(
            [{"code_block": "x = 1", "plan": "Optimize variable assignment"}]
        )
        client.send_message = AsyncMock(return_value=valid_json)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_extractor(
                summary="summary",
                solution=_make_solution(content="x = 1\ny = 2"),
                previous_blocks=[],
                client=client,
            )

        assert isinstance(result, ExtractorOutput)

    async def test_result_contains_correct_plans(self) -> None:
        """Parsed ExtractorOutput contains the plans from the JSON response."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        valid_json = _make_extractor_json(
            [
                {"code_block": "x = 1", "plan": "Plan A"},
                {"code_block": "y = 2", "plan": "Plan B"},
            ]
        )
        client.send_message = AsyncMock(return_value=valid_json)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_extractor(
                summary="summary",
                solution=_make_solution(content="x = 1\ny = 2"),
                previous_blocks=[],
                client=client,
            )

        assert result is not None
        assert len(result.plans) == 2
        assert result.plans[0].code_block == "x = 1"
        assert result.plans[0].plan == "Plan A"
        assert result.plans[1].code_block == "y = 2"
        assert result.plans[1].plan == "Plan B"

    async def test_single_plan_in_response(self) -> None:
        """Works correctly with exactly one plan."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        valid_json = _make_extractor_json(
            [{"code_block": "model.fit(X, y)", "plan": "Use cross-validation"}]
        )
        client.send_message = AsyncMock(return_value=valid_json)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_extractor(
                summary="summary",
                solution=_make_solution(content="model.fit(X, y)"),
                previous_blocks=[],
                client=client,
            )

        assert result is not None
        assert len(result.plans) == 1


# ===========================================================================
# REQ-P2O-016: invoke_extractor -- Uses Structured Output Format
# ===========================================================================


@pytest.mark.unit
class TestInvokeExtractorOutputFormat:
    """invoke_extractor uses output_format parameter in send_message (REQ-P2O-016)."""

    async def test_send_message_does_not_include_output_format(self) -> None:
        """Client.send_message no longer includes explicit output_format (auto-applied by client)."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        valid_json = _make_extractor_json([{"code_block": "x = 1", "plan": "Improve"}])
        client.send_message = AsyncMock(return_value=valid_json)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await invoke_extractor(
                summary="summary",
                solution=_make_solution(),
                previous_blocks=[],
                client=client,
            )

        call_kwargs = client.send_message.call_args[1]
        assert "output_format" not in call_kwargs


# ===========================================================================
# REQ-P2O-034: invoke_extractor -- Retry on JSON Parse Failure
# ===========================================================================


@pytest.mark.unit
class TestInvokeExtractorRetry:
    """invoke_extractor retries once on JSON parse failure (REQ-P2O-034)."""

    async def test_retries_once_on_malformed_json(self) -> None:
        """On malformed JSON, retries exactly once before returning None."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        client.send_message = AsyncMock(
            side_effect=["not valid json {{{", "still not valid"]
        )

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_extractor(
                summary="summary",
                solution=_make_solution(),
                previous_blocks=[],
                client=client,
            )

        assert result is None
        assert client.send_message.call_count == 2

    async def test_returns_none_when_both_attempts_fail(self) -> None:
        """Returns None when both parsing attempts fail."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=["garbage", "more garbage"])

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_extractor(
                summary="summary",
                solution=_make_solution(),
                previous_blocks=[],
                client=client,
            )

        assert result is None

    async def test_succeeds_on_retry_after_first_failure(self) -> None:
        """Returns valid ExtractorOutput when retry succeeds."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        valid_json = _make_extractor_json([{"code_block": "x = 1", "plan": "Fix it"}])
        client.send_message = AsyncMock(side_effect=["not json at all!", valid_json])

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_extractor(
                summary="summary",
                solution=_make_solution(content="x = 1"),
                previous_blocks=[],
                client=client,
            )

        assert isinstance(result, ExtractorOutput)
        assert result.plans[0].code_block == "x = 1"

    async def test_no_retry_on_first_success(self) -> None:
        """Does not retry when first attempt succeeds."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        valid_json = _make_extractor_json([{"code_block": "x = 1", "plan": "Improve"}])
        client.send_message = AsyncMock(return_value=valid_json)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await invoke_extractor(
                summary="summary",
                solution=_make_solution(),
                previous_blocks=[],
                client=client,
            )

        assert client.send_message.call_count == 1

    async def test_return_is_none_type_on_double_failure(self) -> None:
        """Return on double failure is None (not empty list or False)."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=["bad1", "bad2"])

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_extractor(
                summary="summary",
                solution=_make_solution(),
                previous_blocks=[],
                client=client,
            )

        assert result is None
        assert not isinstance(result, (ExtractorOutput, list, str))

    async def test_empty_plans_list_triggers_validation_error_and_retry(self) -> None:
        """Empty plans list in JSON triggers Pydantic validation error and retry."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        empty_plans_json = json.dumps({"plans": []})
        valid_json = _make_extractor_json([{"code_block": "x = 1", "plan": "Fix"}])
        client.send_message = AsyncMock(side_effect=[empty_plans_json, valid_json])

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_extractor(
                summary="summary",
                solution=_make_solution(content="x = 1"),
                previous_blocks=[],
                client=client,
            )

        assert isinstance(result, ExtractorOutput)
        assert client.send_message.call_count == 2

    async def test_retry_uses_same_prompt(self) -> None:
        """Retry sends the same rendered prompt as the first attempt."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        valid_json = _make_extractor_json([{"code_block": "x = 1", "plan": "Fix"}])
        client.send_message = AsyncMock(side_effect=["invalid json", valid_json])

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "the same prompt"
            mock_registry.get.return_value = mock_template

            await invoke_extractor(
                summary="summary",
                solution=_make_solution(),
                previous_blocks=[],
                client=client,
            )

        first_call = client.send_message.call_args_list[0]
        second_call = client.send_message.call_args_list[1]
        assert first_call[1]["message"] == second_call[1]["message"]


# ===========================================================================
# REQ-P2O-018: invoke_extractor -- Previous code blocks formatting
# ===========================================================================


@pytest.mark.unit
class TestInvokeExtractorPreviousBlocks:
    """invoke_extractor formats previous_blocks for template (REQ-P2O-018)."""

    async def test_empty_blocks_produce_empty_formatted_string(self) -> None:
        """Empty previous_blocks produces empty string for template variable."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        valid_json = _make_extractor_json([{"code_block": "x = 1", "plan": "Improve"}])
        client.send_message = AsyncMock(return_value=valid_json)

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_extractor(
                summary="summary",
                solution=_make_solution(),
                previous_blocks=[],
                client=client,
            )

        assert render_kwargs_captured[0]["previous_code_blocks"] == ""

    async def test_non_empty_blocks_formatted_in_template(self) -> None:
        """Non-empty previous_blocks are formatted and included in template."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        valid_json = _make_extractor_json([{"code_block": "x = 1", "plan": "Improve"}])
        client.send_message = AsyncMock(return_value=valid_json)

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_extractor(
                summary="summary",
                solution=_make_solution(),
                previous_blocks=["block_alpha", "block_beta"],
                client=client,
            )

        blocks_text = render_kwargs_captured[0]["previous_code_blocks"]
        assert "block_alpha" in blocks_text
        assert "block_beta" in blocks_text

    async def test_solution_content_passed_as_solution_script(self) -> None:
        """solution.content is passed as the solution_script template variable."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        valid_json = _make_extractor_json([{"code_block": "x = 1", "plan": "Improve"}])
        client.send_message = AsyncMock(return_value=valid_json)

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_extractor(
                summary="summary",
                solution=_make_solution(content="unique_solution_marker"),
                previous_blocks=[],
                client=client,
            )

        assert render_kwargs_captured[0]["solution_script"] == "unique_solution_marker"


# ===========================================================================
# _format_previous_blocks -- Empty list
# ===========================================================================


@pytest.mark.unit
class TestFormatPreviousBlocksEmpty:
    """_format_previous_blocks returns '' for empty list."""

    def test_empty_list_returns_empty_string(self) -> None:
        """Empty blocks list returns exactly empty string."""
        from mle_star.phase2_outer import _format_previous_blocks

        result = _format_previous_blocks([])
        assert result == ""

    def test_empty_list_return_type_is_str(self) -> None:
        """Return type is str (not None) for empty list."""
        from mle_star.phase2_outer import _format_previous_blocks

        result = _format_previous_blocks([])
        assert isinstance(result, str)


# ===========================================================================
# _format_previous_blocks -- Non-empty list
# ===========================================================================


@pytest.mark.unit
class TestFormatPreviousBlocksNonEmpty:
    """_format_previous_blocks formats blocks with header and numbering."""

    def test_single_block_has_header(self) -> None:
        """Single block result includes a header."""
        from mle_star.phase2_outer import _format_previous_blocks

        result = _format_previous_blocks(["x = 1"])
        assert "Previously Improved Code Blocks" in result

    def test_single_block_includes_content(self) -> None:
        """Single block result includes the block text."""
        from mle_star.phase2_outer import _format_previous_blocks

        result = _format_previous_blocks(["some_code_block"])
        assert "some_code_block" in result

    def test_single_block_has_number_1(self) -> None:
        """Single block is numbered with 1."""
        from mle_star.phase2_outer import _format_previous_blocks

        result = _format_previous_blocks(["block content"])
        assert "1" in result

    def test_multiple_blocks_all_present(self) -> None:
        """All blocks appear in the output."""
        from mle_star.phase2_outer import _format_previous_blocks

        blocks = ["block_alpha", "block_beta", "block_gamma"]
        result = _format_previous_blocks(blocks)
        for block in blocks:
            assert block in result

    def test_multiple_blocks_numbered_sequentially(self) -> None:
        """Multiple blocks are numbered 1, 2, 3, etc."""
        from mle_star.phase2_outer import _format_previous_blocks

        blocks = ["A", "B", "C"]
        result = _format_previous_blocks(blocks)
        assert "1" in result
        assert "2" in result
        assert "3" in result

    def test_return_type_is_str(self) -> None:
        """Return type is str for non-empty list."""
        from mle_star.phase2_outer import _format_previous_blocks

        result = _format_previous_blocks(["some block"])
        assert isinstance(result, str)

    def test_header_uses_code_blocks_not_ablations(self) -> None:
        """Header says 'Code Blocks', not 'Ablation'."""
        from mle_star.phase2_outer import _format_previous_blocks

        result = _format_previous_blocks(["x"])
        assert "Code Block" in result

    def test_numbered_sections_use_code_block_label(self) -> None:
        """Each numbered section uses 'Code Block N' format."""
        from mle_star.phase2_outer import _format_previous_blocks

        result = _format_previous_blocks(["block1", "block2"])
        assert "Code Block 1" in result
        assert "Code Block 2" in result


# ===========================================================================
# Prompt Template Integration Tests
# ===========================================================================


@pytest.mark.unit
class TestSummarizePromptTemplateIntegration:
    """Validate that the summarize prompt template exists and renders correctly."""

    def test_summarize_template_exists_in_registry(self) -> None:
        """PromptRegistry contains a summarize template."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.SUMMARIZE)
        assert template.agent_type == AgentType.SUMMARIZE

    def test_summarize_template_has_ablation_code_variable(self) -> None:
        """Summarize template declares 'ablation_code' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.SUMMARIZE)
        assert "ablation_code" in template.variables

    def test_summarize_template_has_raw_result_variable(self) -> None:
        """Summarize template declares 'raw_result' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.SUMMARIZE)
        assert "raw_result" in template.variables

    def test_summarize_template_renders_with_variables(self) -> None:
        """Summarize template renders successfully with both required variables."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.SUMMARIZE)
        rendered = template.render(
            ablation_code="print('ablation')",
            raw_result="feature importance: 0.85",
        )
        assert "print('ablation')" in rendered
        assert "feature importance: 0.85" in rendered


@pytest.mark.unit
class TestExtractorPromptTemplateIntegration:
    """Validate that the extractor prompt template exists and renders correctly."""

    def test_extractor_template_exists_in_registry(self) -> None:
        """PromptRegistry contains an extractor template."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.EXTRACTOR)
        assert template.agent_type == AgentType.EXTRACTOR

    def test_extractor_template_has_solution_script_variable(self) -> None:
        """Extractor template declares 'solution_script' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.EXTRACTOR)
        assert "solution_script" in template.variables

    def test_extractor_template_has_ablation_summary_variable(self) -> None:
        """Extractor template declares 'ablation_summary' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.EXTRACTOR)
        assert "ablation_summary" in template.variables

    def test_extractor_template_has_previous_code_blocks_variable(self) -> None:
        """Extractor template declares 'previous_code_blocks' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.EXTRACTOR)
        assert "previous_code_blocks" in template.variables

    def test_extractor_template_renders_with_variables(self) -> None:
        """Extractor template renders successfully with all required variables."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.EXTRACTOR)
        rendered = template.render(
            solution_script="x = 1",
            ablation_summary="Feature engineering is important",
            previous_code_blocks="",
        )
        assert "x = 1" in rendered
        assert "Feature engineering is important" in rendered

    def test_extractor_template_renders_with_previous_blocks(self) -> None:
        """Extractor template renders with non-empty previous_code_blocks."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.EXTRACTOR)
        rendered = template.render(
            solution_script="model.fit(X, y)",
            ablation_summary="Model selection matters",
            previous_code_blocks="# Code Block 1\nmodel.fit(X, y)",
        )
        assert "model.fit(X, y)" in rendered
        assert "Code Block 1" in rendered


# ===========================================================================
# Hypothesis: Property-based tests for validate_code_block
# ===========================================================================


@pytest.mark.unit
class TestValidateCodeBlockPropertyBased:
    """Property-based tests for validate_code_block invariants."""

    @given(
        content=st.text(min_size=1, max_size=200),
    )
    @settings(max_examples=50)
    def test_full_content_always_valid(self, content: str) -> None:
        """The entire content is always a valid code block."""
        from mle_star.phase2_outer import validate_code_block

        solution = _make_solution(content=content)
        assert validate_code_block(content, solution) is True

    @given(
        content=st.text(min_size=0, max_size=200),
    )
    @settings(max_examples=50)
    def test_empty_string_always_valid(self, content: str) -> None:
        """Empty string is always a valid code block for any solution."""
        from mle_star.phase2_outer import validate_code_block

        solution = _make_solution(content=content)
        assert validate_code_block("", solution) is True

    @given(
        content=st.text(min_size=5, max_size=200),
        start=st.integers(min_value=0),
        length=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=50)
    def test_any_substring_is_valid(
        self, content: str, start: int, length: int
    ) -> None:
        """Any substring of content is a valid code block."""
        from mle_star.phase2_outer import validate_code_block

        # Clamp to valid range
        actual_start = start % len(content)
        actual_end = min(actual_start + length, len(content))
        substring = content[actual_start:actual_end]

        solution = _make_solution(content=content)
        assert validate_code_block(substring, solution) is True

    @given(
        content=st.text(min_size=1, max_size=100).filter(lambda s: s.strip()),
    )
    @settings(max_examples=30)
    def test_non_substring_returns_false(self, content: str) -> None:
        """A string not present in content returns False."""
        from mle_star.phase2_outer import validate_code_block

        # Use a sentinel that cannot be in the content
        unique_sentinel = f"SENTINEL_{content}_END_SENTINEL_XYZ"
        solution = _make_solution(content=content)
        assert validate_code_block(unique_sentinel, solution) is False

    @given(
        content=st.text(min_size=0, max_size=200),
    )
    @settings(max_examples=30)
    def test_return_type_is_always_bool(self, content: str) -> None:
        """Return type is always bool."""
        from mle_star.phase2_outer import validate_code_block

        solution = _make_solution(content=content)
        result = validate_code_block("test", solution)
        assert isinstance(result, bool)


# ===========================================================================
# Hypothesis: Property-based tests for invoke_summarize fallback
# ===========================================================================


@pytest.mark.unit
class TestInvokeSummarizeFallbackPropertyBased:
    """Property-based tests for invoke_summarize fallback behavior."""

    @given(
        raw_output=st.text(min_size=0, max_size=5000),
    )
    @settings(max_examples=30)
    async def test_fallback_content_never_exceeds_2000_chars(
        self, raw_output: str
    ) -> None:
        """Fallback content portion is never longer than 2000 characters."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_summarize(
                ablation_code="code",
                raw_output=raw_output,
                client=client,
            )

        prefix = "[Auto-summary from raw output] "
        content_after_prefix = result[len(prefix) :]
        assert len(content_after_prefix) <= 2000

    @given(
        raw_output=st.text(min_size=0, max_size=5000),
    )
    @settings(max_examples=30)
    async def test_fallback_always_starts_with_prefix(self, raw_output: str) -> None:
        """Fallback result always starts with the auto-summary prefix."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_summarize(
                ablation_code="code",
                raw_output=raw_output,
                client=client,
            )

        assert result.startswith("[Auto-summary from raw output] ")

    @given(
        response=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
    )
    @settings(max_examples=30)
    async def test_non_empty_response_returned_as_is(self, response: str) -> None:
        """Non-empty (after strip) response is returned without fallback."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=response)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_summarize(
                ablation_code="code",
                raw_output="output",
                client=client,
            )

        assert result == response


# ===========================================================================
# Hypothesis: Property-based tests for _format_previous_blocks
# ===========================================================================


@pytest.mark.unit
class TestFormatPreviousBlocksPropertyBased:
    """Property-based tests for _format_previous_blocks invariants."""

    @given(
        blocks=st.lists(
            st.text(min_size=1, max_size=100).filter(lambda s: s.strip()),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=30)
    def test_all_blocks_present_in_output(self, blocks: list[str]) -> None:
        """Every block appears in the formatted output."""
        from mle_star.phase2_outer import _format_previous_blocks

        result = _format_previous_blocks(blocks)
        for block in blocks:
            assert block in result

    @given(
        blocks=st.lists(
            st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=30)
    def test_non_empty_result_has_header(self, blocks: list[str]) -> None:
        """Non-empty blocks always produce output with 'Code Block' header."""
        from mle_star.phase2_outer import _format_previous_blocks

        result = _format_previous_blocks(blocks)
        assert "Previously Improved Code Blocks" in result

    @given(
        n_blocks=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=20)
    def test_empty_returns_empty_nonempty_returns_nonempty(self, n_blocks: int) -> None:
        """Empty list returns '', non-empty returns non-empty string."""
        from mle_star.phase2_outer import _format_previous_blocks

        blocks = [f"block_{i}" for i in range(n_blocks)]
        result = _format_previous_blocks(blocks)
        if n_blocks == 0:
            assert result == ""
        else:
            assert len(result) > 0


# ===========================================================================
# Hypothesis: Property-based tests for invoke_extractor
# ===========================================================================


@pytest.mark.unit
class TestInvokeExtractorPropertyBased:
    """Property-based tests for invoke_extractor invariants."""

    @given(
        n_blocks=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=20)
    async def test_any_number_of_previous_blocks_accepted(self, n_blocks: int) -> None:
        """invoke_extractor accepts any number of previous blocks."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        valid_json = _make_extractor_json([{"code_block": "x = 1", "plan": "Improve"}])
        client.send_message = AsyncMock(return_value=valid_json)

        blocks = [f"block_{i}" for i in range(n_blocks)]

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_extractor(
                summary="summary",
                solution=_make_solution(),
                previous_blocks=blocks,
                client=client,
            )

        assert result is None or isinstance(result, ExtractorOutput)

    @given(
        summary=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
    )
    @settings(max_examples=20)
    async def test_template_always_receives_summary(self, summary: str) -> None:
        """Template always receives the summary as ablation_summary."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        valid_json = _make_extractor_json([{"code_block": "x = 1", "plan": "Improve"}])
        client.send_message = AsyncMock(return_value=valid_json)

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_extractor(
                summary=summary,
                solution=_make_solution(),
                previous_blocks=[],
                client=client,
            )

        assert render_kwargs_captured[0]["ablation_summary"] == summary


# ===========================================================================
# Edge cases for invoke_summarize
# ===========================================================================


@pytest.mark.unit
class TestInvokeSummarizeEdgeCases:
    """Edge case tests for invoke_summarize."""

    async def test_very_long_response_returned_fully(self) -> None:
        """Very long agent response is returned in its entirety."""
        from mle_star.phase2_outer import invoke_summarize

        long_response = "A" * 10000
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=long_response)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_summarize(
                ablation_code="code",
                raw_output="output",
                client=client,
            )

        assert result == long_response
        assert len(result) == 10000

    async def test_empty_ablation_code_accepted(self) -> None:
        """Empty ablation_code is accepted without error."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="Summary result")

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_summarize(
                ablation_code="",
                raw_output="output",
                client=client,
            )

        assert result == "Summary result"

    async def test_empty_raw_output_accepted(self) -> None:
        """Empty raw_output is accepted without error."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="Summary")

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_summarize(
                ablation_code="code",
                raw_output="",
                client=client,
            )

        assert result == "Summary"

    async def test_response_with_only_newline_triggers_fallback(self) -> None:
        """Response that is just a newline triggers the fallback."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="\n")

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_summarize(
                ablation_code="code",
                raw_output="fallback data",
                client=client,
            )

        assert result.startswith("[Auto-summary from raw output] ")


# ===========================================================================
# Edge cases for invoke_extractor
# ===========================================================================


@pytest.mark.unit
class TestInvokeExtractorEdgeCases:
    """Edge case tests for invoke_extractor."""

    async def test_multiple_plans_in_single_response(self) -> None:
        """Response with multiple plans is parsed correctly."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        valid_json = _make_extractor_json(
            [
                {"code_block": "block_1", "plan": "Plan 1"},
                {"code_block": "block_2", "plan": "Plan 2"},
                {"code_block": "block_3", "plan": "Plan 3"},
            ]
        )
        client.send_message = AsyncMock(return_value=valid_json)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_extractor(
                summary="summary",
                solution=_make_solution(content="block_1\nblock_2\nblock_3"),
                previous_blocks=[],
                client=client,
            )

        assert result is not None
        assert len(result.plans) == 3

    async def test_unicode_in_plans(self) -> None:
        """Plans with unicode characters are handled correctly."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        valid_json = json.dumps(
            {
                "plans": [
                    {"code_block": "x = 42", "plan": "Optimize with better approach"}
                ]
            }
        )
        client.send_message = AsyncMock(return_value=valid_json)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_extractor(
                summary="summary",
                solution=_make_solution(content="x = 42"),
                previous_blocks=[],
                client=client,
            )

        assert result is not None

    async def test_multiline_code_block_in_plan(self) -> None:
        """Plans with multiline code blocks are parsed correctly."""
        from mle_star.phase2_outer import invoke_extractor

        multiline_block = "def train():\n    model.fit(X, y)\n    return model"
        client = AsyncMock()
        valid_json = json.dumps(
            {"plans": [{"code_block": multiline_block, "plan": "Add validation"}]}
        )
        client.send_message = AsyncMock(return_value=valid_json)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_extractor(
                summary="summary",
                solution=_make_solution(content=multiline_block),
                previous_blocks=[],
                client=client,
            )

        assert result is not None
        assert result.plans[0].code_block == multiline_block

    async def test_empty_string_response_triggers_retry(self) -> None:
        """Empty string response triggers retry (model_validate_json will fail)."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        valid_json = _make_extractor_json([{"code_block": "x", "plan": "fix"}])
        client.send_message = AsyncMock(side_effect=["", valid_json])

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_extractor(
                summary="summary",
                solution=_make_solution(content="x"),
                previous_blocks=[],
                client=client,
            )

        assert isinstance(result, ExtractorOutput)
        assert client.send_message.call_count == 2

    async def test_json_with_extra_fields_still_parsed(self) -> None:
        """JSON response with extra fields is still parsed (Pydantic ignores extras by default)."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        json_with_extras = json.dumps(
            {
                "plans": [
                    {"code_block": "x = 1", "plan": "Improve", "extra_field": "value"}
                ],
                "metadata": "ignored",
            }
        )
        client.send_message = AsyncMock(return_value=json_with_extras)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_extractor(
                summary="summary",
                solution=_make_solution(content="x = 1"),
                previous_blocks=[],
                client=client,
            )

        assert result is not None
        assert result.plans[0].code_block == "x = 1"


# ===========================================================================
# Parametrized tests for invoke_summarize
# ===========================================================================


@pytest.mark.unit
class TestInvokeSummarizeParametrized:
    """Parametrized tests for invoke_summarize covering multiple response formats."""

    @pytest.mark.parametrize(
        "response,should_fallback",
        [
            ("Valid summary text", False),
            ("", True),
            ("   ", True),
            ("\n", True),
            ("\t", True),
            ("  \n  \t  ", True),
            ("a", False),
            ("  x  ", False),
        ],
        ids=[
            "normal-text",
            "empty",
            "spaces",
            "newline",
            "tab",
            "mixed-whitespace",
            "single-char",
            "padded-char",
        ],
    )
    async def test_fallback_behavior_by_response_type(
        self, response: str, should_fallback: bool
    ) -> None:
        """Verify which responses trigger fallback vs return directly."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=response)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_summarize(
                ablation_code="code",
                raw_output="raw_out",
                client=client,
            )

        if should_fallback:
            assert result.startswith("[Auto-summary from raw output] ")
        else:
            assert result == response
