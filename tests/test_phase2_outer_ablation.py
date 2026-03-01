"""Tests for the Phase 2 outer loop ablation agent (Task 31).

Validates ``invoke_ablation``, ``compute_ablation_timeout``,
``execute_ablation_with_retry``, and ``_format_previous_ablations``
which implement the A_abl agent invocation for the Phase 2 outer loop.

Tests are written TDD-first and serve as the executable specification for
REQ-P2O-001 through REQ-P2O-007, REQ-P2O-020, REQ-P2O-021, and REQ-P2O-035.

Refs:
    SRS 05a (Phase 2 Outer Loop), IMPLEMENTATION_PLAN.md Task 31.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    AgentType,
    DataModality,
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


# ===========================================================================
# REQ-P2O-001: invoke_ablation -- Is Async
# ===========================================================================


@pytest.mark.unit
class TestInvokeAblationIsAsync:
    """invoke_ablation is an async function (REQ-P2O-001)."""

    def test_is_coroutine_function(self) -> None:
        """invoke_ablation is defined as an async function."""
        from mle_star.phase2_outer import invoke_ablation

        assert asyncio.iscoroutinefunction(invoke_ablation)


# ===========================================================================
# REQ-P2O-002: invoke_ablation -- Prompt Loading from Registry
# ===========================================================================


@pytest.mark.unit
class TestInvokeAblationPromptRegistry:
    """invoke_ablation loads the A_abl prompt from PromptRegistry (REQ-P2O-002)."""

    async def test_registry_get_called_with_ablation_agent_type(self) -> None:
        """PromptRegistry.get is called with AgentType.ABLATION."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nablation_code\n```")
        solution = _make_solution(content="my_solution_code")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "rendered ablation prompt"
            mock_registry.get.return_value = mock_template

            await invoke_ablation(
                solution=solution,
                previous_summaries=[],
                client=client,
            )

        mock_registry.get.assert_called_once_with(AgentType.ABLATION)

    async def test_template_rendered_with_solution_script_and_previous_ablations(
        self,
    ) -> None:
        """The ablation template is rendered with solution_script and previous_ablations."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nablation_code\n```")
        solution = _make_solution(content="solution_marker_abc")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_ablation(
                solution=solution,
                previous_summaries=[],
                client=client,
            )

        assert len(render_kwargs_captured) == 1
        assert render_kwargs_captured[0]["solution_script"] == "solution_marker_abc"
        assert "previous_ablations" in render_kwargs_captured[0]

    async def test_rendered_prompt_sent_to_client(self) -> None:
        """The rendered prompt is sent via client.send_message."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nablation_code\n```")
        expected_prompt = "rendered ablation prompt content xyz"

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = expected_prompt
            mock_registry.get.return_value = mock_template

            await invoke_ablation(
                solution=_make_solution(),
                previous_summaries=[],
                client=client,
            )

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("message") == expected_prompt


# ===========================================================================
# REQ-P2O-003: invoke_ablation -- Agent Invocation
# ===========================================================================


@pytest.mark.unit
class TestInvokeAblationAgentInvocation:
    """invoke_ablation invokes A_abl via client.send_message (REQ-P2O-003)."""

    async def test_client_invoked_with_ablation_agent_type(self) -> None:
        """Client.send_message is invoked with agent_type=str(AgentType.ABLATION)."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nresult\n```")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await invoke_ablation(
                solution=_make_solution(),
                previous_summaries=[],
                client=client,
            )

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("agent_type") == str(AgentType.ABLATION)

    async def test_client_invoked_exactly_once(self) -> None:
        """Client.send_message is called exactly once per invocation."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nresult\n```")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await invoke_ablation(
                solution=_make_solution(),
                previous_summaries=[],
                client=client,
            )

        assert client.send_message.call_count == 1


# ===========================================================================
# REQ-P2O-004: invoke_ablation -- Code Block Extraction
# ===========================================================================


@pytest.mark.unit
class TestInvokeAblationCodeExtraction:
    """invoke_ablation extracts code from response using extract_code_block (REQ-P2O-004)."""

    async def test_extract_code_block_called_on_response(self) -> None:
        """extract_code_block is invoked on the agent's response."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        agent_response = "Here is the ablation:\n```python\nabl_code\n```"
        client.send_message = AsyncMock(return_value=agent_response)

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="abl_code",
            ) as mock_extract,
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await invoke_ablation(
                solution=_make_solution(),
                previous_summaries=[],
                client=client,
            )

        mock_extract.assert_called_once_with(agent_response)


# ===========================================================================
# REQ-P2O-005: invoke_ablation -- Returns SolutionScript on success
# ===========================================================================


@pytest.mark.unit
class TestInvokeAblationReturnsSolutionScript:
    """invoke_ablation returns SolutionScript with correct attributes on success (REQ-P2O-005)."""

    async def test_returns_solution_script_on_success(self) -> None:
        """Returns a SolutionScript when extraction succeeds."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value="```python\nablation_code_xyz\n```"
        )

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ablation(
                solution=_make_solution(),
                previous_summaries=[],
                client=client,
            )

        assert isinstance(result, SolutionScript)

    async def test_result_has_phase_refined(self) -> None:
        """Returned SolutionScript has phase=SolutionPhase.REFINED."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nablation_code\n```")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ablation(
                solution=_make_solution(),
                previous_summaries=[],
                client=client,
            )

        assert result is not None
        assert result.phase == SolutionPhase.REFINED

    async def test_result_has_is_executable_true(self) -> None:
        """Returned SolutionScript has is_executable=True."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nablation_code\n```")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ablation(
                solution=_make_solution(),
                previous_summaries=[],
                client=client,
            )

        assert result is not None
        assert result.is_executable is True

    async def test_result_content_matches_extracted_code(self) -> None:
        """Returned SolutionScript.content matches the extracted code."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        expected_code = "import pandas as pd\nprint('ablation')"
        client.send_message = AsyncMock(return_value=f"```python\n{expected_code}\n```")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ablation(
                solution=_make_solution(),
                previous_summaries=[],
                client=client,
            )

        assert result is not None
        assert result.content == expected_code


# ===========================================================================
# REQ-P2O-006: invoke_ablation -- Returns None on empty response
# ===========================================================================


@pytest.mark.unit
class TestInvokeAblationReturnsNone:
    """invoke_ablation returns None when extraction fails (REQ-P2O-006)."""

    async def test_returns_none_on_empty_response(self) -> None:
        """Returns None when agent response is empty string."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ablation(
                solution=_make_solution(),
                previous_summaries=[],
                client=client,
            )

        assert result is None

    async def test_returns_none_on_whitespace_only_response(self) -> None:
        """Returns None when agent response is whitespace only."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="   \n\n  ")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ablation(
                solution=_make_solution(),
                previous_summaries=[],
                client=client,
            )

        assert result is None

    async def test_returns_none_on_empty_extracted_code(self) -> None:
        """Returns None when extract_code_block returns empty string."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\n\n```")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ablation(
                solution=_make_solution(),
                previous_summaries=[],
                client=client,
            )

        assert result is None

    async def test_return_is_none_type_not_empty_string(self) -> None:
        """Returns None (not empty string or other falsy) on failure."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ablation(
                solution=_make_solution(),
                previous_summaries=[],
                client=client,
            )

        assert result is None
        assert not isinstance(result, (str, SolutionScript))


# ===========================================================================
# REQ-P2O-007: invoke_ablation -- Previous summaries formatting
# ===========================================================================


@pytest.mark.unit
class TestInvokeAblationPreviousSummaries:
    """invoke_ablation formats previous_summaries correctly (REQ-P2O-007)."""

    async def test_empty_summaries_produce_empty_string(self) -> None:
        """When previous_summaries is empty, previous_ablations is empty string."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_ablation(
                solution=_make_solution(),
                previous_summaries=[],
                client=client,
            )

        assert render_kwargs_captured[0]["previous_ablations"] == ""

    async def test_single_summary_formatted_with_numbering(self) -> None:
        """Single previous summary is formatted with header and numbering."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_ablation(
                solution=_make_solution(),
                previous_summaries=["Feature engineering was most impactful"],
                client=client,
            )

        ablations_text = render_kwargs_captured[0]["previous_ablations"]
        assert "Feature engineering was most impactful" in ablations_text
        assert "1" in ablations_text

    async def test_multiple_summaries_all_numbered(self) -> None:
        """Multiple previous summaries are each numbered sequentially."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_ablation(
                solution=_make_solution(),
                previous_summaries=[
                    "Summary alpha",
                    "Summary beta",
                    "Summary gamma",
                ],
                client=client,
            )

        ablations_text = render_kwargs_captured[0]["previous_ablations"]
        assert "Summary alpha" in ablations_text
        assert "Summary beta" in ablations_text
        assert "Summary gamma" in ablations_text

    async def test_previous_summaries_has_header(self) -> None:
        """Non-empty previous summaries include a header line."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_ablation(
                solution=_make_solution(),
                previous_summaries=["Some summary"],
                client=client,
            )

        ablations_text = render_kwargs_captured[0]["previous_ablations"]
        assert "Previous Ablation" in ablations_text

    async def test_solution_content_passed_as_solution_script(self) -> None:
        """solution.content is passed as the solution_script template variable."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_ablation(
                solution=_make_solution(content="unique_content_marker"),
                previous_summaries=[],
                client=client,
            )

        assert render_kwargs_captured[0]["solution_script"] == "unique_content_marker"


# ===========================================================================
# REQ-P2O-007: Prompt template integration
# ===========================================================================


@pytest.mark.unit
class TestInvokeAblationPromptTemplateIntegration:
    """Validate that the ablation prompt template exists and renders correctly."""

    def test_ablation_template_exists_in_registry(self) -> None:
        """PromptRegistry contains an ablation template."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.ABLATION)
        assert template.agent_type == AgentType.ABLATION

    def test_ablation_template_has_solution_script_variable(self) -> None:
        """Ablation template declares 'solution_script' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.ABLATION)
        assert "solution_script" in template.variables

    def test_ablation_template_has_previous_ablations_variable(self) -> None:
        """Ablation template declares 'previous_ablations' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.ABLATION)
        assert "previous_ablations" in template.variables

    def test_ablation_template_renders_with_variables(self) -> None:
        """Ablation template renders successfully with both required variables."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.ABLATION)
        rendered = template.render(
            solution_script="x = 1",
            previous_ablations="",
            notes_context="",
        )
        assert "x = 1" in rendered

    def test_ablation_template_renders_with_previous_ablations(self) -> None:
        """Ablation template renders successfully with non-empty previous_ablations."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.ABLATION)
        rendered = template.render(
            solution_script="code here",
            previous_ablations="# Previous\n1. Summary A\n2. Summary B",
            notes_context="",
        )
        assert "code here" in rendered
        assert "Summary A" in rendered


# ===========================================================================
# REQ-P2O-020: _format_previous_ablations -- Empty list
# ===========================================================================


@pytest.mark.unit
class TestFormatPreviousAblationsEmpty:
    """_format_previous_ablations returns '' for empty list (REQ-P2O-020)."""

    def test_empty_list_returns_empty_string(self) -> None:
        """Empty summaries list returns exactly empty string."""
        from mle_star.phase2_outer import _format_previous_ablations

        result = _format_previous_ablations([])
        assert result == ""

    def test_empty_list_return_type_is_str(self) -> None:
        """Return type is str (not None) for empty list."""
        from mle_star.phase2_outer import _format_previous_ablations

        result = _format_previous_ablations([])
        assert isinstance(result, str)


# ===========================================================================
# REQ-P2O-021: _format_previous_ablations -- Non-empty list
# ===========================================================================


@pytest.mark.unit
class TestFormatPreviousAblationsNonEmpty:
    """_format_previous_ablations formats summaries with header and numbering (REQ-P2O-021)."""

    def test_single_summary_has_header(self) -> None:
        """Single summary result includes a header."""
        from mle_star.phase2_outer import _format_previous_ablations

        result = _format_previous_ablations(["Feature engineering is key"])
        assert "Previous Ablation" in result

    def test_single_summary_includes_content(self) -> None:
        """Single summary result includes the summary text."""
        from mle_star.phase2_outer import _format_previous_ablations

        result = _format_previous_ablations(["Feature engineering is key"])
        assert "Feature engineering is key" in result

    def test_single_summary_has_number_1(self) -> None:
        """Single summary is numbered with 1."""
        from mle_star.phase2_outer import _format_previous_ablations

        result = _format_previous_ablations(["Summary content"])
        assert "1" in result

    def test_multiple_summaries_all_present(self) -> None:
        """All summaries appear in the output."""
        from mle_star.phase2_outer import _format_previous_ablations

        summaries = ["Alpha summary", "Beta summary", "Gamma summary"]
        result = _format_previous_ablations(summaries)
        for summary in summaries:
            assert summary in result

    def test_multiple_summaries_numbered_sequentially(self) -> None:
        """Multiple summaries are numbered 1, 2, 3, etc."""
        from mle_star.phase2_outer import _format_previous_ablations

        summaries = ["A", "B", "C"]
        result = _format_previous_ablations(summaries)
        assert "1" in result
        assert "2" in result
        assert "3" in result

    def test_return_type_is_str(self) -> None:
        """Return type is str for non-empty list."""
        from mle_star.phase2_outer import _format_previous_ablations

        result = _format_previous_ablations(["Some summary"])
        assert isinstance(result, str)


# ===========================================================================
# REQ-P2O-035: compute_ablation_timeout -- Default config
# ===========================================================================


@pytest.mark.unit
class TestComputeAblationTimeoutDefault:
    """compute_ablation_timeout with default config (REQ-P2O-035)."""

    def test_default_config_returns_600(self) -> None:
        """Default config (86400s, 4 steps): min(86400/8, 600) = 600."""
        from mle_star.phase2_outer import compute_ablation_timeout

        config = _make_config()
        result = compute_ablation_timeout(config)
        assert result == 600

    def test_returns_int_type(self) -> None:
        """Return type is int (not float)."""
        from mle_star.phase2_outer import compute_ablation_timeout

        config = _make_config()
        result = compute_ablation_timeout(config)
        assert isinstance(result, int)


# ===========================================================================
# REQ-P2O-035: compute_ablation_timeout -- Various configs
# ===========================================================================


@pytest.mark.unit
class TestComputeAblationTimeoutVariousConfigs:
    """compute_ablation_timeout with various config parameters (REQ-P2O-035)."""

    def test_short_time_limit_below_cap(self) -> None:
        """Short time limit: min(1000/(4*2), 600) = 125."""
        from mle_star.phase2_outer import compute_ablation_timeout

        config = _make_config(time_limit_seconds=1000, outer_loop_steps=4)
        result = compute_ablation_timeout(config)
        assert result == 125

    def test_many_outer_steps_below_cap(self) -> None:
        """Many outer steps: min(86400/(100*2), 600) = 432."""
        from mle_star.phase2_outer import compute_ablation_timeout

        config = _make_config(time_limit_seconds=86400, outer_loop_steps=100)
        result = compute_ablation_timeout(config)
        assert result == 432

    def test_very_short_time_limit(self) -> None:
        """Very short time limit: min(100/(4*2), 600) = 12."""
        from mle_star.phase2_outer import compute_ablation_timeout

        config = _make_config(time_limit_seconds=100, outer_loop_steps=4)
        result = compute_ablation_timeout(config)
        assert result == 12

    def test_single_outer_step(self) -> None:
        """Single outer step: min(86400/(1*2), 600) = 600 (capped)."""
        from mle_star.phase2_outer import compute_ablation_timeout

        config = _make_config(time_limit_seconds=86400, outer_loop_steps=1)
        result = compute_ablation_timeout(config)
        assert result == 600

    def test_large_time_with_few_steps(self) -> None:
        """Large time with few steps: always capped at 600."""
        from mle_star.phase2_outer import compute_ablation_timeout

        config = _make_config(time_limit_seconds=100000, outer_loop_steps=2)
        result = compute_ablation_timeout(config)
        assert result == 600

    @pytest.mark.parametrize(
        "time_limit,outer_steps,expected",
        [
            (86400, 4, 600),
            (1000, 4, 125),
            (86400, 100, 432),
            (100, 4, 12),
            (4800, 4, 600),
            (4799, 4, 599),
            (200, 1, 100),
        ],
        ids=[
            "default-capped",
            "short-time",
            "many-steps",
            "very-short",
            "exactly-at-cap",
            "just-below-cap",
            "single-step-short",
        ],
    )
    def test_parametrized_timeout_values(
        self, time_limit: int, outer_steps: int, expected: int
    ) -> None:
        """Parametrized test for various timeout computations."""
        from mle_star.phase2_outer import compute_ablation_timeout

        config = _make_config(
            time_limit_seconds=time_limit,
            outer_loop_steps=outer_steps,
        )
        result = compute_ablation_timeout(config)
        assert result == expected

    def test_integer_division_truncation(self) -> None:
        """Uses integer division (truncates toward zero)."""
        from mle_star.phase2_outer import compute_ablation_timeout

        # 1001 / (4*2) = 1001/8 = 125.125 => integer division = 125
        config = _make_config(time_limit_seconds=1001, outer_loop_steps=4)
        result = compute_ablation_timeout(config)
        assert result == 125


# ===========================================================================
# REQ-P2O-035: execute_ablation_with_retry -- Is Async
# ===========================================================================


@pytest.mark.unit
class TestExecuteAblationWithRetryIsAsync:
    """execute_ablation_with_retry is an async function."""

    def test_is_coroutine_function(self) -> None:
        """execute_ablation_with_retry is defined as an async function."""
        from mle_star.phase2_outer import execute_ablation_with_retry

        assert asyncio.iscoroutinefunction(execute_ablation_with_retry)


# ===========================================================================
# REQ-P2O-035: execute_ablation_with_retry -- Success on first attempt
# ===========================================================================


@pytest.mark.unit
class TestExecuteAblationWithRetrySuccess:
    """execute_ablation_with_retry returns stdout/stderr on success."""

    async def test_success_returns_stdout_stderr(self) -> None:
        """On successful execution, returns (stdout, stderr) from raw result."""
        from mle_star.execution import ExecutionRawResult
        from mle_star.phase2_outer import execute_ablation_with_retry

        client = AsyncMock()
        task = _make_task()
        config = _make_config()
        ablation_script = _make_solution(content="print('ablation')")

        raw_result = ExecutionRawResult(
            stdout="ablation output line 1\nablation output line 2",
            stderr="",
            exit_code=0,
            duration_seconds=5.0,
            timed_out=False,
        )

        with (
            patch(f"{_MODULE}.compute_ablation_timeout", return_value=600),
            patch(f"{_MODULE}.write_script", return_value="/tmp/ablation_study.py"),
            patch(
                f"{_MODULE}.execute_script",
                new_callable=AsyncMock,
                return_value=raw_result,
            ),
            patch(
                f"{_MODULE}.setup_working_directory",
                return_value="/tmp/workdir",
            ),
            patch(
                f"{_MODULE}.build_execution_env",
                return_value={"PATH": "/usr/bin"},
            ),
        ):
            stdout, stderr = await execute_ablation_with_retry(
                ablation_script=ablation_script,
                task=task,
                config=config,
                client=client,
            )

        assert stdout == "ablation output line 1\nablation output line 2"
        assert stderr == ""

    async def test_writes_script_as_ablation_study_py(self) -> None:
        """The script is written with filename 'ablation_study.py'."""
        from mle_star.execution import ExecutionRawResult
        from mle_star.phase2_outer import execute_ablation_with_retry

        client = AsyncMock()
        task = _make_task()
        config = _make_config()
        ablation_script = _make_solution(content="print('test')")

        raw_result = ExecutionRawResult(
            stdout="output",
            stderr="",
            exit_code=0,
            duration_seconds=1.0,
            timed_out=False,
        )

        with (
            patch(f"{_MODULE}.compute_ablation_timeout", return_value=600),
            patch(
                f"{_MODULE}.write_script",
                return_value="/tmp/ablation_study.py",
            ) as mock_write,
            patch(
                f"{_MODULE}.execute_script",
                new_callable=AsyncMock,
                return_value=raw_result,
            ),
            patch(
                f"{_MODULE}.setup_working_directory",
                return_value="/tmp/workdir",
            ),
            patch(
                f"{_MODULE}.build_execution_env",
                return_value={"PATH": "/usr/bin"},
            ),
        ):
            await execute_ablation_with_retry(
                ablation_script=ablation_script,
                task=task,
                config=config,
                client=client,
            )

        # Verify write_script called with "ablation_study.py" filename
        write_call = mock_write.call_args
        assert write_call[0][2] == "ablation_study.py" or (
            "filename" in (write_call[1] if write_call[1] else {})
            and write_call[1]["filename"] == "ablation_study.py"
        )

    async def test_uses_compute_ablation_timeout_for_timeout(self) -> None:
        """Uses compute_ablation_timeout(config) for the execution timeout."""
        from mle_star.execution import ExecutionRawResult
        from mle_star.phase2_outer import execute_ablation_with_retry

        client = AsyncMock()
        task = _make_task()
        config = _make_config()
        ablation_script = _make_solution()

        raw_result = ExecutionRawResult(
            stdout="output",
            stderr="",
            exit_code=0,
            duration_seconds=1.0,
            timed_out=False,
        )

        with (
            patch(
                f"{_MODULE}.compute_ablation_timeout",
                return_value=42,
            ) as mock_timeout,
            patch(
                f"{_MODULE}.write_script",
                return_value="/tmp/ablation_study.py",
            ),
            patch(
                f"{_MODULE}.execute_script",
                new_callable=AsyncMock,
                return_value=raw_result,
            ) as mock_exec,
            patch(
                f"{_MODULE}.setup_working_directory",
                return_value="/tmp/workdir",
            ),
            patch(
                f"{_MODULE}.build_execution_env",
                return_value={"PATH": "/usr/bin"},
            ),
        ):
            await execute_ablation_with_retry(
                ablation_script=ablation_script,
                task=task,
                config=config,
                client=client,
            )

        mock_timeout.assert_called_once_with(config)
        # The timeout value (42) should be passed to execute_script
        exec_call = mock_exec.call_args
        assert 42 in exec_call[0] or exec_call[1].get("timeout_seconds") == 42


# ===========================================================================
# REQ-P2O-035: execute_ablation_with_retry -- Retry on error
# ===========================================================================


@pytest.mark.unit
class TestExecuteAblationWithRetryOnError:
    """execute_ablation_with_retry retries on execution error using debug callback."""

    async def test_error_then_success_returns_fixed_output(self) -> None:
        """On error, debugger fixes script and re-execution succeeds."""
        from mle_star.execution import ExecutionRawResult
        from mle_star.phase2_outer import execute_ablation_with_retry

        client = AsyncMock()
        task = _make_task()
        config = _make_config(max_debug_attempts=3)
        ablation_script = _make_solution(content="broken_code")

        error_result = ExecutionRawResult(
            stdout="",
            stderr="Traceback (most recent call last):\n  File ...\nValueError: bad",
            exit_code=1,
            duration_seconds=1.0,
            timed_out=False,
        )
        success_result = ExecutionRawResult(
            stdout="fixed output",
            stderr="",
            exit_code=0,
            duration_seconds=2.0,
            timed_out=False,
        )

        fixed_script = _make_solution(content="fixed_code")
        mock_debug_cb = AsyncMock(return_value=fixed_script)

        with (
            patch(f"{_MODULE}.compute_ablation_timeout", return_value=600),
            patch(
                f"{_MODULE}.write_script",
                return_value="/tmp/ablation_study.py",
            ),
            patch(
                f"{_MODULE}.execute_script",
                new_callable=AsyncMock,
                side_effect=[error_result, success_result],
            ),
            patch(
                f"{_MODULE}.setup_working_directory",
                return_value="/tmp/workdir",
            ),
            patch(
                f"{_MODULE}.build_execution_env",
                return_value={"PATH": "/usr/bin"},
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=mock_debug_cb,
            ),
            patch(
                f"{_MODULE}.extract_traceback",
                return_value="Traceback: ValueError: bad",
            ),
        ):
            stdout, stderr = await execute_ablation_with_retry(
                ablation_script=ablation_script,
                task=task,
                config=config,
                client=client,
            )

        assert stdout == "fixed output"
        assert stderr == ""

    async def test_all_debug_attempts_exhausted_returns_empty(self) -> None:
        """When all debug attempts fail, returns ('', '')."""
        from mle_star.execution import ExecutionRawResult
        from mle_star.phase2_outer import execute_ablation_with_retry

        client = AsyncMock()
        task = _make_task()
        config = _make_config(max_debug_attempts=2)
        ablation_script = _make_solution(content="broken_code")

        error_result = ExecutionRawResult(
            stdout="",
            stderr="Traceback (most recent call last):\nError",
            exit_code=1,
            duration_seconds=1.0,
            timed_out=False,
        )

        still_broken = _make_solution(content="still_broken")
        mock_debug_cb = AsyncMock(return_value=still_broken)

        with (
            patch(f"{_MODULE}.compute_ablation_timeout", return_value=600),
            patch(
                f"{_MODULE}.write_script",
                return_value="/tmp/ablation_study.py",
            ),
            patch(
                f"{_MODULE}.execute_script",
                new_callable=AsyncMock,
                return_value=error_result,
            ),
            patch(
                f"{_MODULE}.setup_working_directory",
                return_value="/tmp/workdir",
            ),
            patch(
                f"{_MODULE}.build_execution_env",
                return_value={"PATH": "/usr/bin"},
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=mock_debug_cb,
            ),
            patch(
                f"{_MODULE}.extract_traceback",
                return_value="Traceback: Error",
            ),
        ):
            stdout, stderr = await execute_ablation_with_retry(
                ablation_script=ablation_script,
                task=task,
                config=config,
                client=client,
            )

        assert stdout == ""
        assert stderr == ""

    async def test_uses_max_debug_attempts_from_config(self) -> None:
        """Retries exactly config.max_debug_attempts times before returning empty."""
        from mle_star.execution import ExecutionRawResult
        from mle_star.phase2_outer import execute_ablation_with_retry

        client = AsyncMock()
        task = _make_task()
        config = _make_config(max_debug_attempts=3)
        ablation_script = _make_solution(content="broken_code")

        error_result = ExecutionRawResult(
            stdout="",
            stderr="Traceback (most recent call last):\nError",
            exit_code=1,
            duration_seconds=1.0,
            timed_out=False,
        )

        broken_fix = _make_solution(content="still_broken")
        mock_debug_cb = AsyncMock(return_value=broken_fix)

        with (
            patch(f"{_MODULE}.compute_ablation_timeout", return_value=600),
            patch(
                f"{_MODULE}.write_script",
                return_value="/tmp/ablation_study.py",
            ),
            patch(
                f"{_MODULE}.execute_script",
                new_callable=AsyncMock,
                return_value=error_result,
            ) as mock_exec,
            patch(
                f"{_MODULE}.setup_working_directory",
                return_value="/tmp/workdir",
            ),
            patch(
                f"{_MODULE}.build_execution_env",
                return_value={"PATH": "/usr/bin"},
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=mock_debug_cb,
            ),
            patch(
                f"{_MODULE}.extract_traceback",
                return_value="Traceback: Error",
            ),
        ):
            await execute_ablation_with_retry(
                ablation_script=ablation_script,
                task=task,
                config=config,
                client=client,
            )

        # 1 initial + 3 retries = 4 total execute_script calls
        assert mock_exec.call_count == 1 + 3

    async def test_uses_extract_traceback_for_error_info(self) -> None:
        """Uses extract_traceback to get error information from stderr."""
        from mle_star.execution import ExecutionRawResult
        from mle_star.phase2_outer import execute_ablation_with_retry

        client = AsyncMock()
        task = _make_task()
        config = _make_config(max_debug_attempts=1)
        ablation_script = _make_solution(content="broken")

        error_result = ExecutionRawResult(
            stdout="",
            stderr="Traceback (most recent call last):\nNameError: x is not defined",
            exit_code=1,
            duration_seconds=1.0,
            timed_out=False,
        )
        success_result = ExecutionRawResult(
            stdout="fixed",
            stderr="",
            exit_code=0,
            duration_seconds=1.0,
            timed_out=False,
        )

        fixed = _make_solution(content="fixed_code")
        mock_debug_cb = AsyncMock(return_value=fixed)

        with (
            patch(f"{_MODULE}.compute_ablation_timeout", return_value=600),
            patch(
                f"{_MODULE}.write_script",
                return_value="/tmp/ablation_study.py",
            ),
            patch(
                f"{_MODULE}.execute_script",
                new_callable=AsyncMock,
                side_effect=[error_result, success_result],
            ),
            patch(
                f"{_MODULE}.setup_working_directory",
                return_value="/tmp/workdir",
            ),
            patch(
                f"{_MODULE}.build_execution_env",
                return_value={"PATH": "/usr/bin"},
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=mock_debug_cb,
            ),
            patch(
                f"{_MODULE}.extract_traceback",
                return_value="NameError: x is not defined",
            ) as mock_extract_tb,
        ):
            await execute_ablation_with_retry(
                ablation_script=ablation_script,
                task=task,
                config=config,
                client=client,
            )

        mock_extract_tb.assert_called()

    async def test_uses_make_debug_callback(self) -> None:
        """Uses make_debug_callback(task, config, client) for debug callback."""
        from mle_star.execution import ExecutionRawResult
        from mle_star.phase2_outer import execute_ablation_with_retry

        client = AsyncMock()
        task = _make_task()
        config = _make_config(max_debug_attempts=1)
        ablation_script = _make_solution(content="broken")

        error_result = ExecutionRawResult(
            stdout="",
            stderr="Traceback (most recent call last):\nError",
            exit_code=1,
            duration_seconds=1.0,
            timed_out=False,
        )
        success_result = ExecutionRawResult(
            stdout="fixed",
            stderr="",
            exit_code=0,
            duration_seconds=1.0,
            timed_out=False,
        )

        fixed = _make_solution(content="fixed_code")
        mock_debug_cb = AsyncMock(return_value=fixed)

        with (
            patch(f"{_MODULE}.compute_ablation_timeout", return_value=600),
            patch(
                f"{_MODULE}.write_script",
                return_value="/tmp/ablation_study.py",
            ),
            patch(
                f"{_MODULE}.execute_script",
                new_callable=AsyncMock,
                side_effect=[error_result, success_result],
            ),
            patch(
                f"{_MODULE}.setup_working_directory",
                return_value="/tmp/workdir",
            ),
            patch(
                f"{_MODULE}.build_execution_env",
                return_value={"PATH": "/usr/bin"},
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=mock_debug_cb,
            ) as mock_make_cb,
            patch(
                f"{_MODULE}.extract_traceback",
                return_value="Error",
            ),
        ):
            await execute_ablation_with_retry(
                ablation_script=ablation_script,
                task=task,
                config=config,
                client=client,
            )

        mock_make_cb.assert_called_once_with(task, config, client)


# ===========================================================================
# REQ-P2O-035: execute_ablation_with_retry -- No error (exit_code=0)
# ===========================================================================


@pytest.mark.unit
class TestExecuteAblationWithRetryNoRetryNeeded:
    """execute_ablation_with_retry does not retry when no error occurs."""

    async def test_no_retry_on_success(self) -> None:
        """When first execution succeeds, debug callback is not invoked."""
        from mle_star.execution import ExecutionRawResult
        from mle_star.phase2_outer import execute_ablation_with_retry

        client = AsyncMock()
        task = _make_task()
        config = _make_config()
        ablation_script = _make_solution()

        success_result = ExecutionRawResult(
            stdout="ablation complete",
            stderr="",
            exit_code=0,
            duration_seconds=1.0,
            timed_out=False,
        )

        with (
            patch(f"{_MODULE}.compute_ablation_timeout", return_value=600),
            patch(
                f"{_MODULE}.write_script",
                return_value="/tmp/ablation_study.py",
            ),
            patch(
                f"{_MODULE}.execute_script",
                new_callable=AsyncMock,
                return_value=success_result,
            ) as mock_exec,
            patch(
                f"{_MODULE}.setup_working_directory",
                return_value="/tmp/workdir",
            ),
            patch(
                f"{_MODULE}.build_execution_env",
                return_value={"PATH": "/usr/bin"},
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ) as mock_make_cb,
        ):
            stdout, _stderr = await execute_ablation_with_retry(
                ablation_script=ablation_script,
                task=task,
                config=config,
                client=client,
            )

        assert mock_exec.call_count == 1
        assert stdout == "ablation complete"
        # make_debug_callback should not be called when no error
        mock_make_cb.assert_not_called()


# ===========================================================================
# Parametrized empty-response tests for invoke_ablation
# ===========================================================================


@pytest.mark.unit
class TestInvokeAblationParametrized:
    """Parametrized tests for invoke_ablation covering multiple empty response formats."""

    @pytest.mark.parametrize(
        "empty_response",
        ["", "   ", "\n\n", "  \n  \n  "],
        ids=["empty", "spaces", "newlines", "mixed-whitespace"],
    )
    async def test_empty_responses_return_none(self, empty_response: str) -> None:
        """invoke_ablation returns None for various empty/whitespace responses."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=empty_response)

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ablation(
                solution=_make_solution(),
                previous_summaries=[],
                client=client,
            )

        assert result is None


# ===========================================================================
# Hypothesis: Property-based tests for compute_ablation_timeout
# ===========================================================================


@pytest.mark.unit
class TestComputeAblationTimeoutPropertyBased:
    """Property-based tests for compute_ablation_timeout invariants."""

    @given(
        time_limit=st.integers(min_value=1, max_value=1_000_000),
        outer_steps=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=50)
    def test_result_is_always_positive_int(
        self, time_limit: int, outer_steps: int
    ) -> None:
        """compute_ablation_timeout always returns a positive integer."""
        from mle_star.phase2_outer import compute_ablation_timeout

        config = _make_config(
            time_limit_seconds=time_limit,
            outer_loop_steps=outer_steps,
        )
        result = compute_ablation_timeout(config)
        assert isinstance(result, int)
        assert result >= 0

    @given(
        time_limit=st.integers(min_value=1, max_value=1_000_000),
        outer_steps=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=50)
    def test_result_never_exceeds_600(self, time_limit: int, outer_steps: int) -> None:
        """compute_ablation_timeout never returns more than 600."""
        from mle_star.phase2_outer import compute_ablation_timeout

        config = _make_config(
            time_limit_seconds=time_limit,
            outer_loop_steps=outer_steps,
        )
        result = compute_ablation_timeout(config)
        assert result <= 600

    @given(
        time_limit=st.integers(min_value=1, max_value=1_000_000),
        outer_steps=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=50)
    def test_result_matches_formula(self, time_limit: int, outer_steps: int) -> None:
        """compute_ablation_timeout equals min(time_limit//(outer_steps*2), 600)."""
        from mle_star.phase2_outer import compute_ablation_timeout

        config = _make_config(
            time_limit_seconds=time_limit,
            outer_loop_steps=outer_steps,
        )
        result = compute_ablation_timeout(config)
        expected = min(time_limit // (outer_steps * 2), 600)
        assert result == expected

    @given(
        time_limit=st.integers(min_value=4800, max_value=1_000_000),
    )
    @settings(max_examples=20)
    def test_large_time_with_default_steps_always_capped(self, time_limit: int) -> None:
        """With default 4 outer steps and time >= 4800, result is always 600."""
        from mle_star.phase2_outer import compute_ablation_timeout

        config = _make_config(time_limit_seconds=time_limit, outer_loop_steps=4)
        result = compute_ablation_timeout(config)
        assert result == 600


# ===========================================================================
# Hypothesis: Property-based tests for _format_previous_ablations
# ===========================================================================


@pytest.mark.unit
class TestFormatPreviousAblationsPropertyBased:
    """Property-based tests for _format_previous_ablations invariants."""

    @given(
        summaries=st.lists(
            st.text(min_size=1, max_size=100).filter(lambda s: s.strip()),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=30)
    def test_all_summaries_present_in_output(self, summaries: list[str]) -> None:
        """Every summary appears in the formatted output."""
        from mle_star.phase2_outer import _format_previous_ablations

        result = _format_previous_ablations(summaries)
        for summary in summaries:
            assert summary in result

    @given(
        summaries=st.lists(
            st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=30)
    def test_non_empty_result_has_header(self, summaries: list[str]) -> None:
        """Non-empty summaries always produce output with 'Previous Ablation' header."""
        from mle_star.phase2_outer import _format_previous_ablations

        result = _format_previous_ablations(summaries)
        assert "Previous Ablation" in result

    @given(
        n_summaries=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=20)
    def test_empty_returns_empty_nonempty_returns_nonempty(
        self, n_summaries: int
    ) -> None:
        """Empty list returns '', non-empty returns non-empty string."""
        from mle_star.phase2_outer import _format_previous_ablations

        summaries = [f"summary_{i}" for i in range(n_summaries)]
        result = _format_previous_ablations(summaries)
        if n_summaries == 0:
            assert result == ""
        else:
            assert len(result) > 0


# ===========================================================================
# Hypothesis: Property-based tests for invoke_ablation
# ===========================================================================


@pytest.mark.unit
class TestInvokeAblationPropertyBased:
    """Property-based tests for invoke_ablation invariants."""

    @given(
        content=st.text(min_size=1, max_size=200).filter(
            lambda s: s.strip() and "```" not in s
        ),
    )
    @settings(max_examples=30)
    async def test_valid_solution_always_produces_result_or_none(
        self, content: str
    ) -> None:
        """invoke_ablation always returns SolutionScript or None for valid inputs."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nimproved\n```")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ablation(
                solution=_make_solution(content=content),
                previous_summaries=[],
                client=client,
            )

        assert result is None or isinstance(result, SolutionScript)

    @given(
        n_summaries=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=20)
    async def test_any_number_of_summaries_accepted(self, n_summaries: int) -> None:
        """invoke_ablation accepts any number of previous summaries."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        summaries = [f"summary_{i}" for i in range(n_summaries)]

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ablation(
                solution=_make_solution(),
                previous_summaries=summaries,
                client=client,
            )

        assert result is None or isinstance(result, SolutionScript)

    @given(
        content=st.text(min_size=1, max_size=200).filter(
            lambda s: s.strip() and "```" not in s
        ),
        n_summaries=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=20)
    async def test_template_always_rendered_with_correct_variables(
        self, content: str, n_summaries: int
    ) -> None:
        """Template is always rendered with solution_script and previous_ablations."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        summaries = [f"summary_{i}" for i in range(n_summaries)]
        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_ablation(
                solution=_make_solution(content=content),
                previous_summaries=summaries,
                client=client,
            )

        assert len(render_kwargs_captured) == 1
        assert render_kwargs_captured[0]["solution_script"] == content
        assert "previous_ablations" in render_kwargs_captured[0]


# ===========================================================================
# Edge cases for invoke_ablation
# ===========================================================================


@pytest.mark.unit
class TestInvokeAblationEdgeCases:
    """Edge case tests for invoke_ablation."""

    async def test_response_with_multiple_code_blocks_returns_longest(self) -> None:
        """When response has multiple code blocks, extract_code_block picks the longest."""
        from mle_star.phase2_outer import invoke_ablation

        short_code = "x = 1"
        long_code = "import numpy as np\nx = np.array([1, 2])\nprint(x.shape)"
        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value=(
                f"```python\n{short_code}\n```\nExplanation.\n"
                f"```python\n{long_code}\n```"
            )
        )

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ablation(
                solution=_make_solution(),
                previous_summaries=[],
                client=client,
            )

        assert result is not None
        assert result.content == long_code

    async def test_success_result_has_refined_phase(self) -> None:
        """Even with multiple summaries, result phase is always REFINED."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ablation(
                solution=_make_solution(phase=SolutionPhase.MERGED),
                previous_summaries=["summary_1", "summary_2"],
                client=client,
            )

        assert result is not None
        assert result.phase == SolutionPhase.REFINED

    async def test_multiline_code_extraction(self) -> None:
        """Multiline code inside fenced blocks is extracted correctly."""
        from mle_star.phase2_outer import invoke_ablation

        multiline = "import os\nfor i in range(10):\n    print(i)"
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=f"```python\n{multiline}\n```")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ablation(
                solution=_make_solution(),
                previous_summaries=[],
                client=client,
            )

        assert result is not None
        assert result.content == multiline


# ===========================================================================
# Edge cases for execute_ablation_with_retry
# ===========================================================================


@pytest.mark.unit
class TestExecuteAblationWithRetryEdgeCases:
    """Edge case tests for execute_ablation_with_retry."""

    async def test_timeout_treated_as_error(self) -> None:
        """A timed-out execution triggers retry (non-zero exit code)."""
        from mle_star.execution import ExecutionRawResult
        from mle_star.phase2_outer import execute_ablation_with_retry

        client = AsyncMock()
        task = _make_task()
        config = _make_config(max_debug_attempts=1)
        ablation_script = _make_solution()

        timeout_result = ExecutionRawResult(
            stdout="",
            stderr="",
            exit_code=-1,
            duration_seconds=600.0,
            timed_out=True,
        )

        fixed = _make_solution(content="fixed")
        mock_debug_cb = AsyncMock(return_value=fixed)

        success_result = ExecutionRawResult(
            stdout="success after fix",
            stderr="",
            exit_code=0,
            duration_seconds=5.0,
            timed_out=False,
        )

        with (
            patch(f"{_MODULE}.compute_ablation_timeout", return_value=600),
            patch(
                f"{_MODULE}.write_script",
                return_value="/tmp/ablation_study.py",
            ),
            patch(
                f"{_MODULE}.execute_script",
                new_callable=AsyncMock,
                side_effect=[timeout_result, success_result],
            ),
            patch(
                f"{_MODULE}.setup_working_directory",
                return_value="/tmp/workdir",
            ),
            patch(
                f"{_MODULE}.build_execution_env",
                return_value={"PATH": "/usr/bin"},
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=mock_debug_cb,
            ),
            patch(
                f"{_MODULE}.extract_traceback",
                return_value=None,
            ),
        ):
            stdout, _stderr = await execute_ablation_with_retry(
                ablation_script=ablation_script,
                task=task,
                config=config,
                client=client,
            )

        assert stdout == "success after fix"

    async def test_return_type_is_tuple_of_strings(self) -> None:
        """Return type is always tuple[str, str]."""
        from mle_star.execution import ExecutionRawResult
        from mle_star.phase2_outer import execute_ablation_with_retry

        client = AsyncMock()
        task = _make_task()
        config = _make_config()
        ablation_script = _make_solution()

        result = ExecutionRawResult(
            stdout="output",
            stderr="warn",
            exit_code=0,
            duration_seconds=1.0,
            timed_out=False,
        )

        with (
            patch(f"{_MODULE}.compute_ablation_timeout", return_value=600),
            patch(
                f"{_MODULE}.write_script",
                return_value="/tmp/ablation_study.py",
            ),
            patch(
                f"{_MODULE}.execute_script",
                new_callable=AsyncMock,
                return_value=result,
            ),
            patch(
                f"{_MODULE}.setup_working_directory",
                return_value="/tmp/workdir",
            ),
            patch(
                f"{_MODULE}.build_execution_env",
                return_value={"PATH": "/usr/bin"},
            ),
        ):
            out = await execute_ablation_with_retry(
                ablation_script=ablation_script,
                task=task,
                config=config,
                client=client,
            )

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], str)
        assert isinstance(out[1], str)
