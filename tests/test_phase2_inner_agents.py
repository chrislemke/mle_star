"""Tests for the Phase 2 inner loop coder and planner agents (Task 23).

Validates ``invoke_coder`` and ``invoke_planner`` which implement the
A_coder and A_planner agents for the Phase 2 targeted refinement inner
loop. The coder takes a code block and plan, returning improved code.
The planner takes a code block and history of prior plans/scores,
returning a new refinement plan.

Tests are written TDD-first and serve as the executable specification for
REQ-P2I-001 through REQ-P2I-015.

Refs:
    SRS 02b (Phase 2 Inner Loop), IMPLEMENTATION_PLAN.md Task 23.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import given, settings, strategies as st
from mle_star.models import AgentType
import pytest

# ---------------------------------------------------------------------------
# Module path constant for patching
# ---------------------------------------------------------------------------

_MODULE = "mle_star.phase2_inner"


# ===========================================================================
# REQ-P2I-001: invoke_coder -- Is Async
# ===========================================================================


@pytest.mark.unit
class TestInvokeCoderIsAsync:
    """invoke_coder is an async function (REQ-P2I-001)."""

    def test_is_coroutine_function(self) -> None:
        """invoke_coder is defined as an async function."""
        from mle_star.phase2_inner import invoke_coder

        assert asyncio.iscoroutinefunction(invoke_coder)


# ===========================================================================
# REQ-P2I-002: invoke_coder -- Input Validation
# ===========================================================================


@pytest.mark.unit
class TestInvokeCoderInputValidation:
    """invoke_coder raises ValueError for empty inputs (REQ-P2I-002)."""

    async def test_raises_on_empty_code_block(self) -> None:
        """Raises ValueError when code_block is an empty string."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        with pytest.raises(ValueError, match="code_block"):
            await invoke_coder(code_block="", plan="some plan", client=client)

    async def test_raises_on_whitespace_only_code_block(self) -> None:
        """Raises ValueError when code_block contains only whitespace."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        with pytest.raises(ValueError, match="code_block"):
            await invoke_coder(code_block="   \n  ", plan="some plan", client=client)

    async def test_raises_on_empty_plan(self) -> None:
        """Raises ValueError when plan is an empty string."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        with pytest.raises(ValueError, match="plan"):
            await invoke_coder(code_block="x = 1", plan="", client=client)

    async def test_raises_on_whitespace_only_plan(self) -> None:
        """Raises ValueError when plan contains only whitespace."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        with pytest.raises(ValueError, match="plan"):
            await invoke_coder(code_block="x = 1", plan="   \n  ", client=client)

    async def test_does_not_call_client_on_invalid_input(self) -> None:
        """Client is not invoked when input validation fails."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        with pytest.raises(ValueError):
            await invoke_coder(code_block="", plan="plan", client=client)

        client.send_message.assert_not_called()


# ===========================================================================
# REQ-P2I-003: invoke_coder -- Prompt Loading from Registry
# ===========================================================================


@pytest.mark.unit
class TestInvokeCoderPromptRegistry:
    """invoke_coder loads the A_coder prompt from PromptRegistry (REQ-P2I-003)."""

    async def test_registry_get_called_with_coder_agent_type(self) -> None:
        """PromptRegistry.get is called with AgentType.CODER."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nimproved_code\n```")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "rendered coder prompt"
            mock_registry.get.return_value = mock_template

            await invoke_coder(code_block="x = 1", plan="improve x", client=client)

        mock_registry.get.assert_called_once_with(AgentType.CODER)

    async def test_template_rendered_with_code_block_and_plan(self) -> None:
        """The coder template is rendered with code_block and plan variables."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nimproved\n```")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_coder(
                code_block="block_marker_abc",
                plan="plan_marker_xyz",
                client=client,
            )

        assert len(render_kwargs_captured) == 1
        assert render_kwargs_captured[0]["code_block"] == "block_marker_abc"
        assert render_kwargs_captured[0]["plan"] == "plan_marker_xyz"

    async def test_rendered_prompt_sent_to_client(self) -> None:
        """The rendered prompt is sent via client.send_message."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        expected_prompt = "rendered coder prompt content xyz"

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = expected_prompt
            mock_registry.get.return_value = mock_template

            await invoke_coder(code_block="x = 1", plan="improve x", client=client)

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("message") == expected_prompt


# ===========================================================================
# REQ-P2I-004: invoke_coder -- Agent Invocation
# ===========================================================================


@pytest.mark.unit
class TestInvokeCoderAgentInvocation:
    """invoke_coder invokes the coder agent via client.send_message (REQ-P2I-004)."""

    async def test_client_invoked_with_coder_agent_type(self) -> None:
        """Client.send_message is invoked with agent_type=str(AgentType.CODER)."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nresult\n```")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await invoke_coder(code_block="x = 1", plan="improve x", client=client)

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("agent_type") == str(AgentType.CODER)

    async def test_client_invoked_exactly_once(self) -> None:
        """Client.send_message is called exactly once per invocation."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nresult\n```")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await invoke_coder(code_block="x = 1", plan="improve x", client=client)

        assert client.send_message.call_count == 1


# ===========================================================================
# REQ-P2I-005: invoke_coder -- Code Block Extraction
# ===========================================================================


@pytest.mark.unit
class TestInvokeCoderCodeExtraction:
    """invoke_coder extracts code from response using extract_code_block (REQ-P2I-005)."""

    async def test_extract_code_block_called_on_response(self) -> None:
        """extract_code_block is invoked on the agent's response."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        agent_response = "Here is the fix:\n```python\nimproved_code\n```"
        client.send_message = AsyncMock(return_value=agent_response)

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="improved_code",
            ) as mock_extract,
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await invoke_coder(code_block="x = 1", plan="improve x", client=client)

        mock_extract.assert_called_once_with(agent_response)

    async def test_returns_extracted_code_block(self) -> None:
        """Returns the string extracted by extract_code_block."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value="```python\nimproved_code_xyz\n```"
        )

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_coder(
                code_block="x = 1", plan="improve x", client=client
            )

        assert result == "improved_code_xyz"

    async def test_returns_code_from_multiline_fenced_block(self) -> None:
        """Extracts multiline code from a fenced block in the response."""
        from mle_star.phase2_inner import invoke_coder

        multiline_code = "import numpy as np\nx = np.array([1, 2])\nprint(x)"
        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value=f"Here is the improved code:\n```python\n{multiline_code}\n```\nDone."
        )

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_coder(
                code_block="x = 1", plan="use numpy", client=client
            )

        assert result == multiline_code


# ===========================================================================
# REQ-P2I-006: invoke_coder -- Returns None on Failure
# ===========================================================================


@pytest.mark.unit
class TestInvokeCoderReturnsNoneOnFailure:
    """invoke_coder returns None when extraction fails (REQ-P2I-006)."""

    async def test_returns_none_on_empty_response(self) -> None:
        """Returns None when agent response is empty string."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_coder(
                code_block="x = 1", plan="improve x", client=client
            )

        assert result is None

    async def test_returns_none_on_whitespace_only_response(self) -> None:
        """Returns None when agent response is whitespace only."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="   \n\n  ")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_coder(
                code_block="x = 1", plan="improve x", client=client
            )

        assert result is None

    async def test_returns_none_on_empty_extracted_code(self) -> None:
        """Returns None when extract_code_block returns empty string."""
        from mle_star.phase2_inner import invoke_coder

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

            result = await invoke_coder(
                code_block="x = 1", plan="improve x", client=client
            )

        assert result is None


# ===========================================================================
# REQ-P2I-007: invoke_coder -- Return Type
# ===========================================================================


@pytest.mark.unit
class TestInvokeCoderReturnType:
    """invoke_coder returns str | None (REQ-P2I-007)."""

    async def test_returns_string_on_success(self) -> None:
        """Returns a string when extraction succeeds."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nimproved\n```")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_coder(
                code_block="x = 1", plan="improve x", client=client
            )

        assert isinstance(result, str)
        assert result == "improved"

    async def test_return_type_is_none_on_failure(self) -> None:
        """Returns None (not empty string or other falsy) on failure."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_coder(
                code_block="x = 1", plan="improve x", client=client
            )

        assert result is None


# ===========================================================================
# REQ-P2I-008: invoke_coder -- Prompt Template Integration
# ===========================================================================


@pytest.mark.unit
class TestInvokeCoderPromptTemplateIntegration:
    """invoke_coder uses the actual coder prompt template from the registry."""

    def test_coder_template_exists_in_registry(self) -> None:
        """PromptRegistry contains a coder template (no variant)."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.CODER)
        assert template.agent_type == AgentType.CODER

    def test_coder_template_has_code_block_variable(self) -> None:
        """Coder template declares 'code_block' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.CODER)
        assert "code_block" in template.variables

    def test_coder_template_has_plan_variable(self) -> None:
        """Coder template declares 'plan' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.CODER)
        assert "plan" in template.variables

    def test_coder_template_renders_with_variables(self) -> None:
        """Coder template renders successfully with all required variables."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.CODER)
        rendered = template.render(
            code_block="x = 1",
            plan="improve x",
            task_description="Predict target",
            evaluation_metric="accuracy",
            metric_direction="maximize",
            data_modality="tabular",
            current_score="0.85",
        )
        assert "x = 1" in rendered
        assert "improve x" in rendered

    def test_coder_template_mentions_subsampling(self) -> None:
        """Coder template mentions subsampling preservation instruction."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.CODER)
        rendered = template.render(
            code_block="code",
            plan="plan",
            task_description="Predict target",
            evaluation_metric="accuracy",
            metric_direction="maximize",
            data_modality="tabular",
            current_score="0.85",
        )
        assert "subsampling" in rendered.lower()

    def test_coder_template_mentions_dummy_variables(self) -> None:
        """Coder template mentions not introducing dummy variables."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.CODER)
        rendered = template.render(
            code_block="code",
            plan="plan",
            task_description="Predict target",
            evaluation_metric="accuracy",
            metric_direction="maximize",
            data_modality="tabular",
            current_score="0.85",
        )
        assert "dummy" in rendered.lower()


# ===========================================================================
# REQ-P2I-009: invoke_planner -- Is Async
# ===========================================================================


@pytest.mark.unit
class TestInvokePlannerIsAsync:
    """invoke_planner is an async function (REQ-P2I-009)."""

    def test_is_coroutine_function(self) -> None:
        """invoke_planner is defined as an async function."""
        from mle_star.phase2_inner import invoke_planner

        assert asyncio.iscoroutinefunction(invoke_planner)


# ===========================================================================
# REQ-P2I-010: invoke_planner -- Input Validation
# ===========================================================================


@pytest.mark.unit
class TestInvokePlannerInputValidation:
    """invoke_planner raises ValueError for invalid inputs (REQ-P2I-010)."""

    async def test_raises_on_empty_plans_list(self) -> None:
        """Raises ValueError when plans is an empty list."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        with pytest.raises(
            ValueError,
            match="At least one previous plan is required for A_planner",
        ):
            await invoke_planner(code_block="x = 1", plans=[], scores=[], client=client)

    async def test_raises_on_mismatched_plans_scores_lengths(self) -> None:
        """Raises ValueError when len(plans) != len(scores)."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        with pytest.raises(ValueError):
            await invoke_planner(
                code_block="x = 1",
                plans=["plan1", "plan2"],
                scores=[0.5],
                client=client,
            )

    async def test_raises_on_more_scores_than_plans(self) -> None:
        """Raises ValueError when scores list is longer than plans list."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        with pytest.raises(ValueError):
            await invoke_planner(
                code_block="x = 1",
                plans=["plan1"],
                scores=[0.5, 0.6],
                client=client,
            )

    async def test_raises_on_empty_code_block(self) -> None:
        """Raises ValueError when code_block is an empty string."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        with pytest.raises(ValueError, match="code_block"):
            await invoke_planner(
                code_block="",
                plans=["plan1"],
                scores=[0.5],
                client=client,
            )

    async def test_raises_on_whitespace_only_code_block(self) -> None:
        """Raises ValueError when code_block contains only whitespace."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        with pytest.raises(ValueError, match="code_block"):
            await invoke_planner(
                code_block="   \n  ",
                plans=["plan1"],
                scores=[0.5],
                client=client,
            )

    async def test_does_not_call_client_on_invalid_input(self) -> None:
        """Client is not invoked when input validation fails."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        with pytest.raises(ValueError):
            await invoke_planner(code_block="x = 1", plans=[], scores=[], client=client)

        client.send_message.assert_not_called()


# ===========================================================================
# REQ-P2I-011: invoke_planner -- Prompt Loading from Registry
# ===========================================================================


@pytest.mark.unit
class TestInvokePlannerPromptRegistry:
    """invoke_planner loads the A_planner prompt from PromptRegistry (REQ-P2I-011)."""

    async def test_registry_get_called_with_planner_agent_type(self) -> None:
        """PromptRegistry.get is called with AgentType.PLANNER."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="new plan text")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "rendered planner prompt"
            mock_registry.get.return_value = mock_template

            await invoke_planner(
                code_block="x = 1",
                plans=["previous plan"],
                scores=[0.5],
                client=client,
            )

        mock_registry.get.assert_called_once_with(AgentType.PLANNER)

    async def test_template_rendered_with_code_block_and_plan_history(self) -> None:
        """The planner template is rendered with code_block and plan_history variables."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="new plan")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_planner(
                code_block="block_marker",
                plans=["plan_a"],
                scores=[0.7],
                client=client,
            )

        assert len(render_kwargs_captured) == 1
        assert render_kwargs_captured[0]["code_block"] == "block_marker"
        assert "plan_history" in render_kwargs_captured[0]

    async def test_rendered_prompt_sent_to_client(self) -> None:
        """The rendered prompt is sent via client.send_message."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="new plan")

        expected_prompt = "rendered planner prompt content abc"

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = expected_prompt
            mock_registry.get.return_value = mock_template

            await invoke_planner(
                code_block="x = 1",
                plans=["plan"],
                scores=[0.5],
                client=client,
            )

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("message") == expected_prompt


# ===========================================================================
# REQ-P2I-012: invoke_planner -- History Formatting
# ===========================================================================


@pytest.mark.unit
class TestInvokePlannerHistoryFormatting:
    """invoke_planner formats plan history with Plan:/Score: labels (REQ-P2I-012)."""

    async def test_single_plan_score_in_history(self) -> None:
        """Single plan/score pair is formatted with numbered entry."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="new plan")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_planner(
                code_block="x = 1",
                plans=["use random forest"],
                scores=[0.85],
                client=client,
            )

        history = render_kwargs_captured[0]["plan_history"]
        assert "## Plan:" in history
        assert "use random forest" in history
        assert "## Score:" in history
        assert "0.85" in history

    async def test_multiple_plans_scores_numbered(self) -> None:
        """Multiple plan/score pairs are numbered sequentially."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="new plan")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_planner(
                code_block="x = 1",
                plans=["plan alpha", "plan beta", "plan gamma"],
                scores=[0.7, 0.8, 0.75],
                client=client,
            )

        history = render_kwargs_captured[0]["plan_history"]
        assert "plan alpha" in history
        assert "plan beta" in history
        assert "plan gamma" in history
        assert "0.7" in history
        assert "0.8" in history
        assert "0.75" in history

    async def test_none_score_rendered_as_na(self) -> None:
        """None scores are rendered as 'N/A (evaluation failed)'."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="new plan")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_planner(
                code_block="x = 1",
                plans=["plan with failure"],
                scores=[None],
                client=client,
            )

        history = render_kwargs_captured[0]["plan_history"]
        assert "N/A (evaluation failed)" in history

    async def test_mixed_none_and_float_scores(self) -> None:
        """History correctly mixes None and float scores."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="new plan")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_planner(
                code_block="x = 1",
                plans=["plan1", "plan2", "plan3"],
                scores=[0.5, None, 0.9],
                client=client,
            )

        history = render_kwargs_captured[0]["plan_history"]
        assert "0.5" in history
        assert "N/A (evaluation failed)" in history
        assert "0.9" in history

    async def test_history_format_uses_plan_score_labels(self) -> None:
        """Each entry contains '## Plan:' and '## Score:' labels."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="new plan")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_planner(
                code_block="x = 1",
                plans=["plan_a", "plan_b"],
                scores=[0.6, 0.7],
                client=client,
            )

        history = render_kwargs_captured[0]["plan_history"]
        # Should have two Plan/Score pairs
        assert history.count("## Plan:") == 2
        assert history.count("## Score:") == 2


# ===========================================================================
# REQ-P2I-013: invoke_planner -- Agent Invocation
# ===========================================================================


@pytest.mark.unit
class TestInvokePlannerAgentInvocation:
    """invoke_planner invokes the planner agent via client.send_message (REQ-P2I-013)."""

    async def test_client_invoked_with_planner_agent_type(self) -> None:
        """Client.send_message is invoked with agent_type=str(AgentType.PLANNER)."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="new plan text")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await invoke_planner(
                code_block="x = 1",
                plans=["plan1"],
                scores=[0.5],
                client=client,
            )

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("agent_type") == str(AgentType.PLANNER)

    async def test_client_invoked_exactly_once(self) -> None:
        """Client.send_message is called exactly once per invocation."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="new plan text")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await invoke_planner(
                code_block="x = 1",
                plans=["plan1"],
                scores=[0.5],
                client=client,
            )

        assert client.send_message.call_count == 1


# ===========================================================================
# REQ-P2I-014: invoke_planner -- Returns Stripped Text (No Code Extraction)
# ===========================================================================


@pytest.mark.unit
class TestInvokePlannerReturnsText:
    """invoke_planner returns stripped text, no code block extraction (REQ-P2I-014)."""

    async def test_returns_stripped_response_text(self) -> None:
        """Returns the agent's response text stripped of whitespace."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value="  Use gradient boosting with early stopping.  "
        )

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_planner(
                code_block="x = 1",
                plans=["plan1"],
                scores=[0.5],
                client=client,
            )

        assert result == "Use gradient boosting with early stopping."

    async def test_returns_multiline_response_stripped(self) -> None:
        """Returns multiline responses with leading/trailing whitespace stripped."""
        from mle_star.phase2_inner import invoke_planner

        response = "\n  Try using XGBoost.\nAlso increase the learning rate.\n  "
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=response)

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_planner(
                code_block="x = 1",
                plans=["plan1"],
                scores=[0.5],
                client=client,
            )

        assert result == "Try using XGBoost.\nAlso increase the learning rate."

    async def test_does_not_use_extract_code_block(self) -> None:
        """The planner does NOT call extract_code_block on the response."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="plan text")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block") as mock_extract,
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await invoke_planner(
                code_block="x = 1",
                plans=["plan1"],
                scores=[0.5],
                client=client,
            )

        mock_extract.assert_not_called()

    async def test_returns_string_type(self) -> None:
        """Returns a string (not None) on successful response."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="plan text here")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_planner(
                code_block="x = 1",
                plans=["plan1"],
                scores=[0.5],
                client=client,
            )

        assert isinstance(result, str)


# ===========================================================================
# REQ-P2I-015: invoke_planner -- Returns None on Empty Response
# ===========================================================================


@pytest.mark.unit
class TestInvokePlannerReturnsNoneOnFailure:
    """invoke_planner returns None when response is empty (REQ-P2I-015)."""

    async def test_returns_none_on_empty_response(self) -> None:
        """Returns None when agent response is empty string."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_planner(
                code_block="x = 1",
                plans=["plan1"],
                scores=[0.5],
                client=client,
            )

        assert result is None

    async def test_returns_none_on_whitespace_only_response(self) -> None:
        """Returns None when agent response is whitespace only."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="   \n\n  ")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_planner(
                code_block="x = 1",
                plans=["plan1"],
                scores=[0.5],
                client=client,
            )

        assert result is None

    async def test_return_is_none_type_not_empty_string(self) -> None:
        """Returns None (not empty string or other falsy) on failure."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_planner(
                code_block="x = 1",
                plans=["plan1"],
                scores=[0.5],
                client=client,
            )

        assert result is None
        assert not isinstance(result, str)


# ===========================================================================
# Prompt template integration tests for planner
# ===========================================================================


@pytest.mark.unit
class TestInvokePlannerPromptTemplateIntegration:
    """Validate that the planner prompt template exists and renders correctly."""

    def test_planner_template_exists_in_registry(self) -> None:
        """PromptRegistry contains a planner template (no variant)."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.PLANNER)
        assert template.agent_type == AgentType.PLANNER

    def test_planner_template_has_code_block_variable(self) -> None:
        """Planner template declares 'code_block' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.PLANNER)
        assert "code_block" in template.variables

    def test_planner_template_has_plan_history_variable(self) -> None:
        """Planner template declares 'plan_history' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.PLANNER)
        assert "plan_history" in template.variables

    def test_planner_template_renders_with_variables(self) -> None:
        """Planner template renders successfully with all required variables."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.PLANNER)
        rendered = template.render(
            code_block="x = 1",
            plan_history="## Plan: old\n## Score: 0.5",
            notes_context="",
            task_description="Predict target",
            evaluation_metric="accuracy",
            metric_direction="maximize",
            data_modality="tabular",
            current_score="0.85",
        )
        assert "x = 1" in rendered
        assert "## Plan: old" in rendered


# ===========================================================================
# Edge cases
# ===========================================================================


@pytest.mark.unit
class TestInvokeCoderEdgeCases:
    """Edge case tests for invoke_coder."""

    async def test_response_with_no_fences_returns_stripped_text(self) -> None:
        """When response has no code fences, returns stripped text via extract_code_block."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="  just some plain text code  ")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_coder(
                code_block="x = 1", plan="improve x", client=client
            )

        # extract_code_block returns stripped text when no fences
        assert result == "just some plain text code"

    async def test_response_with_multiple_fences_returns_longest(self) -> None:
        """When response has multiple code blocks, returns the longest one."""
        from mle_star.phase2_inner import invoke_coder

        short_code = "x = 1"
        long_code = "import numpy as np\nx = np.array([1, 2, 3])\nprint(x.shape)"
        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value=f"```python\n{short_code}\n```\nSome explanation.\n```python\n{long_code}\n```"
        )

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_coder(
                code_block="x = 1", plan="use numpy", client=client
            )

        assert result == long_code


@pytest.mark.unit
class TestInvokePlannerEdgeCases:
    """Edge case tests for invoke_planner."""

    async def test_single_plan_with_none_score(self) -> None:
        """Works correctly with a single plan that has None score."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="try something new")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_planner(
                code_block="x = 1",
                plans=["failed plan"],
                scores=[None],
                client=client,
            )

        assert result == "try something new"

    async def test_many_plans_scores(self) -> None:
        """Works correctly with many plan/score pairs (e.g., 10)."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="final plan")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            plans = [f"plan_{i}" for i in range(10)]
            scores: list[float | None] = [float(i) * 0.1 for i in range(10)]

            await invoke_planner(
                code_block="x = 1",
                plans=plans,
                scores=scores,
                client=client,
            )

        history = render_kwargs_captured[0]["plan_history"]
        # All 10 plans should appear in the history
        for i in range(10):
            assert f"plan_{i}" in history

    async def test_all_none_scores(self) -> None:
        """Works correctly when all scores are None."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="try a new approach")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_planner(
                code_block="x = 1",
                plans=["plan1", "plan2"],
                scores=[None, None],
                client=client,
            )

        history = render_kwargs_captured[0]["plan_history"]
        assert history.count("N/A (evaluation failed)") == 2

    async def test_response_with_code_fences_not_extracted(self) -> None:
        """Response with code fences is returned as-is (stripped), not extracted."""
        from mle_star.phase2_inner import invoke_planner

        response_with_fences = "Try this approach:\n```python\nsome code\n```\nThis should improve performance."
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=response_with_fences)

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_planner(
                code_block="x = 1",
                plans=["plan1"],
                scores=[0.5],
                client=client,
            )

        # Planner returns full text, NOT extracted code
        assert result == response_with_fences.strip()
        assert "Try this approach:" in result
        assert "This should improve performance." in result


# ===========================================================================
# Property-based tests
# ===========================================================================


@pytest.mark.unit
class TestInvokeCoderPropertyBased:
    """Property-based tests for invoke_coder invariants."""

    @given(
        code_block=st.text(min_size=1, max_size=200).filter(
            lambda s: s.strip() and "```" not in s
        ),
        plan=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
    )
    @settings(max_examples=30)
    async def test_valid_inputs_always_produce_result(
        self, code_block: str, plan: str
    ) -> None:
        """invoke_coder always returns str or None for valid non-empty inputs."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nimproved\n```")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_coder(code_block=code_block, plan=plan, client=client)

        assert result is None or isinstance(result, str)

    @given(
        code_block=st.text(min_size=1, max_size=200).filter(
            lambda s: s.strip() and "```" not in s
        ),
        plan=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
    )
    @settings(max_examples=30)
    async def test_template_always_rendered_with_both_variables(
        self, code_block: str, plan: str
    ) -> None:
        """Coder template is always rendered with both code_block and plan."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_coder(code_block=code_block, plan=plan, client=client)

        assert len(render_kwargs_captured) == 1
        assert render_kwargs_captured[0]["code_block"] == code_block
        assert render_kwargs_captured[0]["plan"] == plan


@pytest.mark.unit
class TestInvokePlannerPropertyBased:
    """Property-based tests for invoke_planner invariants."""

    @given(
        n_plans=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20)
    async def test_matching_plans_scores_accepted(self, n_plans: int) -> None:
        """invoke_planner accepts any plans/scores lists of equal length >= 1."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="new plan text")

        plans = [f"plan_{i}" for i in range(n_plans)]
        scores: list[float | None] = [float(i) * 0.1 for i in range(n_plans)]

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_planner(
                code_block="x = 1",
                plans=plans,
                scores=scores,
                client=client,
            )

        assert result is None or isinstance(result, str)

    @given(
        n_plans=st.integers(min_value=1, max_value=5),
        scores_strategy=st.data(),
    )
    @settings(max_examples=20)
    async def test_history_contains_all_plans(
        self, n_plans: int, scores_strategy: st.DataObject
    ) -> None:
        """History passed to the template contains every plan string."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="new plan")

        plans = [f"unique_plan_marker_{i}" for i in range(n_plans)]
        scores: list[float | None] = [
            scores_strategy.draw(
                st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0))
            )
            for _ in range(n_plans)
        ]

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_planner(
                code_block="x = 1",
                plans=plans,
                scores=scores,
                client=client,
            )

        history = render_kwargs_captured[0]["plan_history"]
        for plan in plans:
            assert plan in history

    @given(
        score=st.one_of(
            st.none(),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ),
    )
    @settings(max_examples=20)
    async def test_none_scores_rendered_as_na_float_scores_as_numbers(
        self, score: float | None
    ) -> None:
        """None scores render as 'N/A (evaluation failed)', floats as their string repr."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="new plan")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_planner(
                code_block="x = 1",
                plans=["plan"],
                scores=[score],
                client=client,
            )

        history = render_kwargs_captured[0]["plan_history"]
        if score is None:
            assert "N/A (evaluation failed)" in history
        else:
            assert str(score) in history


# ===========================================================================
# Parametrized tests for symmetry / exhaustiveness
# ===========================================================================


@pytest.mark.unit
class TestInvokeCoderParametrized:
    """Parametrized tests for invoke_coder covering multiple response formats."""

    @pytest.mark.parametrize(
        "response,expected",
        [
            ("```python\nresult_code\n```", "result_code"),
            ("```\nresult_code\n```", "result_code"),
            ("  plain code output  ", "plain code output"),
        ],
        ids=["python-fence", "generic-fence", "no-fence"],
    )
    async def test_various_response_formats(self, response: str, expected: str) -> None:
        """invoke_coder handles various response formats correctly."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=response)

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_coder(
                code_block="x = 1", plan="improve x", client=client
            )

        assert result == expected

    @pytest.mark.parametrize(
        "empty_response",
        ["", "   ", "\n\n", "  \n  \n  "],
        ids=["empty", "spaces", "newlines", "mixed-whitespace"],
    )
    async def test_empty_responses_return_none(self, empty_response: str) -> None:
        """invoke_coder returns None for various empty/whitespace responses."""
        from mle_star.phase2_inner import invoke_coder

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=empty_response)

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_coder(
                code_block="x = 1", plan="improve x", client=client
            )

        assert result is None


@pytest.mark.unit
class TestInvokePlannerParametrized:
    """Parametrized tests for invoke_planner covering multiple score formats."""

    @pytest.mark.parametrize(
        "score_value,expected_in_history",
        [
            (0.85, "0.85"),
            (0.0, "0.0"),
            (1.0, "1.0"),
            (None, "N/A (evaluation failed)"),
        ],
        ids=["float-normal", "float-zero", "float-one", "none"],
    )
    async def test_score_rendering_formats(
        self, score_value: float | None, expected_in_history: str
    ) -> None:
        """invoke_planner renders various score types correctly in history."""
        from mle_star.phase2_inner import invoke_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="new plan")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_planner(
                code_block="x = 1",
                plans=["plan"],
                scores=[score_value],
                client=client,
            )

        history = render_kwargs_captured[0]["plan_history"]
        assert expected_in_history in history
