"""Tests for Phase 3 ensemble planner and ensembler agents (Task 35).

Validates ``invoke_ens_planner``, ``invoke_ensembler``,
``_format_solutions``, and ``_format_ensemble_history`` which implement
A_ens_planner and A_ensembler for Phase 3 ensemble construction.

Tests are written TDD-first and serve as the executable specification for
REQ-P3-003, REQ-P3-009, and REQ-P3-012.

Refs:
    SRS 06a (Phase 3 Ensemble), IMPLEMENTATION_PLAN.md Task 35.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import given, settings, strategies as st
from mle_star.models import AgentType, SolutionPhase, SolutionScript
import pytest

# ---------------------------------------------------------------------------
# Module path constant for patching
# ---------------------------------------------------------------------------

_MODULE = "mle_star.phase3"


# ---------------------------------------------------------------------------
# Reusable test helpers
# ---------------------------------------------------------------------------


def _make_solution(
    content: str = "print('hello')",
    phase: SolutionPhase = SolutionPhase.REFINED,
) -> SolutionScript:
    """Create a SolutionScript for testing."""
    return SolutionScript(content=content, phase=phase)


# ===========================================================================
# TestInvokeEnsPlanner_IsAsync
# ===========================================================================


@pytest.mark.unit
class TestInvokeEnsPlannerIsAsync:
    """invoke_ens_planner is an async function."""

    def test_is_coroutine_function(self) -> None:
        """invoke_ens_planner is defined as an async function."""
        from mle_star.phase3 import invoke_ens_planner

        assert asyncio.iscoroutinefunction(invoke_ens_planner)


# ===========================================================================
# TestInvokeEnsPlanner_InputValidation (REQ-P3-003, REQ-P3-009)
# ===========================================================================


@pytest.mark.unit
class TestInvokeEnsPlannerInputValidation:
    """invoke_ens_planner raises ValueError for invalid inputs."""

    async def test_raises_on_empty_solutions(self) -> None:
        """Raises ValueError when solutions list is empty."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        with pytest.raises(
            ValueError,
            match="A_ens_planner requires at least 2 solutions for ensembling",
        ):
            await invoke_ens_planner(solutions=[], plans=[], scores=[], client=client)

    async def test_raises_on_single_solution(self) -> None:
        """Raises ValueError when solutions list has only 1 element (REQ-P3-003)."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        with pytest.raises(
            ValueError,
            match="A_ens_planner requires at least 2 solutions for ensembling",
        ):
            await invoke_ens_planner(
                solutions=[_make_solution()],
                plans=[],
                scores=[],
                client=client,
            )

    async def test_raises_on_mismatched_plans_scores(self) -> None:
        """Raises ValueError when len(plans) != len(scores) (REQ-P3-009)."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        with pytest.raises(ValueError):
            await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=["plan1", "plan2"],
                scores=[0.5],
                client=client,
            )

    async def test_raises_on_more_scores_than_plans(self) -> None:
        """Raises ValueError when scores list is longer than plans list."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        with pytest.raises(ValueError):
            await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=["plan1"],
                scores=[0.5, 0.6],
                client=client,
            )

    async def test_does_not_call_client_on_invalid_input(self) -> None:
        """Client is not invoked when input validation fails."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        with pytest.raises(ValueError):
            await invoke_ens_planner(
                solutions=[_make_solution()],
                plans=[],
                scores=[],
                client=client,
            )

        client.send_message.assert_not_called()

    async def test_accepts_exactly_two_solutions(self) -> None:
        """Does not raise when solutions has exactly 2 elements."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="ensemble plan text")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "rendered prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=[],
                scores=[],
                client=client,
            )

        assert result is not None or result is None  # No exception raised


# ===========================================================================
# TestInvokeEnsPlanner_PromptRendering
# ===========================================================================


@pytest.mark.unit
class TestInvokeEnsPlannerPromptRendering:
    """invoke_ens_planner loads the ens_planner prompt from PromptRegistry."""

    async def test_registry_get_called_with_ens_planner_agent_type(self) -> None:
        """PromptRegistry.get is called with AgentType.ENS_PLANNER."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="ensemble plan")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "rendered ens_planner prompt"
            mock_registry.get.return_value = mock_template

            await invoke_ens_planner(
                solutions=[_make_solution("s1"), _make_solution("s2")],
                plans=[],
                scores=[],
                client=client,
            )

        mock_registry.get.assert_called_once_with(AgentType.ENS_PLANNER)

    async def test_template_rendered_with_count_solutions_and_history(
        self,
    ) -> None:
        """Template is rendered with L, solutions_text, and plan_history."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="plan text")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            solutions = [_make_solution("code_a"), _make_solution("code_b")]
            await invoke_ens_planner(
                solutions=solutions,
                plans=[],
                scores=[],
                client=client,
            )

        assert len(render_kwargs_captured) == 1
        assert render_kwargs_captured[0]["L"] == 2
        assert "solutions_text" in render_kwargs_captured[0]
        assert "plan_history" in render_kwargs_captured[0]

    async def test_solution_count_equals_number_of_solutions(self) -> None:
        """L template variable equals len(solutions)."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="plan text")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            solutions = [_make_solution() for _ in range(5)]
            await invoke_ens_planner(
                solutions=solutions,
                plans=[],
                scores=[],
                client=client,
            )

        assert render_kwargs_captured[0]["L"] == 5

    async def test_rendered_prompt_sent_to_client(self) -> None:
        """The rendered prompt is sent via client.send_message."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="plan text")

        expected_prompt = "rendered ens_planner prompt xyz"

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = expected_prompt
            mock_registry.get.return_value = mock_template

            await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=[],
                scores=[],
                client=client,
            )

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("message") == expected_prompt

    async def test_client_invoked_with_ens_planner_agent_type(self) -> None:
        """Client.send_message is invoked with agent_type=str(AgentType.ENS_PLANNER)."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="plan text")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=[],
                scores=[],
                client=client,
            )

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("agent_type") == str(AgentType.ENS_PLANNER)

    async def test_client_invoked_exactly_once(self) -> None:
        """Client.send_message is called exactly once per invocation."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="plan text")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=[],
                scores=[],
                client=client,
            )

        assert client.send_message.call_count == 1


# ===========================================================================
# TestInvokeEnsPlanner_FirstInvocation (plans=[], scores=[])
# ===========================================================================


@pytest.mark.unit
class TestInvokeEnsPlannerFirstInvocation:
    """invoke_ens_planner with empty plans/scores (first round)."""

    async def test_empty_plans_scores_accepted(self) -> None:
        """Does not raise when plans=[] and scores=[]."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="first plan")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=[],
                scores=[],
                client=client,
            )

        assert isinstance(result, str)

    async def test_plan_history_empty_or_no_previous(self) -> None:
        """Plan history for first invocation is empty or indicates no previous."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="first plan")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=[],
                scores=[],
                client=client,
            )

        plan_history = render_kwargs_captured[0]["plan_history"]
        # Empty history should produce empty string or "No previous plans" sentinel
        assert plan_history == "" or "No previous" in plan_history


# ===========================================================================
# TestInvokeEnsPlanner_SubsequentInvocation (plans with history)
# ===========================================================================


@pytest.mark.unit
class TestInvokeEnsPlannerSubsequentInvocation:
    """invoke_ens_planner with previous plans/scores (subsequent rounds)."""

    async def test_history_contains_previous_plans(self) -> None:
        """Plan history includes text of previous plans."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="next plan")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=["use voting ensemble"],
                scores=[0.85],
                client=client,
            )

        plan_history = render_kwargs_captured[0]["plan_history"]
        assert "use voting ensemble" in plan_history
        assert "0.85" in plan_history

    async def test_history_contains_multiple_plans_scores(self) -> None:
        """History includes all previous plan/score pairs."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="next plan")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=["plan_alpha", "plan_beta", "plan_gamma"],
                scores=[0.7, 0.8, 0.75],
                client=client,
            )

        plan_history = render_kwargs_captured[0]["plan_history"]
        assert "plan_alpha" in plan_history
        assert "plan_beta" in plan_history
        assert "plan_gamma" in plan_history
        assert "0.7" in plan_history
        assert "0.8" in plan_history
        assert "0.75" in plan_history

    async def test_none_score_rendered_as_na(self) -> None:
        """None scores are rendered as 'N/A (evaluation failed)' in history."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="next plan")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=["failed plan"],
                scores=[None],
                client=client,
            )

        plan_history = render_kwargs_captured[0]["plan_history"]
        assert "N/A (evaluation failed)" in plan_history

    async def test_history_has_plan_score_labels(self) -> None:
        """History entries contain '## Plan:' and '## Score:' labels."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="next plan")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=["plan_a", "plan_b"],
                scores=[0.6, 0.7],
                client=client,
            )

        plan_history = render_kwargs_captured[0]["plan_history"]
        assert plan_history.count("## Plan:") == 2
        assert plan_history.count("## Score:") == 2

    async def test_mixed_none_and_float_scores(self) -> None:
        """History correctly mixes None and float scores."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="next plan")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=["plan1", "plan2", "plan3"],
                scores=[0.5, None, 0.9],
                client=client,
            )

        plan_history = render_kwargs_captured[0]["plan_history"]
        assert "0.5" in plan_history
        assert "N/A (evaluation failed)" in plan_history
        assert "0.9" in plan_history


# ===========================================================================
# TestInvokeEnsPlanner_Returns
# ===========================================================================


@pytest.mark.unit
class TestInvokeEnsPlannerReturns:
    """invoke_ens_planner returns stripped text or None."""

    async def test_returns_stripped_response_text(self) -> None:
        """Returns the agent's response text stripped of whitespace."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value="  Use weighted averaging of predictions.  "
        )

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=[],
                scores=[],
                client=client,
            )

        assert result == "Use weighted averaging of predictions."

    async def test_returns_multiline_response_stripped(self) -> None:
        """Returns multiline responses with leading/trailing whitespace stripped."""
        from mle_star.phase3 import invoke_ens_planner

        response = "\n  Stack model outputs.\nUse logistic regression.\n  "
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=response)

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=[],
                scores=[],
                client=client,
            )

        assert result == "Stack model outputs.\nUse logistic regression."

    async def test_returns_none_on_empty_response(self) -> None:
        """Returns None when agent response is empty string."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=[],
                scores=[],
                client=client,
            )

        assert result is None

    async def test_returns_none_on_whitespace_only_response(self) -> None:
        """Returns None when agent response is whitespace only."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="   \n\n  ")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=[],
                scores=[],
                client=client,
            )

        assert result is None

    async def test_return_is_none_type_not_empty_string(self) -> None:
        """Returns None (not empty string or other falsy) on empty response."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=[],
                scores=[],
                client=client,
            )

        assert result is None
        assert not isinstance(result, str)

    async def test_returns_string_type_on_success(self) -> None:
        """Returns a string when response is non-empty."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="ensemble plan here")

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=[],
                scores=[],
                client=client,
            )

        assert isinstance(result, str)


# ===========================================================================
# TestInvokeEnsembler_IsAsync
# ===========================================================================


@pytest.mark.unit
class TestInvokeEnsemblerIsAsync:
    """invoke_ensembler is an async function."""

    def test_is_coroutine_function(self) -> None:
        """invoke_ensembler is defined as an async function."""
        from mle_star.phase3 import invoke_ensembler

        assert asyncio.iscoroutinefunction(invoke_ensembler)


# ===========================================================================
# TestInvokeEnsembler_InputValidation (REQ-P3-012)
# ===========================================================================


@pytest.mark.unit
class TestInvokeEnsemblerInputValidation:
    """invoke_ensembler raises ValueError for invalid inputs (REQ-P3-012)."""

    async def test_raises_on_empty_plan(self) -> None:
        """Raises ValueError when plan is an empty string."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        with pytest.raises(
            ValueError,
            match="A_ensembler requires a non-empty ensemble plan",
        ):
            await invoke_ensembler(
                plan="",
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

    async def test_raises_on_whitespace_only_plan(self) -> None:
        """Raises ValueError when plan contains only whitespace."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        with pytest.raises(
            ValueError,
            match="A_ensembler requires a non-empty ensemble plan",
        ):
            await invoke_ensembler(
                plan="   \n  ",
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

    async def test_raises_on_single_solution(self) -> None:
        """Raises ValueError when solutions has only 1 element (REQ-P3-012)."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        with pytest.raises(
            ValueError,
            match="A_ensembler requires at least 2 solutions for ensembling",
        ):
            await invoke_ensembler(
                plan="use voting",
                solutions=[_make_solution()],
                client=client,
            )

    async def test_raises_on_empty_solutions(self) -> None:
        """Raises ValueError when solutions list is empty."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        with pytest.raises(
            ValueError,
            match="A_ensembler requires at least 2 solutions for ensembling",
        ):
            await invoke_ensembler(
                plan="use voting",
                solutions=[],
                client=client,
            )

    async def test_does_not_call_client_on_invalid_input(self) -> None:
        """Client is not invoked when input validation fails."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        with pytest.raises(ValueError):
            await invoke_ensembler(
                plan="",
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

        client.send_message.assert_not_called()

    async def test_accepts_exactly_two_solutions_with_valid_plan(self) -> None:
        """Does not raise when solutions has 2 elements and plan is non-empty."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nensemble_code\n```")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="ensemble_code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ensembler(
                plan="use voting",
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

        # No exception raised; result should be a SolutionScript or None
        assert result is None or isinstance(result, SolutionScript)


# ===========================================================================
# TestInvokeEnsembler_PromptRendering
# ===========================================================================


@pytest.mark.unit
class TestInvokeEnsemblerPromptRendering:
    """invoke_ensembler loads the ensembler prompt from PromptRegistry."""

    async def test_registry_get_called_with_ensembler_agent_type(self) -> None:
        """PromptRegistry.get is called with AgentType.ENSEMBLER."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nensemble\n```")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="ensemble"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "rendered ensembler prompt"
            mock_registry.get.return_value = mock_template

            await invoke_ensembler(
                plan="use stacking",
                solutions=[_make_solution("s1"), _make_solution("s2")],
                client=client,
            )

        mock_registry.get.assert_called_once_with(AgentType.ENSEMBLER)

    async def test_template_rendered_with_count_solutions_and_plan(self) -> None:
        """Template is rendered with L, solutions_text, and plan."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        render_kwargs_captured: list[dict[str, Any]] = []

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            solutions = [_make_solution("code_a"), _make_solution("code_b")]
            await invoke_ensembler(
                plan="use voting ensemble",
                solutions=solutions,
                client=client,
            )

        assert len(render_kwargs_captured) == 1
        assert render_kwargs_captured[0]["L"] == 2
        assert "solutions_text" in render_kwargs_captured[0]
        assert render_kwargs_captured[0]["plan"] == "use voting ensemble"

    async def test_rendered_prompt_sent_to_client(self) -> None:
        """The rendered prompt is sent via client.send_message."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        expected_prompt = "rendered ensembler prompt abc"

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = expected_prompt
            mock_registry.get.return_value = mock_template

            await invoke_ensembler(
                plan="plan",
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("message") == expected_prompt

    async def test_client_invoked_with_ensembler_agent_type(self) -> None:
        """Client.send_message is invoked with agent_type=str(AgentType.ENSEMBLER)."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await invoke_ensembler(
                plan="plan",
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("agent_type") == str(AgentType.ENSEMBLER)

    async def test_client_invoked_exactly_once(self) -> None:
        """Client.send_message is called exactly once per invocation."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await invoke_ensembler(
                plan="plan",
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

        assert client.send_message.call_count == 1


# ===========================================================================
# TestInvokeEnsembler_OutputContract
# ===========================================================================


@pytest.mark.unit
class TestInvokeEnsemblerOutputContract:
    """invoke_ensembler returns SolutionScript with correct fields."""

    async def test_returns_solution_script_on_success(self) -> None:
        """Returns a SolutionScript when code extraction succeeds."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value="```python\nensemble_code_here\n```"
        )

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="ensemble_code_here",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ensembler(
                plan="use weighted voting",
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

        assert isinstance(result, SolutionScript)

    async def test_phase_is_ensemble(self) -> None:
        """Returned SolutionScript has phase=SolutionPhase.ENSEMBLE."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ensembler(
                plan="plan",
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

        assert result is not None
        assert result.phase == SolutionPhase.ENSEMBLE

    async def test_score_is_none(self) -> None:
        """Returned SolutionScript has score=None."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ensembler(
                plan="plan",
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

        assert result is not None
        assert result.score is None

    async def test_is_executable_is_true(self) -> None:
        """Returned SolutionScript has is_executable=True."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ensembler(
                plan="plan",
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

        assert result is not None
        assert result.is_executable is True

    async def test_source_model_is_none(self) -> None:
        """Returned SolutionScript has source_model=None."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ensembler(
                plan="plan",
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

        assert result is not None
        assert result.source_model is None

    async def test_content_matches_extracted_code(self) -> None:
        """Returned SolutionScript.content matches the extracted code block."""
        from mle_star.phase3 import invoke_ensembler

        extracted_code = "import numpy as np\nresult = np.mean(predictions, axis=0)"
        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value=f"```python\n{extracted_code}\n```"
        )

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value=extracted_code,
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ensembler(
                plan="average predictions",
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

        assert result is not None
        assert result.content == extracted_code

    async def test_extract_code_block_called_on_response(self) -> None:
        """extract_code_block is invoked on the agent's response."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        agent_response = "Here is the ensemble:\n```python\nensemble_code\n```"
        client.send_message = AsyncMock(return_value=agent_response)

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="ensemble_code",
            ) as mock_extract,
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await invoke_ensembler(
                plan="plan",
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

        mock_extract.assert_called_once_with(agent_response)


# ===========================================================================
# TestInvokeEnsembler_Returns
# ===========================================================================


@pytest.mark.unit
class TestInvokeEnsemblerReturns:
    """invoke_ensembler returns None when code block extraction fails."""

    async def test_returns_none_on_empty_extraction(self) -> None:
        """Returns None when extract_code_block returns empty string."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="no code here")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ensembler(
                plan="plan",
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

        assert result is None

    async def test_returns_none_on_empty_response(self) -> None:
        """Returns None when agent response is empty string."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ensembler(
                plan="plan",
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

        assert result is None

    async def test_returns_none_on_whitespace_only_extraction(self) -> None:
        """Returns None when extract_code_block returns whitespace only."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\n   \n```")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ensembler(
                plan="plan",
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

        assert result is None

    async def test_return_is_none_type_not_solution_on_failure(self) -> None:
        """Returns None (not empty SolutionScript) on extraction failure."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ensembler(
                plan="plan",
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

        assert result is None
        assert not isinstance(result, SolutionScript)


# ===========================================================================
# TestFormatSolutions
# ===========================================================================


@pytest.mark.unit
class TestFormatSolutions:
    """_format_solutions formats solution texts as numbered sections."""

    def test_two_solutions_formatted(self) -> None:
        """Two solutions are formatted with numbered headers."""
        from mle_star.phase3 import _format_solutions

        solutions = [
            _make_solution("import pandas as pd\ndf = pd.read_csv('data.csv')"),
            _make_solution("import numpy as np\nx = np.array([1, 2])"),
        ]

        result = _format_solutions(solutions)
        assert "# 1th Python Solution" in result
        assert "# 2th Python Solution" in result

    def test_solution_content_inside_code_fence(self) -> None:
        """Solution content appears inside code fences."""
        from mle_star.phase3 import _format_solutions

        solutions = [
            _make_solution("code_marker_alpha"),
            _make_solution("code_marker_beta"),
        ]

        result = _format_solutions(solutions)
        assert "code_marker_alpha" in result
        assert "code_marker_beta" in result
        assert "```" in result

    def test_three_solutions_numbered_sequentially(self) -> None:
        """Three solutions are numbered 1, 2, 3."""
        from mle_star.phase3 import _format_solutions

        solutions = [_make_solution(f"code_{i}") for i in range(3)]

        result = _format_solutions(solutions)
        assert "# 1th Python Solution" in result
        assert "# 2th Python Solution" in result
        assert "# 3th Python Solution" in result

    def test_return_type_is_str(self) -> None:
        """Return type is str."""
        from mle_star.phase3 import _format_solutions

        solutions = [_make_solution(), _make_solution()]
        result = _format_solutions(solutions)
        assert isinstance(result, str)

    def test_each_solution_has_header_and_fence(self) -> None:
        """Each solution has a numbered header followed by a fenced code block."""
        from mle_star.phase3 import _format_solutions

        solutions = [
            _make_solution("first_solution"),
            _make_solution("second_solution"),
        ]

        result = _format_solutions(solutions)
        # Check structure: header, fence open, content, fence close
        lines = result.split("\n")
        header_count = sum(1 for line in lines if "Python Solution" in line)
        assert header_count == 2

    def test_multiline_content_preserved(self) -> None:
        """Multiline solution content is preserved inside the fence."""
        from mle_star.phase3 import _format_solutions

        multiline = "import os\nimport sys\nprint(os.getcwd())"
        solutions = [_make_solution(multiline), _make_solution("x = 1")]

        result = _format_solutions(solutions)
        assert "import os" in result
        assert "import sys" in result
        assert "print(os.getcwd())" in result


# ===========================================================================
# TestFormatEnsembleHistory
# ===========================================================================


@pytest.mark.unit
class TestFormatEnsembleHistory:
    """_format_ensemble_history formats plan/score history."""

    def test_empty_lists_returns_empty_or_no_previous(self) -> None:
        """Empty plans/scores returns empty string or 'No previous' sentinel."""
        from mle_star.phase3 import _format_ensemble_history

        result = _format_ensemble_history([], [])
        assert result == "" or "No previous" in result

    def test_single_plan_score_formatted(self) -> None:
        """Single plan/score pair is formatted with Plan/Score labels."""
        from mle_star.phase3 import _format_ensemble_history

        result = _format_ensemble_history(["use voting"], [0.85])
        assert "## Plan:" in result
        assert "use voting" in result
        assert "## Score:" in result
        assert "0.85" in result

    def test_multiple_plans_scores_all_present(self) -> None:
        """Multiple plan/score pairs are all present in the output."""
        from mle_star.phase3 import _format_ensemble_history

        plans = ["plan_alpha", "plan_beta"]
        scores: list[float | None] = [0.7, 0.8]
        result = _format_ensemble_history(plans, scores)
        assert "plan_alpha" in result
        assert "plan_beta" in result
        assert "0.7" in result
        assert "0.8" in result

    def test_none_score_rendered_as_na(self) -> None:
        """None scores are rendered as 'N/A (evaluation failed)'."""
        from mle_star.phase3 import _format_ensemble_history

        result = _format_ensemble_history(["failed plan"], [None])
        assert "N/A (evaluation failed)" in result

    def test_mixed_none_and_float_scores(self) -> None:
        """History correctly mixes None and float scores."""
        from mle_star.phase3 import _format_ensemble_history

        result = _format_ensemble_history(["plan1", "plan2", "plan3"], [0.5, None, 0.9])
        assert "0.5" in result
        assert "N/A (evaluation failed)" in result
        assert "0.9" in result

    def test_plan_score_labels_count_matches(self) -> None:
        """Number of Plan/Score labels matches number of entries."""
        from mle_star.phase3 import _format_ensemble_history

        result = _format_ensemble_history(["p1", "p2", "p3"], [0.1, 0.2, 0.3])
        assert result.count("## Plan:") == 3
        assert result.count("## Score:") == 3

    def test_return_type_is_str(self) -> None:
        """Return type is always str."""
        from mle_star.phase3 import _format_ensemble_history

        result = _format_ensemble_history([], [])
        assert isinstance(result, str)

    def test_all_none_scores(self) -> None:
        """All None scores are all rendered as N/A."""
        from mle_star.phase3 import _format_ensemble_history

        result = _format_ensemble_history(["plan1", "plan2"], [None, None])
        assert result.count("N/A (evaluation failed)") == 2


# ===========================================================================
# Property-based tests: _format_solutions
# ===========================================================================


@pytest.mark.unit
class TestFormatSolutionsPropertyBased:
    """Property-based tests for _format_solutions invariants."""

    @given(
        n_solutions=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30)
    def test_all_solution_contents_present(self, n_solutions: int) -> None:
        """Every solution's content appears in the formatted output."""
        from mle_star.phase3 import _format_solutions

        solutions = [
            _make_solution(f"unique_content_marker_{i}") for i in range(n_solutions)
        ]
        result = _format_solutions(solutions)
        for i in range(n_solutions):
            assert f"unique_content_marker_{i}" in result

    @given(
        n_solutions=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30)
    def test_all_solutions_numbered(self, n_solutions: int) -> None:
        """Every solution has a numbered header."""
        from mle_star.phase3 import _format_solutions

        solutions = [_make_solution(f"code_{i}") for i in range(n_solutions)]
        result = _format_solutions(solutions)
        for i in range(1, n_solutions + 1):
            assert f"# {i}th Python Solution" in result

    @given(
        content=st.text(min_size=1, max_size=200).filter(
            lambda s: s.strip() and "```" not in s
        ),
    )
    @settings(max_examples=30)
    def test_content_always_inside_fences(self, content: str) -> None:
        """Solution content is always enclosed in code fences."""
        from mle_star.phase3 import _format_solutions

        solutions = [_make_solution(content), _make_solution("other")]
        result = _format_solutions(solutions)
        assert content in result
        assert "```" in result


# ===========================================================================
# Property-based tests: _format_ensemble_history
# ===========================================================================


@pytest.mark.unit
class TestFormatEnsembleHistoryPropertyBased:
    """Property-based tests for _format_ensemble_history invariants."""

    @given(
        n_plans=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30)
    def test_all_plans_present_in_output(self, n_plans: int) -> None:
        """Every plan text appears in the formatted history."""
        from mle_star.phase3 import _format_ensemble_history

        plans = [f"unique_plan_marker_{i}" for i in range(n_plans)]
        scores: list[float | None] = [float(i) * 0.1 for i in range(n_plans)]
        result = _format_ensemble_history(plans, scores)
        for plan in plans:
            assert plan in result

    @given(
        score=st.one_of(
            st.none(),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ),
    )
    @settings(max_examples=20)
    def test_none_scores_rendered_as_na_float_as_numbers(
        self, score: float | None
    ) -> None:
        """None scores render as 'N/A (evaluation failed)', floats as their repr."""
        from mle_star.phase3 import _format_ensemble_history

        result = _format_ensemble_history(["plan"], [score])
        if score is None:
            assert "N/A (evaluation failed)" in result
        else:
            assert str(score) in result

    @given(
        n_entries=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=20)
    def test_empty_returns_empty_nonempty_returns_nonempty(
        self, n_entries: int
    ) -> None:
        """Empty lists return empty/'No previous', non-empty returns non-empty."""
        from mle_star.phase3 import _format_ensemble_history

        plans = [f"plan_{i}" for i in range(n_entries)]
        scores: list[float | None] = [float(i) * 0.1 for i in range(n_entries)]
        result = _format_ensemble_history(plans, scores)
        if n_entries == 0:
            assert result == "" or "No previous" in result
        else:
            assert len(result) > 0

    @given(
        n_plans=st.integers(min_value=1, max_value=5),
        scores_strategy=st.data(),
    )
    @settings(max_examples=20)
    def test_plan_score_label_count_equals_entries(
        self, n_plans: int, scores_strategy: st.DataObject
    ) -> None:
        """Number of Plan/Score labels matches number of entries."""
        from mle_star.phase3 import _format_ensemble_history

        plans = [f"plan_{i}" for i in range(n_plans)]
        scores: list[float | None] = [
            scores_strategy.draw(
                st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0))
            )
            for _ in range(n_plans)
        ]
        result = _format_ensemble_history(plans, scores)
        assert result.count("## Plan:") == n_plans
        assert result.count("## Score:") == n_plans


# ===========================================================================
# Property-based tests: invoke_ens_planner
# ===========================================================================


@pytest.mark.unit
class TestInvokeEnsPlannerPropertyBased:
    """Property-based tests for invoke_ens_planner invariants."""

    @given(
        n_solutions=st.integers(min_value=2, max_value=8),
    )
    @settings(max_examples=20)
    async def test_valid_solution_count_accepted(self, n_solutions: int) -> None:
        """invoke_ens_planner accepts any number of solutions >= 2."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="plan text")

        solutions = [_make_solution(f"code_{i}") for i in range(n_solutions)]

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ens_planner(
                solutions=solutions,
                plans=[],
                scores=[],
                client=client,
            )

        assert result is None or isinstance(result, str)

    @given(
        n_plans=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=20)
    async def test_matching_plans_scores_accepted(self, n_plans: int) -> None:
        """invoke_ens_planner accepts any plans/scores lists of equal length."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="plan text")

        plans = [f"plan_{i}" for i in range(n_plans)]
        scores: list[float | None] = [float(i) * 0.1 for i in range(n_plans)]

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=plans,
                scores=scores,
                client=client,
            )

        assert result is None or isinstance(result, str)


# ===========================================================================
# Property-based tests: invoke_ensembler
# ===========================================================================


@pytest.mark.unit
class TestInvokeEnsemblerPropertyBased:
    """Property-based tests for invoke_ensembler invariants."""

    @given(
        n_solutions=st.integers(min_value=2, max_value=8),
    )
    @settings(max_examples=20)
    async def test_valid_solution_count_accepted(self, n_solutions: int) -> None:
        """invoke_ensembler accepts any number of solutions >= 2."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nensemble_code\n```")

        solutions = [_make_solution(f"code_{i}") for i in range(n_solutions)]

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="ensemble_code",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ensembler(
                plan="use voting",
                solutions=solutions,
                client=client,
            )

        assert result is None or isinstance(result, SolutionScript)

    @given(
        plan=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
    )
    @settings(max_examples=20)
    async def test_valid_plan_always_produces_result(self, plan: str) -> None:
        """invoke_ensembler always returns SolutionScript or None for valid plan."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ensembler(
                plan=plan,
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

        assert result is None or isinstance(result, SolutionScript)

    @given(
        plan=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
    )
    @settings(max_examples=20)
    async def test_successful_result_always_has_ensemble_phase(self, plan: str) -> None:
        """When result is not None, phase is always ENSEMBLE."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ensembler(
                plan=plan,
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

        if result is not None:
            assert result.phase == SolutionPhase.ENSEMBLE
            assert result.score is None
            assert result.source_model is None
            assert result.is_executable is True


# ===========================================================================
# Parametrized tests
# ===========================================================================


@pytest.mark.unit
class TestInvokeEnsPlannerParametrized:
    """Parametrized tests for invoke_ens_planner response handling."""

    @pytest.mark.parametrize(
        "empty_response",
        ["", "   ", "\n\n", "  \n  \n  "],
        ids=["empty", "spaces", "newlines", "mixed-whitespace"],
    )
    async def test_empty_responses_return_none(self, empty_response: str) -> None:
        """invoke_ens_planner returns None for various empty/whitespace responses."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=empty_response)

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=[],
                scores=[],
                client=client,
            )

        assert result is None

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
        """invoke_ens_planner renders various score types correctly in history."""
        from mle_star.phase3 import invoke_ens_planner

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

            await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=["plan"],
                scores=[score_value],
                client=client,
            )

        plan_history = render_kwargs_captured[0]["plan_history"]
        assert expected_in_history in plan_history


@pytest.mark.unit
class TestInvokeEnsemblerParametrized:
    """Parametrized tests for invoke_ensembler response handling."""

    @pytest.mark.parametrize(
        "invalid_plan",
        ["", "   ", "\n\n", "  \n  \t  "],
        ids=["empty", "spaces", "newlines", "mixed-whitespace"],
    )
    async def test_various_empty_plans_raise_value_error(
        self, invalid_plan: str
    ) -> None:
        """invoke_ensembler raises ValueError for various empty/whitespace plans."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        with pytest.raises(ValueError, match="non-empty ensemble plan"):
            await invoke_ensembler(
                plan=invalid_plan,
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

    @pytest.mark.parametrize(
        "n_solutions",
        [0, 1],
        ids=["zero-solutions", "one-solution"],
    )
    async def test_insufficient_solutions_raise_value_error(
        self, n_solutions: int
    ) -> None:
        """invoke_ensembler raises ValueError for < 2 solutions."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        solutions = [_make_solution() for _ in range(n_solutions)]
        with pytest.raises(ValueError, match="at least 2 solutions"):
            await invoke_ensembler(
                plan="use voting",
                solutions=solutions,
                client=client,
            )


# ===========================================================================
# Edge cases
# ===========================================================================


@pytest.mark.unit
class TestInvokeEnsPlannerEdgeCases:
    """Edge case tests for invoke_ens_planner."""

    async def test_many_solutions_accepted(self) -> None:
        """Works correctly with many solutions (e.g., 10)."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="big ensemble plan")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            solutions = [_make_solution(f"code_{i}") for i in range(10)]
            await invoke_ens_planner(
                solutions=solutions,
                plans=[],
                scores=[],
                client=client,
            )

        assert render_kwargs_captured[0]["L"] == 10

    async def test_solutions_text_contains_all_solution_contents(self) -> None:
        """solutions_text template variable contains all solution contents."""
        from mle_star.phase3 import invoke_ens_planner

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="plan")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            solutions = [
                _make_solution("content_unique_alpha"),
                _make_solution("content_unique_beta"),
            ]
            await invoke_ens_planner(
                solutions=solutions,
                plans=[],
                scores=[],
                client=client,
            )

        solutions_text = render_kwargs_captured[0]["solutions_text"]
        assert "content_unique_alpha" in solutions_text
        assert "content_unique_beta" in solutions_text

    async def test_many_plans_scores_history(self) -> None:
        """Works correctly with many plan/score pairs (e.g., 10)."""
        from mle_star.phase3 import invoke_ens_planner

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

            await invoke_ens_planner(
                solutions=[_make_solution(), _make_solution()],
                plans=plans,
                scores=scores,
                client=client,
            )

        plan_history = render_kwargs_captured[0]["plan_history"]
        for i in range(10):
            assert f"plan_{i}" in plan_history


@pytest.mark.unit
class TestInvokeEnsemblerEdgeCases:
    """Edge case tests for invoke_ensembler."""

    async def test_many_solutions_accepted(self) -> None:
        """Works correctly with many solutions (e.g., 10)."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nensemble_code\n```")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="ensemble_code",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            solutions = [_make_solution(f"code_{i}") for i in range(10)]
            result = await invoke_ensembler(
                plan="big ensemble",
                solutions=solutions,
                client=client,
            )

        assert isinstance(result, SolutionScript)
        assert result.content == "ensemble_code"

    async def test_solutions_text_contains_all_contents(self) -> None:
        """solutions_text template variable contains all solution contents."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\ncode\n```")

        render_kwargs_captured: list[dict[str, Any]] = []

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            solutions = [
                _make_solution("unique_sol_one"),
                _make_solution("unique_sol_two"),
            ]
            await invoke_ensembler(
                plan="plan",
                solutions=solutions,
                client=client,
            )

        solutions_text = render_kwargs_captured[0]["solutions_text"]
        assert "unique_sol_one" in solutions_text
        assert "unique_sol_two" in solutions_text

    async def test_response_with_multiple_code_blocks_uses_extract(self) -> None:
        """Response with multiple code blocks delegates to extract_code_block."""
        from mle_star.phase3 import invoke_ensembler

        response = "```python\nshort\n```\nText\n```python\nimport numpy\nx = 1\n```"
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=response)

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="import numpy\nx = 1",
            ) as mock_extract,
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await invoke_ensembler(
                plan="plan",
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

        mock_extract.assert_called_once_with(response)
        assert result is not None
        assert result.content == "import numpy\nx = 1"

    async def test_plan_validation_checked_before_solutions(self) -> None:
        """Plan validation error is raised even with valid solutions."""
        from mle_star.phase3 import invoke_ensembler

        client = AsyncMock()
        with pytest.raises(ValueError, match="non-empty ensemble plan"):
            await invoke_ensembler(
                plan="",
                solutions=[_make_solution(), _make_solution()],
                client=client,
            )

        client.send_message.assert_not_called()
