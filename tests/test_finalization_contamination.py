"""Tests for contamination check and run_finalization orchestration (Task 40).

Validates ``check_contamination`` which invokes the A_test agent with
``variant="contamination_check"`` for each reference discussion and aggregates
verdicts, and ``run_finalization`` which orchestrates the full finalization
pipeline: subsampling removal, test submission generation, leakage check,
evaluation, submission verification, and contamination check.

Tests are written TDD-first and serve as the executable specification for
REQ-FN-026 through REQ-FN-036 and REQ-FN-041.

Refs:
    SRS 08c -- Finalization Contamination.
    SRS 08d -- Finalization Orchestration.
    IMPLEMENTATION_PLAN.md Task 40.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    AgentType,
    CodeBlock,
    CodeBlockCategory,
    DataContaminationResult,
    DataModality,
    EvaluationResult,
    FinalResult,
    MetricDirection,
    Phase1Result,
    Phase2Result,
    Phase3Result,
    PipelineConfig,
    RetrievedModel,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
    TaskType,
)
import pytest

# ---------------------------------------------------------------------------
# Module path constant for patching
# ---------------------------------------------------------------------------

_MODULE = "mle_star.finalization"

# ---------------------------------------------------------------------------
# Reusable test helpers
# ---------------------------------------------------------------------------


def _make_task(**overrides: Any) -> TaskDescription:
    """Build a valid TaskDescription with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed TaskDescription instance.
    """
    defaults: dict[str, Any] = {
        "competition_id": "test-comp",
        "task_type": TaskType.CLASSIFICATION,
        "data_modality": DataModality.TABULAR,
        "evaluation_metric": "accuracy",
        "metric_direction": MetricDirection.MAXIMIZE,
        "description": "Test task description",
        "data_dir": "./input",
        "output_dir": "./final",
    }
    defaults.update(overrides)
    return TaskDescription(**defaults)


def _make_solution(**overrides: Any) -> SolutionScript:
    """Build a valid SolutionScript with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed SolutionScript instance.
    """
    defaults: dict[str, Any] = {
        "content": "import pandas as pd\nprint('hello')\n",
        "phase": SolutionPhase.REFINED,
    }
    defaults.update(overrides)
    return SolutionScript(**defaults)


def _make_config(**overrides: Any) -> PipelineConfig:
    """Build a valid PipelineConfig with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed PipelineConfig instance.
    """
    defaults: dict[str, Any] = {}
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_phase1_result(**overrides: Any) -> Phase1Result:
    """Build a valid Phase1Result with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed Phase1Result instance.
    """
    defaults: dict[str, Any] = {
        "retrieved_models": [
            RetrievedModel(model_name="xgb", example_code="import xgb")
        ],
        "candidate_solutions": [_make_solution(phase=SolutionPhase.INIT)],
        "candidate_scores": [0.85],
        "initial_solution": _make_solution(phase=SolutionPhase.INIT),
        "initial_score": 0.85,
    }
    defaults.update(overrides)
    return Phase1Result(**defaults)


def _make_phase2_result(**overrides: Any) -> Phase2Result:
    """Build a valid Phase2Result with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed Phase2Result instance.
    """
    defaults: dict[str, Any] = {
        "ablation_summaries": ["summary"],
        "refined_blocks": [
            CodeBlock(content="block", category=CodeBlockCategory.TRAINING)
        ],
        "best_solution": _make_solution(phase=SolutionPhase.REFINED),
        "best_score": 0.90,
        "step_history": [],
    }
    defaults.update(overrides)
    return Phase2Result(**defaults)


def _make_phase3_result(**overrides: Any) -> Phase3Result:
    """Build a valid Phase3Result with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed Phase3Result instance.
    """
    defaults: dict[str, Any] = {
        "input_solutions": [
            _make_solution(phase=SolutionPhase.REFINED),
            _make_solution(phase=SolutionPhase.REFINED, content="sol2"),
        ],
        "ensemble_plans": ["plan1"],
        "ensemble_scores": [0.92],
        "best_ensemble": _make_solution(phase=SolutionPhase.ENSEMBLE),
        "best_ensemble_score": 0.92,
    }
    defaults.update(overrides)
    return Phase3Result(**defaults)


def _make_eval_result(**overrides: Any) -> EvaluationResult:
    """Build a valid EvaluationResult with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed EvaluationResult instance.
    """
    defaults: dict[str, Any] = {
        "stdout": "Final Validation Performance: 0.90",
        "stderr": "",
        "exit_code": 0,
        "duration_seconds": 1.0,
        "is_error": False,
        "score": 0.90,
    }
    defaults.update(overrides)
    return EvaluationResult(**defaults)


# ===========================================================================
# check_contamination -- Is Async
# ===========================================================================


@pytest.mark.unit
class TestCheckContaminationIsAsync:
    """check_contamination is an async function (REQ-FN-026)."""

    def test_is_coroutine_function(self) -> None:
        """check_contamination is defined as an async function."""
        from mle_star.finalization import check_contamination

        assert asyncio.iscoroutinefunction(check_contamination)


# ===========================================================================
# REQ-FN-027: None references -> returns None, logs INFO
# ===========================================================================


@pytest.mark.unit
class TestCheckContaminationNoneReferences:
    """check_contamination returns None when reference_discussions is None (REQ-FN-027)."""

    async def test_none_references_returns_none(self) -> None:
        """Returns None when reference_discussions is None."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        solution = _make_solution()

        result = await check_contamination(client, solution, None)

        assert result is None

    async def test_none_references_no_client_call(self) -> None:
        """No client.send_message call when reference_discussions is None."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        client.send_message = AsyncMock()
        solution = _make_solution()

        await check_contamination(client, solution, None)

        client.send_message.assert_not_called()

    async def test_none_references_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Logs INFO when skipping due to None references."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        solution = _make_solution()

        with caplog.at_level(logging.INFO, logger=_MODULE):
            await check_contamination(client, solution, None)

        assert any(r.levelno == logging.INFO for r in caplog.records)


# ===========================================================================
# REQ-FN-027: Empty references -> returns None, logs INFO
# ===========================================================================


@pytest.mark.unit
class TestCheckContaminationEmptyReferences:
    """check_contamination returns None when reference_discussions is [] (REQ-FN-027)."""

    async def test_empty_list_returns_none(self) -> None:
        """Returns None when reference_discussions is an empty list."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        solution = _make_solution()

        result = await check_contamination(client, solution, [])

        assert result is None

    async def test_empty_list_no_client_call(self) -> None:
        """No client.send_message call when reference_discussions is empty."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        client.send_message = AsyncMock()
        solution = _make_solution()

        await check_contamination(client, solution, [])

        client.send_message.assert_not_called()

    async def test_empty_list_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Logs INFO when skipping due to empty references."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        solution = _make_solution()

        with caplog.at_level(logging.INFO, logger=_MODULE):
            await check_contamination(client, solution, [])

        assert any(r.levelno == logging.INFO for r in caplog.records)


# ===========================================================================
# REQ-FN-028: Single reference returning Novel
# ===========================================================================


@pytest.mark.unit
class TestCheckContaminationSingleNovel:
    """Single reference returning Novel verdict (REQ-FN-028)."""

    async def test_single_novel_returns_novel_result(self) -> None:
        """Single Novel reference produces overall Novel verdict."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        novel_json = DataContaminationResult(verdict="Novel").model_dump_json()
        client.send_message = AsyncMock(return_value=novel_json)

        solution = _make_solution()

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_contamination(
                client, solution, ["reference discussion 1"]
            )

        assert result is not None
        assert isinstance(result, DataContaminationResult)
        assert result.verdict == "Novel"

    async def test_single_novel_one_client_call(self) -> None:
        """Exactly one client.send_message call for a single reference."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        novel_json = DataContaminationResult(verdict="Novel").model_dump_json()
        client.send_message = AsyncMock(return_value=novel_json)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await check_contamination(client, _make_solution(), ["single ref"])

        assert client.send_message.call_count == 1


# ===========================================================================
# REQ-FN-028: Single reference returning Same
# ===========================================================================


@pytest.mark.unit
class TestCheckContaminationSingleSame:
    """Single reference returning Same verdict (REQ-FN-028)."""

    async def test_single_same_returns_same_result(self) -> None:
        """Single Same reference produces overall Same verdict."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        same_json = DataContaminationResult(verdict="Same").model_dump_json()
        client.send_message = AsyncMock(return_value=same_json)

        solution = _make_solution()

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_contamination(
                client, solution, ["reference discussion"]
            )

        assert result is not None
        assert result.verdict == "Same"


# ===========================================================================
# REQ-FN-030: Multiple references, all Novel -> overall Novel
# ===========================================================================


@pytest.mark.unit
class TestCheckContaminationMultipleAllNovel:
    """Multiple references all returning Novel produce overall Novel (REQ-FN-030)."""

    async def test_all_novel_returns_novel(self) -> None:
        """All Novel verdicts produce overall Novel verdict."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        novel_json = DataContaminationResult(verdict="Novel").model_dump_json()
        client.send_message = AsyncMock(return_value=novel_json)

        refs = ["ref1", "ref2", "ref3"]

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_contamination(client, _make_solution(), refs)

        assert result is not None
        assert result.verdict == "Novel"

    async def test_all_novel_calls_client_per_reference(self) -> None:
        """Client is called once per reference discussion."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        novel_json = DataContaminationResult(verdict="Novel").model_dump_json()
        client.send_message = AsyncMock(return_value=novel_json)

        refs = ["ref1", "ref2", "ref3"]

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await check_contamination(client, _make_solution(), refs)

        assert client.send_message.call_count == 3


# ===========================================================================
# REQ-FN-031: Multiple references, any Same -> overall Same
# ===========================================================================


@pytest.mark.unit
class TestCheckContaminationMultipleAnySame:
    """At least one Same verdict produces overall Same (REQ-FN-031)."""

    async def test_any_same_returns_same(self) -> None:
        """One Same among Novels produces overall Same verdict."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        novel_json = DataContaminationResult(verdict="Novel").model_dump_json()
        same_json = DataContaminationResult(verdict="Same").model_dump_json()
        # Second reference returns Same
        client.send_message = AsyncMock(side_effect=[novel_json, same_json, novel_json])

        refs = ["ref1", "ref2", "ref3"]

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_contamination(client, _make_solution(), refs)

        assert result is not None
        assert result.verdict == "Same"

    async def test_first_same_returns_same(self) -> None:
        """Same as first verdict produces overall Same."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        same_json = DataContaminationResult(verdict="Same").model_dump_json()
        novel_json = DataContaminationResult(verdict="Novel").model_dump_json()
        client.send_message = AsyncMock(side_effect=[same_json, novel_json])

        refs = ["ref1", "ref2"]

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_contamination(client, _make_solution(), refs)

        assert result is not None
        assert result.verdict == "Same"

    async def test_last_same_returns_same(self) -> None:
        """Same as last verdict produces overall Same."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        novel_json = DataContaminationResult(verdict="Novel").model_dump_json()
        same_json = DataContaminationResult(verdict="Same").model_dump_json()
        client.send_message = AsyncMock(side_effect=[novel_json, novel_json, same_json])

        refs = ["ref1", "ref2", "ref3"]

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_contamination(client, _make_solution(), refs)

        assert result is not None
        assert result.verdict == "Same"

    async def test_all_same_returns_same(self) -> None:
        """All Same verdicts produce overall Same."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        same_json = DataContaminationResult(verdict="Same").model_dump_json()
        client.send_message = AsyncMock(return_value=same_json)

        refs = ["ref1", "ref2"]

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_contamination(client, _make_solution(), refs)

        assert result is not None
        assert result.verdict == "Same"


# ===========================================================================
# REQ-FN-029: Agent invocation uses correct template variables
# ===========================================================================


@pytest.mark.unit
class TestCheckContaminationPromptRendering:
    """Agent prompt uses correct template variables (REQ-FN-029)."""

    async def test_template_rendered_with_reference_discussion(self) -> None:
        """Template render receives reference_discussion variable."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        novel_json = DataContaminationResult(verdict="Novel").model_dump_json()
        client.send_message = AsyncMock(return_value=novel_json)

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await check_contamination(client, _make_solution(), ["my discussion text"])

        assert len(render_kwargs_captured) == 1
        assert "reference_discussion" in render_kwargs_captured[0]
        assert render_kwargs_captured[0]["reference_discussion"] == "my discussion text"

    async def test_template_rendered_with_final_solution(self) -> None:
        """Template render receives final_solution from solution.content."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        novel_json = DataContaminationResult(verdict="Novel").model_dump_json()
        client.send_message = AsyncMock(return_value=novel_json)

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            custom_content = "unique_solution_content_marker"
            await check_contamination(
                client,
                _make_solution(content=custom_content),
                ["ref"],
            )

        assert len(render_kwargs_captured) == 1
        assert "final_solution" in render_kwargs_captured[0]
        assert render_kwargs_captured[0]["final_solution"] == custom_content

    async def test_each_reference_rendered_with_own_discussion(self) -> None:
        """Each reference gets its own discussion text in the rendered prompt."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        novel_json = DataContaminationResult(verdict="Novel").model_dump_json()
        client.send_message = AsyncMock(return_value=novel_json)

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            refs = ["discussion_A", "discussion_B", "discussion_C"]
            await check_contamination(client, _make_solution(), refs)

        assert len(render_kwargs_captured) == 3
        assert render_kwargs_captured[0]["reference_discussion"] == "discussion_A"
        assert render_kwargs_captured[1]["reference_discussion"] == "discussion_B"
        assert render_kwargs_captured[2]["reference_discussion"] == "discussion_C"


# ===========================================================================
# REQ-FN-028: Uses contamination_check variant and output_format
# ===========================================================================


@pytest.mark.unit
class TestCheckContaminationAgentInvocation:
    """Agent invoked with correct variant and output_format (REQ-FN-028)."""

    async def test_registry_get_uses_contamination_check_variant(self) -> None:
        """PromptRegistry.get called with variant='contamination_check'."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        novel_json = DataContaminationResult(verdict="Novel").model_dump_json()
        client.send_message = AsyncMock(return_value=novel_json)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await check_contamination(client, _make_solution(), ["ref"])

        mock_registry.get.assert_called_with(
            AgentType.TEST, variant="contamination_check"
        )

    async def test_client_called_with_test_agent_type(self) -> None:
        """client.send_message uses agent_type=str(AgentType.TEST)."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        novel_json = DataContaminationResult(verdict="Novel").model_dump_json()
        client.send_message = AsyncMock(return_value=novel_json)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await check_contamination(client, _make_solution(), ["ref"])

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("agent_type") == str(AgentType.TEST)

    async def test_client_called_with_output_format(self) -> None:
        """client.send_message includes output_format with DataContaminationResult schema."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        novel_json = DataContaminationResult(verdict="Novel").model_dump_json()
        client.send_message = AsyncMock(return_value=novel_json)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await check_contamination(client, _make_solution(), ["ref"])

        call_kwargs = client.send_message.call_args[1]
        expected_format = {
            "type": "json_schema",
            "schema": DataContaminationResult.model_json_schema(),
        }
        assert call_kwargs.get("output_format") == expected_format


# ===========================================================================
# REQ-FN-032: Logs results at INFO
# ===========================================================================


@pytest.mark.unit
class TestCheckContaminationLogging:
    """check_contamination logs results at INFO level (REQ-FN-032)."""

    async def test_logs_skip_on_none_references(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """INFO log emitted when references is None."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()

        with caplog.at_level(logging.INFO, logger=_MODULE):
            await check_contamination(client, _make_solution(), None)

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_records) >= 1

    async def test_logs_skip_on_empty_references(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """INFO log emitted when references is []."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()

        with caplog.at_level(logging.INFO, logger=_MODULE):
            await check_contamination(client, _make_solution(), [])

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_records) >= 1

    async def test_logs_results_on_success(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """INFO log emitted with per-reference verdicts when check succeeds."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        novel_json = DataContaminationResult(verdict="Novel").model_dump_json()
        client.send_message = AsyncMock(return_value=novel_json)

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            caplog.at_level(logging.INFO, logger=_MODULE),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await check_contamination(client, _make_solution(), ["ref1", "ref2"])

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_records) >= 1
        # Should contain reference count or verdict info
        log_text = " ".join(r.message for r in info_records)
        assert "2" in log_text or "Novel" in log_text


# ===========================================================================
# REQ-FN-041: Graceful degradation on parse failure
# ===========================================================================


@pytest.mark.unit
class TestCheckContaminationGracefulDegradation:
    """Malformed response returns None (REQ-FN-041)."""

    async def test_malformed_json_returns_none(self) -> None:
        """Malformed JSON response returns None instead of crashing."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="not valid json at all")

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_contamination(client, _make_solution(), ["ref"])

        assert result is None

    async def test_invalid_verdict_value_returns_none(self) -> None:
        """JSON with invalid verdict value returns None."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        client.send_message = AsyncMock(return_value='{"verdict": "Invalid"}')

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_contamination(client, _make_solution(), ["ref"])

        assert result is None

    async def test_client_exception_returns_none(self) -> None:
        """Client exception returns None instead of propagating."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=RuntimeError("API down"))

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_contamination(client, _make_solution(), ["ref"])

        assert result is None

    async def test_empty_string_response_returns_none(self) -> None:
        """Empty string response returns None."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_contamination(client, _make_solution(), ["ref"])

        assert result is None

    async def test_missing_verdict_key_returns_none(self) -> None:
        """JSON without verdict key returns None."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        client.send_message = AsyncMock(return_value='{"wrong_key": "Novel"}')

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_contamination(client, _make_solution(), ["ref"])

        assert result is None


# ===========================================================================
# check_contamination -- Parametrized tests
# ===========================================================================


@pytest.mark.unit
class TestCheckContaminationParametrized:
    """Parametrized tests covering multiple verdict combinations."""

    @pytest.mark.parametrize(
        "verdicts,expected_overall",
        [
            (["Novel"], "Novel"),
            (["Same"], "Same"),
            (["Novel", "Novel"], "Novel"),
            (["Novel", "Same"], "Same"),
            (["Same", "Novel"], "Same"),
            (["Same", "Same"], "Same"),
            (["Novel", "Novel", "Novel"], "Novel"),
            (["Novel", "Novel", "Same"], "Same"),
            (["Same", "Novel", "Novel"], "Same"),
            (["Novel", "Same", "Novel"], "Same"),
        ],
        ids=[
            "single-novel",
            "single-same",
            "two-novel",
            "novel-then-same",
            "same-then-novel",
            "two-same",
            "three-novel",
            "two-novel-one-same",
            "same-first-then-novel",
            "same-middle",
        ],
    )
    async def test_verdict_aggregation(
        self,
        verdicts: list[str],
        expected_overall: str,
    ) -> None:
        """Verdict aggregation follows ANY-Same -> Same, ALL-Novel -> Novel."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        responses = [
            DataContaminationResult(verdict=v).model_dump_json()  # type: ignore[arg-type]
            for v in verdicts
        ]
        client.send_message = AsyncMock(side_effect=responses)

        refs = [f"ref{i}" for i in range(len(verdicts))]

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_contamination(client, _make_solution(), refs)

        assert result is not None
        assert result.verdict == expected_overall


# ===========================================================================
# check_contamination -- Property-based tests
# ===========================================================================


@pytest.mark.unit
class TestCheckContaminationPropertyBased:
    """Property-based tests for check_contamination invariants."""

    @given(
        num_refs=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20)
    async def test_all_novel_always_produces_novel(self, num_refs: int) -> None:
        """When all references return Novel, overall verdict is always Novel."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        novel_json = DataContaminationResult(verdict="Novel").model_dump_json()
        client.send_message = AsyncMock(return_value=novel_json)

        refs = [f"ref{i}" for i in range(num_refs)]

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_contamination(client, _make_solution(), refs)

        assert result is not None
        assert result.verdict == "Novel"

    @given(
        num_refs=st.integers(min_value=2, max_value=10),
        same_index=st.integers(min_value=0),
    )
    @settings(max_examples=20)
    async def test_any_same_always_produces_same(
        self, num_refs: int, same_index: int
    ) -> None:
        """When any reference returns Same, overall verdict is always Same."""
        from mle_star.finalization import check_contamination

        same_idx = same_index % num_refs

        client = AsyncMock()
        novel_json = DataContaminationResult(verdict="Novel").model_dump_json()
        same_json = DataContaminationResult(verdict="Same").model_dump_json()

        responses = [novel_json] * num_refs
        responses[same_idx] = same_json
        client.send_message = AsyncMock(side_effect=responses)

        refs = [f"ref{i}" for i in range(num_refs)]

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_contamination(client, _make_solution(), refs)

        assert result is not None
        assert result.verdict == "Same"

    @given(
        num_refs=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20)
    async def test_client_called_exactly_per_reference(self, num_refs: int) -> None:
        """Client is called exactly once per reference discussion."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        novel_json = DataContaminationResult(verdict="Novel").model_dump_json()
        client.send_message = AsyncMock(return_value=novel_json)

        refs = [f"ref{i}" for i in range(num_refs)]

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await check_contamination(client, _make_solution(), refs)

        assert client.send_message.call_count == num_refs

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                whitelist_characters="_= \n",
            ),
            min_size=5,
            max_size=100,
        ),
    )
    @settings(max_examples=15)
    async def test_solution_content_passed_to_template(self, content: str) -> None:
        """Solution content is always passed as final_solution to template."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        novel_json = DataContaminationResult(verdict="Novel").model_dump_json()
        client.send_message = AsyncMock(return_value=novel_json)

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await check_contamination(client, _make_solution(content=content), ["ref"])

        assert len(render_kwargs_captured) == 1
        assert render_kwargs_captured[0]["final_solution"] == content


# ===========================================================================
# run_finalization -- Is Async
# ===========================================================================


@pytest.mark.unit
class TestRunFinalizationIsAsync:
    """run_finalization is an async function (REQ-FN-034)."""

    def test_is_coroutine_function(self) -> None:
        """run_finalization is defined as an async function."""
        from mle_star.finalization import run_finalization

        assert asyncio.iscoroutinefunction(run_finalization)


# ===========================================================================
# REQ-FN-034: run_finalization happy path
# ===========================================================================


@pytest.mark.unit
class TestRunFinalizationHappyPath:
    """run_finalization returns FinalResult on success (REQ-FN-034)."""

    async def test_returns_final_result(self) -> None:
        """Returns a FinalResult instance on successful pipeline."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        solution = _make_solution()
        task = _make_task()
        config = _make_config()
        phase1 = _make_phase1_result()
        phase2_list = [_make_phase2_result()]
        phase3 = _make_phase3_result()

        no_subsample = _make_solution(
            content="no_subsample", phase=SolutionPhase.REFINED
        )
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)
        leakage_checked = _make_solution(
            content="leakage_checked", phase=SolutionPhase.FINAL
        )
        eval_result = _make_eval_result(score=0.91)

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=no_subsample,
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=leakage_checked,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(leakage_checked, eval_result),
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=MagicMock(),
            ),
            patch(
                f"{_MODULE}.verify_submission",
                return_value=True,
            ),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/abs/path/final/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await run_finalization(
                client=client,
                solution=solution,
                task=task,
                config=config,
                phase1_result=phase1,
                phase2_results=phase2_list,
                phase3_result=phase3,
            )

        assert isinstance(result, FinalResult)


# ===========================================================================
# REQ-FN-035: Pipeline step order
# ===========================================================================


@pytest.mark.unit
class TestRunFinalizationPipelineOrder:
    """run_finalization calls steps in correct order (REQ-FN-035)."""

    async def test_pipeline_step_order(self) -> None:
        """Steps called in order: remove_subsampling -> generate_test_submission -> check_and_fix_leakage -> evaluate_with_retry -> verify/get_submission."""
        from mle_star.finalization import run_finalization

        call_order: list[str] = []
        client = AsyncMock()
        solution = _make_solution()
        task = _make_task()
        config = _make_config()

        no_subsample = _make_solution(
            content="no_subsample", phase=SolutionPhase.REFINED
        )
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)
        leakage_checked = _make_solution(
            content="leakage_checked", phase=SolutionPhase.FINAL
        )
        eval_result = _make_eval_result(score=0.91)

        async def mock_remove_subsampling(*args: Any, **kwargs: Any) -> SolutionScript:
            call_order.append("remove_subsampling")
            return no_subsample

        async def mock_generate_test(*args: Any, **kwargs: Any) -> SolutionScript:
            call_order.append("generate_test_submission")
            return test_script

        async def mock_leakage(*args: Any, **kwargs: Any) -> SolutionScript:
            call_order.append("check_and_fix_leakage")
            return leakage_checked

        async def mock_eval(
            *args: Any, **kwargs: Any
        ) -> tuple[SolutionScript, EvaluationResult]:
            call_order.append("evaluate_with_retry")
            return (leakage_checked, eval_result)

        def mock_verify(*args: Any, **kwargs: Any) -> bool:
            call_order.append("verify_submission")
            return True

        def mock_submission_info(*args: Any, **kwargs: Any) -> dict[str, Any]:
            call_order.append("get_submission_info")
            return {
                "exists": True,
                "path": "/abs/path/final/submission.csv",
                "size_bytes": 100,
                "row_count": 10,
            }

        with (
            patch(f"{_MODULE}.remove_subsampling", side_effect=mock_remove_subsampling),
            patch(
                f"{_MODULE}.generate_test_submission", side_effect=mock_generate_test
            ),
            patch(f"{_MODULE}.check_and_fix_leakage", side_effect=mock_leakage),
            patch(f"{_MODULE}.evaluate_with_retry", side_effect=mock_eval),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", side_effect=mock_verify),
            patch(f"{_MODULE}.get_submission_info", side_effect=mock_submission_info),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            await run_finalization(
                client=client,
                solution=solution,
                task=task,
                config=config,
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        assert call_order.index("remove_subsampling") < call_order.index(
            "generate_test_submission"
        )
        assert call_order.index("generate_test_submission") < call_order.index(
            "check_and_fix_leakage"
        )
        assert call_order.index("check_and_fix_leakage") < call_order.index(
            "evaluate_with_retry"
        )
        assert call_order.index("evaluate_with_retry") < call_order.index(
            "verify_submission"
        )


# ===========================================================================
# REQ-FN-025: Fallback on evaluation failure
# ===========================================================================


@pytest.mark.unit
class TestRunFinalizationFallback:
    """run_finalization falls back to original solution on eval failure (REQ-FN-025)."""

    async def test_fallback_on_eval_failure(self) -> None:
        """When evaluation fails, the original solution is used as final."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        original_solution = _make_solution(content="original_content")
        task = _make_task()
        config = _make_config()

        no_subsample = _make_solution(
            content="no_subsample", phase=SolutionPhase.REFINED
        )
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)
        leakage_checked = _make_solution(
            content="leakage_checked", phase=SolutionPhase.FINAL
        )
        failed_eval = _make_eval_result(
            score=None, is_error=True, exit_code=1, stderr="Error"
        )

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=no_subsample,
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=leakage_checked,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(leakage_checked, failed_eval),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=False),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": False,
                    "path": "",
                    "size_bytes": 0,
                    "row_count": None,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await run_finalization(
                client=client,
                solution=original_solution,
                task=task,
                config=config,
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        assert result.final_solution.content == "original_content"

    async def test_fallback_on_eval_none_score(self) -> None:
        """When evaluation returns None score, the original solution is used."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        original_solution = _make_solution(content="fallback_content")
        task = _make_task()
        config = _make_config()

        no_subsample = _make_solution(
            content="no_subsample", phase=SolutionPhase.REFINED
        )
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)
        leakage_checked = _make_solution(
            content="leakage_checked", phase=SolutionPhase.FINAL
        )
        none_score_eval = _make_eval_result(
            score=None, is_error=True, exit_code=1, stderr="Traceback..."
        )

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=no_subsample,
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=leakage_checked,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(leakage_checked, none_score_eval),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=False),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": False,
                    "path": "",
                    "size_bytes": 0,
                    "row_count": None,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await run_finalization(
                client=client,
                solution=original_solution,
                task=task,
                config=config,
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        assert result.final_solution.content == "fallback_content"


# ===========================================================================
# REQ-FN-036: Contamination check integration
# ===========================================================================


@pytest.mark.unit
class TestRunFinalizationContaminationCheck:
    """run_finalization invokes check_contamination when references provided (REQ-FN-036)."""

    async def test_no_contamination_check_when_no_references(self) -> None:
        """check_contamination not called when reference_discussions is None."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        eval_result = _make_eval_result(score=0.91)
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(test_script, eval_result),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=True),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/path/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_contam,
        ):
            await run_finalization(
                client=client,
                solution=_make_solution(),
                task=_make_task(),
                config=_make_config(),
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
                reference_discussions=None,
            )

        # check_contamination should not be called, or called with None/empty
        # which returns None immediately. The key point is no agent invocations.
        if mock_contam.called:
            # If called, it should be with None or empty references
            refs_arg = (
                mock_contam.call_args[0][2]
                if len(mock_contam.call_args[0]) > 2
                else mock_contam.call_args[1].get("reference_discussions")
            )
            assert refs_arg is None or refs_arg == []

    async def test_contamination_check_called_with_references(self) -> None:
        """check_contamination called when reference_discussions provided."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        eval_result = _make_eval_result(score=0.91)
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(test_script, eval_result),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=True),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/path/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=DataContaminationResult(verdict="Novel"),
            ) as mock_contam,
        ):
            await run_finalization(
                client=client,
                solution=_make_solution(),
                task=_make_task(),
                config=_make_config(),
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
                reference_discussions=["ref1", "ref2"],
            )

        mock_contam.assert_called_once()


# ===========================================================================
# REQ-FN-034: FinalResult fields populated correctly
# ===========================================================================


@pytest.mark.unit
class TestRunFinalizationFinalResultFields:
    """FinalResult fields are populated correctly (REQ-FN-034)."""

    async def test_task_field(self) -> None:
        """FinalResult.task matches the input task."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        task = _make_task(competition_id="my-comp")
        eval_result = _make_eval_result(score=0.91)
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(test_script, eval_result),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=True),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/abs/path/final/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await run_finalization(
                client=client,
                solution=_make_solution(),
                task=task,
                config=_make_config(),
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        assert result.task == task

    async def test_config_field(self) -> None:
        """FinalResult.config matches the input config."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        config = _make_config(max_debug_attempts=5)
        eval_result = _make_eval_result(score=0.91)
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(test_script, eval_result),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=True),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/abs/path/final/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await run_finalization(
                client=client,
                solution=_make_solution(),
                task=_make_task(),
                config=config,
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        assert result.config == config

    async def test_phase1_field(self) -> None:
        """FinalResult.phase1 matches the input phase1_result."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        phase1 = _make_phase1_result(initial_score=0.77)
        eval_result = _make_eval_result(score=0.91)
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(test_script, eval_result),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=True),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/abs/path/final/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await run_finalization(
                client=client,
                solution=_make_solution(),
                task=_make_task(),
                config=_make_config(),
                phase1_result=phase1,
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        assert result.phase1 == phase1

    async def test_phase2_results_field(self) -> None:
        """FinalResult.phase2_results matches the input list."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        phase2_list = [_make_phase2_result(), _make_phase2_result(best_score=0.95)]
        eval_result = _make_eval_result(score=0.91)
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(test_script, eval_result),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=True),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/abs/path/final/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await run_finalization(
                client=client,
                solution=_make_solution(),
                task=_make_task(),
                config=_make_config(),
                phase1_result=_make_phase1_result(),
                phase2_results=phase2_list,
                phase3_result=None,
            )

        assert result.phase2_results == phase2_list

    async def test_phase3_none(self) -> None:
        """FinalResult.phase3 is None when no Phase 3 result provided."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        eval_result = _make_eval_result(score=0.91)
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(test_script, eval_result),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=True),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/abs/path/final/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await run_finalization(
                client=client,
                solution=_make_solution(),
                task=_make_task(),
                config=_make_config(),
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        assert result.phase3 is None

    async def test_phase3_set_when_provided(self) -> None:
        """FinalResult.phase3 matches input when Phase 3 result provided."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        phase3 = _make_phase3_result()
        eval_result = _make_eval_result(score=0.91)
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(test_script, eval_result),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=True),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/abs/path/final/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await run_finalization(
                client=client,
                solution=_make_solution(),
                task=_make_task(),
                config=_make_config(),
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=phase3,
            )

        assert result.phase3 == phase3

    async def test_final_solution_from_evaluation(self) -> None:
        """FinalResult.final_solution comes from successful evaluation."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        eval_result = _make_eval_result(score=0.91)
        evaluated_solution = _make_solution(
            content="evaluated_final", phase=SolutionPhase.FINAL
        )

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=_make_solution(phase=SolutionPhase.FINAL),
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=evaluated_solution,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(evaluated_solution, eval_result),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=True),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/abs/path/final/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await run_finalization(
                client=client,
                solution=_make_solution(),
                task=_make_task(),
                config=_make_config(),
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        assert result.final_solution.content == "evaluated_final"

    async def test_total_cost_usd_is_none(self) -> None:
        """FinalResult.total_cost_usd is None (not tracked at this level)."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        eval_result = _make_eval_result(score=0.91)
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(test_script, eval_result),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=True),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/abs/path/final/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await run_finalization(
                client=client,
                solution=_make_solution(),
                task=_make_task(),
                config=_make_config(),
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        assert result.total_cost_usd is None


# ===========================================================================
# REQ-FN-034: Submission path and duration
# ===========================================================================


@pytest.mark.unit
class TestRunFinalizationSubmissionAndDuration:
    """Submission path and duration fields (REQ-FN-034)."""

    async def test_submission_path_on_success(self) -> None:
        """Submission path is set from get_submission_info on success."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        eval_result = _make_eval_result(score=0.91)
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(test_script, eval_result),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=True),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/abs/path/final/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await run_finalization(
                client=client,
                solution=_make_solution(),
                task=_make_task(),
                config=_make_config(),
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        # Submission path should be non-empty when verify_submission is True
        assert result.submission_path != ""

    async def test_submission_path_empty_on_failure(self) -> None:
        """Submission path is empty string when verify_submission returns False."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        eval_result = _make_eval_result(
            score=None, is_error=True, exit_code=1, stderr="Error"
        )
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(test_script, eval_result),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=False),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": False,
                    "path": "",
                    "size_bytes": 0,
                    "row_count": None,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await run_finalization(
                client=client,
                solution=_make_solution(),
                task=_make_task(),
                config=_make_config(),
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        assert result.submission_path == ""

    async def test_duration_positive(self) -> None:
        """FinalResult.total_duration_seconds is positive."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        eval_result = _make_eval_result(score=0.91)
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(test_script, eval_result),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=True),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/abs/path/final/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await run_finalization(
                client=client,
                solution=_make_solution(),
                task=_make_task(),
                config=_make_config(),
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        assert result.total_duration_seconds >= 0.0


# ===========================================================================
# REQ-SF-022: Leakage check in finalization pipeline
# ===========================================================================


@pytest.mark.unit
class TestRunFinalizationLeakageCheck:
    """run_finalization applies leakage check after test submission (REQ-SF-022)."""

    async def test_leakage_check_called_with_test_script(self) -> None:
        """check_and_fix_leakage is called with the test submission script."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        test_script = _make_solution(
            content="test_submission_code", phase=SolutionPhase.FINAL
        )
        eval_result = _make_eval_result(score=0.91)

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=test_script,
            ) as mock_leakage,
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(test_script, eval_result),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=True),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/path/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            await run_finalization(
                client=client,
                solution=_make_solution(),
                task=_make_task(),
                config=_make_config(),
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        mock_leakage.assert_called_once()
        # First arg should be the test script
        leakage_args = mock_leakage.call_args[0]
        assert leakage_args[0].content == "test_submission_code"

    async def test_evaluate_with_retry_receives_leakage_checked_script(self) -> None:
        """evaluate_with_retry receives the leakage-checked script."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)
        leakage_checked = _make_solution(
            content="leakage_fixed_script", phase=SolutionPhase.FINAL
        )
        eval_result = _make_eval_result(score=0.91)

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=leakage_checked,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(leakage_checked, eval_result),
            ) as mock_eval,
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=True),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/path/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            await run_finalization(
                client=client,
                solution=_make_solution(),
                task=_make_task(),
                config=_make_config(),
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        mock_eval.assert_called_once()
        # First arg to evaluate_with_retry should be the leakage-checked script
        eval_args = mock_eval.call_args[0]
        assert eval_args[0].content == "leakage_fixed_script"


# ===========================================================================
# run_finalization -- make_debug_callback integration
# ===========================================================================


@pytest.mark.unit
class TestRunFinalizationDebugCallback:
    """run_finalization creates debug callback for evaluate_with_retry."""

    async def test_make_debug_callback_called(self) -> None:
        """make_debug_callback is invoked to create the debug callback."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        eval_result = _make_eval_result(score=0.91)
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)
        mock_cb = MagicMock()

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(test_script, eval_result),
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=mock_cb,
            ) as mock_make_cb,
            patch(f"{_MODULE}.verify_submission", return_value=True),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/path/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            await run_finalization(
                client=client,
                solution=_make_solution(),
                task=_make_task(),
                config=_make_config(),
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        mock_make_cb.assert_called_once()

    async def test_debug_callback_passed_to_evaluate_with_retry(self) -> None:
        """The debug callback from make_debug_callback is passed to evaluate_with_retry."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        eval_result = _make_eval_result(score=0.91)
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)
        mock_cb = MagicMock()

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(test_script, eval_result),
            ) as mock_eval,
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=mock_cb,
            ),
            patch(f"{_MODULE}.verify_submission", return_value=True),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/path/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            await run_finalization(
                client=client,
                solution=_make_solution(),
                task=_make_task(),
                config=_make_config(),
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        eval_args = mock_eval.call_args
        # debug_callback should be the 4th positional arg or keyword arg
        if len(eval_args[0]) >= 4:
            assert eval_args[0][3] is mock_cb
        else:
            assert eval_args[1].get("debug_callback") is mock_cb


# ===========================================================================
# run_finalization -- Data flow: subsampling output -> test submission input
# ===========================================================================


@pytest.mark.unit
class TestRunFinalizationDataFlow:
    """Verify data flows correctly between pipeline steps."""

    async def test_subsampling_output_fed_to_test_submission(self) -> None:
        """Output of remove_subsampling is passed to generate_test_submission."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        no_subsample = _make_solution(
            content="subsampling_removed", phase=SolutionPhase.REFINED
        )
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)
        eval_result = _make_eval_result(score=0.91)

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=no_subsample,
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ) as mock_gen_test,
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(test_script, eval_result),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=True),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/path/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            await run_finalization(
                client=client,
                solution=_make_solution(),
                task=_make_task(),
                config=_make_config(),
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        mock_gen_test.assert_called_once()
        gen_test_args = mock_gen_test.call_args[0]
        # The solution passed to generate_test_submission should be the
        # no_subsample result (second or third positional arg depending on signature)
        found_subsample_solution = False
        for arg in gen_test_args:
            if isinstance(arg, SolutionScript) and arg.content == "subsampling_removed":
                found_subsample_solution = True
                break
        # Also check kwargs
        for arg in mock_gen_test.call_args[1].values():
            if isinstance(arg, SolutionScript) and arg.content == "subsampling_removed":
                found_subsample_solution = True
                break
        assert found_subsample_solution


# ===========================================================================
# run_finalization -- Parametrized tests
# ===========================================================================


@pytest.mark.unit
class TestRunFinalizationParametrized:
    """Parametrized tests for run_finalization."""

    @pytest.mark.parametrize(
        "eval_is_error,eval_score,expect_fallback",
        [
            (False, 0.91, False),
            (True, None, True),
            (True, 0.50, True),
        ],
        ids=[
            "success-no-fallback",
            "error-none-score-fallback",
            "error-with-score-fallback",
        ],
    )
    async def test_fallback_depends_on_eval_outcome(
        self,
        eval_is_error: bool,
        eval_score: float | None,
        expect_fallback: bool,
    ) -> None:
        """Fallback to original solution depends on evaluation outcome."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        original_solution = _make_solution(content="original")
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)
        eval_result = _make_eval_result(
            score=eval_score,
            is_error=eval_is_error,
            exit_code=1 if eval_is_error else 0,
            stderr="Error" if eval_is_error else "",
        )

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(test_script, eval_result),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(
                f"{_MODULE}.verify_submission",
                return_value=not eval_is_error,
            ),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": not eval_is_error,
                    "path": "/path/submission.csv" if not eval_is_error else "",
                    "size_bytes": 100 if not eval_is_error else 0,
                    "row_count": 10 if not eval_is_error else None,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await run_finalization(
                client=client,
                solution=original_solution,
                task=_make_task(),
                config=_make_config(),
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        if expect_fallback:
            assert result.final_solution.content == "original"
        else:
            assert result.final_solution.content == "test_script"


# ===========================================================================
# run_finalization -- Property-based tests
# ===========================================================================


@pytest.mark.unit
class TestRunFinalizationPropertyBased:
    """Property-based tests for run_finalization invariants."""

    @given(
        score=st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=10)
    async def test_duration_always_non_negative(self, score: float) -> None:
        """Duration is always non-negative regardless of evaluation score."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        eval_result = _make_eval_result(score=score)
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(test_script, eval_result),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=True),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/path/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await run_finalization(
                client=client,
                solution=_make_solution(),
                task=_make_task(),
                config=_make_config(),
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        assert result.total_duration_seconds >= 0.0

    @given(
        score=st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=10)
    async def test_total_cost_always_none(self, score: float) -> None:
        """total_cost_usd is always None (not tracked at this level)."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        eval_result = _make_eval_result(score=score)
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(test_script, eval_result),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=True),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/path/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await run_finalization(
                client=client,
                solution=_make_solution(),
                task=_make_task(),
                config=_make_config(),
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        assert result.total_cost_usd is None

    @given(
        phase3_present=st.booleans(),
    )
    @settings(max_examples=6)
    async def test_result_always_final_result(self, phase3_present: bool) -> None:
        """Return type is always FinalResult regardless of Phase 3 presence."""
        from mle_star.finalization import run_finalization

        client = AsyncMock()
        eval_result = _make_eval_result(score=0.91)
        test_script = _make_solution(content="test_script", phase=SolutionPhase.FINAL)

        with (
            patch(
                f"{_MODULE}.remove_subsampling",
                new_callable=AsyncMock,
                return_value=_make_solution(),
            ),
            patch(
                f"{_MODULE}.generate_test_submission",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=test_script,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(test_script, eval_result),
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=MagicMock()),
            patch(f"{_MODULE}.verify_submission", return_value=True),
            patch(
                f"{_MODULE}.get_submission_info",
                return_value={
                    "exists": True,
                    "path": "/path/submission.csv",
                    "size_bytes": 100,
                    "row_count": 10,
                },
            ),
            patch(
                f"{_MODULE}.check_contamination",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await run_finalization(
                client=client,
                solution=_make_solution(),
                task=_make_task(),
                config=_make_config(),
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=_make_phase3_result() if phase3_present else None,
            )

        assert isinstance(result, FinalResult)
        if phase3_present:
            assert result.phase3 is not None
        else:
            assert result.phase3 is None
