"""Tests for the internet research phase.

Validates ``conduct_research``, ``_parse_research_output``, and
``_format_research_context`` which implement the deep web research
step before model retrieval.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

from mle_star.models import (
    AgentType,
    DataModality,
    EvaluationResult,
    MetricDirection,
    ResearchFindings,
    RetrievedModel,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
    TaskType,
)
import pytest

_PHASE1 = "mle_star.phase1"


def _make_task(**overrides: Any) -> TaskDescription:
    defaults: dict[str, Any] = {
        "competition_id": "test-comp",
        "task_type": TaskType.CLASSIFICATION,
        "data_modality": DataModality.TABULAR,
        "evaluation_metric": "accuracy",
        "metric_direction": MetricDirection.MAXIMIZE,
        "description": "Predict target.",
        "data_dir": "/tmp/test_data",
        "output_dir": "./final",
    }
    defaults.update(overrides)
    return TaskDescription(**defaults)


def _make_config(**overrides: Any) -> Any:
    from mle_star.models import PipelineConfig

    defaults: dict[str, Any] = {
        "num_retrieved_models": 2,
        "max_debug_attempts": 1,
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)


_SAMPLE_FINDINGS = ResearchFindings(
    model_recommendations=["XGBoost", "LightGBM", "CatBoost"],
    feature_engineering_ideas=["Target encoding", "Interaction features"],
    preprocessing_ideas=["StandardScaler", "Missing value imputation"],
    other_insights=["Use cross-validation", "Feature selection helps"],
    raw_summary="Summary of research findings for tabular classification.",
)

_SAMPLE_FINDINGS_JSON = _SAMPLE_FINDINGS.model_dump_json()


# ===========================================================================
# conduct_research tests
# ===========================================================================


@pytest.mark.unit
class TestConductResearch:
    """Tests for the conduct_research function."""

    async def test_conduct_research_returns_findings(self) -> None:
        """conduct_research returns ResearchFindings on success."""
        from mle_star.phase1 import conduct_research

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_SAMPLE_FINDINGS_JSON)

        with patch(f"{_PHASE1}.get_registry") as mock_reg_cls:
            mock_reg = mock_reg_cls.return_value
            mock_tmpl = AsyncMock()
            mock_tmpl.render = lambda **kwargs: "research prompt"
            mock_reg.get.return_value = mock_tmpl

            result = await conduct_research(
                _make_task(), _make_config(), client, baseline_score=0.80
            )

        assert result is not None
        assert isinstance(result, ResearchFindings)
        assert len(result.model_recommendations) == 3
        assert "XGBoost" in result.model_recommendations

    async def test_conduct_research_uses_researcher_agent_type(self) -> None:
        """conduct_research uses AgentType.RESEARCHER for the prompt registry."""
        from mle_star.phase1 import conduct_research

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_SAMPLE_FINDINGS_JSON)

        with patch(f"{_PHASE1}.get_registry") as mock_reg_cls:
            mock_reg = mock_reg_cls.return_value
            mock_tmpl = AsyncMock()
            mock_tmpl.render = lambda **kwargs: "prompt"
            mock_reg.get.return_value = mock_tmpl

            await conduct_research(_make_task(), _make_config(), client)

        mock_reg.get.assert_called_once_with(AgentType.RESEARCHER)

    async def test_conduct_research_passes_baseline_score(self) -> None:
        """conduct_research passes baseline_score to the template."""
        from mle_star.phase1 import conduct_research

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_SAMPLE_FINDINGS_JSON)
        render_kwargs: dict[str, Any] = {}

        with patch(f"{_PHASE1}.get_registry") as mock_reg_cls:
            mock_reg = mock_reg_cls.return_value
            mock_tmpl = AsyncMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs.update(kwargs)
                return "prompt"

            mock_tmpl.render = capture_render
            mock_reg.get.return_value = mock_tmpl

            await conduct_research(
                _make_task(), _make_config(), client, baseline_score=0.82
            )

        assert render_kwargs["baseline_score"] == 0.82

    async def test_conduct_research_none_baseline_score(self) -> None:
        """conduct_research uses 'N/A' when baseline_score is None."""
        from mle_star.phase1 import conduct_research

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_SAMPLE_FINDINGS_JSON)
        render_kwargs: dict[str, Any] = {}

        with patch(f"{_PHASE1}.get_registry") as mock_reg_cls:
            mock_reg = mock_reg_cls.return_value
            mock_tmpl = AsyncMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs.update(kwargs)
                return "prompt"

            mock_tmpl.render = capture_render
            mock_reg.get.return_value = mock_tmpl

            await conduct_research(_make_task(), _make_config(), client)

        assert render_kwargs["baseline_score"] == "N/A"


# ===========================================================================
# _parse_research_output tests
# ===========================================================================


@pytest.mark.unit
class TestParseResearchOutput:
    """Tests for the _parse_research_output function."""

    def test_parse_research_output_direct_json(self) -> None:
        """Direct JSON parsing works for well-formed response."""
        from mle_star.phase1 import _parse_research_output

        result = _parse_research_output(_SAMPLE_FINDINGS_JSON)
        assert result is not None
        assert result.model_recommendations == ["XGBoost", "LightGBM", "CatBoost"]

    def test_parse_research_output_embedded_json(self) -> None:
        """Embedded JSON in text is extracted and parsed."""
        from mle_star.phase1 import _parse_research_output

        response = f"Here are my findings:\n{_SAMPLE_FINDINGS_JSON}\nEnd of findings."
        result = _parse_research_output(response)
        assert result is not None
        assert len(result.model_recommendations) == 3

    def test_parse_research_output_json_code_block(self) -> None:
        """JSON in a code block is extracted and parsed."""
        from mle_star.phase1 import _parse_research_output

        response = f"```json\n{_SAMPLE_FINDINGS_JSON}\n```"
        result = _parse_research_output(response)
        assert result is not None
        assert len(result.model_recommendations) == 3

    def test_parse_research_output_fallback_raw(self) -> None:
        """Unparseable response falls back to raw_summary."""
        from mle_star.phase1 import _parse_research_output

        response = "Just some free text about ML models and techniques."
        result = _parse_research_output(response)
        assert result is not None
        assert result.raw_summary == response
        assert result.model_recommendations == []

    def test_parse_research_output_empty_returns_none(self) -> None:
        """Empty response returns None."""
        from mle_star.phase1 import _parse_research_output

        assert _parse_research_output("") is None
        assert _parse_research_output("   ") is None


# ===========================================================================
# _format_research_context tests
# ===========================================================================


@pytest.mark.unit
class TestFormatResearchContext:
    """Tests for the _format_research_context function."""

    def test_format_research_context_with_findings(self) -> None:
        """Formats ResearchFindings into readable text."""
        from mle_star.phase1 import _format_research_context

        result = _format_research_context(_SAMPLE_FINDINGS)
        assert "# Research Findings" in result
        assert "XGBoost" in result
        assert "Target encoding" in result
        assert "StandardScaler" in result
        assert "cross-validation" in result
        assert "Summary" in result

    def test_format_research_context_none(self) -> None:
        """Returns empty string for None findings."""
        from mle_star.phase1 import _format_research_context

        assert _format_research_context(None) == ""

    def test_format_research_context_empty_lists(self) -> None:
        """Handles findings with empty lists gracefully."""
        from mle_star.phase1 import _format_research_context

        findings = ResearchFindings(
            model_recommendations=[],
            feature_engineering_ideas=[],
            preprocessing_ideas=[],
            other_insights=[],
            raw_summary="Just a summary.",
        )
        result = _format_research_context(findings)
        assert "# Research Findings" in result
        assert "Just a summary." in result
        assert "## Recommended Models" not in result


# ===========================================================================
# run_phase1 research integration
# ===========================================================================


@pytest.mark.unit
class TestRunPhase1ResearchIntegration:
    """Integration tests for research in run_phase1."""

    async def test_run_phase1_continues_when_research_fails(self) -> None:
        """Pipeline continues normally when research raises an exception."""
        from mle_star.phase1 import run_phase1

        task = _make_task()
        config = _make_config()
        client = AsyncMock()

        models = [RetrievedModel(model_name="XGBoost", example_code="import xgb")]
        candidate_sol = SolutionScript(
            content="import xgb\ncode",
            phase=SolutionPhase.INIT,
            source_model="XGBoost",
        )
        candidate_eval = EvaluationResult(
            score=0.85,
            stdout="",
            stderr="",
            exit_code=0,
            duration_seconds=2.0,
            is_error=False,
        )

        with (
            patch(f"{_PHASE1}.generate_baseline", return_value=None),
            patch(
                f"{_PHASE1}.conduct_research",
                side_effect=RuntimeError("Research failed"),
            ),
            patch(
                f"{_PHASE1}.check_and_fix_leakage",
                side_effect=lambda s, t, c: s,
            ),
            patch(
                f"{_PHASE1}.evaluate_with_retry",
                return_value=(candidate_sol, candidate_eval),
            ),
            patch(f"{_PHASE1}.retrieve_models", return_value=models),
            patch(
                f"{_PHASE1}.generate_candidate", return_value=candidate_sol
            ),
            patch(f"{_PHASE1}.rank_solutions") as mock_rank,
            patch(f"{_PHASE1}._run_merge_loop") as mock_merge,
            patch(f"{_PHASE1}._apply_post_merge_safety") as mock_safety,
            patch(f"{_PHASE1}.make_debug_callback", return_value=AsyncMock()),
        ):
            mock_rank.return_value = [(candidate_sol, candidate_eval)]
            mock_merge.return_value = (candidate_sol, 0.85, 0)
            mock_safety.return_value = (candidate_sol, 0.85)

            result = await run_phase1(task, config, client)

        assert result.research_findings is None
        assert result.initial_score == pytest.approx(0.85)

    async def test_research_context_passed_to_retrieve_models(self) -> None:
        """Research context is passed through to retrieve_models."""
        from mle_star.phase1 import run_phase1

        task = _make_task()
        config = _make_config()
        client = AsyncMock()

        models = [RetrievedModel(model_name="XGBoost", example_code="import xgb")]
        candidate_sol = SolutionScript(
            content="import xgb\ncode",
            phase=SolutionPhase.INIT,
            source_model="XGBoost",
        )
        candidate_eval = EvaluationResult(
            score=0.85,
            stdout="",
            stderr="",
            exit_code=0,
            duration_seconds=2.0,
            is_error=False,
        )

        with (
            patch(f"{_PHASE1}.generate_baseline", return_value=None),
            patch(
                f"{_PHASE1}.conduct_research", return_value=_SAMPLE_FINDINGS
            ),
            patch(
                f"{_PHASE1}.check_and_fix_leakage",
                side_effect=lambda s, t, c: s,
            ),
            patch(
                f"{_PHASE1}.evaluate_with_retry",
                return_value=(candidate_sol, candidate_eval),
            ),
            patch(f"{_PHASE1}.retrieve_models", return_value=models) as mock_retrieve,
            patch(
                f"{_PHASE1}.generate_candidate", return_value=candidate_sol
            ),
            patch(f"{_PHASE1}.rank_solutions") as mock_rank,
            patch(f"{_PHASE1}._run_merge_loop") as mock_merge,
            patch(f"{_PHASE1}._apply_post_merge_safety") as mock_safety,
            patch(f"{_PHASE1}.make_debug_callback", return_value=AsyncMock()),
        ):
            mock_rank.return_value = [(candidate_sol, candidate_eval)]
            mock_merge.return_value = (candidate_sol, 0.85, 0)
            mock_safety.return_value = (candidate_sol, 0.85)

            result = await run_phase1(task, config, client)

        # Verify research_context was passed to retrieve_models
        call_kwargs = mock_retrieve.call_args[1]
        assert "research_context" in call_kwargs
        assert "XGBoost" in call_kwargs["research_context"]
        assert result.research_findings is not None
