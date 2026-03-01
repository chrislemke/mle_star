"""Tests for the baseline model generation phase.

Validates ``generate_baseline`` which creates a simple default model solution
before web retrieval to establish a benchmark score.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

from mle_star.models import (
    AgentType,
    DataModality,
    EvaluationResult,
    MetricDirection,
    Phase1Result,
    RetrievedModel,
    RetrieverOutput,
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


_BASELINE_CODE = (
    "import catboost\n"
    "model = catboost.CatBoostClassifier()\n"
    "model.fit(X_train, y_train)\n"
    'print(f"Final Validation Performance: {0.85}")\n'
)


# ===========================================================================
# generate_baseline tests
# ===========================================================================


@pytest.mark.unit
class TestGenerateBaseline:
    """Tests for the generate_baseline function."""

    async def test_generate_baseline_returns_solution(self) -> None:
        """generate_baseline returns a SolutionScript on success."""
        from mle_star.phase1 import generate_baseline

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value=f"```python\n{_BASELINE_CODE}\n```"
        )

        with patch(f"{_PHASE1}.get_registry") as mock_reg_cls:
            mock_reg = mock_reg_cls.return_value
            mock_tmpl = AsyncMock()
            mock_tmpl.render = lambda **kwargs: "baseline prompt"
            mock_reg.get.return_value = mock_tmpl

            result = await generate_baseline(_make_task(), _make_config(), client)

        assert result is not None
        assert isinstance(result, SolutionScript)
        assert result.source_model == "baseline"
        assert result.phase == SolutionPhase.INIT
        assert "catboost" in result.content

    async def test_generate_baseline_none_on_empty(self) -> None:
        """generate_baseline returns None when agent returns empty response."""
        from mle_star.phase1 import generate_baseline

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with patch(f"{_PHASE1}.get_registry") as mock_reg_cls:
            mock_reg = mock_reg_cls.return_value
            mock_tmpl = AsyncMock()
            mock_tmpl.render = lambda **kwargs: "baseline prompt"
            mock_reg.get.return_value = mock_tmpl

            result = await generate_baseline(_make_task(), _make_config(), client)

        assert result is None

    async def test_generate_baseline_uses_baseline_agent_type(self) -> None:
        """generate_baseline uses AgentType.BASELINE for the prompt registry."""
        from mle_star.phase1 import generate_baseline

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value=f"```python\n{_BASELINE_CODE}\n```"
        )

        with patch(f"{_PHASE1}.get_registry") as mock_reg_cls:
            mock_reg = mock_reg_cls.return_value
            mock_tmpl = AsyncMock()
            mock_tmpl.render = lambda **kwargs: "prompt"
            mock_reg.get.return_value = mock_tmpl

            await generate_baseline(_make_task(), _make_config(), client)

        mock_reg.get.assert_called_once_with(AgentType.BASELINE)


# ===========================================================================
# Phase1Result baseline fields
# ===========================================================================


@pytest.mark.unit
class TestPhase1ResultBaselineFields:
    """Tests for baseline_score and baseline_solution in Phase1Result."""

    def test_baseline_score_in_phase1_result(self) -> None:
        """Phase1Result accepts baseline_score and baseline_solution fields."""
        solution = SolutionScript(
            content="code", phase=SolutionPhase.INIT, source_model="baseline"
        )
        result = Phase1Result(
            retrieved_models=[
                RetrievedModel(model_name="test", example_code="code")
            ],
            candidate_solutions=[solution],
            candidate_scores=[0.85],
            initial_solution=solution,
            initial_score=0.85,
            baseline_score=0.80,
            baseline_solution=solution,
        )
        assert result.baseline_score == pytest.approx(0.80)
        assert result.baseline_solution is not None

    def test_baseline_fields_default_to_none(self) -> None:
        """Phase1Result baseline fields default to None."""
        solution = SolutionScript(content="code", phase=SolutionPhase.INIT)
        result = Phase1Result(
            retrieved_models=[
                RetrievedModel(model_name="test", example_code="code")
            ],
            candidate_solutions=[solution],
            candidate_scores=[0.85],
            initial_solution=solution,
            initial_score=0.85,
        )
        assert result.baseline_score is None
        assert result.baseline_solution is None


# ===========================================================================
# run_phase1 baseline integration
# ===========================================================================


@pytest.mark.unit
class TestRunPhase1BaselineIntegration:
    """Integration tests for baseline in run_phase1."""

    async def test_baseline_injected_into_candidates(self) -> None:
        """Successful baseline is injected into candidate ranking."""
        from mle_star.phase1 import run_phase1

        task = _make_task()
        config = _make_config()
        client = AsyncMock()

        # Mock generate_baseline
        baseline_sol = SolutionScript(
            content=_BASELINE_CODE, phase=SolutionPhase.INIT, source_model="baseline"
        )
        baseline_eval = EvaluationResult(
            score=0.80,
            stdout="",
            stderr="",
            exit_code=0,
            duration_seconds=1.0,
            is_error=False,
        )

        # Mock retrieve_models
        models = [RetrievedModel(model_name="XGBoost", example_code="import xgb")]

        # Mock candidate generation
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
            patch(f"{_PHASE1}.generate_baseline", return_value=baseline_sol),
            patch(
                f"{_PHASE1}.check_and_fix_leakage",
                side_effect=lambda s, t, c: s,
            ),
            patch(
                f"{_PHASE1}.evaluate_with_retry",
                side_effect=[
                    (baseline_sol, baseline_eval),
                    (candidate_sol, candidate_eval),
                ],
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

        assert result.baseline_score == pytest.approx(0.80)
        assert result.baseline_solution is not None
        # rank_solutions should have received the baseline in successful_solutions
        call_args = mock_rank.call_args
        solutions_arg = call_args[0][0]
        assert any(s.source_model == "baseline" for s in solutions_arg)

    async def test_run_phase1_continues_when_baseline_fails(self) -> None:
        """Pipeline continues normally when baseline generation raises an exception."""
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
            patch(
                f"{_PHASE1}.generate_baseline",
                side_effect=RuntimeError("Baseline failed"),
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

        assert result.baseline_score is None
        assert result.baseline_solution is None
        assert result.initial_score == pytest.approx(0.85)
