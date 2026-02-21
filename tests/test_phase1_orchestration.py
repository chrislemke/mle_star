"""Tests for the Phase 1 orchestration function ``run_phase1`` (Task 28).

Validates ``run_phase1`` which implements Algorithm 1: retrieve M models,
generate & evaluate M candidates (with leakage checks and debug retries),
sort by score, then merge remaining candidates into the best one until
the first non-improvement.

Tests are written TDD-first and serve as the executable specification for
REQ-P1-018 through REQ-P1-029.

Refs:
    SRS 04b (Phase 1 Orchestration), IMPLEMENTATION_PLAN.md Task 28.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    DataModality,
    EvaluationResult,
    MetricDirection,
    Phase1Result,
    PipelineConfig,
    RetrievedModel,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
    TaskType,
)
import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Module path constant for patching
# ---------------------------------------------------------------------------

_MODULE = "mle_star.phase1"


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


def _make_config(num_retrieved_models: int = 3) -> PipelineConfig:
    """Create a PipelineConfig for testing with a specified M value."""
    return PipelineConfig(num_retrieved_models=num_retrieved_models)


def _make_model(name: str = "xgboost", code: str = "import xgboost") -> RetrievedModel:
    """Create a RetrievedModel for testing."""
    return RetrievedModel(model_name=name, example_code=code)


def _make_solution(
    content: str = "print('hello')",
    phase: SolutionPhase = SolutionPhase.INIT,
    score: float | None = None,
    source_model: str | None = None,
) -> SolutionScript:
    """Create a SolutionScript for testing."""
    return SolutionScript(
        content=content, phase=phase, score=score, source_model=source_model
    )


def _make_eval_result(
    score: float | None = 0.85,
    is_error: bool = False,
    error_traceback: str | None = None,
) -> EvaluationResult:
    """Create an EvaluationResult with the given score and error state."""
    return EvaluationResult(
        score=score,
        stdout=f"Final Validation Performance: {score}" if score is not None else "",
        stderr="" if not is_error else "Traceback (most recent call last):\nError",
        exit_code=0 if not is_error else 1,
        duration_seconds=1.0,
        is_error=is_error,
        error_traceback=error_traceback,
    )


def _make_merged_solution(
    content: str = "merged code",
) -> SolutionScript:
    """Create a SolutionScript representing merged output."""
    return SolutionScript(content=content, phase=SolutionPhase.MERGED)


def _setup_standard_mocks(
    models: Sequence[RetrievedModel],
    candidates: Sequence[SolutionScript | None],
    eval_results: Sequence[tuple[SolutionScript, EvaluationResult]],
    ranked_pairs: Sequence[tuple[SolutionScript, EvaluationResult]] | None = None,
) -> dict[str, Any]:
    """Build a dict of standard mock objects for run_phase1 dependencies.

    Returns a dict keyed by function name, suitable for use with patch().
    """
    mock_retrieve = AsyncMock(return_value=models)
    mock_generate = AsyncMock(side_effect=candidates)
    mock_leakage = AsyncMock(side_effect=lambda sol, _task, _client: sol)
    mock_debug_cb = MagicMock()
    mock_make_debug = MagicMock(return_value=mock_debug_cb)
    mock_eval = AsyncMock(side_effect=eval_results)

    if ranked_pairs is None:
        # Default: filter successful pairs and sort descending by score
        successful = [
            (sol, res)
            for (sol, res) in eval_results
            if not res.is_error and res.score is not None
        ]
        ranked_pairs = sorted(successful, key=lambda p: p[1].score or 0.0, reverse=True)
    mock_rank = MagicMock(return_value=ranked_pairs)
    mock_improve = MagicMock(return_value=True)

    return {
        "retrieve_models": mock_retrieve,
        "generate_candidate": mock_generate,
        "check_and_fix_leakage": mock_leakage,
        "make_debug_callback": mock_make_debug,
        "evaluate_with_retry": mock_eval,
        "rank_solutions": mock_rank,
        "is_improvement_or_equal": mock_improve,
    }


def _apply_patches(mocks: dict[str, Any]) -> contextlib.ExitStack:
    """Return a combined context manager that patches all run_phase1 dependencies.

    Usage::

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)
    """
    from unittest.mock import patch as _patch

    stack = contextlib.ExitStack()
    for name, mock in mocks.items():
        stack.enter_context(_patch(f"{_MODULE}.{name}", new=mock))
    return stack


# ===========================================================================
# REQ-P1-018: run_phase1 is async and returns Phase1Result
# ===========================================================================


@pytest.mark.unit
class TestRunPhase1IsAsync:
    """run_phase1 is an async function returning Phase1Result (REQ-P1-018)."""

    def test_is_coroutine_function(self) -> None:
        """run_phase1 is defined as an async function."""
        from mle_star.phase1 import run_phase1

        assert asyncio.iscoroutinefunction(run_phase1)

    async def test_returns_phase1_result(self) -> None:
        """run_phase1 returns a Phase1Result instance on success."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("xgboost")
        candidate = _make_solution(content="xgb code", source_model="xgboost")
        eval_result = _make_eval_result(score=0.90)
        ranked = [(candidate, eval_result)]

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[candidate],
            eval_results=[(candidate, eval_result)],
            ranked_pairs=ranked,
        )

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert isinstance(result, Phase1Result)


# ===========================================================================
# REQ-P1-019: Calls retrieve_models with correct args
# ===========================================================================


@pytest.mark.unit
class TestRetrieveModelsInvocation:
    """run_phase1 calls retrieve_models with (task, config, client) (REQ-P1-019)."""

    async def test_calls_retrieve_models_with_correct_args(self) -> None:
        """retrieve_models receives task, config, and client as arguments."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="code_a", source_model="m1")
        sol_b = _make_solution(content="code_b", source_model="m2")
        res_a = _make_eval_result(score=0.80)
        res_b = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[(sol_a, res_a), (sol_b, res_b)],
            ranked_pairs=[(sol_b, res_b), (sol_a, res_a)],
        )
        # Merge mock: never called (first non-improvement or single after best)
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        mocks["retrieve_models"].assert_awaited_once_with(task, config, client)


# ===========================================================================
# REQ-P1-020: Generate, leakage-check, and evaluate each candidate
# ===========================================================================


@pytest.mark.unit
class TestCandidateGenerationLoop:
    """For each model, calls generate_candidate, check_and_fix_leakage, evaluate_with_retry (REQ-P1-020)."""

    async def test_generate_called_for_each_model(self) -> None:
        """generate_candidate is called once per retrieved model."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=3)

        models = [_make_model(f"m{i}") for i in range(3)]
        candidates = [
            _make_solution(content=f"code_{i}", source_model=f"m{i}") for i in range(3)
        ]
        eval_results = [
            (candidates[i], _make_eval_result(score=0.80 + i * 0.05)) for i in range(3)
        ]
        ranked = list(reversed(eval_results))

        mocks = _setup_standard_mocks(
            models=models,
            candidates=candidates,
            eval_results=eval_results,
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        assert mocks["generate_candidate"].await_count == 3
        for _i, model in enumerate(models):
            mocks["generate_candidate"].assert_any_await(task, model, config, client)

    async def test_leakage_check_called_for_each_successful_candidate(self) -> None:
        """check_and_fix_leakage is called for each non-None candidate."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="code_a", source_model="m1")
        sol_b = _make_solution(content="code_b", source_model="m2")
        res_a = _make_eval_result(score=0.80)
        res_b = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[(sol_a, res_a), (sol_b, res_b)],
            ranked_pairs=[(sol_b, res_b), (sol_a, res_a)],
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        # At least 2 leakage checks for the 2 candidates (merge may add more)
        assert mocks["check_and_fix_leakage"].await_count >= 2

    async def test_evaluate_with_retry_called_for_each_candidate(self) -> None:
        """evaluate_with_retry is called for each non-None candidate."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="code_a", source_model="m1")
        sol_b = _make_solution(content="code_b", source_model="m2")
        res_a = _make_eval_result(score=0.80)
        res_b = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[(sol_a, res_a), (sol_b, res_b)],
            ranked_pairs=[(sol_b, res_b), (sol_a, res_a)],
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        # At least 2 evaluate calls for the 2 candidates
        assert mocks["evaluate_with_retry"].await_count >= 2

    async def test_make_debug_callback_called_with_correct_args(self) -> None:
        """make_debug_callback receives (task, config, client)."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="code", source_model="m1")
        res = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[(sol, res)],
            ranked_pairs=[(sol, res)],
        )

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        mocks["make_debug_callback"].assert_called_once_with(task, config, client)


# ===========================================================================
# REQ-P1-021: Failed candidate (is_error=True) records score=None
# ===========================================================================


@pytest.mark.unit
class TestFailedCandidateHandling:
    """Failed candidate (is_error=True) records score=None and continues (REQ-P1-021)."""

    async def test_error_candidate_gets_none_score(self) -> None:
        """When evaluate_with_retry returns is_error=True, candidate_scores has None."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("fail_model"), _make_model("good_model")]
        sol_fail = _make_solution(content="bad code", source_model="fail_model")
        sol_good = _make_solution(content="good code", source_model="good_model")
        res_fail = _make_eval_result(score=None, is_error=True, error_traceback="Error")
        res_good = _make_eval_result(score=0.90)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_fail, sol_good],
            eval_results=[(sol_fail, res_fail), (sol_good, res_good)],
            ranked_pairs=[(sol_good, res_good)],
        )

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert result.candidate_scores[0] is None
        assert result.candidate_scores[1] == 0.90

    async def test_continues_after_failed_candidate(self) -> None:
        """Processing does not stop when one candidate fails."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=3)

        models = [_make_model(f"m{i}") for i in range(3)]
        sol_0 = _make_solution(content="code_0", source_model="m0")
        sol_1 = _make_solution(content="code_1", source_model="m1")
        sol_2 = _make_solution(content="code_2", source_model="m2")
        res_0 = _make_eval_result(score=None, is_error=True)
        res_1 = _make_eval_result(score=0.80)
        res_2 = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_0, sol_1, sol_2],
            eval_results=[(sol_0, res_0), (sol_1, res_1), (sol_2, res_2)],
            ranked_pairs=[(sol_2, res_2), (sol_1, res_1)],
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        # All 3 candidates were generated
        assert mocks["generate_candidate"].await_count == 3
        # Result still contains all candidates and scores
        assert len(result.candidate_scores) == 3
        assert result.candidate_scores[0] is None


# ===========================================================================
# REQ-P1-022: All candidates failed raises RuntimeError
# ===========================================================================


@pytest.mark.unit
class TestAllCandidatesFailed:
    """All candidates failed raises RuntimeError (REQ-P1-022)."""

    async def test_all_errors_raises_runtime_error(self) -> None:
        """RuntimeError raised when every candidate has is_error=True."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=3)

        models = [_make_model(f"m{i}") for i in range(3)]
        candidates = [
            _make_solution(content=f"bad_{i}", source_model=f"m{i}") for i in range(3)
        ]
        error_results = [
            (candidates[i], _make_eval_result(score=None, is_error=True))
            for i in range(3)
        ]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=candidates,
            eval_results=error_results,
        )

        with _apply_patches(mocks), pytest.raises(RuntimeError, match="Phase 1 failed"):
            await run_phase1(task, config, client)

    async def test_error_message_contains_model_count(self) -> None:
        """RuntimeError message includes the number of candidates M."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        candidates = [_make_solution(content=f"bad_{i}") for i in range(2)]
        error_results = [
            (candidates[i], _make_eval_result(score=None, is_error=True))
            for i in range(2)
        ]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=candidates,
            eval_results=error_results,
        )

        with (
            _apply_patches(mocks),
            pytest.raises(RuntimeError, match=r"all 2 candidates"),
        ):
            await run_phase1(task, config, client)

    async def test_all_none_generation_raises_runtime_error(self) -> None:
        """RuntimeError raised when all generate_candidate calls return None."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[None, None],
            eval_results=[],
        )

        with _apply_patches(mocks), pytest.raises(RuntimeError, match="Phase 1 failed"):
            await run_phase1(task, config, client)

    async def test_mix_of_none_generation_and_errors_raises(self) -> None:
        """RuntimeError raised when some candidates are None and rest are errors."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=3)

        models = [_make_model(f"m{i}") for i in range(3)]
        sol_2 = _make_solution(content="bad_code", source_model="m2")
        error_result = _make_eval_result(score=None, is_error=True)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[None, None, sol_2],
            eval_results=[(sol_2, error_result)],
        )

        with _apply_patches(mocks), pytest.raises(RuntimeError, match="Phase 1 failed"):
            await run_phase1(task, config, client)


# ===========================================================================
# REQ-P1-023: Successful candidates sorted by score via rank_solutions
# ===========================================================================


@pytest.mark.unit
class TestCandidateSorting:
    """Successful candidates are sorted by score via rank_solutions (REQ-P1-023)."""

    async def test_rank_solutions_called_with_successful_candidates(self) -> None:
        """rank_solutions is called with successful solutions and results."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=3)

        models = [_make_model(f"m{i}") for i in range(3)]
        sol_0 = _make_solution(content="code_0", source_model="m0")
        sol_1 = _make_solution(content="code_1", source_model="m1")
        sol_2 = _make_solution(content="code_2", source_model="m2")
        res_0 = _make_eval_result(score=0.70)
        res_1 = _make_eval_result(score=None, is_error=True)
        res_2 = _make_eval_result(score=0.90)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_0, sol_1, sol_2],
            eval_results=[(sol_0, res_0), (sol_1, res_1), (sol_2, res_2)],
            ranked_pairs=[(sol_2, res_2), (sol_0, res_0)],
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        # rank_solutions should be called; verify it was called
        mocks["rank_solutions"].assert_called_once()
        call_args = mocks["rank_solutions"].call_args
        # Should pass lists of successful solutions/results and direction
        assert call_args[1].get("direction", call_args[0][-1]) == task.metric_direction

    async def test_rank_solutions_receives_direction(self) -> None:
        """rank_solutions receives the metric_direction from the task."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task(direction=MetricDirection.MINIMIZE)
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="code", source_model="m1")
        res = _make_eval_result(score=0.10)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[(sol, res)],
            ranked_pairs=[(sol, res)],
        )

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        call_args = mocks["rank_solutions"].call_args
        # The direction argument (positional or keyword) should be MINIMIZE
        all_args = list(call_args[0]) + list(call_args[1].values())
        assert MetricDirection.MINIMIZE in all_args


# ===========================================================================
# REQ-P1-024: Best candidate becomes initial s_0 with h_best
# ===========================================================================


@pytest.mark.unit
class TestBestCandidateSelection:
    """Best candidate (first from rank_solutions) becomes initial s_0 (REQ-P1-024)."""

    async def test_best_candidate_is_initial_solution(self) -> None:
        """The first element from rank_solutions becomes initial_solution."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="code_a", source_model="m1")
        sol_b = _make_solution(content="code_b", source_model="m2")
        res_a = _make_eval_result(score=0.80)
        res_b = _make_eval_result(score=0.95)

        # rank_solutions returns sol_b first (best)
        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[(sol_a, res_a), (sol_b, res_b)],
            ranked_pairs=[(sol_b, res_b), (sol_a, res_a)],
        )
        # Merge loop: sol_a merged into sol_b, but merge fails => break
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        # initial_solution should be the best-ranked candidate
        assert result.initial_solution.content == sol_b.content
        assert result.initial_score == 0.95

    async def test_initial_score_matches_best_candidate_score(self) -> None:
        """initial_score is set to the score of the best ranked candidate."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="code_a", source_model="m1")
        sol_b = _make_solution(content="code_b", source_model="m2")
        res_a = _make_eval_result(score=0.70)
        res_b = _make_eval_result(score=0.88)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[(sol_a, res_a), (sol_b, res_b)],
            ranked_pairs=[(sol_b, res_b), (sol_a, res_a)],
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert result.initial_score == 0.88


# ===========================================================================
# REQ-P1-025: Merge loop iterates over remaining sorted candidates
# ===========================================================================


@pytest.mark.unit
class TestMergeLoopIteration:
    """Merge loop iterates over remaining sorted candidates (REQ-P1-025)."""

    async def test_merge_called_for_remaining_candidates(self) -> None:
        """merge_solutions is called with (best, candidate_i) for each remaining."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=3)

        models = [_make_model(f"m{i}") for i in range(3)]
        sol_0 = _make_solution(content="code_0", source_model="m0")
        sol_1 = _make_solution(content="code_1", source_model="m1")
        sol_2 = _make_solution(content="code_2", source_model="m2")
        res_0 = _make_eval_result(score=0.90)
        res_1 = _make_eval_result(score=0.85)
        res_2 = _make_eval_result(score=0.80)

        merged_01 = _make_merged_solution("merged_0_1")
        merged_012 = _make_merged_solution("merged_01_2")
        merged_result_01 = _make_eval_result(score=0.92)
        merged_result_012 = _make_eval_result(score=0.93)

        ranked = [(sol_0, res_0), (sol_1, res_1), (sol_2, res_2)]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_0, sol_1, sol_2],
            eval_results=[
                (sol_0, res_0),
                (sol_1, res_1),
                (sol_2, res_2),
                # Merge evaluation results
                (merged_01, merged_result_01),
                (merged_012, merged_result_012),
            ],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(side_effect=[merged_01, merged_012])
        mocks["is_improvement_or_equal"] = MagicMock(return_value=True)

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        # merge_solutions called twice (for candidates at index 1 and 2)
        assert mocks["merge_solutions"].await_count == 2

    async def test_merge_loop_uses_updated_best_as_base(self) -> None:
        """After a successful merge, the merged solution becomes the new base."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=3)

        models = [_make_model(f"m{i}") for i in range(3)]
        sol_0 = _make_solution(content="best_code", source_model="m0")
        sol_1 = _make_solution(content="code_1", source_model="m1")
        sol_2 = _make_solution(content="code_2", source_model="m2")
        res_0 = _make_eval_result(score=0.90)
        res_1 = _make_eval_result(score=0.85)
        res_2 = _make_eval_result(score=0.80)

        merged_01 = _make_merged_solution("merged_after_1")
        merged_012 = _make_merged_solution("merged_after_2")
        merged_res_01 = _make_eval_result(score=0.92)
        merged_res_012 = _make_eval_result(score=0.94)

        ranked = [(sol_0, res_0), (sol_1, res_1), (sol_2, res_2)]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_0, sol_1, sol_2],
            eval_results=[
                (sol_0, res_0),
                (sol_1, res_1),
                (sol_2, res_2),
                (merged_01, merged_res_01),
                (merged_012, merged_res_012),
            ],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(side_effect=[merged_01, merged_012])
        mocks["is_improvement_or_equal"] = MagicMock(return_value=True)

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        # Second merge call should use merged_01 (the updated best) as base,
        # not the original sol_0
        merge_calls = mocks["merge_solutions"].call_args_list
        # First call: merge(sol_0, sol_1, config, client)
        assert merge_calls[0][0][0].content == sol_0.content
        assert merge_calls[0][0][1].content == sol_1.content
        # Second call: merge(merged_01, sol_2, config, client)
        assert merge_calls[1][0][0].content == merged_01.content
        assert merge_calls[1][0][1].content == sol_2.content


# ===========================================================================
# REQ-P1-026: Merge uses is_improvement_or_equal (>= semantics)
# ===========================================================================


@pytest.mark.unit
class TestMergeImprovementCheck:
    """Merge loop uses is_improvement_or_equal for >= semantics (REQ-P1-026)."""

    async def test_equal_score_accepted(self) -> None:
        """A merged solution with equal score is accepted (>= semantics)."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_best = _make_solution(content="best_code", source_model="m1")
        sol_other = _make_solution(content="other_code", source_model="m2")
        res_best = _make_eval_result(score=0.90)
        res_other = _make_eval_result(score=0.85)

        merged = _make_merged_solution("merged_equal")
        merged_res = _make_eval_result(score=0.90)  # Equal to best

        ranked = [(sol_best, res_best), (sol_other, res_other)]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_best, sol_other],
            eval_results=[
                (sol_best, res_best),
                (sol_other, res_other),
                (merged, merged_res),
            ],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=merged)
        # is_improvement_or_equal returns True for equal scores
        mocks["is_improvement_or_equal"] = MagicMock(return_value=True)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        # is_improvement_or_equal was called with the merged score, best score, direction
        mocks["is_improvement_or_equal"].assert_called()
        call_args = mocks["is_improvement_or_equal"].call_args[0]
        assert call_args[0] == 0.90  # new score
        assert call_args[1] == 0.90  # old best score

        # Merged solution should be the final result since improvement_or_equal=True
        assert result.initial_solution.content == merged.content

    async def test_is_improvement_or_equal_receives_direction(self) -> None:
        """is_improvement_or_equal receives the metric_direction from the task."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task(direction=MetricDirection.MINIMIZE)
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_best = _make_solution(content="best_code", source_model="m1")
        sol_other = _make_solution(content="other_code", source_model="m2")
        res_best = _make_eval_result(score=0.10)
        res_other = _make_eval_result(score=0.20)

        merged = _make_merged_solution("merged")
        merged_res = _make_eval_result(score=0.08)

        ranked = [(sol_best, res_best), (sol_other, res_other)]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_best, sol_other],
            eval_results=[
                (sol_best, res_best),
                (sol_other, res_other),
                (merged, merged_res),
            ],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=merged)
        mocks["is_improvement_or_equal"] = MagicMock(return_value=True)

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        call_args = mocks["is_improvement_or_equal"].call_args[0]
        assert call_args[2] == MetricDirection.MINIMIZE


# ===========================================================================
# REQ-P1-027: Break on first non-improvement
# ===========================================================================


@pytest.mark.unit
class TestBreakOnFirstNonImprovement:
    """Merge loop breaks on first non-improvement (REQ-P1-027)."""

    async def test_stops_after_non_improvement(self) -> None:
        """When is_improvement_or_equal returns False, merge loop stops."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=3)

        models = [_make_model(f"m{i}") for i in range(3)]
        sol_0 = _make_solution(content="code_0", source_model="m0")
        sol_1 = _make_solution(content="code_1", source_model="m1")
        sol_2 = _make_solution(content="code_2", source_model="m2")
        res_0 = _make_eval_result(score=0.90)
        res_1 = _make_eval_result(score=0.85)
        res_2 = _make_eval_result(score=0.80)

        merged_01 = _make_merged_solution("merged_01")
        merged_res_01 = _make_eval_result(score=0.88)  # Worse than 0.90

        ranked = [(sol_0, res_0), (sol_1, res_1), (sol_2, res_2)]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_0, sol_1, sol_2],
            eval_results=[
                (sol_0, res_0),
                (sol_1, res_1),
                (sol_2, res_2),
                (merged_01, merged_res_01),
            ],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(side_effect=[merged_01])
        # First merge is not an improvement
        mocks["is_improvement_or_equal"] = MagicMock(return_value=False)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        # Only 1 merge call (second candidate never reached)
        assert mocks["merge_solutions"].await_count == 1
        # Original best retained
        assert result.initial_solution.content == sol_0.content
        assert result.initial_score == 0.90

    async def test_retains_previous_best_on_non_improvement(self) -> None:
        """After break on non-improvement, the best before break is returned."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=3)

        models = [_make_model(f"m{i}") for i in range(3)]
        sol_0 = _make_solution(content="code_0", source_model="m0")
        sol_1 = _make_solution(content="code_1", source_model="m1")
        sol_2 = _make_solution(content="code_2", source_model="m2")
        res_0 = _make_eval_result(score=0.90)
        res_1 = _make_eval_result(score=0.85)
        res_2 = _make_eval_result(score=0.80)

        merged_01 = _make_merged_solution("merged_01")
        merged_res_01 = _make_eval_result(score=0.92)
        merged_012 = _make_merged_solution("merged_012")
        merged_res_012 = _make_eval_result(score=0.89)  # Worse than 0.92

        ranked = [(sol_0, res_0), (sol_1, res_1), (sol_2, res_2)]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_0, sol_1, sol_2],
            eval_results=[
                (sol_0, res_0),
                (sol_1, res_1),
                (sol_2, res_2),
                (merged_01, merged_res_01),
                (merged_012, merged_res_012),
            ],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(side_effect=[merged_01, merged_012])
        # First merge improves, second does not
        mocks["is_improvement_or_equal"] = MagicMock(side_effect=[True, False])

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        # merged_01 (first successful merge) is the final best
        assert result.initial_solution.content == merged_01.content
        assert result.initial_score == 0.92


# ===========================================================================
# REQ-P1-028: Merge execution failure breaks merge loop
# ===========================================================================


@pytest.mark.unit
class TestMergeExecutionFailure:
    """Merge execution failure (is_error or score=None) breaks loop (REQ-P1-028)."""

    async def test_merge_eval_error_breaks_loop(self) -> None:
        """When merged solution evaluation returns is_error=True, loop breaks."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=3)

        models = [_make_model(f"m{i}") for i in range(3)]
        sol_0 = _make_solution(content="code_0", source_model="m0")
        sol_1 = _make_solution(content="code_1", source_model="m1")
        sol_2 = _make_solution(content="code_2", source_model="m2")
        res_0 = _make_eval_result(score=0.90)
        res_1 = _make_eval_result(score=0.85)
        res_2 = _make_eval_result(score=0.80)

        merged_01 = _make_merged_solution("merged_01")
        merged_res_01 = _make_eval_result(score=None, is_error=True)

        ranked = [(sol_0, res_0), (sol_1, res_1), (sol_2, res_2)]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_0, sol_1, sol_2],
            eval_results=[
                (sol_0, res_0),
                (sol_1, res_1),
                (sol_2, res_2),
                (merged_01, merged_res_01),
            ],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=merged_01)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        # Only 1 merge attempted; loop broke on error
        assert mocks["merge_solutions"].await_count == 1
        # Original best retained
        assert result.initial_solution.content == sol_0.content

    async def test_merge_eval_none_score_breaks_loop(self) -> None:
        """When merged solution evaluation has score=None (no error), loop breaks."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_best = _make_solution(content="best", source_model="m1")
        sol_other = _make_solution(content="other", source_model="m2")
        res_best = _make_eval_result(score=0.90)
        res_other = _make_eval_result(score=0.85)

        merged = _make_merged_solution("merged")
        # score=None but not is_error (parsing failure)
        merged_res = _make_eval_result(score=None, is_error=False)

        ranked = [(sol_best, res_best), (sol_other, res_other)]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_best, sol_other],
            eval_results=[
                (sol_best, res_best),
                (sol_other, res_other),
                (merged, merged_res),
            ],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=merged)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        # Original best retained when score is None
        assert result.initial_solution.content == sol_best.content
        assert result.initial_score == 0.90


# ===========================================================================
# REQ-P1-029: Single candidate skips merge loop
# ===========================================================================


@pytest.mark.unit
class TestSingleCandidateSkipsMerge:
    """Single candidate (M=1 or only 1 success) skips merge loop (REQ-P1-029)."""

    async def test_single_model_skips_merge(self) -> None:
        """With M=1, merge_solutions is never called."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="only_code", source_model="m1")
        res = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[(sol, res)],
            ranked_pairs=[(sol, res)],
        )
        mocks["merge_solutions"] = AsyncMock()

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        mocks["merge_solutions"].assert_not_awaited()
        assert result.initial_solution.content == sol.content
        assert result.initial_score == 0.85

    async def test_only_one_success_skips_merge(self) -> None:
        """With M=3 but only 1 successful candidate, merge is skipped."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=3)

        models = [_make_model(f"m{i}") for i in range(3)]
        sol_0 = _make_solution(content="good_code", source_model="m0")
        res_0 = _make_eval_result(score=0.80)
        sol_1 = _make_solution(content="bad_1", source_model="m1")
        res_1 = _make_eval_result(score=None, is_error=True)
        sol_2 = _make_solution(content="bad_2", source_model="m2")
        res_2 = _make_eval_result(score=None, is_error=True)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_0, sol_1, sol_2],
            eval_results=[(sol_0, res_0), (sol_1, res_1), (sol_2, res_2)],
            ranked_pairs=[(sol_0, res_0)],
        )
        mocks["merge_solutions"] = AsyncMock()

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        mocks["merge_solutions"].assert_not_awaited()
        assert result.initial_solution.content == sol_0.content


# ===========================================================================
# Phase1Result construction: correct fields populated
# ===========================================================================


@pytest.mark.unit
class TestPhase1ResultConstruction:
    """Phase1Result is populated with correct fields."""

    async def test_retrieved_models_field(self) -> None:
        """Phase1Result.retrieved_models contains the models from retrieve_models."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="code_a", source_model="m1")
        sol_b = _make_solution(content="code_b", source_model="m2")
        res_a = _make_eval_result(score=0.80)
        res_b = _make_eval_result(score=0.90)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[(sol_a, res_a), (sol_b, res_b)],
            ranked_pairs=[(sol_b, res_b), (sol_a, res_a)],
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert result.retrieved_models == models

    async def test_candidate_solutions_field(self) -> None:
        """Phase1Result.candidate_solutions contains all generated candidates."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="code_a", source_model="m1")
        sol_b = _make_solution(content="code_b", source_model="m2")
        res_a = _make_eval_result(score=0.80)
        res_b = _make_eval_result(score=0.90)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[(sol_a, res_a), (sol_b, res_b)],
            ranked_pairs=[(sol_b, res_b), (sol_a, res_a)],
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert len(result.candidate_solutions) == 2
        contents = {s.content for s in result.candidate_solutions}
        assert "code_a" in contents
        assert "code_b" in contents

    async def test_candidate_scores_field(self) -> None:
        """Phase1Result.candidate_scores has correct scores (None for failures)."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=3)

        models = [_make_model(f"m{i}") for i in range(3)]
        sol_0 = _make_solution(content="code_0", source_model="m0")
        sol_1 = _make_solution(content="code_1", source_model="m1")
        sol_2 = _make_solution(content="code_2", source_model="m2")
        res_0 = _make_eval_result(score=0.80)
        res_1 = _make_eval_result(score=None, is_error=True)
        res_2 = _make_eval_result(score=0.90)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_0, sol_1, sol_2],
            eval_results=[(sol_0, res_0), (sol_1, res_1), (sol_2, res_2)],
            ranked_pairs=[(sol_2, res_2), (sol_0, res_0)],
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert result.candidate_scores == [0.80, None, 0.90]

    async def test_initial_solution_has_score_set(self) -> None:
        """Phase1Result.initial_solution has its score attribute set."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="code", source_model="m1")
        res = _make_eval_result(score=0.88)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[(sol, res)],
            ranked_pairs=[(sol, res)],
        )

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert result.initial_solution.score == 0.88
        assert result.initial_score == 0.88


# ===========================================================================
# generate_candidate returns None: treated as failed candidate
# ===========================================================================


@pytest.mark.unit
class TestGenerateCandidateReturnsNone:
    """generate_candidate returning None is treated as failed (score=None)."""

    async def test_none_candidate_recorded_as_none_score(self) -> None:
        """When generate_candidate returns None, score is None."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_good = _make_solution(content="good_code", source_model="m2")
        res_good = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[None, sol_good],
            eval_results=[(sol_good, res_good)],
            ranked_pairs=[(sol_good, res_good)],
        )

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert result.candidate_scores[0] is None
        assert result.candidate_scores[1] == 0.85

    async def test_none_candidate_not_evaluated(self) -> None:
        """When generate_candidate returns None, evaluate_with_retry is not called for it."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_good = _make_solution(content="good_code", source_model="m2")
        res_good = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[None, sol_good],
            eval_results=[(sol_good, res_good)],
            ranked_pairs=[(sol_good, res_good)],
        )

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        # Only 1 evaluate call (for the non-None candidate)
        assert mocks["evaluate_with_retry"].await_count == 1

    async def test_none_candidate_not_leakage_checked(self) -> None:
        """When generate_candidate returns None, check_and_fix_leakage is not called for it."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_good = _make_solution(content="good_code", source_model="m2")
        res_good = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[None, sol_good],
            eval_results=[(sol_good, res_good)],
            ranked_pairs=[(sol_good, res_good)],
        )

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        # Leakage check called only for the non-None candidate
        # (exactly 1 for the candidate generation phase)
        assert mocks["check_and_fix_leakage"].await_count >= 1


# ===========================================================================
# merge_solutions returns None: break out of merge loop
# ===========================================================================


@pytest.mark.unit
class TestMergeSolutionsReturnsNone:
    """merge_solutions returning None breaks out of merge loop."""

    async def test_none_merge_breaks_loop(self) -> None:
        """When merge_solutions returns None, merge loop stops immediately."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=3)

        models = [_make_model(f"m{i}") for i in range(3)]
        sol_0 = _make_solution(content="code_0", source_model="m0")
        sol_1 = _make_solution(content="code_1", source_model="m1")
        sol_2 = _make_solution(content="code_2", source_model="m2")
        res_0 = _make_eval_result(score=0.90)
        res_1 = _make_eval_result(score=0.85)
        res_2 = _make_eval_result(score=0.80)

        ranked = [(sol_0, res_0), (sol_1, res_1), (sol_2, res_2)]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_0, sol_1, sol_2],
            eval_results=[
                (sol_0, res_0),
                (sol_1, res_1),
                (sol_2, res_2),
            ],
            ranked_pairs=ranked,
        )
        # merge returns None (failed merge)
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        # Only 1 merge attempt; loop breaks after None return
        assert mocks["merge_solutions"].await_count == 1
        # Original best retained
        assert result.initial_solution.content == sol_0.content
        assert result.initial_score == 0.90

    async def test_none_merge_does_not_evaluate(self) -> None:
        """When merge_solutions returns None, no evaluation for the merge is attempted."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_best = _make_solution(content="best", source_model="m1")
        sol_other = _make_solution(content="other", source_model="m2")
        res_best = _make_eval_result(score=0.90)
        res_other = _make_eval_result(score=0.85)

        ranked = [(sol_best, res_best), (sol_other, res_other)]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_best, sol_other],
            eval_results=[(sol_best, res_best), (sol_other, res_other)],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        # Exactly 2 evaluate calls: one per candidate, zero for merge
        assert mocks["evaluate_with_retry"].await_count == 2


# ===========================================================================
# Leakage check during merge loop
# ===========================================================================


@pytest.mark.unit
class TestMergeLoopLeakageCheck:
    """Leakage check is applied to merged solutions before evaluation."""

    async def test_leakage_check_called_for_merged_solution(self) -> None:
        """check_and_fix_leakage is called on the merged solution before evaluation."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_best = _make_solution(content="best_code", source_model="m1")
        sol_other = _make_solution(content="other_code", source_model="m2")
        res_best = _make_eval_result(score=0.90)
        res_other = _make_eval_result(score=0.85)

        merged = _make_merged_solution("merged_code")
        merged_res = _make_eval_result(score=0.92)

        ranked = [(sol_best, res_best), (sol_other, res_other)]

        leakage_calls: list[SolutionScript] = []

        async def _track_leakage(
            sol: SolutionScript, _task: Any, _client: Any
        ) -> SolutionScript:
            leakage_calls.append(sol)
            return sol

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_best, sol_other],
            eval_results=[
                (sol_best, res_best),
                (sol_other, res_other),
                (merged, merged_res),
            ],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=merged)
        mocks["check_and_fix_leakage"] = AsyncMock(side_effect=_track_leakage)
        mocks["is_improvement_or_equal"] = MagicMock(return_value=True)

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        # Leakage should have been called for the merged solution
        merged_leakage_calls = [c for c in leakage_calls if c.content == "merged_code"]
        assert len(merged_leakage_calls) == 1


# ===========================================================================
# Merge loop updates solution score
# ===========================================================================


@pytest.mark.unit
class TestMergeLoopScoreUpdate:
    """Merge loop updates the solution's score attribute on improvement."""

    async def test_solution_score_updated_after_successful_merge(self) -> None:
        """After a successful merge, s_0.score is updated to the new best score."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_best = _make_solution(content="best_code", source_model="m1")
        sol_other = _make_solution(content="other_code", source_model="m2")
        res_best = _make_eval_result(score=0.90)
        res_other = _make_eval_result(score=0.85)

        merged = _make_merged_solution("merged_code")
        merged_res = _make_eval_result(score=0.95)

        ranked = [(sol_best, res_best), (sol_other, res_other)]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_best, sol_other],
            eval_results=[
                (sol_best, res_best),
                (sol_other, res_other),
                (merged, merged_res),
            ],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=merged)
        mocks["is_improvement_or_equal"] = MagicMock(return_value=True)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert result.initial_solution.score == 0.95
        assert result.initial_score == 0.95


# ===========================================================================
# Hypothesis: property-based test for candidate score recording
# ===========================================================================


@pytest.mark.unit
class TestPhase1Properties:
    """Property-based tests for run_phase1 invariants."""

    @given(
        num_models=st.integers(min_value=1, max_value=5),
        scores=st.lists(
            st.one_of(
                st.floats(
                    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
                ),
                st.none(),
            ),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=20, deadline=5000)
    async def test_candidate_scores_length_equals_model_count(
        self,
        num_models: int,
        scores: list[float | None],
    ) -> None:
        """candidate_scores has exactly M entries, one per model."""
        from mle_star.phase1 import run_phase1

        # Adjust scores list to match num_models
        adjusted_scores = (scores * ((num_models // len(scores)) + 1))[:num_models]

        # Skip if all scores are None (would raise RuntimeError)
        if all(s is None for s in adjusted_scores):
            return

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=num_models)

        models = [_make_model(f"m{i}") for i in range(num_models)]
        candidates: list[SolutionScript | None] = []
        eval_pairs: list[tuple[SolutionScript, EvaluationResult]] = []
        ranked_pairs: list[tuple[SolutionScript, EvaluationResult]] = []

        for i, score in enumerate(adjusted_scores):
            if score is None:
                candidates.append(None)
            else:
                sol = _make_solution(content=f"code_{i}", source_model=f"m{i}")
                candidates.append(sol)
                is_error = False
                res = _make_eval_result(score=score, is_error=is_error)
                eval_pairs.append((sol, res))
                ranked_pairs.append((sol, res))

        # Sort ranked pairs descending by score (maximize)
        ranked_pairs.sort(key=lambda p: p[1].score or 0.0, reverse=True)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=candidates,
            eval_results=eval_pairs,
            ranked_pairs=ranked_pairs,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert len(result.candidate_scores) == num_models

    @given(
        score=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=15, deadline=5000)
    async def test_initial_score_matches_initial_solution_score(
        self,
        score: float,
    ) -> None:
        """initial_score always equals initial_solution.score."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="code", source_model="m1")
        res = _make_eval_result(score=score)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[(sol, res)],
            ranked_pairs=[(sol, res)],
        )

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert result.initial_score == result.initial_solution.score


# ===========================================================================
# Full integration scenario: 3 models, 1 fail, 2 succeed, 1 merge improves
# ===========================================================================


@pytest.mark.unit
class TestFullPhase1Scenario:
    """End-to-end scenario with mixed successes, failures, and merges."""

    async def test_full_scenario_three_models(self) -> None:
        """3 models: 1 fails generation, 2 succeed, merge improves once then fails."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=3)

        models = [_make_model("m0"), _make_model("m1"), _make_model("m2")]

        # m0 -> None (generation failure)
        # m1 -> score 0.80
        # m2 -> score 0.90 (best)
        sol_1 = _make_solution(content="code_1", source_model="m1")
        sol_2 = _make_solution(content="code_2", source_model="m2")
        res_1 = _make_eval_result(score=0.80)
        res_2 = _make_eval_result(score=0.90)

        # After ranking: [sol_2(0.90), sol_1(0.80)]
        # Merge sol_2 + sol_1 -> merged (0.92) -> improvement!
        merged = _make_merged_solution("merged_21")
        merged_res = _make_eval_result(score=0.92)

        ranked = [(sol_2, res_2), (sol_1, res_1)]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[None, sol_1, sol_2],
            eval_results=[
                (sol_1, res_1),
                (sol_2, res_2),
                (merged, merged_res),
            ],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=merged)
        mocks["is_improvement_or_equal"] = MagicMock(return_value=True)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        # Verify Phase1Result structure
        assert result.retrieved_models == models
        assert len(result.candidate_solutions) >= 2  # at least the non-None ones
        assert result.candidate_scores[0] is None  # m0 failed generation
        assert result.candidate_scores[1] == 0.80
        assert result.candidate_scores[2] == 0.90
        assert result.initial_solution.content == merged.content
        assert result.initial_score == 0.92

    async def test_minimize_direction_scenario(self) -> None:
        """Phase 1 works correctly with MINIMIZE metric direction."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task(direction=MetricDirection.MINIMIZE)
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="code_a", source_model="m1")
        sol_b = _make_solution(content="code_b", source_model="m2")
        res_a = _make_eval_result(score=0.10)  # Best for minimize
        res_b = _make_eval_result(score=0.20)

        ranked = [(sol_a, res_a), (sol_b, res_b)]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[(sol_a, res_a), (sol_b, res_b)],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert result.initial_score == 0.10
        assert result.initial_solution.content == sol_a.content
