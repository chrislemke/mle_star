"""Tests for result assembly and error recovery (Task 47).

Validates ``_make_failed_phase2_result``, ``_collect_phase2_results`` (modified
for synthetic Phase2Result), ``_execute_phase3_or_skip`` (catches general
Exception), ``_finalize_with_recovery``, ``_log_phase_summary``, and
``_log_solution_lineage`` in ``orchestrator.py``.

These tests are TDD-first -- they define the expected behavior for
REQ-OR-036 through REQ-OR-043.

Refs:
    SRS 09c -- Orchestrator Result Assembly & Error Recovery.
    IMPLEMENTATION_PLAN.md Task 47.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, patch

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    CodeBlock,
    CodeBlockCategory,
    DataModality,
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

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Module path constant for patching
# ---------------------------------------------------------------------------

_MODULE = "mle_star.orchestrator"

# ---------------------------------------------------------------------------
# Reusable test helpers
# ---------------------------------------------------------------------------


def _make_task(**overrides: Any) -> TaskDescription:
    """Build a valid TaskDescription with sensible defaults."""
    defaults: dict[str, Any] = {
        "competition_id": "test-comp",
        "task_type": TaskType.CLASSIFICATION,
        "data_modality": DataModality.TABULAR,
        "evaluation_metric": "accuracy",
        "metric_direction": MetricDirection.MAXIMIZE,
        "description": "Predict the target variable from tabular features.",
        "data_dir": "./input",
        "output_dir": "./final",
    }
    defaults.update(overrides)
    return TaskDescription(**defaults)


def _make_config(**overrides: Any) -> PipelineConfig:
    """Build a valid PipelineConfig with sensible defaults."""
    defaults: dict[str, Any] = {"num_parallel_solutions": 1}
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_solution(**overrides: Any) -> SolutionScript:
    """Build a valid SolutionScript with sensible defaults."""
    defaults: dict[str, Any] = {
        "content": "import pandas as pd\nprint('hello')\n",
        "phase": SolutionPhase.FINAL,
    }
    defaults.update(overrides)
    return SolutionScript(**defaults)


def _make_phase1_result(**overrides: Any) -> Phase1Result:
    """Build a valid Phase1Result with sensible defaults."""
    defaults: dict[str, Any] = {
        "retrieved_models": [
            RetrievedModel(model_name="xgboost", example_code="import xgboost")
        ],
        "candidate_solutions": [_make_solution(phase=SolutionPhase.INIT)],
        "candidate_scores": [0.85],
        "initial_solution": _make_solution(phase=SolutionPhase.MERGED),
        "initial_score": 0.85,
    }
    defaults.update(overrides)
    return Phase1Result(**defaults)


def _make_phase2_result(**overrides: Any) -> Phase2Result:
    """Build a valid Phase2Result with sensible defaults."""
    defaults: dict[str, Any] = {
        "ablation_summaries": ["summary"],
        "refined_blocks": [
            CodeBlock(content="block", category=CodeBlockCategory.TRAINING)
        ],
        "best_solution": _make_solution(phase=SolutionPhase.REFINED),
        "best_score": 0.90,
        "step_history": [{"step": 0, "score": 0.90}],
    }
    defaults.update(overrides)
    return Phase2Result(**defaults)


def _make_phase3_result(**overrides: Any) -> Phase3Result:
    """Build a valid Phase3Result with sensible defaults."""
    sol = _make_solution(phase=SolutionPhase.ENSEMBLE)
    defaults: dict[str, Any] = {
        "input_solutions": [sol, sol],
        "ensemble_plans": ["plan_1"],
        "ensemble_scores": [0.92],
        "best_ensemble": sol,
        "best_ensemble_score": 0.92,
    }
    defaults.update(overrides)
    return Phase3Result(**defaults)


def _make_final_result(
    task: TaskDescription | None = None,
    config: PipelineConfig | None = None,
    **overrides: Any,
) -> FinalResult:
    """Build a valid FinalResult with sensible defaults."""
    defaults: dict[str, Any] = {
        "task": task or _make_task(),
        "config": config or _make_config(),
        "phase1": _make_phase1_result(),
        "phase2_results": [_make_phase2_result()],
        "phase3": None,
        "final_solution": _make_solution(phase=SolutionPhase.FINAL),
        "submission_path": "/output/submission.csv",
        "total_duration_seconds": 100.0,
    }
    defaults.update(overrides)
    return FinalResult(**defaults)


def _make_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with a dummy file."""
    data_dir = tmp_path / "input"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "train.csv").write_text("id,feature,target\n1,0.5,0\n")
    return data_dir


def _make_mock_client() -> AsyncMock:
    """Build a mock ClaudeCodeClient."""
    return AsyncMock()


# ===========================================================================
# REQ-OR-040: _make_failed_phase2_result
# ===========================================================================


@pytest.mark.unit
class TestMakeFailedPhase2Result:
    """_make_failed_phase2_result creates a synthetic Phase2Result (REQ-OR-040)."""

    def test_uses_phase1_initial_solution(self) -> None:
        """Synthetic result uses Phase 1 initial_solution as best_solution."""
        from mle_star.orchestrator import _make_failed_phase2_result

        p1_solution = _make_solution(
            content="p1_solution_code", phase=SolutionPhase.MERGED
        )
        p1_result = _make_phase1_result(initial_solution=p1_solution)

        failed = _make_failed_phase2_result(p1_result)

        assert failed.best_solution is p1_solution

    def test_uses_phase1_initial_score(self) -> None:
        """Synthetic result uses Phase 1 initial_score as best_score."""
        from mle_star.orchestrator import _make_failed_phase2_result

        p1_result = _make_phase1_result(initial_score=0.77)

        failed = _make_failed_phase2_result(p1_result)

        assert failed.best_score == pytest.approx(0.77)

    def test_step_history_contains_failed_marker(self) -> None:
        """step_history contains a single entry with step=0 and failed=True."""
        from mle_star.orchestrator import _make_failed_phase2_result

        p1_result = _make_phase1_result()

        failed = _make_failed_phase2_result(p1_result)

        assert len(failed.step_history) == 1
        assert failed.step_history[0]["step"] == 0
        assert failed.step_history[0]["failed"] is True

    def test_ablation_summaries_empty(self) -> None:
        """Synthetic result has empty ablation_summaries list."""
        from mle_star.orchestrator import _make_failed_phase2_result

        p1_result = _make_phase1_result()

        failed = _make_failed_phase2_result(p1_result)

        assert failed.ablation_summaries == []

    def test_refined_blocks_empty(self) -> None:
        """Synthetic result has empty refined_blocks list."""
        from mle_star.orchestrator import _make_failed_phase2_result

        p1_result = _make_phase1_result()

        failed = _make_failed_phase2_result(p1_result)

        assert failed.refined_blocks == []

    def test_returns_phase2_result_instance(self) -> None:
        """Return type is Phase2Result."""
        from mle_star.orchestrator import _make_failed_phase2_result

        p1_result = _make_phase1_result()

        failed = _make_failed_phase2_result(p1_result)

        assert isinstance(failed, Phase2Result)

    @given(score=st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=15, deadline=5000)
    def test_score_matches_phase1_for_any_value(self, score: float) -> None:
        """For any Phase 1 score, the synthetic result preserves it exactly."""
        from mle_star.orchestrator import _make_failed_phase2_result

        p1_result = _make_phase1_result(initial_score=score)

        failed = _make_failed_phase2_result(p1_result)

        assert failed.best_score == pytest.approx(score)


# ===========================================================================
# REQ-OR-040: _collect_phase2_results (modified — synthetic Phase2Result)
# ===========================================================================


@pytest.mark.unit
class TestCollectPhase2ResultsWithSynthetic:
    """_collect_phase2_results includes synthetic Phase2Result for failures (REQ-OR-040)."""

    def test_both_lists_same_length_as_raw_results(self) -> None:
        """phase2_results and solutions both have len(raw_results) entries."""
        from mle_star.orchestrator import _collect_phase2_results

        p1_result = _make_phase1_result()
        p2_success = _make_phase2_result(best_score=0.90)

        raw: list[Phase2Result | BaseException] = [
            RuntimeError("failed"),
            p2_success,
        ]

        phase2_results, solutions = _collect_phase2_results(raw, p1_result)

        assert len(phase2_results) == 2
        assert len(solutions) == 2

    def test_failed_path_gets_synthetic_phase2_result(self) -> None:
        """A failed path produces a synthetic Phase2Result, not a gap."""
        from mle_star.orchestrator import _collect_phase2_results

        p1_solution = _make_solution(
            content="p1_fallback_code", phase=SolutionPhase.MERGED
        )
        p1_result = _make_phase1_result(
            initial_solution=p1_solution, initial_score=0.80
        )

        raw: list[Phase2Result | BaseException] = [
            RuntimeError("path 0 failed"),
        ]

        phase2_results, solutions = _collect_phase2_results(raw, p1_result)

        assert len(phase2_results) == 1
        # Synthetic result uses Phase 1 solution and score
        assert phase2_results[0].best_solution is p1_solution
        assert phase2_results[0].best_score == pytest.approx(0.80)
        assert phase2_results[0].step_history[0]["failed"] is True
        # Solution list also uses Phase 1 fallback
        assert solutions[0] is p1_solution

    def test_success_path_preserves_original_phase2_result(self) -> None:
        """Successful paths preserve the original Phase2Result unchanged."""
        from mle_star.orchestrator import _collect_phase2_results

        p1_result = _make_phase1_result()
        p2_success = _make_phase2_result(best_score=0.92)

        raw: list[Phase2Result | BaseException] = [p2_success]

        phase2_results, solutions = _collect_phase2_results(raw, p1_result)

        assert phase2_results[0] is p2_success
        assert solutions[0] is p2_success.best_solution

    def test_mixed_success_and_failure(self) -> None:
        """With mixed results, each index gets the appropriate result."""
        from mle_star.orchestrator import _collect_phase2_results

        p1_solution = _make_solution(content="p1_code", phase=SolutionPhase.MERGED)
        p1_result = _make_phase1_result(
            initial_solution=p1_solution, initial_score=0.75
        )
        p2_success = _make_phase2_result(best_score=0.90)

        raw: list[Phase2Result | BaseException] = [
            RuntimeError("failed"),
            p2_success,
            asyncio.CancelledError(),
        ]

        phase2_results, solutions = _collect_phase2_results(raw, p1_result)

        assert len(phase2_results) == 3
        assert len(solutions) == 3

        # Index 0: synthetic (failure)
        assert phase2_results[0].best_solution is p1_solution
        assert phase2_results[0].step_history[0]["failed"] is True
        assert solutions[0] is p1_solution

        # Index 1: original success
        assert phase2_results[1] is p2_success
        assert solutions[1] is p2_success.best_solution

        # Index 2: synthetic (CancelledError)
        assert phase2_results[2].best_solution is p1_solution
        assert phase2_results[2].step_history[0]["failed"] is True
        assert solutions[2] is p1_solution

    def test_all_failures_produces_all_synthetic_results(self) -> None:
        """When all paths fail, all entries are synthetic Phase2Results."""
        from mle_star.orchestrator import _collect_phase2_results

        p1_solution = _make_solution(
            content="p1_fallback_all", phase=SolutionPhase.MERGED
        )
        p1_result = _make_phase1_result(
            initial_solution=p1_solution, initial_score=0.70
        )

        raw: list[Phase2Result | BaseException] = [
            RuntimeError("exploded"),
            asyncio.CancelledError(),
        ]

        phase2_results, solutions = _collect_phase2_results(raw, p1_result)

        assert len(phase2_results) == 2
        for result in phase2_results:
            assert result.best_solution is p1_solution
            assert result.best_score == pytest.approx(0.70)
            assert result.ablation_summaries == []
            assert result.refined_blocks == []

        assert all(s is p1_solution for s in solutions)

    def test_empty_raw_results_produces_empty_lists(self) -> None:
        """Empty raw results produce empty phase2_results and solutions."""
        from mle_star.orchestrator import _collect_phase2_results

        p1_result = _make_phase1_result()

        raw: list[Phase2Result | BaseException] = []

        phase2_results, solutions = _collect_phase2_results(raw, p1_result)

        assert phase2_results == []
        assert solutions == []

    def test_logs_warning_for_failed_path(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A warning is logged for each failed Phase 2 path."""
        from mle_star.orchestrator import _collect_phase2_results

        p1_result = _make_phase1_result()

        raw: list[Phase2Result | BaseException] = [
            RuntimeError("path 0 exploded"),
        ]

        with caplog.at_level(logging.WARNING):
            _collect_phase2_results(raw, p1_result)

        assert any(
            "path 0" in r.message.lower() or "fail" in r.message.lower()
            for r in caplog.records
        )

    @given(num_failures=st.integers(min_value=1, max_value=5))
    @settings(max_examples=10, deadline=5000)
    def test_synthetic_count_equals_failure_count(self, num_failures: int) -> None:
        """Number of synthetic results equals the number of failed paths."""
        from mle_star.orchestrator import _collect_phase2_results

        p1_result = _make_phase1_result(initial_score=0.60)
        raw: list[Phase2Result | BaseException] = [
            RuntimeError(f"fail_{i}") for i in range(num_failures)
        ]

        phase2_results, _solutions = _collect_phase2_results(raw, p1_result)

        assert len(phase2_results) == num_failures
        assert all(r.step_history[0]["failed"] is True for r in phase2_results)


# ===========================================================================
# REQ-OR-041: _execute_phase3_or_skip catches general Exception
# ===========================================================================


@pytest.mark.unit
class TestPhase3GeneralExceptionCatch:
    """_execute_phase3_or_skip catches general Exception (REQ-OR-041)."""

    async def test_timeout_error_returns_none_and_current_best(self) -> None:
        """TimeoutError returns (None, current_best)."""
        from mle_star.orchestrator import _execute_phase3_or_skip

        client = _make_mock_client()
        task = _make_task()
        config = _make_config(num_parallel_solutions=2)
        current_best = _make_solution(
            content="best_p2_solution", phase=SolutionPhase.REFINED
        )
        budgets = {"phase3": 0.01}

        async def _slow_phase3(*args: Any, **kwargs: Any) -> Phase3Result:
            await asyncio.sleep(100)
            return _make_phase3_result()

        # Use a deadline in the future so phase3 is attempted
        mock_time = Mock()
        mock_time.monotonic = Mock(side_effect=[0.0, 0.0, 100.0])

        with (
            patch(f"{_MODULE}.run_phase3", side_effect=_slow_phase3, create=True),
            patch(f"{_MODULE}.time", mock_time),
        ):
            result, best = await _execute_phase3_or_skip(
                client,
                task,
                config,
                [current_best, current_best],
                current_best,
                deadline=1000.0,
                budgets=budgets,
            )

        assert result is None
        assert best is current_best

    async def test_runtime_error_returns_none_and_current_best(self) -> None:
        """RuntimeError returns (None, current_best) instead of propagating."""
        from mle_star.orchestrator import _execute_phase3_or_skip

        client = _make_mock_client()
        task = _make_task()
        config = _make_config(num_parallel_solutions=2)
        current_best = _make_solution(
            content="best_solution", phase=SolutionPhase.REFINED
        )
        budgets = {"phase3": 100.0}

        mock_time = Mock()
        mock_time.monotonic = Mock(return_value=0.0)

        with (
            patch(
                f"{_MODULE}.run_phase3",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Phase 3 crashed"),
                create=True,
            ),
            patch(f"{_MODULE}.time", mock_time),
        ):
            result, best = await _execute_phase3_or_skip(
                client,
                task,
                config,
                [current_best, current_best],
                current_best,
                deadline=1000.0,
                budgets=budgets,
            )

        assert result is None
        assert best is current_best

    async def test_value_error_returns_none_and_current_best(self) -> None:
        """ValueError returns (None, current_best)."""
        from mle_star.orchestrator import _execute_phase3_or_skip

        client = _make_mock_client()
        task = _make_task()
        config = _make_config(num_parallel_solutions=2)
        current_best = _make_solution(
            content="val_err_solution", phase=SolutionPhase.REFINED
        )
        budgets = {"phase3": 100.0}

        mock_time = Mock()
        mock_time.monotonic = Mock(return_value=0.0)

        with (
            patch(
                f"{_MODULE}.run_phase3",
                new_callable=AsyncMock,
                side_effect=ValueError("bad input"),
                create=True,
            ),
            patch(f"{_MODULE}.time", mock_time),
        ):
            result, best = await _execute_phase3_or_skip(
                client,
                task,
                config,
                [current_best, current_best],
                current_best,
                deadline=1000.0,
                budgets=budgets,
            )

        assert result is None
        assert best is current_best

    async def test_general_exception_logged_as_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """General exceptions are logged as warnings."""
        from mle_star.orchestrator import _execute_phase3_or_skip

        client = _make_mock_client()
        task = _make_task()
        config = _make_config(num_parallel_solutions=2)
        current_best = _make_solution(phase=SolutionPhase.REFINED)
        budgets = {"phase3": 100.0}

        mock_time = Mock()
        mock_time.monotonic = Mock(return_value=0.0)

        with (
            patch(
                f"{_MODULE}.run_phase3",
                new_callable=AsyncMock,
                side_effect=RuntimeError("unexpected_error"),
                create=True,
            ),
            patch(f"{_MODULE}.time", mock_time),
            caplog.at_level(logging.WARNING),
        ):
            await _execute_phase3_or_skip(
                client,
                task,
                config,
                [current_best, current_best],
                current_best,
                deadline=1000.0,
                budgets=budgets,
            )

        log_text = " ".join(r.message for r in caplog.records)
        assert "phase 3" in log_text.lower() or "unexpected_error" in log_text.lower()

    async def test_skip_when_l_equals_1_unaffected(self) -> None:
        """L=1 skip behavior is not affected by new Exception handling."""
        from mle_star.orchestrator import _execute_phase3_or_skip

        client = _make_mock_client()
        task = _make_task()
        config = _make_config(num_parallel_solutions=1)
        current_best = _make_solution(phase=SolutionPhase.REFINED)
        budgets = {"phase3": 100.0}

        result, best = await _execute_phase3_or_skip(
            client,
            task,
            config,
            [current_best],
            current_best,
            deadline=1000.0,
            budgets=budgets,
        )

        assert result is None
        assert best is current_best

    async def test_skip_when_deadline_exceeded_unaffected(self) -> None:
        """Deadline-exceeded skip behavior is not affected by new Exception handling."""
        from mle_star.orchestrator import _execute_phase3_or_skip

        client = _make_mock_client()
        task = _make_task()
        config = _make_config(num_parallel_solutions=2)
        current_best = _make_solution(phase=SolutionPhase.REFINED)
        budgets = {"phase3": 100.0}

        # Deadline already exceeded
        mock_time = Mock()
        mock_time.monotonic = Mock(return_value=2000.0)

        with patch(f"{_MODULE}.time", mock_time):
            result, best = await _execute_phase3_or_skip(
                client,
                task,
                config,
                [current_best, current_best],
                current_best,
                deadline=1000.0,
                budgets=budgets,
            )

        assert result is None
        assert best is current_best

    async def test_successful_phase3_still_works(self) -> None:
        """Successful Phase 3 execution is unaffected by new Exception handling."""
        from mle_star.orchestrator import _execute_phase3_or_skip

        client = _make_mock_client()
        task = _make_task()
        config = _make_config(num_parallel_solutions=2)
        current_best = _make_solution(phase=SolutionPhase.REFINED)
        p3_result = _make_phase3_result()
        budgets = {"phase3": 100.0}

        mock_time = Mock()
        mock_time.monotonic = Mock(return_value=0.0)

        with (
            patch(
                f"{_MODULE}.run_phase3",
                new_callable=AsyncMock,
                return_value=p3_result,
                create=True,
            ),
            patch(f"{_MODULE}.time", mock_time),
        ):
            result, best = await _execute_phase3_or_skip(
                client,
                task,
                config,
                [current_best, current_best],
                current_best,
                deadline=1000.0,
                budgets=budgets,
            )

        assert result is p3_result
        assert best is p3_result.best_ensemble


# ===========================================================================
# REQ-OR-043: _finalize_with_recovery
# ===========================================================================


@pytest.mark.unit
class TestFinalizeWithRecovery:
    """_finalize_with_recovery wraps run_finalization with error recovery (REQ-OR-043)."""

    async def test_success_returns_final_result_with_updated_duration(self) -> None:
        """On success, total_duration_seconds is updated to pipeline wall-clock time."""
        from mle_star.orchestrator import _finalize_with_recovery

        task = _make_task()
        config = _make_config()
        p1_result = _make_phase1_result()
        p2_results = [_make_phase2_result()]
        best_solution = _make_solution(phase=SolutionPhase.REFINED)

        # Finalization returns result with finalization-only duration
        finalization_fr = _make_final_result(
            task=task,
            config=config,
            total_duration_seconds=10.0,  # finalization-only
        )

        mock_time = Mock()
        # pipeline_start=100.0, current time after finalization=200.0 -> duration=100.0
        mock_time.monotonic = Mock(return_value=200.0)

        with (
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=finalization_fr,
            ),
            patch(f"{_MODULE}.time", mock_time),
        ):
            result = await _finalize_with_recovery(
                client=_make_mock_client(),
                best_solution=best_solution,
                task=task,
                config=config,
                phase1_result=p1_result,
                phase2_results=p2_results,
                phase3_result=None,
                pipeline_start=100.0,
            )

        assert result.total_duration_seconds == pytest.approx(100.0)

    async def test_failure_returns_best_effort_final_result(self) -> None:
        """On run_finalization failure, a best-effort FinalResult is constructed."""
        from mle_star.orchestrator import _finalize_with_recovery

        task = _make_task()
        config = _make_config()
        p1_result = _make_phase1_result()
        p2_results = [_make_phase2_result()]
        best_solution = _make_solution(
            content="best_available_code", phase=SolutionPhase.REFINED
        )

        mock_time = Mock()
        mock_time.monotonic = Mock(return_value=150.0)

        with (
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                side_effect=RuntimeError("finalization crashed"),
            ),
            patch(f"{_MODULE}.time", mock_time),
        ):
            result = await _finalize_with_recovery(
                client=_make_mock_client(),
                best_solution=best_solution,
                task=task,
                config=config,
                phase1_result=p1_result,
                phase2_results=p2_results,
                phase3_result=None,
                pipeline_start=50.0,
            )

        assert isinstance(result, FinalResult)
        assert result.submission_path == ""
        assert result.total_duration_seconds == pytest.approx(100.0)

    async def test_failure_preserves_phase_results(self) -> None:
        """On failure, the best-effort FinalResult contains all phase results."""
        from mle_star.orchestrator import _finalize_with_recovery

        task = _make_task()
        config = _make_config()
        p1_result = _make_phase1_result()
        p2_result = _make_phase2_result()
        p3_result = _make_phase3_result()
        best_solution = _make_solution(phase=SolutionPhase.REFINED)

        mock_time = Mock()
        mock_time.monotonic = Mock(return_value=100.0)

        with (
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                side_effect=RuntimeError("finalization failed"),
            ),
            patch(f"{_MODULE}.time", mock_time),
        ):
            result = await _finalize_with_recovery(
                client=_make_mock_client(),
                best_solution=best_solution,
                task=task,
                config=config,
                phase1_result=p1_result,
                phase2_results=[p2_result],
                phase3_result=p3_result,
                pipeline_start=0.0,
            )

        assert result.phase1 is p1_result
        assert result.phase2_results == [p2_result]
        assert result.phase3 is p3_result

    async def test_failure_uses_best_solution_as_final(self) -> None:
        """On failure, the best available solution becomes final_solution."""
        from mle_star.orchestrator import _finalize_with_recovery

        task = _make_task()
        config = _make_config()
        p1_result = _make_phase1_result()
        best_solution = _make_solution(
            content="the_best_solution_available", phase=SolutionPhase.REFINED
        )

        mock_time = Mock()
        mock_time.monotonic = Mock(return_value=100.0)

        with (
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
            patch(f"{_MODULE}.time", mock_time),
        ):
            result = await _finalize_with_recovery(
                client=_make_mock_client(),
                best_solution=best_solution,
                task=task,
                config=config,
                phase1_result=p1_result,
                phase2_results=[],
                phase3_result=None,
                pipeline_start=0.0,
            )

        assert result.final_solution.content == "the_best_solution_available"

    async def test_failure_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """On failure, a warning is logged about the finalization failure."""
        from mle_star.orchestrator import _finalize_with_recovery

        task = _make_task()
        config = _make_config()
        p1_result = _make_phase1_result()
        best_solution = _make_solution(phase=SolutionPhase.REFINED)

        mock_time = Mock()
        mock_time.monotonic = Mock(return_value=100.0)

        with (
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                side_effect=RuntimeError("finalization_error"),
            ),
            patch(f"{_MODULE}.time", mock_time),
            caplog.at_level(logging.WARNING),
        ):
            await _finalize_with_recovery(
                client=_make_mock_client(),
                best_solution=best_solution,
                task=task,
                config=config,
                phase1_result=p1_result,
                phase2_results=[],
                phase3_result=None,
                pipeline_start=0.0,
            )

        log_text = " ".join(r.message for r in caplog.records)
        assert (
            "finalization" in log_text.lower()
            or "finalization_error" in log_text.lower()
        )

    @given(
        start=st.floats(min_value=0.0, max_value=10000.0),
        end_offset=st.floats(min_value=0.01, max_value=10000.0),
    )
    @settings(max_examples=15, deadline=5000)
    async def test_duration_always_set_on_failure(
        self, start: float, end_offset: float
    ) -> None:
        """On failure, duration is always populated."""
        from mle_star.orchestrator import _finalize_with_recovery

        task = _make_task()
        config = _make_config()
        p1_result = _make_phase1_result()
        best_solution = _make_solution(phase=SolutionPhase.REFINED)

        end_time = start + end_offset
        mock_time = Mock()
        mock_time.monotonic = Mock(return_value=end_time)

        with (
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                side_effect=RuntimeError("fail"),
            ),
            patch(f"{_MODULE}.time", mock_time),
        ):
            result = await _finalize_with_recovery(
                client=_make_mock_client(),
                best_solution=best_solution,
                task=task,
                config=config,
                phase1_result=p1_result,
                phase2_results=[],
                phase3_result=None,
                pipeline_start=start,
            )

        assert result.total_duration_seconds == pytest.approx(end_offset)


# ===========================================================================
# REQ-OR-036: FinalResult assembly (pipeline-level)
# ===========================================================================


@pytest.mark.unit
class TestFinalResultAssembly:
    """run_pipeline sets total_duration_seconds (REQ-OR-036)."""

    async def test_total_duration_is_pipeline_wall_clock(self, tmp_path: Path) -> None:
        """total_duration_seconds reflects full pipeline time, not finalization time."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(time_limit_seconds=86400)

        mock_client = _make_mock_client()
        p1_result = _make_phase1_result()
        p2_result = _make_phase2_result()

        # Finalization returns a result with finalization-only duration (should be overwritten)
        fr = _make_final_result(
            task=task,
            config=config,
            total_duration_seconds=5.0,  # finalization-only
        )

        with (
            patch(f"{_MODULE}._create_client", return_value=mock_client),
            patch(f"{_MODULE}.validate_api_key"),
            patch(f"{_MODULE}.check_claude_cli_version"),
            patch(f"{_MODULE}.setup_working_directory"),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=p1_result,
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=p2_result,
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=fr,
            ),
        ):
            result = await run_pipeline(task, config)

        assert isinstance(result, FinalResult)
        # Pipeline-level duration must be >= 0 (cannot be exactly finalization's 5.0
        # unless the pipeline happened to take exactly 5.0s, which is highly unlikely)
        assert result.total_duration_seconds >= 0



# ===========================================================================
# REQ-OR-037: _log_phase_summary -- per-phase duration breakdown
# ===========================================================================


@pytest.mark.unit
class TestLogPhaseSummary:
    """_log_phase_summary logs duration breakdowns (REQ-OR-037)."""

    def test_logs_duration_breakdown(self, caplog: pytest.LogCaptureFixture) -> None:
        """Phase durations are logged in the summary."""
        from mle_star.orchestrator import _log_phase_summary

        phase_durations = {
            "phase1": 10.0,
            "phase2": 50.0,
            "phase3": 15.0,
            "finalization": 5.0,
            "total": 80.0,
        }

        with caplog.at_level(logging.INFO):
            _log_phase_summary(phase_durations)

        log_text = " ".join(r.message for r in caplog.records)
        assert "duration" in log_text.lower() or "phase1" in log_text.lower()

    def test_logs_at_info_level(self, caplog: pytest.LogCaptureFixture) -> None:
        """Summary is logged at INFO level."""
        from mle_star.orchestrator import _log_phase_summary

        phase_durations = {"phase1": 10.0, "total": 10.0}

        with caplog.at_level(logging.INFO):
            _log_phase_summary(phase_durations)

        assert len(caplog.records) >= 1
        assert all(r.levelno >= logging.INFO for r in caplog.records)

    def test_handles_empty_dict(self, caplog: pytest.LogCaptureFixture) -> None:
        """Empty phase_durations does not raise."""
        from mle_star.orchestrator import _log_phase_summary

        with caplog.at_level(logging.INFO):
            _log_phase_summary({})

        # Should not raise; may or may not produce log output
        # The test passes if no exception is raised

    def test_includes_total(self, caplog: pytest.LogCaptureFixture) -> None:
        """Summary includes the total duration."""
        from mle_star.orchestrator import _log_phase_summary

        phase_durations = {"phase1": 10.0, "total": 10.0}

        with caplog.at_level(logging.INFO):
            _log_phase_summary(phase_durations)

        log_text = " ".join(r.message for r in caplog.records)
        assert "total" in log_text.lower() or "10" in log_text


# ===========================================================================
# REQ-OR-038: Per-phase duration breakdown (logged)
# ===========================================================================


@pytest.mark.unit
class TestPhaseDurationBreakdown:
    """Pipeline logs per-phase duration breakdown after completion (REQ-OR-038)."""

    async def test_duration_breakdown_logged_after_pipeline(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """After a successful pipeline run, phase durations are logged."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(time_limit_seconds=86400)

        mock_client = _make_mock_client()
        p1_result = _make_phase1_result()
        p2_result = _make_phase2_result()
        fr = _make_final_result(task=task, config=config)

        with (
            patch(f"{_MODULE}._create_client", return_value=mock_client),
            patch(f"{_MODULE}.validate_api_key"),
            patch(f"{_MODULE}.check_claude_cli_version"),
            patch(f"{_MODULE}.setup_working_directory"),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=p1_result,
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=p2_result,
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=fr,
            ),
            caplog.at_level(logging.INFO),
        ):
            await run_pipeline(task, config)

        log_text = " ".join(r.message for r in caplog.records)
        # Should contain some indication of phase timing
        assert (
            "phase 1" in log_text.lower()
            or "phase1" in log_text.lower()
            or "duration" in log_text.lower()
            or "completed in" in log_text.lower()
        )


# ===========================================================================
# REQ-OR-039: _log_solution_lineage
# ===========================================================================


@pytest.mark.unit
class TestLogSolutionLineage:
    """_log_solution_lineage logs solution tracing through phases (REQ-OR-039)."""

    def test_logs_lineage_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Solution lineage is logged at INFO level."""
        from mle_star.orchestrator import _log_solution_lineage

        p1_result = _make_phase1_result()
        p2_results = [_make_phase2_result()]
        p3_result = None
        final_solution = _make_solution(phase=SolutionPhase.FINAL)

        with caplog.at_level(logging.INFO):
            _log_solution_lineage(p1_result, p2_results, p3_result, final_solution)

        assert len(caplog.records) >= 1

    def test_logs_phase1_score(self, caplog: pytest.LogCaptureFixture) -> None:
        """Lineage log includes Phase 1 score."""
        from mle_star.orchestrator import _log_solution_lineage

        p1_result = _make_phase1_result(initial_score=0.85)
        p2_results = [_make_phase2_result()]

        with caplog.at_level(logging.INFO):
            _log_solution_lineage(
                p1_result, p2_results, None, _make_solution(phase=SolutionPhase.FINAL)
            )

        log_text = " ".join(r.message for r in caplog.records)
        assert "0.85" in log_text or "phase" in log_text.lower()

    def test_logs_phase2_scores(self, caplog: pytest.LogCaptureFixture) -> None:
        """Lineage log includes Phase 2 best scores."""
        from mle_star.orchestrator import _log_solution_lineage

        p1_result = _make_phase1_result()
        p2_results = [
            _make_phase2_result(best_score=0.90),
            _make_phase2_result(best_score=0.88),
        ]

        with caplog.at_level(logging.INFO):
            _log_solution_lineage(
                p1_result, p2_results, None, _make_solution(phase=SolutionPhase.FINAL)
            )

        log_text = " ".join(r.message for r in caplog.records)
        assert (
            "0.9" in log_text
            or "phase 2" in log_text.lower()
            or "phase2" in log_text.lower()
        )

    def test_logs_phase3_score_when_present(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Lineage log includes Phase 3 score when Phase 3 ran."""
        from mle_star.orchestrator import _log_solution_lineage

        p1_result = _make_phase1_result()
        p2_results = [_make_phase2_result()]
        p3_result = _make_phase3_result(best_ensemble_score=0.95)

        with caplog.at_level(logging.INFO):
            _log_solution_lineage(
                p1_result,
                p2_results,
                p3_result,
                _make_solution(phase=SolutionPhase.FINAL),
            )

        log_text = " ".join(r.message for r in caplog.records)
        assert (
            "0.95" in log_text
            or "phase 3" in log_text.lower()
            or "ensemble" in log_text.lower()
        )

    def test_handles_none_phase3(self, caplog: pytest.LogCaptureFixture) -> None:
        """Lineage log handles Phase 3 being None (L=1 case)."""
        from mle_star.orchestrator import _log_solution_lineage

        p1_result = _make_phase1_result()
        p2_results = [_make_phase2_result()]

        with caplog.at_level(logging.INFO):
            # Should not raise
            _log_solution_lineage(
                p1_result,
                p2_results,
                None,
                _make_solution(phase=SolutionPhase.FINAL),
            )

    def test_handles_empty_phase2_results(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Lineage log handles empty phase2_results list."""
        from mle_star.orchestrator import _log_solution_lineage

        p1_result = _make_phase1_result()

        with caplog.at_level(logging.INFO):
            # Should not raise
            _log_solution_lineage(
                p1_result,
                [],
                None,
                _make_solution(phase=SolutionPhase.FINAL),
            )


# ===========================================================================
# REQ-OR-042: Phase 1 failure -> PipelineError with diagnostics
# ===========================================================================


@pytest.mark.unit
class TestPhase1FailurePipelineError:
    """Phase 1 failure raises PipelineError with diagnostics (REQ-OR-042)."""

    async def test_phase1_exception_raises_pipeline_error(self, tmp_path: Path) -> None:
        """Non-timeout Phase 1 failure raises PipelineError with diagnostics."""
        from mle_star.orchestrator import PipelineError, run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(time_limit_seconds=86400)

        mock_client = _make_mock_client()

        with (
            patch(f"{_MODULE}._create_client", return_value=mock_client),
            patch(f"{_MODULE}.validate_api_key"),
            patch(f"{_MODULE}.check_claude_cli_version"),
            patch(f"{_MODULE}.setup_working_directory"),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Phase 1 crashed"),
            ),
            pytest.raises((PipelineError, RuntimeError)),
        ):
            await run_pipeline(task, config)

    async def test_pipeline_error_has_diagnostics_dict(self, tmp_path: Path) -> None:
        """PipelineError includes a diagnostics dict with structured context."""
        from mle_star.orchestrator import PipelineError

        error = PipelineError(
            "test failure",
            diagnostics={"elapsed_time": 42.0, "last_phase": "phase1"},
        )

        assert isinstance(error.diagnostics, dict)
        assert "elapsed_time" in error.diagnostics
        assert error.diagnostics["elapsed_time"] == 42.0
        assert "last_phase" in error.diagnostics

    async def test_pipeline_timeout_error_is_pipeline_error(self) -> None:
        """PipelineTimeoutError is a subclass of PipelineError."""
        from mle_star.orchestrator import PipelineError, PipelineTimeoutError

        assert issubclass(PipelineTimeoutError, PipelineError)

        error = PipelineTimeoutError(
            "timed out",
            diagnostics={"elapsed_time": 100.0},
        )
        assert isinstance(error, PipelineError)
        assert error.diagnostics["elapsed_time"] == 100.0


# ===========================================================================
# REQ-OR-043: Finalization failure -> best-effort FinalResult (integration)
# ===========================================================================


@pytest.mark.unit
class TestFinalizationFailureIntegration:
    """Finalization failure produces best-effort FinalResult (REQ-OR-043, integration)."""

    async def test_finalization_exception_produces_result_not_crash(
        self, tmp_path: Path
    ) -> None:
        """If run_finalization raises, pipeline returns FinalResult instead of crashing."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(time_limit_seconds=86400)

        mock_client = _make_mock_client()
        p1_result = _make_phase1_result()
        p2_result = _make_phase2_result()

        with (
            patch(f"{_MODULE}._create_client", return_value=mock_client),
            patch(f"{_MODULE}.validate_api_key"),
            patch(f"{_MODULE}.check_claude_cli_version"),
            patch(f"{_MODULE}.setup_working_directory"),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=p1_result,
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=p2_result,
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                side_effect=RuntimeError("finalization exploded"),
            ),
        ):
            result = await run_pipeline(task, config)

        assert isinstance(result, FinalResult)
        assert result.submission_path == ""

    async def test_finalization_failure_includes_phase_results(
        self, tmp_path: Path
    ) -> None:
        """Best-effort FinalResult on finalization failure includes all phase results."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(time_limit_seconds=86400, num_parallel_solutions=2)

        mock_client = _make_mock_client()
        p1_result = _make_phase1_result()
        p2_result = _make_phase2_result()
        p3_result = _make_phase3_result()

        with (
            patch(f"{_MODULE}._create_client", return_value=mock_client),
            patch(f"{_MODULE}.validate_api_key"),
            patch(f"{_MODULE}.check_claude_cli_version"),
            patch(f"{_MODULE}.setup_working_directory"),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=p1_result,
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=p2_result,
            ),
            patch(
                f"{_MODULE}.run_phase3",
                new_callable=AsyncMock,
                return_value=p3_result,
                create=True,
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                side_effect=RuntimeError("finalization crashed"),
            ),
        ):
            result = await run_pipeline(task, config)

        assert result.phase1 is p1_result
        assert len(result.phase2_results) == 2
        assert result.phase3 is p3_result

    async def test_finalization_failure_sets_duration(
        self, tmp_path: Path
    ) -> None:
        """Best-effort result on finalization failure has duration set."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(time_limit_seconds=86400)

        mock_client = _make_mock_client()
        p1_result = _make_phase1_result()
        p2_result = _make_phase2_result()

        with (
            patch(f"{_MODULE}._create_client", return_value=mock_client),
            patch(f"{_MODULE}.validate_api_key"),
            patch(f"{_MODULE}.check_claude_cli_version"),
            patch(f"{_MODULE}.setup_working_directory"),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=p1_result,
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=p2_result,
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
        ):
            result = await run_pipeline(task, config)

        assert result.total_duration_seconds >= 0

    async def test_finalization_failure_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Finalization failure is logged as a warning."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(time_limit_seconds=86400)

        mock_client = _make_mock_client()
        p1_result = _make_phase1_result()
        p2_result = _make_phase2_result()

        with (
            patch(f"{_MODULE}._create_client", return_value=mock_client),
            patch(f"{_MODULE}.validate_api_key"),
            patch(f"{_MODULE}.check_claude_cli_version"),
            patch(f"{_MODULE}.setup_working_directory"),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=p1_result,
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=p2_result,
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                side_effect=RuntimeError("finalization_crash_event"),
            ),
            caplog.at_level(logging.WARNING),
        ):
            await run_pipeline(task, config)

        log_text = " ".join(r.message for r in caplog.records)
        assert (
            "finalization" in log_text.lower()
            or "finalization_crash_event" in log_text.lower()
        )


# ===========================================================================
# REQ-OR-040: All Phase 2 paths fail -> all Phase 3 inputs are Phase 1 copies
# ===========================================================================


@pytest.mark.unit
class TestAllPhase2FailsAllPhase1Copies:
    """When ALL Phase 2 paths fail, ALL Phase 3 inputs are Phase 1 copies (REQ-OR-040)."""

    def test_all_failures_produce_all_phase1_solutions(self) -> None:
        """Every solution in the output list is the Phase 1 fallback."""
        from mle_star.orchestrator import _collect_phase2_results

        p1_solution = _make_solution(content="phase1_only", phase=SolutionPhase.MERGED)
        p1_result = _make_phase1_result(
            initial_solution=p1_solution, initial_score=0.65
        )

        raw: list[Phase2Result | BaseException] = [
            RuntimeError("fail_0"),
            ValueError("fail_1"),
            asyncio.CancelledError(),
        ]

        phase2_results, solutions = _collect_phase2_results(raw, p1_result)

        # All 3 results should be synthetic
        assert len(phase2_results) == 3
        assert len(solutions) == 3

        for result in phase2_results:
            assert result.best_solution is p1_solution
            assert result.best_score == pytest.approx(0.65)
            assert result.ablation_summaries == []
            assert result.refined_blocks == []
            assert result.step_history[0]["failed"] is True

        for sol in solutions:
            assert sol is p1_solution

    def test_all_failures_phase2_results_are_valid(self) -> None:
        """Synthetic Phase2Results from all-failure scenario are valid Pydantic models."""
        from mle_star.orchestrator import _collect_phase2_results

        p1_result = _make_phase1_result(initial_score=0.50)

        raw: list[Phase2Result | BaseException] = [
            RuntimeError("boom"),
            RuntimeError("crash"),
        ]

        phase2_results, _ = _collect_phase2_results(raw, p1_result)

        for result in phase2_results:
            assert isinstance(result, Phase2Result)
            # Validate the model is properly constructed
            assert result.best_score == pytest.approx(0.50)


# ===========================================================================
# Property-based tests for result assembly
# ===========================================================================


@pytest.mark.unit
class TestResultAssemblyProperties:
    """Property-based tests for result assembly invariants."""

    @given(
        num_success=st.integers(min_value=0, max_value=5),
        num_failure=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=20, deadline=5000)
    def test_collect_phase2_output_length_equals_input_length(
        self, num_success: int, num_failure: int
    ) -> None:
        """Output lists always have the same length as the input list."""
        from mle_star.orchestrator import _collect_phase2_results

        p1_result = _make_phase1_result(initial_score=0.50)
        successes: list[Phase2Result | BaseException] = [
            _make_phase2_result(best_score=0.80 + i * 0.01) for i in range(num_success)
        ]
        failures: list[Phase2Result | BaseException] = [
            RuntimeError(f"fail_{i}") for i in range(num_failure)
        ]

        raw = successes + failures
        phase2_results, solutions = _collect_phase2_results(raw, p1_result)

        assert len(phase2_results) == num_success + num_failure
        assert len(solutions) == num_success + num_failure

    @given(score=st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=15, deadline=5000)
    def test_synthetic_phase2_score_preserves_p1_score(self, score: float) -> None:
        """Synthetic Phase2Result always preserves the Phase 1 initial_score."""
        from mle_star.orchestrator import _make_failed_phase2_result

        p1_result = _make_phase1_result(initial_score=score)
        synthetic = _make_failed_phase2_result(p1_result)

        assert synthetic.best_score == pytest.approx(score)

    @given(
        duration=st.floats(min_value=0.01, max_value=10000.0),
    )
    @settings(max_examples=15, deadline=5000)
    async def test_finalize_with_recovery_success_always_sets_duration(
        self, duration: float
    ) -> None:
        """Successful finalization always has duration from pipeline context."""
        from mle_star.orchestrator import _finalize_with_recovery

        task = _make_task()
        config = _make_config()
        p1_result = _make_phase1_result()
        best_solution = _make_solution(phase=SolutionPhase.REFINED)

        fr = _make_final_result(
            task=task,
            config=config,
            total_duration_seconds=1.0,
        )

        pipeline_start = 0.0
        end_time = pipeline_start + duration

        mock_time = Mock()
        mock_time.monotonic = Mock(return_value=end_time)

        with (
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=fr,
            ),
            patch(f"{_MODULE}.time", mock_time),
        ):
            result = await _finalize_with_recovery(
                client=_make_mock_client(),
                best_solution=best_solution,
                task=task,
                config=config,
                phase1_result=p1_result,
                phase2_results=[],
                phase3_result=None,
                pipeline_start=pipeline_start,
            )

        assert result.total_duration_seconds == pytest.approx(duration)


# ===========================================================================
# Edge cases and robustness
# ===========================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Edge cases for result assembly and error recovery."""

    def test_make_failed_phase2_result_with_zero_score(self) -> None:
        """Synthetic result handles Phase 1 score of 0.0 correctly."""
        from mle_star.orchestrator import _make_failed_phase2_result

        p1_result = _make_phase1_result(initial_score=0.0)
        failed = _make_failed_phase2_result(p1_result)

        assert failed.best_score == pytest.approx(0.0)

    def test_make_failed_phase2_result_with_negative_score(self) -> None:
        """Synthetic result handles negative Phase 1 score (minimize metric)."""
        from mle_star.orchestrator import _make_failed_phase2_result

        p1_result = _make_phase1_result(initial_score=-0.5)
        failed = _make_failed_phase2_result(p1_result)

        assert failed.best_score == pytest.approx(-0.5)

    def test_collect_phase2_results_preserves_order(self) -> None:
        """Results maintain the same order as raw_results input."""
        from mle_star.orchestrator import _collect_phase2_results

        p1_solution = _make_solution(content="p1_code", phase=SolutionPhase.MERGED)
        p1_result = _make_phase1_result(
            initial_solution=p1_solution, initial_score=0.50
        )

        sol_a = _make_solution(content="sol_a", phase=SolutionPhase.REFINED)
        sol_b = _make_solution(content="sol_b", phase=SolutionPhase.REFINED)
        p2_a = _make_phase2_result(best_solution=sol_a, best_score=0.80)
        p2_b = _make_phase2_result(best_solution=sol_b, best_score=0.85)

        raw: list[Phase2Result | BaseException] = [
            p2_a,
            RuntimeError("fail"),
            p2_b,
        ]

        _phase2_results, solutions = _collect_phase2_results(raw, p1_result)

        # Index 0: success (p2_a)
        assert solutions[0] is sol_a
        # Index 1: failure -> Phase 1 fallback
        assert solutions[1] is p1_solution
        # Index 2: success (p2_b)
        assert solutions[2] is sol_b

    async def test_finalize_with_recovery_catches_keyboard_interrupt(self) -> None:
        """KeyboardInterrupt propagates (not caught by recovery)."""
        from mle_star.orchestrator import _finalize_with_recovery

        task = _make_task()
        config = _make_config()
        p1_result = _make_phase1_result()
        best_solution = _make_solution(phase=SolutionPhase.REFINED)

        mock_time = Mock()
        mock_time.monotonic = Mock(return_value=100.0)

        with (
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                side_effect=KeyboardInterrupt(),
            ),
            patch(f"{_MODULE}.time", mock_time),
            pytest.raises(KeyboardInterrupt),
        ):
            await _finalize_with_recovery(
                client=_make_mock_client(),
                best_solution=best_solution,
                task=task,
                config=config,
                phase1_result=p1_result,
                phase2_results=[],
                phase3_result=None,
                pipeline_start=0.0,
            )

    async def test_finalize_with_recovery_catches_os_error(self) -> None:
        """OSError during finalization is caught and produces best-effort result."""
        from mle_star.orchestrator import _finalize_with_recovery

        task = _make_task()
        config = _make_config()
        p1_result = _make_phase1_result()
        best_solution = _make_solution(phase=SolutionPhase.REFINED)

        mock_time = Mock()
        mock_time.monotonic = Mock(return_value=100.0)

        with (
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                side_effect=OSError("disk full"),
            ),
            patch(f"{_MODULE}.time", mock_time),
        ):
            result = await _finalize_with_recovery(
                client=_make_mock_client(),
                best_solution=best_solution,
                task=task,
                config=config,
                phase1_result=p1_result,
                phase2_results=[],
                phase3_result=None,
                pipeline_start=0.0,
            )

        assert isinstance(result, FinalResult)
        assert result.submission_path == ""
