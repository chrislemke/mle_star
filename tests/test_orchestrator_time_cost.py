"""Tests for time budgeting and graceful shutdown (Task 45).

Validates ``_compute_phase_budgets``, deadline enforcement,
proportional time allocation, per-path Phase 2 budgets,
and graceful shutdown on timeout.

These tests are TDD-first — they define the expected behavior for
REQ-OR-024 through REQ-OR-030.

Refs:
    SRS 09c -- Orchestrator Budgets & Hooks.
    IMPLEMENTATION_PLAN.md Task 45.
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
    PhaseTimeBudget,
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
    mock_client = AsyncMock()
    return mock_client


# ===========================================================================
# REQ-OR-024: _compute_phase_budgets
# ===========================================================================


@pytest.mark.unit
class TestComputePhaseBudgets:
    """_compute_phase_budgets computes per-phase time budgets (REQ-OR-025)."""

    def test_default_proportions(self) -> None:
        """Default budget splits remaining time as 65/15/10 (normalized to 90)."""
        from mle_star.orchestrator import _compute_phase_budgets

        config = _make_config()
        budgets = _compute_phase_budgets(config, remaining_seconds=900.0)

        # After Phase 1, remaining phases get: 65/(65+15+10), 15/(65+15+10), 10/(65+15+10)
        # = 72.2%, 16.7%, 11.1%
        assert budgets["phase2"] == pytest.approx(650.0, rel=0.01)
        assert budgets["phase3"] == pytest.approx(150.0, rel=0.01)
        assert budgets["finalization"] == pytest.approx(100.0, rel=0.01)

    def test_custom_phase_time_budget(self) -> None:
        """Custom PhaseTimeBudget proportions are respected."""
        from mle_star.orchestrator import _compute_phase_budgets

        custom_budget = PhaseTimeBudget(
            phase1_pct=20.0,
            phase2_pct=50.0,
            phase3_pct=20.0,
            finalization_pct=10.0,
        )
        config = _make_config(phase_time_budget=custom_budget)
        budgets = _compute_phase_budgets(config, remaining_seconds=800.0)

        # Remaining proportions: 50/(50+20+10) = 62.5%, 20/80 = 25%, 10/80 = 12.5%
        assert budgets["phase2"] == pytest.approx(500.0, rel=0.01)
        assert budgets["phase3"] == pytest.approx(200.0, rel=0.01)
        assert budgets["finalization"] == pytest.approx(100.0, rel=0.01)

    def test_budgets_sum_to_remaining(self) -> None:
        """All phase budgets sum to the total remaining time."""
        from mle_star.orchestrator import _compute_phase_budgets

        config = _make_config()
        remaining = 1000.0
        budgets = _compute_phase_budgets(config, remaining_seconds=remaining)

        total = budgets["phase2"] + budgets["phase3"] + budgets["finalization"]
        assert total == pytest.approx(remaining, rel=1e-6)

    def test_zero_remaining_seconds(self) -> None:
        """Zero remaining time produces zero budgets for all phases."""
        from mle_star.orchestrator import _compute_phase_budgets

        config = _make_config()
        budgets = _compute_phase_budgets(config, remaining_seconds=0.0)

        assert budgets["phase2"] == 0.0
        assert budgets["phase3"] == 0.0
        assert budgets["finalization"] == 0.0

    @given(remaining=st.floats(min_value=0.0, max_value=100000.0))
    @settings(max_examples=20, deadline=5000)
    def test_budgets_always_sum_to_remaining(self, remaining: float) -> None:
        """For any remaining time, budgets always sum to that total."""
        from mle_star.orchestrator import _compute_phase_budgets

        config = _make_config()
        budgets = _compute_phase_budgets(config, remaining_seconds=remaining)

        total = budgets["phase2"] + budgets["phase3"] + budgets["finalization"]
        assert total == pytest.approx(remaining, rel=1e-6)


# ===========================================================================
# REQ-OR-026: Per-path Phase 2 budget
# ===========================================================================


@pytest.mark.unit
class TestPhase2PerPathBudget:
    """Per-path Phase 2 budget = phase2_budget / L (REQ-OR-026)."""

    def test_per_path_budget_with_l2(self) -> None:
        """With L=2, each path gets half the Phase 2 budget."""
        from mle_star.orchestrator import _compute_phase_budgets

        config = _make_config(num_parallel_solutions=2)
        budgets = _compute_phase_budgets(config, remaining_seconds=900.0)

        # Phase 2 total budget is 650.0 (72.2% of 900), per-path = 325.0
        assert "phase2_per_path" in budgets
        assert budgets["phase2_per_path"] == pytest.approx(
            budgets["phase2"] / 2, rel=1e-6
        )

    def test_per_path_budget_with_l3(self) -> None:
        """With L=3, each path gets one-third the Phase 2 budget."""
        from mle_star.orchestrator import _compute_phase_budgets

        config = _make_config(num_parallel_solutions=3)
        budgets = _compute_phase_budgets(config, remaining_seconds=900.0)

        assert budgets["phase2_per_path"] == pytest.approx(
            budgets["phase2"] / 3, rel=1e-6
        )

    def test_per_path_budget_with_l1(self) -> None:
        """With L=1, per-path budget equals the full Phase 2 budget."""
        from mle_star.orchestrator import _compute_phase_budgets

        config = _make_config(num_parallel_solutions=1)
        budgets = _compute_phase_budgets(config, remaining_seconds=900.0)

        assert budgets["phase2_per_path"] == pytest.approx(budgets["phase2"], rel=1e-6)


# ===========================================================================
# REQ-OR-024: Deadline computed at pipeline start
# ===========================================================================


@pytest.mark.unit
class TestDeadlineComputation:
    """Pipeline computes deadline at start using time.monotonic() (REQ-OR-024)."""

    async def test_deadline_checked_before_phase2(self, tmp_path: Path) -> None:
        """If deadline expires before Phase 2, pipeline skips to finalization."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(time_limit_seconds=100)

        mock_client = _make_mock_client()
        p1_result = _make_phase1_result()
        fr = _make_final_result(task=task, config=config)

        phase2_mock = AsyncMock(return_value=_make_phase2_result())

        # Mock time.monotonic() to simulate deadline exceeded after Phase 1.
        # Calls 1-5 (through Phase 1 completion): return 0.0 (well within deadline).
        # Call 6+ (_execute_post_phase1 deadline check): return 200.0 (past deadline=100).
        mock_time = Mock()
        mock_time.monotonic = Mock(side_effect=[0.0] * 5 + [200.0] * 15)

        with (
            patch(f"{_MODULE}._create_client", return_value=mock_client),
            patch(f"{_MODULE}.check_claude_cli_version"),
            patch(f"{_MODULE}.configure_logging"),
            patch(f"{_MODULE}.setup_working_directory"),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=p1_result,
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", phase2_mock),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=fr,
            ),
            patch(f"{_MODULE}.time", mock_time),
        ):
            result = await run_pipeline(task, config)

        # Phase 2 should have been skipped (deadline exceeded after Phase 1)
        phase2_mock.assert_not_called()
        assert isinstance(result, FinalResult)

    async def test_phase1_timeout_raises_pipeline_timeout_error(
        self, tmp_path: Path
    ) -> None:
        """If Phase 1 does not complete before deadline, PipelineTimeoutError raised."""
        from mle_star.orchestrator import PipelineTimeoutError, run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(time_limit_seconds=1)

        mock_client = _make_mock_client()

        async def _very_slow_phase1(*args: Any, **kwargs: Any) -> Phase1Result:
            await asyncio.sleep(10.0)
            return _make_phase1_result()

        with (
            patch(f"{_MODULE}._create_client", return_value=mock_client),
            patch(f"{_MODULE}.check_claude_cli_version"),
            patch(f"{_MODULE}.configure_logging"),
            patch(f"{_MODULE}.setup_working_directory"),
            patch(f"{_MODULE}.run_phase1", side_effect=_very_slow_phase1),
            pytest.raises(PipelineTimeoutError),
        ):
            await run_pipeline(task, config)


# ===========================================================================
# REQ-OR-030: Graceful shutdown
# ===========================================================================


@pytest.mark.unit
class TestGracefulShutdown:
    """Graceful shutdown on timeout or budget exceeded (REQ-OR-030)."""

    async def test_timeout_during_phase2_skips_to_finalization(
        self, tmp_path: Path
    ) -> None:
        """If deadline expires during Phase 2, skip to finalization with Phase 1 solution."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(time_limit_seconds=2, num_parallel_solutions=1)

        mock_client = _make_mock_client()
        p1_solution = _make_solution(
            content="p1_code_for_fallback", phase=SolutionPhase.MERGED
        )
        p1_result = _make_phase1_result(initial_solution=p1_solution)
        fr = _make_final_result(task=task, config=config)

        finalization_mock = AsyncMock(return_value=fr)

        async def _slow_phase2(*args: Any, **kwargs: Any) -> Phase2Result:
            await asyncio.sleep(10.0)
            return _make_phase2_result()

        with (
            patch(f"{_MODULE}._create_client", return_value=mock_client),
            patch(f"{_MODULE}.check_claude_cli_version"),
            patch(f"{_MODULE}.configure_logging"),
            patch(f"{_MODULE}.setup_working_directory"),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=p1_result,
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", side_effect=_slow_phase2),
            patch(f"{_MODULE}.run_finalization", finalization_mock),
        ):
            result = await run_pipeline(task, config)

        assert isinstance(result, FinalResult)
        # Finalization should have been called with Phase 1 solution as fallback
        fin_args = finalization_mock.call_args
        solution_arg = (
            fin_args[0][1] if len(fin_args[0]) > 1 else fin_args[1].get("solution")
        )
        assert solution_arg.content == "p1_code_for_fallback"

    async def test_timeout_during_phase3_skips_to_finalization(
        self, tmp_path: Path
    ) -> None:
        """If deadline expires during Phase 3, skip to finalization with Phase 2 best."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(time_limit_seconds=2, num_parallel_solutions=2)

        mock_client = _make_mock_client()
        p2_solution = _make_solution(
            content="p2_best_code", phase=SolutionPhase.REFINED
        )
        p2_result = _make_phase2_result(best_solution=p2_solution, best_score=0.90)
        fr = _make_final_result(task=task, config=config)
        finalization_mock = AsyncMock(return_value=fr)

        async def _slow_phase3(*args: Any, **kwargs: Any) -> Phase3Result:
            await asyncio.sleep(10.0)
            return _make_phase3_result()

        with (
            patch(f"{_MODULE}._create_client", return_value=mock_client),
            patch(f"{_MODULE}.check_claude_cli_version"),
            patch(f"{_MODULE}.configure_logging"),
            patch(f"{_MODULE}.setup_working_directory"),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=_make_phase1_result(),
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=p2_result,
            ),
            patch(f"{_MODULE}.run_phase3", side_effect=_slow_phase3, create=True),
            patch(f"{_MODULE}.run_finalization", finalization_mock),
        ):
            result = await run_pipeline(task, config)

        assert isinstance(result, FinalResult)
        # Finalization called — should have received one of the Phase 2 solutions
        finalization_mock.assert_awaited_once()

    async def test_pipeline_timeout_error_has_diagnostics(self, tmp_path: Path) -> None:
        """PipelineTimeoutError includes elapsed_time in diagnostics."""
        from mle_star.orchestrator import PipelineTimeoutError, run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(time_limit_seconds=1)

        mock_client = _make_mock_client()

        async def _very_slow_phase1(*args: Any, **kwargs: Any) -> Phase1Result:
            await asyncio.sleep(10.0)
            return _make_phase1_result()

        with (
            patch(f"{_MODULE}._create_client", return_value=mock_client),
            patch(f"{_MODULE}.check_claude_cli_version"),
            patch(f"{_MODULE}.configure_logging"),
            patch(f"{_MODULE}.setup_working_directory"),
            patch(f"{_MODULE}.run_phase1", side_effect=_very_slow_phase1),
            pytest.raises(PipelineTimeoutError) as exc_info,
        ):
            await run_pipeline(task, config)

        assert "elapsed_time" in exc_info.value.diagnostics


# ===========================================================================
# REQ-OR-025: Phase time budgets passed to Phase 2
# ===========================================================================


@pytest.mark.unit
class TestPhase2ReceivesTimeBudget:
    """Phase 2 dispatch receives computed time budget (REQ-OR-025, REQ-OR-026)."""

    async def test_phase2_timeout_passed_to_dispatch(self, tmp_path: Path) -> None:
        """_dispatch_phase2 receives the computed phase2 budget as timeout."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(
            num_parallel_solutions=1,
            time_limit_seconds=86400,
        )

        mock_client = _make_mock_client()
        p1_result = _make_phase1_result()
        fr = _make_final_result(task=task, config=config)

        captured_timeout: list[float | None] = []

        async def _capture_dispatch(
            *args: Any, **kwargs: Any
        ) -> list[Phase2Result | BaseException]:
            captured_timeout.append(kwargs.get("phase2_timeout"))
            return [_make_phase2_result()]

        with (
            patch(f"{_MODULE}._create_client", return_value=mock_client),
            patch(f"{_MODULE}.check_claude_cli_version"),
            patch(f"{_MODULE}.configure_logging"),
            patch(f"{_MODULE}.setup_working_directory"),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=p1_result,
            ),
            patch(f"{_MODULE}._dispatch_phase2", side_effect=_capture_dispatch),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=fr,
            ),
        ):
            await run_pipeline(task, config)

        # A timeout should have been passed (not None)
        assert len(captured_timeout) == 1
        assert captured_timeout[0] is not None
        assert captured_timeout[0] > 0


# ===========================================================================
# REQ-OR-030: Phase 1 solution used as fallback on Phase 2 timeout
# ===========================================================================


@pytest.mark.unit
class TestPhase1FallbackOnTimeout:
    """Phase 1 solution is the fallback when later phases time out (REQ-OR-030)."""

    async def test_phase1_solution_used_when_phase2_times_out(
        self, tmp_path: Path
    ) -> None:
        """Phase 1 initial_solution passed to finalization when Phase 2 times out."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(time_limit_seconds=2, num_parallel_solutions=1)

        mock_client = _make_mock_client()
        p1_solution = _make_solution(
            content="p1_fallback_solution", phase=SolutionPhase.MERGED
        )
        p1_result = _make_phase1_result(initial_solution=p1_solution)
        fr = _make_final_result(task=task, config=config)
        finalization_mock = AsyncMock(return_value=fr)

        async def _slow_phase2(*args: Any, **kwargs: Any) -> Phase2Result:
            await asyncio.sleep(10.0)
            return _make_phase2_result()

        with (
            patch(f"{_MODULE}._create_client", return_value=mock_client),
            patch(f"{_MODULE}.check_claude_cli_version"),
            patch(f"{_MODULE}.configure_logging"),
            patch(f"{_MODULE}.setup_working_directory"),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=p1_result,
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", side_effect=_slow_phase2),
            patch(f"{_MODULE}.run_finalization", finalization_mock),
        ):
            result = await run_pipeline(task, config)

        assert isinstance(result, FinalResult)
        # Verify Phase 1 solution was used for finalization
        fin_args = finalization_mock.call_args
        solution_arg = (
            fin_args[0][1] if len(fin_args[0]) > 1 else fin_args[1].get("solution")
        )
        assert solution_arg.content == "p1_fallback_solution"


# ===========================================================================
# Hypothesis: Property-based tests
# ===========================================================================


@pytest.mark.unit
class TestTimeCostProperties:
    """Property-based tests for time and cost budget computation."""

    @given(
        remaining=st.floats(min_value=1.0, max_value=100000.0),
        l_paths=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20, deadline=5000)
    def test_per_path_budget_equals_total_divided_by_l(
        self, remaining: float, l_paths: int
    ) -> None:
        """Per-path budget always equals phase2_budget / L."""
        from mle_star.orchestrator import _compute_phase_budgets

        config = _make_config(num_parallel_solutions=l_paths)
        budgets = _compute_phase_budgets(config, remaining_seconds=remaining)

        expected = budgets["phase2"] / l_paths
        assert budgets["phase2_per_path"] == pytest.approx(expected, rel=1e-6)



# ===========================================================================
# Integration: Full pipeline with time limit
# ===========================================================================


@pytest.mark.unit
class TestPipelineTimeLimitIntegration:
    """Integration tests for pipeline with time_limit_seconds."""

    async def test_normal_pipeline_completes_with_time_limit(
        self, tmp_path: Path
    ) -> None:
        """A fast pipeline completes normally even with a time limit."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(
            time_limit_seconds=86400,
            num_parallel_solutions=1,
        )

        mock_client = _make_mock_client()
        p1_result = _make_phase1_result()
        p2_result = _make_phase2_result()
        fr = _make_final_result(task=task, config=config)

        with (
            patch(f"{_MODULE}._create_client", return_value=mock_client),
            patch(f"{_MODULE}.check_claude_cli_version"),
            patch(f"{_MODULE}.configure_logging"),
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

    async def test_phase_boundary_logging_on_timeout(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Timeout produces appropriate log messages about skipped phases."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(time_limit_seconds=2, num_parallel_solutions=1)

        mock_client = _make_mock_client()
        p1_result = _make_phase1_result()
        fr = _make_final_result(task=task, config=config)

        async def _slow_phase2(*args: Any, **kwargs: Any) -> Phase2Result:
            await asyncio.sleep(10.0)
            return _make_phase2_result()

        with (
            patch(f"{_MODULE}._create_client", return_value=mock_client),
            patch(f"{_MODULE}.check_claude_cli_version"),
            patch(f"{_MODULE}.configure_logging"),
            patch(f"{_MODULE}.setup_working_directory"),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=p1_result,
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", side_effect=_slow_phase2),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=fr,
            ),
            caplog.at_level(logging.WARNING),
        ):
            await run_pipeline(task, config)

        # Should log about timeout/skipping
        log_text = " ".join(r.message for r in caplog.records)
        assert (
            "deadline" in log_text.lower()
            or "timeout" in log_text.lower()
            or "skip" in log_text.lower()
        )
