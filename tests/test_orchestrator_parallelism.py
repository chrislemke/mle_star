"""Tests for asyncio parallelism in Phase 2 path dispatch (Task 44).

Validates deep-copy isolation of Phase 1 solutions, per-path working
directory creation, overtime cancellation via ``asyncio.Task.cancel()``,
and error handling for cancelled paths.

These tests are TDD-first — they define the expected behavior for the
parallelism enhancements to ``_dispatch_phase2`` and ``_collect_phase2_results``
in ``orchestrator.py``.

Refs:
    SRS 09b -- Parallelism Requirements (REQ-OR-018 through REQ-OR-023).
    IMPLEMENTATION_PLAN.md Task 44.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

from mle_star.models import (
    CodeBlock,
    CodeBlockCategory,
    DataModality,
    MetricDirection,
    Phase1Result,
    Phase2Result,
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
        "description": "Predict the target.",
        "data_dir": "./input",
        "output_dir": "./final",
    }
    defaults.update(overrides)
    return TaskDescription(**defaults)


def _make_config(**overrides: Any) -> PipelineConfig:
    """Build a valid PipelineConfig with sensible defaults."""
    return PipelineConfig(**overrides)


def _make_solution(**overrides: Any) -> SolutionScript:
    """Build a valid SolutionScript with sensible defaults."""
    defaults: dict[str, Any] = {
        "content": "import pandas as pd\nprint('hello')\n",
        "phase": SolutionPhase.INIT,
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


def _make_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with a dummy file."""
    data_dir = tmp_path / "input"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "train.csv").write_text("id,feature,target\n1,0.5,0\n")
    return data_dir


# ===========================================================================
# REQ-OR-020: Deep copy isolation — each path gets an independent copy
# ===========================================================================


@pytest.mark.unit
class TestDeepCopyIsolation:
    """Each Phase 2 path receives a deep copy of Phase 1 solution (REQ-OR-020)."""

    async def test_each_path_receives_distinct_solution_object(
        self, tmp_path: Path
    ) -> None:
        """Verify each path gets a different object (not the same reference)."""
        from mle_star.orchestrator import _dispatch_phase2

        p1_solution = _make_solution(
            content="original_code", phase=SolutionPhase.MERGED
        )
        p1_result = _make_phase1_result(
            initial_solution=p1_solution, initial_score=0.85
        )
        config = _make_config(num_parallel_solutions=3)
        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))

        received_solutions: list[SolutionScript] = []

        async def _capture_solution(
            _client: Any,
            _task: Any,
            _config: Any,
            initial_solution: SolutionScript,
            initial_score: float,
            session_id: str,
        ) -> Phase2Result:
            received_solutions.append(initial_solution)
            return _make_phase2_result()

        mock_client = AsyncMock()

        with patch(
            f"{_MODULE}.run_phase2_outer_loop",
            side_effect=_capture_solution,
        ):
            await _dispatch_phase2(mock_client, task, config, p1_result)

        assert len(received_solutions) == 3
        # Each solution should be a distinct object (deep copy)
        for i, sol in enumerate(received_solutions):
            for j, other_sol in enumerate(received_solutions):
                if i != j:
                    assert sol is not other_sol, (
                        f"Path {i} and path {j} share the same solution object"
                    )

    async def test_deep_copy_mutation_does_not_affect_other_paths(
        self, tmp_path: Path
    ) -> None:
        """Mutating one path's copy does not affect any other path's copy."""
        from mle_star.orchestrator import _dispatch_phase2

        p1_solution = _make_solution(
            content="original_code", phase=SolutionPhase.MERGED
        )
        p1_result = _make_phase1_result(
            initial_solution=p1_solution, initial_score=0.85
        )
        config = _make_config(num_parallel_solutions=2)
        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))

        received_solutions: list[SolutionScript] = []

        async def _mutate_first_path(
            _client: Any,
            _task: Any,
            _config: Any,
            initial_solution: SolutionScript,
            initial_score: float,
            session_id: str,
        ) -> Phase2Result:
            received_solutions.append(initial_solution)
            if session_id == "path-0":
                # Mutate the solution in path 0
                initial_solution.content = "mutated_by_path_0"
            return _make_phase2_result()

        mock_client = AsyncMock()

        with patch(
            f"{_MODULE}.run_phase2_outer_loop",
            side_effect=_mutate_first_path,
        ):
            await _dispatch_phase2(mock_client, task, config, p1_result)

        # Path 1's solution should be unaffected by path 0's mutation
        assert received_solutions[1].content == "original_code"
        # Original Phase 1 solution should also be unaffected
        assert p1_solution.content == "original_code"

    async def test_deep_copy_preserves_content_and_score(self, tmp_path: Path) -> None:
        """Each deep copy has the same content and score as the original."""
        from mle_star.orchestrator import _dispatch_phase2

        p1_solution = _make_solution(
            content="specific_code_content",
            phase=SolutionPhase.MERGED,
            score=0.77,
        )
        p1_result = _make_phase1_result(
            initial_solution=p1_solution, initial_score=0.77
        )
        config = _make_config(num_parallel_solutions=2)
        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))

        received_solutions: list[SolutionScript] = []

        async def _capture(
            _client: Any,
            _task: Any,
            _config: Any,
            initial_solution: SolutionScript,
            initial_score: float,
            session_id: str,
        ) -> Phase2Result:
            received_solutions.append(initial_solution)
            return _make_phase2_result()

        mock_client = AsyncMock()

        with patch(
            f"{_MODULE}.run_phase2_outer_loop",
            side_effect=_capture,
        ):
            await _dispatch_phase2(mock_client, task, config, p1_result)

        for sol in received_solutions:
            assert sol.content == "specific_code_content"
            assert sol.phase == SolutionPhase.MERGED
            assert sol.score == 0.77


# ===========================================================================
# REQ-OR-020: Per-path working directories
# ===========================================================================


@pytest.mark.unit
class TestPerPathWorkingDirectories:
    """Each path operates in its own working subdirectory (REQ-OR-020)."""

    async def test_work_directories_created_for_each_path(self, tmp_path: Path) -> None:
        """Dispatch creates ./work/path-{i}/ directories for each path."""
        from mle_star.orchestrator import _dispatch_phase2

        config = _make_config(num_parallel_solutions=3)
        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        p1_result = _make_phase1_result()

        mock_client = AsyncMock()
        phase2_mock = AsyncMock(return_value=_make_phase2_result())

        with patch(f"{_MODULE}.run_phase2_outer_loop", phase2_mock):
            await _dispatch_phase2(mock_client, task, config, p1_result)

        # Verify work directories were created
        work_base = data_dir.parent / "work"
        for i in range(3):
            path_dir = work_base / f"path-{i}"
            assert path_dir.exists(), f"work/path-{i}/ not created"
            assert path_dir.is_dir()

    async def test_work_directories_are_unique_per_path(self, tmp_path: Path) -> None:
        """Each path gets its own distinct directory, not shared."""
        from mle_star.orchestrator import _dispatch_phase2

        config = _make_config(num_parallel_solutions=2)
        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        p1_result = _make_phase1_result()

        mock_client = AsyncMock()
        phase2_mock = AsyncMock(return_value=_make_phase2_result())

        with patch(f"{_MODULE}.run_phase2_outer_loop", phase2_mock):
            await _dispatch_phase2(mock_client, task, config, p1_result)

        work_base = data_dir.parent / "work"
        path_0 = work_base / "path-0"
        path_1 = work_base / "path-1"
        assert path_0 != path_1
        assert path_0.exists()
        assert path_1.exists()


# ===========================================================================
# REQ-OR-023: Overtime path cancellation
# ===========================================================================


@pytest.mark.unit
class TestOvertimeCancellation:
    """Overtime paths are cancelled gracefully via asyncio.Task.cancel() (REQ-OR-023)."""

    async def test_overtime_paths_cancelled_when_timeout_exceeded(
        self, tmp_path: Path
    ) -> None:
        """Paths still running after the timeout are cancelled."""
        from mle_star.orchestrator import _dispatch_phase2

        config = _make_config(num_parallel_solutions=2)
        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        p1_result = _make_phase1_result()

        fast_result = _make_phase2_result(best_score=0.88)

        async def _fast_path(*args: Any, **kwargs: Any) -> Phase2Result:
            return fast_result

        async def _slow_path(*args: Any, **kwargs: Any) -> Phase2Result:
            await asyncio.sleep(100)  # Will be cancelled
            return _make_phase2_result()

        call_idx = 0

        async def _mixed_speed(*args: Any, **kwargs: Any) -> Phase2Result:
            nonlocal call_idx
            idx = call_idx
            call_idx += 1
            if idx == 0:
                return fast_result
            await asyncio.sleep(100)
            return _make_phase2_result()

        mock_client = AsyncMock()

        with patch(
            f"{_MODULE}.run_phase2_outer_loop",
            side_effect=_mixed_speed,
        ):
            results = await _dispatch_phase2(
                mock_client,
                task,
                config,
                p1_result,
                phase2_timeout=0.1,
            )

        # Should have 2 results: one success and one CancelledError
        assert len(results) == 2
        successes = [r for r in results if isinstance(r, Phase2Result)]
        cancellations = [r for r in results if isinstance(r, asyncio.CancelledError)]
        assert len(successes) >= 1
        assert len(cancellations) >= 1

    async def test_no_timeout_means_no_cancellation(self, tmp_path: Path) -> None:
        """When phase2_timeout is None, all paths complete normally."""
        from mle_star.orchestrator import _dispatch_phase2

        config = _make_config(num_parallel_solutions=2)
        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        p1_result = _make_phase1_result()

        mock_client = AsyncMock()
        phase2_mock = AsyncMock(return_value=_make_phase2_result())

        with patch(f"{_MODULE}.run_phase2_outer_loop", phase2_mock):
            results = await _dispatch_phase2(
                mock_client,
                task,
                config,
                p1_result,
                phase2_timeout=None,
            )

        assert len(results) == 2
        assert all(isinstance(r, Phase2Result) for r in results)

    async def test_all_paths_complete_before_timeout(self, tmp_path: Path) -> None:
        """When all paths finish before timeout, all succeed."""
        from mle_star.orchestrator import _dispatch_phase2

        config = _make_config(num_parallel_solutions=2)
        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        p1_result = _make_phase1_result()

        mock_client = AsyncMock()
        phase2_mock = AsyncMock(return_value=_make_phase2_result())

        with patch(f"{_MODULE}.run_phase2_outer_loop", phase2_mock):
            results = await _dispatch_phase2(
                mock_client,
                task,
                config,
                p1_result,
                phase2_timeout=10.0,  # Large timeout, all paths complete fast
            )

        assert len(results) == 2
        assert all(isinstance(r, Phase2Result) for r in results)


# ===========================================================================
# REQ-OR-022: Cancelled paths treated as failures
# ===========================================================================


@pytest.mark.unit
class TestCancelledPathsHandledAsFailures:
    """Cancelled paths are treated as failed paths in result collection (REQ-OR-022)."""

    def test_cancelled_error_substitutes_phase1_solution(self) -> None:
        """CancelledError in results triggers Phase 1 fallback via synthetic result."""
        from mle_star.orchestrator import _collect_phase2_results

        p1_solution = _make_solution(content="p1_fallback", phase=SolutionPhase.MERGED)
        p1_result = _make_phase1_result(initial_solution=p1_solution)
        p2_success = _make_phase2_result(best_score=0.90)

        raw_results: list[Phase2Result | BaseException] = [
            asyncio.CancelledError(),
            p2_success,
        ]

        phase2_results, solutions = _collect_phase2_results(raw_results, p1_result)

        # Both paths produce a Phase2Result (synthetic for cancelled + original)
        assert len(phase2_results) == 2
        # First is synthetic (failed), second is original success
        assert phase2_results[0].step_history[0]["failed"] is True
        assert phase2_results[1] is p2_success

        # Two solutions: fallback for cancelled + success
        assert len(solutions) == 2
        assert solutions[0].content == "p1_fallback"
        assert solutions[1] is p2_success.best_solution

    def test_all_cancelled_falls_back_entirely(self) -> None:
        """All CancelledErrors result in all-fallback solutions."""
        from mle_star.orchestrator import _collect_phase2_results

        p1_solution = _make_solution(content="p1_code", phase=SolutionPhase.MERGED)
        p1_result = _make_phase1_result(initial_solution=p1_solution)

        raw_results: list[Phase2Result | BaseException] = [
            asyncio.CancelledError(),
            asyncio.CancelledError(),
        ]

        phase2_results, solutions = _collect_phase2_results(raw_results, p1_result)

        # Both paths produce synthetic Phase2Results
        assert len(phase2_results) == 2
        assert all(r.step_history[0]["failed"] is True for r in phase2_results)
        assert len(solutions) == 2
        assert all(s.content == "p1_code" for s in solutions)

    def test_mixed_errors_and_cancellations(self) -> None:
        """Mix of RuntimeError, CancelledError, and success all handled."""
        from mle_star.orchestrator import _collect_phase2_results

        p1_solution = _make_solution(content="p1_code", phase=SolutionPhase.MERGED)
        p1_result = _make_phase1_result(initial_solution=p1_solution)
        p2_success = _make_phase2_result(best_score=0.90)

        raw_results: list[Phase2Result | BaseException] = [
            RuntimeError("exploded"),
            asyncio.CancelledError(),
            p2_success,
        ]

        phase2_results, solutions = _collect_phase2_results(raw_results, p1_result)

        # All three paths produce Phase2Results (2 synthetic + 1 original)
        assert len(phase2_results) == 3
        assert len(solutions) == 3
        assert solutions[0].content == "p1_code"  # RuntimeError fallback
        assert solutions[1].content == "p1_code"  # CancelledError fallback
        assert solutions[2] is p2_success.best_solution


# ===========================================================================
# REQ-OR-021: Session ID uniqueness
# ===========================================================================


@pytest.mark.unit
class TestSessionIdUniqueness:
    """Each path uses a unique session_id 'path-{i}' (REQ-OR-021)."""

    async def test_session_ids_follow_pattern(self, tmp_path: Path) -> None:
        """Session IDs are 'path-0', 'path-1', ... 'path-{L-1}'."""
        from mle_star.orchestrator import _dispatch_phase2

        config = _make_config(num_parallel_solutions=4)
        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        p1_result = _make_phase1_result()

        received_session_ids: list[str] = []

        async def _capture_session_id(
            _client: Any,
            _task: Any,
            _config: Any,
            _solution: Any,
            _score: float,
            session_id: str,
        ) -> Phase2Result:
            received_session_ids.append(session_id)
            return _make_phase2_result()

        mock_client = AsyncMock()

        with patch(
            f"{_MODULE}.run_phase2_outer_loop",
            side_effect=_capture_session_id,
        ):
            await _dispatch_phase2(mock_client, task, config, p1_result)

        assert set(received_session_ids) == {"path-0", "path-1", "path-2", "path-3"}
        # All unique
        assert len(received_session_ids) == len(set(received_session_ids))


# ===========================================================================
# REQ-OR-020: initial_score passed to each path
# ===========================================================================


@pytest.mark.unit
class TestInitialScorePassedToEachPath:
    """Each Phase 2 path receives Phase1Result.initial_score (REQ-OR-020)."""

    async def test_initial_score_passed_to_all_paths(self, tmp_path: Path) -> None:
        """All paths receive the Phase 1 initial_score value."""
        from mle_star.orchestrator import _dispatch_phase2

        config = _make_config(num_parallel_solutions=3)
        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        p1_result = _make_phase1_result(initial_score=0.42)

        received_scores: list[float] = []

        async def _capture_score(
            _client: Any,
            _task: Any,
            _config: Any,
            _solution: Any,
            initial_score: float,
            session_id: str,
        ) -> Phase2Result:
            received_scores.append(initial_score)
            return _make_phase2_result()

        mock_client = AsyncMock()

        with patch(
            f"{_MODULE}.run_phase2_outer_loop",
            side_effect=_capture_score,
        ):
            await _dispatch_phase2(mock_client, task, config, p1_result)

        assert len(received_scores) == 3
        assert all(s == 0.42 for s in received_scores)


# ===========================================================================
# Integration: Pipeline with deep copy and working dirs
# ===========================================================================


@pytest.mark.unit
class TestPipelineParallelismIntegration:
    """Integration test: full pipeline with deep copy and working directory creation."""

    async def test_pipeline_creates_work_directories(self, tmp_path: Path) -> None:
        """Running the full pipeline creates per-path working directories."""
        from mle_star.models import FinalResult
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(num_parallel_solutions=2)

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()

        p1_result = _make_phase1_result()
        p2_result = _make_phase2_result()

        from mle_star.models import Phase3Result

        p3_sol = _make_solution(phase=SolutionPhase.ENSEMBLE)
        p3_result_obj = Phase3Result(
            input_solutions=[p3_sol, p3_sol],
            ensemble_plans=["plan"],
            ensemble_scores=[0.92],
            best_ensemble=p3_sol,
            best_ensemble_score=0.92,
        )

        fr = FinalResult(
            task=task,
            config=config,
            phase1=p1_result,
            phase2_results=[p2_result],
            phase3=p3_result_obj,
            final_solution=_make_solution(phase=SolutionPhase.FINAL),
            submission_path="/output/submission.csv",
            total_duration_seconds=10.0,
            total_cost_usd=None,
        )

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=mock_client),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False},
            ),
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
                return_value=p3_result_obj,
                create=True,
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=fr,
            ),
        ):
            result = await run_pipeline(task, config)

        assert isinstance(result, FinalResult)

        # Verify per-path working directories exist
        work_base = data_dir.parent / "work"
        for i in range(2):
            path_dir = work_base / f"path-{i}"
            assert path_dir.exists(), f"work/path-{i}/ not created"
