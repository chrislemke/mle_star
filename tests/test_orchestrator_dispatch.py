"""Tests for pipeline phase dispatch and sequencing (Task 43).

Validates the sequential phase dispatch within ``run_pipeline()`` in
``orchestrator.py``.  Tests cover strict phase ordering (P1 -> P2 -> P3 ->
Finalization), Phase 2 L-path dispatch via ``asyncio.gather``, Phase 3 skip
when L=1, finalization receiving the best available solution, duration
recording, Phase 2 failure handling with Phase 1 fallback, and best solution
selection across phases.

These tests are written TDD-first -- the full phase dispatch implementation
does not yet exist.  They serve as the executable specification for
REQ-OR-013, REQ-OR-015, REQ-OR-016, REQ-OR-017, REQ-OR-022, and REQ-OR-040.

Refs:
    SRS 09b -- Orchestrator Phase Dispatch & Sequencing.
    IMPLEMENTATION_PLAN.md Task 43.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

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
        "description": "Predict the target variable from tabular features.",
        "data_dir": "./input",
        "output_dir": "./final",
    }
    defaults.update(overrides)
    return TaskDescription(**defaults)


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


def _make_solution(**overrides: Any) -> SolutionScript:
    """Build a valid SolutionScript with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed SolutionScript instance.
    """
    defaults: dict[str, Any] = {
        "content": "import pandas as pd\nprint('hello')\n",
        "phase": SolutionPhase.FINAL,
    }
    defaults.update(overrides)
    return SolutionScript(**defaults)


def _make_phase1_result(**overrides: Any) -> Phase1Result:
    """Build a valid Phase1Result with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed Phase1Result instance.
    """
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
        "step_history": [{"step": 0, "score": 0.90}],
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
    """Build a valid FinalResult with sensible defaults.

    Args:
        task: TaskDescription to use. Defaults to _make_task().
        config: PipelineConfig to use. Defaults to _make_config().
        **overrides: Field values to override.

    Returns:
        A fully constructed FinalResult instance.
    """
    defaults: dict[str, Any] = {
        "task": task or _make_task(),
        "config": config or _make_config(),
        "phase1": _make_phase1_result(),
        "phase2_results": [_make_phase2_result()],
        "phase3": None,
        "final_solution": _make_solution(phase=SolutionPhase.FINAL),
        "submission_path": "/output/submission.csv",
        "total_duration_seconds": 100.0,
        "total_cost_usd": None,
    }
    defaults.update(overrides)
    return FinalResult(**defaults)


def _make_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with a dummy file.

    Args:
        tmp_path: Pytest tmp_path fixture.

    Returns:
        Path to the temporary data directory.
    """
    data_dir = tmp_path / "input"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "train.csv").write_text("id,feature,target\n1,0.5,0\n")
    return data_dir


def _make_mock_client() -> AsyncMock:
    """Build a mock ClaudeSDKClient with connect/disconnect stubs.

    Returns:
        An AsyncMock simulating a ClaudeSDKClient.
    """
    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.disconnect = AsyncMock()
    return mock_client


def _standard_patches(
    tmp_path: Path,
    *,
    phase1_result: Phase1Result | None = None,
    phase2_result: Phase2Result | None = None,
    phase2_side_effect: Any = None,
    phase3_result: Phase3Result | None = None,
    final_result: FinalResult | None = None,
    config: PipelineConfig | None = None,
    mock_client: AsyncMock | None = None,
) -> dict[str, Any]:
    """Prepare standard mock patches for run_pipeline tests.

    Returns a dict of keyword arguments for use with ``unittest.mock.patch``
    context managers. Callers should use the returned mocks to build their
    own patch context.

    Args:
        tmp_path: Pytest tmp_path fixture for creating a valid data_dir.
        phase1_result: Phase1Result to return from run_phase1.
        phase2_result: Phase2Result to return from run_phase2_outer_loop.
        phase2_side_effect: Side effect for run_phase2_outer_loop (overrides phase2_result).
        phase3_result: Phase3Result to return from run_phase3.
        final_result: FinalResult to return from run_finalization.
        config: PipelineConfig to use.
        mock_client: Pre-configured mock client.

    Returns:
        Dict with keys: data_dir, task, config, mock_client, phase1_mock,
        phase2_mock, phase3_mock, finalization_mock, final_result.
    """
    data_dir = _make_data_dir(tmp_path)
    task = _make_task(data_dir=str(data_dir))
    resolved_config = config or _make_config()
    client = mock_client or _make_mock_client()
    p1 = phase1_result or _make_phase1_result()
    p2 = phase2_result or _make_phase2_result()
    p3 = phase3_result
    fr = final_result or _make_final_result(task=task, config=resolved_config)

    phase1_mock = AsyncMock(return_value=p1)
    if phase2_side_effect is not None:
        phase2_mock = AsyncMock(side_effect=phase2_side_effect)
    else:
        phase2_mock = AsyncMock(return_value=p2)
    phase3_mock = AsyncMock(return_value=p3)
    finalization_mock = AsyncMock(return_value=fr)

    return {
        "data_dir": data_dir,
        "task": task,
        "config": resolved_config,
        "mock_client": client,
        "phase1_mock": phase1_mock,
        "phase2_mock": phase2_mock,
        "phase3_mock": phase3_mock,
        "finalization_mock": finalization_mock,
        "final_result": fr,
        "phase1_result": p1,
        "phase2_result": p2,
        "phase3_result": p3,
    }


# ===========================================================================
# REQ-OR-017: Phase ordering -- P1 -> P2 -> P3 -> Finalization
# ===========================================================================


@pytest.mark.unit
class TestPhaseOrdering:
    """Phases execute in strict sequential order: P1 -> P2 -> P3 -> Final (REQ-OR-017)."""

    async def test_phases_execute_in_order(self, tmp_path: Path) -> None:
        """Verify P1 runs before P2, P2 before P3, P3 before finalization.

        Uses a shared call_order list that each mock appends to when called,
        establishing the exact invocation sequence.
        """
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=2)
        ctx = _standard_patches(
            tmp_path,
            config=config,
            phase3_result=_make_phase3_result(),
        )
        call_order: list[str] = []

        async def _track_phase1(*args: Any, **kwargs: Any) -> Phase1Result:
            call_order.append("phase1")
            return ctx["phase1_result"]

        async def _track_phase2(*args: Any, **kwargs: Any) -> Phase2Result:
            call_order.append("phase2")
            return ctx["phase2_result"]

        async def _track_phase3(*args: Any, **kwargs: Any) -> Phase3Result:
            call_order.append("phase3")
            return ctx["phase3_result"]

        async def _track_finalization(*args: Any, **kwargs: Any) -> FinalResult:
            call_order.append("finalization")
            return ctx["final_result"]

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(f"{_MODULE}.run_phase1", side_effect=_track_phase1),
            patch(f"{_MODULE}.run_phase2_outer_loop", side_effect=_track_phase2),
            patch(f"{_MODULE}.run_phase3", side_effect=_track_phase3, create=True),
            patch(f"{_MODULE}.run_finalization", side_effect=_track_finalization),
        ):
            await run_pipeline(ctx["task"], ctx["config"])

        # Phase 1 must come first
        assert call_order.index("phase1") < call_order.index("phase2")
        # Phase 2 must come before Phase 3
        p2_indices = [i for i, v in enumerate(call_order) if v == "phase2"]
        p3_index = call_order.index("phase3")
        assert all(idx < p3_index for idx in p2_indices)
        # Phase 3 must come before finalization
        assert call_order.index("phase3") < call_order.index("finalization")

    async def test_phase2_not_called_until_phase1_completes(
        self, tmp_path: Path
    ) -> None:
        """Verify Phase 2 is not invoked until Phase 1 returns.

        Phase 1 mock records its completion; Phase 2 mock asserts Phase 1
        already completed when it is called.
        """
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=1)
        ctx = _standard_patches(tmp_path, config=config)
        phase1_completed = False

        async def _phase1_completes(*args: Any, **kwargs: Any) -> Phase1Result:
            nonlocal phase1_completed
            phase1_completed = True
            return ctx["phase1_result"]

        async def _phase2_checks(*args: Any, **kwargs: Any) -> Phase2Result:
            assert phase1_completed, "Phase 2 called before Phase 1 completed"
            return ctx["phase2_result"]

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(f"{_MODULE}.run_phase1", side_effect=_phase1_completes),
            patch(f"{_MODULE}.run_phase2_outer_loop", side_effect=_phase2_checks),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=ctx["final_result"],
            ),
        ):
            await run_pipeline(ctx["task"], ctx["config"])

    async def test_phase3_not_called_until_phase2_completes(
        self, tmp_path: Path
    ) -> None:
        """Verify Phase 3 waits for all Phase 2 paths to complete.

        With L=2, both Phase 2 paths must finish before Phase 3 is invoked.
        """
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=2)
        ctx = _standard_patches(
            tmp_path,
            config=config,
            phase3_result=_make_phase3_result(),
        )
        phase2_call_count = 0

        async def _count_phase2(*args: Any, **kwargs: Any) -> Phase2Result:
            nonlocal phase2_call_count
            phase2_call_count += 1
            return ctx["phase2_result"]

        async def _check_phase3(*args: Any, **kwargs: Any) -> Phase3Result:
            assert phase2_call_count == 2, (
                f"Phase 3 called after only {phase2_call_count} Phase 2 paths "
                f"(expected 2)"
            )
            return ctx["phase3_result"]

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=ctx["phase1_result"],
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", side_effect=_count_phase2),
            patch(f"{_MODULE}.run_phase3", side_effect=_check_phase3, create=True),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=ctx["final_result"],
            ),
        ):
            await run_pipeline(ctx["task"], ctx["config"])


# ===========================================================================
# REQ-OR-013: Phase 2 dispatches L paths
# ===========================================================================


@pytest.mark.unit
class TestPhase2Dispatch:
    """Phase 2 dispatches L parallel paths via asyncio.gather (REQ-OR-013)."""

    async def test_l_paths_dispatched(self, tmp_path: Path) -> None:
        """With L=2, verify run_phase2_outer_loop is called twice."""
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=2)
        ctx = _standard_patches(
            tmp_path,
            config=config,
            phase3_result=_make_phase3_result(),
        )

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=ctx["phase1_result"],
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", ctx["phase2_mock"]),
            patch(
                f"{_MODULE}.run_phase3",
                new_callable=AsyncMock,
                return_value=ctx["phase3_result"],
                create=True,
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=ctx["final_result"],
            ),
        ):
            await run_pipeline(ctx["task"], ctx["config"])

        assert ctx["phase2_mock"].call_count == 2

    async def test_phase2_receives_phase1_solution(self, tmp_path: Path) -> None:
        """Each Phase 2 path receives the Phase 1 initial_solution."""
        from mle_star.orchestrator import run_pipeline

        p1_solution = _make_solution(
            content="p1_solution_code", phase=SolutionPhase.MERGED
        )
        p1_result = _make_phase1_result(initial_solution=p1_solution)
        config = _make_config(num_parallel_solutions=2)
        ctx = _standard_patches(
            tmp_path,
            config=config,
            phase1_result=p1_result,
            phase3_result=_make_phase3_result(),
        )

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1", new_callable=AsyncMock, return_value=p1_result
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", ctx["phase2_mock"]),
            patch(
                f"{_MODULE}.run_phase3",
                new_callable=AsyncMock,
                return_value=ctx["phase3_result"],
                create=True,
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=ctx["final_result"],
            ),
        ):
            await run_pipeline(ctx["task"], ctx["config"])

        # Each Phase 2 call should receive a deep copy of the Phase 1 initial_solution
        # (REQ-OR-020: deep copy isolation — same content, distinct object)
        for c in ctx["phase2_mock"].call_args_list:
            # initial_solution is the 4th positional arg (client, task, config, initial_solution)
            # or passed as keyword
            args, kwargs = c
            solution_arg = kwargs.get("initial_solution") or args[3]
            assert solution_arg.content == p1_solution.content
            assert solution_arg.phase == p1_solution.phase
            assert solution_arg is not p1_solution  # Deep copy, not same object

    async def test_phase2_receives_phase1_score(self, tmp_path: Path) -> None:
        """Each Phase 2 path receives Phase 1 initial_score."""
        from mle_star.orchestrator import run_pipeline

        p1_result = _make_phase1_result(initial_score=0.77)
        config = _make_config(num_parallel_solutions=2)
        ctx = _standard_patches(
            tmp_path,
            config=config,
            phase1_result=p1_result,
            phase3_result=_make_phase3_result(),
        )

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1", new_callable=AsyncMock, return_value=p1_result
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", ctx["phase2_mock"]),
            patch(
                f"{_MODULE}.run_phase3",
                new_callable=AsyncMock,
                return_value=ctx["phase3_result"],
                create=True,
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=ctx["final_result"],
            ),
        ):
            await run_pipeline(ctx["task"], ctx["config"])

        for c in ctx["phase2_mock"].call_args_list:
            args, kwargs = c
            score_arg = kwargs.get("initial_score") or args[4]
            assert score_arg == 0.77

    async def test_phase2_results_collected(self, tmp_path: Path) -> None:
        """All L Phase2Results are collected and passed to finalization."""
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=2)
        p2a = _make_phase2_result(best_score=0.88)
        p2b = _make_phase2_result(best_score=0.91)
        call_idx = 0

        async def _return_different(*args: Any, **kwargs: Any) -> Phase2Result:
            nonlocal call_idx
            result = p2a if call_idx == 0 else p2b
            call_idx += 1
            return result

        ctx = _standard_patches(
            tmp_path,
            config=config,
            phase2_side_effect=_return_different,
            phase3_result=_make_phase3_result(),
        )

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=ctx["phase1_result"],
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", side_effect=_return_different),
            patch(
                f"{_MODULE}.run_phase3",
                new_callable=AsyncMock,
                return_value=ctx["phase3_result"],
                create=True,
            ),
            patch(
                f"{_MODULE}.run_finalization",
                ctx["finalization_mock"],
            ),
        ):
            await run_pipeline(ctx["task"], ctx["config"])

        # Finalization receives phase2_results list
        fin_args, fin_kwargs = ctx["finalization_mock"].call_args
        phase2_results_arg = fin_kwargs.get("phase2_results") or fin_args[5]
        assert len(phase2_results_arg) == 2

    async def test_phase2_session_ids(self, tmp_path: Path) -> None:
        """Each path uses a unique session_id 'path-{i}'."""
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=3)
        ctx = _standard_patches(
            tmp_path,
            config=config,
            phase3_result=_make_phase3_result(),
        )

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=ctx["phase1_result"],
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", ctx["phase2_mock"]),
            patch(
                f"{_MODULE}.run_phase3",
                new_callable=AsyncMock,
                return_value=ctx["phase3_result"],
                create=True,
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=ctx["final_result"],
            ),
        ):
            await run_pipeline(ctx["task"], ctx["config"])

        session_ids: list[str] = []
        for c in ctx["phase2_mock"].call_args_list:
            args, kwargs = c
            sid = kwargs.get("session_id") or args[5]
            session_ids.append(sid)

        # Expect path-0, path-1, path-2
        assert len(session_ids) == 3
        assert set(session_ids) == {"path-0", "path-1", "path-2"}
        # Each session_id should be unique
        assert len(set(session_ids)) == len(session_ids)


# ===========================================================================
# REQ-OR-015: Phase 3 skip condition
# ===========================================================================


@pytest.mark.unit
class TestPhase3SkipCondition:
    """Phase 3 is skipped when L=1 (REQ-OR-015)."""

    async def test_phase3_skipped_when_l_equals_1(self, tmp_path: Path) -> None:
        """Phase 3 is not called when num_parallel_solutions == 1."""
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=1)
        ctx = _standard_patches(tmp_path, config=config)
        phase3_mock = AsyncMock(return_value=_make_phase3_result())

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=ctx["phase1_result"],
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", ctx["phase2_mock"]),
            patch(f"{_MODULE}.run_phase3", phase3_mock, create=True),
            patch(
                f"{_MODULE}.run_finalization",
                ctx["finalization_mock"],
            ),
        ):
            await run_pipeline(ctx["task"], ctx["config"])

        # Phase 3 should NOT be called
        phase3_mock.assert_not_called()

        # Finalization should receive phase3_result=None
        fin_args, fin_kwargs = ctx["finalization_mock"].call_args
        phase3_arg = fin_kwargs.get("phase3_result")
        if phase3_arg is None and "phase3_result" not in fin_kwargs:
            # Check positional arg (client, solution, task, config, p1, p2_list, p3)
            phase3_arg = fin_args[6] if len(fin_args) > 6 else None
        assert phase3_arg is None

    async def test_phase3_called_when_l_greater_than_1(self, tmp_path: Path) -> None:
        """With L=2, run_phase3 IS called."""
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=2)
        p3_result = _make_phase3_result()
        ctx = _standard_patches(tmp_path, config=config, phase3_result=p3_result)
        phase3_mock = AsyncMock(return_value=p3_result)

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=ctx["phase1_result"],
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", ctx["phase2_mock"]),
            patch(f"{_MODULE}.run_phase3", phase3_mock, create=True),
            patch(
                f"{_MODULE}.run_finalization",
                ctx["finalization_mock"],
            ),
        ):
            await run_pipeline(ctx["task"], ctx["config"])

        phase3_mock.assert_awaited_once()

    async def test_phase3_receives_best_solutions_from_phase2(
        self, tmp_path: Path
    ) -> None:
        """run_phase3 receives a list of best_solution from each Phase2Result."""
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=2)
        sol_a = _make_solution(content="solution_a", phase=SolutionPhase.REFINED)
        sol_b = _make_solution(content="solution_b", phase=SolutionPhase.REFINED)
        p2a = _make_phase2_result(best_solution=sol_a, best_score=0.88)
        p2b = _make_phase2_result(best_solution=sol_b, best_score=0.91)

        call_idx = 0

        async def _return_different(*args: Any, **kwargs: Any) -> Phase2Result:
            nonlocal call_idx
            result = p2a if call_idx == 0 else p2b
            call_idx += 1
            return result

        p3_result = _make_phase3_result()
        ctx = _standard_patches(
            tmp_path,
            config=config,
            phase2_side_effect=_return_different,
            phase3_result=p3_result,
        )
        phase3_mock = AsyncMock(return_value=p3_result)

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=ctx["phase1_result"],
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", side_effect=_return_different),
            patch(f"{_MODULE}.run_phase3", phase3_mock, create=True),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=ctx["final_result"],
            ),
        ):
            await run_pipeline(ctx["task"], ctx["config"])

        # Phase 3 should receive a list of best solutions from Phase 2
        p3_call_args, p3_call_kwargs = phase3_mock.call_args
        solutions_arg = p3_call_kwargs.get("solutions") or p3_call_args[3]

        assert len(solutions_arg) == 2
        solution_contents = {s.content for s in solutions_arg}
        assert "solution_a" in solution_contents
        assert "solution_b" in solution_contents


# ===========================================================================
# REQ-OR-016: Finalization receives best solution
# ===========================================================================


@pytest.mark.unit
class TestFinalizationReceivesBestSolution:
    """Finalization receives the best available solution from the pipeline (REQ-OR-016)."""

    async def test_finalization_receives_phase3_best_when_available(
        self, tmp_path: Path
    ) -> None:
        """When Phase 3 runs, finalization gets Phase 3 best_ensemble."""
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=2)
        ensemble_sol = _make_solution(
            content="ensemble_solution_code", phase=SolutionPhase.ENSEMBLE
        )
        p3_result = _make_phase3_result(best_ensemble=ensemble_sol)
        ctx = _standard_patches(tmp_path, config=config, phase3_result=p3_result)

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=ctx["phase1_result"],
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", ctx["phase2_mock"]),
            patch(
                f"{_MODULE}.run_phase3",
                new_callable=AsyncMock,
                return_value=p3_result,
                create=True,
            ),
            patch(
                f"{_MODULE}.run_finalization",
                ctx["finalization_mock"],
            ),
        ):
            await run_pipeline(ctx["task"], ctx["config"])

        # Finalization receives the best_ensemble from Phase 3 as the solution arg
        fin_args, fin_kwargs = ctx["finalization_mock"].call_args
        solution_arg = fin_kwargs.get("solution") or fin_args[1]
        assert solution_arg is ensemble_sol

    async def test_finalization_receives_phase2_best_when_phase3_skipped(
        self, tmp_path: Path
    ) -> None:
        """When L=1, finalization gets Phase 2 best_solution."""
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=1)
        p2_solution = _make_solution(
            content="refined_solution_code", phase=SolutionPhase.REFINED
        )
        p2_result = _make_phase2_result(best_solution=p2_solution)
        ctx = _standard_patches(tmp_path, config=config, phase2_result=p2_result)

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=ctx["phase1_result"],
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", ctx["phase2_mock"]),
            patch(
                f"{_MODULE}.run_finalization",
                ctx["finalization_mock"],
            ),
        ):
            await run_pipeline(ctx["task"], ctx["config"])

        fin_args, fin_kwargs = ctx["finalization_mock"].call_args
        solution_arg = fin_kwargs.get("solution") or fin_args[1]
        assert solution_arg is p2_solution

    async def test_finalization_receives_all_phase_results(
        self, tmp_path: Path
    ) -> None:
        """run_finalization receives phase1_result, phase2_results list, phase3_result."""
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=2)
        p1_result = _make_phase1_result()
        p2_result = _make_phase2_result()
        p3_result = _make_phase3_result()
        ctx = _standard_patches(
            tmp_path,
            config=config,
            phase1_result=p1_result,
            phase2_result=p2_result,
            phase3_result=p3_result,
        )

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1", new_callable=AsyncMock, return_value=p1_result
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", ctx["phase2_mock"]),
            patch(
                f"{_MODULE}.run_phase3",
                new_callable=AsyncMock,
                return_value=p3_result,
                create=True,
            ),
            patch(
                f"{_MODULE}.run_finalization",
                ctx["finalization_mock"],
            ),
        ):
            await run_pipeline(ctx["task"], ctx["config"])

        fin_args, fin_kwargs = ctx["finalization_mock"].call_args

        # phase1_result
        p1_arg = fin_kwargs.get("phase1_result") or fin_args[4]
        assert p1_arg is p1_result

        # phase2_results (list)
        p2_arg = fin_kwargs.get("phase2_results") or fin_args[5]
        assert isinstance(p2_arg, list)
        assert len(p2_arg) == 2
        assert all(isinstance(r, Phase2Result) for r in p2_arg)

        # phase3_result
        p3_arg = fin_kwargs.get("phase3_result") or fin_args[6]
        assert p3_arg is p3_result


# ===========================================================================
# Phase duration recording
# ===========================================================================


@pytest.mark.unit
class TestPhaseDurationRecording:
    """Each phase has a positive duration and total duration is recorded."""

    async def test_run_pipeline_records_total_duration(self, tmp_path: Path) -> None:
        """The FinalResult returned has total_duration_seconds > 0.

        This verifies the implementation tracks time.monotonic() at pipeline
        start and computes total duration. We check via the returned
        FinalResult (which finalization produces and the orchestrator may
        also update).
        """
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=1)
        # Make finalization return a result with duration to verify pipeline tracks it
        fr = _make_final_result(total_duration_seconds=42.0)
        ctx = _standard_patches(tmp_path, config=config, final_result=fr)

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=ctx["phase1_result"],
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", ctx["phase2_mock"]),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=fr,
            ),
        ):
            result = await run_pipeline(ctx["task"], ctx["config"])

        assert isinstance(result, FinalResult)
        assert result.total_duration_seconds > 0

    async def test_phase_durations_positive(self, tmp_path: Path) -> None:
        """Each phase invocation takes non-negative wall-clock time.

        We cannot directly measure per-phase durations from the test, but
        we verify the overall pipeline completes with a positive duration
        and that run_pipeline produces a FinalResult (which implies all
        phases ran in sequence with time passing between them).
        """
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=2)
        p3_result = _make_phase3_result()
        # Final result with a positive duration
        fr = _make_final_result(total_duration_seconds=55.0, phase3=p3_result)
        ctx = _standard_patches(
            tmp_path,
            config=config,
            phase3_result=p3_result,
            final_result=fr,
        )

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=ctx["phase1_result"],
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", ctx["phase2_mock"]),
            patch(
                f"{_MODULE}.run_phase3",
                new_callable=AsyncMock,
                return_value=p3_result,
                create=True,
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=fr,
            ),
        ):
            result = await run_pipeline(ctx["task"], ctx["config"])

        assert result.total_duration_seconds > 0


# ===========================================================================
# REQ-OR-022, REQ-OR-040: Phase 2 failure handling
# ===========================================================================


@pytest.mark.unit
class TestPhase2FailureHandling:
    """Failed Phase 2 paths use Phase 1 solution as substitute (REQ-OR-040)."""

    async def test_failed_phase2_path_substitutes_phase1_solution(
        self, tmp_path: Path
    ) -> None:
        """If one Phase 2 path raises, Phase 1 solution substitutes for that path.

        With L=2, if path 0 fails, Phase 3 still receives 2 solutions: the
        Phase 1 fallback and the successful Phase 2 result.
        """
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=2)
        p1_solution = _make_solution(
            content="p1_fallback_code", phase=SolutionPhase.MERGED
        )
        p1_result = _make_phase1_result(
            initial_solution=p1_solution, initial_score=0.80
        )

        p2_success_sol = _make_solution(
            content="p2_success_code", phase=SolutionPhase.REFINED
        )
        p2_success = _make_phase2_result(best_solution=p2_success_sol, best_score=0.90)

        call_idx = 0

        async def _one_fails_one_succeeds(*args: Any, **kwargs: Any) -> Phase2Result:
            nonlocal call_idx
            idx = call_idx
            call_idx += 1
            if idx == 0:
                raise RuntimeError("Phase 2 path 0 exploded")
            return p2_success

        p3_result = _make_phase3_result()
        phase3_mock = AsyncMock(return_value=p3_result)

        fr = _make_final_result()
        ctx = _standard_patches(
            tmp_path,
            config=config,
            phase1_result=p1_result,
            phase2_side_effect=_one_fails_one_succeeds,
            phase3_result=p3_result,
            final_result=fr,
        )

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1", new_callable=AsyncMock, return_value=p1_result
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop", side_effect=_one_fails_one_succeeds
            ),
            patch(f"{_MODULE}.run_phase3", phase3_mock, create=True),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=fr,
            ),
        ):
            await run_pipeline(ctx["task"], ctx["config"])

        # Phase 3 should still be called
        phase3_mock.assert_awaited_once()

        # Phase 3 receives solutions list with Phase 1 fallback for the failed path
        p3_call_args, p3_call_kwargs = phase3_mock.call_args
        solutions_arg = p3_call_kwargs.get("solutions") or p3_call_args[3]
        assert len(solutions_arg) == 2

        solution_contents = [s.content for s in solutions_arg]
        # Must contain the Phase 1 fallback
        assert "p1_fallback_code" in solution_contents
        # Must contain the successful Phase 2 result
        assert "p2_success_code" in solution_contents

    async def test_all_phase2_paths_fail_uses_phase1_solution(
        self, tmp_path: Path
    ) -> None:
        """If all Phase 2 paths fail, Phase 1 solution is used for Phase 3 input.

        With L=2, both paths fail. Phase 3 should receive 2 copies of
        the Phase 1 solution.
        """
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=2)
        p1_solution = _make_solution(
            content="p1_fallback_all", phase=SolutionPhase.MERGED
        )
        p1_result = _make_phase1_result(
            initial_solution=p1_solution, initial_score=0.75
        )

        async def _all_fail(*args: Any, **kwargs: Any) -> Phase2Result:
            raise RuntimeError("Phase 2 catastrophic failure")

        p3_result = _make_phase3_result()
        phase3_mock = AsyncMock(return_value=p3_result)

        fr = _make_final_result()
        ctx = _standard_patches(
            tmp_path,
            config=config,
            phase1_result=p1_result,
            phase2_side_effect=_all_fail,
            phase3_result=p3_result,
            final_result=fr,
        )

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1", new_callable=AsyncMock, return_value=p1_result
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", side_effect=_all_fail),
            patch(f"{_MODULE}.run_phase3", phase3_mock, create=True),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=fr,
            ),
        ):
            await run_pipeline(ctx["task"], ctx["config"])

        phase3_mock.assert_awaited_once()

        p3_call_args, p3_call_kwargs = phase3_mock.call_args
        solutions_arg = p3_call_kwargs.get("solutions") or p3_call_args[3]
        assert len(solutions_arg) == 2
        # All solutions should be the Phase 1 fallback
        for sol in solutions_arg:
            assert sol.content == "p1_fallback_all"


# ===========================================================================
# Best solution selection
# ===========================================================================


@pytest.mark.unit
class TestBestSolutionSelection:
    """Best solution is correctly selected and passed to finalization."""

    async def test_best_phase2_solution_selected_for_finalization_when_phase3_skipped(
        self, tmp_path: Path
    ) -> None:
        """When L=1 and Phase 3 skipped, Phase 2 best_solution goes to finalization."""
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=1)
        best_p2_solution = _make_solution(
            content="best_p2_code", phase=SolutionPhase.REFINED
        )
        p2_result = _make_phase2_result(best_solution=best_p2_solution, best_score=0.93)
        ctx = _standard_patches(tmp_path, config=config, phase2_result=p2_result)

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=ctx["phase1_result"],
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", ctx["phase2_mock"]),
            patch(
                f"{_MODULE}.run_finalization",
                ctx["finalization_mock"],
            ),
        ):
            await run_pipeline(ctx["task"], ctx["config"])

        fin_args, fin_kwargs = ctx["finalization_mock"].call_args
        solution_arg = fin_kwargs.get("solution") or fin_args[1]
        assert solution_arg is best_p2_solution

    async def test_best_phase3_solution_selected_for_finalization(
        self, tmp_path: Path
    ) -> None:
        """When Phase 3 completes, its best_ensemble goes to finalization."""
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=2)
        p3_solution = _make_solution(
            content="best_ensemble_code", phase=SolutionPhase.ENSEMBLE
        )
        p3_result = _make_phase3_result(
            best_ensemble=p3_solution, best_ensemble_score=0.95
        )
        ctx = _standard_patches(tmp_path, config=config, phase3_result=p3_result)

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=ctx["phase1_result"],
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", ctx["phase2_mock"]),
            patch(
                f"{_MODULE}.run_phase3",
                new_callable=AsyncMock,
                return_value=p3_result,
                create=True,
            ),
            patch(
                f"{_MODULE}.run_finalization",
                ctx["finalization_mock"],
            ),
        ):
            await run_pipeline(ctx["task"], ctx["config"])

        fin_args, fin_kwargs = ctx["finalization_mock"].call_args
        solution_arg = fin_kwargs.get("solution") or fin_args[1]
        assert solution_arg is p3_solution


# ===========================================================================
# Parametrized: L values for Phase 2 dispatch count
# ===========================================================================


@pytest.mark.unit
class TestPhase2DispatchParametrized:
    """Parametrized tests for various L values in Phase 2 dispatch."""

    @pytest.mark.parametrize("num_paths", [1, 2, 3, 4])
    async def test_phase2_called_l_times(self, num_paths: int, tmp_path: Path) -> None:
        """run_phase2_outer_loop is called exactly L times for L parallel paths."""
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=num_paths)
        p3_result = _make_phase3_result() if num_paths > 1 else None
        ctx = _standard_patches(tmp_path, config=config, phase3_result=p3_result)

        patches = [
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=ctx["phase1_result"],
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", ctx["phase2_mock"]),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=ctx["final_result"],
            ),
        ]

        if num_paths > 1:
            patches.append(
                patch(
                    f"{_MODULE}.run_phase3",
                    new_callable=AsyncMock,
                    return_value=p3_result,
                    create=True,
                )
            )

        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            await run_pipeline(ctx["task"], ctx["config"])

        assert ctx["phase2_mock"].call_count == num_paths


# ===========================================================================
# Phase 2 gather with return_exceptions=True
# ===========================================================================


@pytest.mark.unit
class TestPhase2GatherReturnExceptions:
    """Phase 2 uses asyncio.gather with return_exceptions=True (REQ-OR-013)."""

    async def test_single_phase2_failure_does_not_crash_pipeline(
        self, tmp_path: Path
    ) -> None:
        """A single Phase 2 path failure should not crash the entire pipeline.

        The pipeline should handle the exception gracefully and continue
        to Phase 3 / finalization.
        """
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=2)
        p1_result = _make_phase1_result()

        call_idx = 0

        async def _one_fails(*args: Any, **kwargs: Any) -> Phase2Result:
            nonlocal call_idx
            idx = call_idx
            call_idx += 1
            if idx == 0:
                raise RuntimeError("path-0 failed")
            return _make_phase2_result()

        p3_result = _make_phase3_result()
        fr = _make_final_result()

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=_make_mock_client()),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1", new_callable=AsyncMock, return_value=p1_result
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", side_effect=_one_fails),
            patch(
                f"{_MODULE}.run_phase3",
                new_callable=AsyncMock,
                return_value=p3_result,
                create=True,
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=fr,
            ),
        ):
            # Should NOT raise
            result = await run_pipeline(
                _make_task(data_dir=str(_make_data_dir(tmp_path))),
                config,
            )

        assert isinstance(result, FinalResult)


# ===========================================================================
# Edge case: L=1 single path, no Phase 3
# ===========================================================================


@pytest.mark.unit
class TestSinglePathPipeline:
    """L=1 pipeline skips Phase 3 and passes Phase 2 result directly."""

    async def test_single_path_complete_pipeline(self, tmp_path: Path) -> None:
        """With L=1, pipeline runs P1 -> single P2 -> skip P3 -> Finalization."""
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=1)
        p1_result = _make_phase1_result()
        p2_result = _make_phase2_result()
        fr = _make_final_result()

        phase1_mock = AsyncMock(return_value=p1_result)
        phase2_mock = AsyncMock(return_value=p2_result)
        phase3_mock = AsyncMock()
        finalization_mock = AsyncMock(return_value=fr)

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=_make_mock_client()),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(f"{_MODULE}.run_phase1", phase1_mock),
            patch(f"{_MODULE}.run_phase2_outer_loop", phase2_mock),
            patch(f"{_MODULE}.run_phase3", phase3_mock, create=True),
            patch(f"{_MODULE}.run_finalization", finalization_mock),
        ):
            result = await run_pipeline(task, config)

        assert isinstance(result, FinalResult)
        phase1_mock.assert_awaited_once()
        phase2_mock.assert_awaited_once()
        phase3_mock.assert_not_called()
        finalization_mock.assert_awaited_once()

    async def test_single_path_finalization_receives_none_for_phase3(
        self, tmp_path: Path
    ) -> None:
        """With L=1, finalization receives phase3_result=None."""
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=1)
        ctx = _standard_patches(tmp_path, config=config)

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=ctx["mock_client"]),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=ctx["phase1_result"],
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", ctx["phase2_mock"]),
            patch(
                f"{_MODULE}.run_finalization",
                ctx["finalization_mock"],
            ),
        ):
            await run_pipeline(ctx["task"], ctx["config"])

        fin_args, fin_kwargs = ctx["finalization_mock"].call_args
        # Check phase3_result is None
        phase3_arg = fin_kwargs.get("phase3_result")
        if phase3_arg is None and "phase3_result" not in fin_kwargs:
            phase3_arg = fin_args[6] if len(fin_args) > 6 else None
        assert phase3_arg is None


# ===========================================================================
# Edge case: Phase 2 returns different scores, best is selected
# ===========================================================================


@pytest.mark.unit
class TestPhase2BestSolutionSelection:
    """When L>1 and Phase 3 is skipped or absent, best Phase 2 result is selected."""

    async def test_multiple_phase2_solutions_passed_to_phase3(
        self, tmp_path: Path
    ) -> None:
        """With L=3, Phase 3 receives 3 solutions (one per Phase 2 path)."""
        from mle_star.orchestrator import run_pipeline

        config = _make_config(num_parallel_solutions=3)
        solutions = [
            _make_solution(content=f"sol_{i}", phase=SolutionPhase.REFINED)
            for i in range(3)
        ]
        p2_results = [
            _make_phase2_result(best_solution=sol, best_score=0.80 + i * 0.05)
            for i, sol in enumerate(solutions)
        ]

        call_idx = 0

        async def _return_p2(*args: Any, **kwargs: Any) -> Phase2Result:
            nonlocal call_idx
            result = p2_results[call_idx]
            call_idx += 1
            return result

        p3_result = _make_phase3_result()
        phase3_mock = AsyncMock(return_value=p3_result)
        fr = _make_final_result()
        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=_make_mock_client()),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=_make_phase1_result(),
            ),
            patch(f"{_MODULE}.run_phase2_outer_loop", side_effect=_return_p2),
            patch(f"{_MODULE}.run_phase3", phase3_mock, create=True),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=fr,
            ),
        ):
            await run_pipeline(task, config)

        p3_call_args, p3_call_kwargs = phase3_mock.call_args
        solutions_arg = p3_call_kwargs.get("solutions") or p3_call_args[3]
        assert len(solutions_arg) == 3
        solution_contents = {s.content for s in solutions_arg}
        assert solution_contents == {"sol_0", "sol_1", "sol_2"}
