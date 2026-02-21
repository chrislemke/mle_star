"""Tests for Phase 1 post-merge safety checks and Phase1Result construction (Task 29).

Validates that ``run_phase1`` performs post-merge safety checks:

1. ``check_data_usage`` called EXACTLY ONCE on the merged solution (REQ-P1-030).
2. ``check_and_fix_leakage`` called after data check on the final solution (REQ-P1-031).
3. ``Phase1Result`` constructed with correct fields after all safety checks (REQ-P1-032).
4. ``Phase1Result.initial_score`` reflects the final re-evaluated score (REQ-P1-033).

Tests are written TDD-first and serve as the executable specification for
REQ-P1-030 through REQ-P1-033.

Refs:
    SRS 04b (Phase 1 Orchestration), IMPLEMENTATION_PLAN.md Task 29.
"""

from __future__ import annotations

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
# Reusable test helpers (mirrored from test_phase1_orchestration.py)
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
# REQ-P1-030: check_data_usage called EXACTLY ONCE after merge loop
# ===========================================================================


@pytest.mark.unit
class TestDataUsageCheckInvocation:
    """check_data_usage called exactly once on merged solution (REQ-P1-030)."""

    async def test_check_data_usage_called_once_after_merge(self) -> None:
        """check_data_usage is invoked exactly once with (s_0, task, client)."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="code_m1", source_model="m1")
        res = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[(sol, res)],
            ranked_pairs=[(sol, res)],
        )
        mock_data = AsyncMock(side_effect=lambda s, _t, _c: s)
        mocks["check_data_usage"] = mock_data

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        mock_data.assert_awaited_once()

    async def test_check_data_usage_receives_s0_task_client(self) -> None:
        """check_data_usage is called with the merged s_0, task, and client."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="code_a", source_model="m1")
        sol_b = _make_solution(content="code_b", source_model="m2")
        res_a = _make_eval_result(score=0.80)
        res_b = _make_eval_result(score=0.90)

        merged = _make_merged_solution("merged_ab")
        merged_res = _make_eval_result(score=0.92)

        ranked = [(sol_b, res_b), (sol_a, res_a)]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[
                (sol_a, res_a),
                (sol_b, res_b),
                (merged, merged_res),
            ],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=merged)
        mocks["is_improvement_or_equal"] = MagicMock(return_value=True)

        data_call_args: list[tuple[Any, ...]] = []

        async def _capture_data(
            s: SolutionScript, t: TaskDescription, c: Any
        ) -> SolutionScript:
            data_call_args.append((s, t, c))
            return s

        mocks["check_data_usage"] = AsyncMock(side_effect=_capture_data)

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        assert len(data_call_args) == 1
        captured_sol, captured_task, captured_client = data_call_args[0]
        # After merge, s_0 should be the merged solution
        assert captured_sol.content == "merged_ab"
        assert captured_task is task
        assert captured_client is client

    async def test_check_data_usage_not_called_multiple_times(self) -> None:
        """check_data_usage is called at most once, even with many merges."""
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
        merged_012 = _make_merged_solution("merged_012")
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

        mock_data = AsyncMock(side_effect=lambda s, _t, _c: s)
        mocks["check_data_usage"] = mock_data

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        # Exactly once, not once per merge iteration
        assert mock_data.await_count == 1


# ===========================================================================
# REQ-P1-030: Data check unchanged => no re-evaluation
# ===========================================================================


@pytest.mark.unit
class TestDataCheckUnchangedNoReeval:
    """When check_data_usage returns unchanged solution, no re-evaluation (REQ-P1-030)."""

    async def test_no_reeval_when_data_check_returns_same_solution(self) -> None:
        """If check_data_usage returns the exact same solution, no extra evaluate call."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="code_m1", source_model="m1")
        res = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[(sol, res)],
            ranked_pairs=[(sol, res)],
        )
        # check_data_usage returns the SAME solution (unchanged)
        mock_data = AsyncMock(side_effect=lambda s, _t, _c: s)
        mocks["check_data_usage"] = mock_data

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        # Only 1 evaluate_with_retry call: the initial candidate evaluation
        # No extra call for re-evaluation after data check
        assert mocks["evaluate_with_retry"].await_count == 1
        assert result.initial_score == 0.85


# ===========================================================================
# REQ-P1-030: Data check modified => re-evaluation
# ===========================================================================


@pytest.mark.unit
class TestDataCheckModifiedTriggersReeval:
    """When check_data_usage returns modified solution, re-evaluation occurs (REQ-P1-030)."""

    async def test_reeval_when_data_check_modifies_solution(self) -> None:
        """If check_data_usage returns a different solution, evaluate_with_retry is called again."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="original_code", source_model="m1")
        res = _make_eval_result(score=0.85)

        modified_sol = _make_solution(
            content="data_modified_code", phase=SolutionPhase.INIT
        )
        modified_res = _make_eval_result(score=0.88)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[
                (sol, res),
                (modified_sol, modified_res),
            ],
            ranked_pairs=[(sol, res)],
        )
        # check_data_usage returns a DIFFERENT solution
        mock_data = AsyncMock(return_value=modified_sol)
        mocks["check_data_usage"] = mock_data

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        # 2 evaluate calls: initial candidate + re-eval after data modification
        assert mocks["evaluate_with_retry"].await_count == 2
        # Score should reflect the re-evaluated score
        assert result.initial_score == 0.88
        assert result.initial_solution.content == "data_modified_code"

    async def test_reeval_score_update_reflected_in_result(self) -> None:
        """Re-evaluation after data check updates Phase1Result.initial_score."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="orig", source_model="m1")
        res = _make_eval_result(score=0.80)

        data_sol = _make_solution(content="data_fixed", phase=SolutionPhase.INIT)
        data_res = _make_eval_result(score=0.92)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[
                (sol, res),
                (data_sol, data_res),
            ],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_data_usage"] = AsyncMock(return_value=data_sol)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert result.initial_score == 0.92
        assert result.initial_solution.score == 0.92


# ===========================================================================
# REQ-P1-030: Data check re-evaluation fails => fallback to pre-modification
# ===========================================================================


@pytest.mark.unit
class TestDataCheckReevalFailureFallback:
    """When re-evaluation after data check fails, fall back to pre-modification (REQ-P1-030)."""

    async def test_fallback_on_reeval_error(self) -> None:
        """If re-evaluation returns is_error=True, fall back to pre-A_data solution."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="original_code", source_model="m1")
        res = _make_eval_result(score=0.85)

        data_sol = _make_solution(content="data_broken_code", phase=SolutionPhase.INIT)
        data_res = _make_eval_result(score=None, is_error=True)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[
                (sol, res),
                (data_sol, data_res),
            ],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_data_usage"] = AsyncMock(return_value=data_sol)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        # Fell back to pre-A_data version
        assert result.initial_solution.content == "original_code"
        assert result.initial_score == 0.85

    async def test_fallback_on_reeval_none_score(self) -> None:
        """If re-evaluation returns score=None (no error), fall back to pre-A_data solution."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="original_code", source_model="m1")
        res = _make_eval_result(score=0.85)

        data_sol = _make_solution(
            content="data_modified_code", phase=SolutionPhase.INIT
        )
        data_res = _make_eval_result(score=None, is_error=False)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[
                (sol, res),
                (data_sol, data_res),
            ],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_data_usage"] = AsyncMock(return_value=data_sol)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        # Fell back to pre-A_data version
        assert result.initial_solution.content == "original_code"
        assert result.initial_score == 0.85

    async def test_fallback_preserves_score_on_solution(self) -> None:
        """After fallback, initial_solution.score matches initial_score."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="original_code", source_model="m1")
        res = _make_eval_result(score=0.85)

        data_sol = _make_solution(content="broken", phase=SolutionPhase.INIT)
        data_res = _make_eval_result(score=None, is_error=True)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[
                (sol, res),
                (data_sol, data_res),
            ],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_data_usage"] = AsyncMock(return_value=data_sol)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert result.initial_solution.score == result.initial_score


# ===========================================================================
# REQ-P1-031: check_and_fix_leakage called after data check
# ===========================================================================


@pytest.mark.unit
class TestPostMergeLeakageCheck:
    """check_and_fix_leakage called after data check on final solution (REQ-P1-031)."""

    async def test_leakage_check_called_after_data_check(self) -> None:
        """check_and_fix_leakage runs on the solution after check_data_usage."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="code_m1", source_model="m1")
        res = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[(sol, res)],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_data_usage"] = AsyncMock(side_effect=lambda s, _t, _c: s)

        # Track ordering: data check before leakage check
        call_order: list[str] = []

        original_data = mocks["check_data_usage"]
        original_leakage = mocks["check_and_fix_leakage"]

        async def _data_wrapper(s: Any, t: Any, c: Any) -> SolutionScript:
            call_order.append("data")
            return await original_data(s, t, c)

        async def _leakage_wrapper(s: Any, t: Any, c: Any) -> SolutionScript:
            call_order.append("leakage")
            return await original_leakage(s, t, c)

        mocks["check_data_usage"] = AsyncMock(side_effect=_data_wrapper)
        mocks["check_and_fix_leakage"] = AsyncMock(side_effect=_leakage_wrapper)

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        # The post-merge data check should come before the post-merge leakage check
        # There may be leakage calls from the candidate generation phase too,
        # so we just verify data appears before the LAST leakage call
        assert "data" in call_order
        data_idx = call_order.index("data")
        last_leakage_idx = len(call_order) - 1 - call_order[::-1].index("leakage")
        assert data_idx < last_leakage_idx

    async def test_leakage_check_receives_post_data_solution(self) -> None:
        """check_and_fix_leakage receives the solution output from check_data_usage."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="original_code", source_model="m1")
        res = _make_eval_result(score=0.85)

        # Data check modifies the solution
        data_modified = _make_solution(
            content="data_checked_code", phase=SolutionPhase.INIT
        )
        data_res = _make_eval_result(score=0.87)

        leakage_calls: list[SolutionScript] = []

        async def _track_leakage(s: SolutionScript, _t: Any, _c: Any) -> SolutionScript:
            leakage_calls.append(s)
            return s

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[
                (sol, res),
                (data_modified, data_res),
            ],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_data_usage"] = AsyncMock(return_value=data_modified)
        mocks["check_and_fix_leakage"] = AsyncMock(side_effect=_track_leakage)

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        # The post-merge leakage check should receive the data-modified solution
        post_merge_leakage = [
            c for c in leakage_calls if c.content == "data_checked_code"
        ]
        assert len(post_merge_leakage) >= 1

    async def test_leakage_modification_triggers_reeval(self) -> None:
        """When post-merge leakage check modifies solution, re-evaluation occurs."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="original_code", source_model="m1")
        res = _make_eval_result(score=0.85)

        leakage_fixed = _make_solution(
            content="leakage_fixed_code", phase=SolutionPhase.INIT
        )
        leakage_res = _make_eval_result(score=0.83)

        eval_count = 0

        async def _counting_leakage(
            s: SolutionScript, _t: Any, _c: Any
        ) -> SolutionScript:
            nonlocal eval_count
            eval_count = mocks["evaluate_with_retry"].await_count
            # Only modify on the post-merge call (not during candidate gen)
            if s.content == "original_code":
                return leakage_fixed
            return s

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[
                (sol, res),
                (leakage_fixed, leakage_res),
            ],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_data_usage"] = AsyncMock(side_effect=lambda s, _t, _c: s)
        mocks["check_and_fix_leakage"] = AsyncMock(side_effect=_counting_leakage)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        # Re-evaluation should have happened
        assert mocks["evaluate_with_retry"].await_count >= 2
        assert result.initial_score == 0.83
        assert result.initial_solution.content == "leakage_fixed_code"

    async def test_leakage_unchanged_no_extra_reeval(self) -> None:
        """When post-merge leakage check returns same solution, no extra re-evaluation."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="code_m1", source_model="m1")
        res = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[(sol, res)],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_data_usage"] = AsyncMock(side_effect=lambda s, _t, _c: s)
        # Leakage returns same solution (no modification)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        # Only 1 evaluate call: initial candidate (no re-eval for unchanged leakage)
        assert mocks["evaluate_with_retry"].await_count == 1
        assert result.initial_score == 0.85


# ===========================================================================
# REQ-P1-032: Phase1Result constructed correctly after safety checks
# ===========================================================================


@pytest.mark.unit
class TestPhase1ResultAfterSafetyChecks:
    """Phase1Result fields correct after all safety checks (REQ-P1-032)."""

    async def test_result_has_correct_fields_after_data_and_leakage(self) -> None:
        """Phase1Result contains correct models, solutions, scores after safety."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="code_a", source_model="m1")
        sol_b = _make_solution(content="code_b", source_model="m2")
        res_a = _make_eval_result(score=0.80)
        res_b = _make_eval_result(score=0.90)

        ranked = [(sol_b, res_b), (sol_a, res_a)]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[(sol_a, res_a), (sol_b, res_b)],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)
        mocks["check_data_usage"] = AsyncMock(side_effect=lambda s, _t, _c: s)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert isinstance(result, Phase1Result)
        assert result.retrieved_models == models
        assert len(result.candidate_solutions) == 2
        assert len(result.candidate_scores) == 2
        assert result.initial_score == 0.90

    async def test_result_initial_solution_is_post_safety_solution(self) -> None:
        """Phase1Result.initial_solution is the solution AFTER all safety checks."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="original", source_model="m1")
        res = _make_eval_result(score=0.80)

        # Data check modifies solution
        data_sol = _make_solution(content="data_fixed", phase=SolutionPhase.INIT)
        data_res = _make_eval_result(score=0.85)

        # Leakage check modifies solution further
        leak_sol = _make_solution(content="leak_fixed", phase=SolutionPhase.INIT)
        leak_res = _make_eval_result(score=0.83)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[
                (sol, res),
                (data_sol, data_res),
                (leak_sol, leak_res),
            ],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_data_usage"] = AsyncMock(return_value=data_sol)

        # Leakage modifies the data_sol further
        async def _leakage_side_effect(
            s: SolutionScript, _t: Any, _c: Any
        ) -> SolutionScript:
            if s.content == "data_fixed":
                return leak_sol
            return s

        mocks["check_and_fix_leakage"] = AsyncMock(side_effect=_leakage_side_effect)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        # Final solution is the one after BOTH safety checks
        assert result.initial_solution.content == "leak_fixed"
        assert result.initial_score == 0.83

    async def test_candidate_scores_unchanged_by_safety_checks(self) -> None:
        """Phase1Result.candidate_scores reflects pre-safety scores."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="code_a", source_model="m1")
        sol_b = _make_solution(content="code_b", source_model="m2")
        res_a = _make_eval_result(score=0.80)
        res_b = _make_eval_result(score=0.90)

        ranked = [(sol_b, res_b), (sol_a, res_a)]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[(sol_a, res_a), (sol_b, res_b)],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)
        mocks["check_data_usage"] = AsyncMock(side_effect=lambda s, _t, _c: s)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        # candidate_scores are from the generation phase, not affected by safety
        assert result.candidate_scores == [0.80, 0.90]


# ===========================================================================
# REQ-P1-033: initial_score reflects final re-evaluated score
# ===========================================================================


@pytest.mark.unit
class TestInitialScoreReflectsFinal:
    """Phase1Result.initial_score reflects final score after safety (REQ-P1-033)."""

    async def test_initial_score_after_data_modification_and_reeval(self) -> None:
        """initial_score is the re-evaluated score after data check modification."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="orig", source_model="m1")
        res = _make_eval_result(score=0.80)

        data_sol = _make_solution(content="data_mod", phase=SolutionPhase.INIT)
        data_res = _make_eval_result(score=0.88)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[
                (sol, res),
                (data_sol, data_res),
            ],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_data_usage"] = AsyncMock(return_value=data_sol)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert result.initial_score == 0.88

    async def test_initial_score_after_leakage_modification_and_reeval(self) -> None:
        """initial_score is the re-evaluated score after leakage check modification."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="orig", source_model="m1")
        res = _make_eval_result(score=0.85)

        leak_sol = _make_solution(content="leak_fixed", phase=SolutionPhase.INIT)
        leak_res = _make_eval_result(score=0.82)

        async def _leakage(s: SolutionScript, _t: Any, _c: Any) -> SolutionScript:
            if s.content == "orig":
                return leak_sol
            return s

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[
                (sol, res),
                (leak_sol, leak_res),
            ],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_data_usage"] = AsyncMock(side_effect=lambda s, _t, _c: s)
        mocks["check_and_fix_leakage"] = AsyncMock(side_effect=_leakage)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert result.initial_score == 0.82
        assert result.initial_solution.score == 0.82

    async def test_score_consistency_initial_score_equals_solution_score(self) -> None:
        """initial_score always equals initial_solution.score after safety checks."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="orig", source_model="m1")
        res = _make_eval_result(score=0.80)

        data_sol = _make_solution(content="data_mod", phase=SolutionPhase.INIT)
        data_res = _make_eval_result(score=0.91)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[
                (sol, res),
                (data_sol, data_res),
            ],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_data_usage"] = AsyncMock(return_value=data_sol)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert result.initial_score == result.initial_solution.score

    async def test_initial_score_fallback_after_both_checks_fail_reeval(self) -> None:
        """Fallback score preserved if data re-eval fails and leakage returns same."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="original_code", source_model="m1")
        res = _make_eval_result(score=0.85)

        data_sol = _make_solution(content="data_broken", phase=SolutionPhase.INIT)
        data_res = _make_eval_result(score=None, is_error=True)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[
                (sol, res),
                (data_sol, data_res),
            ],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_data_usage"] = AsyncMock(return_value=data_sol)
        # Leakage returns same solution (no modification after fallback)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        # Fell back to original; leakage found nothing to change
        assert result.initial_score == 0.85
        assert result.initial_solution.score == 0.85


# ===========================================================================
# Combined scenario: data + leakage both modify
# ===========================================================================


@pytest.mark.unit
class TestCombinedDataAndLeakageSafety:
    """Combined scenario where both data and leakage checks modify the solution."""

    async def test_both_data_and_leakage_modify_solution(self) -> None:
        """When both safety checks modify the solution, final reflects last modification."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="original", source_model="m1")
        res = _make_eval_result(score=0.80)

        # Data check modifies
        data_sol = _make_solution(content="data_enhanced", phase=SolutionPhase.INIT)
        data_res = _make_eval_result(score=0.88)

        # Leakage check modifies the data-enhanced solution
        leak_sol = _make_solution(content="final_safe", phase=SolutionPhase.INIT)
        leak_res = _make_eval_result(score=0.86)

        async def _leakage(s: SolutionScript, _t: Any, _c: Any) -> SolutionScript:
            if s.content == "data_enhanced":
                return leak_sol
            return s

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[
                (sol, res),
                (data_sol, data_res),
                (leak_sol, leak_res),
            ],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_data_usage"] = AsyncMock(return_value=data_sol)
        mocks["check_and_fix_leakage"] = AsyncMock(side_effect=_leakage)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert result.initial_solution.content == "final_safe"
        assert result.initial_score == 0.86
        assert result.initial_solution.score == 0.86

    async def test_data_modifies_leakage_unchanged(self) -> None:
        """Data check modifies solution, leakage returns same; only 1 re-eval."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="original", source_model="m1")
        res = _make_eval_result(score=0.80)

        data_sol = _make_solution(content="data_enhanced", phase=SolutionPhase.INIT)
        data_res = _make_eval_result(score=0.88)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[
                (sol, res),
                (data_sol, data_res),
            ],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_data_usage"] = AsyncMock(return_value=data_sol)
        # Leakage passthrough for the data_sol
        mocks["check_and_fix_leakage"] = AsyncMock(side_effect=lambda s, _t, _c: s)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        # 2 evaluate calls: initial + data re-eval (no leakage re-eval)
        assert mocks["evaluate_with_retry"].await_count == 2
        assert result.initial_solution.content == "data_enhanced"
        assert result.initial_score == 0.88


# ===========================================================================
# Full end-to-end scenario with merges + safety checks
# ===========================================================================


@pytest.mark.unit
class TestFullScenarioWithSafety:
    """End-to-end: merge loop followed by data + leakage checks."""

    async def test_merge_then_safety_checks(self) -> None:
        """Full pipeline: 2 models, merge improves, then data and leakage checks run."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="code_a", source_model="m1")
        sol_b = _make_solution(content="code_b", source_model="m2")
        res_a = _make_eval_result(score=0.80)
        res_b = _make_eval_result(score=0.90)

        merged = _make_merged_solution("merged_ab")
        merged_res = _make_eval_result(score=0.92)

        ranked = [(sol_b, res_b), (sol_a, res_a)]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[
                (sol_a, res_a),
                (sol_b, res_b),
                (merged, merged_res),
            ],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=merged)
        mocks["is_improvement_or_equal"] = MagicMock(return_value=True)

        # Safety checks: passthrough
        mocks["check_data_usage"] = AsyncMock(side_effect=lambda s, _t, _c: s)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        # Data check was called
        mocks["check_data_usage"].assert_awaited_once()
        # Result reflects the merged solution (unchanged by safety)
        assert result.initial_solution.content == "merged_ab"
        assert result.initial_score == 0.92

    async def test_merge_then_data_modifies_then_leakage_modifies(self) -> None:
        """Full pipeline: merge, data modifies, leakage modifies further."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="code_a", source_model="m1")
        sol_b = _make_solution(content="code_b", source_model="m2")
        res_a = _make_eval_result(score=0.80)
        res_b = _make_eval_result(score=0.90)

        merged = _make_merged_solution("merged_code")
        merged_res = _make_eval_result(score=0.92)

        # Data modifies merged
        data_sol = _make_solution(
            content="merged_data_fixed", phase=SolutionPhase.MERGED
        )
        data_res = _make_eval_result(score=0.93)

        # Leakage modifies data-fixed
        leak_sol = _make_solution(
            content="merged_data_leak_fixed", phase=SolutionPhase.MERGED
        )
        leak_res = _make_eval_result(score=0.91)

        ranked = [(sol_b, res_b), (sol_a, res_a)]

        async def _leakage(s: SolutionScript, _t: Any, _c: Any) -> SolutionScript:
            if s.content == "merged_data_fixed":
                return leak_sol
            return s

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[
                (sol_a, res_a),
                (sol_b, res_b),
                (merged, merged_res),
                (data_sol, data_res),
                (leak_sol, leak_res),
            ],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=merged)
        mocks["is_improvement_or_equal"] = MagicMock(return_value=True)
        mocks["check_data_usage"] = AsyncMock(return_value=data_sol)
        mocks["check_and_fix_leakage"] = AsyncMock(side_effect=_leakage)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert result.initial_solution.content == "merged_data_leak_fixed"
        assert result.initial_score == 0.91
        assert result.initial_solution.score == 0.91


# ===========================================================================
# Hypothesis: property-based tests for safety check invariants
# ===========================================================================


@pytest.mark.unit
class TestPhase1SafetyProperties:
    """Property-based tests for Phase 1 safety check invariants."""

    @given(
        pre_score=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        post_score=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=20, deadline=5000)
    async def test_initial_score_equals_solution_score_invariant(
        self,
        pre_score: float,
        post_score: float,
    ) -> None:
        """initial_score always equals initial_solution.score regardless of safety outcomes."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="orig", source_model="m1")
        res = _make_eval_result(score=pre_score)

        data_sol = _make_solution(content="data_mod", phase=SolutionPhase.INIT)
        data_res = _make_eval_result(score=post_score)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[
                (sol, res),
                (data_sol, data_res),
            ],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_data_usage"] = AsyncMock(return_value=data_sol)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert result.initial_score == result.initial_solution.score

    @given(
        pre_score=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=15, deadline=5000)
    async def test_data_check_called_exactly_once_property(
        self,
        pre_score: float,
    ) -> None:
        """check_data_usage is always called exactly once, regardless of score."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="code", source_model="m1")
        res = _make_eval_result(score=pre_score)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[(sol, res)],
            ranked_pairs=[(sol, res)],
        )
        mock_data = AsyncMock(side_effect=lambda s, _t, _c: s)
        mocks["check_data_usage"] = mock_data

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        assert mock_data.await_count == 1


# ===========================================================================
# Edge case: MINIMIZE direction with safety checks
# ===========================================================================


@pytest.mark.unit
class TestSafetyChecksWithMinimize:
    """Safety checks work correctly with MINIMIZE metric direction."""

    async def test_data_modification_reeval_with_minimize(self) -> None:
        """Data check re-evaluation works with MINIMIZE direction."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task(direction=MetricDirection.MINIMIZE)
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="orig", source_model="m1")
        res = _make_eval_result(score=0.20)

        data_sol = _make_solution(content="data_mod", phase=SolutionPhase.INIT)
        data_res = _make_eval_result(score=0.15)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[
                (sol, res),
                (data_sol, data_res),
            ],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_data_usage"] = AsyncMock(return_value=data_sol)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert result.initial_score == 0.15
        assert result.initial_solution.content == "data_mod"

    async def test_fallback_preserves_minimize_score(self) -> None:
        """Fallback after data re-eval failure preserves original score for MINIMIZE."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task(direction=MetricDirection.MINIMIZE)
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="orig", source_model="m1")
        res = _make_eval_result(score=0.10)

        data_sol = _make_solution(content="broken", phase=SolutionPhase.INIT)
        data_res = _make_eval_result(score=None, is_error=True)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[
                (sol, res),
                (data_sol, data_res),
            ],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_data_usage"] = AsyncMock(return_value=data_sol)

        with _apply_patches(mocks):
            result = await run_phase1(task, config, client)

        assert result.initial_score == 0.10
        assert result.initial_solution.content == "orig"
