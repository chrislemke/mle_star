"""Tests for the Phase 3 orchestration function ``run_phase3`` (Task 36).

Validates ``run_phase3`` which implements Algorithm 3: ensemble construction.
Given L solution scripts, execute R ensemble rounds.  Each round invokes
A_ens_planner to propose a strategy, A_ensembler to implement it, then
applies leakage check and evaluation with debug retry.  The best ensemble
solution is selected using ``is_improvement_or_equal`` (>= semantics).

Tests are written TDD-first and serve as the executable specification for
REQ-P3-017 through REQ-P3-035.

Refs:
    SRS 06b (Phase 3 Orchestration), IMPLEMENTATION_PLAN.md Task 36.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    DataModality,
    EvaluationResult,
    MetricDirection,
    Phase3Result,
    PipelineConfig,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
    TaskType,
)
from pydantic import ValidationError
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
    score: float | None = None,
) -> SolutionScript:
    """Create a SolutionScript for testing."""
    return SolutionScript(content=content, phase=phase, score=score)


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


def _make_config(ensemble_rounds: int = 5) -> PipelineConfig:
    """Create a PipelineConfig for testing with optional ensemble_rounds."""
    return PipelineConfig(ensemble_rounds=ensemble_rounds)


def _make_eval_result(
    score: float | None = 0.85,
    is_error: bool = False,
) -> EvaluationResult:
    """Create an EvaluationResult with the given score and error state."""
    return EvaluationResult(
        score=score,
        stdout=f"Final Validation Performance: {score}" if score else "",
        stderr="" if not is_error else "Traceback (most recent call last):\nError",
        exit_code=0 if not is_error else 1,
        duration_seconds=1.0,
        is_error=is_error,
        error_traceback="Traceback..." if is_error else None,
    )


def _make_ensemble_solution(
    content: str = "ensemble_code",
    score: float | None = None,
) -> SolutionScript:
    """Create an ENSEMBLE-phase SolutionScript."""
    return SolutionScript(content=content, phase=SolutionPhase.ENSEMBLE, score=score)


def _patch_phase3_dependencies(
    ens_planner_rv: str | None = "Use weighted averaging",
    ensembler_rv: SolutionScript | None = None,
    leakage_rv: SolutionScript | None = None,
    debug_callback_rv: Any = None,
    eval_rv: tuple[SolutionScript, EvaluationResult] | None = None,
    is_improvement_rv: bool = True,
) -> dict[str, Any]:
    """Build a dict of mock objects for all run_phase3 dependencies.

    Returns a dict of mock objects keyed by function name for assertions.
    """
    if ensembler_rv is None:
        ensembler_rv = _make_ensemble_solution()
    if leakage_rv is None:
        leakage_rv = ensembler_rv
    if debug_callback_rv is None:
        debug_callback_rv = MagicMock()
    if eval_rv is None:
        scored = SolutionScript(
            content=ensembler_rv.content,
            phase=SolutionPhase.ENSEMBLE,
            score=0.85,
        )
        eval_rv = (scored, _make_eval_result(score=0.85))

    return {
        "invoke_ens_planner": AsyncMock(return_value=ens_planner_rv),
        "invoke_ensembler": AsyncMock(return_value=ensembler_rv),
        "check_and_fix_leakage": AsyncMock(return_value=leakage_rv),
        "make_debug_callback": MagicMock(return_value=debug_callback_rv),
        "evaluate_with_retry": AsyncMock(return_value=eval_rv),
        "is_improvement_or_equal": MagicMock(return_value=is_improvement_rv),
    }


async def _run_phase3(
    mocks: dict[str, Any],
    ensemble_rounds: int = 5,
    solutions: list[SolutionScript] | None = None,
    task: TaskDescription | None = None,
    config: PipelineConfig | None = None,
) -> Phase3Result:
    """Run run_phase3 with all dependencies mocked.

    Args:
        mocks: Dict from _patch_phase3_dependencies().
        ensemble_rounds: Number of ensemble rounds (R).
        solutions: Input solution scripts.
        task: TaskDescription (defaults to maximize).
        config: PipelineConfig (defaults with ensemble_rounds).

    Returns:
        The Phase3Result from the orchestration function.
    """
    from mle_star.phase3 import run_phase3

    if solutions is None:
        solutions = [
            _make_solution(content="sol_a", score=0.80),
            _make_solution(content="sol_b", score=0.75),
        ]
    if task is None:
        task = _make_task()
    if config is None:
        config = _make_config(ensemble_rounds=ensemble_rounds)

    client = AsyncMock()

    with (
        patch(f"{_MODULE}.invoke_ens_planner", mocks["invoke_ens_planner"]),
        patch(f"{_MODULE}.invoke_ensembler", mocks["invoke_ensembler"]),
        patch(f"{_MODULE}.check_and_fix_leakage", mocks["check_and_fix_leakage"]),
        patch(f"{_MODULE}.make_debug_callback", mocks["make_debug_callback"]),
        patch(f"{_MODULE}.evaluate_with_retry", mocks["evaluate_with_retry"]),
        patch(
            f"{_MODULE}.is_improvement_or_equal",
            mocks["is_improvement_or_equal"],
        ),
    ):
        return await run_phase3(
            client=client,
            task=task,
            config=config,
            solutions=solutions,
        )


# ===========================================================================
# TestRunPhase3IsAsync
# ===========================================================================


@pytest.mark.unit
class TestRunPhase3IsAsync:
    """run_phase3 is an async function returning Phase3Result."""

    def test_is_coroutine_function(self) -> None:
        """run_phase3 is defined as an async function."""
        from mle_star.phase3 import run_phase3

        assert asyncio.iscoroutinefunction(run_phase3)

    async def test_returns_phase3_result(self) -> None:
        """run_phase3 returns a Phase3Result instance on normal invocation."""
        mocks = _patch_phase3_dependencies()
        result = await _run_phase3(mocks, ensemble_rounds=1)

        assert isinstance(result, Phase3Result)


# ===========================================================================
# TestRunPhase3InputValidation
# ===========================================================================


@pytest.mark.unit
class TestRunPhase3InputValidation:
    """run_phase3 raises ValueError for empty solutions list."""

    async def test_raises_on_empty_solutions(self) -> None:
        """Raises ValueError when solutions list is empty."""
        from mle_star.phase3 import run_phase3

        client = AsyncMock()
        task = _make_task()
        config = _make_config()

        with pytest.raises(ValueError):
            await run_phase3(
                client=client,
                task=task,
                config=config,
                solutions=[],
            )

    async def test_does_not_raise_on_single_solution(self) -> None:
        """Does NOT raise for a single solution (handled as skip, not error)."""
        from mle_star.phase3 import run_phase3

        client = AsyncMock()
        task = _make_task()
        config = _make_config()
        sol = _make_solution(score=0.80)

        # Should not raise -- single solution is a valid skip case
        result = await run_phase3(
            client=client,
            task=task,
            config=config,
            solutions=[sol],
        )

        assert isinstance(result, Phase3Result)

    async def test_does_not_invoke_agents_on_empty_solutions(self) -> None:
        """No agents are invoked when solutions is empty."""
        from mle_star.phase3 import run_phase3

        client = AsyncMock()
        task = _make_task()
        config = _make_config()

        with pytest.raises(ValueError):
            await run_phase3(
                client=client,
                task=task,
                config=config,
                solutions=[],
            )

        client.send_message.assert_not_called()


# ===========================================================================
# TestRunPhase3SingleSolutionSkip (REQ-P3-018)
# ===========================================================================


@pytest.mark.unit
class TestRunPhase3SingleSolutionSkip:
    """Single solution skips ensemble and returns immediately (REQ-P3-018)."""

    async def test_single_solution_returns_phase3_result(self) -> None:
        """Returns a Phase3Result with the single solution as best_ensemble."""
        from mle_star.phase3 import run_phase3

        client = AsyncMock()
        task = _make_task()
        config = _make_config(ensemble_rounds=5)
        sol = _make_solution(content="only_solution", score=0.80)

        result = await run_phase3(
            client=client,
            task=task,
            config=config,
            solutions=[sol],
        )

        assert isinstance(result, Phase3Result)

    async def test_single_solution_best_ensemble_is_input(self) -> None:
        """best_ensemble is the single input solution."""
        from mle_star.phase3 import run_phase3

        client = AsyncMock()
        task = _make_task()
        config = _make_config(ensemble_rounds=5)
        sol = _make_solution(content="only_solution", score=0.80)

        result = await run_phase3(
            client=client,
            task=task,
            config=config,
            solutions=[sol],
        )

        assert result.best_ensemble.content == "only_solution"

    async def test_single_solution_best_ensemble_score_matches(self) -> None:
        """best_ensemble_score matches the single solution's score."""
        from mle_star.phase3 import run_phase3

        client = AsyncMock()
        task = _make_task()
        config = _make_config(ensemble_rounds=5)
        sol = _make_solution(content="only_solution", score=0.80)

        result = await run_phase3(
            client=client,
            task=task,
            config=config,
            solutions=[sol],
        )

        assert result.best_ensemble_score == 0.80

    async def test_single_solution_input_solutions_preserved(self) -> None:
        """input_solutions contains the original single solution."""
        from mle_star.phase3 import run_phase3

        client = AsyncMock()
        task = _make_task()
        config = _make_config(ensemble_rounds=5)
        sol = _make_solution(content="only_solution", score=0.80)

        result = await run_phase3(
            client=client,
            task=task,
            config=config,
            solutions=[sol],
        )

        assert len(result.input_solutions) == 1
        assert result.input_solutions[0].content == "only_solution"

    async def test_single_solution_empty_plans(self) -> None:
        """ensemble_plans is empty for single solution skip."""
        from mle_star.phase3 import run_phase3

        client = AsyncMock()
        task = _make_task()
        config = _make_config(ensemble_rounds=5)
        sol = _make_solution(content="only_solution", score=0.80)

        result = await run_phase3(
            client=client,
            task=task,
            config=config,
            solutions=[sol],
        )

        assert result.ensemble_plans == []

    async def test_single_solution_empty_scores(self) -> None:
        """ensemble_scores is empty for single solution skip."""
        from mle_star.phase3 import run_phase3

        client = AsyncMock()
        task = _make_task()
        config = _make_config(ensemble_rounds=5)
        sol = _make_solution(content="only_solution", score=0.80)

        result = await run_phase3(
            client=client,
            task=task,
            config=config,
            solutions=[sol],
        )

        assert result.ensemble_scores == []

    async def test_single_solution_does_not_invoke_ens_planner(self) -> None:
        """invoke_ens_planner is NOT called for single solution."""
        from mle_star.phase3 import run_phase3

        client = AsyncMock()
        task = _make_task()
        config = _make_config(ensemble_rounds=5)
        sol = _make_solution(content="only_solution", score=0.80)

        with patch(f"{_MODULE}.invoke_ens_planner") as mock_planner:
            await run_phase3(
                client=client,
                task=task,
                config=config,
                solutions=[sol],
            )

        mock_planner.assert_not_called()

    async def test_single_solution_does_not_invoke_ensembler(self) -> None:
        """invoke_ensembler is NOT called for single solution."""
        from mle_star.phase3 import run_phase3

        client = AsyncMock()
        task = _make_task()
        config = _make_config(ensemble_rounds=5)
        sol = _make_solution(content="only_solution", score=0.80)

        with patch(f"{_MODULE}.invoke_ensembler") as mock_ensembler:
            await run_phase3(
                client=client,
                task=task,
                config=config,
                solutions=[sol],
            )

        mock_ensembler.assert_not_called()


# ===========================================================================
# TestRunPhase3NormalFlow (REQ-P3-019 through REQ-P3-024)
# ===========================================================================


@pytest.mark.unit
class TestRunPhase3NormalFlow:
    """Normal flow executes R rounds: plan -> implement -> leakage -> evaluate."""

    @pytest.mark.parametrize("r_rounds", [1, 2, 3, 5])
    async def test_ens_planner_called_r_times(self, r_rounds: int) -> None:
        """invoke_ens_planner is called exactly R times."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3(mocks, ensemble_rounds=r_rounds)

        assert mocks["invoke_ens_planner"].call_count == r_rounds

    @pytest.mark.parametrize("r_rounds", [1, 2, 3, 5])
    async def test_ensembler_called_r_times(self, r_rounds: int) -> None:
        """invoke_ensembler is called exactly R times."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3(mocks, ensemble_rounds=r_rounds)

        assert mocks["invoke_ensembler"].call_count == r_rounds

    async def test_round_0_planner_receives_empty_history(self) -> None:
        """First round passes empty plans and empty scores to ens_planner."""
        planner_calls: list[tuple[Any, ...]] = []

        async def capture_planner(*args: Any, **kwargs: Any) -> str:
            planner_calls.append((args, kwargs))
            return "plan_r0"

        mocks = _patch_phase3_dependencies()
        mocks["invoke_ens_planner"] = AsyncMock(side_effect=capture_planner)

        await _run_phase3(mocks, ensemble_rounds=1)

        first_call = planner_calls[0]
        # plans argument (2nd positional or kwarg)
        plans_arg = (
            first_call[0][1]
            if len(first_call[0]) > 1
            else first_call[1].get("plans", [])
        )
        scores_arg = (
            first_call[0][2]
            if len(first_call[0]) > 2
            else first_call[1].get("scores", [])
        )
        assert len(plans_arg) == 0
        assert len(scores_arg) == 0

    async def test_ensembler_receives_plan_from_planner(self) -> None:
        """invoke_ensembler receives the plan returned by invoke_ens_planner."""
        mocks = _patch_phase3_dependencies(ens_planner_rv="stacking approach")

        await _run_phase3(mocks, ensemble_rounds=1)

        ensembler_call = mocks["invoke_ensembler"].call_args
        call_str = str(ensembler_call)
        assert "stacking approach" in call_str

    async def test_ensembler_receives_solutions(self) -> None:
        """invoke_ensembler receives the input solutions list."""
        solutions = [
            _make_solution(content="sol_alpha", score=0.80),
            _make_solution(content="sol_beta", score=0.75),
        ]
        mocks = _patch_phase3_dependencies()

        await _run_phase3(mocks, ensemble_rounds=1, solutions=solutions)

        ensembler_call = mocks["invoke_ensembler"].call_args
        call_str = str(ensembler_call)
        assert "sol_alpha" in call_str or "sol_beta" in call_str

    async def test_leakage_check_called_for_each_round(self) -> None:
        """check_and_fix_leakage is called once per round."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3(mocks, ensemble_rounds=3)

        assert mocks["check_and_fix_leakage"].call_count == 3

    async def test_evaluate_with_retry_called_for_each_round(self) -> None:
        """evaluate_with_retry is called once per round."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3(mocks, ensemble_rounds=3)

        assert mocks["evaluate_with_retry"].call_count == 3


# ===========================================================================
# TestRunPhase3HistoryAccumulation (REQ-P3-023)
# ===========================================================================


@pytest.mark.unit
class TestRunPhase3HistoryAccumulation:
    """History grows each round: accumulated plans and scores passed to ens_planner."""

    async def test_round_1_receives_round_0_history(self) -> None:
        """After round 0, ens_planner at round 1 receives one plan and one score."""
        planner_calls: list[tuple[Any, ...]] = []
        call_count = 0

        async def capture_planner(*args: Any, **kwargs: Any) -> str:
            nonlocal call_count
            planner_calls.append((args, kwargs))
            call_count += 1
            return f"plan_r{call_count - 1}"

        ensemble_sol = _make_ensemble_solution(content="ens_code")
        scored_sol = SolutionScript(
            content="ens_code", phase=SolutionPhase.ENSEMBLE, score=0.85
        )
        eval_result = _make_eval_result(score=0.85)

        mocks = _patch_phase3_dependencies()
        mocks["invoke_ens_planner"] = AsyncMock(side_effect=capture_planner)
        mocks["invoke_ensembler"] = AsyncMock(return_value=ensemble_sol)
        mocks["check_and_fix_leakage"] = AsyncMock(return_value=ensemble_sol)
        mocks["evaluate_with_retry"] = AsyncMock(return_value=(scored_sol, eval_result))

        await _run_phase3(mocks, ensemble_rounds=2)

        # Second call should have 1 plan and 1 score
        second_call = planner_calls[1]
        plans_arg = (
            second_call[0][1]
            if len(second_call[0]) > 1
            else second_call[1].get("plans", [])
        )
        scores_arg = (
            second_call[0][2]
            if len(second_call[0]) > 2
            else second_call[1].get("scores", [])
        )
        assert len(plans_arg) == 1
        assert len(scores_arg) == 1

    async def test_history_grows_by_one_each_round(self) -> None:
        """History size grows by 1 with each round: 0, 1, 2, ..."""
        planner_calls: list[tuple[Any, ...]] = []
        call_idx = 0

        async def capture_planner(*args: Any, **kwargs: Any) -> str:
            nonlocal call_idx
            planner_calls.append((args, kwargs))
            call_idx += 1
            return f"plan_r{call_idx - 1}"

        ensemble_sol = _make_ensemble_solution()
        scored_sol = SolutionScript(
            content="ens", phase=SolutionPhase.ENSEMBLE, score=0.85
        )
        eval_result = _make_eval_result(score=0.85)

        mocks = _patch_phase3_dependencies()
        mocks["invoke_ens_planner"] = AsyncMock(side_effect=capture_planner)
        mocks["invoke_ensembler"] = AsyncMock(return_value=ensemble_sol)
        mocks["check_and_fix_leakage"] = AsyncMock(return_value=ensemble_sol)
        mocks["evaluate_with_retry"] = AsyncMock(return_value=(scored_sol, eval_result))

        await _run_phase3(mocks, ensemble_rounds=4)

        for r in range(4):
            c = planner_calls[r]
            plans_arg = c[0][1] if len(c[0]) > 1 else c[1].get("plans", [])
            assert len(plans_arg) == r

    async def test_accumulated_plans_contain_actual_plan_texts(self) -> None:
        """Accumulated plans passed to ens_planner contain the actual plan texts."""
        planner_calls: list[tuple[Any, ...]] = []
        plan_texts = ["voting_plan", "stacking_plan", "blending_plan"]
        plan_idx = 0

        async def capture_planner(*args: Any, **kwargs: Any) -> str:
            nonlocal plan_idx
            planner_calls.append((args, kwargs))
            plan = plan_texts[plan_idx]
            plan_idx += 1
            return plan

        ensemble_sol = _make_ensemble_solution()
        scored_sol = SolutionScript(
            content="ens", phase=SolutionPhase.ENSEMBLE, score=0.85
        )
        eval_result = _make_eval_result(score=0.85)

        mocks = _patch_phase3_dependencies()
        mocks["invoke_ens_planner"] = AsyncMock(side_effect=capture_planner)
        mocks["invoke_ensembler"] = AsyncMock(return_value=ensemble_sol)
        mocks["check_and_fix_leakage"] = AsyncMock(return_value=ensemble_sol)
        mocks["evaluate_with_retry"] = AsyncMock(return_value=(scored_sol, eval_result))

        await _run_phase3(mocks, ensemble_rounds=3)

        # Third call should have plans ["voting_plan", "stacking_plan"]
        third_call = planner_calls[2]
        plans_arg = (
            third_call[0][1]
            if len(third_call[0]) > 1
            else third_call[1].get("plans", [])
        )
        assert "voting_plan" in plans_arg
        assert "stacking_plan" in plans_arg

    async def test_accumulated_scores_contain_actual_scores(self) -> None:
        """Accumulated scores passed to ens_planner contain the actual eval scores."""
        planner_calls: list[tuple[Any, ...]] = []
        round_scores = [0.80, 0.85, 0.90]
        round_idx = 0

        async def capture_planner(*args: Any, **kwargs: Any) -> str:
            nonlocal round_idx
            planner_calls.append((args, kwargs))
            plan = f"plan_{round_idx}"
            round_idx += 1
            return plan

        async def eval_side_effect(
            *args: Any, **kwargs: Any
        ) -> tuple[SolutionScript, EvaluationResult]:
            idx = mocks["evaluate_with_retry"].call_count - 1
            s = round_scores[idx] if idx < len(round_scores) else 0.85
            sol = SolutionScript(content="ens", phase=SolutionPhase.ENSEMBLE, score=s)
            return (sol, _make_eval_result(score=s))

        ensemble_sol = _make_ensemble_solution()

        mocks = _patch_phase3_dependencies()
        mocks["invoke_ens_planner"] = AsyncMock(side_effect=capture_planner)
        mocks["invoke_ensembler"] = AsyncMock(return_value=ensemble_sol)
        mocks["check_and_fix_leakage"] = AsyncMock(return_value=ensemble_sol)
        mocks["evaluate_with_retry"] = AsyncMock(side_effect=eval_side_effect)

        await _run_phase3(mocks, ensemble_rounds=3)

        # Third call should have scores [0.80, 0.85]
        third_call = planner_calls[2]
        scores_arg = (
            third_call[0][2]
            if len(third_call[0]) > 2
            else third_call[1].get("scores", [])
        )
        assert 0.80 in scores_arg
        assert 0.85 in scores_arg


# ===========================================================================
# TestRunPhase3SafetyIntegration (REQ-P3-027, REQ-P3-028)
# ===========================================================================


@pytest.mark.unit
class TestRunPhase3SafetyIntegration:
    """Safety integration: leakage check before every eval, debug callback."""

    async def test_leakage_check_called_before_every_evaluation(self) -> None:
        """check_and_fix_leakage is called R times, once per round."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3(mocks, ensemble_rounds=3)

        assert mocks["check_and_fix_leakage"].call_count == 3

    async def test_make_debug_callback_called_once(self) -> None:
        """make_debug_callback is called exactly ONCE at the start (REQ-P3-028)."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3(mocks, ensemble_rounds=3)

        mocks["make_debug_callback"].assert_called_once()

    async def test_debug_callback_passed_to_evaluate_with_retry(self) -> None:
        """The debug callback from make_debug_callback is passed to evaluate_with_retry."""
        sentinel_callback = MagicMock(name="sentinel_debug_cb")
        mocks = _patch_phase3_dependencies(debug_callback_rv=sentinel_callback)

        await _run_phase3(mocks, ensemble_rounds=1)

        eval_call = mocks["evaluate_with_retry"].call_args
        # The debug callback should appear somewhere in the call args
        assert sentinel_callback in eval_call[0] or any(
            v is sentinel_callback for v in (eval_call[1] or {}).values()
        )

    async def test_leakage_check_receives_ensemble_solution(self) -> None:
        """check_and_fix_leakage receives the solution from invoke_ensembler."""
        ens_sol = _make_ensemble_solution(content="leakage_target_code")
        mocks = _patch_phase3_dependencies(ensembler_rv=ens_sol, leakage_rv=ens_sol)

        await _run_phase3(mocks, ensemble_rounds=1)

        leakage_call = mocks["check_and_fix_leakage"].call_args
        # First positional arg should be the ensemble solution
        passed_solution = (
            leakage_call[0][0] if leakage_call[0] else leakage_call[1].get("solution")
        )
        assert passed_solution.content == "leakage_target_code"

    async def test_evaluate_receives_leakage_checked_solution(self) -> None:
        """evaluate_with_retry receives the solution AFTER leakage check."""
        ens_sol = _make_ensemble_solution(content="before_leakage")
        leakage_fixed = _make_ensemble_solution(content="after_leakage_fix")
        scored = SolutionScript(
            content="after_leakage_fix", phase=SolutionPhase.ENSEMBLE, score=0.85
        )

        mocks = _patch_phase3_dependencies(
            ensembler_rv=ens_sol,
            leakage_rv=leakage_fixed,
        )
        mocks["evaluate_with_retry"] = AsyncMock(
            return_value=(scored, _make_eval_result(score=0.85))
        )

        await _run_phase3(mocks, ensemble_rounds=1)

        eval_call = mocks["evaluate_with_retry"].call_args
        passed_solution = (
            eval_call[0][0] if eval_call[0] else eval_call[1].get("solution")
        )
        assert passed_solution.content == "after_leakage_fix"


# ===========================================================================
# TestRunPhase3BestSelection (REQ-P3-025)
# ===========================================================================


@pytest.mark.unit
class TestRunPhase3BestSelection:
    """Best ensemble selected using is_improvement_or_equal (>= semantics)."""

    async def test_maximize_highest_score_wins(self) -> None:
        """For maximize, highest scoring ensemble is selected as best."""
        round_scores = [0.80, 0.90, 0.85]
        round_contents = ["ens_r0", "ens_r1", "ens_r2"]
        eval_idx = 0

        async def eval_side_effect(
            *args: Any, **kwargs: Any
        ) -> tuple[SolutionScript, EvaluationResult]:
            nonlocal eval_idx
            s = round_scores[eval_idx]
            c = round_contents[eval_idx]
            eval_idx += 1
            sol = SolutionScript(content=c, phase=SolutionPhase.ENSEMBLE, score=s)
            return (sol, _make_eval_result(score=s))

        mocks = _patch_phase3_dependencies()
        # Use real comparison for maximize
        mocks["is_improvement_or_equal"] = lambda new, old, d: (
            new >= old if d == MetricDirection.MAXIMIZE else new <= old
        )
        mocks["evaluate_with_retry"] = AsyncMock(side_effect=eval_side_effect)

        result = await _run_phase3(mocks, ensemble_rounds=3)

        assert result.best_ensemble_score == 0.90
        assert result.best_ensemble.content == "ens_r1"

    async def test_minimize_lowest_score_wins(self) -> None:
        """For minimize, lowest scoring ensemble is selected as best."""
        round_scores = [0.30, 0.20, 0.25]
        round_contents = ["ens_r0", "ens_r1", "ens_r2"]
        eval_idx = 0

        async def eval_side_effect(
            *args: Any, **kwargs: Any
        ) -> tuple[SolutionScript, EvaluationResult]:
            nonlocal eval_idx
            s = round_scores[eval_idx]
            c = round_contents[eval_idx]
            eval_idx += 1
            sol = SolutionScript(content=c, phase=SolutionPhase.ENSEMBLE, score=s)
            return (sol, _make_eval_result(score=s))

        task = _make_task(direction=MetricDirection.MINIMIZE)
        mocks = _patch_phase3_dependencies()
        mocks["is_improvement_or_equal"] = lambda new, old, d: (
            new >= old if d == MetricDirection.MAXIMIZE else new <= old
        )
        mocks["evaluate_with_retry"] = AsyncMock(side_effect=eval_side_effect)

        result = await _run_phase3(mocks, ensemble_rounds=3, task=task)

        assert result.best_ensemble_score == 0.20
        assert result.best_ensemble.content == "ens_r1"

    async def test_tie_last_occurrence_wins(self) -> None:
        """On tie, LAST occurrence wins (>= semantics picks the later one)."""
        round_scores = [0.85, 0.85, 0.85]
        round_contents = ["ens_first", "ens_second", "ens_third"]
        eval_idx = 0

        async def eval_side_effect(
            *args: Any, **kwargs: Any
        ) -> tuple[SolutionScript, EvaluationResult]:
            nonlocal eval_idx
            s = round_scores[eval_idx]
            c = round_contents[eval_idx]
            eval_idx += 1
            sol = SolutionScript(content=c, phase=SolutionPhase.ENSEMBLE, score=s)
            return (sol, _make_eval_result(score=s))

        mocks = _patch_phase3_dependencies()
        mocks["is_improvement_or_equal"] = lambda new, old, d: (
            new >= old if d == MetricDirection.MAXIMIZE else new <= old
        )
        mocks["evaluate_with_retry"] = AsyncMock(side_effect=eval_side_effect)

        result = await _run_phase3(mocks, ensemble_rounds=3)

        # Last occurrence (ens_third) wins due to >= semantics
        assert result.best_ensemble.content == "ens_third"
        assert result.best_ensemble_score == 0.85

    async def test_progressive_improvement_selects_last_best(self) -> None:
        """Progressive improvement: 0.80 -> 0.85 -> 0.90 selects round 2."""
        round_scores = [0.80, 0.85, 0.90]
        round_contents = ["ens_r0", "ens_r1", "ens_r2"]
        eval_idx = 0

        async def eval_side_effect(
            *args: Any, **kwargs: Any
        ) -> tuple[SolutionScript, EvaluationResult]:
            nonlocal eval_idx
            s = round_scores[eval_idx]
            c = round_contents[eval_idx]
            eval_idx += 1
            sol = SolutionScript(content=c, phase=SolutionPhase.ENSEMBLE, score=s)
            return (sol, _make_eval_result(score=s))

        mocks = _patch_phase3_dependencies()
        mocks["is_improvement_or_equal"] = lambda new, old, d: (
            new >= old if d == MetricDirection.MAXIMIZE else new <= old
        )
        mocks["evaluate_with_retry"] = AsyncMock(side_effect=eval_side_effect)

        result = await _run_phase3(mocks, ensemble_rounds=3)

        assert result.best_ensemble_score == 0.90
        assert result.best_ensemble.content == "ens_r2"


# ===========================================================================
# TestRunPhase3EnsPlannerFailure (REQ-P3-031)
# ===========================================================================


@pytest.mark.unit
class TestRunPhase3EnsPlannerFailure:
    """When ens_planner returns None, record placeholder and continue."""

    async def test_planner_none_records_failure_plan(self) -> None:
        """When invoke_ens_planner returns None, a failure placeholder plan is recorded."""
        mocks = _patch_phase3_dependencies(ens_planner_rv=None)

        result = await _run_phase3(mocks, ensemble_rounds=1)

        assert len(result.ensemble_plans) == 1
        # Plan should be some failure sentinel
        assert (
            "[ens_planner failed]" in result.ensemble_plans[0]
            or result.ensemble_plans[0] != ""
        )

    async def test_planner_none_records_none_score(self) -> None:
        """When ens_planner returns None, the corresponding score is None."""
        mocks = _patch_phase3_dependencies(ens_planner_rv=None)
        # Ensure ensembler is NOT called when planner fails
        mocks["invoke_ensembler"] = AsyncMock(return_value=None)
        mocks["evaluate_with_retry"] = AsyncMock(return_value=None)

        result = await _run_phase3(mocks, ensemble_rounds=1)

        assert len(result.ensemble_scores) == 1
        assert result.ensemble_scores[0] is None

    async def test_planner_none_continues_to_next_round(self) -> None:
        """When ens_planner returns None in round 0, round 1 still executes."""
        call_count = 0

        async def planner_side_effect(*args: Any, **kwargs: Any) -> str | None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None
            return "plan_round_1"

        mocks = _patch_phase3_dependencies()
        mocks["invoke_ens_planner"] = AsyncMock(side_effect=planner_side_effect)

        result = await _run_phase3(mocks, ensemble_rounds=2)

        assert len(result.ensemble_plans) == 2
        # Second round should have a valid plan
        assert result.ensemble_plans[1] == "plan_round_1"

    async def test_planner_none_does_not_call_ensembler(self) -> None:
        """When ens_planner returns None, ensembler is NOT invoked for that round."""
        mocks = _patch_phase3_dependencies(ens_planner_rv=None)

        await _run_phase3(mocks, ensemble_rounds=1)

        mocks["invoke_ensembler"].assert_not_called()

    async def test_planner_failure_history_propagated(self) -> None:
        """Failure plan and None score are included in history for next round."""
        planner_calls: list[tuple[Any, ...]] = []
        call_count = 0

        async def planner_side_effect(*args: Any, **kwargs: Any) -> str | None:
            nonlocal call_count
            planner_calls.append((args, kwargs))
            call_count += 1
            if call_count == 1:
                return None
            return "plan_round_1"

        mocks = _patch_phase3_dependencies()
        mocks["invoke_ens_planner"] = AsyncMock(side_effect=planner_side_effect)

        await _run_phase3(mocks, ensemble_rounds=2)

        # Second call should have 1 plan entry (the failure plan) and 1 score (None)
        second_call = planner_calls[1]
        plans_arg = (
            second_call[0][1]
            if len(second_call[0]) > 1
            else second_call[1].get("plans", [])
        )
        scores_arg = (
            second_call[0][2]
            if len(second_call[0]) > 2
            else second_call[1].get("scores", [])
        )
        assert len(plans_arg) == 1
        assert len(scores_arg) == 1
        assert scores_arg[0] is None


# ===========================================================================
# TestRunPhase3EnsemblerFailure (REQ-P3-030)
# ===========================================================================


@pytest.mark.unit
class TestRunPhase3EnsemblerFailure:
    """When ensembler returns None, record score=None and empty solution."""

    async def test_ensembler_none_records_none_score(self) -> None:
        """When invoke_ensembler returns None, score for that round is None."""
        mocks = _patch_phase3_dependencies()
        mocks["invoke_ensembler"] = AsyncMock(return_value=None)

        result = await _run_phase3(mocks, ensemble_rounds=1)

        assert len(result.ensemble_scores) == 1
        assert result.ensemble_scores[0] is None

    async def test_ensembler_none_does_not_evaluate(self) -> None:
        """When ensembler returns None, evaluate_with_retry is NOT called."""
        mocks = _patch_phase3_dependencies()
        mocks["invoke_ensembler"] = AsyncMock(return_value=None)

        await _run_phase3(mocks, ensemble_rounds=1)

        mocks["evaluate_with_retry"].assert_not_called()

    async def test_ensembler_none_does_not_call_leakage_check(self) -> None:
        """When ensembler returns None, check_and_fix_leakage is NOT called."""
        mocks = _patch_phase3_dependencies()
        mocks["invoke_ensembler"] = AsyncMock(return_value=None)

        await _run_phase3(mocks, ensemble_rounds=1)

        mocks["check_and_fix_leakage"].assert_not_called()

    async def test_ensembler_none_continues_to_next_round(self) -> None:
        """When ensembler returns None, loop continues to next round."""
        call_count = 0

        async def ensembler_side_effect(
            *args: Any, **kwargs: Any
        ) -> SolutionScript | None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None
            return _make_ensemble_solution(content="round_1_code")

        mocks = _patch_phase3_dependencies()
        mocks["invoke_ensembler"] = AsyncMock(side_effect=ensembler_side_effect)

        result = await _run_phase3(mocks, ensemble_rounds=2)

        assert len(result.ensemble_plans) == 2
        assert len(result.ensemble_scores) == 2

    async def test_ensembler_none_plan_still_recorded(self) -> None:
        """When ensembler returns None, the plan is still recorded."""
        mocks = _patch_phase3_dependencies(ens_planner_rv="a_valid_plan")
        mocks["invoke_ensembler"] = AsyncMock(return_value=None)

        result = await _run_phase3(mocks, ensemble_rounds=1)

        assert result.ensemble_plans[0] == "a_valid_plan"


# ===========================================================================
# TestRunPhase3EvaluationFailure (REQ-P3-029)
# ===========================================================================


@pytest.mark.unit
class TestRunPhase3EvaluationFailure:
    """When evaluation fails (is_error=True, score=None), record None and continue."""

    async def test_eval_error_records_none_score(self) -> None:
        """When evaluation returns is_error=True with score=None, records None."""
        failed_sol = _make_ensemble_solution(content="ens_code")
        failed_eval = _make_eval_result(score=None, is_error=True)

        mocks = _patch_phase3_dependencies()
        mocks["evaluate_with_retry"] = AsyncMock(return_value=(failed_sol, failed_eval))

        result = await _run_phase3(mocks, ensemble_rounds=1)

        assert len(result.ensemble_scores) == 1
        assert result.ensemble_scores[0] is None

    async def test_eval_error_continues_to_next_round(self) -> None:
        """Evaluation failure in round 0 does not prevent round 1."""
        eval_idx = 0

        async def eval_side_effect(
            *args: Any, **kwargs: Any
        ) -> tuple[SolutionScript, EvaluationResult]:
            nonlocal eval_idx
            eval_idx += 1
            if eval_idx == 1:
                # First round fails
                sol = _make_ensemble_solution()
                return (sol, _make_eval_result(score=None, is_error=True))
            # Second round succeeds
            sol = SolutionScript(
                content="ens_r1", phase=SolutionPhase.ENSEMBLE, score=0.90
            )
            return (sol, _make_eval_result(score=0.90))

        mocks = _patch_phase3_dependencies()
        mocks["evaluate_with_retry"] = AsyncMock(side_effect=eval_side_effect)
        mocks["is_improvement_or_equal"] = lambda new, old, d: (
            new >= old if d == MetricDirection.MAXIMIZE else new <= old
        )

        result = await _run_phase3(mocks, ensemble_rounds=2)

        assert len(result.ensemble_scores) == 2
        assert result.ensemble_scores[0] is None
        assert result.ensemble_scores[1] == 0.90

    async def test_eval_error_does_not_update_best(self) -> None:
        """Evaluation failure (score=None) does not update best ensemble."""
        solutions = [
            _make_solution(content="sol_a", score=0.80),
            _make_solution(content="sol_b", score=0.75),
        ]
        failed_sol = _make_ensemble_solution()
        failed_eval = _make_eval_result(score=None, is_error=True)

        mocks = _patch_phase3_dependencies()
        mocks["evaluate_with_retry"] = AsyncMock(return_value=(failed_sol, failed_eval))

        result = await _run_phase3(mocks, ensemble_rounds=1, solutions=solutions)

        # Since all rounds failed, should fall back to best input
        # best_ensemble_score should come from the best input solution
        assert result.best_ensemble_score == 0.80


# ===========================================================================
# TestRunPhase3AllRoundsFailed (REQ-P3-026)
# ===========================================================================


@pytest.mark.unit
class TestRunPhase3AllRoundsFailed:
    """When all rounds fail, fallback to best INPUT solution (REQ-P3-026)."""

    async def test_all_rounds_failed_fallback_to_best_input_maximize(self) -> None:
        """When all rounds fail, best_ensemble is the highest-scoring input solution."""
        solutions = [
            _make_solution(content="sol_worst", score=0.70),
            _make_solution(content="sol_best", score=0.90),
            _make_solution(content="sol_middle", score=0.80),
        ]
        failed_sol = _make_ensemble_solution()
        failed_eval = _make_eval_result(score=None, is_error=True)

        mocks = _patch_phase3_dependencies()
        mocks["evaluate_with_retry"] = AsyncMock(return_value=(failed_sol, failed_eval))

        result = await _run_phase3(mocks, ensemble_rounds=3, solutions=solutions)

        assert result.best_ensemble.content == "sol_best"
        assert result.best_ensemble_score == 0.90

    async def test_all_rounds_failed_fallback_to_best_input_minimize(self) -> None:
        """When all rounds fail for minimize, best_ensemble is the lowest-scoring input."""
        solutions = [
            _make_solution(content="sol_high", score=0.90),
            _make_solution(content="sol_low", score=0.10),
            _make_solution(content="sol_mid", score=0.50),
        ]
        task = _make_task(direction=MetricDirection.MINIMIZE)
        failed_sol = _make_ensemble_solution()
        failed_eval = _make_eval_result(score=None, is_error=True)

        mocks = _patch_phase3_dependencies()
        mocks["evaluate_with_retry"] = AsyncMock(return_value=(failed_sol, failed_eval))

        result = await _run_phase3(
            mocks, ensemble_rounds=3, solutions=solutions, task=task
        )

        assert result.best_ensemble.content == "sol_low"
        assert result.best_ensemble_score == 0.10

    async def test_all_planner_failures_fallback(self) -> None:
        """When all ens_planner calls return None, fallback to best input."""
        solutions = [
            _make_solution(content="sol_a", score=0.85),
            _make_solution(content="sol_b", score=0.80),
        ]

        mocks = _patch_phase3_dependencies(ens_planner_rv=None)

        result = await _run_phase3(mocks, ensemble_rounds=3, solutions=solutions)

        assert result.best_ensemble.content == "sol_a"
        assert result.best_ensemble_score == 0.85

    async def test_all_ensembler_failures_fallback(self) -> None:
        """When all ensembler calls return None, fallback to best input."""
        solutions = [
            _make_solution(content="sol_a", score=0.85),
            _make_solution(content="sol_b", score=0.80),
        ]

        mocks = _patch_phase3_dependencies()
        mocks["invoke_ensembler"] = AsyncMock(return_value=None)

        result = await _run_phase3(mocks, ensemble_rounds=2, solutions=solutions)

        assert result.best_ensemble.content == "sol_a"
        assert result.best_ensemble_score == 0.85

    async def test_all_rounds_failed_does_not_raise(self) -> None:
        """All-rounds-failed scenario returns a result, not an exception."""
        solutions = [
            _make_solution(content="sol_a", score=0.80),
            _make_solution(content="sol_b", score=0.75),
        ]

        mocks = _patch_phase3_dependencies(ens_planner_rv=None)

        result = await _run_phase3(mocks, ensemble_rounds=3, solutions=solutions)

        assert isinstance(result, Phase3Result)

    async def test_mixed_failures_first_success_wins(self) -> None:
        """Mix of planner failures and eval failures; single success is used."""
        eval_idx = 0

        async def planner_side_effect(*args: Any, **kwargs: Any) -> str | None:
            nonlocal eval_idx
            # First round: planner fails; second round: planner succeeds
            if eval_idx == 0:
                eval_idx += 1
                return None
            eval_idx += 1
            return "working_plan"

        scored_sol = SolutionScript(
            content="ens_success", phase=SolutionPhase.ENSEMBLE, score=0.88
        )

        solutions = [
            _make_solution(content="sol_a", score=0.80),
            _make_solution(content="sol_b", score=0.75),
        ]

        mocks = _patch_phase3_dependencies()
        mocks["invoke_ens_planner"] = AsyncMock(side_effect=planner_side_effect)
        mocks["evaluate_with_retry"] = AsyncMock(
            return_value=(scored_sol, _make_eval_result(score=0.88))
        )
        mocks["is_improvement_or_equal"] = lambda new, old, d: (
            new >= old if d == MetricDirection.MAXIMIZE else new <= old
        )

        result = await _run_phase3(mocks, ensemble_rounds=2, solutions=solutions)

        assert result.best_ensemble_score == 0.88


# ===========================================================================
# TestRunPhase3EnsembleAttemptRecords (REQ-P3-032, REQ-P3-033, REQ-P3-044)
# ===========================================================================


@pytest.mark.unit
class TestRunPhase3EnsembleAttemptRecords:
    """Exactly R records in ensemble_plans and ensemble_scores, ordered by round."""

    @pytest.mark.parametrize("r_rounds", [1, 2, 3, 5])
    async def test_ensemble_plans_has_r_entries(self, r_rounds: int) -> None:
        """ensemble_plans has exactly R entries."""
        mocks = _patch_phase3_dependencies()
        result = await _run_phase3(mocks, ensemble_rounds=r_rounds)

        assert len(result.ensemble_plans) == r_rounds

    @pytest.mark.parametrize("r_rounds", [1, 2, 3, 5])
    async def test_ensemble_scores_has_r_entries(self, r_rounds: int) -> None:
        """ensemble_scores has exactly R entries."""
        mocks = _patch_phase3_dependencies()
        result = await _run_phase3(mocks, ensemble_rounds=r_rounds)

        assert len(result.ensemble_scores) == r_rounds

    async def test_plans_and_scores_same_length(self) -> None:
        """ensemble_plans and ensemble_scores always have equal length."""
        mocks = _patch_phase3_dependencies()
        result = await _run_phase3(mocks, ensemble_rounds=4)

        assert len(result.ensemble_plans) == len(result.ensemble_scores)

    async def test_plans_ordered_by_round(self) -> None:
        """ensemble_plans are in round order (round 0 first, round R-1 last)."""
        plan_idx = 0

        async def planner_side_effect(*args: Any, **kwargs: Any) -> str:
            nonlocal plan_idx
            plan = f"plan_round_{plan_idx}"
            plan_idx += 1
            return plan

        mocks = _patch_phase3_dependencies()
        mocks["invoke_ens_planner"] = AsyncMock(side_effect=planner_side_effect)

        result = await _run_phase3(mocks, ensemble_rounds=3)

        assert result.ensemble_plans[0] == "plan_round_0"
        assert result.ensemble_plans[1] == "plan_round_1"
        assert result.ensemble_plans[2] == "plan_round_2"

    async def test_scores_ordered_by_round(self) -> None:
        """ensemble_scores are in round order."""
        round_scores = [0.80, 0.85, 0.90]
        eval_idx = 0

        async def eval_side_effect(
            *args: Any, **kwargs: Any
        ) -> tuple[SolutionScript, EvaluationResult]:
            nonlocal eval_idx
            s = round_scores[eval_idx]
            eval_idx += 1
            sol = SolutionScript(content="ens", phase=SolutionPhase.ENSEMBLE, score=s)
            return (sol, _make_eval_result(score=s))

        mocks = _patch_phase3_dependencies()
        mocks["evaluate_with_retry"] = AsyncMock(side_effect=eval_side_effect)

        result = await _run_phase3(mocks, ensemble_rounds=3)

        assert result.ensemble_scores == [0.80, 0.85, 0.90]

    async def test_failed_rounds_have_none_scores(self) -> None:
        """Failed rounds have None in ensemble_scores at their position."""
        eval_idx = 0

        async def eval_side_effect(
            *args: Any, **kwargs: Any
        ) -> tuple[SolutionScript, EvaluationResult]:
            nonlocal eval_idx
            eval_idx += 1
            if eval_idx == 2:
                sol = _make_ensemble_solution()
                return (sol, _make_eval_result(score=None, is_error=True))
            s = 0.85
            sol = SolutionScript(content="ens", phase=SolutionPhase.ENSEMBLE, score=s)
            return (sol, _make_eval_result(score=s))

        mocks = _patch_phase3_dependencies()
        mocks["evaluate_with_retry"] = AsyncMock(side_effect=eval_side_effect)

        result = await _run_phase3(mocks, ensemble_rounds=3)

        assert result.ensemble_scores[0] == 0.85
        assert result.ensemble_scores[1] is None
        assert result.ensemble_scores[2] == 0.85


# ===========================================================================
# TestRunPhase3ResultConstruction (REQ-P3-034, REQ-P3-035)
# ===========================================================================


@pytest.mark.unit
class TestRunPhase3ResultConstruction:
    """Phase3Result fields are correctly populated."""

    async def test_input_solutions_preserved(self) -> None:
        """input_solutions contains the original solutions list."""
        solutions = [
            _make_solution(content="preserved_sol_a", score=0.80),
            _make_solution(content="preserved_sol_b", score=0.75),
        ]
        mocks = _patch_phase3_dependencies()

        result = await _run_phase3(mocks, ensemble_rounds=1, solutions=solutions)

        assert len(result.input_solutions) == 2
        assert result.input_solutions[0].content == "preserved_sol_a"
        assert result.input_solutions[1].content == "preserved_sol_b"

    async def test_input_solutions_not_mutated(self) -> None:
        """Original solutions list is not mutated by run_phase3."""
        solutions = [
            _make_solution(content="immutable_a", score=0.80),
            _make_solution(content="immutable_b", score=0.75),
        ]
        original_contents = [s.content for s in solutions]

        mocks = _patch_phase3_dependencies()

        await _run_phase3(mocks, ensemble_rounds=2, solutions=solutions)

        assert [s.content for s in solutions] == original_contents

    async def test_best_ensemble_is_solution_script(self) -> None:
        """best_ensemble is a SolutionScript instance."""
        mocks = _patch_phase3_dependencies()
        result = await _run_phase3(mocks, ensemble_rounds=1)

        assert isinstance(result.best_ensemble, SolutionScript)

    async def test_best_ensemble_score_is_float(self) -> None:
        """best_ensemble_score is a float."""
        mocks = _patch_phase3_dependencies()
        result = await _run_phase3(mocks, ensemble_rounds=1)

        assert isinstance(result.best_ensemble_score, float)

    async def test_ensemble_plans_are_strings(self) -> None:
        """All entries in ensemble_plans are strings."""
        mocks = _patch_phase3_dependencies()
        result = await _run_phase3(mocks, ensemble_rounds=3)

        for plan in result.ensemble_plans:
            assert isinstance(plan, str)

    async def test_ensemble_scores_are_float_or_none(self) -> None:
        """All entries in ensemble_scores are float or None."""
        mocks = _patch_phase3_dependencies()
        result = await _run_phase3(mocks, ensemble_rounds=3)

        for score in result.ensemble_scores:
            assert score is None or isinstance(score, float)

    async def test_result_is_frozen(self) -> None:
        """Phase3Result is frozen (immutable)."""
        mocks = _patch_phase3_dependencies()
        result = await _run_phase3(mocks, ensemble_rounds=1)

        with pytest.raises(ValidationError):
            result.best_ensemble_score = 999.0  # type: ignore[misc]

    async def test_ensemble_plans_length_equals_rounds(self) -> None:
        """len(ensemble_plans) == config.ensemble_rounds."""
        mocks = _patch_phase3_dependencies()
        result = await _run_phase3(mocks, ensemble_rounds=4)

        assert len(result.ensemble_plans) == 4

    async def test_ensemble_scores_length_equals_rounds(self) -> None:
        """len(ensemble_scores) == config.ensemble_rounds."""
        mocks = _patch_phase3_dependencies()
        result = await _run_phase3(mocks, ensemble_rounds=4)

        assert len(result.ensemble_scores) == 4


# ===========================================================================
# TestRunPhase3PropertyBased
# ===========================================================================


@pytest.mark.unit
class TestRunPhase3PropertyBased:
    """Property-based tests for run_phase3 invariants."""

    @given(
        r_rounds=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=15)
    async def test_plans_length_equals_r(self, r_rounds: int) -> None:
        """ensemble_plans always has exactly R entries."""
        mocks = _patch_phase3_dependencies()
        result = await _run_phase3(mocks, ensemble_rounds=r_rounds)

        assert len(result.ensemble_plans) == r_rounds

    @given(
        r_rounds=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=15)
    async def test_scores_length_equals_r(self, r_rounds: int) -> None:
        """ensemble_scores always has exactly R entries."""
        mocks = _patch_phase3_dependencies()
        result = await _run_phase3(mocks, ensemble_rounds=r_rounds)

        assert len(result.ensemble_scores) == r_rounds

    @given(
        r_rounds=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=15)
    async def test_plans_scores_same_length(self, r_rounds: int) -> None:
        """ensemble_plans and ensemble_scores always have equal length."""
        mocks = _patch_phase3_dependencies()
        result = await _run_phase3(mocks, ensemble_rounds=r_rounds)

        assert len(result.ensemble_plans) == len(result.ensemble_scores)

    @given(
        r_rounds=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=15)
    async def test_result_always_has_best_ensemble(self, r_rounds: int) -> None:
        """Phase3Result always has a best_ensemble (never None)."""
        mocks = _patch_phase3_dependencies()
        result = await _run_phase3(mocks, ensemble_rounds=r_rounds)

        assert result.best_ensemble is not None
        assert isinstance(result.best_ensemble, SolutionScript)

    @given(
        r_rounds=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=15)
    async def test_best_ensemble_score_is_always_float(self, r_rounds: int) -> None:
        """best_ensemble_score is always a float (never None)."""
        mocks = _patch_phase3_dependencies()
        result = await _run_phase3(mocks, ensemble_rounds=r_rounds)

        assert isinstance(result.best_ensemble_score, float)

    @given(
        n_solutions=st.integers(min_value=2, max_value=6),
    )
    @settings(max_examples=15)
    async def test_input_solutions_preserved_for_any_count(
        self, n_solutions: int
    ) -> None:
        """input_solutions always has the same length as the input solutions list."""
        solutions = [
            _make_solution(content=f"sol_{i}", score=0.50 + 0.05 * i)
            for i in range(n_solutions)
        ]
        mocks = _patch_phase3_dependencies()
        result = await _run_phase3(mocks, ensemble_rounds=1, solutions=solutions)

        assert len(result.input_solutions) == n_solutions

    @given(
        r_rounds=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=15)
    async def test_all_rounds_failed_still_produces_valid_result(
        self, r_rounds: int
    ) -> None:
        """When all rounds fail, result is still valid with a fallback solution."""
        solutions = [
            _make_solution(content="fb_sol_a", score=0.85),
            _make_solution(content="fb_sol_b", score=0.80),
        ]
        failed_sol = _make_ensemble_solution()
        failed_eval = _make_eval_result(score=None, is_error=True)

        mocks = _patch_phase3_dependencies()
        mocks["evaluate_with_retry"] = AsyncMock(return_value=(failed_sol, failed_eval))

        result = await _run_phase3(mocks, ensemble_rounds=r_rounds, solutions=solutions)

        assert isinstance(result, Phase3Result)
        assert isinstance(result.best_ensemble_score, float)
        assert result.best_ensemble_score == 0.85

    @given(
        r_rounds=st.integers(min_value=1, max_value=4),
        score=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=20)
    async def test_best_score_at_least_as_good_as_best_input_maximize(
        self, r_rounds: int, score: float
    ) -> None:
        """For maximize, best_ensemble_score >= max(input solution scores)."""
        solutions = [
            _make_solution(content="sol_a", score=score),
            _make_solution(content="sol_b", score=score - 0.1 if score >= 0.1 else 0.0),
        ]
        scored_sol = SolutionScript(
            content="ens", phase=SolutionPhase.ENSEMBLE, score=score + 0.05
        )

        mocks = _patch_phase3_dependencies()
        mocks["evaluate_with_retry"] = AsyncMock(
            return_value=(scored_sol, _make_eval_result(score=score + 0.05))
        )
        mocks["is_improvement_or_equal"] = lambda new, old, d: (
            new >= old if d == MetricDirection.MAXIMIZE else new <= old
        )

        result = await _run_phase3(mocks, ensemble_rounds=r_rounds, solutions=solutions)

        max_input = max(s.score for s in solutions if s.score is not None)
        assert result.best_ensemble_score >= max_input


# ===========================================================================
# Edge cases and additional scenarios
# ===========================================================================


@pytest.mark.unit
class TestRunPhase3EdgeCases:
    """Edge case tests for run_phase3."""

    async def test_single_round(self) -> None:
        """Works correctly with ensemble_rounds=1."""
        mocks = _patch_phase3_dependencies()
        result = await _run_phase3(mocks, ensemble_rounds=1)

        assert isinstance(result, Phase3Result)
        assert len(result.ensemble_plans) == 1
        assert len(result.ensemble_scores) == 1

    async def test_input_solutions_with_none_scores(self) -> None:
        """Handles input solutions where some have score=None."""
        solutions = [
            _make_solution(content="scored_sol", score=0.80),
            _make_solution(content="unscored_sol", score=None),
        ]
        mocks = _patch_phase3_dependencies()

        result = await _run_phase3(mocks, ensemble_rounds=1, solutions=solutions)

        assert isinstance(result, Phase3Result)

    async def test_many_solutions_accepted(self) -> None:
        """Works correctly with many input solutions (e.g., 10)."""
        solutions = [
            _make_solution(content=f"sol_{i}", score=0.50 + 0.05 * i) for i in range(10)
        ]
        mocks = _patch_phase3_dependencies()

        result = await _run_phase3(mocks, ensemble_rounds=1, solutions=solutions)

        assert len(result.input_solutions) == 10

    async def test_exactly_two_solutions(self) -> None:
        """Works correctly with the minimum 2 solutions for ensemble."""
        solutions = [
            _make_solution(content="sol_a", score=0.80),
            _make_solution(content="sol_b", score=0.75),
        ]
        mocks = _patch_phase3_dependencies()

        result = await _run_phase3(mocks, ensemble_rounds=2, solutions=solutions)

        assert isinstance(result, Phase3Result)
        assert len(result.input_solutions) == 2

    async def test_negative_scores(self) -> None:
        """Works correctly with negative evaluation scores."""
        solutions = [
            _make_solution(content="sol_a", score=-0.50),
            _make_solution(content="sol_b", score=-0.80),
        ]
        scored_sol = SolutionScript(
            content="ens", phase=SolutionPhase.ENSEMBLE, score=-0.30
        )
        mocks = _patch_phase3_dependencies()
        mocks["evaluate_with_retry"] = AsyncMock(
            return_value=(scored_sol, _make_eval_result(score=-0.30))
        )
        mocks["is_improvement_or_equal"] = lambda new, old, d: (
            new >= old if d == MetricDirection.MAXIMIZE else new <= old
        )

        result = await _run_phase3(mocks, ensemble_rounds=1, solutions=solutions)

        assert result.best_ensemble_score == -0.30

    async def test_zero_scores(self) -> None:
        """Works correctly when scores are 0.0."""
        solutions = [
            _make_solution(content="sol_a", score=0.0),
            _make_solution(content="sol_b", score=0.0),
        ]
        scored_sol = SolutionScript(
            content="ens", phase=SolutionPhase.ENSEMBLE, score=0.0
        )
        mocks = _patch_phase3_dependencies()
        mocks["evaluate_with_retry"] = AsyncMock(
            return_value=(scored_sol, _make_eval_result(score=0.0))
        )
        mocks["is_improvement_or_equal"] = lambda new, old, d: (
            new >= old if d == MetricDirection.MAXIMIZE else new <= old
        )

        result = await _run_phase3(mocks, ensemble_rounds=1, solutions=solutions)

        assert result.best_ensemble_score == 0.0


# ===========================================================================
# TestRunPhase3LeakageModifiesSolution
# ===========================================================================


@pytest.mark.unit
class TestRunPhase3LeakageModifiesSolution:
    """Tests that the leakage-fixed solution (not original) is used for evaluation."""

    async def test_leakage_fixed_solution_evaluated(self) -> None:
        """evaluate_with_retry receives the post-leakage solution, not the original."""
        original_ens = _make_ensemble_solution(content="original_ens_code")
        leakage_fixed = _make_ensemble_solution(content="fixed_by_leakage")
        scored = SolutionScript(
            content="fixed_by_leakage", phase=SolutionPhase.ENSEMBLE, score=0.85
        )

        mocks = _patch_phase3_dependencies(
            ensembler_rv=original_ens,
            leakage_rv=leakage_fixed,
        )
        mocks["evaluate_with_retry"] = AsyncMock(
            return_value=(scored, _make_eval_result(score=0.85))
        )

        await _run_phase3(mocks, ensemble_rounds=1)

        eval_call = mocks["evaluate_with_retry"].call_args
        passed_solution = (
            eval_call[0][0] if eval_call[0] else eval_call[1].get("solution")
        )
        assert passed_solution.content == "fixed_by_leakage"


# ===========================================================================
# TestRunPhase3RoundSequencing
# ===========================================================================


@pytest.mark.unit
class TestRunPhase3RoundSequencing:
    """Tests that the round sequence is plan -> implement -> leakage -> eval."""

    async def test_leakage_not_called_when_ensembler_fails(self) -> None:
        """check_and_fix_leakage is skipped when ensembler returns None."""
        mocks = _patch_phase3_dependencies()
        mocks["invoke_ensembler"] = AsyncMock(return_value=None)

        await _run_phase3(mocks, ensemble_rounds=1)

        mocks["check_and_fix_leakage"].assert_not_called()
        mocks["evaluate_with_retry"].assert_not_called()

    async def test_eval_not_called_when_ensembler_fails(self) -> None:
        """evaluate_with_retry is skipped when ensembler returns None."""
        mocks = _patch_phase3_dependencies()
        mocks["invoke_ensembler"] = AsyncMock(return_value=None)

        await _run_phase3(mocks, ensemble_rounds=1)

        mocks["evaluate_with_retry"].assert_not_called()

    async def test_planner_called_before_ensembler(self) -> None:
        """invoke_ens_planner is called before invoke_ensembler in each round."""
        call_order: list[str] = []

        async def track_planner(*a: Any, **kw: Any) -> str:
            call_order.append("planner")
            return "plan"

        async def track_ensembler(*a: Any, **kw: Any) -> SolutionScript:
            call_order.append("ensembler")
            return _make_ensemble_solution()

        mocks = _patch_phase3_dependencies()
        mocks["invoke_ens_planner"] = AsyncMock(side_effect=track_planner)
        mocks["invoke_ensembler"] = AsyncMock(side_effect=track_ensembler)

        await _run_phase3(mocks, ensemble_rounds=1)

        assert call_order.index("planner") < call_order.index("ensembler")

    async def test_leakage_called_before_eval(self) -> None:
        """check_and_fix_leakage is called before evaluate_with_retry."""
        call_order: list[str] = []

        async def track_leakage(*a: Any, **kw: Any) -> SolutionScript:
            call_order.append("leakage")
            return _make_ensemble_solution()

        async def track_eval(
            *a: Any, **kw: Any
        ) -> tuple[SolutionScript, EvaluationResult]:
            call_order.append("eval")
            sol = SolutionScript(
                content="ens", phase=SolutionPhase.ENSEMBLE, score=0.85
            )
            return (sol, _make_eval_result(score=0.85))

        mocks = _patch_phase3_dependencies()
        mocks["check_and_fix_leakage"] = AsyncMock(side_effect=track_leakage)
        mocks["evaluate_with_retry"] = AsyncMock(side_effect=track_eval)

        await _run_phase3(mocks, ensemble_rounds=1)

        assert call_order.index("leakage") < call_order.index("eval")


# ===========================================================================
# Parametrized: various R values
# ===========================================================================


@pytest.mark.unit
class TestRunPhase3Parametrized:
    """Parametrized tests across various R values."""

    @pytest.mark.parametrize("r", [1, 2, 3, 4, 5])
    async def test_returns_valid_phase3_result_for_various_r(self, r: int) -> None:
        """run_phase3 returns a valid Phase3Result for R={1,2,3,4,5}."""
        mocks = _patch_phase3_dependencies()
        result = await _run_phase3(mocks, ensemble_rounds=r)

        assert isinstance(result, Phase3Result)
        assert len(result.ensemble_plans) == r
        assert len(result.ensemble_scores) == r

    @pytest.mark.parametrize(
        "direction",
        [MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE],
        ids=["maximize", "minimize"],
    )
    async def test_works_for_both_metric_directions(
        self, direction: MetricDirection
    ) -> None:
        """run_phase3 handles both maximize and minimize correctly."""
        task = _make_task(direction=direction)
        mocks = _patch_phase3_dependencies()

        result = await _run_phase3(mocks, ensemble_rounds=2, task=task)

        assert isinstance(result, Phase3Result)

    @pytest.mark.parametrize(
        "n_solutions",
        [2, 3, 5, 10],
        ids=["2-solutions", "3-solutions", "5-solutions", "10-solutions"],
    )
    async def test_works_for_various_solution_counts(self, n_solutions: int) -> None:
        """run_phase3 handles various numbers of input solutions."""
        solutions = [
            _make_solution(content=f"sol_{i}", score=0.50 + 0.05 * i)
            for i in range(n_solutions)
        ]
        mocks = _patch_phase3_dependencies()

        result = await _run_phase3(mocks, ensemble_rounds=1, solutions=solutions)

        assert len(result.input_solutions) == n_solutions
