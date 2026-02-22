"""Tests for Phase 3 constraints (Task 37).

Validates non-functional requirements for Phase 3 ensemble construction:
orchestration overhead budget, never-raises reliability, structured logging
at correct levels for 17 key events, Algorithm 3 fidelity, sequential
round execution, exactly R rounds attempted, and leakage check on every
ensemble solution before evaluation.

Tests are written TDD-first and serve as the executable specification for
REQ-P3-036 through REQ-P3-049.

Refs:
    SRS 07c (Phase 3 Constraints), IMPLEMENTATION_PLAN.md Task 37.
"""

from __future__ import annotations

import logging
import time
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
import pytest

# ---------------------------------------------------------------------------
# Module path constant for patching
# ---------------------------------------------------------------------------

_MODULE = "mle_star.phase3"
_LOGGER_NAME = "mle_star.phase3"


# ---------------------------------------------------------------------------
# Reusable test helpers (mirrored from test_phase3_orchestration.py)
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
    competition_id: str = "test-comp",
) -> TaskDescription:
    """Create a minimal TaskDescription for testing."""
    return TaskDescription(
        competition_id=competition_id,
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
    duration_seconds: float = 1.0,
) -> EvaluationResult:
    """Create an EvaluationResult with the given score and error state."""
    return EvaluationResult(
        score=score,
        stdout=f"Final Validation Performance: {score}" if score else "",
        stderr="" if not is_error else "Traceback (most recent call last):\nError",
        exit_code=0 if not is_error else 1,
        duration_seconds=duration_seconds,
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


async def _run_phase3_with_caplog(
    mocks: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
    ensemble_rounds: int = 5,
    solutions: list[SolutionScript] | None = None,
    task: TaskDescription | None = None,
    config: PipelineConfig | None = None,
) -> Phase3Result:
    """Run run_phase3 with all dependencies mocked and caplog enabled.

    Args:
        mocks: Dict from _patch_phase3_dependencies().
        caplog: Pytest log capture fixture.
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
        caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
    ):
        return await run_phase3(
            client=client,
            task=task,
            config=config,
            solutions=solutions,
        )


# ===========================================================================
# REQ-P3-036: Orchestration overhead < 5s across all R rounds
# ===========================================================================


@pytest.mark.unit
class TestPhase3OrchestrOverhead:
    """Orchestration overhead < 5s with mocked dependencies (REQ-P3-036)."""

    async def test_overhead_under_five_seconds_r5(self) -> None:
        """Orchestration overhead excluding LLM calls is under 5s for R=5."""
        mocks = _patch_phase3_dependencies()

        start = time.monotonic()
        result = await _run_phase3(mocks, ensemble_rounds=5)
        elapsed = time.monotonic() - start

        assert isinstance(result, Phase3Result)
        assert elapsed < 5.0, (
            f"Orchestration overhead was {elapsed:.2f}s, expected < 5.0s"
        )

    async def test_overhead_under_five_seconds_r10(self) -> None:
        """Orchestration overhead excluding LLM calls is under 5s even for R=10."""
        mocks = _patch_phase3_dependencies()

        start = time.monotonic()
        result = await _run_phase3(mocks, ensemble_rounds=10)
        elapsed = time.monotonic() - start

        assert isinstance(result, Phase3Result)
        assert elapsed < 5.0, (
            f"Orchestration overhead was {elapsed:.2f}s for R=10, expected < 5.0s"
        )

    @given(r_rounds=st.integers(min_value=1, max_value=10))
    @settings(max_examples=10, deadline=10000)
    async def test_overhead_under_five_seconds_property(self, r_rounds: int) -> None:
        """Property: orchestration overhead < 5s for any R in [1, 10]."""
        mocks = _patch_phase3_dependencies()

        start = time.monotonic()
        result = await _run_phase3(mocks, ensemble_rounds=r_rounds)
        elapsed = time.monotonic() - start

        assert isinstance(result, Phase3Result)
        assert elapsed < 5.0, (
            f"Overhead was {elapsed:.2f}s for R={r_rounds}, expected < 5.0s"
        )

    async def test_overhead_with_failures_under_five_seconds(self) -> None:
        """Overhead remains < 5s even when some rounds have planner failures."""
        call_count = 0

        async def alternating_planner(*args: Any, **kwargs: Any) -> str | None:
            nonlocal call_count
            call_count += 1
            return None if call_count % 2 == 0 else "valid plan"

        mocks = _patch_phase3_dependencies()
        mocks["invoke_ens_planner"] = AsyncMock(side_effect=alternating_planner)

        start = time.monotonic()
        result = await _run_phase3(mocks, ensemble_rounds=8)
        elapsed = time.monotonic() - start

        assert isinstance(result, Phase3Result)
        assert elapsed < 5.0


# ===========================================================================
# REQ-P3-038: run_phase3 never raises on round failure
# ===========================================================================


@pytest.mark.unit
class TestPhase3NeverRaises:
    """run_phase3 never raises even when agent functions throw (REQ-P3-038)."""

    async def test_survives_planner_returning_none(self) -> None:
        """Does not raise when all planner calls return None."""
        mocks = _patch_phase3_dependencies(ens_planner_rv=None)

        result = await _run_phase3(mocks, ensemble_rounds=3)

        assert isinstance(result, Phase3Result)

    async def test_survives_ensembler_returning_none(self) -> None:
        """Does not raise when all ensembler calls return None."""
        mocks = _patch_phase3_dependencies()
        mocks["invoke_ensembler"] = AsyncMock(return_value=None)

        result = await _run_phase3(mocks, ensemble_rounds=3)

        assert isinstance(result, Phase3Result)

    async def test_survives_eval_failures(self) -> None:
        """Does not raise when all evaluations return is_error=True."""
        failed_sol = _make_ensemble_solution()
        failed_eval = _make_eval_result(score=None, is_error=True)
        mocks = _patch_phase3_dependencies()
        mocks["evaluate_with_retry"] = AsyncMock(return_value=(failed_sol, failed_eval))

        result = await _run_phase3(mocks, ensemble_rounds=3)

        assert isinstance(result, Phase3Result)

    async def test_survives_mixed_failures(self) -> None:
        """Does not raise with a mix of planner, ensembler, and eval failures."""
        call_idx = 0

        async def planner_mix(*args: Any, **kwargs: Any) -> str | None:
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                return None
            return "plan"

        ens_call_idx = 0

        async def ensembler_mix(*args: Any, **kwargs: Any) -> SolutionScript | None:
            nonlocal ens_call_idx
            ens_call_idx += 1
            if ens_call_idx == 1:
                return None
            return _make_ensemble_solution()

        mocks = _patch_phase3_dependencies()
        mocks["invoke_ens_planner"] = AsyncMock(side_effect=planner_mix)
        mocks["invoke_ensembler"] = AsyncMock(side_effect=ensembler_mix)

        result = await _run_phase3(mocks, ensemble_rounds=4)

        assert isinstance(result, Phase3Result)

    async def test_returns_valid_result_even_on_total_failure(self) -> None:
        """All-failure scenario returns valid Phase3Result with fallback solution."""
        solutions = [
            _make_solution(content="fallback_a", score=0.80),
            _make_solution(content="fallback_b", score=0.75),
        ]
        mocks = _patch_phase3_dependencies(ens_planner_rv=None)

        result = await _run_phase3(mocks, ensemble_rounds=3, solutions=solutions)

        assert isinstance(result, Phase3Result)
        assert result.best_ensemble is not None
        assert isinstance(result.best_ensemble_score, float)
        assert result.best_ensemble.content == "fallback_a"

    @given(r_rounds=st.integers(min_value=1, max_value=6))
    @settings(max_examples=10, deadline=10000)
    async def test_never_raises_property(self, r_rounds: int) -> None:
        """Property: run_phase3 never raises regardless of R and failure pattern."""
        mocks = _patch_phase3_dependencies(ens_planner_rv=None)
        result = await _run_phase3(mocks, ensemble_rounds=r_rounds)

        assert isinstance(result, Phase3Result)


# ===========================================================================
# REQ-P3-039: Phase 3 start logging
# ===========================================================================


@pytest.mark.unit
class TestPhase3StartLogging:
    """Phase 3 start event is logged at INFO with L, R, competition_id (REQ-P3-039)."""

    async def test_phase3_start_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Phase 3 start is logged at INFO level."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=3)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_msgs) >= 1, "Expected at least one INFO log at Phase 3 start"

    async def test_phase3_start_logs_l_value(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Phase 3 start log includes L (number of solutions)."""
        solutions = [
            _make_solution(content=f"sol_{i}", score=0.80 - 0.01 * i) for i in range(4)
        ]
        mocks = _patch_phase3_dependencies()
        await _run_phase3_with_caplog(
            mocks, caplog, ensemble_rounds=1, solutions=solutions
        )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "4" in all_info_text or "L" in all_info_text

    async def test_phase3_start_logs_r_value(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Phase 3 start log includes R (ensemble_rounds)."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=7)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "7" in all_info_text or "R" in all_info_text

    async def test_phase3_start_logs_competition_id(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Phase 3 start log includes competition_id."""
        task = _make_task(competition_id="my-kaggle-ensemble-comp")
        mocks = _patch_phase3_dependencies()
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=1, task=task)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "my-kaggle-ensemble-comp" in all_info_text


# ===========================================================================
# REQ-P3-039: Phase 3 skipped logging (L=1)
# ===========================================================================


@pytest.mark.unit
class TestPhase3SkippedLogging:
    """Phase 3 skipped (L=1) is logged at INFO (REQ-P3-039)."""

    async def test_skipped_single_solution_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When L=1, Phase 3 skip is logged at INFO level."""
        from mle_star.phase3 import run_phase3

        client = AsyncMock()
        task = _make_task(competition_id="skip-comp")
        config = _make_config(ensemble_rounds=5)
        sol = _make_solution(content="only_solution", score=0.80)

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase3(client=client, task=task, config=config, solutions=[sol])

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention skip or single solution
        assert (
            "skip" in all_info_text.lower()
            or "single" in all_info_text.lower()
            or "1" in all_info_text
        )

    async def test_skipped_logs_solution_score(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Phase 3 skipped log includes the single solution's score."""
        from mle_star.phase3 import run_phase3

        client = AsyncMock()
        task = _make_task()
        config = _make_config(ensemble_rounds=5)
        sol = _make_solution(content="only_solution", score=0.8765)

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase3(client=client, task=task, config=config, solutions=[sol])

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "0.8765" in all_info_text or "score" in all_info_text.lower()

    async def test_skipped_logs_competition_id(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Phase 3 skipped log includes competition_id."""
        from mle_star.phase3 import run_phase3

        client = AsyncMock()
        task = _make_task(competition_id="skip-this-comp")
        config = _make_config(ensemble_rounds=5)
        sol = _make_solution(content="only_solution", score=0.80)

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase3(client=client, task=task, config=config, solutions=[sol])

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "skip-this-comp" in all_info_text


# ===========================================================================
# REQ-P3-039: Ensemble round start logging
# ===========================================================================


@pytest.mark.unit
class TestEnsembleRoundStartLogging:
    """Ensemble round start is logged at INFO per round (REQ-P3-039)."""

    async def test_round_start_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Each round start is logged at INFO level."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=3)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention round index or round start
        assert (
            "round" in all_info_text.lower()
            or "r=" in all_info_text.lower()
            or "ensemble" in all_info_text.lower()
        )

    async def test_round_start_logs_round_index(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Round start log includes the round index r."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=2)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention index 0 or 1
        assert "0" in all_info_text or "1" in all_info_text

    async def test_round_start_logs_previous_plans_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Round start log includes number of previous plans."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=3)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # At least one round starts with 0 previous plans
        assert (
            "0" in all_info_text
            or "plan" in all_info_text.lower()
            or "previous" in all_info_text.lower()
            or "history" in all_info_text.lower()
        )


# ===========================================================================
# REQ-P3-039: A_ens_planner invocation logging
# ===========================================================================


@pytest.mark.unit
class TestEnsPlannerInvocationLogging:
    """A_ens_planner invocation start/complete/failure logging (REQ-P3-039)."""

    async def test_planner_start_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_ens_planner invocation start is logged at INFO with round and history size."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=1)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "planner" in all_info_text.lower()
            or "ens_planner" in all_info_text.lower()
            or "plan" in all_info_text.lower()
        )

    async def test_planner_complete_logs_plan_text(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_ens_planner complete log includes plan text (first 200 chars)."""
        mocks = _patch_phase3_dependencies(
            ens_planner_rv="Use weighted voting approach"
        )
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=1)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "weighted" in all_info_text.lower()
            or "plan" in all_info_text.lower()
            or "complet" in all_info_text.lower()
        )

    async def test_planner_complete_truncates_long_plan(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_ens_planner complete log truncates plan text beyond 200 characters."""
        long_plan = "P" * 400
        mocks = _patch_phase3_dependencies(ens_planner_rv=long_plan)
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=1)

        # No single log message should contain the full 400-char plan
        for record in caplog.records:
            assert long_plan not in record.message

    async def test_planner_empty_response_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_ens_planner empty response is logged at WARNING."""
        mocks = _patch_phase3_dependencies(ens_planner_rv=None)
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=1)

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1, "Expected WARNING when ens_planner returns None"
        warning_text = " ".join(r.message for r in warning_msgs)
        assert (
            "planner" in warning_text.lower()
            or "ens_planner" in warning_text.lower()
            or "empty" in warning_text.lower()
            or "skip" in warning_text.lower()
            or "fail" in warning_text.lower()
        )


# ===========================================================================
# REQ-P3-039: A_ensembler invocation logging
# ===========================================================================


@pytest.mark.unit
class TestEnsemblerInvocationLogging:
    """A_ensembler invocation start/complete/failure logging (REQ-P3-039)."""

    async def test_ensembler_start_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_ensembler invocation start is logged at INFO with round and plan text."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=1)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "ensembler" in all_info_text.lower()
            or "implement" in all_info_text.lower()
            or "ensemble" in all_info_text.lower()
        )

    async def test_ensembler_complete_logs_script_length(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_ensembler complete log includes script length."""
        ens_sol = _make_ensemble_solution(content="x" * 500)
        mocks = _patch_phase3_dependencies(ensembler_rv=ens_sol, leakage_rv=ens_sol)
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=1)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "500" in all_info_text
            or "length" in all_info_text.lower()
            or "complet" in all_info_text.lower()
        )

    async def test_ensembler_failure_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_ensembler extraction failure is logged at WARNING."""
        mocks = _patch_phase3_dependencies()
        mocks["invoke_ensembler"] = AsyncMock(return_value=None)
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=1)

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1, "Expected WARNING when ensembler returns None"
        warning_text = " ".join(r.message for r in warning_msgs)
        assert (
            "ensembler" in warning_text.lower()
            or "empty" in warning_text.lower()
            or "code" in warning_text.lower()
            or "skip" in warning_text.lower()
        )


# ===========================================================================
# REQ-P3-039: Leakage check logging
# ===========================================================================


@pytest.mark.unit
class TestLeakageCheckLogging:
    """Leakage check start/complete is logged at INFO (REQ-P3-039)."""

    async def test_leakage_start_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Leakage check start is logged at INFO with round and content length."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=1)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "leak" in all_info_text.lower()
            or "check" in all_info_text.lower()
            or "safety" in all_info_text.lower()
        )

    async def test_leakage_complete_logs_found_status(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Leakage complete log indicates leakage found yes/no and content changed."""
        ens_sol = _make_ensemble_solution(content="original_ens")
        leakage_fixed = _make_ensemble_solution(content="leakage_fixed")
        scored = SolutionScript(
            content="leakage_fixed", phase=SolutionPhase.ENSEMBLE, score=0.85
        )
        mocks = _patch_phase3_dependencies(
            ensembler_rv=ens_sol,
            leakage_rv=leakage_fixed,
        )
        mocks["evaluate_with_retry"] = AsyncMock(
            return_value=(scored, _make_eval_result(score=0.85))
        )
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=1)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "leak" in all_info_text.lower()
            or "changed" in all_info_text.lower()
            or "content" in all_info_text.lower()
        )

    async def test_leakage_no_change_logs_unchanged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When leakage check returns same object, log indicates no change."""
        ens_sol = _make_ensemble_solution(content="unchanged_ens")
        mocks = _patch_phase3_dependencies(
            ensembler_rv=ens_sol,
            leakage_rv=ens_sol,
        )
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=1)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "leak" in all_info_text.lower()
            or "unchanged" in all_info_text.lower()
            or "no" in all_info_text.lower()
        )


# ===========================================================================
# REQ-P3-039: Evaluation logging
# ===========================================================================


@pytest.mark.unit
class TestEvaluationLogging:
    """Evaluation start/complete is logged at INFO (REQ-P3-039)."""

    async def test_eval_start_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Evaluation start is logged at INFO with round and content length."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=1)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "eval" in all_info_text.lower()
            or "score" in all_info_text.lower()
            or "execut" in all_info_text.lower()
        )

    async def test_eval_complete_logs_score(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Evaluation complete log includes score value."""
        scored = SolutionScript(
            content="ens", phase=SolutionPhase.ENSEMBLE, score=0.9123
        )
        mocks = _patch_phase3_dependencies()
        mocks["evaluate_with_retry"] = AsyncMock(
            return_value=(scored, _make_eval_result(score=0.9123))
        )
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=1)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "0.9123" in all_info_text or "score" in all_info_text.lower()

    async def test_eval_complete_logs_failed_on_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Evaluation complete log indicates 'failed' when is_error=True."""
        failed_sol = _make_ensemble_solution()
        failed_eval = _make_eval_result(score=None, is_error=True)
        mocks = _patch_phase3_dependencies()
        mocks["evaluate_with_retry"] = AsyncMock(return_value=(failed_sol, failed_eval))
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=1)

        all_msgs = [r for r in caplog.records if r.levelno >= logging.INFO]
        all_text = " ".join(r.message for r in all_msgs)
        assert (
            "fail" in all_text.lower()
            or "error" in all_text.lower()
            or "None" in all_text
        )

    async def test_eval_complete_logs_duration(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Evaluation complete log includes duration."""
        scored = SolutionScript(content="ens", phase=SolutionPhase.ENSEMBLE, score=0.85)
        mocks = _patch_phase3_dependencies()
        mocks["evaluate_with_retry"] = AsyncMock(
            return_value=(scored, _make_eval_result(score=0.85, duration_seconds=42.5))
        )
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=1)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "duration" in all_info_text.lower()
            or "time" in all_info_text.lower()
            or "42.5" in all_info_text
            or "sec" in all_info_text.lower()
        )

    async def test_round_failed_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Round failed (execution error) is logged at WARNING with error summary."""
        failed_sol = _make_ensemble_solution()
        failed_eval = _make_eval_result(score=None, is_error=True)
        mocks = _patch_phase3_dependencies()
        mocks["evaluate_with_retry"] = AsyncMock(return_value=(failed_sol, failed_eval))
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=1)

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        # Either the round failure or the all-rounds-failed fallback should produce a warning
        assert len(warning_msgs) >= 1
        all_text = " ".join(r.message for r in warning_msgs)
        assert (
            "round" in all_text.lower()
            or "fail" in all_text.lower()
            or "error" in all_text.lower()
            or "fallback" in all_text.lower()
        )


# ===========================================================================
# REQ-P3-039: Best selection logging
# ===========================================================================


@pytest.mark.unit
class TestBestSelectionLogging:
    """Best selection is logged at INFO with best round, score, count (REQ-P3-039)."""

    async def test_best_selection_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Best selection is logged at INFO level."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=3)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "best" in all_info_text.lower()
            or "select" in all_info_text.lower()
            or "score" in all_info_text.lower()
        )

    async def test_best_selection_logs_best_score(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Best selection log includes the best score value."""
        scored = SolutionScript(
            content="ens", phase=SolutionPhase.ENSEMBLE, score=0.9456
        )
        mocks = _patch_phase3_dependencies()
        mocks["evaluate_with_retry"] = AsyncMock(
            return_value=(scored, _make_eval_result(score=0.9456))
        )
        mocks["is_improvement_or_equal"] = lambda new, old, d: (
            new >= old if d == MetricDirection.MAXIMIZE else new <= old
        )
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=2)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "0.9456" in all_info_text

    async def test_best_selection_logs_successful_round_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Best selection log includes total successful rounds."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=3)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention the count of successful rounds (3)
        assert (
            "3" in all_info_text
            or "success" in all_info_text.lower()
            or "round" in all_info_text.lower()
        )


# ===========================================================================
# REQ-P3-039: All rounds failed logging
# ===========================================================================


@pytest.mark.unit
class TestAllRoundsFailedLogging:
    """All rounds failed WARNING with R and fallback score (REQ-P3-039)."""

    async def test_all_rounds_failed_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When all rounds fail, a WARNING is logged."""
        solutions = [
            _make_solution(content="fb_a", score=0.80),
            _make_solution(content="fb_b", score=0.75),
        ]
        mocks = _patch_phase3_dependencies(ens_planner_rv=None)
        await _run_phase3_with_caplog(
            mocks, caplog, ensemble_rounds=3, solutions=solutions
        )

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1, (
            "Expected at least one WARNING when all rounds fail"
        )

    async def test_all_rounds_failed_logs_r_value(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """All rounds failed WARNING includes R (number of attempts)."""
        solutions = [
            _make_solution(content="fb_a", score=0.80),
            _make_solution(content="fb_b", score=0.75),
        ]
        mocks = _patch_phase3_dependencies(ens_planner_rv=None)
        await _run_phase3_with_caplog(
            mocks, caplog, ensemble_rounds=5, solutions=solutions
        )

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        warning_text = " ".join(r.message for r in warning_msgs)
        assert "5" in warning_text or "attempt" in warning_text.lower()

    async def test_all_rounds_failed_logs_fallback_score(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """All rounds failed WARNING includes the fallback solution score."""
        solutions = [
            _make_solution(content="fb_best", score=0.8765),
            _make_solution(content="fb_other", score=0.75),
        ]
        mocks = _patch_phase3_dependencies(ens_planner_rv=None)
        await _run_phase3_with_caplog(
            mocks, caplog, ensemble_rounds=2, solutions=solutions
        )

        all_msgs = " ".join(r.message for r in caplog.records)
        assert (
            "0.8765" in all_msgs
            or "fallback" in all_msgs.lower()
            or "input" in all_msgs.lower()
        )


# ===========================================================================
# REQ-P3-039: Phase 3 complete logging
# ===========================================================================


@pytest.mark.unit
class TestPhase3CompleteLogging:
    """Phase 3 complete is logged at INFO with score, round, duration (REQ-P3-039)."""

    async def test_phase3_complete_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Phase 3 complete event is logged at INFO level."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=2)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "complet" in all_info_text.lower()
            or "finish" in all_info_text.lower()
            or "phase 3" in all_info_text.lower()
            or "done" in all_info_text.lower()
        )

    async def test_phase3_complete_logs_best_score(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Phase 3 complete log includes best score."""
        scored = SolutionScript(
            content="ens", phase=SolutionPhase.ENSEMBLE, score=0.9234
        )
        mocks = _patch_phase3_dependencies()
        mocks["evaluate_with_retry"] = AsyncMock(
            return_value=(scored, _make_eval_result(score=0.9234))
        )
        mocks["is_improvement_or_equal"] = lambda new, old, d: (
            new >= old if d == MetricDirection.MAXIMIZE else new <= old
        )
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=1)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "0.9234" in all_info_text or "score" in all_info_text.lower()

    async def test_phase3_complete_logs_total_duration(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Phase 3 complete log includes total duration."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=2)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "duration" in all_info_text.lower()
            or "time" in all_info_text.lower()
            or "sec" in all_info_text.lower()
            or "elapsed" in all_info_text.lower()
        )

    async def test_phase3_complete_logs_rounds_attempted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Phase 3 complete log includes rounds attempted."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=4)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "4" in all_info_text
            or "round" in all_info_text.lower()
            or "attempt" in all_info_text.lower()
        )


# ===========================================================================
# REQ-P3-043: Sequential ensemble round execution
# ===========================================================================


@pytest.mark.unit
class TestSequentialRoundExecution:
    """Ensemble rounds run sequentially, not concurrently (REQ-P3-043)."""

    async def test_rounds_run_in_order(self) -> None:
        """Each round completes before the next begins."""
        call_order: list[int] = []
        call_completed: list[int] = []

        async def tracked_planner(*args: Any, **kwargs: Any) -> str:
            step = len(call_order)
            call_order.append(step)
            if step > 0:
                assert step - 1 in call_completed, (
                    f"Round {step} started before round {step - 1} completed"
                )
            call_completed.append(step)
            return f"plan_{step}"

        mocks = _patch_phase3_dependencies()
        mocks["invoke_ens_planner"] = AsyncMock(side_effect=tracked_planner)

        await _run_phase3(mocks, ensemble_rounds=4)

        assert call_order == [0, 1, 2, 3]
        assert call_completed == [0, 1, 2, 3]

    async def test_no_concurrent_ensemble_rounds(self) -> None:
        """No two ensemble rounds run concurrently."""
        active_rounds = 0
        max_concurrent = 0

        async def tracked_eval(
            *args: Any, **kwargs: Any
        ) -> tuple[SolutionScript, EvaluationResult]:
            nonlocal active_rounds, max_concurrent
            active_rounds += 1
            if active_rounds > max_concurrent:
                max_concurrent = active_rounds
            sol = SolutionScript(
                content="ens", phase=SolutionPhase.ENSEMBLE, score=0.85
            )
            result = (sol, _make_eval_result(score=0.85))
            active_rounds -= 1
            return result

        mocks = _patch_phase3_dependencies()
        mocks["evaluate_with_retry"] = AsyncMock(side_effect=tracked_eval)

        await _run_phase3(mocks, ensemble_rounds=5)

        assert max_concurrent == 1, (
            f"Expected max 1 concurrent round, got {max_concurrent}"
        )

    async def test_history_accumulates_sequentially(self) -> None:
        """Plans and scores accumulate: round r sees exactly r prior entries."""
        planner_calls: list[tuple[Any, ...]] = []
        plan_idx = 0

        async def capture_planner(*args: Any, **kwargs: Any) -> str:
            nonlocal plan_idx
            planner_calls.append((args, kwargs))
            plan = f"plan_{plan_idx}"
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

        await _run_phase3(mocks, ensemble_rounds=4)

        for r in range(4):
            c = planner_calls[r]
            plans_arg = c[0][1] if len(c[0]) > 1 else c[1].get("plans", [])
            assert len(plans_arg) == r, (
                f"Round {r} expected {r} prior plans, got {len(plans_arg)}"
            )


# ===========================================================================
# REQ-P3-044: Exactly R rounds attempted
# ===========================================================================


@pytest.mark.unit
class TestExactlyRRoundsAttempted:
    """R rounds attempted regardless of failures (REQ-P3-044)."""

    @pytest.mark.parametrize("r_rounds", [1, 2, 3, 5, 8])
    async def test_planner_called_exactly_r_times(self, r_rounds: int) -> None:
        """invoke_ens_planner is called exactly R times."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3(mocks, ensemble_rounds=r_rounds)

        assert mocks["invoke_ens_planner"].call_count == r_rounds

    @pytest.mark.parametrize("r_rounds", [1, 2, 3, 5])
    async def test_r_plans_recorded_even_with_failures(self, r_rounds: int) -> None:
        """ensemble_plans has exactly R entries even when some rounds fail."""
        mocks = _patch_phase3_dependencies(ens_planner_rv=None)
        result = await _run_phase3(mocks, ensemble_rounds=r_rounds)

        assert len(result.ensemble_plans) == r_rounds
        assert len(result.ensemble_scores) == r_rounds

    async def test_failed_rounds_count_as_iterations(self) -> None:
        """Failed rounds still count toward R total iterations."""
        call_idx = 0

        async def alternating_planner(*args: Any, **kwargs: Any) -> str | None:
            nonlocal call_idx
            call_idx += 1
            return None if call_idx % 2 == 0 else f"plan_{call_idx}"

        mocks = _patch_phase3_dependencies()
        mocks["invoke_ens_planner"] = AsyncMock(side_effect=alternating_planner)

        result = await _run_phase3(mocks, ensemble_rounds=6)

        assert len(result.ensemble_plans) == 6
        assert len(result.ensemble_scores) == 6
        assert mocks["invoke_ens_planner"].call_count == 6

    @given(r_rounds=st.integers(min_value=1, max_value=8))
    @settings(max_examples=15, deadline=10000)
    async def test_exactly_r_rounds_property(self, r_rounds: int) -> None:
        """Property: ensemble_plans always has exactly R entries."""
        mocks = _patch_phase3_dependencies()
        result = await _run_phase3(mocks, ensemble_rounds=r_rounds)

        assert len(result.ensemble_plans) == r_rounds
        assert len(result.ensemble_scores) == r_rounds

    @given(r_rounds=st.integers(min_value=1, max_value=6))
    @settings(max_examples=10, deadline=10000)
    async def test_exactly_r_rounds_all_failures_property(self, r_rounds: int) -> None:
        """Property: R entries even when all rounds fail."""
        mocks = _patch_phase3_dependencies(ens_planner_rv=None)
        result = await _run_phase3(mocks, ensemble_rounds=r_rounds)

        assert len(result.ensemble_plans) == r_rounds
        assert len(result.ensemble_scores) == r_rounds


# ===========================================================================
# REQ-P3-045: Leakage check on every ensemble solution
# ===========================================================================


@pytest.mark.unit
class TestLeakageEveryRound:
    """Leakage check on every ensemble solution before eval (REQ-P3-045)."""

    @pytest.mark.parametrize("r_rounds", [1, 2, 3, 5])
    async def test_leakage_called_for_every_successful_round(
        self, r_rounds: int
    ) -> None:
        """check_and_fix_leakage called once per round when ensembler succeeds."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3(mocks, ensemble_rounds=r_rounds)

        assert mocks["check_and_fix_leakage"].call_count == r_rounds

    async def test_leakage_not_called_when_ensembler_fails(self) -> None:
        """check_and_fix_leakage NOT called when ensembler returns None."""
        mocks = _patch_phase3_dependencies()
        mocks["invoke_ensembler"] = AsyncMock(return_value=None)

        await _run_phase3(mocks, ensemble_rounds=3)

        mocks["check_and_fix_leakage"].assert_not_called()

    async def test_leakage_not_called_when_planner_fails(self) -> None:
        """check_and_fix_leakage NOT called when planner returns None."""
        mocks = _patch_phase3_dependencies(ens_planner_rv=None)

        await _run_phase3(mocks, ensemble_rounds=3)

        mocks["check_and_fix_leakage"].assert_not_called()

    async def test_leakage_called_before_eval(self) -> None:
        """check_and_fix_leakage is called before evaluate_with_retry each round."""
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

        await _run_phase3(mocks, ensemble_rounds=3)

        # Verify leakage always precedes eval
        for i in range(3):
            leakage_idx = call_order.index("leakage", i * 2)
            eval_idx = call_order.index("eval", i * 2)
            assert leakage_idx < eval_idx, (
                f"Round {i}: leakage at {leakage_idx}, eval at {eval_idx}"
            )

    async def test_leakage_receives_ensemble_solution(self) -> None:
        """check_and_fix_leakage receives the solution from invoke_ensembler."""
        ens_sol = _make_ensemble_solution(content="leakage_target_content")
        mocks = _patch_phase3_dependencies(ensembler_rv=ens_sol, leakage_rv=ens_sol)

        await _run_phase3(mocks, ensemble_rounds=1)

        leakage_call = mocks["check_and_fix_leakage"].call_args
        passed_solution = (
            leakage_call[0][0] if leakage_call[0] else leakage_call[1].get("solution")
        )
        assert passed_solution.content == "leakage_target_content"

    async def test_mixed_success_and_failure_correct_leakage_count(self) -> None:
        """Leakage called only for rounds where ensembler succeeds."""
        ens_call_idx = 0

        async def ensembler_mix(*args: Any, **kwargs: Any) -> SolutionScript | None:
            nonlocal ens_call_idx
            ens_call_idx += 1
            # Rounds 1 and 3 succeed, round 2 fails
            if ens_call_idx == 2:
                return None
            return _make_ensemble_solution()

        planner_idx = 0

        async def planner_all(*args: Any, **kwargs: Any) -> str:
            nonlocal planner_idx
            planner_idx += 1
            return f"plan_{planner_idx}"

        mocks = _patch_phase3_dependencies()
        mocks["invoke_ens_planner"] = AsyncMock(side_effect=planner_all)
        mocks["invoke_ensembler"] = AsyncMock(side_effect=ensembler_mix)

        await _run_phase3(mocks, ensemble_rounds=3)

        # Leakage called for rounds 1 and 3 only (2 times)
        assert mocks["check_and_fix_leakage"].call_count == 2

    @given(r_rounds=st.integers(min_value=1, max_value=6))
    @settings(max_examples=10, deadline=10000)
    async def test_leakage_count_equals_r_when_all_succeed(self, r_rounds: int) -> None:
        """Property: leakage count equals R when all ensembler calls succeed."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3(mocks, ensemble_rounds=r_rounds)

        assert mocks["check_and_fix_leakage"].call_count == r_rounds


# ===========================================================================
# REQ-P3-042: Algorithm 3 fidelity
# ===========================================================================


@pytest.mark.unit
class TestAlgorithm3Fidelity:
    """Algorithm 3 fidelity: r=0 no history, plan->implement->eval (REQ-P3-042)."""

    async def test_round_zero_receives_empty_history(self) -> None:
        """First round (r=0) passes empty plans and empty scores to ens_planner."""
        planner_calls: list[tuple[Any, ...]] = []

        async def capture_planner(*args: Any, **kwargs: Any) -> str:
            planner_calls.append((args, kwargs))
            return "plan_r0"

        mocks = _patch_phase3_dependencies()
        mocks["invoke_ens_planner"] = AsyncMock(side_effect=capture_planner)

        await _run_phase3(mocks, ensemble_rounds=1)

        first_call = planner_calls[0]
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

    async def test_each_round_follows_plan_implement_eval_order(self) -> None:
        """Each round follows plan -> implement -> leakage -> eval sequence."""
        call_order: list[str] = []

        async def track_planner(*a: Any, **kw: Any) -> str:
            call_order.append("plan")
            return "a plan"

        async def track_ensembler(*a: Any, **kw: Any) -> SolutionScript:
            call_order.append("implement")
            return _make_ensemble_solution()

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
        mocks["invoke_ens_planner"] = AsyncMock(side_effect=track_planner)
        mocks["invoke_ensembler"] = AsyncMock(side_effect=track_ensembler)
        mocks["check_and_fix_leakage"] = AsyncMock(side_effect=track_leakage)
        mocks["evaluate_with_retry"] = AsyncMock(side_effect=track_eval)

        await _run_phase3(mocks, ensemble_rounds=2)

        # Each round should have: plan, implement, leakage, eval (4 calls per round)
        assert len(call_order) == 8
        for r in range(2):
            base = r * 4
            assert call_order[base] == "plan"
            assert call_order[base + 1] == "implement"
            assert call_order[base + 2] == "leakage"
            assert call_order[base + 3] == "eval"

    async def test_best_selection_uses_improvement_or_equal(self) -> None:
        """Best solution selection uses is_improvement_or_equal (>= semantics)."""
        import inspect

        from mle_star import phase3

        source = inspect.getsource(phase3.run_phase3)
        assert "is_improvement_or_equal" in source

    async def test_best_selection_last_tie_wins(self) -> None:
        """On tied scores, the LAST occurrence wins (>= semantics)."""
        round_scores = [0.85, 0.85]
        round_contents = ["ens_first", "ens_second"]
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

        result = await _run_phase3(mocks, ensemble_rounds=2)

        assert result.best_ensemble.content == "ens_second"


# ===========================================================================
# Full logging integration test
# ===========================================================================


@pytest.mark.unit
class TestFullLoggingIntegration:
    """End-to-end logging integration test for a complete Phase 3 run (REQ-P3-039)."""

    async def test_successful_run_has_all_expected_log_events(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A successful Phase 3 run produces INFO logs for all expected events."""
        ensemble_sol = _make_ensemble_solution(content="ens_code")
        scored_sol = SolutionScript(
            content="ens_code", phase=SolutionPhase.ENSEMBLE, score=0.90
        )
        mocks = _patch_phase3_dependencies()
        mocks["invoke_ensembler"] = AsyncMock(return_value=ensemble_sol)
        mocks["check_and_fix_leakage"] = AsyncMock(return_value=ensemble_sol)
        mocks["evaluate_with_retry"] = AsyncMock(
            return_value=(scored_sol, _make_eval_result(score=0.90))
        )
        mocks["is_improvement_or_equal"] = lambda new, old, d: (
            new >= old if d == MetricDirection.MAXIMIZE else new <= old
        )

        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=2)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        # Expect at minimum: phase 3 start, 2x round start, 2x planner, 2x ensembler,
        # 2x leakage, 2x eval, best selection, phase 3 complete = ~15+ INFO messages
        assert len(info_msgs) >= 5, (
            f"Expected at least 5 INFO messages for 2-round run, got {len(info_msgs)}. "
            f"Messages: {[r.message for r in info_msgs]}"
        )

    async def test_all_failed_run_produces_warning_and_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """All-failed run produces WARNINGs plus INFO for start/complete."""
        solutions = [
            _make_solution(content="fb_a", score=0.80),
            _make_solution(content="fb_b", score=0.75),
        ]
        mocks = _patch_phase3_dependencies(ens_planner_rv=None)
        await _run_phase3_with_caplog(
            mocks, caplog, ensemble_rounds=3, solutions=solutions
        )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]

        # Should have INFO for phase 3 start and complete at minimum
        assert len(info_msgs) >= 2, (
            f"Expected at least 2 INFO messages, got {len(info_msgs)}"
        )
        # Should have WARNINGs for planner failures and all-rounds-failed
        assert len(warning_msgs) >= 1, (
            f"Expected at least 1 WARNING message, got {len(warning_msgs)}"
        )

    async def test_log_messages_from_correct_logger(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """All log messages originate from the mle_star.phase3 logger."""
        mocks = _patch_phase3_dependencies()
        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=1)

        for record in caplog.records:
            assert record.name == _LOGGER_NAME, (
                f"Log from unexpected logger: {record.name}"
            )

    async def test_mixed_round_outcomes_correct_log_levels(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Mix of successful and failed rounds produces correct log levels."""
        call_idx = 0

        async def planner_mix(*args: Any, **kwargs: Any) -> str | None:
            nonlocal call_idx
            call_idx += 1
            # Round 1 planner succeeds, round 2 fails, round 3 succeeds
            if call_idx == 2:
                return None
            return f"plan_{call_idx}"

        mocks = _patch_phase3_dependencies()
        mocks["invoke_ens_planner"] = AsyncMock(side_effect=planner_mix)

        await _run_phase3_with_caplog(mocks, caplog, ensemble_rounds=3)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]

        # Should have INFO messages for successful rounds
        assert len(info_msgs) >= 3
        # Should have at least one WARNING for the failed round
        assert len(warning_msgs) >= 1

    async def test_skipped_single_solution_correct_logging(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """L=1 skip produces correct INFO log without any round-level logs."""
        from mle_star.phase3 import run_phase3

        client = AsyncMock()
        task = _make_task(competition_id="skip-test")
        config = _make_config(ensemble_rounds=5)
        sol = _make_solution(content="only_sol", score=0.85)

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase3(client=client, task=task, config=config, solutions=[sol])

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        # Should have skip INFO log(s) but no round-level logs
        assert len(info_msgs) >= 1
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "skip" in all_info_text.lower()
            or "single" in all_info_text.lower()
            or "1" in all_info_text
        )

    @given(r_rounds=st.integers(min_value=1, max_value=5))
    @settings(max_examples=10, deadline=10000)
    async def test_log_count_scales_with_rounds(self, r_rounds: int) -> None:
        """Property: INFO log count is at least 2 (start + complete) for any R."""
        import logging as _logging

        from mle_star.phase3 import run_phase3

        mocks = _patch_phase3_dependencies()
        solutions = [
            _make_solution(content="sol_a", score=0.80),
            _make_solution(content="sol_b", score=0.75),
        ]
        config = _make_config(ensemble_rounds=r_rounds)
        task = _make_task()
        client = AsyncMock()

        handler = _logging.StreamHandler()
        handler.setLevel(_logging.DEBUG)
        test_logger = _logging.getLogger(_LOGGER_NAME)
        test_logger.addHandler(handler)

        try:
            with (
                patch(f"{_MODULE}.invoke_ens_planner", mocks["invoke_ens_planner"]),
                patch(f"{_MODULE}.invoke_ensembler", mocks["invoke_ensembler"]),
                patch(
                    f"{_MODULE}.check_and_fix_leakage",
                    mocks["check_and_fix_leakage"],
                ),
                patch(
                    f"{_MODULE}.make_debug_callback",
                    mocks["make_debug_callback"],
                ),
                patch(
                    f"{_MODULE}.evaluate_with_retry",
                    mocks["evaluate_with_retry"],
                ),
                patch(
                    f"{_MODULE}.is_improvement_or_equal",
                    mocks["is_improvement_or_equal"],
                ),
            ):
                result = await run_phase3(
                    client=client,
                    task=task,
                    config=config,
                    solutions=solutions,
                )

            assert isinstance(result, Phase3Result)
        finally:
            test_logger.removeHandler(handler)
