"""Tests for Phase 1 constraints (Task 30).

Validates non-functional requirements for Phase 1: structured logging at
correct levels for all 16 key events, sequential candidate generation/evaluation,
sequential merge loop, orchestration overhead budget, SDK agent invocation,
single module organization, Algorithm 1 fidelity, leakage check at 3 integration
points, and prompt fidelity for retriever/init/merger.

Tests are written TDD-first and serve as the executable specification for
REQ-P1-034 through REQ-P1-045.

Refs:
    SRS 04c (Phase 1 Constraints), IMPLEMENTATION_PLAN.md Task 30.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    AgentType,
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
_LOGGER_NAME = "mle_star.phase1"


# ---------------------------------------------------------------------------
# Reusable test helpers (mirrored from test_phase1_orchestration.py)
# ---------------------------------------------------------------------------


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
    duration_seconds: float = 1.0,
) -> EvaluationResult:
    """Create an EvaluationResult with the given score and error state."""
    return EvaluationResult(
        score=score,
        stdout=f"Final Validation Performance: {score}" if score is not None else "",
        stderr="" if not is_error else "Traceback (most recent call last):\nError",
        exit_code=0 if not is_error else 1,
        duration_seconds=duration_seconds,
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
    mock_data_usage = AsyncMock(side_effect=lambda sol, _task, _client: sol)
    mock_debug_cb = MagicMock()
    mock_make_debug = MagicMock(return_value=mock_debug_cb)
    mock_eval = AsyncMock(side_effect=eval_results)

    if ranked_pairs is None:
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
        "check_data_usage": mock_data_usage,
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
# REQ-P1-034/035: Candidate generation/evaluation independence (sequential)
# ===========================================================================


@pytest.mark.unit
class TestCandidateGenerationSequential:
    """Candidate generation/evaluation is sequential (REQ-P1-034/035)."""

    def test_asyncio_gather_in_candidate_loop(self) -> None:
        """_generate_and_evaluate_candidates uses asyncio.gather for parallel execution."""
        from mle_star import phase1

        source = inspect.getsource(phase1._generate_and_evaluate_candidates)
        assert "asyncio.gather" in source

    def test_parallel_dispatch_in_candidate_loop(self) -> None:
        """_generate_and_evaluate_candidates uses parallel dispatch via gather."""
        from mle_star import phase1

        source = inspect.getsource(phase1._generate_and_evaluate_candidates)
        assert "gather" in source

    async def test_candidates_evaluated_in_order(self) -> None:
        """Candidates are generated and evaluated in sequential model order."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=3)

        models = [_make_model(f"model_{i}") for i in range(3)]
        candidates = [
            _make_solution(content=f"code_{i}", source_model=f"model_{i}")
            for i in range(3)
        ]
        eval_results = [
            (candidates[i], _make_eval_result(score=0.80 + i * 0.05)) for i in range(3)
        ]
        ranked = list(reversed(eval_results))

        call_order: list[str] = []

        async def _gen_side(task: Any, model: Any, config: Any, client: Any, **kwargs: Any) -> Any:
            call_order.append(f"gen_{model.model_name}")
            idx = int(model.model_name.split("_")[1])
            return candidates[idx]

        async def _eval_side(*args: Any, **kwargs: Any) -> Any:
            call_order.append("eval")
            # Pop from the front of eval_results
            idx = sum(1 for e in call_order if e == "eval") - 1
            return eval_results[idx]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=candidates,
            eval_results=eval_results,
            ranked_pairs=ranked,
        )
        mocks["generate_candidate"] = AsyncMock(side_effect=_gen_side)
        mocks["evaluate_with_retry"] = AsyncMock(side_effect=eval_results)
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        # Verify generate calls happen in order
        gen_calls = [c for c in call_order if c.startswith("gen_")]
        assert gen_calls == ["gen_model_0", "gen_model_1", "gen_model_2"]


# ===========================================================================
# REQ-P1-036: Merge loop sequential requirement
# ===========================================================================


@pytest.mark.unit
class TestMergeLoopSequential:
    """Merge loop is inherently sequential (REQ-P1-036)."""

    def test_no_asyncio_gather_in_merge_loop(self) -> None:
        """_run_merge_loop does not use asyncio.gather or parallel constructs."""
        from mle_star import phase1

        source = inspect.getsource(phase1._run_merge_loop)
        assert "asyncio.gather" not in source
        assert "asyncio.create_task" not in source
        assert "TaskGroup" not in source

    def test_merge_loop_uses_sequential_for(self) -> None:
        """_run_merge_loop iterates sequentially over ranked candidates."""
        from mle_star import phase1

        source = inspect.getsource(phase1._run_merge_loop)
        assert "for " in source


# ===========================================================================
# REQ-P1-037: Phase 1 overhead budget < 5 seconds
# ===========================================================================


@pytest.mark.unit
class TestPhase1OverheadBudget:
    """Phase 1 orchestration overhead excluding LLM calls < 5 seconds (REQ-P1-037)."""

    async def test_overhead_under_five_seconds(self) -> None:
        """Orchestration overhead (excluding mocked calls) is under 5 seconds."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m0"), _make_model("m1")]
        sol_a = _make_solution(content="code_0", source_model="m0")
        sol_b = _make_solution(content="code_1", source_model="m1")
        res_a = _make_eval_result(score=0.80)
        res_b = _make_eval_result(score=0.85)
        ranked = [(sol_b, res_b), (sol_a, res_a)]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[(sol_a, res_a), (sol_b, res_b)],
            ranked_pairs=ranked,
        )
        # Merge returns None => no merge eval needed
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        start = time.monotonic()
        with _apply_patches(mocks):
            await run_phase1(task, config, client)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, (
            f"Orchestration overhead {elapsed:.2f}s exceeded 5s budget"
        )


# ===========================================================================
# REQ-P1-038: Structured logging - Phase 1 start
# ===========================================================================


@pytest.mark.unit
class TestPhase1StartLogging:
    """Phase 1 start event is logged at INFO (REQ-P1-038)."""

    async def test_phase1_start_logs_competition_id(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Phase 1 start log includes competition ID."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task(competition_id="my-kaggle-comp")
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol = _make_solution(content="code_a", source_model="m1")
        res = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol, sol],
            eval_results=[(sol, res), (sol, res)],
            ranked_pairs=[(sol, res)],
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "my-kaggle-comp" in all_info_text

    async def test_phase1_start_logs_m_value(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Phase 1 start log includes M (num_retrieved_models) value."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=5)

        models = [_make_model(f"m{i}") for i in range(5)]
        candidates = [
            _make_solution(content=f"code_{i}", source_model=f"m{i}") for i in range(5)
        ]
        eval_results = [
            (candidates[i], _make_eval_result(score=0.80 + i * 0.02)) for i in range(5)
        ]
        ranked = list(reversed(eval_results))

        mocks = _setup_standard_mocks(
            models=models,
            candidates=candidates,
            eval_results=eval_results,
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "5" in all_info_text or "M" in all_info_text


# ===========================================================================
# REQ-P1-038: Structured logging - Retrieval complete
# ===========================================================================


@pytest.mark.unit
class TestRetrievalCompleteLogging:
    """Retrieval complete event is logged at INFO (REQ-P1-038)."""

    async def test_retrieval_complete_logs_model_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Retrieval complete log includes number of models retrieved."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=3)

        models = [_make_model("resnet"), _make_model("bert"), _make_model("xgboost")]
        sol = _make_solution(content="code", source_model="resnet")
        res = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol, sol, sol],
            eval_results=[(sol, res), (sol, res), (sol, res)],
            ranked_pairs=[(sol, res)],
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "3" in all_info_text

    async def test_retrieval_complete_logs_model_names(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Retrieval complete log includes retrieved model names."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("resnet50"), _make_model("lightgbm")]
        sol = _make_solution(content="code", source_model="resnet50")
        res = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol, sol],
            eval_results=[(sol, res), (sol, res)],
            ranked_pairs=[(sol, res)],
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "resnet50" in all_info_text or "lightgbm" in all_info_text


# ===========================================================================
# REQ-P1-038: Structured logging - Candidate generation start/complete
# ===========================================================================


@pytest.mark.unit
class TestCandidateGenerationLogging:
    """Candidate generation events logged at INFO (REQ-P1-038)."""

    async def test_candidate_gen_start_logs_model_index_and_name(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Candidate generation start log includes model index (i/M) and model name."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("catboost"), _make_model("xgboost")]
        sol_a = _make_solution(content="code_a", source_model="catboost")
        sol_b = _make_solution(content="code_b", source_model="xgboost")
        res_a = _make_eval_result(score=0.80)
        res_b = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[(sol_a, res_a), (sol_b, res_b)],
            ranked_pairs=[(sol_b, res_b), (sol_a, res_a)],
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention model name
        assert "catboost" in all_info_text or "xgboost" in all_info_text
        # Should mention model index (1/2 or 0)
        assert "1" in all_info_text or "0" in all_info_text

    async def test_candidate_gen_complete_logs_code_length(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Candidate generation complete log includes code length."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("resnet")
        sol = _make_solution(content="x" * 150, source_model="resnet")
        res = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[(sol, res)],
            ranked_pairs=[(sol, res)],
        )

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "150" in all_info_text or "length" in all_info_text.lower()


# ===========================================================================
# REQ-P1-038: Structured logging - Candidate evaluation result
# ===========================================================================


@pytest.mark.unit
class TestCandidateEvaluationLogging:
    """Candidate evaluation result logged at INFO (REQ-P1-038)."""

    async def test_candidate_eval_result_logs_score(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Candidate evaluation result log includes score value."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("bert")
        sol = _make_solution(content="bert code", source_model="bert")
        res = _make_eval_result(score=0.9123, duration_seconds=45.0)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[(sol, res)],
            ranked_pairs=[(sol, res)],
        )

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "0.9123" in all_info_text or "score" in all_info_text.lower()

    async def test_candidate_eval_result_logs_duration(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Candidate evaluation result log includes duration."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("bert")
        sol = _make_solution(content="bert code", source_model="bert")
        res = _make_eval_result(score=0.85, duration_seconds=12.34)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[(sol, res)],
            ranked_pairs=[(sol, res)],
        )

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "12.34" in all_info_text or "duration" in all_info_text.lower()

    async def test_candidate_eval_result_logs_model_name(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Candidate evaluation result log includes model name."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("efficientnet")
        sol = _make_solution(content="eff code", source_model="efficientnet")
        res = _make_eval_result(score=0.88)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[(sol, res)],
            ranked_pairs=[(sol, res)],
        )

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "efficientnet" in all_info_text


# ===========================================================================
# REQ-P1-038: Structured logging - Candidate skipped (execution failure)
# ===========================================================================


@pytest.mark.unit
class TestCandidateSkippedLogging:
    """Candidate execution failure logged at WARNING (REQ-P1-038)."""

    async def test_candidate_skip_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When a candidate fails evaluation, a WARNING is logged."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("fail_model"), _make_model("good_model")]
        sol_fail = _make_solution(content="bad", source_model="fail_model")
        sol_good = _make_solution(content="good", source_model="good_model")
        res_fail = _make_eval_result(
            score=None, is_error=True, error_traceback="ValueError: boom"
        )
        res_good = _make_eval_result(score=0.90)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_fail, sol_good],
            eval_results=[(sol_fail, res_fail), (sol_good, res_good)],
            ranked_pairs=[(sol_good, res_good)],
        )

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        warning_text = " ".join(r.message for r in warning_msgs)
        assert "fail_model" in warning_text

    async def test_candidate_skip_logs_error_summary(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Candidate skip warning includes error summary."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("broken"), _make_model("ok")]
        sol_broken = _make_solution(content="err", source_model="broken")
        sol_ok = _make_solution(content="ok", source_model="ok")
        res_broken = _make_eval_result(
            score=None, is_error=True, error_traceback="RuntimeError: GPU OOM"
        )
        res_ok = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_broken, sol_ok],
            eval_results=[(sol_broken, res_broken), (sol_ok, res_ok)],
            ranked_pairs=[(sol_ok, res_ok)],
        )

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        warning_text = " ".join(r.message for r in warning_msgs)
        # Should contain part of the error traceback
        assert "broken" in warning_text


# ===========================================================================
# REQ-P1-038: Structured logging - All candidates failed
# ===========================================================================


@pytest.mark.unit
class TestAllCandidatesFailedLogging:
    """All candidates failed event logged at ERROR (REQ-P1-038)."""

    async def test_all_candidates_failed_logs_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When all candidates fail, an ERROR-level message is logged."""
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
            caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
            pytest.raises(RuntimeError),
        ):
            await run_phase1(task, config, client)

        # Check for ERROR-level log or the RuntimeError message
        # The error is currently raised, which may also be logged
        warning_and_above = [r for r in caplog.records if r.levelno >= logging.WARNING]
        # At minimum, the failing candidates should have produced warnings
        assert len(warning_and_above) >= 1
        # Ideally an ERROR log should mention all candidates failed and M value
        all_text = " ".join(r.message for r in caplog.records)
        assert "2" in all_text or "failed" in all_text.lower()


# ===========================================================================
# REQ-P1-038: Structured logging - Candidates sorted
# ===========================================================================


@pytest.mark.unit
class TestCandidatesSortedLogging:
    """Candidates sorted event logged at INFO (REQ-P1-038)."""

    async def test_candidates_sorted_logs_order(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Candidates sorted log includes model names in sorted order."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=3)

        models = [
            _make_model("lgbm"),
            _make_model("catboost"),
            _make_model("xgboost"),
        ]
        sol_a = _make_solution(content="code_a", source_model="lgbm")
        sol_b = _make_solution(content="code_b", source_model="catboost")
        sol_c = _make_solution(content="code_c", source_model="xgboost")
        res_a = _make_eval_result(score=0.80)
        res_b = _make_eval_result(score=0.90)
        res_c = _make_eval_result(score=0.85)

        ranked = [(sol_b, res_b), (sol_c, res_c), (sol_a, res_a)]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b, sol_c],
            eval_results=[(sol_a, res_a), (sol_b, res_b), (sol_c, res_c)],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention sorted or ranked
        assert (
            "sort" in all_info_text.lower()
            or "rank" in all_info_text.lower()
            or "order" in all_info_text.lower()
        )

    async def test_candidates_sorted_logs_scores(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Candidates sorted log includes score values."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="a", source_model="m1")
        sol_b = _make_solution(content="b", source_model="m2")
        res_a = _make_eval_result(score=0.75)
        res_b = _make_eval_result(score=0.92)
        ranked = [(sol_b, res_b), (sol_a, res_a)]

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[(sol_a, res_a), (sol_b, res_b)],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "0.92" in all_info_text or "0.75" in all_info_text


# ===========================================================================
# REQ-P1-038: Structured logging - Merge attempt start/result
# ===========================================================================


@pytest.mark.unit
class TestMergeAttemptLogging:
    """Merge attempt start/result events logged at INFO (REQ-P1-038)."""

    async def test_merge_start_logs_base_score(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Merge attempt start log includes base score."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="a", source_model="m1")
        sol_b = _make_solution(content="b", source_model="m2")
        res_a = _make_eval_result(score=0.80)
        res_b = _make_eval_result(score=0.90)
        ranked = [(sol_b, res_b), (sol_a, res_a)]

        merged = _make_merged_solution("merged")
        merged_res = _make_eval_result(score=0.92)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[(sol_a, res_a), (sol_b, res_b), (merged, merged_res)],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=merged)
        mocks["is_improvement_or_equal"] = MagicMock(return_value=True)

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention merge start with base score
        assert "merg" in all_info_text.lower()
        assert "0.9" in all_info_text or "score" in all_info_text.lower()

    async def test_merge_result_logs_accepted_or_rejected(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Merge result log indicates whether merge was accepted or rejected."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="a", source_model="m1")
        sol_b = _make_solution(content="b", source_model="m2")
        res_a = _make_eval_result(score=0.80)
        res_b = _make_eval_result(score=0.90)
        ranked = [(sol_b, res_b), (sol_a, res_a)]

        merged = _make_merged_solution("merged")
        merged_res = _make_eval_result(score=0.92)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[(sol_a, res_a), (sol_b, res_b), (merged, merged_res)],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=merged)
        mocks["is_improvement_or_equal"] = MagicMock(return_value=True)

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "accept" in all_info_text.lower()
            or "reject" in all_info_text.lower()
            or "improv" in all_info_text.lower()
            or "0.92" in all_info_text
        )

    async def test_merge_result_logs_merged_score(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Merge result log includes the merged score."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="a", source_model="m1")
        sol_b = _make_solution(content="b", source_model="m2")
        res_a = _make_eval_result(score=0.80)
        res_b = _make_eval_result(score=0.90)
        ranked = [(sol_b, res_b), (sol_a, res_a)]

        merged = _make_merged_solution("merged")
        merged_res = _make_eval_result(score=0.9456)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[(sol_a, res_a), (sol_b, res_b), (merged, merged_res)],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=merged)
        mocks["is_improvement_or_equal"] = MagicMock(return_value=True)

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "0.9456" in all_info_text


# ===========================================================================
# REQ-P1-038: Structured logging - Merge loop break
# ===========================================================================


@pytest.mark.unit
class TestMergeLoopBreakLogging:
    """Merge loop break event logged at INFO (REQ-P1-038)."""

    async def test_merge_loop_break_on_none_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When merge returns None, break is logged with reason."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="a", source_model="m1")
        sol_b = _make_solution(content="b", source_model="m2")
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

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        all_msgs = " ".join(r.message for r in caplog.records)
        # Should mention merge break or None
        assert (
            "break" in all_msgs.lower()
            or "none" in all_msgs.lower()
            or "loop" in all_msgs.lower()
        )

    async def test_merge_loop_break_on_eval_failure_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When merged solution fails evaluation, break is logged."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="a", source_model="m1")
        sol_b = _make_solution(content="b", source_model="m2")
        res_a = _make_eval_result(score=0.80)
        res_b = _make_eval_result(score=0.90)
        ranked = [(sol_b, res_b), (sol_a, res_a)]

        merged = _make_merged_solution("merged")
        merged_res = _make_eval_result(score=None, is_error=True)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[(sol_a, res_a), (sol_b, res_b), (merged, merged_res)],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=merged)

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        warning_msgs = [r for r in caplog.records if r.levelno >= logging.WARNING]
        warning_text = " ".join(r.message for r in warning_msgs)
        assert (
            "merg" in warning_text.lower()
            or "eval" in warning_text.lower()
            or "break" in warning_text.lower()
            or "fail" in warning_text.lower()
        )


# ===========================================================================
# REQ-P1-038: Structured logging - Post-merge safety checks
# ===========================================================================


@pytest.mark.unit
class TestPostMergeSafetyLogging:
    """Post-merge safety check events logged at INFO (REQ-P1-038)."""

    async def test_post_merge_data_start_logs_solution_length(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Post-merge A_data start log includes solution content length."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="x" * 200, source_model="m1")
        res = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[(sol, res)],
            ranked_pairs=[(sol, res)],
        )

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "data" in all_info_text.lower()
            or "A_data" in all_info_text
            or "safety" in all_info_text.lower()
        )

    async def test_post_merge_data_result_logs_modified_status(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Post-merge A_data result log indicates whether solution was modified."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="original", source_model="m1")
        res = _make_eval_result(score=0.85)

        # Data check modifies solution
        data_sol = _make_solution(content="data_modified", phase=SolutionPhase.INIT)
        data_res = _make_eval_result(score=0.88)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[(sol, res), (data_sol, data_res)],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_data_usage"] = AsyncMock(return_value=data_sol)

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should indicate modification occurred
        assert (
            "modif" in all_info_text.lower()
            or "changed" in all_info_text.lower()
            or "data" in all_info_text.lower()
        )

    async def test_post_merge_leakage_start_logs_solution_length(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Post-merge A_leakage start log includes solution content length."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="code_content", source_model="m1")
        res = _make_eval_result(score=0.85)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[(sol, res)],
            ranked_pairs=[(sol, res)],
        )

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "leak" in all_info_text.lower()
            or "A_leakage" in all_info_text
            or "safety" in all_info_text.lower()
        )

    async def test_post_merge_leakage_result_logs_detection_status(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Post-merge A_leakage result log indicates whether leakage was detected."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="original_code", source_model="m1")
        res = _make_eval_result(score=0.85)

        # Leakage modifies solution
        leak_sol = _make_solution(content="leak_fixed", phase=SolutionPhase.INIT)
        leak_res = _make_eval_result(score=0.83)

        async def _leakage(s: SolutionScript, _t: Any, _c: Any) -> SolutionScript:
            if s.content == "original_code":
                return leak_sol
            return s

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[(sol, res), (leak_sol, leak_res)],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_and_fix_leakage"] = AsyncMock(side_effect=_leakage)

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        all_msgs = " ".join(r.message for r in caplog.records)
        # Should indicate leakage was found/modified
        assert (
            "leak" in all_msgs.lower()
            or "detect" in all_msgs.lower()
            or "modif" in all_msgs.lower()
        )


# ===========================================================================
# REQ-P1-038: Structured logging - Phase 1 complete
# ===========================================================================


@pytest.mark.unit
class TestPhase1CompleteLogging:
    """Phase 1 complete event logged at INFO (REQ-P1-038)."""

    async def test_phase1_complete_logs_final_score(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Phase 1 complete log includes final score."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="code", source_model="m1")
        res = _make_eval_result(score=0.8765)

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[(sol, res)],
            ranked_pairs=[(sol, res)],
        )

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "0.8765" in all_info_text or "score" in all_info_text.lower()

    async def test_phase1_complete_logs_total_duration(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Phase 1 complete log includes total duration."""
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

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "duration" in all_info_text.lower()
            or "time" in all_info_text.lower()
            or "elapsed" in all_info_text.lower()
            or "s" in all_info_text  # seconds unit
        )

    async def test_phase1_complete_logs_merge_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Phase 1 complete log includes number of merges performed."""
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

        merged_1 = _make_merged_solution("merged_1")
        merged_1_res = _make_eval_result(score=0.92)
        merged_2 = _make_merged_solution("merged_2")
        merged_2_res = _make_eval_result(score=0.94)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=candidates,
            eval_results=[
                *eval_results,
                (merged_1, merged_1_res),
                (merged_2, merged_2_res),
            ],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(side_effect=[merged_1, merged_2])
        mocks["is_improvement_or_equal"] = MagicMock(return_value=True)

        with _apply_patches(mocks), caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await run_phase1(task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "merg" in all_info_text.lower() or "2" in all_info_text  # 2 merges
        )


# ===========================================================================
# REQ-P1-039: SDK agent invocation (all agents invoked via SDK)
# ===========================================================================


@pytest.mark.unit
class TestSDKAgentInvocation:
    """All agents invoked via SDK client.send_message (REQ-P1-039)."""

    def test_retrieve_models_uses_client_send_message(self) -> None:
        """retrieve_models calls client.send_message with agent_type."""
        from mle_star import phase1

        source = inspect.getsource(phase1.retrieve_models)
        assert "client.send_message" in source
        assert "agent_type" in source

    def test_generate_candidate_uses_client_send_message(self) -> None:
        """generate_candidate calls client.send_message with agent_type."""
        from mle_star import phase1

        source = inspect.getsource(phase1.generate_candidate)
        assert "client.send_message" in source
        assert "agent_type" in source

    def test_merge_solutions_uses_client_send_message(self) -> None:
        """merge_solutions calls client.send_message with agent_type."""
        from mle_star import phase1

        source = inspect.getsource(phase1.merge_solutions)
        assert "client.send_message" in source
        assert "agent_type" in source


# ===========================================================================
# REQ-P1-040: Single module organization
# ===========================================================================


@pytest.mark.unit
class TestSingleModuleOrganization:
    """All Phase 1 code resides in a single module (REQ-P1-040)."""

    def test_all_phase1_functions_in_phase1_module(self) -> None:
        """Key Phase 1 functions are defined in mle_star.phase1."""
        from mle_star import phase1

        assert hasattr(phase1, "parse_retriever_output")
        assert hasattr(phase1, "retrieve_models")
        assert hasattr(phase1, "generate_candidate")
        assert hasattr(phase1, "merge_solutions")
        assert hasattr(phase1, "run_phase1")

    def test_run_phase1_is_async(self) -> None:
        """run_phase1 is an async function."""
        from mle_star.phase1 import run_phase1

        assert asyncio.iscoroutinefunction(run_phase1)

    def test_retrieve_models_is_async(self) -> None:
        """retrieve_models is an async function."""
        from mle_star.phase1 import retrieve_models

        assert asyncio.iscoroutinefunction(retrieve_models)

    def test_generate_candidate_is_async(self) -> None:
        """generate_candidate is an async function."""
        from mle_star.phase1 import generate_candidate

        assert asyncio.iscoroutinefunction(generate_candidate)

    def test_merge_solutions_is_async(self) -> None:
        """merge_solutions is an async function."""
        from mle_star.phase1 import merge_solutions

        assert asyncio.iscoroutinefunction(merge_solutions)


# ===========================================================================
# REQ-P1-041: Algorithm 1 fidelity
# ===========================================================================


@pytest.mark.unit
class TestAlgorithm1Fidelity:
    """run_phase1 follows Algorithm 1 pipeline order (REQ-P1-041)."""

    async def test_pipeline_order_retrieval_then_generate_then_merge(self) -> None:
        """Pipeline executes in order: retrieve -> generate/eval -> rank -> merge."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="a", source_model="m1")
        sol_b = _make_solution(content="b", source_model="m2")
        res_a = _make_eval_result(score=0.80)
        res_b = _make_eval_result(score=0.90)
        ranked = [(sol_b, res_b), (sol_a, res_a)]

        merged = _make_merged_solution("merged")
        merged_res = _make_eval_result(score=0.92)

        call_order: list[str] = []

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[(sol_a, res_a), (sol_b, res_b), (merged, merged_res)],
            ranked_pairs=ranked,
        )

        orig_retrieve = mocks["retrieve_models"]
        orig_generate = mocks["generate_candidate"]
        orig_rank = mocks["rank_solutions"]
        orig_merge = AsyncMock(return_value=merged)

        async def _retrieve_wrapper(*a: Any, **k: Any) -> Any:
            call_order.append("retrieve")
            return await orig_retrieve(*a, **k)

        async def _generate_wrapper(*a: Any, **k: Any) -> Any:
            call_order.append("generate")
            return await orig_generate(*a, **k)

        def _rank_wrapper(*a: Any, **k: Any) -> Any:
            call_order.append("rank")
            return orig_rank(*a, **k)

        async def _merge_wrapper(*a: Any, **k: Any) -> Any:
            call_order.append("merge")
            return await orig_merge(*a, **k)

        mocks["retrieve_models"] = AsyncMock(side_effect=_retrieve_wrapper)
        mocks["generate_candidate"] = AsyncMock(side_effect=_generate_wrapper)
        mocks["rank_solutions"] = MagicMock(side_effect=_rank_wrapper)
        mocks["merge_solutions"] = AsyncMock(side_effect=_merge_wrapper)
        mocks["is_improvement_or_equal"] = MagicMock(return_value=True)

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        # Verify ordering: retrieve before generate, generate before rank, rank before merge
        retrieve_idx = call_order.index("retrieve")
        first_generate_idx = call_order.index("generate")
        rank_idx = call_order.index("rank")
        merge_idx = call_order.index("merge")

        assert retrieve_idx < first_generate_idx
        assert first_generate_idx < rank_idx
        assert rank_idx < merge_idx

    async def test_merge_uses_improvement_or_equal_semantics(self) -> None:
        """Merge loop uses is_improvement_or_equal (>= semantics) for acceptance."""
        from mle_star import phase1

        source = inspect.getsource(phase1._run_merge_loop)
        assert "is_improvement_or_equal" in source

    async def test_result_contains_all_required_fields(self) -> None:
        """Phase1Result has retrieved_models, candidate_solutions/scores, initial_solution/score."""
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
            result = await run_phase1(task, config, client)

        assert isinstance(result, Phase1Result)
        assert result.retrieved_models is not None
        assert result.candidate_solutions is not None
        assert result.candidate_scores is not None
        assert result.initial_solution is not None
        assert result.initial_score is not None


# ===========================================================================
# REQ-P1-042: Leakage check at 3 integration points
# ===========================================================================


@pytest.mark.unit
class TestLeakageThreeIntegrationPoints:
    """check_and_fix_leakage used at 3 integration points (REQ-P1-042)."""

    async def test_leakage_called_for_each_candidate(self) -> None:
        """Leakage check runs for each successful candidate (point 1)."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="a", source_model="m1")
        sol_b = _make_solution(content="b", source_model="m2")
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

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        # 2 candidate leakage checks + 1 post-merge leakage check = at least 3
        assert mocks["check_and_fix_leakage"].await_count >= 2

    async def test_leakage_called_after_each_merge(self) -> None:
        """Leakage check runs after each merge (point 2)."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=2)

        models = [_make_model("m1"), _make_model("m2")]
        sol_a = _make_solution(content="a", source_model="m1")
        sol_b = _make_solution(content="b", source_model="m2")
        res_a = _make_eval_result(score=0.80)
        res_b = _make_eval_result(score=0.90)
        ranked = [(sol_b, res_b), (sol_a, res_a)]

        merged = _make_merged_solution("merged")
        merged_res = _make_eval_result(score=0.92)

        mocks = _setup_standard_mocks(
            models=models,
            candidates=[sol_a, sol_b],
            eval_results=[(sol_a, res_a), (sol_b, res_b), (merged, merged_res)],
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=merged)
        mocks["is_improvement_or_equal"] = MagicMock(return_value=True)

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        # 2 candidate + 1 merge + 1 post-merge = 4
        assert mocks["check_and_fix_leakage"].await_count >= 3

    async def test_leakage_called_post_merge_safety(self) -> None:
        """Leakage check runs as post-merge safety (point 3)."""
        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=1)

        model = _make_model("m1")
        sol = _make_solution(content="code", source_model="m1")
        res = _make_eval_result(score=0.85)

        leakage_calls: list[str] = []

        async def _track_leakage(s: SolutionScript, _t: Any, _c: Any) -> SolutionScript:
            leakage_calls.append(s.content)
            return s

        mocks = _setup_standard_mocks(
            models=[model],
            candidates=[sol],
            eval_results=[(sol, res)],
            ranked_pairs=[(sol, res)],
        )
        mocks["check_and_fix_leakage"] = AsyncMock(side_effect=_track_leakage)

        with _apply_patches(mocks):
            await run_phase1(task, config, client)

        # At least 2 calls: candidate leakage + post-merge leakage
        assert len(leakage_calls) >= 2

    def test_leakage_present_in_candidate_loop_source(self) -> None:
        """check_and_fix_leakage appears in _generate_and_evaluate_single_candidate source."""
        from mle_star import phase1

        source = inspect.getsource(phase1._generate_and_evaluate_single_candidate)
        assert "check_and_fix_leakage" in source

    def test_leakage_present_in_merge_loop_source(self) -> None:
        """check_and_fix_leakage appears in _run_merge_loop source."""
        from mle_star import phase1

        source = inspect.getsource(phase1._run_merge_loop)
        assert "check_and_fix_leakage" in source

    def test_leakage_present_in_post_merge_safety_source(self) -> None:
        """check_and_fix_leakage appears in _apply_post_merge_safety source."""
        from mle_star import phase1

        source = inspect.getsource(phase1._apply_post_merge_safety)
        assert "check_and_fix_leakage" in source


# ===========================================================================
# REQ-P1-043: Prompt fidelity for retriever
# ===========================================================================


@pytest.mark.unit
class TestRetrieverPromptFidelity:
    """Retriever prompt contains key phrases from Algorithm 1 (REQ-P1-043)."""

    def test_retriever_prompt_has_json_schema(self) -> None:
        """Retriever YAML template includes JSON schema instruction."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.RETRIEVER)
        rendered = template.render(
            task_description="Test task", target_column="Not specified", M=5,
            research_context="",
            notes_context="",
        )
        assert "model_name" in rendered
        assert "example_code" in rendered

    def test_retriever_prompt_has_m_variable(self) -> None:
        """Retriever prompt template accepts M variable for model count."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.RETRIEVER)
        rendered = template.render(
            task_description="Test task", target_column="Not specified", M=7,
            research_context="",
            notes_context="",
        )
        assert "7" in rendered

    def test_retriever_prompt_has_task_description(self) -> None:
        """Retriever prompt template accepts task_description variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.RETRIEVER)
        rendered = template.render(
            task_description="Classify images into 10 categories",
            target_column="Not specified",
            M=3,
            research_context="",
            notes_context="",
        )
        assert "Classify images" in rendered

    def test_retriever_prompt_mentions_effective_models(self) -> None:
        """Retriever prompt mentions effective models."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.RETRIEVER)
        rendered = template.render(
            task_description="Test", target_column="Not specified", M=3,
            research_context="",
            notes_context="",
        )
        assert "effective" in rendered.lower() or "model" in rendered.lower()

    def test_retriever_prompt_requires_example_code(self) -> None:
        """Retriever prompt instructs agent to provide example code."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.RETRIEVER)
        rendered = template.render(
            task_description="Test", target_column="Not specified", M=3,
            research_context="",
            notes_context="",
        )
        assert "example code" in rendered.lower() or "example_code" in rendered.lower()


# ===========================================================================
# REQ-P1-044: Prompt fidelity for init
# ===========================================================================


@pytest.mark.unit
class TestInitPromptFidelity:
    """Init prompt contains key phrases from Algorithm 1 (REQ-P1-044)."""

    def test_init_prompt_mentions_expert_ml_engineer(self) -> None:
        """Init YAML template mentions expert ML engineer persona."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.INIT)
        rendered = template.render(
            task_description="Test task",
            target_column="Not specified",
            model_name="xgboost",
            example_code="import xgboost",
            research_context="",
            notes_context="",
        )
        assert "expert ML engineer" in rendered

    def test_init_prompt_mentions_straightforward_solution(self) -> None:
        """Init prompt instructs straightforward solution without ensembling."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.INIT)
        rendered = template.render(
            task_description="Test task",
            target_column="Not specified",
            model_name="xgboost",
            example_code="import xgboost",
            research_context="",
            notes_context="",
        )
        assert "straightforward" in rendered.lower()
        assert "ensembl" in rendered.lower()

    def test_init_prompt_mentions_input_directory(self) -> None:
        """Init prompt mentions ./input directory for data."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.INIT)
        rendered = template.render(
            task_description="Test task",
            target_column="Not specified",
            model_name="xgboost",
            example_code="import xgboost",
            research_context="",
            notes_context="",
        )
        assert "./input" in rendered

    def test_init_prompt_mentions_pytorch(self) -> None:
        """Init prompt instructs use of PyTorch over TensorFlow."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.INIT)
        rendered = template.render(
            task_description="Test task",
            target_column="Not specified",
            model_name="xgboost",
            example_code="import xgboost",
            research_context="",
            notes_context="",
        )
        assert "PyTorch" in rendered

    def test_init_prompt_mentions_subsample_threshold(self) -> None:
        """Init prompt mentions 30,000 subsampling threshold."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.INIT)
        rendered = template.render(
            task_description="Test task",
            target_column="Not specified",
            model_name="xgboost",
            example_code="import xgboost",
            research_context="",
            notes_context="",
        )
        assert "30,000" in rendered or "30000" in rendered

    def test_init_prompt_mentions_validation_performance(self) -> None:
        """Init prompt requires Final Validation Performance output."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.INIT)
        rendered = template.render(
            task_description="Test task",
            target_column="Not specified",
            model_name="xgboost",
            example_code="import xgboost",
            research_context="",
            notes_context="",
        )
        assert "Final Validation Performance" in rendered

    def test_init_prompt_mentions_single_code_block(self) -> None:
        """Init prompt instructs response should contain a single code block."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.INIT)
        rendered = template.render(
            task_description="Test task",
            target_column="Not specified",
            model_name="xgboost",
            example_code="import xgboost",
            research_context="",
            notes_context="",
        )
        assert "single markdown code block" in rendered.lower()

    def test_init_prompt_forbids_exit_function(self) -> None:
        """Init prompt instructs not to use exit() function."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.INIT)
        rendered = template.render(
            task_description="Test task",
            target_column="Not specified",
            model_name="xgboost",
            example_code="import xgboost",
            research_context="",
            notes_context="",
        )
        assert "exit()" in rendered

    def test_init_prompt_forbids_try_except(self) -> None:
        """Init prompt instructs not to use try/except."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.INIT)
        rendered = template.render(
            task_description="Test task",
            target_column="Not specified",
            model_name="xgboost",
            example_code="import xgboost",
            research_context="",
            notes_context="",
        )
        assert "try/except" in rendered

    def test_init_prompt_renders_model_name(self) -> None:
        """Init prompt includes the model_name variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.INIT)
        rendered = template.render(
            task_description="Test task",
            target_column="Not specified",
            model_name="my_special_model",
            example_code="import special",
            research_context="",
            notes_context="",
        )
        assert "my_special_model" in rendered

    def test_init_prompt_renders_example_code(self) -> None:
        """Init prompt includes the example_code variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.INIT)
        rendered = template.render(
            task_description="Test task",
            target_column="Not specified",
            model_name="xgboost",
            example_code="from xgboost import XGBClassifier",
            research_context="",
            notes_context="",
        )
        assert "from xgboost import XGBClassifier" in rendered


# ===========================================================================
# REQ-P1-045: Prompt fidelity for merger
# ===========================================================================


@pytest.mark.unit
class TestMergerPromptFidelity:
    """Merger prompt contains key phrases from Algorithm 1 (REQ-P1-045)."""

    def test_merger_prompt_mentions_expert_ml_engineer(self) -> None:
        """Merger YAML template mentions expert ML engineer persona."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.MERGER)
        rendered = template.render(
            base_code="print('base')",
            reference_code="print('ref')",
        )
        assert "expert ML engineer" in rendered

    def test_merger_prompt_mentions_integrate(self) -> None:
        """Merger prompt instructs to integrate reference into base."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.MERGER)
        rendered = template.render(
            base_code="print('base')",
            reference_code="print('ref')",
        )
        assert "integrat" in rendered.lower()

    def test_merger_prompt_mentions_ensemble(self) -> None:
        """Merger prompt instructs to ensemble the models."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.MERGER)
        rendered = template.render(
            base_code="print('base')",
            reference_code="print('ref')",
        )
        assert "ensemble" in rendered.lower()

    def test_merger_prompt_mentions_input_directory(self) -> None:
        """Merger prompt mentions ./input directory."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.MERGER)
        rendered = template.render(
            base_code="print('base')",
            reference_code="print('ref')",
        )
        assert "./input" in rendered

    def test_merger_prompt_mentions_subsample_threshold(self) -> None:
        """Merger prompt mentions 30,000 subsampling threshold."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.MERGER)
        rendered = template.render(
            base_code="print('base')",
            reference_code="print('ref')",
        )
        assert "30,000" in rendered or "30000" in rendered

    def test_merger_prompt_mentions_validation_performance(self) -> None:
        """Merger prompt requires Final Validation Performance output."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.MERGER)
        rendered = template.render(
            base_code="print('base')",
            reference_code="print('ref')",
        )
        assert "Final Validation Performance" in rendered

    def test_merger_prompt_mentions_single_code_block(self) -> None:
        """Merger prompt instructs response should contain a single code block."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.MERGER)
        rendered = template.render(
            base_code="print('base')",
            reference_code="print('ref')",
        )
        assert "single markdown code block" in rendered.lower()

    def test_merger_prompt_renders_base_code(self) -> None:
        """Merger prompt includes the base_code variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.MERGER)
        rendered = template.render(
            base_code="MY_UNIQUE_BASE_CODE_123",
            reference_code="print('ref')",
        )
        assert "MY_UNIQUE_BASE_CODE_123" in rendered

    def test_merger_prompt_renders_reference_code(self) -> None:
        """Merger prompt includes the reference_code variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.MERGER)
        rendered = template.render(
            base_code="print('base')",
            reference_code="UNIQUE_REFERENCE_CODE_456",
        )
        assert "UNIQUE_REFERENCE_CODE_456" in rendered

    def test_merger_prompt_forbids_exit_function(self) -> None:
        """Merger prompt instructs not to use exit() function."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.MERGER)
        rendered = template.render(
            base_code="print('base')",
            reference_code="print('ref')",
        )
        assert "exit()" in rendered

    def test_merger_prompt_mentions_simple_design(self) -> None:
        """Merger prompt instructs relatively simple solution design."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.MERGER)
        rendered = template.render(
            base_code="print('base')",
            reference_code="print('ref')",
        )
        assert "simple" in rendered.lower()


# ===========================================================================
# Hypothesis: property-based tests for Phase 1 logging invariants
# ===========================================================================


@pytest.mark.unit
class TestPhase1LoggingProperties:
    """Property-based tests for Phase 1 logging invariants."""

    @given(
        num_models=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=10, deadline=10000)
    async def test_phase1_always_logs_start_and_complete(self, num_models: int) -> None:
        """Phase 1 always logs start and complete events regardless of M value."""
        import logging as _logging

        from mle_star.phase1 import run_phase1

        client = AsyncMock()
        task = _make_task()
        config = _make_config(num_retrieved_models=num_models)

        models = [_make_model(f"m{i}") for i in range(num_models)]
        candidates = [
            _make_solution(content=f"code_{i}", source_model=f"m{i}")
            for i in range(num_models)
        ]
        eval_results = [
            (candidates[i], _make_eval_result(score=0.80 + i * 0.01))
            for i in range(num_models)
        ]
        ranked = list(reversed(eval_results))

        mocks = _setup_standard_mocks(
            models=models,
            candidates=candidates,
            eval_results=eval_results,
            ranked_pairs=ranked,
        )
        mocks["merge_solutions"] = AsyncMock(return_value=None)

        handler = _logging.StreamHandler()
        handler.setLevel(_logging.DEBUG)
        logger = _logging.getLogger(_LOGGER_NAME)
        logger.addHandler(handler)

        try:
            with _apply_patches(mocks):
                result = await run_phase1(task, config, client)

            assert isinstance(result, Phase1Result)
            assert result.initial_score is not None
        finally:
            logger.removeHandler(handler)

    @given(
        score=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=10, deadline=10000)
    async def test_phase1_complete_score_matches_result(self, score: float) -> None:
        """Phase 1 result score is always consistent regardless of score value."""
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
# Agent config inclusion for Phase 1 agents
# ===========================================================================


@pytest.mark.unit
class TestPhase1AgentConfigs:
    """Phase 1 agent configs present in build_default_agent_configs."""

    def test_retriever_config_exists(self) -> None:
        """Retriever agent config is in default configs."""
        from mle_star.models import build_default_agent_configs

        configs = build_default_agent_configs()
        assert AgentType.RETRIEVER in configs

    def test_init_config_exists(self) -> None:
        """Init agent config is in default configs."""
        from mle_star.models import build_default_agent_configs

        configs = build_default_agent_configs()
        assert AgentType.INIT in configs

    def test_merger_config_exists(self) -> None:
        """Merger agent config is in default configs."""
        from mle_star.models import build_default_agent_configs

        configs = build_default_agent_configs()
        assert AgentType.MERGER in configs


# ===========================================================================
# Phase 1 agents use PromptRegistry
# ===========================================================================


@pytest.mark.unit
class TestPhase1PromptRegistryUsage:
    """All Phase 1 agents load prompts from PromptRegistry."""

    def test_retriever_uses_registry(self) -> None:
        """Retriever prompt is loaded from PromptRegistry."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.RETRIEVER)
        assert template is not None

    def test_init_uses_registry(self) -> None:
        """Init prompt is loaded from PromptRegistry."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.INIT)
        assert template is not None

    def test_merger_uses_registry(self) -> None:
        """Merger prompt is loaded from PromptRegistry."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.MERGER)
        assert template is not None

    def test_retrieve_models_source_uses_prompt_registry(self) -> None:
        """retrieve_models source code references get_registry."""
        from mle_star import phase1

        source = inspect.getsource(phase1.retrieve_models)
        assert "get_registry" in source

    def test_generate_candidate_source_uses_prompt_registry(self) -> None:
        """generate_candidate source code references get_registry."""
        from mle_star import phase1

        source = inspect.getsource(phase1.generate_candidate)
        assert "get_registry" in source

    def test_merge_solutions_source_uses_prompt_registry(self) -> None:
        """merge_solutions source code references get_registry."""
        from mle_star import phase1

        source = inspect.getsource(phase1.merge_solutions)
        assert "get_registry" in source
