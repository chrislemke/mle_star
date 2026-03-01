"""Tests for Phase 2 inner loop constraints (Task 26).

Validates non-functional requirements for ``run_phase2_inner_loop``:
graceful error handling (never raises on agent failure), structured
logging at correct levels, sequential iteration execution, monotonic
best-score invariant, exact iteration count, and immutability of
input parameters.

Tests are written TDD-first and serve as the executable specification
for REQ-P2I-039 through REQ-P2I-050.

Refs:
    SRS 06d (Phase 2 Inner Constraints), IMPLEMENTATION_PLAN.md Task 26.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, patch

from mle_star.models import (
    CodeBlock,
    CodeBlockCategory,
    DataModality,
    EvaluationResult,
    InnerLoopResult,
    MetricDirection,
    PipelineConfig,
    RefinementAttempt,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
    TaskType,
)
import pytest

# ---------------------------------------------------------------------------
# Module path constant for patching
# ---------------------------------------------------------------------------

_MODULE = "mle_star.phase2_inner"


# ---------------------------------------------------------------------------
# Helper factories
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
        description="Test task",
    )


def _make_config(inner_loop_steps: int = 4) -> PipelineConfig:
    """Create a PipelineConfig with a specified K value."""
    return PipelineConfig(inner_loop_steps=inner_loop_steps)


def _make_solution(
    content: str = "original code with TARGET_BLOCK here",
) -> SolutionScript:
    """Create a mutable SolutionScript for testing."""
    return SolutionScript(content=content, phase=SolutionPhase.REFINED)


def _make_code_block(content: str = "TARGET_BLOCK") -> CodeBlock:
    """Create a CodeBlock targeting a known substring."""
    return CodeBlock(content=content, category=CodeBlockCategory.TRAINING)


def _make_eval_result(
    score: float | None = 0.85,
    is_error: bool = False,
    duration_seconds: float = 1.0,
) -> EvaluationResult:
    """Create an EvaluationResult with the given score."""
    return EvaluationResult(
        score=score,
        stdout="Final Validation Performance: 0.85",
        stderr="" if not is_error else "Traceback: Error",
        exit_code=0 if not is_error else 1,
        duration_seconds=duration_seconds,
        is_error=is_error,
    )


# ===========================================================================
# REQ-P2I-041: Never Raises on Agent Failure
# ===========================================================================


@pytest.mark.unit
class TestNeverRaisesOnAgentFailure:
    """run_phase2_inner_loop never raises on agent failure (REQ-P2I-041)."""

    async def test_all_coder_failures_returns_valid_result(self) -> None:
        """When all K coder calls return None, returns valid InnerLoopResult."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=4)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="some plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert isinstance(result, InnerLoopResult)
        assert result.improved is False
        # Early stopping (patience=2): k=0 fails (streak=1), k=1 fails (streak=2, break)
        assert len(result.attempts) == 2

    async def test_all_planner_failures_returns_valid_result(self) -> None:
        """When all planner calls return None (k>=1), returns valid InnerLoopResult."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=4)

        # k=0 uses initial_plan (no planner), but coder fails;
        # k=1..3 planner returns None => all skip.
        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert isinstance(result, InnerLoopResult)
        assert result.improved is False
        # Early stopping (patience=2): k=0 coder fails (streak=1), k=1 planner fails (streak=2, break)
        assert len(result.attempts) == 2

    async def test_mixed_failures_returns_valid_result(self) -> None:
        """When a mix of coder/planner failures occur, returns valid InnerLoopResult."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=4)

        # k=0: coder returns code (success)
        # k=1: planner returns None (fail)
        # k=2: planner returns plan but coder returns None (fail)
        # k=3: planner returns plan, coder returns code (success)
        coder_returns = [
            "improved_code_v1",
            None,  # won't be called at k=1 (planner fails)
            None,
            "improved_code_v2",
        ]
        planner_returns = [
            None,  # Not called at k=0
            None,  # Fails at k=1
            "plan for k=2",
            "plan for k=3",
        ]

        coder_mock = AsyncMock(side_effect=coder_returns)
        planner_mock = AsyncMock(side_effect=planner_returns[1:])  # Only called at k>=1

        eval_solution = _make_solution(content="evaluated solution")
        eval_result = _make_eval_result(0.85)

        with (
            patch(f"{_MODULE}.invoke_coder", coder_mock),
            patch(f"{_MODULE}.invoke_planner", planner_mock),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert isinstance(result, InnerLoopResult)
        # Early stopping (patience=2): k=0 succeeds (streak=0), k=1 planner fails (streak=1),
        # k=2 coder fails (streak=2, break)
        assert len(result.attempts) == 3
        # At least some attempts should have score=None (failed)
        failed_attempts = [a for a in result.attempts if a.score is None]
        assert len(failed_attempts) >= 1

    async def test_all_evaluations_fail_returns_valid_result(self) -> None:
        """When all evaluations return is_error=True, returns valid InnerLoopResult."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=4)

        error_eval = _make_eval_result(score=None, is_error=True)
        eval_solution = _make_solution(content="failed solution")

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="a plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, error_eval),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert isinstance(result, InnerLoopResult)
        assert result.improved is False
        # Early stopping (patience=2): k=0 eval error (streak=1), k=1 eval error (streak=2, break)
        assert len(result.attempts) == 2


# ===========================================================================
# REQ-P2I-043: Structured Logging -- Inner Loop Start/Complete
# ===========================================================================


@pytest.mark.unit
class TestInnerLoopStartLogging:
    """Inner loop start event is logged at INFO (REQ-P2I-043)."""

    async def test_inner_loop_start_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Inner loop start is logged at INFO with code block length, plan, h_best, K."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block(content="TARGET_BLOCK")
        task = _make_task()
        config = _make_config(inner_loop_steps=3)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan text here",
                best_score=0.80,
                task=task,
                config=config,
            )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_msgs) >= 1, "Expected at least one INFO log at inner loop start"
        # The first INFO log should mention start-related content
        all_info_text = " ".join(r.message for r in info_msgs)
        # Must mention code block length
        assert (
            "12" in all_info_text
            or "length" in all_info_text.lower()
            or "block" in all_info_text.lower()
        )
        # Must mention K value
        assert (
            "3" in all_info_text
            or "K" in all_info_text
            or "steps" in all_info_text.lower()
        )

    async def test_inner_loop_start_logs_initial_plan_truncated(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Inner loop start log includes initial plan text truncated to 200 chars."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        long_plan = "A" * 300

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan=long_plan,
                best_score=0.80,
                task=task,
                config=config,
            )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # The full 300-char plan should NOT appear in any single log message
        assert long_plan not in all_info_text


@pytest.mark.unit
class TestInnerLoopCompleteLogging:
    """Inner loop complete event is logged at INFO (REQ-P2I-043)."""

    async def test_inner_loop_complete_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Inner loop complete is logged with total attempts, best score, improved."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=2)

        eval_solution = _make_solution(
            content="evaluated solution with TARGET_BLOCK here"
        )
        eval_result = _make_eval_result(0.90)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="new plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        # The last INFO message should be the completion message
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention total attempts
        assert "2" in all_info_text or "attempt" in all_info_text.lower()
        # Should mention best score
        assert "0.9" in all_info_text or "score" in all_info_text.lower()
        # Should mention improved status
        assert "improved" in all_info_text.lower() or "yes" in all_info_text.lower()

    async def test_inner_loop_complete_logs_successful_eval_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Inner loop complete log includes count of successful evaluations."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=3)

        # k=0: coder None (fail), k=1: plan ok, coder ok (success), k=2: plan ok, coder ok (success)
        coder_side = [None, "improved_v1", "improved_v2"]
        coder_mock = AsyncMock(side_effect=coder_side)

        eval_solution = _make_solution(content="evaluated with TARGET_BLOCK here")
        eval_result = _make_eval_result(0.85)

        with (
            patch(f"{_MODULE}.invoke_coder", coder_mock),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention number of successful evaluations (2 out of 3)
        assert "2" in all_info_text or "success" in all_info_text.lower()


# ===========================================================================
# REQ-P2I-043: Structured Logging -- Coder Invocation
# ===========================================================================


@pytest.mark.unit
class TestCoderLogging:
    """Coder invocation events are logged at correct levels (REQ-P2I-043)."""

    async def test_coder_invocation_start_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_coder invocation start is logged at INFO with step k and plan text."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="optimize the model",
                best_score=0.80,
                task=task,
                config=config,
            )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention coder invocation at step k=0
        assert "coder" in all_info_text.lower()
        assert "0" in all_info_text  # step k=0

    async def test_coder_invocation_start_truncates_plan(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_coder invocation start log truncates plan text to first 200 chars."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        long_plan = "X" * 400

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan=long_plan,
                best_score=0.80,
                task=task,
                config=config,
            )

        # No single log message should contain the full 400-char plan
        for record in caplog.records:
            assert long_plan not in record.message

    async def test_coder_invocation_complete_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_coder invocation complete is logged at INFO with step k and output length."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        coder_output = "improved_code_block"

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value=coder_output,
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention coder completion with output length
        assert "coder" in all_info_text.lower()
        assert (
            str(len(coder_output)) in all_info_text or "length" in all_info_text.lower()
        )

    async def test_coder_failure_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When coder returns None, a WARNING is logged mentioning the step."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1
        warning_text = " ".join(r.message for r in warning_msgs)
        assert (
            "coder" in warning_text.lower()
            or "k=0" in warning_text
            or "skip" in warning_text.lower()
        )

    async def test_coder_complete_logs_failed_to_parse_on_none(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When coder returns None, completion log mentions 'failed to parse' or equivalent."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        all_msgs = " ".join(r.message for r in caplog.records)
        # Should indicate failure to parse or None result
        assert (
            "failed" in all_msgs.lower()
            or "none" in all_msgs.lower()
            or "skip" in all_msgs.lower()
        )


# ===========================================================================
# REQ-P2I-043: Structured Logging -- Planner Invocation
# ===========================================================================


@pytest.mark.unit
class TestPlannerLogging:
    """Planner invocation events are logged at correct levels (REQ-P2I-043)."""

    async def test_planner_invocation_start_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_planner invocation start is logged at INFO with step k and history count."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=2)

        eval_solution = _make_solution(content="evaluated with TARGET_BLOCK here")
        eval_result = _make_eval_result(0.85)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="new plan from planner",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention planner invocation at step k=1
        assert "planner" in all_info_text.lower()
        # Should mention the history size (1 previous attempt at k=1)
        assert (
            "1" in all_info_text
            or "history" in all_info_text.lower()
            or "previous" in all_info_text.lower()
        )

    async def test_planner_invocation_complete_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_planner invocation complete is logged at INFO with step k and plan text."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=2)

        eval_solution = _make_solution(content="evaluated with TARGET_BLOCK here")
        eval_result = _make_eval_result(0.85)
        plan_text = "use gradient clipping for stability"

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value=plan_text,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention planner completion
        assert "planner" in all_info_text.lower()
        # Should include (truncated) plan text
        assert "gradient" in all_info_text.lower() or "plan" in all_info_text.lower()

    async def test_planner_empty_response_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When planner returns None, a WARNING is logged mentioning step k."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=2)

        eval_solution = _make_solution(content="evaluated with TARGET_BLOCK here")
        eval_result = _make_eval_result(0.85)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1
        warning_text = " ".join(r.message for r in warning_msgs)
        assert "planner" in warning_text.lower() or "k=1" in warning_text


# ===========================================================================
# REQ-P2I-043: Structured Logging -- Code Block Replacement
# ===========================================================================


@pytest.mark.unit
class TestReplacementLogging:
    """Code block replacement events are logged at correct levels (REQ-P2I-043)."""

    async def test_replacement_success_logs_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Successful code block replacement is logged at DEBUG with block lengths."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution(content="original code with TARGET_BLOCK here")
        code_block = _make_code_block(content="TARGET_BLOCK")
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        new_code = "IMPROVED_TARGET_BLOCK"

        eval_solution = _make_solution(
            content="evaluated with IMPROVED_TARGET_BLOCK here"
        )
        eval_result = _make_eval_result(0.85)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value=new_code,
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        debug_msgs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        all_debug_text = " ".join(r.message for r in debug_msgs)
        # Should mention replacement success with original and new block lengths
        assert "replac" in all_debug_text.lower() or "block" in all_debug_text.lower()
        # Should include length values: original "TARGET_BLOCK" = 12 chars
        assert (
            str(len("TARGET_BLOCK")) in all_debug_text
            or "length" in all_debug_text.lower()
        )

    async def test_replacement_failure_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Failed code block replacement is logged at WARNING with error message."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        # Solution does NOT contain the code block content -> replace_block raises
        solution = _make_solution(content="no matching block here")
        code_block = _make_code_block(content="MISSING_BLOCK")
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="new code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1
        warning_text = " ".join(r.message for r in warning_msgs)
        assert "replace" in warning_text.lower() or "block" in warning_text.lower()


# ===========================================================================
# REQ-P2I-043: Structured Logging -- Leakage Check
# ===========================================================================


@pytest.mark.unit
class TestLeakageLogging:
    """Leakage check events are logged at correct levels (REQ-P2I-043)."""

    async def test_leakage_check_start_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Leakage check start is logged at INFO with solution content length."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        eval_solution = _make_solution(content="improved with TARGET_BLOCK here")

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "leak" in all_info_text.lower()

    async def test_leakage_check_complete_no_change_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Leakage check complete is logged at INFO indicating no change."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        eval_solution = _make_solution(content="improved with TARGET_BLOCK here")

        # check_and_fix_leakage returns the same solution (no change)
        # Need is_improvement_or_equal=True so leakage check is triggered post-loop
        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention leakage check result
        assert "leak" in all_info_text.lower()

    async def test_leakage_check_complete_with_change_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Leakage check complete is logged at INFO indicating content changed."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        eval_solution = _make_solution(content="improved with TARGET_BLOCK here")

        # check_and_fix_leakage returns a DIFFERENT solution
        modified_solution = _make_solution(
            content="leakage-fixed code with TARGET_BLOCK here"
        )

        # Need is_improvement_or_equal=True so leakage check is triggered post-loop
        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=modified_solution,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention leakage and indicate content was changed
        assert "leak" in all_info_text.lower()
        assert (
            "yes" in all_info_text.lower()
            or "changed" in all_info_text.lower()
            or "found" in all_info_text.lower()
            or "detected" in all_info_text.lower()
            or "corrected" in all_info_text.lower()
        )


# ===========================================================================
# REQ-P2I-043: Structured Logging -- Evaluation
# ===========================================================================


@pytest.mark.unit
class TestEvaluationLogging:
    """Evaluation events are logged at correct levels (REQ-P2I-043)."""

    async def test_evaluation_start_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Evaluation start is logged at INFO with step k and solution content length."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        eval_solution = _make_solution(content="evaluated with TARGET_BLOCK here")
        eval_result = _make_eval_result(0.90, duration_seconds=2.5)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention evaluation start
        assert "eval" in all_info_text.lower()
        assert "k=0" in all_info_text or "0" in all_info_text

    async def test_evaluation_complete_with_score_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Evaluation complete is logged at INFO with score and duration."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        eval_solution = _make_solution(content="evaluated with TARGET_BLOCK here")
        eval_result = _make_eval_result(0.92, duration_seconds=3.5)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention score
        assert "0.92" in all_info_text or "score" in all_info_text.lower()
        # Should mention duration
        assert "3.5" in all_info_text or "duration" in all_info_text.lower()

    async def test_evaluation_complete_with_error_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Evaluation complete with is_error logs at INFO with 'failed' indication."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        eval_solution = _make_solution(content="error code with TARGET_BLOCK here")
        eval_result = _make_eval_result(score=None, is_error=True, duration_seconds=1.0)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention evaluation failure or error
        assert "failed" in all_info_text.lower() or "error" in all_info_text.lower()


# ===========================================================================
# REQ-P2I-043: Structured Logging -- Best Score Update
# ===========================================================================


@pytest.mark.unit
class TestBestScoreLogging:
    """Best score update events are logged at INFO (REQ-P2I-043)."""

    async def test_best_score_updated_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When best score is updated, an INFO log includes old and new scores."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        eval_solution = _make_solution(content="improved with TARGET_BLOCK here")
        eval_result = _make_eval_result(0.95)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention best score update with old and new values
        assert "best" in all_info_text.lower() or "update" in all_info_text.lower()
        assert "0.8" in all_info_text  # old best
        assert "0.95" in all_info_text  # new best

    async def test_best_score_not_updated_no_update_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When best score is NOT updated, no 'best score updated' INFO log."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        eval_solution = _make_solution(content="worse with TARGET_BLOCK here")
        eval_result = _make_eval_result(0.70)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        # Check that no INFO log mentions "best score updated" or similar
        for record in info_msgs:
            msg_lower = record.message.lower()
            # Allow "best" in the completion message, but not a specific
            # "updated" message about best score changing
            if "update" in msg_lower and "best" in msg_lower:
                pytest.fail(
                    f"Unexpected 'best score updated' INFO log when score did not improve: "
                    f"{record.message!r}"
                )


# ===========================================================================
# REQ-P2I-043: Structured Logging -- Attempt Skipped
# ===========================================================================


@pytest.mark.unit
class TestAttemptSkippedLogging:
    """Attempt skipped events are logged at WARNING (REQ-P2I-043)."""

    async def test_attempt_skipped_coder_failure_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When coder fails and attempt is skipped, WARNING is logged with reason."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1
        warning_text = " ".join(r.message for r in warning_msgs)
        # Should mention the step number and the reason for skipping
        assert "0" in warning_text or "k=" in warning_text
        assert (
            "skip" in warning_text.lower()
            or "coder" in warning_text.lower()
            or "none" in warning_text.lower()
        )

    async def test_attempt_skipped_planner_failure_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When planner fails and attempt is skipped, WARNING is logged."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=2)

        eval_solution = _make_solution(content="evaluated with TARGET_BLOCK here")
        eval_result = _make_eval_result(0.85)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
            caplog.at_level(logging.DEBUG, logger="mle_star.phase2_inner"),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1
        warning_text = " ".join(r.message for r in warning_msgs)
        assert "planner" in warning_text.lower() or "k=1" in warning_text


# ===========================================================================
# REQ-P2I-046: Sequential Iteration Execution
# ===========================================================================


@pytest.mark.unit
class TestSequentialExecution:
    """Iterations run sequentially, not concurrently (REQ-P2I-046)."""

    async def test_iterations_run_sequentially(self) -> None:
        """Each iteration completes before the next begins (no concurrent execution)."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=3)

        # Track the order of coder invocations to verify sequential execution
        call_order: list[int] = []
        call_completed: list[int] = []

        async def tracked_coder(code: str, plan: str, client: Any, **kwargs: Any) -> str:
            step = len(call_order)
            call_order.append(step)
            # Verify the previous call completed before this one started
            if step > 0:
                assert step - 1 in call_completed, (
                    f"Step {step} started before step {step - 1} completed"
                )
            call_completed.append(step)
            return "improved code"

        eval_solution = _make_solution(content="evaluated with TARGET_BLOCK here")
        eval_result = _make_eval_result(0.85)

        # Use is_improvement_or_equal=True to avoid early stopping with K=3
        with (
            patch(f"{_MODULE}.invoke_coder", side_effect=tracked_coder),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="new plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # All 3 iterations should have been called in order
        assert call_order == [0, 1, 2]
        assert call_completed == [0, 1, 2]

    async def test_no_concurrent_evaluations(self) -> None:
        """Evaluations are never running concurrently."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=3)

        active_evals = 0
        max_concurrent = 0

        async def tracked_eval(
            sol: Any, task: Any, config: Any, callback: Any
        ) -> tuple[Any, Any]:
            nonlocal active_evals, max_concurrent
            active_evals += 1
            if active_evals > max_concurrent:
                max_concurrent = active_evals
            result = (
                _make_solution(content="eval result with TARGET_BLOCK here"),
                _make_eval_result(0.85),
            )
            active_evals -= 1
            return result

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="new plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(f"{_MODULE}.evaluate_with_retry", side_effect=tracked_eval),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert max_concurrent == 1, (
            f"Expected max 1 concurrent eval, got {max_concurrent}"
        )


# ===========================================================================
# REQ-P2I-047: Monotonic Best Score
# ===========================================================================


@pytest.mark.unit
class TestMonotonicBestScore:
    """local_best_score is monotonically non-decreasing (maximize) or non-increasing (minimize) (REQ-P2I-047)."""

    async def test_best_score_never_decreases_maximize(self) -> None:
        """With MAXIMIZE, best score never decreases across iterations."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        # Use "improved code" as the code block so replace_block works after compounding
        solution = _make_solution(content="original code with improved code here")
        code_block = _make_code_block(content="improved code")
        task = _make_task(direction=MetricDirection.MAXIMIZE)
        config = _make_config(inner_loop_steps=4)

        # Scores: 0.82, 0.78 (worse), 0.90, 0.85 (worse than 0.90)
        scores = [0.82, 0.78, 0.90, 0.85]
        eval_call_count = 0

        async def varying_eval(sol: Any, t: Any, c: Any, cb: Any) -> tuple[Any, Any]:
            nonlocal eval_call_count
            score = scores[eval_call_count]
            eval_call_count += 1
            eval_result = _make_eval_result(score)
            # Return solution containing "improved code" so replace_block works on next iteration
            return (
                _make_solution(content=f"sol-{eval_call_count} with improved code here"),
                eval_result,
            )

        # Use real is_improvement_or_equal to test actual monotonic behavior
        from mle_star.scoring import (
            is_improvement as real_is_improvement,
            is_improvement_or_equal as real_is_improvement_or_equal,
        )

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="new plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(f"{_MODULE}.evaluate_with_retry", side_effect=varying_eval),
            patch(
                f"{_MODULE}.is_improvement_or_equal",
                side_effect=real_is_improvement_or_equal,
            ),
            patch(f"{_MODULE}.is_improvement", side_effect=real_is_improvement),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # Best score should be 0.90 (the highest seen)
        assert result.best_score == 0.90
        # The was_improvement flags should track correctly
        # k=0: 0.82 >= 0.80 -> True
        # k=1: 0.78 >= 0.82 -> False
        # k=2: 0.90 >= 0.82 -> True
        # k=3: 0.85 >= 0.90 -> False
        assert result.attempts[0].was_improvement is True
        assert result.attempts[1].was_improvement is False
        assert result.attempts[2].was_improvement is True
        assert result.attempts[3].was_improvement is False

    async def test_best_score_never_increases_minimize(self) -> None:
        """With MINIMIZE, best score never increases across iterations."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        # Use "improved code" as the code block so replace_block works after compounding
        solution = _make_solution(content="original code with improved code here")
        code_block = _make_code_block(content="improved code")
        task = _make_task(direction=MetricDirection.MINIMIZE)
        config = _make_config(inner_loop_steps=4)

        # Scores: 0.45, 0.50 (worse for minimize), 0.30, 0.35 (worse than 0.30)
        scores = [0.45, 0.50, 0.30, 0.35]
        eval_call_count = 0

        async def varying_eval(sol: Any, t: Any, c: Any, cb: Any) -> tuple[Any, Any]:
            nonlocal eval_call_count
            score = scores[eval_call_count]
            eval_call_count += 1
            eval_result = _make_eval_result(score)
            return (
                _make_solution(content=f"sol-{eval_call_count} with improved code here"),
                eval_result,
            )

        from mle_star.scoring import (
            is_improvement as real_is_improvement,
            is_improvement_or_equal as real_is_improvement_or_equal,
        )

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="new plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(f"{_MODULE}.evaluate_with_retry", side_effect=varying_eval),
            patch(
                f"{_MODULE}.is_improvement_or_equal",
                side_effect=real_is_improvement_or_equal,
            ),
            patch(f"{_MODULE}.is_improvement", side_effect=real_is_improvement),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.50,
                task=task,
                config=config,
            )

        # Best score should be 0.30 (the lowest seen, since minimizing)
        assert result.best_score == 0.30

    async def test_best_score_monotonic_with_failed_evaluations(self) -> None:
        """Best score stays monotonic even when some evaluations fail (score=None)."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        # Use "improved code" as the code block so replace_block works after compounding
        solution = _make_solution(content="original code with improved code here")
        code_block = _make_code_block(content="improved code")
        task = _make_task(direction=MetricDirection.MAXIMIZE)
        config = _make_config(inner_loop_steps=3)

        # k=0: score 0.85, k=1: None (fail), k=2: score 0.90
        scores_iter = iter([0.85, None, 0.90])
        eval_call_count = 0

        async def varying_eval(sol: Any, t: Any, c: Any, cb: Any) -> tuple[Any, Any]:
            nonlocal eval_call_count
            score = next(scores_iter)
            eval_call_count += 1
            eval_result = _make_eval_result(score=score, is_error=(score is None))
            return (
                _make_solution(content=f"sol-{eval_call_count} with improved code here"),
                eval_result,
            )

        from mle_star.scoring import (
            is_improvement as real_is_improvement,
            is_improvement_or_equal as real_is_improvement_or_equal,
        )

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="new plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(f"{_MODULE}.evaluate_with_retry", side_effect=varying_eval),
            patch(
                f"{_MODULE}.is_improvement_or_equal",
                side_effect=real_is_improvement_or_equal,
            ),
            patch(f"{_MODULE}.is_improvement", side_effect=real_is_improvement),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # Best should be 0.90, not regressed by the None in between
        assert result.best_score == 0.90
        assert len(result.attempts) == 3


# ===========================================================================
# REQ-P2I-048: Iteration Count = K
# ===========================================================================


@pytest.mark.unit
class TestIterationCount:
    """Given K, exactly K RefinementAttempt records are produced (REQ-P2I-048)."""

    async def test_exactly_k_attempts_all_success(self) -> None:
        """All K iterations succeed and produce K attempts."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=4)

        # eval_solution must contain the coder output so replace_block works
        # after compounding improvements
        eval_solution = _make_solution(content="evaluated with improved code here")
        eval_result = _make_eval_result(0.85)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="new plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert len(result.attempts) == 4

    async def test_exactly_k_attempts_all_failure(self) -> None:
        """All K iterations fail (coder returns None) and early stop produces fewer attempts."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=4)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # Early stopping (patience=2): k=0 fails (streak=1), k=1 fails (streak=2, break)
        assert len(result.attempts) == 2

    async def test_exactly_k_attempts_mixed_planner_coder_failures(self) -> None:
        """Mixed planner and coder failures with early stopping produce correct attempt count."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=5)

        # k=0: coder None (fail), k=1: planner None (fail) -> streak=2, early stop
        coder_returns = [None, "code_v1", "code_v2", None, "code_v3"]
        planner_returns = [None, "plan2", None, "plan4"]  # k>=1 only

        coder_mock = AsyncMock(side_effect=coder_returns)
        planner_mock = AsyncMock(side_effect=planner_returns)

        eval_solution = _make_solution(content="eval with TARGET_BLOCK here")
        eval_result = _make_eval_result(0.85)

        with (
            patch(f"{_MODULE}.invoke_coder", coder_mock),
            patch(f"{_MODULE}.invoke_planner", planner_mock),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # Early stopping (patience=2): k=0 coder fails (streak=1), k=1 planner fails (streak=2, break)
        assert len(result.attempts) == 2

    async def test_k_equals_1_produces_single_attempt(self) -> None:
        """K=1 produces exactly 1 RefinementAttempt."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        eval_solution = _make_solution(content="eval with TARGET_BLOCK here")
        eval_result = _make_eval_result(0.85)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert len(result.attempts) == 1

    async def test_attempt_records_are_refinement_attempt_instances(self) -> None:
        """Each element in attempts is a RefinementAttempt instance."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=3)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert all(isinstance(a, RefinementAttempt) for a in result.attempts)


# ===========================================================================
# REQ-P2I-049: Immutable Input Solution
# ===========================================================================


@pytest.mark.unit
class TestImmutableInputSolution:
    """The solution parameter must not be mutated during the loop (REQ-P2I-049)."""

    async def test_input_solution_not_mutated(self) -> None:
        """Original solution object's content is unchanged after inner loop completes."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        original_content = "original code with TARGET_BLOCK here"
        solution = _make_solution(content=original_content)
        original_score = solution.score
        original_phase = solution.phase
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=4)

        eval_solution = _make_solution(content="modified with TARGET_BLOCK here")
        eval_result = _make_eval_result(0.95)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved TARGET_BLOCK",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="new plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # Verify the original solution was NOT mutated
        assert solution.content == original_content
        assert solution.score == original_score
        assert solution.phase == original_phase

    async def test_input_solution_content_preserved_across_all_iterations(self) -> None:
        """Solution content remains identical across all K iterations."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        original_content = "original code with TARGET_BLOCK here"
        solution = _make_solution(content=original_content)
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=4)

        # Track solution content at each coder invocation
        observed_contents: list[str] = []

        async def tracking_coder(code: str, plan: str, client: Any, **kwargs: Any) -> str:
            # At each call, record the original solution's content
            # (we capture via closure)
            observed_contents.append(solution.content)
            return "improved code"

        # eval_solution must contain coder output for compounding to work
        eval_solution = _make_solution(content="evaluated with improved code here")
        eval_result = _make_eval_result(0.90)

        with (
            patch(f"{_MODULE}.invoke_coder", side_effect=tracking_coder),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="new plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # All observed contents should match the original
        assert all(c == original_content for c in observed_contents)


# ===========================================================================
# REQ-P2I-050: Immutable Code Block
# ===========================================================================


@pytest.mark.unit
class TestImmutableCodeBlock:
    """code_block.content must not be modified during the loop (REQ-P2I-050)."""

    async def test_code_block_content_not_modified(self) -> None:
        """CodeBlock's content field remains unchanged after inner loop completes."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        original_block_content = "TARGET_BLOCK"
        code_block = _make_code_block(content=original_block_content)
        task = _make_task()
        config = _make_config(inner_loop_steps=4)

        eval_solution = _make_solution(content="evaluated with TARGET_BLOCK here")
        eval_result = _make_eval_result(0.90)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="new plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert code_block.content == original_block_content

    async def test_code_block_category_not_modified(self) -> None:
        """CodeBlock's category field remains unchanged after inner loop completes."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block(content="TARGET_BLOCK")
        original_category = code_block.category
        task = _make_task()
        config = _make_config(inner_loop_steps=2)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert code_block.category == original_category

    async def test_coder_receives_original_code_block_every_iteration(self) -> None:
        """When no improvement occurs, invoke_coder always receives the original code_block.content."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        original_block = "TARGET_BLOCK"
        code_block = _make_code_block(content=original_block)
        task = _make_task()
        config = _make_config(inner_loop_steps=3)

        received_code_blocks: list[str] = []

        async def tracking_coder(code: str, plan: str, client: Any, **kwargs: Any) -> str:
            received_code_blocks.append(code)
            return "improved_" + code

        eval_solution = _make_solution(content="evaluated with TARGET_BLOCK here")
        eval_result = _make_eval_result(0.85)

        with (
            patch(f"{_MODULE}.invoke_coder", side_effect=tracking_coder),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(eval_solution, eval_result),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # All coder calls should receive the original code block content (no compounding when no improvement)
        assert all(cb == original_block for cb in received_code_blocks)
        # Early stopping (patience=2): k=0 fails (streak=1), k=1 fails (streak=2, break)
        assert len(received_code_blocks) == 2
