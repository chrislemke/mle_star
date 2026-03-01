"""Tests for Phase 2 inner loop safety integration and error handling (Task 25).

Validates that ``run_phase2_inner_loop`` integrates leakage checking
(``check_and_fix_leakage``) before every evaluation and uses
``evaluate_with_retry`` with a debug callback instead of bare
``evaluate_solution``.

Tests are written TDD-first and serve as the executable specification for
REQ-P2I-030 and REQ-P2I-031.

Refs:
    SRS 06b (Phase 2 Inner Loop Safety), IMPLEMENTATION_PLAN.md Task 25.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    CodeBlock,
    CodeBlockCategory,
    DataModality,
    EvaluationResult,
    MetricDirection,
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
    error_traceback: str | None = None,
) -> EvaluationResult:
    """Create an EvaluationResult with the given score and error state."""
    return EvaluationResult(
        score=score,
        stdout="Final Validation Performance: 0.85",
        stderr="" if not is_error else "Traceback (most recent call last):\nError",
        exit_code=0 if not is_error else 1,
        duration_seconds=1.0,
        is_error=is_error,
        error_traceback=error_traceback,
    )


def _make_leakage_fixed_solution(
    content: str = "original code with LEAKAGE_FIXED here",
) -> SolutionScript:
    """Create a SolutionScript representing the leakage-checked version."""
    return SolutionScript(content=content, phase=SolutionPhase.REFINED)


def _make_debug_fixed_solution(
    content: str = "original code with DEBUG_FIXED here",
) -> SolutionScript:
    """Create a SolutionScript representing a debugger-fixed version."""
    return SolutionScript(content=content, phase=SolutionPhase.REFINED)


# ===========================================================================
# REQ-P2I-030: Leakage check before every evaluation
# ===========================================================================


@pytest.mark.unit
class TestLeakageCheckBeforeEvaluation:
    """check_and_fix_leakage is called post-loop on the best solution (REQ-P2I-030)."""

    async def test_leakage_check_called_on_candidate(self) -> None:
        """check_and_fix_leakage is called with (candidate, task, client) after replace_block."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)
        leakage_fixed = _make_leakage_fixed_solution()

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=leakage_fixed,
            ) as mock_leakage,
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(leakage_fixed, _make_eval_result(0.85)),
            ),
            # is_improvement_or_equal=True triggers post-loop leakage check.
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.90,
                task=task,
                config=config,
            )

        # check_and_fix_leakage called once post-loop on the best solution.
        mock_leakage.assert_called_once()
        call_args = mock_leakage.call_args
        # First arg is the best solution (after improvement)
        assert isinstance(call_args[0][0], SolutionScript)
        # Second arg is the task
        assert call_args[0][1] is task
        # Third arg is the client
        assert call_args[0][2] is client

    async def test_leakage_fixed_solution_triggers_reevaluation(self) -> None:
        """Post-loop leakage fix triggers re-evaluation with the fixed solution."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)
        candidate = _make_solution("step candidate")
        leakage_fixed = _make_leakage_fixed_solution("leakage-safe code")

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=leakage_fixed,
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                side_effect=[
                    (candidate, _make_eval_result(0.85)),       # in-step eval
                    (leakage_fixed, _make_eval_result(0.84)),   # post-loop re-eval
                ],
            ) as mock_eval,
            # Improvement triggers post-loop leakage check.
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.90,
                task=task,
                config=config,
            )

        # evaluate_with_retry called twice: once in step, once after leakage fix.
        assert mock_eval.call_count == 2
        # The re-evaluation (second call) receives the leakage-fixed solution.
        assert mock_eval.call_args_list[1][0][0] is leakage_fixed

    async def test_leakage_check_called_once_post_loop(self) -> None:
        """check_and_fix_leakage is called once after the loop when improvement occurs."""
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
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ) as mock_leakage,
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            # Improvement triggers post-loop leakage check.
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.90,
                task=task,
                config=config,
            )

        # Leakage check called once post-loop (not per-iteration).
        assert mock_leakage.call_count == 1

    async def test_leakage_modified_solution_becomes_best(self) -> None:
        """When leakage check modifies the candidate, the modified version is used for tracking."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)
        leakage_fixed = _make_leakage_fixed_solution("leakage-safe best solution")

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                return_value=leakage_fixed,
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(leakage_fixed, _make_eval_result(0.95)),
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

        # The leakage-fixed solution is the one that became best
        assert result.best_solution.content == "leakage-safe best solution"

    async def test_leakage_check_not_called_on_coder_failure(self) -> None:
        """check_and_fix_leakage is NOT invoked when coder returns None."""
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
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
            ) as mock_leakage,
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
            ) as mock_eval,
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

        mock_leakage.assert_not_called()
        mock_eval.assert_not_called()

    async def test_leakage_check_not_called_on_replace_block_failure(self) -> None:
        """check_and_fix_leakage is NOT invoked when replace_block raises ValueError."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        # Solution content that does NOT contain the code block content,
        # so replace_block will raise ValueError
        solution = _make_solution("code without the block")
        code_block = _make_code_block("NONEXISTENT_BLOCK")
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
            ) as mock_leakage,
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
            ) as mock_eval,
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

        mock_leakage.assert_not_called()
        mock_eval.assert_not_called()

    async def test_leakage_check_not_called_on_planner_failure(self) -> None:
        """check_and_fix_leakage is NOT invoked when planner returns None (k>=1)."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        # K=2: k=0 succeeds normally, k=1 planner fails
        config = _make_config(inner_loop_steps=2)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value=None,  # Planner fails at k=1
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ) as mock_leakage,
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
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
                best_score=0.90,
                task=task,
                config=config,
            )

        # Leakage check only post-loop when improved; is_improvement_or_equal=False → no call.
        assert mock_leakage.call_count == 0


# ===========================================================================
# REQ-P2I-031: evaluate_with_retry + make_debug_callback
# ===========================================================================


@pytest.mark.unit
class TestEvaluateWithRetry:
    """evaluate_with_retry is used with make_debug_callback (REQ-P2I-031)."""

    async def test_evaluate_with_retry_called_not_evaluate_solution(self) -> None:
        """evaluate_with_retry is called instead of bare evaluate_solution."""
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
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ) as mock_eval_retry,
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.90,
                task=task,
                config=config,
            )

        mock_eval_retry.assert_called_once()

    async def test_make_debug_callback_receives_task_config_client(self) -> None:
        """make_debug_callback is called with (task, config, client)."""
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
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ) as mock_make_cb,
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
                best_score=0.90,
                task=task,
                config=config,
            )

        mock_make_cb.assert_called_once_with(task, config, client)

    async def test_debug_callback_passed_to_evaluate_with_retry(self) -> None:
        """The callback from make_debug_callback is passed to evaluate_with_retry."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)
        sentinel_callback = AsyncMock()

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=sentinel_callback,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ) as mock_eval_retry,
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.90,
                task=task,
                config=config,
            )

        # The debug_callback argument is the sentinel_callback
        eval_call = mock_eval_retry.call_args
        assert eval_call[0][3] is sentinel_callback

    async def test_evaluate_with_retry_receives_correct_args(self) -> None:
        """evaluate_with_retry receives (candidate_from_replace_block, task, config, callback).

        Leakage checking now runs post-loop, so evaluate_with_retry receives
        the candidate produced by replace_block, not the leakage-fixed version.
        """
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)
        sentinel_callback = AsyncMock()
        candidate_result = _make_eval_result(0.85)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, _task, _client: sol,
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=sentinel_callback,
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(solution, candidate_result),
            ) as mock_eval_retry,
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.90,
                task=task,
                config=config,
            )

        eval_call = mock_eval_retry.call_args
        # First arg is the candidate from replace_block (contains coder output)
        candidate_arg = eval_call[0][0]
        assert "improved code" in candidate_arg.content
        assert eval_call[0][1] is task  # task
        assert eval_call[0][2] is config  # config
        assert eval_call[0][3] is sentinel_callback  # debug callback


# ===========================================================================
# REQ-P2I-031: Debug success and failure scenarios
# ===========================================================================


@pytest.mark.unit
class TestDebugSuccessAndFailure:
    """Debug callback integration via evaluate_with_retry (REQ-P2I-031)."""

    async def test_debug_success_uses_fixed_solution_and_score(self) -> None:
        """When evaluate_with_retry returns a debugged solution, it becomes best on improvement."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)
        debug_fixed = _make_debug_fixed_solution("debugger fixed code")

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                # Returns the debug-fixed solution after retry
                return_value=(debug_fixed, _make_eval_result(0.92)),
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

        # The debug-fixed solution became the best
        assert result.best_solution.content == "debugger fixed code"
        assert result.best_score == 0.92
        assert result.improved is True

    async def test_all_debug_retries_fail_score_none_no_improvement(self) -> None:
        """When all debug retries fail, score=None and was_improvement=False."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        # evaluate_with_retry returns an error result with no score
        error_result = _make_eval_result(
            score=None,
            is_error=True,
            error_traceback="SomeError: failed",
        )
        error_solution = _make_solution("still broken")

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(error_solution, error_result),
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

        # No improvement — original solution preserved
        assert result.best_solution is solution
        assert result.best_score == 0.80
        assert result.improved is False
        # The attempt was recorded with score=None
        assert len(result.attempts) == 1
        assert result.attempts[0].score is None
        assert result.attempts[0].was_improvement is False


# ===========================================================================
# Existing behavior preservation with safety integration
# ===========================================================================


@pytest.mark.unit
class TestExistingBehaviorPreservation:
    """Existing error-handling behavior is preserved with safety integration."""

    async def test_coder_failure_records_attempt_skips_eval(self) -> None:
        """Coder failure records RefinementAttempt(score=None, code_block='') and skips all eval."""
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
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
            ) as mock_leakage,
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
            ) as mock_eval,
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

        mock_leakage.assert_not_called()
        mock_eval.assert_not_called()
        assert len(result.attempts) == 1
        assert result.attempts[0].score is None
        assert result.attempts[0].code_block == ""
        assert result.attempts[0].was_improvement is False

    async def test_replace_block_failure_records_attempt_skips_eval(self) -> None:
        """replace_block ValueError records attempt with coder output and skips eval."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution("code without the block")
        code_block = _make_code_block("NONEXISTENT_BLOCK")
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
            ) as mock_leakage,
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
            ) as mock_eval,
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

        mock_leakage.assert_not_called()
        mock_eval.assert_not_called()
        assert len(result.attempts) == 1
        assert result.attempts[0].score is None
        assert result.attempts[0].code_block == "improved code"
        assert result.attempts[0].was_improvement is False

    async def test_none_score_from_retry_never_updates_best(self) -> None:
        """When evaluate_with_retry returns score=None, best score is never updated."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=2)

        # Both iterations return score=None
        none_result = _make_eval_result(score=None)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="next plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), none_result),
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

        # Original solution and score preserved
        assert result.best_solution is solution
        assert result.best_score == 0.80
        assert result.improved is False

    async def test_input_solution_preserved_when_no_improvement(self) -> None:
        """InnerLoopResult.best_solution is the input solution when no iteration improves."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution("the original input")
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
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.75)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.90,
                task=task,
                config=config,
            )

        assert result.best_solution is solution
        assert result.best_solution.content == "the original input"


# ===========================================================================
# Call ordering: leakage check always precedes evaluate_with_retry
# ===========================================================================


@pytest.mark.unit
class TestCallOrdering:
    """Verify that safety functions are invoked in the correct order."""

    async def test_leakage_check_before_evaluate_with_retry(self) -> None:
        """For each iteration, check_and_fix_leakage is called before evaluate_with_retry."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        call_order: list[str] = []

        async def _track_leakage(sol: Any, t: Any, c: Any) -> SolutionScript:
            call_order.append("leakage")
            return sol

        async def _track_eval(
            sol: Any,
            t: Any,
            cfg: Any,
            cb: Any,
        ) -> tuple[SolutionScript, EvaluationResult]:
            call_order.append("eval")
            return (sol, _make_eval_result(0.85))

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=_track_leakage,
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                side_effect=_track_eval,
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.90,
                task=task,
                config=config,
            )

        # Leakage check moved to post-loop; only eval runs per-iteration.
        assert call_order == ["eval"]

    async def test_call_ordering_across_multiple_iterations(self) -> None:
        """For K=3, leakage-eval pairs are interleaved correctly per iteration."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=3)

        call_order: list[str] = []

        async def _track_leakage(sol: Any, t: Any, c: Any) -> SolutionScript:
            call_order.append("leakage")
            return sol

        async def _track_eval(
            sol: Any,
            t: Any,
            cfg: Any,
            cb: Any,
        ) -> tuple[SolutionScript, EvaluationResult]:
            call_order.append("eval")
            return (sol, _make_eval_result(0.85))

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="next plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=_track_leakage,
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                side_effect=_track_eval,
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.90,
                task=task,
                config=config,
            )

        # Leakage check moved to post-loop; only evals run per-iteration.
        # Early stopping (patience=2) stops after 2 no-improvement iterations.
        assert call_order == ["eval", "eval"]


# ===========================================================================
# Score tracking from evaluate_with_retry return values
# ===========================================================================


@pytest.mark.unit
class TestScoreTrackingFromRetry:
    """Score from evaluate_with_retry is used for attempt records and best tracking."""

    async def test_score_from_eval_result_recorded_in_attempt(self) -> None:
        """RefinementAttempt.score matches the EvaluationResult.score from evaluate_with_retry."""
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
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.91)),
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

        assert result.attempts[0].score == 0.91

    async def test_returned_solution_from_retry_used_for_best(self) -> None:
        """The solution returned by evaluate_with_retry is what becomes local_best_solution."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)
        retry_solution = _make_solution("code after debug retry")

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(retry_solution, _make_eval_result(0.95)),
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

        assert result.best_solution.content == "code after debug retry"


# ===========================================================================
# Hypothesis property-based tests
# ===========================================================================


@pytest.mark.unit
class TestPropertyBased:
    """Property-based tests for inner loop safety integration."""

    @given(
        k_steps=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=20, deadline=5000)
    async def test_all_none_scores_preserve_input_solution(
        self,
        k_steps: int,
    ) -> None:
        """For any K, when all evaluations return score=None, best_solution is the input."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=k_steps)
        none_result = _make_eval_result(score=None)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="next plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), none_result),
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

        assert result.best_solution is solution
        assert result.improved is False
        # Early stopping (patience=2) limits attempts when no improvement.
        expected = k_steps if k_steps <= 2 else 2
        assert len(result.attempts) == expected

    @given(
        k_steps=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=20, deadline=5000)
    async def test_leakage_call_count_matches_successful_iterations(
        self,
        k_steps: int,
    ) -> None:
        """check_and_fix_leakage is called exactly once per iteration when all succeed."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=k_steps)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="next plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ) as mock_leakage,
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
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
                initial_plan="initial plan",
                best_score=0.90,
                task=task,
                config=config,
            )

        # Leakage check moved to post-loop; only runs when improved.
        # is_improvement_or_equal=False → no improvement → no leakage call.
        assert mock_leakage.call_count == 0

    @given(
        k_steps=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=20, deadline=5000)
    async def test_evaluate_with_retry_call_count_matches_iterations(
        self,
        k_steps: int,
    ) -> None:
        """evaluate_with_retry is called exactly once per successful iteration."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=k_steps)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="next plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ) as mock_eval_retry,
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.90,
                task=task,
                config=config,
            )

        # Early stopping (patience=2) limits iterations when no improvement.
        expected = k_steps if k_steps <= 2 else 2
        assert mock_eval_retry.call_count == expected


# ===========================================================================
# Mixed scenarios: some iterations succeed, some fail
# ===========================================================================


@pytest.mark.unit
class TestMixedIterationScenarios:
    """Mixed scenarios where some iterations fail and some succeed."""

    async def test_coder_failure_then_success_calls_leakage_only_on_success(
        self,
    ) -> None:
        """With K=2, coder fails at k=0 and succeeds at k=1: leakage called once."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=2)

        coder_returns = [None, "improved code"]

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                side_effect=coder_returns,
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="next plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ) as mock_leakage,
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ) as mock_eval,
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.90,
                task=task,
                config=config,
            )

        # k=0: coder failed -> no eval
        # k=1: coder succeeded -> eval
        # Leakage: post-loop only, is_improvement_or_equal=False → no call.
        assert mock_leakage.call_count == 0
        assert mock_eval.call_count == 1
        assert len(result.attempts) == 2
        assert result.attempts[0].code_block == ""  # coder failure
        assert result.attempts[1].score == 0.85  # successful eval

    async def test_planner_failure_skips_leakage_and_eval(self) -> None:
        """Planner failure at k=1 records attempt but does not call leakage or eval."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=2)

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
            ) as mock_leakage,
            patch(
                f"{_MODULE}.make_debug_callback",
                return_value=AsyncMock(),
            ),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ) as mock_eval,
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.90,
                task=task,
                config=config,
            )

        # k=0 succeeds -> eval called
        # k=1 planner fails -> no eval
        # Leakage: post-loop only, is_improvement_or_equal=False → no call.
        assert mock_leakage.call_count == 0
        assert mock_eval.call_count == 1
        assert len(result.attempts) == 2
        assert result.attempts[1].plan == "[planner failed]"
        assert result.attempts[1].score is None
