"""Tests for solution evaluation pipeline (Task 15).

Validates ``evaluate_solution``, ``evaluate_with_retry``, and
``is_better_solution`` which orchestrate end-to-end solution evaluation,
retry-on-failure with debug callbacks, and score comparison against a
previous best.

Tests are written TDD-first and serve as the executable specification for
REQ-EX-015 through REQ-EX-018.

Refs:
    SRS 02d (Evaluation Pipeline), IMPLEMENTATION_PLAN.md Task 15.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import given, settings, strategies as st
from mle_star.execution import (
    evaluate_solution,
    evaluate_with_retry,
    is_better_solution,
)
from mle_star.models import (
    DataModality,
    EvaluationResult,
    MetricDirection,
    PipelineConfig,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
    TaskType,
)
from mle_star.scoring import is_improvement
import pytest

# ---------------------------------------------------------------------------
# Helpers -- factory functions for building valid model instances
# ---------------------------------------------------------------------------


def _make_solution(**overrides: Any) -> SolutionScript:
    """Build a valid SolutionScript with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed SolutionScript instance.
    """
    defaults: dict[str, Any] = {
        "content": "import pandas as pd\nprint('Final Validation Performance: 0.85')\n",
        "phase": SolutionPhase.INIT,
    }
    defaults.update(overrides)
    return SolutionScript(**defaults)


def _make_task(**overrides: Any) -> TaskDescription:
    """Build a valid TaskDescription with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed TaskDescription instance.
    """
    defaults: dict[str, Any] = {
        "competition_id": "spaceship-titanic",
        "task_type": TaskType.CLASSIFICATION,
        "data_modality": DataModality.TABULAR,
        "evaluation_metric": "accuracy",
        "metric_direction": MetricDirection.MAXIMIZE,
        "description": "Predict which passengers were transported.",
        "data_dir": "/tmp/test_data",
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
    defaults: dict[str, Any] = {
        "time_limit_seconds": 300,
        "max_debug_attempts": 3,
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_eval_result(**overrides: Any) -> EvaluationResult:
    """Build a valid EvaluationResult with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed EvaluationResult instance.
    """
    defaults: dict[str, Any] = {
        "score": 0.85,
        "stdout": "Final Validation Performance: 0.85\n",
        "stderr": "",
        "exit_code": 0,
        "duration_seconds": 10.0,
        "is_error": False,
        "error_traceback": None,
    }
    defaults.update(overrides)
    return EvaluationResult(**defaults)


# ---------------------------------------------------------------------------
# Common patch target prefix
# ---------------------------------------------------------------------------
_EXEC = "mle_star.execution"


# ===========================================================================
# REQ-EX-015: evaluate_solution -- Is Async
# ===========================================================================


@pytest.mark.unit
class TestEvaluateSolutionIsAsync:
    """evaluate_solution is an async function (REQ-EX-015)."""

    def test_evaluate_solution_is_coroutine_function(self) -> None:
        """evaluate_solution is defined as an async function."""
        assert asyncio.iscoroutinefunction(evaluate_solution)


# ===========================================================================
# REQ-EX-015: evaluate_solution -- Orchestration Order
# ===========================================================================


@pytest.mark.unit
class TestEvaluateSolutionOrchestration:
    """evaluate_solution calls all 6 steps in correct order (REQ-EX-015)."""

    async def test_calls_all_six_steps(self) -> None:
        """All six internal functions are called during evaluation."""
        solution = _make_solution()
        task = _make_task()
        config = _make_config()
        expected_result = _make_eval_result()

        with (
            patch(
                f"{_EXEC}.setup_working_directory", return_value="/abs/path"
            ) as m_setup,
            patch(f"{_EXEC}.clean_output_directory") as m_clean,
            patch(
                f"{_EXEC}.write_script", return_value="/abs/path/solution.py"
            ) as m_write,
            patch(
                f"{_EXEC}.build_execution_env", return_value={"PATH": "/usr/bin"}
            ) as m_env,
            patch(f"{_EXEC}.execute_script", new_callable=AsyncMock) as m_exec,
            patch(
                f"{_EXEC}.build_evaluation_result", return_value=expected_result
            ) as m_build,
        ):
            m_exec.return_value = MagicMock()  # ExecutionRawResult mock

            await evaluate_solution(solution, task, config)

            m_setup.assert_called_once()
            m_clean.assert_called_once()
            m_write.assert_called_once()
            m_env.assert_called_once()
            m_exec.assert_called_once()
            m_build.assert_called_once()

    async def test_returns_evaluation_result(self) -> None:
        """evaluate_solution returns an EvaluationResult."""
        solution = _make_solution()
        task = _make_task()
        config = _make_config()
        expected_result = _make_eval_result()

        with (
            patch(f"{_EXEC}.setup_working_directory", return_value="/abs/path"),
            patch(f"{_EXEC}.clean_output_directory"),
            patch(f"{_EXEC}.write_script", return_value="/abs/path/solution.py"),
            patch(f"{_EXEC}.build_execution_env", return_value={}),
            patch(f"{_EXEC}.execute_script", new_callable=AsyncMock) as m_exec,
            patch(f"{_EXEC}.build_evaluation_result", return_value=expected_result),
        ):
            m_exec.return_value = MagicMock()

            result = await evaluate_solution(solution, task, config)

            assert isinstance(result, EvaluationResult)
            assert result is expected_result

    async def test_setup_uses_task_data_dir(self) -> None:
        """setup_working_directory is called with task.data_dir as base path."""
        task = _make_task(data_dir="/my/data/dir")
        solution = _make_solution()
        config = _make_config()

        with (
            patch(
                f"{_EXEC}.setup_working_directory", return_value="/my/data/dir"
            ) as m_setup,
            patch(f"{_EXEC}.clean_output_directory"),
            patch(f"{_EXEC}.write_script", return_value="/my/data/dir/solution.py"),
            patch(f"{_EXEC}.build_execution_env", return_value={}),
            patch(f"{_EXEC}.execute_script", new_callable=AsyncMock) as m_exec,
            patch(f"{_EXEC}.build_evaluation_result", return_value=_make_eval_result()),
        ):
            m_exec.return_value = MagicMock()

            await evaluate_solution(solution, task, config)

            m_setup.assert_called_once_with(task.data_dir)

    async def test_steps_are_called_in_order(self) -> None:
        """The six steps are called in the correct sequential order."""
        solution = _make_solution()
        task = _make_task()
        config = _make_config()
        call_order: list[str] = []

        def track(name: str, return_value: Any = None) -> Any:
            """Create a side_effect that records call order."""

            def side_effect(*args: Any, **kwargs: Any) -> Any:
                call_order.append(name)
                return return_value

            return side_effect

        raw_mock = MagicMock()

        async def execute_side_effect(*args: Any, **kwargs: Any) -> Any:
            """Async side_effect that records call order for execute_script."""
            call_order.append("execute")
            return raw_mock

        with (
            patch(
                f"{_EXEC}.setup_working_directory",
                side_effect=track("setup", "/abs/path"),
            ),
            patch(
                f"{_EXEC}.clean_output_directory",
                side_effect=track("clean"),
            ),
            patch(
                f"{_EXEC}.write_script",
                side_effect=track("write", "/abs/path/solution.py"),
            ),
            patch(
                f"{_EXEC}.build_execution_env",
                side_effect=track("build_env", {}),
            ),
            patch(f"{_EXEC}.execute_script", new_callable=AsyncMock) as m_exec,
            patch(
                f"{_EXEC}.build_evaluation_result",
                side_effect=track("build_result", _make_eval_result()),
            ),
        ):
            m_exec.side_effect = execute_side_effect

            await evaluate_solution(solution, task, config)

            assert call_order == [
                "setup",
                "clean",
                "write",
                "build_env",
                "execute",
                "build_result",
            ]


# ===========================================================================
# REQ-EX-015: evaluate_solution -- Timeout Override
# ===========================================================================


@pytest.mark.unit
class TestEvaluateSolutionTimeoutOverride:
    """evaluate_solution respects timeout_override over config.time_limit_seconds (REQ-EX-015)."""

    async def test_timeout_override_takes_precedence(self) -> None:
        """When timeout_override is provided, it is used instead of config.time_limit_seconds."""
        solution = _make_solution()
        task = _make_task()
        config = _make_config(time_limit_seconds=300)

        with (
            patch(f"{_EXEC}.setup_working_directory", return_value="/abs/path"),
            patch(f"{_EXEC}.clean_output_directory"),
            patch(f"{_EXEC}.write_script", return_value="/abs/path/solution.py"),
            patch(f"{_EXEC}.build_execution_env", return_value={}),
            patch(f"{_EXEC}.execute_script", new_callable=AsyncMock) as m_exec,
            patch(f"{_EXEC}.build_evaluation_result", return_value=_make_eval_result()),
        ):
            m_exec.return_value = MagicMock()

            await evaluate_solution(solution, task, config, timeout_override=60)

            # execute_script should have been called with timeout_seconds=60
            _, kwargs = m_exec.call_args
            if "timeout_seconds" in kwargs:
                assert kwargs["timeout_seconds"] == 60
            else:
                args = m_exec.call_args[0]
                # timeout_seconds is the 3rd positional arg
                assert args[2] == 60

    async def test_uses_config_time_limit_when_override_is_none(self) -> None:
        """When timeout_override is None, config.time_limit_seconds is used."""
        solution = _make_solution()
        task = _make_task()
        config = _make_config(time_limit_seconds=500)

        with (
            patch(f"{_EXEC}.setup_working_directory", return_value="/abs/path"),
            patch(f"{_EXEC}.clean_output_directory"),
            patch(f"{_EXEC}.write_script", return_value="/abs/path/solution.py"),
            patch(f"{_EXEC}.build_execution_env", return_value={}),
            patch(f"{_EXEC}.execute_script", new_callable=AsyncMock) as m_exec,
            patch(f"{_EXEC}.build_evaluation_result", return_value=_make_eval_result()),
        ):
            m_exec.return_value = MagicMock()

            await evaluate_solution(solution, task, config, timeout_override=None)

            _, kwargs = m_exec.call_args
            if "timeout_seconds" in kwargs:
                assert kwargs["timeout_seconds"] == 500
            else:
                args = m_exec.call_args[0]
                assert args[2] == 500


# ===========================================================================
# REQ-EX-016: evaluate_solution -- Does NOT Mutate Input
# ===========================================================================


@pytest.mark.unit
class TestEvaluateSolutionNoMutation:
    """evaluate_solution does NOT mutate the input SolutionScript (REQ-EX-016)."""

    async def test_solution_content_unchanged(self) -> None:
        """The content field of the input SolutionScript is not modified."""
        original_content = "import pandas as pd\nprint('hello')\n"
        solution = _make_solution(content=original_content)

        with (
            patch(f"{_EXEC}.setup_working_directory", return_value="/abs/path"),
            patch(f"{_EXEC}.clean_output_directory"),
            patch(f"{_EXEC}.write_script", return_value="/abs/path/solution.py"),
            patch(f"{_EXEC}.build_execution_env", return_value={}),
            patch(f"{_EXEC}.execute_script", new_callable=AsyncMock) as m_exec,
            patch(f"{_EXEC}.build_evaluation_result", return_value=_make_eval_result()),
        ):
            m_exec.return_value = MagicMock()

            await evaluate_solution(solution, _make_task(), _make_config())

            assert solution.content == original_content

    async def test_solution_score_unchanged(self) -> None:
        """The score field of the input SolutionScript is not modified."""
        solution = _make_solution(score=None)

        with (
            patch(f"{_EXEC}.setup_working_directory", return_value="/abs/path"),
            patch(f"{_EXEC}.clean_output_directory"),
            patch(f"{_EXEC}.write_script", return_value="/abs/path/solution.py"),
            patch(f"{_EXEC}.build_execution_env", return_value={}),
            patch(f"{_EXEC}.execute_script", new_callable=AsyncMock) as m_exec,
            patch(
                f"{_EXEC}.build_evaluation_result",
                return_value=_make_eval_result(score=0.99),
            ),
        ):
            m_exec.return_value = MagicMock()

            await evaluate_solution(solution, _make_task(), _make_config())

            assert solution.score is None

    async def test_solution_phase_unchanged(self) -> None:
        """The phase field of the input SolutionScript is not modified."""
        solution = _make_solution(phase=SolutionPhase.INIT)

        with (
            patch(f"{_EXEC}.setup_working_directory", return_value="/abs/path"),
            patch(f"{_EXEC}.clean_output_directory"),
            patch(f"{_EXEC}.write_script", return_value="/abs/path/solution.py"),
            patch(f"{_EXEC}.build_execution_env", return_value={}),
            patch(f"{_EXEC}.execute_script", new_callable=AsyncMock) as m_exec,
            patch(f"{_EXEC}.build_evaluation_result", return_value=_make_eval_result()),
        ):
            m_exec.return_value = MagicMock()

            await evaluate_solution(solution, _make_task(), _make_config())

            assert solution.phase == SolutionPhase.INIT

    async def test_deep_copy_equality_after_evaluation(self) -> None:
        """A deep copy of the solution taken before eval matches after eval."""
        solution = _make_solution(
            content="x = 1\n", score=None, phase=SolutionPhase.REFINED
        )
        snapshot = copy.deepcopy(solution)

        with (
            patch(f"{_EXEC}.setup_working_directory", return_value="/abs/path"),
            patch(f"{_EXEC}.clean_output_directory"),
            patch(f"{_EXEC}.write_script", return_value="/abs/path/solution.py"),
            patch(f"{_EXEC}.build_execution_env", return_value={}),
            patch(f"{_EXEC}.execute_script", new_callable=AsyncMock) as m_exec,
            patch(f"{_EXEC}.build_evaluation_result", return_value=_make_eval_result()),
        ):
            m_exec.return_value = MagicMock()

            await evaluate_solution(solution, _make_task(), _make_config())

            assert solution.content == snapshot.content
            assert solution.phase == snapshot.phase
            assert solution.score == snapshot.score
            assert solution.is_executable == snapshot.is_executable


# ===========================================================================
# REQ-EX-015: evaluate_solution -- Error Propagation
# ===========================================================================


@pytest.mark.unit
class TestEvaluateSolutionErrorPropagation:
    """evaluate_solution propagates ValueError from write_script (REQ-EX-015)."""

    async def test_propagates_value_error_from_write_script(self) -> None:
        """ValueError raised by write_script propagates to the caller."""
        solution = _make_solution(content="   ")  # empty after strip
        task = _make_task()
        config = _make_config()

        with (
            patch(f"{_EXEC}.setup_working_directory", return_value="/abs/path"),
            patch(f"{_EXEC}.clean_output_directory"),
            patch(
                f"{_EXEC}.write_script",
                side_effect=ValueError(
                    "Script content is empty after stripping whitespace"
                ),
            ),
            pytest.raises(ValueError, match="Script content is empty"),
        ):
            await evaluate_solution(solution, task, config)

    async def test_propagates_value_error_for_forbidden_call(self) -> None:
        """ValueError for forbidden exit calls propagates to the caller."""
        solution = _make_solution(content="exit(0)")
        task = _make_task()
        config = _make_config()

        with (
            patch(f"{_EXEC}.setup_working_directory", return_value="/abs/path"),
            patch(f"{_EXEC}.clean_output_directory"),
            patch(
                f"{_EXEC}.write_script",
                side_effect=ValueError("Script contains forbidden call"),
            ),
            pytest.raises(ValueError, match="forbidden"),
        ):
            await evaluate_solution(solution, task, config)


# ===========================================================================
# REQ-EX-017: evaluate_with_retry -- Is Async
# ===========================================================================


@pytest.mark.unit
class TestEvaluateWithRetryIsAsync:
    """evaluate_with_retry is an async function (REQ-EX-017)."""

    def test_evaluate_with_retry_is_coroutine_function(self) -> None:
        """evaluate_with_retry is defined as an async function."""
        assert asyncio.iscoroutinefunction(evaluate_with_retry)


# ===========================================================================
# REQ-EX-017: evaluate_with_retry -- Immediate Success
# ===========================================================================


@pytest.mark.unit
class TestEvaluateWithRetrySuccess:
    """evaluate_with_retry returns immediately on success (REQ-EX-017)."""

    async def test_returns_on_first_success(self) -> None:
        """When first evaluation succeeds, returns immediately without retries."""
        solution = _make_solution()
        task = _make_task()
        config = _make_config(max_debug_attempts=3)
        success_result = _make_eval_result(is_error=False, score=0.85)
        debug_callback = AsyncMock()

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=success_result,
        ) as m_eval:
            _sol, result = await evaluate_with_retry(
                solution, task, config, debug_callback
            )

            m_eval.assert_called_once()
            debug_callback.assert_not_called()
            assert result.is_error is False
            assert result.score == pytest.approx(0.85)

    async def test_returns_original_solution_on_success(self) -> None:
        """On success, the returned solution is the one that was evaluated."""
        solution = _make_solution(content="original_code\n")
        task = _make_task()
        config = _make_config()
        success_result = _make_eval_result(is_error=False)
        debug_callback = AsyncMock()

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=success_result,
        ):
            sol, _result = await evaluate_with_retry(
                solution, task, config, debug_callback
            )

            assert sol.content == "original_code\n"

    async def test_returns_tuple_type(self) -> None:
        """Return value is a tuple of (SolutionScript, EvaluationResult)."""
        solution = _make_solution()
        task = _make_task()
        config = _make_config()
        success_result = _make_eval_result(is_error=False)
        debug_callback = AsyncMock()

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=success_result,
        ):
            result_tuple = await evaluate_with_retry(
                solution, task, config, debug_callback
            )

            assert isinstance(result_tuple, tuple)
            assert len(result_tuple) == 2
            assert isinstance(result_tuple[0], SolutionScript)
            assert isinstance(result_tuple[1], EvaluationResult)


# ===========================================================================
# REQ-EX-017: evaluate_with_retry -- Retry on Failure
# ===========================================================================


@pytest.mark.unit
class TestEvaluateWithRetryRetries:
    """evaluate_with_retry calls debug_callback and retries on failure (REQ-EX-017)."""

    async def test_calls_debug_callback_on_failure(self) -> None:
        """debug_callback is called when evaluation fails."""
        solution = _make_solution()
        task = _make_task()
        config = _make_config(max_debug_attempts=1)
        error_tb = "Traceback (most recent call last):\n  ...\nValueError: test"
        fail_result = _make_eval_result(
            is_error=True,
            error_traceback=error_tb,
            score=None,
        )
        success_result = _make_eval_result(is_error=False, score=0.9)
        fixed_solution = _make_solution(content="fixed_code\n")

        debug_callback = AsyncMock(return_value=fixed_solution)

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            side_effect=[fail_result, success_result],
        ):
            _sol, result = await evaluate_with_retry(
                solution, task, config, debug_callback
            )

            debug_callback.assert_called_once()
            assert result.is_error is False

    async def test_debug_callback_receives_solution_and_traceback(self) -> None:
        """debug_callback is called with the current solution and error_traceback."""
        solution = _make_solution(content="buggy_code\n")
        task = _make_task()
        config = _make_config(max_debug_attempts=1)
        error_tb = "Traceback (most recent call last):\n  ValueError: oops"
        fail_result = _make_eval_result(is_error=True, error_traceback=error_tb)
        success_result = _make_eval_result(is_error=False)
        fixed_solution = _make_solution(content="fixed_code\n")

        debug_callback = AsyncMock(return_value=fixed_solution)

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            side_effect=[fail_result, success_result],
        ):
            await evaluate_with_retry(solution, task, config, debug_callback)

            call_args = debug_callback.call_args
            passed_solution = call_args[0][0]
            passed_traceback = call_args[0][1]
            assert passed_solution.content == "buggy_code\n"
            assert "ValueError" in passed_traceback

    async def test_retry_uses_fixed_solution_from_callback(self) -> None:
        """The solution returned by debug_callback is used for the retry evaluation."""
        solution = _make_solution(content="original\n")
        task = _make_task()
        config = _make_config(max_debug_attempts=1)
        fail_result = _make_eval_result(is_error=True, error_traceback="error")
        success_result = _make_eval_result(is_error=False, score=0.9)
        fixed_solution = _make_solution(content="fixed_content\n")

        debug_callback = AsyncMock(return_value=fixed_solution)

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            side_effect=[fail_result, success_result],
        ) as m_eval:
            _sol, _result = await evaluate_with_retry(
                solution, task, config, debug_callback
            )

            # Second call should use the fixed solution
            second_call_args = m_eval.call_args_list[1]
            assert second_call_args[0][0].content == "fixed_content\n"

    async def test_multiple_retries_before_success(self) -> None:
        """Multiple failures followed by success: all retries executed."""
        solution = _make_solution()
        task = _make_task()
        config = _make_config(max_debug_attempts=3)
        fail_result = _make_eval_result(is_error=True, error_traceback="error")
        success_result = _make_eval_result(is_error=False, score=0.85)

        fix1 = _make_solution(content="fix1\n")
        fix2 = _make_solution(content="fix2\n")
        debug_callback = AsyncMock(side_effect=[fix1, fix2])

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            side_effect=[fail_result, fail_result, success_result],
        ) as m_eval:
            _sol, result = await evaluate_with_retry(
                solution, task, config, debug_callback
            )

            assert m_eval.call_count == 3
            assert debug_callback.call_count == 2
            assert result.is_error is False


# ===========================================================================
# REQ-EX-017: evaluate_with_retry -- Retries Exhausted
# ===========================================================================


@pytest.mark.unit
class TestEvaluateWithRetryExhausted:
    """evaluate_with_retry returns last result when all retries exhausted (REQ-EX-017)."""

    async def test_returns_last_result_when_all_retries_fail(self) -> None:
        """When all retries fail, returns the last (solution, result) pair."""
        solution = _make_solution()
        task = _make_task()
        config = _make_config(max_debug_attempts=2)
        fail1 = _make_eval_result(is_error=True, error_traceback="error1")
        fail2 = _make_eval_result(is_error=True, error_traceback="error2")
        fail3 = _make_eval_result(is_error=True, error_traceback="error3")

        fix1 = _make_solution(content="fix1\n")
        fix2 = _make_solution(content="fix2\n")
        debug_callback = AsyncMock(side_effect=[fix1, fix2])

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            side_effect=[fail1, fail2, fail3],
        ):
            sol, result = await evaluate_with_retry(
                solution, task, config, debug_callback
            )

            assert result.is_error is True
            assert sol.content == "fix2\n"

    async def test_evaluate_called_max_retries_plus_one_times(self) -> None:
        """evaluate_solution is called 1 (initial) + max_retries times total."""
        solution = _make_solution()
        task = _make_task()
        config = _make_config(max_debug_attempts=2)
        fail_result = _make_eval_result(is_error=True, error_traceback="error")

        debug_callback = AsyncMock(return_value=_make_solution(content="fix\n"))

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=fail_result,
        ) as m_eval:
            await evaluate_with_retry(solution, task, config, debug_callback)

            # 1 initial + 2 retries = 3 total
            assert m_eval.call_count == 3

    async def test_debug_callback_called_max_retries_times(self) -> None:
        """debug_callback is called exactly max_retries times when all fail."""
        solution = _make_solution()
        task = _make_task()
        config = _make_config(max_debug_attempts=3)
        fail_result = _make_eval_result(is_error=True, error_traceback="error")

        debug_callback = AsyncMock(return_value=_make_solution(content="fix\n"))

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=fail_result,
        ):
            await evaluate_with_retry(solution, task, config, debug_callback)

            assert debug_callback.call_count == 3


# ===========================================================================
# REQ-EX-017: evaluate_with_retry -- max_retries Parameter
# ===========================================================================


@pytest.mark.unit
class TestEvaluateWithRetryMaxRetries:
    """evaluate_with_retry handles max_retries parameter (REQ-EX-017)."""

    async def test_defaults_to_config_max_debug_attempts(self) -> None:
        """When max_retries is None, uses config.max_debug_attempts."""
        solution = _make_solution()
        task = _make_task()
        config = _make_config(max_debug_attempts=2)
        fail_result = _make_eval_result(is_error=True, error_traceback="error")

        debug_callback = AsyncMock(return_value=_make_solution(content="fix\n"))

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=fail_result,
        ) as m_eval:
            await evaluate_with_retry(
                solution, task, config, debug_callback, max_retries=None
            )

            # 1 initial + 2 retries (from config.max_debug_attempts)
            assert m_eval.call_count == 3

    async def test_explicit_max_retries_overrides_config(self) -> None:
        """When max_retries is explicitly provided, it overrides config.max_debug_attempts."""
        solution = _make_solution()
        task = _make_task()
        config = _make_config(max_debug_attempts=5)
        fail_result = _make_eval_result(is_error=True, error_traceback="error")

        debug_callback = AsyncMock(return_value=_make_solution(content="fix\n"))

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=fail_result,
        ) as m_eval:
            await evaluate_with_retry(
                solution, task, config, debug_callback, max_retries=1
            )

            # 1 initial + 1 retry (explicit max_retries=1)
            assert m_eval.call_count == 2

    async def test_max_retries_zero_means_no_retries(self) -> None:
        """When max_retries=0, only the initial evaluation is performed."""
        solution = _make_solution()
        task = _make_task()
        config = _make_config(max_debug_attempts=5)
        fail_result = _make_eval_result(is_error=True, error_traceback="error")

        debug_callback = AsyncMock()

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=fail_result,
        ) as m_eval:
            _sol, result = await evaluate_with_retry(
                solution, task, config, debug_callback, max_retries=0
            )

            assert m_eval.call_count == 1
            debug_callback.assert_not_called()
            assert result.is_error is True


# ===========================================================================
# REQ-EX-017: evaluate_with_retry -- Error Traceback Handling
# ===========================================================================


@pytest.mark.unit
class TestEvaluateWithRetryTraceback:
    """evaluate_with_retry passes error_traceback to debug_callback (REQ-EX-017)."""

    async def test_passes_traceback_when_present(self) -> None:
        """debug_callback receives the error_traceback from the failed evaluation."""
        solution = _make_solution()
        task = _make_task()
        config = _make_config(max_debug_attempts=1)
        traceback_str = (
            "Traceback (most recent call last):\n  File ...\nValueError: bad"
        )
        fail_result = _make_eval_result(is_error=True, error_traceback=traceback_str)
        success_result = _make_eval_result(is_error=False)

        debug_callback = AsyncMock(return_value=_make_solution())

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            side_effect=[fail_result, success_result],
        ):
            await evaluate_with_retry(solution, task, config, debug_callback)

            passed_tb = debug_callback.call_args[0][1]
            assert passed_tb == traceback_str

    async def test_passes_empty_or_none_traceback_when_absent(self) -> None:
        """debug_callback receives error_traceback value even when None (e.g., timeout)."""
        solution = _make_solution()
        task = _make_task()
        config = _make_config(max_debug_attempts=1)
        fail_result = _make_eval_result(
            is_error=True, error_traceback=None, exit_code=-1
        )
        success_result = _make_eval_result(is_error=False)

        debug_callback = AsyncMock(return_value=_make_solution())

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            side_effect=[fail_result, success_result],
        ):
            await evaluate_with_retry(solution, task, config, debug_callback)

            passed_tb = debug_callback.call_args[0][1]
            # Should be None or an empty string -- the important thing is
            # the callback was still invoked
            assert passed_tb is None or isinstance(passed_tb, str)


# ===========================================================================
# REQ-EX-017: evaluate_with_retry -- Chained Debug Fixes
# ===========================================================================


@pytest.mark.unit
class TestEvaluateWithRetryChainedFixes:
    """evaluate_with_retry chains debug fixes through multiple retries (REQ-EX-017)."""

    async def test_each_retry_uses_latest_fixed_solution(self) -> None:
        """Each retry uses the solution from the most recent debug_callback call."""
        solution = _make_solution(content="v0\n")
        task = _make_task()
        config = _make_config(max_debug_attempts=3)
        fail_result = _make_eval_result(is_error=True, error_traceback="error")
        success_result = _make_eval_result(is_error=False, score=0.9)

        fix_v1 = _make_solution(content="v1\n")
        fix_v2 = _make_solution(content="v2\n")
        fix_v3 = _make_solution(content="v3\n")
        debug_callback = AsyncMock(side_effect=[fix_v1, fix_v2, fix_v3])

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            side_effect=[fail_result, fail_result, fail_result, success_result],
        ) as m_eval:
            await evaluate_with_retry(solution, task, config, debug_callback)

            eval_calls = m_eval.call_args_list
            assert eval_calls[0][0][0].content == "v0\n"
            assert eval_calls[1][0][0].content == "v1\n"
            assert eval_calls[2][0][0].content == "v2\n"
            assert eval_calls[3][0][0].content == "v3\n"


# ===========================================================================
# REQ-EX-018: is_better_solution -- Score is None
# ===========================================================================


@pytest.mark.unit
class TestIsBetterSolutionScoreNone:
    """is_better_solution returns False when score is None (REQ-EX-018)."""

    def test_returns_false_when_score_is_none(self) -> None:
        """Returns False when new_result.score is None."""
        result = _make_eval_result(score=None, is_error=False)
        assert is_better_solution(result, 0.5, MetricDirection.MAXIMIZE) is False

    def test_returns_false_when_score_is_none_minimize(self) -> None:
        """Returns False when new_result.score is None for minimize direction."""
        result = _make_eval_result(score=None, is_error=False)
        assert is_better_solution(result, 0.5, MetricDirection.MINIMIZE) is False


# ===========================================================================
# REQ-EX-018: is_better_solution -- Error Results
# ===========================================================================


@pytest.mark.unit
class TestIsBetterSolutionError:
    """is_better_solution returns False when is_error is True (REQ-EX-018)."""

    def test_returns_false_when_is_error_true(self) -> None:
        """Returns False when new_result.is_error is True."""
        result = _make_eval_result(is_error=True, score=0.99)
        assert is_better_solution(result, 0.5, MetricDirection.MAXIMIZE) is False

    def test_returns_false_when_is_error_true_minimize(self) -> None:
        """Returns False when is_error is True for minimize direction."""
        result = _make_eval_result(is_error=True, score=0.1)
        assert is_better_solution(result, 0.5, MetricDirection.MINIMIZE) is False

    def test_returns_false_when_is_error_and_score_none(self) -> None:
        """Returns False when both is_error is True and score is None."""
        result = _make_eval_result(is_error=True, score=None)
        assert is_better_solution(result, 0.5, MetricDirection.MAXIMIZE) is False

    def test_returns_false_when_is_error_true_even_with_high_score(self) -> None:
        """Returns False even when score appears to be an improvement but is_error is True."""
        result = _make_eval_result(is_error=True, score=1.0)
        assert is_better_solution(result, 0.0, MetricDirection.MAXIMIZE) is False


# ===========================================================================
# REQ-EX-018: is_better_solution -- Maximize Direction
# ===========================================================================


@pytest.mark.unit
class TestIsBetterSolutionMaximize:
    """is_better_solution delegates to is_improvement for maximize (REQ-EX-018)."""

    def test_higher_score_is_better(self) -> None:
        """For maximize: a higher new score is better."""
        result = _make_eval_result(score=0.9, is_error=False)
        assert is_better_solution(result, 0.8, MetricDirection.MAXIMIZE) is True

    def test_lower_score_is_not_better(self) -> None:
        """For maximize: a lower new score is not better."""
        result = _make_eval_result(score=0.7, is_error=False)
        assert is_better_solution(result, 0.8, MetricDirection.MAXIMIZE) is False

    def test_equal_score_is_not_better(self) -> None:
        """For maximize: equal score is not better (strict comparison)."""
        result = _make_eval_result(score=0.8, is_error=False)
        assert is_better_solution(result, 0.8, MetricDirection.MAXIMIZE) is False


# ===========================================================================
# REQ-EX-018: is_better_solution -- Minimize Direction
# ===========================================================================


@pytest.mark.unit
class TestIsBetterSolutionMinimize:
    """is_better_solution delegates to is_improvement for minimize (REQ-EX-018)."""

    def test_lower_score_is_better(self) -> None:
        """For minimize: a lower new score is better."""
        result = _make_eval_result(score=0.3, is_error=False)
        assert is_better_solution(result, 0.5, MetricDirection.MINIMIZE) is True

    def test_higher_score_is_not_better(self) -> None:
        """For minimize: a higher new score is not better."""
        result = _make_eval_result(score=0.7, is_error=False)
        assert is_better_solution(result, 0.5, MetricDirection.MINIMIZE) is False

    def test_equal_score_is_not_better(self) -> None:
        """For minimize: equal score is not better (strict comparison)."""
        result = _make_eval_result(score=0.5, is_error=False)
        assert is_better_solution(result, 0.5, MetricDirection.MINIMIZE) is False


# ===========================================================================
# REQ-EX-018: is_better_solution -- Return Type
# ===========================================================================


@pytest.mark.unit
class TestIsBetterSolutionReturnType:
    """is_better_solution always returns a bool (REQ-EX-018)."""

    def test_returns_bool_for_success(self) -> None:
        """Return type is bool when evaluation succeeded."""
        result = _make_eval_result(score=0.9, is_error=False)
        ret = is_better_solution(result, 0.8, MetricDirection.MAXIMIZE)
        assert isinstance(ret, bool)

    def test_returns_bool_for_error(self) -> None:
        """Return type is bool when evaluation had an error."""
        result = _make_eval_result(score=None, is_error=True)
        ret = is_better_solution(result, 0.8, MetricDirection.MAXIMIZE)
        assert isinstance(ret, bool)

    def test_returns_bool_for_none_score(self) -> None:
        """Return type is bool when score is None."""
        result = _make_eval_result(score=None, is_error=False)
        ret = is_better_solution(result, 0.8, MetricDirection.MAXIMIZE)
        assert isinstance(ret, bool)


# ===========================================================================
# REQ-EX-018: is_better_solution -- Consistency with is_improvement
# ===========================================================================


@pytest.mark.unit
class TestIsBetterSolutionConsistency:
    """is_better_solution is consistent with is_improvement from scoring module (REQ-EX-018)."""

    @pytest.mark.parametrize(
        "new_score,old_score,direction",
        [
            (0.9, 0.8, MetricDirection.MAXIMIZE),
            (0.7, 0.8, MetricDirection.MAXIMIZE),
            (0.8, 0.8, MetricDirection.MAXIMIZE),
            (0.3, 0.5, MetricDirection.MINIMIZE),
            (0.7, 0.5, MetricDirection.MINIMIZE),
            (0.5, 0.5, MetricDirection.MINIMIZE),
        ],
        ids=[
            "max_improvement",
            "max_no_improvement",
            "max_equal",
            "min_improvement",
            "min_no_improvement",
            "min_equal",
        ],
    )
    def test_delegates_to_is_improvement_when_valid(
        self,
        new_score: float,
        old_score: float,
        direction: MetricDirection,
    ) -> None:
        """When score is not None and is_error is False, delegates to is_improvement."""
        result = _make_eval_result(score=new_score, is_error=False)
        expected = is_improvement(new_score, old_score, direction)
        assert is_better_solution(result, old_score, direction) is expected


# ===========================================================================
# REQ-EX-018: is_better_solution -- Property-based Tests
# ===========================================================================


@pytest.mark.unit
class TestIsBetterSolutionPropertyBased:
    """Property-based tests for is_better_solution using Hypothesis (REQ-EX-018)."""

    @given(
        old_score=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        direction=st.sampled_from([MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]),
    )
    @settings(max_examples=50)
    def test_error_always_returns_false(
        self, old_score: float, direction: MetricDirection
    ) -> None:
        """Property: is_error=True always returns False regardless of score."""
        result = _make_eval_result(is_error=True, score=old_score + 1.0)
        assert is_better_solution(result, old_score, direction) is False

    @given(
        old_score=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        direction=st.sampled_from([MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]),
    )
    @settings(max_examples=50)
    def test_none_score_always_returns_false(
        self, old_score: float, direction: MetricDirection
    ) -> None:
        """Property: score=None always returns False."""
        result = _make_eval_result(is_error=False, score=None)
        assert is_better_solution(result, old_score, direction) is False

    @given(
        new_score=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        old_score=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        direction=st.sampled_from([MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]),
    )
    @settings(max_examples=50)
    def test_consistent_with_is_improvement(
        self,
        new_score: float,
        old_score: float,
        direction: MetricDirection,
    ) -> None:
        """Property: for valid results, is_better_solution matches is_improvement."""
        result = _make_eval_result(score=new_score, is_error=False)
        expected = is_improvement(new_score, old_score, direction)
        assert is_better_solution(result, old_score, direction) is expected

    @given(
        new_score=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        old_score=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        direction=st.sampled_from([MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]),
    )
    @settings(max_examples=50)
    def test_always_returns_bool(
        self,
        new_score: float,
        old_score: float,
        direction: MetricDirection,
    ) -> None:
        """Property: is_better_solution always returns a bool."""
        result = _make_eval_result(score=new_score, is_error=False)
        ret = is_better_solution(result, old_score, direction)
        assert isinstance(ret, bool)

    @given(
        score=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        direction=st.sampled_from([MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]),
    )
    @settings(max_examples=30)
    def test_equal_scores_never_better(
        self, score: float, direction: MetricDirection
    ) -> None:
        """Property: equal scores never yield True (strict comparison)."""
        result = _make_eval_result(score=score, is_error=False)
        assert is_better_solution(result, score, direction) is False


# ===========================================================================
# REQ-EX-018: is_better_solution -- Edge Cases
# ===========================================================================


@pytest.mark.unit
class TestIsBetterSolutionEdgeCases:
    """is_better_solution edge cases (REQ-EX-018)."""

    def test_negative_scores_maximize(self) -> None:
        """For maximize with negative scores: -0.3 > -0.5 is better."""
        result = _make_eval_result(score=-0.3, is_error=False)
        assert is_better_solution(result, -0.5, MetricDirection.MAXIMIZE) is True

    def test_negative_scores_minimize(self) -> None:
        """For minimize with negative scores: -0.7 < -0.3 is better."""
        result = _make_eval_result(score=-0.7, is_error=False)
        assert is_better_solution(result, -0.3, MetricDirection.MINIMIZE) is True

    def test_zero_old_score_maximize(self) -> None:
        """For maximize: positive new score > 0 old score is better."""
        result = _make_eval_result(score=0.1, is_error=False)
        assert is_better_solution(result, 0.0, MetricDirection.MAXIMIZE) is True

    def test_zero_old_score_minimize(self) -> None:
        """For minimize: negative new score < 0 old score is better."""
        result = _make_eval_result(score=-0.1, is_error=False)
        assert is_better_solution(result, 0.0, MetricDirection.MINIMIZE) is True

    def test_very_small_improvement_maximize(self) -> None:
        """For maximize: even a tiny improvement counts."""
        result = _make_eval_result(score=0.800001, is_error=False)
        assert is_better_solution(result, 0.8, MetricDirection.MAXIMIZE) is True

    def test_very_small_improvement_minimize(self) -> None:
        """For minimize: even a tiny decrease counts."""
        result = _make_eval_result(score=0.799999, is_error=False)
        assert is_better_solution(result, 0.8, MetricDirection.MINIMIZE) is True
