"""Tests for execution harness constraints and interface compliance (Task 18).

Validates ``detect_error_masking``, structured logging, large output
truncation, and performance constraints.

Tests are written TDD-first and serve as the executable specification for
REQ-EX-028 through REQ-EX-047.

Refs:
    SRS 02c (Advanced Operations), SRS 02d (Constraints),
    IMPLEMENTATION_PLAN.md Task 18.
"""

from __future__ import annotations

import inspect
import logging
import textwrap
import time
from typing import TYPE_CHECKING
from unittest.mock import patch

from hypothesis import HealthCheck, given, settings, strategies as st
from mle_star.execution import ExecutionRawResult
from mle_star.models import EvaluationResult, SolutionScript
import pytest

from tests.conftest import make_config, make_solution, make_task

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_script(tmp_path: Path, content: str, name: str = "script.py") -> str:
    """Write a Python script to tmp_path and return its absolute path."""
    script_path = tmp_path / name
    script_path.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(script_path.resolve())


# ===========================================================================
# REQ-EX-045: detect_error_masking
# ===========================================================================


@pytest.mark.unit
class TestDetectErrorMasking:
    """detect_error_masking identifies broad try/except patterns (REQ-EX-045)."""

    def test_bare_except_detected(self) -> None:
        """Bare ``except:`` clause is detected."""
        from mle_star.execution import detect_error_masking

        code = textwrap.dedent("""\
            try:
                x = 1 / 0
            except:
                pass
        """)
        warnings = detect_error_masking(code)
        assert len(warnings) >= 1
        assert any("bare" in w.lower() or "except:" in w.lower() for w in warnings)

    def test_except_exception_with_pass_detected(self) -> None:
        """``except Exception:`` with only ``pass`` body is detected."""
        from mle_star.execution import detect_error_masking

        code = textwrap.dedent("""\
            try:
                x = 1 / 0
            except Exception:
                pass
        """)
        warnings = detect_error_masking(code)
        assert len(warnings) >= 1
        assert any("Exception" in w for w in warnings)

    def test_except_base_exception_with_pass_detected(self) -> None:
        """``except BaseException:`` with only ``pass`` body is detected."""
        from mle_star.execution import detect_error_masking

        code = textwrap.dedent("""\
            try:
                x = 1 / 0
            except BaseException:
                pass
        """)
        warnings = detect_error_masking(code)
        assert len(warnings) >= 1
        assert any("BaseException" in w for w in warnings)

    def test_clean_code_returns_empty(self) -> None:
        """Clean code without error masking returns empty list."""
        from mle_star.execution import detect_error_masking

        code = textwrap.dedent("""\
            try:
                x = 1 / 0
            except ZeroDivisionError:
                print("oops")
        """)
        warnings = detect_error_masking(code)
        assert warnings == []

    def test_except_exception_with_real_handler_not_detected(self) -> None:
        """``except Exception:`` with a real handler body is not detected."""
        from mle_star.execution import detect_error_masking

        code = textwrap.dedent("""\
            try:
                x = 1 / 0
            except Exception:
                logger.warning("error occurred")
                raise
        """)
        warnings = detect_error_masking(code)
        assert warnings == []

    def test_multiple_patterns_found(self) -> None:
        """Multiple error masking patterns are each reported."""
        from mle_star.execution import detect_error_masking

        code = textwrap.dedent("""\
            try:
                a = 1
            except:
                pass

            try:
                b = 2
            except Exception:
                pass
        """)
        warnings = detect_error_masking(code)
        assert len(warnings) >= 2

    def test_returns_list_of_strings(self) -> None:
        """Return type is list[str] even for clean code."""
        from mle_star.execution import detect_error_masking

        result = detect_error_masking("x = 1")
        assert isinstance(result, list)

    def test_advisory_only_does_not_raise(self) -> None:
        """Function is advisory only; never raises exceptions."""
        from mle_star.execution import detect_error_masking

        # Should not raise even with problematic code
        result = detect_error_masking("except:\n    pass")
        assert isinstance(result, list)

    @given(st.text(min_size=0, max_size=500))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_never_raises_on_arbitrary_input(self, content: str) -> None:
        """Function never raises on arbitrary string input."""
        from mle_star.execution import detect_error_masking

        result = detect_error_masking(content)
        assert isinstance(result, list)
        assert all(isinstance(w, str) for w in result)


# ===========================================================================
# REQ-EX-034: evaluate_solution uses subprocess execution
# ===========================================================================


@pytest.mark.unit
class TestEvaluateSolutionSubprocess:
    """evaluate_solution uses subprocess execution (REQ-EX-034)."""

    @pytest.mark.asyncio
    async def test_subprocess_execution(self, tmp_path: Path) -> None:
        """evaluate_solution uses subprocess execution and returns correct score."""
        from mle_star.execution import evaluate_solution

        with patch("mle_star.execution.execute_script") as mock_exec:
            mock_exec.return_value = ExecutionRawResult(
                stdout="Final Validation Performance: 0.85\n",
                stderr="",
                exit_code=0,
                duration_seconds=1.0,
                timed_out=False,
            )
            task = make_task(data_dir=str(tmp_path))
            solution = make_solution()
            config = make_config()

            result = await evaluate_solution(solution, task, config)
            assert result.score == 0.85


# ===========================================================================
# REQ-EX-038: Large output handling with truncation
# ===========================================================================


@pytest.mark.unit
class TestLargeOutputHandling:
    """Harness handles large stdout/stderr with truncation (REQ-EX-038)."""

    def test_truncate_output_function_exists(self) -> None:
        """A truncation helper exists in the execution module."""
        from mle_star.execution import _truncate_output

        assert callable(_truncate_output)

    def test_small_output_unchanged(self) -> None:
        """Output below limit is returned unchanged."""
        from mle_star.execution import _truncate_output

        output = "hello world"
        assert _truncate_output(output) == output

    def test_large_output_truncated(self) -> None:
        """Output exceeding 100MB is truncated."""
        from mle_star.execution import _MAX_OUTPUT_BYTES, _truncate_output

        large_output = "x" * (_MAX_OUTPUT_BYTES + 1000)
        result = _truncate_output(large_output)
        assert len(result) <= _MAX_OUTPUT_BYTES + 200  # margin for warning

    def test_truncated_output_has_warning(self) -> None:
        """Truncated output includes a warning message."""
        from mle_star.execution import _MAX_OUTPUT_BYTES, _truncate_output

        large_output = "x" * (_MAX_OUTPUT_BYTES + 1000)
        result = _truncate_output(large_output)
        assert "truncated" in result.lower()


# ===========================================================================
# REQ-EX-039: Structured logging
# ===========================================================================


@pytest.mark.unit
class TestExecutionLogging:
    """Execution harness logs events at correct levels (REQ-EX-039)."""

    def test_write_script_logs_debug(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """write_script logs script path and content length at DEBUG."""
        from mle_star.execution import write_script

        solution = make_solution(content="print('hello')\n")
        with caplog.at_level(logging.DEBUG, logger="mle_star.execution"):
            write_script(solution, str(tmp_path))

        debug_msgs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(debug_msgs) >= 1
        msg_text = " ".join(r.message for r in debug_msgs)
        assert "script" in msg_text.lower() or "write" in msg_text.lower()

    @pytest.mark.asyncio
    async def test_execute_script_logs_info_start(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """execute_script logs execution start at INFO."""
        from mle_star.execution import execute_script

        script = _write_script(tmp_path, "print('hello')")
        with caplog.at_level(logging.INFO, logger="mle_star.execution"):
            await execute_script(script, str(tmp_path), timeout_seconds=10)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_msgs) >= 1

    @pytest.mark.asyncio
    async def test_execute_script_logs_info_complete(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """execute_script logs completion at INFO with exit code and duration."""
        from mle_star.execution import execute_script

        script = _write_script(tmp_path, "print('done')")
        with caplog.at_level(logging.INFO, logger="mle_star.execution"):
            await execute_script(script, str(tmp_path), timeout_seconds=10)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        msg_text = " ".join(r.message for r in info_msgs)
        assert (
            "exit" in msg_text.lower()
            or "complete" in msg_text.lower()
            or "duration" in msg_text.lower()
        )

    @pytest.mark.asyncio
    async def test_timeout_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Timeout triggers a WARNING log."""
        from mle_star.execution import execute_script

        script = _write_script(tmp_path, "import time; time.sleep(60)")
        with caplog.at_level(logging.WARNING, logger="mle_star.execution"):
            await execute_script(script, str(tmp_path), timeout_seconds=1)

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1

    @pytest.mark.asyncio
    async def test_error_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Error execution logs a WARNING."""
        from mle_star.execution import execute_script

        script = _write_script(tmp_path, "raise ValueError('boom')")
        with caplog.at_level(logging.WARNING, logger="mle_star.execution"):
            await execute_script(script, str(tmp_path), timeout_seconds=10)

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1

    @pytest.mark.asyncio
    async def test_retry_logs_info(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Retry attempt is logged at INFO."""
        from mle_star.execution import evaluate_with_retry

        async def mock_debug_callback(
            sol: SolutionScript, tb: str | None
        ) -> SolutionScript:
            return sol

        task = make_task(data_dir=str(tmp_path))
        solution = make_solution(content="print('Final Validation Performance: 0.5')\n")
        config = make_config(max_debug_attempts=1)

        with (
            patch("mle_star.execution.evaluate_solution") as mock_eval,
            caplog.at_level(logging.INFO, logger="mle_star.execution"),
        ):
            mock_eval.side_effect = [
                EvaluationResult(
                    score=None,
                    stdout="",
                    stderr="Traceback (most recent call last):\nValueError",
                    exit_code=1,
                    duration_seconds=1.0,
                    is_error=True,
                    error_traceback="Traceback (most recent call last):\nValueError",
                ),
                EvaluationResult(
                    score=0.5,
                    stdout="Final Validation Performance: 0.5\n",
                    stderr="",
                    exit_code=0,
                    duration_seconds=1.0,
                    is_error=False,
                    error_traceback=None,
                ),
            ]

            await evaluate_with_retry(
                solution, task, config, mock_debug_callback, max_retries=1
            )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        msg_text = " ".join(r.message for r in info_msgs)
        assert "retry" in msg_text.lower()


# ===========================================================================
# REQ-EX-036: Score parsing performance
# ===========================================================================


@pytest.mark.unit
class TestScoreParsingPerformance:
    """parse_score executes in < 10ms for stdout up to 1MB (REQ-EX-036)."""

    def test_parse_score_under_10ms_for_1mb(self) -> None:
        """Score parsing completes in under 10ms for 1MB input."""
        from mle_star.scoring import parse_score

        # Create ~1MB of output with a score at the end
        filler = "Training epoch 1: loss=0.5\n" * 38_000  # ~1MB
        stdout = filler + "Final Validation Performance: 0.8196\n"
        assert len(stdout) >= 1_000_000

        start = time.perf_counter()
        result = parse_score(stdout)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result == pytest.approx(0.8196)
        assert elapsed_ms < 10.0, f"parse_score took {elapsed_ms:.2f}ms (limit: 10ms)"


# ===========================================================================
# REQ-EX-035: Execution overhead
# ===========================================================================


@pytest.mark.unit
class TestExecutionOverhead:
    """Execution overhead < 2s excluding script runtime (REQ-EX-035)."""

    @pytest.mark.asyncio
    async def test_overhead_under_2_seconds(self, tmp_path: Path) -> None:
        """Write + parse overhead is under 2 seconds."""
        from mle_star.execution import (
            build_evaluation_result,
            build_execution_env,
            write_script,
        )

        solution = make_solution(
            content="print('Final Validation Performance: 0.85')\n"
        )

        start = time.perf_counter()

        write_script(solution, str(tmp_path))
        build_execution_env()

        raw = ExecutionRawResult(
            stdout="Final Validation Performance: 0.85\n",
            stderr="",
            exit_code=0,
            duration_seconds=0.1,
            timed_out=False,
        )
        build_evaluation_result(raw)

        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"Overhead took {elapsed:.3f}s (limit: 2s)"


# ===========================================================================
# REQ-EX-032: ScoreFunction protocol compliance
# ===========================================================================


@pytest.mark.unit
class TestScoreFunctionCompliance:
    """evaluate_solution satisfies ScoreFunction protocol (REQ-EX-032)."""

    def test_evaluate_solution_signature_compatible(self) -> None:
        """evaluate_solution can be wrapped to satisfy ScoreFunction protocol."""
        from mle_star.execution import evaluate_solution

        sig = inspect.signature(evaluate_solution)
        params = list(sig.parameters.keys())
        assert "solution" in params
        assert "task" in params
        assert "config" in params


# ===========================================================================
# REQ-EX-028 to REQ-EX-031: Type alignment verification
# ===========================================================================


@pytest.mark.unit
class TestTypeAlignment:
    """All public functions use correct Spec 01 types (REQ-EX-028 to 031)."""

    def test_write_script_accepts_solution_script(self) -> None:
        """write_script accepts SolutionScript (REQ-EX-028)."""
        from mle_star.execution import write_script

        sig = inspect.signature(write_script)
        assert "solution" in sig.parameters

    def test_evaluate_solution_returns_evaluation_result(self) -> None:
        """evaluate_solution return type is EvaluationResult (REQ-EX-029)."""
        from mle_star.execution import evaluate_solution

        hints = inspect.get_annotations(evaluate_solution)
        assert "return" in hints

    def test_evaluate_solution_accepts_task_description(self) -> None:
        """evaluate_solution accepts TaskDescription (REQ-EX-030)."""
        from mle_star.execution import evaluate_solution

        sig = inspect.signature(evaluate_solution)
        assert "task" in sig.parameters

    def test_evaluate_solution_accepts_pipeline_config(self) -> None:
        """evaluate_solution accepts PipelineConfig (REQ-EX-031)."""
        from mle_star.execution import evaluate_solution

        sig = inspect.signature(evaluate_solution)
        assert "config" in sig.parameters


# ===========================================================================
# REQ-EX-042: No persistent state
# ===========================================================================


@pytest.mark.unit
class TestNoPersistentState:
    """No persistent state between executions (REQ-EX-042)."""

    @pytest.mark.asyncio
    async def test_independent_executions(self, tmp_path: Path) -> None:
        """Two sequential executions do not share state."""
        from mle_star.execution import execute_script

        script1 = _write_script(tmp_path, "print('run1')", name="script1.py")
        script2 = _write_script(tmp_path, "print('run2')", name="script2.py")

        result1 = await execute_script(script1, str(tmp_path), timeout_seconds=10)
        result2 = await execute_script(script2, str(tmp_path), timeout_seconds=10)

        assert "run1" in result1.stdout
        assert "run2" in result2.stdout
        assert "run1" not in result2.stdout
        assert "run2" not in result1.stdout
