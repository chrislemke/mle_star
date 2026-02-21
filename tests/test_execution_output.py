"""Tests for output parsing and evaluation result construction (Task 14).

Validates ``extract_traceback``, ``detect_error``, and
``build_evaluation_result`` which parse subprocess output into structured
``EvaluationResult`` instances.

Tests are written TDD-first and serve as the executable specification for
REQ-EX-012 through REQ-EX-014.

Refs:
    SRS 02c (Output Parsing), IMPLEMENTATION_PLAN.md Task 14.
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings, strategies as st
from mle_star.execution import (
    ExecutionRawResult,
    build_evaluation_result,
    detect_error,
    extract_traceback,
)
from mle_star.models import EvaluationResult
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw(**overrides: Any) -> ExecutionRawResult:
    """Build a valid ExecutionRawResult with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed ExecutionRawResult instance.
    """
    defaults: dict[str, Any] = {
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
        "duration_seconds": 1.0,
        "timed_out": False,
    }
    defaults.update(overrides)
    return ExecutionRawResult(**defaults)


# ---------------------------------------------------------------------------
# Realistic traceback fixtures used across multiple test classes
# ---------------------------------------------------------------------------

_SIMPLE_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "solution.py", line 10, in <module>\n'
    "    result = 1 / 0\n"
    "ZeroDivisionError: division by zero\n"
)

_VALUE_ERROR_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "solution.py", line 5, in <module>\n'
    "    int('abc')\n"
    "ValueError: invalid literal for int() with base 10: 'abc'\n"
)

_MULTILINE_EXCEPTION_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "solution.py", line 3, in <module>\n'
    "    raise RuntimeError(\n"
    '        "This is a multi-line\\n"\n'
    '        "error message"\n'
    "    )\n"
    "RuntimeError: This is a multi-line\n"
    "error message\n"
)

_CHAINED_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "solution.py", line 3, in <module>\n'
    "    int('abc')\n"
    "ValueError: invalid literal for int() with base 10: 'abc'\n"
    "\n"
    "During handling of the above exception, another exception occurred:\n"
    "\n"
    "Traceback (most recent call last):\n"
    '  File "solution.py", line 5, in <module>\n'
    "    raise RuntimeError('conversion failed')\n"
    "RuntimeError: conversion failed\n"
)

_NESTED_TRACEBACK_WITH_PREFIX = (
    "Loading data...\n"
    "Processing features...\n"
    "Traceback (most recent call last):\n"
    '  File "solution.py", line 20, in <module>\n'
    "    model.fit(X, y)\n"
    '  File "sklearn/base.py", line 100, in fit\n'
    "    self._fit(X, y)\n"
    "ValueError: could not convert string to float\n"
)


# ===========================================================================
# REQ-EX-012: extract_traceback -- Happy Path
# ===========================================================================


@pytest.mark.unit
class TestExtractTracebackHappyPath:
    """extract_traceback extracts Python traceback from stderr (REQ-EX-012)."""

    def test_extracts_simple_traceback(self) -> None:
        """Extracts a complete single-frame traceback."""
        result = extract_traceback(_SIMPLE_TRACEBACK)
        assert result is not None
        assert "Traceback (most recent call last):" in result
        assert "ZeroDivisionError: division by zero" in result

    def test_extracts_value_error_traceback(self) -> None:
        """Extracts a ValueError traceback."""
        result = extract_traceback(_VALUE_ERROR_TRACEBACK)
        assert result is not None
        assert "Traceback (most recent call last):" in result
        assert "ValueError" in result

    def test_extracts_multi_frame_traceback(self) -> None:
        """Extracts a traceback with multiple stack frames."""
        result = extract_traceback(_NESTED_TRACEBACK_WITH_PREFIX)
        assert result is not None
        assert "Traceback (most recent call last):" in result
        assert "ValueError: could not convert string to float" in result

    def test_returns_str_type(self) -> None:
        """Return type is str when traceback is found."""
        result = extract_traceback(_SIMPLE_TRACEBACK)
        assert result is not None
        assert isinstance(result, str)

    def test_traceback_with_prefix_text_excludes_prefix(self) -> None:
        """Text before the traceback marker is not included in the result."""
        result = extract_traceback(_NESTED_TRACEBACK_WITH_PREFIX)
        assert result is not None
        assert "Loading data..." not in result
        assert "Processing features..." not in result

    def test_traceback_starts_at_marker(self) -> None:
        """Extracted traceback starts with 'Traceback (most recent call last):'."""
        result = extract_traceback(_NESTED_TRACEBACK_WITH_PREFIX)
        assert result is not None
        assert result.startswith("Traceback (most recent call last):")


# ===========================================================================
# REQ-EX-012: extract_traceback -- Multiple Tracebacks (returns LAST)
# ===========================================================================


@pytest.mark.unit
class TestExtractTracebackMultiple:
    """extract_traceback returns the LAST traceback when multiple exist (REQ-EX-012)."""

    def test_two_tracebacks_returns_last(self) -> None:
        """When two independent tracebacks exist, returns the second one."""
        stderr = _SIMPLE_TRACEBACK + "\n" + _VALUE_ERROR_TRACEBACK
        result = extract_traceback(stderr)
        assert result is not None
        assert "ValueError" in result
        # The last traceback should NOT contain ZeroDivisionError
        assert "ZeroDivisionError" not in result

    def test_chained_traceback_returns_last(self) -> None:
        """Chained exception tracebacks: returns the last traceback block."""
        result = extract_traceback(_CHAINED_TRACEBACK)
        assert result is not None
        assert "RuntimeError: conversion failed" in result

    def test_three_tracebacks_returns_last(self) -> None:
        """Three tracebacks: returns the third (last) one."""
        tb1 = (
            "Traceback (most recent call last):\n"
            '  File "a.py", line 1, in <module>\n'
            "    x()\n"
            "NameError: name 'x' is not defined\n"
        )
        tb2 = (
            "Traceback (most recent call last):\n"
            '  File "b.py", line 2, in <module>\n'
            "    y()\n"
            "TypeError: y() missing 1 required argument\n"
        )
        tb3 = (
            "Traceback (most recent call last):\n"
            '  File "c.py", line 3, in <module>\n'
            "    z()\n"
            "AttributeError: 'NoneType' has no attribute 'z'\n"
        )
        stderr = tb1 + "\n" + tb2 + "\n" + tb3
        result = extract_traceback(stderr)
        assert result is not None
        assert "AttributeError" in result
        assert "NameError" not in result
        assert "TypeError" not in result


# ===========================================================================
# REQ-EX-012: extract_traceback -- No Match (returns None)
# ===========================================================================


@pytest.mark.unit
class TestExtractTracebackNoMatch:
    """extract_traceback returns None when no traceback is found (REQ-EX-012)."""

    def test_empty_string_returns_none(self) -> None:
        """Empty stderr returns None."""
        result = extract_traceback("")
        assert result is None

    def test_no_traceback_marker_returns_none(self) -> None:
        """Stderr without traceback marker returns None."""
        result = extract_traceback("Warning: deprecated function used\nDone.\n")
        assert result is None

    def test_whitespace_only_returns_none(self) -> None:
        """Whitespace-only stderr returns None."""
        result = extract_traceback("   \n\t  \n")
        assert result is None

    def test_partial_marker_returns_none(self) -> None:
        """Partial traceback marker without full text returns None."""
        result = extract_traceback("Traceback:\n  some error\n")
        assert result is None

    def test_case_mismatch_returns_none(self) -> None:
        """Wrong case for traceback marker returns None."""
        result = extract_traceback("traceback (most recent call last):\n  Error\n")
        assert result is None

    def test_warning_text_without_traceback_returns_none(self) -> None:
        """Stderr with warnings but no traceback returns None."""
        stderr = (
            "/usr/lib/python3/warnings.py:42: UserWarning: low accuracy\n"
            "  warnings.warn('low accuracy')\n"
            "Training complete with warnings.\n"
        )
        result = extract_traceback(stderr)
        assert result is None

    def test_returns_none_type(self) -> None:
        """Return value is exactly None (not empty string or other falsy)."""
        result = extract_traceback("no traceback here")
        assert result is None
        assert type(result) is type(None)


# ===========================================================================
# REQ-EX-012: extract_traceback -- Edge Cases
# ===========================================================================


@pytest.mark.unit
class TestExtractTracebackEdgeCases:
    """extract_traceback handles edge cases correctly (REQ-EX-012)."""

    def test_traceback_at_start_of_stderr(self) -> None:
        """Traceback that starts at the very beginning of stderr."""
        result = extract_traceback(_SIMPLE_TRACEBACK)
        assert result is not None
        assert "ZeroDivisionError" in result

    def test_traceback_at_end_of_stderr_without_trailing_newline(self) -> None:
        """Traceback at end of stderr without trailing newline."""
        stderr = (
            "Some output\n"
            "Traceback (most recent call last):\n"
            '  File "script.py", line 1, in <module>\n'
            "    raise ValueError('oops')\n"
            "ValueError: oops"
        )
        result = extract_traceback(stderr)
        assert result is not None
        assert "ValueError: oops" in result

    def test_traceback_with_deep_call_stack(self) -> None:
        """Traceback with many stack frames is fully captured."""
        frames = ""
        for i in range(20):
            frames += f'  File "module_{i}.py", line {i + 1}, in func_{i}\n'
            frames += f"    func_{i + 1}()\n"
        stderr = (
            "Traceback (most recent call last):\n"
            f"{frames}"
            "RecursionError: maximum recursion depth exceeded\n"
        )
        result = extract_traceback(stderr)
        assert result is not None
        assert "RecursionError" in result
        assert "Traceback (most recent call last):" in result

    def test_traceback_with_syntax_error_format(self) -> None:
        """SyntaxError tracebacks have a slightly different format but still extracted."""
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "script.py", line 1\n'
            "    def foo(:\n"
            "           ^\n"
            "SyntaxError: invalid syntax\n"
        )
        result = extract_traceback(stderr)
        assert result is not None
        assert "SyntaxError: invalid syntax" in result

    def test_stderr_with_interleaved_output_and_traceback(self) -> None:
        """Stderr with mixed warning output and a traceback."""
        stderr = (
            "WARNING: something happened\n"
            "DEBUG: processing step 3\n"
            "Traceback (most recent call last):\n"
            '  File "solution.py", line 42, in <module>\n'
            "    model.predict(X_test)\n"
            "RuntimeError: CUDA out of memory\n"
            "Cleanup complete.\n"
        )
        result = extract_traceback(stderr)
        assert result is not None
        assert "RuntimeError: CUDA out of memory" in result


# ===========================================================================
# REQ-EX-012: extract_traceback -- Property-based Tests
# ===========================================================================


@pytest.mark.unit
class TestExtractTracebackPropertyBased:
    """Property-based tests for extract_traceback using Hypothesis (REQ-EX-012)."""

    @given(
        text=st.text(
            alphabet=st.characters(
                blacklist_categories=("Cs",),  # type: ignore[arg-type]
            ),
            min_size=0,
            max_size=200,
        ),
    )
    @settings(max_examples=50)
    def test_no_marker_always_returns_none(self, text: str) -> None:
        """Property: stderr without 'Traceback (most recent call last):' returns None."""
        if "Traceback (most recent call last):" in text:
            return
        result = extract_traceback(text)
        assert result is None

    @given(
        prefix=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters=" _-.\n",
            ),
            min_size=0,
            max_size=100,
        ),
        exception_name=st.sampled_from(
            [
                "ValueError",
                "TypeError",
                "RuntimeError",
                "KeyError",
                "IndexError",
                "AttributeError",
                "ImportError",
                "OSError",
                "FileNotFoundError",
                "ZeroDivisionError",
            ]
        ),
        message=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters=" _-",
            ),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(max_examples=30)
    def test_constructed_traceback_always_extracted(
        self, prefix: str, exception_name: str, message: str
    ) -> None:
        """Property: a properly formed traceback is always extracted."""
        traceback_text = (
            "Traceback (most recent call last):\n"
            '  File "test.py", line 1, in <module>\n'
            "    x()\n"
            f"{exception_name}: {message}\n"
        )
        stderr = prefix + traceback_text
        result = extract_traceback(stderr)
        assert result is not None
        assert exception_name in result
        assert "Traceback (most recent call last):" in result

    @given(
        text=st.text(
            alphabet=st.characters(
                blacklist_categories=("Cs",),  # type: ignore[arg-type]
            ),
            min_size=0,
            max_size=200,
        ),
    )
    @settings(max_examples=30)
    def test_return_type_is_str_or_none(self, text: str) -> None:
        """Property: return type is always str or None."""
        result = extract_traceback(text)
        assert result is None or isinstance(result, str)


# ===========================================================================
# REQ-EX-013: detect_error -- Returns True
# ===========================================================================


@pytest.mark.unit
class TestDetectErrorTrue:
    """detect_error returns True when error conditions are present (REQ-EX-013)."""

    def test_nonzero_exit_code_is_error(self) -> None:
        """Non-zero exit code indicates an error."""
        raw = _make_raw(exit_code=1)
        assert detect_error(raw) is True

    def test_negative_exit_code_is_error(self) -> None:
        """Negative exit code (e.g., timeout) indicates an error."""
        raw = _make_raw(exit_code=-1)
        assert detect_error(raw) is True

    def test_timed_out_is_error(self) -> None:
        """Timed-out execution is an error."""
        raw = _make_raw(timed_out=True, exit_code=-1)
        assert detect_error(raw) is True

    def test_traceback_in_stderr_is_error(self) -> None:
        """Traceback marker in stderr indicates an error."""
        raw = _make_raw(
            stderr=_SIMPLE_TRACEBACK,
            exit_code=0,
        )
        assert detect_error(raw) is True

    def test_exit_code_and_traceback_both_present(self) -> None:
        """Multiple error conditions: non-zero exit code AND traceback."""
        raw = _make_raw(
            exit_code=1,
            stderr=_SIMPLE_TRACEBACK,
        )
        assert detect_error(raw) is True

    def test_all_three_conditions_true(self) -> None:
        """All three error conditions true simultaneously."""
        raw = _make_raw(
            exit_code=-1,
            timed_out=True,
            stderr=_SIMPLE_TRACEBACK,
        )
        assert detect_error(raw) is True

    def test_exit_code_and_timed_out(self) -> None:
        """Non-zero exit code and timed_out together."""
        raw = _make_raw(exit_code=-1, timed_out=True)
        assert detect_error(raw) is True

    def test_traceback_with_exit_zero(self) -> None:
        """Traceback in stderr with exit_code=0 is still an error.

        Some scripts may catch exceptions but still print tracebacks.
        """
        raw = _make_raw(
            exit_code=0,
            stderr=(
                "Traceback (most recent call last):\n"
                '  File "script.py", line 1, in <module>\n'
                "    raise ValueError('oops')\n"
                "ValueError: oops\n"
            ),
        )
        assert detect_error(raw) is True

    def test_returns_bool_type(self) -> None:
        """Return type is exactly bool."""
        raw = _make_raw(exit_code=1)
        result = detect_error(raw)
        assert isinstance(result, bool)


# ===========================================================================
# REQ-EX-013: detect_error -- Returns False
# ===========================================================================


@pytest.mark.unit
class TestDetectErrorFalse:
    """detect_error returns False when no error conditions are present (REQ-EX-013)."""

    def test_success_no_stderr(self) -> None:
        """Clean successful execution with empty stderr returns False."""
        raw = _make_raw(exit_code=0, stderr="", timed_out=False)
        assert detect_error(raw) is False

    def test_success_with_warning_stderr(self) -> None:
        """Successful execution with non-traceback stderr returns False."""
        raw = _make_raw(
            exit_code=0,
            stderr="UserWarning: deprecated function used\n",
            timed_out=False,
        )
        assert detect_error(raw) is False

    def test_success_with_debug_output_in_stderr(self) -> None:
        """Successful execution with debug output in stderr returns False."""
        raw = _make_raw(
            exit_code=0,
            stderr="DEBUG: loading model\nDEBUG: training started\nDEBUG: done\n",
            timed_out=False,
        )
        assert detect_error(raw) is False

    def test_success_with_stderr_containing_word_traceback(self) -> None:
        """Stderr containing 'traceback' (lowercase) does not trigger error detection.

        Only the exact marker 'Traceback (most recent call last):' counts.
        """
        raw = _make_raw(
            exit_code=0,
            stderr="See traceback above for details\n",
            timed_out=False,
        )
        assert detect_error(raw) is False

    def test_returns_bool_type_false(self) -> None:
        """Return type is exactly bool for False case."""
        raw = _make_raw(exit_code=0)
        result = detect_error(raw)
        assert isinstance(result, bool)


# ===========================================================================
# REQ-EX-013: detect_error -- Parametrized Exit Codes
# ===========================================================================


@pytest.mark.unit
class TestDetectErrorParametrized:
    """detect_error correctly identifies errors for various exit codes (REQ-EX-013)."""

    @pytest.mark.parametrize(
        "exit_code,expected",
        [
            (0, False),
            (1, True),
            (2, True),
            (42, True),
            (127, True),
            (255, True),
            (-1, True),
            (-9, True),
            (-15, True),
        ],
        ids=[
            "zero_success",
            "one_error",
            "two_error",
            "forty_two_error",
            "command_not_found",
            "max_unsigned_byte",
            "negative_one_timeout",
            "sigkill",
            "sigterm",
        ],
    )
    def test_exit_code_detection(self, exit_code: int, expected: bool) -> None:
        """Various exit codes correctly detected as error or success."""
        raw = _make_raw(exit_code=exit_code)
        assert detect_error(raw) is expected


# ===========================================================================
# REQ-EX-013: detect_error -- Property-based Tests
# ===========================================================================


@pytest.mark.unit
class TestDetectErrorPropertyBased:
    """Property-based tests for detect_error using Hypothesis (REQ-EX-013)."""

    @given(
        exit_code=st.integers(min_value=1, max_value=255),
    )
    @settings(max_examples=30)
    def test_positive_exit_code_always_error(self, exit_code: int) -> None:
        """Property: any positive exit code is always detected as error."""
        raw = _make_raw(exit_code=exit_code)
        assert detect_error(raw) is True

    @given(
        exit_code=st.integers(min_value=-128, max_value=-1),
    )
    @settings(max_examples=30)
    def test_negative_exit_code_always_error(self, exit_code: int) -> None:
        """Property: any negative exit code is always detected as error."""
        raw = _make_raw(exit_code=exit_code)
        assert detect_error(raw) is True

    @given(
        stderr=st.text(
            alphabet=st.characters(
                blacklist_categories=("Cs",),  # type: ignore[arg-type]
            ),
            min_size=0,
            max_size=200,
        ),
    )
    @settings(max_examples=30)
    def test_exit_zero_no_timeout_no_traceback_is_not_error(self, stderr: str) -> None:
        """Property: exit_code=0, not timed_out, no traceback in stderr is not an error."""
        if "Traceback (most recent call last):" in stderr:
            return
        raw = _make_raw(exit_code=0, timed_out=False, stderr=stderr)
        assert detect_error(raw) is False

    @given(
        stdout=st.text(max_size=100),
        duration=st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_timed_out_always_error(self, stdout: str, duration: float) -> None:
        """Property: timed_out=True always means error regardless of other fields."""
        raw = _make_raw(
            stdout=stdout,
            timed_out=True,
            exit_code=-1,
            duration_seconds=duration,
        )
        assert detect_error(raw) is True

    @given(
        exit_code=st.integers(min_value=-128, max_value=255),
        timed_out=st.booleans(),
        stderr=st.text(max_size=100),
    )
    @settings(max_examples=50)
    def test_returns_bool_type(
        self, exit_code: int, timed_out: bool, stderr: str
    ) -> None:
        """Property: detect_error always returns a bool."""
        raw = _make_raw(
            exit_code=exit_code,
            timed_out=timed_out,
            stderr=stderr,
        )
        result = detect_error(raw)
        assert isinstance(result, bool)


# ===========================================================================
# REQ-EX-014: build_evaluation_result -- Happy Path
# ===========================================================================


@pytest.mark.unit
class TestBuildEvaluationResultHappyPath:
    """build_evaluation_result constructs EvaluationResult correctly (REQ-EX-014)."""

    def test_returns_evaluation_result_type(self) -> None:
        """Return type is EvaluationResult."""
        raw = _make_raw(
            stdout="Final Validation Performance: 0.8196\n",
        )
        result = build_evaluation_result(raw)
        assert isinstance(result, EvaluationResult)

    def test_score_parsed_from_stdout(self) -> None:
        """Score is parsed from stdout using parse_score."""
        raw = _make_raw(
            stdout="Final Validation Performance: 0.8196\n",
        )
        result = build_evaluation_result(raw)
        assert result.score is not None
        assert result.score == pytest.approx(0.8196)

    def test_stdout_mapped_directly(self) -> None:
        """Stdout field is mapped directly from raw."""
        raw = _make_raw(stdout="some output text\n")
        result = build_evaluation_result(raw)
        assert result.stdout == "some output text\n"

    def test_stderr_mapped_directly(self) -> None:
        """Stderr field is mapped directly from raw."""
        raw = _make_raw(stderr="some error text\n")
        result = build_evaluation_result(raw)
        assert result.stderr == "some error text\n"

    def test_exit_code_mapped_directly(self) -> None:
        """exit_code field is mapped directly from raw."""
        raw = _make_raw(exit_code=0)
        result = build_evaluation_result(raw)
        assert result.exit_code == 0

    def test_duration_seconds_mapped_directly(self) -> None:
        """duration_seconds field is mapped directly from raw."""
        raw = _make_raw(duration_seconds=42.5)
        result = build_evaluation_result(raw)
        assert result.duration_seconds == pytest.approx(42.5)

    def test_is_error_false_for_success(self) -> None:
        """is_error is False for a successful execution."""
        raw = _make_raw(
            stdout="Final Validation Performance: 0.9\n",
            exit_code=0,
            timed_out=False,
            stderr="",
        )
        result = build_evaluation_result(raw)
        assert result.is_error is False

    def test_error_traceback_none_for_success(self) -> None:
        """error_traceback is None when is_error is False."""
        raw = _make_raw(
            stdout="Final Validation Performance: 0.9\n",
            exit_code=0,
            timed_out=False,
            stderr="",
        )
        result = build_evaluation_result(raw)
        assert result.error_traceback is None


# ===========================================================================
# REQ-EX-014: build_evaluation_result -- Score Parsing Integration
# ===========================================================================


@pytest.mark.unit
class TestBuildEvaluationResultScoreParsing:
    """build_evaluation_result integrates with parse_score (REQ-EX-014)."""

    def test_acceptance_criterion_single_score(self) -> None:
        """Acceptance: parse_score('Final Validation Performance: 0.8196') returns 0.8196."""
        raw = _make_raw(
            stdout="Final Validation Performance: 0.8196\n",
        )
        result = build_evaluation_result(raw)
        assert result.score == pytest.approx(0.8196)

    def test_acceptance_criterion_multiple_scores_returns_last(self) -> None:
        """Acceptance: multiple 'Performance:' lines returns the LAST match."""
        raw = _make_raw(
            stdout=(
                "Final Validation Performance: 0.5\n"
                "Final Validation Performance: 0.8196\n"
            ),
        )
        result = build_evaluation_result(raw)
        assert result.score == pytest.approx(0.8196)

    def test_acceptance_criterion_no_score_returns_none(self) -> None:
        """Acceptance: 'Training complete.' with no pattern returns None."""
        raw = _make_raw(
            stdout="Training complete.\n",
        )
        result = build_evaluation_result(raw)
        assert result.score is None

    def test_no_stdout_returns_none_score(self) -> None:
        """Empty stdout produces None score."""
        raw = _make_raw(stdout="")
        result = build_evaluation_result(raw)
        assert result.score is None

    def test_score_with_scientific_notation(self) -> None:
        """Score in scientific notation is parsed correctly."""
        raw = _make_raw(
            stdout="Final Validation Performance: 1.5e-3\n",
        )
        result = build_evaluation_result(raw)
        assert result.score is not None
        assert result.score == pytest.approx(1.5e-3)

    def test_score_with_negative_value(self) -> None:
        """Negative score is parsed correctly (e.g., neg_MSE metrics)."""
        raw = _make_raw(
            stdout="Final Validation Performance: -0.5432\n",
        )
        result = build_evaluation_result(raw)
        assert result.score is not None
        assert result.score == pytest.approx(-0.5432)


# ===========================================================================
# REQ-EX-014: build_evaluation_result -- Error Detection Integration
# ===========================================================================


@pytest.mark.unit
class TestBuildEvaluationResultErrorDetection:
    """build_evaluation_result integrates with detect_error (REQ-EX-014)."""

    def test_is_error_true_for_nonzero_exit(self) -> None:
        """is_error is True when exit_code != 0."""
        raw = _make_raw(exit_code=1, stderr="Error occurred\n")
        result = build_evaluation_result(raw)
        assert result.is_error is True

    def test_is_error_true_for_timeout(self) -> None:
        """is_error is True when timed_out is True."""
        raw = _make_raw(timed_out=True, exit_code=-1)
        result = build_evaluation_result(raw)
        assert result.is_error is True

    def test_is_error_true_for_traceback_in_stderr(self) -> None:
        """is_error is True when stderr contains traceback marker."""
        raw = _make_raw(stderr=_SIMPLE_TRACEBACK)
        result = build_evaluation_result(raw)
        assert result.is_error is True

    def test_is_error_false_for_clean_run(self) -> None:
        """is_error is False for a clean successful run."""
        raw = _make_raw(
            stdout="Final Validation Performance: 0.85\n",
            stderr="",
            exit_code=0,
            timed_out=False,
        )
        result = build_evaluation_result(raw)
        assert result.is_error is False


# ===========================================================================
# REQ-EX-014: build_evaluation_result -- Error Traceback Integration
# ===========================================================================


@pytest.mark.unit
class TestBuildEvaluationResultTraceback:
    """build_evaluation_result sets error_traceback only when is_error=True (REQ-EX-014)."""

    def test_error_traceback_set_when_is_error(self) -> None:
        """error_traceback is set from stderr traceback when is_error is True."""
        raw = _make_raw(
            exit_code=1,
            stderr=_SIMPLE_TRACEBACK,
        )
        result = build_evaluation_result(raw)
        assert result.is_error is True
        assert result.error_traceback is not None
        assert "ZeroDivisionError" in result.error_traceback

    def test_error_traceback_none_when_not_error(self) -> None:
        """error_traceback is None when is_error is False."""
        raw = _make_raw(
            exit_code=0,
            stderr="",
            timed_out=False,
        )
        result = build_evaluation_result(raw)
        assert result.is_error is False
        assert result.error_traceback is None

    def test_error_traceback_none_when_error_but_no_traceback(self) -> None:
        """error_traceback is None when is_error=True but stderr has no traceback.

        This happens for timeouts or exit codes without Python exceptions.
        """
        raw = _make_raw(
            exit_code=1,
            stderr="Segmentation fault\n",
        )
        result = build_evaluation_result(raw)
        assert result.is_error is True
        assert result.error_traceback is None

    def test_error_traceback_none_when_timed_out_no_traceback(self) -> None:
        """error_traceback is None for timeout without traceback in stderr."""
        raw = _make_raw(
            exit_code=-1,
            timed_out=True,
            stderr="",
        )
        result = build_evaluation_result(raw)
        assert result.is_error is True
        assert result.error_traceback is None

    def test_error_traceback_returns_last_traceback(self) -> None:
        """error_traceback contains the LAST traceback from stderr."""
        raw = _make_raw(
            exit_code=1,
            stderr=_CHAINED_TRACEBACK,
        )
        result = build_evaluation_result(raw)
        assert result.is_error is True
        assert result.error_traceback is not None
        assert "RuntimeError: conversion failed" in result.error_traceback

    def test_error_traceback_not_set_when_success_with_warnings(self) -> None:
        """error_traceback is None even if stderr has content but no error/traceback."""
        raw = _make_raw(
            exit_code=0,
            stderr="UserWarning: low accuracy detected\n",
            timed_out=False,
        )
        result = build_evaluation_result(raw)
        assert result.is_error is False
        assert result.error_traceback is None


# ===========================================================================
# REQ-EX-014: build_evaluation_result -- Full Integration Scenarios
# ===========================================================================


@pytest.mark.unit
class TestBuildEvaluationResultIntegration:
    """Full integration scenarios for build_evaluation_result (REQ-EX-014)."""

    def test_successful_run_with_score(self) -> None:
        """Complete successful run: score parsed, no error, no traceback."""
        raw = _make_raw(
            stdout=(
                "Loading data...\n"
                "Training model...\n"
                "Epoch 10/10 - loss: 0.1234\n"
                "Final Validation Performance: 0.8196\n"
            ),
            stderr="",
            exit_code=0,
            duration_seconds=120.5,
            timed_out=False,
        )
        result = build_evaluation_result(raw)
        assert result.score == pytest.approx(0.8196)
        assert result.is_error is False
        assert result.error_traceback is None
        assert result.exit_code == 0
        assert result.duration_seconds == pytest.approx(120.5)
        assert "Loading data..." in result.stdout
        assert result.stderr == ""

    def test_failed_run_with_traceback(self) -> None:
        """Failed run: no score, error detected, traceback extracted."""
        raw = _make_raw(
            stdout="Loading data...\n",
            stderr=_SIMPLE_TRACEBACK,
            exit_code=1,
            duration_seconds=5.0,
            timed_out=False,
        )
        result = build_evaluation_result(raw)
        assert result.score is None
        assert result.is_error is True
        assert result.error_traceback is not None
        assert "ZeroDivisionError" in result.error_traceback
        assert result.exit_code == 1
        assert result.duration_seconds == pytest.approx(5.0)

    def test_timed_out_run_with_partial_output(self) -> None:
        """Timed-out run with partial stdout but no score pattern."""
        raw = _make_raw(
            stdout="Epoch 1/100 - loss: 0.9876\n",
            stderr="",
            exit_code=-1,
            duration_seconds=300.0,
            timed_out=True,
        )
        result = build_evaluation_result(raw)
        assert result.score is None
        assert result.is_error is True
        assert result.error_traceback is None
        assert result.exit_code == -1
        assert result.duration_seconds == pytest.approx(300.0)

    def test_run_with_score_but_also_error(self) -> None:
        """Run that printed a score but also had a non-zero exit code.

        Score may still be parsed even if the run ultimately failed.
        """
        raw = _make_raw(
            stdout=("Final Validation Performance: 0.75\nError during cleanup\n"),
            stderr=_VALUE_ERROR_TRACEBACK,
            exit_code=1,
            duration_seconds=60.0,
            timed_out=False,
        )
        result = build_evaluation_result(raw)
        assert result.score == pytest.approx(0.75)
        assert result.is_error is True
        assert result.error_traceback is not None
        assert "ValueError" in result.error_traceback

    def test_run_with_stderr_warnings_but_success(self) -> None:
        """Successful run with warnings in stderr (common in ML workflows)."""
        raw = _make_raw(
            stdout="Final Validation Performance: 0.92\n",
            stderr=(
                "/usr/lib/python3/site-packages/sklearn/utils.py:42: "
                "FutureWarning: use the new API\n"
                "  warnings.warn('use the new API', FutureWarning)\n"
            ),
            exit_code=0,
            duration_seconds=45.0,
            timed_out=False,
        )
        result = build_evaluation_result(raw)
        assert result.score == pytest.approx(0.92)
        assert result.is_error is False
        assert result.error_traceback is None

    def test_result_is_frozen(self) -> None:
        """Returned EvaluationResult is frozen (immutable)."""
        raw = _make_raw(
            stdout="Final Validation Performance: 0.5\n",
        )
        result = build_evaluation_result(raw)
        with pytest.raises(Exception):  # noqa: B017
            result.score = 0.99  # type: ignore[misc]


# ===========================================================================
# REQ-EX-014: build_evaluation_result -- Property-based Tests
# ===========================================================================


@pytest.mark.unit
class TestBuildEvaluationResultPropertyBased:
    """Property-based tests for build_evaluation_result using Hypothesis (REQ-EX-014)."""

    @given(
        stdout=st.text(max_size=200),
        stderr=st.text(max_size=200),
        exit_code=st.integers(min_value=-128, max_value=255),
        duration=st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
        timed_out=st.booleans(),
    )
    @settings(max_examples=50)
    def test_always_returns_evaluation_result(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration: float,
        timed_out: bool,
    ) -> None:
        """Property: build_evaluation_result always returns an EvaluationResult."""
        raw = _make_raw(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=duration,
            timed_out=timed_out,
        )
        result = build_evaluation_result(raw)
        assert isinstance(result, EvaluationResult)

    @given(
        stdout=st.text(max_size=200),
        stderr=st.text(max_size=200),
        exit_code=st.integers(min_value=-128, max_value=255),
        duration=st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
        timed_out=st.booleans(),
    )
    @settings(max_examples=50)
    def test_stdout_stderr_always_mapped(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration: float,
        timed_out: bool,
    ) -> None:
        """Property: stdout, stderr, exit_code, duration are always mapped directly."""
        raw = _make_raw(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=duration,
            timed_out=timed_out,
        )
        result = build_evaluation_result(raw)
        assert result.stdout == stdout
        assert result.stderr == stderr
        assert result.exit_code == exit_code
        assert result.duration_seconds == duration

    @given(
        stdout=st.text(max_size=200),
        duration=st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_not_error_implies_no_traceback(self, stdout: str, duration: float) -> None:
        """Property: when is_error is False, error_traceback is always None."""
        raw = _make_raw(
            stdout=stdout,
            stderr="",
            exit_code=0,
            duration_seconds=duration,
            timed_out=False,
        )
        result = build_evaluation_result(raw)
        if not result.is_error:
            assert result.error_traceback is None

    @given(
        score=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=30)
    def test_formatted_score_round_trips(self, score: float) -> None:
        """Property: a formatted score in stdout is correctly parsed into result.score."""
        raw = _make_raw(
            stdout=f"Final Validation Performance: {score}\n",
        )
        result = build_evaluation_result(raw)
        assert result.score is not None
        assert result.score == pytest.approx(score, rel=1e-6, abs=1e-12)

    @given(
        exit_code=st.integers(min_value=1, max_value=255),
        stderr=st.text(max_size=100),
        duration=st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_nonzero_exit_always_is_error(
        self, exit_code: int, stderr: str, duration: float
    ) -> None:
        """Property: non-zero exit code always results in is_error=True."""
        raw = _make_raw(
            exit_code=exit_code,
            stderr=stderr,
            duration_seconds=duration,
        )
        result = build_evaluation_result(raw)
        assert result.is_error is True

    @given(
        stdout=st.text(max_size=100),
        stderr=st.text(max_size=100),
        exit_code=st.integers(min_value=-128, max_value=255),
        duration=st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
        timed_out=st.booleans(),
    )
    @settings(max_examples=50)
    def test_is_error_consistent_with_detect_error(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration: float,
        timed_out: bool,
    ) -> None:
        """Property: is_error field always matches detect_error(raw)."""
        raw = _make_raw(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=duration,
            timed_out=timed_out,
        )
        result = build_evaluation_result(raw)
        assert result.is_error == detect_error(raw)
