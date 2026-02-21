"""Tests for the debugger safety agent (Task 19).

Validates ``extract_code_block``, ``debug_solution``, and
``make_debug_callback`` which implement the A_debugger agent for
fixing execution errors in solution scripts.

Tests are written TDD-first and serve as the executable specification for
REQ-SF-001 through REQ-SF-010.

Refs:
    SRS 03a (Safety Debugger), IMPLEMENTATION_PLAN.md Task 19.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

from hypothesis import given, settings, strategies as st
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
        "content": (
            'import pandas as pd\nprint(f"Final Validation Performance: {0.85}")\n'
        ),
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


_TRACEBACK_SAMPLE = (
    "Traceback (most recent call last):\n"
    '  File "solution.py", line 10, in <module>\n'
    "    result = 1 / 0\n"
    "ZeroDivisionError: division by zero"
)

_SAFETY = "mle_star.safety"


# ===========================================================================
# REQ-SF-005: extract_code_block -- Single Fenced Block
# ===========================================================================


@pytest.mark.unit
class TestExtractCodeBlockSingleFence:
    """extract_code_block extracts content from a single fenced block (REQ-SF-005)."""

    def test_python_tagged_fence(self) -> None:
        """Extracts code from a ```python ... ``` block."""
        from mle_star.safety import extract_code_block

        response = "```python\nprint('hello')\n```"
        assert extract_code_block(response) == "print('hello')"

    def test_generic_fence(self) -> None:
        """Extracts code from a ``` ... ``` block without language tag."""
        from mle_star.safety import extract_code_block

        response = "```\nprint('hello')\n```"
        assert extract_code_block(response) == "print('hello')"

    def test_strips_whitespace_in_block(self) -> None:
        """Strips leading/trailing whitespace from extracted code."""
        from mle_star.safety import extract_code_block

        response = "```python\n  \nprint('hello')\n  \n```"
        result = extract_code_block(response)
        assert result.strip() == result


# ===========================================================================
# REQ-SF-005: extract_code_block -- Multiple Fenced Blocks
# ===========================================================================


@pytest.mark.unit
class TestExtractCodeBlockMultipleFences:
    """extract_code_block returns the longest block when multiple exist (REQ-SF-005)."""

    def test_returns_longest_block(self) -> None:
        """When multiple fenced blocks exist, returns the longest one."""
        from mle_star.safety import extract_code_block

        response = (
            "```python\nshort\n```\n"
            "Some text\n"
            "```python\nthis is a much longer code block with more content\n```"
        )
        result = extract_code_block(response)
        assert "much longer" in result
        assert result != "short"

    def test_returns_longest_by_char_count(self) -> None:
        """Length comparison is by character count."""
        from mle_star.safety import extract_code_block

        response = "```\nab\n```\n```\nabcde\n```\n```\nabc\n```"
        assert extract_code_block(response) == "abcde"


# ===========================================================================
# REQ-SF-005: extract_code_block -- No Fences
# ===========================================================================


@pytest.mark.unit
class TestExtractCodeBlockNoFences:
    """extract_code_block returns stripped response when no fences exist (REQ-SF-005)."""

    def test_returns_stripped_text(self) -> None:
        """Returns the entire response stripped when no code fences found."""
        from mle_star.safety import extract_code_block

        response = "  just plain text  "
        assert extract_code_block(response) == "just plain text"

    def test_empty_response(self) -> None:
        """Returns empty string for empty input."""
        from mle_star.safety import extract_code_block

        assert extract_code_block("") == ""

    def test_whitespace_only(self) -> None:
        """Returns empty string for whitespace-only input."""
        from mle_star.safety import extract_code_block

        assert extract_code_block("   \n\n  ") == ""

    def test_multiline_no_fences(self) -> None:
        """Returns stripped multiline response when no fences."""
        from mle_star.safety import extract_code_block

        response = "  line1\nline2\nline3  "
        assert extract_code_block(response) == "line1\nline2\nline3"


# ===========================================================================
# REQ-SF-005: extract_code_block -- Edge Cases
# ===========================================================================


@pytest.mark.unit
class TestExtractCodeBlockEdgeCases:
    """extract_code_block handles edge cases correctly (REQ-SF-005)."""

    def test_surrounding_text_ignored(self) -> None:
        """Text surrounding a fenced block is not included."""
        from mle_star.safety import extract_code_block

        response = (
            "Here is the fix:\n```python\nprint('fixed')\n```\nThis should work now."
        )
        assert extract_code_block(response) == "print('fixed')"

    def test_multiline_code_block(self) -> None:
        """Multi-line code within fences is fully extracted."""
        from mle_star.safety import extract_code_block

        code = "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.shape)"
        response = f"```python\n{code}\n```"
        assert extract_code_block(response) == code


# ===========================================================================
# REQ-SF-010: _ensure_score_line -- Already Present
# ===========================================================================


@pytest.mark.unit
class TestEnsureScoreLinePresent:
    """_ensure_score_line returns code unchanged when score line exists (REQ-SF-010)."""

    def test_code_with_score_line_unchanged(self) -> None:
        """Code already containing 'Final Validation Performance' is returned as-is."""
        from mle_star.safety import _ensure_score_line

        code = 'print(f"Final Validation Performance: {score}")'
        assert _ensure_score_line(code) == code

    def test_partial_match_sufficient(self) -> None:
        """Any occurrence of the marker string is sufficient."""
        from mle_star.safety import _ensure_score_line

        code = "# Final Validation Performance is printed below\nprint(score)"
        assert _ensure_score_line(code) == code


# ===========================================================================
# REQ-SF-010: _ensure_score_line -- Missing, Appended
# ===========================================================================


@pytest.mark.unit
class TestEnsureScoreLineMissing:
    """_ensure_score_line appends score print when missing (REQ-SF-010)."""

    def test_appended_at_end(self) -> None:
        """When no score line and no __main__ block, appends at end."""
        from mle_star.safety import _ensure_score_line

        code = "x = 1\ny = 2"
        result = _ensure_score_line(code)
        assert "Final Validation Performance" in result
        assert result.startswith("x = 1")

    def test_inserted_before_main_block(self) -> None:
        """When no score line but __main__ exists, inserts before __main__."""
        from mle_star.safety import _ensure_score_line

        code = 'x = 1\n\nif __name__ == "__main__":\n    main()'
        result = _ensure_score_line(code)
        assert "Final Validation Performance" in result
        score_idx = result.find("Final Validation Performance")
        main_idx = result.find("if __name__")
        assert score_idx < main_idx


# ===========================================================================
# REQ-SF-003: debug_solution -- Input Validation
# ===========================================================================


@pytest.mark.unit
class TestDebugSolutionInputValidation:
    """debug_solution raises ValueError for invalid traceback input (REQ-SF-003)."""

    async def test_raises_on_empty_traceback(self) -> None:
        """Raises ValueError when traceback is empty string."""
        from mle_star.safety import debug_solution

        client = AsyncMock()
        with pytest.raises(ValueError, match="No traceback provided"):
            await debug_solution(
                _make_solution(), "", _make_task(), _make_config(), client
            )

    async def test_raises_on_none_traceback(self) -> None:
        """Raises ValueError when traceback is None."""
        from mle_star.safety import debug_solution

        client = AsyncMock()
        with pytest.raises(ValueError, match="No traceback provided"):
            await debug_solution(
                _make_solution(),
                None,  # type: ignore[arg-type]
                _make_task(),
                _make_config(),
                client,
            )


# ===========================================================================
# REQ-SF-006: debug_solution -- Is Async
# ===========================================================================


@pytest.mark.unit
class TestDebugSolutionIsAsync:
    """debug_solution is an async function (REQ-SF-006)."""

    def test_is_coroutine_function(self) -> None:
        """debug_solution is defined as an async function."""
        from mle_star.safety import debug_solution

        assert asyncio.iscoroutinefunction(debug_solution)


# ===========================================================================
# REQ-SF-006: debug_solution -- Success on First Attempt
# ===========================================================================


@pytest.mark.unit
class TestDebugSolutionFirstAttemptSuccess:
    """debug_solution returns on first successful fix (REQ-SF-006)."""

    async def test_success_on_first_attempt(self) -> None:
        """Fixed solution evaluates successfully on first debug attempt."""
        from mle_star.safety import debug_solution

        client = AsyncMock()
        fixed_code = 'print(f"Final Validation Performance: {0.85}")\n'
        client.send_message = AsyncMock(return_value=f"```python\n{fixed_code}\n```")

        success_result = _make_eval_result(is_error=False, score=0.85)

        with patch(
            f"{_SAFETY}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=success_result,
        ):
            _sol, result = await debug_solution(
                _make_solution(),
                _TRACEBACK_SAMPLE,
                _make_task(),
                _make_config(),
                client,
            )

        assert not result.is_error
        assert result.score == pytest.approx(0.85)
        assert client.send_message.call_count == 1

    async def test_returns_fixed_solution(self) -> None:
        """Returned solution contains the debugger's fixed code."""
        from mle_star.safety import debug_solution

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value='```python\nfixed_code_here\nprint(f"Final Validation Performance: {s}")\n```'
        )

        success_result = _make_eval_result(is_error=False, score=0.9)

        with patch(
            f"{_SAFETY}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=success_result,
        ):
            sol, _result = await debug_solution(
                _make_solution(),
                _TRACEBACK_SAMPLE,
                _make_task(),
                _make_config(),
                client,
            )

        assert "fixed_code_here" in sol.content


# ===========================================================================
# REQ-SF-006: debug_solution -- Retry on Failure
# ===========================================================================


@pytest.mark.unit
class TestDebugSolutionRetry:
    """debug_solution retries on evaluation failure (REQ-SF-006)."""

    async def test_retries_on_first_failure(self) -> None:
        """When first fix fails, debugger is invoked again with new traceback."""
        from mle_star.safety import debug_solution

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value='```python\nprint("fixed")\nprint(f"Final Validation Performance: {s}")\n```'
        )

        error_result = _make_eval_result(
            is_error=True,
            error_traceback="Traceback...\nTypeError: bad",
            score=None,
        )
        success_result = _make_eval_result(is_error=False, score=0.9)

        with patch(
            f"{_SAFETY}.evaluate_solution",
            new_callable=AsyncMock,
            side_effect=[error_result, success_result],
        ):
            _sol, result = await debug_solution(
                _make_solution(),
                _TRACEBACK_SAMPLE,
                _make_task(),
                _make_config(),
                client,
            )

        assert not result.is_error
        assert client.send_message.call_count == 2

    async def test_uses_new_traceback_for_retry(self) -> None:
        """Subsequent retries use the traceback from the most recent failure."""
        from mle_star.safety import debug_solution

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value='```python\ncode\nprint(f"Final Validation Performance: {s}")\n```'
        )

        error1 = _make_eval_result(is_error=True, error_traceback="first traceback")
        error2 = _make_eval_result(is_error=True, error_traceback="second traceback")
        success = _make_eval_result(is_error=False, score=0.8)

        with patch(
            f"{_SAFETY}.evaluate_solution",
            new_callable=AsyncMock,
            side_effect=[error1, error2, success],
        ):
            await debug_solution(
                _make_solution(),
                _TRACEBACK_SAMPLE,
                _make_task(),
                _make_config(),
                client,
            )

        # Third invocation should have received "second traceback"
        third_call_prompt = (
            client.send_message.call_args_list[2][1].get("message")
            or client.send_message.call_args_list[2][0][1]
        )
        assert "second traceback" in third_call_prompt


# ===========================================================================
# REQ-SF-006: debug_solution -- All Attempts Exhausted
# ===========================================================================


@pytest.mark.unit
class TestDebugSolutionExhausted:
    """debug_solution returns last pair when all attempts fail (REQ-SF-006)."""

    async def test_returns_last_pair_on_all_failures(self) -> None:
        """Returns the final (solution, result) even if still broken."""
        from mle_star.safety import debug_solution

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value='```python\nstill_broken\nprint(f"Final Validation Performance: {s}")\n```'
        )

        error_result = _make_eval_result(is_error=True, error_traceback="Traceback...")
        config = _make_config(max_debug_attempts=3)

        with patch(
            f"{_SAFETY}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=error_result,
        ):
            _sol, result = await debug_solution(
                _make_solution(),
                _TRACEBACK_SAMPLE,
                _make_task(),
                config,
                client,
            )

        assert result.is_error
        assert client.send_message.call_count == config.max_debug_attempts

    async def test_respects_max_debug_attempts(self) -> None:
        """Number of debug invocations equals config.max_debug_attempts."""
        from mle_star.safety import debug_solution

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value='```python\nx\nprint(f"Final Validation Performance: {s}")\n```'
        )

        error_result = _make_eval_result(is_error=True, error_traceback="Traceback...")

        for max_attempts in (1, 2, 5):
            client.send_message.reset_mock()
            config = _make_config(max_debug_attempts=max_attempts)

            with patch(
                f"{_SAFETY}.evaluate_solution",
                new_callable=AsyncMock,
                return_value=error_result,
            ):
                await debug_solution(
                    _make_solution(),
                    _TRACEBACK_SAMPLE,
                    _make_task(),
                    config,
                    client,
                )

            assert client.send_message.call_count == max_attempts


# ===========================================================================
# REQ-SF-004: debug_solution -- Output Contract
# ===========================================================================


@pytest.mark.unit
class TestDebugSolutionOutputContract:
    """debug_solution returns correct types and preserves phase (REQ-SF-004)."""

    async def test_returns_tuple(self) -> None:
        """Return value is a tuple of (SolutionScript, EvaluationResult)."""
        from mle_star.safety import debug_solution

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value='```python\nfixed\nprint(f"Final Validation Performance: {s}")\n```'
        )

        with patch(
            f"{_SAFETY}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=_make_eval_result(),
        ):
            result = await debug_solution(
                _make_solution(),
                _TRACEBACK_SAMPLE,
                _make_task(),
                _make_config(),
                client,
            )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], SolutionScript)
        assert isinstance(result[1], EvaluationResult)

    async def test_preserves_solution_phase(self) -> None:
        """Returned solution has the same phase as the input solution."""
        from mle_star.safety import debug_solution

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value='```python\nfixed\nprint(f"Final Validation Performance: {s}")\n```'
        )

        for phase in (
            SolutionPhase.INIT,
            SolutionPhase.REFINED,
            SolutionPhase.ENSEMBLE,
        ):
            with patch(
                f"{_SAFETY}.evaluate_solution",
                new_callable=AsyncMock,
                return_value=_make_eval_result(),
            ):
                sol, _ = await debug_solution(
                    _make_solution(phase=phase),
                    _TRACEBACK_SAMPLE,
                    _make_task(),
                    _make_config(),
                    client,
                )

            assert sol.phase == phase

    async def test_sets_is_executable_true(self) -> None:
        """Returned solution has is_executable=True (optimistic)."""
        from mle_star.safety import debug_solution

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value='```python\nfixed\nprint(f"Final Validation Performance: {s}")\n```'
        )

        with patch(
            f"{_SAFETY}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=_make_eval_result(),
        ):
            sol, _ = await debug_solution(
                _make_solution(is_executable=False),
                _TRACEBACK_SAMPLE,
                _make_task(),
                _make_config(),
                client,
            )

        assert sol.is_executable is True


# ===========================================================================
# REQ-SF-002: debug_solution -- Prompt Construction
# ===========================================================================


@pytest.mark.unit
class TestDebugSolutionPrompt:
    """debug_solution constructs prompt from PromptRegistry (REQ-SF-002)."""

    async def test_prompt_contains_solution_code(self) -> None:
        """The prompt sent to the agent contains the solution's source code."""
        from mle_star.safety import debug_solution

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value='```python\nfixed\nprint(f"Final Validation Performance: {s}")\n```'
        )

        solution = _make_solution(content="unique_code_marker_xyz = 42")

        with patch(
            f"{_SAFETY}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=_make_eval_result(),
        ):
            await debug_solution(
                solution,
                _TRACEBACK_SAMPLE,
                _make_task(),
                _make_config(),
                client,
            )

        call_args = client.send_message.call_args
        prompt = call_args[1].get("message") or call_args[0][1]
        assert "unique_code_marker_xyz" in prompt

    async def test_prompt_contains_traceback(self) -> None:
        """The prompt sent to the agent contains the error traceback."""
        from mle_star.safety import debug_solution

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value='```python\nfixed\nprint(f"Final Validation Performance: {s}")\n```'
        )

        with patch(
            f"{_SAFETY}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=_make_eval_result(),
        ):
            await debug_solution(
                _make_solution(),
                _TRACEBACK_SAMPLE,
                _make_task(),
                _make_config(),
                client,
            )

        call_args = client.send_message.call_args
        prompt = call_args[1].get("message") or call_args[0][1]
        assert "ZeroDivisionError" in prompt


# ===========================================================================
# REQ-SF-009: debug_solution -- Subsampling Preservation
# ===========================================================================


@pytest.mark.unit
class TestDebugSolutionSubsampling:
    """Debug prompt includes subsampling preservation instruction (REQ-SF-009)."""

    async def test_prompt_includes_subsampling_instruction(self) -> None:
        """The rendered prompt contains 'Do not remove subsampling'."""
        from mle_star.safety import debug_solution

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value='```python\nfixed\nprint(f"Final Validation Performance: {s}")\n```'
        )

        with patch(
            f"{_SAFETY}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=_make_eval_result(),
        ):
            await debug_solution(
                _make_solution(),
                _TRACEBACK_SAMPLE,
                _make_task(),
                _make_config(),
                client,
            )

        call_args = client.send_message.call_args
        prompt = call_args[1].get("message") or call_args[0][1]
        assert "subsampling" in prompt.lower()


# ===========================================================================
# REQ-SF-010: debug_solution -- Score Line Appended
# ===========================================================================


@pytest.mark.unit
class TestDebugSolutionScoreLine:
    """debug_solution appends score line if missing from debugged code (REQ-SF-010)."""

    async def test_appends_score_line_when_missing(self) -> None:
        """When debugged code lacks score print, it is appended."""
        from mle_star.safety import debug_solution

        client = AsyncMock()
        # Response without Final Validation Performance
        client.send_message = AsyncMock(
            return_value="```python\nresult = train_model()\nprint(result)\n```"
        )

        with patch(
            f"{_SAFETY}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=_make_eval_result(),
        ):
            sol, _ = await debug_solution(
                _make_solution(),
                _TRACEBACK_SAMPLE,
                _make_task(),
                _make_config(),
                client,
            )

        assert "Final Validation Performance" in sol.content

    async def test_preserves_existing_score_line(self) -> None:
        """When debugged code already has score print, content is unchanged."""
        from mle_star.safety import debug_solution

        client = AsyncMock()
        code_with_score = (
            'result = train_model()\nprint(f"Final Validation Performance: {result}")\n'
        )
        client.send_message = AsyncMock(
            return_value=f"```python\n{code_with_score}\n```"
        )

        with patch(
            f"{_SAFETY}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=_make_eval_result(),
        ):
            sol, _ = await debug_solution(
                _make_solution(),
                _TRACEBACK_SAMPLE,
                _make_task(),
                _make_config(),
                client,
            )

        # Should appear exactly once
        count = sol.content.count("Final Validation Performance")
        assert count == 1


# ===========================================================================
# REQ-SF-007: make_debug_callback -- Factory
# ===========================================================================


@pytest.mark.unit
class TestMakeDebugCallbackFactory:
    """make_debug_callback returns a callable compatible with evaluate_with_retry (REQ-SF-007)."""

    def test_returns_callable(self) -> None:
        """make_debug_callback returns a callable."""
        from mle_star.safety import make_debug_callback

        client = AsyncMock()
        callback = make_debug_callback(_make_task(), _make_config(), client)
        assert callable(callback)

    def test_callback_is_async(self) -> None:
        """The returned callback is an async function."""
        from mle_star.safety import make_debug_callback

        client = AsyncMock()
        callback = make_debug_callback(_make_task(), _make_config(), client)
        assert asyncio.iscoroutinefunction(callback)


# ===========================================================================
# REQ-SF-007: make_debug_callback -- Single Invocation
# ===========================================================================


@pytest.mark.unit
class TestMakeDebugCallbackInvocation:
    """make_debug_callback wraps a single debugger invocation (REQ-SF-007)."""

    async def test_invokes_debugger_once(self) -> None:
        """Each callback call invokes the debugger agent exactly once."""
        from mle_star.safety import make_debug_callback

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value='```python\nfixed_code\nprint(f"Final Validation Performance: {s}")\n```'
        )

        callback = make_debug_callback(_make_task(), _make_config(), client)
        result = await callback(_make_solution(), _TRACEBACK_SAMPLE)

        assert isinstance(result, SolutionScript)
        assert client.send_message.call_count == 1

    async def test_returns_original_on_empty_traceback(self) -> None:
        """When traceback is None, returns the original solution unchanged."""
        from mle_star.safety import make_debug_callback

        client = AsyncMock()
        callback = make_debug_callback(_make_task(), _make_config(), client)

        solution = _make_solution()
        result = await callback(solution, None)
        assert result is solution
        client.send_message.assert_not_called()

    async def test_returns_original_on_empty_string_traceback(self) -> None:
        """When traceback is empty string, returns the original solution."""
        from mle_star.safety import make_debug_callback

        client = AsyncMock()
        callback = make_debug_callback(_make_task(), _make_config(), client)

        solution = _make_solution()
        result = await callback(solution, "")
        assert result is solution
        client.send_message.assert_not_called()


# ===========================================================================
# REQ-SF-007: make_debug_callback -- Signature Compatibility
# ===========================================================================


@pytest.mark.unit
class TestMakeDebugCallbackSignature:
    """make_debug_callback signature is compatible with evaluate_with_retry (REQ-SF-007)."""

    async def test_compatible_with_evaluate_with_retry(self) -> None:
        """Callback can be passed directly to evaluate_with_retry."""
        from mle_star.safety import make_debug_callback

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value='```python\nfixed\nprint(f"Final Validation Performance: {s}")\n```'
        )
        callback = make_debug_callback(_make_task(), _make_config(), client)
        success_result = _make_eval_result(is_error=False)

        from mle_star.execution import evaluate_with_retry

        with patch(
            "mle_star.execution.evaluate_solution",
            new_callable=AsyncMock,
            return_value=success_result,
        ):
            _sol, result = await evaluate_with_retry(
                _make_solution(),
                _make_task(),
                _make_config(),
                debug_callback=callback,
            )

        assert not result.is_error


# ===========================================================================
# Property-based tests
# ===========================================================================


@pytest.mark.unit
class TestExtractCodeBlockPropertyBased:
    """Property-based tests for extract_code_block."""

    @given(code=st.text(min_size=1, max_size=200).filter(lambda s: "```" not in s))
    @settings(max_examples=50)
    def test_fenced_code_roundtrips(self, code: str) -> None:
        """Code wrapped in fences is extracted back (if no nested fences)."""
        from mle_star.safety import extract_code_block

        response = f"```python\n{code}\n```"
        result = extract_code_block(response)
        assert result == code.strip()

    @given(text=st.text(max_size=200).filter(lambda s: "```" not in s))
    @settings(max_examples=50)
    def test_no_fences_returns_stripped(self, text: str) -> None:
        """Without fences, returns the stripped input."""
        from mle_star.safety import extract_code_block

        assert extract_code_block(text) == text.strip()
