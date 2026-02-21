"""Tests for submission verification and batch evaluation (Task 17).

Validates ``verify_submission``, ``get_submission_info``,
``evaluate_batch``, and ``rank_solutions`` which handle submission
file checks, batch sequential evaluation, and solution ranking by
score.

Tests are written TDD-first and serve as the executable specification for
REQ-EX-024 through REQ-EX-027.

Refs:
    SRS 02d (Submission & Batch Evaluation), IMPLEMENTATION_PLAN.md Task 17.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from hypothesis import given, settings, strategies as st
from mle_star.execution import (
    evaluate_batch,
    get_submission_info,
    rank_solutions,
    verify_submission,
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
# REQ-EX-024: verify_submission -- File Exists and Non-Empty
# ===========================================================================


@pytest.mark.unit
class TestVerifySubmissionFileExists:
    """verify_submission returns True when file exists and has size > 0 (REQ-EX-024)."""

    def test_returns_true_for_existing_nonempty_file(self, tmp_path: Path) -> None:
        """Returns True when submission.csv exists in final/ with content."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        submission = final_dir / "submission.csv"
        submission.write_text("id,target\n1,0\n2,1\n", encoding="utf-8")

        result = verify_submission(str(tmp_path))

        assert result is True

    def test_returns_true_for_single_byte_file(self, tmp_path: Path) -> None:
        """Returns True when file has exactly 1 byte (size > 0)."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        submission = final_dir / "submission.csv"
        submission.write_text("x", encoding="utf-8")

        result = verify_submission(str(tmp_path))

        assert result is True

    def test_returns_true_for_custom_filename(self, tmp_path: Path) -> None:
        """Returns True for a custom expected filename."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        custom = final_dir / "output.csv"
        custom.write_text("id,prediction\n1,0.5\n", encoding="utf-8")

        result = verify_submission(str(tmp_path), expected_filename="output.csv")

        assert result is True


# ===========================================================================
# REQ-EX-024: verify_submission -- File Missing
# ===========================================================================


@pytest.mark.unit
class TestVerifySubmissionFileMissing:
    """verify_submission returns False when the file is missing (REQ-EX-024)."""

    def test_returns_false_when_file_does_not_exist(self, tmp_path: Path) -> None:
        """Returns False when submission.csv does not exist in final/."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()

        result = verify_submission(str(tmp_path))

        assert result is False

    def test_returns_false_when_final_dir_does_not_exist(self, tmp_path: Path) -> None:
        """Returns False when the final/ directory itself does not exist."""
        result = verify_submission(str(tmp_path))

        assert result is False

    def test_returns_false_for_wrong_filename(self, tmp_path: Path) -> None:
        """Returns False when a different file exists but not the expected one."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        (final_dir / "other.csv").write_text("data\n", encoding="utf-8")

        result = verify_submission(str(tmp_path))

        assert result is False


# ===========================================================================
# REQ-EX-024: verify_submission -- Empty File
# ===========================================================================


@pytest.mark.unit
class TestVerifySubmissionEmptyFile:
    """verify_submission returns False for an empty (0-byte) file (REQ-EX-024)."""

    def test_returns_false_for_empty_file(self, tmp_path: Path) -> None:
        """Returns False when submission.csv exists but is 0 bytes."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        submission = final_dir / "submission.csv"
        submission.write_text("", encoding="utf-8")

        result = verify_submission(str(tmp_path))

        assert result is False


# ===========================================================================
# REQ-EX-024: verify_submission -- Return Type
# ===========================================================================


@pytest.mark.unit
class TestVerifySubmissionReturnType:
    """verify_submission always returns a bool (REQ-EX-024)."""

    def test_returns_bool_when_file_exists(self, tmp_path: Path) -> None:
        """Return type is bool when file exists."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        (final_dir / "submission.csv").write_text("data\n", encoding="utf-8")

        result = verify_submission(str(tmp_path))

        assert isinstance(result, bool)

    def test_returns_bool_when_file_missing(self, tmp_path: Path) -> None:
        """Return type is bool when file is missing."""
        result = verify_submission(str(tmp_path))

        assert isinstance(result, bool)


# ===========================================================================
# REQ-EX-025: get_submission_info -- File Exists
# ===========================================================================


@pytest.mark.unit
class TestGetSubmissionInfoFileExists:
    """get_submission_info returns correct dict when file exists (REQ-EX-025)."""

    def test_exists_is_true(self, tmp_path: Path) -> None:
        """The 'exists' key is True when file is present."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        submission = final_dir / "submission.csv"
        submission.write_text("id,target\n1,0\n2,1\n", encoding="utf-8")

        info = get_submission_info(str(tmp_path))

        assert info["exists"] is True

    def test_path_is_absolute(self, tmp_path: Path) -> None:
        """The 'path' key is an absolute path string."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        submission = final_dir / "submission.csv"
        submission.write_text("id,target\n1,0\n", encoding="utf-8")

        info = get_submission_info(str(tmp_path))
        path_str = info["path"]
        assert isinstance(path_str, str)

        assert Path(path_str).is_absolute()

    def test_path_points_to_submission_file(self, tmp_path: Path) -> None:
        """The 'path' key ends with final/submission.csv."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        submission = final_dir / "submission.csv"
        submission.write_text("id,target\n1,0\n", encoding="utf-8")

        info = get_submission_info(str(tmp_path))
        path_str = info["path"]
        assert isinstance(path_str, str)

        assert path_str.endswith("final/submission.csv") or path_str.endswith(
            "final\\submission.csv"
        )

    def test_size_bytes_is_positive(self, tmp_path: Path) -> None:
        """The 'size_bytes' key is a positive integer."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        content = "id,target\n1,0\n2,1\n"
        submission = final_dir / "submission.csv"
        submission.write_text(content, encoding="utf-8")

        info = get_submission_info(str(tmp_path))

        assert isinstance(info["size_bytes"], int)
        assert info["size_bytes"] > 0

    def test_size_bytes_matches_actual_size(self, tmp_path: Path) -> None:
        """The 'size_bytes' matches the actual file size on disk."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        content = "id,target\n1,0\n2,1\n3,0\n"
        submission = final_dir / "submission.csv"
        submission.write_text(content, encoding="utf-8")
        expected_size = submission.stat().st_size

        info = get_submission_info(str(tmp_path))

        assert info["size_bytes"] == expected_size

    def test_row_count_excludes_header(self, tmp_path: Path) -> None:
        """The 'row_count' is lines minus header (data rows only)."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        content = "id,target\n1,0\n2,1\n3,0\n"
        submission = final_dir / "submission.csv"
        submission.write_text(content, encoding="utf-8")

        info = get_submission_info(str(tmp_path))

        assert info["row_count"] == 3

    def test_row_count_single_data_row(self, tmp_path: Path) -> None:
        """The 'row_count' is 1 when file has header + 1 data row."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        content = "id,target\n1,0\n"
        submission = final_dir / "submission.csv"
        submission.write_text(content, encoding="utf-8")

        info = get_submission_info(str(tmp_path))

        assert info["row_count"] == 1

    def test_row_count_header_only(self, tmp_path: Path) -> None:
        """The 'row_count' is 0 when file has only a header line."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        content = "id,target\n"
        submission = final_dir / "submission.csv"
        submission.write_text(content, encoding="utf-8")

        info = get_submission_info(str(tmp_path))

        assert info["row_count"] == 0

    def test_custom_filename(self, tmp_path: Path) -> None:
        """Works correctly with a custom expected_filename."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        content = "id,prediction\n1,0.5\n2,0.3\n"
        custom = final_dir / "predictions.csv"
        custom.write_text(content, encoding="utf-8")

        info = get_submission_info(str(tmp_path), expected_filename="predictions.csv")

        assert info["exists"] is True
        assert info["row_count"] == 2


# ===========================================================================
# REQ-EX-025: get_submission_info -- File Missing
# ===========================================================================


@pytest.mark.unit
class TestGetSubmissionInfoFileMissing:
    """get_submission_info returns correct dict when file is missing (REQ-EX-025)."""

    def test_exists_is_false(self, tmp_path: Path) -> None:
        """The 'exists' key is False when file is missing."""
        info = get_submission_info(str(tmp_path))

        assert info["exists"] is False

    def test_size_bytes_is_zero(self, tmp_path: Path) -> None:
        """The 'size_bytes' is 0 when file is missing."""
        info = get_submission_info(str(tmp_path))

        assert info["size_bytes"] == 0

    def test_row_count_is_none(self, tmp_path: Path) -> None:
        """The 'row_count' is None when file is missing."""
        info = get_submission_info(str(tmp_path))

        assert info["row_count"] is None

    def test_path_is_string(self, tmp_path: Path) -> None:
        """The 'path' key is a string even when file does not exist."""
        info = get_submission_info(str(tmp_path))

        assert isinstance(info["path"], str)

    def test_all_required_keys_present(self, tmp_path: Path) -> None:
        """All required keys (exists, path, size_bytes, row_count) are present."""
        info = get_submission_info(str(tmp_path))

        assert "exists" in info
        assert "path" in info
        assert "size_bytes" in info
        assert "row_count" in info


# ===========================================================================
# REQ-EX-025: get_submission_info -- Return Dict Shape
# ===========================================================================


@pytest.mark.unit
class TestGetSubmissionInfoDictShape:
    """get_submission_info returns a dict with the expected keys and types (REQ-EX-025)."""

    def test_returns_dict(self, tmp_path: Path) -> None:
        """Return type is a dict."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        (final_dir / "submission.csv").write_text("h\n1\n", encoding="utf-8")

        info = get_submission_info(str(tmp_path))

        assert isinstance(info, dict)

    def test_exists_is_bool(self, tmp_path: Path) -> None:
        """The 'exists' value is a bool."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        (final_dir / "submission.csv").write_text("h\n1\n", encoding="utf-8")

        info = get_submission_info(str(tmp_path))

        assert isinstance(info["exists"], bool)

    def test_path_is_str(self, tmp_path: Path) -> None:
        """The 'path' value is a str."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        (final_dir / "submission.csv").write_text("h\n1\n", encoding="utf-8")

        info = get_submission_info(str(tmp_path))

        assert isinstance(info["path"], str)

    def test_size_bytes_is_int(self, tmp_path: Path) -> None:
        """The 'size_bytes' value is an int."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        (final_dir / "submission.csv").write_text("h\n1\n", encoding="utf-8")

        info = get_submission_info(str(tmp_path))

        assert isinstance(info["size_bytes"], int)

    def test_row_count_is_int_when_exists(self, tmp_path: Path) -> None:
        """The 'row_count' value is an int when the file exists."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        (final_dir / "submission.csv").write_text("h\n1\n", encoding="utf-8")

        info = get_submission_info(str(tmp_path))

        assert isinstance(info["row_count"], int)

    def test_row_count_is_none_when_missing(self, tmp_path: Path) -> None:
        """The 'row_count' value is None when the file does not exist."""
        info = get_submission_info(str(tmp_path))

        assert info["row_count"] is None


# ===========================================================================
# REQ-EX-025: get_submission_info -- Edge Cases
# ===========================================================================


@pytest.mark.unit
class TestGetSubmissionInfoEdgeCases:
    """get_submission_info edge cases (REQ-EX-025)."""

    def test_file_with_trailing_newline(self, tmp_path: Path) -> None:
        """Trailing newline does not count as an extra row."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        content = "id,target\n1,0\n2,1\n"
        (final_dir / "submission.csv").write_text(content, encoding="utf-8")

        info = get_submission_info(str(tmp_path))

        assert info["row_count"] == 2

    def test_file_without_trailing_newline(self, tmp_path: Path) -> None:
        """File without trailing newline still counts rows correctly."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        content = "id,target\n1,0\n2,1"
        (final_dir / "submission.csv").write_text(content, encoding="utf-8")

        info = get_submission_info(str(tmp_path))

        assert info["row_count"] == 2

    def test_empty_file(self, tmp_path: Path) -> None:
        """An empty file has size_bytes=0 and appropriate row_count."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        (final_dir / "submission.csv").write_text("", encoding="utf-8")

        info = get_submission_info(str(tmp_path))

        assert info["exists"] is True
        assert info["size_bytes"] == 0


# ===========================================================================
# REQ-EX-026: evaluate_batch -- Is Async
# ===========================================================================


@pytest.mark.unit
class TestEvaluateBatchIsAsync:
    """evaluate_batch is an async function (REQ-EX-026)."""

    def test_evaluate_batch_is_coroutine_function(self) -> None:
        """evaluate_batch is defined as an async function."""
        assert asyncio.iscoroutinefunction(evaluate_batch)


# ===========================================================================
# REQ-EX-026: evaluate_batch -- Sequential Evaluation
# ===========================================================================


@pytest.mark.unit
class TestEvaluateBatchSequential:
    """evaluate_batch evaluates solutions sequentially, not concurrently (REQ-EX-026)."""

    async def test_evaluates_each_solution_once(self) -> None:
        """Each solution in the input list is evaluated exactly once."""
        solutions = [_make_solution(content=f"print({i})\n") for i in range(3)]
        task = _make_task()
        config = _make_config()
        result = _make_eval_result()

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=result,
        ) as m_eval:
            await evaluate_batch(solutions, task, config)

            assert m_eval.call_count == 3

    async def test_returns_results_in_same_order(self) -> None:
        """Results are returned in the same order as the input list."""
        solutions = [_make_solution(content=f"print({i})\n") for i in range(3)]
        task = _make_task()
        config = _make_config()
        results = [_make_eval_result(score=0.1 * (i + 1)) for i in range(3)]

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            side_effect=results,
        ):
            actual = await evaluate_batch(solutions, task, config)

            assert len(actual) == 3
            assert actual[0].score == pytest.approx(0.1)
            assert actual[1].score == pytest.approx(0.2)
            assert actual[2].score == pytest.approx(0.3)

    async def test_sequential_call_order(self) -> None:
        """Solutions are evaluated in sequential order, not concurrently."""
        call_order: list[str] = []

        async def track_eval(
            solution: Any, task: Any, config: Any, **kwargs: Any
        ) -> EvaluationResult:
            """Track evaluation call order."""
            call_order.append(solution.content)
            return _make_eval_result()

        solutions = [_make_solution(content=f"solution_{i}\n") for i in range(4)]
        task = _make_task()
        config = _make_config()

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            side_effect=track_eval,
        ):
            await evaluate_batch(solutions, task, config)

            assert call_order == [
                "solution_0\n",
                "solution_1\n",
                "solution_2\n",
                "solution_3\n",
            ]

    async def test_passes_correct_solution_to_each_call(self) -> None:
        """Each call to evaluate_solution receives the correct solution."""
        solutions = [_make_solution(content=f"code_{i}\n") for i in range(3)]
        task = _make_task()
        config = _make_config()

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=_make_eval_result(),
        ) as m_eval:
            await evaluate_batch(solutions, task, config)

            for i, call_args in enumerate(m_eval.call_args_list):
                assert call_args[0][0].content == f"code_{i}\n"

    async def test_passes_task_and_config_to_each_call(self) -> None:
        """Each call to evaluate_solution receives the same task and config."""
        solutions = [_make_solution(), _make_solution()]
        task = _make_task(competition_id="test-comp")
        config = _make_config(time_limit_seconds=600)

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=_make_eval_result(),
        ) as m_eval:
            await evaluate_batch(solutions, task, config)

            for call_args in m_eval.call_args_list:
                assert call_args[0][1].competition_id == "test-comp"
                assert call_args[0][2].time_limit_seconds == 600


# ===========================================================================
# REQ-EX-026: evaluate_batch -- Empty Input
# ===========================================================================


@pytest.mark.unit
class TestEvaluateBatchEmptyInput:
    """evaluate_batch handles an empty input list gracefully (REQ-EX-026)."""

    async def test_returns_empty_list_for_empty_input(self) -> None:
        """Returns an empty list when given an empty solutions list."""
        task = _make_task()
        config = _make_config()

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
        ) as m_eval:
            result = await evaluate_batch([], task, config)

            assert result == []
            m_eval.assert_not_called()


# ===========================================================================
# REQ-EX-026: evaluate_batch -- Single Solution
# ===========================================================================


@pytest.mark.unit
class TestEvaluateBatchSingleSolution:
    """evaluate_batch handles a single-solution input correctly (REQ-EX-026)."""

    async def test_single_solution_returns_single_result(self) -> None:
        """A single-element input list returns a single-element result list."""
        solution = _make_solution()
        task = _make_task()
        config = _make_config()
        expected = _make_eval_result(score=0.92)

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=expected,
        ):
            results = await evaluate_batch([solution], task, config)

            assert len(results) == 1
            assert results[0].score == pytest.approx(0.92)


# ===========================================================================
# REQ-EX-026: evaluate_batch -- Return Type
# ===========================================================================


@pytest.mark.unit
class TestEvaluateBatchReturnType:
    """evaluate_batch returns a list of EvaluationResult (REQ-EX-026)."""

    async def test_returns_list(self) -> None:
        """Return type is a list."""
        solutions = [_make_solution()]
        task = _make_task()
        config = _make_config()

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=_make_eval_result(),
        ):
            result = await evaluate_batch(solutions, task, config)

            assert isinstance(result, list)

    async def test_all_elements_are_evaluation_results(self) -> None:
        """Every element in the returned list is an EvaluationResult."""
        solutions = [_make_solution() for _ in range(3)]
        task = _make_task()
        config = _make_config()

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=_make_eval_result(),
        ):
            results = await evaluate_batch(solutions, task, config)

            for r in results:
                assert isinstance(r, EvaluationResult)

    async def test_result_length_matches_input_length(self) -> None:
        """The number of results matches the number of input solutions."""
        n = 5
        solutions = [_make_solution() for _ in range(n)]
        task = _make_task()
        config = _make_config()

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=_make_eval_result(),
        ):
            results = await evaluate_batch(solutions, task, config)

            assert len(results) == n


# ===========================================================================
# REQ-EX-026: evaluate_batch -- Error Propagation
# ===========================================================================


@pytest.mark.unit
class TestEvaluateBatchErrorResults:
    """evaluate_batch includes error results in the output list (REQ-EX-026)."""

    async def test_error_results_included(self) -> None:
        """Solutions that produce errors are included in results with is_error=True."""
        solutions = [_make_solution(), _make_solution()]
        task = _make_task()
        config = _make_config()
        success = _make_eval_result(is_error=False, score=0.85)
        error = _make_eval_result(is_error=True, score=None)

        with patch(
            f"{_EXEC}.evaluate_solution",
            new_callable=AsyncMock,
            side_effect=[success, error],
        ):
            results = await evaluate_batch(solutions, task, config)

            assert results[0].is_error is False
            assert results[1].is_error is True


# ===========================================================================
# REQ-EX-027: rank_solutions -- Maximize Direction
# ===========================================================================


@pytest.mark.unit
class TestRankSolutionsMaximize:
    """rank_solutions sorts by score descending for MAXIMIZE (REQ-EX-027)."""

    def test_highest_score_first(self) -> None:
        """Solution with the highest score is first for maximize."""
        sol_a = _make_solution(content="a\n")
        sol_b = _make_solution(content="b\n")
        sol_c = _make_solution(content="c\n")
        res_a = _make_eval_result(score=0.7, is_error=False)
        res_b = _make_eval_result(score=0.9, is_error=False)
        res_c = _make_eval_result(score=0.8, is_error=False)

        ranked = rank_solutions(
            [sol_a, sol_b, sol_c],
            [res_a, res_b, res_c],
            MetricDirection.MAXIMIZE,
        )

        assert ranked[0][1].score == pytest.approx(0.9)
        assert ranked[1][1].score == pytest.approx(0.8)
        assert ranked[2][1].score == pytest.approx(0.7)

    def test_preserves_solution_result_pairing(self) -> None:
        """Each solution is paired with its corresponding result after ranking."""
        sol_a = _make_solution(content="a\n")
        sol_b = _make_solution(content="b\n")
        res_a = _make_eval_result(score=0.3, is_error=False)
        res_b = _make_eval_result(score=0.9, is_error=False)

        ranked = rank_solutions(
            [sol_a, sol_b], [res_a, res_b], MetricDirection.MAXIMIZE
        )

        assert ranked[0][0].content == "b\n"
        assert ranked[0][1].score == pytest.approx(0.9)
        assert ranked[1][0].content == "a\n"
        assert ranked[1][1].score == pytest.approx(0.3)


# ===========================================================================
# REQ-EX-027: rank_solutions -- Minimize Direction
# ===========================================================================


@pytest.mark.unit
class TestRankSolutionsMinimize:
    """rank_solutions sorts by score ascending for MINIMIZE (REQ-EX-027)."""

    def test_lowest_score_first(self) -> None:
        """Solution with the lowest score is first for minimize."""
        sol_a = _make_solution(content="a\n")
        sol_b = _make_solution(content="b\n")
        sol_c = _make_solution(content="c\n")
        res_a = _make_eval_result(score=0.7, is_error=False)
        res_b = _make_eval_result(score=0.2, is_error=False)
        res_c = _make_eval_result(score=0.5, is_error=False)

        ranked = rank_solutions(
            [sol_a, sol_b, sol_c],
            [res_a, res_b, res_c],
            MetricDirection.MINIMIZE,
        )

        assert ranked[0][1].score == pytest.approx(0.2)
        assert ranked[1][1].score == pytest.approx(0.5)
        assert ranked[2][1].score == pytest.approx(0.7)

    def test_preserves_solution_result_pairing_minimize(self) -> None:
        """Each solution is paired with its corresponding result for minimize."""
        sol_a = _make_solution(content="a\n")
        sol_b = _make_solution(content="b\n")
        res_a = _make_eval_result(score=0.8, is_error=False)
        res_b = _make_eval_result(score=0.1, is_error=False)

        ranked = rank_solutions(
            [sol_a, sol_b], [res_a, res_b], MetricDirection.MINIMIZE
        )

        assert ranked[0][0].content == "b\n"
        assert ranked[0][1].score == pytest.approx(0.1)
        assert ranked[1][0].content == "a\n"
        assert ranked[1][1].score == pytest.approx(0.8)


# ===========================================================================
# REQ-EX-027: rank_solutions -- None Scores Placed After Valid Scores
# ===========================================================================


@pytest.mark.unit
class TestRankSolutionsNoneScores:
    """rank_solutions places None scores after valid scores (REQ-EX-027)."""

    def test_none_score_after_valid_maximize(self) -> None:
        """None-scored results appear after all valid-scored results for maximize."""
        sol_a = _make_solution(content="a\n")
        sol_b = _make_solution(content="b\n")
        sol_c = _make_solution(content="c\n")
        res_a = _make_eval_result(score=0.5, is_error=False)
        res_b = _make_eval_result(score=None, is_error=False)
        res_c = _make_eval_result(score=0.8, is_error=False)

        ranked = rank_solutions(
            [sol_a, sol_b, sol_c],
            [res_a, res_b, res_c],
            MetricDirection.MAXIMIZE,
        )

        assert ranked[0][1].score == pytest.approx(0.8)
        assert ranked[1][1].score == pytest.approx(0.5)
        assert ranked[2][1].score is None

    def test_none_score_after_valid_minimize(self) -> None:
        """None-scored results appear after all valid-scored results for minimize."""
        sol_a = _make_solution(content="a\n")
        sol_b = _make_solution(content="b\n")
        res_a = _make_eval_result(score=None, is_error=False)
        res_b = _make_eval_result(score=0.3, is_error=False)

        ranked = rank_solutions(
            [sol_a, sol_b], [res_a, res_b], MetricDirection.MINIMIZE
        )

        assert ranked[0][1].score == pytest.approx(0.3)
        assert ranked[1][1].score is None

    def test_multiple_none_scores(self) -> None:
        """Multiple None-scored results are all placed after valid ones."""
        sol_a = _make_solution(content="a\n")
        sol_b = _make_solution(content="b\n")
        sol_c = _make_solution(content="c\n")
        res_a = _make_eval_result(score=None, is_error=False)
        res_b = _make_eval_result(score=0.5, is_error=False)
        res_c = _make_eval_result(score=None, is_error=False)

        ranked = rank_solutions(
            [sol_a, sol_b, sol_c],
            [res_a, res_b, res_c],
            MetricDirection.MAXIMIZE,
        )

        assert ranked[0][1].score == pytest.approx(0.5)
        assert ranked[1][1].score is None
        assert ranked[2][1].score is None


# ===========================================================================
# REQ-EX-027: rank_solutions -- Error Results Placed Last
# ===========================================================================


@pytest.mark.unit
class TestRankSolutionsErrorResults:
    """rank_solutions places is_error=True results after None-scored results (REQ-EX-027)."""

    def test_error_after_none(self) -> None:
        """Error results appear after None-scored results."""
        sol_a = _make_solution(content="a\n")
        sol_b = _make_solution(content="b\n")
        sol_c = _make_solution(content="c\n")
        res_a = _make_eval_result(score=0.5, is_error=False)
        res_b = _make_eval_result(score=None, is_error=False)
        res_c = _make_eval_result(score=None, is_error=True)

        ranked = rank_solutions(
            [sol_a, sol_b, sol_c],
            [res_a, res_b, res_c],
            MetricDirection.MAXIMIZE,
        )

        # Valid score first
        assert ranked[0][1].score == pytest.approx(0.5)
        assert ranked[0][1].is_error is False
        # None score second
        assert ranked[1][1].score is None
        assert ranked[1][1].is_error is False
        # Error last
        assert ranked[2][1].is_error is True

    def test_error_after_none_and_valid_scores(self) -> None:
        """Ordering is: valid scores (sorted) > None scores > error results."""
        sol_a = _make_solution(content="a\n")
        sol_b = _make_solution(content="b\n")
        sol_c = _make_solution(content="c\n")
        sol_d = _make_solution(content="d\n")
        res_a = _make_eval_result(score=0.3, is_error=False)
        res_b = _make_eval_result(score=0.9, is_error=False)
        res_c = _make_eval_result(score=None, is_error=False)
        res_d = _make_eval_result(score=0.5, is_error=True)

        ranked = rank_solutions(
            [sol_a, sol_b, sol_c, sol_d],
            [res_a, res_b, res_c, res_d],
            MetricDirection.MAXIMIZE,
        )

        # Valid scores sorted descending
        assert ranked[0][1].score == pytest.approx(0.9)
        assert ranked[1][1].score == pytest.approx(0.3)
        # None score next
        assert ranked[2][1].score is None
        assert ranked[2][1].is_error is False
        # Error result last
        assert ranked[3][1].is_error is True

    def test_multiple_errors_all_at_end(self) -> None:
        """Multiple error results are all placed at the end."""
        sol_a = _make_solution(content="a\n")
        sol_b = _make_solution(content="b\n")
        sol_c = _make_solution(content="c\n")
        res_a = _make_eval_result(score=0.5, is_error=False)
        res_b = _make_eval_result(score=None, is_error=True)
        res_c = _make_eval_result(score=0.8, is_error=True)

        ranked = rank_solutions(
            [sol_a, sol_b, sol_c],
            [res_a, res_b, res_c],
            MetricDirection.MAXIMIZE,
        )

        assert ranked[0][1].score == pytest.approx(0.5)
        assert ranked[0][1].is_error is False
        assert ranked[1][1].is_error is True
        assert ranked[2][1].is_error is True

    def test_error_with_score_still_ranked_last(self) -> None:
        """Error results with high scores are still ranked after non-error results."""
        sol_a = _make_solution(content="a\n")
        sol_b = _make_solution(content="b\n")
        res_a = _make_eval_result(score=0.1, is_error=False)
        res_b = _make_eval_result(score=0.99, is_error=True)

        ranked = rank_solutions(
            [sol_a, sol_b], [res_a, res_b], MetricDirection.MAXIMIZE
        )

        assert ranked[0][0].content == "a\n"
        assert ranked[0][1].is_error is False
        assert ranked[1][0].content == "b\n"
        assert ranked[1][1].is_error is True


# ===========================================================================
# REQ-EX-027: rank_solutions -- Return Type
# ===========================================================================


@pytest.mark.unit
class TestRankSolutionsReturnType:
    """rank_solutions returns a list of (SolutionScript, EvaluationResult) tuples (REQ-EX-027)."""

    def test_returns_list(self) -> None:
        """Return type is a list."""
        sol = _make_solution()
        res = _make_eval_result()

        ranked = rank_solutions([sol], [res], MetricDirection.MAXIMIZE)

        assert isinstance(ranked, list)

    def test_elements_are_tuples(self) -> None:
        """Each element is a tuple."""
        sol = _make_solution()
        res = _make_eval_result()

        ranked = rank_solutions([sol], [res], MetricDirection.MAXIMIZE)

        assert isinstance(ranked[0], tuple)

    def test_tuple_contains_solution_and_result(self) -> None:
        """Each tuple contains (SolutionScript, EvaluationResult)."""
        sol = _make_solution()
        res = _make_eval_result()

        ranked = rank_solutions([sol], [res], MetricDirection.MAXIMIZE)

        assert isinstance(ranked[0][0], SolutionScript)
        assert isinstance(ranked[0][1], EvaluationResult)

    def test_empty_input_returns_empty_list(self) -> None:
        """Empty input lists return an empty list."""
        ranked = rank_solutions([], [], MetricDirection.MAXIMIZE)

        assert ranked == []

    def test_output_length_matches_input_length(self) -> None:
        """Output list length matches input list length."""
        solutions = [_make_solution() for _ in range(5)]
        results = [_make_eval_result(score=float(i)) for i in range(5)]

        ranked = rank_solutions(solutions, results, MetricDirection.MAXIMIZE)

        assert len(ranked) == 5


# ===========================================================================
# REQ-EX-027: rank_solutions -- Edge Cases
# ===========================================================================


@pytest.mark.unit
class TestRankSolutionsEdgeCases:
    """rank_solutions edge cases (REQ-EX-027)."""

    def test_single_solution(self) -> None:
        """A single solution is returned as-is in a list."""
        sol = _make_solution(content="only\n")
        res = _make_eval_result(score=0.5)

        ranked = rank_solutions([sol], [res], MetricDirection.MAXIMIZE)

        assert len(ranked) == 1
        assert ranked[0][0].content == "only\n"
        assert ranked[0][1].score == pytest.approx(0.5)

    def test_all_none_scores(self) -> None:
        """When all scores are None, all solutions are returned."""
        solutions = [_make_solution(content=f"s{i}\n") for i in range(3)]
        results = [_make_eval_result(score=None, is_error=False) for _ in range(3)]

        ranked = rank_solutions(solutions, results, MetricDirection.MAXIMIZE)

        assert len(ranked) == 3
        for _sol, res in ranked:
            assert res.score is None

    def test_all_error_results(self) -> None:
        """When all results are errors, all solutions are returned."""
        solutions = [_make_solution(content=f"s{i}\n") for i in range(3)]
        results = [_make_eval_result(is_error=True, score=None) for _ in range(3)]

        ranked = rank_solutions(solutions, results, MetricDirection.MAXIMIZE)

        assert len(ranked) == 3
        for _sol, res in ranked:
            assert res.is_error is True

    def test_equal_scores_all_returned(self) -> None:
        """When multiple solutions have the same score, all are included."""
        solutions = [_make_solution(content=f"s{i}\n") for i in range(3)]
        results = [_make_eval_result(score=0.5, is_error=False) for _ in range(3)]

        ranked = rank_solutions(solutions, results, MetricDirection.MAXIMIZE)

        assert len(ranked) == 3
        for _sol, res in ranked:
            assert res.score == pytest.approx(0.5)

    def test_negative_scores_maximize(self) -> None:
        """Negative scores are sorted correctly for maximize."""
        sol_a = _make_solution(content="a\n")
        sol_b = _make_solution(content="b\n")
        res_a = _make_eval_result(score=-0.5, is_error=False)
        res_b = _make_eval_result(score=-0.1, is_error=False)

        ranked = rank_solutions(
            [sol_a, sol_b], [res_a, res_b], MetricDirection.MAXIMIZE
        )

        assert ranked[0][1].score == pytest.approx(-0.1)
        assert ranked[1][1].score == pytest.approx(-0.5)

    def test_negative_scores_minimize(self) -> None:
        """Negative scores are sorted correctly for minimize."""
        sol_a = _make_solution(content="a\n")
        sol_b = _make_solution(content="b\n")
        res_a = _make_eval_result(score=-0.1, is_error=False)
        res_b = _make_eval_result(score=-0.5, is_error=False)

        ranked = rank_solutions(
            [sol_a, sol_b], [res_a, res_b], MetricDirection.MINIMIZE
        )

        assert ranked[0][1].score == pytest.approx(-0.5)
        assert ranked[1][1].score == pytest.approx(-0.1)


# ===========================================================================
# REQ-EX-027: rank_solutions -- Property-Based Tests
# ===========================================================================


@pytest.mark.unit
class TestRankSolutionsPropertyBased:
    """Property-based tests for rank_solutions using Hypothesis (REQ-EX-027)."""

    @given(
        scores=st.lists(
            st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            min_size=0,
            max_size=20,
        ),
        direction=st.sampled_from([MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]),
    )
    @settings(max_examples=50)
    def test_output_length_equals_input_length(
        self, scores: list[float], direction: MetricDirection
    ) -> None:
        """Property: output length always equals input length."""
        solutions = [_make_solution(content=f"s{i}\n") for i in range(len(scores))]
        results = [_make_eval_result(score=s, is_error=False) for s in scores]

        ranked = rank_solutions(solutions, results, direction)

        assert len(ranked) == len(scores)

    @given(
        scores=st.lists(
            st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=50)
    def test_best_score_first_maximize(self, scores: list[float]) -> None:
        """Property: for maximize, the first result has the highest score."""
        solutions = [_make_solution(content=f"s{i}\n") for i in range(len(scores))]
        results = [_make_eval_result(score=s, is_error=False) for s in scores]

        ranked = rank_solutions(solutions, results, MetricDirection.MAXIMIZE)

        assert ranked[0][1].score == pytest.approx(max(scores))

    @given(
        scores=st.lists(
            st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=50)
    def test_best_score_first_minimize(self, scores: list[float]) -> None:
        """Property: for minimize, the first result has the lowest score."""
        solutions = [_make_solution(content=f"s{i}\n") for i in range(len(scores))]
        results = [_make_eval_result(score=s, is_error=False) for s in scores]

        ranked = rank_solutions(solutions, results, MetricDirection.MINIMIZE)

        assert ranked[0][1].score == pytest.approx(min(scores))

    @given(
        n_valid=st.integers(min_value=0, max_value=5),
        n_none=st.integers(min_value=0, max_value=5),
        n_error=st.integers(min_value=0, max_value=5),
        direction=st.sampled_from([MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]),
    )
    @settings(max_examples=50)
    def test_valid_before_none_before_error(
        self,
        n_valid: int,
        n_none: int,
        n_error: int,
        direction: MetricDirection,
    ) -> None:
        """Property: valid scores come before None scores, which come before errors."""
        solutions: list[SolutionScript] = []
        results: list[EvaluationResult] = []

        for i in range(n_valid):
            solutions.append(_make_solution(content=f"valid_{i}\n"))
            results.append(_make_eval_result(score=float(i), is_error=False))

        for i in range(n_none):
            solutions.append(_make_solution(content=f"none_{i}\n"))
            results.append(_make_eval_result(score=None, is_error=False))

        for i in range(n_error):
            solutions.append(_make_solution(content=f"error_{i}\n"))
            results.append(_make_eval_result(score=None, is_error=True))

        ranked = rank_solutions(solutions, results, direction)

        total = n_valid + n_none + n_error
        assert len(ranked) == total

        # Check ordering tiers
        idx = 0
        # First tier: valid scores
        while (
            idx < total
            and ranked[idx][1].score is not None
            and not ranked[idx][1].is_error
        ):
            idx += 1
        # Second tier: None scores (non-error)
        while (
            idx < total and ranked[idx][1].score is None and not ranked[idx][1].is_error
        ):
            idx += 1
        # Third tier: errors
        while idx < total and ranked[idx][1].is_error:
            idx += 1
        assert idx == total

    @given(
        scores=st.lists(
            st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            min_size=0,
            max_size=15,
        ),
        direction=st.sampled_from([MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]),
    )
    @settings(max_examples=50)
    def test_idempotent_ranking(
        self, scores: list[float], direction: MetricDirection
    ) -> None:
        """Property: ranking an already-ranked list produces the same order."""
        solutions = [_make_solution(content=f"s{i}\n") for i in range(len(scores))]
        results = [_make_eval_result(score=s, is_error=False) for s in scores]

        ranked_once = rank_solutions(solutions, results, direction)
        ranked_twice = rank_solutions(
            [s for s, _r in ranked_once],
            [r for _s, r in ranked_once],
            direction,
        )

        for i in range(len(ranked_once)):
            assert ranked_once[i][0].content == ranked_twice[i][0].content
            assert ranked_once[i][1].score == ranked_twice[i][1].score

    @given(
        scores=st.lists(
            st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=15,
        ),
        direction=st.sampled_from([MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]),
    )
    @settings(max_examples=50)
    def test_all_input_solutions_present_in_output(
        self, scores: list[float], direction: MetricDirection
    ) -> None:
        """Property: all input solutions appear in the output (no duplication/loss)."""
        contents = [f"unique_content_{i}\n" for i in range(len(scores))]
        solutions = [_make_solution(content=c) for c in contents]
        results = [_make_eval_result(score=s, is_error=False) for s in scores]

        ranked = rank_solutions(solutions, results, direction)

        ranked_contents = sorted(s.content for s, _r in ranked)
        original_contents = sorted(contents)
        assert ranked_contents == original_contents


# ===========================================================================
# REQ-EX-027: rank_solutions -- Sorted Valid Scores
# ===========================================================================


@pytest.mark.unit
class TestRankSolutionsValidScoresSorted:
    """rank_solutions sorts valid scores in the correct order (REQ-EX-027)."""

    def test_maximize_descending_order(self) -> None:
        """Valid scores are in strictly descending order for maximize."""
        solutions = [_make_solution(content=f"s{i}\n") for i in range(5)]
        results = [
            _make_eval_result(score=s, is_error=False)
            for s in [0.1, 0.5, 0.3, 0.9, 0.7]
        ]

        ranked = rank_solutions(solutions, results, MetricDirection.MAXIMIZE)

        valid_scores = [r.score for _s, r in ranked if r.score is not None]
        for i in range(len(valid_scores) - 1):
            assert valid_scores[i] >= valid_scores[i + 1]

    def test_minimize_ascending_order(self) -> None:
        """Valid scores are in strictly ascending order for minimize."""
        solutions = [_make_solution(content=f"s{i}\n") for i in range(5)]
        results = [
            _make_eval_result(score=s, is_error=False)
            for s in [0.1, 0.5, 0.3, 0.9, 0.7]
        ]

        ranked = rank_solutions(solutions, results, MetricDirection.MINIMIZE)

        valid_scores = [r.score for _s, r in ranked if r.score is not None]
        for i in range(len(valid_scores) - 1):
            assert valid_scores[i] <= valid_scores[i + 1]


# ===========================================================================
# Cross-Function: verify_submission and get_submission_info Consistency
# ===========================================================================


@pytest.mark.unit
class TestVerifyAndInfoConsistency:
    """verify_submission and get_submission_info agree on file existence (REQ-EX-024, REQ-EX-025)."""

    def test_both_agree_file_exists(self, tmp_path: Path) -> None:
        """Both return consistent results when the file exists."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        (final_dir / "submission.csv").write_text("h\n1\n", encoding="utf-8")

        verified = verify_submission(str(tmp_path))
        info = get_submission_info(str(tmp_path))

        assert verified is True
        assert info["exists"] is True

    def test_both_agree_file_missing(self, tmp_path: Path) -> None:
        """Both return consistent results when the file is missing."""
        verified = verify_submission(str(tmp_path))
        info = get_submission_info(str(tmp_path))

        assert verified is False
        assert info["exists"] is False

    def test_both_agree_empty_file(self, tmp_path: Path) -> None:
        """Both return consistent results for an empty file."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        (final_dir / "submission.csv").write_text("", encoding="utf-8")

        verified = verify_submission(str(tmp_path))
        info = get_submission_info(str(tmp_path))

        # verify_submission returns False for empty file (size == 0)
        assert verified is False
        # get_submission_info reports exists=True but size_bytes=0
        assert info["exists"] is True
        assert info["size_bytes"] == 0
