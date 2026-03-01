"""Tests for the MLE-STAR validation module.

Validates ``check_reproducibility``, ``check_sanity``, ``check_deep_leakage``,
``check_overfitting``, and ``validate_solution`` defined in
``src/mle_star/validation.py``.

Tests mock external dependencies (evaluate_solution, ClaudeCodeClient) and
verify correct ValidationCheck/ValidationResult construction for various
scenarios.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from mle_star.models import (
    AgentType,
    DataModality,
    EvaluationResult,
    MetricDirection,
    PipelineConfig,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
    TaskType,
    ValidationCheck,
    ValidationResult,
    ValidationStatus,
)
import pytest

_MODULE = "mle_star.validation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(**overrides: Any) -> TaskDescription:
    defaults: dict[str, Any] = {
        "competition_id": "test-comp",
        "task_type": TaskType.CLASSIFICATION,
        "data_modality": DataModality.TABULAR,
        "evaluation_metric": "accuracy",
        "metric_direction": MetricDirection.MAXIMIZE,
        "description": "Predict outcome.",
    }
    defaults.update(overrides)
    return TaskDescription(**defaults)


def _make_solution(**overrides: Any) -> SolutionScript:
    defaults: dict[str, Any] = {
        "content": "import pandas as pd\nprint('Final Validation Performance: 0.85')\n",
        "phase": SolutionPhase.INIT,
        "score": 0.85,
    }
    defaults.update(overrides)
    return SolutionScript(**defaults)


def _make_config(**overrides: Any) -> PipelineConfig:
    return PipelineConfig(**overrides)


def _make_eval_result(**overrides: Any) -> EvaluationResult:
    defaults: dict[str, Any] = {
        "score": 0.85,
        "stdout": "Final Validation Performance: 0.85",
        "stderr": "",
        "exit_code": 0,
        "duration_seconds": 10.0,
        "is_error": False,
        "error_traceback": None,
    }
    defaults.update(overrides)
    return EvaluationResult(**defaults)


# ===========================================================================
# ValidationCheck and ValidationResult models
# ===========================================================================


@pytest.mark.unit
class TestValidationModels:
    """ValidationCheck and ValidationResult Pydantic models."""

    def test_validation_check_construction(self) -> None:
        """ValidationCheck accepts all required fields."""
        check = ValidationCheck(
            name="test",
            status=ValidationStatus.PASSED,
            details="OK",
        )
        assert check.name == "test"
        assert check.status == ValidationStatus.PASSED
        assert check.details == "OK"
        assert check.scores is None

    def test_validation_check_with_scores(self) -> None:
        """ValidationCheck accepts optional scores list."""
        check = ValidationCheck(
            name="reproducibility",
            status=ValidationStatus.PASSED,
            details="OK",
            scores=[0.85, 0.84, 0.86],
        )
        assert check.scores == [0.85, 0.84, 0.86]

    def test_validation_result_construction(self) -> None:
        """ValidationResult aggregates checks with passed/baseline flags."""
        solution = _make_solution()
        checks = [
            ValidationCheck(
                name="a", status=ValidationStatus.PASSED, details="ok"
            ),
        ]
        result = ValidationResult(
            solution=solution,
            checks=checks,
            passed=True,
            baseline_beaten=True,
        )
        assert result.passed is True
        assert result.baseline_beaten is True
        assert len(result.checks) == 1

    def test_validation_status_values(self) -> None:
        """ValidationStatus has exactly 3 values."""
        assert len(ValidationStatus) == 3
        assert ValidationStatus.PASSED == "passed"
        assert ValidationStatus.FAILED == "failed"
        assert ValidationStatus.SKIPPED == "skipped"


# ===========================================================================
# check_reproducibility
# ===========================================================================


@pytest.mark.unit
class TestCheckReproducibility:
    """check_reproducibility re-runs with seeds and checks variance."""

    @pytest.mark.asyncio
    async def test_passed_when_scores_are_consistent(self) -> None:
        """Consistent scores across seed runs produce PASSED status."""
        from mle_star.validation import check_reproducibility

        consistent_results = [
            _make_eval_result(score=0.85),
            _make_eval_result(score=0.85),
            _make_eval_result(score=0.85),
        ]

        with patch(
            f"{_MODULE}.evaluate_solution",
            new_callable=AsyncMock,
            side_effect=consistent_results,
        ):
            result = await check_reproducibility(
                _make_solution(), _make_task(), _make_config()
            )

        assert result.name == "reproducibility"
        assert result.status == ValidationStatus.PASSED
        assert result.scores is not None
        assert len(result.scores) == 3

    @pytest.mark.asyncio
    async def test_failed_when_scores_vary_widely(self) -> None:
        """Highly variable scores produce FAILED status."""
        from mle_star.validation import check_reproducibility

        varied_results = [
            _make_eval_result(score=0.5),
            _make_eval_result(score=0.9),
            _make_eval_result(score=0.2),
        ]

        with patch(
            f"{_MODULE}.evaluate_solution",
            new_callable=AsyncMock,
            side_effect=varied_results,
        ):
            result = await check_reproducibility(
                _make_solution(), _make_task(), _make_config()
            )

        assert result.name == "reproducibility"
        assert result.status == ValidationStatus.FAILED

    @pytest.mark.asyncio
    async def test_skipped_when_too_few_scores(self) -> None:
        """SKIPPED when fewer than 2 runs produce scores."""
        from mle_star.validation import check_reproducibility

        error_results = [
            _make_eval_result(score=None, is_error=True),
            _make_eval_result(score=None, is_error=True),
            _make_eval_result(score=0.85),
        ]

        with patch(
            f"{_MODULE}.evaluate_solution",
            new_callable=AsyncMock,
            side_effect=error_results,
        ):
            result = await check_reproducibility(
                _make_solution(), _make_task(), _make_config()
            )

        assert result.name == "reproducibility"
        assert result.status == ValidationStatus.SKIPPED


# ===========================================================================
# check_sanity
# ===========================================================================


@pytest.mark.unit
class TestCheckSanity:
    """check_sanity generates shuffled-labels test via AI agent."""

    @pytest.mark.asyncio
    async def test_passed_when_shuffled_score_much_worse(self) -> None:
        """PASSED when shuffled labels degrade performance significantly."""
        from mle_star.validation import check_sanity

        client = MagicMock()
        client.send_message = AsyncMock(
            return_value="```python\nprint('Final Validation Performance: 0.5')\n```"
        )

        with patch(
            f"{_MODULE}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=_make_eval_result(score=0.5),
        ), patch(
            f"{_MODULE}.get_registry",
            return_value=MagicMock(
                get=MagicMock(
                    return_value=MagicMock(render=MagicMock(return_value="prompt"))
                )
            ),
        ):
            result = await check_sanity(
                _make_solution(), _make_task(), _make_config(), client
            )

        assert result.name == "sanity"
        assert result.status == ValidationStatus.PASSED

    @pytest.mark.asyncio
    async def test_failed_when_shuffled_score_similar(self) -> None:
        """FAILED when shuffled labels produce similar performance."""
        from mle_star.validation import check_sanity

        client = MagicMock()
        client.send_message = AsyncMock(
            return_value="```python\nprint('Final Validation Performance: 0.84')\n```"
        )

        with patch(
            f"{_MODULE}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=_make_eval_result(score=0.84),
        ), patch(
            f"{_MODULE}.get_registry",
            return_value=MagicMock(
                get=MagicMock(
                    return_value=MagicMock(render=MagicMock(return_value="prompt"))
                )
            ),
        ):
            result = await check_sanity(
                _make_solution(), _make_task(), _make_config(), client
            )

        assert result.name == "sanity"
        assert result.status == ValidationStatus.FAILED

    @pytest.mark.asyncio
    async def test_skipped_when_agent_returns_empty(self) -> None:
        """SKIPPED when agent returns empty code block."""
        from mle_star.validation import check_sanity

        client = MagicMock()
        client.send_message = AsyncMock(return_value="No code here")

        with patch(
            f"{_MODULE}.get_registry",
            return_value=MagicMock(
                get=MagicMock(
                    return_value=MagicMock(render=MagicMock(return_value="prompt"))
                )
            ),
        ), patch(
            f"{_MODULE}.extract_code_block",
            return_value="",
        ):
            result = await check_sanity(
                _make_solution(), _make_task(), _make_config(), client
            )

        assert result.name == "sanity"
        assert result.status == ValidationStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_skipped_on_exception(self) -> None:
        """SKIPPED when an exception occurs during the check."""
        from mle_star.validation import check_sanity

        client = MagicMock()
        client.send_message = AsyncMock(side_effect=RuntimeError("API error"))

        with patch(
            f"{_MODULE}.get_registry",
            return_value=MagicMock(
                get=MagicMock(
                    return_value=MagicMock(render=MagicMock(return_value="prompt"))
                )
            ),
        ):
            result = await check_sanity(
                _make_solution(), _make_task(), _make_config(), client
            )

        assert result.name == "sanity"
        assert result.status == ValidationStatus.SKIPPED


# ===========================================================================
# check_deep_leakage
# ===========================================================================


@pytest.mark.unit
class TestCheckDeepLeakage:
    """check_deep_leakage performs thorough AI-powered leakage analysis."""

    @pytest.mark.asyncio
    async def test_passed_when_clean(self) -> None:
        """PASSED when agent reports clean analysis."""
        from mle_star.validation import check_deep_leakage

        client = MagicMock()
        client.send_message = AsyncMock(
            return_value="Overall verdict: clean\nNo issues found."
        )

        with patch(
            f"{_MODULE}.get_registry",
            return_value=MagicMock(
                get=MagicMock(
                    return_value=MagicMock(render=MagicMock(return_value="prompt"))
                )
            ),
        ):
            result = await check_deep_leakage(
                _make_solution(), _make_task(), _make_config(), client
            )

        assert result.name == "deep_leakage"
        assert result.status == ValidationStatus.PASSED

    @pytest.mark.asyncio
    async def test_failed_when_leakage_detected(self) -> None:
        """FAILED when agent detects leakage."""
        from mle_star.validation import check_deep_leakage

        client = MagicMock()
        client.send_message = AsyncMock(
            return_value="leakage_detected: preprocessing uses full dataset"
        )

        with patch(
            f"{_MODULE}.get_registry",
            return_value=MagicMock(
                get=MagicMock(
                    return_value=MagicMock(render=MagicMock(return_value="prompt"))
                )
            ),
        ):
            result = await check_deep_leakage(
                _make_solution(), _make_task(), _make_config(), client
            )

        assert result.name == "deep_leakage"
        assert result.status == ValidationStatus.FAILED

    @pytest.mark.asyncio
    async def test_skipped_on_exception(self) -> None:
        """SKIPPED when an exception occurs."""
        from mle_star.validation import check_deep_leakage

        client = MagicMock()
        client.send_message = AsyncMock(side_effect=RuntimeError("timeout"))

        with patch(
            f"{_MODULE}.get_registry",
            return_value=MagicMock(
                get=MagicMock(
                    return_value=MagicMock(render=MagicMock(return_value="prompt"))
                )
            ),
        ):
            result = await check_deep_leakage(
                _make_solution(), _make_task(), _make_config(), client
            )

        assert result.name == "deep_leakage"
        assert result.status == ValidationStatus.SKIPPED


# ===========================================================================
# check_overfitting
# ===========================================================================


@pytest.mark.unit
class TestCheckOverfitting:
    """check_overfitting checks train/validation gap."""

    @pytest.mark.asyncio
    async def test_passed_when_gap_within_threshold(self) -> None:
        """PASSED when train-val gap is within threshold."""
        from mle_star.validation import check_overfitting

        client = MagicMock()
        client.send_message = AsyncMock(
            return_value="```python\nprint('Training Performance: 0.88')\nprint('Final Validation Performance: 0.85')\n```"
        )

        stdout = "Training Performance: 0.88\nFinal Validation Performance: 0.85"
        eval_result = _make_eval_result(
            score=0.85,
            stdout=stdout,
        )

        with patch(
            f"{_MODULE}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=eval_result,
        ), patch(
            f"{_MODULE}.get_registry",
            return_value=MagicMock(
                get=MagicMock(
                    return_value=MagicMock(render=MagicMock(return_value="prompt"))
                )
            ),
        ):
            result = await check_overfitting(
                _make_solution(), _make_task(), _make_config(), client
            )

        assert result.name == "overfitting"
        assert result.status == ValidationStatus.PASSED
        assert result.scores is not None
        assert len(result.scores) == 2

    @pytest.mark.asyncio
    async def test_failed_when_gap_exceeds_threshold(self) -> None:
        """FAILED when train-val gap exceeds threshold."""
        from mle_star.validation import check_overfitting

        client = MagicMock()
        client.send_message = AsyncMock(
            return_value="```python\nprint('done')\n```"
        )

        stdout = "Training Performance: 0.99\nFinal Validation Performance: 0.70"
        eval_result = _make_eval_result(
            score=0.70,
            stdout=stdout,
        )

        with patch(
            f"{_MODULE}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=eval_result,
        ), patch(
            f"{_MODULE}.get_registry",
            return_value=MagicMock(
                get=MagicMock(
                    return_value=MagicMock(render=MagicMock(return_value="prompt"))
                )
            ),
        ):
            result = await check_overfitting(
                _make_solution(), _make_task(), _make_config(), client
            )

        assert result.name == "overfitting"
        assert result.status == ValidationStatus.FAILED

    @pytest.mark.asyncio
    async def test_skipped_when_script_fails(self) -> None:
        """SKIPPED when modified script fails to execute."""
        from mle_star.validation import check_overfitting

        client = MagicMock()
        client.send_message = AsyncMock(
            return_value="```python\nraise Exception('fail')\n```"
        )

        with patch(
            f"{_MODULE}.evaluate_solution",
            new_callable=AsyncMock,
            return_value=_make_eval_result(is_error=True, score=None),
        ), patch(
            f"{_MODULE}.get_registry",
            return_value=MagicMock(
                get=MagicMock(
                    return_value=MagicMock(render=MagicMock(return_value="prompt"))
                )
            ),
        ):
            result = await check_overfitting(
                _make_solution(), _make_task(), _make_config(), client
            )

        assert result.name == "overfitting"
        assert result.status == ValidationStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_skipped_on_exception(self) -> None:
        """SKIPPED when an exception occurs."""
        from mle_star.validation import check_overfitting

        client = MagicMock()
        client.send_message = AsyncMock(side_effect=RuntimeError("error"))

        with patch(
            f"{_MODULE}.get_registry",
            return_value=MagicMock(
                get=MagicMock(
                    return_value=MagicMock(render=MagicMock(return_value="prompt"))
                )
            ),
        ):
            result = await check_overfitting(
                _make_solution(), _make_task(), _make_config(), client
            )

        assert result.name == "overfitting"
        assert result.status == ValidationStatus.SKIPPED


# ===========================================================================
# validate_solution -- top-level entry point
# ===========================================================================


@pytest.mark.unit
class TestValidateSolution:
    """validate_solution runs all 4 checks in parallel and aggregates results."""

    @pytest.mark.asyncio
    async def test_all_passed_returns_passed(self) -> None:
        """When all checks pass, result.passed is True."""
        from mle_star.validation import validate_solution

        passed_check = ValidationCheck(
            name="test", status=ValidationStatus.PASSED, details="ok"
        )

        with patch(
            f"{_MODULE}.check_reproducibility",
            new_callable=AsyncMock,
            return_value=passed_check,
        ), patch(
            f"{_MODULE}.check_sanity",
            new_callable=AsyncMock,
            return_value=passed_check,
        ), patch(
            f"{_MODULE}.check_deep_leakage",
            new_callable=AsyncMock,
            return_value=passed_check,
        ), patch(
            f"{_MODULE}.check_overfitting",
            new_callable=AsyncMock,
            return_value=passed_check,
        ):
            result = await validate_solution(
                _make_solution(),
                _make_task(),
                _make_config(),
                MagicMock(),
            )

        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert len(result.checks) == 4

    @pytest.mark.asyncio
    async def test_one_failed_returns_not_passed(self) -> None:
        """When any check fails, result.passed is False."""
        from mle_star.validation import validate_solution

        passed_check = ValidationCheck(
            name="ok", status=ValidationStatus.PASSED, details="ok"
        )
        failed_check = ValidationCheck(
            name="fail", status=ValidationStatus.FAILED, details="bad"
        )

        with patch(
            f"{_MODULE}.check_reproducibility",
            new_callable=AsyncMock,
            return_value=passed_check,
        ), patch(
            f"{_MODULE}.check_sanity",
            new_callable=AsyncMock,
            return_value=failed_check,
        ), patch(
            f"{_MODULE}.check_deep_leakage",
            new_callable=AsyncMock,
            return_value=passed_check,
        ), patch(
            f"{_MODULE}.check_overfitting",
            new_callable=AsyncMock,
            return_value=passed_check,
        ):
            result = await validate_solution(
                _make_solution(),
                _make_task(),
                _make_config(),
                MagicMock(),
            )

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_skipped_does_not_fail(self) -> None:
        """SKIPPED checks do not cause overall failure."""
        from mle_star.validation import validate_solution

        passed_check = ValidationCheck(
            name="ok", status=ValidationStatus.PASSED, details="ok"
        )
        skipped_check = ValidationCheck(
            name="skip", status=ValidationStatus.SKIPPED, details="skipped"
        )

        with patch(
            f"{_MODULE}.check_reproducibility",
            new_callable=AsyncMock,
            return_value=skipped_check,
        ), patch(
            f"{_MODULE}.check_sanity",
            new_callable=AsyncMock,
            return_value=passed_check,
        ), patch(
            f"{_MODULE}.check_deep_leakage",
            new_callable=AsyncMock,
            return_value=skipped_check,
        ), patch(
            f"{_MODULE}.check_overfitting",
            new_callable=AsyncMock,
            return_value=passed_check,
        ):
            result = await validate_solution(
                _make_solution(),
                _make_task(),
                _make_config(),
                MagicMock(),
            )

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_exception_in_check_becomes_skipped(self) -> None:
        """When a check raises an exception, it becomes SKIPPED in results."""
        from mle_star.validation import validate_solution

        passed_check = ValidationCheck(
            name="ok", status=ValidationStatus.PASSED, details="ok"
        )

        with patch(
            f"{_MODULE}.check_reproducibility",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ), patch(
            f"{_MODULE}.check_sanity",
            new_callable=AsyncMock,
            return_value=passed_check,
        ), patch(
            f"{_MODULE}.check_deep_leakage",
            new_callable=AsyncMock,
            return_value=passed_check,
        ), patch(
            f"{_MODULE}.check_overfitting",
            new_callable=AsyncMock,
            return_value=passed_check,
        ):
            result = await validate_solution(
                _make_solution(),
                _make_task(),
                _make_config(),
                MagicMock(),
            )

        # Should still pass because the exception becomes SKIPPED, not FAILED
        assert result.passed is True
        assert len(result.checks) == 4
        skipped = [c for c in result.checks if c.status == ValidationStatus.SKIPPED]
        assert len(skipped) == 1

    @pytest.mark.asyncio
    async def test_baseline_beaten_flag(self) -> None:
        """baseline_beaten reflects whether score beats the baseline."""
        from mle_star.validation import validate_solution

        passed_check = ValidationCheck(
            name="ok", status=ValidationStatus.PASSED, details="ok"
        )

        solution = _make_solution(score=0.9)
        task = _make_task(baseline_value=0.8)

        with patch(
            f"{_MODULE}.check_reproducibility",
            new_callable=AsyncMock,
            return_value=passed_check,
        ), patch(
            f"{_MODULE}.check_sanity",
            new_callable=AsyncMock,
            return_value=passed_check,
        ), patch(
            f"{_MODULE}.check_deep_leakage",
            new_callable=AsyncMock,
            return_value=passed_check,
        ), patch(
            f"{_MODULE}.check_overfitting",
            new_callable=AsyncMock,
            return_value=passed_check,
        ):
            result = await validate_solution(
                solution, task, _make_config(), MagicMock()
            )

        assert result.baseline_beaten is True

    @pytest.mark.asyncio
    async def test_baseline_not_beaten_flag(self) -> None:
        """baseline_beaten is False when score doesn't beat baseline."""
        from mle_star.validation import validate_solution

        passed_check = ValidationCheck(
            name="ok", status=ValidationStatus.PASSED, details="ok"
        )

        solution = _make_solution(score=0.7)
        task = _make_task(baseline_value=0.8)

        with patch(
            f"{_MODULE}.check_reproducibility",
            new_callable=AsyncMock,
            return_value=passed_check,
        ), patch(
            f"{_MODULE}.check_sanity",
            new_callable=AsyncMock,
            return_value=passed_check,
        ), patch(
            f"{_MODULE}.check_deep_leakage",
            new_callable=AsyncMock,
            return_value=passed_check,
        ), patch(
            f"{_MODULE}.check_overfitting",
            new_callable=AsyncMock,
            return_value=passed_check,
        ):
            result = await validate_solution(
                solution, task, _make_config(), MagicMock()
            )

        assert result.baseline_beaten is False
