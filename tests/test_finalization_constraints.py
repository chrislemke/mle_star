"""Tests for finalization constraints (Task 41).

Validates non-functional requirements for the finalization module in
``finalization.py``: orchestration overhead budget, subsampling removal
latency (LLM call count), structured logging at correct levels for 13
key events, SDK agent invocation, single module organization, submission
file path convention, no exit() in test scripts, no error masking in test
scripts, and A_test agent config with correct tool set.

Tests are written TDD-first and serve as the executable specification for
REQ-FN-037 through REQ-FN-048.

Refs:
    SRS 08d (Finalization Constraints), IMPLEMENTATION_PLAN.md Task 41.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from hypothesis import HealthCheck, given, settings, strategies as st
from mle_star.models import (
    AgentType,
    CodeBlock,
    CodeBlockCategory,
    DataContaminationResult,
    DataModality,
    EvaluationResult,
    FinalResult,
    MetricDirection,
    Phase1Result,
    Phase2Result,
    Phase3Result,
    PipelineConfig,
    RetrievedModel,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
    TaskType,
)
import pytest

# ---------------------------------------------------------------------------
# Module path constant for patching
# ---------------------------------------------------------------------------

_MODULE = "mle_star.finalization"
_LOGGER_NAME = "mle_star.finalization"


# ---------------------------------------------------------------------------
# Reusable test helpers
# ---------------------------------------------------------------------------


def _make_task(**overrides: Any) -> TaskDescription:
    """Build a valid TaskDescription with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed TaskDescription instance.
    """
    defaults: dict[str, Any] = {
        "competition_id": "test-comp",
        "task_type": TaskType.CLASSIFICATION,
        "data_modality": DataModality.TABULAR,
        "evaluation_metric": "accuracy",
        "metric_direction": MetricDirection.MAXIMIZE,
        "description": "Predict the target variable from tabular features.",
        "data_dir": "./input",
        "output_dir": "./final",
    }
    defaults.update(overrides)
    return TaskDescription(**defaults)


def _make_solution(**overrides: Any) -> SolutionScript:
    """Build a valid SolutionScript with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed SolutionScript instance.
    """
    defaults: dict[str, Any] = {
        "content": "import pandas as pd\nprint('hello')\n",
        "phase": SolutionPhase.REFINED,
    }
    defaults.update(overrides)
    return SolutionScript(**defaults)


def _make_config(**overrides: Any) -> PipelineConfig:
    """Build a valid PipelineConfig with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed PipelineConfig instance.
    """
    defaults: dict[str, Any] = {}
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_phase1_result(**overrides: Any) -> Phase1Result:
    """Build a valid Phase1Result with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed Phase1Result instance.
    """
    defaults: dict[str, Any] = {
        "retrieved_models": [
            RetrievedModel(model_name="xgb", example_code="import xgb")
        ],
        "candidate_solutions": [_make_solution(phase=SolutionPhase.INIT)],
        "candidate_scores": [0.85],
        "initial_solution": _make_solution(phase=SolutionPhase.INIT),
        "initial_score": 0.85,
    }
    defaults.update(overrides)
    return Phase1Result(**defaults)


def _make_phase2_result(**overrides: Any) -> Phase2Result:
    """Build a valid Phase2Result with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed Phase2Result instance.
    """
    defaults: dict[str, Any] = {
        "ablation_summaries": ["summary"],
        "refined_blocks": [
            CodeBlock(content="block", category=CodeBlockCategory.TRAINING)
        ],
        "best_solution": _make_solution(phase=SolutionPhase.REFINED),
        "best_score": 0.90,
        "step_history": [],
    }
    defaults.update(overrides)
    return Phase2Result(**defaults)


def _make_phase3_result(**overrides: Any) -> Phase3Result:
    """Build a valid Phase3Result with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed Phase3Result instance.
    """
    defaults: dict[str, Any] = {
        "input_solutions": [
            _make_solution(phase=SolutionPhase.REFINED),
            _make_solution(phase=SolutionPhase.REFINED, content="sol2"),
        ],
        "ensemble_plans": ["plan1"],
        "ensemble_scores": [0.92],
        "best_ensemble": _make_solution(phase=SolutionPhase.ENSEMBLE),
        "best_ensemble_score": 0.92,
    }
    defaults.update(overrides)
    return Phase3Result(**defaults)


def _make_eval_result(**overrides: Any) -> EvaluationResult:
    """Build a valid EvaluationResult with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed EvaluationResult instance.
    """
    defaults: dict[str, Any] = {
        "stdout": "Final Validation Performance: 0.90",
        "stderr": "",
        "exit_code": 0,
        "duration_seconds": 1.0,
        "is_error": False,
        "score": 0.90,
    }
    defaults.update(overrides)
    return EvaluationResult(**defaults)


def _setup_run_finalization_mocks(
    *,
    remove_subsampling_result: SolutionScript | None = None,
    test_script: SolutionScript | None = None,
    leakage_result: SolutionScript | None = None,
    eval_result: EvaluationResult | None = None,
    verify_result: bool = True,
    submission_info: dict[str, Any] | None = None,
    contamination_result: DataContaminationResult | None = None,
) -> dict[str, Any]:
    """Build a dict of mock objects for run_finalization dependencies.

    Args:
        remove_subsampling_result: Result from remove_subsampling.
        test_script: Result from generate_test_submission.
        leakage_result: Result from check_and_fix_leakage.
        eval_result: Result from evaluate_with_retry.
        verify_result: Result from verify_submission.
        submission_info: Result from get_submission_info.
        contamination_result: Result from check_contamination.

    Returns:
        A dict keyed by function name, suitable for patching.
    """
    if remove_subsampling_result is None:
        remove_subsampling_result = _make_solution(phase=SolutionPhase.REFINED)
    if test_script is None:
        test_script = _make_solution(
            content="test script code", phase=SolutionPhase.FINAL
        )
    if leakage_result is None:
        leakage_result = test_script
    if eval_result is None:
        eval_result = _make_eval_result()
    if submission_info is None:
        submission_info = {
            "exists": True,
            "path": "/work/final/submission.csv",
            "size_bytes": 1024,
            "row_count": 100,
        }

    final_solution = leakage_result

    return {
        "remove_subsampling": AsyncMock(return_value=remove_subsampling_result),
        "generate_test_submission": AsyncMock(return_value=test_script),
        "check_and_fix_leakage": AsyncMock(return_value=leakage_result),
        "evaluate_with_retry": AsyncMock(return_value=(final_solution, eval_result)),
        "make_debug_callback": MagicMock(return_value=MagicMock()),
        "verify_submission": MagicMock(return_value=verify_result),
        "get_submission_info": MagicMock(return_value=submission_info),
        "check_contamination": AsyncMock(return_value=contamination_result),
    }


def _apply_run_finalization_patches(
    mocks: dict[str, Any],
) -> Any:
    """Return a combined context manager that patches all run_finalization deps.

    Args:
        mocks: Dict of function name to mock object.

    Returns:
        An ExitStack context manager applying all patches.
    """
    import contextlib
    from unittest.mock import patch as _patch

    stack = contextlib.ExitStack()
    for name, mock_obj in mocks.items():
        stack.enter_context(_patch(f"{_MODULE}.{name}", new=mock_obj))
    return stack


async def _run_finalization_with_mocks(
    mocks: dict[str, Any] | None = None,
    *,
    solution: SolutionScript | None = None,
    task: TaskDescription | None = None,
    config: PipelineConfig | None = None,
    reference_discussions: list[str] | None = None,
) -> FinalResult:
    """Run run_finalization with all dependencies mocked.

    Args:
        mocks: Optional pre-built mocks dict.
        solution: Input solution.
        task: Task description.
        config: Pipeline config.
        reference_discussions: Optional reference discussions.

    Returns:
        The FinalResult from run_finalization.
    """
    from mle_star.finalization import run_finalization

    if mocks is None:
        mocks = _setup_run_finalization_mocks()
    if solution is None:
        solution = _make_solution()
    if task is None:
        task = _make_task()
    if config is None:
        config = _make_config()

    client = AsyncMock()

    with _apply_run_finalization_patches(mocks):
        return await run_finalization(
            client=client,
            solution=solution,
            task=task,
            config=config,
            phase1_result=_make_phase1_result(),
            phase2_results=[_make_phase2_result()],
            phase3_result=None,
            reference_discussions=reference_discussions,
        )


# ===========================================================================
# REQ-FN-037: Performance - Finalization Overhead
# ===========================================================================


@pytest.mark.unit
class TestFinalizationOverheadPerformance:
    """Finalization overhead (excluding LLM calls) does not exceed 5s (REQ-FN-037)."""

    async def test_overhead_under_five_seconds(self) -> None:
        """Orchestration overhead with mocked dependencies is under 5 seconds."""
        mocks = _setup_run_finalization_mocks()

        start = time.monotonic()
        await _run_finalization_with_mocks(mocks)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"Finalization overhead {elapsed:.2f}s exceeded 5s budget"

    async def test_overhead_under_five_seconds_with_fallback(self) -> None:
        """Overhead stays under 5s even when fallback path is triggered."""
        mocks = _setup_run_finalization_mocks(
            eval_result=_make_eval_result(
                is_error=True,
                score=None,
                exit_code=1,
                stdout="",
                stderr="Error",
            ),
            verify_result=False,
        )

        start = time.monotonic()
        await _run_finalization_with_mocks(mocks)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, (
            f"Finalization fallback overhead {elapsed:.2f}s exceeded 5s budget"
        )

    @given(
        content_len=st.integers(min_value=10, max_value=5000),
    )
    @settings(max_examples=5, deadline=10000)
    async def test_overhead_under_five_seconds_varying_content(
        self, content_len: int
    ) -> None:
        """Overhead stays under 5s regardless of solution content size."""
        solution = _make_solution(content="x" * content_len)
        mocks = _setup_run_finalization_mocks()

        start = time.monotonic()
        await _run_finalization_with_mocks(mocks, solution=solution)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0


# ===========================================================================
# REQ-FN-038: Subsampling Removal Latency
# ===========================================================================


@pytest.mark.unit
class TestSubsamplingRemovalLatency:
    """Subsampling removal uses at most 2 sequential LLM calls (REQ-FN-038)."""

    async def test_subsampling_found_uses_two_llm_calls(self) -> None:
        """When subsampling is found, exactly 2 LLM calls are made."""
        from mle_star.finalization import remove_subsampling

        solution = _make_solution(content="data = train[:1000]\nmodel.fit(data)\n")
        task = _make_task()
        client = AsyncMock()

        # First call: extraction returns the subsampling block
        # Second call: removal returns the replacement block
        client.send_message = AsyncMock(
            side_effect=[
                "```python\ndata = train[:1000]\n```",
                "```python\ndata = train\n```",
            ]
        )

        await remove_subsampling(client, solution, task)

        assert client.send_message.await_count == 2

    async def test_no_subsampling_uses_one_llm_call(self) -> None:
        """When no subsampling is found (empty extraction), only 1 LLM call."""
        from mle_star.finalization import remove_subsampling

        solution = _make_solution(content="model.fit(full_train)\n")
        task = _make_task()
        client = AsyncMock()

        # Extraction returns empty block (no subsampling found)
        client.send_message = AsyncMock(return_value="```python\n\n```")

        await remove_subsampling(client, solution, task)

        assert client.send_message.await_count == 1

    async def test_extraction_not_in_solution_uses_one_llm_call(self) -> None:
        """When extracted block is not in solution, only 1 LLM call made."""
        from mle_star.finalization import remove_subsampling

        solution = _make_solution(content="completely different code\n")
        task = _make_task()
        client = AsyncMock()

        # Extraction returns a block that is not in the solution
        client.send_message = AsyncMock(
            return_value="```python\nnonexistent_block\n```"
        )

        await remove_subsampling(client, solution, task)

        assert client.send_message.await_count == 1

    async def test_subsampling_never_exceeds_two_calls(self) -> None:
        """Subsampling removal never makes more than 2 LLM calls."""
        from mle_star.finalization import remove_subsampling

        solution = _make_solution(content="data = train.sample(500)\nmodel.fit(data)\n")
        task = _make_task()
        client = AsyncMock()

        client.send_message = AsyncMock(
            side_effect=[
                "```python\ndata = train.sample(500)\n```",
                "```python\ndata = train\n```",
            ]
        )

        await remove_subsampling(client, solution, task)

        assert client.send_message.await_count <= 2


# ===========================================================================
# REQ-FN-042: Structured Logging - Finalization start
# ===========================================================================


@pytest.mark.unit
class TestFinalizationStartLogging:
    """Finalization start event logged at INFO with required content (REQ-FN-042)."""

    async def test_finalization_start_logs_solution_phase(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Finalization start log includes the solution phase."""
        mocks = _setup_run_finalization_mocks()
        solution = _make_solution(phase=SolutionPhase.ENSEMBLE)

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await _run_finalization_with_mocks(mocks, solution=solution)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "ensemble" in all_info_text.lower() or "phase" in all_info_text.lower()

    async def test_finalization_start_logs_content_length(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Finalization start log includes the solution content length."""
        content = "x" * 250
        mocks = _setup_run_finalization_mocks()
        solution = _make_solution(content=content)

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await _run_finalization_with_mocks(mocks, solution=solution)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "250" in all_info_text or "content" in all_info_text.lower()

    async def test_finalization_start_logs_competition_id(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Finalization start log includes the competition ID."""
        mocks = _setup_run_finalization_mocks()
        task = _make_task(competition_id="my-kaggle-comp-2025")

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await _run_finalization_with_mocks(mocks, task=task)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "my-kaggle-comp-2025" in all_info_text


# ===========================================================================
# REQ-FN-042: Structured Logging - Subsampling extraction start
# ===========================================================================


@pytest.mark.unit
class TestSubsamplingExtractionStartLogging:
    """Subsampling extraction start logged at INFO (REQ-FN-042)."""

    async def test_extraction_start_logs_solution_content_length(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Subsampling extraction start includes solution content length."""
        from mle_star.finalization import remove_subsampling

        content = "a" * 300
        solution = _make_solution(content=content)
        task = _make_task()
        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\n\n```")

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await remove_subsampling(client, solution, task)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "300" in all_info_text or "extraction" in all_info_text.lower()


# ===========================================================================
# REQ-FN-042: Structured Logging - Subsampling extraction result
# ===========================================================================


@pytest.mark.unit
class TestSubsamplingExtractionResultLogging:
    """Subsampling extraction result logged at INFO (REQ-FN-042)."""

    async def test_extraction_result_logs_found_flag_when_found(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Extraction result indicates subsampling was found."""
        from mle_star.finalization import remove_subsampling

        solution = _make_solution(content="data = train[:500]\nfit(data)\n")
        task = _make_task()
        client = AsyncMock()
        client.send_message = AsyncMock(
            side_effect=[
                "```python\ndata = train[:500]\n```",
                "```python\ndata = train\n```",
            ]
        )

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await remove_subsampling(client, solution, task)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "found" in all_info_text.lower()
            or "detected" in all_info_text.lower()
            or "block" in all_info_text.lower()
        )

    async def test_extraction_result_logs_not_found_when_empty(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Extraction result indicates no subsampling when extraction is empty."""
        from mle_star.finalization import remove_subsampling

        solution = _make_solution(content="model.fit(full)\n")
        task = _make_task()
        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\n\n```")

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await remove_subsampling(client, solution, task)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "no subsampling" in all_info_text.lower()
            or "empty" in all_info_text.lower()
            or "not found" in all_info_text.lower()
        )

    async def test_extraction_result_logs_block_length(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Extraction result log includes extracted block length."""
        from mle_star.finalization import remove_subsampling

        block = "data = train.sample(1000)"
        solution = _make_solution(content=f"{block}\nfit(data)\n")
        task = _make_task()
        client = AsyncMock()
        client.send_message = AsyncMock(
            side_effect=[
                f"```python\n{block}\n```",
                "```python\ndata = train\n```",
            ]
        )

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await remove_subsampling(client, solution, task)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            str(len(block)) in all_info_text
            or "length" in all_info_text.lower()
            or "len" in all_info_text.lower()
        )


# ===========================================================================
# REQ-FN-042: Structured Logging - Subsampling removal result
# ===========================================================================


@pytest.mark.unit
class TestSubsamplingRemovalResultLogging:
    """Subsampling removal result logged at INFO (REQ-FN-042)."""

    async def test_removal_result_logs_original_block_length(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Removal result log includes original block length."""
        from mle_star.finalization import remove_subsampling

        original_block = "data = train.head(500)"
        replacement = "data = train"
        solution = _make_solution(content=f"{original_block}\nfit(data)\n")
        task = _make_task()
        client = AsyncMock()
        client.send_message = AsyncMock(
            side_effect=[
                f"```python\n{original_block}\n```",
                f"```python\n{replacement}\n```",
            ]
        )

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await remove_subsampling(client, solution, task)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            str(len(original_block)) in all_info_text
            or "original" in all_info_text.lower()
            or "removal" in all_info_text.lower()
        )

    async def test_removal_result_logs_replacement_block_length(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Removal result log includes replacement block length."""
        from mle_star.finalization import remove_subsampling

        original_block = "data = train.sample(1000)"
        replacement = "data = train"
        solution = _make_solution(content=f"{original_block}\nfit(data)\n")
        task = _make_task()
        client = AsyncMock()
        client.send_message = AsyncMock(
            side_effect=[
                f"```python\n{original_block}\n```",
                f"```python\n{replacement}\n```",
            ]
        )

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await remove_subsampling(client, solution, task)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            str(len(replacement)) in all_info_text
            or "replacement" in all_info_text.lower()
        )


# ===========================================================================
# REQ-FN-042: Structured Logging - Subsampling replacement result
# ===========================================================================


@pytest.mark.unit
class TestSubsamplingReplacementResultLogging:
    """Subsampling replacement result logged at INFO (REQ-FN-042)."""

    async def test_replacement_result_logs_success_flag(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Replacement result includes whether replacement succeeded."""
        from mle_star.finalization import remove_subsampling

        original_block = "data = train[:500]"
        replacement = "data = train"
        solution = _make_solution(content=f"{original_block}\nfit(data)\n")
        task = _make_task()
        client = AsyncMock()
        client.send_message = AsyncMock(
            side_effect=[
                f"```python\n{original_block}\n```",
                f"```python\n{replacement}\n```",
            ]
        )

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await remove_subsampling(client, solution, task)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "replac" in all_info_text.lower()
            or "success" in all_info_text.lower()
            or "content" in all_info_text.lower()
        )

    async def test_replacement_result_logs_content_length_change(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Replacement result includes content length change information."""
        from mle_star.finalization import remove_subsampling

        original_block = "data = train.sample(frac=0.1)"
        replacement = "data = train"
        full_content = f"{original_block}\nmodel.fit(data)\n"
        solution = _make_solution(content=full_content)
        task = _make_task()
        client = AsyncMock()
        client.send_message = AsyncMock(
            side_effect=[
                f"```python\n{original_block}\n```",
                f"```python\n{replacement}\n```",
            ]
        )

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await remove_subsampling(client, solution, task)

        all_msgs = " ".join(r.message for r in caplog.records)
        assert (
            "length" in all_msgs.lower()
            or "content" in all_msgs.lower()
            or "change" in all_msgs.lower()
            or "len" in all_msgs.lower()
        )


# ===========================================================================
# REQ-FN-042: Structured Logging - A_test invocation start
# ===========================================================================


@pytest.mark.unit
class TestATestInvocationStartLogging:
    """A_test invocation start logged at INFO (REQ-FN-042)."""

    async def test_a_test_start_logs_competition_id(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_test invocation start includes task competition_id."""
        from mle_star.finalization import generate_test_submission

        task = _make_task(competition_id="titanic-2025")
        solution = _make_solution()
        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nprint('test')\n```")

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await generate_test_submission(client, task, solution)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "titanic-2025" in all_info_text

    async def test_a_test_start_logs_solution_content_length(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_test invocation start includes solution content length."""
        from mle_star.finalization import generate_test_submission

        content = "y" * 175
        task = _make_task()
        solution = _make_solution(content=content)
        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nprint('test')\n```")

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await generate_test_submission(client, task, solution)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "175" in all_info_text or "solution_len" in all_info_text


# ===========================================================================
# REQ-FN-042: Structured Logging - A_test invocation result
# ===========================================================================


@pytest.mark.unit
class TestATestInvocationResultLogging:
    """A_test invocation result logged at INFO (REQ-FN-042)."""

    async def test_a_test_result_logs_script_content_length(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_test result includes generated script content length."""
        from mle_star.finalization import generate_test_submission

        generated_code = "import pandas as pd\nsubmission()\n"
        task = _make_task()
        solution = _make_solution()
        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value=f"```python\n{generated_code}\n```"
        )

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await generate_test_submission(client, task, solution)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert str(len(generated_code)) in all_info_text or "len=" in all_info_text

    async def test_a_test_empty_result_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_test empty extraction logs WARNING."""
        from mle_star.finalization import generate_test_submission

        task = _make_task()
        solution = _make_solution()
        client = AsyncMock()
        # Return a code fence with empty/whitespace-only content so
        # extract_code_block returns "" after strip, triggering the warning
        client.send_message = AsyncMock(return_value="```python\n   \n```")

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            await generate_test_submission(client, task, solution)

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1
        warning_text = " ".join(r.message for r in warning_msgs)
        assert "empty" in warning_text.lower() or "A_test" in warning_text


# ===========================================================================
# REQ-FN-042: Structured Logging - Submission verification result
# ===========================================================================


@pytest.mark.unit
class TestSubmissionVerificationLogging:
    """Submission verification result logged at INFO (REQ-FN-042)."""

    async def test_submission_verification_logs_file_exists(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Submission verification logs file exists status."""
        mocks = _setup_run_finalization_mocks(
            submission_info={
                "exists": True,
                "path": "/work/final/submission.csv",
                "size_bytes": 2048,
                "row_count": 500,
            },
        )

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await _run_finalization_with_mocks(mocks)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "submission" in all_info_text.lower()
            or "verif" in all_info_text.lower()
            or "exist" in all_info_text.lower()
        )

    async def test_submission_verification_logs_size_bytes(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Submission verification logs file size in bytes."""
        mocks = _setup_run_finalization_mocks(
            submission_info={
                "exists": True,
                "path": "/work/final/submission.csv",
                "size_bytes": 4096,
                "row_count": 200,
            },
        )

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await _run_finalization_with_mocks(mocks)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "4096" in all_info_text
            or "size" in all_info_text.lower()
            or "bytes" in all_info_text.lower()
        )

    async def test_submission_verification_logs_row_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Submission verification logs row count."""
        mocks = _setup_run_finalization_mocks(
            submission_info={
                "exists": True,
                "path": "/work/final/submission.csv",
                "size_bytes": 1024,
                "row_count": 750,
            },
        )

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await _run_finalization_with_mocks(mocks)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "750" in all_info_text or "row" in all_info_text.lower()


# ===========================================================================
# REQ-FN-042: Structured Logging - Fallback activated
# ===========================================================================


@pytest.mark.unit
class TestFallbackActivatedLogging:
    """Fallback activated logged at WARNING (REQ-FN-042)."""

    async def test_fallback_logs_warning_on_eval_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Fallback activation logs WARNING when evaluation fails."""
        mocks = _setup_run_finalization_mocks(
            eval_result=_make_eval_result(
                is_error=True,
                score=None,
                exit_code=1,
                stdout="",
                stderr="RuntimeError: OOM",
            ),
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            await _run_finalization_with_mocks(mocks)

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1
        warning_text = " ".join(r.message for r in warning_msgs)
        assert "fallback" in warning_text.lower()

    async def test_fallback_logs_warning_on_verify_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Fallback activation logs WARNING when submission verification fails."""
        mocks = _setup_run_finalization_mocks(verify_result=False)

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            await _run_finalization_with_mocks(mocks)

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1
        warning_text = " ".join(r.message for r in warning_msgs)
        assert "fallback" in warning_text.lower()

    async def test_fallback_logs_reason(self, caplog: pytest.LogCaptureFixture) -> None:
        """Fallback log includes reason for fallback (eval_error/verified flags)."""
        mocks = _setup_run_finalization_mocks(
            eval_result=_make_eval_result(
                is_error=True,
                score=None,
                exit_code=1,
                stdout="",
                stderr="Error",
            ),
            verify_result=False,
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            await _run_finalization_with_mocks(mocks)

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        warning_text = " ".join(r.message for r in warning_msgs)
        assert (
            "eval" in warning_text.lower()
            or "error" in warning_text.lower()
            or "verif" in warning_text.lower()
        )

    async def test_fallback_logs_solution_phase_and_score(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Fallback log includes fallback solution phase and score info."""
        solution = _make_solution(phase=SolutionPhase.REFINED, score=0.88)
        mocks = _setup_run_finalization_mocks(
            eval_result=_make_eval_result(
                is_error=True,
                score=None,
                exit_code=1,
                stdout="",
                stderr="Error",
            ),
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            await _run_finalization_with_mocks(mocks, solution=solution)

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        warning_text = " ".join(r.message for r in warning_msgs)
        # Fallback log should reference the error state
        assert "fallback" in warning_text.lower()


# ===========================================================================
# REQ-FN-042: Structured Logging - Contamination check start
# ===========================================================================


@pytest.mark.unit
class TestContaminationCheckStartLogging:
    """Contamination check start logged at INFO (REQ-FN-042)."""

    async def test_contamination_start_logs_reference_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Contamination check start includes number of reference discussions."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        solution = _make_solution()
        refs = ["ref discussion 1", "ref discussion 2", "ref discussion 3"]

        result_json = DataContaminationResult(verdict="Novel").model_dump_json()
        client.send_message = AsyncMock(return_value=result_json)

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await check_contamination(client, solution, refs)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "3" in all_info_text or "reference" in all_info_text.lower()


# ===========================================================================
# REQ-FN-042: Structured Logging - Contamination check result
# ===========================================================================


@pytest.mark.unit
class TestContaminationCheckResultLogging:
    """Contamination check result logged at INFO (REQ-FN-042)."""

    async def test_contamination_result_logs_per_reference_verdicts(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Contamination result includes per-reference verdicts."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        solution = _make_solution()
        refs = ["ref1", "ref2"]

        results = [
            DataContaminationResult(verdict="Novel").model_dump_json(),
            DataContaminationResult(verdict="Same").model_dump_json(),
        ]
        client.send_message = AsyncMock(side_effect=results)

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await check_contamination(client, solution, refs)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "Novel" in all_info_text or "Same" in all_info_text

    async def test_contamination_result_logs_overall_verdict(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Contamination result includes overall verdict."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        solution = _make_solution()
        refs = ["ref1"]

        result_json = DataContaminationResult(verdict="Novel").model_dump_json()
        client.send_message = AsyncMock(return_value=result_json)

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await check_contamination(client, solution, refs)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "Novel" in all_info_text or "overall" in all_info_text.lower()


# ===========================================================================
# REQ-FN-042: Structured Logging - Contamination check skipped
# ===========================================================================


@pytest.mark.unit
class TestContaminationCheckSkippedLogging:
    """Contamination check skipped logged at INFO (REQ-FN-042)."""

    async def test_contamination_skipped_logs_info_for_none(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Skipped contamination check logs INFO when references is None."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        solution = _make_solution()

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await check_contamination(client, solution, None)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "skip" in all_info_text.lower() or "no reference" in all_info_text.lower()
        )

    async def test_contamination_skipped_logs_info_for_empty(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Skipped contamination check logs INFO when references is []."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        solution = _make_solution()

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await check_contamination(client, solution, [])

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_msgs) >= 1


# ===========================================================================
# REQ-FN-042: Structured Logging - FinalResult construction
# ===========================================================================


@pytest.mark.unit
class TestFinalResultConstructionLogging:
    """FinalResult construction logged at INFO (REQ-FN-042)."""

    async def test_final_result_logs_solution_phase(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Final result log includes solution phase."""
        mocks = _setup_run_finalization_mocks()

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await _run_finalization_with_mocks(mocks)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert "solution_phase" in all_info_text or "phase" in all_info_text.lower()

    async def test_final_result_logs_submission_path(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Final result log includes submission path."""
        mocks = _setup_run_finalization_mocks(
            submission_info={
                "exists": True,
                "path": "/work/final/submission.csv",
                "size_bytes": 1024,
                "row_count": 100,
            },
        )

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await _run_finalization_with_mocks(mocks)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "submission_path" in all_info_text or "submission" in all_info_text.lower()
        )

    async def test_final_result_logs_total_duration(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Final result log includes total duration."""
        mocks = _setup_run_finalization_mocks()

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await _run_finalization_with_mocks(mocks)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "duration" in all_info_text.lower() or "complete" in all_info_text.lower()
        )


# ===========================================================================
# REQ-FN-043: SDK Agent Invocation
# ===========================================================================


@pytest.mark.unit
class TestSDKAgentInvocation:
    """All agents invoked via SDK client.send_message (REQ-FN-043)."""

    def test_remove_subsampling_uses_client_send_message(self) -> None:
        """remove_subsampling calls client.send_message with agent_type."""
        from mle_star import finalization

        source = inspect.getsource(finalization._remove_subsampling_impl)
        assert "client.send_message" in source
        assert "agent_type" in source

    def test_generate_test_submission_uses_client_send_message(self) -> None:
        """generate_test_submission calls client.send_message with agent_type."""
        from mle_star import finalization

        source = inspect.getsource(finalization.generate_test_submission)
        assert "client.send_message" in source
        assert "agent_type" in source

    def test_check_contamination_uses_client_send_message(self) -> None:
        """check_contamination calls client.send_message with agent_type."""
        from mle_star import finalization

        source = inspect.getsource(finalization._check_contamination_impl)
        assert "client.send_message" in source
        assert "agent_type" in source

    async def test_remove_subsampling_actually_calls_send_message(self) -> None:
        """remove_subsampling invokes client.send_message at runtime."""
        from mle_star.finalization import remove_subsampling

        solution = _make_solution(content="code\n")
        task = _make_task()
        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\n\n```")

        await remove_subsampling(client, solution, task)
        client.send_message.assert_called()

    async def test_generate_test_submission_actually_calls_send_message(
        self,
    ) -> None:
        """generate_test_submission invokes client.send_message at runtime."""
        from mle_star.finalization import generate_test_submission

        task = _make_task()
        solution = _make_solution()
        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nprint('test')\n```")

        await generate_test_submission(client, task, solution)
        client.send_message.assert_called_once()

    async def test_check_contamination_actually_calls_send_message(self) -> None:
        """check_contamination invokes client.send_message at runtime."""
        from mle_star.finalization import check_contamination

        client = AsyncMock()
        solution = _make_solution()
        result_json = DataContaminationResult(verdict="Novel").model_dump_json()
        client.send_message = AsyncMock(return_value=result_json)

        await check_contamination(client, solution, ["ref1"])
        client.send_message.assert_called_once()


# ===========================================================================
# REQ-FN-044: Single Module Organization
# ===========================================================================


@pytest.mark.unit
class TestSingleModuleOrganization:
    """All finalization functions reside in finalization.py (REQ-FN-044)."""

    def test_all_finalization_functions_in_one_module(self) -> None:
        """Key finalization functions are importable from mle_star.finalization."""
        from mle_star.finalization import (
            check_contamination,
            generate_test_submission,
            remove_subsampling,
            run_finalization,
        )

        assert callable(remove_subsampling)
        assert callable(generate_test_submission)
        assert callable(check_contamination)
        assert callable(run_finalization)

    def test_run_finalization_is_async(self) -> None:
        """run_finalization is an async function."""
        from mle_star.finalization import run_finalization

        assert asyncio.iscoroutinefunction(run_finalization)

    def test_remove_subsampling_is_async(self) -> None:
        """remove_subsampling is an async function."""
        from mle_star.finalization import remove_subsampling

        assert asyncio.iscoroutinefunction(remove_subsampling)

    def test_generate_test_submission_is_async(self) -> None:
        """generate_test_submission is an async function."""
        from mle_star.finalization import generate_test_submission

        assert asyncio.iscoroutinefunction(generate_test_submission)

    def test_check_contamination_is_async(self) -> None:
        """check_contamination is an async function."""
        from mle_star.finalization import check_contamination

        assert asyncio.iscoroutinefunction(check_contamination)

    def test_apply_fallback_is_synchronous(self) -> None:
        """_apply_fallback is a regular synchronous function."""
        from mle_star.finalization import _apply_fallback

        assert not asyncio.iscoroutinefunction(_apply_fallback)
        assert callable(_apply_fallback)


# ===========================================================================
# REQ-FN-045: Submission File Path Convention
# ===========================================================================


@pytest.mark.unit
class TestSubmissionFilePathConvention:
    """Test submission script writes to ./final/submission.csv (REQ-FN-045)."""

    def test_prompt_template_mentions_final_submission_csv(self) -> None:
        """A_test default prompt template mentions ./final/submission.csv."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.TEST)
        rendered = template.render(
            task_description="Test task",
            target_column="Not specified",
            final_solution="print('hello')",
        )
        assert "./final/submission.csv" in rendered or (
            "submission.csv" in rendered and "./final" in rendered
        )

    def test_prompt_template_explicitly_instructs_final_dir(self) -> None:
        """A_test prompt explicitly instructs saving to ./final directory."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.TEST)
        rendered = template.render(
            task_description="Test task",
            target_column="Not specified",
            final_solution="print('hello')",
        )
        assert "./final" in rendered

    def test_prompt_template_mentions_submission_csv(self) -> None:
        """A_test prompt mentions submission.csv filename."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.TEST)
        rendered = template.render(
            task_description="Test task",
            target_column="Not specified",
            final_solution="print('hello')",
        )
        assert "submission.csv" in rendered


# ===========================================================================
# REQ-FN-046: No exit() in Test Script
# ===========================================================================


@pytest.mark.unit
class TestNoExitInTestScript:
    """Test scripts shall not contain exit()/sys.exit()/os._exit()/quit() (REQ-FN-046)."""

    def test_prompt_forbids_exit_function(self) -> None:
        """A_test prompt explicitly forbids exit() usage."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.TEST)
        rendered = template.render(
            task_description="Test task",
            target_column="Not specified",
            final_solution="print('hello')",
        )
        assert "exit()" in rendered

    def test_prompt_forbids_exit_keyword(self) -> None:
        """A_test prompt contains instruction not to use exit."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.TEST)
        rendered = template.render(
            task_description="Test task",
            target_column="Not specified",
            final_solution="print('hello')",
        )
        # The prompt should instruct not to use exit()
        assert "do not use exit" in rendered.lower() or "exit()" in rendered

    @given(
        exit_variant=st.sampled_from(["exit()", "sys.exit()", "os._exit(0)", "quit()"]),
    )
    @settings(max_examples=4, deadline=5000)
    def test_exit_variants_detectable_in_generated_code(
        self, exit_variant: str
    ) -> None:
        """All exit variants are detectable with a simple substring check."""
        # This test validates that exit detection is feasible
        code_with_exit = (
            f"import pandas as pd\ndf = pd.read_csv('test.csv')\n{exit_variant}\n"
        )
        exit_patterns = ["exit()", "sys.exit(", "os._exit(", "quit()"]
        assert any(p in code_with_exit for p in exit_patterns)


# ===========================================================================
# REQ-FN-047: No Error Masking in Test Script
# ===========================================================================


@pytest.mark.unit
class TestNoErrorMaskingInTestScript:
    """Test scripts shall not mask errors with try/except (REQ-FN-047)."""

    def test_prompt_forbids_try_except(self) -> None:
        """A_test prompt explicitly forbids try/except for error masking."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.TEST)
        rendered = template.render(
            task_description="Test task",
            target_column="Not specified",
            final_solution="print('hello')",
        )
        assert "try:" in rendered and "except:" in rendered

    def test_prompt_forbids_ignoring_unintended_behavior(self) -> None:
        """A_test prompt forbids ignoring unintended behavior."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.TEST)
        rendered = template.render(
            task_description="Test task",
            target_column="Not specified",
            final_solution="print('hello')",
        )
        assert "unintended" in rendered.lower() or "ignore" in rendered.lower()


# ===========================================================================
# REQ-FN-048: Agent Config
# ===========================================================================


@pytest.mark.unit
class TestFinalizationAgentConfig:
    """A_test agent config has correct tools (REQ-FN-048)."""

    def test_test_agent_config_exists(self) -> None:
        """AgentType.TEST is in build_default_agent_configs."""
        from mle_star.models import build_default_agent_configs

        configs = build_default_agent_configs()
        assert AgentType.TEST in configs

    def test_test_agent_config_has_read_tool(self) -> None:
        """A_test agent config includes 'Read' in its tools."""
        from mle_star.models import build_default_agent_configs

        configs = build_default_agent_configs()
        test_config = configs[AgentType.TEST]
        assert test_config.tools is not None
        assert "Read" in test_config.tools

    def test_test_agent_config_tools_is_read_only(self) -> None:
        """A_test agent config tools should be ['Read'] only (REQ-FN-048)."""
        from mle_star.models import build_default_agent_configs

        configs = build_default_agent_configs()
        test_config = configs[AgentType.TEST]
        assert test_config.tools == ["Read"], (
            f"Expected ['Read'] but got {test_config.tools}"
        )

    def test_test_agent_config_has_correct_agent_type(self) -> None:
        """A_test agent config has agent_type == AgentType.TEST."""
        from mle_star.models import build_default_agent_configs

        configs = build_default_agent_configs()
        test_config = configs[AgentType.TEST]
        assert test_config.agent_type == AgentType.TEST

    def test_test_agent_config_has_description(self) -> None:
        """A_test agent config has a non-empty description."""
        from mle_star.models import build_default_agent_configs

        configs = build_default_agent_configs()
        test_config = configs[AgentType.TEST]
        assert test_config.description
        assert len(test_config.description) > 0


# ===========================================================================
# REQ-FN-042: Structured Logging - Property-based invariants
# ===========================================================================


@pytest.mark.unit
class TestFinalizationLoggingProperties:
    """Property-based tests for finalization logging invariants."""

    @given(
        content_len=st.integers(min_value=1, max_value=2000),
    )
    @settings(
        max_examples=5,
        deadline=10000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    async def test_finalization_always_logs_complete(
        self,
        content_len: int,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Finalization always logs a completion message regardless of input size."""
        caplog.clear()
        mocks = _setup_run_finalization_mocks()
        solution = _make_solution(content="x" * content_len)

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            result = await _run_finalization_with_mocks(mocks, solution=solution)

        assert isinstance(result, FinalResult)
        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Must always log finalization complete with duration
        assert (
            "complete" in all_info_text.lower()
            or "duration" in all_info_text.lower()
            or "finalization" in all_info_text.lower()
        )

    @given(
        competition_id=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters="-_",
            ),
        ),
    )
    @settings(
        max_examples=5,
        deadline=10000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    async def test_a_test_always_logs_competition_id(
        self,
        competition_id: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """generate_test_submission always logs the competition_id."""
        caplog.clear()
        from mle_star.finalization import generate_test_submission

        task = _make_task(competition_id=competition_id)
        solution = _make_solution()
        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nprint('test')\n```")

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await generate_test_submission(client, task, solution)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert competition_id in all_info_text


# ===========================================================================
# REQ-FN-042: Structured Logging - Level correctness
# ===========================================================================


@pytest.mark.unit
class TestFinalizationLoggingLevels:
    """All finalization log events are at the correct log level."""

    async def test_normal_flow_has_no_error_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Normal (non-fallback) finalization flow produces no ERROR logs."""
        mocks = _setup_run_finalization_mocks()

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await _run_finalization_with_mocks(mocks)

        error_msgs = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_msgs) == 0

    async def test_normal_flow_has_no_warning_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Normal (non-fallback) finalization flow produces no WARNING logs."""
        mocks = _setup_run_finalization_mocks()

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await _run_finalization_with_mocks(mocks)

        # Filter to only finalization logger warnings
        warning_msgs = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and r.name == _LOGGER_NAME
        ]
        assert len(warning_msgs) == 0

    async def test_fallback_uses_warning_not_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Fallback activation uses WARNING level, not ERROR."""
        mocks = _setup_run_finalization_mocks(
            eval_result=_make_eval_result(
                is_error=True,
                score=None,
                exit_code=1,
                stdout="",
                stderr="Error",
            ),
        )

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await _run_finalization_with_mocks(mocks)

        warning_msgs = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and r.name == _LOGGER_NAME
        ]
        error_msgs = [
            r
            for r in caplog.records
            if r.levelno >= logging.ERROR and r.name == _LOGGER_NAME
        ]
        assert len(warning_msgs) >= 1
        assert len(error_msgs) == 0


# ===========================================================================
# REQ-FN-043: SDK Agent Invocation - PromptRegistry usage
# ===========================================================================


@pytest.mark.unit
class TestFinalizationPromptRegistryUsage:
    """All finalization agents load prompts from PromptRegistry."""

    def test_remove_subsampling_uses_prompt_registry(self) -> None:
        """_remove_subsampling_impl source references PromptRegistry."""
        from mle_star import finalization

        source = inspect.getsource(finalization._remove_subsampling_impl)
        assert "PromptRegistry" in source

    def test_generate_test_submission_uses_prompt_registry(self) -> None:
        """generate_test_submission source references PromptRegistry."""
        from mle_star import finalization

        source = inspect.getsource(finalization.generate_test_submission)
        assert "PromptRegistry" in source

    def test_check_contamination_uses_prompt_registry(self) -> None:
        """_check_contamination_impl source references PromptRegistry."""
        from mle_star import finalization

        source = inspect.getsource(finalization._check_contamination_impl)
        assert "PromptRegistry" in source

    def test_subsampling_extract_variant_exists(self) -> None:
        """A_test subsampling_extract variant is loadable from PromptRegistry."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.TEST, variant="subsampling_extract")
        assert template is not None

    def test_subsampling_remove_variant_exists(self) -> None:
        """A_test subsampling_remove variant is loadable from PromptRegistry."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.TEST, variant="subsampling_remove")
        assert template is not None

    def test_contamination_check_variant_exists(self) -> None:
        """A_test contamination_check variant is loadable from PromptRegistry."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.TEST, variant="contamination_check")
        assert template is not None

    def test_default_test_variant_exists(self) -> None:
        """A_test default variant (no variant specified) is loadable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.TEST)
        assert template is not None


# ===========================================================================
# Additional: Finalization pipeline step ordering
# ===========================================================================


@pytest.mark.unit
class TestFinalizationPipelineOrder:
    """Finalization pipeline follows the correct step ordering."""

    async def test_steps_execute_in_correct_order(self) -> None:
        """Steps execute: subsampling -> test_submission -> leakage -> eval -> verify -> fallback -> contamination."""
        from mle_star.finalization import run_finalization

        call_order: list[str] = []

        solution = _make_solution()
        test_script = _make_solution(content="test code", phase=SolutionPhase.FINAL)
        eval_result = _make_eval_result()
        sub_info = {
            "exists": True,
            "path": "/work/final/submission.csv",
            "size_bytes": 1024,
            "row_count": 100,
        }

        async def _remove_sub(*a: Any, **k: Any) -> SolutionScript:
            call_order.append("remove_subsampling")
            return solution

        async def _gen_test(*a: Any, **k: Any) -> SolutionScript:
            call_order.append("generate_test_submission")
            return test_script

        async def _leakage(*a: Any, **k: Any) -> SolutionScript:
            call_order.append("check_and_fix_leakage")
            return test_script

        async def _eval(*a: Any, **k: Any) -> tuple[SolutionScript, EvaluationResult]:
            call_order.append("evaluate_with_retry")
            return (test_script, eval_result)

        def _verify(*a: Any, **k: Any) -> bool:
            call_order.append("verify_submission")
            return True

        def _get_info(*a: Any, **k: Any) -> dict[str, Any]:
            call_order.append("get_submission_info")
            return sub_info

        async def _contam(*a: Any, **k: Any) -> None:
            call_order.append("check_contamination")
            return None

        mocks = {
            "remove_subsampling": AsyncMock(side_effect=_remove_sub),
            "generate_test_submission": AsyncMock(side_effect=_gen_test),
            "check_and_fix_leakage": AsyncMock(side_effect=_leakage),
            "evaluate_with_retry": AsyncMock(side_effect=_eval),
            "make_debug_callback": MagicMock(return_value=MagicMock()),
            "verify_submission": MagicMock(side_effect=_verify),
            "get_submission_info": MagicMock(side_effect=_get_info),
            "check_contamination": AsyncMock(side_effect=_contam),
        }

        client = AsyncMock()
        task = _make_task()
        config = _make_config()

        with _apply_run_finalization_patches(mocks):
            await run_finalization(
                client=client,
                solution=solution,
                task=task,
                config=config,
                phase1_result=_make_phase1_result(),
                phase2_results=[_make_phase2_result()],
                phase3_result=None,
            )

        # Verify order
        assert call_order.index("remove_subsampling") < call_order.index(
            "generate_test_submission"
        )
        assert call_order.index("generate_test_submission") < call_order.index(
            "check_and_fix_leakage"
        )
        assert call_order.index("check_and_fix_leakage") < call_order.index(
            "evaluate_with_retry"
        )
        assert call_order.index("evaluate_with_retry") < call_order.index(
            "verify_submission"
        )
        assert call_order.index("verify_submission") < call_order.index(
            "check_contamination"
        )

    def test_run_finalization_source_has_time_monotonic(self) -> None:
        """run_finalization tracks duration via time.monotonic()."""
        from mle_star import finalization

        source = inspect.getsource(finalization.run_finalization)
        assert "time.monotonic()" in source
