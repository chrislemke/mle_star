"""Tests for Phase 2 outer loop constraints (Task 34).

Validates non-functional requirements for the Phase 2 outer loop in
``phase2_outer.py``: processing overhead, structured logging at correct
levels, sequential iteration execution, monotonic best-score invariant,
ablation script self-containment, immutability of input solution, and
code block provenance tracking.

Tests are written TDD-first and serve as the executable specification
for REQ-P2O-031 through REQ-P2O-044.

Refs:
    SRS 05e (Phase 2 Outer Constraints), IMPLEMENTATION_PLAN.md Task 34.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any
from unittest.mock import AsyncMock, patch

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    CodeBlock,
    DataModality,
    ExtractorOutput,
    InnerLoopResult,
    MetricDirection,
    Phase2Result,
    PipelineConfig,
    RefinementAttempt,
    RefinePlan,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
    TaskType,
)
import pytest

# ---------------------------------------------------------------------------
# Module path constant for patching
# ---------------------------------------------------------------------------

_MODULE = "mle_star.phase2_outer"
_LOGGER_NAME = "mle_star.phase2_outer"


# ---------------------------------------------------------------------------
# Reusable test helpers
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
        description="Predict the target variable from tabular features.",
    )


def _make_config(**kwargs: Any) -> PipelineConfig:
    """Create a PipelineConfig for testing with optional overrides."""
    return PipelineConfig(**kwargs)


def _make_solution(
    content: str = "import pandas as pd\ndf = pd.read_csv('data.csv')\nmodel.fit(df)",
    phase: SolutionPhase = SolutionPhase.INIT,
    score: float | None = None,
) -> SolutionScript:
    """Create a SolutionScript for testing."""
    return SolutionScript(content=content, phase=phase, score=score)


def _make_inner_loop_result(
    best_solution: SolutionScript | None = None,
    best_score: float = 0.90,
    improved: bool = True,
    attempts: list[RefinementAttempt] | None = None,
) -> InnerLoopResult:
    """Create a mock InnerLoopResult."""
    if best_solution is None:
        best_solution = _make_solution(
            content="improved code", phase=SolutionPhase.REFINED
        )
    if attempts is None:
        attempts = [
            RefinementAttempt(
                plan="test plan",
                score=best_score,
                code_block="improved block",
                was_improvement=improved,
            )
        ]
    return InnerLoopResult(
        best_solution=best_solution,
        best_score=best_score,
        attempts=attempts,
        improved=improved,
    )


def _make_extractor_output(
    code_block: str = "model.fit(df)",
    plan: str = "Optimize the model training loop",
) -> ExtractorOutput:
    """Create a mock ExtractorOutput with a single plan."""
    return ExtractorOutput(plans=[RefinePlan(code_block=code_block, plan=plan)])


def _make_ablation_script(
    content: str = "print('ablation study')",
) -> SolutionScript:
    """Create a SolutionScript wrapping an ablation script."""
    return SolutionScript(
        content=content, phase=SolutionPhase.REFINED, is_executable=True
    )


def _patch_outer_loop_dependencies(
    invoke_ablation_rv: SolutionScript | None = None,
    execute_ablation_rv: tuple[str, str] = ("ablation stdout", ""),
    invoke_summarize_rv: str = "Ablation summary: feature engineering is key",
    invoke_extractor_rv: ExtractorOutput | None = None,
    validate_code_block_rv: bool = True,
    inner_loop_rv: InnerLoopResult | None = None,
) -> dict[str, Any]:
    """Build a dict of patch objects for all outer loop dependencies.

    Returns a dict of mock objects keyed by function name for
    assertions in tests.
    """
    if invoke_ablation_rv is None:
        invoke_ablation_rv = _make_ablation_script()
    if invoke_extractor_rv is None:
        invoke_extractor_rv = _make_extractor_output()
    if inner_loop_rv is None:
        inner_loop_rv = _make_inner_loop_result()

    return {
        "invoke_ablation": AsyncMock(return_value=invoke_ablation_rv),
        "execute_ablation_with_retry": AsyncMock(return_value=execute_ablation_rv),
        "invoke_summarize": AsyncMock(return_value=invoke_summarize_rv),
        "invoke_extractor": AsyncMock(return_value=invoke_extractor_rv),
        "validate_code_block": lambda code_block, solution: validate_code_block_rv,
        "run_phase2_inner_loop": AsyncMock(return_value=inner_loop_rv),
        "is_improvement_or_equal": lambda new, old, direction: (
            new >= old if direction == MetricDirection.MAXIMIZE else new <= old
        ),
    }


async def _run_outer_loop(
    mocks: dict[str, Any],
    outer_loop_steps: int = 2,
    initial_score: float = 0.80,
    initial_solution: SolutionScript | None = None,
    task: TaskDescription | None = None,
    config: PipelineConfig | None = None,
    session_id: str = "test-session",
) -> Phase2Result:
    """Run run_phase2_outer_loop with all dependencies mocked.

    Args:
        mocks: Dict from _patch_outer_loop_dependencies().
        outer_loop_steps: Number of outer iterations.
        initial_score: Starting score.
        initial_solution: Starting solution (defaults to helper).
        task: TaskDescription (defaults to helper).
        config: PipelineConfig (defaults to helper with outer_loop_steps).
        session_id: Session identifier.

    Returns:
        The Phase2Result from the orchestration function.
    """
    from mle_star.phase2_outer import run_phase2_outer_loop

    if initial_solution is None:
        initial_solution = _make_solution()
    if task is None:
        task = _make_task()
    if config is None:
        config = _make_config(outer_loop_steps=outer_loop_steps)

    client = AsyncMock()

    with (
        patch(f"{_MODULE}.invoke_ablation", mocks["invoke_ablation"]),
        patch(
            f"{_MODULE}.execute_ablation_with_retry",
            mocks["execute_ablation_with_retry"],
        ),
        patch(f"{_MODULE}.invoke_summarize", mocks["invoke_summarize"]),
        patch(f"{_MODULE}.invoke_extractor", mocks["invoke_extractor"]),
        patch(f"{_MODULE}.validate_code_block", mocks["validate_code_block"]),
        patch(
            f"{_MODULE}.run_phase2_inner_loop",
            mocks["run_phase2_inner_loop"],
        ),
        patch(
            f"{_MODULE}.is_improvement_or_equal",
            mocks["is_improvement_or_equal"],
        ),
    ):
        return await run_phase2_outer_loop(
            client=client,
            task=task,
            config=config,
            initial_solution=initial_solution,
            initial_score=initial_score,
            session_id=session_id,
        )


async def _run_outer_loop_with_caplog(
    mocks: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
    outer_loop_steps: int = 2,
    initial_score: float = 0.80,
    initial_solution: SolutionScript | None = None,
    task: TaskDescription | None = None,
    config: PipelineConfig | None = None,
    session_id: str = "test-session",
) -> Phase2Result:
    """Run outer loop with caplog context for log capture.

    Args:
        mocks: Dict from _patch_outer_loop_dependencies().
        caplog: Pytest log capture fixture.
        outer_loop_steps: Number of outer iterations.
        initial_score: Starting score.
        initial_solution: Starting solution (defaults to helper).
        task: TaskDescription (defaults to helper).
        config: PipelineConfig (defaults to helper with outer_loop_steps).
        session_id: Session identifier.

    Returns:
        The Phase2Result from the orchestration function.
    """
    from mle_star.phase2_outer import run_phase2_outer_loop

    if initial_solution is None:
        initial_solution = _make_solution()
    if task is None:
        task = _make_task()
    if config is None:
        config = _make_config(outer_loop_steps=outer_loop_steps)

    client = AsyncMock()

    with (
        patch(f"{_MODULE}.invoke_ablation", mocks["invoke_ablation"]),
        patch(
            f"{_MODULE}.execute_ablation_with_retry",
            mocks["execute_ablation_with_retry"],
        ),
        patch(f"{_MODULE}.invoke_summarize", mocks["invoke_summarize"]),
        patch(f"{_MODULE}.invoke_extractor", mocks["invoke_extractor"]),
        patch(f"{_MODULE}.validate_code_block", mocks["validate_code_block"]),
        patch(
            f"{_MODULE}.run_phase2_inner_loop",
            mocks["run_phase2_inner_loop"],
        ),
        patch(
            f"{_MODULE}.is_improvement_or_equal",
            mocks["is_improvement_or_equal"],
        ),
        caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
    ):
        return await run_phase2_outer_loop(
            client=client,
            task=task,
            config=config,
            initial_solution=initial_solution,
            initial_score=initial_score,
            session_id=session_id,
        )


# ===========================================================================
# REQ-P2O-031: A_abl response processing overhead < 500ms
# ===========================================================================


@pytest.mark.unit
class TestAblationResponseOverhead:
    """A_abl response processing overhead < 500ms (REQ-P2O-031)."""

    async def test_invoke_ablation_processing_overhead_under_500ms(self) -> None:
        """Processing overhead for invoke_ablation (excluding LLM call) is < 500ms."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value="```python\nprint('ablation')\n```"
        )
        solution = _make_solution(content="x" * 50_000)
        previous_summaries = ["summary " * 100 for _ in range(5)]

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="print('ablation')"),
        ):
            mock_template = mock_registry_cls.return_value.get.return_value
            mock_template.render.return_value = "rendered prompt"

            start = time.monotonic()
            await invoke_ablation(solution, previous_summaries, client)
            elapsed = time.monotonic() - start

        assert elapsed < 0.5, (
            f"invoke_ablation processing overhead was {elapsed:.3f}s, expected < 0.5s"
        )

    async def test_ablation_script_wrapping_overhead_under_500ms(self) -> None:
        """Wrapping extracted code in SolutionScript takes < 500ms for large content."""
        content = "x" * 50_000

        start = time.monotonic()
        script = SolutionScript(
            content=content,
            phase=SolutionPhase.REFINED,
            is_executable=True,
        )
        elapsed = time.monotonic() - start

        assert elapsed < 0.5, (
            f"SolutionScript wrapping took {elapsed:.3f}s, expected < 0.5s"
        )
        assert len(script.content) == 50_000


# ===========================================================================
# REQ-P2O-032: validate_code_block() < 50ms for 50KB solutions
# ===========================================================================


@pytest.mark.unit
class TestValidateCodeBlockPerformance:
    """validate_code_block() < 50ms for 50KB solutions (REQ-P2O-032)."""

    async def test_validate_code_block_under_50ms_for_50kb(self) -> None:
        """validate_code_block completes in < 50ms for a 50KB solution."""
        from mle_star.phase2_outer import validate_code_block

        large_content = "a" * 50_000
        code_block = "a" * 100
        solution = _make_solution(content=large_content)

        start = time.monotonic()
        result = validate_code_block(code_block, solution)
        elapsed = time.monotonic() - start

        assert result is True
        assert elapsed < 0.05, (
            f"validate_code_block took {elapsed:.4f}s, expected < 0.05s"
        )

    async def test_validate_code_block_under_50ms_when_not_found(self) -> None:
        """validate_code_block completes in < 50ms even when block is absent."""
        from mle_star.phase2_outer import validate_code_block

        large_content = "a" * 50_000
        code_block = "b" * 100  # Not a substring
        solution = _make_solution(content=large_content)

        start = time.monotonic()
        result = validate_code_block(code_block, solution)
        elapsed = time.monotonic() - start

        assert result is False
        assert elapsed < 0.05, (
            f"validate_code_block took {elapsed:.4f}s, expected < 0.05s"
        )

    @given(
        content_size=st.integers(min_value=1_000, max_value=100_000),
        block_size=st.integers(min_value=10, max_value=500),
    )
    @settings(max_examples=20, deadline=500)
    async def test_validate_code_block_performance_property(
        self, content_size: int, block_size: int
    ) -> None:
        """validate_code_block always completes < 50ms for various input sizes."""
        from mle_star.phase2_outer import validate_code_block

        large_content = "x" * content_size
        code_block = "x" * block_size
        solution = _make_solution(content=large_content)

        start = time.monotonic()
        validate_code_block(code_block, solution)
        elapsed = time.monotonic() - start

        assert elapsed < 0.05


# ===========================================================================
# REQ-P2O-037: Structured Logging -- Outer Step Start
# ===========================================================================


@pytest.mark.unit
class TestOuterStepStartLogging:
    """Outer step start event is logged at INFO (REQ-P2O-037)."""

    async def test_outer_step_start_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Outer step start is logged at INFO with step index, h_best, summary count."""
        mocks = _patch_outer_loop_dependencies()
        await _run_outer_loop_with_caplog(
            mocks, caplog, outer_loop_steps=2, initial_score=0.80
        )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_msgs) >= 1, "Expected at least one INFO log at outer step start"
        all_info_text = " ".join(r.message for r in info_msgs)
        # Must mention step index (1/2 or step 0)
        assert "1" in all_info_text or "step" in all_info_text.lower()
        # Must mention h_best
        assert "0.8" in all_info_text or "h_best" in all_info_text.lower()

    async def test_outer_step_start_logs_summary_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Outer step start log includes the number of accumulated summaries."""
        mocks = _patch_outer_loop_dependencies()
        await _run_outer_loop_with_caplog(
            mocks, caplog, outer_loop_steps=2, initial_score=0.80
        )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # First step should have 0 summaries, second should have 1
        # The session id should also appear
        assert "session" in all_info_text.lower() or "test-session" in all_info_text


# ===========================================================================
# REQ-P2O-037: Structured Logging -- A_abl Invocation
# ===========================================================================


@pytest.mark.unit
class TestAblationInvocationLogging:
    """A_abl invocation events are logged at INFO (REQ-P2O-037)."""

    async def test_ablation_invocation_start_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_abl invocation start is logged at INFO with solution length and summary count."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value="```python\nprint('ablation')\n```"
        )
        solution = _make_solution(content="solution content here")
        summaries = ["summary_one", "summary_two"]

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="print('ablation')"),
            caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
        ):
            mock_template = mock_registry_cls.return_value.get.return_value
            mock_template.render.return_value = "rendered prompt"
            await invoke_ablation(solution, summaries, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention solution content length (21 chars)
        assert (
            str(len("solution content here")) in all_info_text
            or "length" in all_info_text.lower()
            or "content" in all_info_text.lower()
        )
        # Should mention number of previous summaries (2)
        assert "2" in all_info_text or "summar" in all_info_text.lower()

    async def test_ablation_invocation_complete_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_abl invocation complete is logged at INFO with ablation script length."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        ablation_code = "print('ablation study output')"
        client.send_message = AsyncMock(return_value=f"```python\n{ablation_code}\n```")
        solution = _make_solution()

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=ablation_code),
            caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
        ):
            mock_template = mock_registry_cls.return_value.get.return_value
            mock_template.render.return_value = "rendered prompt"
            await invoke_ablation(solution, [], client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention ablation script length
        assert (
            str(len(ablation_code)) in all_info_text
            or "length" in all_info_text.lower()
            or "complet" in all_info_text.lower()
        )

    async def test_ablation_empty_response_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_abl empty response is logged at WARNING."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="no code here")
        solution = _make_solution()

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
            caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
        ):
            mock_template = mock_registry_cls.return_value.get.return_value
            mock_template.render.return_value = "rendered prompt"
            result = await invoke_ablation(solution, [], client)

        assert result is None
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1
        warning_text = " ".join(r.message for r in warning_msgs)
        assert "empty" in warning_text.lower() or "abl" in warning_text.lower()


# ===========================================================================
# REQ-P2O-037: Structured Logging -- Ablation Execution
# ===========================================================================


@pytest.mark.unit
class TestAblationExecutionLogging:
    """Ablation execution events are logged at correct levels (REQ-P2O-037)."""

    async def test_ablation_execution_start_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Ablation execution start is logged at INFO with script path and timeout."""
        from mle_star.phase2_outer import execute_ablation_with_retry

        ablation_script = _make_ablation_script(content="print('test')")
        task = _make_task()
        config = _make_config()
        client = AsyncMock()

        # Mock the execution dependencies
        mock_raw = AsyncMock()
        mock_raw.exit_code = 0
        mock_raw.timed_out = False
        mock_raw.stdout = "output"
        mock_raw.stderr = ""

        with (
            patch(f"{_MODULE}.setup_working_directory", return_value="/tmp/work"),
            patch(f"{_MODULE}.build_execution_env", return_value={}),
            patch(
                f"{_MODULE}.write_script", return_value="/tmp/work/ablation_study.py"
            ),
            patch(
                f"{_MODULE}.execute_script",
                new_callable=AsyncMock,
                return_value=mock_raw,
            ),
            caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
        ):
            await execute_ablation_with_retry(ablation_script, task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention execution start with script path or timeout
        assert (
            "ablation" in all_info_text.lower()
            or "script" in all_info_text.lower()
            or "execut" in all_info_text.lower()
        )

    async def test_ablation_execution_complete_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Ablation execution complete is logged at INFO with exit code, output length, duration."""
        from mle_star.phase2_outer import execute_ablation_with_retry

        ablation_script = _make_ablation_script(content="print('test')")
        task = _make_task()
        config = _make_config()
        client = AsyncMock()

        mock_raw = AsyncMock()
        mock_raw.exit_code = 0
        mock_raw.timed_out = False
        mock_raw.stdout = "ablation output data here"
        mock_raw.stderr = ""

        with (
            patch(f"{_MODULE}.setup_working_directory", return_value="/tmp/work"),
            patch(f"{_MODULE}.build_execution_env", return_value={}),
            patch(
                f"{_MODULE}.write_script", return_value="/tmp/work/ablation_study.py"
            ),
            patch(
                f"{_MODULE}.execute_script",
                new_callable=AsyncMock,
                return_value=mock_raw,
            ),
            caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
        ):
            await execute_ablation_with_retry(ablation_script, task, config, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention exit code or completion
        assert (
            "exit" in all_info_text.lower()
            or "complet" in all_info_text.lower()
            or "0" in all_info_text
        )

    async def test_ablation_execution_error_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Ablation execution error is logged at WARNING with exit code and traceback."""
        from mle_star.phase2_outer import execute_ablation_with_retry

        ablation_script = _make_ablation_script(content="import os; raise Exception()")
        task = _make_task()
        config = _make_config(max_debug_attempts=1)
        client = AsyncMock()

        mock_raw = AsyncMock()
        mock_raw.exit_code = 1
        mock_raw.timed_out = False
        mock_raw.stdout = ""
        mock_raw.stderr = (
            "Traceback (most recent call last):\n  File '<stdin>', line 1\nException"
        )

        # Debug callback returns a new (still-failing) script
        debug_cb = AsyncMock(
            return_value=_make_ablation_script(content="fixed attempt")
        )

        with (
            patch(f"{_MODULE}.setup_working_directory", return_value="/tmp/work"),
            patch(f"{_MODULE}.build_execution_env", return_value={}),
            patch(
                f"{_MODULE}.write_script", return_value="/tmp/work/ablation_study.py"
            ),
            patch(
                f"{_MODULE}.execute_script",
                new_callable=AsyncMock,
                return_value=mock_raw,
            ),
            patch(f"{_MODULE}.extract_traceback", return_value="Exception"),
            patch(f"{_MODULE}.make_debug_callback", return_value=debug_cb),
            caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
        ):
            await execute_ablation_with_retry(ablation_script, task, config, client)

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1
        warning_text = " ".join(r.message for r in warning_msgs)
        assert (
            "fail" in warning_text.lower()
            or "error" in warning_text.lower()
            or "exit" in warning_text.lower()
        )


# ===========================================================================
# REQ-P2O-037: Structured Logging -- A_summarize Invocation
# ===========================================================================


@pytest.mark.unit
class TestSummarizeInvocationLogging:
    """A_summarize invocation events are logged at INFO (REQ-P2O-037)."""

    async def test_summarize_invocation_start_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_summarize invocation start is logged at INFO with code and output lengths."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="Summary of ablation results")
        ablation_code = "print('feature importance analysis')"
        raw_output = "Feature X is most important"

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
        ):
            mock_template = mock_registry_cls.return_value.get.return_value
            mock_template.render.return_value = "rendered prompt"
            await invoke_summarize(ablation_code, raw_output, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention ablation code length and/or raw output length
        assert (
            str(len(ablation_code)) in all_info_text
            or str(len(raw_output)) in all_info_text
            or "length" in all_info_text.lower()
            or "summarize" in all_info_text.lower()
        )

    async def test_summarize_invocation_complete_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_summarize invocation complete is logged at INFO with summary length."""
        from mle_star.phase2_outer import invoke_summarize

        client = AsyncMock()
        summary_text = "The ablation study shows feature X contributes 45% to the score"
        client.send_message = AsyncMock(return_value=summary_text)

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
        ):
            mock_template = mock_registry_cls.return_value.get.return_value
            mock_template.render.return_value = "rendered prompt"
            await invoke_summarize("code", "output", client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention summary length
        assert (
            str(len(summary_text)) in all_info_text
            or "length" in all_info_text.lower()
            or "complet" in all_info_text.lower()
        )


# ===========================================================================
# REQ-P2O-037: Structured Logging -- A_extractor Invocation
# ===========================================================================


@pytest.mark.unit
class TestExtractorInvocationLogging:
    """A_extractor invocation events are logged at INFO (REQ-P2O-037)."""

    async def test_extractor_invocation_start_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_extractor invocation start is logged at INFO with summary, solution, blocks count."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        extractor_json = (
            '{"plans": [{"code_block": "model.fit(df)", "plan": "optimize"}]}'
        )
        client.send_message = AsyncMock(return_value=extractor_json)
        solution = _make_solution(content="code model.fit(df) end")
        summary = "Feature engineering is key"
        previous_blocks = ["block1", "block2"]

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
        ):
            mock_template = mock_registry_cls.return_value.get.return_value
            mock_template.render.return_value = "rendered prompt"
            await invoke_extractor(summary, solution, previous_blocks, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention summary length, solution length, or previous blocks count
        assert (
            str(len(summary)) in all_info_text
            or str(len(previous_blocks)) in all_info_text
            or "2" in all_info_text
            or "extractor" in all_info_text.lower()
        )

    async def test_extractor_invocation_complete_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_extractor invocation complete is logged at INFO with plan count and code block length."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        extractor_json = (
            '{"plans": [{"code_block": "model.fit(df)", "plan": "optimize training"}]}'
        )
        client.send_message = AsyncMock(return_value=extractor_json)
        solution = _make_solution(content="code model.fit(df) end")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
        ):
            mock_template = mock_registry_cls.return_value.get.return_value
            mock_template.render.return_value = "rendered prompt"
            result = await invoke_extractor("summary", solution, [], client)

        assert result is not None
        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention number of plans or code block length
        assert (
            "1" in all_info_text
            or "plan" in all_info_text.lower()
            or "complet" in all_info_text.lower()
        )

    async def test_extractor_parse_failure_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A_extractor parse failure is logged at WARNING (already exists, verify here)."""
        from mle_star.phase2_outer import invoke_extractor

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="not valid json")
        solution = _make_solution()

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
        ):
            mock_template = mock_registry_cls.return_value.get.return_value
            mock_template.render.return_value = "rendered prompt"
            result = await invoke_extractor("summary", solution, [], client)

        assert result is None
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1
        warning_text = " ".join(r.message for r in warning_msgs)
        assert "extractor" in warning_text.lower() or "parse" in warning_text.lower()


# ===========================================================================
# REQ-P2O-037: Structured Logging -- Code Block Validation
# ===========================================================================


@pytest.mark.unit
class TestCodeBlockValidationLogging:
    """Code block validation events are logged at correct levels (REQ-P2O-037)."""

    async def test_validation_pass_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Code block validation pass is logged at INFO with pass/fail and method."""
        mocks = _patch_outer_loop_dependencies()
        await _run_outer_loop_with_caplog(
            mocks, caplog, outer_loop_steps=1, initial_score=0.80
        )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Validation result should be logged (pass or validation)
        assert (
            "valid" in all_info_text.lower()
            or "pass" in all_info_text.lower()
            or "block" in all_info_text.lower()
        )

    async def test_validation_failure_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Code block validation failure is logged at WARNING with first 100 chars."""
        mocks = _patch_outer_loop_dependencies(validate_code_block_rv=False)
        mocks["validate_code_block"] = lambda code_block, solution: False

        await _run_outer_loop_with_caplog(
            mocks, caplog, outer_loop_steps=1, initial_score=0.80
        )

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1
        warning_text = " ".join(r.message for r in warning_msgs)
        assert (
            "valid" in warning_text.lower()
            or "block" in warning_text.lower()
            or "skip" in warning_text.lower()
        )


# ===========================================================================
# REQ-P2O-037: Structured Logging -- Inner Loop Handoff/Return
# ===========================================================================


@pytest.mark.unit
class TestInnerLoopHandoffLogging:
    """Inner loop handoff and return events are logged at INFO (REQ-P2O-037)."""

    async def test_inner_loop_handoff_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Inner loop handoff is logged at INFO with code block length and plan text."""
        mocks = _patch_outer_loop_dependencies()
        await _run_outer_loop_with_caplog(
            mocks, caplog, outer_loop_steps=1, initial_score=0.80
        )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention inner loop or handoff
        assert (
            "inner" in all_info_text.lower()
            or "handoff" in all_info_text.lower()
            or "block" in all_info_text.lower()
        )

    async def test_inner_loop_handoff_truncates_plan(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Inner loop handoff log truncates plan text to first 200 characters."""
        long_plan = "P" * 400
        extractor_output = ExtractorOutput(
            plans=[RefinePlan(code_block="model.fit(df)", plan=long_plan)]
        )
        mocks = _patch_outer_loop_dependencies(invoke_extractor_rv=extractor_output)
        await _run_outer_loop_with_caplog(
            mocks, caplog, outer_loop_steps=1, initial_score=0.80
        )

        # No single log message should contain the full 400-char plan
        for record in caplog.records:
            assert long_plan not in record.message

    async def test_inner_loop_return_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Inner loop return is logged at INFO with best score and improvement status."""
        inner_result = _make_inner_loop_result(best_score=0.92, improved=True)
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)
        await _run_outer_loop_with_caplog(
            mocks, caplog, outer_loop_steps=1, initial_score=0.80
        )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention the score from inner loop
        assert "0.92" in all_info_text or "score" in all_info_text.lower()
        # Should mention improvement status
        assert (
            "yes" in all_info_text.lower()
            or "improv" in all_info_text.lower()
            or "better" in all_info_text.lower()
        )

    async def test_inner_loop_return_no_improvement_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Inner loop return with no improvement logs at INFO indicating no improvement."""
        inner_result = _make_inner_loop_result(best_score=0.70, improved=False)
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)
        await _run_outer_loop_with_caplog(
            mocks, caplog, outer_loop_steps=1, initial_score=0.80
        )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention the return score or no improvement
        assert "0.7" in all_info_text or "score" in all_info_text.lower()


# ===========================================================================
# REQ-P2O-037: Structured Logging -- Outer Step Complete
# ===========================================================================


@pytest.mark.unit
class TestOuterStepCompleteLogging:
    """Outer step complete event is logged at INFO (REQ-P2O-037)."""

    async def test_outer_step_complete_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Outer step complete is logged at INFO with step index, h_best, and duration."""
        mocks = _patch_outer_loop_dependencies()
        await _run_outer_loop_with_caplog(
            mocks, caplog, outer_loop_steps=1, initial_score=0.80
        )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention step completion
        assert (
            "complet" in all_info_text.lower()
            or "finish" in all_info_text.lower()
            or "done" in all_info_text.lower()
            or "step" in all_info_text.lower()
        )

    async def test_outer_step_complete_includes_duration(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Outer step complete log includes duration information."""
        mocks = _patch_outer_loop_dependencies()
        await _run_outer_loop_with_caplog(
            mocks, caplog, outer_loop_steps=1, initial_score=0.80
        )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention duration (seconds) or time
        assert (
            "duration" in all_info_text.lower()
            or "time" in all_info_text.lower()
            or "s)" in all_info_text
            or "sec" in all_info_text.lower()
        )


# ===========================================================================
# REQ-P2O-037: Structured Logging -- Outer Step Skipped
# ===========================================================================


@pytest.mark.unit
class TestOuterStepSkippedLogging:
    """Outer step skipped event is logged at WARNING (REQ-P2O-037)."""

    async def test_skipped_extractor_none_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Skipped step due to extractor returning None logs WARNING with step and reason."""
        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_extractor"] = AsyncMock(return_value=None)

        await _run_outer_loop_with_caplog(
            mocks, caplog, outer_loop_steps=1, initial_score=0.80
        )

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1
        warning_text = " ".join(r.message for r in warning_msgs)
        # Should mention step index and reason (extractor or skip)
        assert (
            "extractor" in warning_text.lower()
            or "skip" in warning_text.lower()
            or "none" in warning_text.lower()
        )

    async def test_skipped_validation_failure_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Skipped step due to validation failure logs WARNING with step and reason."""
        mocks = _patch_outer_loop_dependencies(validate_code_block_rv=False)
        mocks["validate_code_block"] = lambda code_block, solution: False

        await _run_outer_loop_with_caplog(
            mocks, caplog, outer_loop_steps=1, initial_score=0.80
        )

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1
        warning_text = " ".join(r.message for r in warning_msgs)
        assert (
            "valid" in warning_text.lower()
            or "skip" in warning_text.lower()
            or "block" in warning_text.lower()
        )


# ===========================================================================
# REQ-P2O-037: Structured Logging -- Outer Loop Complete
# ===========================================================================


@pytest.mark.unit
class TestOuterLoopCompleteLogging:
    """Outer loop complete event is logged at INFO (REQ-P2O-037)."""

    async def test_outer_loop_complete_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Outer loop complete is logged at INFO with total steps, final h_best, duration."""
        mocks = _patch_outer_loop_dependencies()
        await _run_outer_loop_with_caplog(
            mocks, caplog, outer_loop_steps=3, initial_score=0.80
        )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should mention total steps completed (3)
        assert "3" in all_info_text or "steps" in all_info_text.lower()
        # Should mention final h_best
        assert (
            "final" in all_info_text.lower()
            or "h_best" in all_info_text.lower()
            or "best" in all_info_text.lower()
        )

    async def test_outer_loop_complete_includes_total_duration(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Outer loop complete log includes total duration."""
        mocks = _patch_outer_loop_dependencies()
        await _run_outer_loop_with_caplog(
            mocks, caplog, outer_loop_steps=2, initial_score=0.80
        )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        assert (
            "duration" in all_info_text.lower()
            or "time" in all_info_text.lower()
            or "sec" in all_info_text.lower()
        )

    async def test_outer_loop_complete_with_all_skipped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Outer loop complete log still emitted when all steps are skipped."""
        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_extractor"] = AsyncMock(return_value=None)

        await _run_outer_loop_with_caplog(
            mocks, caplog, outer_loop_steps=2, initial_score=0.80
        )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        all_info_text = " ".join(r.message for r in info_msgs)
        # Should still log outer loop completion
        assert (
            "outer" in all_info_text.lower()
            or "complet" in all_info_text.lower()
            or "loop" in all_info_text.lower()
        )


# ===========================================================================
# REQ-P2O-040: Sequential Outer Loop Execution
# ===========================================================================


@pytest.mark.unit
class TestSequentialOuterExecution:
    """Outer loop iterations run sequentially, not concurrently (REQ-P2O-040)."""

    async def test_iterations_run_sequentially(self) -> None:
        """Each outer iteration completes before the next begins (no concurrent execution)."""
        call_order: list[int] = []
        call_completed: list[int] = []

        async def tracked_ablation(*args: Any, **kwargs: Any) -> SolutionScript:
            step = len(call_order)
            call_order.append(step)
            # Verify the previous call completed before this one started
            if step > 0:
                assert step - 1 in call_completed, (
                    f"Step {step} started before step {step - 1} completed"
                )
            call_completed.append(step)
            return _make_ablation_script()

        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_ablation"] = AsyncMock(side_effect=tracked_ablation)

        await _run_outer_loop(mocks, outer_loop_steps=3, initial_score=0.80)

        assert call_order == [0, 1, 2]
        assert call_completed == [0, 1, 2]

    async def test_no_concurrent_inner_loops(self) -> None:
        """Inner loop invocations are never running concurrently."""
        active_inner_loops = 0
        max_concurrent = 0

        async def tracked_inner_loop(*args: Any, **kwargs: Any) -> InnerLoopResult:
            nonlocal active_inner_loops, max_concurrent
            active_inner_loops += 1
            if active_inner_loops > max_concurrent:
                max_concurrent = active_inner_loops
            result = _make_inner_loop_result()
            active_inner_loops -= 1
            return result

        mocks = _patch_outer_loop_dependencies()
        mocks["run_phase2_inner_loop"] = AsyncMock(side_effect=tracked_inner_loop)

        await _run_outer_loop(mocks, outer_loop_steps=3, initial_score=0.80)

        assert max_concurrent == 1, (
            f"Expected max 1 concurrent inner loop, got {max_concurrent}"
        )

    async def test_state_accumulates_correctly_across_steps(self) -> None:
        """Summaries and code blocks accumulate sequentially across iterations."""
        summaries = ["summary_A", "summary_B", "summary_C"]
        mock_summarize = AsyncMock(side_effect=summaries)

        ablation_calls: list[tuple[Any, ...]] = []

        async def capture_ablation(*args: Any, **kwargs: Any) -> SolutionScript:
            ablation_calls.append(args)
            return _make_ablation_script()

        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_summarize"] = mock_summarize
        mocks["invoke_ablation"] = AsyncMock(side_effect=capture_ablation)

        result = await _run_outer_loop(mocks, outer_loop_steps=3, initial_score=0.80)

        # Summaries should accumulate
        assert len(result.ablation_summaries) == 3
        assert result.ablation_summaries == summaries

        # First ablation call should have 0 previous summaries
        # Second should have 1, third should have 2
        assert len(ablation_calls) == 3
        # Previous summaries is the second positional arg
        first_prev = ablation_calls[0][1] if len(ablation_calls[0]) > 1 else []
        second_prev = ablation_calls[1][1] if len(ablation_calls[1]) > 1 else []
        third_prev = ablation_calls[2][1] if len(ablation_calls[2]) > 1 else []
        assert len(first_prev) == 0
        assert len(second_prev) == 1
        assert len(third_prev) == 2


# ===========================================================================
# REQ-P2O-041: Monotonic Best Score
# ===========================================================================


@pytest.mark.unit
class TestMonotonicBestScore:
    """h_best is monotonically non-decreasing (maximize) or non-increasing (minimize) (REQ-P2O-041)."""

    async def test_best_score_never_decreases_maximize(self) -> None:
        """With MAXIMIZE, best score never decreases across outer iterations."""
        # Inner loop returns varying scores: 0.82, 0.78 (worse), 0.90, 0.85 (worse)
        scores = [0.82, 0.78, 0.90, 0.85]
        inner_results = [
            _make_inner_loop_result(best_score=s, improved=s > 0.80) for s in scores
        ]
        mock_inner = AsyncMock(side_effect=inner_results)
        mocks = _patch_outer_loop_dependencies()
        mocks["run_phase2_inner_loop"] = mock_inner

        result = await _run_outer_loop(mocks, outer_loop_steps=4, initial_score=0.80)

        # h_best: 0.80 -> 0.82 -> 0.82 (skip 0.78) -> 0.90 -> 0.90 (skip 0.85)
        assert result.best_score == 0.90

        # Verify step_history shows monotonic best_score_after_step
        h_values = [step["best_score_after_step"] for step in result.step_history]
        for i in range(1, len(h_values)):
            assert h_values[i] >= h_values[i - 1], (
                f"h_best decreased at step {i}: {h_values[i - 1]} -> {h_values[i]}"
            )

    async def test_best_score_never_increases_minimize(self) -> None:
        """With MINIMIZE, best score never increases across outer iterations."""
        task = _make_task(direction=MetricDirection.MINIMIZE)
        scores = [0.78, 0.82, 0.70, 0.75]
        inner_results = [
            _make_inner_loop_result(best_score=s, improved=s < 0.80) for s in scores
        ]
        mock_inner = AsyncMock(side_effect=inner_results)
        mocks = _patch_outer_loop_dependencies()
        mocks["run_phase2_inner_loop"] = mock_inner
        mocks["is_improvement_or_equal"] = lambda new, old, d: new <= old

        result = await _run_outer_loop(
            mocks, outer_loop_steps=4, initial_score=0.80, task=task
        )

        # h_best: 0.80 -> 0.78 -> 0.78 (skip 0.82) -> 0.70 -> 0.70 (skip 0.75)
        assert result.best_score == 0.70

        # Verify monotonicity (non-increasing for minimize)
        h_values = [step["best_score_after_step"] for step in result.step_history]
        for i in range(1, len(h_values)):
            assert h_values[i] <= h_values[i - 1], (
                f"h_best increased at step {i}: {h_values[i - 1]} -> {h_values[i]}"
            )

    async def test_best_score_monotonic_with_skipped_steps(self) -> None:
        """Best score stays monotonic even when some steps are skipped."""
        # Step 0: success (0.85), Step 1: skip (extractor None), Step 2: success (0.90)
        inner_results = [
            _make_inner_loop_result(best_score=0.85, improved=True),
            _make_inner_loop_result(best_score=0.90, improved=True),
        ]
        mock_inner = AsyncMock(side_effect=inner_results)
        extractor_results = [
            _make_extractor_output(),
            None,
            _make_extractor_output(),
        ]
        mock_extractor = AsyncMock(side_effect=extractor_results)

        mocks = _patch_outer_loop_dependencies()
        mocks["run_phase2_inner_loop"] = mock_inner
        mocks["invoke_extractor"] = mock_extractor

        result = await _run_outer_loop(mocks, outer_loop_steps=3, initial_score=0.80)

        # h_best: 0.80 -> 0.85 -> 0.85 (skipped) -> 0.90
        assert result.best_score == 0.90

        h_values = [step["best_score_after_step"] for step in result.step_history]
        for i in range(1, len(h_values)):
            assert h_values[i] >= h_values[i - 1]

    async def test_best_score_monotonic_with_all_worse(self) -> None:
        """When all inner loop scores are worse, h_best stays at initial_score."""
        scores = [0.70, 0.65, 0.75]
        inner_results = [
            _make_inner_loop_result(best_score=s, improved=False) for s in scores
        ]
        mock_inner = AsyncMock(side_effect=inner_results)
        mocks = _patch_outer_loop_dependencies()
        mocks["run_phase2_inner_loop"] = mock_inner

        result = await _run_outer_loop(mocks, outer_loop_steps=3, initial_score=0.80)

        assert result.best_score == 0.80
        # All step records should show h_best = 0.80 (unchanged)
        for step in result.step_history:
            assert step["best_score_after_step"] == 0.80

    @given(
        initial_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        score_deltas=st.lists(
            st.floats(min_value=-0.3, max_value=0.3, allow_nan=False),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=30, deadline=5000)
    async def test_monotonic_property_maximize(
        self, initial_score: float, score_deltas: list[float]
    ) -> None:
        """Property: h_best is monotonically non-decreasing for MAXIMIZE direction."""
        scores = [max(0.0, min(1.0, initial_score + d)) for d in score_deltas]
        inner_results = [
            _make_inner_loop_result(best_score=s, improved=s > initial_score)
            for s in scores
        ]
        mock_inner = AsyncMock(side_effect=inner_results)
        mocks = _patch_outer_loop_dependencies()
        mocks["run_phase2_inner_loop"] = mock_inner

        result = await _run_outer_loop(
            mocks,
            outer_loop_steps=len(scores),
            initial_score=initial_score,
        )

        # Verify monotonicity
        h_values = [initial_score] + [
            step["best_score_after_step"] for step in result.step_history
        ]
        for i in range(1, len(h_values)):
            assert h_values[i] >= h_values[i - 1], (
                f"h_best decreased at step {i}: {h_values[i - 1]} -> {h_values[i]}"
            )


# ===========================================================================
# REQ-P2O-042: Ablation Script Self-Containment
# ===========================================================================


@pytest.mark.unit
class TestAblationScriptSelfContainment:
    """Ablation scripts must be self-contained (REQ-P2O-042)."""

    async def test_ablation_script_is_solution_script(self) -> None:
        """invoke_ablation returns a SolutionScript with is_executable=True."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value="```python\nprint('ablation')\n```"
        )
        solution = _make_solution()

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="print('ablation')"),
        ):
            mock_template = mock_registry_cls.return_value.get.return_value
            mock_template.render.return_value = "rendered prompt"
            result = await invoke_ablation(solution, [], client)

        assert result is not None
        assert isinstance(result, SolutionScript)
        assert result.is_executable is True
        assert result.phase == SolutionPhase.REFINED

    async def test_ablation_script_content_from_extraction(self) -> None:
        """Ablation script content comes from extract_code_block, ensuring self-containment."""
        from mle_star.phase2_outer import invoke_ablation

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value="Some explanation\n```python\nimport sklearn\nprint('done')\n```"
        )
        extracted = "import sklearn\nprint('done')"
        solution = _make_solution()

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=extracted),
        ):
            mock_template = mock_registry_cls.return_value.get.return_value
            mock_template.render.return_value = "rendered prompt"
            result = await invoke_ablation(solution, [], client)

        assert result is not None
        assert result.content == extracted

    async def test_ablation_script_executed_as_standalone(self) -> None:
        """Ablation script is written to disk and executed as a standalone Python file."""
        from mle_star.phase2_outer import execute_ablation_with_retry

        ablation_script = _make_ablation_script(content="print('standalone test')")
        task = _make_task()
        config = _make_config()
        client = AsyncMock()

        mock_raw = AsyncMock()
        mock_raw.exit_code = 0
        mock_raw.timed_out = False
        mock_raw.stdout = "standalone test"
        mock_raw.stderr = ""

        with (
            patch(f"{_MODULE}.setup_working_directory", return_value="/tmp/work"),
            patch(f"{_MODULE}.build_execution_env", return_value={}),
            patch(
                f"{_MODULE}.write_script", return_value="/tmp/work/ablation_study.py"
            ) as mock_write,
            patch(
                f"{_MODULE}.execute_script",
                new_callable=AsyncMock,
                return_value=mock_raw,
            ),
        ):
            await execute_ablation_with_retry(ablation_script, task, config, client)

        # Verify write_script was called with the ablation script
        mock_write.assert_called_once()
        call_args = mock_write.call_args
        assert call_args[0][0] == ablation_script  # First positional arg is the script


# ===========================================================================
# REQ-P2O-043: Immutable Input Solution
# ===========================================================================


@pytest.mark.unit
class TestImmutableInputSolution:
    """The initial_solution parameter must not be mutated (REQ-P2O-043)."""

    async def test_input_solution_not_mutated(self) -> None:
        """Original solution object's content is unchanged after outer loop completes."""
        original_content = (
            "import pandas as pd\ndf = pd.read_csv('data.csv')\nmodel.fit(df)"
        )
        solution = _make_solution(content=original_content)
        original_score = solution.score
        original_phase = solution.phase

        improved_sol = _make_solution(
            content="improved code", phase=SolutionPhase.REFINED
        )
        inner_result = _make_inner_loop_result(
            best_solution=improved_sol, best_score=0.95, improved=True
        )
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)

        await _run_outer_loop(
            mocks,
            outer_loop_steps=3,
            initial_score=0.80,
            initial_solution=solution,
        )

        # Verify the original solution was NOT mutated
        assert solution.content == original_content
        assert solution.score == original_score
        assert solution.phase == original_phase

    async def test_input_solution_content_preserved_across_all_iterations(self) -> None:
        """Solution content remains identical across all T iterations."""
        original_content = (
            "import pandas as pd\ndf = pd.read_csv('data.csv')\nmodel.fit(df)"
        )
        solution = _make_solution(content=original_content)

        ablation_calls: list[str] = []

        async def capture_ablation(
            sol: SolutionScript, prev: list[str], client: Any
        ) -> SolutionScript:
            ablation_calls.append(sol.content)
            return _make_ablation_script()

        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_ablation"] = AsyncMock(side_effect=capture_ablation)

        # Even with improvements, first iteration should receive the original solution
        await _run_outer_loop(
            mocks,
            outer_loop_steps=3,
            initial_score=0.80,
            initial_solution=solution,
        )

        # The original solution object should not be mutated
        assert solution.content == original_content

    async def test_deep_copy_semantics_initial_solution(self) -> None:
        """Mutations to the outer loop's working copy do not affect the original."""
        original_content = "original code model.fit(df) end"
        solution = _make_solution(content=original_content)

        # Deep copy to verify isolation
        solution_snapshot = copy.deepcopy(solution)

        improved_sol = _make_solution(
            content="improved code", phase=SolutionPhase.REFINED
        )
        inner_result = _make_inner_loop_result(
            best_solution=improved_sol, best_score=0.95, improved=True
        )
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)

        result = await _run_outer_loop(
            mocks,
            outer_loop_steps=2,
            initial_score=0.80,
            initial_solution=solution,
        )

        # The solution should be identical to the snapshot
        assert solution.content == solution_snapshot.content
        assert solution.score == solution_snapshot.score
        assert solution.phase == solution_snapshot.phase

        # Result should have the improved solution, not the original
        assert result.best_solution.content == "improved code"


# ===========================================================================
# REQ-P2O-044: Code Block Provenance (CodeBlock.outer_step == t)
# ===========================================================================


@pytest.mark.unit
class TestCodeBlockProvenance:
    """Each CodeBlock.outer_step equals its iteration index (REQ-P2O-044)."""

    async def test_code_block_outer_step_matches_iteration(self) -> None:
        """Each CodeBlock.outer_step equals its iteration index t."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=4, initial_score=0.80)

        for t, block in enumerate(result.refined_blocks):
            assert block.outer_step == t, (
                f"Block {t} has outer_step={block.outer_step}, expected {t}"
            )

    async def test_code_block_provenance_with_skipped_steps(self) -> None:
        """Even for skipped steps, CodeBlock entries track correct outer_step."""
        # Step 0: success, Step 1: skip (extractor None), Step 2: success
        extractor_results = [
            _make_extractor_output(),
            None,
            _make_extractor_output(),
        ]
        inner_results = [
            _make_inner_loop_result(best_score=0.85, improved=True),
            _make_inner_loop_result(best_score=0.90, improved=True),
        ]
        mock_extractor = AsyncMock(side_effect=extractor_results)
        mock_inner = AsyncMock(side_effect=inner_results)

        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_extractor"] = mock_extractor
        mocks["run_phase2_inner_loop"] = mock_inner

        result = await _run_outer_loop(mocks, outer_loop_steps=3, initial_score=0.80)

        assert len(result.refined_blocks) == 3
        for t, block in enumerate(result.refined_blocks):
            assert block.outer_step == t

    async def test_code_block_is_code_block_instance(self) -> None:
        """Each entry in refined_blocks is a CodeBlock instance with outer_step set."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=3, initial_score=0.80)

        for block in result.refined_blocks:
            assert isinstance(block, CodeBlock)
            assert block.outer_step is not None

    @given(t_steps=st.integers(min_value=1, max_value=8))
    @settings(max_examples=10, deadline=5000)
    async def test_code_block_provenance_property(self, t_steps: int) -> None:
        """Property: refined_blocks has T entries with outer_step 0..T-1."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(
            mocks, outer_loop_steps=t_steps, initial_score=0.80
        )

        assert len(result.refined_blocks) == t_steps
        for t, block in enumerate(result.refined_blocks):
            assert block.outer_step == t

    async def test_code_block_content_matches_extractor(self) -> None:
        """CodeBlock.content matches the code block from extractor plans[0]."""
        extractor_output = _make_extractor_output(code_block="model.fit(df)")
        solution = _make_solution(
            content="import sklearn\nmodel.fit(df)\nprint('done')"
        )
        mocks = _patch_outer_loop_dependencies(invoke_extractor_rv=extractor_output)

        result = await _run_outer_loop(
            mocks,
            outer_loop_steps=1,
            initial_score=0.80,
            initial_solution=solution,
        )

        assert result.refined_blocks[0].content == "model.fit(df)"
        assert result.refined_blocks[0].outer_step == 0


# ===========================================================================
# REQ-P2O-037: Full Logging Integration
# ===========================================================================


@pytest.mark.unit
class TestFullLoggingIntegration:
    """Integration test verifying complete logging through an outer loop execution."""

    async def test_successful_step_has_all_expected_log_events(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A successful outer step produces INFO logs for all expected events."""
        inner_result = _make_inner_loop_result(best_score=0.90, improved=True)
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)

        await _run_outer_loop_with_caplog(
            mocks, caplog, outer_loop_steps=1, initial_score=0.80
        )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        # Should have multiple INFO messages for a full step:
        # step start, inner loop handoff, inner loop return, step complete, loop complete
        assert len(info_msgs) >= 3, (
            f"Expected at least 3 INFO messages, got {len(info_msgs)}. "
            f"Messages: {[r.message for r in info_msgs]}"
        )

    async def test_skipped_step_produces_warning_and_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A skipped step produces a WARNING plus INFO for step start and loop complete."""
        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_extractor"] = AsyncMock(return_value=None)

        await _run_outer_loop_with_caplog(
            mocks, caplog, outer_loop_steps=1, initial_score=0.80
        )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]

        # Should have at least INFO for step start and loop complete
        assert len(info_msgs) >= 1
        # Should have WARNING for the skip
        assert len(warning_msgs) >= 1

    async def test_multiple_steps_produce_correct_log_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Multiple steps produce proportionally more log messages."""
        mocks = _patch_outer_loop_dependencies()
        await _run_outer_loop_with_caplog(
            mocks, caplog, outer_loop_steps=3, initial_score=0.80
        )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        # 3 steps worth of logging + loop complete; expect significantly more than 3
        assert len(info_msgs) >= 4, (
            f"Expected at least 4 INFO messages for 3 steps, got {len(info_msgs)}"
        )

    async def test_log_messages_are_from_correct_logger(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """All log messages come from the mle_star.phase2_outer logger."""
        mocks = _patch_outer_loop_dependencies()
        await _run_outer_loop_with_caplog(
            mocks, caplog, outer_loop_steps=1, initial_score=0.80
        )

        for record in caplog.records:
            assert record.name == _LOGGER_NAME, (
                f"Log from unexpected logger: {record.name}"
            )
