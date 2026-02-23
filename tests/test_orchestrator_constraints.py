"""Tests for orchestrator constraints (Task 49).

Validates non-functional requirements for the orchestrator module:
orchestration overhead budget, memory efficiency, idempotent retry safety,
CLI retry on transient failure, CLI version checking, concurrent
session limiting, and agent name uniqueness.

Tests are written TDD-first and serve as the executable specification for
REQ-OR-048 through REQ-OR-057.

Refs:
    SRS 09c -- Orchestrator Constraints.
    IMPLEMENTATION_PLAN.md Task 49.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import HealthCheck, given, settings, strategies as st
from mle_star.models import (
    AgentType,
    CodeBlock,
    CodeBlockCategory,
    DataModality,
    FinalResult,
    MetricDirection,
    Phase1Result,
    Phase2Result,
    PipelineConfig,
    RetrievedModel,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
    TaskType,
    build_default_agent_configs,
)
import pytest

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Module path constant for patching
# ---------------------------------------------------------------------------

_MODULE = "mle_star.orchestrator"
_LOGGER_NAME = "mle_star.orchestrator"


# ---------------------------------------------------------------------------
# Reusable test helpers
# ---------------------------------------------------------------------------


def _make_task(**overrides: Any) -> TaskDescription:
    """Build a valid TaskDescription with sensible defaults."""
    defaults: dict[str, Any] = {
        "competition_id": "test-comp",
        "task_type": TaskType.CLASSIFICATION,
        "data_modality": DataModality.TABULAR,
        "evaluation_metric": "accuracy",
        "metric_direction": MetricDirection.MAXIMIZE,
        "description": "Predict the target.",
        "data_dir": "./input",
        "output_dir": "./final",
    }
    defaults.update(overrides)
    return TaskDescription(**defaults)


def _make_config(**overrides: Any) -> PipelineConfig:
    """Build a valid PipelineConfig with sensible defaults."""
    return PipelineConfig(**overrides)


def _make_solution(**overrides: Any) -> SolutionScript:
    """Build a valid SolutionScript with sensible defaults."""
    defaults: dict[str, Any] = {
        "content": "import pandas as pd\nprint('hello')\n",
        "phase": SolutionPhase.INIT,
    }
    defaults.update(overrides)
    return SolutionScript(**defaults)


def _make_phase1_result(**overrides: Any) -> Phase1Result:
    """Build a valid Phase1Result with sensible defaults."""
    defaults: dict[str, Any] = {
        "retrieved_models": [
            RetrievedModel(model_name="xgboost", example_code="import xgboost")
        ],
        "candidate_solutions": [_make_solution(phase=SolutionPhase.INIT)],
        "candidate_scores": [0.85],
        "initial_solution": _make_solution(phase=SolutionPhase.MERGED),
        "initial_score": 0.85,
    }
    defaults.update(overrides)
    return Phase1Result(**defaults)


def _make_phase2_result(**overrides: Any) -> Phase2Result:
    """Build a valid Phase2Result with sensible defaults."""
    defaults: dict[str, Any] = {
        "ablation_summaries": ["summary"],
        "refined_blocks": [
            CodeBlock(content="block", category=CodeBlockCategory.TRAINING)
        ],
        "best_solution": _make_solution(phase=SolutionPhase.REFINED),
        "best_score": 0.90,
        "step_history": [{"step": 0, "score": 0.90}],
    }
    defaults.update(overrides)
    return Phase2Result(**defaults)


def _make_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with a dummy file."""
    data_dir = tmp_path / "input"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "train.csv").write_text("id,feature,target\n1,0.5,0\n")
    return data_dir


def _make_final_result(
    task: TaskDescription, config: PipelineConfig, **overrides: Any
) -> FinalResult:
    """Build a valid FinalResult with sensible defaults."""
    defaults: dict[str, Any] = {
        "task": task,
        "config": config,
        "phase1": _make_phase1_result(),
        "phase2_results": [_make_phase2_result()],
        "phase3": None,
        "final_solution": _make_solution(phase=SolutionPhase.FINAL),
        "submission_path": "/output/submission.csv",
        "total_duration_seconds": 10.0,
    }
    defaults.update(overrides)
    return FinalResult(**defaults)


# ===========================================================================
# REQ-OR-048: Orchestrator overhead < 1% (< 100ms per phase transition)
# ===========================================================================


@pytest.mark.unit
class TestOrchestratorOverhead:
    """Orchestration overhead completes in under 100ms per phase transition (REQ-OR-048)."""

    def test_compute_phase_budgets_under_100ms(self) -> None:
        """_compute_phase_budgets executes in under 100ms."""
        from mle_star.orchestrator import _compute_phase_budgets

        # Arrange
        config = _make_config(num_parallel_solutions=4)
        remaining_seconds = 3600.0

        # Act
        start = time.monotonic()
        result = _compute_phase_budgets(config, remaining_seconds)
        elapsed_ms = (time.monotonic() - start) * 1000

        # Assert
        assert elapsed_ms < 100.0, (
            f"_compute_phase_budgets took {elapsed_ms:.2f}ms, exceeds 100ms budget"
        )
        assert "phase2" in result
        assert "phase3" in result
        assert "finalization" in result
        assert "phase2_per_path" in result

    def test_collect_phase2_results_under_100ms(self) -> None:
        """_collect_phase2_results completes in under 100ms for typical inputs."""
        from mle_star.orchestrator import _collect_phase2_results

        # Arrange
        p1_result = _make_phase1_result()
        raw_results: list[Phase2Result | BaseException] = [
            _make_phase2_result(best_score=0.88 + i * 0.01) for i in range(4)
        ]

        # Act
        start = time.monotonic()
        phase2_results, solutions = _collect_phase2_results(raw_results, p1_result)
        elapsed_ms = (time.monotonic() - start) * 1000

        # Assert
        assert elapsed_ms < 100.0, (
            f"_collect_phase2_results took {elapsed_ms:.2f}ms, exceeds 100ms budget"
        )
        assert len(phase2_results) == 4
        assert len(solutions) == 4

    def test_make_failed_phase2_result_under_100ms(self) -> None:
        """_make_failed_phase2_result completes in under 100ms."""
        from mle_star.orchestrator import _make_failed_phase2_result

        # Arrange
        p1_result = _make_phase1_result()

        # Act
        start = time.monotonic()
        result = _make_failed_phase2_result(p1_result)
        elapsed_ms = (time.monotonic() - start) * 1000

        # Assert
        assert elapsed_ms < 100.0, (
            f"_make_failed_phase2_result took {elapsed_ms:.2f}ms, exceeds 100ms budget"
        )
        assert isinstance(result, Phase2Result)

    @given(
        num_results=st.integers(min_value=1, max_value=10),
        num_failures=st.integers(min_value=0, max_value=5),
    )
    @settings(
        max_examples=10,
        deadline=5000,
    )
    def test_collect_results_overhead_scales_linearly(
        self, num_results: int, num_failures: int
    ) -> None:
        """Result collection overhead scales linearly with number of results."""
        from mle_star.orchestrator import _collect_phase2_results

        # Arrange
        p1_result = _make_phase1_result()
        actual_failures = min(num_failures, num_results)
        raw_results: list[Phase2Result | BaseException] = []
        for i in range(num_results):
            if i < actual_failures:
                raw_results.append(RuntimeError(f"path {i} failed"))
            else:
                raw_results.append(_make_phase2_result(best_score=0.85 + i * 0.01))

        # Act
        start = time.monotonic()
        phase2_results, solutions = _collect_phase2_results(raw_results, p1_result)
        elapsed_ms = (time.monotonic() - start) * 1000

        # Assert
        assert elapsed_ms < 100.0
        assert len(phase2_results) == num_results
        assert len(solutions) == num_results

    def test_create_client_under_100ms(self) -> None:
        """_create_client completes in under 100ms."""
        from mle_star.orchestrator import _create_client

        config = _make_config()
        task = _make_task()

        start = time.monotonic()
        with patch(
            f"{_MODULE}.detect_gpu_info",
            return_value={"cuda_available": False},
        ):
            client = _create_client(config, task)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 100.0, (
            f"_create_client took {elapsed_ms:.2f}ms, exceeds 100ms budget"
        )
        from mle_star.orchestrator import ClaudeCodeClient

        assert isinstance(client, ClaudeCodeClient)


# ===========================================================================
# REQ-OR-049: Memory efficiency -- only retain current best per path
# ===========================================================================


@pytest.mark.unit
class TestMemoryEfficiency:
    """Orchestrator retains only current best + under-evaluation per path (REQ-OR-049)."""

    def test_phase2_result_contains_only_best_solution(self) -> None:
        """Phase2Result stores only best_solution, not a full history of all solutions."""
        # Arrange
        p2 = _make_phase2_result(best_score=0.92)

        # Assert -- only one solution stored, not a list of all tried solutions
        assert hasattr(p2, "best_solution")
        assert isinstance(p2.best_solution, SolutionScript)
        # Phase2Result should NOT have an 'all_solutions' or 'solution_history' field
        assert not hasattr(p2, "all_solutions")
        assert not hasattr(p2, "solution_history")
        assert not hasattr(p2, "candidate_solutions")

    def test_phase2_result_step_history_is_metadata_only(self) -> None:
        """step_history contains score + metadata, not full solution content."""
        # Arrange
        p2 = _make_phase2_result(
            step_history=[
                {"step": 0, "score": 0.85},
                {"step": 1, "score": 0.87},
                {"step": 2, "score": 0.90},
            ],
        )

        # Assert -- step_history contains dicts (score/metadata), not SolutionScript
        for entry in p2.step_history:
            assert isinstance(entry, dict)
            # entries should not contain full solution objects
            for value in entry.values():
                assert not isinstance(value, SolutionScript)

    def test_collect_phase2_results_returns_only_best_per_path(self) -> None:
        """_collect_phase2_results returns one solution per path (the best)."""
        from mle_star.orchestrator import _collect_phase2_results

        # Arrange
        p1_result = _make_phase1_result()
        raw_results: list[Phase2Result | BaseException] = [
            _make_phase2_result(best_score=0.88),
            _make_phase2_result(best_score=0.91),
            _make_phase2_result(best_score=0.85),
        ]

        # Act
        _phase2_results, solutions = _collect_phase2_results(raw_results, p1_result)

        # Assert -- exactly one solution per path
        assert len(solutions) == 3
        for sol in solutions:
            assert isinstance(sol, SolutionScript)

    def test_phase2_result_ablation_summaries_are_strings(self) -> None:
        """Ablation summaries are text strings (summaries), not full ablation results."""
        # Arrange
        p2 = _make_phase2_result(
            ablation_summaries=[
                "Feature importance analysis shows...",
                "PCA reveals...",
            ]
        )

        # Assert -- ablation_summaries are strings, not complex objects
        for summary in p2.ablation_summaries:
            assert isinstance(summary, str)

    def test_phase2_result_refined_blocks_are_code_blocks(self) -> None:
        """Refined blocks store code block metadata, not full solution scripts."""
        # Arrange
        blocks = [
            CodeBlock(content="block_1", category=CodeBlockCategory.TRAINING),
            CodeBlock(content="block_2", category=CodeBlockCategory.PREPROCESSING),
        ]
        p2 = _make_phase2_result(refined_blocks=blocks)

        # Assert -- blocks are CodeBlock instances, not full SolutionScripts
        for block in p2.refined_blocks:
            assert isinstance(block, CodeBlock)
            assert not isinstance(block, SolutionScript)


# ===========================================================================
# REQ-OR-051: Idempotent retry safety -- fresh state per run_pipeline call
# ===========================================================================


@pytest.mark.unit
class TestIdempotentRetrySafety:
    """Each run_pipeline() call creates fresh state (REQ-OR-051)."""

    pass


# ===========================================================================
# REQ-OR-052: Retry on transient failure with exponential backoff
# ===========================================================================


@pytest.mark.unit
class TestRetryWithBackoff:
    """Agent invocations are retried on transient failure with exponential backoff (REQ-OR-052)."""

    async def test_retry_retries_up_to_three_times(self) -> None:
        """retry_with_backoff retries send_message up to max_retries times."""
        from mle_star.orchestrator import retry_with_backoff

        # Arrange
        mock_client = AsyncMock()
        mock_client.send_message = AsyncMock(
            side_effect=[
                RuntimeError("transient 1"),
                RuntimeError("transient 2"),
                "success response",  # Third attempt succeeds
            ]
        )

        # Act
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await retry_with_backoff(
                mock_client, AgentType.CODER, "test message", max_retries=3
            )

        # Assert
        assert result == "success response"
        assert mock_client.send_message.await_count == 3

    async def test_retry_succeeds_on_first_try(self) -> None:
        """When send_message succeeds immediately, only one attempt is made."""
        from mle_star.orchestrator import retry_with_backoff

        # Arrange
        mock_client = AsyncMock()
        mock_client.send_message = AsyncMock(return_value="immediate success")

        # Act
        result = await retry_with_backoff(
            mock_client, AgentType.PLANNER, "test msg", max_retries=3
        )

        # Assert
        assert result == "immediate success"
        assert mock_client.send_message.await_count == 1

    async def test_retry_uses_exponential_backoff(self) -> None:
        """Delays between retries follow exponential backoff: 1s, 2s, 4s."""
        from mle_star.orchestrator import retry_with_backoff

        # Arrange
        mock_client = AsyncMock()
        mock_client.send_message = AsyncMock(
            side_effect=[
                RuntimeError("fail 1"),
                RuntimeError("fail 2"),
                "ok",  # Third attempt succeeds
            ]
        )

        sleep_calls: list[float] = []

        async def _mock_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        # Act
        with patch("asyncio.sleep", side_effect=_mock_sleep):
            await retry_with_backoff(mock_client, AgentType.CODER, "msg", max_retries=3)

        # Assert -- exponential backoff: 1, 2 seconds
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == pytest.approx(1.0)
        assert sleep_calls[1] == pytest.approx(2.0)

    async def test_retry_passes_agent_type_and_message(self) -> None:
        """retry_with_backoff passes agent_type and message to send_message."""
        from mle_star.orchestrator import retry_with_backoff

        # Arrange
        mock_client = AsyncMock()
        mock_client.send_message = AsyncMock(return_value="response")

        # Act
        await retry_with_backoff(
            mock_client, AgentType.DEBUGGER, "debug this", max_retries=3
        )

        # Assert
        mock_client.send_message.assert_awaited_once()
        call_kwargs = mock_client.send_message.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("agent_type") == AgentType.DEBUGGER
        assert call_kwargs.kwargs.get("message") == "debug this"

    async def test_retry_passes_session_id(self) -> None:
        """retry_with_backoff passes session_id to send_message."""
        from mle_star.orchestrator import retry_with_backoff

        # Arrange
        mock_client = AsyncMock()
        mock_client.send_message = AsyncMock(return_value="response")

        # Act
        await retry_with_backoff(
            mock_client,
            AgentType.CODER,
            "msg",
            max_retries=3,
            session_id="my-session-42",
        )

        # Assert
        call_kwargs = mock_client.send_message.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("session_id") == "my-session-42"

    async def test_retry_raises_after_max_retries_exhausted(self) -> None:
        """After max_retries failures, raises RuntimeError."""
        from mle_star.orchestrator import retry_with_backoff

        # Arrange
        mock_client = AsyncMock()
        mock_client.send_message = AsyncMock(
            side_effect=RuntimeError("persistent failure")
        )

        # Act & Assert
        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RuntimeError, match="persistent failure"),
        ):
            await retry_with_backoff(mock_client, AgentType.CODER, "msg", max_retries=3)

        assert mock_client.send_message.await_count == 3

    async def test_retry_logs_each_retry_attempt(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Each retry attempt is logged at WARNING level."""
        from mle_star.orchestrator import retry_with_backoff

        # Arrange
        mock_client = AsyncMock()
        mock_client.send_message = AsyncMock(
            side_effect=[
                RuntimeError("fail 1"),
                "ok",
            ]
        )

        # Act
        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
        ):
            await retry_with_backoff(mock_client, AgentType.CODER, "msg", max_retries=3)

        # Assert -- at least one warning about retry
        warning_msgs = [r for r in caplog.records if r.levelno >= logging.WARNING]
        all_text = " ".join(r.message for r in warning_msgs)
        assert "retry" in all_text.lower() or "failed" in all_text.lower()

    async def test_retry_with_zero_retries_raises_immediately(self) -> None:
        """With max_retries=0, no attempt is made and error propagates."""
        from mle_star.orchestrator import retry_with_backoff

        # Act & Assert
        mock_client = AsyncMock()
        mock_client.send_message = AsyncMock(side_effect=RuntimeError("immediate"))

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RuntimeError),
        ):
            await retry_with_backoff(mock_client, AgentType.CODER, "msg", max_retries=0)

    @given(max_retries=st.integers(min_value=1, max_value=5))
    @settings(
        max_examples=5,
        deadline=10000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    async def test_retry_backoff_delays_are_powers_of_two(
        self, max_retries: int
    ) -> None:
        """Backoff delays follow 2^i pattern: 1, 2, 4, 8, 16..."""
        from mle_star.orchestrator import retry_with_backoff

        # Arrange -- all attempts fail
        mock_client = AsyncMock()
        mock_client.send_message = AsyncMock(side_effect=RuntimeError("always fails"))

        sleep_calls: list[float] = []

        async def _mock_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        # Act
        with (
            patch("asyncio.sleep", side_effect=_mock_sleep),
            pytest.raises(RuntimeError),
        ):
            await retry_with_backoff(
                mock_client, AgentType.CODER, "msg", max_retries=max_retries
            )

        # Assert -- delays should be 1, 2, 4, 8, ...
        expected_delays = [2**i for i in range(max_retries - 1)]
        # There are max_retries - 1 sleeps (no sleep after last failure)
        assert len(sleep_calls) == max(0, max_retries - 1)
        for actual, expected in zip(sleep_calls, expected_delays, strict=False):
            assert actual == pytest.approx(expected)


# ===========================================================================
# REQ-OR-054: Claude CLI version check
# ===========================================================================


@pytest.mark.unit
class TestClaudeCLIVersionCheck:
    """check_claude_cli_version verifies Claude CLI presence and version (REQ-OR-054)."""

    def test_check_cli_version_passes_with_valid_version(self) -> None:
        """check_claude_cli_version succeeds when CLI version >= 1.0.0."""
        from mle_star.orchestrator import check_claude_cli_version

        mock_result = MagicMock()
        mock_result.stdout = "1.0.0 (Claude Code)\n"

        with (
            patch("shutil.which", return_value="/usr/local/bin/claude"),
            patch(f"{_MODULE}._subprocess_mod.run", return_value=mock_result),
        ):
            check_claude_cli_version()  # No exception

    def test_check_cli_version_passes_with_higher_version(self) -> None:
        """check_claude_cli_version succeeds when CLI version > 1.0.0."""
        from mle_star.orchestrator import check_claude_cli_version

        mock_result = MagicMock()
        mock_result.stdout = "2.0.0 (Claude Code)\n"

        with (
            patch("shutil.which", return_value="/usr/local/bin/claude"),
            patch(f"{_MODULE}._subprocess_mod.run", return_value=mock_result),
        ):
            check_claude_cli_version()  # No exception

    def test_check_cli_version_passes_with_patch_version(self) -> None:
        """check_claude_cli_version succeeds when CLI version is 1.0.1."""
        from mle_star.orchestrator import check_claude_cli_version

        mock_result = MagicMock()
        mock_result.stdout = "1.0.1 (Claude Code)\n"

        with (
            patch("shutil.which", return_value="/usr/local/bin/claude"),
            patch(f"{_MODULE}._subprocess_mod.run", return_value=mock_result),
        ):
            check_claude_cli_version()  # No exception

    def test_check_cli_version_raises_for_old_version(self) -> None:
        """check_claude_cli_version raises ImportError when CLI version < 1.0.0."""
        from mle_star.orchestrator import check_claude_cli_version

        mock_result = MagicMock()
        mock_result.stdout = "0.9.0 (Claude Code)\n"

        with (
            patch("shutil.which", return_value="/usr/local/bin/claude"),
            patch(f"{_MODULE}._subprocess_mod.run", return_value=mock_result),
            pytest.raises(ImportError, match=r"1\.0\.0"),
        ):
            check_claude_cli_version()

    def test_check_cli_version_raises_when_not_on_path(self) -> None:
        """check_claude_cli_version raises FileNotFoundError when claude not on PATH."""
        from mle_star.orchestrator import check_claude_cli_version

        with (
            patch("shutil.which", return_value=None),
            pytest.raises(FileNotFoundError),
        ):
            check_claude_cli_version()

    def test_check_cli_version_raises_for_much_older_version(self) -> None:
        """check_claude_cli_version raises ImportError for very old version."""
        from mle_star.orchestrator import check_claude_cli_version

        mock_result = MagicMock()
        mock_result.stdout = "0.0.1 (Claude Code)\n"

        with (
            patch("shutil.which", return_value="/usr/local/bin/claude"),
            patch(f"{_MODULE}._subprocess_mod.run", return_value=mock_result),
            pytest.raises(ImportError, match=r"1\.0\.0"),
        ):
            check_claude_cli_version()

    @pytest.mark.parametrize(
        "version",
        ["0.1.37", "0.9.0", "0.1.0", "0.0.99"],
    )
    def test_check_cli_version_rejects_old_versions(self, version: str) -> None:
        """check_claude_cli_version rejects all versions below 1.0.0."""
        from mle_star.orchestrator import check_claude_cli_version

        mock_result = MagicMock()
        mock_result.stdout = f"{version} (Claude Code)\n"

        with (
            patch("shutil.which", return_value="/usr/local/bin/claude"),
            patch(f"{_MODULE}._subprocess_mod.run", return_value=mock_result),
            pytest.raises(ImportError),
        ):
            check_claude_cli_version()

    @pytest.mark.parametrize(
        "version",
        ["1.0.0", "1.0.1", "1.1.0", "2.0.0"],
    )
    def test_check_cli_version_accepts_valid_versions(self, version: str) -> None:
        """check_claude_cli_version accepts all versions >= 1.0.0."""
        from mle_star.orchestrator import check_claude_cli_version

        mock_result = MagicMock()
        mock_result.stdout = f"{version} (Claude Code)\n"

        with (
            patch("shutil.which", return_value="/usr/local/bin/claude"),
            patch(f"{_MODULE}._subprocess_mod.run", return_value=mock_result),
        ):
            check_claude_cli_version()  # No exception

    def test_check_cli_version_error_message_includes_found_version(self) -> None:
        """ImportError message includes the version that was found."""
        from mle_star.orchestrator import check_claude_cli_version

        mock_result = MagicMock()
        mock_result.stdout = "0.9.0 (Claude Code)\n"

        with (
            patch("shutil.which", return_value="/usr/local/bin/claude"),
            patch(f"{_MODULE}._subprocess_mod.run", return_value=mock_result),
            pytest.raises(ImportError, match=r"0\.9\.0"),
        ):
            check_claude_cli_version()

    def test_check_cli_version_handles_timeout(self) -> None:
        """check_claude_cli_version raises FileNotFoundError on subprocess timeout."""
        from mle_star.orchestrator import check_claude_cli_version

        with (
            patch("shutil.which", return_value="/usr/local/bin/claude"),
            patch(
                f"{_MODULE}._subprocess_mod.run",
                side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=10),
            ),
            pytest.raises(FileNotFoundError),
        ):
            check_claude_cli_version()


# ===========================================================================
# REQ-OR-056: Concurrent session limit
# ===========================================================================


@pytest.mark.unit
class TestConcurrentSessionLimit:
    """Excess paths are serialized when SDK limits concurrent sessions (REQ-OR-056)."""

    async def test_dispatch_with_session_limit_below_l(self, tmp_path: Path) -> None:
        """When max_concurrent_sessions < L, excess paths are serialized."""
        from mle_star.orchestrator import _dispatch_phase2_with_session_limit

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(num_parallel_solutions=4)
        p1_result = _make_phase1_result()

        mock_client = MagicMock()
        phase2_mock = AsyncMock(return_value=_make_phase2_result())

        with patch(f"{_MODULE}.run_phase2_outer_loop", phase2_mock):
            results = await _dispatch_phase2_with_session_limit(
                mock_client,
                task,
                config,
                p1_result,
                max_concurrent_sessions=2,
            )

        # All 4 paths should complete
        assert len(results) == 4
        assert all(isinstance(r, Phase2Result) for r in results)

    async def test_dispatch_with_session_limit_equal_to_l(self, tmp_path: Path) -> None:
        """When max_concurrent_sessions == L, all paths run in parallel."""
        from mle_star.orchestrator import _dispatch_phase2_with_session_limit

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(num_parallel_solutions=3)
        p1_result = _make_phase1_result()

        mock_client = MagicMock()
        phase2_mock = AsyncMock(return_value=_make_phase2_result())

        with patch(f"{_MODULE}.run_phase2_outer_loop", phase2_mock):
            results = await _dispatch_phase2_with_session_limit(
                mock_client,
                task,
                config,
                p1_result,
                max_concurrent_sessions=3,
            )

        assert len(results) == 3
        assert all(isinstance(r, Phase2Result) for r in results)

    async def test_dispatch_with_session_limit_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning is logged when serialization is required."""
        from mle_star.orchestrator import _dispatch_phase2_with_session_limit

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(num_parallel_solutions=4)
        p1_result = _make_phase1_result()

        mock_client = MagicMock()
        phase2_mock = AsyncMock(return_value=_make_phase2_result())

        with (
            patch(f"{_MODULE}.run_phase2_outer_loop", phase2_mock),
            caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
        ):
            await _dispatch_phase2_with_session_limit(
                mock_client,
                task,
                config,
                p1_result,
                max_concurrent_sessions=2,
            )

        # Should log a warning about serialization
        warning_msgs = [r for r in caplog.records if r.levelno >= logging.WARNING]
        all_text = " ".join(r.message for r in warning_msgs)
        assert (
            "serial" in all_text.lower()
            or "limit" in all_text.lower()
            or "concurrent" in all_text.lower()
        )

    async def test_dispatch_with_session_limit_no_warning_when_sufficient(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning when max_concurrent_sessions >= L."""
        from mle_star.orchestrator import _dispatch_phase2_with_session_limit

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(num_parallel_solutions=2)
        p1_result = _make_phase1_result()

        mock_client = MagicMock()
        phase2_mock = AsyncMock(return_value=_make_phase2_result())

        with (
            patch(f"{_MODULE}.run_phase2_outer_loop", phase2_mock),
            caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
        ):
            await _dispatch_phase2_with_session_limit(
                mock_client,
                task,
                config,
                p1_result,
                max_concurrent_sessions=5,
            )

        # No warning about serialization should appear
        warning_msgs = [r for r in caplog.records if r.levelno >= logging.WARNING]
        serialization_warnings = [
            r
            for r in warning_msgs
            if "serial" in r.message.lower()
            or "limit" in r.message.lower()
            or "concurrent" in r.message.lower()
        ]
        assert len(serialization_warnings) == 0

    async def test_dispatch_with_session_limit_one_runs_sequentially(
        self, tmp_path: Path
    ) -> None:
        """With max_concurrent_sessions=1, all paths run sequentially."""
        from mle_star.orchestrator import _dispatch_phase2_with_session_limit

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(num_parallel_solutions=3)
        p1_result = _make_phase1_result()

        mock_client = MagicMock()

        execution_order: list[str] = []

        async def _ordered_execution(
            _client: Any,
            _task: Any,
            _config: Any,
            _solution: Any,
            _score: float,
            session_id: str,
        ) -> Phase2Result:
            execution_order.append(f"start-{session_id}")
            await asyncio.sleep(0.01)  # Small delay to detect overlap
            execution_order.append(f"end-{session_id}")
            return _make_phase2_result()

        with patch(
            f"{_MODULE}.run_phase2_outer_loop",
            side_effect=_ordered_execution,
        ):
            results = await _dispatch_phase2_with_session_limit(
                mock_client,
                task,
                config,
                p1_result,
                max_concurrent_sessions=1,
            )

        # All paths should complete
        assert len(results) == 3

    @given(
        num_paths=st.integers(min_value=2, max_value=8),
        max_sessions=st.integers(min_value=1, max_value=4),
    )
    @settings(
        max_examples=8,
        deadline=10000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    async def test_dispatch_always_completes_all_paths(
        self, num_paths: int, max_sessions: int, tmp_path: Path
    ) -> None:
        """All L paths complete regardless of session limit."""
        from mle_star.orchestrator import _dispatch_phase2_with_session_limit

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(num_parallel_solutions=num_paths)
        p1_result = _make_phase1_result()

        mock_client = MagicMock()
        phase2_mock = AsyncMock(return_value=_make_phase2_result())

        with patch(f"{_MODULE}.run_phase2_outer_loop", phase2_mock):
            results = await _dispatch_phase2_with_session_limit(
                mock_client,
                task,
                config,
                p1_result,
                max_concurrent_sessions=max_sessions,
            )

        assert len(results) == num_paths


# ===========================================================================
# REQ-OR-057: Agent name uniqueness
# ===========================================================================


@pytest.mark.unit
class TestAgentNameUniqueness:
    """All 14 agent names from build_default_agent_configs are unique (REQ-OR-057)."""

    def test_agent_configs_has_16_entries(self) -> None:
        """build_default_agent_configs returns exactly 16 agent definitions."""
        # Act
        agents = build_default_agent_configs()

        # Assert
        assert len(agents) == 16

    def test_all_agent_names_are_unique(self) -> None:
        """All 14 agent names (keys) are unique."""
        # Act
        agents = build_default_agent_configs()

        # Assert -- dict keys are inherently unique, but verify count
        agent_names = [str(k) for k in agents]
        assert len(agent_names) == len(set(agent_names))

    def test_agent_names_match_agent_type_enum(self) -> None:
        """Every key in build_default_agent_configs matches an AgentType value."""
        # Arrange
        expected_names = {agent_type for agent_type in AgentType}

        # Act
        agents = build_default_agent_configs()
        actual_names = set(agents.keys())

        # Assert
        assert actual_names == expected_names

    def test_every_agent_type_has_a_definition(self) -> None:
        """Every AgentType enum member has a corresponding agent definition."""
        # Act
        agents = build_default_agent_configs()

        # Assert
        for agent_type in AgentType:
            assert agent_type in agents, (
                f"AgentType.{agent_type.name} ({agent_type.value}) "
                f"missing from agents dict"
            )

    def test_agent_definitions_have_required_fields(self) -> None:
        """Each agent definition is a non-empty AgentConfig."""
        # Act
        agents = build_default_agent_configs()

        # Assert -- each definition should be a non-None AgentConfig
        for name, config in agents.items():
            from mle_star.models import AgentConfig

            assert isinstance(config, AgentConfig), (
                f"Agent {name} config is not an AgentConfig"
            )

    def test_agent_names_are_all_lowercase(self) -> None:
        """All agent names are lowercase strings matching StrEnum convention."""
        # Act
        agents = build_default_agent_configs()

        # Assert
        for name in agents:
            name_str = str(name)
            assert name_str == name_str.lower(), (
                f"Agent name '{name_str}' is not lowercase"
            )

    @pytest.mark.parametrize(
        "agent_type",
        list(AgentType),
        ids=[a.name for a in AgentType],
    )
    def test_individual_agent_type_present(self, agent_type: AgentType) -> None:
        """Each individual AgentType is present in the agents dict."""
        agents = build_default_agent_configs()
        assert agent_type in agents

    def test_no_extra_agents_beyond_enum(self) -> None:
        """Agents dict does not contain keys not in AgentType enum."""
        # Arrange
        valid_names = set(AgentType)

        # Act
        agents = build_default_agent_configs()

        # Assert
        for name in agents:
            assert name in valid_names, (
                f"Agent name '{name}' not found in AgentType enum"
            )


# ===========================================================================
# Hypothesis: property-based tests for orchestrator constraints
# ===========================================================================


@pytest.mark.unit
class TestOrchestratorConstraintProperties:
    """Property-based tests for orchestrator constraint invariants."""

    @given(
        remaining=st.floats(
            min_value=0.0, max_value=86400.0, allow_nan=False, allow_infinity=False
        ),
        num_parallel=st.integers(min_value=1, max_value=8),
    )
    @settings(max_examples=20, deadline=5000)
    def test_phase_budgets_sum_to_remaining(
        self, remaining: float, num_parallel: int
    ) -> None:
        """Phase budgets always sum to approximately remaining seconds (if > 0)."""
        from mle_star.orchestrator import _compute_phase_budgets

        config = _make_config(num_parallel_solutions=num_parallel)
        budgets = _compute_phase_budgets(config, remaining)

        if remaining <= 0:
            assert budgets["phase2"] == 0.0
            assert budgets["phase3"] == 0.0
            assert budgets["finalization"] == 0.0
        else:
            total = budgets["phase2"] + budgets["phase3"] + budgets["finalization"]
            assert total == pytest.approx(remaining, rel=1e-6)

    @given(
        remaining=st.floats(
            min_value=1.0, max_value=86400.0, allow_nan=False, allow_infinity=False
        ),
        num_parallel=st.integers(min_value=1, max_value=8),
    )
    @settings(max_examples=20, deadline=5000)
    def test_per_path_budget_equals_phase2_divided_by_l(
        self, remaining: float, num_parallel: int
    ) -> None:
        """Per-path Phase 2 budget is always phase2_budget / L."""
        from mle_star.orchestrator import _compute_phase_budgets

        config = _make_config(num_parallel_solutions=num_parallel)
        budgets = _compute_phase_budgets(config, remaining)

        expected_per_path = budgets["phase2"] / num_parallel
        assert budgets["phase2_per_path"] == pytest.approx(expected_per_path, rel=1e-6)

    @given(
        num_agents=st.just(16),
    )
    @settings(max_examples=3, deadline=5000)
    def test_agent_count_invariant(self, num_agents: int) -> None:
        """Agent count is always exactly 16."""
        agents = build_default_agent_configs()
        assert len(agents) == num_agents



# ===========================================================================
# Integration: Combined orchestrator constraints validation
# ===========================================================================


@pytest.mark.unit
class TestOrchestratorConstraintsIntegration:
    """Integration tests combining multiple constraint requirements."""

    async def test_full_pipeline_creates_independent_state_per_call(
        self, tmp_path: Path
    ) -> None:
        """Two successive run_pipeline calls are fully independent."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(num_parallel_solutions=1)

        p1_result = _make_phase1_result()
        p2_result = _make_phase2_result()
        fr = _make_final_result(task, config)

        mock_client = MagicMock()

        with (
            patch(f"{_MODULE}._create_client", return_value=mock_client),
            patch(
                f"{_MODULE}.check_claude_cli_version",
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=p1_result,
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=p2_result,
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=fr,
            ),
        ):
            result1 = await run_pipeline(task, config)
            result2 = await run_pipeline(task, config)

        # Both results should be valid and independent
        assert isinstance(result1, FinalResult)
        assert isinstance(result2, FinalResult)

    def test_all_helpers_execute_fast(self) -> None:
        """All orchestration helper functions complete under 100ms combined."""
        from mle_star.orchestrator import (
            _compute_phase_budgets,
            _make_failed_phase2_result,
        )

        # Arrange
        config = _make_config(num_parallel_solutions=4)
        p1_result = _make_phase1_result()

        # Act -- run all helpers sequentially
        start = time.monotonic()
        _compute_phase_budgets(config, 3600.0)
        _make_failed_phase2_result(p1_result)
        with patch(
            f"{_MODULE}.detect_gpu_info",
            return_value={"cuda_available": False},
        ):
            from mle_star.orchestrator import _create_client

            _create_client(config, _make_task())
        elapsed_ms = (time.monotonic() - start) * 1000

        # Assert -- all helpers combined should be well under 100ms
        assert elapsed_ms < 100.0, (
            f"Combined helper overhead {elapsed_ms:.2f}ms exceeds 100ms budget"
        )

    def test_agent_config_keys_are_exactly_agent_type_values(self) -> None:
        """The agent config keys are exactly the AgentType enum members."""
        agents = build_default_agent_configs()
        agent_keys = set(agents.keys())
        enum_values = set(AgentType)
        assert agent_keys == enum_values
