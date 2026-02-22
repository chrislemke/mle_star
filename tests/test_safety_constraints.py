"""Tests for safety module constraints (Task 22).

Validates structured logging for all three safety agents, graceful
degradation behavior, agent config inclusion, PromptRegistry usage,
single module organization, and performance constraints.

Tests are written TDD-first and serve as the executable specification for
REQ-SF-032 through REQ-SF-046.

Refs:
    SRS 03d (Safety Constraints), IMPLEMENTATION_PLAN.md Task 22.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

from mle_star.models import (
    AgentType,
    EvaluationResult,
    LeakageAnswer,
    LeakageDetectionOutput,
    SolutionScript,
)
import pytest

from tests.conftest import make_config, make_solution, make_task

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_solution_with_score() -> SolutionScript:
    """Build a solution that prints a score line."""
    return make_solution(
        content='import pandas as pd\nprint(f"Final Validation Performance: {0.85}")\n'
    )


def _make_eval_success() -> EvaluationResult:
    """Build a successful evaluation result."""
    return EvaluationResult(
        score=0.85,
        stdout="Final Validation Performance: 0.85\n",
        stderr="",
        exit_code=0,
        duration_seconds=1.0,
        is_error=False,
        error_traceback=None,
    )


def _make_eval_error() -> EvaluationResult:
    """Build a failed evaluation result."""
    return EvaluationResult(
        score=None,
        stdout="",
        stderr="Traceback (most recent call last):\nValueError: boom",
        exit_code=1,
        duration_seconds=1.0,
        is_error=True,
        error_traceback="Traceback (most recent call last):\nValueError: boom",
    )


# ===========================================================================
# REQ-SF-032: Safety agent configs in build_default_agent_configs
# ===========================================================================


@pytest.mark.unit
class TestSafetyAgentConfigs:
    """Safety agent configs included in build_default_agent_configs (REQ-SF-032)."""

    def test_debugger_config_exists(self) -> None:
        """Debugger agent config is in default configs."""
        from mle_star.models import build_default_agent_configs

        configs = build_default_agent_configs()
        assert AgentType.DEBUGGER in configs

    def test_leakage_config_exists(self) -> None:
        """Leakage agent config is in default configs."""
        from mle_star.models import build_default_agent_configs

        configs = build_default_agent_configs()
        assert AgentType.LEAKAGE in configs

    def test_data_config_exists(self) -> None:
        """Data agent config is in default configs."""
        from mle_star.models import build_default_agent_configs

        configs = build_default_agent_configs()
        assert AgentType.DATA in configs


# ===========================================================================
# REQ-SF-033: All safety agents load prompts from PromptRegistry
# ===========================================================================


@pytest.mark.unit
class TestPromptRegistryUsage:
    """All safety agents load prompts from PromptRegistry (REQ-SF-033)."""

    def test_debugger_uses_registry(self) -> None:
        """Debugger prompt is loaded from PromptRegistry."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.DEBUGGER)
        assert template is not None

    def test_leakage_detection_uses_registry(self) -> None:
        """Leakage detection prompt loaded from PromptRegistry."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.LEAKAGE, variant="detection")
        assert template is not None

    def test_leakage_correction_uses_registry(self) -> None:
        """Leakage correction prompt loaded from PromptRegistry."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.LEAKAGE, variant="correction")
        assert template is not None

    def test_data_uses_registry(self) -> None:
        """Data agent prompt loaded from PromptRegistry."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.DATA)
        assert template is not None


# ===========================================================================
# REQ-SF-038: Leakage graceful degradation
# ===========================================================================


@pytest.mark.unit
class TestLeakageGracefulDegradation:
    """Leakage detection failure returns original solution (REQ-SF-038)."""

    @pytest.mark.asyncio
    async def test_malformed_detection_returns_original(self) -> None:
        """Malformed detection response returns original solution."""
        from mle_star.safety import check_and_fix_leakage

        solution = _make_solution_with_score()
        task = make_task()
        client = AsyncMock()
        client.send_message = AsyncMock(return_value="not valid json")

        result = await check_and_fix_leakage(solution, task, client)
        assert result is solution

    @pytest.mark.asyncio
    async def test_sdk_failure_returns_original(self) -> None:
        """SDK client failure returns original solution."""
        from mle_star.safety import check_and_fix_leakage

        solution = _make_solution_with_score()
        task = make_task()
        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=RuntimeError("SDK down"))

        result = await check_and_fix_leakage(solution, task, client)
        assert result is solution


# ===========================================================================
# REQ-SF-039: Data agent graceful degradation
# ===========================================================================


@pytest.mark.unit
class TestDataAgentGracefulDegradation:
    """Data agent failure returns original solution (REQ-SF-039)."""

    @pytest.mark.asyncio
    async def test_unparseable_response_returns_original(self) -> None:
        """Unparseable data agent response returns original solution."""
        from mle_star.safety import check_data_usage

        solution = _make_solution_with_score()
        task = make_task()
        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=RuntimeError("SDK error"))

        result = await check_data_usage(solution, task, client)
        assert result is solution


# ===========================================================================
# REQ-SF-040: Debugger graceful degradation
# ===========================================================================


@pytest.mark.unit
class TestDebuggerGracefulDegradation:
    """Debugger returns solution allowing retry/fallback (REQ-SF-040)."""

    def test_extract_code_block_returns_full_response_when_no_fences(self) -> None:
        """extract_code_block returns stripped response when no code fences."""
        from mle_star.safety import extract_code_block

        response = "  some plain text response  "
        result = extract_code_block(response)
        assert result == "some plain text response"


# ===========================================================================
# REQ-SF-041: Structured logging
# ===========================================================================


@pytest.mark.unit
class TestDebuggerLogging:
    """Debugger agent logs invocation events at correct levels (REQ-SF-041)."""

    @pytest.mark.asyncio
    async def test_debug_invocation_start_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Debug invocation start is logged at INFO."""
        from mle_star.safety import _invoke_debugger_agent

        solution = _make_solution_with_score()
        task = make_task()
        config = make_config()
        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value='```python\nprint("fixed")\nprint(f"Final Validation Performance: {0.9}")\n```'
        )

        with caplog.at_level(logging.INFO, logger="mle_star.safety"):
            await _invoke_debugger_agent(
                solution, "Traceback: ValueError", task, config, client
            )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_msgs) >= 1
        msg_text = " ".join(r.message for r in info_msgs)
        assert "debug" in msg_text.lower()

    @pytest.mark.asyncio
    async def test_debug_invocation_result_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Debug invocation result is logged at INFO."""
        from mle_star.safety import _invoke_debugger_agent

        solution = _make_solution_with_score()
        task = make_task()
        config = make_config()
        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value='```python\nprint("fixed code")\nprint(f"Final Validation Performance: {0.9}")\n```'
        )

        with caplog.at_level(logging.INFO, logger="mle_star.safety"):
            await _invoke_debugger_agent(
                solution, "Traceback: ValueError", task, config, client
            )

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        msg_text = " ".join(r.message for r in info_msgs)
        assert "result" in msg_text.lower() or "length" in msg_text.lower()


@pytest.mark.unit
class TestLeakageLogging:
    """Leakage agent logs detection/correction events (REQ-SF-041)."""

    @pytest.mark.asyncio
    async def test_leakage_detection_start_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Leakage detection start is logged at INFO."""
        from mle_star.safety import check_and_fix_leakage

        solution = _make_solution_with_score()
        task = make_task()
        client = AsyncMock()

        # Return no-leakage detection response
        no_leak = LeakageDetectionOutput(
            answers=[
                LeakageAnswer(
                    leakage_status="No Data Leakage",
                    code_block="some code",
                )
            ]
        )
        client.send_message = AsyncMock(return_value=no_leak.model_dump_json())

        with caplog.at_level(logging.INFO, logger="mle_star.safety"):
            await check_and_fix_leakage(solution, task, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_msgs) >= 1
        msg_text = " ".join(r.message for r in info_msgs)
        assert "leak" in msg_text.lower() or "detection" in msg_text.lower()

    @pytest.mark.asyncio
    async def test_leakage_detection_result_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Leakage detection result is logged at INFO."""
        from mle_star.safety import check_and_fix_leakage

        solution = _make_solution_with_score()
        task = make_task()
        client = AsyncMock()

        no_leak = LeakageDetectionOutput(
            answers=[
                LeakageAnswer(
                    leakage_status="No Data Leakage",
                    code_block="some code",
                )
            ]
        )
        client.send_message = AsyncMock(return_value=no_leak.model_dump_json())

        with caplog.at_level(logging.INFO, logger="mle_star.safety"):
            await check_and_fix_leakage(solution, task, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        msg_text = " ".join(r.message for r in info_msgs)
        assert "answer" in msg_text.lower() or "result" in msg_text.lower()

    @pytest.mark.asyncio
    async def test_leakage_parse_failure_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Parse failure in leakage detection logs WARNING."""
        from mle_star.safety import check_and_fix_leakage

        solution = _make_solution_with_score()
        task = make_task()
        client = AsyncMock()
        client.send_message = AsyncMock(return_value="not json at all")

        with caplog.at_level(logging.WARNING, logger="mle_star.safety"):
            await check_and_fix_leakage(solution, task, client)

        warning_msgs = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_msgs) >= 1

    @pytest.mark.asyncio
    async def test_leakage_replacement_skipped_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Replacement skip for unmatched block logs WARNING."""
        from mle_star.safety import check_and_fix_leakage

        # Solution that doesn't contain the detected code block
        solution = make_solution(content="totally different code\n")
        task = make_task()
        client = AsyncMock()

        leak = LeakageDetectionOutput(
            answers=[
                LeakageAnswer(
                    leakage_status="Yes Data Leakage",
                    code_block="nonexistent block",
                )
            ]
        )

        # First call: detection response, second call: correction response
        client.send_message = AsyncMock(
            side_effect=[
                leak.model_dump_json(),
                "```python\nfixed_code\n```",
            ]
        )

        with caplog.at_level(logging.WARNING, logger="mle_star.safety"):
            await check_and_fix_leakage(solution, task, client)

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1


@pytest.mark.unit
class TestDataAgentLogging:
    """Data agent logs check events at correct levels (REQ-SF-041)."""

    @pytest.mark.asyncio
    async def test_data_check_start_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Data check start is logged at INFO."""
        from mle_star.safety import check_data_usage

        solution = _make_solution_with_score()
        task = make_task()
        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value="All the provided information is used."
        )

        with caplog.at_level(logging.INFO, logger="mle_star.safety"):
            await check_data_usage(solution, task, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_msgs) >= 1
        msg_text = " ".join(r.message for r in info_msgs)
        assert "data" in msg_text.lower()

    @pytest.mark.asyncio
    async def test_data_check_result_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Data check result is logged at INFO."""
        from mle_star.safety import check_data_usage

        solution = _make_solution_with_score()
        task = make_task()
        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value="All the provided information is used."
        )

        with caplog.at_level(logging.INFO, logger="mle_star.safety"):
            await check_data_usage(solution, task, client)

        info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
        msg_text = " ".join(r.message for r in info_msgs)
        assert (
            "confirmed" in msg_text.lower()
            or "result" in msg_text.lower()
            or "modified" in msg_text.lower()
        )

    @pytest.mark.asyncio
    async def test_data_parse_failure_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Parse failure in data agent logs WARNING."""
        from mle_star.safety import check_data_usage

        solution = _make_solution_with_score()
        task = make_task()
        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=RuntimeError("SDK fail"))

        with caplog.at_level(logging.WARNING, logger="mle_star.safety"):
            await check_data_usage(solution, task, client)

        warning_msgs = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_msgs) >= 1


# ===========================================================================
# REQ-SF-042: SDK agent invocation
# ===========================================================================


@pytest.mark.unit
class TestSdkInvocation:
    """All safety agents use SDK agent mechanism (REQ-SF-042)."""

    @pytest.mark.asyncio
    async def test_debugger_uses_send_message(self) -> None:
        """Debugger invokes client.send_message."""
        from mle_star.safety import _invoke_debugger_agent

        solution = _make_solution_with_score()
        task = make_task()
        config = make_config()
        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value='```python\nprint("fix")\nprint(f"Final Validation Performance: {0.9}")\n```'
        )

        await _invoke_debugger_agent(solution, "Traceback: Error", task, config, client)
        client.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_leakage_uses_send_message(self) -> None:
        """Leakage agent invokes client.send_message."""
        from mle_star.safety import check_and_fix_leakage

        solution = _make_solution_with_score()
        task = make_task()
        client = AsyncMock()

        no_leak = LeakageDetectionOutput(
            answers=[
                LeakageAnswer(
                    leakage_status="No Data Leakage",
                    code_block="code",
                )
            ]
        )
        client.send_message = AsyncMock(return_value=no_leak.model_dump_json())

        await check_and_fix_leakage(solution, task, client)
        client.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_data_uses_send_message(self) -> None:
        """Data agent invokes client.send_message."""
        from mle_star.safety import check_data_usage

        solution = _make_solution_with_score()
        task = make_task()
        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value="All the provided information is used."
        )

        await check_data_usage(solution, task, client)
        client.send_message.assert_called_once()


# ===========================================================================
# REQ-SF-043: Single module organization
# ===========================================================================


@pytest.mark.unit
class TestModuleOrganization:
    """All safety functions reside in safety.py (REQ-SF-043)."""

    def test_all_safety_functions_in_one_module(self) -> None:
        """All public safety functions importable from mle_star.safety."""
        from mle_star.safety import (
            check_and_fix_leakage,
            check_data_usage,
            debug_solution,
            extract_code_block,
            make_debug_callback,
            parse_data_agent_response,
        )

        assert callable(extract_code_block)
        assert callable(debug_solution)
        assert callable(make_debug_callback)
        assert callable(check_and_fix_leakage)
        assert callable(check_data_usage)
        assert callable(parse_data_agent_response)
