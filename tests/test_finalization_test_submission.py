"""Tests for the finalization test submission agent function (Task 39).

Validates ``generate_test_submission`` which invokes a single A_test agent
call (default variant) to produce a final test-submission script from the
refined solution and task description.

Tests are written TDD-first and serve as the executable specification for
REQ-FN-010, REQ-FN-011, REQ-FN-013, REQ-FN-019, and REQ-FN-040.

Refs:
    SRS 08b — Finalization Test Submission.
    IMPLEMENTATION_PLAN.md Task 39.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    AgentType,
    DataModality,
    MetricDirection,
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

# ---------------------------------------------------------------------------
# Example content for tests
# ---------------------------------------------------------------------------

SAMPLE_SOLUTION_CONTENT = (
    "import pandas as pd\n"
    "df_train = pd.read_csv('./input/train.csv')\n"
    "model.fit(df_train)\n"
    'print(f"Final Validation Performance: {score}")\n'
)

SAMPLE_TEST_SUBMISSION_CODE = (
    "import pandas as pd\n"
    "df_train = pd.read_csv('./input/train.csv')\n"
    "model.fit(df_train)\n"
    "df_test = pd.read_csv('./input/test.csv')\n"
    "preds = model.predict(df_test)\n"
    "submission = pd.DataFrame({'id': df_test['id'], 'target': preds})\n"
    "submission.to_csv('./final/submission.csv', index=False)\n"
)

SAMPLE_TASK_DESCRIPTION = "Predict which passengers were transported."


# ---------------------------------------------------------------------------
# Reusable test helpers
# ---------------------------------------------------------------------------


def _make_solution(**overrides: Any) -> SolutionScript:
    """Build a valid SolutionScript with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed SolutionScript instance.
    """
    defaults: dict[str, Any] = {
        "content": SAMPLE_SOLUTION_CONTENT,
        "phase": SolutionPhase.FINAL,
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
        "description": SAMPLE_TASK_DESCRIPTION,
        "data_dir": "/tmp/test_data",
        "output_dir": "./final",
    }
    defaults.update(overrides)
    return TaskDescription(**defaults)


def _make_agent_response(code: str = SAMPLE_TEST_SUBMISSION_CODE) -> str:
    """Build a fenced-code-block response wrapping the given code.

    Args:
        code: The code to wrap in markdown fences.

    Returns:
        A markdown-formatted string with fenced code.
    """
    return f"Here is the test submission script:\n```python\n{code}\n```"


# ===========================================================================
# REQ-FN-019: generate_test_submission -- Is Async
# ===========================================================================


@pytest.mark.unit
class TestGenerateTestSubmissionIsAsync:
    """generate_test_submission is an async function (REQ-FN-019)."""

    def test_is_coroutine_function(self) -> None:
        """generate_test_submission is defined as an async function."""
        from mle_star.finalization import generate_test_submission

        assert asyncio.iscoroutinefunction(generate_test_submission)


# ===========================================================================
# REQ-FN-019: generate_test_submission -- Return Type
# ===========================================================================


@pytest.mark.unit
class TestGenerateTestSubmissionReturnType:
    """generate_test_submission returns a SolutionScript (REQ-FN-013)."""

    async def test_returns_solution_script_on_happy_path(self) -> None:
        """Returns a SolutionScript instance on successful extraction."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value=_make_agent_response(),
        )

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value=SAMPLE_TEST_SUBMISSION_CODE,
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "rendered prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_test_submission(
                client, _make_task(), _make_solution()
            )

        assert isinstance(result, SolutionScript)

    async def test_returned_solution_has_final_phase(self) -> None:
        """Returned SolutionScript has phase=SolutionPhase.FINAL (REQ-FN-013)."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value=_make_agent_response(),
        )

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value=SAMPLE_TEST_SUBMISSION_CODE,
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "rendered prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_test_submission(
                client, _make_task(), _make_solution()
            )

        assert result.phase == SolutionPhase.FINAL

    async def test_returned_solution_has_is_executable_true(self) -> None:
        """Returned SolutionScript has is_executable=True (REQ-FN-013)."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value=_make_agent_response(),
        )

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value=SAMPLE_TEST_SUBMISSION_CODE,
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "rendered prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_test_submission(
                client, _make_task(), _make_solution()
            )

        assert result.is_executable is True

    async def test_returned_solution_content_is_extracted_code(self) -> None:
        """Returned SolutionScript content equals the code extracted by extract_code_block (REQ-FN-013)."""
        from mle_star.finalization import generate_test_submission

        extracted_code = "print('hello test submission')"
        client = AsyncMock()
        client.send_message = AsyncMock(return_value="some raw response")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value=extracted_code,
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "rendered prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_test_submission(
                client, _make_task(), _make_solution()
            )

        assert result.content == extracted_code


# ===========================================================================
# REQ-FN-010, REQ-FN-011: Prompt template usage -- default variant
# ===========================================================================


@pytest.mark.unit
class TestPromptTemplateUsage:
    """Prompt is loaded via PromptRegistry with AgentType.TEST default variant (REQ-FN-010, REQ-FN-011)."""

    async def test_registry_get_called_with_test_agent_no_variant(self) -> None:
        """PromptRegistry.get is called with AgentType.TEST and no variant (default)."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="raw response")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await generate_test_submission(client, _make_task(), _make_solution())

        # Should be called with AgentType.TEST only (default variant, no variant kwarg)
        # or with variant=None explicitly. Either way, only one get call.
        mock_registry.get.assert_called_once()
        call_args = mock_registry.get.call_args
        assert call_args[0][0] == AgentType.TEST
        # Variant should be None (default) or not passed at all
        variant = call_args[1].get("variant") if call_args[1] else None
        if variant is not None:
            assert variant is None

    async def test_registry_get_not_called_with_subsampling_variants(self) -> None:
        """PromptRegistry.get is NOT called with subsampling_extract or subsampling_remove variants."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="raw response")

        get_calls: list[tuple[Any, ...]] = []

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"

            def track_get(*args: Any, **kwargs: Any) -> Any:
                get_calls.append((args, kwargs))
                return mock_template

            mock_registry.get.side_effect = track_get

            await generate_test_submission(client, _make_task(), _make_solution())

        for _args, kwargs in get_calls:
            variant = kwargs.get("variant")
            assert variant != "subsampling_extract"
            assert variant != "subsampling_remove"


# ===========================================================================
# REQ-FN-011: Template rendered with correct variables
# ===========================================================================


@pytest.mark.unit
class TestTemplateRendering:
    """Template is rendered with task_description and final_solution (REQ-FN-011)."""

    async def test_template_rendered_with_task_description(self) -> None:
        """Template render receives task_description from task.description."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        render_kwargs_captured: list[dict[str, Any]] = []

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            custom_description = "Predict house prices based on features."
            task = _make_task(description=custom_description)
            await generate_test_submission(client, task, _make_solution())

        assert len(render_kwargs_captured) == 1
        assert "task_description" in render_kwargs_captured[0]
        assert render_kwargs_captured[0]["task_description"] == custom_description

    async def test_template_rendered_with_final_solution(self) -> None:
        """Template render receives final_solution from solution.content."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        render_kwargs_captured: list[dict[str, Any]] = []

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            custom_content = "unique_solution_content_marker_xyz"
            solution = _make_solution(content=custom_content)
            await generate_test_submission(client, _make_task(), solution)

        assert len(render_kwargs_captured) == 1
        assert "final_solution" in render_kwargs_captured[0]
        assert render_kwargs_captured[0]["final_solution"] == custom_content

    async def test_template_rendered_with_both_variables(self) -> None:
        """Template render receives both task_description and final_solution."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        render_kwargs_captured: list[dict[str, Any]] = []

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            task = _make_task(description="my_task_desc")
            solution = _make_solution(content="my_solution_code")
            await generate_test_submission(client, task, solution)

        assert len(render_kwargs_captured) == 1
        kwargs = render_kwargs_captured[0]
        assert kwargs["task_description"] == "my_task_desc"
        assert kwargs["final_solution"] == "my_solution_code"


# ===========================================================================
# REQ-FN-019: Client invocation
# ===========================================================================


@pytest.mark.unit
class TestClientInvocation:
    """Rendered prompt is sent via client.send_message with AgentType.TEST (REQ-FN-019)."""

    async def test_client_called_with_rendered_prompt(self) -> None:
        """The rendered prompt is sent as the message parameter."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")
        expected_prompt = "the rendered prompt content"

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = expected_prompt
            mock_registry.get.return_value = mock_template

            await generate_test_submission(client, _make_task(), _make_solution())

        client.send_message.assert_called_once()
        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("message") == expected_prompt

    async def test_client_called_with_test_agent_type(self) -> None:
        """client.send_message is called with agent_type=str(AgentType.TEST)."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await generate_test_submission(client, _make_task(), _make_solution())

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("agent_type") == str(AgentType.TEST)

    async def test_exactly_one_client_call(self) -> None:
        """Exactly one client.send_message call per invocation."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await generate_test_submission(client, _make_task(), _make_solution())

        assert client.send_message.call_count == 1


# ===========================================================================
# REQ-FN-013: Response parsing via extract_code_block
# ===========================================================================


@pytest.mark.unit
class TestResponseParsing:
    """Response is parsed via extract_code_block (REQ-FN-013)."""

    async def test_extract_code_block_called_on_agent_response(self) -> None:
        """extract_code_block is invoked with the raw client response."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        raw_response = "Here is code:\n```python\nprint('hi')\n```"
        client.send_message = AsyncMock(return_value=raw_response)

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block", return_value="print('hi')"
            ) as mock_extract,
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await generate_test_submission(client, _make_task(), _make_solution())

        mock_extract.assert_called_once_with(raw_response)

    async def test_extracted_code_becomes_solution_content(self) -> None:
        """The string returned by extract_code_block is set as the content of the returned SolutionScript."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="raw agent response")

        extracted = "import os\nprint(os.getcwd())"

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=extracted),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_test_submission(
                client, _make_task(), _make_solution()
            )

        assert result.content == extracted


# ===========================================================================
# REQ-FN-040: Graceful degradation on empty extraction
# ===========================================================================


@pytest.mark.unit
class TestGracefulDegradationEmptyExtraction:
    """Empty/whitespace extraction logs warning, returns SolutionScript with empty content (REQ-FN-040)."""

    async def test_empty_extraction_returns_solution_with_empty_content(self) -> None:
        """Empty extraction result returns SolutionScript with empty content."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="no code here at all")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_test_submission(
                client, _make_task(), _make_solution()
            )

        assert isinstance(result, SolutionScript)
        assert result.content == ""

    async def test_whitespace_extraction_returns_solution_with_empty_content(
        self,
    ) -> None:
        """Whitespace-only extraction result returns SolutionScript with empty content."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="   \n\t  ")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="   \n\t  "),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_test_submission(
                client, _make_task(), _make_solution()
            )

        assert isinstance(result, SolutionScript)
        assert result.content.strip() == ""

    async def test_empty_extraction_still_has_final_phase(self) -> None:
        """Even on empty extraction, returned SolutionScript has phase=FINAL."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="empty response")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_test_submission(
                client, _make_task(), _make_solution()
            )

        assert result.phase == SolutionPhase.FINAL

    async def test_empty_extraction_still_has_is_executable_true(self) -> None:
        """Even on empty extraction, returned SolutionScript has is_executable=True."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="empty response")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_test_submission(
                client, _make_task(), _make_solution()
            )

        assert result.is_executable is True

    async def test_empty_extraction_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Empty extraction logs a warning message."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="empty")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
            caplog.at_level(logging.WARNING, logger=_MODULE),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await generate_test_submission(client, _make_task(), _make_solution())

        assert any("warning" in r.levelname.lower() for r in caplog.records)

    async def test_whitespace_extraction_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Whitespace-only extraction logs a warning message."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="whitespace response")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="   "),
            caplog.at_level(logging.WARNING, logger=_MODULE),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await generate_test_submission(client, _make_task(), _make_solution())

        assert any("warning" in r.levelname.lower() for r in caplog.records)


# ===========================================================================
# Exception propagation -- SDK errors propagate to caller
# ===========================================================================


@pytest.mark.unit
class TestExceptionPropagation:
    """SDK client errors propagate to the caller (caller handles fallback)."""

    async def test_runtime_error_propagates(self) -> None:
        """RuntimeError from client.send_message propagates to the caller."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=RuntimeError("API down"))

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            with pytest.raises(RuntimeError, match="API down"):
                await generate_test_submission(client, _make_task(), _make_solution())

    async def test_timeout_error_propagates(self) -> None:
        """TimeoutError from client.send_message propagates to the caller."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=TimeoutError("Timed out"))

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            with pytest.raises(TimeoutError, match="Timed out"):
                await generate_test_submission(client, _make_task(), _make_solution())

    async def test_connection_error_propagates(self) -> None:
        """ConnectionError from client.send_message propagates to the caller."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            with pytest.raises(ConnectionError, match="Connection refused"):
                await generate_test_submission(client, _make_task(), _make_solution())

    async def test_keyboard_interrupt_not_caught(self) -> None:
        """KeyboardInterrupt is not caught and propagates normally."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=KeyboardInterrupt)

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            with pytest.raises(KeyboardInterrupt):
                await generate_test_submission(client, _make_task(), _make_solution())


# ===========================================================================
# Happy path -- complete single-step pipeline
# ===========================================================================


@pytest.mark.unit
class TestHappyPathPipeline:
    """Full single-step pipeline: render -> invoke -> parse -> return."""

    async def test_complete_pipeline_produces_test_submission(self) -> None:
        """Complete pipeline successfully produces a test submission script."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value=_make_agent_response(SAMPLE_TEST_SUBMISSION_CODE),
        )

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value=SAMPLE_TEST_SUBMISSION_CODE,
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "rendered prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_test_submission(
                client, _make_task(), _make_solution()
            )

        assert result.content == SAMPLE_TEST_SUBMISSION_CODE
        assert result.phase == SolutionPhase.FINAL
        assert result.is_executable is True

    async def test_pipeline_step_order(self) -> None:
        """Steps execute in order: registry.get -> template.render -> client.send_message -> extract_code_block."""
        from mle_star.finalization import generate_test_submission

        call_order: list[str] = []
        client = AsyncMock()

        async def mock_send(**kwargs: Any) -> str:
            call_order.append("send_message")
            return "raw response"

        client.send_message = mock_send

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block") as mock_extract,
        ):
            mock_registry = mock_registry_cls.return_value

            def mock_get(*args: Any, **kwargs: Any) -> Any:
                call_order.append("registry_get")
                mock_tmpl = MagicMock()

                def mock_render(**rkwargs: Any) -> str:
                    call_order.append("template_render")
                    return "prompt"

                mock_tmpl.render = mock_render
                return mock_tmpl

            mock_registry.get.side_effect = mock_get

            def mock_extract_fn(response: str) -> str:
                call_order.append("extract_code_block")
                return "code"

            mock_extract.side_effect = mock_extract_fn

            await generate_test_submission(client, _make_task(), _make_solution())

        assert call_order == [
            "registry_get",
            "template_render",
            "send_message",
            "extract_code_block",
        ]

    async def test_input_solution_not_mutated(self) -> None:
        """Input solution is not modified by generate_test_submission."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        original_content = "original_code_here"
        solution = _make_solution(content=original_content, phase=SolutionPhase.REFINED)

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="new_code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_test_submission(client, _make_task(), solution)

        # Original solution should not be mutated
        assert solution.content == original_content
        assert solution.phase == SolutionPhase.REFINED
        # Result is a different object
        assert result is not solution
        assert result.content == "new_code"

    async def test_score_not_set_on_returned_solution(self) -> None:
        """Returned SolutionScript does not have a score set (score is None)."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_test_submission(
                client, _make_task(), _make_solution(score=0.95)
            )

        assert result.score is None


# ===========================================================================
# Parametrized tests
# ===========================================================================


@pytest.mark.unit
class TestGenerateTestSubmissionParametrized:
    """Parametrized tests covering multiple extraction outcomes."""

    @pytest.mark.parametrize(
        "extracted_code,expected_content_empty",
        [
            ("valid_code_here", False),
            ("", True),
            ("   ", True),
            ("\n\t\n", True),
            ("import pandas as pd\ndf = pd.read_csv('test.csv')", False),
        ],
        ids=[
            "valid-code",
            "empty-extraction",
            "whitespace-only",
            "newlines-tabs-only",
            "multiline-code",
        ],
    )
    async def test_extraction_result_determines_content(
        self,
        extracted_code: str,
        expected_content_empty: bool,
    ) -> None:
        """Extraction result determines whether the returned content is empty or valid."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value=extracted_code,
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_test_submission(
                client, _make_task(), _make_solution()
            )

        if expected_content_empty:
            assert result.content.strip() == ""
        else:
            assert result.content == extracted_code

    @pytest.mark.parametrize(
        "exception_type",
        [RuntimeError, TimeoutError, ValueError, ConnectionError, OSError],
        ids=["runtime", "timeout", "value", "connection", "os"],
    )
    async def test_various_exceptions_propagate(
        self, exception_type: type[Exception]
    ) -> None:
        """Various exception types all propagate to the caller."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=exception_type("Test failure"))

        with patch(f"{_MODULE}.get_registry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            with pytest.raises(exception_type, match="Test failure"):
                await generate_test_submission(client, _make_task(), _make_solution())

    @pytest.mark.parametrize(
        "input_phase",
        list(SolutionPhase),
        ids=[p.value for p in SolutionPhase],
    )
    async def test_output_always_has_final_phase_regardless_of_input(
        self, input_phase: SolutionPhase
    ) -> None:
        """Regardless of input solution phase, output always has FINAL phase."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="test code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_test_submission(
                client,
                _make_task(),
                _make_solution(phase=input_phase),
            )

        assert result.phase == SolutionPhase.FINAL


# ===========================================================================
# Edge cases
# ===========================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Edge case tests for generate_test_submission."""

    async def test_very_long_solution_content_passed_to_template(self) -> None:
        """Very long solution content is passed through without truncation."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        render_kwargs_captured: list[dict[str, Any]] = []
        long_content = "x = 1\n" * 10000

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await generate_test_submission(
                client,
                _make_task(),
                _make_solution(content=long_content),
            )

        assert render_kwargs_captured[0]["final_solution"] == long_content

    async def test_empty_solution_content(self) -> None:
        """Works correctly with empty solution content string."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="generated_code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_test_submission(
                client,
                _make_task(),
                _make_solution(content=""),
            )

        assert result.content == "generated_code"
        assert result.phase == SolutionPhase.FINAL

    async def test_empty_task_description(self) -> None:
        """Works correctly with empty task description string."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        render_kwargs_captured: list[dict[str, Any]] = []

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await generate_test_submission(
                client,
                _make_task(description=""),
                _make_solution(),
            )

        assert render_kwargs_captured[0]["task_description"] == ""

    async def test_solution_with_special_characters_in_content(self) -> None:
        """Solution content with special characters is passed correctly."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        render_kwargs_captured: list[dict[str, Any]] = []
        special_content = "x = '{curly}'\ny = \"quotes\"\nz = `backticks`\n# comment"

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await generate_test_submission(
                client,
                _make_task(),
                _make_solution(content=special_content),
            )

        assert render_kwargs_captured[0]["final_solution"] == special_content

    async def test_created_at_auto_set_on_returned_solution(self) -> None:
        """Returned SolutionScript has created_at automatically set."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_test_submission(
                client, _make_task(), _make_solution()
            )

        assert result.created_at is not None


# ===========================================================================
# Property-based tests
# ===========================================================================


@pytest.mark.unit
class TestGenerateTestSubmissionPropertyBased:
    """Property-based tests for generate_test_submission invariants."""

    @given(
        extracted_code=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                whitelist_characters="_= \n",
            ),
            min_size=1,
            max_size=200,
        ),
    )
    @settings(max_examples=30)
    async def test_extracted_code_always_becomes_content(
        self, extracted_code: str
    ) -> None:
        """Whatever extract_code_block returns is always set as the content (when non-empty)."""
        from mle_star.finalization import generate_test_submission

        # Skip whitespace-only — those trigger the empty-extraction path
        if not extracted_code.strip():
            return

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value=extracted_code,
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_test_submission(
                client, _make_task(), _make_solution()
            )

        assert result.content == extracted_code

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                whitelist_characters="_= \n",
            ),
            min_size=5,
            max_size=200,
        ),
        description=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                whitelist_characters="_= \n",
            ),
            min_size=5,
            max_size=200,
        ),
    )
    @settings(max_examples=30)
    async def test_template_always_receives_correct_variables(
        self, content: str, description: str
    ) -> None:
        """Template is always rendered with task_description and final_solution from inputs."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        render_kwargs_captured: list[dict[str, Any]] = []

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            task = _make_task(description=description)
            solution = _make_solution(content=content)
            await generate_test_submission(client, task, solution)

        assert len(render_kwargs_captured) == 1
        assert render_kwargs_captured[0]["task_description"] == description
        assert render_kwargs_captured[0]["final_solution"] == content

    @given(
        phase=st.sampled_from(list(SolutionPhase)),
    )
    @settings(max_examples=10)
    async def test_output_always_final_phase_regardless_of_input(
        self, phase: SolutionPhase
    ) -> None:
        """Output phase is always FINAL regardless of input solution phase."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_test_submission(
                client, _make_task(), _make_solution(phase=phase)
            )

        assert result.phase == SolutionPhase.FINAL

    @given(
        extracted_code=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                whitelist_characters="_= \n",
            ),
            min_size=1,
            max_size=200,
        ),
    )
    @settings(max_examples=30)
    async def test_output_always_has_is_executable_true(
        self, extracted_code: str
    ) -> None:
        """Output is_executable is always True regardless of extraction result."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value=extracted_code,
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_test_submission(
                client, _make_task(), _make_solution()
            )

        assert result.is_executable is True

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                whitelist_characters="_= \n",
            ),
            min_size=5,
            max_size=200,
        ),
    )
    @settings(max_examples=20)
    async def test_exactly_one_client_call_per_invocation(self, content: str) -> None:
        """Exactly one client.send_message call is made per invocation."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await generate_test_submission(
                client, _make_task(), _make_solution(content=content)
            )

        assert client.send_message.call_count == 1

    @given(
        score=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
    )
    @settings(max_examples=20)
    async def test_input_score_does_not_affect_output_score(
        self, score: float | None
    ) -> None:
        """Input solution score does not transfer to output solution."""
        from mle_star.finalization import generate_test_submission

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        with (
            patch(f"{_MODULE}.get_registry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="code"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_test_submission(
                client, _make_task(), _make_solution(score=score)
            )

        assert result.score is None
