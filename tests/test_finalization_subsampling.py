"""Tests for the finalization subsampling removal function (Task 38).

Validates ``remove_subsampling`` which implements a two-step agent pipeline:
1. Invoke A_test with variant="subsampling_extract" to identify the subsampling
   code block in the solution.
2. Invoke A_test with variant="subsampling_remove" to generate a replacement
   block without subsampling.

Tests are written TDD-first and serve as the executable specification for
REQ-FN-001, REQ-FN-003, REQ-FN-004, REQ-FN-006, REQ-FN-007, REQ-FN-008,
REQ-FN-009, and REQ-FN-039.

Refs:
    SRS 07 (Finalization), IMPLEMENTATION_PLAN.md Task 38.
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

SOLUTION_WITH_SUBSAMPLING = (
    "import pandas as pd\n"
    "df_train = pd.read_csv('./input/train.csv')\n"
    "df_train = df_train.sample(n=30000, random_state=42)\n"
    "model.fit(df_train)\n"
    'print(f"Final Validation Performance: {score}")\n'
)

SUBSAMPLING_BLOCK = "df_train = df_train.sample(n=30000, random_state=42)"

REPLACEMENT_BLOCK = "# subsampling removed"

SOLUTION_AFTER_REMOVAL = SOLUTION_WITH_SUBSAMPLING.replace(
    SUBSAMPLING_BLOCK, REPLACEMENT_BLOCK, 1
)

SOLUTION_WITHOUT_SUBSAMPLING = (
    "import pandas as pd\n"
    "df_train = pd.read_csv('./input/train.csv')\n"
    "model.fit(df_train)\n"
    'print(f"Final Validation Performance: {score}")\n'
)


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
        "content": SOLUTION_WITH_SUBSAMPLING,
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
        "description": "Predict which passengers were transported.",
        "data_dir": "/tmp/test_data",
        "output_dir": "./final",
    }
    defaults.update(overrides)
    return TaskDescription(**defaults)


def _make_extraction_response(block: str = SUBSAMPLING_BLOCK) -> str:
    """Build a fenced-code-block response wrapping the given block.

    Args:
        block: The code block to wrap in markdown fences.

    Returns:
        A markdown-formatted string with fenced code.
    """
    return f"Here is the subsampling block:\n```python\n{block}\n```"


def _make_removal_response(replacement: str = REPLACEMENT_BLOCK) -> str:
    """Build a fenced-code-block response wrapping the replacement block.

    Args:
        replacement: The replacement code to wrap in markdown fences.

    Returns:
        A markdown-formatted string with fenced code.
    """
    return f"Here is the code without subsampling:\n```python\n{replacement}\n```"


# ===========================================================================
# REQ-FN-009: remove_subsampling -- Is Async
# ===========================================================================


@pytest.mark.unit
class TestRemoveSubsamplingIsAsync:
    """remove_subsampling is an async function (REQ-FN-009)."""

    def test_is_coroutine_function(self) -> None:
        """remove_subsampling is defined as an async function."""
        from mle_star.finalization import remove_subsampling

        assert asyncio.iscoroutinefunction(remove_subsampling)


# ===========================================================================
# REQ-FN-009: remove_subsampling -- Signature and Return Type
# ===========================================================================


@pytest.mark.unit
class TestRemoveSubsamplingReturnType:
    """remove_subsampling returns a SolutionScript (REQ-FN-009)."""

    async def test_returns_solution_script_on_happy_path(self) -> None:
        """Returns a SolutionScript instance on successful removal."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        extraction_response = _make_extraction_response()
        removal_response = _make_removal_response()
        client.send_message = AsyncMock(
            side_effect=[extraction_response, removal_response]
        )

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "rendered prompt"
            mock_registry.get.return_value = mock_template

            result = await remove_subsampling(client, _make_solution(), _make_task())

        assert isinstance(result, SolutionScript)

    async def test_returns_solution_script_on_no_subsampling(self) -> None:
        """Returns a SolutionScript instance when no subsampling found."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        # Extraction returns empty block
        client.send_message = AsyncMock(return_value="```python\n\n```")

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "rendered prompt"
            mock_registry.get.return_value = mock_template

            with patch(f"{_MODULE}.extract_code_block", return_value=""):
                result = await remove_subsampling(
                    client, _make_solution(), _make_task()
                )

        assert isinstance(result, SolutionScript)


# ===========================================================================
# REQ-FN-001: Extraction uses PromptRegistry with subsampling_extract variant
# ===========================================================================


@pytest.mark.unit
class TestExtractionPromptVariant:
    """Extraction prompt is loaded via PromptRegistry with variant='subsampling_extract' (REQ-FN-001)."""

    async def test_registry_get_called_with_subsampling_extract_variant(self) -> None:
        """PromptRegistry.get is called with AgentType.TEST and variant='subsampling_extract'."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="no code found")

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            with patch(f"{_MODULE}.extract_code_block", return_value=""):
                await remove_subsampling(client, _make_solution(), _make_task())

        mock_registry.get.assert_any_call(AgentType.TEST, variant="subsampling_extract")

    async def test_extraction_template_rendered_with_final_solution(self) -> None:
        """Extraction template is rendered with final_solution variable containing solution content."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="no code found")

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            solution = _make_solution(content="unique_solution_marker_xyz")
            with patch(f"{_MODULE}.extract_code_block", return_value=""):
                await remove_subsampling(client, solution, _make_task())

        # The first render call (extraction) should have final_solution
        assert len(render_kwargs_captured) >= 1
        assert "final_solution" in render_kwargs_captured[0]
        assert (
            render_kwargs_captured[0]["final_solution"] == "unique_solution_marker_xyz"
        )

    async def test_extraction_prompt_sent_to_client(self) -> None:
        """The rendered extraction prompt is sent via client.send_message."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="no block")
        expected_prompt = "rendered extraction prompt content"

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = expected_prompt
            mock_registry.get.return_value = mock_template

            with patch(f"{_MODULE}.extract_code_block", return_value=""):
                await remove_subsampling(client, _make_solution(), _make_task())

        first_call_kwargs = client.send_message.call_args_list[0][1]
        assert first_call_kwargs.get("message") == expected_prompt


# ===========================================================================
# REQ-FN-003: Extraction response parsed via extract_code_block
# ===========================================================================


@pytest.mark.unit
class TestExtractionParsing:
    """Extraction response is parsed via extract_code_block (REQ-FN-003)."""

    async def test_extract_code_block_called_on_extraction_response(self) -> None:
        """extract_code_block is invoked on the extraction agent's response."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        extraction_raw_response = "Here is the block:\n```python\nsome_code\n```"
        client.send_message = AsyncMock(return_value=extraction_raw_response)

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="") as mock_extract,
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await remove_subsampling(client, _make_solution(), _make_task())

        # extract_code_block should have been called with the raw extraction response
        mock_extract.assert_called_once_with(extraction_raw_response)

    async def test_extracted_block_verified_as_substring_of_solution(self) -> None:
        """Extracted block must be a non-empty substring of solution.content."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        extraction_response = _make_extraction_response(SUBSAMPLING_BLOCK)
        removal_response = _make_removal_response()
        client.send_message = AsyncMock(
            side_effect=[extraction_response, removal_response]
        )

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            solution = _make_solution()
            result = await remove_subsampling(client, solution, _make_task())

        # Block was found in solution, so removal should have proceeded
        assert result.content != solution.content


# ===========================================================================
# REQ-FN-008: No-subsampling passthrough -- empty extraction
# ===========================================================================


@pytest.mark.unit
class TestNoSubsamplingPassthroughEmpty:
    """Returns original unchanged when extraction finds empty block (REQ-FN-008)."""

    async def test_empty_extraction_returns_original_unchanged(self) -> None:
        """Empty extraction result causes immediate return of original solution."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="   ")

        solution = _make_solution(content="original_content_marker")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await remove_subsampling(client, solution, _make_task())

        assert result.content == "original_content_marker"

    async def test_removal_agent_not_called_when_empty_extraction(self) -> None:
        """Removal agent is not invoked when extraction returns empty."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="nothing here")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await remove_subsampling(client, _make_solution(), _make_task())

        # Only one call for extraction, no removal call
        assert client.send_message.call_count == 1


# ===========================================================================
# REQ-FN-008: No-subsampling passthrough -- block not in solution
# ===========================================================================


@pytest.mark.unit
class TestNoSubsamplingPassthroughNotSubstring:
    """Returns original unchanged when extracted block is not in solution (REQ-FN-003)."""

    async def test_block_not_in_solution_returns_original(self) -> None:
        """When extracted block is not a substring of solution.content, return original."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="some response")

        solution = _make_solution(content="completely different content")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="not_in_solution_at_all",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await remove_subsampling(client, solution, _make_task())

        assert result.content == "completely different content"

    async def test_removal_not_invoked_when_block_not_substring(self) -> None:
        """Removal agent is not called when extracted block is not a substring."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="not_present_in_content",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await remove_subsampling(
                client,
                _make_solution(content="actual code here"),
                _make_task(),
            )

        # Only the extraction call, no removal call
        assert client.send_message.call_count == 1


# ===========================================================================
# REQ-FN-004: Removal uses PromptRegistry with subsampling_remove variant
# ===========================================================================


@pytest.mark.unit
class TestRemovalPromptVariant:
    """Removal prompt is loaded via PromptRegistry with variant='subsampling_remove' (REQ-FN-004)."""

    async def test_registry_get_called_with_subsampling_remove_variant(self) -> None:
        """PromptRegistry.get is called with AgentType.TEST and variant='subsampling_remove'."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        extraction_response = _make_extraction_response()
        removal_response = _make_removal_response()
        client.send_message = AsyncMock(
            side_effect=[extraction_response, removal_response]
        )

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await remove_subsampling(client, _make_solution(), _make_task())

        mock_registry.get.assert_any_call(AgentType.TEST, variant="subsampling_remove")

    async def test_removal_template_rendered_with_code_block_with_subsampling(
        self,
    ) -> None:
        """Removal template is rendered with code_block_with_subsampling variable."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        extraction_response = _make_extraction_response()
        removal_response = _make_removal_response()
        client.send_message = AsyncMock(
            side_effect=[extraction_response, removal_response]
        )

        render_kwargs_captured: list[dict[str, Any]] = []
        get_call_variants: list[str | None] = []

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value

            def make_template_for_variant(
                agent_type: AgentType, variant: str | None = None
            ) -> Any:
                get_call_variants.append(variant)
                mock_tmpl = MagicMock()

                def render(**kwargs: Any) -> str:
                    render_kwargs_captured.append(
                        {"variant": variant, "kwargs": kwargs}
                    )
                    return f"prompt for {variant}"

                mock_tmpl.render = render
                return mock_tmpl

            mock_registry.get.side_effect = make_template_for_variant

            await remove_subsampling(client, _make_solution(), _make_task())

        # Find the removal render call
        removal_renders = [
            c for c in render_kwargs_captured if c["variant"] == "subsampling_remove"
        ]
        assert len(removal_renders) >= 1
        assert "code_block_with_subsampling" in removal_renders[0]["kwargs"]
        assert (
            removal_renders[0]["kwargs"]["code_block_with_subsampling"]
            == SUBSAMPLING_BLOCK
        )


# ===========================================================================
# REQ-FN-006: Removal response parsed via extract_code_block
# ===========================================================================


@pytest.mark.unit
class TestRemovalParsing:
    """Removal response is parsed via extract_code_block (REQ-FN-006)."""

    async def test_extract_code_block_called_on_removal_response(self) -> None:
        """extract_code_block is called on the removal agent's response."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        extraction_raw = _make_extraction_response()
        removal_raw = _make_removal_response()
        client.send_message = AsyncMock(side_effect=[extraction_raw, removal_raw])

        extract_calls: list[str] = []

        def tracking_extract(response: str) -> str:
            extract_calls.append(response)
            if response == extraction_raw:
                return SUBSAMPLING_BLOCK
            return REPLACEMENT_BLOCK

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                side_effect=tracking_extract,
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await remove_subsampling(client, _make_solution(), _make_task())

        # extract_code_block called twice: once for extraction, once for removal
        assert len(extract_calls) == 2
        assert extract_calls[0] == extraction_raw
        assert extract_calls[1] == removal_raw


# ===========================================================================
# REQ-FN-007: Replacement via SolutionScript.replace_block
# ===========================================================================


@pytest.mark.unit
class TestReplacementViaReplaceBlock:
    """Replacement uses SolutionScript.replace_block(old, new) (REQ-FN-007)."""

    async def test_happy_path_replacement_succeeds(self) -> None:
        """Successful replacement changes solution content."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        extraction_response = _make_extraction_response()
        removal_response = _make_removal_response()
        client.send_message = AsyncMock(
            side_effect=[extraction_response, removal_response]
        )

        solution = _make_solution()

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await remove_subsampling(client, solution, _make_task())

        assert SUBSAMPLING_BLOCK not in result.content
        assert REPLACEMENT_BLOCK in result.content

    async def test_replace_block_called_with_old_and_new(self) -> None:
        """replace_block is called with the extracted block and the removal result."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        extraction_response = _make_extraction_response()
        removal_response = _make_removal_response()
        client.send_message = AsyncMock(
            side_effect=[extraction_response, removal_response]
        )

        solution = _make_solution()

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch.object(
                SolutionScript,
                "replace_block",
                wraps=solution.replace_block,
            ) as mock_replace,
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await remove_subsampling(client, solution, _make_task())

        mock_replace.assert_called_once()
        call_args = mock_replace.call_args[0]
        assert call_args[0] == SUBSAMPLING_BLOCK
        assert call_args[1] == REPLACEMENT_BLOCK


# ===========================================================================
# REQ-FN-007: replace_block ValueError -- log warning, return original
# ===========================================================================


@pytest.mark.unit
class TestReplaceBlockValueErrorHandling:
    """replace_block ValueError is caught, logged, and original returned (REQ-FN-007)."""

    async def test_value_error_returns_original(self) -> None:
        """When replace_block raises ValueError, original solution is returned."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        extraction_response = _make_extraction_response()
        removal_response = _make_removal_response()
        client.send_message = AsyncMock(
            side_effect=[extraction_response, removal_response]
        )

        solution = _make_solution()

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch.object(
                SolutionScript,
                "replace_block",
                side_effect=ValueError("Block not found"),
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await remove_subsampling(client, solution, _make_task())

        assert result.content == solution.content

    async def test_value_error_logged_as_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """ValueError from replace_block is logged at WARNING level."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        extraction_response = _make_extraction_response()
        removal_response = _make_removal_response()
        client.send_message = AsyncMock(
            side_effect=[extraction_response, removal_response]
        )

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch.object(
                SolutionScript,
                "replace_block",
                side_effect=ValueError("Block not found"),
            ),
            caplog.at_level(logging.WARNING, logger=_MODULE),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await remove_subsampling(client, _make_solution(), _make_task())

        assert any("warning" in r.levelname.lower() for r in caplog.records)

    async def test_value_error_does_not_propagate(self) -> None:
        """ValueError from replace_block does not propagate to the caller."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        extraction_response = _make_extraction_response()
        removal_response = _make_removal_response()
        client.send_message = AsyncMock(
            side_effect=[extraction_response, removal_response]
        )

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch.object(
                SolutionScript,
                "replace_block",
                side_effect=ValueError("Block not found"),
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            # Should NOT raise
            result = await remove_subsampling(client, _make_solution(), _make_task())

        assert isinstance(result, SolutionScript)


# ===========================================================================
# REQ-FN-039: Graceful degradation -- extraction agent failure
# ===========================================================================


@pytest.mark.unit
class TestGracefulDegradationExtractionFailure:
    """Extraction agent failure returns original solution (REQ-FN-039)."""

    async def test_extraction_exception_returns_original(self) -> None:
        """When extraction agent raises an exception, original is returned."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=RuntimeError("API down"))

        solution = _make_solution(content="original_code_safe")

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await remove_subsampling(client, solution, _make_task())

        assert result.content == "original_code_safe"

    async def test_extraction_timeout_returns_original(self) -> None:
        """When extraction agent raises TimeoutError, original is returned."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=TimeoutError("Timed out"))

        solution = _make_solution(content="timeout_safe_content")

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await remove_subsampling(client, solution, _make_task())

        assert result.content == "timeout_safe_content"

    async def test_exception_does_not_propagate(self) -> None:
        """General exceptions from the client do not propagate to the caller."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        client.send_message = AsyncMock(
            side_effect=Exception("Unexpected internal error")
        )

        solution = _make_solution()

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            # Should NOT raise
            result = await remove_subsampling(client, solution, _make_task())

        assert isinstance(result, SolutionScript)


# ===========================================================================
# REQ-FN-039: Graceful degradation -- removal agent failure
# ===========================================================================


@pytest.mark.unit
class TestGracefulDegradationRemovalFailure:
    """Removal agent failure returns original solution (REQ-FN-039)."""

    async def test_removal_exception_returns_original(self) -> None:
        """When removal agent raises an exception, original is returned."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        extraction_response = _make_extraction_response()
        client.send_message = AsyncMock(
            side_effect=[extraction_response, RuntimeError("Removal failed")]
        )

        solution = _make_solution(content=SOLUTION_WITH_SUBSAMPLING)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await remove_subsampling(client, solution, _make_task())

        assert result.content == SOLUTION_WITH_SUBSAMPLING

    async def test_removal_timeout_returns_original(self) -> None:
        """When removal agent raises TimeoutError, original is returned."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        extraction_response = _make_extraction_response()
        client.send_message = AsyncMock(
            side_effect=[extraction_response, TimeoutError("Timed out")]
        )

        solution = _make_solution()

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await remove_subsampling(client, solution, _make_task())

        assert result.content == solution.content


# ===========================================================================
# Agent type usage -- client.send_message called with TEST agent_type
# ===========================================================================


@pytest.mark.unit
class TestAgentTypeUsage:
    """remove_subsampling invokes the client with AgentType.TEST (REQ-FN-001, REQ-FN-004)."""

    async def test_extraction_uses_test_agent_type(self) -> None:
        """The extraction call uses str(AgentType.TEST) as agent_type."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="empty")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await remove_subsampling(client, _make_solution(), _make_task())

        call_kwargs = client.send_message.call_args_list[0][1]
        assert call_kwargs.get("agent_type") == str(AgentType.TEST)

    async def test_removal_uses_test_agent_type(self) -> None:
        """The removal call uses str(AgentType.TEST) as agent_type."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        extraction_response = _make_extraction_response()
        removal_response = _make_removal_response()
        client.send_message = AsyncMock(
            side_effect=[extraction_response, removal_response]
        )

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await remove_subsampling(client, _make_solution(), _make_task())

        # Both calls should use the TEST agent type
        for call in client.send_message.call_args_list:
            assert call[1].get("agent_type") == str(AgentType.TEST)

    async def test_exactly_two_calls_on_happy_path(self) -> None:
        """Exactly two client calls are made on the happy path: extraction + removal."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        extraction_response = _make_extraction_response()
        removal_response = _make_removal_response()
        client.send_message = AsyncMock(
            side_effect=[extraction_response, removal_response]
        )

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await remove_subsampling(client, _make_solution(), _make_task())

        assert client.send_message.call_count == 2


# ===========================================================================
# Happy path -- complete two-step pipeline
# ===========================================================================


@pytest.mark.unit
class TestHappyPathPipeline:
    """Full two-step pipeline: extraction -> removal -> replace_block."""

    async def test_complete_pipeline_replaces_subsampling(self) -> None:
        """Complete pipeline successfully removes subsampling from solution."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        extraction_response = _make_extraction_response(SUBSAMPLING_BLOCK)
        removal_response = _make_removal_response(REPLACEMENT_BLOCK)
        client.send_message = AsyncMock(
            side_effect=[extraction_response, removal_response]
        )

        solution = _make_solution(content=SOLUTION_WITH_SUBSAMPLING)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await remove_subsampling(client, solution, _make_task())

        assert SUBSAMPLING_BLOCK not in result.content
        assert REPLACEMENT_BLOCK in result.content
        # Rest of the solution preserved
        assert "import pandas as pd" in result.content
        assert "model.fit(df_train)" in result.content

    async def test_pipeline_step_order(self) -> None:
        """Extraction is invoked first, then removal."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        extraction_response = _make_extraction_response()
        removal_response = _make_removal_response()
        client.send_message = AsyncMock(
            side_effect=[extraction_response, removal_response]
        )

        get_call_variants: list[str | None] = []

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value

            def make_template_for_variant(
                agent_type: AgentType, variant: str | None = None
            ) -> Any:
                get_call_variants.append(variant)
                mock_tmpl = MagicMock()
                mock_tmpl.render.return_value = f"prompt for {variant}"
                return mock_tmpl

            mock_registry.get.side_effect = make_template_for_variant

            await remove_subsampling(client, _make_solution(), _make_task())

        assert get_call_variants[0] == "subsampling_extract"
        assert get_call_variants[1] == "subsampling_remove"

    async def test_solution_phase_preserved(self) -> None:
        """The result preserves the solution's phase from replace_block."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        extraction_response = _make_extraction_response()
        removal_response = _make_removal_response()

        for phase in (
            SolutionPhase.FINAL,
            SolutionPhase.ENSEMBLE,
            SolutionPhase.REFINED,
        ):
            client.send_message = AsyncMock(
                side_effect=[extraction_response, removal_response]
            )

            with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
                mock_registry = mock_registry_cls.return_value
                mock_template = MagicMock()
                mock_template.render.return_value = "prompt"
                mock_registry.get.return_value = mock_template

                result = await remove_subsampling(
                    client,
                    _make_solution(phase=phase),
                    _make_task(),
                )

            assert result.phase == phase


# ===========================================================================
# Edge cases
# ===========================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Edge case tests for remove_subsampling."""

    async def test_whitespace_only_extraction_treated_as_empty(self) -> None:
        """Whitespace-only extraction result is treated as no subsampling."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="   ")

        solution = _make_solution(content="my code")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="   "),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await remove_subsampling(client, solution, _make_task())

        # Whitespace-only extraction should be treated as "no subsampling"
        # (stripped to empty), or if not stripped, it won't be a meaningful
        # substring check. The function should return original.
        assert result.content == "my code"

    async def test_solution_with_score_preserved_on_passthrough(self) -> None:
        """Solution score attribute is preserved when returning original."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="no block found")

        solution = _make_solution(score=0.95)

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await remove_subsampling(client, solution, _make_task())

        assert result.score == pytest.approx(0.95)

    async def test_removal_response_without_code_fences_uses_stripped_text(
        self,
    ) -> None:
        """Removal response without fences uses entire stripped text from extract_code_block."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        extraction_response = _make_extraction_response()
        # Raw text without fences
        raw_removal = "# no subsampling needed"
        client.send_message = AsyncMock(side_effect=[extraction_response, raw_removal])

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await remove_subsampling(client, _make_solution(), _make_task())

        # extract_code_block on unfenced text returns stripped text
        # The result should contain whatever extract_code_block returned
        assert isinstance(result, SolutionScript)

    async def test_empty_solution_content(self) -> None:
        """Works correctly with empty solution content string."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="some response")

        solution = _make_solution(content="")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="some_block"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await remove_subsampling(client, solution, _make_task())

        # Block not in empty solution -> passthrough
        assert result.content == ""

    async def test_keyboard_interrupt_not_caught(self) -> None:
        """KeyboardInterrupt is not caught and propagates normally."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=KeyboardInterrupt)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            with pytest.raises(KeyboardInterrupt):
                await remove_subsampling(client, _make_solution(), _make_task())


# ===========================================================================
# Prompt template rendering integration
# ===========================================================================


@pytest.mark.unit
class TestPromptTemplateRendering:
    """Verify correct prompt template usage and variable passing."""

    async def test_extraction_prompt_renders_with_solution_content(self) -> None:
        """Extraction template render receives the full solution content."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        rendered_kwargs: list[dict[str, Any]] = []

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                rendered_kwargs.append(kwargs)
                return "prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            solution_content = "very_specific_marker_abc123"
            await remove_subsampling(
                client,
                _make_solution(content=solution_content),
                _make_task(),
            )

        assert len(rendered_kwargs) >= 1
        assert rendered_kwargs[0].get("final_solution") == solution_content

    async def test_removal_prompt_receives_extracted_block(self) -> None:
        """Removal template render receives the extracted subsampling block."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        extraction_response = _make_extraction_response()
        removal_response = _make_removal_response()
        client.send_message = AsyncMock(
            side_effect=[extraction_response, removal_response]
        )

        render_calls: list[dict[str, Any]] = []
        call_count = 0

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value

            def make_template(agent_type: AgentType, variant: str | None = None) -> Any:
                nonlocal call_count
                current_idx = call_count
                call_count += 1
                mock_tmpl = MagicMock()

                def render(**kwargs: Any) -> str:
                    render_calls.append(
                        {"index": current_idx, "variant": variant, "kwargs": kwargs}
                    )
                    return "prompt"

                mock_tmpl.render = render
                return mock_tmpl

            mock_registry.get.side_effect = make_template

            await remove_subsampling(client, _make_solution(), _make_task())

        # The removal render (second call) should have code_block_with_subsampling
        removal_renders = [
            c for c in render_calls if c["variant"] == "subsampling_remove"
        ]
        assert len(removal_renders) == 1
        assert (
            removal_renders[0]["kwargs"]["code_block_with_subsampling"]
            == SUBSAMPLING_BLOCK
        )


# ===========================================================================
# Parametrized tests
# ===========================================================================


@pytest.mark.unit
class TestRemoveSubsamplingParametrized:
    """Parametrized tests covering multiple extraction outcomes."""

    @pytest.mark.parametrize(
        "extracted_block,should_proceed_to_removal",
        [
            ("", False),
            ("   ", False),
            ("not_in_solution", False),
            (SUBSAMPLING_BLOCK, True),
        ],
        ids=[
            "empty-extraction",
            "whitespace-extraction",
            "block-not-in-solution",
            "valid-block-found",
        ],
    )
    async def test_removal_invocation_depends_on_extraction(
        self,
        extracted_block: str,
        should_proceed_to_removal: bool,
    ) -> None:
        """Removal agent is only invoked when extraction yields a valid, present block."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        if should_proceed_to_removal:
            extraction_response = _make_extraction_response(extracted_block)
            removal_response = _make_removal_response()
            client.send_message = AsyncMock(
                side_effect=[extraction_response, removal_response]
            )
        else:
            client.send_message = AsyncMock(return_value="response")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value=extracted_block,
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await remove_subsampling(client, _make_solution(), _make_task())

        if should_proceed_to_removal:
            assert client.send_message.call_count == 2
        else:
            assert client.send_message.call_count == 1

    @pytest.mark.parametrize(
        "exception_type",
        [RuntimeError, TimeoutError, ValueError, ConnectionError, OSError],
        ids=["runtime", "timeout", "value", "connection", "os"],
    )
    async def test_various_exceptions_return_original(
        self, exception_type: type[Exception]
    ) -> None:
        """Various exception types all trigger graceful degradation."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=exception_type("Test failure"))

        solution = _make_solution(content="safe_original")

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await remove_subsampling(client, solution, _make_task())

        assert result.content == "safe_original"


# ===========================================================================
# Property-based tests
# ===========================================================================


@pytest.mark.unit
class TestRemoveSubsamplingPropertyBased:
    """Property-based tests for remove_subsampling invariants."""

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                whitelist_characters="_= \n",
            ),
            min_size=10,
            max_size=200,
        ),
    )
    @settings(max_examples=30)
    async def test_no_subsampling_preserves_content(self, content: str) -> None:
        """When extraction finds nothing, content is always preserved exactly."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="no block")

        solution = _make_solution(content=content)

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await remove_subsampling(client, solution, _make_task())

        assert result.content == content

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
    @settings(max_examples=30)
    async def test_exception_always_returns_solution_script(self, content: str) -> None:
        """On any exception, a SolutionScript is always returned."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=RuntimeError("Random failure"))

        solution = _make_solution(content=content)

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await remove_subsampling(client, solution, _make_task())

        assert isinstance(result, SolutionScript)
        assert result.content == content

    @given(
        phase=st.sampled_from(list(SolutionPhase)),
    )
    @settings(max_examples=10)
    async def test_phase_preserved_across_all_phases(
        self, phase: SolutionPhase
    ) -> None:
        """Solution phase is preserved regardless of input phase value."""
        from mle_star.finalization import remove_subsampling

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        solution = _make_solution(phase=phase)

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await remove_subsampling(client, solution, _make_task())

        assert result.phase == phase

    @given(
        block=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters="_",
            ),
            min_size=3,
            max_size=50,
        ),
        prefix=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters="_\n",
            ),
            min_size=1,
            max_size=30,
        ),
        suffix=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters="_\n",
            ),
            min_size=1,
            max_size=30,
        ),
        replacement=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters="_",
            ),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(max_examples=30)
    async def test_replacement_produces_content_without_original_block(
        self,
        block: str,
        prefix: str,
        suffix: str,
        replacement: str,
    ) -> None:
        """After replacement, original block is replaced by new content."""
        from mle_star.finalization import remove_subsampling

        content = f"{prefix}\n{block}\n{suffix}"
        solution = _make_solution(content=content)

        client = AsyncMock()
        extraction_raw = _make_extraction_response(block)
        removal_raw = _make_removal_response(replacement)
        client.send_message = AsyncMock(side_effect=[extraction_raw, removal_raw])

        def mock_extract(response: str) -> str:
            if response == extraction_raw:
                return block
            return replacement

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", side_effect=mock_extract),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await remove_subsampling(client, solution, _make_task())

        # The replacement should be present and the block replaced
        assert replacement in result.content
        # Prefix and suffix should still be present
        assert prefix in result.content
        assert suffix in result.content

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters="_= \n",
            ),
            min_size=10,
            max_size=200,
        ),
    )
    @settings(max_examples=20)
    async def test_block_not_in_content_always_returns_original(
        self, content: str
    ) -> None:
        """When extracted block is not a substring of content, original is returned."""
        from mle_star.finalization import remove_subsampling

        # Use a sentinel that cannot be in the content
        unique_sentinel = f"SENTINEL_NOT_IN_CONTENT_{hash(content)}"

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="response")

        solution = _make_solution(content=content)

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value=unique_sentinel,
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await remove_subsampling(client, solution, _make_task())

        assert result.content == content
