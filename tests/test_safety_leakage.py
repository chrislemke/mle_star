"""Tests for the leakage detection and correction agent (Task 20).

Validates ``check_and_fix_leakage`` which implements the A_leakage agent for
detecting and correcting data leakage in solution scripts via a two-step
detection-then-correction pipeline.

Tests are written TDD-first and serve as the executable specification for
REQ-SF-011 through REQ-SF-023.

Refs:
    SRS 03b (Safety Leakage), IMPLEMENTATION_PLAN.md Task 20.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal
from unittest.mock import AsyncMock, patch

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    AgentType,
    DataModality,
    LeakageAnswer,
    LeakageDetectionOutput,
    MetricDirection,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
    TaskType,
)
from pydantic import ValidationError
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
            "import pandas as pd\n"
            "df_train = pd.read_csv('train.csv')\n"
            "df_test = pd.read_csv('test.csv')\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "scaler = StandardScaler()\n"
            "scaler.fit(df_train)\n"
            "df_train = scaler.transform(df_train)\n"
            "df_test = scaler.transform(df_test)\n"
            'print(f"Final Validation Performance: {0.85}")\n'
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


_LEAKY_CODE_BLOCK = (
    "scaler = StandardScaler()\n"
    "scaler.fit(df_train)\n"
    "df_train = scaler.transform(df_train)\n"
    "df_test = scaler.transform(df_test)"
)

_CORRECTED_CODE_BLOCK = (
    "scaler = StandardScaler()\n"
    "scaler.fit(df_train[train_idx])\n"
    "df_train = scaler.transform(df_train)\n"
    "df_test = scaler.transform(df_test)"
)

_SAFETY = "mle_star.safety"


def _make_detection_output(
    *,
    leakage: bool = True,
    code_block: str = _LEAKY_CODE_BLOCK,
) -> LeakageDetectionOutput:
    """Build a LeakageDetectionOutput with a single answer.

    Args:
        leakage: Whether the answer indicates leakage.
        code_block: The code block in the answer.

    Returns:
        A LeakageDetectionOutput with one LeakageAnswer.
    """
    status: Literal["Yes Data Leakage", "No Data Leakage"] = (
        "Yes Data Leakage" if leakage else "No Data Leakage"
    )
    return LeakageDetectionOutput(
        answers=[LeakageAnswer(leakage_status=status, code_block=code_block)]
    )


def _make_correction_response(corrected_block: str = _CORRECTED_CODE_BLOCK) -> str:
    """Build a markdown response containing a corrected code block.

    Args:
        corrected_block: The corrected code to wrap in fences.

    Returns:
        A markdown-formatted string with fenced code.
    """
    return f"Here is the corrected code:\n```python\n{corrected_block}\n```"


# ===========================================================================
# REQ-SF-011: check_and_fix_leakage -- Is Async
# ===========================================================================


@pytest.mark.unit
class TestCheckAndFixLeakageIsAsync:
    """check_and_fix_leakage is an async function (REQ-SF-011)."""

    def test_is_coroutine_function(self) -> None:
        """check_and_fix_leakage is defined as an async function."""
        from mle_star.safety import check_and_fix_leakage

        assert asyncio.iscoroutinefunction(check_and_fix_leakage)


# ===========================================================================
# REQ-SF-012: check_and_fix_leakage -- Return Type
# ===========================================================================


@pytest.mark.unit
class TestCheckAndFixLeakageReturnType:
    """check_and_fix_leakage returns a SolutionScript (REQ-SF-012)."""

    async def test_returns_solution_script_on_no_leakage(self) -> None:
        """Returns a SolutionScript instance when no leakage is detected."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=False)
        client.send_message = AsyncMock(return_value=detection.model_dump_json())

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "detection prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(_make_solution(), _make_task(), client)

        assert isinstance(result, SolutionScript)

    async def test_returns_solution_script_on_leakage_found(self) -> None:
        """Returns a SolutionScript instance when leakage is detected and corrected."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=True)
        correction_response = _make_correction_response()
        client.send_message = AsyncMock(
            side_effect=[detection.model_dump_json(), correction_response]
        )

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(_make_solution(), _make_task(), client)

        assert isinstance(result, SolutionScript)


# ===========================================================================
# REQ-SF-013: Detection uses LeakageDetectionOutput structured output
# ===========================================================================


@pytest.mark.unit
class TestDetectionStructuredOutput:
    """Detection agent uses LeakageDetectionOutput structured output (REQ-SF-013)."""

    async def test_detection_response_parsed_as_leakage_detection_output(self) -> None:
        """The detection agent response is parsed as LeakageDetectionOutput."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=False)
        client.send_message = AsyncMock(return_value=detection.model_dump_json())

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(_make_solution(), _make_task(), client)

        # No leakage => original returned unchanged
        assert result.content == _make_solution().content

    async def test_detection_with_multiple_answers(self) -> None:
        """Detection output can contain multiple LeakageAnswer entries."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = LeakageDetectionOutput(
            answers=[
                LeakageAnswer(
                    leakage_status="No Data Leakage",
                    code_block="clean_block = 1",
                ),
                LeakageAnswer(
                    leakage_status="Yes Data Leakage",
                    code_block=_LEAKY_CODE_BLOCK,
                ),
            ]
        )
        correction_response = _make_correction_response()
        client.send_message = AsyncMock(
            side_effect=[detection.model_dump_json(), correction_response]
        )

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(_make_solution(), _make_task(), client)

        # Only the leaky block should have been corrected
        assert _LEAKY_CODE_BLOCK not in result.content


# ===========================================================================
# REQ-SF-014: Detection prompt loaded from registry with variant="detection"
# ===========================================================================


@pytest.mark.unit
class TestDetectionPromptVariant:
    """Detection prompt is loaded from PromptRegistry with variant='detection' (REQ-SF-014)."""

    async def test_registry_get_called_with_detection_variant(self) -> None:
        """PromptRegistry.get is called with AgentType.LEAKAGE and variant='detection'."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=False)
        client.send_message = AsyncMock(return_value=detection.model_dump_json())

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            await check_and_fix_leakage(_make_solution(), _make_task(), client)

        mock_registry.get.assert_any_call(AgentType.LEAKAGE, variant="detection")

    async def test_detection_prompt_contains_solution_code(self) -> None:
        """The detection prompt is rendered with the solution's source code."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=False)
        client.send_message = AsyncMock(return_value=detection.model_dump_json())

        rendered_prompts: list[str] = []

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()

            def capture_render(**kwargs: Any) -> str:
                prompt = f"code={kwargs.get('code', '')}"
                rendered_prompts.append(prompt)
                return prompt

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            solution = _make_solution(content="unique_marker_abc_123")
            await check_and_fix_leakage(solution, _make_task(), client)

        # At least one render call should have contained the solution code
        assert any("unique_marker_abc_123" in p for p in rendered_prompts)


# ===========================================================================
# REQ-SF-015: Correction prompt loaded from registry with variant="correction"
# ===========================================================================


@pytest.mark.unit
class TestCorrectionPromptVariant:
    """Correction prompt is loaded from PromptRegistry with variant='correction' (REQ-SF-015)."""

    async def test_registry_get_called_with_correction_variant(self) -> None:
        """PromptRegistry.get is called with AgentType.LEAKAGE and variant='correction'."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=True)
        correction_response = _make_correction_response()
        client.send_message = AsyncMock(
            side_effect=[detection.model_dump_json(), correction_response]
        )

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            await check_and_fix_leakage(_make_solution(), _make_task(), client)

        mock_registry.get.assert_any_call(AgentType.LEAKAGE, variant="correction")

    async def test_correction_prompt_rendered_with_current_solution(self) -> None:
        """The correction prompt is rendered with the current solution's code."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=True)
        correction_response = _make_correction_response()
        client.send_message = AsyncMock(
            side_effect=[detection.model_dump_json(), correction_response]
        )

        render_calls: list[dict[str, Any]] = []

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value

            def make_template_for_variant(
                agent_type: AgentType, variant: str | None = None
            ) -> Any:
                mock_tmpl = AsyncMock()

                def render(**kwargs: Any) -> str:
                    render_calls.append({"variant": variant, "kwargs": kwargs})
                    return f"prompt for {variant}"

                mock_tmpl.render = render
                return mock_tmpl

            mock_registry.get.side_effect = make_template_for_variant

            await check_and_fix_leakage(_make_solution(), _make_task(), client)

        correction_renders = [c for c in render_calls if c["variant"] == "correction"]
        assert len(correction_renders) >= 1
        # The correction render should contain the code kwarg
        assert "code" in correction_renders[0]["kwargs"]


# ===========================================================================
# REQ-SF-016: Returns original solution when no leakage found
# ===========================================================================


@pytest.mark.unit
class TestNoLeakageReturnsOriginal:
    """check_and_fix_leakage returns original when no leakage found (REQ-SF-016)."""

    async def test_returns_original_content_unchanged(self) -> None:
        """Solution content is unchanged when no leakage detected."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=False)
        client.send_message = AsyncMock(return_value=detection.model_dump_json())

        solution = _make_solution(content="perfectly_clean_code")

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(solution, _make_task(), client)

        assert result.content == "perfectly_clean_code"

    async def test_correction_agent_not_invoked_when_no_leakage(self) -> None:
        """The correction agent is not called when no leakage is detected."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=False)
        client.send_message = AsyncMock(return_value=detection.model_dump_json())

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            await check_and_fix_leakage(_make_solution(), _make_task(), client)

        # Only one call for detection, no correction call
        assert client.send_message.call_count == 1

    async def test_all_answers_no_leakage_returns_original(self) -> None:
        """When multiple answers all say 'No Data Leakage', original is returned."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = LeakageDetectionOutput(
            answers=[
                LeakageAnswer(leakage_status="No Data Leakage", code_block="block_a"),
                LeakageAnswer(leakage_status="No Data Leakage", code_block="block_b"),
            ]
        )
        client.send_message = AsyncMock(return_value=detection.model_dump_json())

        solution = _make_solution(content="all_clean_code")

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(solution, _make_task(), client)

        assert result.content == "all_clean_code"
        assert client.send_message.call_count == 1


# ===========================================================================
# REQ-SF-017: Returns corrected solution when leakage found
# ===========================================================================


@pytest.mark.unit
class TestLeakageFoundReturnsCorrected:
    """check_and_fix_leakage returns corrected solution when leakage found (REQ-SF-017)."""

    async def test_content_differs_from_original_when_corrected(self) -> None:
        """The returned solution's content differs from the original when corrected."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=True)
        correction_response = _make_correction_response()
        client.send_message = AsyncMock(
            side_effect=[detection.model_dump_json(), correction_response]
        )

        original = _make_solution()

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(original, _make_task(), client)

        assert result.content != original.content

    async def test_corrected_content_contains_replacement(self) -> None:
        """The corrected solution contains the replacement code block."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=True)
        correction_response = _make_correction_response()
        client.send_message = AsyncMock(
            side_effect=[detection.model_dump_json(), correction_response]
        )

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(_make_solution(), _make_task(), client)

        assert _CORRECTED_CODE_BLOCK in result.content

    async def test_preserves_solution_phase(self) -> None:
        """The returned solution preserves the original phase."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=True)
        correction_response = _make_correction_response()

        for phase in (
            SolutionPhase.INIT,
            SolutionPhase.REFINED,
            SolutionPhase.ENSEMBLE,
        ):
            client.send_message = AsyncMock(
                side_effect=[detection.model_dump_json(), correction_response]
            )

            with patch(
                f"{_SAFETY}.PromptRegistry",
            ) as mock_registry_cls:
                mock_registry = mock_registry_cls.return_value
                mock_template = AsyncMock()
                mock_template.render = lambda **kwargs: "prompt"
                mock_registry.get.return_value = mock_template

                result = await check_and_fix_leakage(
                    _make_solution(phase=phase), _make_task(), client
                )

            assert result.phase == phase

    async def test_correction_invoked_per_leaky_answer(self) -> None:
        """Correction agent is invoked once for each 'Yes Data Leakage' answer."""
        from mle_star.safety import check_and_fix_leakage

        leaky_block_a = "leaky_block_a = fit_transform(all_data)"
        leaky_block_b = "leaky_block_b = normalize(all_data)"
        corrected_a = "corrected_a = fit_transform(train_data)"
        corrected_b = "corrected_b = normalize(train_data)"

        solution_content = (
            "import pandas as pd\n"
            f"{leaky_block_a}\n"
            f"{leaky_block_b}\n"
            'print(f"Final Validation Performance: {{0.85}}")\n'
        )

        client = AsyncMock()
        detection = LeakageDetectionOutput(
            answers=[
                LeakageAnswer(
                    leakage_status="Yes Data Leakage",
                    code_block=leaky_block_a,
                ),
                LeakageAnswer(
                    leakage_status="No Data Leakage",
                    code_block="some_clean_block",
                ),
                LeakageAnswer(
                    leakage_status="Yes Data Leakage",
                    code_block=leaky_block_b,
                ),
            ]
        )
        client.send_message = AsyncMock(
            side_effect=[
                detection.model_dump_json(),
                _make_correction_response(corrected_a),
                _make_correction_response(corrected_b),
            ]
        )

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(
                _make_solution(content=solution_content), _make_task(), client
            )

        # 1 detection call + 2 correction calls = 3
        assert client.send_message.call_count == 3
        assert corrected_a in result.content
        assert corrected_b in result.content


# ===========================================================================
# REQ-SF-018: Correction uses SolutionScript.replace_block for targeted fix
# ===========================================================================


@pytest.mark.unit
class TestCorrectionUsesReplaceBlock:
    """Correction uses SolutionScript.replace_block() for targeted fix (REQ-SF-018)."""

    async def test_replace_block_called_with_leaky_and_corrected(self) -> None:
        """replace_block is called with the leaky code_block and the corrected block."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=True)
        correction_response = _make_correction_response()
        client.send_message = AsyncMock(
            side_effect=[detection.model_dump_json(), correction_response]
        )

        original = _make_solution()

        with (
            patch(
                f"{_SAFETY}.PromptRegistry",
            ) as mock_registry_cls,
            patch.object(
                SolutionScript,
                "replace_block",
                wraps=original.replace_block,
            ) as mock_replace,
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            await check_and_fix_leakage(original, _make_task(), client)

        mock_replace.assert_called_once()
        call_args = mock_replace.call_args
        assert call_args[0][0] == _LEAKY_CODE_BLOCK
        assert call_args[0][1] == _CORRECTED_CODE_BLOCK


# ===========================================================================
# REQ-SF-019: Correction uses extract_code_block on free-form response
# ===========================================================================


@pytest.mark.unit
class TestCorrectionUsesExtractCodeBlock:
    """Correction step extracts code from free-form response via extract_code_block (REQ-SF-019)."""

    async def test_extract_code_block_called_for_correction(self) -> None:
        """extract_code_block is invoked on the correction agent's response."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=True)
        correction_response = _make_correction_response()
        client.send_message = AsyncMock(
            side_effect=[detection.model_dump_json(), correction_response]
        )

        with (
            patch(
                f"{_SAFETY}.PromptRegistry",
            ) as mock_registry_cls,
            patch(
                f"{_SAFETY}.extract_code_block",
                return_value=_CORRECTED_CODE_BLOCK,
            ) as mock_extract,
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            await check_and_fix_leakage(_make_solution(), _make_task(), client)

        mock_extract.assert_called_once_with(correction_response)


# ===========================================================================
# REQ-SF-020: Agent sends to LEAKAGE agent_type
# ===========================================================================


@pytest.mark.unit
class TestAgentTypeUsed:
    """check_and_fix_leakage invokes the client with AgentType.LEAKAGE (REQ-SF-020)."""

    async def test_detection_uses_leakage_agent_type(self) -> None:
        """The detection call uses str(AgentType.LEAKAGE) as agent_type."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=False)
        client.send_message = AsyncMock(return_value=detection.model_dump_json())

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            await check_and_fix_leakage(_make_solution(), _make_task(), client)

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("agent_type") == str(AgentType.LEAKAGE)

    async def test_correction_uses_leakage_agent_type(self) -> None:
        """The correction call also uses str(AgentType.LEAKAGE) as agent_type."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=True)
        correction_response = _make_correction_response()
        client.send_message = AsyncMock(
            side_effect=[detection.model_dump_json(), correction_response]
        )

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            await check_and_fix_leakage(_make_solution(), _make_task(), client)

        # Both calls should use the leakage agent type
        for call in client.send_message.call_args_list:
            assert call[1].get("agent_type") == str(AgentType.LEAKAGE)


# ===========================================================================
# REQ-SF-021: replace_block ValueError caught, logged as warning, skipped
# ===========================================================================


@pytest.mark.unit
class TestReplaceBlockValueErrorHandling:
    """replace_block ValueError is caught, logged, and skipped (REQ-SF-021)."""

    async def test_value_error_logged_as_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """ValueError from replace_block is logged at WARNING level."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        # Detection finds leakage in a block that does NOT exist in the solution
        nonexistent_block = "this_block_does_not_exist_in_solution"
        detection = _make_detection_output(leakage=True, code_block=nonexistent_block)
        correction_response = _make_correction_response("corrected_code")
        client.send_message = AsyncMock(
            side_effect=[detection.model_dump_json(), correction_response]
        )

        with (
            patch(
                f"{_SAFETY}.PromptRegistry",
            ) as mock_registry_cls,
            caplog.at_level(logging.WARNING, logger=_SAFETY),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            await check_and_fix_leakage(_make_solution(), _make_task(), client)

        # Should log a warning about the block not found
        assert any("warning" in r.levelname.lower() for r in caplog.records)

    async def test_value_error_returns_solution_unchanged(self) -> None:
        """When replace_block raises ValueError, solution is returned unchanged."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        nonexistent_block = "not_in_content_at_all"
        detection = _make_detection_output(leakage=True, code_block=nonexistent_block)
        correction_response = _make_correction_response("some_correction")
        client.send_message = AsyncMock(
            side_effect=[detection.model_dump_json(), correction_response]
        )

        original = _make_solution()

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(original, _make_task(), client)

        # Original content should be preserved since replace_block failed
        assert result.content == original.content

    async def test_value_error_does_not_propagate(self) -> None:
        """ValueError from replace_block does not propagate to the caller."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        nonexistent_block = "missing_block"
        detection = _make_detection_output(leakage=True, code_block=nonexistent_block)
        correction_response = _make_correction_response("fix")
        client.send_message = AsyncMock(
            side_effect=[detection.model_dump_json(), correction_response]
        )

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            # Should NOT raise
            result = await check_and_fix_leakage(_make_solution(), _make_task(), client)

        assert isinstance(result, SolutionScript)

    async def test_subsequent_corrections_proceed_after_value_error(self) -> None:
        """After a failed replace_block, later corrections still proceed."""
        from mle_star.safety import check_and_fix_leakage

        missing_block = "not_in_content"
        present_block = _LEAKY_CODE_BLOCK

        client = AsyncMock()
        detection = LeakageDetectionOutput(
            answers=[
                LeakageAnswer(
                    leakage_status="Yes Data Leakage",
                    code_block=missing_block,
                ),
                LeakageAnswer(
                    leakage_status="Yes Data Leakage",
                    code_block=present_block,
                ),
            ]
        )
        correction_for_missing = _make_correction_response("fix_missing")
        correction_for_present = _make_correction_response(_CORRECTED_CODE_BLOCK)
        client.send_message = AsyncMock(
            side_effect=[
                detection.model_dump_json(),
                correction_for_missing,
                correction_for_present,
            ]
        )

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(_make_solution(), _make_task(), client)

        # The second correction (for the present block) should succeed
        assert _CORRECTED_CODE_BLOCK in result.content
        # 1 detection + 2 corrections
        assert client.send_message.call_count == 3


# ===========================================================================
# REQ-SF-022: Graceful degradation on agent failure
# ===========================================================================


@pytest.mark.unit
class TestGracefulDegradationOnFailure:
    """Agent failure returns original solution (graceful degradation) (REQ-SF-022)."""

    async def test_detection_exception_returns_original(self) -> None:
        """When detection agent raises an exception, original solution is returned."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=RuntimeError("API down"))

        solution = _make_solution(content="original_code_here")

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(solution, _make_task(), client)

        assert result.content == "original_code_here"

    async def test_correction_exception_returns_original(self) -> None:
        """When correction agent raises an exception, original solution is returned."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=True)
        client.send_message = AsyncMock(
            side_effect=[
                detection.model_dump_json(),
                RuntimeError("Correction failed"),
            ]
        )

        solution = _make_solution()

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(solution, _make_task(), client)

        assert result.content == solution.content

    async def test_invalid_json_detection_returns_original(self) -> None:
        """When detection response is invalid JSON, original solution is returned."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="not valid json {{{")

        solution = _make_solution(content="my_code_content")

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(solution, _make_task(), client)

        assert result.content == "my_code_content"

    async def test_malformed_detection_schema_returns_original(self) -> None:
        """When detection response has valid JSON but wrong schema, original is returned."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        # Valid JSON but missing required 'answers' field
        client.send_message = AsyncMock(
            return_value=json.dumps({"wrong_field": "data"})
        )

        solution = _make_solution(content="safe_code")

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(solution, _make_task(), client)

        assert result.content == "safe_code"

    async def test_timeout_error_returns_original(self) -> None:
        """When client raises TimeoutError, original solution is returned."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=TimeoutError("Timed out"))

        solution = _make_solution(content="timeout_safe_code")

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(solution, _make_task(), client)

        assert result.content == "timeout_safe_code"

    async def test_keyboard_interrupt_not_caught(self) -> None:
        """KeyboardInterrupt is not caught and propagates normally."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=KeyboardInterrupt)

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            with pytest.raises(KeyboardInterrupt):
                await check_and_fix_leakage(_make_solution(), _make_task(), client)

    async def test_exception_does_not_propagate(self) -> None:
        """General exceptions from the client do not propagate to the caller."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        client.send_message = AsyncMock(
            side_effect=Exception("Unexpected internal error")
        )

        solution = _make_solution()

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            # Should NOT raise
            result = await check_and_fix_leakage(solution, _make_task(), client)

        assert isinstance(result, SolutionScript)


# ===========================================================================
# REQ-SF-023: Detection prompt rendered with solution code
# ===========================================================================


@pytest.mark.unit
class TestDetectionPromptRendering:
    """Detection prompt is rendered with the solution's code (REQ-SF-023)."""

    async def test_detection_message_sent_to_client(self) -> None:
        """The rendered detection prompt is sent via client.send_message."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=False)
        client.send_message = AsyncMock(return_value=detection.model_dump_json())

        expected_prompt = "rendered detection prompt with code"

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: expected_prompt
            mock_registry.get.return_value = mock_template

            await check_and_fix_leakage(_make_solution(), _make_task(), client)

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("message") == expected_prompt


# ===========================================================================
# Two-step pipeline integration tests
# ===========================================================================


@pytest.mark.unit
class TestTwoStepPipelineFlow:
    """Verifies the complete two-step detection-then-correction pipeline."""

    async def test_detection_then_correction_sequence(self) -> None:
        """Detection is invoked first, then correction for each leaky block."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=True)
        correction_response = _make_correction_response()
        client.send_message = AsyncMock(
            side_effect=[detection.model_dump_json(), correction_response]
        )

        call_order: list[str] = []

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value

            def make_template(agent_type: AgentType, variant: str | None = None) -> Any:
                mock_tmpl = AsyncMock()

                def render(**kwargs: Any) -> str:
                    call_order.append(variant or "default")
                    return f"prompt for {variant}"

                mock_tmpl.render = render
                return mock_tmpl

            mock_registry.get.side_effect = make_template

            await check_and_fix_leakage(_make_solution(), _make_task(), client)

        assert call_order[0] == "detection"
        assert "correction" in call_order[1:]

    async def test_no_correction_when_all_clean(self) -> None:
        """No correction step runs when all blocks are clean."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=False)
        client.send_message = AsyncMock(return_value=detection.model_dump_json())

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            await check_and_fix_leakage(_make_solution(), _make_task(), client)

        assert client.send_message.call_count == 1

    async def test_multiple_leaky_blocks_corrected_sequentially(self) -> None:
        """Multiple leaky blocks are each corrected in sequence."""
        from mle_star.safety import check_and_fix_leakage

        block_a = "scaler.fit(df_train)"
        block_b = "df_test = scaler.transform(df_test)"
        fix_a = "scaler.fit(df_train[train_idx])"
        fix_b = "df_test = scaler.transform(df_test)  # fixed"

        client = AsyncMock()
        detection = LeakageDetectionOutput(
            answers=[
                LeakageAnswer(
                    leakage_status="Yes Data Leakage",
                    code_block=block_a,
                ),
                LeakageAnswer(
                    leakage_status="Yes Data Leakage",
                    code_block=block_b,
                ),
            ]
        )
        client.send_message = AsyncMock(
            side_effect=[
                detection.model_dump_json(),
                _make_correction_response(fix_a),
                _make_correction_response(fix_b),
            ]
        )

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(_make_solution(), _make_task(), client)

        # Both corrections applied
        assert fix_a in result.content
        assert fix_b in result.content
        # 1 detection + 2 corrections
        assert client.send_message.call_count == 3


# ===========================================================================
# Edge cases
# ===========================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Edge case tests for check_and_fix_leakage."""

    async def test_empty_solution_content(self) -> None:
        """Works correctly with an empty solution content string."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=False, code_block="")
        client.send_message = AsyncMock(return_value=detection.model_dump_json())

        solution = _make_solution(content="")

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(solution, _make_task(), client)

        assert isinstance(result, SolutionScript)

    async def test_solution_with_score_preserved(self) -> None:
        """Solution score attribute is preserved through the pipeline."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=False)
        client.send_message = AsyncMock(return_value=detection.model_dump_json())

        solution = _make_solution(score=0.92)

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(solution, _make_task(), client)

        assert result.score == pytest.approx(0.92)

    async def test_correction_response_without_code_fences(self) -> None:
        """Correction response without fences uses entire stripped text as code."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=True)
        # No markdown fences -- extract_code_block returns stripped text
        raw_correction = _CORRECTED_CODE_BLOCK
        client.send_message = AsyncMock(
            side_effect=[detection.model_dump_json(), raw_correction]
        )

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(_make_solution(), _make_task(), client)

        assert _CORRECTED_CODE_BLOCK in result.content


# ===========================================================================
# Property-based tests
# ===========================================================================


@pytest.mark.unit
class TestLeakageDetectionPropertyBased:
    """Property-based tests for leakage detection and correction invariants."""

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
    async def test_no_leakage_returns_identical_content(self, content: str) -> None:
        """When detection finds no leakage, returned content equals original."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=False, code_block="")
        client.send_message = AsyncMock(return_value=detection.model_dump_json())

        solution = _make_solution(content=content)

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(solution, _make_task(), client)

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
        """On any exception, a SolutionScript is always returned (graceful degradation)."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        client.send_message = AsyncMock(side_effect=RuntimeError("Random failure"))

        solution = _make_solution(content=content)

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(solution, _make_task(), client)

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
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        detection = _make_detection_output(leakage=False)
        client.send_message = AsyncMock(return_value=detection.model_dump_json())

        solution = _make_solution(phase=phase)

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(solution, _make_task(), client)

        assert result.phase == phase

    @given(
        n_clean=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=10)
    async def test_all_clean_answers_means_no_correction(self, n_clean: int) -> None:
        """N clean answers results in zero correction calls."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        answers = [
            LeakageAnswer(
                leakage_status="No Data Leakage",
                code_block=f"clean_block_{i}",
            )
            for i in range(n_clean)
        ]
        detection = LeakageDetectionOutput(answers=answers)
        client.send_message = AsyncMock(return_value=detection.model_dump_json())

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            await check_and_fix_leakage(_make_solution(), _make_task(), client)

        # Only 1 call for detection, no correction calls
        assert client.send_message.call_count == 1


# ===========================================================================
# Model validation tests (LeakageAnswer, LeakageDetectionOutput)
# ===========================================================================


@pytest.mark.unit
class TestLeakageModels:
    """Validate LeakageAnswer and LeakageDetectionOutput models."""

    def test_leakage_answer_yes(self) -> None:
        """LeakageAnswer accepts 'Yes Data Leakage' status."""
        answer = LeakageAnswer(
            leakage_status="Yes Data Leakage",
            code_block="some code",
        )
        assert answer.leakage_status == "Yes Data Leakage"

    def test_leakage_answer_no(self) -> None:
        """LeakageAnswer accepts 'No Data Leakage' status."""
        answer = LeakageAnswer(
            leakage_status="No Data Leakage",
            code_block="safe code",
        )
        assert answer.leakage_status == "No Data Leakage"

    def test_leakage_answer_invalid_status_rejected(self) -> None:
        """LeakageAnswer rejects invalid leakage_status values."""
        with pytest.raises(ValidationError):
            LeakageAnswer(
                leakage_status="Maybe",  # type: ignore[arg-type]
                code_block="code",
            )

    def test_leakage_detection_output_empty_answers_rejected(self) -> None:
        """LeakageDetectionOutput rejects an empty answers list."""
        with pytest.raises(ValueError, match="at least 1"):
            LeakageDetectionOutput(answers=[])

    def test_leakage_detection_output_single_answer(self) -> None:
        """LeakageDetectionOutput accepts a single answer."""
        output = LeakageDetectionOutput(
            answers=[
                LeakageAnswer(
                    leakage_status="No Data Leakage",
                    code_block="code",
                )
            ]
        )
        assert len(output.answers) == 1

    def test_leakage_detection_output_multiple_answers(self) -> None:
        """LeakageDetectionOutput accepts multiple answers."""
        output = LeakageDetectionOutput(
            answers=[
                LeakageAnswer(
                    leakage_status="No Data Leakage",
                    code_block="block1",
                ),
                LeakageAnswer(
                    leakage_status="Yes Data Leakage",
                    code_block="block2",
                ),
            ]
        )
        assert len(output.answers) == 2

    def test_leakage_answer_frozen(self) -> None:
        """LeakageAnswer is immutable (frozen)."""
        answer = LeakageAnswer(
            leakage_status="No Data Leakage",
            code_block="code",
        )
        with pytest.raises(ValidationError):
            answer.code_block = "new_code"  # type: ignore[misc]

    def test_leakage_detection_output_frozen(self) -> None:
        """LeakageDetectionOutput is immutable (frozen)."""
        output = _make_detection_output(leakage=False)
        with pytest.raises(ValidationError):
            output.answers = []  # type: ignore[misc]


# ===========================================================================
# Prompt template integration tests
# ===========================================================================


@pytest.mark.unit
class TestLeakagePromptTemplates:
    """Validate that leakage prompt templates exist and render correctly."""

    def test_detection_template_exists_in_registry(self) -> None:
        """PromptRegistry contains a leakage template with variant='detection'."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.LEAKAGE, variant="detection")
        assert template.agent_type == AgentType.LEAKAGE

    def test_correction_template_exists_in_registry(self) -> None:
        """PromptRegistry contains a leakage template with variant='correction'."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.LEAKAGE, variant="correction")
        assert template.agent_type == AgentType.LEAKAGE

    def test_detection_template_has_code_variable(self) -> None:
        """Detection template declares 'code' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.LEAKAGE, variant="detection")
        assert "code" in template.variables

    def test_correction_template_has_code_variable(self) -> None:
        """Correction template declares 'code' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.LEAKAGE, variant="correction")
        assert "code" in template.variables

    def test_detection_template_renders_with_code(self) -> None:
        """Detection template renders successfully with code kwarg."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.LEAKAGE, variant="detection")
        rendered = template.render(code="print('hello')")
        assert "print('hello')" in rendered

    def test_correction_template_renders_with_code(self) -> None:
        """Correction template renders successfully with code kwarg."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.LEAKAGE, variant="correction")
        rendered = template.render(code="print('hello')")
        assert "print('hello')" in rendered

    def test_detection_template_mentions_leakage(self) -> None:
        """Detection template content mentions data leakage checking."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.LEAKAGE, variant="detection")
        rendered = template.render(code="x = 1")
        assert "leakage" in rendered.lower()

    def test_correction_template_mentions_leakage(self) -> None:
        """Correction template content mentions preventing data leakage."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.LEAKAGE, variant="correction")
        rendered = template.render(code="x = 1")
        assert "leakage" in rendered.lower()


# ===========================================================================
# _parse_leakage_output — empty / whitespace / free-text handling
# ===========================================================================


@pytest.mark.unit
class TestParseLeakageOutputEmptyResponse:
    """_parse_leakage_output handles empty and whitespace responses."""

    def test_empty_response_defaults_to_no_leakage(self) -> None:
        """Empty string response defaults to 'No Data Leakage'."""
        from mle_star.safety import _parse_leakage_output

        result = _parse_leakage_output("")
        assert len(result.answers) == 1
        assert result.answers[0].leakage_status == "No Data Leakage"
        assert result.answers[0].code_block == "(empty response)"

    def test_whitespace_response_defaults_to_no_leakage(self) -> None:
        """Whitespace-only response defaults to 'No Data Leakage'."""
        from mle_star.safety import _parse_leakage_output

        result = _parse_leakage_output("   \n\t  ")
        assert len(result.answers) == 1
        assert result.answers[0].leakage_status == "No Data Leakage"

    def test_none_response_defaults_to_no_leakage(self) -> None:
        """None-like empty response defaults to 'No Data Leakage'."""
        from mle_star.safety import _parse_leakage_output

        result = _parse_leakage_output("")
        assert result.answers[0].leakage_status == "No Data Leakage"


@pytest.mark.unit
class TestParseLeakageOutputFreeTextPositive:
    """_parse_leakage_output handles free-text positive leakage responses."""

    def test_free_text_positive_leakage_parsed(self) -> None:
        """Free-text 'data leakage detected' is parsed as positive leakage."""
        from mle_star.safety import _parse_leakage_output

        response = (
            "After reviewing the code, I found that there is data leakage detected "
            "in the preprocessing pipeline. The scaler is fit on all data."
        )
        result = _parse_leakage_output(response)
        assert len(result.answers) == 1
        assert result.answers[0].leakage_status == "Yes Data Leakage"

    def test_free_text_with_code_block_extracts_code(self) -> None:
        """Free-text with a code fence extracts the code block."""
        from mle_star.safety import _parse_leakage_output

        response = (
            "Data leakage detected in the following code:\n"
            "```python\n"
            "scaler.fit(all_data)\n"
            "```\n"
            "The scaler should only be fit on training data."
        )
        result = _parse_leakage_output(response)
        assert result.answers[0].leakage_status == "Yes Data Leakage"
        assert result.answers[0].code_block == "scaler.fit(all_data)"

    def test_free_text_without_code_block_uses_placeholder(self) -> None:
        """Free-text without code fence uses placeholder code block."""
        from mle_star.safety import _parse_leakage_output

        response = "Yes, there is data leakage detected in the code."
        result = _parse_leakage_output(response)
        assert result.answers[0].code_block == "(leakage detected in free-text response)"

    @pytest.mark.parametrize(
        "pattern",
        [
            "data leakage detected",
            "yes data leakage",
            "leakage detected",
            "leakage is present",
            "leakage found",
            "data leakage",
        ],
    )
    def test_various_positive_leakage_patterns(self, pattern: str) -> None:
        """Various positive leakage phrases are detected."""
        from mle_star.safety import _parse_leakage_output

        response = f"After analysis: {pattern} in the preprocessing step."
        result = _parse_leakage_output(response)
        assert result.answers[0].leakage_status == "Yes Data Leakage"

    @pytest.mark.parametrize(
        "pattern",
        [
            "Data Leakage Detected",
            "YES DATA LEAKAGE",
            "Leakage Detected",
        ],
    )
    def test_case_insensitive_pattern_matching(self, pattern: str) -> None:
        """Pattern matching is case-insensitive."""
        from mle_star.safety import _parse_leakage_output

        response = f"Result: {pattern}."
        result = _parse_leakage_output(response)
        assert result.answers[0].leakage_status == "Yes Data Leakage"


@pytest.mark.unit
class TestFreeTextDetectionCorrectionFlow:
    """Integration test for free-text leakage detection triggering correction."""

    async def test_free_text_detection_triggers_correction_flow(self) -> None:
        """Free-text positive detection triggers the correction agent."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        # Detection returns free-text with positive leakage and a code block
        free_text_detection = (
            "Data leakage detected in the preprocessing:\n"
            "```python\n"
            "scaler.fit(all_data)\n"
            "```"
        )
        corrected_code = (
            "import pandas as pd\n"
            "scaler.fit(train_data_only)\n"
            'print(f"Final Validation Performance: {0.9}")\n'
        )
        correction_response = f"```python\n{corrected_code}\n```"
        client.send_message = AsyncMock(
            side_effect=[free_text_detection, correction_response]
        )

        solution = _make_solution(content="scaler.fit(all_data)\nprint('done')")

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(solution, _make_task(), client)

        # Correction agent should have been called
        assert client.send_message.call_count == 2
        assert isinstance(result, SolutionScript)

    async def test_placeholder_code_block_uses_full_corrected_code(self) -> None:
        """When code_block is a placeholder, full corrected code replaces solution."""
        from mle_star.safety import check_and_fix_leakage

        client = AsyncMock()
        # Free-text without code fence -> placeholder code block
        free_text_detection = "Yes, data leakage detected in this solution."
        corrected_code = (
            "import pandas as pd\n"
            "df = pd.read_csv('train.csv')\n"
            "scaler.fit(df[train_idx])\n"
            'print(f"Final Validation Performance: {0.9}")\n'
        )
        correction_response = f"```python\n{corrected_code}\n```"
        client.send_message = AsyncMock(
            side_effect=[free_text_detection, correction_response]
        )

        solution = _make_solution(content="original code here")

        with patch(
            f"{_SAFETY}.PromptRegistry",
        ) as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = AsyncMock()
            mock_template.render = lambda **kwargs: "prompt"
            mock_registry.get.return_value = mock_template

            result = await check_and_fix_leakage(solution, _make_task(), client)

        # Should use the full corrected code since the code_block was a placeholder
        # extract_code_block strips trailing whitespace
        assert result.content == corrected_code.strip()
