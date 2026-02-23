"""Tests for the Phase 1 agent invocation functions (Task 27).

Validates ``retrieve_models``, ``generate_candidate``, ``merge_solutions``,
and the ``parse_retriever_output`` helper that will be implemented in
``src/mle_star/phase1.py``.  These agents handle model retrieval (A_retriever),
initial solution generation (A_init), and solution merging (A_merger).

Tests are written TDD-first and serve as the executable specification for
REQ-P1-001 through REQ-P1-017.

Refs:
    SRS 04a (Phase 1 Agents), IMPLEMENTATION_PLAN.md Task 27.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    AgentType,
    RetrievedModel,
    RetrieverOutput,
    SolutionPhase,
    SolutionScript,
)
import pytest

# ---------------------------------------------------------------------------
# Module path constant for patching
# ---------------------------------------------------------------------------

_MODULE = "mle_star.phase1"


# ---------------------------------------------------------------------------
# Reusable test helpers
# ---------------------------------------------------------------------------


def _make_task() -> Any:
    """Create a minimal TaskDescription-like object for testing."""
    from mle_star.models import DataModality, MetricDirection, TaskDescription, TaskType

    return TaskDescription(
        competition_id="test-comp",
        task_type=TaskType.CLASSIFICATION,
        data_modality=DataModality.TABULAR,
        evaluation_metric="accuracy",
        metric_direction=MetricDirection.MAXIMIZE,
        description="Predict the target variable from tabular features.",
    )


def _make_config(num_retrieved_models: int = 4) -> Any:
    """Create a minimal PipelineConfig for testing."""
    from mle_star.models import PipelineConfig

    return PipelineConfig(num_retrieved_models=num_retrieved_models)


def _make_retrieved_model(
    name: str = "xgboost", code: str = "import xgboost as xgb"
) -> RetrievedModel:
    """Create a RetrievedModel for testing."""
    return RetrievedModel(model_name=name, example_code=code)


def _make_solution(
    content: str = "print('hello')",
    phase: SolutionPhase = SolutionPhase.INIT,
) -> SolutionScript:
    """Create a SolutionScript for testing."""
    return SolutionScript(content=content, phase=phase)


def _make_retriever_json(
    models: list[dict[str, str]] | None = None,
) -> str:
    """Create a valid RetrieverOutput JSON string."""
    if models is None:
        models = [
            {"model_name": "xgboost", "example_code": "import xgboost as xgb"},
            {"model_name": "lightgbm", "example_code": "import lightgbm as lgb"},
        ]
    return json.dumps({"models": models})


# ===========================================================================
# REQ-P1-001: retrieve_models -- Is Async
# ===========================================================================


@pytest.mark.unit
class TestRetrieveModelsIsAsync:
    """retrieve_models is an async function (REQ-P1-001)."""

    def test_is_coroutine_function(self) -> None:
        """retrieve_models is defined as an async function."""
        from mle_star.phase1 import retrieve_models

        assert asyncio.iscoroutinefunction(retrieve_models)


# ===========================================================================
# REQ-P1-002: retrieve_models -- Prompt Loading from Registry
# ===========================================================================


@pytest.mark.unit
class TestRetrieveModelsPromptRegistry:
    """retrieve_models loads the A_retriever prompt from PromptRegistry (REQ-P1-002)."""

    async def test_registry_get_called_with_retriever_agent_type(self) -> None:
        """PromptRegistry.get is called with AgentType.RETRIEVER."""
        from mle_star.phase1 import retrieve_models

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json())

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "rendered retriever prompt"
            mock_registry.get.return_value = mock_template

            await retrieve_models(
                task=_make_task(), config=_make_config(), client=client
            )

        mock_registry.get.assert_called_once_with(AgentType.RETRIEVER)

    async def test_template_rendered_with_task_description_and_m(self) -> None:
        """The retriever template is rendered with task_description and M variables."""
        from mle_star.phase1 import retrieve_models

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json())

        render_kwargs_captured: list[dict[str, Any]] = []

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            task = _make_task()
            config = _make_config(num_retrieved_models=5)
            await retrieve_models(task=task, config=config, client=client)

        assert len(render_kwargs_captured) == 1
        assert render_kwargs_captured[0]["task_description"] == task.description
        assert render_kwargs_captured[0]["M"] == 5

    async def test_rendered_prompt_sent_to_client(self) -> None:
        """The rendered prompt is sent via client.send_message."""
        from mle_star.phase1 import retrieve_models

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json())

        expected_prompt = "rendered retriever prompt content xyz"

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = expected_prompt
            mock_registry.get.return_value = mock_template

            await retrieve_models(
                task=_make_task(), config=_make_config(), client=client
            )

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("message") == expected_prompt


# ===========================================================================
# REQ-P1-003: retrieve_models -- Agent Invocation
# ===========================================================================


@pytest.mark.unit
class TestRetrieveModelsAgentInvocation:
    """retrieve_models invokes A_retriever via client.send_message (REQ-P1-003)."""

    async def test_client_invoked_with_retriever_agent_type(self) -> None:
        """Client.send_message is invoked with agent_type=str(AgentType.RETRIEVER)."""
        from mle_star.phase1 import retrieve_models

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json())

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await retrieve_models(
                task=_make_task(), config=_make_config(), client=client
            )

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("agent_type") == str(AgentType.RETRIEVER)

    async def test_client_invoked_exactly_once(self) -> None:
        """Client.send_message is called exactly once per invocation."""
        from mle_star.phase1 import retrieve_models

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json())

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await retrieve_models(
                task=_make_task(), config=_make_config(), client=client
            )

        assert client.send_message.call_count == 1


# ===========================================================================
# REQ-P1-004: retrieve_models -- Response Parsing
# ===========================================================================


@pytest.mark.unit
class TestRetrieveModelsResponseParsing:
    """retrieve_models parses response via RetrieverOutput.model_validate_json (REQ-P1-004)."""

    async def test_returns_list_of_retrieved_models(self) -> None:
        """Returns a list of RetrievedModel objects from valid JSON response."""
        from mle_star.phase1 import retrieve_models

        models_data = [
            {"model_name": "xgboost", "example_code": "import xgboost"},
            {"model_name": "lightgbm", "example_code": "import lightgbm"},
        ]
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json(models_data))

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await retrieve_models(
                task=_make_task(), config=_make_config(), client=client
            )

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(m, RetrievedModel) for m in result)
        assert result[0].model_name == "xgboost"
        assert result[1].model_name == "lightgbm"

    async def test_return_type_is_list_of_retrieved_model(self) -> None:
        """Return type is list[RetrievedModel]."""
        from mle_star.phase1 import retrieve_models

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json())

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await retrieve_models(
                task=_make_task(), config=_make_config(), client=client
            )

        assert isinstance(result, list)
        for model in result:
            assert isinstance(model, RetrievedModel)


# ===========================================================================
# REQ-P1-005: retrieve_models -- Fewer Than M Models Warning
# ===========================================================================


@pytest.mark.unit
class TestRetrieveModelsFewerThanM:
    """retrieve_models logs warning when fewer than M models returned (REQ-P1-005)."""

    async def test_fewer_models_than_m_returns_available(self) -> None:
        """Returns available models when fewer than M are in the response."""
        from mle_star.phase1 import retrieve_models

        # Config requests 4 but only 2 returned
        models_data = [
            {"model_name": "xgboost", "example_code": "import xgboost"},
            {"model_name": "lightgbm", "example_code": "import lightgbm"},
        ]
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json(models_data))

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await retrieve_models(
                task=_make_task(),
                config=_make_config(num_retrieved_models=4),
                client=client,
            )

        assert len(result) == 2

    async def test_fewer_models_than_m_logs_warning(self) -> None:
        """Logs a warning when fewer than M models are returned."""
        from mle_star.phase1 import retrieve_models

        models_data = [
            {"model_name": "xgboost", "example_code": "import xgboost"},
        ]
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json(models_data))

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.logger") as mock_logger,
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await retrieve_models(
                task=_make_task(),
                config=_make_config(num_retrieved_models=4),
                client=client,
            )

        mock_logger.warning.assert_called()

    async def test_exact_m_models_no_warning(self) -> None:
        """No warning logged when exactly M models are returned."""
        from mle_star.phase1 import retrieve_models

        models_data = [
            {"model_name": f"model_{i}", "example_code": f"code_{i}"} for i in range(4)
        ]
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json(models_data))

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.logger") as mock_logger,
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await retrieve_models(
                task=_make_task(),
                config=_make_config(num_retrieved_models=4),
                client=client,
            )

        # Should not have a "fewer" warning when count matches
        for call in mock_logger.warning.call_args_list:
            assert "fewer" not in str(call).lower() or not mock_logger.warning.called


# ===========================================================================
# REQ-P1-006: retrieve_models -- Zero Models Raises ValueError
# ===========================================================================


@pytest.mark.unit
class TestRetrieveModelsZeroModels:
    """retrieve_models raises ValueError when 0 models after filtering (REQ-P1-006)."""

    async def test_raises_on_all_models_filtered(self) -> None:
        """Raises ValueError when all models are filtered out (empty name/code)."""
        from mle_star.phase1 import retrieve_models

        # All models have empty names or codes
        models_data = [
            {"model_name": "", "example_code": "import xgboost"},
            {"model_name": "lightgbm", "example_code": ""},
            {"model_name": "  ", "example_code": "import catboost"},
        ]
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json(models_data))

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            with pytest.raises(ValueError, match="zero models"):
                await retrieve_models(
                    task=_make_task(), config=_make_config(), client=client
                )

    async def test_error_message_mentions_retriever(self) -> None:
        """The ValueError message mentions A_retriever."""
        from mle_star.phase1 import retrieve_models

        models_data = [
            {"model_name": "", "example_code": ""},
        ]
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json(models_data))

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            with pytest.raises(ValueError, match="A_retriever"):
                await retrieve_models(
                    task=_make_task(), config=_make_config(), client=client
                )


# ===========================================================================
# REQ-P1-007: retrieve_models -- Filtering Invalid Models
# ===========================================================================


@pytest.mark.unit
class TestRetrieveModelsFiltering:
    """retrieve_models excludes models with empty name/code after strip (REQ-P1-007)."""

    async def test_filters_empty_model_name(self) -> None:
        """Models with empty model_name are excluded."""
        from mle_star.phase1 import retrieve_models

        models_data = [
            {"model_name": "", "example_code": "import xgboost"},
            {"model_name": "lightgbm", "example_code": "import lightgbm"},
        ]
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json(models_data))

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await retrieve_models(
                task=_make_task(), config=_make_config(), client=client
            )

        assert len(result) == 1
        assert result[0].model_name == "lightgbm"

    async def test_filters_whitespace_only_model_name(self) -> None:
        """Models with whitespace-only model_name are excluded."""
        from mle_star.phase1 import retrieve_models

        models_data = [
            {"model_name": "   ", "example_code": "import xgboost"},
            {"model_name": "catboost", "example_code": "import catboost"},
        ]
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json(models_data))

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await retrieve_models(
                task=_make_task(), config=_make_config(), client=client
            )

        assert len(result) == 1
        assert result[0].model_name == "catboost"

    async def test_filters_empty_example_code(self) -> None:
        """Models with empty example_code are excluded."""
        from mle_star.phase1 import retrieve_models

        models_data = [
            {"model_name": "xgboost", "example_code": ""},
            {"model_name": "lightgbm", "example_code": "import lightgbm"},
        ]
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json(models_data))

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await retrieve_models(
                task=_make_task(), config=_make_config(), client=client
            )

        assert len(result) == 1
        assert result[0].model_name == "lightgbm"

    async def test_filters_whitespace_only_example_code(self) -> None:
        """Models with whitespace-only example_code are excluded."""
        from mle_star.phase1 import retrieve_models

        models_data = [
            {"model_name": "xgboost", "example_code": "  \n  "},
            {"model_name": "lightgbm", "example_code": "import lightgbm"},
        ]
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json(models_data))

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await retrieve_models(
                task=_make_task(), config=_make_config(), client=client
            )

        assert len(result) == 1
        assert result[0].model_name == "lightgbm"

    async def test_filtering_logs_warning_per_invalid_model(self) -> None:
        """Logs a warning for each model that is filtered out."""
        from mle_star.phase1 import retrieve_models

        models_data = [
            {"model_name": "", "example_code": "import xgboost"},
            {"model_name": "lightgbm", "example_code": ""},
            {"model_name": "catboost", "example_code": "import catboost"},
        ]
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json(models_data))

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.logger") as mock_logger,
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await retrieve_models(
                task=_make_task(), config=_make_config(), client=client
            )

        # Should have logged at least 2 warnings for the 2 filtered models
        assert mock_logger.warning.call_count >= 2

    async def test_valid_models_preserved_after_filtering(self) -> None:
        """Valid models are preserved after filtering out invalid ones."""
        from mle_star.phase1 import retrieve_models

        models_data = [
            {"model_name": "", "example_code": "import a"},
            {"model_name": "xgboost", "example_code": "import xgboost"},
            {"model_name": "  ", "example_code": "import b"},
            {"model_name": "lightgbm", "example_code": "import lightgbm"},
            {"model_name": "catboost", "example_code": "  "},
        ]
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json(models_data))

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await retrieve_models(
                task=_make_task(), config=_make_config(), client=client
            )

        assert len(result) == 2
        assert result[0].model_name == "xgboost"
        assert result[1].model_name == "lightgbm"


# ===========================================================================
# parse_retriever_output -- Helper Function Tests
# ===========================================================================


@pytest.mark.unit
class TestParseRetrieverOutput:
    """parse_retriever_output parses JSON into RetrieverOutput."""

    def test_valid_json_returns_retriever_output(self) -> None:
        """Valid JSON with models list returns RetrieverOutput."""
        from mle_star.phase1 import parse_retriever_output

        response = _make_retriever_json()
        result = parse_retriever_output(response)
        assert isinstance(result, RetrieverOutput)
        assert len(result.models) == 2

    def test_models_are_retrieved_model_instances(self) -> None:
        """Each model in the output is a RetrievedModel instance."""
        from mle_star.phase1 import parse_retriever_output

        response = _make_retriever_json()
        result = parse_retriever_output(response)
        for model in result.models:
            assert isinstance(model, RetrievedModel)

    def test_invalid_json_raises_value_error(self) -> None:
        """Raises ValueError on invalid JSON."""
        from mle_star.phase1 import parse_retriever_output

        with pytest.raises(ValueError):
            parse_retriever_output("not valid json at all")

    def test_missing_models_key_raises_value_error(self) -> None:
        """Raises ValueError when JSON lacks 'models' key."""
        from mle_star.phase1 import parse_retriever_output

        with pytest.raises(ValueError):
            parse_retriever_output('{"other_key": []}')

    def test_empty_models_list_raises_value_error(self) -> None:
        """Raises ValueError when models list is empty."""
        from mle_star.phase1 import parse_retriever_output

        with pytest.raises(ValueError):
            parse_retriever_output('{"models": []}')

    def test_malformed_model_entry_raises_value_error(self) -> None:
        """Raises ValueError when a model entry is missing required fields."""
        from mle_star.phase1 import parse_retriever_output

        with pytest.raises(ValueError):
            parse_retriever_output('{"models": [{"model_name": "xgboost"}]}')

    def test_preserves_model_name_and_example_code(self) -> None:
        """Parsed models preserve original model_name and example_code."""
        from mle_star.phase1 import parse_retriever_output

        models_data = [
            {
                "model_name": "unique_marker_abc",
                "example_code": "unique_code_xyz",
            }
        ]
        result = parse_retriever_output(_make_retriever_json(models_data))
        assert result.models[0].model_name == "unique_marker_abc"
        assert result.models[0].example_code == "unique_code_xyz"

    def test_multiple_models_parsed_correctly(self) -> None:
        """Multiple models in JSON are all parsed."""
        from mle_star.phase1 import parse_retriever_output

        models_data = [
            {"model_name": f"model_{i}", "example_code": f"code_{i}"} for i in range(5)
        ]
        result = parse_retriever_output(_make_retriever_json(models_data))
        assert len(result.models) == 5
        for i, model in enumerate(result.models):
            assert model.model_name == f"model_{i}"
            assert model.example_code == f"code_{i}"


# ===========================================================================
# REQ-P1-008: generate_candidate -- Is Async
# ===========================================================================


@pytest.mark.unit
class TestGenerateCandidateIsAsync:
    """generate_candidate is an async function (REQ-P1-008)."""

    def test_is_coroutine_function(self) -> None:
        """generate_candidate is defined as an async function."""
        from mle_star.phase1 import generate_candidate

        assert asyncio.iscoroutinefunction(generate_candidate)


# ===========================================================================
# REQ-P1-009: generate_candidate -- Prompt Loading from Registry
# ===========================================================================


@pytest.mark.unit
class TestGenerateCandidatePromptRegistry:
    """generate_candidate loads the A_init prompt from PromptRegistry (REQ-P1-009)."""

    async def test_registry_get_called_with_init_agent_type(self) -> None:
        """PromptRegistry.get is called with AgentType.INIT."""
        from mle_star.phase1 import generate_candidate

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nimport xgboost\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="import xgboost",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "rendered init prompt"
            mock_registry.get.return_value = mock_template

            await generate_candidate(
                task=_make_task(),
                model=_make_retrieved_model(),
                config=_make_config(),
                client=client,
            )

        mock_registry.get.assert_called_once_with(AgentType.INIT)

    async def test_template_rendered_with_task_model_code(self) -> None:
        """The init template is rendered with task_description, model_name, example_code."""
        from mle_star.phase1 import generate_candidate

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nimport xgboost\n```")

        render_kwargs_captured: list[dict[str, Any]] = []

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="import xgboost",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            task = _make_task()
            model = _make_retrieved_model(
                name="marker_model_name", code="marker_example_code"
            )
            await generate_candidate(
                task=task, model=model, config=_make_config(), client=client
            )

        assert len(render_kwargs_captured) == 1
        assert render_kwargs_captured[0]["task_description"] == task.description
        assert render_kwargs_captured[0]["model_name"] == "marker_model_name"
        assert render_kwargs_captured[0]["example_code"] == "marker_example_code"

    async def test_rendered_prompt_sent_to_client(self) -> None:
        """The rendered prompt is sent via client.send_message."""
        from mle_star.phase1 import generate_candidate

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nimport xgboost\n```")

        expected_prompt = "rendered init prompt content abc"

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="import xgboost",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = expected_prompt
            mock_registry.get.return_value = mock_template

            await generate_candidate(
                task=_make_task(),
                model=_make_retrieved_model(),
                config=_make_config(),
                client=client,
            )

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("message") == expected_prompt


# ===========================================================================
# REQ-P1-010: generate_candidate -- Agent Invocation
# ===========================================================================


@pytest.mark.unit
class TestGenerateCandidateAgentInvocation:
    """generate_candidate invokes A_init via client.send_message (REQ-P1-010)."""

    async def test_client_invoked_with_init_agent_type(self) -> None:
        """Client.send_message is invoked with agent_type=str(AgentType.INIT)."""
        from mle_star.phase1 import generate_candidate

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nimport xgboost\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="import xgboost",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await generate_candidate(
                task=_make_task(),
                model=_make_retrieved_model(),
                config=_make_config(),
                client=client,
            )

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("agent_type") == str(AgentType.INIT)

    async def test_client_invoked_exactly_once(self) -> None:
        """Client.send_message is called exactly once per invocation."""
        from mle_star.phase1 import generate_candidate

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nimport xgboost\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="import xgboost",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await generate_candidate(
                task=_make_task(),
                model=_make_retrieved_model(),
                config=_make_config(),
                client=client,
            )

        assert client.send_message.call_count == 1


# ===========================================================================
# REQ-P1-011: generate_candidate -- Code Extraction and Return Type
# ===========================================================================


@pytest.mark.unit
class TestGenerateCandidateCodeExtraction:
    """generate_candidate extracts code and returns SolutionScript (REQ-P1-011)."""

    async def test_extract_code_block_called_on_response(self) -> None:
        """extract_code_block is invoked on the agent's response."""
        from mle_star.phase1 import generate_candidate

        client = AsyncMock()
        agent_response = "Here is code:\n```python\nimport xgboost\n```"
        client.send_message = AsyncMock(return_value=agent_response)

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="import xgboost",
            ) as mock_extract,
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await generate_candidate(
                task=_make_task(),
                model=_make_retrieved_model(),
                config=_make_config(),
                client=client,
            )

        mock_extract.assert_called_once_with(agent_response)

    async def test_returns_solution_script(self) -> None:
        """Returns a SolutionScript on success."""
        from mle_star.phase1 import generate_candidate

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nimport xgboost\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="import xgboost",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_candidate(
                task=_make_task(),
                model=_make_retrieved_model(),
                config=_make_config(),
                client=client,
            )

        assert isinstance(result, SolutionScript)

    async def test_solution_script_content_from_extraction(self) -> None:
        """SolutionScript content matches extracted code block."""
        from mle_star.phase1 import generate_candidate

        extracted_code = "import xgboost as xgb\nmodel = xgb.XGBClassifier()"
        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value=f"```python\n{extracted_code}\n```"
        )

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value=extracted_code,
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_candidate(
                task=_make_task(),
                model=_make_retrieved_model(),
                config=_make_config(),
                client=client,
            )

        assert result is not None
        assert result.content == extracted_code

    async def test_solution_script_phase_is_init(self) -> None:
        """SolutionScript has phase=SolutionPhase.INIT."""
        from mle_star.phase1 import generate_candidate

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nimport xgboost\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="import xgboost",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_candidate(
                task=_make_task(),
                model=_make_retrieved_model(),
                config=_make_config(),
                client=client,
            )

        assert result is not None
        assert result.phase == SolutionPhase.INIT

    async def test_solution_script_score_is_none(self) -> None:
        """SolutionScript has score=None (not yet evaluated)."""
        from mle_star.phase1 import generate_candidate

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nimport xgboost\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="import xgboost",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_candidate(
                task=_make_task(),
                model=_make_retrieved_model(),
                config=_make_config(),
                client=client,
            )

        assert result is not None
        assert result.score is None

    async def test_solution_script_is_executable_true(self) -> None:
        """SolutionScript has is_executable=True."""
        from mle_star.phase1 import generate_candidate

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nimport xgboost\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="import xgboost",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_candidate(
                task=_make_task(),
                model=_make_retrieved_model(),
                config=_make_config(),
                client=client,
            )

        assert result is not None
        assert result.is_executable is True

    async def test_solution_script_source_model_set(self) -> None:
        """SolutionScript source_model matches model.model_name."""
        from mle_star.phase1 import generate_candidate

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nimport xgboost\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="import xgboost",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            model = _make_retrieved_model(name="unique_model_marker")
            result = await generate_candidate(
                task=_make_task(),
                model=model,
                config=_make_config(),
                client=client,
            )

        assert result is not None
        assert result.source_model == "unique_model_marker"


# ===========================================================================
# REQ-P1-012: generate_candidate -- Returns None on Failure
# ===========================================================================


@pytest.mark.unit
class TestGenerateCandidateReturnsNoneOnFailure:
    """generate_candidate returns None when extraction fails (REQ-P1-012)."""

    async def test_returns_none_on_empty_extracted_code(self) -> None:
        """Returns None when extract_code_block returns empty string."""
        from mle_star.phase1 import generate_candidate

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="no code here")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_candidate(
                task=_make_task(),
                model=_make_retrieved_model(),
                config=_make_config(),
                client=client,
            )

        assert result is None

    async def test_returns_none_on_empty_response(self) -> None:
        """Returns None when agent response is empty string."""
        from mle_star.phase1 import generate_candidate

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_candidate(
                task=_make_task(),
                model=_make_retrieved_model(),
                config=_make_config(),
                client=client,
            )

        assert result is None

    async def test_returns_none_on_whitespace_only_extracted(self) -> None:
        """Returns None when extract_code_block returns whitespace only."""
        from mle_star.phase1 import generate_candidate

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="   \n  ")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="   "),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_candidate(
                task=_make_task(),
                model=_make_retrieved_model(),
                config=_make_config(),
                client=client,
            )

        assert result is None

    async def test_return_is_none_type_not_solution_script(self) -> None:
        """Returns None (not empty SolutionScript) on failure."""
        from mle_star.phase1 import generate_candidate

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_candidate(
                task=_make_task(),
                model=_make_retrieved_model(),
                config=_make_config(),
                client=client,
            )

        assert result is None
        assert not isinstance(result, SolutionScript)


# ===========================================================================
# REQ-P1-013: merge_solutions -- Is Async
# ===========================================================================


@pytest.mark.unit
class TestMergeSolutionsIsAsync:
    """merge_solutions is an async function (REQ-P1-013)."""

    def test_is_coroutine_function(self) -> None:
        """merge_solutions is defined as an async function."""
        from mle_star.phase1 import merge_solutions

        assert asyncio.iscoroutinefunction(merge_solutions)


# ===========================================================================
# REQ-P1-014: merge_solutions -- Prompt Loading from Registry
# ===========================================================================


@pytest.mark.unit
class TestMergeSolutionsPromptRegistry:
    """merge_solutions loads the A_merger prompt from PromptRegistry (REQ-P1-014)."""

    async def test_registry_get_called_with_merger_agent_type(self) -> None:
        """PromptRegistry.get is called with AgentType.MERGER."""
        from mle_star.phase1 import merge_solutions

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nmerged code\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="merged code",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "rendered merger prompt"
            mock_registry.get.return_value = mock_template

            await merge_solutions(
                base=_make_solution(content="base code"),
                reference=_make_solution(content="reference code"),
                config=_make_config(),
                client=client,
            )

        mock_registry.get.assert_called_once_with(AgentType.MERGER)

    async def test_template_rendered_with_base_and_reference_code(self) -> None:
        """The merger template is rendered with base_code and reference_code."""
        from mle_star.phase1 import merge_solutions

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nmerged code\n```")

        render_kwargs_captured: list[dict[str, Any]] = []

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="merged code",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "rendered prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await merge_solutions(
                base=_make_solution(content="base_marker_content"),
                reference=_make_solution(content="reference_marker_content"),
                config=_make_config(),
                client=client,
            )

        assert len(render_kwargs_captured) == 1
        assert render_kwargs_captured[0]["base_code"] == "base_marker_content"
        assert render_kwargs_captured[0]["reference_code"] == "reference_marker_content"

    async def test_rendered_prompt_sent_to_client(self) -> None:
        """The rendered prompt is sent via client.send_message."""
        from mle_star.phase1 import merge_solutions

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nmerged code\n```")

        expected_prompt = "rendered merger prompt content def"

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="merged code",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = expected_prompt
            mock_registry.get.return_value = mock_template

            await merge_solutions(
                base=_make_solution(),
                reference=_make_solution(),
                config=_make_config(),
                client=client,
            )

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("message") == expected_prompt


# ===========================================================================
# REQ-P1-015: merge_solutions -- Agent Invocation
# ===========================================================================


@pytest.mark.unit
class TestMergeSolutionsAgentInvocation:
    """merge_solutions invokes A_merger via client.send_message (REQ-P1-015)."""

    async def test_client_invoked_with_merger_agent_type(self) -> None:
        """Client.send_message is invoked with agent_type=str(AgentType.MERGER)."""
        from mle_star.phase1 import merge_solutions

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nmerged code\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="merged code",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await merge_solutions(
                base=_make_solution(),
                reference=_make_solution(),
                config=_make_config(),
                client=client,
            )

        call_kwargs = client.send_message.call_args[1]
        assert call_kwargs.get("agent_type") == str(AgentType.MERGER)

    async def test_client_invoked_exactly_once(self) -> None:
        """Client.send_message is called exactly once per invocation."""
        from mle_star.phase1 import merge_solutions

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nmerged code\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="merged code",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await merge_solutions(
                base=_make_solution(),
                reference=_make_solution(),
                config=_make_config(),
                client=client,
            )

        assert client.send_message.call_count == 1


# ===========================================================================
# REQ-P1-016: merge_solutions -- Code Extraction and Return Type
# ===========================================================================


@pytest.mark.unit
class TestMergeSolutionsCodeExtraction:
    """merge_solutions extracts code and returns SolutionScript (REQ-P1-016)."""

    async def test_extract_code_block_called_on_response(self) -> None:
        """extract_code_block is invoked on the agent's response."""
        from mle_star.phase1 import merge_solutions

        client = AsyncMock()
        agent_response = "Here is the merged solution:\n```python\nmerged\n```"
        client.send_message = AsyncMock(return_value=agent_response)

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="merged",
            ) as mock_extract,
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            await merge_solutions(
                base=_make_solution(),
                reference=_make_solution(),
                config=_make_config(),
                client=client,
            )

        mock_extract.assert_called_once_with(agent_response)

    async def test_returns_solution_script(self) -> None:
        """Returns a SolutionScript on success."""
        from mle_star.phase1 import merge_solutions

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nmerged code\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="merged code",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await merge_solutions(
                base=_make_solution(),
                reference=_make_solution(),
                config=_make_config(),
                client=client,
            )

        assert isinstance(result, SolutionScript)

    async def test_solution_script_content_from_extraction(self) -> None:
        """SolutionScript content matches extracted code block."""
        from mle_star.phase1 import merge_solutions

        merged_code = "import xgboost\nimport lightgbm\nensemble()"
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=f"```python\n{merged_code}\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value=merged_code,
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await merge_solutions(
                base=_make_solution(),
                reference=_make_solution(),
                config=_make_config(),
                client=client,
            )

        assert result is not None
        assert result.content == merged_code

    async def test_solution_script_phase_is_merged(self) -> None:
        """SolutionScript has phase=SolutionPhase.MERGED."""
        from mle_star.phase1 import merge_solutions

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nmerged code\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="merged code",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await merge_solutions(
                base=_make_solution(),
                reference=_make_solution(),
                config=_make_config(),
                client=client,
            )

        assert result is not None
        assert result.phase == SolutionPhase.MERGED

    async def test_solution_script_score_is_none(self) -> None:
        """SolutionScript has score=None (not yet evaluated)."""
        from mle_star.phase1 import merge_solutions

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nmerged code\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="merged code",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await merge_solutions(
                base=_make_solution(),
                reference=_make_solution(),
                config=_make_config(),
                client=client,
            )

        assert result is not None
        assert result.score is None

    async def test_solution_script_is_executable_true(self) -> None:
        """SolutionScript has is_executable=True."""
        from mle_star.phase1 import merge_solutions

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nmerged code\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="merged code",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await merge_solutions(
                base=_make_solution(),
                reference=_make_solution(),
                config=_make_config(),
                client=client,
            )

        assert result is not None
        assert result.is_executable is True

    async def test_solution_script_source_model_is_none(self) -> None:
        """SolutionScript source_model is None for merged solutions."""
        from mle_star.phase1 import merge_solutions

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nmerged code\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="merged code",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await merge_solutions(
                base=_make_solution(),
                reference=_make_solution(),
                config=_make_config(),
                client=client,
            )

        assert result is not None
        assert result.source_model is None


# ===========================================================================
# REQ-P1-017: merge_solutions -- Returns None on Failure
# ===========================================================================


@pytest.mark.unit
class TestMergeSolutionsReturnsNoneOnFailure:
    """merge_solutions returns None when extraction fails (REQ-P1-017)."""

    async def test_returns_none_on_empty_extracted_code(self) -> None:
        """Returns None when extract_code_block returns empty string."""
        from mle_star.phase1 import merge_solutions

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="no code")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await merge_solutions(
                base=_make_solution(),
                reference=_make_solution(),
                config=_make_config(),
                client=client,
            )

        assert result is None

    async def test_returns_none_on_empty_response(self) -> None:
        """Returns None when agent response is empty string."""
        from mle_star.phase1 import merge_solutions

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await merge_solutions(
                base=_make_solution(),
                reference=_make_solution(),
                config=_make_config(),
                client=client,
            )

        assert result is None

    async def test_returns_none_on_whitespace_only_extracted(self) -> None:
        """Returns None when extract_code_block returns whitespace only."""
        from mle_star.phase1 import merge_solutions

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="  \n  ")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="   "),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await merge_solutions(
                base=_make_solution(),
                reference=_make_solution(),
                config=_make_config(),
                client=client,
            )

        assert result is None

    async def test_return_is_none_type_not_solution_script(self) -> None:
        """Returns None (not empty SolutionScript) on failure."""
        from mle_star.phase1 import merge_solutions

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await merge_solutions(
                base=_make_solution(),
                reference=_make_solution(),
                config=_make_config(),
                client=client,
            )

        assert result is None
        assert not isinstance(result, SolutionScript)


# ===========================================================================
# Prompt Template Integration Tests
# ===========================================================================


@pytest.mark.unit
class TestRetrieverPromptTemplateIntegration:
    """Validate that the retriever prompt template exists and renders correctly."""

    def test_retriever_template_exists_in_registry(self) -> None:
        """PromptRegistry contains a retriever template."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.RETRIEVER)
        assert template.agent_type == AgentType.RETRIEVER

    def test_retriever_template_has_task_description_variable(self) -> None:
        """Retriever template declares 'task_description' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.RETRIEVER)
        assert "task_description" in template.variables

    def test_retriever_template_has_m_variable(self) -> None:
        """Retriever template declares 'M' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.RETRIEVER)
        assert "M" in template.variables

    def test_retriever_template_renders_with_variables(self) -> None:
        """Retriever template renders successfully with both required variables."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.RETRIEVER)
        rendered = template.render(
            task_description="Predict house prices", target_column="Not specified", M=4,
            research_context="",
        )
        assert "Predict house prices" in rendered
        assert "4" in rendered


@pytest.mark.unit
class TestInitPromptTemplateIntegration:
    """Validate that the init prompt template exists and renders correctly."""

    def test_init_template_exists_in_registry(self) -> None:
        """PromptRegistry contains an init template."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.INIT)
        assert template.agent_type == AgentType.INIT

    def test_init_template_has_task_description_variable(self) -> None:
        """Init template declares 'task_description' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.INIT)
        assert "task_description" in template.variables

    def test_init_template_has_model_name_variable(self) -> None:
        """Init template declares 'model_name' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.INIT)
        assert "model_name" in template.variables

    def test_init_template_has_example_code_variable(self) -> None:
        """Init template declares 'example_code' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.INIT)
        assert "example_code" in template.variables

    def test_init_template_renders_with_variables(self) -> None:
        """Init template renders successfully with all required variables."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.INIT)
        rendered = template.render(
            task_description="Classify images",
            target_column="Not specified",
            model_name="xgboost",
            example_code="import xgboost",
            research_context="",
        )
        assert "Classify images" in rendered
        assert "xgboost" in rendered
        assert "import xgboost" in rendered


@pytest.mark.unit
class TestMergerPromptTemplateIntegration:
    """Validate that the merger prompt template exists and renders correctly."""

    def test_merger_template_exists_in_registry(self) -> None:
        """PromptRegistry contains a merger template."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.MERGER)
        assert template.agent_type == AgentType.MERGER

    def test_merger_template_has_base_code_variable(self) -> None:
        """Merger template declares 'base_code' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.MERGER)
        assert "base_code" in template.variables

    def test_merger_template_has_reference_code_variable(self) -> None:
        """Merger template declares 'reference_code' as a required variable."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.MERGER)
        assert "reference_code" in template.variables

    def test_merger_template_renders_with_variables(self) -> None:
        """Merger template renders successfully with both required variables."""
        from mle_star.prompts import PromptRegistry

        registry = PromptRegistry()
        template = registry.get(AgentType.MERGER)
        rendered = template.render(
            base_code="base_code_content",
            reference_code="reference_code_content",
        )
        assert "base_code_content" in rendered
        assert "reference_code_content" in rendered


# ===========================================================================
# Edge Cases
# ===========================================================================


@pytest.mark.unit
class TestRetrieveModelsEdgeCases:
    """Edge case tests for retrieve_models."""

    async def test_single_valid_model_returned(self) -> None:
        """Works correctly with exactly one valid model."""
        from mle_star.phase1 import retrieve_models

        models_data = [
            {"model_name": "xgboost", "example_code": "import xgboost"},
        ]
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json(models_data))

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await retrieve_models(
                task=_make_task(), config=_make_config(), client=client
            )

        assert len(result) == 1
        assert result[0].model_name == "xgboost"

    async def test_more_models_than_m_all_returned(self) -> None:
        """When more than M models returned, all valid ones are included."""
        from mle_star.phase1 import retrieve_models

        models_data = [
            {"model_name": f"model_{i}", "example_code": f"code_{i}"} for i in range(6)
        ]
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json(models_data))

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await retrieve_models(
                task=_make_task(),
                config=_make_config(num_retrieved_models=4),
                client=client,
            )

        # All 6 valid models returned (no truncation)
        assert len(result) == 6

    async def test_mixed_valid_and_invalid_models(self) -> None:
        """Only valid models survive filtering from a mix."""
        from mle_star.phase1 import retrieve_models

        models_data = [
            {"model_name": "valid1", "example_code": "code1"},
            {"model_name": "", "example_code": "code2"},
            {"model_name": "valid2", "example_code": "code3"},
            {"model_name": "valid3", "example_code": "  "},
        ]
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json(models_data))

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await retrieve_models(
                task=_make_task(), config=_make_config(), client=client
            )

        assert len(result) == 2
        assert [m.model_name for m in result] == ["valid1", "valid2"]


@pytest.mark.unit
class TestGenerateCandidateEdgeCases:
    """Edge case tests for generate_candidate."""

    async def test_multiline_extracted_code(self) -> None:
        """Handles multiline extracted code correctly."""
        from mle_star.phase1 import generate_candidate

        multiline_code = "import numpy as np\nx = np.array([1, 2])\nprint(x)"
        client = AsyncMock()
        client.send_message = AsyncMock(
            return_value=f"```python\n{multiline_code}\n```"
        )

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value=multiline_code,
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_candidate(
                task=_make_task(),
                model=_make_retrieved_model(),
                config=_make_config(),
                client=client,
            )

        assert result is not None
        assert result.content == multiline_code

    async def test_source_model_preserved_from_different_models(self) -> None:
        """source_model is set from model.model_name for different model names."""
        from mle_star.phase1 import generate_candidate

        for model_name in ["xgboost", "lightgbm", "catboost", "random_forest"]:
            client = AsyncMock()
            client.send_message = AsyncMock(return_value="```python\ncode\n```")

            with (
                patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
                patch(
                    f"{_MODULE}.extract_code_block",
                    return_value="code",
                ),
            ):
                mock_registry = mock_registry_cls.return_value
                mock_template = MagicMock()
                mock_template.render.return_value = "prompt"
                mock_registry.get.return_value = mock_template

                model = _make_retrieved_model(name=model_name)
                result = await generate_candidate(
                    task=_make_task(),
                    model=model,
                    config=_make_config(),
                    client=client,
                )

            assert result is not None
            assert result.source_model == model_name


@pytest.mark.unit
class TestMergeSolutionsEdgeCases:
    """Edge case tests for merge_solutions."""

    async def test_merged_solution_does_not_inherit_base_score(self) -> None:
        """Merged solution does not carry over the base solution's score."""
        from mle_star.phase1 import merge_solutions

        base = _make_solution(content="base code")
        base.score = 0.85

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nmerged\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="merged",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await merge_solutions(
                base=base,
                reference=_make_solution(),
                config=_make_config(),
                client=client,
            )

        assert result is not None
        assert result.score is None

    async def test_merged_solution_does_not_inherit_source_model(self) -> None:
        """Merged solution has source_model=None regardless of inputs."""
        from mle_star.phase1 import merge_solutions

        base = _make_solution(content="base code")
        base.source_model = "xgboost"

        ref = _make_solution(content="ref code")
        ref.source_model = "lightgbm"

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nmerged\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(
                f"{_MODULE}.extract_code_block",
                return_value="merged",
            ),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await merge_solutions(
                base=base,
                reference=ref,
                config=_make_config(),
                client=client,
            )

        assert result is not None
        assert result.source_model is None


# ===========================================================================
# Property-Based Tests
# ===========================================================================


@pytest.mark.unit
class TestRetrieveModelsPropertyBased:
    """Property-based tests for retrieve_models invariants."""

    @given(
        n_models=st.integers(min_value=1, max_value=10),
        m=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20)
    async def test_valid_models_always_returns_list(
        self, n_models: int, m: int
    ) -> None:
        """retrieve_models always returns a list for valid model responses."""
        from mle_star.phase1 import retrieve_models

        models_data = [
            {"model_name": f"model_{i}", "example_code": f"code_{i}"}
            for i in range(n_models)
        ]
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json(models_data))

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await retrieve_models(
                task=_make_task(),
                config=_make_config(num_retrieved_models=m),
                client=client,
            )

        assert isinstance(result, list)
        assert len(result) == n_models
        assert all(isinstance(m_item, RetrievedModel) for m_item in result)

    @given(
        n_valid=st.integers(min_value=1, max_value=5),
        n_invalid=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=20)
    async def test_filtering_preserves_only_valid_models(
        self, n_valid: int, n_invalid: int
    ) -> None:
        """After filtering, only models with non-empty name and code remain."""
        from mle_star.phase1 import retrieve_models

        valid = [
            {"model_name": f"valid_{i}", "example_code": f"valid_code_{i}"}
            for i in range(n_valid)
        ]
        invalid = [
            {"model_name": "", "example_code": f"invalid_code_{i}"}
            for i in range(n_invalid)
        ]
        models_data = valid + invalid
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=_make_retriever_json(models_data))

        with patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await retrieve_models(
                task=_make_task(), config=_make_config(), client=client
            )

        assert len(result) == n_valid
        for model in result:
            assert model.model_name.strip()
            assert model.example_code.strip()


@pytest.mark.unit
class TestGenerateCandidatePropertyBased:
    """Property-based tests for generate_candidate invariants."""

    @given(
        model_name=st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
        code=st.text(min_size=1, max_size=200).filter(
            lambda s: s.strip() and "```" not in s
        ),
    )
    @settings(max_examples=20)
    async def test_successful_generation_always_returns_solution_script(
        self, model_name: str, code: str
    ) -> None:
        """generate_candidate returns SolutionScript with correct phase for valid inputs."""
        from mle_star.phase1 import generate_candidate

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=f"```python\n{code}\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=code),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_candidate(
                task=_make_task(),
                model=_make_retrieved_model(name=model_name, code="example"),
                config=_make_config(),
                client=client,
            )

        assert result is not None
        assert isinstance(result, SolutionScript)
        assert result.phase == SolutionPhase.INIT
        assert result.source_model == model_name
        assert result.score is None
        assert result.is_executable is True

    @given(
        model_name=st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
    )
    @settings(max_examples=20)
    async def test_failure_always_returns_none(self, model_name: str) -> None:
        """generate_candidate returns None when extraction yields empty code."""
        from mle_star.phase1 import generate_candidate

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_candidate(
                task=_make_task(),
                model=_make_retrieved_model(name=model_name),
                config=_make_config(),
                client=client,
            )

        assert result is None


@pytest.mark.unit
class TestMergeSolutionsPropertyBased:
    """Property-based tests for merge_solutions invariants."""

    @given(
        base_content=st.text(min_size=1, max_size=200).filter(
            lambda s: s.strip() and "```" not in s
        ),
        ref_content=st.text(min_size=1, max_size=200).filter(
            lambda s: s.strip() and "```" not in s
        ),
    )
    @settings(max_examples=20)
    async def test_successful_merge_always_returns_merged_phase(
        self, base_content: str, ref_content: str
    ) -> None:
        """merge_solutions always returns SolutionScript with MERGED phase on success."""
        from mle_star.phase1 import merge_solutions

        merged_code = "merged_output"
        client = AsyncMock()
        client.send_message = AsyncMock(return_value=f"```python\n{merged_code}\n```")

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=merged_code),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await merge_solutions(
                base=_make_solution(content=base_content),
                reference=_make_solution(content=ref_content),
                config=_make_config(),
                client=client,
            )

        assert result is not None
        assert result.phase == SolutionPhase.MERGED
        assert result.source_model is None
        assert result.score is None
        assert result.is_executable is True

    @given(
        base_content=st.text(min_size=1, max_size=100).filter(lambda s: s.strip()),
        ref_content=st.text(min_size=1, max_size=100).filter(lambda s: s.strip()),
    )
    @settings(max_examples=20)
    async def test_template_always_receives_both_codes(
        self, base_content: str, ref_content: str
    ) -> None:
        """Merger template always receives base_code and reference_code."""
        from mle_star.phase1 import merge_solutions

        client = AsyncMock()
        client.send_message = AsyncMock(return_value="```python\nmerged\n```")

        render_kwargs_captured: list[dict[str, Any]] = []

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value="merged"),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()

            def capture_render(**kwargs: Any) -> str:
                render_kwargs_captured.append(kwargs)
                return "prompt"

            mock_template.render = capture_render
            mock_registry.get.return_value = mock_template

            await merge_solutions(
                base=_make_solution(content=base_content),
                reference=_make_solution(content=ref_content),
                config=_make_config(),
                client=client,
            )

        assert len(render_kwargs_captured) == 1
        assert render_kwargs_captured[0]["base_code"] == base_content
        assert render_kwargs_captured[0]["reference_code"] == ref_content


# ===========================================================================
# Parametrized Tests
# ===========================================================================


@pytest.mark.unit
class TestParseRetrieverOutputParametrized:
    """Parametrized tests for parse_retriever_output covering various invalid inputs."""

    @pytest.mark.parametrize(
        "invalid_input",
        [
            "",
            "not json",
            "42",
            "null",
            "[]",
            '{"models": []}',
            '{"wrong_key": [{"model_name": "a", "example_code": "b"}]}',
            '{"models": [{"model_name": "a"}]}',
            '{"models": "not a list"}',
        ],
        ids=[
            "empty-string",
            "plain-text",
            "number",
            "null",
            "array",
            "empty-models",
            "wrong-key",
            "missing-example-code",
            "models-not-list",
        ],
    )
    def test_invalid_inputs_raise_value_error(self, invalid_input: str) -> None:
        """parse_retriever_output raises ValueError for various invalid inputs."""
        from mle_star.phase1 import parse_retriever_output

        with pytest.raises(ValueError):
            parse_retriever_output(invalid_input)


@pytest.mark.unit
class TestGenerateCandidateParametrized:
    """Parametrized tests for generate_candidate covering multiple response formats."""

    @pytest.mark.parametrize(
        "empty_response",
        ["", "   ", "\n\n", "  \n  \n  "],
        ids=["empty", "spaces", "newlines", "mixed-whitespace"],
    )
    async def test_empty_responses_return_none(self, empty_response: str) -> None:
        """generate_candidate returns None for various empty/whitespace responses."""
        from mle_star.phase1 import generate_candidate

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=empty_response)

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await generate_candidate(
                task=_make_task(),
                model=_make_retrieved_model(),
                config=_make_config(),
                client=client,
            )

        assert result is None


@pytest.mark.unit
class TestMergeSolutionsParametrized:
    """Parametrized tests for merge_solutions covering multiple response formats."""

    @pytest.mark.parametrize(
        "empty_response",
        ["", "   ", "\n\n", "  \n  \n  "],
        ids=["empty", "spaces", "newlines", "mixed-whitespace"],
    )
    async def test_empty_responses_return_none(self, empty_response: str) -> None:
        """merge_solutions returns None for various empty/whitespace responses."""
        from mle_star.phase1 import merge_solutions

        client = AsyncMock()
        client.send_message = AsyncMock(return_value=empty_response)

        with (
            patch(f"{_MODULE}.PromptRegistry") as mock_registry_cls,
            patch(f"{_MODULE}.extract_code_block", return_value=""),
        ):
            mock_registry = mock_registry_cls.return_value
            mock_template = MagicMock()
            mock_template.render.return_value = "prompt"
            mock_registry.get.return_value = mock_template

            result = await merge_solutions(
                base=_make_solution(),
                reference=_make_solution(),
                config=_make_config(),
                client=client,
            )

        assert result is None
