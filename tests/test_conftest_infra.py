"""Tests for shared test infrastructure in conftest.py (Task 51).

Validates the factory functions (``make_solution``, ``make_task``,
``make_config``, ``make_eval_result``), the ``mock_client`` fixture,
the ``mock_registry`` fixture, and the ``tmp_working_dir`` fixture
that provide reusable test infrastructure for the entire test suite.

Tests are written TDD-first and serve as the executable specification
for the shared test infrastructure contract.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from hypothesis import given, settings, strategies as st
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
)
import pytest

# Import factory functions from conftest -- they are plain functions,
# not fixtures, so they can be imported directly.
from tests.conftest import make_config, make_eval_result, make_solution, make_task

# ===========================================================================
# make_solution factory
# ===========================================================================


@pytest.mark.unit
class TestMakeSolution:
    """Tests for the make_solution() factory function."""

    def test_returns_solution_script_instance(self) -> None:
        """make_solution returns a SolutionScript instance with no args."""
        result = make_solution()
        assert isinstance(result, SolutionScript)

    def test_default_content_is_nonempty(self) -> None:
        """Default content is a non-empty string."""
        result = make_solution()
        assert isinstance(result.content, str)
        assert len(result.content) > 0

    def test_default_phase_is_init(self) -> None:
        """Default phase is SolutionPhase.INIT."""
        result = make_solution()
        assert result.phase == SolutionPhase.INIT

    def test_default_score_is_none(self) -> None:
        """Default score is None (unevaluated)."""
        result = make_solution()
        assert result.score is None

    def test_default_is_executable_is_true(self) -> None:
        """Default is_executable is True."""
        result = make_solution()
        assert result.is_executable is True

    def test_default_created_at_is_set(self) -> None:
        """created_at is auto-set to a datetime."""
        result = make_solution()
        assert isinstance(result.created_at, datetime)

    def test_override_content(self) -> None:
        """Content can be overridden via keyword argument."""
        result = make_solution(content="custom code")
        assert result.content == "custom code"

    def test_override_phase(self) -> None:
        """Phase can be overridden via keyword argument."""
        result = make_solution(phase=SolutionPhase.REFINED)
        assert result.phase == SolutionPhase.REFINED

    def test_override_score(self) -> None:
        """Score can be overridden via keyword argument."""
        result = make_solution(score=0.95)
        assert result.score == 0.95

    def test_override_is_executable(self) -> None:
        """is_executable can be overridden via keyword argument."""
        result = make_solution(is_executable=False)
        assert result.is_executable is False

    def test_override_source_model(self) -> None:
        """source_model can be overridden via keyword argument."""
        result = make_solution(source_model="xgboost")
        assert result.source_model == "xgboost"

    def test_multiple_overrides(self) -> None:
        """Multiple fields can be overridden simultaneously."""
        result = make_solution(
            content="import xgb",
            phase=SolutionPhase.ENSEMBLE,
            score=0.42,
            is_executable=False,
            source_model="lightgbm",
        )
        assert result.content == "import xgb"
        assert result.phase == SolutionPhase.ENSEMBLE
        assert result.score == 0.42
        assert result.is_executable is False
        assert result.source_model == "lightgbm"

    def test_each_call_returns_independent_instance(self) -> None:
        """Successive calls return distinct objects (no shared state)."""
        a = make_solution()
        b = make_solution()
        assert a is not b

    @given(st.text(min_size=1, max_size=200))
    @settings(max_examples=10)
    def test_arbitrary_content_produces_valid_solution(self, content: str) -> None:
        """Any non-empty content string produces a valid SolutionScript."""
        result = make_solution(content=content)
        assert result.content == content
        assert isinstance(result, SolutionScript)


# ===========================================================================
# make_task factory
# ===========================================================================


@pytest.mark.unit
class TestMakeTask:
    """Tests for the make_task() factory function."""

    def test_returns_task_description_instance(self) -> None:
        """make_task returns a TaskDescription instance with no args."""
        result = make_task()
        assert isinstance(result, TaskDescription)

    def test_default_competition_id_is_nonempty(self) -> None:
        """Default competition_id is a non-empty string."""
        result = make_task()
        assert isinstance(result.competition_id, str)
        assert len(result.competition_id) > 0

    def test_default_task_type_is_valid(self) -> None:
        """Default task_type is a valid TaskType enum member."""
        result = make_task()
        assert isinstance(result.task_type, TaskType)

    def test_default_data_modality_is_valid(self) -> None:
        """Default data_modality is a valid DataModality enum member."""
        result = make_task()
        assert isinstance(result.data_modality, DataModality)

    def test_default_metric_direction_is_valid(self) -> None:
        """Default metric_direction is a valid MetricDirection enum member."""
        result = make_task()
        assert isinstance(result.metric_direction, MetricDirection)

    def test_default_evaluation_metric_is_nonempty(self) -> None:
        """Default evaluation_metric is a non-empty string."""
        result = make_task()
        assert isinstance(result.evaluation_metric, str)
        assert len(result.evaluation_metric) > 0

    def test_default_description_is_nonempty(self) -> None:
        """Default description is a non-empty string."""
        result = make_task()
        assert isinstance(result.description, str)
        assert len(result.description) > 0

    def test_default_data_dir(self) -> None:
        """Default data_dir is set."""
        result = make_task()
        assert isinstance(result.data_dir, str)
        assert len(result.data_dir) > 0

    def test_default_output_dir(self) -> None:
        """Default output_dir is set."""
        result = make_task()
        assert isinstance(result.output_dir, str)
        assert len(result.output_dir) > 0

    def test_override_competition_id(self) -> None:
        """competition_id can be overridden."""
        result = make_task(competition_id="my-comp-2024")
        assert result.competition_id == "my-comp-2024"

    def test_override_task_type(self) -> None:
        """task_type can be overridden."""
        result = make_task(task_type=TaskType.REGRESSION)
        assert result.task_type == TaskType.REGRESSION

    def test_override_data_modality(self) -> None:
        """data_modality can be overridden."""
        result = make_task(data_modality=DataModality.IMAGE)
        assert result.data_modality == DataModality.IMAGE

    def test_override_metric_direction(self) -> None:
        """metric_direction can be overridden."""
        result = make_task(metric_direction=MetricDirection.MINIMIZE)
        assert result.metric_direction == MetricDirection.MINIMIZE

    def test_override_evaluation_metric(self) -> None:
        """evaluation_metric can be overridden."""
        result = make_task(evaluation_metric="rmse")
        assert result.evaluation_metric == "rmse"

    def test_override_description(self) -> None:
        """Description can be overridden."""
        result = make_task(description="Custom task description")
        assert result.description == "Custom task description"

    def test_override_data_dir(self) -> None:
        """data_dir can be overridden."""
        result = make_task(data_dir="/data/train")
        assert result.data_dir == "/data/train"

    def test_override_output_dir(self) -> None:
        """output_dir can be overridden."""
        result = make_task(output_dir="/submissions")
        assert result.output_dir == "/submissions"

    def test_multiple_overrides(self) -> None:
        """Multiple fields can be overridden simultaneously."""
        result = make_task(
            competition_id="custom",
            task_type=TaskType.IMAGE_CLASSIFICATION,
            metric_direction=MetricDirection.MINIMIZE,
        )
        assert result.competition_id == "custom"
        assert result.task_type == TaskType.IMAGE_CLASSIFICATION
        assert result.metric_direction == MetricDirection.MINIMIZE

    def test_each_call_returns_independent_instance(self) -> None:
        """Successive calls return distinct objects."""
        a = make_task()
        b = make_task()
        assert a is not b

    def test_result_is_frozen(self) -> None:
        """TaskDescription is immutable (frozen)."""
        result = make_task()
        with pytest.raises(Exception):  # noqa: B017
            result.competition_id = "changed"  # type: ignore[misc]


# ===========================================================================
# make_config factory
# ===========================================================================


@pytest.mark.unit
class TestMakeConfig:
    """Tests for the make_config() factory function."""

    def test_returns_pipeline_config_instance(self) -> None:
        """make_config returns a PipelineConfig instance with no args."""
        result = make_config()
        assert isinstance(result, PipelineConfig)

    def test_default_num_retrieved_models(self) -> None:
        """Default num_retrieved_models matches PipelineConfig default."""
        result = make_config()
        assert result.num_retrieved_models == 4

    def test_default_time_limit_seconds(self) -> None:
        """Default time_limit_seconds matches PipelineConfig default."""
        result = make_config()
        assert result.time_limit_seconds == 86400

    def test_default_max_debug_attempts(self) -> None:
        """Default max_debug_attempts matches PipelineConfig default."""
        result = make_config()
        assert result.max_debug_attempts == 3

    def test_override_num_retrieved_models(self) -> None:
        """num_retrieved_models can be overridden."""
        result = make_config(num_retrieved_models=8)
        assert result.num_retrieved_models == 8

    def test_override_outer_loop_steps(self) -> None:
        """outer_loop_steps can be overridden."""
        result = make_config(outer_loop_steps=10)
        assert result.outer_loop_steps == 10

    def test_override_inner_loop_steps(self) -> None:
        """inner_loop_steps can be overridden."""
        result = make_config(inner_loop_steps=6)
        assert result.inner_loop_steps == 6

    def test_override_time_limit_seconds(self) -> None:
        """time_limit_seconds can be overridden."""
        result = make_config(time_limit_seconds=3600)
        assert result.time_limit_seconds == 3600

    def test_override_max_budget_usd(self) -> None:
        """max_budget_usd can be overridden."""
        result = make_config(max_budget_usd=50.0)
        assert result.max_budget_usd == 50.0

    def test_override_model(self) -> None:
        """Model can be overridden."""
        result = make_config(model="opus")
        assert result.model == "opus"

    def test_multiple_overrides(self) -> None:
        """Multiple fields can be overridden simultaneously."""
        result = make_config(
            num_retrieved_models=2,
            inner_loop_steps=8,
            model="haiku",
        )
        assert result.num_retrieved_models == 2
        assert result.inner_loop_steps == 8
        assert result.model == "haiku"

    def test_each_call_returns_independent_instance(self) -> None:
        """Successive calls return distinct objects."""
        a = make_config()
        b = make_config()
        assert a is not b

    def test_result_is_frozen(self) -> None:
        """PipelineConfig is immutable (frozen)."""
        result = make_config()
        with pytest.raises(Exception):  # noqa: B017
            result.model = "changed"  # type: ignore[misc]

    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=10)
    def test_arbitrary_positive_num_retrieved_models(self, n: int) -> None:
        """Any positive integer produces a valid config."""
        result = make_config(num_retrieved_models=n)
        assert result.num_retrieved_models == n
        assert isinstance(result, PipelineConfig)


# ===========================================================================
# make_eval_result factory
# ===========================================================================


@pytest.mark.unit
class TestMakeEvalResult:
    """Tests for the make_eval_result() factory function."""

    def test_returns_evaluation_result_instance(self) -> None:
        """make_eval_result returns an EvaluationResult instance with no args."""
        result = make_eval_result()
        assert isinstance(result, EvaluationResult)

    def test_default_score_is_numeric(self) -> None:
        """Default score is a numeric value (not None)."""
        result = make_eval_result()
        assert result.score is not None
        assert isinstance(result.score, float)

    def test_default_stdout_is_string(self) -> None:
        """Default stdout is a string."""
        result = make_eval_result()
        assert isinstance(result.stdout, str)

    def test_default_stderr_is_empty(self) -> None:
        """Default stderr is an empty string."""
        result = make_eval_result()
        assert result.stderr == ""

    def test_default_exit_code_is_zero(self) -> None:
        """Default exit_code is 0 (success)."""
        result = make_eval_result()
        assert result.exit_code == 0

    def test_default_duration_is_positive(self) -> None:
        """Default duration_seconds is positive."""
        result = make_eval_result()
        assert result.duration_seconds > 0

    def test_default_is_error_is_false(self) -> None:
        """Default is_error is False."""
        result = make_eval_result()
        assert result.is_error is False

    def test_default_error_traceback_is_none(self) -> None:
        """Default error_traceback is None."""
        result = make_eval_result()
        assert result.error_traceback is None

    def test_override_score(self) -> None:
        """Score can be overridden."""
        result = make_eval_result(score=0.42)
        assert result.score == 0.42

    def test_override_score_to_none(self) -> None:
        """Score can be overridden to None (parse failure)."""
        result = make_eval_result(score=None)
        assert result.score is None

    def test_override_stdout(self) -> None:
        """Stdout can be overridden."""
        result = make_eval_result(stdout="custom output")
        assert result.stdout == "custom output"

    def test_override_stderr(self) -> None:
        """Stderr can be overridden."""
        result = make_eval_result(stderr="error output")
        assert result.stderr == "error output"

    def test_override_exit_code(self) -> None:
        """exit_code can be overridden."""
        result = make_eval_result(exit_code=1)
        assert result.exit_code == 1

    def test_override_duration_seconds(self) -> None:
        """duration_seconds can be overridden."""
        result = make_eval_result(duration_seconds=99.9)
        assert result.duration_seconds == 99.9

    def test_override_is_error(self) -> None:
        """is_error can be overridden."""
        result = make_eval_result(is_error=True)
        assert result.is_error is True

    def test_override_error_traceback(self) -> None:
        """error_traceback can be overridden."""
        result = make_eval_result(error_traceback="Traceback...")
        assert result.error_traceback == "Traceback..."

    def test_multiple_overrides_for_error_case(self) -> None:
        """Can construct an error result with multiple overrides."""
        result = make_eval_result(
            score=None,
            exit_code=1,
            is_error=True,
            stderr="Traceback (most recent call last):\nValueError",
            error_traceback="ValueError: bad input",
        )
        assert result.score is None
        assert result.exit_code == 1
        assert result.is_error is True
        assert "Traceback" in result.stderr
        assert result.error_traceback == "ValueError: bad input"

    def test_each_call_returns_independent_instance(self) -> None:
        """Successive calls return distinct objects."""
        a = make_eval_result()
        b = make_eval_result()
        assert a is not b

    def test_result_is_frozen(self) -> None:
        """EvaluationResult is immutable (frozen)."""
        result = make_eval_result()
        with pytest.raises(Exception):  # noqa: B017
            result.score = 0.99  # type: ignore[misc]

    @given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    @settings(max_examples=10)
    def test_arbitrary_score_produces_valid_result(self, score: float) -> None:
        """Any float score in [0,1] produces a valid EvaluationResult."""
        result = make_eval_result(score=score)
        assert result.score == score
        assert isinstance(result, EvaluationResult)


# ===========================================================================
# mock_client fixture
# ===========================================================================


@pytest.mark.unit
class TestMockClientFixture:
    """Tests for the mock_client fixture."""

    def test_is_async_mock(self, mock_client: Any) -> None:
        """mock_client is an AsyncMock instance."""
        assert isinstance(mock_client, AsyncMock)

    def test_has_send_message_attribute(self, mock_client: Any) -> None:
        """mock_client has a send_message attribute."""
        assert hasattr(mock_client, "send_message")

    def test_send_message_is_async_mock(self, mock_client: Any) -> None:
        """mock_client.send_message is an AsyncMock."""
        assert isinstance(mock_client.send_message, AsyncMock)

    async def test_send_message_returns_string(self, mock_client: Any) -> None:
        """send_message returns a string by default."""
        result = await mock_client.send_message(agent_type="test", message="hi")
        assert isinstance(result, str)

    async def test_send_message_default_return_value(self, mock_client: Any) -> None:
        """send_message returns a predictable default response."""
        result = await mock_client.send_message(agent_type="test", message="hi")
        assert result == "response"

    async def test_send_message_is_awaitable(self, mock_client: Any) -> None:
        """send_message can be awaited (is a coroutine)."""
        coro = mock_client.send_message(agent_type="test", message="hi")
        assert asyncio.iscoroutine(coro) or hasattr(coro, "__await__")
        result = await coro
        assert isinstance(result, str)

    async def test_send_message_tracks_calls(self, mock_client: Any) -> None:
        """send_message records call arguments for assertion."""
        await mock_client.send_message(agent_type="coder", message="fix bug")
        mock_client.send_message.assert_called_once_with(
            agent_type="coder", message="fix bug"
        )

    async def test_send_message_return_value_configurable(
        self, mock_client: Any
    ) -> None:
        """send_message return value can be reconfigured per-test."""
        mock_client.send_message.return_value = "custom response"
        result = await mock_client.send_message(agent_type="test", message="hi")
        assert result == "custom response"

    async def test_send_message_side_effect_configurable(
        self, mock_client: Any
    ) -> None:
        """send_message side_effect can be set to simulate errors."""
        mock_client.send_message.side_effect = RuntimeError("API error")
        with pytest.raises(RuntimeError, match="API error"):
            await mock_client.send_message(agent_type="test", message="hi")

    def test_each_test_gets_fresh_mock(self, mock_client: Any) -> None:
        """mock_client is not called yet (fresh per test)."""
        mock_client.send_message.assert_not_called()


# ===========================================================================
# mock_registry fixture
# ===========================================================================


@pytest.mark.unit
class TestMockRegistryFixture:
    """Tests for the mock_registry fixture."""

    def test_is_mock_instance(self, mock_registry: Any) -> None:
        """mock_registry is a MagicMock instance."""
        assert isinstance(mock_registry, MagicMock)

    def test_has_get_method(self, mock_registry: Any) -> None:
        """mock_registry has a get() method."""
        assert hasattr(mock_registry, "get")
        assert callable(mock_registry.get)

    def test_get_returns_object_with_render(self, mock_registry: Any) -> None:
        """mock_registry.get() returns an object with a render() method."""
        template = mock_registry.get(AgentType.CODER)
        assert hasattr(template, "render")
        assert callable(template.render)

    def test_get_render_returns_string(self, mock_registry: Any) -> None:
        """The template from mock_registry.get().render() returns a string."""
        template = mock_registry.get(AgentType.CODER)
        rendered = template.render(code_block="x = 1", plan="improve it")
        assert isinstance(rendered, str)

    def test_get_render_returns_predictable_string(self, mock_registry: Any) -> None:
        """render() returns a predictable, non-empty string."""
        template = mock_registry.get(AgentType.PLANNER)
        rendered = template.render(some_var="value")
        assert len(rendered) > 0

    def test_get_accepts_agent_type_arg(self, mock_registry: Any) -> None:
        """get() accepts an AgentType as the first argument."""
        # Should not raise
        mock_registry.get(AgentType.RETRIEVER)
        mock_registry.get(AgentType.DEBUGGER)

    def test_get_accepts_variant_kwarg(self, mock_registry: Any) -> None:
        """get() accepts a variant keyword argument."""
        # Should not raise
        mock_registry.get(AgentType.LEAKAGE, variant="detection")

    def test_each_test_gets_fresh_registry(self, mock_registry: Any) -> None:
        """mock_registry is fresh per test (no prior calls recorded)."""
        # After fixture creation, get has not been called yet
        # (the fixture creates it, but does not call get)
        # We verify by calling get once and checking call count
        mock_registry.get(AgentType.INIT)
        assert mock_registry.get.call_count == 1


# ===========================================================================
# tmp_working_dir fixture
# ===========================================================================


@pytest.mark.unit
class TestTmpWorkingDirFixture:
    """Tests for the tmp_working_dir fixture."""

    def test_returns_path_object(self, tmp_working_dir: Path) -> None:
        """tmp_working_dir returns a Path object."""
        assert isinstance(tmp_working_dir, Path)

    def test_directory_exists(self, tmp_working_dir: Path) -> None:
        """The returned path exists and is a directory."""
        assert tmp_working_dir.exists()
        assert tmp_working_dir.is_dir()

    def test_input_subdir_exists(self, tmp_working_dir: Path) -> None:
        """An input/ subdirectory exists within the working dir."""
        input_dir = tmp_working_dir / "input"
        assert input_dir.exists()
        assert input_dir.is_dir()

    def test_final_subdir_exists(self, tmp_working_dir: Path) -> None:
        """A final/ subdirectory exists within the working dir."""
        final_dir = tmp_working_dir / "final"
        assert final_dir.exists()
        assert final_dir.is_dir()

    def test_input_has_dummy_file(self, tmp_working_dir: Path) -> None:
        """The input/ subdirectory contains at least one file."""
        input_dir = tmp_working_dir / "input"
        files = list(input_dir.iterdir())
        assert len(files) >= 1

    def test_dummy_file_is_readable(self, tmp_working_dir: Path) -> None:
        """The dummy file in input/ is readable and non-empty."""
        input_dir = tmp_working_dir / "input"
        files = list(input_dir.iterdir())
        assert len(files) >= 1
        content = files[0].read_text()
        assert len(content) > 0

    def test_final_subdir_is_initially_empty(self, tmp_working_dir: Path) -> None:
        """The final/ subdirectory starts empty."""
        final_dir = tmp_working_dir / "final"
        files = list(final_dir.iterdir())
        assert len(files) == 0

    def test_can_write_to_final_dir(self, tmp_working_dir: Path) -> None:
        """Files can be written into the final/ subdirectory."""
        final_dir = tmp_working_dir / "final"
        output_file = final_dir / "submission.csv"
        output_file.write_text("id,target\n1,0\n")
        assert output_file.exists()
        assert output_file.read_text() == "id,target\n1,0\n"

    def test_each_test_gets_independent_directory(
        self, tmp_working_dir: Path, tmp_path: Path
    ) -> None:
        """tmp_working_dir is unique per test invocation."""
        # Write a sentinel file
        sentinel = tmp_working_dir / "sentinel.txt"
        sentinel.write_text("marker")
        assert sentinel.exists()
        # The directory is within tmp_path, which is unique per test
        assert str(tmp_working_dir).startswith(str(tmp_path))
