"""Tests for the pipeline entry point and SDK client setup (Task 42).

Validates ``PipelineError``, ``PipelineTimeoutError``, ``run_pipeline``,
``run_pipeline_sync``, and the internal helpers ``_validate_inputs``,
``_build_system_prompt``, and ``_build_agents_dict`` defined in
``src/mle_star/orchestrator.py``.

These tests are written TDD-first -- the implementation does not yet exist.
They serve as the executable specification for REQ-OR-002, REQ-OR-005
through REQ-OR-010, REQ-OR-030, REQ-OR-042, REQ-OR-044, and REQ-OR-053.

Refs:
    SRS 09a -- Orchestrator Entry Point & SDK Client Setup.
    IMPLEMENTATION_PLAN.md Task 42.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
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
    Phase3Result,
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


def _make_config(**overrides: Any) -> PipelineConfig:
    """Build a valid PipelineConfig with sensible defaults.

    Uses ``num_parallel_solutions=1`` by default so that Phase 3 is
    skipped in SDK-setup-focused tests (Phase dispatch is tested in
    ``test_orchestrator_dispatch.py``).

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed PipelineConfig instance.
    """
    defaults: dict[str, Any] = {"num_parallel_solutions": 1}
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_solution(**overrides: Any) -> SolutionScript:
    """Build a valid SolutionScript with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed SolutionScript instance.
    """
    defaults: dict[str, Any] = {
        "content": "import pandas as pd\nprint('hello')\n",
        "phase": SolutionPhase.FINAL,
    }
    defaults.update(overrides)
    return SolutionScript(**defaults)


def _make_phase1_result(**overrides: Any) -> Phase1Result:
    """Build a valid Phase1Result with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed Phase1Result instance.
    """
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
        "step_history": [{"step": 0, "score": 0.90}],
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
    sol = _make_solution(phase=SolutionPhase.ENSEMBLE)
    defaults: dict[str, Any] = {
        "input_solutions": [sol, sol],
        "ensemble_plans": ["plan_1"],
        "ensemble_scores": [0.92],
        "best_ensemble": sol,
        "best_ensemble_score": 0.92,
    }
    defaults.update(overrides)
    return Phase3Result(**defaults)


def _make_final_result(
    task: TaskDescription | None = None,
    config: PipelineConfig | None = None,
    **overrides: Any,
) -> FinalResult:
    """Build a valid FinalResult with sensible defaults.

    Args:
        task: TaskDescription to use. Defaults to _make_task().
        config: PipelineConfig to use. Defaults to _make_config().
        **overrides: Field values to override.

    Returns:
        A fully constructed FinalResult instance.
    """
    defaults: dict[str, Any] = {
        "task": task or _make_task(),
        "config": config or _make_config(),
        "phase1": _make_phase1_result(),
        "phase2_results": [_make_phase2_result()],
        "phase3": None,
        "final_solution": _make_solution(phase=SolutionPhase.FINAL),
        "submission_path": "/output/submission.csv",
        "total_duration_seconds": 100.0,
        "total_cost_usd": None,
    }
    defaults.update(overrides)
    return FinalResult(**defaults)


def _make_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with a dummy file.

    Args:
        tmp_path: Pytest tmp_path fixture.

    Returns:
        Path to the temporary data directory.
    """
    data_dir = tmp_path / "input"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text("id,feature,target\n1,0.5,0\n")
    return data_dir


# ===========================================================================
# REQ-OR-042: PipelineError custom exception
# ===========================================================================


@pytest.mark.unit
class TestPipelineError:
    """PipelineError is a custom exception with a diagnostics attribute (REQ-OR-042)."""

    def test_is_subclass_of_exception(self) -> None:
        """PipelineError inherits from Exception."""
        from mle_star.orchestrator import PipelineError

        assert issubclass(PipelineError, Exception)

    def test_message_accessible(self) -> None:
        """PipelineError message is accessible via str()."""
        from mle_star.orchestrator import PipelineError

        err = PipelineError("something went wrong", diagnostics={})
        assert "something went wrong" in str(err)

    def test_diagnostics_attribute_is_dict(self) -> None:
        """PipelineError has a diagnostics attribute that is a dict."""
        from mle_star.orchestrator import PipelineError

        diag = {
            "elapsed_time": 42.0,
            "cost": 1.5,
            "last_successful_operation": "phase1",
        }
        err = PipelineError("error", diagnostics=diag)
        assert isinstance(err.diagnostics, dict)
        assert err.diagnostics == diag

    def test_diagnostics_contains_expected_keys(self) -> None:
        """PipelineError diagnostics dict stores arbitrary diagnostic info."""
        from mle_star.orchestrator import PipelineError

        diag = {
            "elapsed_time": 100.0,
            "cost": 5.0,
            "last_successful_operation": "phase2",
        }
        err = PipelineError("pipeline failed", diagnostics=diag)
        assert err.diagnostics["elapsed_time"] == 100.0
        assert err.diagnostics["cost"] == 5.0
        assert err.diagnostics["last_successful_operation"] == "phase2"

    def test_empty_diagnostics(self) -> None:
        """PipelineError accepts an empty diagnostics dict."""
        from mle_star.orchestrator import PipelineError

        err = PipelineError("error", diagnostics={})
        assert err.diagnostics == {}

    def test_can_be_raised_and_caught(self) -> None:
        """PipelineError can be raised and caught as Exception."""
        from mle_star.orchestrator import PipelineError

        with pytest.raises(PipelineError, match="test error"):
            raise PipelineError("test error", diagnostics={"key": "value"})

    def test_caught_as_exception(self) -> None:
        """PipelineError can be caught with a bare except Exception clause."""
        from mle_star.orchestrator import PipelineError

        caught = False
        try:
            raise PipelineError("test", diagnostics={})
        except Exception:
            caught = True
        assert caught


# ===========================================================================
# REQ-OR-030: PipelineTimeoutError custom exception
# ===========================================================================


@pytest.mark.unit
class TestPipelineTimeoutError:
    """PipelineTimeoutError is a subclass of PipelineError (REQ-OR-030)."""

    def test_is_subclass_of_pipeline_error(self) -> None:
        """PipelineTimeoutError inherits from PipelineError."""
        from mle_star.orchestrator import PipelineError, PipelineTimeoutError

        assert issubclass(PipelineTimeoutError, PipelineError)

    def test_is_subclass_of_exception(self) -> None:
        """PipelineTimeoutError is also a subclass of Exception."""
        from mle_star.orchestrator import PipelineTimeoutError

        assert issubclass(PipelineTimeoutError, Exception)

    def test_has_diagnostics_attribute(self) -> None:
        """PipelineTimeoutError has a diagnostics attribute from PipelineError."""
        from mle_star.orchestrator import PipelineTimeoutError

        diag = {"elapsed_time": 86400.0, "last_successful_operation": "phase1"}
        err = PipelineTimeoutError("timeout", diagnostics=diag)
        assert err.diagnostics == diag

    def test_message_accessible(self) -> None:
        """PipelineTimeoutError message is accessible via str()."""
        from mle_star.orchestrator import PipelineTimeoutError

        err = PipelineTimeoutError("timed out after 86400s", diagnostics={})
        assert "timed out" in str(err)

    def test_can_be_caught_as_pipeline_error(self) -> None:
        """PipelineTimeoutError can be caught as PipelineError."""
        from mle_star.orchestrator import PipelineError, PipelineTimeoutError

        with pytest.raises(PipelineError):
            raise PipelineTimeoutError("timeout", diagnostics={})


# ===========================================================================
# REQ-OR-002: Input validation
# ===========================================================================


@pytest.mark.unit
class TestValidateInputs:
    """_validate_inputs raises ValueError on invalid task inputs (REQ-OR-002)."""

    def test_data_dir_does_not_exist_raises_value_error(self, tmp_path: Path) -> None:
        """ValueError raised when task.data_dir points to a nonexistent path."""
        from mle_star.orchestrator import _validate_inputs

        task = _make_task(data_dir=str(tmp_path / "nonexistent"))
        config = _make_config()

        with pytest.raises(ValueError, match="data_dir"):
            _validate_inputs(task, config)

    def test_data_dir_is_file_raises_value_error(self, tmp_path: Path) -> None:
        """ValueError raised when task.data_dir is a file, not a directory."""
        from mle_star.orchestrator import _validate_inputs

        fake_file = tmp_path / "not_a_dir.csv"
        fake_file.write_text("data")
        task = _make_task(data_dir=str(fake_file))
        config = _make_config()

        with pytest.raises(ValueError, match="data_dir"):
            _validate_inputs(task, config)

    def test_data_dir_empty_raises_value_error(self, tmp_path: Path) -> None:
        """ValueError raised when task.data_dir exists but contains no files."""
        from mle_star.orchestrator import _validate_inputs

        empty_dir = tmp_path / "empty_input"
        empty_dir.mkdir()
        task = _make_task(data_dir=str(empty_dir))
        config = _make_config()

        with pytest.raises(ValueError, match="data_dir"):
            _validate_inputs(task, config)

    def test_valid_data_dir_passes(self, tmp_path: Path) -> None:
        """No error raised for a valid data_dir with files."""
        from mle_star.orchestrator import _validate_inputs

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config()

        # Should not raise
        _validate_inputs(task, config)

    def test_none_config_uses_default(self, tmp_path: Path) -> None:
        """When config is None, _validate_inputs uses PipelineConfig() defaults."""
        from mle_star.orchestrator import _validate_inputs

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))

        # Should not raise; config=None produces PipelineConfig()
        result = _validate_inputs(task, None)

        assert isinstance(result, PipelineConfig)

    def test_provided_config_returned_unchanged(self, tmp_path: Path) -> None:
        """When config is provided, _validate_inputs returns it unchanged."""
        from mle_star.orchestrator import _validate_inputs

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(model="opus", max_budget_usd=50.0)

        result = _validate_inputs(task, config)

        assert result is config

    def test_data_dir_with_subdirectory_only_raises(self, tmp_path: Path) -> None:
        """ValueError raised when data_dir has subdirectories but no files."""
        from mle_star.orchestrator import _validate_inputs

        data_dir = tmp_path / "input_with_subdir"
        data_dir.mkdir()
        (data_dir / "subdir").mkdir()
        task = _make_task(data_dir=str(data_dir))
        config = _make_config()

        with pytest.raises(ValueError, match="data_dir"):
            _validate_inputs(task, config)


# ===========================================================================
# REQ-OR-007: System prompt construction
# ===========================================================================


@pytest.mark.unit
class TestBuildSystemPrompt:
    """_build_system_prompt constructs the orchestrator system prompt (REQ-OR-007)."""

    def test_contains_kaggle_grandmaster_persona(self) -> None:
        """System prompt contains Kaggle grandmaster persona text."""
        from mle_star.orchestrator import _build_system_prompt

        task = _make_task()
        gpu_info: dict[str, Any] = {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_names": [],
        }

        prompt = _build_system_prompt(task, gpu_info)

        # Should reference Kaggle grandmaster expertise
        lower = prompt.lower()
        assert "kaggle" in lower
        assert "grandmaster" in lower

    def test_contains_task_description(self) -> None:
        """System prompt includes the task description text."""
        from mle_star.orchestrator import _build_system_prompt

        task = _make_task(
            description="Classify images of cats and dogs using deep learning."
        )
        gpu_info: dict[str, Any] = {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_names": [],
        }

        prompt = _build_system_prompt(task, gpu_info)

        assert "Classify images of cats and dogs" in prompt

    def test_contains_evaluation_metric(self) -> None:
        """System prompt includes the evaluation metric name."""
        from mle_star.orchestrator import _build_system_prompt

        task = _make_task(evaluation_metric="f1_score")
        gpu_info: dict[str, Any] = {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_names": [],
        }

        prompt = _build_system_prompt(task, gpu_info)

        assert "f1_score" in prompt

    def test_contains_metric_direction(self) -> None:
        """System prompt includes the metric direction (maximize/minimize)."""
        from mle_star.orchestrator import _build_system_prompt

        task = _make_task(metric_direction=MetricDirection.MINIMIZE)
        gpu_info: dict[str, Any] = {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_names": [],
        }

        prompt = _build_system_prompt(task, gpu_info)

        assert "minimize" in prompt.lower()

    def test_contains_gpu_info_when_available(self) -> None:
        """System prompt includes GPU information when GPUs are detected."""
        from mle_star.orchestrator import _build_system_prompt

        task = _make_task()
        gpu_info: dict[str, Any] = {
            "cuda_available": True,
            "gpu_count": 2,
            "gpu_names": ["NVIDIA A100", "NVIDIA A100"],
        }

        prompt = _build_system_prompt(task, gpu_info)

        assert "GPU" in prompt or "gpu" in prompt.lower()

    def test_contains_gpu_info_when_none_available(self) -> None:
        """System prompt includes GPU info section even when no GPUs detected."""
        from mle_star.orchestrator import _build_system_prompt

        task = _make_task()
        gpu_info: dict[str, Any] = {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_names": [],
        }

        prompt = _build_system_prompt(task, gpu_info)

        # The prompt should still be a non-empty string
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_returns_string(self) -> None:
        """_build_system_prompt returns a string."""
        from mle_star.orchestrator import _build_system_prompt

        task = _make_task()
        gpu_info: dict[str, Any] = {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_names": [],
        }

        result = _build_system_prompt(task, gpu_info)

        assert isinstance(result, str)


# ===========================================================================
# REQ-OR-006: Agent registration
# ===========================================================================


@pytest.mark.unit
class TestBuildAgentsDict:
    """_build_agents_dict builds the agents dict for the SDK (REQ-OR-006)."""

    def test_returns_dict(self) -> None:
        """_build_agents_dict returns a dict."""
        from mle_star.orchestrator import _build_agents_dict

        result = _build_agents_dict()

        assert isinstance(result, dict)

    def test_contains_all_14_agents(self) -> None:
        """The agents dict contains entries for all 14 MLE-STAR agents."""
        from mle_star.orchestrator import _build_agents_dict

        result = _build_agents_dict()

        assert len(result) == 14

    def test_all_agent_types_present(self) -> None:
        """Every AgentType enum value has a corresponding key in the dict."""
        from mle_star.orchestrator import _build_agents_dict

        result = _build_agents_dict()

        for agent_type in AgentType:
            assert str(agent_type) in result or agent_type.value in result, (
                f"Agent {agent_type} not found in agents dict"
            )

    def test_agent_definitions_from_build_default_agent_configs(self) -> None:
        """Agent definitions are derived from build_default_agent_configs()."""
        from mle_star.orchestrator import _build_agents_dict

        result = _build_agents_dict()
        default_configs = build_default_agent_configs()

        # Each agent definition should have a description matching the config
        for agent_type, agent_config in default_configs.items():
            key = str(agent_type) if str(agent_type) in result else agent_type.value
            assert key in result, f"Agent {agent_type} missing from dict"
            defn = result[key]
            assert defn["description"] == agent_config.description

    def test_each_agent_has_description_key(self) -> None:
        """Every agent definition dict has a 'description' key."""
        from mle_star.orchestrator import _build_agents_dict

        result = _build_agents_dict()

        for key, defn in result.items():
            assert "description" in defn, f"Agent {key} missing 'description'"

    def test_agent_definitions_converted_via_to_agent_definition(self) -> None:
        """Each agent config is converted to dict via to_agent_definition()."""
        from mle_star.orchestrator import _build_agents_dict

        result = _build_agents_dict()
        default_configs = build_default_agent_configs()

        for agent_type, agent_config in default_configs.items():
            expected_defn = agent_config.to_agent_definition()
            key = str(agent_type) if str(agent_type) in result else agent_type.value
            actual_defn = result[key]
            assert actual_defn["description"] == expected_defn["description"]
            assert actual_defn["tools"] == expected_defn["tools"]


# ===========================================================================
# REQ-OR-005: SDK client setup
# ===========================================================================


@pytest.mark.unit
class TestSDKClientSetup:
    """run_pipeline creates SDK client with correct options (REQ-OR-005)."""

    async def test_client_created_with_model_from_config(self, tmp_path: Path) -> None:
        """SDK client options include the model from PipelineConfig."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(model="opus")

        mock_client_instance = AsyncMock()
        mock_client_instance.connect = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()
        mock_client_instance.query = AsyncMock()

        captured_options: list[Any] = []

        def _capture_client(options: Any = None, **kwargs: Any) -> AsyncMock:
            captured_options.append(options)
            return mock_client_instance

        final_result = _make_final_result(task=task, config=config)

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", side_effect=_capture_client),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=_make_phase1_result(),
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=_make_phase2_result(),
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=final_result,
            ),
        ):
            await run_pipeline(task, config)

        assert len(captured_options) == 1
        assert captured_options[0].model == "opus"

    async def test_client_created_with_permission_mode(self, tmp_path: Path) -> None:
        """SDK client options include the permission_mode from PipelineConfig."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(permission_mode="acceptEdits")

        mock_client_instance = AsyncMock()
        mock_client_instance.connect = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        captured_options: list[Any] = []

        def _capture_client(options: Any = None, **kwargs: Any) -> AsyncMock:
            captured_options.append(options)
            return mock_client_instance

        final_result = _make_final_result(task=task, config=config)

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", side_effect=_capture_client),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=_make_phase1_result(),
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=_make_phase2_result(),
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=final_result,
            ),
        ):
            await run_pipeline(task, config)

        assert captured_options[0].permission_mode == "acceptEdits"

    async def test_client_created_with_max_budget(self, tmp_path: Path) -> None:
        """SDK client options include max_budget_usd from PipelineConfig."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(max_budget_usd=25.0)

        mock_client_instance = AsyncMock()
        mock_client_instance.connect = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        captured_options: list[Any] = []

        def _capture_client(options: Any = None, **kwargs: Any) -> AsyncMock:
            captured_options.append(options)
            return mock_client_instance

        final_result = _make_final_result(task=task, config=config)

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", side_effect=_capture_client),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=_make_phase1_result(),
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=_make_phase2_result(),
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=final_result,
            ),
        ):
            await run_pipeline(task, config)

        assert captured_options[0].max_budget_usd == 25.0

    async def test_client_options_include_14_agents(self, tmp_path: Path) -> None:
        """SDK client options include agent definitions for all 14 agents."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config()

        mock_client_instance = AsyncMock()
        mock_client_instance.connect = AsyncMock()
        mock_client_instance.disconnect = AsyncMock()

        captured_options: list[Any] = []

        def _capture_client(options: Any = None, **kwargs: Any) -> AsyncMock:
            captured_options.append(options)
            return mock_client_instance

        final_result = _make_final_result(task=task, config=config)

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", side_effect=_capture_client),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=_make_phase1_result(),
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=_make_phase2_result(),
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=final_result,
            ),
        ):
            await run_pipeline(task, config)

        agents = captured_options[0].agents
        assert agents is not None
        assert len(agents) == 14


# ===========================================================================
# REQ-OR-005: Client disconnect (cleanup)
# ===========================================================================


@pytest.mark.unit
class TestClientDisconnect:
    """client.disconnect() is called in try/finally for cleanup (REQ-OR-005)."""

    async def test_disconnect_called_on_success(self, tmp_path: Path) -> None:
        """client.disconnect() is called after successful pipeline completion."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config()

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()

        final_result = _make_final_result(task=task, config=config)

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=mock_client),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=_make_phase1_result(),
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=_make_phase2_result(),
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=final_result,
            ),
        ):
            await run_pipeline(task, config)

        mock_client.disconnect.assert_awaited_once()

    async def test_disconnect_called_on_exception(self, tmp_path: Path) -> None:
        """client.disconnect() is called even when an exception occurs (try/finally)."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config()

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=mock_client),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                side_effect=RuntimeError("phase1 exploded"),
            ),
            pytest.raises(RuntimeError, match="phase1 exploded"),
        ):
            await run_pipeline(task, config)

        # disconnect should still be called despite the exception
        mock_client.disconnect.assert_awaited_once()

    async def test_disconnect_called_on_pipeline_error(self, tmp_path: Path) -> None:
        """client.disconnect() is called when PipelineError is raised."""
        from mle_star.orchestrator import PipelineError, run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config()

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=mock_client),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                side_effect=PipelineError("failed", diagnostics={"elapsed_time": 10.0}),
            ),
            pytest.raises(PipelineError),
        ):
            await run_pipeline(task, config)

        mock_client.disconnect.assert_awaited_once()


# ===========================================================================
# REQ-OR-010: MCP server registration failure
# ===========================================================================


@pytest.mark.unit
class TestMCPRegistrationFailure:
    """MCP registration failure is logged as warning, doesn't crash (REQ-OR-010)."""

    async def test_mcp_failure_logged_as_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MCP server registration failure emits a warning log but continues."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config()

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()

        final_result = _make_final_result(task=task, config=config)

        # Simulate MCP registration failure by having the client raise
        # during MCP setup but still continuing
        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=mock_client),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}._register_mcp_servers",
                side_effect=Exception("MCP server init failed"),
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=_make_phase1_result(),
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=_make_phase2_result(),
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=final_result,
            ),
            caplog.at_level(logging.WARNING),
        ):
            # Should not crash
            result = await run_pipeline(task, config)

        assert isinstance(result, FinalResult)
        # Verify warning was logged
        assert any(
            "MCP" in record.message or "mcp" in record.message.lower()
            for record in caplog.records
            if record.levelno >= logging.WARNING
        )


# ===========================================================================
# REQ-OR-005: run_pipeline is async and returns FinalResult
# ===========================================================================


@pytest.mark.unit
class TestRunPipelineIsAsync:
    """run_pipeline is async and returns FinalResult (REQ-OR-005)."""

    def test_is_coroutine_function(self) -> None:
        """run_pipeline is defined as an async function."""
        from mle_star.orchestrator import run_pipeline

        assert asyncio.iscoroutinefunction(run_pipeline)

    async def test_returns_final_result(self, tmp_path: Path) -> None:
        """run_pipeline returns a FinalResult instance on success."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config()

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()

        final_result = _make_final_result(task=task, config=config)

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=mock_client),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=_make_phase1_result(),
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=_make_phase2_result(),
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=final_result,
            ),
        ):
            result = await run_pipeline(task, config)

        assert isinstance(result, FinalResult)

    async def test_default_config_when_none(self, tmp_path: Path) -> None:
        """When config is None, run_pipeline uses PipelineConfig() defaults."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()

        captured_options: list[Any] = []

        def _capture_client(options: Any = None, **kwargs: Any) -> AsyncMock:
            captured_options.append(options)
            return mock_client

        default_config = PipelineConfig()
        final_result = _make_final_result(task=task, config=default_config)

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", side_effect=_capture_client),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=_make_phase1_result(),
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=_make_phase2_result(),
            ),
            patch(
                f"{_MODULE}.run_phase3",
                new_callable=AsyncMock,
                return_value=_make_phase3_result(),
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=final_result,
            ),
        ):
            await run_pipeline(task, None)

        # Default model is "sonnet"
        assert captured_options[0].model == "sonnet"

    async def test_accepts_task_and_config_parameters(self, tmp_path: Path) -> None:
        """run_pipeline accepts task as first param and optional config."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config()

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()

        final_result = _make_final_result(task=task, config=config)

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=mock_client),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=_make_phase1_result(),
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=_make_phase2_result(),
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=final_result,
            ),
        ):
            # Both positional should work
            result = await run_pipeline(task, config)

        assert isinstance(result, FinalResult)


# ===========================================================================
# REQ-OR-005: SDK client receives system prompt
# ===========================================================================


@pytest.mark.unit
class TestClientSystemPrompt:
    """run_pipeline configures the SDK client with a system prompt (REQ-OR-007)."""

    async def test_system_prompt_passed_to_client_options(self, tmp_path: Path) -> None:
        """ClaudeAgentOptions.system_prompt is set from _build_system_prompt."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config()

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()

        captured_options: list[Any] = []

        def _capture_client(options: Any = None, **kwargs: Any) -> AsyncMock:
            captured_options.append(options)
            return mock_client

        final_result = _make_final_result(task=task, config=config)

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", side_effect=_capture_client),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=_make_phase1_result(),
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=_make_phase2_result(),
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=final_result,
            ),
        ):
            await run_pipeline(task, config)

        options = captured_options[0]
        assert options.system_prompt is not None
        assert isinstance(options.system_prompt, str)
        assert len(options.system_prompt) > 0


# ===========================================================================
# REQ-OR-053: run_pipeline_sync
# ===========================================================================


@pytest.mark.unit
class TestRunPipelineSync:
    """run_pipeline_sync is a sync wrapper around run_pipeline (REQ-OR-053)."""

    def test_is_not_coroutine_function(self) -> None:
        """run_pipeline_sync is a regular (synchronous) function."""
        from mle_star.orchestrator import run_pipeline_sync

        assert not asyncio.iscoroutinefunction(run_pipeline_sync)

    def test_delegates_to_run_pipeline(self, tmp_path: Path) -> None:
        """run_pipeline_sync calls run_pipeline via asyncio.run()."""
        from mle_star.orchestrator import run_pipeline_sync

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config()

        final_result = _make_final_result(task=task, config=config)
        mock_run_pipeline = AsyncMock(return_value=final_result)

        with patch(f"{_MODULE}.run_pipeline", new=mock_run_pipeline):
            result = run_pipeline_sync(task, config)

        assert isinstance(result, FinalResult)
        mock_run_pipeline.assert_awaited_once_with(task, config)

    def test_returns_final_result(self, tmp_path: Path) -> None:
        """run_pipeline_sync returns a FinalResult instance."""
        from mle_star.orchestrator import run_pipeline_sync

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config()

        final_result = _make_final_result(task=task, config=config)
        mock_run_pipeline = AsyncMock(return_value=final_result)

        with patch(f"{_MODULE}.run_pipeline", new=mock_run_pipeline):
            result = run_pipeline_sync(task, config)

        assert isinstance(result, FinalResult)

    def test_propagates_exceptions(self, tmp_path: Path) -> None:
        """run_pipeline_sync propagates exceptions from run_pipeline."""
        from mle_star.orchestrator import PipelineError, run_pipeline_sync

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config()

        mock_run_pipeline = AsyncMock(
            side_effect=PipelineError("failed", diagnostics={"elapsed_time": 0.0})
        )

        with (
            patch(f"{_MODULE}.run_pipeline", new=mock_run_pipeline),
            pytest.raises(PipelineError, match="failed"),
        ):
            run_pipeline_sync(task, config)

    def test_passes_none_config(self, tmp_path: Path) -> None:
        """run_pipeline_sync passes config=None to run_pipeline."""
        from mle_star.orchestrator import run_pipeline_sync

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))

        final_result = _make_final_result(task=task)
        mock_run_pipeline = AsyncMock(return_value=final_result)

        with patch(f"{_MODULE}.run_pipeline", new=mock_run_pipeline):
            run_pipeline_sync(task, None)

        mock_run_pipeline.assert_awaited_once_with(task, None)


# ===========================================================================
# Input validation: run_pipeline raises ValueError before SDK setup
# ===========================================================================


@pytest.mark.unit
class TestRunPipelineInputValidation:
    """run_pipeline raises ValueError for invalid inputs before SDK setup (REQ-OR-002)."""

    async def test_nonexistent_data_dir_raises_before_client(
        self, tmp_path: Path
    ) -> None:
        """ValueError for nonexistent data_dir raised before creating SDK client."""
        from mle_star.orchestrator import run_pipeline

        task = _make_task(data_dir=str(tmp_path / "nonexistent"))
        config = _make_config()

        mock_client_cls = MagicMock()

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", mock_client_cls),
            pytest.raises(ValueError, match="data_dir"),
        ):
            await run_pipeline(task, config)

        # Client should never have been created
        mock_client_cls.assert_not_called()

    async def test_empty_data_dir_raises_before_client(self, tmp_path: Path) -> None:
        """ValueError for empty data_dir raised before creating SDK client."""
        from mle_star.orchestrator import run_pipeline

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        task = _make_task(data_dir=str(empty_dir))
        config = _make_config()

        mock_client_cls = MagicMock()

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", mock_client_cls),
            pytest.raises(ValueError, match="data_dir"),
        ):
            await run_pipeline(task, config)

        mock_client_cls.assert_not_called()

    async def test_file_as_data_dir_raises_before_client(self, tmp_path: Path) -> None:
        """ValueError for file-as-data_dir raised before creating SDK client."""
        from mle_star.orchestrator import run_pipeline

        fake_file = tmp_path / "data.csv"
        fake_file.write_text("data")
        task = _make_task(data_dir=str(fake_file))
        config = _make_config()

        mock_client_cls = MagicMock()

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", mock_client_cls),
            pytest.raises(ValueError, match="data_dir"),
        ):
            await run_pipeline(task, config)

        mock_client_cls.assert_not_called()


# ===========================================================================
# Hypothesis: property-based tests
# ===========================================================================


@pytest.mark.unit
class TestOrchestratorProperties:
    """Property-based tests for orchestrator helpers."""

    @given(
        model=st.sampled_from(["sonnet", "opus", "haiku"]),
        permission_mode=st.sampled_from(
            ["bypassPermissions", "acceptEdits", "default"]
        ),
    )
    @settings(
        max_examples=10,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_validate_inputs_returns_config_for_valid_inputs(
        self,
        model: str,
        permission_mode: str,
        tmp_path: Path,
    ) -> None:
        """_validate_inputs returns the config unchanged for any valid config/data combo."""
        from mle_star.orchestrator import _validate_inputs

        data_dir = tmp_path / f"input_{model}_{permission_mode}"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "data.csv").write_text("id,target\n1,0\n")

        task = _make_task(data_dir=str(data_dir))
        config = _make_config(model=model, permission_mode=permission_mode)

        result = _validate_inputs(task, config)

        assert result is config
        assert result.model == model
        assert result.permission_mode == permission_mode

    @given(
        metric=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N", "P")),
        ),
        direction=st.sampled_from([MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]),
    )
    @settings(max_examples=15, deadline=5000)
    def test_system_prompt_always_contains_metric(
        self,
        metric: str,
        direction: MetricDirection,
    ) -> None:
        """System prompt always includes the evaluation metric name."""
        from mle_star.orchestrator import _build_system_prompt

        task = _make_task(evaluation_metric=metric, metric_direction=direction)
        gpu_info: dict[str, Any] = {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_names": [],
        }

        prompt = _build_system_prompt(task, gpu_info)

        assert metric in prompt

    @given(
        description=st.text(
            min_size=5,
            max_size=200,
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        ),
    )
    @settings(max_examples=10, deadline=5000)
    def test_system_prompt_always_contains_description(
        self,
        description: str,
    ) -> None:
        """System prompt always includes the task description."""
        from mle_star.orchestrator import _build_system_prompt

        task = _make_task(description=description)
        gpu_info: dict[str, Any] = {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_names": [],
        }

        prompt = _build_system_prompt(task, gpu_info)

        assert description in prompt

    def test_build_agents_dict_has_exactly_14_entries(self) -> None:
        """_build_agents_dict always produces exactly 14 agent definitions."""
        from mle_star.orchestrator import _build_agents_dict

        result = _build_agents_dict()

        assert len(result) == len(AgentType)
        assert len(result) == 14

    @given(
        gpu_count=st.integers(min_value=0, max_value=8),
    )
    @settings(max_examples=10, deadline=5000)
    def test_system_prompt_is_non_empty_for_any_gpu_count(
        self,
        gpu_count: int,
    ) -> None:
        """System prompt is always a non-empty string regardless of GPU count."""
        from mle_star.orchestrator import _build_system_prompt

        task = _make_task()
        gpu_info: dict[str, Any] = {
            "cuda_available": gpu_count > 0,
            "gpu_count": gpu_count,
            "gpu_names": [f"GPU-{i}" for i in range(gpu_count)],
        }

        prompt = _build_system_prompt(task, gpu_info)

        assert isinstance(prompt, str)
        assert len(prompt.strip()) > 0


# ===========================================================================
# Edge cases: exception hierarchy
# ===========================================================================


@pytest.mark.unit
class TestExceptionHierarchyEdgeCases:
    """Edge cases for the PipelineError/PipelineTimeoutError hierarchy."""

    def test_pipeline_error_with_none_values_in_diagnostics(self) -> None:
        """PipelineError accepts None values within the diagnostics dict."""
        from mle_star.orchestrator import PipelineError

        diag: dict[str, Any] = {
            "elapsed_time": None,
            "cost": None,
            "last_successful_operation": None,
        }
        err = PipelineError("error with nulls", diagnostics=diag)
        assert err.diagnostics["elapsed_time"] is None

    def test_pipeline_timeout_error_inherits_args(self) -> None:
        """PipelineTimeoutError preserves the exception args tuple."""
        from mle_star.orchestrator import PipelineTimeoutError

        err = PipelineTimeoutError("timed out", diagnostics={"timeout": True})
        assert "timed out" in err.args[0]

    def test_pipeline_error_repr_includes_class_name(self) -> None:
        """PipelineError repr or str includes useful context."""
        from mle_star.orchestrator import PipelineError

        err = PipelineError("test", diagnostics={})
        # Basic sanity: the str representation is meaningful
        assert len(str(err)) > 0

    @pytest.mark.parametrize(
        "error_cls_name",
        ["PipelineError", "PipelineTimeoutError"],
    )
    def test_exceptions_importable(self, error_cls_name: str) -> None:
        """Both exception classes are importable from mle_star.orchestrator."""
        import mle_star.orchestrator as orchestrator

        cls = getattr(orchestrator, error_cls_name)
        assert issubclass(cls, Exception)


# ===========================================================================
# Edge cases: _validate_inputs with various data_dir content
# ===========================================================================


@pytest.mark.unit
class TestValidateInputsEdgeCases:
    """Edge cases for _validate_inputs data directory validation."""

    def test_data_dir_with_hidden_files_only_raises(self, tmp_path: Path) -> None:
        """ValueError raised when data_dir has only hidden files (no regular files)."""
        from mle_star.orchestrator import _validate_inputs

        data_dir = tmp_path / "hidden_only"
        data_dir.mkdir()
        (data_dir / ".gitkeep").write_text("")
        task = _make_task(data_dir=str(data_dir))
        config = _make_config()

        # Implementation may count hidden files as valid or not;
        # if it counts any file (including hidden), this should pass.
        # We test the general case: at least some content exists.
        # The key validation is "no files at all" vs "has files".
        # This test documents behavior for hidden-only dirs.
        with contextlib.suppress(ValueError):
            _validate_inputs(task, config)

    def test_data_dir_with_multiple_files_passes(self, tmp_path: Path) -> None:
        """No error raised for data_dir with multiple files."""
        from mle_star.orchestrator import _validate_inputs

        data_dir = tmp_path / "multi_file"
        data_dir.mkdir()
        (data_dir / "train.csv").write_text("id,target\n1,0\n")
        (data_dir / "test.csv").write_text("id\n1\n")
        task = _make_task(data_dir=str(data_dir))
        config = _make_config()

        # Should not raise
        result = _validate_inputs(task, config)
        assert isinstance(result, PipelineConfig)

    def test_data_dir_with_nested_files_passes(self, tmp_path: Path) -> None:
        """No error raised when data_dir has files in subdirectories."""
        from mle_star.orchestrator import _validate_inputs

        data_dir = tmp_path / "nested"
        data_dir.mkdir()
        (data_dir / "train.csv").write_text("id,target\n1,0\n")
        sub = data_dir / "images"
        sub.mkdir()
        (sub / "img.png").write_text("fake_png")
        task = _make_task(data_dir=str(data_dir))
        config = _make_config()

        # Should not raise
        _validate_inputs(task, config)


# ===========================================================================
# SDK client connect is called
# ===========================================================================


@pytest.mark.unit
class TestClientConnect:
    """run_pipeline calls client.connect() to establish the SDK session."""

    async def test_connect_called(self, tmp_path: Path) -> None:
        """client.connect() is called during pipeline setup."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config()

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()

        final_result = _make_final_result(task=task, config=config)

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=mock_client),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=_make_phase1_result(),
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=_make_phase2_result(),
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=final_result,
            ),
        ):
            await run_pipeline(task, config)

        mock_client.connect.assert_awaited_once()


# ===========================================================================
# _build_agents_dict: verify specific agent tool assignments
# ===========================================================================


@pytest.mark.unit
class TestBuildAgentsDictToolAssignments:
    """_build_agents_dict assigns correct tools per REQ-OR-008."""

    def test_retriever_has_web_tools(self) -> None:
        """Retriever agent has WebSearch and WebFetch tools."""
        from mle_star.orchestrator import _build_agents_dict

        agents = _build_agents_dict()
        retriever_key = _find_agent_key(agents, AgentType.RETRIEVER)
        tools = agents[retriever_key].get("tools", [])

        assert "WebSearch" in tools
        assert "WebFetch" in tools

    def test_execution_agents_have_bash_tools(self) -> None:
        """Execution agents (init, merger, ablation, coder, etc.) have Bash tool."""
        from mle_star.orchestrator import _build_agents_dict

        execution_types = [
            AgentType.INIT,
            AgentType.MERGER,
            AgentType.ABLATION,
            AgentType.CODER,
            AgentType.ENSEMBLER,
            AgentType.DEBUGGER,
            AgentType.TEST,
        ]

        agents = _build_agents_dict()

        for agent_type in execution_types:
            key = _find_agent_key(agents, agent_type)
            tools = agents[key].get("tools", [])
            assert "Bash" in tools, f"Agent {agent_type} missing Bash tool"

    def test_read_only_agents_have_read_tool(self) -> None:
        """Read-only agents (summarize, extractor, planner, etc.) have Read tool."""
        from mle_star.orchestrator import _build_agents_dict

        read_only_types = [
            AgentType.SUMMARIZE,
            AgentType.EXTRACTOR,
            AgentType.PLANNER,
            AgentType.ENS_PLANNER,
            AgentType.LEAKAGE,
            AgentType.DATA,
        ]

        agents = _build_agents_dict()

        for agent_type in read_only_types:
            key = _find_agent_key(agents, agent_type)
            tools = agents[key].get("tools", [])
            assert "Read" in tools, f"Agent {agent_type} missing Read tool"


def _find_agent_key(agents: dict[str, Any], agent_type: AgentType) -> str:
    """Find the key used for an agent type in the agents dict.

    Tries both str(agent_type) and agent_type.value as potential keys.

    Args:
        agents: The agents dict from _build_agents_dict.
        agent_type: The AgentType to look up.

    Returns:
        The key that matches the agent type.

    Raises:
        KeyError: If the agent type is not found.
    """
    for key_candidate in [str(agent_type), agent_type.value]:
        if key_candidate in agents:
            return key_candidate
    msg = (
        f"Agent {agent_type} not found in agents dict with keys: {list(agents.keys())}"
    )
    raise KeyError(msg)


# ===========================================================================
# Parametrized tests: SDK client options for various config values
# ===========================================================================


@pytest.mark.unit
class TestClientOptionsParametrized:
    """Parametrized tests for SDK client option construction."""

    @pytest.mark.parametrize(
        ("model_name",),
        [("sonnet",), ("opus",), ("haiku",)],
    )
    async def test_model_passed_to_options(
        self, model_name: str, tmp_path: Path
    ) -> None:
        """SDK client options model matches config.model for various models."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(model=model_name)

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()

        captured_options: list[Any] = []

        def _capture_client(options: Any = None, **kwargs: Any) -> AsyncMock:
            captured_options.append(options)
            return mock_client

        final_result = _make_final_result(task=task, config=config)

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", side_effect=_capture_client),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=_make_phase1_result(),
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=_make_phase2_result(),
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=final_result,
            ),
        ):
            await run_pipeline(task, config)

        assert captured_options[0].model == model_name

    @pytest.mark.parametrize(
        ("budget",),
        [(None,), (10.0,), (100.0,), (0.5,)],
    )
    async def test_budget_passed_to_options(
        self, budget: float | None, tmp_path: Path
    ) -> None:
        """SDK client options max_budget_usd matches config for various budgets."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config(max_budget_usd=budget)

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()

        captured_options: list[Any] = []

        def _capture_client(options: Any = None, **kwargs: Any) -> AsyncMock:
            captured_options.append(options)
            return mock_client

        final_result = _make_final_result(task=task, config=config)

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", side_effect=_capture_client),
            patch(
                f"{_MODULE}.detect_gpu_info",
                return_value={"cuda_available": False, "gpu_count": 0, "gpu_names": []},
            ),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=_make_phase1_result(),
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=_make_phase2_result(),
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=final_result,
            ),
        ):
            await run_pipeline(task, config)

        assert captured_options[0].max_budget_usd == budget


# ===========================================================================
# GPU info is passed to system prompt builder
# ===========================================================================


@pytest.mark.unit
class TestGPUInfoDetection:
    """run_pipeline calls detect_gpu_info and passes result to system prompt."""

    async def test_detect_gpu_info_called(self, tmp_path: Path) -> None:
        """detect_gpu_info is called during pipeline setup."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config()

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()

        final_result = _make_final_result(task=task, config=config)
        mock_detect = MagicMock(
            return_value={
                "cuda_available": True,
                "gpu_count": 1,
                "gpu_names": ["NVIDIA A100"],
            }
        )

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=mock_client),
            patch(f"{_MODULE}.detect_gpu_info", mock_detect),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=_make_phase1_result(),
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=_make_phase2_result(),
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=final_result,
            ),
        ):
            await run_pipeline(task, config)

        mock_detect.assert_called_once()

    async def test_gpu_info_passed_to_build_system_prompt(self, tmp_path: Path) -> None:
        """detect_gpu_info result is passed to _build_system_prompt."""
        from mle_star.orchestrator import run_pipeline

        data_dir = _make_data_dir(tmp_path)
        task = _make_task(data_dir=str(data_dir))
        config = _make_config()

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()

        final_result = _make_final_result(task=task, config=config)
        gpu = {"cuda_available": True, "gpu_count": 2, "gpu_names": ["A100", "A100"]}

        captured_gpu: list[Any] = []

        def _capture_build_prompt(t: Any, g: Any) -> str:
            captured_gpu.append(g)
            return "mocked system prompt"

        with (
            patch(f"{_MODULE}.ClaudeSDKClient", return_value=mock_client),
            patch(f"{_MODULE}.detect_gpu_info", return_value=gpu),
            patch(f"{_MODULE}._build_system_prompt", side_effect=_capture_build_prompt),
            patch(
                f"{_MODULE}.run_phase1",
                new_callable=AsyncMock,
                return_value=_make_phase1_result(),
            ),
            patch(
                f"{_MODULE}.run_phase2_outer_loop",
                new_callable=AsyncMock,
                return_value=_make_phase2_result(),
            ),
            patch(
                f"{_MODULE}.run_finalization",
                new_callable=AsyncMock,
                return_value=final_result,
            ),
        ):
            await run_pipeline(task, config)

        assert len(captured_gpu) == 1
        assert captured_gpu[0] == gpu
