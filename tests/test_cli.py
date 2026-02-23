"""Tests for the CLI entry point integration (Task 50).

Validates that ``main()`` in ``src/mle_star/cli.py`` correctly wires
``argparse`` argument parsing to ``run_pipeline_sync()`` from
``mle_star.orchestrator``. Covers happy path, config loading, help/usage,
error handling, and output formatting.

These tests are TDD-first -- they define the expected behavior for
the CLI entry point before ``cli.py`` is implemented.

Refs:
    IMPLEMENTATION_PLAN.md Task 50.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from hypothesis import HealthCheck, given, settings, strategies as st
from mle_star.models import (
    DataModality,
    FinalResult,
    MetricDirection,
    Phase1Result,
    PipelineConfig,
    RetrievedModel,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
    TaskType,
)
from mle_star.orchestrator import PipelineError
import pytest
import yaml

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Module path constant for patching
# ---------------------------------------------------------------------------

_MODULE = "mle_star.cli"

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

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed PipelineConfig instance.
    """
    defaults: dict[str, Any] = {}
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
            RetrievedModel(model_name="xgboost", example_code="import xgb")
        ],
        "candidate_solutions": [_make_solution(phase=SolutionPhase.INIT)],
        "candidate_scores": [0.85],
        "initial_solution": _make_solution(phase=SolutionPhase.INIT),
        "initial_score": 0.85,
    }
    defaults.update(overrides)
    return Phase1Result(**defaults)


def _make_final_result(**overrides: Any) -> FinalResult:
    """Build a valid FinalResult with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed FinalResult instance.
    """
    task = _make_task()
    config = _make_config()
    phase1 = _make_phase1_result()
    solution = _make_solution(phase=SolutionPhase.FINAL)

    defaults: dict[str, Any] = {
        "task": task,
        "config": config,
        "phase1": phase1,
        "phase2_results": [],
        "phase3": None,
        "final_solution": solution,
        "submission_path": "/path/to/submission.csv",
        "total_duration_seconds": 120.5,
    }
    defaults.update(overrides)
    return FinalResult(**defaults)


def _task_yaml_data(**overrides: Any) -> dict[str, Any]:
    """Return a dict suitable for writing to a task YAML file.

    Args:
        **overrides: Field values to override.

    Returns:
        A dict matching TaskDescription field names and valid values.
    """
    defaults: dict[str, Any] = {
        "competition_id": "test-comp",
        "task_type": "classification",
        "data_modality": "tabular",
        "evaluation_metric": "accuracy",
        "metric_direction": "maximize",
        "description": "Predict the target variable from tabular features.",
        "data_dir": "./input",
        "output_dir": "./final",
    }
    defaults.update(overrides)
    return defaults


def _config_yaml_data(**overrides: Any) -> dict[str, Any]:
    """Return a dict suitable for writing to a config YAML file.

    Args:
        **overrides: Field values to override.

    Returns:
        A dict matching PipelineConfig field names and valid values.
    """
    defaults: dict[str, Any] = {
        "num_retrieved_models": 3,
        "outer_loop_steps": 2,
        "inner_loop_steps": 3,
    }
    defaults.update(overrides)
    return defaults


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write a dict to a YAML file.

    Args:
        path: Destination file path.
        data: Data to serialize as YAML.
    """
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


# ===========================================================================
# Test: Happy path -- task YAML loaded and pipeline runs successfully
# ===========================================================================


@pytest.mark.unit
class TestCliHappyPath:
    """main() loads a task YAML, calls run_pipeline_sync, and exits 0."""

    def test_happy_path_runs_pipeline_and_exits_zero(self, tmp_path: Path) -> None:
        """Given a valid task YAML, main() calls run_pipeline_sync and exits 0."""
        from mle_star.cli import main

        # Arrange -- write a valid task YAML file
        task_file = tmp_path / "task.yaml"
        _write_yaml(task_file, _task_yaml_data())
        mock_result = _make_final_result()

        with (
            patch(f"{_MODULE}.run_pipeline_sync", return_value=mock_result) as mock_run,
            patch("sys.argv", ["mle_star", "--task", str(task_file)]),
        ):
            # Act
            exit_code = main()

        # Assert -- run_pipeline_sync was called with a TaskDescription, no config
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        task_arg = call_args[0][0] if call_args[0] else call_args[1]["task"]
        assert isinstance(task_arg, TaskDescription)
        assert task_arg.competition_id == "test-comp"
        assert task_arg.task_type == TaskType.CLASSIFICATION
        # Config should be None when --config is not provided
        if len(call_args[0]) > 1:
            config_arg = call_args[0][1]
        else:
            config_arg = call_args[1].get("config")
        assert config_arg is None
        assert exit_code == 0

    def test_happy_path_prints_submission_path_to_stdout(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """On success, main() prints the submission path to stdout."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "task.yaml"
        _write_yaml(task_file, _task_yaml_data())
        mock_result = _make_final_result(submission_path="/out/submission.csv")

        with (
            patch(f"{_MODULE}.run_pipeline_sync", return_value=mock_result),
            patch("sys.argv", ["mle_star", "--task", str(task_file)]),
        ):
            # Act
            main()

        # Assert -- stdout should contain the submission path
        captured = capsys.readouterr()
        assert "/out/submission.csv" in captured.out

    def test_happy_path_prints_duration_to_stdout(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """On success, main() prints the total duration to stdout."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "task.yaml"
        _write_yaml(task_file, _task_yaml_data())
        mock_result = _make_final_result(total_duration_seconds=42.7)

        with (
            patch(f"{_MODULE}.run_pipeline_sync", return_value=mock_result),
            patch("sys.argv", ["mle_star", "--task", str(task_file)]),
        ):
            # Act
            main()

        # Assert -- stdout should contain the duration
        captured = capsys.readouterr()
        assert "42.7" in captured.out


# ===========================================================================
# Test: Config loading -- --config flag provides a PipelineConfig
# ===========================================================================


@pytest.mark.unit
class TestCliWithConfig:
    """main() loads and passes PipelineConfig when --config is provided."""

    def test_config_flag_loads_and_passes_pipeline_config(self, tmp_path: Path) -> None:
        """--config <file> loads the YAML and passes a PipelineConfig to run_pipeline_sync."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "task.yaml"
        _write_yaml(task_file, _task_yaml_data())

        config_file = tmp_path / "config.yaml"
        config_data = _config_yaml_data(
            num_retrieved_models=3,
            outer_loop_steps=2,
        )
        _write_yaml(config_file, config_data)

        mock_result = _make_final_result()

        with (
            patch(f"{_MODULE}.run_pipeline_sync", return_value=mock_result) as mock_run,
            patch(
                "sys.argv",
                ["mle_star", "--task", str(task_file), "--config", str(config_file)],
            ),
        ):
            # Act
            exit_code = main()

        # Assert -- config was passed as PipelineConfig
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        if len(call_args[0]) > 1:
            config_arg = call_args[0][1]
        else:
            config_arg = call_args[1].get("config")
        assert isinstance(config_arg, PipelineConfig)
        assert config_arg.num_retrieved_models == 3
        assert config_arg.outer_loop_steps == 2
        assert exit_code == 0

    def test_config_with_all_defaults_passes_default_pipeline_config(
        self, tmp_path: Path
    ) -> None:
        """--config with empty YAML passes PipelineConfig with all defaults."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "task.yaml"
        _write_yaml(task_file, _task_yaml_data())

        config_file = tmp_path / "config.yaml"
        _write_yaml(config_file, {})

        mock_result = _make_final_result()

        with (
            patch(f"{_MODULE}.run_pipeline_sync", return_value=mock_result) as mock_run,
            patch(
                "sys.argv",
                ["mle_star", "--task", str(task_file), "--config", str(config_file)],
            ),
        ):
            # Act
            main()

        # Assert
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        if len(call_args[0]) > 1:
            config_arg = call_args[0][1]
        else:
            config_arg = call_args[1].get("config")
        assert isinstance(config_arg, PipelineConfig)
        # All defaults preserved
        assert config_arg.num_retrieved_models == 4
        assert config_arg.time_limit_seconds == 86400


# ===========================================================================
# Test: --help flag prints usage and exits cleanly
# ===========================================================================


@pytest.mark.unit
class TestCliHelp:
    """main() handles --help flag by printing usage and exiting 0."""

    def test_help_flag_exits_with_code_zero(self) -> None:
        """--help triggers SystemExit(0) from argparse."""
        from mle_star.cli import main

        with (
            patch("sys.argv", ["mle_star", "--help"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 0

    def test_help_flag_prints_usage_to_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--help prints usage information containing '--task'."""
        from mle_star.cli import main

        with (
            patch("sys.argv", ["mle_star", "--help"]),
            pytest.raises(SystemExit),
        ):
            main()

        captured = capsys.readouterr()
        assert "--task" in captured.out
        assert "--config" in captured.out


# ===========================================================================
# Test: Missing --task argument
# ===========================================================================


@pytest.mark.unit
class TestCliMissingTask:
    """main() exits with code 2 when --task is not provided."""

    def test_missing_task_exits_with_code_two(self) -> None:
        """Omitting --task triggers argparse error (SystemExit(2))."""
        from mle_star.cli import main

        with (
            patch("sys.argv", ["mle_star"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 2

    def test_missing_task_prints_error_to_stderr(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Omitting --task prints an error message to stderr."""
        from mle_star.cli import main

        with (
            patch("sys.argv", ["mle_star"]),
            pytest.raises(SystemExit),
        ):
            main()

        captured = capsys.readouterr()
        assert "error" in captured.err.lower() or "--task" in captured.err


# ===========================================================================
# Test: PipelineError causes non-zero exit and stderr output
# ===========================================================================


@pytest.mark.unit
class TestCliPipelineError:
    """main() handles PipelineError with non-zero exit and stderr diagnostics."""

    def test_pipeline_error_exits_with_code_one(self, tmp_path: Path) -> None:
        """When run_pipeline_sync raises PipelineError, main() returns 1."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "task.yaml"
        _write_yaml(task_file, _task_yaml_data())

        error = PipelineError(
            "Phase 1 failed",
            diagnostics={"elapsed_time": 42.0, "phase": "phase1"},
        )

        with (
            patch(f"{_MODULE}.run_pipeline_sync", side_effect=error),
            patch("sys.argv", ["mle_star", "--task", str(task_file)]),
        ):
            # Act
            exit_code = main()

        # Assert
        assert exit_code == 1

    def test_pipeline_error_prints_message_to_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """PipelineError message is printed to stderr."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "task.yaml"
        _write_yaml(task_file, _task_yaml_data())

        error = PipelineError(
            "Phase 1 failed",
            diagnostics={"elapsed_time": 42.0},
        )

        with (
            patch(f"{_MODULE}.run_pipeline_sync", side_effect=error),
            patch("sys.argv", ["mle_star", "--task", str(task_file)]),
        ):
            main()

        # Assert
        captured = capsys.readouterr()
        assert "Phase 1 failed" in captured.err

    def test_pipeline_error_prints_diagnostics_to_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """PipelineError diagnostics are included in stderr output."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "task.yaml"
        _write_yaml(task_file, _task_yaml_data())

        error = PipelineError(
            "Pipeline crashed",
            diagnostics={"elapsed_time": 99.9, "phase": "phase2"},
        )

        with (
            patch(f"{_MODULE}.run_pipeline_sync", side_effect=error),
            patch("sys.argv", ["mle_star", "--task", str(task_file)]),
        ):
            main()

        # Assert
        captured = capsys.readouterr()
        assert "elapsed_time" in captured.err or "99.9" in captured.err


# ===========================================================================
# Test: Generic exception causes non-zero exit
# ===========================================================================


@pytest.mark.unit
class TestCliGenericException:
    """main() handles unexpected exceptions with non-zero exit and stderr."""

    def test_generic_exception_exits_with_code_one(self, tmp_path: Path) -> None:
        """An unexpected exception from run_pipeline_sync yields exit code 1."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "task.yaml"
        _write_yaml(task_file, _task_yaml_data())

        with (
            patch(
                f"{_MODULE}.run_pipeline_sync",
                side_effect=RuntimeError("unexpected crash"),
            ),
            patch("sys.argv", ["mle_star", "--task", str(task_file)]),
        ):
            # Act
            exit_code = main()

        # Assert
        assert exit_code == 1

    def test_generic_exception_prints_to_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An unexpected exception message is printed to stderr."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "task.yaml"
        _write_yaml(task_file, _task_yaml_data())

        with (
            patch(
                f"{_MODULE}.run_pipeline_sync",
                side_effect=RuntimeError("unexpected crash"),
            ),
            patch("sys.argv", ["mle_star", "--task", str(task_file)]),
        ):
            main()

        # Assert
        captured = capsys.readouterr()
        assert "unexpected crash" in captured.err


# ===========================================================================
# Test: Task file not found
# ===========================================================================


@pytest.mark.unit
class TestCliTaskFileNotFound:
    """main() handles non-existent task file path gracefully."""

    def test_nonexistent_task_file_exits_with_code_one(self, tmp_path: Path) -> None:
        """A non-existent --task path yields exit code 1."""
        from mle_star.cli import main

        # Arrange
        missing = tmp_path / "does_not_exist.yaml"

        with patch("sys.argv", ["mle_star", "--task", str(missing)]):
            # Act
            exit_code = main()

        # Assert
        assert exit_code == 1

    def test_nonexistent_task_file_prints_error_to_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A non-existent --task path prints an error message to stderr."""
        from mle_star.cli import main

        # Arrange
        missing = tmp_path / "does_not_exist.yaml"

        with patch("sys.argv", ["mle_star", "--task", str(missing)]):
            main()

        # Assert
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower() or "does_not_exist" in captured.err


# ===========================================================================
# Test: Invalid YAML in task file
# ===========================================================================


@pytest.mark.unit
class TestCliInvalidYaml:
    """main() handles malformed YAML in the task file."""

    def test_invalid_yaml_exits_with_code_one(self, tmp_path: Path) -> None:
        """Malformed YAML in --task file yields exit code 1."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "bad.yaml"
        task_file.write_text("{{not: valid: yaml: [", encoding="utf-8")

        with patch("sys.argv", ["mle_star", "--task", str(task_file)]):
            # Act
            exit_code = main()

        # Assert
        assert exit_code == 1

    def test_invalid_yaml_prints_error_to_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Malformed YAML prints a parse error to stderr."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "bad.yaml"
        task_file.write_text("{{invalid yaml content", encoding="utf-8")

        with patch("sys.argv", ["mle_star", "--task", str(task_file)]):
            main()

        # Assert
        captured = capsys.readouterr()
        assert captured.err.strip() != ""


# ===========================================================================
# Test: Valid YAML but missing required task fields
# ===========================================================================


@pytest.mark.unit
class TestCliInvalidTaskData:
    """main() handles valid YAML with missing required TaskDescription fields."""

    def test_missing_required_fields_exits_with_code_one(self, tmp_path: Path) -> None:
        """Valid YAML missing required TaskDescription fields yields exit code 1."""
        from mle_star.cli import main

        # Arrange -- only competition_id, missing everything else
        task_file = tmp_path / "incomplete.yaml"
        _write_yaml(task_file, {"competition_id": "only-this"})

        with patch("sys.argv", ["mle_star", "--task", str(task_file)]):
            # Act
            exit_code = main()

        # Assert
        assert exit_code == 1

    def test_missing_required_fields_prints_validation_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Missing required fields prints a validation error to stderr."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "incomplete.yaml"
        _write_yaml(task_file, {"competition_id": "only-this"})

        with patch("sys.argv", ["mle_star", "--task", str(task_file)]):
            main()

        # Assert -- stderr should mention the validation problem
        captured = capsys.readouterr()
        assert captured.err.strip() != ""

    def test_invalid_enum_value_exits_with_code_one(self, tmp_path: Path) -> None:
        """An invalid enum value in the task YAML yields exit code 1."""
        from mle_star.cli import main

        # Arrange
        task_data = _task_yaml_data(task_type="not_a_real_type")
        task_file = tmp_path / "bad_enum.yaml"
        _write_yaml(task_file, task_data)

        with patch("sys.argv", ["mle_star", "--task", str(task_file)]):
            # Act
            exit_code = main()

        # Assert
        assert exit_code == 1


# ===========================================================================
# Test: Config file not found
# ===========================================================================


@pytest.mark.unit
class TestCliConfigFileNotFound:
    """main() handles non-existent config file path gracefully."""

    def test_nonexistent_config_file_exits_with_code_one(self, tmp_path: Path) -> None:
        """A non-existent --config path yields exit code 1."""
        from mle_star.cli import main

        # Arrange -- valid task file, missing config file
        task_file = tmp_path / "task.yaml"
        _write_yaml(task_file, _task_yaml_data())
        missing_config = tmp_path / "no_config.yaml"

        with patch(
            "sys.argv",
            ["mle_star", "--task", str(task_file), "--config", str(missing_config)],
        ):
            # Act
            exit_code = main()

        # Assert
        assert exit_code == 1

    def test_nonexistent_config_file_prints_error_to_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A non-existent --config path prints an error to stderr."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "task.yaml"
        _write_yaml(task_file, _task_yaml_data())
        missing_config = tmp_path / "no_config.yaml"

        with patch(
            "sys.argv",
            ["mle_star", "--task", str(task_file), "--config", str(missing_config)],
        ):
            main()

        # Assert
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower() or "no_config" in captured.err


# ===========================================================================
# Test: Invalid config YAML data
# ===========================================================================


@pytest.mark.unit
class TestCliInvalidConfigData:
    """main() handles valid YAML with invalid PipelineConfig field values."""

    def test_invalid_config_value_exits_with_code_one(self, tmp_path: Path) -> None:
        """Config YAML with invalid values (e.g., negative int) yields exit code 1."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "task.yaml"
        _write_yaml(task_file, _task_yaml_data())

        config_file = tmp_path / "bad_config.yaml"
        _write_yaml(config_file, {"num_retrieved_models": -1})

        with patch(
            "sys.argv",
            ["mle_star", "--task", str(task_file), "--config", str(config_file)],
        ):
            # Act
            exit_code = main()

        # Assert
        assert exit_code == 1


# ===========================================================================
# Test: Success output content
# ===========================================================================


@pytest.mark.unit
class TestCliSuccessOutput:
    """main() prints useful result information on successful pipeline run."""

    def test_success_prints_score_when_available(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When final_solution has a score, it appears in stdout."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "task.yaml"
        _write_yaml(task_file, _task_yaml_data())

        solution = _make_solution(score=0.9123)
        mock_result = _make_final_result(final_solution=solution)

        with (
            patch(f"{_MODULE}.run_pipeline_sync", return_value=mock_result),
            patch("sys.argv", ["mle_star", "--task", str(task_file)]),
        ):
            main()

        # Assert
        captured = capsys.readouterr()
        assert "0.9123" in captured.out



# ===========================================================================
# Test: Task YAML with all fields including optional overrides
# ===========================================================================


@pytest.mark.unit
class TestCliTaskFieldMapping:
    """All TaskDescription fields from the YAML are correctly mapped."""

    def test_all_task_fields_are_passed_through(self, tmp_path: Path) -> None:
        """Every field in the task YAML is correctly mapped to TaskDescription."""
        from mle_star.cli import main

        # Arrange
        task_data = _task_yaml_data(
            competition_id="my-comp",
            task_type="regression",
            data_modality="image",
            evaluation_metric="rmse",
            metric_direction="minimize",
            description="Predict house prices.",
            data_dir="/data/input",
            output_dir="/data/output",
        )
        task_file = tmp_path / "task.yaml"
        _write_yaml(task_file, task_data)

        mock_result = _make_final_result()

        with (
            patch(f"{_MODULE}.run_pipeline_sync", return_value=mock_result) as mock_run,
            patch("sys.argv", ["mle_star", "--task", str(task_file)]),
        ):
            main()

        # Assert
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        task_arg = call_args[0][0] if call_args[0] else call_args[1]["task"]
        assert task_arg.competition_id == "my-comp"
        assert task_arg.task_type == TaskType.REGRESSION
        assert task_arg.data_modality == DataModality.IMAGE
        assert task_arg.evaluation_metric == "rmse"
        assert task_arg.metric_direction == MetricDirection.MINIMIZE
        assert task_arg.description == "Predict house prices."
        assert task_arg.data_dir == "/data/input"
        assert task_arg.output_dir == "/data/output"


# ===========================================================================
# Test: YAML file is None / not a dict after parsing
# ===========================================================================


@pytest.mark.unit
class TestCliYamlNotDict:
    """main() handles YAML files that parse to non-dict types."""

    def test_yaml_scalar_exits_with_code_one(self, tmp_path: Path) -> None:
        """A YAML file containing a plain scalar (not a dict) yields exit code 1."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "scalar.yaml"
        task_file.write_text("just a string\n", encoding="utf-8")

        with patch("sys.argv", ["mle_star", "--task", str(task_file)]):
            exit_code = main()

        assert exit_code == 1

    def test_yaml_list_exits_with_code_one(self, tmp_path: Path) -> None:
        """A YAML file containing a list (not a dict) yields exit code 1."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "list.yaml"
        task_file.write_text("- item1\n- item2\n", encoding="utf-8")

        with patch("sys.argv", ["mle_star", "--task", str(task_file)]):
            exit_code = main()

        assert exit_code == 1

    def test_yaml_null_exits_with_code_one(self, tmp_path: Path) -> None:
        """An empty YAML file (parsed as None) yields exit code 1."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "empty.yaml"
        task_file.write_text("", encoding="utf-8")

        with patch("sys.argv", ["mle_star", "--task", str(task_file)]):
            exit_code = main()

        assert exit_code == 1


# ===========================================================================
# Test: Config YAML that is not a dict
# ===========================================================================


@pytest.mark.unit
class TestCliConfigYamlNotDict:
    """main() handles config YAML files that parse to non-dict types."""

    def test_config_yaml_list_exits_with_code_one(self, tmp_path: Path) -> None:
        """A config YAML file containing a list (not a dict) yields exit code 1."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "task.yaml"
        _write_yaml(task_file, _task_yaml_data())

        config_file = tmp_path / "list_config.yaml"
        config_file.write_text("- item1\n- item2\n", encoding="utf-8")

        with patch(
            "sys.argv",
            ["mle_star", "--task", str(task_file), "--config", str(config_file)],
        ):
            exit_code = main()

        assert exit_code == 1


# ===========================================================================
# Hypothesis: Property-based tests
# ===========================================================================


@pytest.mark.unit
class TestCliPropertyBased:
    """Property-based tests for CLI argument parsing and error handling."""

    @given(comp_id=st.text(min_size=1, max_size=50))
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_arbitrary_competition_id_is_passed_through(
        self, comp_id: str, tmp_path: Path
    ) -> None:
        """Any non-empty competition_id string is passed through to the task."""
        from mle_star.cli import main

        # Arrange
        task_data = _task_yaml_data(competition_id=comp_id)
        task_file = tmp_path / "task.yaml"
        _write_yaml(task_file, task_data)

        mock_result = _make_final_result()

        with (
            patch(f"{_MODULE}.run_pipeline_sync", return_value=mock_result) as mock_run,
            patch("sys.argv", ["mle_star", "--task", str(task_file)]),
        ):
            exit_code = main()

        # Assert
        if exit_code == 0:
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            task_arg = call_args[0][0] if call_args[0] else call_args[1]["task"]
            assert task_arg.competition_id == comp_id

    @given(
        models=st.integers(min_value=1, max_value=20),
        steps=st.integers(min_value=1, max_value=20),
    )
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_arbitrary_config_values_are_passed_through(
        self, models: int, steps: int, tmp_path: Path
    ) -> None:
        """Positive config integer values are correctly passed to PipelineConfig."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "task.yaml"
        _write_yaml(task_file, _task_yaml_data())

        config_file = tmp_path / "config.yaml"
        _write_yaml(
            config_file,
            {"num_retrieved_models": models, "outer_loop_steps": steps},
        )

        mock_result = _make_final_result()

        with (
            patch(f"{_MODULE}.run_pipeline_sync", return_value=mock_result) as mock_run,
            patch(
                "sys.argv",
                ["mle_star", "--task", str(task_file), "--config", str(config_file)],
            ),
        ):
            main()

        # Assert
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        if len(call_args[0]) > 1:
            config_arg = call_args[0][1]
        else:
            config_arg = call_args[1].get("config")
        assert config_arg.num_retrieved_models == models
        assert config_arg.outer_loop_steps == steps


# ===========================================================================
# Test: Return type invariant -- main always returns int
# ===========================================================================


@pytest.mark.unit
class TestCliReturnType:
    """main() always returns an integer exit code."""

    def test_success_returns_int_zero(self, tmp_path: Path) -> None:
        """Successful runs return integer 0."""
        from mle_star.cli import main

        # Arrange
        task_file = tmp_path / "task.yaml"
        _write_yaml(task_file, _task_yaml_data())

        mock_result = _make_final_result()

        with (
            patch(f"{_MODULE}.run_pipeline_sync", return_value=mock_result),
            patch("sys.argv", ["mle_star", "--task", str(task_file)]),
        ):
            result = main()

        assert isinstance(result, int)
        assert result == 0

    def test_error_returns_int_one(self, tmp_path: Path) -> None:
        """Errored runs return integer 1."""
        from mle_star.cli import main

        # Arrange -- non-existent file
        missing = tmp_path / "nope.yaml"

        with patch("sys.argv", ["mle_star", "--task", str(missing)]):
            result = main()

        assert isinstance(result, int)
        assert result == 1
