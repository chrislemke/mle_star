"""Shared fixtures for the mle_star test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from mle_star.models import (
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
import yaml

# ---------------------------------------------------------------------------
# Factory functions (plain functions, importable from conftest)
# ---------------------------------------------------------------------------


def make_solution(**overrides: Any) -> SolutionScript:
    """Build a valid SolutionScript with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed SolutionScript instance.
    """
    defaults: dict[str, Any] = {
        "content": "import pandas as pd\nprint('hello')\n",
        "phase": SolutionPhase.INIT,
    }
    defaults.update(overrides)
    return SolutionScript(**defaults)


def make_task(**overrides: Any) -> TaskDescription:
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


def make_config(**overrides: Any) -> PipelineConfig:
    """Build a valid PipelineConfig with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed PipelineConfig instance.
    """
    defaults: dict[str, Any] = {}
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def make_eval_result(**overrides: Any) -> EvaluationResult:
    """Build a valid EvaluationResult with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed EvaluationResult instance.
    """
    defaults: dict[str, Any] = {
        "score": 0.85,
        "stdout": "Final Validation Performance: 0.85\n",
        "stderr": "",
        "exit_code": 0,
        "duration_seconds": 10.0,
        "is_error": False,
        "error_traceback": None,
    }
    defaults.update(overrides)
    return EvaluationResult(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_client() -> AsyncMock:
    """Return an AsyncMock simulating a ClaudeSDKClient.

    The mock has a ``send_message`` attribute that is an ``AsyncMock``
    returning ``"response"`` by default.
    """
    client = AsyncMock()
    client.send_message = AsyncMock(return_value="response")
    return client


@pytest.fixture()
def mock_registry() -> MagicMock:
    """Return a stub PromptRegistry with a ``get()`` returning a renderable template.

    The template returned by ``get()`` has a ``render()`` method that
    returns a predictable string (``"rendered prompt"``).
    """
    registry = MagicMock()
    mock_template = MagicMock()
    mock_template.render = MagicMock(return_value="rendered prompt")
    registry.get = MagicMock(return_value=mock_template)
    return registry


@pytest.fixture()
def tmp_working_dir(tmp_path: Path) -> Path:
    """Provide a temporary working directory with standard layout.

    Creates ``input/`` and ``final/`` subdirectories and places a dummy
    CSV file in ``input/``.

    Args:
        tmp_path: Pytest built-in temp directory fixture.

    Returns:
        Path to the temporary working directory.
    """
    work_dir = tmp_path / "workdir"
    work_dir.mkdir()

    input_dir = work_dir / "input"
    input_dir.mkdir()
    (input_dir / "train.csv").write_text("id,feature,target\n1,0.5,0\n2,0.3,1\n")

    final_dir = work_dir / "final"
    final_dir.mkdir()

    return work_dir


@pytest.fixture(scope="session")
def prompts_dir() -> Path:
    """Return the path to the prompts package directory.

    Locates the prompts directory relative to the mle_star package source.
    Uses importlib to find the installed package, falling back to a
    path-based lookup from the project root.
    """
    try:
        import importlib.resources

        prompts_package = importlib.resources.files("mle_star.prompts")
        return Path(str(prompts_package))
    except (ModuleNotFoundError, TypeError):
        # Package not yet created -- fall back to source path
        import mle_star

        pkg_dir = Path(mle_star.__file__).parent
        return pkg_dir / "prompts"


@pytest.fixture(scope="session")
def all_yaml_files(prompts_dir: Path) -> list[Path]:
    """Return all YAML files in the prompts directory.

    Returns an empty list if the prompts directory does not exist yet.
    """
    if not prompts_dir.exists():
        return []
    return sorted(prompts_dir.glob("*.yaml"))


@pytest.fixture(scope="session")
def loaded_templates(all_yaml_files: list[Path]) -> dict[str, Any]:
    """Load and return all YAML templates keyed by filename stem.

    Returns:
        A dict mapping filename stems (e.g. 'retriever') to parsed YAML content.
    """
    templates: dict[str, Any] = {}
    for yaml_file in all_yaml_files:
        with yaml_file.open("r", encoding="utf-8") as f:
            templates[yaml_file.stem] = yaml.safe_load(f)
    return templates
