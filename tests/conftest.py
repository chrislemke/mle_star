"""Shared fixtures for the mle_star test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


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
