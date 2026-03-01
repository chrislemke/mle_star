"""Prompt template registry for MLE-STAR agents (REQ-DM-032 through REQ-DM-035).

Loads prompt templates from YAML files in this package directory and provides
keyed access by ``AgentType`` and optional variant string.

Refs:
    SRS 01c — Prompt Template Registry (REQ-DM-032 through REQ-DM-035).
    IMPLEMENTATION_PLAN.md Task 08.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from mle_star.models import AgentType, PromptTemplate

logger = logging.getLogger(__name__)

_PROMPTS_DIR: Path = Path(__file__).parent


class PromptRegistry:
    """Registry of prompt templates keyed by agent type and optional variant (REQ-DM-032).

    Loads all ``.yaml`` files from the ``prompts/`` package directory on
    construction. Templates are stored in a flat dict keyed by
    ``(AgentType, variant)`` where *variant* is ``None`` for single-template
    agents and for the default variant of multi-template agents.

    Attributes:
        _templates: Internal mapping from ``(AgentType, variant)`` to template.
    """

    def __init__(self) -> None:
        """Load all YAML prompt templates from the prompts package directory."""
        self._templates: dict[tuple[AgentType, str | None], PromptTemplate] = {}
        self._agent_types: set[AgentType] = set()
        self._load_all()

    def _load_all(self) -> None:
        """Scan the prompts directory for YAML files and load each one."""
        for yaml_path in sorted(_PROMPTS_DIR.glob("*.yaml")):
            self._load_yaml(yaml_path)

    def _load_yaml(self, path: Path) -> None:
        """Load a single YAML file and register its template(s).

        Supports two formats:
        - Single-template: top-level keys ``agent_type``, ``figure_ref``,
          ``template``, ``variables``.
        - Multi-template: top-level key ``templates`` containing a list of
          template dicts, each with an additional ``variant`` key.

        Args:
            path: Path to the YAML file.
        """
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if "templates" in data:
            self._load_multi_template(data["templates"])
        else:
            self._load_single_template(data)

    def _load_single_template(self, data: dict[str, Any]) -> None:
        """Register a single-template YAML entry.

        Args:
            data: Parsed YAML dict with agent_type, figure_ref, template, variables.
        """
        try:
            agent_type = AgentType(str(data["agent_type"]))
        except ValueError:
            logger.warning("Skipping unknown agent_type %r in prompt template", data["agent_type"])
            return
        variables: list[str] = [str(v) for v in data["variables"]]
        pt = PromptTemplate(
            agent_type=agent_type,
            figure_ref=str(data["figure_ref"]),
            template=str(data["template"]),
            variables=variables,
        )
        self._templates[(agent_type, None)] = pt
        self._agent_types.add(agent_type)

    @staticmethod
    def _parse_variant(entry: dict[str, Any]) -> str | None:
        """Extract the variant string from a YAML entry, normalizing falsy to None.

        Args:
            entry: A single template dict from a multi-template YAML file.

        Returns:
            The variant string, or None if absent/empty.
        """
        raw: str | None = entry.get("variant") or None
        return str(raw) if raw else None

    def _load_multi_template(self, templates: list[dict[str, Any]]) -> None:
        """Register templates from a multi-template YAML entry.

        The first template with ``variant=None`` (or the first in the list if
        none has a null variant) is also stored as the default for that agent.

        Args:
            templates: List of template dicts, each with an optional variant key.
        """
        first_per_agent: dict[AgentType, PromptTemplate] = {}

        for entry in templates:
            try:
                agent_type = AgentType(str(entry["agent_type"]))
            except ValueError:
                logger.warning("Skipping unknown agent_type %r in multi-template YAML", entry["agent_type"])
                continue
            variant = self._parse_variant(entry)
            variables: list[str] = [str(v) for v in entry["variables"]]
            pt = PromptTemplate(
                agent_type=agent_type,
                figure_ref=str(entry["figure_ref"]),
                template=str(entry["template"]),
                variables=variables,
            )
            self._templates[(agent_type, variant)] = pt
            self._agent_types.add(agent_type)
            first_per_agent.setdefault(agent_type, pt)

        self._assign_defaults(first_per_agent)

    def _assign_defaults(
        self, first_per_agent: dict[AgentType, PromptTemplate]
    ) -> None:
        """Ensure every multi-template agent has a default (variant=None) entry.

        If a template with ``variant=None`` was loaded, it already occupies the
        default slot. Otherwise, fall back to the first template for that agent.

        Args:
            first_per_agent: Mapping of agent types to their first loaded template.
        """
        for agent_type, fallback in first_per_agent.items():
            if (agent_type, None) not in self._templates:
                self._templates[(agent_type, None)] = fallback

    def get(self, agent_type: AgentType, variant: str | None = None) -> PromptTemplate:
        """Retrieve a prompt template by agent type and optional variant (REQ-DM-032).

        Args:
            agent_type: The agent type to look up.
            variant: Optional variant name (e.g., ``"detection"``, ``"correction"``).

        Returns:
            The matching ``PromptTemplate``.

        Raises:
            KeyError: If no template is registered for the given agent type/variant.
        """
        key = (agent_type, variant)
        if key not in self._templates:
            if variant is None:
                msg = f"No template registered for agent type: {agent_type!r}"
            else:
                msg = f"No template registered for agent type {agent_type!r} with variant {variant!r}"
            raise KeyError(msg)
        return self._templates[key]

    def __len__(self) -> int:
        """Return the number of unique agent types in the registry (REQ-DM-033)."""
        return len(self._agent_types)


# Module-level singleton for reuse across agent invocations.
_singleton_registry: PromptRegistry | None = None


def get_registry() -> PromptRegistry:
    """Return the singleton PromptRegistry instance.

    Lazily creates the registry on first call. Subsequent calls return
    the cached instance, avoiding repeated YAML file reads.

    Returns:
        The shared ``PromptRegistry`` singleton.
    """
    global _singleton_registry  # noqa: PLW0603
    if _singleton_registry is None:
        _singleton_registry = PromptRegistry()
    return _singleton_registry


def _reset_registry() -> None:
    """Reset the singleton registry (for testing only)."""
    global _singleton_registry  # noqa: PLW0603
    _singleton_registry = None
