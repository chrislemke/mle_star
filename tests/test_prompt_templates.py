"""Tests for MLE-STAR prompt template YAML files.

Validates that all 14 agent types have properly structured YAML prompt
templates in src/mle_star/prompts/, with correct variables, figure references,
and placeholder consistency. This test module follows TDD -- these tests are
written before the YAML files exist, defining the specification that the
prompt templates must satisfy.
"""

from __future__ import annotations

import importlib.resources
import re
from typing import TYPE_CHECKING, Any

import pytest
import yaml

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Constants -- the single source of truth for what must exist
# ---------------------------------------------------------------------------

ALL_AGENT_TYPES: list[str] = [
    "retriever",
    "init",
    "merger",
    "ablation",
    "summarize",
    "extractor",
    "coder",
    "planner",
    "ens_planner",
    "ensembler",
    "debugger",
    "leakage",
    "data",
    "test",
]

# Agents that use the single-template YAML format (no variants)
SINGLE_TEMPLATE_AGENTS: list[str] = [
    "retriever",
    "init",
    "merger",
    "ablation",
    "summarize",
    "extractor",
    "coder",
    "planner",
    "ens_planner",
    "ensembler",
    "debugger",
    "data",
]

# Agents that use the multi-template YAML format (with variants)
MULTI_TEMPLATE_AGENTS: list[str] = [
    "leakage",
    "test",
]

# Expected variables for each agent type / variant
EXPECTED_VARIABLES: dict[tuple[str, str | None], list[str]] = {
    ("retriever", None): ["task_description", "M"],
    ("init", None): ["task_description", "model_name", "example_code"],
    ("merger", None): ["base_code", "reference_code"],
    ("ablation", None): ["solution_script", "previous_ablations"],
    ("summarize", None): ["ablation_code", "raw_result"],
    ("extractor", None): [
        "solution_script",
        "ablation_summary",
        "previous_code_blocks",
    ],
    ("coder", None): ["code_block", "plan"],
    ("planner", None): ["code_block", "plan_history"],
    ("ens_planner", None): ["L", "solutions_text", "plan_history"],
    ("ensembler", None): ["L", "solutions_text", "plan"],
    ("debugger", None): ["code", "bug"],
    ("leakage", "detection"): ["code"],
    ("leakage", "correction"): ["code"],
    ("data", None): ["initial_solution", "task_description"],
    ("test", None): ["task_description", "final_solution"],
    ("test", "subsampling_extract"): ["final_solution"],
    ("test", "subsampling_remove"): ["code_block_with_subsampling"],
    ("test", "contamination_check"): ["reference_discussion", "final_solution"],
}

# Expected figure references
EXPECTED_FIGURES: dict[tuple[str, str | None], str] = {
    ("retriever", None): "Figure 9",
    ("init", None): "Figure 10",
    ("merger", None): "Figure 11",
    ("ablation", None): "Figure 12",
    ("summarize", None): "Figure 13",
    ("extractor", None): "Figure 14",
    ("coder", None): "Figure 15",
    ("planner", None): "Figure 16",
    ("ens_planner", None): "Figure 17",
    ("ensembler", None): "Figure 18",
    ("debugger", None): "Figure 19",
    ("leakage", "detection"): "Figure 20",
    ("leakage", "correction"): "Figure 21",
    ("data", None): "Figure 22",
    ("test", None): "Figure 25",
    ("test", "subsampling_extract"): "Figure 26",
    ("test", "subsampling_remove"): "Figure 27",
    ("test", "contamination_check"): "Figure 28",
}

TOTAL_TEMPLATE_COUNT = 18

LEAKAGE_VARIANTS: list[str] = ["detection", "correction"]
TEST_VARIANTS: list[str] = [
    "subsampling_extract",
    "subsampling_remove",
    "contamination_check",
]

# Required keys for every template entry (single or within multi-template list)
REQUIRED_TEMPLATE_KEYS: set[str] = {"agent_type", "figure_ref", "template", "variables"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_placeholders(template_text: str) -> set[str]:
    """Extract all {variable} placeholders from a template string.

    Ignores escaped braces (``{{...}}``) which represent literal braces
    in Python ``str.format()`` syntax.

    Args:
        template_text: The template string potentially containing {var} placeholders.

    Returns:
        A set of variable names found as placeholders.
    """
    # Remove escaped braces first to avoid false positives
    cleaned = template_text.replace("{{", "").replace("}}", "")
    return set(re.findall(r"\{(\w+)\}", cleaned))


def _get_all_template_entries(
    loaded_templates: dict[str, Any],
) -> list[dict[str, Any]]:
    """Flatten all template entries from all YAML files into a single list.

    Handles both single-template files (with top-level keys) and
    multi-template files (with a 'templates' list).

    Args:
        loaded_templates: Dict mapping filename stems to parsed YAML content.

    Returns:
        A flat list of all template entry dicts.
    """
    entries: list[dict[str, Any]] = []
    for data in loaded_templates.values():
        if "templates" in data:
            entries.extend(data["templates"])
        else:
            entries.append(data)
    return entries


# ---------------------------------------------------------------------------
# Test: Package structure
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPromptsPackageStructure:
    """Tests that the prompts directory is a proper Python package."""

    def test_prompts_directory_exists(self, prompts_dir: Path) -> None:
        """Verify that the prompts directory exists within mle_star package."""
        assert prompts_dir.exists(), (
            f"Prompts directory does not exist at {prompts_dir}"
        )
        assert prompts_dir.is_dir(), (
            f"Expected a directory at {prompts_dir}, found a file"
        )

    def test_prompts_init_py_exists(self, prompts_dir: Path) -> None:
        """Verify that __init__.py exists to make prompts a Python package."""
        init_file = prompts_dir / "__init__.py"
        assert init_file.exists(), (
            f"Missing __init__.py in {prompts_dir}; "
            "the prompts directory must be a proper Python package"
        )

    def test_prompts_package_is_importable(self) -> None:
        """Verify that mle_star.prompts can be imported as a Python package."""
        try:
            importlib.import_module("mle_star.prompts")
        except ImportError as exc:
            pytest.fail(
                f"Could not import mle_star.prompts: {exc}. "
                "Ensure __init__.py exists in src/mle_star/prompts/."
            )


# ---------------------------------------------------------------------------
# Test: YAML file existence
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestYamlFileExistence:
    """Tests that YAML files exist for all 14 agent types."""

    @pytest.mark.parametrize("agent_type", ALL_AGENT_TYPES)
    def test_yaml_file_exists_for_agent(
        self, prompts_dir: Path, agent_type: str
    ) -> None:
        """Verify a YAML file exists for each of the 14 agent types.

        Args:
            prompts_dir: Path to the prompts directory.
            agent_type: The agent type name to check.
        """
        yaml_path = prompts_dir / f"{agent_type}.yaml"
        assert yaml_path.exists(), (
            f"Missing YAML file for agent type '{agent_type}' at {yaml_path}"
        )

    def test_total_yaml_file_count(self, all_yaml_files: list[Path]) -> None:
        """Verify exactly 14 YAML files exist (one per agent type)."""
        assert len(all_yaml_files) == len(ALL_AGENT_TYPES), (
            f"Expected {len(ALL_AGENT_TYPES)} YAML files, "
            f"found {len(all_yaml_files)}: "
            f"{[f.name for f in all_yaml_files]}"
        )


# ---------------------------------------------------------------------------
# Test: YAML validity and loadability
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestYamlValidity:
    """Tests that all YAML files are valid and parseable."""

    @pytest.mark.parametrize("agent_type", ALL_AGENT_TYPES)
    def test_yaml_is_parseable(self, prompts_dir: Path, agent_type: str) -> None:
        """Verify each YAML file can be parsed without errors.

        Args:
            prompts_dir: Path to the prompts directory.
            agent_type: The agent type name to check.
        """
        yaml_path = prompts_dir / f"{agent_type}.yaml"
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data is not None, (
            f"YAML file for '{agent_type}' parsed as None (empty file?)"
        )
        assert isinstance(data, dict), (
            f"YAML file for '{agent_type}' did not parse to a dict, "
            f"got {type(data).__name__}"
        )

    @pytest.mark.parametrize("agent_type", ALL_AGENT_TYPES)
    def test_yaml_uses_utf8_encoding(self, prompts_dir: Path, agent_type: str) -> None:
        """Verify each YAML file is valid UTF-8.

        Args:
            prompts_dir: Path to the prompts directory.
            agent_type: The agent type name to check.
        """
        yaml_path = prompts_dir / f"{agent_type}.yaml"
        try:
            yaml_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            pytest.fail(
                f"YAML file for '{agent_type}' contains invalid UTF-8 characters"
            )


# ---------------------------------------------------------------------------
# Test: Single-template YAML structure
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSingleTemplateStructure:
    """Tests for agents that use the single-template YAML format."""

    @pytest.mark.parametrize("agent_type", SINGLE_TEMPLATE_AGENTS)
    def test_has_required_top_level_keys(
        self, loaded_templates: dict[str, Any], agent_type: str
    ) -> None:
        """Verify single-template files have all required top-level keys.

        Args:
            loaded_templates: All loaded YAML data keyed by filename stem.
            agent_type: The agent type name to check.
        """
        data = loaded_templates[agent_type]
        missing_keys = REQUIRED_TEMPLATE_KEYS - set(data.keys())
        assert not missing_keys, (
            f"Agent '{agent_type}' YAML is missing required keys: {missing_keys}"
        )

    @pytest.mark.parametrize("agent_type", SINGLE_TEMPLATE_AGENTS)
    def test_agent_type_field_matches_filename(
        self, loaded_templates: dict[str, Any], agent_type: str
    ) -> None:
        """Verify agent_type field matches the YAML filename.

        Args:
            loaded_templates: All loaded YAML data keyed by filename stem.
            agent_type: The agent type name to check.
        """
        data = loaded_templates[agent_type]
        assert data["agent_type"] == agent_type, (
            f"agent_type field is '{data['agent_type']}', "
            f"expected '{agent_type}' to match filename"
        )

    @pytest.mark.parametrize("agent_type", SINGLE_TEMPLATE_AGENTS)
    def test_template_is_nonempty_string(
        self, loaded_templates: dict[str, Any], agent_type: str
    ) -> None:
        """Verify the template field is a non-empty string.

        Args:
            loaded_templates: All loaded YAML data keyed by filename stem.
            agent_type: The agent type name to check.
        """
        data = loaded_templates[agent_type]
        template = data["template"]
        assert isinstance(template, str), (
            f"Agent '{agent_type}' template is not a string, "
            f"got {type(template).__name__}"
        )
        assert len(template.strip()) > 0, f"Agent '{agent_type}' has an empty template"

    @pytest.mark.parametrize("agent_type", SINGLE_TEMPLATE_AGENTS)
    def test_variables_is_nonempty_list(
        self, loaded_templates: dict[str, Any], agent_type: str
    ) -> None:
        """Verify the variables field is a non-empty list of strings.

        Args:
            loaded_templates: All loaded YAML data keyed by filename stem.
            agent_type: The agent type name to check.
        """
        data = loaded_templates[agent_type]
        variables = data["variables"]
        assert isinstance(variables, list), (
            f"Agent '{agent_type}' variables is not a list, "
            f"got {type(variables).__name__}"
        )
        assert len(variables) > 0, f"Agent '{agent_type}' has an empty variables list"
        for var in variables:
            assert isinstance(var, str), (
                f"Agent '{agent_type}' has non-string variable: {var!r}"
            )

    @pytest.mark.parametrize("agent_type", SINGLE_TEMPLATE_AGENTS)
    def test_figure_ref_is_string(
        self, loaded_templates: dict[str, Any], agent_type: str
    ) -> None:
        """Verify figure_ref is a non-empty string matching 'Figure N' pattern.

        Args:
            loaded_templates: All loaded YAML data keyed by filename stem.
            agent_type: The agent type name to check.
        """
        data = loaded_templates[agent_type]
        figure_ref = data["figure_ref"]
        assert isinstance(figure_ref, str), (
            f"Agent '{agent_type}' figure_ref is not a string"
        )
        assert re.match(r"^Figure \d+$", figure_ref), (
            f"Agent '{agent_type}' figure_ref '{figure_ref}' "
            "does not match expected 'Figure N' format"
        )

    @pytest.mark.parametrize("agent_type", SINGLE_TEMPLATE_AGENTS)
    def test_does_not_contain_templates_list(
        self, loaded_templates: dict[str, Any], agent_type: str
    ) -> None:
        """Verify single-template files do not have a 'templates' key.

        Args:
            loaded_templates: All loaded YAML data keyed by filename stem.
            agent_type: The agent type name to check.
        """
        data = loaded_templates[agent_type]
        assert "templates" not in data, (
            f"Agent '{agent_type}' should use single-template format, "
            "but contains a 'templates' list"
        )


# ---------------------------------------------------------------------------
# Test: Multi-template (variant) YAML structure
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiTemplateStructure:
    """Tests for agents that use the multi-template YAML format (leakage, test)."""

    @pytest.mark.parametrize("agent_type", MULTI_TEMPLATE_AGENTS)
    def test_has_templates_list(
        self, loaded_templates: dict[str, Any], agent_type: str
    ) -> None:
        """Verify multi-template files have a top-level 'templates' key.

        Args:
            loaded_templates: All loaded YAML data keyed by filename stem.
            agent_type: The agent type name to check.
        """
        data = loaded_templates[agent_type]
        assert "templates" in data, (
            f"Agent '{agent_type}' YAML should have a 'templates' list "
            "for multi-variant agents"
        )
        assert isinstance(data["templates"], list), (
            f"Agent '{agent_type}' 'templates' should be a list, "
            f"got {type(data['templates']).__name__}"
        )

    @pytest.mark.parametrize("agent_type", MULTI_TEMPLATE_AGENTS)
    def test_each_variant_has_required_keys(
        self, loaded_templates: dict[str, Any], agent_type: str
    ) -> None:
        """Verify each variant in a multi-template file has all required keys.

        Args:
            loaded_templates: All loaded YAML data keyed by filename stem.
            agent_type: The agent type name to check.
        """
        variant_keys = REQUIRED_TEMPLATE_KEYS | {"variant"}
        for entry in loaded_templates[agent_type]["templates"]:
            missing = variant_keys - set(entry.keys())
            variant_label = entry.get("variant", "<unknown>")
            assert not missing, (
                f"Agent '{agent_type}' variant '{variant_label}' "
                f"is missing required keys: {missing}"
            )

    @pytest.mark.parametrize("agent_type", MULTI_TEMPLATE_AGENTS)
    def test_each_variant_agent_type_matches_filename(
        self, loaded_templates: dict[str, Any], agent_type: str
    ) -> None:
        """Verify agent_type in each variant matches the YAML filename.

        Args:
            loaded_templates: All loaded YAML data keyed by filename stem.
            agent_type: The agent type name to check.
        """
        for entry in loaded_templates[agent_type]["templates"]:
            assert entry["agent_type"] == agent_type, (
                f"Variant '{entry.get('variant')}' has agent_type "
                f"'{entry['agent_type']}', expected '{agent_type}'"
            )


# ---------------------------------------------------------------------------
# Test: Leakage agent variants
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLeakageVariants:
    """Tests specific to the leakage agent's detection and correction variants."""

    def test_leakage_has_exactly_two_variants(
        self, loaded_templates: dict[str, Any]
    ) -> None:
        """Verify leakage.yaml contains exactly 2 template variants."""
        templates_list = loaded_templates["leakage"]["templates"]
        assert len(templates_list) == 2, (
            f"Expected 2 leakage variants, found {len(templates_list)}"
        )

    @pytest.mark.parametrize("variant", LEAKAGE_VARIANTS)
    def test_leakage_variant_exists(
        self, loaded_templates: dict[str, Any], variant: str
    ) -> None:
        """Verify each expected leakage variant is present.

        Args:
            loaded_templates: All loaded YAML data keyed by filename stem.
            variant: The expected variant name.
        """
        templates_list = loaded_templates["leakage"]["templates"]
        variant_names = [t["variant"] for t in templates_list]
        assert variant in variant_names, (
            f"Leakage variant '{variant}' not found. "
            f"Available variants: {variant_names}"
        )

    def test_leakage_detection_figure_ref(
        self, loaded_templates: dict[str, Any]
    ) -> None:
        """Verify leakage detection variant references Figure 20."""
        templates_list = loaded_templates["leakage"]["templates"]
        detection = next(t for t in templates_list if t["variant"] == "detection")
        assert detection["figure_ref"] == "Figure 20"

    def test_leakage_correction_figure_ref(
        self, loaded_templates: dict[str, Any]
    ) -> None:
        """Verify leakage correction variant references Figure 21."""
        templates_list = loaded_templates["leakage"]["templates"]
        correction = next(t for t in templates_list if t["variant"] == "correction")
        assert correction["figure_ref"] == "Figure 21"


# ---------------------------------------------------------------------------
# Test: Test agent variants
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTestAgentVariants:
    """Tests specific to the test agent's base and additional variants."""

    def test_test_agent_has_four_variants(
        self, loaded_templates: dict[str, Any]
    ) -> None:
        """Verify test.yaml contains exactly 4 template variants.

        The test agent has 1 base variant (no variant key or variant=None)
        plus 3 additional variants: subsampling_extract, subsampling_remove,
        contamination_check.
        """
        templates_list = loaded_templates["test"]["templates"]
        assert len(templates_list) == 4, (
            f"Expected 4 test variants (1 base + 3 additional), "
            f"found {len(templates_list)}"
        )

    @pytest.mark.parametrize("variant", TEST_VARIANTS)
    def test_test_additional_variant_exists(
        self, loaded_templates: dict[str, Any], variant: str
    ) -> None:
        """Verify each expected additional test variant is present.

        Args:
            loaded_templates: All loaded YAML data keyed by filename stem.
            variant: The expected variant name.
        """
        templates_list = loaded_templates["test"]["templates"]
        variant_names = [t.get("variant") for t in templates_list]
        assert variant in variant_names, (
            f"Test variant '{variant}' not found. Available variants: {variant_names}"
        )

    def test_test_base_variant_exists(self, loaded_templates: dict[str, Any]) -> None:
        """Verify the test agent has a base variant (variant=None or 'base')."""
        templates_list = loaded_templates["test"]["templates"]
        base_entries = [t for t in templates_list if t.get("variant") is None]
        assert len(base_entries) == 1, (
            "Expected exactly 1 base test variant (variant=None), "
            f"found {len(base_entries)}"
        )

    def test_test_base_figure_ref(self, loaded_templates: dict[str, Any]) -> None:
        """Verify test base variant references Figure 25."""
        templates_list = loaded_templates["test"]["templates"]
        base = next(t for t in templates_list if t.get("variant") is None)
        assert base["figure_ref"] == "Figure 25"

    def test_test_subsampling_extract_figure_ref(
        self, loaded_templates: dict[str, Any]
    ) -> None:
        """Verify test subsampling_extract variant references Figure 26."""
        templates_list = loaded_templates["test"]["templates"]
        entry = next(t for t in templates_list if t["variant"] == "subsampling_extract")
        assert entry["figure_ref"] == "Figure 26"

    def test_test_subsampling_remove_figure_ref(
        self, loaded_templates: dict[str, Any]
    ) -> None:
        """Verify test subsampling_remove variant references Figure 27."""
        templates_list = loaded_templates["test"]["templates"]
        entry = next(t for t in templates_list if t["variant"] == "subsampling_remove")
        assert entry["figure_ref"] == "Figure 27"

    def test_test_contamination_check_figure_ref(
        self, loaded_templates: dict[str, Any]
    ) -> None:
        """Verify test contamination_check variant references Figure 28."""
        templates_list = loaded_templates["test"]["templates"]
        entry = next(t for t in templates_list if t["variant"] == "contamination_check")
        assert entry["figure_ref"] == "Figure 28"


# ---------------------------------------------------------------------------
# Test: Variable correctness
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVariableCorrectness:
    """Tests that each template declares the correct variables."""

    @pytest.mark.parametrize(
        ("agent_type", "variant", "expected_vars"),
        [(agent, var, vars_) for (agent, var), vars_ in EXPECTED_VARIABLES.items()],
        ids=[f"{agent}-{var}" if var else agent for agent, var in EXPECTED_VARIABLES],
    )
    def test_expected_variables_match(
        self,
        loaded_templates: dict[str, Any],
        agent_type: str,
        variant: str | None,
        expected_vars: list[str],
    ) -> None:
        """Verify each template declares exactly the expected variables.

        Args:
            loaded_templates: All loaded YAML data keyed by filename stem.
            agent_type: The agent type name.
            variant: The variant name, or None for single-template agents.
            expected_vars: The expected list of variable names.
        """
        data = loaded_templates[agent_type]

        if "templates" in data:
            # Multi-template: find the matching variant
            matching = [t for t in data["templates"] if t.get("variant") == variant]
            assert len(matching) == 1, (
                f"Expected exactly 1 entry for {agent_type}/{variant}, "
                f"found {len(matching)}"
            )
            entry = matching[0]
        else:
            entry = data

        actual_vars = sorted(entry["variables"])
        expected_sorted = sorted(expected_vars)
        assert actual_vars == expected_sorted, (
            f"Agent '{agent_type}'"
            f"{f' variant {variant!r}' if variant else ''} "
            f"has variables {actual_vars}, expected {expected_sorted}"
        )


# ---------------------------------------------------------------------------
# Test: Placeholder consistency
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPlaceholderConsistency:
    """Tests that {variable} placeholders in templates match the variables list."""

    @pytest.mark.parametrize(
        ("agent_type", "variant"),
        list(EXPECTED_VARIABLES.keys()),
        ids=[f"{agent}-{var}" if var else agent for agent, var in EXPECTED_VARIABLES],
    )
    def test_placeholders_match_variables(
        self,
        loaded_templates: dict[str, Any],
        agent_type: str,
        variant: str | None,
    ) -> None:
        """Verify all {var} placeholders in the template match the variables list.

        This checks two things:
        1. Every declared variable appears as a placeholder in the template.
        2. Every placeholder in the template is declared in the variables list.

        Args:
            loaded_templates: All loaded YAML data keyed by filename stem.
            agent_type: The agent type name.
            variant: The variant name, or None for single-template agents.
        """
        data = loaded_templates[agent_type]

        if "templates" in data:
            entry = next(t for t in data["templates"] if t.get("variant") == variant)
        else:
            entry = data

        template_text: str = entry["template"]
        declared_vars = set(entry["variables"])
        placeholders = _extract_placeholders(template_text)

        # Every declared variable must appear as a placeholder
        missing_in_template = declared_vars - placeholders
        assert not missing_in_template, (
            f"Agent '{agent_type}'"
            f"{f' variant {variant!r}' if variant else ''}: "
            f"variables declared but not used as placeholders: "
            f"{missing_in_template}"
        )

        # Every placeholder must be declared as a variable
        undeclared_placeholders = placeholders - declared_vars
        assert not undeclared_placeholders, (
            f"Agent '{agent_type}'"
            f"{f' variant {variant!r}' if variant else ''}: "
            f"placeholders found in template but not declared in variables: "
            f"{undeclared_placeholders}"
        )


# ---------------------------------------------------------------------------
# Test: Figure reference correctness
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFigureReferences:
    """Tests that each template references the correct figure."""

    @pytest.mark.parametrize(
        ("agent_type", "variant", "expected_figure"),
        [(agent, var, fig) for (agent, var), fig in EXPECTED_FIGURES.items()],
        ids=[f"{agent}-{var}" if var else agent for agent, var in EXPECTED_FIGURES],
    )
    def test_figure_ref_matches_expected(
        self,
        loaded_templates: dict[str, Any],
        agent_type: str,
        variant: str | None,
        expected_figure: str,
    ) -> None:
        """Verify each template references the correct figure from the paper.

        Args:
            loaded_templates: All loaded YAML data keyed by filename stem.
            agent_type: The agent type name.
            variant: The variant name, or None for single-template agents.
            expected_figure: The expected figure reference string.
        """
        data = loaded_templates[agent_type]

        if "templates" in data:
            entry = next(t for t in data["templates"] if t.get("variant") == variant)
        else:
            entry = data

        assert entry["figure_ref"] == expected_figure, (
            f"Agent '{agent_type}'"
            f"{f' variant {variant!r}' if variant else ''} "
            f"has figure_ref '{entry['figure_ref']}', "
            f"expected '{expected_figure}'"
        )


# ---------------------------------------------------------------------------
# Test: Total template count
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTotalTemplateCount:
    """Tests that the total number of prompt templates is correct."""

    def test_total_templates_is_18(self, loaded_templates: dict[str, Any]) -> None:
        """Verify there are exactly 18 prompt templates across all YAML files.

        Breakdown: 11 single-template agents + 2 leakage variants
        + 4 test variants (1 base + 3 additional) + 1 data = 18.
        """
        entries = _get_all_template_entries(loaded_templates)
        assert len(entries) == TOTAL_TEMPLATE_COUNT, (
            f"Expected {TOTAL_TEMPLATE_COUNT} total templates, found {len(entries)}"
        )


# ---------------------------------------------------------------------------
# Test: Template content quality
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTemplateContentQuality:
    """Tests for template content quality and consistency."""

    def test_no_duplicate_figure_references(
        self, loaded_templates: dict[str, Any]
    ) -> None:
        """Verify no two templates reference the same figure."""
        entries = _get_all_template_entries(loaded_templates)
        figure_refs = [e["figure_ref"] for e in entries]
        duplicates = [fig for fig in figure_refs if figure_refs.count(fig) > 1]
        assert not duplicates, f"Duplicate figure references found: {set(duplicates)}"

    def test_all_templates_are_nonempty_strings(
        self, loaded_templates: dict[str, Any]
    ) -> None:
        """Verify every template field is a non-empty string."""
        entries = _get_all_template_entries(loaded_templates)
        for entry in entries:
            agent = entry["agent_type"]
            variant = entry.get("variant", "base")
            template = entry["template"]
            assert isinstance(template, str), (
                f"{agent}/{variant}: template is not a string"
            )
            assert len(template.strip()) > 0, f"{agent}/{variant}: template is empty"

    def test_all_variables_lists_contain_only_strings(
        self, loaded_templates: dict[str, Any]
    ) -> None:
        """Verify all variables lists contain only string elements."""
        entries = _get_all_template_entries(loaded_templates)
        for entry in entries:
            agent = entry["agent_type"]
            variant = entry.get("variant", "base")
            for var in entry["variables"]:
                assert isinstance(var, str), (
                    f"{agent}/{variant}: variable {var!r} is not a string"
                )

    def test_no_duplicate_variables_within_template(
        self, loaded_templates: dict[str, Any]
    ) -> None:
        """Verify no template has duplicate entries in its variables list."""
        entries = _get_all_template_entries(loaded_templates)
        for entry in entries:
            agent = entry["agent_type"]
            variant = entry.get("variant", "base")
            variables = entry["variables"]
            assert len(variables) == len(set(variables)), (
                f"{agent}/{variant}: duplicate variables found in {variables}"
            )

    def test_variable_names_are_valid_identifiers(
        self, loaded_templates: dict[str, Any]
    ) -> None:
        """Verify all variable names are valid Python identifiers."""
        entries = _get_all_template_entries(loaded_templates)
        for entry in entries:
            agent = entry["agent_type"]
            variant = entry.get("variant", "base")
            for var in entry["variables"]:
                assert var.isidentifier(), (
                    f"{agent}/{variant}: variable '{var}' is not a valid "
                    "Python identifier"
                )

    def test_figure_ref_format_consistency(
        self, loaded_templates: dict[str, Any]
    ) -> None:
        """Verify all figure_ref fields follow 'Figure N' format."""
        entries = _get_all_template_entries(loaded_templates)
        for entry in entries:
            agent = entry["agent_type"]
            variant = entry.get("variant", "base")
            fig = entry["figure_ref"]
            assert re.match(r"^Figure \d+$", fig), (
                f"{agent}/{variant}: figure_ref '{fig}' does not match "
                "'Figure N' format"
            )


# ---------------------------------------------------------------------------
# Test: Agent type coverage (cross-check)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgentTypeCoverage:
    """Cross-checks that all expected agent types appear in loaded templates."""

    def test_all_agent_types_present_in_templates(
        self, loaded_templates: dict[str, Any]
    ) -> None:
        """Verify every expected agent type has at least one template entry."""
        entries = _get_all_template_entries(loaded_templates)
        found_agent_types = {e["agent_type"] for e in entries}
        expected_set = set(ALL_AGENT_TYPES)
        missing = expected_set - found_agent_types
        assert not missing, f"Agent types missing from templates: {missing}"

    def test_no_unexpected_agent_types(self, loaded_templates: dict[str, Any]) -> None:
        """Verify no unexpected agent types appear in the templates."""
        entries = _get_all_template_entries(loaded_templates)
        found_agent_types = {e["agent_type"] for e in entries}
        expected_set = set(ALL_AGENT_TYPES)
        unexpected = found_agent_types - expected_set
        assert not unexpected, (
            f"Unexpected agent types found in templates: {unexpected}"
        )
