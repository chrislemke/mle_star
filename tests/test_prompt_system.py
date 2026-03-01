"""Tests for MLE-STAR prompt template system (Task 08).

Validates the ``PromptTemplate`` frozen Pydantic model and ``PromptRegistry``
loader defined in ``src/mle_star/models.py`` and ``src/mle_star/prompts/__init__.py``.
These tests are written TDD-first -- the implementation does not yet exist.
They serve as the executable specification for REQ-PT-001 through REQ-PT-008.

Refs:
    SRS 01a (Prompt Template System), IMPLEMENTATION_PLAN.md Task 08.
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings, strategies as st
from mle_star.models import AgentType, PromptTemplate
from mle_star.prompts import PromptRegistry
from pydantic import ValidationError
import pytest

# ---------------------------------------------------------------------------
# Constants -- canonical values from the spec
# ---------------------------------------------------------------------------

AGENT_TYPE_VALUES: list[str] = [
    "baseline",
    "researcher",
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
    "validator",
]

# Single-template agents (one YAML file -> one PromptTemplate)
SINGLE_TEMPLATE_AGENTS: list[AgentType] = [
    AgentType.BASELINE,
    AgentType.RESEARCHER,
    AgentType.RETRIEVER,
    AgentType.INIT,
    AgentType.MERGER,
    AgentType.ABLATION,
    AgentType.SUMMARIZE,
    AgentType.EXTRACTOR,
    AgentType.CODER,
    AgentType.PLANNER,
    AgentType.ENS_PLANNER,
    AgentType.ENSEMBLER,
    AgentType.DEBUGGER,
    AgentType.DATA,
]

# Leakage variants
LEAKAGE_VARIANTS: list[str] = ["detection", "correction", "deep_analysis"]

# Test variants (including the default with variant=None)
TEST_VARIANTS: list[str] = [
    "subsampling_extract",
    "subsampling_remove",
    "contamination_check",
]

# Validator variants
VALIDATOR_VARIANTS: list[str] = ["sanity", "overfitting"]

# Total unique agent types = 17 (14 original + baseline + researcher + validator)
TOTAL_AGENT_TYPES: int = 17

# Total template variants = 14 single + 3 leakage + 4 test + 2 validator = 23
TOTAL_TEMPLATE_VARIANTS: int = 23


# ---------------------------------------------------------------------------
# Helpers -- factory function for building valid PromptTemplate instances
# ---------------------------------------------------------------------------


def _make_prompt_template(**overrides: Any) -> PromptTemplate:
    """Build a valid PromptTemplate with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed PromptTemplate instance.
    """
    defaults: dict[str, Any] = {
        "agent_type": AgentType.RETRIEVER,
        "figure_ref": "Figure 9",
        "template": "List {M} models for {task_description}.",
        "variables": ["M", "task_description"],
    }
    defaults.update(overrides)
    return PromptTemplate(**defaults)


# ===========================================================================
# PromptTemplate -- Model Construction & Fields
# ===========================================================================


@pytest.mark.unit
class TestPromptTemplateConstruction:
    """PromptTemplate is a frozen Pydantic model with required fields."""

    def test_valid_construction(self) -> None:
        """Constructing with all required fields succeeds."""
        pt = _make_prompt_template()
        assert pt.agent_type == AgentType.RETRIEVER
        assert pt.figure_ref == "Figure 9"
        assert pt.template == "List {M} models for {task_description}."
        assert pt.variables == ["M", "task_description"]

    def test_agent_type_field_is_agent_type_enum(self) -> None:
        """The agent_type field is of type AgentType."""
        pt = _make_prompt_template()
        assert isinstance(pt.agent_type, AgentType)

    def test_figure_ref_field_is_string(self) -> None:
        """The figure_ref field is a string."""
        pt = _make_prompt_template()
        assert isinstance(pt.figure_ref, str)

    def test_template_field_is_string(self) -> None:
        """The template field is a string."""
        pt = _make_prompt_template()
        assert isinstance(pt.template, str)

    def test_variables_field_is_list_of_strings(self) -> None:
        """The variables field is a list of strings."""
        pt = _make_prompt_template()
        assert isinstance(pt.variables, list)
        for var in pt.variables:
            assert isinstance(var, str)

    @pytest.mark.parametrize(
        "missing_field",
        ["agent_type", "figure_ref", "template", "variables"],
    )
    def test_missing_required_field_raises(self, missing_field: str) -> None:
        """Omitting any required field raises ValidationError."""
        all_fields: dict[str, Any] = {
            "agent_type": "retriever",
            "figure_ref": "Figure 9",
            "template": "List {M} models.",
            "variables": ["M"],
        }
        del all_fields[missing_field]
        with pytest.raises(ValidationError):
            PromptTemplate(**all_fields)

    def test_invalid_agent_type_raises(self) -> None:
        """An invalid agent_type string raises ValidationError."""
        with pytest.raises(ValidationError):
            _make_prompt_template(agent_type="nonexistent_agent")

    @pytest.mark.parametrize("agent_type", list(AgentType))
    def test_accepts_all_agent_types(self, agent_type: AgentType) -> None:
        """Every AgentType value is accepted in the agent_type field."""
        pt = _make_prompt_template(agent_type=agent_type)
        assert pt.agent_type == agent_type

    def test_empty_variables_list_accepted(self) -> None:
        """An empty variables list is accepted (template with no placeholders)."""
        pt = _make_prompt_template(
            template="No placeholders here.",
            variables=[],
        )
        assert pt.variables == []

    def test_single_variable_accepted(self) -> None:
        """A single-element variables list is accepted."""
        pt = _make_prompt_template(
            template="Hello {name}.",
            variables=["name"],
        )
        assert pt.variables == ["name"]


# ===========================================================================
# PromptTemplate -- Frozen (Immutable)
# ===========================================================================


@pytest.mark.unit
class TestPromptTemplateFrozen:
    """PromptTemplate is frozen (immutable) per ConfigDict(frozen=True)."""

    def test_cannot_mutate_agent_type(self) -> None:
        """Assignment to agent_type raises an error."""
        pt = _make_prompt_template()
        with pytest.raises(ValidationError):
            pt.agent_type = AgentType.CODER  # type: ignore[misc]

    def test_cannot_mutate_figure_ref(self) -> None:
        """Assignment to figure_ref raises an error."""
        pt = _make_prompt_template()
        with pytest.raises(ValidationError):
            pt.figure_ref = "Figure 99"  # type: ignore[misc]

    def test_cannot_mutate_template(self) -> None:
        """Assignment to template raises an error."""
        pt = _make_prompt_template()
        with pytest.raises(ValidationError):
            pt.template = "Changed template."  # type: ignore[misc]

    def test_cannot_mutate_variables(self) -> None:
        """Assignment to variables raises an error."""
        pt = _make_prompt_template()
        with pytest.raises(ValidationError):
            pt.variables = ["new_var"]  # type: ignore[misc]


# ===========================================================================
# PromptTemplate -- Serialization
# ===========================================================================


@pytest.mark.unit
class TestPromptTemplateSerialization:
    """PromptTemplate supports JSON round-trip serialization."""

    def test_round_trip_preserves_all_fields(self) -> None:
        """Serialize and deserialize; all fields preserved."""
        original = _make_prompt_template()
        json_str = original.model_dump_json()
        restored = PromptTemplate.model_validate_json(json_str)

        assert restored.agent_type == original.agent_type
        assert restored.figure_ref == original.figure_ref
        assert restored.template == original.template
        assert restored.variables == original.variables

    def test_round_trip_equality(self) -> None:
        """Serialized then deserialized object equals the original."""
        original = _make_prompt_template()
        restored = PromptTemplate.model_validate_json(original.model_dump_json())
        assert original == restored


# ===========================================================================
# PromptTemplate.render() -- Happy Path
# ===========================================================================


@pytest.mark.unit
class TestPromptTemplateRenderHappyPath:
    """PromptTemplate.render() substitutes placeholders using str.format()."""

    def test_render_basic_substitution(self) -> None:
        """render() replaces {variable} placeholders with provided values."""
        pt = _make_prompt_template(
            template="List {M} models for {task_description}.",
            variables=["M", "task_description"],
        )
        result = pt.render(M=4, task_description="classify images")
        assert result == "List 4 models for classify images."

    def test_render_acceptance_criterion(self) -> None:
        """Exact acceptance criterion from spec.

        render(M=4, task_description="classify images") substitutes correctly.
        """
        pt = _make_prompt_template(
            template="List {M} models for {task_description}.",
            variables=["M", "task_description"],
        )
        result = pt.render(M=4, task_description="classify images")
        assert "4" in result
        assert "classify images" in result

    def test_render_single_variable(self) -> None:
        """render() works with a single variable."""
        pt = _make_prompt_template(
            template="Code:\n{code}",
            variables=["code"],
        )
        result = pt.render(code="print('hello')")
        assert result == "Code:\nprint('hello')"

    def test_render_no_variables(self) -> None:
        """render() with empty variables list returns template unchanged."""
        pt = _make_prompt_template(
            template="Static prompt with no variables.",
            variables=[],
        )
        result = pt.render()
        assert result == "Static prompt with no variables."

    def test_render_multiple_occurrences_of_same_variable(self) -> None:
        """render() replaces all occurrences of the same variable."""
        pt = _make_prompt_template(
            template="{name} is great. {name} wins.",
            variables=["name"],
        )
        result = pt.render(name="Alice")
        assert result == "Alice is great. Alice wins."

    def test_render_with_integer_value(self) -> None:
        """render() accepts integer values and converts them to string."""
        pt = _make_prompt_template(
            template="Use {M} models.",
            variables=["M"],
        )
        result = pt.render(M=10)
        assert result == "Use 10 models."

    def test_render_with_multiline_template(self) -> None:
        """render() works with multiline templates (typical YAML block scalar)."""
        pt = _make_prompt_template(
            template="# Competition\n{task_description}\n\n# Task\nUse {M} models.\n",
            variables=["task_description", "M"],
        )
        result = pt.render(task_description="Predict house prices", M=4)
        assert "Predict house prices" in result
        assert "4" in result
        assert result.startswith("# Competition\n")

    def test_render_preserves_literal_braces(self) -> None:
        """render() preserves {{ and }} as literal braces in the output.

        YAML templates use {{ and }} for literal braces, which Python
        str.format() renders as { and }.
        """
        pt = _make_prompt_template(
            template="Schema: {{'name': str}} with {M} items.",
            variables=["M"],
        )
        result = pt.render(M=5)
        assert result == "Schema: {'name': str} with 5 items."

    def test_render_returns_string_type(self) -> None:
        """render() always returns a str."""
        pt = _make_prompt_template()
        result = pt.render(M=4, task_description="test")
        assert isinstance(result, str)

    def test_render_extra_kwargs_ignored_or_allowed(self) -> None:
        """render() does not raise when extra keyword arguments are provided.

        Python str.format() ignores extra keyword arguments that do not
        correspond to any placeholder in the template.
        """
        pt = _make_prompt_template(
            template="Use {M} models.",
            variables=["M"],
        )
        # str.format() silently ignores extra kwargs
        result = pt.render(M=4, extra_arg="ignored")
        assert result == "Use 4 models."


# ===========================================================================
# PromptTemplate.render() -- Error Cases
# ===========================================================================


@pytest.mark.unit
class TestPromptTemplateRenderErrors:
    """PromptTemplate.render() raises KeyError for missing required variables."""

    def test_render_missing_variable_raises_key_error(self) -> None:
        """Calling render() without a required variable raises KeyError."""
        pt = _make_prompt_template(
            template="List {M} models for {task_description}.",
            variables=["M", "task_description"],
        )
        with pytest.raises(KeyError):
            pt.render(M=4)  # Missing task_description

    def test_render_no_args_raises_key_error(self) -> None:
        """Calling render() with no args on a template with variables raises KeyError."""
        pt = _make_prompt_template(
            template="{code}",
            variables=["code"],
        )
        with pytest.raises(KeyError):
            pt.render()

    def test_render_missing_all_variables_raises_key_error(self) -> None:
        """Calling render() missing all required variables raises KeyError."""
        pt = _make_prompt_template(
            template="{M} {task_description}",
            variables=["M", "task_description"],
        )
        with pytest.raises(KeyError):
            pt.render()

    def test_render_missing_one_of_many_raises_key_error(self) -> None:
        """Missing even one of several required variables raises KeyError."""
        pt = _make_prompt_template(
            template="{a} {b} {c}",
            variables=["a", "b", "c"],
        )
        with pytest.raises(KeyError):
            pt.render(a="x", b="y")  # Missing c


# ===========================================================================
# PromptTemplate.render() -- Property-based (Hypothesis)
# ===========================================================================


@pytest.mark.unit
class TestPromptTemplateRenderPropertyBased:
    """Property-based tests for PromptTemplate.render() using Hypothesis."""

    @given(
        value=st.text(min_size=1, max_size=200),
    )
    @settings(max_examples=50)
    def test_rendered_output_contains_substituted_value(self, value: str) -> None:
        """Property: rendered output always contains the substituted value.

        We use a simple template with one variable to verify that
        the value appears verbatim in the output.
        """
        # Avoid values containing format-like braces which would confuse str.format
        if "{" in value or "}" in value:
            return
        pt = _make_prompt_template(
            template="Result: {var}",
            variables=["var"],
        )
        result = pt.render(var=value)
        assert value in result

    @given(
        m_val=st.integers(min_value=1, max_value=10000),
        desc=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),  # type: ignore[arg-type]
            ),
            min_size=1,
            max_size=100,
        ),
    )
    @settings(max_examples=50)
    def test_render_produces_consistent_output(self, m_val: int, desc: str) -> None:
        """Property: rendering with the same arguments always produces the same output."""
        if "{" in desc or "}" in desc:
            return
        pt = _make_prompt_template(
            template="List {M} models for {task_description}.",
            variables=["M", "task_description"],
        )
        result1 = pt.render(M=m_val, task_description=desc)
        result2 = pt.render(M=m_val, task_description=desc)
        assert result1 == result2

    @given(
        m_val=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=30)
    def test_render_output_length_at_least_as_long_as_value(self, m_val: int) -> None:
        """Property: rendered output length is at least the length of the value string."""
        pt = _make_prompt_template(
            template="{M}",
            variables=["M"],
        )
        result = pt.render(M=m_val)
        assert len(result) >= len(str(m_val))


# ===========================================================================
# PromptRegistry -- Construction & Loading
# ===========================================================================


@pytest.mark.unit
class TestPromptRegistryConstruction:
    """PromptRegistry loads all 14 YAML templates on construction."""

    def test_registry_can_be_instantiated(self) -> None:
        """PromptRegistry() creates an instance without errors."""
        registry = PromptRegistry()
        assert registry is not None

    def test_registry_len_is_14(self) -> None:
        """len(registry) returns 14 -- one per unique AgentType."""
        registry = PromptRegistry()
        assert len(registry) == TOTAL_AGENT_TYPES

    def test_registry_len_returns_int(self) -> None:
        """len(registry) returns an int type."""
        registry = PromptRegistry()
        result = len(registry)
        assert isinstance(result, int)


# ===========================================================================
# PromptRegistry.get() -- Single-template Agents
# ===========================================================================


@pytest.mark.unit
class TestPromptRegistryGetSingleTemplate:
    """PromptRegistry.get() retrieves templates for single-template agents."""

    @pytest.mark.parametrize("agent_type", SINGLE_TEMPLATE_AGENTS)
    def test_get_returns_prompt_template(self, agent_type: AgentType) -> None:
        """get() returns a PromptTemplate for each single-template agent."""
        registry = PromptRegistry()
        pt = registry.get(agent_type)
        assert isinstance(pt, PromptTemplate)

    @pytest.mark.parametrize("agent_type", SINGLE_TEMPLATE_AGENTS)
    def test_get_returns_correct_agent_type(self, agent_type: AgentType) -> None:
        """get() returns a template whose agent_type matches the requested one."""
        registry = PromptRegistry()
        pt = registry.get(agent_type)
        assert pt.agent_type == agent_type

    def test_get_retriever_acceptance_criterion(self) -> None:
        """Exact acceptance criterion: get(AgentType.retriever) returns retriever template."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.RETRIEVER)
        assert pt.agent_type == AgentType.RETRIEVER
        assert isinstance(pt.template, str)
        assert len(pt.template) > 0

    @pytest.mark.parametrize("agent_type", SINGLE_TEMPLATE_AGENTS)
    def test_get_template_has_nonempty_template_string(
        self, agent_type: AgentType
    ) -> None:
        """Each single-template agent has a non-empty template string."""
        registry = PromptRegistry()
        pt = registry.get(agent_type)
        assert len(pt.template) > 0

    @pytest.mark.parametrize("agent_type", SINGLE_TEMPLATE_AGENTS)
    def test_get_template_has_figure_ref(self, agent_type: AgentType) -> None:
        """Each single-template agent has a non-empty figure_ref."""
        registry = PromptRegistry()
        pt = registry.get(agent_type)
        assert pt.figure_ref.startswith("Figure ") or pt.figure_ref.startswith("N/A")

    @pytest.mark.parametrize("agent_type", SINGLE_TEMPLATE_AGENTS)
    def test_get_template_has_variables_list(self, agent_type: AgentType) -> None:
        """Each single-template agent has a variables list (may be empty)."""
        registry = PromptRegistry()
        pt = registry.get(agent_type)
        assert isinstance(pt.variables, list)


# ===========================================================================
# PromptRegistry.get() -- Retriever Template Content
# ===========================================================================


@pytest.mark.unit
class TestPromptRegistryRetrieverContent:
    """PromptRegistry returns correct retriever template content from YAML."""

    def test_retriever_figure_ref(self) -> None:
        """Retriever template has figure_ref 'Figure 9'."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.RETRIEVER)
        assert pt.figure_ref == "Figure 9"

    def test_retriever_variables(self) -> None:
        """Retriever template has variables ['task_description', 'target_column', 'M', 'research_context', 'notes_context']."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.RETRIEVER)
        assert sorted(pt.variables) == sorted(
            ["task_description", "target_column", "M", "research_context", "notes_context"]
        )

    def test_retriever_template_contains_placeholders(self) -> None:
        """Retriever template text contains {task_description} and {M}."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.RETRIEVER)
        assert "{task_description}" in pt.template
        assert "{M}" in pt.template

    def test_retriever_render_works(self) -> None:
        """Rendering the retriever template produces expected substitutions."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.RETRIEVER)
        rendered = pt.render(
            task_description="classify images", target_column="label", M=4,
            research_context="", notes_context="",
        )
        assert "classify images" in rendered
        assert "4" in rendered


# ===========================================================================
# PromptRegistry.get() -- Leakage Variants
# ===========================================================================


@pytest.mark.unit
class TestPromptRegistryLeakageVariants:
    """PromptRegistry.get() supports leakage variant templates."""

    def test_get_leakage_detection(self) -> None:
        """get(AgentType.leakage, variant='detection') returns detection template."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.LEAKAGE, variant="detection")
        assert pt.agent_type == AgentType.LEAKAGE
        assert isinstance(pt, PromptTemplate)

    def test_get_leakage_correction(self) -> None:
        """get(AgentType.leakage, variant='correction') returns correction template."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.LEAKAGE, variant="correction")
        assert pt.agent_type == AgentType.LEAKAGE
        assert isinstance(pt, PromptTemplate)

    def test_leakage_detection_figure_ref(self) -> None:
        """Leakage detection template has figure_ref 'Figure 20'."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.LEAKAGE, variant="detection")
        assert pt.figure_ref == "Figure 20"

    def test_leakage_correction_figure_ref(self) -> None:
        """Leakage correction template has figure_ref 'Figure 21'."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.LEAKAGE, variant="correction")
        assert pt.figure_ref == "Figure 21"

    def test_leakage_detection_variables(self) -> None:
        """Leakage detection template has variables ['code']."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.LEAKAGE, variant="detection")
        assert pt.variables == ["code"]

    def test_leakage_correction_variables(self) -> None:
        """Leakage correction template has variables ['code']."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.LEAKAGE, variant="correction")
        assert pt.variables == ["code"]

    @pytest.mark.parametrize("variant", LEAKAGE_VARIANTS)
    def test_leakage_variant_render_works(self, variant: str) -> None:
        """Rendering any leakage variant template produces expected substitutions."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.LEAKAGE, variant=variant)
        # deep_analysis requires extra variables beyond just 'code'
        kwargs: dict[str, str] = {"code": "import pandas as pd"}
        for var in pt.variables:
            if var not in kwargs:
                kwargs[var] = f"<{var}>"
        rendered = pt.render(**kwargs)
        assert "import pandas as pd" in rendered

    def test_leakage_detection_and_correction_are_different(self) -> None:
        """Detection and correction templates have different template text."""
        registry = PromptRegistry()
        detection = registry.get(AgentType.LEAKAGE, variant="detection")
        correction = registry.get(AgentType.LEAKAGE, variant="correction")
        assert detection.template != correction.template

    def test_leakage_detection_and_correction_have_different_figure_refs(self) -> None:
        """Detection and correction templates have different figure_ref values."""
        registry = PromptRegistry()
        detection = registry.get(AgentType.LEAKAGE, variant="detection")
        correction = registry.get(AgentType.LEAKAGE, variant="correction")
        assert detection.figure_ref != correction.figure_ref


# ===========================================================================
# PromptRegistry.get() -- Test Variants
# ===========================================================================


@pytest.mark.unit
class TestPromptRegistryTestVariants:
    """PromptRegistry.get() supports test variant templates."""

    def test_get_test_default_variant(self) -> None:
        """get(AgentType.test) returns the default test template (variant=None)."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.TEST)
        assert pt.agent_type == AgentType.TEST
        assert isinstance(pt, PromptTemplate)

    def test_get_test_subsampling_extract(self) -> None:
        """get(AgentType.test, variant='subsampling_extract') returns correct template."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.TEST, variant="subsampling_extract")
        assert pt.agent_type == AgentType.TEST
        assert isinstance(pt, PromptTemplate)

    def test_get_test_subsampling_remove(self) -> None:
        """get(AgentType.test, variant='subsampling_remove') returns correct template."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.TEST, variant="subsampling_remove")
        assert pt.agent_type == AgentType.TEST
        assert isinstance(pt, PromptTemplate)

    def test_get_test_contamination_check(self) -> None:
        """get(AgentType.test, variant='contamination_check') returns correct template."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.TEST, variant="contamination_check")
        assert pt.agent_type == AgentType.TEST
        assert isinstance(pt, PromptTemplate)

    def test_test_default_figure_ref(self) -> None:
        """Test default template has figure_ref 'Figure 25'."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.TEST)
        assert pt.figure_ref == "Figure 25"

    def test_test_subsampling_extract_figure_ref(self) -> None:
        """Test subsampling_extract template has figure_ref 'Figure 26'."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.TEST, variant="subsampling_extract")
        assert pt.figure_ref == "Figure 26"

    def test_test_subsampling_remove_figure_ref(self) -> None:
        """Test subsampling_remove template has figure_ref 'Figure 27'."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.TEST, variant="subsampling_remove")
        assert pt.figure_ref == "Figure 27"

    def test_test_contamination_check_figure_ref(self) -> None:
        """Test contamination_check template has figure_ref 'Figure 28'."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.TEST, variant="contamination_check")
        assert pt.figure_ref == "Figure 28"

    @pytest.mark.parametrize("variant", TEST_VARIANTS)
    def test_test_variant_has_variables(self, variant: str) -> None:
        """Each test variant template has a non-empty variables list."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.TEST, variant=variant)
        assert isinstance(pt.variables, list)
        assert len(pt.variables) >= 1

    def test_test_default_has_variables(self) -> None:
        """Default test template has variables list with task_description and final_solution."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.TEST)
        assert "task_description" in pt.variables
        assert "final_solution" in pt.variables

    def test_all_test_variants_have_different_templates(self) -> None:
        """All four test variants produce distinct template text."""
        registry = PromptRegistry()
        templates = set()
        templates.add(registry.get(AgentType.TEST).template)
        for variant in TEST_VARIANTS:
            templates.add(registry.get(AgentType.TEST, variant=variant).template)
        assert len(templates) == 4


# ===========================================================================
# PromptRegistry.get() -- Error Cases
# ===========================================================================


@pytest.mark.unit
class TestPromptRegistryGetErrors:
    """PromptRegistry.get() raises KeyError for unknown agent types or variants."""

    def test_unknown_agent_type_raises_key_error(self) -> None:
        """get() with a non-existent agent type raises KeyError."""
        registry = PromptRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent_agent")  # type: ignore[arg-type]

    def test_unknown_variant_raises_key_error(self) -> None:
        """get() with an existing agent but unknown variant raises KeyError."""
        registry = PromptRegistry()
        with pytest.raises(KeyError):
            registry.get(AgentType.LEAKAGE, variant="nonexistent_variant")

    def test_variant_on_single_template_agent_raises_key_error(self) -> None:
        """get() with a variant on a single-template agent raises KeyError."""
        registry = PromptRegistry()
        with pytest.raises(KeyError):
            registry.get(AgentType.RETRIEVER, variant="nonexistent")

    def test_invalid_test_variant_raises_key_error(self) -> None:
        """get() with invalid test variant raises KeyError."""
        registry = PromptRegistry()
        with pytest.raises(KeyError):
            registry.get(AgentType.TEST, variant="bogus_variant")

    def test_invalid_leakage_variant_raises_key_error(self) -> None:
        """get() with invalid leakage variant raises KeyError."""
        registry = PromptRegistry()
        with pytest.raises(KeyError):
            registry.get(AgentType.LEAKAGE, variant="bogus_variant")


# ===========================================================================
# PromptRegistry -- Coverage of All Agent Types
# ===========================================================================


@pytest.mark.unit
class TestPromptRegistryCoverage:
    """PromptRegistry covers all 14 agent types with 18 total template variants."""

    @pytest.mark.parametrize("agent_type", list(AgentType))
    def test_every_agent_type_is_in_registry(self, agent_type: AgentType) -> None:
        """Every AgentType enum value can be retrieved from the registry."""
        registry = PromptRegistry()
        pt = registry.get(agent_type)
        assert pt.agent_type == agent_type

    def test_all_14_agent_types_covered(self) -> None:
        """All 14 agent types are retrievable from the registry."""
        registry = PromptRegistry()
        retrieved_types: set[AgentType] = set()
        for agent_type in AgentType:
            pt = registry.get(agent_type)
            retrieved_types.add(pt.agent_type)
        assert len(retrieved_types) == TOTAL_AGENT_TYPES

    def test_total_template_variants_count(self) -> None:
        """Total template variants across all agents is 23."""
        registry = PromptRegistry()
        count = 0
        # Single-template agents (no variant): 14
        for agent_type in SINGLE_TEMPLATE_AGENTS:
            registry.get(agent_type)
            count += 1
        # Leakage variants: 3 (detection, correction, deep_analysis)
        for variant in LEAKAGE_VARIANTS:
            registry.get(AgentType.LEAKAGE, variant=variant)
            count += 1
        # Test variants: default + 3 named = 4
        registry.get(AgentType.TEST)
        count += 1
        for variant in TEST_VARIANTS:
            registry.get(AgentType.TEST, variant=variant)
            count += 1
        # Validator variants: 2 (sanity, overfitting)
        for variant in VALIDATOR_VARIANTS:
            registry.get(AgentType.VALIDATOR, variant=variant)
            count += 1
        assert count == TOTAL_TEMPLATE_VARIANTS


# ===========================================================================
# PromptRegistry -- Template Content Validation
# ===========================================================================


@pytest.mark.unit
class TestPromptRegistryTemplateContent:
    """All loaded templates have valid, non-empty content."""

    @pytest.mark.parametrize("agent_type", list(AgentType))
    def test_template_is_nonempty_string(self, agent_type: AgentType) -> None:
        """Each template has a non-empty template string."""
        registry = PromptRegistry()
        pt = registry.get(agent_type)
        assert isinstance(pt.template, str)
        assert len(pt.template.strip()) > 0

    @pytest.mark.parametrize("agent_type", list(AgentType))
    def test_figure_ref_is_nonempty_string(self, agent_type: AgentType) -> None:
        """Each template has a non-empty figure_ref string."""
        registry = PromptRegistry()
        pt = registry.get(agent_type)
        assert isinstance(pt.figure_ref, str)
        assert len(pt.figure_ref.strip()) > 0

    @pytest.mark.parametrize("agent_type", list(AgentType))
    def test_variables_match_template_placeholders(self, agent_type: AgentType) -> None:
        """Each template's variables correspond to placeholders in the template.

        Every declared variable should appear as {variable} in the template text.
        """
        registry = PromptRegistry()
        pt = registry.get(agent_type)
        for var in pt.variables:
            assert f"{{{var}}}" in pt.template, (
                f"Variable '{var}' declared but not found "
                f"as placeholder in template for {agent_type}"
            )


# ===========================================================================
# PromptRegistry -- Specific Agent Figure References
# ===========================================================================


_EXPECTED_FIGURE_REFS: list[tuple[AgentType, str]] = [
    (AgentType.RETRIEVER, "Figure 9"),
    (AgentType.INIT, "Figure 10"),
    (AgentType.MERGER, "Figure 11"),
    (AgentType.DATA, "Figure 22"),
]


@pytest.mark.unit
class TestPromptRegistryFigureRefs:
    """Templates have correct figure references matching the paper."""

    @pytest.mark.parametrize(
        "agent_type,expected_ref",
        _EXPECTED_FIGURE_REFS,
        ids=[a.value for a, _ in _EXPECTED_FIGURE_REFS],
    )
    def test_figure_ref_matches_paper(
        self, agent_type: AgentType, expected_ref: str
    ) -> None:
        """Each agent's figure_ref matches the expected paper reference."""
        registry = PromptRegistry()
        pt = registry.get(agent_type)
        assert pt.figure_ref == expected_ref


# ===========================================================================
# PromptRegistry -- Specific Agent Variable Lists
# ===========================================================================


@pytest.mark.unit
class TestPromptRegistryVariables:
    """Templates have correct variable lists matching their YAML definitions."""

    def test_retriever_variables_are_task_description_and_m(self) -> None:
        """Retriever template declares variables task_description, target_column, M, research_context, and notes_context."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.RETRIEVER)
        assert sorted(pt.variables) == sorted(
            ["task_description", "target_column", "M", "research_context", "notes_context"]
        )

    def test_init_variables(self) -> None:
        """Init template declares variables task_description, target_column, model_name, example_code, research_context, notes_context."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.INIT)
        assert sorted(pt.variables) == sorted(
            ["task_description", "target_column", "model_name", "example_code", "research_context", "notes_context"]
        )

    def test_merger_variables(self) -> None:
        """Merger template declares variables base_code and reference_code."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.MERGER)
        assert sorted(pt.variables) == sorted(["base_code", "reference_code"])

    def test_coder_variables(self) -> None:
        """Coder template declares variables including code_block, plan, and task context."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.CODER)
        assert sorted(pt.variables) == sorted([
            "code_block", "current_score", "data_modality",
            "evaluation_metric", "metric_direction", "plan", "task_description",
        ])

    def test_debugger_variables(self) -> None:
        """Debugger template declares variables code and bug."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.DEBUGGER)
        assert sorted(pt.variables) == sorted(["code", "bug"])

    def test_data_variables(self) -> None:
        """Data template declares variables initial_solution, task_description, and target_column."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.DATA)
        assert sorted(pt.variables) == sorted(
            ["initial_solution", "task_description", "target_column"]
        )


# ===========================================================================
# PromptRegistry -- Render Integration (end-to-end)
# ===========================================================================


@pytest.mark.unit
class TestPromptRegistryRenderIntegration:
    """End-to-end tests: load from registry then render."""

    def test_retriever_load_and_render(self) -> None:
        """Load retriever template and render with actual values."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.RETRIEVER)
        rendered = pt.render(
            task_description="Predict house prices", target_column="price", M=4,
            research_context="", notes_context="",
        )
        assert "Predict house prices" in rendered
        assert "4" in rendered

    def test_leakage_detection_load_and_render(self) -> None:
        """Load leakage detection template and render with code."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.LEAKAGE, variant="detection")
        rendered = pt.render(code="df = pd.read_csv('train.csv')")
        assert "df = pd.read_csv('train.csv')" in rendered

    def test_test_default_load_and_render(self) -> None:
        """Load test default template and render with values."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.TEST)
        rendered = pt.render(
            task_description="Classify images",
            target_column="label",
            final_solution="import torch\n",
        )
        assert "Classify images" in rendered
        assert "import torch" in rendered

    def test_debugger_load_and_render(self) -> None:
        """Load debugger template and render with code and bug."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.DEBUGGER)
        rendered = pt.render(code="x = 1/0", bug="ZeroDivisionError")
        assert "x = 1/0" in rendered
        assert "ZeroDivisionError" in rendered

    def test_data_load_and_render(self) -> None:
        """Load data template and render with solution and description."""
        registry = PromptRegistry()
        pt = registry.get(AgentType.DATA)
        rendered = pt.render(
            initial_solution="import pandas as pd\n",
            task_description="Predict survival",
            target_column="Survived",
        )
        assert "import pandas as pd" in rendered
        assert "Predict survival" in rendered


# ===========================================================================
# PromptRegistry -- Consistency Property-based Tests
# ===========================================================================


@pytest.mark.unit
class TestPromptRegistryPropertyBased:
    """Property-based tests for PromptRegistry using Hypothesis."""

    @given(
        agent_type=st.sampled_from(list(AgentType)),
    )
    @settings(max_examples=30)
    def test_get_always_returns_prompt_template(self, agent_type: AgentType) -> None:
        """Property: get() always returns a PromptTemplate for any valid AgentType."""
        registry = PromptRegistry()
        pt = registry.get(agent_type)
        assert isinstance(pt, PromptTemplate)

    @given(
        agent_type=st.sampled_from(list(AgentType)),
    )
    @settings(max_examples=30)
    def test_get_agent_type_matches_request(self, agent_type: AgentType) -> None:
        """Property: returned template's agent_type always matches the requested one."""
        registry = PromptRegistry()
        pt = registry.get(agent_type)
        assert pt.agent_type == agent_type

    @given(
        agent_type=st.sampled_from(list(AgentType)),
    )
    @settings(max_examples=30)
    def test_get_idempotent(self, agent_type: AgentType) -> None:
        """Property: calling get() twice returns equal PromptTemplate objects."""
        registry = PromptRegistry()
        pt1 = registry.get(agent_type)
        pt2 = registry.get(agent_type)
        assert pt1 == pt2

    @given(
        bogus=st.text(
            alphabet=st.characters(whitelist_categories=("L",)),  # type: ignore[arg-type]
            min_size=1,
            max_size=50,
        ),
    )
    @settings(max_examples=30)
    def test_get_invalid_agent_raises_key_error(self, bogus: str) -> None:
        """Property: get() with an invalid agent type string raises KeyError.

        We filter out any string that happens to match a valid AgentType value.
        """
        valid_values = {a.value for a in AgentType}
        if bogus in valid_values:
            return
        registry = PromptRegistry()
        with pytest.raises(KeyError):
            registry.get(bogus)  # type: ignore[arg-type]


# ===========================================================================
# PromptRegistry -- Multiple Instances Consistency
# ===========================================================================


@pytest.mark.unit
class TestPromptRegistryMultipleInstances:
    """Multiple PromptRegistry instances are consistent."""

    def test_two_registries_have_same_length(self) -> None:
        """Two independently constructed registries have the same len()."""
        r1 = PromptRegistry()
        r2 = PromptRegistry()
        assert len(r1) == len(r2)

    def test_two_registries_return_equal_templates(self) -> None:
        """Two registries return equal templates for the same agent type."""
        r1 = PromptRegistry()
        r2 = PromptRegistry()
        for agent_type in AgentType:
            assert r1.get(agent_type) == r2.get(agent_type)


# ===========================================================================
# PromptRegistry -- Resilience to Unknown Agent Types
# ===========================================================================


@pytest.mark.unit
class TestPromptRegistryResilience:
    """PromptRegistry gracefully skips templates with unknown agent_type values."""

    def test_load_single_template_unknown_agent_type_skipped(self) -> None:
        """_load_single_template with unknown agent_type adds no template and raises no error."""
        registry = PromptRegistry()
        count_before = len(registry._templates)
        registry._load_single_template({
            "agent_type": "totally_unknown_agent",
            "figure_ref": "Figure 99",
            "template": "Hello {x}",
            "variables": ["x"],
        })
        assert len(registry._templates) == count_before

    def test_load_multi_template_unknown_agent_type_skipped(self) -> None:
        """_load_multi_template with unknown agent_type skips that entry without error."""
        registry = PromptRegistry()
        count_before = len(registry._templates)
        registry._load_multi_template([
            {
                "agent_type": "totally_unknown_agent",
                "variant": "v1",
                "figure_ref": "Figure 99",
                "template": "Hello {x}",
                "variables": ["x"],
            },
        ])
        assert len(registry._templates) == count_before

    def test_unknown_agent_type_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A warning is logged when an unknown agent_type is encountered."""
        import logging

        registry = PromptRegistry()
        with caplog.at_level(logging.WARNING, logger="mle_star.prompts"):
            registry._load_single_template({
                "agent_type": "totally_unknown_agent",
                "figure_ref": "Figure 99",
                "template": "Hello {x}",
                "variables": ["x"],
            })
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1
        assert "totally_unknown_agent" in warning_msgs[0].message
