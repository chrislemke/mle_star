"""Tests for MLE-STAR agent configuration and SDK integration types (Task 09).

Validates AgentConfig frozen Pydantic model, its ``to_agent_definition()``
and ``to_output_format()`` methods, and the ``build_default_agent_configs()``
factory function defined in ``src/mle_star/models.py``.  These tests are
written TDD-first -- the implementation does not yet exist.  They serve as
the executable specification for REQ-DM-036 through REQ-DM-040.

Refs:
    SRS 01b (Agent Config & SDK Integration), IMPLEMENTATION_PLAN.md Task 09.
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    AgentConfig,
    AgentType,
    ExtractorOutput,
    LeakageDetectionOutput,
    RetrieverOutput,
    build_default_agent_configs,
)
from pydantic import BaseModel, ValidationError
import pytest

# ---------------------------------------------------------------------------
# Constants -- canonical values from the spec
# ---------------------------------------------------------------------------

VALID_MODEL_VALUES: list[str] = ["sonnet", "opus", "haiku", "inherit"]

EXECUTION_AGENT_TYPES: list[AgentType] = [
    AgentType.INIT,
    AgentType.MERGER,
    AgentType.ABLATION,
    AgentType.CODER,
    AgentType.ENSEMBLER,
    AgentType.DEBUGGER,
]

READ_ONLY_AGENT_TYPES: list[AgentType] = [
    AgentType.LEAKAGE,
    AgentType.DATA,
    AgentType.TEST,
]

READ_WRITE_AGENT_TYPES: list[AgentType] = [
    AgentType.SUMMARIZE,
    AgentType.EXTRACTOR,
    AgentType.PLANNER,
    AgentType.ENS_PLANNER,
]

EXECUTION_TOOLS: list[str] = ["Bash", "Edit", "Write", "Read"]
READ_ONLY_TOOLS: list[str] = ["Read"]
READ_WRITE_TOOLS: list[str] = ["Read", "Write"]
RETRIEVER_TOOLS: list[str] = ["WebSearch", "WebFetch", "Write"]

# Mapping of agent types to expected output_schema (None means no schema)
AGENT_OUTPUT_SCHEMAS: dict[AgentType, type[BaseModel] | None] = {
    AgentType.RETRIEVER: None,
    AgentType.INIT: None,
    AgentType.MERGER: None,
    AgentType.ABLATION: None,
    AgentType.SUMMARIZE: None,
    AgentType.EXTRACTOR: ExtractorOutput,
    AgentType.PLANNER: None,
    AgentType.CODER: None,
    AgentType.ENS_PLANNER: None,
    AgentType.ENSEMBLER: None,
    AgentType.DEBUGGER: None,
    AgentType.LEAKAGE: LeakageDetectionOutput,
    AgentType.DATA: None,
    AgentType.TEST: None,
}


# ---------------------------------------------------------------------------
# Helpers -- factory functions for building valid model instances
# ---------------------------------------------------------------------------


class _DummyOutputSchema(BaseModel):
    """A minimal Pydantic model used as a test output schema."""

    value: str


def _make_agent_config(**overrides: Any) -> AgentConfig:
    """Build a valid AgentConfig with sensible defaults.

    All required fields are populated; any keyword argument overrides
    the corresponding default.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed AgentConfig instance.
    """
    defaults: dict[str, Any] = {
        "agent_type": AgentType.RETRIEVER,
        "description": "Retrieves ML models from the web.",
    }
    defaults.update(overrides)
    return AgentConfig(**defaults)


# ===========================================================================
# REQ-DM-036: AgentConfig Model -- Construction and Fields
# ===========================================================================


@pytest.mark.unit
class TestAgentConfigConstruction:
    """AgentConfig has 7 fields with correct types (REQ-DM-036)."""

    def test_valid_construction_with_required_fields_only(self) -> None:
        """Constructing with only required fields succeeds with correct defaults."""
        config = AgentConfig(
            agent_type=AgentType.RETRIEVER,
            description="Retrieves ML models from the web.",
        )
        assert config.agent_type == AgentType.RETRIEVER
        assert config.description == "Retrieves ML models from the web."
        assert config.system_prompt is None
        assert config.tools is None
        assert config.model is None
        assert config.output_schema is None
        assert config.max_turns is None

    def test_valid_construction_with_all_fields(self) -> None:
        """Constructing with all fields succeeds and stores correct values."""
        config = AgentConfig(
            agent_type=AgentType.CODER,
            description="Writes code refinements.",
            system_prompt="You are an expert coder.",
            tools=["Bash", "Edit", "Write", "Read"],
            model="sonnet",
            output_schema=_DummyOutputSchema,
            max_turns=10,
        )
        assert config.agent_type == AgentType.CODER
        assert config.description == "Writes code refinements."
        assert config.system_prompt == "You are an expert coder."
        assert config.tools == ["Bash", "Edit", "Write", "Read"]
        assert config.model == "sonnet"
        assert config.output_schema is _DummyOutputSchema
        assert config.max_turns == 10

    def test_agent_type_is_agent_type_enum(self) -> None:
        """agent_type field holds an AgentType enum value."""
        config = _make_agent_config(agent_type=AgentType.DEBUGGER)
        assert isinstance(config.agent_type, AgentType)
        assert config.agent_type == AgentType.DEBUGGER

    def test_description_is_string(self) -> None:
        """Description field holds a string value."""
        config = _make_agent_config(description="Test description")
        assert isinstance(config.description, str)
        assert config.description == "Test description"

    def test_system_prompt_default_is_none(self) -> None:
        """system_prompt defaults to None when not provided."""
        config = _make_agent_config()
        assert config.system_prompt is None

    def test_system_prompt_accepts_string(self) -> None:
        """system_prompt accepts a string value."""
        config = _make_agent_config(system_prompt="You are a helpful agent.")
        assert config.system_prompt == "You are a helpful agent."

    def test_tools_default_is_none(self) -> None:
        """Tools defaults to None when not provided."""
        config = _make_agent_config()
        assert config.tools is None

    def test_tools_accepts_list_of_strings(self) -> None:
        """Tools accepts a list of strings."""
        config = _make_agent_config(tools=["Bash", "Read"])
        assert config.tools == ["Bash", "Read"]

    def test_tools_accepts_empty_list(self) -> None:
        """Tools accepts an empty list."""
        config = _make_agent_config(tools=[])
        assert config.tools == []

    def test_model_default_is_none(self) -> None:
        """Model defaults to None when not provided."""
        config = _make_agent_config()
        assert config.model is None

    @pytest.mark.parametrize("model_value", VALID_MODEL_VALUES)
    def test_model_accepts_valid_values(self, model_value: str) -> None:
        """Model accepts all valid string values: sonnet, opus, haiku, inherit."""
        config = _make_agent_config(model=model_value)
        assert config.model == model_value

    def test_output_schema_default_is_none(self) -> None:
        """output_schema defaults to None when not provided."""
        config = _make_agent_config()
        assert config.output_schema is None

    def test_output_schema_accepts_pydantic_model_class(self) -> None:
        """output_schema accepts a Pydantic model class (not instance)."""
        config = _make_agent_config(output_schema=RetrieverOutput)
        assert config.output_schema is RetrieverOutput

    def test_output_schema_holds_class_not_instance(self) -> None:
        """output_schema stores the class itself, not an instance."""
        config = _make_agent_config(output_schema=_DummyOutputSchema)
        assert config.output_schema is _DummyOutputSchema
        assert isinstance(config.output_schema, type)

    def test_max_turns_default_is_none(self) -> None:
        """max_turns defaults to None when not provided."""
        config = _make_agent_config()
        assert config.max_turns is None

    def test_max_turns_accepts_positive_int(self) -> None:
        """max_turns accepts a positive integer."""
        config = _make_agent_config(max_turns=5)
        assert config.max_turns == 5

    def test_max_turns_accepts_one(self) -> None:
        """max_turns accepts the value 1."""
        config = _make_agent_config(max_turns=1)
        assert config.max_turns == 1


@pytest.mark.unit
class TestAgentConfigRequiredFields:
    """AgentConfig raises ValidationError when required fields are missing."""

    def test_missing_agent_type_raises(self) -> None:
        """Omitting agent_type raises ValidationError."""
        with pytest.raises(ValidationError):
            AgentConfig(description="A description")  # type: ignore[call-arg]

    def test_missing_description_raises(self) -> None:
        """Omitting description raises ValidationError."""
        with pytest.raises(ValidationError):
            AgentConfig(agent_type=AgentType.RETRIEVER)  # type: ignore[call-arg]

    def test_missing_both_required_raises(self) -> None:
        """Omitting both required fields raises ValidationError."""
        with pytest.raises(ValidationError):
            AgentConfig()  # type: ignore[call-arg]


@pytest.mark.unit
class TestAgentConfigAllAgentTypes:
    """AgentConfig accepts all 14 AgentType enum values."""

    @pytest.mark.parametrize("agent_type", list(AgentType))
    def test_accepts_all_agent_types(self, agent_type: AgentType) -> None:
        """Every AgentType value is accepted by AgentConfig."""
        config = _make_agent_config(agent_type=agent_type)
        assert config.agent_type == agent_type

    def test_invalid_agent_type_raises(self) -> None:
        """An invalid string for agent_type raises ValidationError."""
        with pytest.raises(ValidationError):
            AgentConfig(
                agent_type="nonexistent_agent",  # type: ignore[arg-type]
                description="A description",
            )


# ===========================================================================
# REQ-DM-039: AgentConfig Frozen (Immutable)
# ===========================================================================


@pytest.mark.unit
class TestAgentConfigFrozen:
    """AgentConfig is frozen (immutable) per REQ-DM-039."""

    def test_cannot_mutate_agent_type(self) -> None:
        """Assignment to agent_type raises an error."""
        config = _make_agent_config()
        with pytest.raises(ValidationError):
            config.agent_type = AgentType.CODER  # type: ignore[misc]

    def test_cannot_mutate_description(self) -> None:
        """Assignment to description raises an error."""
        config = _make_agent_config()
        with pytest.raises(ValidationError):
            config.description = "Changed."  # type: ignore[misc]

    def test_cannot_mutate_system_prompt(self) -> None:
        """Assignment to system_prompt raises an error."""
        config = _make_agent_config(system_prompt="original")
        with pytest.raises(ValidationError):
            config.system_prompt = "changed"  # type: ignore[misc]

    def test_cannot_mutate_tools(self) -> None:
        """Assignment to tools raises an error."""
        config = _make_agent_config(tools=["Read"])
        with pytest.raises(ValidationError):
            config.tools = ["Bash"]  # type: ignore[misc]

    def test_cannot_mutate_model(self) -> None:
        """Assignment to model raises an error."""
        config = _make_agent_config(model="sonnet")
        with pytest.raises(ValidationError):
            config.model = "opus"  # type: ignore[misc]

    def test_cannot_mutate_output_schema(self) -> None:
        """Assignment to output_schema raises an error."""
        config = _make_agent_config(output_schema=RetrieverOutput)
        with pytest.raises(ValidationError):
            config.output_schema = ExtractorOutput  # type: ignore[misc]

    def test_cannot_mutate_max_turns(self) -> None:
        """Assignment to max_turns raises an error."""
        config = _make_agent_config(max_turns=5)
        with pytest.raises(ValidationError):
            config.max_turns = 10  # type: ignore[misc]


# ===========================================================================
# REQ-DM-037: AgentConfig.to_agent_definition()
# ===========================================================================


@pytest.mark.unit
class TestAgentConfigToAgentDefinition:
    """to_agent_definition() returns dict with keys: description, prompt, tools, model (REQ-DM-037)."""

    def test_returns_dict(self) -> None:
        """to_agent_definition() returns a dict."""
        config = _make_agent_config()
        result = config.to_agent_definition()
        assert isinstance(result, dict)

    def test_has_exactly_four_keys(self) -> None:
        """Returned dict has exactly the four required keys."""
        config = _make_agent_config()
        result = config.to_agent_definition()
        assert set(result.keys()) == {"description", "prompt", "tools", "model"}

    def test_description_maps_to_self_description(self) -> None:
        """'description' key maps to self.description."""
        config = _make_agent_config(description="Test agent description.")
        result = config.to_agent_definition()
        assert result["description"] == "Test agent description."

    def test_prompt_maps_to_system_prompt_when_set(self) -> None:
        """'prompt' key maps to self.system_prompt when it is not None."""
        config = _make_agent_config(system_prompt="You are an expert.")
        result = config.to_agent_definition()
        assert result["prompt"] == "You are an expert."

    def test_prompt_defaults_to_empty_string_when_system_prompt_is_none(self) -> None:
        """'prompt' key defaults to '' when system_prompt is None."""
        config = _make_agent_config(system_prompt=None)
        result = config.to_agent_definition()
        assert result["prompt"] == ""

    def test_tools_maps_to_self_tools(self) -> None:
        """'tools' key maps to self.tools."""
        config = _make_agent_config(tools=["Bash", "Read"])
        result = config.to_agent_definition()
        assert result["tools"] == ["Bash", "Read"]

    def test_tools_is_none_when_not_set(self) -> None:
        """'tools' key is None when self.tools is None."""
        config = _make_agent_config(tools=None)
        result = config.to_agent_definition()
        assert result["tools"] is None

    def test_model_maps_to_self_model(self) -> None:
        """'model' key maps to self.model."""
        config = _make_agent_config(model="opus")
        result = config.to_agent_definition()
        assert result["model"] == "opus"

    def test_model_is_none_when_not_set(self) -> None:
        """'model' key is None when self.model is None."""
        config = _make_agent_config(model=None)
        result = config.to_agent_definition()
        assert result["model"] is None

    def test_full_config_produces_correct_dict(self) -> None:
        """A fully populated config produces the correct dict."""
        config = AgentConfig(
            agent_type=AgentType.CODER,
            description="Writes code.",
            system_prompt="You are an expert coder.",
            tools=["Bash", "Edit", "Write", "Read"],
            model="sonnet",
            output_schema=_DummyOutputSchema,
            max_turns=10,
        )
        result = config.to_agent_definition()
        assert result == {
            "description": "Writes code.",
            "prompt": "You are an expert coder.",
            "tools": ["Bash", "Edit", "Write", "Read"],
            "model": "sonnet",
        }

    def test_minimal_config_produces_correct_dict(self) -> None:
        """A minimal config (required fields only) produces correct defaults."""
        config = AgentConfig(
            agent_type=AgentType.RETRIEVER,
            description="Retrieves models.",
        )
        result = config.to_agent_definition()
        assert result == {
            "description": "Retrieves models.",
            "prompt": "",
            "tools": None,
            "model": None,
        }


# ===========================================================================
# REQ-DM-038: AgentConfig.to_output_format()
# ===========================================================================


@pytest.mark.unit
class TestAgentConfigToOutputFormat:
    """to_output_format() returns JSON schema dict or None (REQ-DM-038)."""

    def test_returns_none_when_output_schema_is_none(self) -> None:
        """Returns None when output_schema is None."""
        config = _make_agent_config(output_schema=None)
        result = config.to_output_format()
        assert result is None

    def test_returns_dict_when_output_schema_is_set(self) -> None:
        """Returns a dict when output_schema is set."""
        config = _make_agent_config(output_schema=_DummyOutputSchema)
        result = config.to_output_format()
        assert isinstance(result, dict)

    def test_dict_has_type_key(self) -> None:
        """Returned dict has 'type' key set to 'json_schema'."""
        config = _make_agent_config(output_schema=_DummyOutputSchema)
        result = config.to_output_format()
        assert result is not None
        assert result["type"] == "json_schema"

    def test_dict_has_schema_key(self) -> None:
        """Returned dict has 'schema' key containing the JSON schema."""
        config = _make_agent_config(output_schema=_DummyOutputSchema)
        result = config.to_output_format()
        assert result is not None
        assert "schema" in result

    def test_schema_matches_model_json_schema(self) -> None:
        """'schema' value matches output_schema.model_json_schema()."""
        config = _make_agent_config(output_schema=_DummyOutputSchema)
        result = config.to_output_format()
        assert result is not None
        expected_schema = _DummyOutputSchema.model_json_schema()
        assert result["schema"] == expected_schema

    def test_schema_contains_properties(self) -> None:
        """The JSON schema dict contains a 'properties' key from the Pydantic model."""
        config = _make_agent_config(output_schema=_DummyOutputSchema)
        result = config.to_output_format()
        assert result is not None
        assert "properties" in result["schema"]
        assert "value" in result["schema"]["properties"]

    def test_with_retriever_output_schema(self) -> None:
        """to_output_format works correctly with RetrieverOutput schema."""
        config = _make_agent_config(output_schema=RetrieverOutput)
        result = config.to_output_format()
        assert result is not None
        assert result["type"] == "json_schema"
        assert result["schema"] == RetrieverOutput.model_json_schema()

    def test_with_extractor_output_schema(self) -> None:
        """to_output_format works correctly with ExtractorOutput schema."""
        config = _make_agent_config(output_schema=ExtractorOutput)
        result = config.to_output_format()
        assert result is not None
        assert result["type"] == "json_schema"
        assert result["schema"] == ExtractorOutput.model_json_schema()

    def test_with_leakage_detection_output_schema(self) -> None:
        """to_output_format works correctly with LeakageDetectionOutput schema."""
        config = _make_agent_config(output_schema=LeakageDetectionOutput)
        result = config.to_output_format()
        assert result is not None
        assert result["type"] == "json_schema"
        assert result["schema"] == LeakageDetectionOutput.model_json_schema()

    def test_dict_has_exactly_two_keys(self) -> None:
        """Returned dict has exactly two keys: 'type' and 'schema'."""
        config = _make_agent_config(output_schema=_DummyOutputSchema)
        result = config.to_output_format()
        assert result is not None
        assert set(result.keys()) == {"type", "schema"}


# ===========================================================================
# REQ-DM-040: build_default_agent_configs() Factory
# ===========================================================================


@pytest.mark.unit
class TestBuildDefaultAgentConfigsStructure:
    """build_default_agent_configs() returns dict[AgentType, AgentConfig] with 14 agents (REQ-DM-040)."""

    def test_returns_dict(self) -> None:
        """build_default_agent_configs() returns a dict."""
        configs = build_default_agent_configs()
        assert isinstance(configs, dict)

    def test_returns_exactly_16_configs(self) -> None:
        """The dict contains exactly 16 entries (one per agent)."""
        configs = build_default_agent_configs()
        assert len(configs) == 16

    def test_keys_are_all_agent_type_values(self) -> None:
        """Every key is an AgentType enum value."""
        configs = build_default_agent_configs()
        for key in configs:
            assert isinstance(key, AgentType)

    def test_keys_cover_all_agent_types(self) -> None:
        """The dict keys are exactly the set of all AgentType members."""
        configs = build_default_agent_configs()
        expected_keys = set(AgentType)
        actual_keys = set(configs.keys())
        assert actual_keys == expected_keys

    def test_values_are_all_agent_config_instances(self) -> None:
        """Every value is an AgentConfig instance."""
        configs = build_default_agent_configs()
        for value in configs.values():
            assert isinstance(value, AgentConfig)

    def test_each_config_agent_type_matches_its_key(self) -> None:
        """Each AgentConfig's agent_type matches its dict key."""
        configs = build_default_agent_configs()
        for agent_type, config in configs.items():
            assert config.agent_type == agent_type

    def test_each_config_has_nonempty_description(self) -> None:
        """Every AgentConfig has a non-empty description string."""
        configs = build_default_agent_configs()
        for agent_type, config in configs.items():
            assert isinstance(config.description, str), (
                f"{agent_type} has non-string description"
            )
            assert len(config.description) > 0, f"{agent_type} has empty description"

    def test_each_config_has_tools_list(self) -> None:
        """Every AgentConfig has a non-None tools list."""
        configs = build_default_agent_configs()
        for agent_type, config in configs.items():
            assert config.tools is not None, f"{agent_type} has tools=None"
            assert isinstance(config.tools, list), f"{agent_type} tools is not a list"


@pytest.mark.unit
class TestBuildDefaultAgentConfigsRetriever:
    """Retriever agent has specific tools and output_schema (REQ-OR-008)."""

    def test_retriever_tools(self) -> None:
        """A_retriever has tools=['WebSearch', 'WebFetch', 'Write']."""
        configs = build_default_agent_configs()
        retriever = configs[AgentType.RETRIEVER]
        assert retriever.tools == ["WebSearch", "WebFetch", "Write"]

    def test_retriever_output_schema(self) -> None:
        """A_retriever has no output_schema (parsed from prompt text)."""
        configs = build_default_agent_configs()
        retriever = configs[AgentType.RETRIEVER]
        assert retriever.output_schema is None


@pytest.mark.unit
class TestBuildDefaultAgentConfigsExecutionAgents:
    """Execution agents have tools=['Bash', 'Edit', 'Write', 'Read'] (REQ-OR-008)."""

    @pytest.mark.parametrize("agent_type", EXECUTION_AGENT_TYPES)
    def test_execution_agent_tools(self, agent_type: AgentType) -> None:
        """Execution agents (A_init, A_merger, A_abl, A_coder, A_ensembler, A_debugger, A_test) have correct tools."""
        configs = build_default_agent_configs()
        config = configs[agent_type]
        assert config.tools == ["Bash", "Edit", "Write", "Read"]


@pytest.mark.unit
class TestBuildDefaultAgentConfigsReadOnlyAgents:
    """Read-only agents have tools=['Read'] (REQ-OR-008)."""

    @pytest.mark.parametrize("agent_type", READ_ONLY_AGENT_TYPES)
    def test_read_only_agent_tools(self, agent_type: AgentType) -> None:
        """Read-only agents (A_leakage, A_data, A_test) have tools=['Read']."""
        configs = build_default_agent_configs()
        config = configs[agent_type]
        assert config.tools == ["Read"]


@pytest.mark.unit
class TestBuildDefaultAgentConfigsReadWriteAgents:
    """Research/planning agents have tools=['Read', 'Write'] for note-taking."""

    @pytest.mark.parametrize("agent_type", READ_WRITE_AGENT_TYPES)
    def test_read_write_agent_tools(self, agent_type: AgentType) -> None:
        """Research/planning agents (A_summarize, A_extractor, A_planner, A_ens_planner) have tools=['Read', 'Write']."""
        configs = build_default_agent_configs()
        config = configs[agent_type]
        assert config.tools == ["Read", "Write"]


@pytest.mark.unit
class TestBuildDefaultAgentConfigsOutputSchemas:
    """Agents have the correct output_schema assignments (REQ-OR-008)."""

    def test_retriever_output_schema_is_none(self) -> None:
        """A_retriever has no output_schema (parsed from prompt text)."""
        configs = build_default_agent_configs()
        assert configs[AgentType.RETRIEVER].output_schema is None

    def test_extractor_output_schema_is_extractor_output(self) -> None:
        """A_extractor has output_schema=ExtractorOutput."""
        configs = build_default_agent_configs()
        assert configs[AgentType.EXTRACTOR].output_schema is ExtractorOutput

    def test_leakage_output_schema_is_leakage_detection_output(self) -> None:
        """A_leakage has output_schema=LeakageDetectionOutput."""
        configs = build_default_agent_configs()
        assert configs[AgentType.LEAKAGE].output_schema is LeakageDetectionOutput

    def test_data_output_schema_is_none(self) -> None:
        """A_data has output_schema=None."""
        configs = build_default_agent_configs()
        assert configs[AgentType.DATA].output_schema is None

    @pytest.mark.parametrize(
        "agent_type,expected_schema",
        [(at, schema) for at, schema in AGENT_OUTPUT_SCHEMAS.items()],
        ids=[at.value for at in AGENT_OUTPUT_SCHEMAS],
    )
    def test_all_output_schemas_match_spec(
        self,
        agent_type: AgentType,
        expected_schema: type[BaseModel] | None,
    ) -> None:
        """Every agent has the correct output_schema per the spec."""
        configs = build_default_agent_configs()
        config = configs[agent_type]
        if expected_schema is None:
            assert config.output_schema is None, (
                f"{agent_type.value} should have output_schema=None"
            )
        else:
            assert config.output_schema is expected_schema, (
                f"{agent_type.value} should have output_schema={expected_schema.__name__}"
            )


@pytest.mark.unit
class TestBuildDefaultAgentConfigsToolSummary:
    """All agent tool assignments match the spec from REQ-OR-008."""

    def test_all_tools_are_from_known_set(self) -> None:
        """Every tool name across all configs is from a known tool set."""
        known_tools = {"Bash", "Edit", "Write", "Read", "WebSearch", "WebFetch"}
        configs = build_default_agent_configs()
        for agent_type, config in configs.items():
            assert config.tools is not None
            for tool in config.tools:
                assert tool in known_tools, (
                    f"{agent_type.value} has unknown tool: {tool}"
                )

    def test_web_tools_only_on_web_agents(self) -> None:
        """Only A_retriever and A_researcher have WebSearch or WebFetch in their tools."""
        configs = build_default_agent_configs()
        web_tools = {"WebSearch", "WebFetch"}
        web_agents = {AgentType.RETRIEVER, AgentType.RESEARCHER}
        for agent_type, config in configs.items():
            assert config.tools is not None
            has_web_tools = bool(web_tools & set(config.tools))
            if agent_type in web_agents:
                assert has_web_tools, f"{agent_type.value} should have web tools"
            else:
                assert not has_web_tools, (
                    f"{agent_type.value} should not have web tools"
                )


# ===========================================================================
# Property-based tests: AgentConfig with Hypothesis
# ===========================================================================


@pytest.mark.unit
class TestAgentConfigPropertyBased:
    """Property-based tests for AgentConfig using Hypothesis."""

    @given(
        description=st.text(min_size=1, max_size=200),
        agent_type=st.sampled_from(list(AgentType)),
    )
    @settings(max_examples=50)
    def test_any_nonempty_description_with_valid_agent_type_accepted(
        self, description: str, agent_type: AgentType
    ) -> None:
        """Property: any non-empty description with any AgentType creates a valid config."""
        config = AgentConfig(
            agent_type=agent_type,
            description=description,
        )
        assert config.agent_type == agent_type
        assert config.description == description

    @given(
        description=st.text(min_size=1, max_size=100),
        system_prompt=st.one_of(st.none(), st.text(min_size=0, max_size=200)),
        model=st.one_of(st.none(), st.sampled_from(VALID_MODEL_VALUES)),
        max_turns=st.one_of(st.none(), st.integers(min_value=1, max_value=100)),
    )
    @settings(max_examples=30)
    def test_optional_fields_always_produce_valid_config(
        self,
        description: str,
        system_prompt: str | None,
        model: str | None,
        max_turns: int | None,
    ) -> None:
        """Property: any combination of valid optional fields creates a valid config."""
        config = AgentConfig(
            agent_type=AgentType.RETRIEVER,
            description=description,
            system_prompt=system_prompt,
            model=model,
            max_turns=max_turns,
        )
        assert config.description == description
        assert config.system_prompt == system_prompt
        assert config.model == model
        assert config.max_turns == max_turns

    @given(
        agent_type=st.sampled_from(list(AgentType)),
    )
    @settings(max_examples=30)
    def test_to_agent_definition_always_returns_four_keys(
        self, agent_type: AgentType
    ) -> None:
        """Property: to_agent_definition() always returns a dict with exactly 4 keys."""
        config = AgentConfig(
            agent_type=agent_type,
            description="Test description",
        )
        result = config.to_agent_definition()
        assert isinstance(result, dict)
        assert set(result.keys()) == {"description", "prompt", "tools", "model"}

    @given(
        description=st.text(min_size=1, max_size=100),
        system_prompt=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
    )
    @settings(max_examples=30)
    def test_to_agent_definition_prompt_is_system_prompt_or_empty(
        self, description: str, system_prompt: str | None
    ) -> None:
        """Property: to_agent_definition 'prompt' equals system_prompt or '' when None."""
        config = AgentConfig(
            agent_type=AgentType.CODER,
            description=description,
            system_prompt=system_prompt,
        )
        result = config.to_agent_definition()
        if system_prompt is not None:
            assert result["prompt"] == system_prompt
        else:
            assert result["prompt"] == ""


@pytest.mark.unit
class TestBuildDefaultAgentConfigsPropertyBased:
    """Property-based tests for build_default_agent_configs using Hypothesis."""

    def test_factory_is_idempotent(self) -> None:
        """Calling build_default_agent_configs multiple times returns equivalent results."""
        first = build_default_agent_configs()
        second = build_default_agent_configs()
        assert set(first.keys()) == set(second.keys())
        for agent_type in first:
            assert first[agent_type].agent_type == second[agent_type].agent_type
            assert first[agent_type].description == second[agent_type].description
            assert first[agent_type].tools == second[agent_type].tools
            assert first[agent_type].output_schema is second[agent_type].output_schema

    def test_factory_returns_fresh_dict_each_call(self) -> None:
        """Each call returns a new dict instance (not a shared reference)."""
        first = build_default_agent_configs()
        second = build_default_agent_configs()
        assert first is not second

    @given(agent_type=st.sampled_from(list(AgentType)))
    @settings(max_examples=14)
    def test_every_agent_type_has_matching_key_and_field(
        self, agent_type: AgentType
    ) -> None:
        """Property: for every AgentType, the config's agent_type matches its key."""
        configs = build_default_agent_configs()
        assert agent_type in configs
        assert configs[agent_type].agent_type == agent_type


# ===========================================================================
# Integration: to_agent_definition + to_output_format with default configs
# ===========================================================================


@pytest.mark.unit
class TestDefaultConfigIntegration:
    """Default agent configs produce valid agent definitions and output formats."""

    @pytest.mark.parametrize("agent_type", list(AgentType))
    def test_to_agent_definition_succeeds_for_all_defaults(
        self, agent_type: AgentType
    ) -> None:
        """to_agent_definition() succeeds for every default config."""
        configs = build_default_agent_configs()
        result = configs[agent_type].to_agent_definition()
        assert isinstance(result, dict)
        assert "description" in result
        assert "prompt" in result
        assert "tools" in result
        assert "model" in result

    @pytest.mark.parametrize("agent_type", list(AgentType))
    def test_to_output_format_returns_correct_type_for_all_defaults(
        self, agent_type: AgentType
    ) -> None:
        """to_output_format() returns dict or None for every default config."""
        configs = build_default_agent_configs()
        config = configs[agent_type]
        result = config.to_output_format()
        if config.output_schema is not None:
            assert isinstance(result, dict)
            assert result["type"] == "json_schema"
            assert "schema" in result
        else:
            assert result is None

    def test_retriever_to_output_format_is_none(self) -> None:
        """Retriever default config has no output_schema (parsed from prompt text)."""
        configs = build_default_agent_configs()
        result = configs[AgentType.RETRIEVER].to_output_format()
        assert result is None

    def test_extractor_to_output_format_produces_extractor_schema(self) -> None:
        """Extractor default config's to_output_format produces ExtractorOutput schema."""
        configs = build_default_agent_configs()
        result = configs[AgentType.EXTRACTOR].to_output_format()
        assert result is not None
        assert result["schema"] == ExtractorOutput.model_json_schema()

    def test_leakage_to_output_format_produces_leakage_schema(self) -> None:
        """Leakage default config's to_output_format produces LeakageDetectionOutput schema."""
        configs = build_default_agent_configs()
        result = configs[AgentType.LEAKAGE].to_output_format()
        assert result is not None
        assert result["schema"] == LeakageDetectionOutput.model_json_schema()
