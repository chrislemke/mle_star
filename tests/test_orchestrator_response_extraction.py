"""Tests for ``_extract_result_from_json_output()`` and its integration with ``send_message()``.

Validates the JSON output extraction helper that parses Claude CLI
``--output-format json`` responses, extracting the actual result content
from the JSON array of conversation messages.

Tests cover:
- Happy path: typical JSON array with result message
- Fallback paths: plain text, JSON object, empty array, missing result
- Edge cases: dict/list result fields, empty string, whitespace, invalid JSON
- Multiple result messages (last wins)
- Integration with ``ClaudeCodeClient.send_message()``

Refs:
    SRS 09a -- Orchestrator Entry Point & SDK Client Setup.
    IMPLEMENTATION_PLAN.md Task 42.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    AgentType,
    RetrieverOutput,
    build_default_agent_configs,
)
from mle_star.orchestrator import ClaudeCodeClient, _extract_result_from_json_output
import pytest

# ---------------------------------------------------------------------------
# Module path constant for patching
# ---------------------------------------------------------------------------

_MODULE = "mle_star.orchestrator"

# ---------------------------------------------------------------------------
# Reusable test helpers
# ---------------------------------------------------------------------------


def _make_json_array(*messages: dict[str, Any]) -> str:
    """Serialize a sequence of dicts into a JSON array string.

    Args:
        *messages: Dicts representing conversation messages.

    Returns:
        A JSON-encoded array string.
    """
    return json.dumps(list(messages))


def _make_result_message(result_content: Any) -> dict[str, Any]:
    """Build a result-type message dict for the JSON array.

    Args:
        result_content: The value for the ``"result"`` field.

    Returns:
        A message dict with ``"type": "result"`` and the given content.
    """
    return {"type": "result", "result": result_content}


def _make_system_message(text: str = "System initialized") -> dict[str, Any]:
    """Build a system-type message dict.

    Args:
        text: The system message text.

    Returns:
        A message dict with ``"type": "system"``.
    """
    return {"type": "system", "text": text}


def _make_assistant_message(text: str = "Working on it...") -> dict[str, Any]:
    """Build an assistant-type message dict.

    Args:
        text: The assistant message text.

    Returns:
        A message dict with ``"type": "assistant"``.
    """
    return {"type": "assistant", "text": text}


def _make_client() -> ClaudeCodeClient:
    """Build a ClaudeCodeClient with default agent configs.

    Returns:
        A configured ClaudeCodeClient instance.
    """
    return ClaudeCodeClient(
        system_prompt="You are a test agent.",
        agent_configs=build_default_agent_configs(),
        model="sonnet",
    )


# ===========================================================================
# Unit tests: _extract_result_from_json_output
# ===========================================================================


@pytest.mark.unit
class TestExtractResultFromJsonOutput:
    """Unit tests for _extract_result_from_json_output() helper."""

    # -----------------------------------------------------------------------
    # 1. Typical JSON array with system + assistant + result messages
    # -----------------------------------------------------------------------

    def test_typical_json_array_extracts_result_content(self) -> None:
        """Typical JSON array with system, assistant, and result messages extracts the result."""
        # Arrange
        result_content = (
            '{"models": [{"model_name": "xgboost", "example_code": "import xgb"}]}'
        )
        raw_output = _make_json_array(
            _make_system_message(),
            _make_assistant_message("I will retrieve models."),
            _make_result_message(result_content),
        )

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == result_content

    def test_single_result_message_in_array(self) -> None:
        """Array containing only a result message extracts it correctly."""
        # Arrange
        result_content = "Simple text result"
        raw_output = _make_json_array(_make_result_message(result_content))

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == result_content

    # -----------------------------------------------------------------------
    # 2. Plain text input -> returns as-is
    # -----------------------------------------------------------------------

    def test_plain_text_returns_as_is(self) -> None:
        """Plain text input (not JSON) is returned unchanged."""
        # Arrange
        raw_output = "This is a plain text response from the CLI."

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == raw_output

    def test_plain_text_with_newlines_returns_as_is(self) -> None:
        """Multi-line plain text is returned unchanged."""
        # Arrange
        raw_output = "Line 1\nLine 2\nLine 3"

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == raw_output

    def test_empty_string_returns_as_is(self) -> None:
        """Empty string input is returned unchanged."""
        # Arrange
        raw_output = ""

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == ""

    # -----------------------------------------------------------------------
    # 3. JSON object (not array) -> returns as-is
    # -----------------------------------------------------------------------

    def test_json_object_returns_as_is(self) -> None:
        """A JSON object (not array) is returned unchanged."""
        # Arrange
        raw_output = '{"key": "value", "nested": {"a": 1}}'

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == raw_output

    def test_json_object_starting_with_brace_returns_as_is(self) -> None:
        """JSON object starting with '{' does not trigger array parsing."""
        # Arrange
        raw_output = '{"type": "result", "result": "should not be extracted"}'

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert -- the function only handles JSON arrays, so the object is returned as-is
        assert extracted == raw_output

    # -----------------------------------------------------------------------
    # 4. Empty array -> returns raw stdout
    # -----------------------------------------------------------------------

    def test_empty_array_returns_raw_stdout(self) -> None:
        """An empty JSON array returns the raw output unchanged."""
        # Arrange
        raw_output = "[]"

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == raw_output

    # -----------------------------------------------------------------------
    # 5. Array without result message -> returns raw stdout
    # -----------------------------------------------------------------------

    def test_array_without_result_message_returns_raw_stdout(self) -> None:
        """A JSON array with no 'type: result' message returns raw output."""
        # Arrange
        raw_output = _make_json_array(
            _make_system_message(),
            _make_assistant_message("I am working."),
            {"type": "tool_use", "tool": "Bash"},
        )

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == raw_output

    def test_array_with_only_system_messages_returns_raw_stdout(self) -> None:
        """Array containing only system messages returns raw output."""
        # Arrange
        raw_output = _make_json_array(
            _make_system_message("init"),
            _make_system_message("ready"),
        )

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == raw_output

    # -----------------------------------------------------------------------
    # 6. Result field is dict -> re-serializes to JSON string
    # -----------------------------------------------------------------------

    def test_result_field_dict_reserialized_to_json_string(self) -> None:
        """When result field is a dict, it is re-serialized via json.dumps()."""
        # Arrange
        result_dict = {
            "models": [
                {"model_name": "xgboost", "example_code": "import xgboost as xgb"}
            ]
        }
        raw_output = _make_json_array(
            _make_system_message(),
            _make_result_message(result_dict),
        )

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert -- should be a JSON string that can be parsed back
        assert isinstance(extracted, str)
        parsed = json.loads(extracted)
        assert parsed == result_dict
        assert parsed["models"][0]["model_name"] == "xgboost"

    def test_result_field_list_reserialized_to_json_string(self) -> None:
        """When result field is a list, it is re-serialized via json.dumps()."""
        # Arrange
        result_list = [{"model_name": "rf"}, {"model_name": "lgbm"}]
        raw_output = _make_json_array(
            _make_result_message(result_list),
        )

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert isinstance(extracted, str)
        parsed = json.loads(extracted)
        assert parsed == result_list
        assert len(parsed) == 2

    def test_result_field_nested_dict_roundtrips_correctly(self) -> None:
        """Deeply nested dict in result field roundtrips through serialization."""
        # Arrange
        nested = {"a": {"b": {"c": [1, 2, {"d": True}]}}}
        raw_output = _make_json_array(_make_result_message(nested))

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert json.loads(extracted) == nested

    def test_result_field_empty_dict_reserialized(self) -> None:
        """Empty dict in result field is re-serialized to '{}'."""
        # Arrange
        raw_output = _make_json_array(_make_result_message({}))

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == "{}"

    def test_result_field_empty_list_reserialized(self) -> None:
        """Empty list in result field is re-serialized to '[]'."""
        # Arrange
        raw_output = _make_json_array(_make_result_message([]))

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == "[]"

    # -----------------------------------------------------------------------
    # 7. Result field is empty string -> returns empty string
    # -----------------------------------------------------------------------

    def test_result_field_empty_string_returns_empty(self) -> None:
        """When result field is an empty string, returns empty string."""
        # Arrange
        raw_output = _make_json_array(
            _make_system_message(),
            _make_result_message(""),
        )

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == ""

    # -----------------------------------------------------------------------
    # 8. Whitespace around JSON array -> handles correctly
    # -----------------------------------------------------------------------

    def test_leading_whitespace_around_json_array(self) -> None:
        """Leading whitespace before JSON array is handled correctly."""
        # Arrange
        result_content = "extracted value"
        inner = _make_json_array(_make_result_message(result_content))
        raw_output = "   \n  " + inner

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == result_content

    def test_trailing_whitespace_around_json_array(self) -> None:
        """Trailing whitespace after JSON array is handled correctly."""
        # Arrange
        result_content = "extracted value"
        inner = _make_json_array(_make_result_message(result_content))
        raw_output = inner + "   \n\t  "

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == result_content

    def test_both_leading_and_trailing_whitespace(self) -> None:
        """Both leading and trailing whitespace around JSON array handled."""
        # Arrange
        result_content = "the answer"
        inner = _make_json_array(_make_result_message(result_content))
        raw_output = "\n\t  " + inner + "  \n\n"

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == result_content

    def test_whitespace_only_input_returns_as_is(self) -> None:
        """Whitespace-only input is returned unchanged (no bracket start)."""
        # Arrange
        raw_output = "   \n\t  "

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == raw_output

    # -----------------------------------------------------------------------
    # 9. Invalid JSON starting with `[` -> returns raw stdout
    # -----------------------------------------------------------------------

    def test_invalid_json_starting_with_bracket_returns_raw(self) -> None:
        """Invalid JSON that starts with '[' returns raw output as fallback."""
        # Arrange
        raw_output = "[this is not valid json at all"

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == raw_output

    def test_truncated_json_array_returns_raw(self) -> None:
        """Truncated JSON array returns raw output as fallback."""
        # Arrange
        raw_output = '[{"type": "result", "result": "incomplete'

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == raw_output

    def test_bracket_in_middle_of_text_returns_raw(self) -> None:
        """Text that does not start with '[' (even if contains one) returns raw."""
        # Arrange
        raw_output = "some text [with brackets] in it"

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == raw_output

    # -----------------------------------------------------------------------
    # 10. Multiple result messages -> uses last one
    # -----------------------------------------------------------------------

    def test_multiple_result_messages_uses_last(self) -> None:
        """When multiple result messages exist, the last one is used."""
        # Arrange
        raw_output = _make_json_array(
            _make_system_message(),
            _make_result_message("first result"),
            _make_assistant_message("continuing..."),
            _make_result_message("second result"),
            _make_result_message("third and final result"),
        )

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == "third and final result"

    def test_two_result_messages_uses_second(self) -> None:
        """With exactly two result messages, the second is used."""
        # Arrange
        raw_output = _make_json_array(
            _make_result_message("early"),
            _make_result_message("later"),
        )

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == "later"

    # -----------------------------------------------------------------------
    # Additional edge cases
    # -----------------------------------------------------------------------

    def test_result_message_with_missing_result_key_returns_empty(self) -> None:
        """A result-type message missing the 'result' key returns empty string."""
        # Arrange
        raw_output = _make_json_array({"type": "result"})

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert -- msg.get("result", "") yields ""
        assert extracted == ""

    def test_result_field_integer_coerced_to_string(self) -> None:
        """When result field is an integer, it is coerced to string via str()."""
        # Arrange
        raw_output = _make_json_array(_make_result_message(42))

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == "42"
        assert isinstance(extracted, str)

    def test_result_field_boolean_coerced_to_string(self) -> None:
        """When result field is a boolean, it is coerced to string via str()."""
        # Arrange
        raw_output = _make_json_array(_make_result_message(True))

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == "True"
        assert isinstance(extracted, str)

    def test_result_field_none_coerced_to_string(self) -> None:
        """When result field is None, it is coerced to string 'None'."""
        # Arrange
        raw_output = _make_json_array(_make_result_message(None))

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == "None"
        assert isinstance(extracted, str)

    def test_non_dict_elements_in_array_are_skipped(self) -> None:
        """Non-dict elements in the array are skipped gracefully."""
        # Arrange
        raw_output = json.dumps(
            ["string_element", 42, None, _make_result_message("found")]
        )

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == "found"

    def test_array_of_primitives_only_returns_raw(self) -> None:
        """Array containing only primitive values (no dicts) returns raw output."""
        # Arrange
        raw_output = json.dumps([1, 2, 3, "hello"])

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == raw_output

    def test_return_type_is_always_str(self) -> None:
        """Return type is always str regardless of input."""
        # Arrange -- multiple input variants
        inputs = [
            "",
            "plain text",
            "[]",
            _make_json_array(_make_result_message("text")),
            _make_json_array(_make_result_message({"key": "val"})),
            _make_json_array(_make_result_message([1, 2])),
        ]

        # Act & Assert
        for raw in inputs:
            result = _extract_result_from_json_output(raw)
            assert isinstance(result, str), (
                f"Expected str, got {type(result)} for input: {raw!r}"
            )

    @pytest.mark.parametrize(
        ("raw_input", "expected"),
        [
            ("plain text", "plain text"),
            ("[]", "[]"),
            ("{}", "{}"),
            ("   ", "   "),
        ],
        ids=["plain_text", "empty_array", "json_object", "whitespace"],
    )
    def test_fallback_cases_parametrized(self, raw_input: str, expected: str) -> None:
        """All fallback cases return raw input unchanged."""
        # Act
        extracted = _extract_result_from_json_output(raw_input)

        # Assert
        assert extracted == expected


# ===========================================================================
# Hypothesis: property-based tests for extraction
# ===========================================================================


@pytest.mark.unit
class TestExtractResultProperties:
    """Property-based tests for _extract_result_from_json_output invariants."""

    @given(content=st.text(min_size=0, max_size=500))
    @settings(max_examples=30, deadline=5000)
    def test_string_result_roundtrips_through_extraction(self, content: str) -> None:
        """Any string placed in a result message is extracted faithfully."""
        # Arrange
        raw_output = _make_json_array(
            _make_system_message(),
            _make_result_message(content),
        )

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == content

    @given(
        text=st.text(min_size=1, max_size=500).filter(
            lambda s: not s.strip().startswith("[")
        )
    )
    @settings(max_examples=30, deadline=5000)
    def test_non_array_input_returned_unchanged(self, text: str) -> None:
        """Any input not starting with '[' (after strip) is returned unchanged."""
        # Act
        result = _extract_result_from_json_output(text)

        # Assert
        assert result == text

    @given(
        num_messages=st.integers(min_value=1, max_value=10),
        final_content=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=20, deadline=5000)
    def test_last_result_message_always_wins(
        self, num_messages: int, final_content: str
    ) -> None:
        """No matter how many result messages, the last one is always extracted."""
        # Arrange
        messages: list[dict[str, Any]] = []
        for i in range(num_messages - 1):
            messages.append(_make_result_message(f"result_{i}"))
        messages.append(_make_result_message(final_content))
        raw_output = json.dumps(messages)

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert extracted == final_content

    @given(
        data=st.dictionaries(
            keys=st.text(
                min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz"
            ),
            values=st.one_of(st.integers(), st.text(max_size=50), st.booleans()),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=20, deadline=5000)
    def test_dict_result_roundtrips_via_json(self, data: dict[str, Any]) -> None:
        """Dict result fields survive serialization and can be parsed back."""
        # Arrange
        raw_output = _make_json_array(_make_result_message(data))

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert isinstance(extracted, str)
        parsed = json.loads(extracted)
        assert parsed == data

    @given(
        items=st.lists(st.integers(), min_size=0, max_size=10),
    )
    @settings(max_examples=15, deadline=5000)
    def test_list_result_roundtrips_via_json(self, items: list[int]) -> None:
        """List result fields survive serialization and can be parsed back."""
        # Arrange
        raw_output = _make_json_array(_make_result_message(items))

        # Act
        extracted = _extract_result_from_json_output(raw_output)

        # Assert
        assert isinstance(extracted, str)
        parsed = json.loads(extracted)
        assert parsed == items

    @given(raw=st.text(min_size=0, max_size=500))
    @settings(max_examples=50, deadline=5000)
    def test_return_type_is_always_string(self, raw: str) -> None:
        """Return type is always str for any input."""
        # Act
        result = _extract_result_from_json_output(raw)

        # Assert
        assert isinstance(result, str)


# ===========================================================================
# Integration tests: send_message() with mocked subprocess
# ===========================================================================


@pytest.mark.unit
class TestSendMessageExtractionIntegration:
    """Integration tests verifying send_message() uses extraction for structured output."""

    # -----------------------------------------------------------------------
    # 11. send_message with structured output active calls extraction
    # -----------------------------------------------------------------------

    async def test_send_message_with_output_schema_extracts_result(self) -> None:
        """send_message() with output_schema extracts result from JSON array response."""
        # Arrange
        client = _make_client()
        result_payload = (
            '{"models": [{"model_name": "xgb", "example_code": "import xgb"}]}'
        )
        cli_response = _make_json_array(
            _make_system_message(),
            _make_assistant_message("Searching..."),
            _make_result_message(result_payload),
        )

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(cli_response.encode("utf-8"), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            # Act
            response = await client.send_message(
                agent_type=AgentType.RETRIEVER,
                message="Find ML models for classification",
                output_schema=RetrieverOutput,
            )

        # Assert -- the extracted payload, not the raw JSON array
        assert response == result_payload
        # Verify it can be parsed by model_validate_json
        parsed = RetrieverOutput.model_validate_json(response)
        assert parsed.models[0].model_name == "xgb"

    async def test_send_message_with_config_output_schema_extracts_result(self) -> None:
        """send_message() uses agent config output_schema when per-call schema is None."""
        # Arrange -- EXTRACTOR has output_schema in default configs
        client = _make_client()
        result_payload = '{"code_blocks": ["block1"]}'
        cli_response = _make_json_array(
            _make_result_message(result_payload),
        )

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(cli_response.encode("utf-8"), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            # Act -- no explicit output_schema, but EXTRACTOR config has one
            response = await client.send_message(
                agent_type=AgentType.EXTRACTOR,
                message="Extract code blocks",
            )

        # Assert -- extracts from JSON array wrapper
        assert response == result_payload

    async def test_send_message_with_dict_result_field_reserialized(self) -> None:
        """send_message() re-serializes dict result field for model_validate_json compatibility."""
        # Arrange
        client = _make_client()
        result_dict = {
            "models": [{"model_name": "rf", "example_code": "from sklearn import"}]
        }
        cli_response = _make_json_array(
            _make_system_message(),
            _make_result_message(result_dict),
        )

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(cli_response.encode("utf-8"), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            response = await client.send_message(
                agent_type=AgentType.RETRIEVER,
                message="Get models",
                output_schema=RetrieverOutput,
            )

        # Assert -- should be a JSON string that model_validate_json can parse
        parsed = RetrieverOutput.model_validate_json(response)
        assert parsed.models[0].model_name == "rf"

    # -----------------------------------------------------------------------
    # 12. send_message without structured output returns raw text
    # -----------------------------------------------------------------------

    async def test_send_message_without_schema_returns_raw_text(self) -> None:
        """send_message() without structured output returns raw text unchanged."""
        # Arrange -- use CODER agent which has no output_schema
        client = _make_client()
        raw_response = "Here is the updated code:\n```python\nprint('hello')\n```"

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(raw_response.encode("utf-8"), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            response = await client.send_message(
                agent_type=AgentType.CODER,
                message="Write some code",
            )

        # Assert -- raw text returned as-is
        assert response == raw_response

    async def test_send_message_without_schema_does_not_parse_json(self) -> None:
        """Without schema, even JSON array output is returned as raw text."""
        # Arrange -- use CODER agent (no output_schema)
        client = _make_client()
        json_like_response = _make_json_array(
            _make_system_message(),
            _make_result_message("should not be extracted"),
        )

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(json_like_response.encode("utf-8"), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            response = await client.send_message(
                agent_type=AgentType.CODER,
                message="Code something",
            )

        # Assert -- raw JSON array returned, not extracted
        assert response == json_like_response

    async def test_send_message_with_use_structured_output_false_returns_raw(
        self,
    ) -> None:
        """use_structured_output=False bypasses extraction even with agent config schema."""
        # Arrange -- RETRIEVER has output_schema, but we disable structured output
        client = _make_client()
        raw_response = "Unstructured retriever output text"

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(raw_response.encode("utf-8"), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            response = await client.send_message(
                agent_type=AgentType.RETRIEVER,
                message="Find models",
                use_structured_output=False,
            )

        # Assert
        assert response == raw_response

    async def test_send_message_raises_on_nonzero_exit(self) -> None:
        """send_message() raises RuntimeError when subprocess exits with non-zero code."""
        # Arrange
        client = _make_client()

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Error: model not found"))

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            pytest.raises(RuntimeError, match="claude -p failed"),
        ):
            await client.send_message(
                agent_type=AgentType.RETRIEVER,
                message="Find models",
            )

    async def test_send_message_extraction_falls_back_on_invalid_json(self) -> None:
        """When CLI returns invalid JSON with schema active, fallback returns raw output."""
        # Arrange
        client = _make_client()
        invalid_response = "[this is not valid json"

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(invalid_response.encode("utf-8"), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            response = await client.send_message(
                agent_type=AgentType.RETRIEVER,
                message="Find models",
                output_schema=RetrieverOutput,
            )

        # Assert -- fallback to raw output
        assert response == invalid_response

    async def test_send_message_extraction_with_whitespace_wrapped_json(self) -> None:
        """send_message handles CLI output with whitespace around JSON array."""
        # Arrange
        client = _make_client()
        result_payload = '{"models": [{"model_name": "svm", "example_code": "from sklearn.svm import SVC"}]}'
        inner = _make_json_array(_make_result_message(result_payload))
        cli_response = "\n  " + inner + "  \n"

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(cli_response.encode("utf-8"), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            response = await client.send_message(
                agent_type=AgentType.RETRIEVER,
                message="Find models",
                output_schema=RetrieverOutput,
            )

        # Assert
        assert response == result_payload
