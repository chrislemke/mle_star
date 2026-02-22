"""Tests for configuration management and environment (Task 48).

Validates ``apply_env_overrides``, ``validate_api_key``, ``configure_logging``,
and ``PipelineState`` defined in ``src/mle_star/orchestrator.py``.

These tests are written TDD-first -- the implementation does not yet exist.
They serve as the executable specification for REQ-OR-046, REQ-OR-047,
and REQ-OR-050.

Refs:
    SRS 09c -- Orchestrator Budgets & Hooks.
    IMPLEMENTATION_PLAN.md Task 48.
"""

from __future__ import annotations

import logging
from typing import Any

from hypothesis import HealthCheck, given, settings, strategies as st
from mle_star.models import PipelineConfig
import pytest

# ---------------------------------------------------------------------------
# Module path constant for patching
# ---------------------------------------------------------------------------

_MODULE = "mle_star.orchestrator"

# ---------------------------------------------------------------------------
# Reusable test helpers
# ---------------------------------------------------------------------------


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


# ===========================================================================
# REQ-OR-046: Environment variable overrides
# ===========================================================================


@pytest.mark.unit
class TestApplyEnvOverrides:
    """apply_env_overrides reads env vars and applies them to config (REQ-OR-046)."""

    def test_no_env_vars_returns_same_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no MLE_STAR_* env vars are set, config is returned unchanged."""
        from mle_star.orchestrator import apply_env_overrides

        # Arrange -- ensure no MLE_STAR env vars exist
        monkeypatch.delenv("MLE_STAR_MODEL", raising=False)
        monkeypatch.delenv("MLE_STAR_LOG_LEVEL", raising=False)
        monkeypatch.delenv("MLE_STAR_MAX_BUDGET", raising=False)
        monkeypatch.delenv("MLE_STAR_TIME_LIMIT", raising=False)

        config = _make_config()

        # Act
        result = apply_env_overrides(config)

        # Assert -- values match the defaults
        assert result.model == config.model
        assert result.log_level == config.log_level
        assert result.max_budget_usd == config.max_budget_usd
        assert result.time_limit_seconds == config.time_limit_seconds

    def test_mle_star_model_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MLE_STAR_MODEL env var overrides the default model value."""
        from mle_star.orchestrator import apply_env_overrides

        # Arrange
        monkeypatch.setenv("MLE_STAR_MODEL", "opus")
        monkeypatch.delenv("MLE_STAR_LOG_LEVEL", raising=False)
        monkeypatch.delenv("MLE_STAR_MAX_BUDGET", raising=False)
        monkeypatch.delenv("MLE_STAR_TIME_LIMIT", raising=False)

        config = _make_config()  # model defaults to "sonnet"

        # Act
        result = apply_env_overrides(config)

        # Assert
        assert result.model == "opus"

    def test_mle_star_log_level_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MLE_STAR_LOG_LEVEL env var overrides the default log level."""
        from mle_star.orchestrator import apply_env_overrides

        # Arrange
        monkeypatch.setenv("MLE_STAR_LOG_LEVEL", "DEBUG")
        monkeypatch.delenv("MLE_STAR_MODEL", raising=False)
        monkeypatch.delenv("MLE_STAR_MAX_BUDGET", raising=False)
        monkeypatch.delenv("MLE_STAR_TIME_LIMIT", raising=False)

        config = _make_config()  # log_level defaults to "INFO"

        # Act
        result = apply_env_overrides(config)

        # Assert
        assert result.log_level == "DEBUG"

    def test_mle_star_max_budget_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MLE_STAR_MAX_BUDGET env var overrides the default max_budget_usd."""
        from mle_star.orchestrator import apply_env_overrides

        # Arrange
        monkeypatch.setenv("MLE_STAR_MAX_BUDGET", "42.5")
        monkeypatch.delenv("MLE_STAR_MODEL", raising=False)
        monkeypatch.delenv("MLE_STAR_LOG_LEVEL", raising=False)
        monkeypatch.delenv("MLE_STAR_TIME_LIMIT", raising=False)

        config = _make_config()  # max_budget_usd defaults to None

        # Act
        result = apply_env_overrides(config)

        # Assert
        assert result.max_budget_usd == 42.5

    def test_mle_star_time_limit_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MLE_STAR_TIME_LIMIT env var overrides the default time_limit_seconds."""
        from mle_star.orchestrator import apply_env_overrides

        # Arrange
        monkeypatch.setenv("MLE_STAR_TIME_LIMIT", "3600")
        monkeypatch.delenv("MLE_STAR_MODEL", raising=False)
        monkeypatch.delenv("MLE_STAR_LOG_LEVEL", raising=False)
        monkeypatch.delenv("MLE_STAR_MAX_BUDGET", raising=False)

        config = _make_config()  # time_limit_seconds defaults to 86400

        # Act
        result = apply_env_overrides(config)

        # Assert
        assert result.time_limit_seconds == 3600

    def test_env_var_does_not_override_explicit_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env vars do not override values explicitly set in PipelineConfig.

        If a field value differs from the default, it was explicitly set by
        the user and should be preserved over the env var.
        """
        from mle_star.orchestrator import apply_env_overrides

        # Arrange -- set env var AND explicit constructor value
        monkeypatch.setenv("MLE_STAR_MODEL", "haiku")
        monkeypatch.delenv("MLE_STAR_LOG_LEVEL", raising=False)
        monkeypatch.delenv("MLE_STAR_MAX_BUDGET", raising=False)
        monkeypatch.delenv("MLE_STAR_TIME_LIMIT", raising=False)

        config = _make_config(model="opus")  # Explicitly set to "opus"

        # Act
        result = apply_env_overrides(config)

        # Assert -- explicit value wins over env var
        assert result.model == "opus"

    def test_multiple_env_vars_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multiple env vars are applied simultaneously."""
        from mle_star.orchestrator import apply_env_overrides

        # Arrange
        monkeypatch.setenv("MLE_STAR_MODEL", "haiku")
        monkeypatch.setenv("MLE_STAR_LOG_LEVEL", "WARNING")
        monkeypatch.setenv("MLE_STAR_MAX_BUDGET", "100.0")
        monkeypatch.setenv("MLE_STAR_TIME_LIMIT", "7200")

        config = _make_config()  # All defaults

        # Act
        result = apply_env_overrides(config)

        # Assert
        assert result.model == "haiku"
        assert result.log_level == "WARNING"
        assert result.max_budget_usd == 100.0
        assert result.time_limit_seconds == 7200

    def test_invalid_max_budget_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-numeric MLE_STAR_MAX_BUDGET is silently ignored."""
        from mle_star.orchestrator import apply_env_overrides

        # Arrange
        monkeypatch.setenv("MLE_STAR_MAX_BUDGET", "not_a_number")
        monkeypatch.delenv("MLE_STAR_MODEL", raising=False)
        monkeypatch.delenv("MLE_STAR_LOG_LEVEL", raising=False)
        monkeypatch.delenv("MLE_STAR_TIME_LIMIT", raising=False)

        config = _make_config()

        # Act
        result = apply_env_overrides(config)

        # Assert -- default remains
        assert result.max_budget_usd is None

    def test_invalid_time_limit_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-numeric MLE_STAR_TIME_LIMIT is silently ignored."""
        from mle_star.orchestrator import apply_env_overrides

        # Arrange
        monkeypatch.setenv("MLE_STAR_TIME_LIMIT", "fast")
        monkeypatch.delenv("MLE_STAR_MODEL", raising=False)
        monkeypatch.delenv("MLE_STAR_LOG_LEVEL", raising=False)
        monkeypatch.delenv("MLE_STAR_MAX_BUDGET", raising=False)

        config = _make_config()

        # Act
        result = apply_env_overrides(config)

        # Assert -- default remains
        assert result.time_limit_seconds == 86400

    def test_returns_new_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """apply_env_overrides returns a new PipelineConfig (config is frozen)."""
        from mle_star.orchestrator import apply_env_overrides

        # Arrange
        monkeypatch.setenv("MLE_STAR_MODEL", "opus")
        monkeypatch.delenv("MLE_STAR_LOG_LEVEL", raising=False)
        monkeypatch.delenv("MLE_STAR_MAX_BUDGET", raising=False)
        monkeypatch.delenv("MLE_STAR_TIME_LIMIT", raising=False)

        config = _make_config()

        # Act
        result = apply_env_overrides(config)

        # Assert -- must be a different object since PipelineConfig is frozen
        assert result is not config
        assert isinstance(result, PipelineConfig)

    def test_non_overridden_fields_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fields not targeted by env vars are preserved in the returned config."""
        from mle_star.orchestrator import apply_env_overrides

        # Arrange
        monkeypatch.setenv("MLE_STAR_MODEL", "opus")
        monkeypatch.delenv("MLE_STAR_LOG_LEVEL", raising=False)
        monkeypatch.delenv("MLE_STAR_MAX_BUDGET", raising=False)
        monkeypatch.delenv("MLE_STAR_TIME_LIMIT", raising=False)

        config = _make_config(
            num_retrieved_models=8,
            outer_loop_steps=6,
            inner_loop_steps=5,
        )

        # Act
        result = apply_env_overrides(config)

        # Assert -- non-env-var fields carried over
        assert result.num_retrieved_models == 8
        assert result.outer_loop_steps == 6
        assert result.inner_loop_steps == 5

    def test_env_var_does_not_override_explicit_log_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MLE_STAR_LOG_LEVEL does not override explicitly set log_level."""
        from mle_star.orchestrator import apply_env_overrides

        # Arrange
        monkeypatch.setenv("MLE_STAR_LOG_LEVEL", "ERROR")
        monkeypatch.delenv("MLE_STAR_MODEL", raising=False)
        monkeypatch.delenv("MLE_STAR_MAX_BUDGET", raising=False)
        monkeypatch.delenv("MLE_STAR_TIME_LIMIT", raising=False)

        config = _make_config(log_level="DEBUG")  # Explicitly set

        # Act
        result = apply_env_overrides(config)

        # Assert
        assert result.log_level == "DEBUG"

    def test_env_var_does_not_override_explicit_max_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MLE_STAR_MAX_BUDGET does not override explicitly set max_budget_usd."""
        from mle_star.orchestrator import apply_env_overrides

        # Arrange
        monkeypatch.setenv("MLE_STAR_MAX_BUDGET", "999.0")
        monkeypatch.delenv("MLE_STAR_MODEL", raising=False)
        monkeypatch.delenv("MLE_STAR_LOG_LEVEL", raising=False)
        monkeypatch.delenv("MLE_STAR_TIME_LIMIT", raising=False)

        config = _make_config(max_budget_usd=50.0)  # Explicitly set

        # Act
        result = apply_env_overrides(config)

        # Assert
        assert result.max_budget_usd == 50.0

    def test_env_var_does_not_override_explicit_time_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MLE_STAR_TIME_LIMIT does not override explicitly set time_limit_seconds."""
        from mle_star.orchestrator import apply_env_overrides

        # Arrange
        monkeypatch.setenv("MLE_STAR_TIME_LIMIT", "100")
        monkeypatch.delenv("MLE_STAR_MODEL", raising=False)
        monkeypatch.delenv("MLE_STAR_LOG_LEVEL", raising=False)
        monkeypatch.delenv("MLE_STAR_MAX_BUDGET", raising=False)

        config = _make_config(time_limit_seconds=43200)  # Explicitly set

        # Act
        result = apply_env_overrides(config)

        # Assert
        assert result.time_limit_seconds == 43200


# ===========================================================================
# REQ-OR-046: API key validation
# ===========================================================================


@pytest.mark.unit
class TestValidateApiKey:
    """validate_api_key checks for ANTHROPIC_API_KEY env var (REQ-OR-046)."""

    def test_raises_when_api_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EnvironmentError raised when ANTHROPIC_API_KEY is not set."""
        from mle_star.orchestrator import validate_api_key

        # Arrange
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        # Act & Assert
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            validate_api_key()

    def test_passes_when_api_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No error when ANTHROPIC_API_KEY is set to a non-empty value."""
        from mle_star.orchestrator import validate_api_key

        # Arrange
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-12345")

        # Act & Assert -- should not raise
        validate_api_key()

    def test_raises_when_api_key_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EnvironmentError raised when ANTHROPIC_API_KEY is set but empty."""
        from mle_star.orchestrator import validate_api_key

        # Arrange
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

        # Act & Assert
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            validate_api_key()

    def test_raises_when_api_key_whitespace_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """EnvironmentError raised when ANTHROPIC_API_KEY is whitespace only."""
        from mle_star.orchestrator import validate_api_key

        # Arrange
        monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")

        # Act & Assert
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            validate_api_key()


# ===========================================================================
# REQ-OR-047: Logging configuration
# ===========================================================================


@pytest.mark.unit
class TestConfigureLogging:
    """configure_logging sets up Python logging for the pipeline (REQ-OR-047)."""

    def test_sets_log_level_info_by_default(self) -> None:
        """Default log level is INFO when config.log_level is 'INFO'."""
        from mle_star.orchestrator import configure_logging

        # Arrange
        config = _make_config()  # log_level defaults to "INFO"

        # Act
        configure_logging(config)

        # Assert
        mle_logger = logging.getLogger("mle_star")
        assert mle_logger.level == logging.INFO

    def test_sets_log_level_debug(self) -> None:
        """Log level is set to DEBUG when config.log_level is 'DEBUG'."""
        from mle_star.orchestrator import configure_logging

        # Arrange
        config = _make_config(log_level="DEBUG")

        # Act
        configure_logging(config)

        # Assert
        mle_logger = logging.getLogger("mle_star")
        assert mle_logger.level == logging.DEBUG

    def test_adds_file_handler_when_log_file_set(self, tmp_path: Any) -> None:
        """A FileHandler is added to the logger when config.log_file is set."""
        from mle_star.orchestrator import configure_logging

        # Arrange
        log_file = str(tmp_path / "pipeline.log")
        config = _make_config(log_file=log_file)

        # Clean up any previous handlers on the logger
        mle_logger = logging.getLogger("mle_star")
        original_handlers = list(mle_logger.handlers)

        # Act
        configure_logging(config)

        # Assert -- at least one FileHandler should be present
        file_handlers = [
            h for h in mle_logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) >= 1

        # Cleanup -- remove handlers added by this test
        for h in mle_logger.handlers[:]:
            if h not in original_handlers:
                mle_logger.removeHandler(h)
                if isinstance(h, logging.FileHandler):
                    h.close()

    def test_no_file_handler_when_log_file_none(self) -> None:
        """No FileHandler is added when config.log_file is None."""
        from mle_star.orchestrator import configure_logging

        # Arrange
        config = _make_config()  # log_file defaults to None

        # Clean up any previous handlers
        mle_logger = logging.getLogger("mle_star")
        for h in mle_logger.handlers[:]:
            if isinstance(h, logging.FileHandler):
                mle_logger.removeHandler(h)
                h.close()

        # Act
        configure_logging(config)

        # Assert
        file_handlers = [
            h for h in mle_logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 0

    def test_idempotent_multiple_calls(self) -> None:
        """Calling configure_logging multiple times does not duplicate handlers."""
        from mle_star.orchestrator import configure_logging

        # Arrange
        config = _make_config()

        # Clean up any previous handlers
        mle_logger = logging.getLogger("mle_star")
        for h in mle_logger.handlers[:]:
            mle_logger.removeHandler(h)
            if isinstance(h, logging.FileHandler):
                h.close()

        # Act -- call twice
        configure_logging(config)
        handler_count_first = len(mle_logger.handlers)
        configure_logging(config)
        handler_count_second = len(mle_logger.handlers)

        # Assert -- handler count should not increase
        assert handler_count_second == handler_count_first

    def test_console_handler_has_formatter(self) -> None:
        """Console handler has a formatter with timestamp, level, and logger name."""
        from mle_star.orchestrator import configure_logging

        # Arrange
        config = _make_config()

        # Clean up any previous handlers
        mle_logger = logging.getLogger("mle_star")
        for h in mle_logger.handlers[:]:
            mle_logger.removeHandler(h)
            if isinstance(h, logging.FileHandler):
                h.close()

        # Act
        configure_logging(config)

        # Assert -- at least one StreamHandler with a formatter
        stream_handlers = [
            h for h in mle_logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) >= 1
        formatter = stream_handlers[0].formatter
        assert formatter is not None
        fmt_str = formatter._fmt or ""
        # Should contain structured output elements: time, level, name, message
        # Check that the format string references some common fields
        assert "%(message)s" in fmt_str or "message" in fmt_str.lower()

    @pytest.mark.parametrize(
        ("log_level_str", "expected_level"),
        [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
        ],
    )
    def test_parametrized_log_levels(
        self,
        log_level_str: str,
        expected_level: int,
    ) -> None:
        """configure_logging correctly maps string log levels to logging constants."""
        from mle_star.orchestrator import configure_logging

        # Arrange
        config = _make_config(log_level=log_level_str)

        # Act
        configure_logging(config)

        # Assert
        mle_logger = logging.getLogger("mle_star")
        assert mle_logger.level == expected_level


# ===========================================================================
# REQ-OR-050: PipelineState model
# ===========================================================================


@pytest.mark.unit
class TestPipelineState:
    """PipelineState is a mutable Pydantic model for runtime state (REQ-OR-050)."""

    def test_default_values(self) -> None:
        """PipelineState has expected defaults for all fields."""
        from mle_star.orchestrator import PipelineState

        # Act
        state = PipelineState()

        # Assert
        assert state.current_phase == "phase1"
        assert state.elapsed_seconds == 0.0
        assert state.accumulated_cost_usd == 0.0
        assert state.phase2_path_statuses == []
        assert state.best_score_so_far is None
        assert state.agent_call_count == 0

    def test_update_current_phase(self) -> None:
        """current_phase can be updated to track pipeline progression."""
        from mle_star.orchestrator import PipelineState

        # Arrange
        state = PipelineState()

        # Act
        state.current_phase = "phase2"

        # Assert
        assert state.current_phase == "phase2"

    def test_update_elapsed_seconds(self) -> None:
        """elapsed_seconds can be updated with wall-clock time."""
        from mle_star.orchestrator import PipelineState

        # Arrange
        state = PipelineState()

        # Act
        state.elapsed_seconds = 123.456

        # Assert
        assert state.elapsed_seconds == 123.456

    def test_update_accumulated_cost(self) -> None:
        """accumulated_cost_usd can be updated with running cost."""
        from mle_star.orchestrator import PipelineState

        # Arrange
        state = PipelineState()

        # Act
        state.accumulated_cost_usd = 5.75

        # Assert
        assert state.accumulated_cost_usd == 5.75

    def test_update_phase2_path_statuses(self) -> None:
        """phase2_path_statuses can be updated with per-path status strings."""
        from mle_star.orchestrator import PipelineState

        # Arrange
        state = PipelineState()

        # Act
        state.phase2_path_statuses = ["running", "completed", "failed", "cancelled"]

        # Assert
        assert state.phase2_path_statuses == [
            "running",
            "completed",
            "failed",
            "cancelled",
        ]

    def test_update_best_score(self) -> None:
        """best_score_so_far can be updated from None to a value."""
        from mle_star.orchestrator import PipelineState

        # Arrange
        state = PipelineState()
        assert state.best_score_so_far is None

        # Act
        state.best_score_so_far = 0.92

        # Assert
        assert state.best_score_so_far == 0.92

    def test_update_agent_call_count(self) -> None:
        """agent_call_count can be incremented."""
        from mle_star.orchestrator import PipelineState

        # Arrange
        state = PipelineState()

        # Act
        state.agent_call_count = 42

        # Assert
        assert state.agent_call_count == 42

    def test_is_pydantic_base_model(self) -> None:
        """PipelineState is a Pydantic BaseModel."""
        from mle_star.orchestrator import PipelineState
        from pydantic import BaseModel

        assert issubclass(PipelineState, BaseModel)

    def test_is_mutable(self) -> None:
        """PipelineState is NOT frozen -- fields can be mutated at runtime."""
        from mle_star.orchestrator import PipelineState

        # Arrange
        state = PipelineState()

        # Act & Assert -- should NOT raise ValidationError
        state.current_phase = "finalization"
        state.elapsed_seconds = 999.0
        state.accumulated_cost_usd = 10.0
        state.phase2_path_statuses = ["completed"]
        state.best_score_so_far = 0.99
        state.agent_call_count = 100

    def test_complete_phase_transition(self) -> None:
        """PipelineState supports transitioning through all pipeline phases."""
        from mle_star.orchestrator import PipelineState

        # Arrange
        state = PipelineState()

        # Act -- walk through all phases
        phases = ["phase1", "phase2", "phase3", "finalization", "complete"]
        for phase in phases:
            state.current_phase = phase

        # Assert -- final state
        assert state.current_phase == "complete"

    def test_path_statuses_independent_of_current_phase(self) -> None:
        """phase2_path_statuses can be updated regardless of current_phase."""
        from mle_star.orchestrator import PipelineState

        # Arrange
        state = PipelineState()
        state.current_phase = "phase3"

        # Act
        state.phase2_path_statuses = ["completed", "completed"]

        # Assert
        assert state.phase2_path_statuses == ["completed", "completed"]
        assert state.current_phase == "phase3"

    def test_agent_call_count_is_int(self) -> None:
        """agent_call_count is typed as int."""
        from mle_star.orchestrator import PipelineState

        # Arrange
        state = PipelineState()

        # Assert
        assert isinstance(state.agent_call_count, int)

    def test_elapsed_seconds_is_float(self) -> None:
        """elapsed_seconds is typed as float."""
        from mle_star.orchestrator import PipelineState

        # Arrange
        state = PipelineState()

        # Assert
        assert isinstance(state.elapsed_seconds, float)


# ===========================================================================
# Hypothesis: property-based tests for configuration
# ===========================================================================


@pytest.mark.unit
class TestConfigOverrideProperties:
    """Property-based tests for apply_env_overrides behavior."""

    @given(
        model_name=st.sampled_from(["sonnet", "opus", "haiku"]),
    )
    @settings(
        max_examples=10,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_env_model_applied_for_default_config(
        self,
        model_name: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """For any valid model name, env var overrides default config."""
        from mle_star.orchestrator import apply_env_overrides

        monkeypatch.setenv("MLE_STAR_MODEL", model_name)
        monkeypatch.delenv("MLE_STAR_LOG_LEVEL", raising=False)
        monkeypatch.delenv("MLE_STAR_MAX_BUDGET", raising=False)
        monkeypatch.delenv("MLE_STAR_TIME_LIMIT", raising=False)

        config = _make_config()
        result = apply_env_overrides(config)

        assert result.model == model_name

    @given(
        time_limit=st.integers(min_value=1, max_value=604800),
    )
    @settings(
        max_examples=10,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_env_time_limit_applied_for_default_config(
        self,
        time_limit: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """For any valid time limit, env var overrides default config."""
        from mle_star.orchestrator import apply_env_overrides

        monkeypatch.setenv("MLE_STAR_TIME_LIMIT", str(time_limit))
        monkeypatch.delenv("MLE_STAR_MODEL", raising=False)
        monkeypatch.delenv("MLE_STAR_LOG_LEVEL", raising=False)
        monkeypatch.delenv("MLE_STAR_MAX_BUDGET", raising=False)

        config = _make_config()
        result = apply_env_overrides(config)

        assert result.time_limit_seconds == time_limit

    @given(
        budget=st.floats(
            min_value=0.01,
            max_value=10000.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(
        max_examples=10,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_env_budget_applied_for_default_config(
        self,
        budget: float,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """For any valid budget float, env var overrides default config."""
        from mle_star.orchestrator import apply_env_overrides

        monkeypatch.setenv("MLE_STAR_MAX_BUDGET", str(budget))
        monkeypatch.delenv("MLE_STAR_MODEL", raising=False)
        monkeypatch.delenv("MLE_STAR_LOG_LEVEL", raising=False)
        monkeypatch.delenv("MLE_STAR_TIME_LIMIT", raising=False)

        config = _make_config()
        result = apply_env_overrides(config)

        assert result.max_budget_usd is not None
        assert abs(result.max_budget_usd - budget) < 1e-6

    @given(
        explicit_model=st.sampled_from(["opus", "haiku"]),
        env_model=st.sampled_from(["sonnet", "opus", "haiku"]),
    )
    @settings(
        max_examples=10,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_explicit_always_wins_over_env(
        self,
        explicit_model: str,
        env_model: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit config values always take precedence over env vars.

        Property: for any explicit model that differs from the default,
        setting an env var must not change it.
        """
        from mle_star.orchestrator import apply_env_overrides

        monkeypatch.setenv("MLE_STAR_MODEL", env_model)
        monkeypatch.delenv("MLE_STAR_LOG_LEVEL", raising=False)
        monkeypatch.delenv("MLE_STAR_MAX_BUDGET", raising=False)
        monkeypatch.delenv("MLE_STAR_TIME_LIMIT", raising=False)

        # "opus" and "haiku" both differ from default "sonnet"
        config = _make_config(model=explicit_model)
        result = apply_env_overrides(config)

        assert result.model == explicit_model

    @given(
        log_level=st.sampled_from(["DEBUG", "WARNING", "ERROR", "CRITICAL"]),
    )
    @settings(
        max_examples=10,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_env_log_level_applied_for_default_config(
        self,
        log_level: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """For any valid log level, env var overrides default config."""
        from mle_star.orchestrator import apply_env_overrides

        monkeypatch.setenv("MLE_STAR_LOG_LEVEL", log_level)
        monkeypatch.delenv("MLE_STAR_MODEL", raising=False)
        monkeypatch.delenv("MLE_STAR_MAX_BUDGET", raising=False)
        monkeypatch.delenv("MLE_STAR_TIME_LIMIT", raising=False)

        config = _make_config()
        result = apply_env_overrides(config)

        assert result.log_level == log_level


# ===========================================================================
# Hypothesis: property-based tests for PipelineState
# ===========================================================================


@pytest.mark.unit
class TestPipelineStateProperties:
    """Property-based tests for PipelineState mutability and invariants."""

    @given(
        phase=st.sampled_from(
            ["phase1", "phase2", "phase3", "finalization", "complete"]
        ),
        elapsed=st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
        cost=st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
        call_count=st.integers(min_value=0, max_value=100000),
    )
    @settings(max_examples=20, deadline=5000)
    def test_all_fields_independently_settable(
        self,
        phase: str,
        elapsed: float,
        cost: float,
        call_count: int,
    ) -> None:
        """All PipelineState fields can be set to any valid value."""
        from mle_star.orchestrator import PipelineState

        state = PipelineState()

        state.current_phase = phase
        state.elapsed_seconds = elapsed
        state.accumulated_cost_usd = cost
        state.agent_call_count = call_count

        assert state.current_phase == phase
        assert state.elapsed_seconds == elapsed
        assert state.accumulated_cost_usd == cost
        assert state.agent_call_count == call_count

    @given(
        num_paths=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=10, deadline=5000)
    def test_path_statuses_supports_variable_length(
        self,
        num_paths: int,
    ) -> None:
        """phase2_path_statuses supports variable-length lists."""
        from mle_star.orchestrator import PipelineState

        state = PipelineState()
        statuses = ["running"] * num_paths
        state.phase2_path_statuses = statuses

        assert len(state.phase2_path_statuses) == num_paths

    @given(
        score=st.one_of(
            st.none(),
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
        ),
    )
    @settings(max_examples=15, deadline=5000)
    def test_best_score_accepts_none_and_float(
        self,
        score: float | None,
    ) -> None:
        """best_score_so_far accepts both None and float values."""
        from mle_star.orchestrator import PipelineState

        state = PipelineState()
        state.best_score_so_far = score

        assert state.best_score_so_far == score


# ===========================================================================
# Integration: apply_env_overrides returns valid PipelineConfig
# ===========================================================================


@pytest.mark.unit
class TestApplyEnvOverridesIntegration:
    """Integration tests verifying apply_env_overrides produces valid configs."""

    def test_result_passes_pydantic_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returned config passes all PipelineConfig validators."""
        from mle_star.orchestrator import apply_env_overrides

        # Arrange
        monkeypatch.setenv("MLE_STAR_MODEL", "opus")
        monkeypatch.setenv("MLE_STAR_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("MLE_STAR_MAX_BUDGET", "75.0")
        monkeypatch.setenv("MLE_STAR_TIME_LIMIT", "3600")

        config = _make_config()

        # Act
        result = apply_env_overrides(config)

        # Assert -- construct a new config with the same values to verify validity
        validated = PipelineConfig(
            model=result.model,
            log_level=result.log_level,
            max_budget_usd=result.max_budget_usd,
            time_limit_seconds=result.time_limit_seconds,
        )
        assert validated.model == "opus"
        assert validated.log_level == "DEBUG"
        assert validated.max_budget_usd == 75.0
        assert validated.time_limit_seconds == 3600

    def test_env_time_limit_zero_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MLE_STAR_TIME_LIMIT='0' would violate validation, so should be ignored.

        PipelineConfig requires time_limit_seconds >= 1.
        """
        from mle_star.orchestrator import apply_env_overrides

        # Arrange
        monkeypatch.setenv("MLE_STAR_TIME_LIMIT", "0")
        monkeypatch.delenv("MLE_STAR_MODEL", raising=False)
        monkeypatch.delenv("MLE_STAR_LOG_LEVEL", raising=False)
        monkeypatch.delenv("MLE_STAR_MAX_BUDGET", raising=False)

        config = _make_config()

        # Act
        result = apply_env_overrides(config)

        # Assert -- the result should still have a valid time_limit_seconds >= 1
        assert result.time_limit_seconds >= 1
