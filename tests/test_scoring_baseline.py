"""Tests for beats_baseline scoring function.

Validates the ``beats_baseline`` function defined in
``src/mle_star/scoring.py``. Tests cover the None-baseline bypass,
maximize/minimize direction behavior, and consistency with
``is_improvement``.
"""

from __future__ import annotations

from hypothesis import given, settings, strategies as st
from mle_star.models import MetricDirection
from mle_star.scoring import beats_baseline, is_improvement
import pytest


# ===========================================================================
# beats_baseline -- None baseline always passes
# ===========================================================================


@pytest.mark.unit
class TestBeatsBaselineNone:
    """beats_baseline returns True when baseline_value is None."""

    def test_none_baseline_returns_true_maximize(self) -> None:
        """No baseline configured with maximize direction returns True."""
        assert beats_baseline(0.5, None, MetricDirection.MAXIMIZE) is True

    def test_none_baseline_returns_true_minimize(self) -> None:
        """No baseline configured with minimize direction returns True."""
        assert beats_baseline(0.5, None, MetricDirection.MINIMIZE) is True

    def test_none_baseline_any_score_returns_true(self) -> None:
        """Any score passes when baseline is None."""
        assert beats_baseline(-999.0, None, MetricDirection.MAXIMIZE) is True
        assert beats_baseline(0.0, None, MetricDirection.MINIMIZE) is True

    def test_returns_bool(self) -> None:
        """beats_baseline returns a bool type."""
        result = beats_baseline(0.5, None, MetricDirection.MAXIMIZE)
        assert isinstance(result, bool)


# ===========================================================================
# beats_baseline -- Maximize direction
# ===========================================================================


@pytest.mark.unit
class TestBeatsBaselineMaximize:
    """beats_baseline with maximize direction delegates to is_improvement."""

    def test_higher_score_beats_baseline(self) -> None:
        """Score above baseline passes."""
        assert beats_baseline(0.9, 0.8, MetricDirection.MAXIMIZE) is True

    def test_lower_score_does_not_beat_baseline(self) -> None:
        """Score below baseline fails."""
        assert beats_baseline(0.7, 0.8, MetricDirection.MAXIMIZE) is False

    def test_equal_score_does_not_beat_baseline(self) -> None:
        """Score equal to baseline fails (strict comparison)."""
        assert beats_baseline(0.8, 0.8, MetricDirection.MAXIMIZE) is False

    def test_barely_above_beats(self) -> None:
        """Even a tiny improvement counts."""
        assert beats_baseline(0.80001, 0.8, MetricDirection.MAXIMIZE) is True

    def test_barely_below_does_not_beat(self) -> None:
        """Even a tiny shortfall fails."""
        assert beats_baseline(0.79999, 0.8, MetricDirection.MAXIMIZE) is False


# ===========================================================================
# beats_baseline -- Minimize direction
# ===========================================================================


@pytest.mark.unit
class TestBeatsBaselineMinimize:
    """beats_baseline with minimize direction delegates to is_improvement."""

    def test_lower_score_beats_baseline(self) -> None:
        """Score below baseline passes (lower is better)."""
        assert beats_baseline(0.7, 0.8, MetricDirection.MINIMIZE) is True

    def test_higher_score_does_not_beat_baseline(self) -> None:
        """Score above baseline fails."""
        assert beats_baseline(0.9, 0.8, MetricDirection.MINIMIZE) is False

    def test_equal_score_does_not_beat_baseline(self) -> None:
        """Score equal to baseline fails (strict comparison)."""
        assert beats_baseline(0.8, 0.8, MetricDirection.MINIMIZE) is False

    def test_barely_below_beats(self) -> None:
        """Even a tiny improvement counts."""
        assert beats_baseline(0.79999, 0.8, MetricDirection.MINIMIZE) is True


# ===========================================================================
# beats_baseline -- Consistency with is_improvement
# ===========================================================================


@pytest.mark.unit
class TestBeatsBaselineConsistency:
    """beats_baseline delegates to is_improvement when baseline is not None."""

    @pytest.mark.parametrize(
        "direction", [MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]
    )
    def test_matches_is_improvement(self, direction: MetricDirection) -> None:
        """beats_baseline(s, b, d) == is_improvement(s, b, d) when b is not None."""
        for score, baseline in [(0.9, 0.8), (0.7, 0.8), (0.8, 0.8)]:
            expected = is_improvement(score, baseline, direction)
            assert beats_baseline(score, baseline, direction) == expected


# ===========================================================================
# beats_baseline -- Property-based
# ===========================================================================


@pytest.mark.unit
class TestBeatsBaselinePropertyBased:
    """Property-based tests for beats_baseline."""

    @given(
        score=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        direction=st.sampled_from([MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]),
    )
    @settings(max_examples=50)
    def test_none_baseline_always_true(
        self, score: float, direction: MetricDirection
    ) -> None:
        """Property: beats_baseline(score, None, direction) is always True."""
        assert beats_baseline(score, None, direction) is True

    @given(
        score=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        baseline=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        direction=st.sampled_from([MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]),
    )
    @settings(max_examples=50)
    def test_matches_is_improvement_property(
        self, score: float, baseline: float, direction: MetricDirection
    ) -> None:
        """Property: beats_baseline equals is_improvement when baseline is not None."""
        assert beats_baseline(score, baseline, direction) == is_improvement(
            score, baseline, direction
        )

    @given(
        x=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        direction=st.sampled_from([MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]),
    )
    @settings(max_examples=30)
    def test_equal_score_never_beats(
        self, x: float, direction: MetricDirection
    ) -> None:
        """Property: a score equal to the baseline never beats it."""
        assert beats_baseline(x, x, direction) is False

    @given(
        score=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        baseline=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=30)
    def test_returns_bool_type(self, score: float, baseline: float) -> None:
        """Property: beats_baseline always returns a bool."""
        result = beats_baseline(score, baseline, MetricDirection.MAXIMIZE)
        assert isinstance(result, bool)
