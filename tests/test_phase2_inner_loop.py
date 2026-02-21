"""Tests for the Phase 2 inner loop orchestration function (Task 24).

Validates ``run_phase2_inner_loop`` which implements the K-iteration inner
loop of Algorithm 2 for Phase 2 targeted code block refinement.  Each
iteration invokes the coder (and planner for k>=1), replaces the code
block in the original solution, evaluates, and tracks the best score.

Tests are written TDD-first and serve as the executable specification for
REQ-P2I-016 through REQ-P2I-029.

Refs:
    SRS 02b (Phase 2 Inner Loop), IMPLEMENTATION_PLAN.md Task 24.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    CodeBlock,
    CodeBlockCategory,
    DataModality,
    EvaluationResult,
    InnerLoopResult,
    MetricDirection,
    PipelineConfig,
    RefinementAttempt,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
    TaskType,
)
import pytest

# ---------------------------------------------------------------------------
# Module path constant for patching
# ---------------------------------------------------------------------------

_MODULE = "mle_star.phase2_inner"


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_task(
    direction: MetricDirection = MetricDirection.MAXIMIZE,
) -> TaskDescription:
    """Create a minimal TaskDescription for testing."""
    return TaskDescription(
        competition_id="test-comp",
        task_type=TaskType.CLASSIFICATION,
        data_modality=DataModality.TABULAR,
        evaluation_metric="accuracy",
        metric_direction=direction,
        description="Test task",
    )


def _make_config(inner_loop_steps: int = 4) -> PipelineConfig:
    """Create a PipelineConfig with a specified K value."""
    return PipelineConfig(inner_loop_steps=inner_loop_steps)


def _make_solution(
    content: str = "original code with TARGET_BLOCK here",
) -> SolutionScript:
    """Create a mutable SolutionScript for testing."""
    return SolutionScript(content=content, phase=SolutionPhase.REFINED)


def _make_code_block(content: str = "TARGET_BLOCK") -> CodeBlock:
    """Create a CodeBlock targeting a known substring."""
    return CodeBlock(content=content, category=CodeBlockCategory.TRAINING)


def _make_eval_result(score: float | None = 0.85) -> EvaluationResult:
    """Create an EvaluationResult with the given score."""
    return EvaluationResult(
        score=score,
        stdout="Final Validation Performance: 0.85",
        stderr="",
        exit_code=0,
        duration_seconds=1.0,
        is_error=False,
    )


# ===========================================================================
# REQ-P2I-016: run_phase2_inner_loop is async and returns InnerLoopResult
# ===========================================================================


@pytest.mark.unit
class TestRunPhase2InnerLoopIsAsync:
    """run_phase2_inner_loop is an async function returning InnerLoopResult (REQ-P2I-016)."""

    def test_is_coroutine_function(self) -> None:
        """run_phase2_inner_loop is defined as an async function."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        assert asyncio.iscoroutinefunction(run_phase2_inner_loop)

    async def test_returns_inner_loop_result(self) -> None:
        """run_phase2_inner_loop returns an InnerLoopResult instance."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved code",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(
                f"{_MODULE}.is_improvement_or_equal",
                return_value=True,
            ),
            patch(
                f"{_MODULE}.is_improvement",
                return_value=True,
            ),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert isinstance(result, InnerLoopResult)


# ===========================================================================
# REQ-P2I-017/018: k=0 uses initial_plan, no planner call
# ===========================================================================


@pytest.mark.unit
class TestK0UsesInitialPlan:
    """At k=0, initial_plan is used directly without calling A_planner (REQ-P2I-017/018)."""

    async def test_k0_does_not_call_planner(self) -> None:
        """invoke_planner is NOT called at k=0."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ) as mock_coder,
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ) as mock_planner,
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="my initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        mock_planner.assert_not_called()
        mock_coder.assert_called_once()

    async def test_k0_passes_initial_plan_to_coder(self) -> None:
        """invoke_coder receives initial_plan as the plan argument at k=0."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ) as mock_coder,
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="my initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # Verify coder was called with: code_block.content, initial_plan, client
        mock_coder.assert_called_once_with(
            code_block.content, "my initial plan", client
        )

    async def test_attempts_0_plan_is_initial_plan(self) -> None:
        """attempts[0].plan equals initial_plan (REQ-P2I-029)."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)
        initial_plan = "refine the training loop to use cosine annealing"

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(f"{_MODULE}.invoke_planner", new_callable=AsyncMock),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan=initial_plan,
                best_score=0.80,
                task=task,
                config=config,
            )

        assert result.attempts[0].plan == initial_plan


# ===========================================================================
# REQ-P2I-019: k>=1 calls planner, then coder
# ===========================================================================


@pytest.mark.unit
class TestKGe1CallsPlanner:
    """For k=1..K-1: invoke_planner then invoke_coder are called (REQ-P2I-019)."""

    async def test_planner_called_k_minus_1_times(self) -> None:
        """invoke_planner is called exactly K-1 times (once per k>=1 iteration)."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=4)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="new plan",
            ) as mock_planner,
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # K=4, so planner called at k=1, k=2, k=3 => 3 times
        assert mock_planner.call_count == 3

    async def test_coder_called_k_times(self) -> None:
        """invoke_coder is called exactly K times (once per iteration)."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=4)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ) as mock_coder,
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="new plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert mock_coder.call_count == 4


# ===========================================================================
# REQ-P2I-020: Planner receives full history of ALL previous plans/scores
# ===========================================================================


@pytest.mark.unit
class TestPlannerReceivesFullHistory:
    """A_planner receives accumulated plans and scores from ALL prior iterations (REQ-P2I-020)."""

    async def test_planner_at_k2_gets_two_plans_and_scores(self) -> None:
        """At k=2, planner receives plans=[p0, p1] and scores=[s0, s1]."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=3)

        planner_calls: list[tuple[Any, ...]] = []

        async def capture_planner(*args: Any, **kwargs: Any) -> str:
            planner_calls.append(args)
            return f"plan_k{len(planner_calls)}"

        eval_scores = [0.85, 0.87, 0.90]
        eval_index = 0

        async def mock_eval(
            *args: Any, **kwargs: Any
        ) -> tuple[SolutionScript, EvaluationResult]:
            nonlocal eval_index
            score = eval_scores[eval_index]
            eval_index += 1
            return (_make_solution(), _make_eval_result(score))

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(f"{_MODULE}.invoke_planner", side_effect=capture_planner),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(f"{_MODULE}.evaluate_with_retry", side_effect=mock_eval),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # Planner called at k=1 and k=2
        assert len(planner_calls) == 2

        # At k=1: planner gets [initial_plan] and [0.85]
        k1_plans = planner_calls[0][1]  # plans argument
        k1_scores = planner_calls[0][2]  # scores argument
        assert k1_plans == ["initial plan"]
        assert k1_scores == [0.85]

        # At k=2: planner gets [initial_plan, plan_k1] and [0.85, 0.87]
        k2_plans = planner_calls[1][1]
        k2_scores = planner_calls[1][2]
        assert len(k2_plans) == 2
        assert k2_plans[0] == "initial plan"
        assert len(k2_scores) == 2
        assert k2_scores == [0.85, 0.87]

    async def test_planner_history_includes_failed_attempts(self) -> None:
        """Planner history includes attempts where score was None (REQ-P2I-035)."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=3)

        planner_calls: list[tuple[Any, ...]] = []

        async def capture_planner(*args: Any, **kwargs: Any) -> str:
            planner_calls.append(args)
            return "new plan"

        # k=0 returns None score, k=1 returns 0.85
        eval_results = [
            _make_eval_result(None),
            _make_eval_result(0.85),
            _make_eval_result(0.90),
        ]
        eval_index = 0

        async def mock_eval(
            *args: Any, **kwargs: Any
        ) -> tuple[SolutionScript, EvaluationResult]:
            nonlocal eval_index
            r = eval_results[eval_index]
            eval_index += 1
            return (_make_solution(), r)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(f"{_MODULE}.invoke_planner", side_effect=capture_planner),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(f"{_MODULE}.evaluate_with_retry", side_effect=mock_eval),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # At k=1: history includes k=0's None score
        k1_scores = planner_calls[0][2]
        assert k1_scores == [None]


# ===========================================================================
# REQ-P2I-021: Coder always receives ORIGINAL code_block.content
# ===========================================================================


@pytest.mark.unit
class TestCoderAlwaysGetsOriginalCodeBlock:
    """A_coder always receives the original code_block.content at every k (REQ-P2I-021)."""

    async def test_coder_receives_original_code_block_at_every_step(self) -> None:
        """All invoke_coder calls pass the original code_block.content as first arg."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        original_content = "TARGET_BLOCK"
        solution = _make_solution()
        code_block = _make_code_block(content=original_content)
        task = _make_task()
        config = _make_config(inner_loop_steps=3)

        coder_calls: list[tuple[Any, ...]] = []

        async def capture_coder(*args: Any, **kwargs: Any) -> str:
            coder_calls.append(args)
            return f"improved_v{len(coder_calls)}"

        with (
            patch(f"{_MODULE}.invoke_coder", side_effect=capture_coder),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="new plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert len(coder_calls) == 3
        for i, coder_call in enumerate(coder_calls):
            assert coder_call[0] == original_content, (
                f"At k={i}, coder received {coder_call[0]!r} "
                f"instead of original {original_content!r}"
            )


# ===========================================================================
# REQ-P2I-022/023: replace_block called on ORIGINAL solution
# ===========================================================================


@pytest.mark.unit
class TestReplaceBlockOnOriginalSolution:
    """replace_block is called against the ORIGINAL solution, not intermediates (REQ-P2I-022/023)."""

    async def test_replace_block_uses_original_code_block_content(self) -> None:
        """solution.replace_block is called with original code_block.content as old."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        original_content = "TARGET_BLOCK"
        solution = _make_solution(content="prefix TARGET_BLOCK suffix")
        code_block = _make_code_block(content=original_content)
        task = _make_task()
        config = _make_config(inner_loop_steps=2)

        replace_calls: list[tuple[str, str]] = []
        original_replace = SolutionScript.replace_block

        def tracking_replace(
            self_: SolutionScript, old: str, new: str
        ) -> SolutionScript:
            replace_calls.append((old, new))
            return original_replace(self_, old, new)

        with (
            patch.object(SolutionScript, "replace_block", tracking_replace),
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="new plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # Both iterations should replace against the original code_block.content
        assert len(replace_calls) == 2
        for old, _new in replace_calls:
            assert old == original_content


# ===========================================================================
# REQ-P2I-024: Local best score initialized from best_score parameter
# ===========================================================================


@pytest.mark.unit
class TestBestScoreInitialization:
    """local_best_score initializes from best_score; local_best_solution = solution (REQ-P2I-024)."""

    async def test_no_improvement_returns_original_best_score(self) -> None:
        """When no iteration improves, returned best_score equals input best_score."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=2)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.75)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert result.best_score == 0.80


# ===========================================================================
# REQ-P2I-025/026: Update best on >= (is_improvement_or_equal)
# ===========================================================================


@pytest.mark.unit
class TestBestUpdateOnImprovementOrEqual:
    """After eval, compare with is_improvement_or_equal (>= semantics) (REQ-P2I-025/026)."""

    async def test_equal_score_triggers_update_maximize(self) -> None:
        """Equal score updates best when direction is MAXIMIZE."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task(direction=MetricDirection.MAXIMIZE)
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(f"{_MODULE}.invoke_planner", new_callable=AsyncMock),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.80)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # Equal score should update best_score (is_improvement_or_equal=True)
        assert result.best_score == 0.80
        # But was_improvement should reflect whether the attempt was at least as good
        assert result.attempts[0].was_improvement is True

    async def test_better_score_triggers_update_maximize(self) -> None:
        """Higher score updates best when direction is MAXIMIZE."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task(direction=MetricDirection.MAXIMIZE)
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(f"{_MODULE}.invoke_planner", new_callable=AsyncMock),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.90)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert result.best_score == 0.90
        assert result.attempts[0].was_improvement is True

    async def test_lower_score_triggers_update_minimize(self) -> None:
        """Lower-or-equal score updates best when direction is MINIMIZE."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task(direction=MetricDirection.MINIMIZE)
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(f"{_MODULE}.invoke_planner", new_callable=AsyncMock),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.70)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert result.best_score == 0.70
        assert result.attempts[0].was_improvement is True


# ===========================================================================
# REQ-P2I-027: None score never triggers best-score update
# ===========================================================================


@pytest.mark.unit
class TestNoneScoreNoUpdate:
    """None evaluation score does not trigger best update (REQ-P2I-027)."""

    async def test_none_score_keeps_original_best(self) -> None:
        """When eval returns None score, best_score stays unchanged."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(f"{_MODULE}.invoke_planner", new_callable=AsyncMock),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(None)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert result.best_score == 0.80
        assert result.attempts[0].was_improvement is False

    async def test_none_score_was_improvement_false(self) -> None:
        """was_improvement is False when score is None."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=2)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(None)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        for attempt in result.attempts:
            assert attempt.was_improvement is False

    async def test_none_score_does_not_call_is_improvement_or_equal(self) -> None:
        """is_improvement_or_equal is NOT called when score is None."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(f"{_MODULE}.invoke_planner", new_callable=AsyncMock),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(None)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False) as mock_cmp,
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        mock_cmp.assert_not_called()


# ===========================================================================
# REQ-P2I-028: RefinementAttempt fields
# ===========================================================================


@pytest.mark.unit
class TestRefinementAttemptFields:
    """Each attempt captures plan, score, code_block, was_improvement (REQ-P2I-028)."""

    async def test_successful_attempt_fields(self) -> None:
        """Successful attempt has plan, score, code_block from coder, was_improvement."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved_code_here",
            ),
            patch(f"{_MODULE}.invoke_planner", new_callable=AsyncMock),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.90)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan_x",
                best_score=0.80,
                task=task,
                config=config,
            )

        attempt = result.attempts[0]
        assert attempt.plan == "plan_x"
        assert attempt.score == 0.90
        assert attempt.code_block == "improved_code_here"
        assert attempt.was_improvement is True

    async def test_coder_failure_attempt_fields(self) -> None:
        """When coder returns None, attempt has code_block="" and no eval."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(f"{_MODULE}.invoke_planner", new_callable=AsyncMock),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
            ) as mock_eval,
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        attempt = result.attempts[0]
        assert attempt.code_block == ""
        assert attempt.score is None
        assert attempt.was_improvement is False
        # evaluate_with_retry should NOT be called when coder fails
        mock_eval.assert_not_called()

    async def test_replace_block_failure_attempt_fields(self) -> None:
        """When replace_block raises ValueError, attempt records coder output and skips eval."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        # Solution that does NOT contain the code_block content
        solution = _make_solution(content="no match here")
        code_block = _make_code_block(content="NONEXISTENT")
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="coder_output",
            ),
            patch(f"{_MODULE}.invoke_planner", new_callable=AsyncMock),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
            ) as mock_eval,
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        attempt = result.attempts[0]
        assert attempt.code_block == "coder_output"
        assert attempt.score is None
        assert attempt.was_improvement is False
        mock_eval.assert_not_called()


# ===========================================================================
# REQ-P2I-029: Attempts list has exactly K entries, ordered
# ===========================================================================


@pytest.mark.unit
class TestAttemptsListStructure:
    """attempts list has exactly K entries, ordered k=0 first (REQ-P2I-029)."""

    async def test_exactly_k_attempts(self) -> None:
        """With K=4, exactly 4 RefinementAttempts are returned."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=4)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert len(result.attempts) == 4
        assert all(isinstance(a, RefinementAttempt) for a in result.attempts)

    async def test_attempts_ordered_k0_first(self) -> None:
        """attempts[0] corresponds to k=0 (initial_plan)."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=3)

        plan_counter = 0

        async def mock_planner(*args: Any, **kwargs: Any) -> str:
            nonlocal plan_counter
            plan_counter += 1
            return f"planner_plan_{plan_counter}"

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(f"{_MODULE}.invoke_planner", side_effect=mock_planner),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert result.attempts[0].plan == "initial plan"
        assert result.attempts[1].plan == "planner_plan_1"
        assert result.attempts[2].plan == "planner_plan_2"


# ===========================================================================
# InnerLoopResult.improved uses strict is_improvement
# ===========================================================================


@pytest.mark.unit
class TestImprovedFlagUsesStrictImprovement:
    """InnerLoopResult.improved=True only on strict improvement (is_improvement) (REQ-P2I-025)."""

    async def test_improved_true_on_strict_improvement(self) -> None:
        """improved=True when is_improvement returns True."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(f"{_MODULE}.invoke_planner", new_callable=AsyncMock),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.90)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert result.improved is True

    async def test_improved_false_on_equal_score(self) -> None:
        """improved=False when score equals best (is_improvement_or_equal=True but is_improvement=False)."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(f"{_MODULE}.invoke_planner", new_callable=AsyncMock),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.80)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # Equal score: was_improvement=True (>= semantics) but improved=False (strict)
        assert result.improved is False

    async def test_improved_false_on_no_improvement(self) -> None:
        """improved=False when no iteration produces any improvement."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=3)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.70)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert result.improved is False


# ===========================================================================
# Preserves input on no improvement
# ===========================================================================


@pytest.mark.unit
class TestPreservesInputOnNoImprovement:
    """When no improvement, returned solution equals input solution (REQ-P2I-024)."""

    async def test_returns_original_solution_on_no_improvement(self) -> None:
        """best_solution is the original input solution when nothing improves."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution(content="original code with TARGET_BLOCK here")
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=2)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.70)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert result.best_solution.content == solution.content


# ===========================================================================
# K=1 single iteration (only k=0)
# ===========================================================================


@pytest.mark.unit
class TestK1SingleIteration:
    """Works correctly with K=1 -- only k=0, no planner call."""

    async def test_k1_one_attempt_no_planner(self) -> None:
        """K=1 yields one attempt and zero planner calls."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ) as mock_coder,
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
            ) as mock_planner,
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="sole plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert len(result.attempts) == 1
        assert result.attempts[0].plan == "sole plan"
        mock_planner.assert_not_called()
        mock_coder.assert_called_once()


# ===========================================================================
# All coder failures
# ===========================================================================


@pytest.mark.unit
class TestAllCoderFailures:
    """When all coder calls return None, returns original solution (all failures)."""

    async def test_all_failures_returns_original(self) -> None:
        """All coder failures: original solution and score preserved."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=3)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
            ) as mock_eval,
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert result.best_score == 0.80
        assert result.best_solution.content == solution.content
        assert result.improved is False
        assert len(result.attempts) == 3
        # evaluate_with_retry never called when coder fails
        mock_eval.assert_not_called()
        for attempt in result.attempts:
            assert attempt.code_block == ""
            assert attempt.score is None
            assert attempt.was_improvement is False

    async def test_all_failures_planner_still_called(self) -> None:
        """Even when coder fails, planner is called for k>=1 iterations."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=3)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ) as mock_planner,
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(f"{_MODULE}.evaluate_with_retry", new_callable=AsyncMock),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # K=3: planner called at k=1 and k=2
        assert mock_planner.call_count == 2


# ===========================================================================
# Mixed success/failure
# ===========================================================================


@pytest.mark.unit
class TestMixedSuccessFailure:
    """Some iterations succeed, some fail -- best tracking correct."""

    async def test_mixed_iterations(self) -> None:
        """k=0 succeeds (0.85), k=1 coder fails, k=2 succeeds (0.90)."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=3)

        coder_results = ["improved_v1", None, "improved_v3"]
        coder_index = 0

        async def mock_coder(*args: Any, **kwargs: Any) -> str | None:
            nonlocal coder_index
            result = coder_results[coder_index]
            coder_index += 1
            return result

        eval_results = [_make_eval_result(0.85), _make_eval_result(0.90)]
        eval_index = 0

        async def mock_eval(
            *args: Any, **kwargs: Any
        ) -> tuple[SolutionScript, EvaluationResult]:
            nonlocal eval_index
            r = eval_results[eval_index]
            eval_index += 1
            return (_make_solution(), r)

        improvement_or_equal_results = [True, True]  # For k=0 and k=2
        ioer_index = 0

        def mock_improvement_or_equal(*args: Any, **kwargs: Any) -> bool:
            nonlocal ioer_index
            r = improvement_or_equal_results[ioer_index]
            ioer_index += 1
            return r

        improvement_results = [True, True]  # For k=0 and k=2
        ir_index = 0

        def mock_improvement(*args: Any, **kwargs: Any) -> bool:
            nonlocal ir_index
            r = improvement_results[ir_index]
            ir_index += 1
            return r

        with (
            patch(f"{_MODULE}.invoke_coder", side_effect=mock_coder),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(f"{_MODULE}.evaluate_with_retry", side_effect=mock_eval),
            patch(
                f"{_MODULE}.is_improvement_or_equal",
                side_effect=mock_improvement_or_equal,
            ),
            patch(f"{_MODULE}.is_improvement", side_effect=mock_improvement),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert len(result.attempts) == 3

        # k=0: success
        assert result.attempts[0].score == 0.85
        assert result.attempts[0].code_block == "improved_v1"
        assert result.attempts[0].was_improvement is True

        # k=1: coder failure
        assert result.attempts[1].score is None
        assert result.attempts[1].code_block == ""
        assert result.attempts[1].was_improvement is False

        # k=2: success
        assert result.attempts[2].score == 0.90
        assert result.attempts[2].code_block == "improved_v3"
        assert result.attempts[2].was_improvement is True

        # Best should be 0.90
        assert result.best_score == 0.90
        assert result.improved is True


# ===========================================================================
# Planner failure handling
# ===========================================================================


@pytest.mark.unit
class TestPlannerFailure:
    """When planner returns None at k>=1, record attempt with plan='[planner failed]'."""

    async def test_planner_none_records_failure_plan(self) -> None:
        """Planner returning None records attempt with plan='[planner failed]' and skips eval."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=2)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ) as mock_eval,
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # k=0 uses initial_plan, k=1 planner fails
        assert result.attempts[1].plan == "[planner failed]"
        assert result.attempts[1].score is None
        assert result.attempts[1].was_improvement is False
        # evaluate_with_retry called once for k=0 but not for k=1
        assert mock_eval.call_count == 1


# ===========================================================================
# evaluate_with_retry is called (with check_and_fix_leakage and make_debug_callback)
# ===========================================================================


@pytest.mark.unit
class TestUsesEvaluateWithRetry:
    """Inner loop calls evaluate_with_retry with debug callback."""

    async def test_evaluate_with_retry_called(self) -> None:
        """evaluate_with_retry is called for each successful coder+replace iteration."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=2)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ) as mock_eval,
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert mock_eval.call_count == 2


# ===========================================================================
# Parametrized tests for K values
# ===========================================================================


@pytest.mark.unit
class TestParametrizedKValues:
    """Parametrized tests verifying attempt count and planner calls for various K."""

    @pytest.mark.parametrize(
        "k,expected_planner_calls",
        [
            (1, 0),
            (2, 1),
            (3, 2),
            (5, 4),
            (10, 9),
        ],
        ids=["K=1", "K=2", "K=3", "K=5", "K=10"],
    )
    async def test_attempt_count_matches_k(
        self, k: int, expected_planner_calls: int
    ) -> None:
        """Number of attempts equals K and planner calls equal K-1."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=k)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ) as mock_planner,
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert len(result.attempts) == k
        assert mock_planner.call_count == expected_planner_calls


# ===========================================================================
# Best solution update tracks the correct improved solution
# ===========================================================================


@pytest.mark.unit
class TestBestSolutionTracking:
    """Best solution is correctly updated when improvement occurs."""

    async def test_best_solution_updated_on_improvement(self) -> None:
        """best_solution content reflects the improved solution after replace_block."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution(content="original code with TARGET_BLOCK here")
        code_block = _make_code_block(content="TARGET_BLOCK")
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="IMPROVED_BLOCK",
            ),
            patch(f"{_MODULE}.invoke_planner", new_callable=AsyncMock),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c, cb: (sol, _make_eval_result(0.90)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True),
            patch(f"{_MODULE}.is_improvement", return_value=True),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # The best solution should have TARGET_BLOCK replaced with IMPROVED_BLOCK
        assert "IMPROVED_BLOCK" in result.best_solution.content
        assert "TARGET_BLOCK" not in result.best_solution.content

    async def test_later_improvement_overwrites_earlier(self) -> None:
        """When k=2 improves over k=0, best_solution is from k=2."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution(content="code TARGET_BLOCK end")
        code_block = _make_code_block(content="TARGET_BLOCK")
        task = _make_task()
        config = _make_config(inner_loop_steps=3)

        coder_results = ["v1_code", "v2_code", "v3_code"]
        coder_index = 0

        async def mock_coder(*args: Any, **kwargs: Any) -> str:
            nonlocal coder_index
            r = coder_results[coder_index]
            coder_index += 1
            return r

        eval_results = [
            _make_eval_result(0.85),
            _make_eval_result(0.83),
            _make_eval_result(0.95),
        ]
        eval_index = 0

        async def mock_eval(
            sol: Any, *args: Any, **kwargs: Any
        ) -> tuple[SolutionScript, EvaluationResult]:
            nonlocal eval_index
            r = eval_results[eval_index]
            eval_index += 1
            return (sol, r)

        # k=0: improvement, k=1: no improvement, k=2: improvement
        ioer_results = [True, False, True]
        ioer_index = 0

        def mock_ioer(*args: Any, **kwargs: Any) -> bool:
            nonlocal ioer_index
            r = ioer_results[ioer_index]
            ioer_index += 1
            return r

        ir_results = [True, False, True]
        ir_index = 0

        def mock_ir(*args: Any, **kwargs: Any) -> bool:
            nonlocal ir_index
            r = ir_results[ir_index]
            ir_index += 1
            return r

        with (
            patch(f"{_MODULE}.invoke_coder", side_effect=mock_coder),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(f"{_MODULE}.evaluate_with_retry", side_effect=mock_eval),
            patch(f"{_MODULE}.is_improvement_or_equal", side_effect=mock_ioer),
            patch(f"{_MODULE}.is_improvement", side_effect=mock_ir),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert result.best_score == 0.95
        assert "v3_code" in result.best_solution.content
        assert result.improved is True


# ===========================================================================
# check_and_fix_leakage and evaluate_with_retry receive the replaced solution
# ===========================================================================


@pytest.mark.unit
class TestLeakageAndEvalReceiveReplacedSolution:
    """check_and_fix_leakage receives the replaced solution; evaluate_with_retry receives its output."""

    async def test_leakage_check_receives_replaced_solution(self) -> None:
        """check_and_fix_leakage receives the solution after replace_block."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution(content="prefix TARGET_BLOCK suffix")
        code_block = _make_code_block(content="TARGET_BLOCK")
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        leakage_solutions: list[Any] = []
        eval_solutions: list[Any] = []

        async def capture_leakage(sol: Any, t: Any, c: Any) -> SolutionScript:
            leakage_solutions.append(sol)
            return sol

        async def capture_eval(
            sol: Any, *args: Any, **kwargs: Any
        ) -> tuple[SolutionScript, EvaluationResult]:
            eval_solutions.append(sol)
            return (sol, _make_eval_result(0.85))

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="NEW_CODE",
            ),
            patch(f"{_MODULE}.invoke_planner", new_callable=AsyncMock),
            patch(f"{_MODULE}.check_and_fix_leakage", side_effect=capture_leakage),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(f"{_MODULE}.evaluate_with_retry", side_effect=capture_eval),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # check_and_fix_leakage receives the replaced solution
        assert len(leakage_solutions) == 1
        assert "NEW_CODE" in leakage_solutions[0].content
        assert "TARGET_BLOCK" not in leakage_solutions[0].content

        # evaluate_with_retry receives whatever check_and_fix_leakage returned
        assert len(eval_solutions) == 1
        assert "NEW_CODE" in eval_solutions[0].content
        assert "TARGET_BLOCK" not in eval_solutions[0].content


# ===========================================================================
# is_improvement_or_equal receives correct arguments
# ===========================================================================


@pytest.mark.unit
class TestIsImprovementOrEqualArguments:
    """is_improvement_or_equal is called with (new_score, best_score, direction)."""

    async def test_comparison_called_with_correct_args(self) -> None:
        """Verify arguments to is_improvement_or_equal."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task(direction=MetricDirection.MAXIMIZE)
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(f"{_MODULE}.invoke_planner", new_callable=AsyncMock),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.90)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=True) as mock_cmp,
            patch(f"{_MODULE}.is_improvement", return_value=True),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        mock_cmp.assert_called_once_with(0.90, 0.80, MetricDirection.MAXIMIZE)


# ===========================================================================
# Property-based tests
# ===========================================================================


@pytest.mark.unit
class TestInnerLoopPropertyBased:
    """Property-based tests for run_phase2_inner_loop invariants."""

    @given(
        k=st.integers(min_value=1, max_value=8),
    )
    @settings(max_examples=15)
    async def test_attempt_count_always_equals_k(self, k: int) -> None:
        """For any K >= 1, exactly K attempts are returned."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=k)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert len(result.attempts) == k

    @given(
        k=st.integers(min_value=1, max_value=8),
    )
    @settings(max_examples=15)
    async def test_first_attempt_always_has_initial_plan(self, k: int) -> None:
        """For any K >= 1, attempts[0].plan is always the initial_plan."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=k)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="unique_initial_plan_marker",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert result.attempts[0].plan == "unique_initial_plan_marker"

    @given(
        best_score=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=15)
    async def test_no_improvement_preserves_best_score(self, best_score: float) -> None:
        """When is_improvement_or_equal returns False, best_score is preserved."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(f"{_MODULE}.invoke_planner", new_callable=AsyncMock),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.50)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=best_score,
                task=task,
                config=config,
            )

        assert result.best_score == best_score

    @given(
        k=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=10)
    async def test_all_coder_failures_never_calls_eval(self, k: int) -> None:
        """When all coder calls return None, evaluate_with_retry is never called."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=k)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
            ) as mock_eval,
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            result = await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        mock_eval.assert_not_called()
        assert result.improved is False
        assert all(a.score is None for a in result.attempts)


# ===========================================================================
# Calls check_and_fix_leakage on each successful iteration
# ===========================================================================


@pytest.mark.unit
class TestCallsLeakageCheck:
    """Inner loop calls check_and_fix_leakage before every evaluation."""

    async def test_leakage_check_called_per_successful_iteration(self) -> None:
        """check_and_fix_leakage is called once per successful coder+replace iteration."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=2)

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ) as mock_leakage,
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # check_and_fix_leakage called once per successful iteration (K=2)
        assert mock_leakage.call_count == 2


# ===========================================================================
# Accumulated history includes plan text from failed planner
# ===========================================================================


@pytest.mark.unit
class TestAccumulatedHistoryIncludesFailedPlans:
    """Failed attempts are included in the accumulated history (REQ-P2I-035)."""

    async def test_planner_failure_appears_in_later_history(self) -> None:
        """When planner fails at k=1, the '[planner failed]' plan is in k=2 history."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=3)

        planner_calls: list[tuple[Any, ...]] = []
        planner_index = 0

        async def mock_planner(*args: Any, **kwargs: Any) -> str | None:
            nonlocal planner_index
            planner_calls.append(args)
            planner_index += 1
            if planner_index == 1:
                return None  # k=1 planner fails
            return "plan_from_planner"  # k=2 planner succeeds

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(f"{_MODULE}.invoke_planner", side_effect=mock_planner),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="initial plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        # planner_calls[1] is the k=2 call, should have 2 prior plans
        assert len(planner_calls) == 2  # k=1 and k=2

        # k=2 planner receives history that includes the failed k=1 plan
        k2_plans = planner_calls[1][1]
        assert len(k2_plans) == 2
        assert k2_plans[0] == "initial plan"
        assert k2_plans[1] == "[planner failed]"


# ===========================================================================
# Coder client argument
# ===========================================================================


@pytest.mark.unit
class TestCoderClientArgument:
    """Coder receives the client as its third argument at every iteration."""

    async def test_coder_receives_client(self) -> None:
        """invoke_coder is always called with the client as third positional arg."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=2)

        coder_calls: list[tuple[Any, ...]] = []

        async def capture_coder(*args: Any, **kwargs: Any) -> str:
            coder_calls.append(args)
            return "improved"

        with (
            patch(f"{_MODULE}.invoke_coder", side_effect=capture_coder),
            patch(
                f"{_MODULE}.invoke_planner",
                new_callable=AsyncMock,
                return_value="plan",
            ),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        for coder_call in coder_calls:
            assert coder_call[2] is client


# ===========================================================================
# Planner client argument
# ===========================================================================


@pytest.mark.unit
class TestPlannerClientArgument:
    """Planner receives the client as its fourth argument."""

    async def test_planner_receives_client(self) -> None:
        """invoke_planner is called with the client as fourth positional arg."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=2)

        planner_calls: list[tuple[Any, ...]] = []

        async def capture_planner(*args: Any, **kwargs: Any) -> str:
            planner_calls.append(args)
            return "plan"

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(f"{_MODULE}.invoke_planner", side_effect=capture_planner),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=AsyncMock()),
            patch(
                f"{_MODULE}.evaluate_with_retry",
                new_callable=AsyncMock,
                return_value=(_make_solution(), _make_eval_result(0.85)),
            ),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert len(planner_calls) == 1  # K=2, planner called once at k=1
        assert planner_calls[0][3] is client


# ===========================================================================
# evaluate_with_retry receives task, config, and debug callback
# ===========================================================================


@pytest.mark.unit
class TestEvaluateWithRetryArguments:
    """evaluate_with_retry receives the modified solution, task, config, and debug callback."""

    async def test_eval_receives_task_config_and_callback(self) -> None:
        """evaluate_with_retry is called with (modified_solution, task, config, debug_callback)."""
        from mle_star.phase2_inner import run_phase2_inner_loop

        client = AsyncMock()
        solution = _make_solution()
        code_block = _make_code_block()
        task = _make_task()
        config = _make_config(inner_loop_steps=1)

        eval_args: list[tuple[Any, ...]] = []
        mock_callback = AsyncMock()

        async def capture_eval(
            *args: Any, **kwargs: Any
        ) -> tuple[SolutionScript, EvaluationResult]:
            eval_args.append(args)
            return (_make_solution(), _make_eval_result(0.85))

        with (
            patch(
                f"{_MODULE}.invoke_coder",
                new_callable=AsyncMock,
                return_value="improved",
            ),
            patch(f"{_MODULE}.invoke_planner", new_callable=AsyncMock),
            patch(
                f"{_MODULE}.check_and_fix_leakage",
                new_callable=AsyncMock,
                side_effect=lambda sol, t, c: sol,
            ),
            patch(f"{_MODULE}.make_debug_callback", return_value=mock_callback),
            patch(f"{_MODULE}.evaluate_with_retry", side_effect=capture_eval),
            patch(f"{_MODULE}.is_improvement_or_equal", return_value=False),
            patch(f"{_MODULE}.is_improvement", return_value=False),
        ):
            await run_phase2_inner_loop(
                client=client,
                solution=solution,
                code_block=code_block,
                initial_plan="plan",
                best_score=0.80,
                task=task,
                config=config,
            )

        assert len(eval_args) == 1
        # args: (solution, task, config, debug_callback)
        assert isinstance(eval_args[0][0], SolutionScript)
        assert eval_args[0][1] is task
        assert eval_args[0][2] is config
        assert eval_args[0][3] is mock_callback
