"""Tests for the Phase 2 outer loop orchestration function (Task 33).

Validates ``run_phase2_outer_loop`` which implements the T-iteration outer
loop of Algorithm 2 for Phase 2 targeted refinement.  Each iteration invokes
ablation -> summarize -> extractor -> inner loop, tracks the best solution
using ``is_improvement_or_equal`` (>= semantics), and accumulates ablation
summaries and code blocks across iterations.

Tests are written TDD-first and serve as the executable specification for
REQ-P2O-019 through REQ-P2O-030, REQ-P2O-041, REQ-P2O-043, REQ-P2O-044.

Refs:
    SRS 05c (Phase 2 Outer Orchestration), IMPLEMENTATION_PLAN.md Task 33.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    CodeBlock,
    DataModality,
    ExtractorOutput,
    InnerLoopResult,
    MetricDirection,
    Phase2Result,
    PipelineConfig,
    RefinementAttempt,
    RefinePlan,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
    TaskType,
)
import pytest

# ---------------------------------------------------------------------------
# Module path constant for patching
# ---------------------------------------------------------------------------

_MODULE = "mle_star.phase2_outer"


# ---------------------------------------------------------------------------
# Reusable test helpers
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
        description="Predict the target variable from tabular features.",
    )


def _make_config(**kwargs: Any) -> PipelineConfig:
    """Create a PipelineConfig for testing with optional overrides."""
    return PipelineConfig(**kwargs)


def _make_solution(
    content: str = "import pandas as pd\ndf = pd.read_csv('data.csv')\nmodel.fit(df)",
    phase: SolutionPhase = SolutionPhase.INIT,
    score: float | None = None,
) -> SolutionScript:
    """Create a SolutionScript for testing."""
    return SolutionScript(content=content, phase=phase, score=score)


def _make_inner_loop_result(
    best_solution: SolutionScript | None = None,
    best_score: float = 0.90,
    improved: bool = True,
    attempts: list[RefinementAttempt] | None = None,
) -> InnerLoopResult:
    """Create a mock InnerLoopResult."""
    if best_solution is None:
        best_solution = _make_solution(
            content="improved code", phase=SolutionPhase.REFINED
        )
    if attempts is None:
        attempts = [
            RefinementAttempt(
                plan="test plan",
                score=best_score,
                code_block="improved block",
                was_improvement=improved,
            )
        ]
    return InnerLoopResult(
        best_solution=best_solution,
        best_score=best_score,
        attempts=attempts,
        improved=improved,
    )


def _make_extractor_output(
    code_block: str = "model.fit(df)",
    plan: str = "Optimize the model training loop",
) -> ExtractorOutput:
    """Create a mock ExtractorOutput with a single plan."""
    return ExtractorOutput(plans=[RefinePlan(code_block=code_block, plan=plan)])


def _make_ablation_script(
    content: str = "print('ablation study')",
) -> SolutionScript:
    """Create a SolutionScript wrapping an ablation script."""
    return SolutionScript(
        content=content, phase=SolutionPhase.REFINED, is_executable=True
    )


def _patch_outer_loop_dependencies(
    invoke_ablation_rv: SolutionScript | None = None,
    execute_ablation_rv: tuple[str, str] = ("ablation stdout", ""),
    invoke_summarize_rv: str = "Ablation summary: feature engineering is key",
    invoke_extractor_rv: ExtractorOutput | None = None,
    validate_code_block_rv: bool = True,
    inner_loop_rv: InnerLoopResult | None = None,
) -> dict[str, Any]:
    """Build a dict of patch objects for all outer loop dependencies.

    Returns a dict of mock objects keyed by function name for
    assertions in tests.
    """
    if invoke_ablation_rv is None:
        invoke_ablation_rv = _make_ablation_script()
    if invoke_extractor_rv is None:
        invoke_extractor_rv = _make_extractor_output()
    if inner_loop_rv is None:
        inner_loop_rv = _make_inner_loop_result()

    return {
        "invoke_ablation": AsyncMock(return_value=invoke_ablation_rv),
        "execute_ablation_with_retry": AsyncMock(return_value=execute_ablation_rv),
        "invoke_summarize": AsyncMock(return_value=invoke_summarize_rv),
        "invoke_extractor": AsyncMock(return_value=invoke_extractor_rv),
        "validate_code_block": lambda code_block, solution: validate_code_block_rv,
        "run_phase2_inner_loop": AsyncMock(return_value=inner_loop_rv),
        "is_improvement_or_equal": lambda new, old, direction: (
            new >= old if direction == MetricDirection.MAXIMIZE else new <= old
        ),
    }


async def _run_outer_loop(
    mocks: dict[str, Any],
    outer_loop_steps: int = 2,
    initial_score: float = 0.80,
    initial_solution: SolutionScript | None = None,
    task: TaskDescription | None = None,
    config: PipelineConfig | None = None,
    session_id: str = "test-session",
) -> Phase2Result:
    """Run run_phase2_outer_loop with all dependencies mocked.

    Args:
        mocks: Dict from _patch_outer_loop_dependencies().
        outer_loop_steps: Number of outer iterations.
        initial_score: Starting score.
        initial_solution: Starting solution (defaults to helper).
        task: TaskDescription (defaults to helper).
        config: PipelineConfig (defaults to helper with outer_loop_steps).
        session_id: Session identifier.

    Returns:
        The Phase2Result from the orchestration function.
    """
    from mle_star.phase2_outer import run_phase2_outer_loop

    if initial_solution is None:
        initial_solution = _make_solution()
    if task is None:
        task = _make_task()
    if config is None:
        config = _make_config(outer_loop_steps=outer_loop_steps)

    client = AsyncMock()

    with (
        patch(f"{_MODULE}.invoke_ablation", mocks["invoke_ablation"]),
        patch(
            f"{_MODULE}.execute_ablation_with_retry",
            mocks["execute_ablation_with_retry"],
        ),
        patch(f"{_MODULE}.invoke_summarize", mocks["invoke_summarize"]),
        patch(f"{_MODULE}.invoke_extractor", mocks["invoke_extractor"]),
        patch(f"{_MODULE}.validate_code_block", mocks["validate_code_block"]),
        patch(
            f"{_MODULE}.run_phase2_inner_loop",
            mocks["run_phase2_inner_loop"],
        ),
        patch(
            f"{_MODULE}.is_improvement_or_equal",
            mocks["is_improvement_or_equal"],
        ),
    ):
        return await run_phase2_outer_loop(
            client=client,
            task=task,
            config=config,
            initial_solution=initial_solution,
            initial_score=initial_score,
            session_id=session_id,
        )


# ===========================================================================
# REQ-P2O-019: run_phase2_outer_loop signature and async
# ===========================================================================


@pytest.mark.unit
class TestRunPhase2OuterLoopIsAsync:
    """run_phase2_outer_loop is an async function returning Phase2Result (REQ-P2O-019)."""

    def test_is_coroutine_function(self) -> None:
        """run_phase2_outer_loop is defined as an async function."""
        from mle_star.phase2_outer import run_phase2_outer_loop

        assert asyncio.iscoroutinefunction(run_phase2_outer_loop)

    async def test_returns_phase2_result(self) -> None:
        """run_phase2_outer_loop returns a Phase2Result instance."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=1)

        assert isinstance(result, Phase2Result)

    async def test_accepts_initial_score_as_float_parameter(self) -> None:
        """run_phase2_outer_loop accepts initial_score: float explicitly."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=1, initial_score=0.42)

        assert isinstance(result, Phase2Result)


# ===========================================================================
# REQ-P2O-019/h_best: h_best initialized from initial_score, not solution.score
# ===========================================================================


@pytest.mark.unit
class TestHBestInitialization:
    """h_best is initialized from initial_score parameter, not initial_solution.score."""

    async def test_uses_initial_score_not_solution_score(self) -> None:
        """h_best is set to initial_score, ignoring initial_solution.score.

        When initial_solution.score differs from initial_score, the outer
        loop must use initial_score as h_best.
        """
        # Solution has score 0.50 but initial_score is 0.80
        solution = _make_solution(score=0.50)

        # Inner loop returns score worse than initial_score (0.80)
        # but better than solution.score (0.50)
        inner_result = _make_inner_loop_result(best_score=0.70, improved=False)
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)

        result = await _run_outer_loop(
            mocks,
            outer_loop_steps=1,
            initial_score=0.80,
            initial_solution=solution,
        )

        # h_best should remain 0.80 (from initial_score), not 0.70
        assert result.best_score == 0.80

    async def test_initial_score_with_none_solution_score(self) -> None:
        """Works when initial_solution.score is None."""
        solution = _make_solution(score=None)
        inner_result = _make_inner_loop_result(best_score=0.75, improved=False)
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)

        result = await _run_outer_loop(
            mocks,
            outer_loop_steps=1,
            initial_score=0.80,
            initial_solution=solution,
        )

        # h_best should be 0.80 (not crash due to None score)
        assert result.best_score == 0.80


# ===========================================================================
# REQ-P2O-025: Executes exactly T outer iterations
# ===========================================================================


@pytest.mark.unit
class TestOuterLoopIterationCount:
    """The outer loop executes exactly config.outer_loop_steps iterations (REQ-P2O-025)."""

    @pytest.mark.parametrize("t_steps", [1, 2, 3, 4, 5])
    async def test_exact_t_iterations(self, t_steps: int) -> None:
        """invoke_ablation called exactly T times for T outer_loop_steps."""
        mocks = _patch_outer_loop_dependencies()
        await _run_outer_loop(mocks, outer_loop_steps=t_steps)

        assert mocks["invoke_ablation"].call_count == t_steps

    @pytest.mark.parametrize("t_steps", [1, 2, 3, 4])
    async def test_step_history_length_equals_t(self, t_steps: int) -> None:
        """step_history has exactly T entries."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=t_steps)

        assert len(result.step_history) == t_steps

    async def test_inner_loop_called_t_times_when_no_skips(self) -> None:
        """run_phase2_inner_loop called T times when all iterations succeed."""
        mocks = _patch_outer_loop_dependencies()
        await _run_outer_loop(mocks, outer_loop_steps=3)

        assert mocks["run_phase2_inner_loop"].call_count == 3


# ===========================================================================
# REQ-P2O-019 to REQ-P2O-026: Each iteration ablation -> summarize -> extract -> inner
# ===========================================================================


@pytest.mark.unit
class TestIterationPipeline:
    """Each iteration follows ablation -> summarize -> extract -> inner loop."""

    async def test_all_agents_called_in_sequence(self) -> None:
        """All four pipeline stages are invoked per iteration."""
        mocks = _patch_outer_loop_dependencies()
        await _run_outer_loop(mocks, outer_loop_steps=1)

        mocks["invoke_ablation"].assert_called_once()
        mocks["execute_ablation_with_retry"].assert_called_once()
        mocks["invoke_summarize"].assert_called_once()
        mocks["invoke_extractor"].assert_called_once()
        mocks["run_phase2_inner_loop"].assert_called_once()

    async def test_ablation_receives_current_solution(self) -> None:
        """invoke_ablation receives the current best solution."""
        solution = _make_solution(content="specific content xyz")
        mocks = _patch_outer_loop_dependencies()
        await _run_outer_loop(mocks, outer_loop_steps=1, initial_solution=solution)

        call_args = mocks["invoke_ablation"].call_args
        passed_solution = call_args[0][0] if call_args[0] else call_args[1]["solution"]
        assert passed_solution.content == "specific content xyz"

    async def test_summarize_receives_ablation_output(self) -> None:
        """invoke_summarize receives the ablation script code and raw output."""
        ablation_script = _make_ablation_script(content="ablation_code_marker")
        mocks = _patch_outer_loop_dependencies(
            invoke_ablation_rv=ablation_script,
            execute_ablation_rv=("raw_stdout_marker", ""),
        )
        await _run_outer_loop(mocks, outer_loop_steps=1)

        call_args = mocks["invoke_summarize"].call_args
        # First positional args or keyword args
        ablation_code_arg = (
            call_args[0][0] if call_args[0] else call_args[1].get("ablation_code")
        )
        raw_output_arg = (
            call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("raw_output")
        )
        assert "ablation_code_marker" in str(ablation_code_arg)
        assert "raw_stdout_marker" in str(raw_output_arg)

    async def test_extractor_receives_summary_and_solution(self) -> None:
        """invoke_extractor receives the ablation summary and current solution."""
        mocks = _patch_outer_loop_dependencies(
            invoke_summarize_rv="test_summary_marker",
        )
        await _run_outer_loop(mocks, outer_loop_steps=1)

        call_args = mocks["invoke_extractor"].call_args
        # Check summary is passed
        assert "test_summary_marker" in str(call_args)


# ===========================================================================
# REQ-P2O-026: Uses FIRST plan from ExtractorOutput
# ===========================================================================


@pytest.mark.unit
class TestFirstPlanUsed:
    """Outer loop uses plans[0] from ExtractorOutput (REQ-P2O-026)."""

    async def test_inner_loop_receives_first_plan_code_block(self) -> None:
        """Inner loop receives code_block from plans[0].code_block."""
        extractor_output = ExtractorOutput(
            plans=[
                RefinePlan(code_block="first_block", plan="first plan"),
                RefinePlan(code_block="second_block", plan="second plan"),
            ]
        )
        solution = _make_solution(content="code with first_block and second_block")
        mocks = _patch_outer_loop_dependencies(
            invoke_extractor_rv=extractor_output,
        )
        await _run_outer_loop(mocks, outer_loop_steps=1, initial_solution=solution)

        call_args = mocks["run_phase2_inner_loop"].call_args
        # code_block argument should contain "first_block"
        call_str = str(call_args)
        assert "first_block" in call_str

    async def test_inner_loop_receives_first_plan_text(self) -> None:
        """Inner loop receives initial_plan from plans[0].plan."""
        extractor_output = ExtractorOutput(
            plans=[
                RefinePlan(code_block="block", plan="plan_alpha_unique"),
                RefinePlan(code_block="other", plan="plan_beta_unique"),
            ]
        )
        solution = _make_solution(content="code with block and other")
        mocks = _patch_outer_loop_dependencies(
            invoke_extractor_rv=extractor_output,
        )
        await _run_outer_loop(mocks, outer_loop_steps=1, initial_solution=solution)

        call_args = mocks["run_phase2_inner_loop"].call_args
        call_str = str(call_args)
        assert "plan_alpha_unique" in call_str
        # Should NOT include the second plan
        assert "plan_beta_unique" not in call_str

    async def test_single_plan_used_directly(self) -> None:
        """Single plan in ExtractorOutput is used correctly."""
        extractor_output = _make_extractor_output(
            code_block="target_block",
            plan="sole_plan_text",
        )
        solution = _make_solution(content="code target_block end")
        mocks = _patch_outer_loop_dependencies(
            invoke_extractor_rv=extractor_output,
        )
        await _run_outer_loop(mocks, outer_loop_steps=1, initial_solution=solution)

        call_str = str(mocks["run_phase2_inner_loop"].call_args)
        assert "sole_plan_text" in call_str


# ===========================================================================
# REQ-P2O-022: Ablation summaries accumulated across iterations
# ===========================================================================


@pytest.mark.unit
class TestAblationSummaryAccumulation:
    """Ablation summaries are accumulated in T_abl list (REQ-P2O-022)."""

    async def test_summaries_collected_across_iterations(self) -> None:
        """Phase2Result.ablation_summaries has T entries after T steps."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=3)

        assert len(result.ablation_summaries) == 3

    async def test_each_summary_from_invoke_summarize(self) -> None:
        """Each ablation summary comes from invoke_summarize return value."""
        summaries = ["summary_1", "summary_2", "summary_3"]
        mock_summarize = AsyncMock(side_effect=summaries)

        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_summarize"] = mock_summarize

        result = await _run_outer_loop(mocks, outer_loop_steps=3)

        assert result.ablation_summaries == summaries

    async def test_previous_summaries_passed_to_ablation(self) -> None:
        """invoke_ablation receives accumulated summaries from prior steps.

        At step t=0, previous_summaries should be empty.
        At step t=1, previous_summaries should contain the first summary.
        """
        summaries = ["first_summary", "second_summary"]
        mock_summarize = AsyncMock(side_effect=summaries)

        ablation_calls: list[Any] = []
        original_ablation = AsyncMock(return_value=_make_ablation_script())

        async def capture_ablation(*args: Any, **kwargs: Any) -> SolutionScript | None:
            ablation_calls.append((args, kwargs))
            return await original_ablation(*args, **kwargs)

        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_summarize"] = mock_summarize
        mocks["invoke_ablation"] = AsyncMock(
            side_effect=[_make_ablation_script(), _make_ablation_script()]
        )

        # We need to capture the previous_summaries arg
        invoke_ablation_calls: list[Any] = []
        original_invoke = mocks["invoke_ablation"]

        async def capture_invoke(*args: Any, **kwargs: Any) -> Any:
            invoke_ablation_calls.append((args, kwargs))
            return await original_invoke(*args, **kwargs)

        mocks["invoke_ablation"] = AsyncMock(side_effect=capture_invoke)

        await _run_outer_loop(mocks, outer_loop_steps=2)

        # First call should have empty previous summaries
        first_call = invoke_ablation_calls[0]
        # Second call should have first summary in previous_summaries
        second_call = invoke_ablation_calls[1]

        # The previous_summaries arg is the second positional arg
        first_prev = (
            first_call[0][1]
            if len(first_call[0]) > 1
            else first_call[1].get("previous_summaries", [])
        )
        second_prev = (
            second_call[0][1]
            if len(second_call[0]) > 1
            else second_call[1].get("previous_summaries", [])
        )

        assert len(first_prev) == 0
        assert len(second_prev) == 1
        assert "first_summary" in str(second_prev)


# ===========================================================================
# REQ-P2O-023, REQ-P2O-044: Code blocks accumulated with outer_step
# ===========================================================================


@pytest.mark.unit
class TestCodeBlockAccumulation:
    """Code blocks are accumulated in C list with outer_step set (REQ-P2O-023/044)."""

    async def test_refined_blocks_has_t_entries(self) -> None:
        """Phase2Result.refined_blocks has T entries after T steps."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=3)

        assert len(result.refined_blocks) == 3

    async def test_code_blocks_are_code_block_type(self) -> None:
        """Each entry in refined_blocks is a CodeBlock instance."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=2)

        for block in result.refined_blocks:
            assert isinstance(block, CodeBlock)

    async def test_code_block_outer_step_matches_iteration_index(self) -> None:
        """Each CodeBlock.outer_step equals its iteration index (REQ-P2O-044)."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=4)

        for t, block in enumerate(result.refined_blocks):
            assert block.outer_step == t

    async def test_code_block_content_from_extractor_first_plan(self) -> None:
        """CodeBlock.content matches plans[0].code_block from extractor."""
        extractor_output = _make_extractor_output(code_block="target_code_block")
        solution = _make_solution(content="code target_code_block end")
        mocks = _patch_outer_loop_dependencies(
            invoke_extractor_rv=extractor_output,
        )
        result = await _run_outer_loop(
            mocks, outer_loop_steps=1, initial_solution=solution
        )

        assert result.refined_blocks[0].content == "target_code_block"

    async def test_previous_blocks_passed_to_extractor(self) -> None:
        """invoke_extractor receives accumulated code block strings from prior steps."""
        # We track what previous_blocks arg is passed to invoke_extractor
        extractor_calls: list[Any] = []

        async def capture_extractor(*args: Any, **kwargs: Any) -> ExtractorOutput:
            extractor_calls.append((args, kwargs))
            return _make_extractor_output()

        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_extractor"] = AsyncMock(side_effect=capture_extractor)

        await _run_outer_loop(mocks, outer_loop_steps=2)

        # First call: no previous blocks
        first_call = extractor_calls[0]
        # Second call: one previous block
        second_call = extractor_calls[1]

        first_prev = (
            first_call[0][2]
            if len(first_call[0]) > 2
            else first_call[1].get("previous_blocks", [])
        )
        second_prev = (
            second_call[0][2]
            if len(second_call[0]) > 2
            else second_call[1].get("previous_blocks", [])
        )

        assert len(first_prev) == 0
        assert len(second_prev) == 1


# ===========================================================================
# REQ-P2O-027: Best solution updated with is_improvement_or_equal
# ===========================================================================


@pytest.mark.unit
class TestBestSolutionUpdate:
    """Best solution is updated using is_improvement_or_equal (>= for maximize) (REQ-P2O-027)."""

    async def test_improvement_updates_best(self) -> None:
        """When inner loop improves score, best solution is updated."""
        improved_solution = _make_solution(
            content="improved", phase=SolutionPhase.REFINED
        )
        inner_result = _make_inner_loop_result(
            best_solution=improved_solution, best_score=0.90, improved=True
        )
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)

        result = await _run_outer_loop(mocks, outer_loop_steps=1, initial_score=0.80)

        assert result.best_score == 0.90
        assert result.best_solution.content == "improved"

    async def test_equal_score_updates_best(self) -> None:
        """When inner loop returns equal score, best solution is updated (>= semantics)."""
        equal_solution = _make_solution(
            content="equal solution", phase=SolutionPhase.REFINED
        )
        inner_result = _make_inner_loop_result(
            best_solution=equal_solution, best_score=0.80, improved=False
        )
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)

        result = await _run_outer_loop(mocks, outer_loop_steps=1, initial_score=0.80)

        # Equal score should trigger update (>= semantics)
        assert result.best_score == 0.80
        assert result.best_solution.content == "equal solution"

    async def test_worse_score_does_not_update_best(self) -> None:
        """When inner loop returns worse score, best solution is NOT updated."""
        initial = _make_solution(content="initial best")
        inner_result = _make_inner_loop_result(best_score=0.70, improved=False)
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)

        result = await _run_outer_loop(
            mocks,
            outer_loop_steps=1,
            initial_score=0.80,
            initial_solution=initial,
        )

        assert result.best_score == 0.80
        assert result.best_solution.content == "initial best"

    async def test_does_not_rely_on_inner_loop_improved_flag(self) -> None:
        """Best-score update uses is_improvement_or_equal, NOT InnerLoopResult.improved.

        The inner loop improved flag uses strict > (is_improvement), but the
        outer loop must use >= (is_improvement_or_equal).
        """
        equal_solution = _make_solution(
            content="equal but new", phase=SolutionPhase.REFINED
        )
        # improved=False because inner loop uses strict >, but score is EQUAL
        inner_result = _make_inner_loop_result(
            best_solution=equal_solution, best_score=0.80, improved=False
        )
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)

        result = await _run_outer_loop(mocks, outer_loop_steps=1, initial_score=0.80)

        # Should update despite improved=False, because >= semantics
        assert result.best_solution.content == "equal but new"

    async def test_minimize_direction_worse_higher_score_not_updated(self) -> None:
        """For minimize direction, higher score (worse) does not update."""
        task = _make_task(direction=MetricDirection.MINIMIZE)
        inner_result = _make_inner_loop_result(best_score=0.90, improved=False)
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)
        # Override comparison for minimize
        mocks["is_improvement_or_equal"] = lambda new, old, d: new <= old

        result = await _run_outer_loop(
            mocks, outer_loop_steps=1, initial_score=0.80, task=task
        )

        assert result.best_score == 0.80  # 0.90 > 0.80, not improvement for minimize

    async def test_minimize_direction_lower_score_updates(self) -> None:
        """For minimize direction, lower score (better) updates best."""
        task = _make_task(direction=MetricDirection.MINIMIZE)
        improved_solution = _make_solution(
            content="minimized", phase=SolutionPhase.REFINED
        )
        inner_result = _make_inner_loop_result(
            best_solution=improved_solution, best_score=0.70, improved=True
        )
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)
        mocks["is_improvement_or_equal"] = lambda new, old, d: new <= old

        result = await _run_outer_loop(
            mocks, outer_loop_steps=1, initial_score=0.80, task=task
        )

        assert result.best_score == 0.70
        assert result.best_solution.content == "minimized"


# ===========================================================================
# REQ-P2O-026: Inner loop receives correct parameters
# ===========================================================================


@pytest.mark.unit
class TestInnerLoopParameters:
    """Inner loop receives solution=s_t, code_block=c_t, initial_plan=p_0, best_score=h_best."""

    async def test_inner_loop_receives_current_solution(self) -> None:
        """Inner loop receives the current best solution as solution arg."""
        solution = _make_solution(content="current_best_solution")
        mocks = _patch_outer_loop_dependencies()
        await _run_outer_loop(mocks, outer_loop_steps=1, initial_solution=solution)

        call_str = str(mocks["run_phase2_inner_loop"].call_args)
        assert "current_best_solution" in call_str

    async def test_inner_loop_receives_code_block_from_extractor(self) -> None:
        """Inner loop receives CodeBlock with content from extractor plans[0]."""
        extractor_output = _make_extractor_output(code_block="extracted_code_block")
        solution = _make_solution(content="code extracted_code_block end")
        mocks = _patch_outer_loop_dependencies(
            invoke_extractor_rv=extractor_output,
        )
        await _run_outer_loop(mocks, outer_loop_steps=1, initial_solution=solution)

        call_str = str(mocks["run_phase2_inner_loop"].call_args)
        assert "extracted_code_block" in call_str

    async def test_inner_loop_receives_best_score(self) -> None:
        """Inner loop receives h_best as best_score argument."""
        mocks = _patch_outer_loop_dependencies()
        await _run_outer_loop(mocks, outer_loop_steps=1, initial_score=0.75)

        call_str = str(mocks["run_phase2_inner_loop"].call_args)
        assert "0.75" in call_str

    async def test_inner_loop_receives_updated_best_score_next_iteration(self) -> None:
        """In iteration 2, inner loop receives updated h_best from iteration 1."""
        inner_result_1 = _make_inner_loop_result(best_score=0.90, improved=True)
        inner_result_2 = _make_inner_loop_result(best_score=0.85, improved=False)

        mock_inner = AsyncMock(side_effect=[inner_result_1, inner_result_2])
        mocks = _patch_outer_loop_dependencies()
        mocks["run_phase2_inner_loop"] = mock_inner

        await _run_outer_loop(mocks, outer_loop_steps=2, initial_score=0.80)

        # Second call should have best_score=0.90 (updated from first iteration)
        second_call = mock_inner.call_args_list[1]
        call_str = str(second_call)
        assert "0.9" in call_str

    async def test_inner_loop_receives_updated_solution_after_improvement(self) -> None:
        """In iteration 2, inner loop receives updated solution from iteration 1."""
        improved_sol = _make_solution(content="improved_from_iter_1")
        inner_result_1 = _make_inner_loop_result(
            best_solution=improved_sol, best_score=0.90, improved=True
        )
        inner_result_2 = _make_inner_loop_result(best_score=0.85, improved=False)

        mock_inner = AsyncMock(side_effect=[inner_result_1, inner_result_2])
        mocks = _patch_outer_loop_dependencies()
        mocks["run_phase2_inner_loop"] = mock_inner

        await _run_outer_loop(mocks, outer_loop_steps=2, initial_score=0.80)

        # Second call should have the improved solution
        second_call = mock_inner.call_args_list[1]
        call_str = str(second_call)
        assert "improved_from_iter_1" in call_str


# ===========================================================================
# REQ-P2O-030: Skipped iterations with was_skipped=True
# ===========================================================================


@pytest.mark.unit
class TestSkippedIterations:
    """Skipped iterations are recorded with was_skipped=True (REQ-P2O-030)."""

    async def test_extractor_returns_none_marks_skipped(self) -> None:
        """When invoke_extractor returns None, step is skipped."""
        mocks = _patch_outer_loop_dependencies(invoke_extractor_rv=None)
        # Need to set extractor mock to return None directly
        mocks["invoke_extractor"] = AsyncMock(return_value=None)

        result = await _run_outer_loop(mocks, outer_loop_steps=1)

        assert len(result.step_history) == 1
        assert result.step_history[0]["was_skipped"] is True

    async def test_code_block_validation_failure_marks_skipped(self) -> None:
        """When validate_code_block returns False, step is skipped."""
        mocks = _patch_outer_loop_dependencies(validate_code_block_rv=False)
        mocks["validate_code_block"] = lambda code_block, solution: False

        result = await _run_outer_loop(mocks, outer_loop_steps=1)

        assert len(result.step_history) == 1
        assert result.step_history[0]["was_skipped"] is True

    async def test_skipped_step_has_no_inner_loop_attempts(self) -> None:
        """Skipped steps should have empty inner_loop_attempts."""
        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_extractor"] = AsyncMock(return_value=None)

        result = await _run_outer_loop(mocks, outer_loop_steps=1)

        assert result.step_history[0]["inner_loop_attempts"] == []

    async def test_skipped_step_inner_loop_not_called(self) -> None:
        """When step is skipped, run_phase2_inner_loop is NOT called."""
        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_extractor"] = AsyncMock(return_value=None)

        await _run_outer_loop(mocks, outer_loop_steps=1)

        mocks["run_phase2_inner_loop"].assert_not_called()

    async def test_loop_continues_after_skip(self) -> None:
        """Loop continues to next iteration after a skip."""
        # First iteration: extractor fails (skip), second: succeeds
        extractor_calls = [None, _make_extractor_output()]
        mock_extractor = AsyncMock(side_effect=extractor_calls)

        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_extractor"] = mock_extractor

        result = await _run_outer_loop(mocks, outer_loop_steps=2)

        assert len(result.step_history) == 2
        assert result.step_history[0]["was_skipped"] is True
        assert result.step_history[1]["was_skipped"] is False
        # Inner loop called only for the non-skipped iteration
        assert mocks["run_phase2_inner_loop"].call_count == 1

    async def test_ablation_failure_still_proceeds(self) -> None:
        """When invoke_ablation returns None, iteration should still proceed.

        Ablation failure produces empty output, but summarize/extractor
        should still be called.
        """
        mocks = _patch_outer_loop_dependencies(
            invoke_ablation_rv=None,
        )
        mocks["invoke_ablation"] = AsyncMock(return_value=None)
        # execute_ablation should NOT be called if ablation is None
        mocks["execute_ablation_with_retry"] = AsyncMock(return_value=("", ""))

        result = await _run_outer_loop(mocks, outer_loop_steps=1)

        # The iteration should still produce a step history entry
        assert len(result.step_history) == 1

    async def test_skipped_best_score_unchanged(self) -> None:
        """Skipped iteration does not change h_best."""
        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_extractor"] = AsyncMock(return_value=None)

        result = await _run_outer_loop(mocks, outer_loop_steps=1, initial_score=0.80)

        assert result.best_score == 0.80

    async def test_multiple_skips_all_recorded(self) -> None:
        """All skipped iterations are recorded in step_history."""
        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_extractor"] = AsyncMock(return_value=None)

        result = await _run_outer_loop(mocks, outer_loop_steps=3)

        for step in result.step_history:
            assert step["was_skipped"] is True


# ===========================================================================
# REQ-P2O-029: Phase2Result.best_score never worse than initial_score
# ===========================================================================


@pytest.mark.unit
class TestScoreGuarantee:
    """Phase2Result.best_score is never worse than initial_score (REQ-P2O-029)."""

    async def test_no_improvement_returns_initial_score(self) -> None:
        """When no iteration improves, best_score equals initial_score."""
        inner_result = _make_inner_loop_result(best_score=0.70, improved=False)
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)

        result = await _run_outer_loop(mocks, outer_loop_steps=3, initial_score=0.80)

        assert result.best_score == 0.80

    async def test_no_improvement_returns_initial_solution(self) -> None:
        """When no iteration improves, best_solution is the initial solution."""
        initial = _make_solution(content="initial_baseline")
        inner_result = _make_inner_loop_result(best_score=0.70, improved=False)
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)

        result = await _run_outer_loop(
            mocks,
            outer_loop_steps=2,
            initial_score=0.80,
            initial_solution=initial,
        )

        assert result.best_solution.content == "initial_baseline"

    async def test_all_skipped_returns_initial_score(self) -> None:
        """When all iterations are skipped, initial_score is returned."""
        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_extractor"] = AsyncMock(return_value=None)

        result = await _run_outer_loop(mocks, outer_loop_steps=4, initial_score=0.85)

        assert result.best_score == 0.85

    async def test_maximize_score_never_decreases(self) -> None:
        """For maximize, best_score monotonically non-decreasing across steps."""
        scores = [0.82, 0.79, 0.88, 0.85]
        inner_results = [
            _make_inner_loop_result(best_score=s, improved=s > 0.80) for s in scores
        ]
        mock_inner = AsyncMock(side_effect=inner_results)
        mocks = _patch_outer_loop_dependencies()
        mocks["run_phase2_inner_loop"] = mock_inner

        result = await _run_outer_loop(mocks, outer_loop_steps=4, initial_score=0.80)

        # h_best: 0.80 -> 0.82 -> 0.82 (skip 0.79) -> 0.88 -> 0.88 (skip 0.85)
        assert result.best_score == 0.88

    async def test_minimize_score_never_increases(self) -> None:
        """For minimize, best_score monotonically non-increasing across steps."""
        task = _make_task(direction=MetricDirection.MINIMIZE)
        scores = [0.78, 0.82, 0.70, 0.75]
        inner_results = [
            _make_inner_loop_result(best_score=s, improved=s < 0.80) for s in scores
        ]
        mock_inner = AsyncMock(side_effect=inner_results)
        mocks = _patch_outer_loop_dependencies()
        mocks["run_phase2_inner_loop"] = mock_inner
        mocks["is_improvement_or_equal"] = lambda new, old, d: new <= old

        result = await _run_outer_loop(
            mocks, outer_loop_steps=4, initial_score=0.80, task=task
        )

        # h_best: 0.80 -> 0.78 -> 0.78 (skip 0.82) -> 0.70 -> 0.70 (skip 0.75)
        assert result.best_score == 0.70


# ===========================================================================
# REQ-P2O-028: Phase2Result correctly constructed with all fields
# ===========================================================================


@pytest.mark.unit
class TestPhase2ResultConstruction:
    """Phase2Result is correctly constructed with all required fields (REQ-P2O-028)."""

    async def test_has_ablation_summaries(self) -> None:
        """Phase2Result has ablation_summaries field."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=2)

        assert hasattr(result, "ablation_summaries")
        assert isinstance(result.ablation_summaries, list)

    async def test_has_refined_blocks(self) -> None:
        """Phase2Result has refined_blocks field."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=2)

        assert hasattr(result, "refined_blocks")
        assert isinstance(result.refined_blocks, list)

    async def test_has_best_solution(self) -> None:
        """Phase2Result has best_solution field."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=1)

        assert hasattr(result, "best_solution")
        assert isinstance(result.best_solution, SolutionScript)

    async def test_has_best_score(self) -> None:
        """Phase2Result has best_score field as float."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=1)

        assert hasattr(result, "best_score")
        assert isinstance(result.best_score, float)

    async def test_has_step_history(self) -> None:
        """Phase2Result has step_history field as list of dicts."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=1)

        assert hasattr(result, "step_history")
        assert isinstance(result.step_history, list)
        assert all(isinstance(s, dict) for s in result.step_history)


# ===========================================================================
# REQ-P2O-030: Step history record structure
# ===========================================================================


@pytest.mark.unit
class TestStepHistoryStructure:
    """Each step history record has all required fields (REQ-P2O-030)."""

    async def test_step_history_has_outer_step(self) -> None:
        """Each step record has outer_step field."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=2)

        for i, step in enumerate(result.step_history):
            assert "outer_step" in step
            assert step["outer_step"] == i

    async def test_step_history_has_ablation_summary(self) -> None:
        """Each step record has ablation_summary field."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=1)

        assert "ablation_summary" in result.step_history[0]
        assert isinstance(result.step_history[0]["ablation_summary"], str)

    async def test_step_history_has_code_block(self) -> None:
        """Each step record has code_block field."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=1)

        assert "code_block" in result.step_history[0]
        assert isinstance(result.step_history[0]["code_block"], str)

    async def test_step_history_has_plan(self) -> None:
        """Each step record has plan field."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=1)

        assert "plan" in result.step_history[0]
        assert isinstance(result.step_history[0]["plan"], str)

    async def test_step_history_has_inner_loop_attempts(self) -> None:
        """Each step record has inner_loop_attempts field."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=1)

        assert "inner_loop_attempts" in result.step_history[0]
        assert isinstance(result.step_history[0]["inner_loop_attempts"], list)

    async def test_step_history_has_best_score_after_step(self) -> None:
        """Each step record has best_score_after_step field."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=1)

        assert "best_score_after_step" in result.step_history[0]
        assert isinstance(result.step_history[0]["best_score_after_step"], float)

    async def test_step_history_has_was_skipped(self) -> None:
        """Each step record has was_skipped field."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=1)

        assert "was_skipped" in result.step_history[0]
        assert isinstance(result.step_history[0]["was_skipped"], bool)

    async def test_non_skipped_step_was_skipped_false(self) -> None:
        """Normal (non-skipped) steps have was_skipped=False."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=1)

        assert result.step_history[0]["was_skipped"] is False

    async def test_step_history_inner_loop_attempts_from_inner_result(self) -> None:
        """inner_loop_attempts in step history comes from InnerLoopResult.attempts."""
        attempts = [
            RefinementAttempt(
                plan="unique_plan_marker",
                score=0.88,
                code_block="block",
                was_improvement=True,
            )
        ]
        inner_result = _make_inner_loop_result(
            best_score=0.88, improved=True, attempts=attempts
        )
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)
        result = await _run_outer_loop(mocks, outer_loop_steps=1, initial_score=0.80)

        step_attempts = result.step_history[0]["inner_loop_attempts"]
        assert len(step_attempts) == 1
        assert step_attempts[0].plan == "unique_plan_marker"

    async def test_best_score_after_step_reflects_updated_h_best(self) -> None:
        """best_score_after_step reflects h_best AFTER the step completes."""
        inner_result = _make_inner_loop_result(best_score=0.90, improved=True)
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)

        result = await _run_outer_loop(mocks, outer_loop_steps=1, initial_score=0.80)

        assert result.step_history[0]["best_score_after_step"] == 0.90


# ===========================================================================
# REQ-P2O-024: s_t tracking -- solution only updates on improvement
# ===========================================================================


@pytest.mark.unit
class TestSolutionTracking:
    """s_t (current solution) only updates when inner loop improves h_best."""

    async def test_solution_updated_on_improvement(self) -> None:
        """After an improving iteration, s_t is the inner loop's best solution."""
        improved = _make_solution(content="iter1_improvement")
        inner_result = _make_inner_loop_result(
            best_solution=improved, best_score=0.90, improved=True
        )
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)

        result = await _run_outer_loop(mocks, outer_loop_steps=1, initial_score=0.80)

        assert result.best_solution.content == "iter1_improvement"

    async def test_solution_not_updated_on_worse_score(self) -> None:
        """After a non-improving iteration, s_t remains the previous best."""
        initial = _make_solution(content="original_content")
        inner_result = _make_inner_loop_result(best_score=0.70, improved=False)
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)

        result = await _run_outer_loop(
            mocks,
            outer_loop_steps=1,
            initial_score=0.80,
            initial_solution=initial,
        )

        assert result.best_solution.content == "original_content"

    async def test_solution_passed_to_ablation_after_improvement(self) -> None:
        """After improvement in iteration 1, iteration 2 ablation gets the improved solution."""
        improved_sol = _make_solution(content="solution_after_improvement")
        inner_result_1 = _make_inner_loop_result(
            best_solution=improved_sol, best_score=0.90, improved=True
        )
        inner_result_2 = _make_inner_loop_result(best_score=0.85, improved=False)

        mock_inner = AsyncMock(side_effect=[inner_result_1, inner_result_2])
        ablation_calls: list[Any] = []

        async def capture_ablation(*args: Any, **kwargs: Any) -> SolutionScript:
            ablation_calls.append((args, kwargs))
            return _make_ablation_script()

        mocks = _patch_outer_loop_dependencies()
        mocks["run_phase2_inner_loop"] = mock_inner
        mocks["invoke_ablation"] = AsyncMock(side_effect=capture_ablation)

        await _run_outer_loop(mocks, outer_loop_steps=2, initial_score=0.80)

        # Second ablation call should receive the improved solution
        second_call = ablation_calls[1]
        second_sol = second_call[0][0] if second_call[0] else second_call[1]["solution"]
        assert second_sol.content == "solution_after_improvement"

    async def test_solution_not_passed_to_ablation_on_no_improvement(self) -> None:
        """After no improvement in iteration 1, iteration 2 ablation gets the original."""
        initial = _make_solution(content="original_stays")
        inner_result_1 = _make_inner_loop_result(best_score=0.70, improved=False)
        inner_result_2 = _make_inner_loop_result(best_score=0.70, improved=False)

        mock_inner = AsyncMock(side_effect=[inner_result_1, inner_result_2])
        ablation_calls: list[Any] = []

        async def capture_ablation(*args: Any, **kwargs: Any) -> SolutionScript:
            ablation_calls.append((args, kwargs))
            return _make_ablation_script()

        mocks = _patch_outer_loop_dependencies()
        mocks["run_phase2_inner_loop"] = mock_inner
        mocks["invoke_ablation"] = AsyncMock(side_effect=capture_ablation)

        await _run_outer_loop(
            mocks,
            outer_loop_steps=2,
            initial_score=0.80,
            initial_solution=initial,
        )

        # Second ablation call should still receive the original
        second_call = ablation_calls[1]
        second_sol = second_call[0][0] if second_call[0] else second_call[1]["solution"]
        assert second_sol.content == "original_stays"


# ===========================================================================
# REQ-P2O-043: Immutable input solution
# ===========================================================================


@pytest.mark.unit
class TestImmutableInputSolution:
    """run_phase2_outer_loop does not mutate initial_solution (REQ-P2O-043)."""

    async def test_initial_solution_content_unchanged(self) -> None:
        """initial_solution.content is not modified after the loop."""
        original_content = "immutable_content_marker"
        initial = _make_solution(content=original_content)
        content_before = initial.content

        mocks = _patch_outer_loop_dependencies()
        await _run_outer_loop(mocks, outer_loop_steps=2, initial_solution=initial)

        assert initial.content == content_before
        assert initial.content == original_content

    async def test_initial_solution_score_unchanged(self) -> None:
        """initial_solution.score is not modified after the loop."""
        initial = _make_solution(score=0.42)
        score_before = initial.score

        mocks = _patch_outer_loop_dependencies()
        await _run_outer_loop(mocks, outer_loop_steps=2, initial_solution=initial)

        assert initial.score == score_before


# ===========================================================================
# REQ-P2O-041: Monotonic best score
# ===========================================================================


@pytest.mark.unit
class TestMonotonicBestScore:
    """h_best is monotonically non-decreasing (maximize) or non-increasing (minimize)."""

    async def test_step_history_scores_monotonic_maximize(self) -> None:
        """For maximize, best_score_after_step is non-decreasing across steps."""
        scores = [0.82, 0.79, 0.88, 0.85]
        inner_results = [
            _make_inner_loop_result(best_score=s, improved=s > 0.80) for s in scores
        ]
        mock_inner = AsyncMock(side_effect=inner_results)
        mocks = _patch_outer_loop_dependencies()
        mocks["run_phase2_inner_loop"] = mock_inner

        result = await _run_outer_loop(mocks, outer_loop_steps=4, initial_score=0.80)

        step_scores = [s["best_score_after_step"] for s in result.step_history]
        # Should be non-decreasing
        for i in range(1, len(step_scores)):
            assert step_scores[i] >= step_scores[i - 1]

    async def test_step_history_scores_monotonic_minimize(self) -> None:
        """For minimize, best_score_after_step is non-increasing across steps."""
        task = _make_task(direction=MetricDirection.MINIMIZE)
        scores = [0.78, 0.82, 0.70, 0.75]
        inner_results = [
            _make_inner_loop_result(best_score=s, improved=s < 0.80) for s in scores
        ]
        mock_inner = AsyncMock(side_effect=inner_results)
        mocks = _patch_outer_loop_dependencies()
        mocks["run_phase2_inner_loop"] = mock_inner
        mocks["is_improvement_or_equal"] = lambda new, old, d: new <= old

        result = await _run_outer_loop(
            mocks, outer_loop_steps=4, initial_score=0.80, task=task
        )

        step_scores = [s["best_score_after_step"] for s in result.step_history]
        for i in range(1, len(step_scores)):
            assert step_scores[i] <= step_scores[i - 1]


# ===========================================================================
# Multi-iteration end-to-end scenarios
# ===========================================================================


@pytest.mark.unit
class TestMultiIterationScenarios:
    """End-to-end scenarios testing multiple outer loop iterations."""

    async def test_progressive_improvement_across_3_iterations(self) -> None:
        """Score progressively improves: 0.80 -> 0.82 -> 0.85 -> 0.90."""
        scores = [0.82, 0.85, 0.90]
        solutions = [
            _make_solution(content=f"solution_v{i}", phase=SolutionPhase.REFINED)
            for i in range(3)
        ]
        inner_results = [
            _make_inner_loop_result(
                best_solution=solutions[i],
                best_score=scores[i],
                improved=True,
            )
            for i in range(3)
        ]
        mock_inner = AsyncMock(side_effect=inner_results)
        mocks = _patch_outer_loop_dependencies()
        mocks["run_phase2_inner_loop"] = mock_inner

        result = await _run_outer_loop(mocks, outer_loop_steps=3, initial_score=0.80)

        assert result.best_score == 0.90
        assert result.best_solution.content == "solution_v2"
        assert len(result.ablation_summaries) == 3
        assert len(result.refined_blocks) == 3

    async def test_improvement_then_no_improvement(self) -> None:
        """First iteration improves, second does not."""
        improved = _make_solution(content="improved_iter1")
        inner_1 = _make_inner_loop_result(
            best_solution=improved, best_score=0.90, improved=True
        )
        inner_2 = _make_inner_loop_result(best_score=0.85, improved=False)

        mock_inner = AsyncMock(side_effect=[inner_1, inner_2])
        mocks = _patch_outer_loop_dependencies()
        mocks["run_phase2_inner_loop"] = mock_inner

        result = await _run_outer_loop(mocks, outer_loop_steps=2, initial_score=0.80)

        assert result.best_score == 0.90
        assert result.best_solution.content == "improved_iter1"

    async def test_no_improvement_then_improvement(self) -> None:
        """First iteration does not improve, second does."""
        improved = _make_solution(content="improved_iter2")
        inner_1 = _make_inner_loop_result(best_score=0.70, improved=False)
        inner_2 = _make_inner_loop_result(
            best_solution=improved, best_score=0.90, improved=True
        )

        mock_inner = AsyncMock(side_effect=[inner_1, inner_2])
        mocks = _patch_outer_loop_dependencies()
        mocks["run_phase2_inner_loop"] = mock_inner

        result = await _run_outer_loop(mocks, outer_loop_steps=2, initial_score=0.80)

        assert result.best_score == 0.90
        assert result.best_solution.content == "improved_iter2"

    async def test_skip_then_improve(self) -> None:
        """First iteration skipped, second iteration improves."""
        improved = _make_solution(content="improved_after_skip")
        inner_result = _make_inner_loop_result(
            best_solution=improved, best_score=0.90, improved=True
        )

        extractor_calls = [None, _make_extractor_output()]
        mock_extractor = AsyncMock(side_effect=extractor_calls)

        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)
        mocks["invoke_extractor"] = mock_extractor

        result = await _run_outer_loop(mocks, outer_loop_steps=2, initial_score=0.80)

        assert result.best_score == 0.90
        assert result.step_history[0]["was_skipped"] is True
        assert result.step_history[1]["was_skipped"] is False

    async def test_all_iterations_fail_returns_initial(self) -> None:
        """When all iterations skip, returns initial solution and score."""
        initial = _make_solution(content="fallback_baseline")
        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_extractor"] = AsyncMock(return_value=None)

        result = await _run_outer_loop(
            mocks,
            outer_loop_steps=3,
            initial_score=0.80,
            initial_solution=initial,
        )

        assert result.best_score == 0.80
        assert result.best_solution.content == "fallback_baseline"


# ===========================================================================
# Edge cases
# ===========================================================================


@pytest.mark.unit
class TestOuterLoopEdgeCases:
    """Edge cases for run_phase2_outer_loop."""

    async def test_single_iteration(self) -> None:
        """Works correctly with outer_loop_steps=1."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=1)

        assert len(result.step_history) == 1
        assert len(result.ablation_summaries) == 1
        assert len(result.refined_blocks) == 1

    async def test_initial_score_zero(self) -> None:
        """Works with initial_score=0.0."""
        inner_result = _make_inner_loop_result(best_score=0.50, improved=True)
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)

        result = await _run_outer_loop(mocks, outer_loop_steps=1, initial_score=0.0)

        assert result.best_score == 0.50

    async def test_negative_initial_score(self) -> None:
        """Works with negative initial_score."""
        inner_result = _make_inner_loop_result(best_score=-0.30, improved=True)
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)

        result = await _run_outer_loop(mocks, outer_loop_steps=1, initial_score=-0.50)

        assert result.best_score == -0.30

    async def test_empty_ablation_output_does_not_crash(self) -> None:
        """When ablation execution returns empty output, loop continues."""
        mocks = _patch_outer_loop_dependencies(
            execute_ablation_rv=("", ""),
        )

        result = await _run_outer_loop(mocks, outer_loop_steps=1)

        assert isinstance(result, Phase2Result)

    async def test_step_history_code_block_empty_on_skip(self) -> None:
        """Skipped step has empty string for code_block in step history."""
        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_extractor"] = AsyncMock(return_value=None)

        result = await _run_outer_loop(mocks, outer_loop_steps=1)

        assert result.step_history[0]["code_block"] == ""

    async def test_step_history_plan_empty_on_skip(self) -> None:
        """Skipped step has empty string for plan in step history."""
        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_extractor"] = AsyncMock(return_value=None)

        result = await _run_outer_loop(mocks, outer_loop_steps=1)

        assert result.step_history[0]["plan"] == ""


# ===========================================================================
# Hypothesis: Property-based tests for outer loop invariants
# ===========================================================================


@pytest.mark.unit
class TestOuterLoopPropertyBased:
    """Property-based tests for run_phase2_outer_loop invariants."""

    @given(
        t_steps=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=15)
    async def test_step_history_length_equals_t_steps(self, t_steps: int) -> None:
        """step_history always has exactly T entries regardless of skips."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=t_steps)

        assert len(result.step_history) == t_steps

    @given(
        t_steps=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=15)
    async def test_ablation_summaries_length_matches_steps(self, t_steps: int) -> None:
        """ablation_summaries has entries for all steps (including skipped)."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=t_steps)

        # In a normal run (no skips), all summaries should be present
        assert len(result.ablation_summaries) == t_steps

    @given(
        t_steps=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=15)
    async def test_refined_blocks_length_matches_steps(self, t_steps: int) -> None:
        """refined_blocks has entries for all steps (including skipped)."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=t_steps)

        assert len(result.refined_blocks) == t_steps

    @given(
        initial_score=st.floats(
            min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=20)
    async def test_best_score_at_least_initial_score_maximize(
        self, initial_score: float
    ) -> None:
        """For maximize, Phase2Result.best_score >= initial_score always."""
        # Inner loop returns score worse than initial
        inner_result = _make_inner_loop_result(
            best_score=initial_score - 1.0, improved=False
        )
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)

        result = await _run_outer_loop(
            mocks, outer_loop_steps=1, initial_score=initial_score
        )

        assert result.best_score >= initial_score

    @given(
        initial_score=st.floats(
            min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=20)
    async def test_best_score_at_most_initial_score_minimize(
        self, initial_score: float
    ) -> None:
        """For minimize, Phase2Result.best_score <= initial_score always."""
        task = _make_task(direction=MetricDirection.MINIMIZE)
        inner_result = _make_inner_loop_result(
            best_score=initial_score + 1.0, improved=False
        )
        mocks = _patch_outer_loop_dependencies(inner_loop_rv=inner_result)
        mocks["is_improvement_or_equal"] = lambda new, old, d: new <= old

        result = await _run_outer_loop(
            mocks, outer_loop_steps=1, initial_score=initial_score, task=task
        )

        assert result.best_score <= initial_score

    @given(
        t_steps=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=15)
    async def test_step_history_outer_step_indices_sequential(
        self, t_steps: int
    ) -> None:
        """step_history outer_step values are 0, 1, ..., T-1."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=t_steps)

        indices = [s["outer_step"] for s in result.step_history]
        assert indices == list(range(t_steps))

    @given(
        t_steps=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=15)
    async def test_refined_blocks_outer_step_sequential(self, t_steps: int) -> None:
        """refined_blocks[t].outer_step == t for all t."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=t_steps)

        for t, block in enumerate(result.refined_blocks):
            assert block.outer_step == t

    @given(
        t_steps=st.integers(min_value=1, max_value=5),
        initial_score=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=15)
    async def test_step_history_scores_monotonic_nondecreasing(
        self, t_steps: int, initial_score: float
    ) -> None:
        """best_score_after_step is monotonically non-decreasing for maximize."""
        # Use the default mocks (inner loop returns 0.90 which is likely >= initial)
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(
            mocks, outer_loop_steps=t_steps, initial_score=initial_score
        )

        step_scores = [s["best_score_after_step"] for s in result.step_history]
        for i in range(1, len(step_scores)):
            assert step_scores[i] >= step_scores[i - 1]


# ===========================================================================
# Hypothesis: Property-based tests for skip behavior
# ===========================================================================


@pytest.mark.unit
class TestSkipBehaviorPropertyBased:
    """Property-based tests for skip scenarios."""

    @given(
        t_steps=st.integers(min_value=1, max_value=6),
        initial_score=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=15)
    async def test_all_skipped_preserves_initial_score(
        self, t_steps: int, initial_score: float
    ) -> None:
        """When all iterations skip, best_score equals initial_score."""
        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_extractor"] = AsyncMock(return_value=None)

        result = await _run_outer_loop(
            mocks, outer_loop_steps=t_steps, initial_score=initial_score
        )

        assert result.best_score == initial_score

    @given(
        t_steps=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=15)
    async def test_all_skipped_all_was_skipped_true(self, t_steps: int) -> None:
        """When all iterations skip, all step_history entries have was_skipped=True."""
        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_extractor"] = AsyncMock(return_value=None)

        result = await _run_outer_loop(mocks, outer_loop_steps=t_steps)

        for step in result.step_history:
            assert step["was_skipped"] is True

    @given(
        t_steps=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=15)
    async def test_all_skipped_inner_loop_never_called(self, t_steps: int) -> None:
        """When all iterations skip, inner loop is never invoked."""
        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_extractor"] = AsyncMock(return_value=None)

        await _run_outer_loop(mocks, outer_loop_steps=t_steps)

        mocks["run_phase2_inner_loop"].assert_not_called()


# ===========================================================================
# Ablation failure handling
# ===========================================================================


@pytest.mark.unit
class TestAblationFailureHandling:
    """Tests for ablation agent failure scenarios."""

    async def test_ablation_returns_none_proceeds_with_empty(self) -> None:
        """When invoke_ablation returns None, summarize still called (with empty)."""
        mocks = _patch_outer_loop_dependencies()
        mocks["invoke_ablation"] = AsyncMock(return_value=None)

        result = await _run_outer_loop(mocks, outer_loop_steps=1)

        # Summarize should still be called (with empty strings for code/output)
        assert isinstance(result, Phase2Result)
        assert len(result.step_history) == 1

    async def test_empty_ablation_output_passed_to_summarize(self) -> None:
        """When ablation execution produces empty output, empty strings pass to summarize."""
        mocks = _patch_outer_loop_dependencies(
            execute_ablation_rv=("", ""),
        )

        await _run_outer_loop(mocks, outer_loop_steps=1)

        # Verify summarize was called
        mocks["invoke_summarize"].assert_called_once()


# ===========================================================================
# Code block validation in the pipeline
# ===========================================================================


@pytest.mark.unit
class TestCodeBlockValidation:
    """Tests for code block validation within the outer loop pipeline."""

    async def test_validation_passes_proceeds_to_inner_loop(self) -> None:
        """When validate_code_block returns True, inner loop is called."""
        mocks = _patch_outer_loop_dependencies(validate_code_block_rv=True)

        await _run_outer_loop(mocks, outer_loop_steps=1)

        mocks["run_phase2_inner_loop"].assert_called_once()

    async def test_validation_fails_skips_inner_loop(self) -> None:
        """When validate_code_block returns False, inner loop is skipped."""
        mocks = _patch_outer_loop_dependencies()
        mocks["validate_code_block"] = lambda code_block, solution: False

        await _run_outer_loop(mocks, outer_loop_steps=1)

        mocks["run_phase2_inner_loop"].assert_not_called()

    async def test_validation_receives_correct_args(self) -> None:
        """validate_code_block receives code_block and current solution."""
        validation_calls: list[tuple[str, SolutionScript]] = []

        def capture_validate(code_block: str, solution: SolutionScript) -> bool:
            validation_calls.append((code_block, solution))
            return True

        extractor_output = _make_extractor_output(code_block="validate_this_block")
        solution = _make_solution(content="code validate_this_block end")
        mocks = _patch_outer_loop_dependencies(
            invoke_extractor_rv=extractor_output,
        )
        mocks["validate_code_block"] = capture_validate

        await _run_outer_loop(mocks, outer_loop_steps=1, initial_solution=solution)

        assert len(validation_calls) == 1
        assert validation_calls[0][0] == "validate_this_block"


# ===========================================================================
# Session ID parameter
# ===========================================================================


@pytest.mark.unit
class TestSessionIdParameter:
    """Tests that session_id parameter is accepted."""

    async def test_accepts_session_id_string(self) -> None:
        """run_phase2_outer_loop accepts a session_id parameter."""
        mocks = _patch_outer_loop_dependencies()
        result = await _run_outer_loop(mocks, outer_loop_steps=1, session_id="path-0")

        assert isinstance(result, Phase2Result)

    async def test_different_session_ids_produce_results(self) -> None:
        """Different session_id values are accepted without error."""
        for sid in ["path-0", "path-1", "test-session", "phase-2-outer"]:
            mocks = _patch_outer_loop_dependencies()
            result = await _run_outer_loop(mocks, outer_loop_steps=1, session_id=sid)
            assert isinstance(result, Phase2Result)
