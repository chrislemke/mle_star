"""Phase 1: agent invocations for model retrieval, candidate generation, and merging.

Implements ``retrieve_models``, ``generate_candidate``, and ``merge_solutions``
for the initial solution generation pipeline (Algorithm 1, lines 1-17).
A_retriever searches the web for M effective models, A_init generates a
candidate solution script per model, and A_merger integrates a reference
solution into a base solution via ensemble.

Helper ``parse_retriever_output`` validates and deserializes the structured
JSON response from A_retriever.

Refs:
    SRS 04a — Phase 1 Agents (REQ-P1-001 through REQ-P1-017).
    IMPLEMENTATION_PLAN.md Task 27.
"""

from __future__ import annotations

import logging
from typing import Any

from mle_star.models import (
    AgentType,
    PipelineConfig,
    RetrievedModel,
    RetrieverOutput,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
)
from mle_star.prompts import PromptRegistry
from mle_star.safety import extract_code_block

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A_retriever — Model retrieval (REQ-P1-001 to REQ-P1-007)
# ---------------------------------------------------------------------------


def parse_retriever_output(response: str) -> RetrieverOutput:
    """Parse the A_retriever structured JSON response (REQ-P1-004).

    Deserializes *response* into a ``RetrieverOutput`` model using
    ``model_validate_json``.  Raises ``ValueError`` with a descriptive
    message (including the first 500 characters of the raw response)
    on any parsing or validation failure.

    Args:
        response: Raw JSON string from the retriever agent.

    Returns:
        A validated ``RetrieverOutput`` instance.

    Raises:
        ValueError: If the response is not valid JSON or does not
            conform to the ``RetrieverOutput`` schema.
    """
    try:
        return RetrieverOutput.model_validate_json(response)
    except Exception as exc:
        truncated = response[:500]
        msg = f"Failed to parse retriever output: {exc}. Raw response: {truncated}"
        raise ValueError(msg) from exc


def _filter_valid_models(models: list[RetrievedModel]) -> list[RetrievedModel]:
    """Exclude models with empty model_name or example_code (REQ-P1-006).

    Models that fail validation are logged as warnings and excluded.

    Args:
        models: Raw list of retrieved models.

    Returns:
        Filtered list containing only models with non-empty fields.
    """
    valid: list[RetrievedModel] = []
    for model in models:
        if not model.model_name.strip():
            logger.warning(
                "Excluding model with empty model_name: %r",
                model,
            )
            continue
        if not model.example_code.strip():
            logger.warning(
                "Excluding model '%s' with empty example_code",
                model.model_name,
            )
            continue
        valid.append(model)
    return valid


async def retrieve_models(
    task: TaskDescription,
    config: PipelineConfig,
    client: Any,
) -> list[RetrievedModel]:
    """Invoke A_retriever to find M effective models for the task (REQ-P1-007).

    Renders the retriever prompt template with the task description and
    requested model count, sends it to the A_retriever agent via the SDK
    client with structured output, parses and validates the response, and
    filters out models with empty fields.

    Args:
        task: Task description providing competition context.
        config: Pipeline configuration (provides ``num_retrieved_models``).
        client: SDK client for agent invocation.

    Returns:
        A list of validated ``RetrievedModel`` instances (1 to M).

    Raises:
        ValueError: If zero valid models remain after filtering.
    """
    registry = PromptRegistry()
    template = registry.get(AgentType.RETRIEVER)
    prompt = template.render(
        task_description=task.description,
        M=config.num_retrieved_models,
    )

    response: str = await client.send_message(
        agent_type=str(AgentType.RETRIEVER),
        message=prompt,
    )

    output = parse_retriever_output(response)
    valid_models = _filter_valid_models(output.models)

    if len(valid_models) == 0:
        msg = "A_retriever returned zero models"
        raise ValueError(msg)

    if len(valid_models) < config.num_retrieved_models:
        logger.warning(
            "A_retriever returned %d models (requested %d); proceeding with available",
            len(valid_models),
            config.num_retrieved_models,
        )

    return valid_models


# ---------------------------------------------------------------------------
# A_init — Candidate solution generation (REQ-P1-008 to REQ-P1-012)
# ---------------------------------------------------------------------------


async def generate_candidate(
    task: TaskDescription,
    model: RetrievedModel,
    config: PipelineConfig,
    client: Any,
) -> SolutionScript | None:
    """Invoke A_init to generate a candidate solution for a model (REQ-P1-012).

    Renders the init prompt template with the task description and model
    details, sends it to the A_init agent, extracts the code block from
    the response, and constructs a ``SolutionScript`` with
    ``phase=SolutionPhase.INIT``.

    Args:
        task: Task description providing competition context.
        model: The retrieved model to base the solution on.
        config: Pipeline configuration (unused directly but available
            for future extensions).
        client: SDK client for agent invocation.

    Returns:
        A ``SolutionScript`` with the generated code, or ``None`` if the
        agent response is empty or extraction yields no code.
    """
    registry = PromptRegistry()
    template = registry.get(AgentType.INIT)
    prompt = template.render(
        task_description=task.description,
        model_name=model.model_name,
        example_code=model.example_code,
    )

    response: str = await client.send_message(
        agent_type=str(AgentType.INIT),
        message=prompt,
    )

    extracted = extract_code_block(response)
    if not extracted.strip():
        logger.warning("A_init returned empty code for model '%s'", model.model_name)
        return None

    return SolutionScript(
        content=extracted,
        phase=SolutionPhase.INIT,
        source_model=model.model_name,
    )


# ---------------------------------------------------------------------------
# A_merger — Solution merging (REQ-P1-013 to REQ-P1-017)
# ---------------------------------------------------------------------------


async def merge_solutions(
    base: SolutionScript,
    reference: SolutionScript,
    config: PipelineConfig,
    client: Any,
) -> SolutionScript | None:
    """Invoke A_merger to integrate a reference solution into the base (REQ-P1-017).

    Renders the merger prompt template with the base and reference solution
    source code, sends it to the A_merger agent, extracts the code block
    from the response, and constructs a ``SolutionScript`` with
    ``phase=SolutionPhase.MERGED``.

    Args:
        base: Current best solution (code base).
        reference: Next-ranked candidate solution to integrate.
        config: Pipeline configuration (unused directly but available
            for future extensions).
        client: SDK client for agent invocation.

    Returns:
        A ``SolutionScript`` with the merged code, or ``None`` if the
        agent response is empty or extraction yields no code.
    """
    registry = PromptRegistry()
    template = registry.get(AgentType.MERGER)
    prompt = template.render(
        base_code=base.content,
        reference_code=reference.content,
    )

    response: str = await client.send_message(
        agent_type=str(AgentType.MERGER),
        message=prompt,
    )

    extracted = extract_code_block(response)
    if not extracted.strip():
        logger.warning("A_merger returned empty code")
        return None

    return SolutionScript(
        content=extracted,
        phase=SolutionPhase.MERGED,
    )
