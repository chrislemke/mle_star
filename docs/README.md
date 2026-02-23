# MLE-STAR

MLE-STAR (Machine Learning Engineering with Systematic Targeted Ablation and Refinement) is an autonomous ML engineering agent that solves Kaggle competitions end-to-end. It implements a multi-phase pipeline with 14 specialized AI agents that retrieve models, generate and refine solutions, run ablation studies, construct ensembles, and produce validated submissions -- all with built-in safety guardrails and time budgeting.

## How It Works

The pipeline progresses through four phases plus finalization:

```
Phase 1: Model Retrieval & Initial Solution
    |
    |   Retrieve M models -> Generate candidates -> Evaluate -> Merge -> Safety checks
    |
    v
Phase 2: Targeted Refinement (L parallel paths)
    |
    |   For each path, repeat T outer steps:
    |     Ablation study -> Summarize -> Extract code blocks ->
    |       For each block, repeat K inner steps:
    |         Plan refinement -> Implement -> Evaluate -> Keep if improved
    |
    v
Phase 3: Ensemble Construction (skipped if L=1)
    |
    |   Combine L refined solutions over R rounds:
    |     Plan ensemble strategy -> Implement -> Evaluate -> Keep best
    |
    v
Finalization
    |
    |   Remove subsampling -> Generate test submission -> Leakage check ->
    |   Evaluate -> Verify submission -> Contamination check
    |
    v
  Result (submission.csv + scores + diagnostics)
```

## Pipeline Phases

### Phase 1: Model Retrieval and Initial Solution

Phase 1 produces the initial solution `s_0` that seeds all subsequent refinement.

1. **Retrieve** -- `A_retriever` searches the web for M candidate ML models suitable for the competition task, returning structured `(model_name, example_code)` pairs.
2. **Generate** -- For each retrieved model, `A_init` generates a complete solution script. Each candidate is evaluated and scored.
3. **Sort and Merge** -- Candidates are sorted by score. `A_merger` iteratively merges the best solution with each subsequent candidate, keeping the merge only if the score improves or stays equal.
4. **Safety Checks** -- The merged solution passes through `A_data` (data usage verification) and `A_leakage` (data leakage detection and correction). If a safety agent modifies the solution, it is re-evaluated to confirm the fix did not degrade performance.

### Phase 2: Targeted Refinement

Phase 2 runs L independent paths in parallel (via `asyncio.gather`), each starting from a deep copy of `s_0`. Each path executes T outer loop steps:

**Outer Loop** (per step t):
1. `A_ablation` generates and executes an ablation script to identify which components contribute most to performance.
2. `A_summarize` distills the ablation output into a concise analysis.
3. `A_extractor` reads the summary and the current solution, then extracts a targeted code block `c_t` with a refinement plan.

**Inner Loop** (K iterations per code block):
1. `A_planner` creates a detailed refinement plan for `c_t`, informed by previous plans and their outcomes.
2. `A_coder` implements the plan, producing a modified solution.
3. The candidate is checked for leakage (`A_leakage`) and evaluated. If the score improves, it becomes the new best.
4. On evaluation errors, `A_debugger` attempts to fix the script (up to `max_debug_attempts` retries).

### Phase 3: Ensemble Construction

When L > 1, Phase 3 combines the L refined solutions into an ensemble over R rounds:

1. `A_ens_planner` proposes an ensemble strategy (stacking, blending, voting, etc.) given all L solutions and the history of previous attempts.
2. `A_ensembler` implements the strategy as a single script.
3. The ensemble script passes through leakage checking and evaluation with debug retry.
4. The best ensemble across all R rounds is selected (ties broken by last occurrence).

When L = 1, Phase 3 is skipped entirely and the single refined solution proceeds to finalization.

### Finalization

1. **Remove Subsampling** -- `A_test` identifies and removes any subsampling logic that was used during refinement to speed up evaluation.
2. **Generate Test Submission** -- `A_test` adapts the solution to produce a submission on the full test set.
3. **Leakage Check** -- Final `A_leakage` pass on the submission script.
4. **Evaluate** -- The final script is executed and scored. On failure, `A_debugger` retries.
5. **Verify Submission** -- Checks that the submission file exists, is non-empty, and has the expected format.
6. **Contamination Check** -- `A_test` compares the solution against reference discussions to detect data contamination (result: "Novel" or "Same").
7. **Fallback** -- If evaluation fails or the submission is invalid, the pipeline falls back to the pre-finalization solution.

## The 14 Agents

| Agent | Type | Role | Tools | Output Format |
|---|---|---|---|---|
| A_retriever | `retriever` | Search web for ML models | WebSearch, WebFetch | Structured JSON (`RetrieverOutput`) |
| A_init | `init` | Generate initial solution scripts | Bash, Edit, Write, Read | Code block |
| A_merger | `merger` | Merge candidate solutions | Bash, Edit, Write, Read | Code block |
| A_ablation | `ablation` | Run ablation studies | Bash, Edit, Write, Read | Code block |
| A_summarize | `summarize` | Summarize ablation results | Read | Free text |
| A_extractor | `extractor` | Extract code blocks for refinement | Read | Structured JSON (`ExtractorOutput`) |
| A_planner | `planner` | Plan code refinements | Read | Free text |
| A_coder | `coder` | Implement refinement plans | Bash, Edit, Write, Read | Code block |
| A_ens_planner | `ens_planner` | Plan ensemble strategies | Read | Free text |
| A_ensembler | `ensembler` | Implement ensemble scripts | Bash, Edit, Write, Read | Code block |
| A_debugger | `debugger` | Fix failing scripts | Bash, Edit, Write, Read | Code block |
| A_leakage | `leakage` | Detect/correct data leakage | Read | Structured JSON (`LeakageDetectionOutput`) |
| A_data | `data` | Verify correct data usage | Read | Free text |
| A_test | `test` | Generate test submissions, check contamination | Read | Code block / Structured JSON |

## Installation

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The pipeline also requires the [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude`) to be available on PATH. The CLI check runs at startup and exits with an error if not found.

## Usage

```bash
uv run mle_star --task task.yaml [--config config.yaml]
```

**Arguments:**

- `--task` (required) -- Path to a task description YAML file.
- `--config` (optional) -- Path to a pipeline configuration YAML file. Defaults are used if omitted.

**Output:**

On success, the CLI prints:
- Submission file path
- Total duration
- Final score (if available)

Exit code 0 on success, 1 on error.

## Configuration

### PipelineConfig

All hyperparameters have paper-specified defaults:

```yaml
# config.yaml
num_retrieved_models: 4       # M -- candidate models to retrieve
outer_loop_steps: 4           # T -- outer refinement loop iterations
inner_loop_steps: 4           # K -- inner refinement loop iterations per block
num_parallel_solutions: 2     # L -- parallel solution paths (1 = no ensemble)
ensemble_rounds: 5            # R -- ensemble strategy attempts
time_limit_seconds: 86400     # Total pipeline time budget (24 hours)
subsample_limit: 30000        # Max training samples during refinement
max_debug_attempts: 3         # Debug retries before fallback

permission_mode: "dangerously-skip-permissions"
model: "opus"               # Claude model to use
log_level: "INFO"
log_file: null                # Optional log file path
```

### Environment Variable Overrides

These override **default** config values only (explicit YAML values take precedence):

| Variable | Overrides | Notes |
|---|---|---|
| `MLE_STAR_MODEL` | `model` | Claude model identifier |
| `MLE_STAR_LOG_LEVEL` | `log_level` | Logging level string |
| `MLE_STAR_TIME_LIMIT` | `time_limit_seconds` | Time limit in seconds (must be >= 1) |

## Task Description

The `--task` YAML file defines the competition:

```yaml
# task.yaml
competition_id: "titanic"
task_type: "classification"        # classification, regression, image_classification,
                                   # image_to_image, text_classification,
                                   # audio_classification, sequence_to_sequence, tabular
data_modality: "tabular"           # tabular, image, text, audio, mixed
evaluation_metric: "accuracy"
metric_direction: "maximize"       # maximize or minimize
description: |
  Predict survival on the Titanic.
  Training data: train.csv with columns PassengerId, Survived, Pclass, ...
  Test data: test.csv (predict Survived)
  Submission format: PassengerId, Survived
target_column: "Survived"          # Column to predict (optional, aids agents)
data_dir: "./input"                # Must exist with data before running
output_dir: "./final"              # Path for submission output
```

## Project Structure

```
src/mle_star/
  __init__.py             Package init
  cli.py                  CLI entry point (--task, --config)
  orchestrator.py         Pipeline orchestrator: phase dispatch, parallelism,
                          time control, SDK client lifecycle, hooks
  models.py               Pydantic models, enums, configs, agent definitions
  scoring.py              Score parsing, comparison (is_improvement,
                          is_improvement_or_equal), ScoreFunction protocol
  execution.py            Script execution harness: env setup, async subprocess,
                          output parsing, evaluation pipeline, subsampling,
                          submission verification, batch evaluation, ranking
  safety.py               Safety agents: debugger, leakage, data verification,
                          code block extraction
  phase1.py               Phase 1: retrieve_models, generate_candidate,
                          merge_solutions, run_phase1
  phase2_inner.py         Phase 2 inner loop: coder/planner agents,
                          run_phase2_inner_loop with safety integration
  phase2_outer.py         Phase 2 outer loop: ablation, summarize, extractor
                          agents, run_phase2_outer_loop
  phase3.py               Phase 3: ensemble planner/ensembler agents, run_phase3
  finalization.py         Finalization: subsampling removal, test submission,
                          contamination check, run_finalization
  prompts/                YAML prompt templates (one per agent)
    __init__.py           PromptRegistry: loads and renders templates
    retriever.yaml        ... through test.yaml (14 templates)

tests/                    pytest test suite (90% coverage minimum)
```

## Development

### Testing

```bash
uv run pytest
```

Coverage is enforced at 90% minimum via `addopts` in `pyproject.toml`. Tests use [hypothesis](https://hypothesis.readthedocs.io/) for property-based testing and `pytest-asyncio` for async test support.

### Linting and Formatting

```bash
uv run ruff format
uv run ruff check . --fix
```

### Type Checking

```bash
uv run mypy --config-file=pyproject.toml src tests
```

### Complexity Analysis

```bash
uv run xenon --max-average=B --max-modules=B --max-absolute=B src
```

All modules are kept at or below xenon complexity grade B through function decomposition.

### Security Scanning

```bash
uv run bandit -c pyproject.toml -r .
```

### Dependency Audit

```bash
uv run pip-audit
```

### Mutation Testing

```bash
uv run mutmut run
uv run mutmut browse   # Inspect surviving mutants
```

## Architecture Notes

### Immutability

All Pydantic models use `ConfigDict(frozen=True)` except `SolutionScript` (needs mutable `score` after evaluation) and `PipelineState` (runtime introspection). Configuration and results are immutable once created.

### Safety Integration

Safety checks are woven throughout the pipeline, not bolted on at the end:
- **A_leakage** runs before every evaluation in Phase 2 inner loop, after every ensemble in Phase 3, and during finalization.
- **A_data** runs once after Phase 1 merging to verify correct data usage.
- **A_debugger** retries failed evaluations with error tracebacks across all phases.
- Graceful degradation: safety agent failures are caught and logged, never crashing the pipeline. The pre-check solution is preserved as fallback.

### Error Recovery

- Failed Phase 2 paths fall back to the Phase 1 solution.
- Failed Phase 3 falls back to the best Phase 2 solution.
- Failed finalization falls back to the pre-finalization solution.
- Each fallback is logged with the reason for failure.

### Time Budgeting

- A global deadline is computed at pipeline start (`time_limit_seconds`).
- Remaining time after Phase 1 is distributed proportionally (P2: 65%, P3: 15%, Fin: 10%).
- Per-path Phase 2 budget = Phase 2 budget / L.
- If the deadline expires before Phase 2 starts, the pipeline skips directly to finalization with the Phase 1 solution.

### Parallelism

Phase 2 runs L independent solution paths concurrently via `asyncio.gather`. Each path gets a deep copy of the initial solution and its own working directory (`./work/path-{i}/`). An `asyncio.Semaphore` limits concurrent SDK sessions. Overtime paths are cancelled and fall back to the Phase 1 solution.

### Idempotency

Each `run_pipeline()` call creates fresh local state (session maps, failure counts). No global mutable state is shared across invocations.
