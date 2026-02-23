"""CLI entry point for the MLE-STAR pipeline.

Provides ``main()`` as the console-script entry point registered in
``pyproject.toml`` as ``mle_star = "mle_star.cli:main"``. Parses
command-line arguments, loads task and config YAML files, and delegates
to ``run_pipeline_sync()`` from ``mle_star.orchestrator``.

Refs:
    IMPLEMENTATION_PLAN.md Task 50.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml

from mle_star.models import PipelineConfig, TaskDescription
from mle_star.orchestrator import PipelineError, run_pipeline_sync


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured ``ArgumentParser`` with ``--task`` and ``--config`` flags.
    """
    parser = argparse.ArgumentParser(
        prog="mle_star",
        description="MLE-STAR: automated ML pipeline for Kaggle competitions.",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Path to the task description YAML file.",
    )
    parser.add_argument(
        "--config",
        required=False,
        default=None,
        help="Path to an optional PipelineConfig YAML file.",
    )
    return parser


def _load_yaml(path: str, label: str) -> dict[str, Any]:
    """Load and validate a YAML file as a dict.

    Args:
        path: File path to the YAML file.
        label: Human-readable label for error messages (e.g., "task", "config").

    Returns:
        The parsed YAML content as a dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid YAML or does not parse to a dict.
    """
    file_path = Path(path)
    if not file_path.exists():
        msg = f"{label} file not found: {path}"
        raise FileNotFoundError(msg)

    with open(file_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        msg = f"{label} file must contain a YAML mapping, got {type(data).__name__}"
        raise ValueError(msg)

    return data


def _check_claude_cli() -> None:
    """Check that the ``claude`` CLI is available on PATH.

    Raises:
        SystemExit: If ``claude`` is not found.
    """
    if shutil.which("claude") is None:
        print(
            "Error: Claude Code CLI ('claude') not found on PATH. "
            "Install it from https://docs.anthropic.com/en/docs/claude-code",
            file=sys.stderr,
        )
        sys.exit(1)


def _print_startup_summary(task: TaskDescription, config: PipelineConfig) -> None:
    """Print a startup summary banner to stdout.

    Args:
        task: Task description with competition metadata.
        config: Pipeline configuration with hyperparameters.
    """
    sep = "=" * 60
    print(sep)
    print("MLE-STAR Pipeline")
    print(sep)
    print(f"  Competition:  {task.competition_id}")
    print(f"  Task type:    {task.task_type}")
    print(f"  Metric:       {task.evaluation_metric} ({task.metric_direction})")
    print(f"  Data dir:     {task.data_dir}")
    print(f"  Model:        {config.model}")
    print(f"  Time limit:   {config.time_limit_seconds}s")
    print(
        f"  Hyperparams:  M={config.num_retrieved_models}, "
        f"T={config.outer_loop_steps}, K={config.inner_loop_steps}, "
        f"L={config.num_parallel_solutions}, R={config.ensemble_rounds}"
    )
    print(sep)


def main() -> int:
    """Entry point for the mle_star CLI application.

    Parses ``--task`` (required) and ``--config`` (optional) arguments,
    loads and validates YAML files, and runs the pipeline via
    ``run_pipeline_sync()``.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = _build_parser()
    args = parser.parse_args()

    _check_claude_cli()

    try:
        # Load and validate task description
        task_data = _load_yaml(args.task, "task")
        task = TaskDescription(**task_data)

        # Load and validate config if provided
        config: PipelineConfig | None = None
        if args.config is not None:
            config_data = _load_yaml(args.config, "config")
            config = PipelineConfig(**config_data)

        # Print startup summary
        effective_config = config if config is not None else PipelineConfig()
        _print_startup_summary(task, effective_config)

        # Run the pipeline
        result = run_pipeline_sync(task, config)

        # Print success information
        print("Pipeline completed successfully.")
        print(f"Submission path: {result.submission_path}")
        print(f"Total duration: {result.total_duration_seconds}s")
        if result.final_solution.score is not None:
            print(f"Final score: {result.final_solution.score}")

    except PipelineError as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        print(f"Diagnostics: {exc.diagnostics}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
