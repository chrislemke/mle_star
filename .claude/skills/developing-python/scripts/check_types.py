#!/usr/bin/env python3
"""
Type-check a Python file or directory using mypy with strict settings.

Usage:
    check_types.py <path> [--config pyproject.toml]

Runs mypy in strict mode and reports results. Exits 0 if clean, 1 if errors found.
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: check_types.py <path> [--config <config-file>]")
        print("  <path>    File or directory to type-check")
        print("  --config  Path to mypy config file (default: pyproject.toml)")
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Error: Path does not exist: {target}")
        sys.exit(1)

    # Parse optional --config argument
    config_file = None
    if "--config" in sys.argv:
        config_idx = sys.argv.index("--config")
        if config_idx + 1 < len(sys.argv):
            config_file = sys.argv[config_idx + 1]

    cmd = [
        sys.executable, "-m", "mypy",
        "--strict",
        "--pretty",
        "--show-error-codes",
        "--no-error-summary",
        str(target),
    ]

    if config_file:
        cmd.extend(["--config-file", config_file])

    print(f"Running: mypy --strict {target}")
    print("-" * 60)

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print("-" * 60)
        print("All type checks passed.")
    else:
        print("-" * 60)
        print("Type errors found. Fix the issues above.")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
