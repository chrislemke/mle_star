#!/usr/bin/env bash
set -euo pipefail

# RALPH Loop - Autonomous AI Development Loop
# Usage: ./loop.sh [plan] [max_iterations]
#   ./loop.sh          → build mode, unlimited iterations
#   ./loop.sh plan     → planning mode, unlimited iterations
#   ./loop.sh 10       → build mode, max 10 iterations
#   ./loop.sh plan 5   → planning mode, max 5 iterations

MODE="build"
MAX_ITERATIONS=0
ITERATION=0

# Parse arguments
for arg in "$@"; do
    if [[ "$arg" == "plan" ]]; then
        MODE="plan"
    elif [[ "$arg" =~ ^[0-9]+$ ]]; then
        MAX_ITERATIONS=$arg
    fi
done

# Select prompt file based on mode
if [[ "$MODE" == "plan" ]]; then
    PROMPT_FILE="PROMPT_plan.md"
else
    PROMPT_FILE="PROMPT_build.md"
fi

# Verify prompt file exists
if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "Error: $PROMPT_FILE not found"
    exit 1
fi

echo "RALPH Loop Starting"
echo "Mode: $MODE"
echo "Max iterations: $([ $MAX_ITERATIONS -eq 0 ] && echo 'unlimited' || echo $MAX_ITERATIONS)"
echo "---"

while true; do
    ITERATION=$((ITERATION + 1))

    # Check iteration limit
    if [[ $MAX_ITERATIONS -gt 0 && $ITERATION -gt $MAX_ITERATIONS ]]; then
        echo "Reached maximum iterations ($MAX_ITERATIONS)"
        break
    fi

    echo ""
    echo "=== Iteration $ITERATION ==="
    echo ""

    # Run Claude with the appropriate prompt
    if ! claude -p "$(cat "$PROMPT_FILE")" --dangerously-skip-permissions; then
        echo "Claude exited with error, stopping loop"
        exit 1
    fi

    # Push changes after successful iteration
    if git rev-parse --git-dir > /dev/null 2>&1; then
        if [[ -n $(git status --porcelain) ]] || [[ $(git rev-list @{u}..HEAD 2>/dev/null | wc -l) -gt 0 ]]; then
            echo ""
            echo "Pushing changes..."
            git push || echo "Warning: git push failed, continuing..."
        fi
    fi

    echo ""
    echo "=== Iteration $ITERATION Complete ==="
done

echo ""
echo "RALPH Loop Complete"
