# RALPH Build Mode

You are operating in BUILD mode. Your job is to implement ONE task from the plan per iteration.

---

## Phase 1: Orient

1. Read `thoughts/shared/specs/` directory to understand requirements
2. Read `CLAUDE.md` to understand codebase patterns and commands
3. Read the `main.py` this file should be the entry point for the application.

---

## Phase 2: Read Plan

Read `thoughts/shared/plan/IMPLEMENTATION_PLAN.md` to understand:
- Overall project status
- Available tasks
- Task dependencies

---

## Phase 3: Select Task

Choose the highest priority task that:
- Has status `pending` (not `in-progress` or `done`)
- Has no blocking dependencies (or dependencies are `done`)
- Is P1 before P2 before P3

If no tasks are available, commit any pending changes and exit.

---

## Phase 4: Investigate

Before implementing:
1. Mark the task as `in-progress` in `thoughts/shared/plan/IMPLEMENTATION_PLAN.md`
2. Read all code files relevant to this task
3. Understand existing patterns and conventions
4. Identify exactly what changes are needed

---

## Phase 5: Implement

Implement the task:
- Always use the `developing-python` skill
- Follow existing code patterns
- Keep changes minimal and focused
- Use subagents (also using the `developing-python` skill) for parallel work when appropriate
- WRITE TEST FIRST (Classical TDD) using the `tdd-test-writer` agent
- All implemented functions and modules MUST include Google-style Python docstrings
- Respect KISS (keep it stupid simple) and YAGNI (you ain't gonna need it)

---

## Phase 6: Validate

Run all validation commands from `CLAUDE.md`:

1. **Build**: Ensure the project compiles/builds
2. **Test**: Run the test suite
3. **Typecheck**: Run type checking (if applicable)
4. **Lint**: Run linter (if applicable)

If any validation fails:
- Fix the issue
- Re-run validation
- Repeat until all checks pass

---

## Phase 7: Update Plan

In `thoughts/shared/plan/IMPLEMENTATION_PLAN.md`:
1. Mark the completed task as `done`
2. Check all acceptance criteria boxes
3. Add any new tasks discovered during implementation

---

## Phase 8: Update CLAUDE.md

If you learned something important during this iteration:
- Add it to the "Operational Notes" section
- Document any new patterns discovered
- Note any gotchas for future iterations

---

## Phase 9: Commit

Create a git commit with:
- A clear, conventional commit message
- A helpful description
- All changed files
- Reference the task if applicable

Pre-commit will run automatically before committing. Fix any issues that arise.
Do not push!

Example: `feat: implement user authentication`

---

## Phase 10: Exit

Exit cleanly to allow the loop to continue.

---

## Reminders

- ONE task per iteration
- Always validate before committing
- Keep commits atomic and focused
- Update documentation as you go
- If stuck, update the plan and exit (don't spin)
