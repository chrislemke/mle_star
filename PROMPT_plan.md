# RALPH Planning Mode

You are operating in PLANNING mode. Your job is to analyze requirements and create/update the implementation plan.

**CRITICAL: Plan only. Do NOT implement anything.**

---

## Phase 0: Load Context

1. Read all files in `thoughts/shared/specs/` directory to understand requirements
2. Read `CLAUDE.md` to understand the codebase and conventions
3. Read `thoughts/shared/plan/IMPLEMENTATION_PLAN.md` if it exists to see current state
4. Explore the existing source code to understand what's already built

---

## Phase 1: Gap Analysis

Compare the requirements (specs) against:
- The current implementation plan
- The existing codebase

Identify:
- Missing tasks needed to fulfill requirements
- Tasks that are complete but not marked as done
- Tasks that are no longer relevant
- Dependencies between tasks
- Optimal task ordering

---

## Phase 2: Update Plan

Update `thoughts/shared/plan/IMPLEMENTATION_PLAN.md` with:

1. **New tasks** discovered from gap analysis
2. **Status updates** for tasks already complete in code
3. **Refined descriptions** where requirements are now clearer
4. **Priority adjustments** based on dependencies and value

### Task Format

Each task should follow this format:

```markdown
## [P1] Task Title
**Status:** pending | in-progress | done
**Priority:** P1 (critical) | P2 (important) | P3 (nice-to-have)

### Description
What needs to be done and why.

### Acceptance Criteria
- [ ] Specific, testable criterion
- [ ] Another criterion
```

---

## Phase 3: Commit and Exit

1. Commit the updated `thoughts/shared/plan/IMPLEMENTATION_PLAN.md` with message: `docs: update implementation plan` and some helpful description. Do not push.
2. Exit cleanly

Pre-commit will run automatically before committing. Fix any issues that arise.
Do not push!

---

## Reminders

- Do NOT write any implementation code
- Do NOT modify source files
- Only update `thoughts/shared/plan/IMPLEMENTATION_PLAN.md`
- Keep tasks small and focused (1-2 hours of work max)
- Ensure every task has clear acceptance criteria
