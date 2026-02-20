---
name: create-skill
description: Creates new Claude Code skills with proper folder structure, SKILL.md frontmatter, and bundled resources. Use when the user wants to create a new skill, scaffold a skill, generate a SKILL.md, build a slash command, set up a .claude/skills directory, or says "create a skill", "new skill", "scaffold skill", "make a skill for", or asks about skill structure. Also use when converting an existing workflow or prompt into a reusable skill.
---

# Creating Skills for Claude Code

Create well-structured, production-ready Claude Code skills following the Agent Skills open standard.

## Decision: Skill vs Command vs Subagent

Before creating, determine the right type:

- **Skill** (model-invoked): Claude decides when to use it. Best for reusable expertise, workflows, domain knowledge. → `.claude/skills/<name>/SKILL.md`
- **Command** (user-invoked via `/name`): User explicitly triggers it. Best for on-demand actions, one-off tasks. → `.claude/commands/<name>.md` (same frontmatter as skills)
- **Subagent**: Delegated parallel work. Best for independent tasks within a larger workflow. → `.claude/agents/<name>.md`

Default to **Skill** unless the user explicitly wants a command or subagent.

## Quick Start

Initialize a new skill with the bundled script:

```bash
python3 <skill-base-path>/scripts/init_skill.py <skill-name> --path .claude/skills
```

This creates the complete folder structure in one step. Then customize the generated SKILL.md.

For personal (cross-project) skills, use `--path ~/.claude/skills` instead.

## Writing the SKILL.md

### Frontmatter (YAML)

```yaml
---
name: my-skill-name          # kebab-case, max 64 chars, no "anthropic"/"claude"
description: >-               # max 1024 chars, third person, specific triggers
  Processes X and generates Y. Use when the user mentions X,
  asks about Y, or works with Z files.
allowed-tools: Read, Bash     # optional: restrict tool access
---
```

**Critical**: The description is the primary discovery mechanism. Be "pushy" — include specific trigger phrases and file types. Write in third person ("Processes..." not "I process...").

### Body Structure

Keep under 500 lines. Follow one of these patterns:

**Workflow-based** (sequential processes):
```
## Overview → ## Decision Tree → ## Step 1 → ## Step 2...
```

**Task-based** (tool collections):
```
## Overview → ## Quick Start → ## Task 1 → ## Task 2...
```

**Reference/Guidelines** (standards):
```
## Overview → ## Guidelines → ## Specifications → ## Examples
```

### Core Principles

1. **Concise is key** — Only add context Claude doesn't already have. Challenge every paragraph: "Does Claude need this explanation?"
2. **Progressive disclosure** — SKILL.md is the overview. Put details in `references/`, automation in `scripts/`, templates in `assets/`
3. **One level deep** — All reference files link directly from SKILL.md. Never nest references → references → details
4. **Explain the why** — Instead of rigid MUSTs, explain reasoning. Claude is smart; theory of mind beats commands
5. **Feedback loops** — For quality-critical tasks, add validate → fix → repeat cycles
6. **Solve, don't punt** — Scripts should handle errors explicitly, not fail silently

### Naming Convention

Use **gerund form** (verb + -ing): `processing-pdfs`, `generating-reports`, `analyzing-data`

Acceptable alternatives: noun phrases (`pdf-processing`) or action-oriented (`process-pdfs`).

Avoid: vague names (`helper`, `utils`), reserved words (`anthropic-*`, `claude-*`).

## Folder Structure

```
skill-name/
├── SKILL.md              # Required: frontmatter + instructions
├── scripts/              # Optional: executable code (Python/Bash)
├── references/           # Optional: docs loaded into context as needed
└── assets/               # Optional: templates, icons, fonts for output
```

Only create directories the skill actually needs. Not every skill requires all three.

## Bundled Scripts

Pre-made scripts are preferred over asking Claude to generate code:
- More reliable than generated code
- Save tokens (no code generation in context)
- Ensure consistency across uses
- Make execution intent clear: "Run `scripts/build.py`" vs "See `scripts/build.py` for the algorithm"

Scripts should handle errors explicitly with helpful messages. Document magic constants.

## Validation

After creating or editing a skill, validate with:

```bash
python3 <skill-base-path>/scripts/validate_skill.py <path-to-skill-folder>
```

Fix any reported issues before committing.

## Workflow Checklist

```
Skill Creation Progress:
- [ ] Clarify intent: What should the skill do? When should it trigger?
- [ ] Choose type: Skill / Command / Subagent
- [ ] Run init script to scaffold folder structure
- [ ] Write description with specific trigger phrases
- [ ] Write concise SKILL.md body (<500 lines)
- [ ] Add scripts/ for deterministic operations
- [ ] Add references/ for detailed docs (if needed)
- [ ] Validate with validate_skill.py
- [ ] Test with realistic prompts
- [ ] Iterate based on Claude's behavior
```

## Best Practices Reference

For detailed authoring guidance: See [references/best-practices.md](references/best-practices.md)

For common patterns and templates: See [references/patterns.md](references/patterns.md)
