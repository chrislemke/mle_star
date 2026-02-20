#!/usr/bin/env python3
"""
Skill Initializer — Creates a complete Claude Code skill folder structure.

Usage:
    init_skill.py <skill-name> --path <target-directory> [--type skill|command|agent] [--minimal]

Examples:
    init_skill.py processing-pdfs --path .claude/skills
    init_skill.py review-code --path ~/.claude/skills
    init_skill.py deploy-app --path .claude/skills --minimal

The script creates the full folder structure in one step:
  skill-name/
  ├── SKILL.md          (with frontmatter template)
  ├── scripts/          (optional, with example)
  ├── references/       (optional, with example)
  └── assets/           (only if not --minimal)
"""

import argparse
import re
import sys
from pathlib import Path


# ── Validation ────────────────────────────────────────────────────────────────

RESERVED_WORDS = {"anthropic", "claude"}
NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_NAME_LENGTH = 64


def validate_name(name: str) -> list[str]:
    errors = []
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"Name exceeds {MAX_NAME_LENGTH} chars (got {len(name)})")
    if not NAME_PATTERN.match(name):
        errors.append("Name must be lowercase letters, digits, and hyphens only (kebab-case)")
    for word in RESERVED_WORDS:
        if word in name.lower():
            errors.append(f'Name must not contain reserved word "{word}"')
    return errors


# ── Templates ─────────────────────────────────────────────────────────────────

def skill_template(name: str, title: str) -> str:
    return f"""---
name: {name}
description: >-
  [TODO] Describe what this skill does and when to use it. Be specific:
  include trigger phrases, file types, and contexts. Write in third person.
  Example: "Extracts text from PDF files and fills forms. Use when working
  with PDF files, forms, or document extraction."
---

# {title}

[TODO: 1-2 sentences — what does this skill enable?]

## Quick Start

[TODO: The simplest way to use this skill. Show a concrete example.]

## Workflow

[TODO: Step-by-step instructions. Choose the structure that fits:
- Workflow-based: sequential steps for a process
- Task-based: different operations/capabilities
- Reference: guidelines and specifications

Keep it concise. Only add context Claude doesn't already know.]

## Resources

**Scripts**: See `scripts/` for executable utilities
**References**: See `references/` for detailed documentation

Delete any empty resource directories — not every skill needs all of them.
"""


def command_template(name: str, title: str) -> str:
    return f"""---
name: {name}
description: >-
  [TODO] Describe what this command does. Commands are user-invoked via /{name}.
---

# {title}

[TODO: Instructions for when the user invokes /{name}]

## Steps

1. [TODO: First step]
2. [TODO: Second step]
3. [TODO: Present results]
"""


def agent_template(name: str, title: str) -> str:
    return f"""---
name: {name}
description: >-
  [TODO] Describe this subagent's role and what it produces.
---

# {title}

## Role

[TODO: What this agent is responsible for]

## Inputs

[TODO: What this agent receives]

## Procedure

1. [TODO: Step-by-step procedure]

## Outputs

[TODO: What this agent produces — files, JSON, reports]
"""


EXAMPLE_SCRIPT = '''#!/usr/bin/env python3
"""
Example utility script for {name}.

Replace with actual implementation or delete if not needed.
Scripts are preferred over asking Claude to generate code because they are
more reliable, save tokens, and ensure consistency.
"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: {name}_helper.py <input>")
        sys.exit(1)

    input_path = sys.argv[1]
    print(f"Processing: {{input_path}}")
    # TODO: Add actual logic here


if __name__ == "__main__":
    main()
'''


EXAMPLE_REFERENCE = """# {title} — Reference

[TODO: Detailed documentation that SKILL.md references when needed.
This file is only loaded into context when Claude needs it.]

## Contents

- [Section 1](#section-1)
- [Section 2](#section-2)

## Section 1

[TODO: Detailed content]

## Section 2

[TODO: Detailed content]
"""


# ── Scaffolding ───────────────────────────────────────────────────────────────

def title_from_name(name: str) -> str:
    return " ".join(word.capitalize() for word in name.split("-"))


def create_skill(name: str, path: Path, skill_type: str = "skill", minimal: bool = False) -> Path | None:
    title = title_from_name(name)

    # Determine target directory and file
    if skill_type == "command":
        target = path / f"{name}.md"
        if target.exists():
            print(f"❌ Command already exists: {target}")
            return None
        path.mkdir(parents=True, exist_ok=True)
        target.write_text(command_template(name, title))
        print(f"✅ Created command: {target}")
        return target

    if skill_type == "agent":
        target = path / f"{name}.md"
        if target.exists():
            print(f"❌ Agent already exists: {target}")
            return None
        path.mkdir(parents=True, exist_ok=True)
        target.write_text(agent_template(name, title))
        print(f"✅ Created agent: {target}")
        return target

    # Default: Skill (folder with SKILL.md)
    skill_dir = path / name
    if skill_dir.exists():
        print(f"❌ Skill directory already exists: {skill_dir}")
        return None

    try:
        skill_dir.mkdir(parents=True)
        print(f"✅ Created: {skill_dir}/")

        # SKILL.md
        (skill_dir / "SKILL.md").write_text(skill_template(name, title))
        print(f"   ├── SKILL.md")

        # scripts/
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        script_file = scripts_dir / f"{name.replace('-', '_')}_helper.py"
        script_file.write_text(EXAMPLE_SCRIPT.format(name=name))
        script_file.chmod(0o755)
        print(f"   ├── scripts/{script_file.name}")

        # references/
        refs_dir = skill_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "reference.md").write_text(EXAMPLE_REFERENCE.format(title=title))
        print(f"   ├── references/reference.md")

        # assets/ (only if not minimal)
        if not minimal:
            (skill_dir / "assets").mkdir()
            print(f"   └── assets/")
        else:
            print(f"   └── (minimal mode: no assets/)")

        print(f"\n✅ Skill '{name}' scaffolded at {skill_dir}")
        print(f"\nNext steps:")
        print(f"  1. Edit SKILL.md — fill in the [TODO] sections")
        print(f"  2. Write a specific description with trigger phrases")
        print(f"  3. Add scripts or references as needed")
        print(f"  4. Delete empty resource directories")
        print(f"  5. Test with realistic prompts")
        return skill_dir

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scaffold a new Claude Code skill, command, or agent."
    )
    parser.add_argument("name", help="Skill name in kebab-case (e.g. processing-pdfs)")
    parser.add_argument("--path", required=True, help="Target directory (e.g. .claude/skills)")
    parser.add_argument(
        "--type",
        choices=["skill", "command", "agent"],
        default="skill",
        help="Type to create (default: skill)",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Skip assets/ directory for lean skills",
    )
    args = parser.parse_args()

    # Validate name
    errors = validate_name(args.name)
    if errors:
        print(f"❌ Invalid name '{args.name}':")
        for e in errors:
            print(f"   • {e}")
        sys.exit(1)

    print(f"🚀 Creating {args.type}: {args.name}")
    print(f"   Path: {args.path}\n")

    result = create_skill(args.name, Path(args.path), args.type, args.minimal)
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
