#!/usr/bin/env python3
"""
Skill Validator — Checks a Claude Code skill for structural correctness.

Usage:
    validate_skill.py <path-to-skill-folder>

Checks:
  - SKILL.md exists and has valid YAML frontmatter
  - name field: kebab-case, max 64 chars, no reserved words
  - description field: non-empty, max 1024 chars, third person
  - Body length: warns if >500 lines
  - No Windows-style paths in content
  - References are one level deep (no nested chains)
  - Scripts have execute permissions
  - No deeply nested file references
"""

import re
import sys
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────────

RESERVED_WORDS = {"anthropic", "claude"}
NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_NAME_LENGTH = 64
MAX_DESC_LENGTH = 1024
MAX_BODY_LINES = 500
FIRST_PERSON_STARTS = ["i ", "i'll", "i can", "i will", "you can", "you should"]
WINDOWS_PATH = re.compile(r"[a-zA-Z_]+\\[a-zA-Z_]+")


class ValidationResult:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str):
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def report(self) -> str:
        lines = []
        if self.errors:
            lines.append(f"\n❌ {len(self.errors)} error(s):")
            for e in self.errors:
                lines.append(f"   • {e}")
        if self.warnings:
            lines.append(f"\n⚠️  {len(self.warnings)} warning(s):")
            for w in self.warnings:
                lines.append(f"   • {w}")
        if self.ok and not self.warnings:
            lines.append("\n✅ Skill is valid — no issues found.")
        elif self.ok:
            lines.append("\n✅ Skill is valid (with warnings).")
        return "\n".join(lines)


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_frontmatter(content: str) -> tuple[dict, str, int]:
    """Parse YAML frontmatter from SKILL.md content.

    Returns: (frontmatter_dict, body, body_start_line)
    """
    lines = content.split("\n")

    if not lines or lines[0].strip() != "---":
        return {}, content, 1

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        return {}, content, 1

    fm_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1 :])
    body_start = end_idx + 2

    # Simple YAML parsing (no dependency needed)
    fm = {}
    current_key = None
    current_val_lines = []

    for line in fm_lines:
        # Multi-line string continuation
        if current_key and (line.startswith("  ") or line.startswith("\t")):
            current_val_lines.append(line.strip())
            continue

        # Save previous key
        if current_key:
            fm[current_key] = " ".join(current_val_lines).strip()
            current_val_lines = []
            current_key = None

        # New key:value
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if val in (">-", ">", "|", "|-"):
                current_key = key
            elif val:
                fm[key] = val.strip('"').strip("'")
            else:
                fm[key] = ""

    # Flush last key
    if current_key:
        fm[current_key] = " ".join(current_val_lines).strip()

    return fm, body, body_start


# ── Validators ────────────────────────────────────────────────────────────────

def validate_skill(skill_path: Path) -> ValidationResult:
    result = ValidationResult()

    # Check SKILL.md exists
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        result.error(f"SKILL.md not found in {skill_path}")
        return result

    content = skill_md.read_text(encoding="utf-8")
    fm, body, body_start = parse_frontmatter(content)

    # ── Frontmatter checks ────────────────────────────────────────────────

    if not fm:
        result.error("No YAML frontmatter found (must start with --- on line 1)")
        return result

    # name
    name = fm.get("name", "")
    if not name:
        result.error("Missing 'name' field in frontmatter")
    else:
        if len(name) > MAX_NAME_LENGTH:
            result.error(f"name exceeds {MAX_NAME_LENGTH} chars (got {len(name)})")
        if not NAME_PATTERN.match(name):
            result.error("name must be kebab-case: lowercase letters, digits, hyphens")
        for word in RESERVED_WORDS:
            if word in name.lower():
                result.error(f'name must not contain reserved word "{word}"')
        if "<" in name or ">" in name:
            result.error("name must not contain XML tags")

    # description
    desc = fm.get("description", "")
    if not desc:
        result.error("Missing 'description' field in frontmatter")
    elif desc.startswith("[TODO"):
        result.warn("Description still contains [TODO] placeholder")
    else:
        if len(desc) > MAX_DESC_LENGTH:
            result.error(f"description exceeds {MAX_DESC_LENGTH} chars (got {len(desc)})")
        if "<" in desc and ">" in desc:
            result.error("description must not contain XML tags")
        desc_lower = desc.lower().strip()
        for phrase in FIRST_PERSON_STARTS:
            if desc_lower.startswith(phrase):
                result.warn(f'description should use third person (starts with "{phrase}...")')
                break
        if "use when" not in desc_lower and "use for" not in desc_lower:
            result.warn("description should include 'Use when...' trigger guidance")

    # ── Body checks ───────────────────────────────────────────────────────

    body_lines = body.strip().split("\n")
    body_line_count = len(body_lines)

    if body_line_count > MAX_BODY_LINES:
        result.warn(
            f"Body has {body_line_count} lines (recommended <{MAX_BODY_LINES}). "
            "Consider moving details to references/"
        )

    if body_line_count < 5:
        result.warn("Body seems very short — does the skill have enough instructions?")

    # Windows paths
    for i, line in enumerate(body_lines, start=body_start):
        if WINDOWS_PATH.search(line) and "\\n" not in line and "\\t" not in line:
            result.warn(f"Line {i}: possible Windows-style path (use forward slashes)")

    # Check for TODO placeholders in body
    todo_count = body.count("[TODO")
    if todo_count > 0:
        result.warn(f"Body still contains {todo_count} [TODO] placeholder(s)")

    # ── File structure checks ─────────────────────────────────────────────

    # Check for nested references (references in referenced files)
    refs_dir = skill_path / "references"
    if refs_dir.is_dir():
        for ref_file in refs_dir.glob("*.md"):
            ref_content = ref_file.read_text(encoding="utf-8", errors="replace")
            # Strip code blocks before checking for nested references
            stripped = re.sub(r"```[\s\S]*?```", "", ref_content)
            ref_links = re.findall(r"\[.*?\]\((references/.*?)\)", stripped)
            if ref_links:
                result.warn(
                    f"{ref_file.name} references other reference files ({ref_links}). "
                    "Keep references one level deep from SKILL.md."
                )

    # Check script permissions
    scripts_dir = skill_path / "scripts"
    if scripts_dir.is_dir():
        for script in scripts_dir.iterdir():
            if script.suffix in (".py", ".sh", ".bash"):
                import os
                if not os.access(script, os.X_OK):
                    result.warn(f"scripts/{script.name} is not executable (chmod +x)")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: validate_skill.py <path-to-skill-folder>")
        print("\nExample: validate_skill.py .claude/skills/my-skill")
        sys.exit(1)

    skill_path = Path(sys.argv[1]).resolve()

    if not skill_path.is_dir():
        print(f"❌ Not a directory: {skill_path}")
        sys.exit(1)

    print(f"🔍 Validating skill at: {skill_path}\n")

    result = validate_skill(skill_path)
    print(result.report())

    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
