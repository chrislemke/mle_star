#!/usr/bin/env python3
"""
SRS Validator — Checks an SRS document for ISO 29148 structural compliance.

Usage:
    validate_srs.py <path-to-srs.md> [--strict]

Checks:
  - Required and recommended sections are present
  - No empty sections (heading with no content)
  - Requirement IDs follow the CAT-NNN pattern and are unique
  - Requirements contain shall-statements
  - Requirements have priority assignments
  - Requirements have verification criteria
  - Summary statistics for coverage

Exit codes: 0 = pass, 1 = fail (or warnings in --strict mode)
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────────

REQUIRED_SECTIONS = [
    "introduction",
    "purpose",
    "scope",
    "product perspective",
    "functional requirements",
]

RECOMMENDED_SECTIONS = [
    "definitions",
    "external interface",
    "non-functional requirements",
    "data requirements",
    "constraints",
    "change control",
    "stakeholder",
    "user class",
    "assumptions",
    "dependencies",
]

REQ_ID_PATTERN = re.compile(r"\b(FR|DR|NFR|CR|IR)-(\d{3})\b")

PRIORITY_KEYWORDS = re.compile(
    r"\b(Must|Should|Could|Won't|Wont|High|Medium|Low|Critical)\b",
    re.IGNORECASE,
)

VERIFICATION_KEYWORDS = re.compile(
    r"\b(Verification|Verify|Acceptance|Test|Inspection|Analysis|Demonstration|IADT)\b",
    re.IGNORECASE,
)


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
            lines.append(f"\n  {len(self.errors)} error(s):")
            for e in self.errors:
                lines.append(f"   - {e}")
        if self.warnings:
            lines.append(f"\n  {len(self.warnings)} warning(s):")
            for w in self.warnings:
                lines.append(f"   - {w}")
        if self.ok and not self.warnings:
            lines.append("\n  Skill is valid — no issues found.")
        elif self.ok:
            lines.append("\n  Valid (with warnings).")
        return "\n".join(lines)


# ── Section Parsing ──────────────────────────────────────────────────────────

def parse_sections(content: str) -> list[tuple[int, str, str, int]]:
    """Parse markdown headings and their content.

    Returns list of (level, title, body_text, line_number).
    """
    lines = content.split("\n")
    sections = []
    current_level = 0
    current_title = ""
    current_body_lines = []
    current_line = 0

    for i, line in enumerate(lines, start=1):
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            # Save previous section
            if current_title:
                body = "\n".join(current_body_lines).strip()
                sections.append((current_level, current_title, body, current_line))

            current_level = len(heading_match.group(1))
            current_title = heading_match.group(2).strip()
            current_body_lines = []
            current_line = i
        else:
            current_body_lines.append(line)

    # Save last section
    if current_title:
        body = "\n".join(current_body_lines).strip()
        sections.append((current_level, current_title, body, current_line))

    return sections


# ── Validators ───────────────────────────────────────────────────────────────

def check_sections(sections: list[tuple[int, str, str, int]], result: ValidationResult):
    """Check for required and recommended sections."""
    section_titles_lower = [title.lower() for _, title, _, _ in sections]
    all_text = " ".join(section_titles_lower)

    # Required sections
    for req in REQUIRED_SECTIONS:
        if not any(req in title for title in section_titles_lower):
            result.error(f"Missing required section: '{req}'")

    # Recommended sections
    for rec in RECOMMENDED_SECTIONS:
        if not any(rec in title for title in section_titles_lower):
            result.warn(f"Missing recommended section: '{rec}'")

    # Empty sections
    for level, title, body, line_num in sections:
        # Skip table-of-contents and appendix placeholders
        if "table of contents" in title.lower():
            continue
        if not body or body.isspace():
            result.warn(f"Line {line_num}: Section '{title}' appears empty")


def find_requirement_ids(content: str) -> list[tuple[str, str, int]]:
    """Find all requirement IDs in the document.

    Returns list of (full_id, category, line_number).
    """
    ids = []
    for i, line in enumerate(content.split("\n"), start=1):
        for match in REQ_ID_PATTERN.finditer(line):
            category = match.group(1)
            number = match.group(2)
            full_id = f"{category}-{number}"
            ids.append((full_id, category, i))
    return ids


def check_requirement_ids(
    ids: list[tuple[str, str, int]], result: ValidationResult
) -> dict[str, list[str]]:
    """Check requirement IDs for uniqueness and gaps.

    Returns dict of category -> sorted list of IDs.
    """
    # Check duplicates
    id_counts = Counter(full_id for full_id, _, _ in ids)
    for full_id, count in id_counts.items():
        if count > 1:
            result.error(f"Duplicate requirement ID: {full_id} (appears {count} times)")

    # Group by category
    by_category: dict[str, list[int]] = defaultdict(list)
    for full_id, category, _ in ids:
        number = int(full_id.split("-")[1])
        by_category[category].append(number)

    # Check gaps within categories (only for sequential IDs, not sub-ranges)
    for category, numbers in by_category.items():
        sorted_nums = sorted(set(numbers))
        if len(sorted_nums) > 1:
            # Detect clusters (IDs within 10 of each other) vs sub-ranges
            # If IDs span more than 10x the count, they're likely sub-ranges
            span = sorted_nums[-1] - sorted_nums[0] + 1
            if span <= len(sorted_nums) * 3:
                # Likely sequential — check for gaps
                expected = set(range(sorted_nums[0], sorted_nums[-1] + 1))
                missing = expected - set(sorted_nums)
                if missing:
                    missing_sorted = sorted(missing)
                    if len(missing_sorted) > 10:
                        shown = [f"{category}-{n:03d}" for n in missing_sorted[:5]]
                        result.warn(
                            f"Gap in {category} numbering: {len(missing_sorted)} "
                            f"missing IDs ({', '.join(shown)}, ...)"
                        )
                    else:
                        missing_ids = [f"{category}-{n:03d}" for n in missing_sorted]
                        result.warn(
                            f"Gap in {category} numbering: missing {', '.join(missing_ids)}"
                        )

    return {
        cat: [f"{cat}-{n:03d}" for n in sorted(set(nums))]
        for cat, nums in by_category.items()
    }


def check_shall_statements(
    content: str,
    ids: list[tuple[str, str, int]],
    result: ValidationResult,
) -> int:
    """Check that requirements contain shall-statements.

    Returns count of requirements with shall-statements.
    """
    lines = content.split("\n")
    has_shall = 0
    checked = 0

    # For each requirement ID, check surrounding lines for "shall"
    unique_ids = set()
    for full_id, category, line_num in ids:
        if full_id in unique_ids:
            continue
        unique_ids.add(full_id)
        checked += 1

        # Check the line and the next 5 lines for "shall"
        start = max(0, line_num - 1)
        end = min(len(lines), line_num + 5)
        context = " ".join(lines[start:end]).lower()
        if "shall" in context:
            has_shall += 1
        else:
            result.warn(f"{full_id} (line {line_num}): No 'shall' statement found nearby")

    return has_shall


def check_priorities(
    content: str,
    ids: list[tuple[str, str, int]],
    result: ValidationResult,
) -> int:
    """Check that requirements have priority assignments.

    Returns count of requirements with priorities.
    """
    lines = content.split("\n")
    has_priority = 0
    unique_ids = set()

    for full_id, _, line_num in ids:
        if full_id in unique_ids:
            continue
        unique_ids.add(full_id)

        # Check the line and surrounding lines for priority keywords
        start = max(0, line_num - 1)
        end = min(len(lines), line_num + 8)
        context = " ".join(lines[start:end])
        if PRIORITY_KEYWORDS.search(context):
            has_priority += 1

    return has_priority


def check_verification(
    content: str,
    ids: list[tuple[str, str, int]],
    result: ValidationResult,
) -> int:
    """Check that requirements have verification criteria.

    Returns count of requirements with verification criteria.
    """
    lines = content.split("\n")
    has_verification = 0
    unique_ids = set()

    for full_id, _, line_num in ids:
        if full_id in unique_ids:
            continue
        unique_ids.add(full_id)

        # Check the line and surrounding lines for verification keywords
        start = max(0, line_num - 1)
        end = min(len(lines), line_num + 10)
        context = " ".join(lines[start:end])
        if VERIFICATION_KEYWORDS.search(context):
            has_verification += 1

    return has_verification


# ── Validation ───────────────────────────────────────────────────────────────

def validate_srs(filepath: Path) -> tuple[ValidationResult, dict]:
    """Validate an SRS document.

    Returns (ValidationResult, stats_dict).
    """
    result = ValidationResult()
    stats = {
        "total_reqs": 0,
        "by_category": {},
        "shall_count": 0,
        "priority_count": 0,
        "verification_count": 0,
        "sections_found": 0,
        "sections_required": len(REQUIRED_SECTIONS),
    }

    if not filepath.exists():
        result.error(f"File not found: {filepath}")
        return result, stats

    content = filepath.read_text(encoding="utf-8")

    if not content.strip():
        result.error("File is empty")
        return result, stats

    # Parse sections
    sections = parse_sections(content)
    check_sections(sections, result)
    stats["sections_found"] = len(sections)

    # Find and check requirement IDs
    ids = find_requirement_ids(content)
    unique_ids = set(full_id for full_id, _, _ in ids)
    stats["total_reqs"] = len(unique_ids)

    if not ids:
        result.warn(
            "No requirement IDs found (expected pattern: FR-001, NFR-101, etc.)"
        )
        return result, stats

    by_category = check_requirement_ids(ids, result)
    stats["by_category"] = {cat: len(id_list) for cat, id_list in by_category.items()}

    # Check quality attributes
    stats["shall_count"] = check_shall_statements(content, ids, result)
    stats["priority_count"] = check_priorities(content, ids, result)
    stats["verification_count"] = check_verification(content, ids, result)

    # Coverage warnings
    total = stats["total_reqs"]
    if total > 0:
        prio_pct = stats["priority_count"] / total * 100
        verif_pct = stats["verification_count"] / total * 100
        if prio_pct < 80:
            result.warn(
                f"Only {prio_pct:.0f}% of requirements have priority assignments"
            )
        if verif_pct < 80:
            result.warn(
                f"Only {verif_pct:.0f}% of requirements have verification criteria"
            )

    return result, stats


# ── Output ───────────────────────────────────────────────────────────────────

def print_report(filepath: Path, result: ValidationResult, stats: dict):
    """Print formatted validation report."""
    print(f"SRS Validation Report: {filepath.name}")
    print("=" * 50)

    # Sections
    print(f"\nSections:")
    print(f"  Found: {stats['sections_found']} sections")
    required_missing = [
        e for e in result.errors if "Missing required section" in e
    ]
    if required_missing:
        print(f"  Missing required: {len(required_missing)}")
    else:
        print(f"  All {stats['sections_required']} required sections present")

    # Requirements
    total = stats["total_reqs"]
    print(f"\nRequirements:")
    print(f"  Total: {total}")
    if stats["by_category"]:
        cats = "  |  ".join(
            f"{cat}: {count}"
            for cat, count in sorted(stats["by_category"].items())
        )
        print(f"  {cats}")

    duplicates = [e for e in result.errors if "Duplicate" in e]
    print(f"  Duplicates: {len(duplicates) if duplicates else 'None'}")

    # Quality
    if total > 0:
        shall_pct = stats["shall_count"] / total * 100
        prio_pct = stats["priority_count"] / total * 100
        verif_pct = stats["verification_count"] / total * 100
        print(f"\nQuality:")
        print(f"  Shall-statements: {stats['shall_count']}/{total} ({shall_pct:.1f}%)")
        print(f"  Priority assigned: {stats['priority_count']}/{total} ({prio_pct:.1f}%)")
        print(
            f"  Verification criteria: {stats['verification_count']}/{total} ({verif_pct:.1f}%)"
        )

    # Result
    print(result.report())

    error_count = len(result.errors)
    warn_count = len(result.warnings)
    if result.ok and not result.warnings:
        print(f"\nResult: PASS")
    elif result.ok:
        print(f"\nResult: PASS ({warn_count} warning(s))")
    else:
        print(f"\nResult: FAIL ({error_count} error(s), {warn_count} warning(s))")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate an SRS document for ISO 29148 structural compliance."
    )
    parser.add_argument("srs_file", help="Path to the SRS markdown file")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    args = parser.parse_args()

    filepath = Path(args.srs_file).resolve()

    if not filepath.suffix == ".md":
        print(f"Note: File does not have .md extension: {filepath.name}")

    result, stats = validate_srs(filepath)

    if args.strict and result.warnings:
        # Promote warnings to errors in strict mode
        for w in result.warnings:
            result.error(f"[strict] {w}")

    print_report(filepath, result, stats)
    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
