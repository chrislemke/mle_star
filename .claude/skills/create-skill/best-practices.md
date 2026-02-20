# Best Practices Reference

Condensed from [Anthropic's official best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices) and community patterns (2026).

## Contents

- [Conciseness](#conciseness)
- [Degrees of Freedom](#degrees-of-freedom)
- [Description Writing](#description-writing)
- [Progressive Disclosure](#progressive-disclosure)
- [Feedback Loops](#feedback-loops)
- [Scripts](#scripts)
- [Anti-Patterns](#anti-patterns)
- [Evaluation-Driven Development](#evaluation-driven-development)
- [Cross-Model Testing](#cross-model-testing)

## Conciseness

The context window is shared. Every token in your skill competes with conversation history, other skills, and the system prompt.

**Default assumption**: Claude is already very smart. Only add what Claude doesn't know.

Challenge each section:
- "Does Claude really need this explanation?"
- "Can I assume Claude knows this?"
- "Does this paragraph justify its token cost?"

**Good** (~50 tokens):
```
Use pdfplumber for text extraction:
import pdfplumber
with pdfplumber.open("file.pdf") as pdf:
    text = pdf.pages[0].extract_text()
```

**Bad** (~150 tokens): Explaining what PDFs are and why pdfplumber is recommended.

## Degrees of Freedom

Match specificity to task fragility:

| Freedom Level | Use When | Example |
|---------------|----------|---------|
| **High** (text guidance) | Multiple valid approaches | Code review checklist |
| **Medium** (pseudocode) | Preferred pattern exists | Report template with customization |
| **Low** (exact scripts) | Fragile/critical operations | Database migration commands |

Think: narrow bridge with cliffs (low freedom) vs. open field (high freedom).

## Description Writing

The description is how Claude discovers your skill. It's injected into the system prompt at startup.

**Rules**:
- Third person always ("Processes..." not "I process...")
- Include WHAT it does AND WHEN to use it
- Be "pushy" — Claude tends to under-trigger skills
- Include specific file types, trigger phrases, and contexts
- Max 1024 characters

**Good**:
```
description: >-
  Extracts text and tables from PDF files, fills forms, merges documents.
  Use when working with PDF files or when the user mentions PDFs, forms,
  document extraction, or asks to process .pdf files.
```

**Bad**:
```
description: Helps with documents
```

## Progressive Disclosure

Three-level loading prevents context bloat:

1. **Metadata** (always loaded): name + description (~100 tokens)
2. **SKILL.md body** (loaded when triggered): core instructions (<500 lines)
3. **Bundled resources** (loaded as needed): references, scripts, assets

**Key patterns**:
- Reference files link directly from SKILL.md (one level deep)
- For files >100 lines, include a table of contents at the top
- Scripts execute without loading into context (only output consumes tokens)
- Organize by domain: `references/finance.md`, `references/sales.md`

## Feedback Loops

The validate → fix → repeat pattern significantly improves quality.

```
1. Make changes
2. Run validation: python scripts/validate.py
3. If errors: fix issues, go to step 2
4. Only proceed when validation passes
5. Test output
```

For complex tasks, provide a checklist Claude can track:
```
Progress:
- [ ] Step 1: Analyze input
- [ ] Step 2: Generate draft
- [ ] Step 3: Validate output
- [ ] Step 4: Apply fixes
- [ ] Step 5: Final check
```

## Scripts

Utility scripts are preferred over Claude-generated code:

**Benefits**: More reliable, save tokens, ensure consistency, faster execution.

**Requirements**:
- Handle errors explicitly (don't punt to Claude)
- Document all constants (no "voodoo numbers")
- List required packages in SKILL.md
- Make execution intent clear: "Run X" vs "See X for reference"

**Error handling example**:
```python
def process_file(path):
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        print(f"File {path} not found, creating default")
        Path(path).write_text("")
        return ""
```

## Anti-Patterns

| Anti-Pattern | Fix |
|-------------|-----|
| Windows paths (`scripts\helper.py`) | Forward slashes (`scripts/helper.py`) |
| Too many options ("use pypdf, or pdfplumber, or...") | One default + escape hatch |
| Time-sensitive info ("before August 2025, use...") | Current method + "Old patterns" section |
| Inconsistent terminology (mix "field"/"box"/"element") | Pick one term, use consistently |
| Deeply nested references (A → B → C) | All references one level from SKILL.md |
| Vague names (`helper`, `utils`, `tools`) | Descriptive gerunds (`processing-pdfs`) |

## Evaluation-Driven Development

Build evaluations BEFORE extensive documentation:

1. Run Claude on tasks without a skill — document failures
2. Create 3 test scenarios targeting those gaps
3. Measure baseline performance
4. Write minimal instructions to address gaps
5. Test, compare, iterate

**Eval structure**:
```json
{
  "skill_name": "my-skill",
  "evals": [
    {
      "id": 1,
      "prompt": "User's task prompt",
      "expected_output": "Expected result description",
      "assertions": ["Output includes X", "Handles Y correctly"]
    }
  ]
}
```

## Cross-Model Testing

Skills behave differently across models:

- **Haiku** (fast): Needs more explicit guidance
- **Sonnet** (balanced): Good middle ground
- **Opus** (powerful): Avoid over-explaining

Test with all models you plan to deploy with. What works for Opus may need more detail for Haiku.

## Iterative Development with Claude

The most effective workflow uses two Claude instances:

- **Claude A**: Helps design and refine the skill
- **Claude B**: Tests the skill on real tasks

Cycle: observe Claude B's behavior → bring insights to Claude A → refine → test again.

Key insight: Claude A understands agent needs, you provide domain expertise, Claude B reveals gaps through real usage.
