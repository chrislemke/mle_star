# Common Patterns & Templates

Reusable patterns for building Claude Code skills. Pick what fits your use case.

## Contents

- [Template Pattern](#template-pattern)
- [Examples Pattern](#examples-pattern)
- [Conditional Workflow](#conditional-workflow)
- [Validation Loop](#validation-loop)
- [Domain Organization](#domain-organization)
- [MCP Integration](#mcp-integration)
- [Skill Type Templates](#skill-type-templates)

## Template Pattern

Define output format with appropriate strictness.

**Strict** (for API responses, data formats):
```markdown
## Report Structure
ALWAYS use this exact template:
# [Title]
## Executive Summary
[One-paragraph overview]
## Key Findings
- Finding 1 with data
- Finding 2 with data
## Recommendations
1. Actionable recommendation
```

**Flexible** (when adaptation is useful):
```markdown
## Report Structure
Sensible default format — adjust based on the analysis:
# [Title]
## Executive Summary
## Key Findings (adapt sections as needed)
## Recommendations
```

## Examples Pattern

Show input/output pairs to calibrate Claude's output style:

```markdown
## Commit Message Format

**Example 1:**
Input: Added user authentication with JWT tokens
Output:
feat(auth): implement JWT-based authentication
Add login endpoint and token validation middleware

**Example 2:**
Input: Fixed date display bug in reports
Output:
fix(reports): correct date formatting in timezone conversion
Use UTC timestamps consistently across report generation
```

Examples communicate style and detail level better than descriptions.

## Conditional Workflow

Guide Claude through decision points:

```markdown
## Workflow

1. Determine task type:
   **Creating new?** → Follow "Creation" below
   **Editing existing?** → Follow "Editing" below

2. Creation:
   - Use library X
   - Build from scratch
   - Export to format Y

3. Editing:
   - Load existing file
   - Modify content
   - Validate changes
   - Save output
```

When workflows grow large, push them into separate reference files.

## Validation Loop

For quality-critical operations:

```markdown
## Editing Process

1. Make edits
2. **Validate immediately**: `python scripts/validate.py output/`
3. If validation fails:
   - Review error messages carefully
   - Fix the issues
   - Run validation again
4. **Only proceed when validation passes**
5. Build final output
6. Test the result
```

Make validation scripts verbose: "Field 'X' not found. Available fields: A, B, C"

## Domain Organization

For multi-domain skills, organize by variant:

```
my-skill/
├── SKILL.md (overview + navigation)
└── references/
    ├── finance.md (revenue, billing)
    ├── sales.md (pipeline, accounts)
    └── product.md (usage, features)
```

SKILL.md points Claude to the right file:
```markdown
**Finance**: Revenue, billing → See [references/finance.md](references/finance.md)
**Sales**: Pipeline, accounts → See [references/sales.md](references/sales.md)
```

Claude reads only the relevant file, saving context.

## MCP Integration

When skills use MCP tools, use fully qualified names:

```markdown
Use the BigQuery:bigquery_schema tool to retrieve table schemas.
Use the GitHub:create_issue tool to create issues.
```

Format: `ServerName:tool_name`. Without the prefix, Claude may fail to locate tools.

## Skill Type Templates

### Workflow Skill

Best for sequential, multi-step processes.

```yaml
---
name: deploying-services
description: >-
  Deploys services to production with validation and rollback support.
  Use when deploying, releasing, pushing to production, or managing releases.
---
```

```markdown
# Deploying Services

## Pre-deployment Checklist
1. Run tests: `npm test`
2. Check lint: `npm run lint`
3. Verify environment variables

## Deployment Steps
1. Build: `npm run build`
2. Deploy: `scripts/deploy.sh <env>`
3. Validate: `scripts/health_check.sh`
4. If health check fails → rollback: `scripts/rollback.sh`

## Post-deployment
- Monitor logs for 15 minutes
- Verify key user flows
```

### Tool/Utility Skill

Best for a collection of operations.

```yaml
---
name: processing-images
description: >-
  Resizes, converts, and optimizes images. Use when working with
  image files, thumbnails, format conversion, or image optimization.
---
```

```markdown
# Processing Images

## Quick Start
Resize: `python scripts/resize.py input.png 800x600`
Convert: `python scripts/convert.py input.png output.webp`
Optimize: `python scripts/optimize.py input.png --quality 85`

## Batch Operations
Process entire directories:
`python scripts/batch.py images/ --resize 800x600 --format webp`
```

### Guidelines Skill

Best for standards, conventions, brand rules.

```yaml
---
name: code-standards
description: >-
  Enforces project coding standards and conventions. Use when writing
  new code, reviewing PRs, or when the user asks about code style.
---
```

```markdown
# Code Standards

## Naming
- Files: kebab-case (`user-service.ts`)
- Classes: PascalCase (`UserService`)
- Functions: camelCase (`getUserById`)

## Structure
- One export per file for components
- Colocate tests: `foo.ts` → `foo.test.ts`

## Error Handling
Always use typed errors:
try { ... } catch (e) { if (e instanceof AppError) ... }
```

### Research/Analysis Skill

Best for investigative, multi-source tasks.

```yaml
---
name: researching-topics
description: >-
  Conducts structured research across multiple sources and produces
  synthesized reports. Use when the user needs research, analysis,
  competitive intelligence, or literature review.
---
```

```markdown
# Researching Topics

## Process
1. Clarify scope and key questions
2. Search across available sources
3. Cross-reference claims between sources
4. Identify conflicts and gaps
5. Synthesize findings with citations

## Output Format
# [Topic] Research Summary
## Key Findings (ranked by confidence)
## Sources and Methodology
## Open Questions
```
