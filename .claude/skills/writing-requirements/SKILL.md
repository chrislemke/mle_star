---
name: writing-requirements
description: >-
  Writes and maintains Software Requirements Specifications (SRS) aligned
  with ISO/IEC/IEEE 29148. Produces standards-compliant requirements documents
  with shall-statements, unique requirement IDs, verification criteria, and
  traceability matrices. Use when the user wants to write requirements, create
  an SRS, document software requirements, specify system behavior, write
  functional requirements, define non-functional requirements, build a
  requirements traceability matrix (RTM), or mentions ISO 29148, requirements
  engineering, requirements specification, product requirements document (PRD),
  system requirements, or asks to convert user stories into formal requirements.
  Also use when reviewing or improving existing requirements documents.
---

# Writing Requirements (ISO/IEC/IEEE 29148 SRS)

Produce standards-aligned Software Requirements Specifications with testable
shall-statements, verification criteria, and full traceability. Output is a
single markdown file `SRS-<project-name>.md`, with optional companion
`RTM-<project-name>.md` for the traceability matrix.

## Decision Tree

Determine the starting point before proceeding:

| Situation | Entry Point |
|-----------|-------------|
| No existing material | **Phase 1**: Full elicitation from scratch |
| User has notes, user stories, or a PRD | **Transform**: Map existing material to SRS sections, fill gaps |
| User has a partial or draft SRS | **Gap Analysis**: Identify missing sections, then fill them |
| User wants to review an existing SRS | **Validate & Improve**: Run validation, suggest improvements |

For the **Transform** path: read the provided material, map content to SRS
sections (see template), identify gaps, then ask targeted questions to fill
only the missing parts. Do not re-elicit what's already covered.

For **Gap Analysis**: read the existing SRS, compare against the required
sections in the template, report what's missing, and work through those
sections with the user.

For **Validate & Improve**: run `scripts/validate_srs.py` on the document,
then analyze quality beyond structure (ambiguous terms, compound requirements,
missing testability). Present findings grouped by severity.

## Phase 1: Scope & Context

Elicit foundational information. Ask these questions (3-4 per turn max):

**Purpose & Audience**
- What is the system/product name?
- What problem does it solve? What are its goals/objectives?
- Who is the intended audience of this SRS? (developers, stakeholders, testers, regulators)
- What decisions should this SRS support?

**Scope**
- What is explicitly IN scope for this system?
- What is explicitly OUT of scope?
- Is this part of a larger system? If so, how does it relate?

**Stakeholders**
- Who are the user classes? (roles, frequency of use, technical proficiency)
- Who are the other stakeholders? (operators, administrators, regulatory bodies)
- Are there existing systems this replaces or integrates with?

**Context**
- What is the operating environment? (platforms, browsers, infrastructure)
- What are the key assumptions? (technical, organizational, operational)
- What are the critical dependencies? (third-party services, hardware, other projects)

Draft each section immediately after the user answers. Confirm before proceeding.

## Phase 2: Requirements Elicitation

Work through categories **one at a time**. For each:
1. Ask targeted questions to draw out requirements
2. Write them as shall-statements with unique IDs
3. Show the drafted requirements to the user
4. Confirm and adjust before moving to the next category

### Category Order

**2a. External Interfaces (IR-NNN)**
- User interfaces: key screens, flows, interaction patterns
- Software/API interfaces: endpoints, contracts, protocols
- Hardware interfaces: devices, sensors, peripherals
- Communication interfaces: protocols, ports, network constraints

**2b. Functional Requirements (FR-NNN)**
- Core system functions and behaviors
- Input validation and processing rules
- Output generation
- Normal flow and abnormal/edge-case behavior
- Error handling and recovery

**2c. Data Requirements (DR-NNN)**
- Core entities and their relationships
- Validation rules and integrity constraints
- Retention, deletion, and archival policies
- Data migration and import/export needs

**2d. Non-Functional Requirements (NFR-NNN)**
Work through each sub-category:
- Performance (NFR-1xx): latency, throughput, resource limits
- Reliability/Availability (NFR-2xx): uptime SLOs, recovery targets (RTO/RPO)
- Security & Privacy (NFR-3xx): authentication, authorization, encryption, audit, compliance
- Maintainability (NFR-4xx): operability, observability, supportability
- Scalability (NFR-5xx): load growth assumptions, horizontal/vertical limits
- Usability & Accessibility (NFR-6xx): task success metrics, WCAG level
- Portability/Compatibility (NFR-7xx): platforms, browsers, versions

**2e. Constraints (CR-NNN)**
- Regulatory/standards compliance
- Technology/platform constraints (only if truly constrained)
- Operational/site constraints

## Phase 3: Refinement

After all requirements are elicited:

1. **Verification criteria**: Assign a verification method and acceptance criterion to
   every requirement. Methods: Inspection, Analysis, Demonstration, Test (IADT).
   See [references/verification-traceability.md](references/verification-traceability.md)

2. **Priority assignment**: Use MoSCoW for each requirement:
   - **Must** — mandatory for the release
   - **Should** — important but not critical
   - **Could** — desirable if time/budget allows
   - **Won't** — explicitly deferred (document why)

3. **Release allocation**: Assign each requirement to:
   - MVP / Phase 1
   - Phase 2
   - Future / Deferred

4. **Rationale**: For key or controversial requirements, document:
   - Why it exists (source, stakeholder need)
   - Trade-offs considered
   - Alternatives rejected and why

5. **Review completeness**: Check for gaps:
   - Every functional requirement has error handling
   - Every NFR is quantified with a measurable target
   - Every external interface has defined inputs/outputs
   - Assumptions are documented for uncertain items

## Phase 4: Document Assembly & Validation

1. Load the full SRS template:
   See [references/srs-template.md](references/srs-template.md)

2. Assemble all elicited content into the template structure.
   Output file: `SRS-<project-name>.md`

3. Validate the document:
   ```bash
   python3 <skill-path>/scripts/validate_srs.py <path-to-srs.md>
   ```

4. Fix any reported errors. Re-run until validation passes.

5. Present the completed SRS to the user for final review.

## Phase 5: Traceability (Optional)

When the user requests it, or for systems requiring formal traceability:

Generate `RTM-<project-name>.md` with a Requirements Traceability Matrix mapping:

| Req ID | Requirement | Source/Need | Design Ref | Test Case | Status |
|--------|-------------|-------------|------------|-----------|--------|

See [references/verification-traceability.md](references/verification-traceability.md)
for the full RTM template and guidance.

## Requirement Writing Rules

Every requirement must follow these rules. For detailed guidance and examples,
see [references/requirement-writing-guide.md](references/requirement-writing-guide.md).

**ID scheme**:

| Category | Prefix | Example |
|----------|--------|---------|
| Interface | IR | IR-001 |
| Functional | FR | FR-001 |
| Data | DR | DR-001 |
| Non-Functional | NFR | NFR-101 |
| Constraint | CR | CR-001 |

**Statement rules**:
- Use "shall" for mandatory, "should" for recommended, "may" for optional
- One requirement per statement (no compound "and"/"or" requirements)
- Every requirement must be testable — quantify with measurable targets
- Include inputs, processing, outputs, and error handling
- Avoid ambiguous words: "appropriate", "sufficient", "fast", "user-friendly"

**Requirement template** (compact inline format):

> **FR-001**: *Title* — The system shall [action] [object] [qualifier].
> Input: ... | Output: ... | Error: ... | Priority: Must | Verify: Test |
> Release: MVP

## Definitions & Acronyms

Include these standard definitions in every SRS:

| Term | Definition |
|------|-----------|
| SRS | Software Requirements Specification |
| RTM | Requirements Traceability Matrix |
| MoSCoW | Must/Should/Could/Won't prioritization |
| IADT | Inspection, Analysis, Demonstration, Test |
| FR | Functional Requirement |
| NFR | Non-Functional Requirement |
| DR | Data Requirement |
| IR | Interface Requirement |
| CR | Constraint Requirement |

Add project-specific terms as they emerge during elicitation.

## Change Control

Every SRS document must include version control metadata:

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | YYYY-MM-DD | [Author] | Initial draft |

Requirements are baselined when the SRS version reaches 1.0. After baselining,
changes require a change request with impact analysis.

## Validation Checklist

Track SRS completion progress:

```
SRS Progress:
- [ ] Purpose, scope, and audience defined
- [ ] Definitions and acronyms listed
- [ ] Stakeholders and user classes identified
- [ ] System context and product perspective documented
- [ ] Assumptions and dependencies listed
- [ ] All external interfaces documented (IR-NNN)
- [ ] Functional requirements complete (FR-NNN)
- [ ] Data requirements specified (DR-NNN)
- [ ] Non-functional requirements quantified (NFR-NNN)
- [ ] Constraints documented (CR-NNN)
- [ ] Verification criteria assigned to all requirements
- [ ] Priorities (MoSCoW) and release allocation set
- [ ] Rationale documented for key requirements
- [ ] Change control metadata included
- [ ] validate_srs.py passes with no errors
- [ ] Traceability matrix generated (if required)
```
