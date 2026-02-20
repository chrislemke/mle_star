# Requirement Writing Guide

Practical reference for writing testable, unambiguous requirements
aligned with ISO/IEC/IEEE 29148.

## Shall-Statement Syntax

Pattern: **"The system shall `<verb>` `<object>` `<qualifier>`."**

| Keyword | Meaning | When to Use |
|---------|---------|-------------|
| shall | Mandatory | Binding requirement — must be implemented and verified |
| should | Recommended | Important but not critical; deviation needs justification |
| may | Optional | Permitted behavior; included at implementer's discretion |

Rules:
- Active voice, present tense
- One action per requirement (no compound "and"/"or")
- Subject is always "the system" unless specifying a component
- Every shall-statement must have a corresponding verification method

## Requirement ID Scheme

| Category | Prefix | Range | Example | Use For |
|----------|--------|-------|---------|---------|
| Interface | IR | IR-001..IR-399 | IR-015 | UI, API, hardware, communication interfaces |
| Functional | FR | FR-001..FR-999 | FR-042 | System behavior, processing, business logic |
| Data | DR | DR-001..DR-399 | DR-007 | Entities, validation, retention, migration |
| Non-Functional | NFR | NFR-101..NFR-799 | NFR-301 | Performance, security, scalability, usability |
| Constraint | CR | CR-001..CR-399 | CR-012 | Regulatory, technology, operational constraints |

NFR sub-ranges:
- NFR-1xx: Performance
- NFR-2xx: Reliability/Availability
- NFR-3xx: Security/Privacy
- NFR-4xx: Maintainability
- NFR-5xx: Scalability
- NFR-6xx: Usability/Accessibility
- NFR-7xx: Portability/Compatibility

## Testability Criteria

A requirement is testable if you can define:
1. **Inputs**: What triggers or feeds the behavior
2. **Expected outputs**: What the system should produce
3. **Pass/fail criteria**: How to determine success
4. **Repeatability**: The test produces consistent results

### Measurable vs Unmeasurable Terms

| Avoid (Vague) | Use Instead (Measurable) |
|---------------|-------------------------|
| fast | within 200ms |
| user-friendly | complete task in 3 clicks |
| reliable | 99.9% uptime over 30 days |
| flexible | configurable via JSON/YAML |
| scalable | support 10,000 concurrent users |
| secure | encrypted with AES-256 |
| efficient | process 1,000 records/second |
| easy to maintain | deploy in under 5 minutes |
| robust | recover within 30 seconds of failure |
| intuitive | 90% task completion on first attempt |
| real-time | end-to-end latency under 100ms |
| high quality | zero critical defects in production |
| minimal downtime | planned maintenance under 4 hours/month |

## Common Anti-Patterns

| Anti-Pattern | Example | Problem | Fix |
|-------------|---------|---------|-----|
| Compound requirement | "The system shall validate input **and** store results" | Two behaviors in one ID — partial implementation is ambiguous | Split into FR-001 (validate) and FR-002 (store) |
| Vague qualifier | "The system shall respond **appropriately**" | No measurable criterion | "shall return HTTP 400 with error code E-101" |
| Implementation-prescriptive | "The system shall use **PostgreSQL**" | Constrains design without justification | Move to constraints section with rationale, or state "shall persist data in a relational database" |
| Untestable | "The system shall be **easy to use**" | No objective measure | "90% of [user class] shall complete [task] within [time]" |
| Missing error handling | "The system shall accept file uploads" | No abnormal behavior defined | Add: "Files exceeding 10MB shall be rejected with error E-201" |
| Missing boundary | "The system shall accept user input" | No limits defined | "shall accept input strings of 1-256 characters" |
| Wishful thinking | "The system shall **never** fail" | Impossible to verify | "shall maintain 99.95% availability measured monthly" |
| Passive voice | "Input **shall be validated**" | Unclear who/what acts | "The system shall validate input against [rules]" |
| Negation chains | "The system shall not fail to prevent unauthorized access" | Confusing logic | "The system shall reject requests from unauthenticated users" |

## Compound Requirement Splitting

When you find "and", "or", "also", or "additionally" in a requirement,
split it:

**Before** (bad):
> FR-010: The system shall validate the email format, check for
> duplicates, and send a confirmation email.

**After** (good):
> FR-010: The system shall validate that email addresses match RFC 5322 format.
> Error: Invalid format shall return error E-010.
>
> FR-011: The system shall reject registration if the email address
> already exists. Error: Duplicate shall return error E-011.
>
> FR-012: The system shall send a confirmation email to the registered
> address within 30 seconds of successful registration.

## Good vs Bad Examples

### Example 1: Functional Requirement

**Bad**:
> The system should search products and display them nicely.

**Good**:
> **FR-017**: *Product Search* — The system shall return search results
> within 2 seconds for queries matching up to 10,000 records.
>
> - Input: Search query string (1-256 characters)
> - Processing: Full-text search against product catalog
> - Output: Paginated list of matching products (max 50 per page)
> - Error: Empty query shall return validation error E-101
> - Priority: Must | Verify: Test | Release: MVP

### Example 2: Non-Functional Requirement

**Bad**:
> The system should be available and reliable.

**Good**:
> **NFR-201**: *Service Availability* — The system shall maintain 99.9%
> availability measured over each calendar month, excluding scheduled
> maintenance windows of no more than 4 hours per month.
>
> - RTO: 15 minutes
> - RPO: 5 minutes
> - Measurement: Uptime monitoring via health check endpoint
> - Priority: Must | Verify: Analysis + Test | Release: MVP

### Example 3: Data Requirement

**Bad**:
> User data should be kept safe and deleted when no longer needed.

**Good**:
> **DR-201**: *User Data Retention* — The system shall retain user
> account data for 90 days after account deletion. After 90 days, the
> system shall permanently delete all personally identifiable information.
>
> - Scope: All PII fields (name, email, address, phone)
> - Method: Automated batch deletion job, daily at 02:00 UTC
> - Audit: Deletion events shall be logged with timestamp and record count
> - Priority: Must | Verify: Test | Release: MVP

## Ambiguity Checklist

Words that must be eliminated or replaced with measurable terms:

| Word | Ask Instead |
|------|------------|
| appropriate | What specific criteria? |
| sufficient | What quantity or threshold? |
| adequate | What minimum standard? |
| reasonable | What specific limit? |
| as much as possible | What is the measurable target? |
| best effort | What is the minimum acceptable outcome? |
| timely | Within what time period? |
| user-friendly | What task success rate/time? |
| seamless | What is the acceptable transition delay? |
| state-of-the-art | Which specific standard or benchmark? |
| maximize / minimize | What is the target value? |
| support | What specific capability? |
| handle | What specific behavior on what input? |
| process | What transformation, with what inputs/outputs? |
| manage | What CRUD operations, by whom? |
| etc. / and so on | List all items explicitly |
| if possible | Is this a requirement or not? Decide. |
| TBD | Resolve before baselining — track in TBD list |
