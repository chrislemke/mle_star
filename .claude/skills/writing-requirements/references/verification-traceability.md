# Verification Methods & Requirements Traceability

Reference for assigning verification methods to requirements and building
a Requirements Traceability Matrix (RTM).

## Verification Methods (IADT)

ISO/IEC/IEEE 29148 defines four verification methods. Assign at least one
to every requirement.

| Method | Abbr | Definition | When to Use | Example |
|--------|------|-----------|-------------|---------|
| Inspection | I | Visual examination of documents, code, or configuration | Standards compliance, code conventions, config correctness | Review security headers in HTTP config |
| Analysis | A | Mathematical, statistical, or logical evaluation | Performance models, capacity planning, algorithmic correctness | Calculate max throughput from architecture |
| Demonstration | D | Operate the system in a scenario to show capability | User workflows, integration scenarios, UI behavior | Walk through the user registration flow |
| Test | T | Execute with known inputs and compare to expected outputs | Functional behavior, performance benchmarks, edge cases | Run load test with 1,000 concurrent users |

### Selecting the Right Method

| Requirement Type | Primary Method | Secondary Method |
|-----------------|----------------|------------------|
| Functional (FR) | Test | Demonstration |
| Performance (NFR-1xx) | Test | Analysis |
| Reliability (NFR-2xx) | Test + Analysis | Demonstration |
| Security (NFR-3xx) | Test + Inspection | Analysis |
| Maintainability (NFR-4xx) | Inspection | Demonstration |
| Scalability (NFR-5xx) | Test | Analysis |
| Usability (NFR-6xx) | Test (usability testing) | Demonstration |
| Portability (NFR-7xx) | Demonstration | Test |
| Data (DR) | Test | Inspection |
| Interface (IR) | Test | Demonstration |
| Constraint (CR) | Inspection | Analysis |

## Acceptance Criteria Patterns

Use one of these formats for each requirement's acceptance criterion.

### Given/When/Then (Behavioral)

Best for functional requirements and user-facing behavior:

```
Given [precondition]
When [action or trigger]
Then [expected result]
```

Example:
```
Given a registered user with valid credentials
When the user submits the login form
Then the system shall authenticate the user within 2 seconds
  and redirect to the dashboard
  and create an audit log entry
```

### Quantitative (Measurable)

Best for non-functional requirements:

```
Metric:    [what is measured]
Target:    [threshold value]
Method:    [how to measure]
Duration:  [measurement window]
Condition: [under what circumstances]
```

Example:
```
Metric:    API response time (p95)
Target:    < 200ms
Method:    Application performance monitoring
Duration:  Measured over 24-hour window
Condition: Under normal load (up to 500 req/s)
```

### Checklist (Compliance)

Best for constraints and standards compliance:

```
- [ ] [Criterion 1]
- [ ] [Criterion 2]
- [ ] [Criterion 3]
Evidence: [How compliance is documented]
```

## Requirements Traceability Matrix (RTM)

The RTM links requirements across the development lifecycle:

```
Stakeholder Need → System Requirement → Software Requirement → Design → Test Case
```

### RTM Document Template

# Requirements Traceability Matrix: [System Name]

| Field | Value |
|-------|-------|
| Version | [0.1] |
| Date | [YYYY-MM-DD] |
| SRS Reference | [SRS document version] |

### Forward Traceability (Requirements → Tests)

| Req ID | Requirement Summary | Stakeholder Need | Design Component | Test Case(s) | Test Status |
|--------|-------------------|-----------------|-----------------|-------------|-------------|
| FR-001 | [Summary] | [SN-001] | [Module/Class] | [TC-001, TC-002] | [Pass/Fail/Pending] |
| FR-002 | [Summary] | [SN-001] | [Module/Class] | [TC-003] | [Pending] |
| NFR-101 | [Summary] | [SN-002] | [Architecture decision] | [TC-010] | [Pending] |

### Backward Traceability (Tests → Requirements)

| Test Case | Test Description | Requirement(s) Verified | Result | Date |
|-----------|-----------------|------------------------|--------|------|
| TC-001 | [What the test does] | FR-001 | [Pass/Fail] | [Date] |

### Stakeholder Needs

| Need ID | Description | Source | Priority | Linked Requirements |
|---------|-------------|--------|----------|-------------------|
| SN-001 | [Stakeholder need] | [Who raised it] | [Must/Should/Could] | FR-001, FR-002 |

## Traceability Levels

| Level | Direction | Purpose |
|-------|-----------|---------|
| Forward | Requirements → Design → Tests | Ensures every requirement is implemented and tested |
| Backward | Tests → Design → Requirements | Ensures no orphan tests; every test traces to a requirement |
| Bidirectional | Both directions | Full coverage — recommended for safety-critical or regulated systems |

## Coverage Analysis

Use the RTM to identify:

| Issue | Detection Method | Action |
|-------|-----------------|--------|
| Untested requirements | Requirements with no linked test cases | Add test cases or justify exemption |
| Orphan tests | Test cases with no linked requirements | Remove or link to requirements |
| Gold plating | Design components not linked to any requirement | Remove or create justifying requirement |
| Unmet stakeholder needs | Needs with no linked requirements | Elicit missing requirements |
| Missing design | Requirements with no linked design component | Complete design |

## Apportioning of Requirements

When requirements need to be allocated across software elements or subsystems:

| Req ID | Requirement | Allocated To | Rationale |
|--------|-------------|-------------|-----------|
| FR-001 | [Summary] | [Component/Module] | [Why allocated here] |

Include a cross-reference table mapping requirements to software elements.
Explicitly identify deferred requirements and document the reason for deferral:

| Req ID | Requirement | Deferred To | Reason |
|--------|-------------|-------------|--------|
| FR-050 | [Summary] | Phase 2 | [Justification] |
