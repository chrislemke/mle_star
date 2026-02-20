# SRS Document Template (ISO/IEC/IEEE 29148)

Use this template to assemble the final SRS document. Replace all `[...]`
placeholders with project-specific content. Delete sections marked "if
applicable" when they do not apply.

---

# Software Requirements Specification: [System Name]

| Field | Value |
|-------|-------|
| Version | [0.1] |
| Date | [YYYY-MM-DD] |
| Status | [Draft / In Review / Baselined] |
| Authors | [Names] |
| Approved By | [Name, Role] |

## Table of Contents

<!-- Generate from headings after assembly -->

---

## 1. Introduction

### 1.1 Purpose

This SRS describes the software requirements for [system name]. It is intended
for [audience: developers, testers, project managers, stakeholders, regulators].

This document supports the following decisions:
- [e.g., Design and implementation choices]
- [e.g., Test planning and acceptance criteria]
- [e.g., Stakeholder sign-off]

### 1.2 Scope

**Product name**: [Name]

**What the product does**: [High-level description of capabilities, benefits,
objectives, and goals. Must be consistent with any higher-level specifications.]

**In scope**:
- [Capability 1]
- [Capability 2]

**Out of scope**:
- [Explicitly excluded item 1]
- [Explicitly excluded item 2]

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|------|-----------|
| SRS | Software Requirements Specification |
| RTM | Requirements Traceability Matrix |
| [Term] | [Definition] |

### 1.4 References

| ID | Title | Version | Source |
|----|-------|---------|--------|
| [REF-01] | [Document title] | [Version] | [URL or location] |

### 1.5 Document Overview

This SRS is organized as follows:
- Section 2: Product perspective and context
- Section 3: External interface requirements
- Section 4: Functional requirements
- Section 5: Data requirements
- Section 6: Non-functional requirements
- Section 7: Constraints
- Section 8: Change control
- Section 9: Appendices

---

## 2. Product Perspective

### 2.1 System Context

[Describe the system boundary. How does the product relate to the larger
system or environment? Consider a context diagram showing external actors
and systems.]

**Related systems**:
- [System A — relationship/interface description]
- [System B — relationship/interface description]

**System boundary constraints**:
- [Constraint from the larger system]

### 2.2 Product Functions Summary

High-level capabilities (detailed in Section 4):

1. [Capability 1 — brief description]
2. [Capability 2 — brief description]
3. [Capability 3 — brief description]

### 2.3 Stakeholders and User Classes

| User Class | Description | Frequency | Technical Proficiency | Key Needs |
|------------|-------------|-----------|----------------------|-----------|
| [Role] | [Who they are] | [Daily/Weekly/Occasional] | [High/Medium/Low] | [Primary needs] |

### 2.4 Operating Environment

- **Platforms**: [OS, browsers, devices]
- **Infrastructure**: [Cloud provider, on-premises, hybrid]
- **Runtime**: [Language runtime, framework versions]
- **Dependencies**: [Required services, databases, message queues]

### 2.5 Assumptions and Dependencies

**Assumptions** (if proven false, requirements may change):

| ID | Assumption | Impact if Invalid |
|----|-----------|-------------------|
| A-01 | [Assumption] | [Impact] |

**Dependencies** (external factors the project relies on):

| ID | Dependency | Owner | Risk if Unavailable |
|----|-----------|-------|---------------------|
| D-01 | [Dependency] | [Who controls it] | [Impact] |

---

## 3. External Interface Requirements

### 3.1 User Interfaces (IR-0xx)

For each key screen or interaction flow:

> **IR-001**: *[Screen/Flow Name]* — The system shall provide [description].
>
> - Purpose: [Why this interface exists]
> - Inputs: [User-provided data, actions]
> - Outputs: [System responses, displayed information]
> - Behavior: [Key interaction rules]
> - Error handling: [Validation messages, recovery]
> - Priority: [Must/Should/Could/Won't] | Verify: [Method] | Release: [Phase]

### 3.2 Software/API Interfaces (IR-1xx)

For each API endpoint or service contract:

> **IR-101**: *[API/Service Name]* — The system shall [expose/consume]
> [endpoint/service] that [purpose].
>
> - Protocol: [REST/gRPC/GraphQL/etc.]
> - Authentication: [Method]
> - Request: [Format, key fields]
> - Response: [Format, key fields]
> - Error codes: [Expected error responses]
> - Rate limits: [If applicable]
> - Priority: [Must/Should/Could/Won't] | Verify: [Method] | Release: [Phase]

### 3.3 Hardware Interfaces (IR-2xx)

*[If applicable. Delete this section if no hardware interfaces exist.]*

> **IR-201**: *[Hardware Name]* — The system shall interface with [device]
> via [protocol/connection].
>
> - Data format: [Specification]
> - Timing: [Frequency, latency requirements]

### 3.4 Communication Interfaces (IR-3xx)

> **IR-301**: *[Protocol/Channel Name]* — The system shall communicate
> via [protocol] on [port/channel].
>
> - Data format: [JSON/XML/binary/etc.]
> - Security: [TLS version, certificate requirements]
> - Network constraints: [Bandwidth, latency, firewall rules]

---

## 4. Functional Requirements

Organize by feature area or use case. Each requirement follows this format:

> **FR-NNN**: *Title* — The system shall [verb] [object] [qualifier].
>
> - Input: [What triggers or feeds this function]
> - Processing: [Rules, algorithms, validations]
> - Output: [What the system produces]
> - Error handling: [Abnormal conditions and system response]
> - Priority: [Must/Should/Could/Won't]
> - Verification: [Test/Inspection/Analysis/Demonstration]
> - Acceptance criteria: [Measurable pass/fail condition]
> - Rationale: [Why this requirement exists — source, trade-off]
> - Release: [MVP / Phase 2 / Deferred]

### 4.1 [Feature Area 1]

[Group related requirements under feature headings]

### 4.2 [Feature Area 2]

### 4.3 [Feature Area N]

---

## 5. Data Requirements

### 5.1 Data Entities and Relationships (DR-0xx)

| Entity | Description | Key Attributes | Relationships |
|--------|-------------|----------------|---------------|
| [Entity] | [Purpose] | [Key fields] | [Related entities] |

> **DR-001**: The system shall maintain [entity] records with [attributes].
> Integrity constraint: [rule]. Validation: [rules].

### 5.2 Data Validation Rules (DR-1xx)

> **DR-101**: The system shall validate [field] against [rule] before
> [operation]. Invalid data shall [rejection behavior].

### 5.3 Data Retention and Archival (DR-2xx)

> **DR-201**: The system shall retain [data type] for [duration].
> After [duration], the system shall [archive/delete/anonymize].

### 5.4 Data Migration and Import/Export (DR-3xx)

*[If applicable.]*

> **DR-301**: The system shall [import/export] [data type] in [format]
> via [mechanism].

---

## 6. Non-Functional Requirements

Each NFR must have a measurable target.

### 6.1 Performance (NFR-1xx)

> **NFR-101**: The system shall [respond to / process] [operation] within
> [N] [ms/seconds] under [load condition].
>
> - Measurement: [How to measure]
> - Target: [Specific threshold]

### 6.2 Reliability and Availability (NFR-2xx)

> **NFR-201**: The system shall maintain [N]% availability measured over
> [period], excluding planned maintenance windows.
>
> - RTO: [Recovery Time Objective]
> - RPO: [Recovery Point Objective]

### 6.3 Security and Privacy (NFR-3xx)

> **NFR-301**: The system shall [security measure] for [scope].
>
> - Authentication: [Method]
> - Authorization: [Model — RBAC/ABAC/etc.]
> - Encryption: [At rest / in transit — standards]
> - Audit: [What is logged, retention]
> - Compliance: [GDPR/HIPAA/SOC2/etc.]

### 6.4 Maintainability (NFR-4xx)

> **NFR-401**: The system shall [maintainability measure].
>
> - Observability: [Logging, metrics, tracing]
> - Deployment: [Strategy — blue/green, rolling, etc.]
> - Documentation: [Required artifacts]

### 6.5 Scalability (NFR-5xx)

> **NFR-501**: The system shall support [N] [concurrent users / requests
> per second / records] with [growth assumption over time period].

### 6.6 Usability and Accessibility (NFR-6xx)

> **NFR-601**: [User class] shall be able to [task] with [success rate]%
> success rate within [time/attempts].
>
> - Accessibility: [WCAG level, if applicable]

### 6.7 Portability and Compatibility (NFR-7xx)

> **NFR-701**: The system shall operate on [platforms/browsers/versions].

---

## 7. Constraints

### 7.1 Regulatory Constraints (CR-1xx)

> **CR-101**: The system shall comply with [regulation/standard].
> Evidence: [How compliance is demonstrated].

### 7.2 Technology Constraints (CR-2xx)

> **CR-201**: The system shall be implemented using [technology] due to
> [reason — organizational mandate, existing infrastructure, licensing].

### 7.3 Operational Constraints (CR-3xx)

> **CR-301**: The system shall [operational constraint] due to
> [site/operational limitation].

---

## 8. Change Control

### 8.1 Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | [Date] | [Author] | Initial draft |

### 8.2 Baselining Policy

This SRS is baselined at version 1.0. After baselining:
- Changes require a formal change request
- Each change request must include impact analysis
- Approved changes increment the version number
- All changes are recorded in the version history

---

## 9. Appendices

### 9.1 Glossary

*[Project-specific terms not covered in Section 1.3]*

### 9.2 Analysis Models

*[Diagrams, state machines, data flow diagrams — reference or embed]*

### 9.3 To Be Determined (TBD) List

| TBD ID | Description | Owner | Target Resolution Date |
|--------|-------------|-------|----------------------|
| TBD-01 | [Open question] | [Who resolves it] | [Date] |

### 9.4 Supporting Information

*[Any additional information. State explicitly whether this content is
part of the requirements or informational only.]*
