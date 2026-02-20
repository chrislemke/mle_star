# SRS 01 — Data Models: Constraints and Traceability

## 6. Non-Functional Requirements

### 6.1 Performance

> **REQ-DM-044**: *Model Instantiation Speed* — All Pydantic models defined in this spec shall instantiate in under 1 millisecond for typical payloads (solution scripts up to 50 KB, plan texts up to 2 KB).
>
> - Measurement: `timeit` benchmark of model construction
> - Priority: Should | Verify: Test | Release: MVP

### 6.2 Maintainability

> **REQ-DM-045**: *Single Module* — All data models defined in this spec shall reside in a single Python module (e.g., `mle_star/models.py`) to centralize type definitions.
>
> - Priority: Should | Verify: Inspection | Release: MVP

> **REQ-DM-046**: *Re-export Convenience* — The package `__init__.py` shall re-export all public models and enums from the data models module.
>
> - Priority: Should | Verify: Inspection | Release: MVP

### 6.3 Portability

> **REQ-DM-047**: *Python Version Compatibility* — All data models shall be compatible with Python 3.10, 3.11, 3.12, and 3.13.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Acceptance: All models shall pass type checking and instantiation tests on each listed Python version.

---

## 7. Constraints

### 7.1 Technology Constraints

> **REQ-DM-048**: *Pydantic v2 Dependency* — All data models shall use Pydantic v2 (>=2.0.0) BaseModel as their base class.
>
> - Rationale: Pydantic v2 provides `.model_json_schema()` needed for SDK `output_format` integration.
> - Priority: Must | Verify: Inspection | Release: MVP

> **REQ-DM-049**: *SDK Structured Output Compatibility* — All structured output schemas shall conform to the `{"type": "json_schema", "schema": ...}` format expected by `claude-agent-sdk` `ClaudeAgentOptions.output_format`.
>
> - Priority: Must | Verify: Test | Release: MVP
> - Source: REF-06 Section 12

> **REQ-DM-050**: *No External Dependencies Beyond Pydantic* — The data models module shall not depend on any packages other than `pydantic`, Python standard library modules, and type stubs.
>
> - Rationale: Keep the foundation module lightweight; SDK types are referenced by string names, not by import.
> - Priority: Must | Verify: Inspection | Release: MVP

---

## 8. Traceability Matrix

### 8.1 Requirements to Paper Sections

| Req ID | Paper Section | Paper Element | SDK Construct |
|--------|--------------|---------------|---------------|
| REQ-DM-001 | Section 4 | Hyperparameters M,T,K,L,R | — |
| REQ-DM-002 | Section 4 | Validation | Pydantic validator |
| REQ-DM-003 | — | Serialization | — |
| REQ-DM-004 | Table 7 | Competition categories | — |
| REQ-DM-005 | Section 3 | Data modalities | — |
| REQ-DM-006 | Section 3 | Score direction | — |
| REQ-DM-007 | Section 3 | T_task | — |
| REQ-DM-008 | Algorithms 1-3 | Pipeline phases | — |
| REQ-DM-009 | Eq. 2,3,7,9 | Solution scripts | — |
| REQ-DM-010 | Eq. 7 | `s_t.replace(c_t, c_t^k)` | — |
| REQ-DM-011 | Section 3.2 | Code block categories | — |
| REQ-DM-012 | Eq. 6 | Code block c_t | — |
| REQ-DM-013 | Section 6 | All 14 agents | `AgentDefinition` |
| REQ-DM-014 | Figure 9 | Retriever JSON schema | `output_format` |
| REQ-DM-015 | Figure 9 | Retriever output | `output_format` |
| REQ-DM-016 | Figure 14 | Extractor JSON schema | `output_format` |
| REQ-DM-017 | Figure 14 | Extractor output | `output_format` |
| REQ-DM-018 | Figure 20 | Leakage JSON schema | `output_format` |
| REQ-DM-019 | Figure 20 | Leakage output | `output_format` |
| REQ-DM-020 | Figure 28 | Contamination check | `output_format` |
| REQ-DM-021 | Section 3.4 | Execution result + traceback | — |
| REQ-DM-022 | Algorithm 1 | Phase 1 outputs | — |
| REQ-DM-023 | Algorithm 2 | Phase 2 outputs | — |
| REQ-DM-024 | Algorithm 3 | Phase 3 outputs | — |
| REQ-DM-025 | Full pipeline | Final result | `ResultMessage` |
| REQ-DM-026 | Section 3 | h: S -> R | — |
| REQ-DM-027 | Figures 10-19 | "Final Validation Performance" | — |
| REQ-DM-028 | Alg 2 lines 12,21 | Score comparison | — |
| REQ-DM-029 | Alg 1 line 11 | Score comparison (>=) | — |
| REQ-DM-030 | Figures 9-28 | Prompt templates | — |
| REQ-DM-031 | Figures 9-28 | Template rendering | — |
| REQ-DM-032 | Figures 9-28 | Template registry | — |
| REQ-DM-033 | Appendix A | 14 agent templates | — |
| REQ-DM-034 | Figures 20,21 | Leakage dual templates | — |
| REQ-DM-035 | Figures 26,27 | Subsampling templates | — |
| REQ-DM-036 | — | Agent-to-SDK mapping | `AgentDefinition` |
| REQ-DM-037 | — | SDK dict output | `ClaudeAgentOptions.agents` |
| REQ-DM-038 | — | Structured output config | `ClaudeAgentOptions.output_format` |
| REQ-DM-039 | — | Immutability | `ConfigDict(frozen=True)` |
| REQ-DM-040 | Section 6 | All 14 agents configured | `AgentDefinition` |
| REQ-DM-041 | — | JSON schema validity | `output_format` |
| REQ-DM-042 | Algorithm 2 | Refinement history | — |
| REQ-DM-043 | Algorithm 3 | Ensemble history | — |
| REQ-DM-044 | — | Performance | — |
| REQ-DM-045 | — | Module organization | — |
| REQ-DM-046 | — | Re-exports | — |
| REQ-DM-047 | — | Python compat | — |
| REQ-DM-048 | — | Pydantic v2 | — |
| REQ-DM-049 | — | SDK compat | `output_format` |
| REQ-DM-050 | — | Minimal deps | — |

### 8.2 Cross-References to Other Specs

| Req ID | Referenced By |
|--------|--------------|
| REQ-DM-001 (PipelineConfig) | Specs 02-09 |
| REQ-DM-007 (TaskDescription) | Specs 02, 04, 08, 09 |
| REQ-DM-009 (SolutionScript) | Specs 02-09 |
| REQ-DM-010 (replace_block) | Specs 05, 06 |
| REQ-DM-012 (CodeBlock) | Specs 05, 06 |
| REQ-DM-013 (AgentType) | Specs 03-09 |
| REQ-DM-014-019 (Structured Schemas) | Specs 03, 04, 05 |
| REQ-DM-021 (EvaluationResult) | Spec 02 |
| REQ-DM-026-029 (Score Functions) | Spec 02 |
| REQ-DM-030-035 (Prompt Registry) | Specs 03-08 |
| REQ-DM-036-038 (AgentConfig) | Spec 09 |
| REQ-DM-042-043 (History Models) | Specs 05, 06, 07 |

---

## 9. Change Control

### 9.1 Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2026-02-20 | MLE-STAR Team | Initial draft — all 50 requirements |

### 9.2 Baselining Policy

This SRS is baselined at version 1.0. After baselining, changes require impact analysis across all 9 specs due to this document's foundational role.
