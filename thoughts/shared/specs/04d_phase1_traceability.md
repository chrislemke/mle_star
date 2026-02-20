# SRS 04 — Phase 1: Traceability and Change Control

## 10. Traceability Matrix

### 10.1 Requirements to Paper Sections

| Req ID | Paper Section | Paper Element | SDK Construct |
|--------|--------------|---------------|---------------|
| REQ-P1-001 | Section 3.1 | A_retriever agent | `AgentDefinition` |
| REQ-P1-002 | Figure 9 | Retriever prompt template | `prompt` parameter |
| REQ-P1-003 | Figure 9 | Retriever structured output | `output_format` |
| REQ-P1-004 | Figure 9 | Retriever output parsing | `RetrieverOutput.model_validate_json()` |
| REQ-P1-005 | Algorithm 1 line 1 | M models expected | -- |
| REQ-P1-006 | Figure 9 | Model field validation | -- |
| REQ-P1-007 | Algorithm 1 line 1 | `A_retriever(T_task)` invocation | SDK agent call |
| REQ-P1-008 | Section 3.1 | A_init agent | `AgentDefinition` |
| REQ-P1-009 | Figure 10 | Init prompt template | `prompt` parameter |
| REQ-P1-010 | Algorithm 1 line 3 | `s_init^i = A_init(...)` output | -- |
| REQ-P1-011 | Figure 10 | Code block extraction | -- |
| REQ-P1-012 | Algorithm 1 line 3 | `A_init(T_task, T_model^i, T_code^i)` invocation | SDK agent call |
| REQ-P1-013 | Section 3.1 | A_merger agent | `AgentDefinition` |
| REQ-P1-014 | Figure 11 | Merger prompt template | `prompt` parameter |
| REQ-P1-015 | Algorithm 1 line 9 | `s_candidate = A_merger(...)` output | -- |
| REQ-P1-016 | Figure 11 | Code block extraction | -- |
| REQ-P1-017 | Algorithm 1 line 9 | `A_merger(s_0, s_init^{pi(i)})` invocation | SDK agent call |
| REQ-P1-018 | Algorithm 1 | Full Phase 1 entry point | -- |
| REQ-P1-019 | Algorithm 1 line 1 | Model retrieval step | -- |
| REQ-P1-020 | Algorithm 1 lines 2-5 | Candidate generation + evaluation loop | -- |
| REQ-P1-021 | Algorithm 1 lines 2-5 | Failed candidate handling | -- |
| REQ-P1-022 | Algorithm 1 | All candidates failed | -- |
| REQ-P1-023 | Algorithm 1 line 6 | Sort by score (permutation pi) | -- |
| REQ-P1-024 | Algorithm 1 lines 6-7 | Initialize s_0 and h_best | -- |
| REQ-P1-025 | Algorithm 1 lines 8-17 | Merge loop | -- |
| REQ-P1-026 | Algorithm 1 line 11 | `>=` comparison semantics | `is_improvement_or_equal()` |
| REQ-P1-027 | Algorithm 1 lines 14-16 | Break-on-first-failure | -- |
| REQ-P1-028 | Algorithm 1 lines 8-17 | Merge execution failure as failed merge | -- |
| REQ-P1-029 | Algorithm 1 lines 8-17 | Single candidate, no merge | -- |
| REQ-P1-030 | Section 3.4 | `s_0 <- A_data(s_0, T_task)` | -- |
| REQ-P1-031 | Section 3.4 | Leakage check after A_data | -- |
| REQ-P1-032 | Algorithm 1 | Phase1Result construction | -- |
| REQ-P1-033 | Algorithm 1 | Score consistency | -- |
| REQ-P1-034 | Algorithm 1 lines 2-5 | Candidate independence | -- |
| REQ-P1-035 | Algorithm 1 line 4 | Evaluation independence | -- |
| REQ-P1-036 | Algorithm 1 lines 8-17 | Merge loop sequential | -- |
| REQ-P1-037 | -- | Orchestration overhead | -- |
| REQ-P1-038 | -- | Logging | Python `logging` |
| REQ-P1-039 | -- | SDK-only invocation | `claude-agent-sdk` |
| REQ-P1-040 | -- | Module organization | -- |
| REQ-P1-041 | Appendix B | Algorithm 1 fidelity | -- |
| REQ-P1-042 | Section 3.4 | Leakage integration points | -- |
| REQ-P1-043 | Figure 9 | Retriever prompt fidelity | -- |
| REQ-P1-044 | Figure 10 | Init prompt fidelity | -- |
| REQ-P1-045 | Figure 11 | Merger prompt fidelity | -- |

### 10.2 Cross-References to Other Specs

| Req ID | Referenced By |
|--------|--------------|
| REQ-P1-018 (run_phase1) | Spec 09 (Orchestrator invokes Phase 1) |
| REQ-P1-032 (Phase1Result) | Spec 05 (Phase 2 takes Phase1Result.initial_solution as input) |
| REQ-P1-032 (Phase1Result) | Spec 09 (Orchestrator stores Phase1Result in FinalResult) |

### 10.3 Spec 01 Dependencies (Inbound)

| Spec 01 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-DM-001 (PipelineConfig) | REQ-P1-002, REQ-P1-007, REQ-P1-012, REQ-P1-017, REQ-P1-018 | M parameter, config for evaluation |
| REQ-DM-007 (TaskDescription) | REQ-P1-002, REQ-P1-007, REQ-P1-009, REQ-P1-012, REQ-P1-018, REQ-P1-030 | Task description for prompts and evaluation |
| REQ-DM-009 (SolutionScript) | REQ-P1-010, REQ-P1-012, REQ-P1-015, REQ-P1-017, REQ-P1-020 | Solution wrapper for candidate and merged scripts |
| REQ-DM-013 (AgentType) | REQ-P1-001, REQ-P1-008, REQ-P1-013 | Agent identity enum values |
| REQ-DM-014 (RetrievedModel) | REQ-P1-004, REQ-P1-006, REQ-P1-007, REQ-P1-012 | Model schema for retriever output |
| REQ-DM-015 (RetrieverOutput) | REQ-P1-003, REQ-P1-004, REQ-P1-005 | Structured output schema for retriever |
| REQ-DM-022 (Phase1Result) | REQ-P1-018, REQ-P1-032, REQ-P1-033 | Return type of run_phase1 |
| REQ-DM-029 (is_improvement_or_equal) | REQ-P1-025, REQ-P1-026 | Score comparison in merge loop |
| REQ-DM-032 (PromptRegistry) | REQ-P1-002, REQ-P1-007, REQ-P1-009, REQ-P1-012, REQ-P1-014, REQ-P1-017 | Template retrieval for all agents |
| REQ-DM-036 (AgentConfig) | REQ-P1-001, REQ-P1-008, REQ-P1-013 | Agent-to-SDK mapping |

### 10.4 Spec 02 Dependencies (Inbound)

| Spec 02 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-EX-015 (evaluate_solution) | REQ-P1-020, REQ-P1-025, REQ-P1-030, REQ-P1-031 | Evaluate candidate and merged solutions |
| REQ-EX-021 (evaluate_with_retry) | REQ-P1-020, REQ-P1-025, REQ-P1-030, REQ-P1-031 | Evaluate with debug retry support |
| REQ-EX-026 (evaluate_batch) | REQ-P1-035 | Batch evaluation of candidates |
| REQ-EX-027 (rank_solutions) | REQ-P1-023 | Sort candidates by score |

### 10.5 Spec 03 Dependencies (Inbound)

| Spec 03 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-SF-005 (extract_code_block) | REQ-P1-011, REQ-P1-016 | Code extraction from A_init and A_merger responses |
| REQ-SF-007 (make_debug_callback) | REQ-P1-020, REQ-P1-025 | Debug callback for evaluate_with_retry |
| REQ-SF-020 (check_and_fix_leakage) | REQ-P1-020, REQ-P1-025, REQ-P1-031, REQ-P1-042 | Leakage detection and correction before evaluation |
| REQ-SF-022 (leakage integration point) | REQ-P1-042 | Leakage check requirement on every solution |
| REQ-SF-030 (check_data_usage) | REQ-P1-030 | Data usage check after merging |

---

## 11. Change Control

### 11.1 Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2026-02-20 | MLE-STAR Team | Initial draft -- all 45 requirements |

### 11.2 Baselining Policy

This SRS is baselined at version 1.0. After baselining, changes require impact analysis against Specs 05 and 09 (downstream consumers of Phase1Result), Spec 01 (upstream data model dependencies), Spec 02 (upstream execution harness dependencies), and Spec 03 (upstream safety agent dependencies).
