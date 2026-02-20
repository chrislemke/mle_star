# SRS 07 — Phase 3: Traceability & Change Control

---

## 8. Traceability Matrix

### 8.1 Requirements to Paper Sections

| Req ID | Paper Section | Paper Element | SDK Construct |
|--------|--------------|---------------|---------------|
| REQ-P3-001 | Section 3.3 | A_ens_planner agent | `AgentDefinition` |
| REQ-P3-002 | Figure 17 | Ens_planner prompt template | `prompt` parameter |
| REQ-P3-003 | Algorithm 3 input | L solutions input | -- |
| REQ-P3-004 | Algorithm 3 line 1 | First invocation (no history) | -- |
| REQ-P3-005 | Algorithm 3 line 5 | Subsequent invocations (with history) | -- |
| REQ-P3-006 | Figure 17 | Ensemble plan output (natural language) | -- |
| REQ-P3-007 | Figure 17 | "differ from the previous plans" | -- |
| REQ-P3-008 | Figure 17 | "concentrate how to merge" | -- |
| REQ-P3-009 | Algorithm 3 lines 1, 5 | `A_ens_planner(...)` invocation | SDK agent call |
| REQ-P3-010 | Section 3.3 | A_ensembler agent | `AgentDefinition` |
| REQ-P3-011 | Figure 18 | Ensembler prompt template | `prompt` parameter |
| REQ-P3-012 | Algorithm 3 lines 2, 6 | `A_ensembler(e_r, solutions)` input | -- |
| REQ-P3-013 | Algorithm 3 lines 2, 6 | `s_ens^r = A_ensembler(...)` output | -- |
| REQ-P3-014 | Figure 18 | "Do not subsample" | -- |
| REQ-P3-015 | Figure 18 | `./final/submission.csv` | REQ-EX-024 |
| REQ-P3-016 | Algorithm 3 lines 2, 6 | Ensembler invocation function | SDK agent call |
| REQ-P3-017 | Algorithm 3 | Full Phase 3 entry point | -- |
| REQ-P3-018 | Section 3.3 | L=1 skip condition | -- |
| REQ-P3-019 | Algorithm 3 line 1 | Initial ensemble plan (r=0) | -- |
| REQ-P3-020 | Algorithm 3 line 2 | Initial ensemble implementation (r=0) | -- |
| REQ-P3-021 | Algorithm 3 line 3 | Initial ensemble evaluation (r=0) | -- |
| REQ-P3-022 | Algorithm 3 lines 4-8 | Subsequent rounds iteration loop | -- |
| REQ-P3-023 | Algorithm 3 line 5 | Full history to A_ens_planner | -- |
| REQ-P3-024 | Algorithm 3 lines 3, 7 | Score tracking | -- |
| REQ-P3-025 | Algorithm 3 line 9 | `r* = argmax h(s_ens^r)` best selection | -- |
| REQ-P3-026 | -- | All rounds failed fallback | -- |
| REQ-P3-027 | Section 3.4 | Leakage check before evaluation | REQ-SF-022 |
| REQ-P3-028 | Section 3.4 | A_debugger on execution error | REQ-SF-006, REQ-SF-007 |
| REQ-P3-029 | -- | Ensemble script failure handling | -- |
| REQ-P3-030 | -- | Ensembler extraction failure handling | -- |
| REQ-P3-031 | -- | Ens_planner failure handling | -- |
| REQ-P3-032 | Algorithm 3 | EnsembleAttempt record | REQ-DM-043 |
| REQ-P3-033 | Algorithm 3 | History ordering | -- |
| REQ-P3-034 | Algorithm 3 | Phase3Result construction | REQ-DM-024 |
| REQ-P3-035 | Algorithm 3 | Score consistency | -- |
| REQ-P3-036 | -- | Orchestration overhead | -- |
| REQ-P3-037 | Section 4 | Phase 3 duration budget | -- |
| REQ-P3-038 | -- | Never raises on round failure | -- |
| REQ-P3-039 | -- | Logging | Python `logging` |
| REQ-P3-040 | -- | SDK-only invocation | `claude-agent-sdk` |
| REQ-P3-041 | -- | Module organization | -- |
| REQ-P3-042 | Appendix B | Algorithm 3 fidelity | -- |
| REQ-P3-043 | Algorithm 3 lines 4-8 | Sequential rounds | -- |
| REQ-P3-044 | Algorithm 3 | Iteration count = R | -- |
| REQ-P3-045 | Section 3.4 | Leakage integration points | REQ-SF-022 |
| REQ-P3-046 | Figure 17 | Ens_planner prompt fidelity | -- |
| REQ-P3-047 | Figure 18 | Ensembler prompt fidelity | -- |
| REQ-P3-048 | Figure 18 | Ensemble scripts are full programs | -- |
| REQ-P3-049 | Figure 18, Figures 10-19 | "Final Validation Performance" output | REQ-DM-027 |

### 8.2 Cross-References to Other Specs

| Req ID | Referenced By |
|--------|--------------|
| REQ-P3-017 (run_phase3) | Spec 09 (Orchestrator invokes Phase 3) |
| REQ-P3-034 (Phase3Result) | Spec 08 (Finalization takes Phase3Result.best_ensemble as input) |
| REQ-P3-034 (Phase3Result) | Spec 09 (Orchestrator stores Phase3Result in FinalResult) |

### 8.3 Spec 01 Dependencies (Inbound)

| Spec 01 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-DM-001 (PipelineConfig) | REQ-P3-002, REQ-P3-017, REQ-P3-022, REQ-P3-044 | L and R parameters, max_debug_attempts |
| REQ-DM-006 (MetricDirection) | REQ-P3-025 | Score comparison direction for best selection |
| REQ-DM-007 (TaskDescription) | REQ-P3-017, REQ-P3-021, REQ-P3-022, REQ-P3-028 | Task context for evaluation and metric direction |
| REQ-DM-009 (SolutionScript) | REQ-P3-003, REQ-P3-013, REQ-P3-016, REQ-P3-017, REQ-P3-018 | Solution type for input/output |
| REQ-DM-013 (AgentType) | REQ-P3-001, REQ-P3-010 | Agent identity enum values (ens_planner, ensembler) |
| REQ-DM-024 (Phase3Result) | REQ-P3-017, REQ-P3-018, REQ-P3-034, REQ-P3-035 | Return type of run_phase3 |
| REQ-DM-027 (Score Parsing) | REQ-P3-049 | "Final Validation Performance" regex pattern |
| REQ-DM-032 (PromptRegistry) | REQ-P3-002, REQ-P3-009, REQ-P3-011, REQ-P3-016 | Template retrieval for both agents |
| REQ-DM-036 (AgentConfig) | REQ-P3-001, REQ-P3-010 | Agent-to-SDK mapping |
| REQ-DM-043 (EnsembleAttempt) | REQ-P3-032, REQ-P3-033 | Ensemble history records |

### 8.4 Spec 02 Dependencies (Inbound)

| Spec 02 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-EX-015 (evaluate_solution) | REQ-P3-021, REQ-P3-022 | Evaluate ensemble solutions |
| REQ-EX-021 (evaluate_with_retry) | REQ-P3-021, REQ-P3-022, REQ-P3-028 | Evaluate with debug retry support |
| REQ-EX-024 (verify_submission) | REQ-P3-015 | Verify submission.csv after evaluation |

### 8.5 Spec 03 Dependencies (Inbound)

| Spec 03 Req ID | Used By (this spec) | Purpose |
|----------------|---------------------|---------|
| REQ-SF-005 (extract_code_block) | REQ-P3-013, REQ-P3-016 | Extract code block from A_ensembler response |
| REQ-SF-006 (debug_solution) | REQ-P3-028 | Debug execution errors in ensemble scripts |
| REQ-SF-007 (make_debug_callback) | REQ-P3-021, REQ-P3-022, REQ-P3-028 | Debug callback for evaluate_with_retry |
| REQ-SF-020 (check_and_fix_leakage) | REQ-P3-021, REQ-P3-022, REQ-P3-027 | Leakage detection and correction |
| REQ-SF-022 (leakage integration point) | REQ-P3-027, REQ-P3-045 | Leakage check before every evaluation |

---

## 9. Change Control

### 9.1 Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2026-02-20 | MLE-STAR Team | Initial draft -- all 49 requirements |

### 9.2 Baselining Policy

This SRS is baselined at version 1.0. After baselining, changes require impact analysis against Spec 08 (finalization depends on Phase3Result.best_ensemble), Spec 09 (orchestrator invokes Phase 3), Spec 01 (upstream data model dependencies), Spec 02 (upstream execution harness dependencies), and Spec 03 (upstream safety agent dependencies).
