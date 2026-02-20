# MLE-STAR: System Architecture Summary

> Condensed architectural reference for implementation. Last updated: 2026-02-20.

---

## 1. Three-Phase Pipeline Overview

```
Phase 1: INITIAL SOLUTION GENERATION
  A_retriever -> A_init -> A_merger -> A_data -> A_leakage
  (web search)  (eval)   (merge)    (check)   (check)

Phase 2: NESTED LOOP REFINEMENT (T=4 outer x K=4 inner)
  Outer: A_abl -> A_summarize -> A_extractor -> [Inner Loop]
  Inner: A_planner -> A_coder -> evaluate -> feedback
  Safety: A_debugger (on error), A_leakage (always)

Phase 3: ENSEMBLE OPTIMIZATION (L=2 solutions, R=5 rounds)
  A_ens_planner -> A_ensembler -> evaluate -> feedback

Final: A_test -> submission generation (+ subsampling removal via Figs 26-27)
```

---

## 2. Agent Interaction Flow

### Phase 1: Initialization
```
Task Description (T_task) + Datasets (D)
    |
    v
A_retriever(T_task) --> {T_model^i, T_code^i} for i=1..M
    |
    v
A_init(T_task, T_model^i, T_code^i) --> s_init^i (for each i)
    |
    v
Sort by h(s_init^i) descending
    |
    v
A_merger(s_0, s_(k)) --> merged solution (sequential, break on first failure)
    |
    v
A_data(s_0, T_task) --> ensure all data sources used
    |
    v
A_leakage(s_0) --> check/correct data leakage
    |
    v
Initial Solution s_0
```

### Phase 2: Refinement (Nested Loops)
```
For t = 0 to T-1 (outer loop -- code block selection):
    |
    A_abl(s_t, previous_summaries) --> ablation code a_t
    |
    exec(a_t) --> raw results r_t
    |
    A_summarize(a_t, r_t) --> summary T_abl^t
    |
    A_extractor(T_abl^t, s_t, prev_blocks) --> (c_t, p_0)
    |
    For k = 0 to K-1 (inner loop -- strategy exploration):
        |
        A_coder(c_t, p_k) --> c_t^k
        |
        s_t^k = s_t.replace(c_t, c_t^k)
        |
        A_debugger(s_t^k, traceback) --> fix if error
        |
        A_leakage(s_t^k) --> check leakage
        |
        evaluate h(s_t^k)
        |
        A_planner(c_t, history) --> p_{k+1}  (for next iteration)
    |
    s_{t+1} = best of {s_t^0, ..., s_t^{K-1}} if improvement
```

### Phase 3: Ensemble
```
L parallel solutions (each from Phase 1 + Phase 2)
    |
    v
For r = 0 to R-1:
    |
    A_ens_planner(solutions, history) --> ensemble plan e_r
    |
    A_ensembler(e_r, solutions) --> s_ens^r
    |
    evaluate h(s_ens^r) --> feedback
    |
    v
Final: s_ens* = best ensemble
```

---

## 3. Hyperparameter Defaults

| Parameter | Value | Description |
|-----------|-------|-------------|
| **M** | 4 | Number of candidate models retrieved from web search |
| **T** | 4 | Outer loop iterations (different code blocks targeted) |
| **K** | 4 | Inner loop iterations (refinement strategies per code block) |
| **L** | 2 | Number of parallel solutions maintained for ensembling |
| **R** | 5 | Ensemble strategy exploration iterations |
| **Time limit** | 24 hours | Maximum runtime per competition |
| **Subsample** | 30,000 | Max training samples during refinement (removed for final) |
| **Ablation parts** | 2-3 | Components tested per ablation study |

### Total Agent Calls per Competition (Approximate)

- **Phase 1:** 1 retriever + M init + (M-1) merger + 1 data + 1 leakage = ~10 calls
- **Phase 2:** T * (1 abl + 1 summarize + 1 extractor + K * (1 coder + 1 planner)) = T*(3 + 2K) = 4*(3+8) = ~44 calls per solution
- **Phase 3:** R * (1 ens_planner + 1 ensembler) = 10 calls
- **Plus:** Debugging calls as needed, leakage checks after each solution
- **Total:** ~64+ agent LLM calls per solution path, ~128+ for L=2 parallel paths + ensemble

---

## 4. Safety Modules

### Debugging Agent (A_debugger)
- **Trigger:** Python execution error (traceback)
- **Scope:** All phases
- **Behavior:** Iterative fix attempts; falls back to last working version
- **Input:** Code + error traceback
- **Output:** Fixed code

### Data Leakage Checker (A_leakage)
- **Trigger:** Every generated solution before evaluation
- **Scope:** All phases
- **Behavior:** Two-step process:
  1. Extract preprocessing code block + detect leakage (Figure 20)
  2. If leakage found, correct it (Figure 21)
- **Evidence:** Without checker on spaceship-titanic: validation +5.0% but test -8.9%
- **Common issue:** LLM preprocesses test data using combined train+test statistics

### Data Usage Checker (A_data)
- **Trigger:** After initial solution generation
- **Scope:** Phase 1 only
- **Behavior:** Compares solution against task description; adds unused data sources
- **Evidence:** Recovered performance on nomad2018 by adding .xyz geometry files
- **Common issue:** LLM only reads train.csv, ignoring auxiliary data files

---

## 5. Benchmark Results Summary

### Headline Numbers (MLE-bench Lite, 22 Kaggle Competitions)

| Configuration | Any Medal (%) | Gold (%) |
|---------------|---------------|----------|
| AIDE (Gemini-2.0-Flash) | 25.8 | 12.1 |
| AIDE (o1-preview) | 36.6 | 20.7 |
| MLE-STAR (Gemini-2.0-Flash) | **43.9** | **30.3** |
| MLE-STAR (Gemini-2.5-Pro) | **63.6** | **36.4** |

### Key Performance Insights

1. **Web search impact:** MLE-STAR uses EfficientNet/ViT instead of ResNet; wins 37% vs 26% on image tasks
2. **Targeted refinement value:** Even without ensemble, MLE-STAR (37.9%) exceeds AIDE (25.8%)
3. **Ensemble contribution:** Adds ~6% medal rate and ~4.5% gold medals over no-ensemble
4. **Cross-model compatibility:** Works with Gemini, Claude-Sonnet-4, and other LLMs
5. **Computation time:** 14.1 hours average (vs AIDE's 15.4 hours -- not slower)
6. **Progressive improvement:** ~22.3% average error reduction over 4 refinement steps

### Hardware (Appendix F)
- 96 vCPUs with 360 GB Memory (Intel Xeon CPU)
- 8 NVIDIA V100 GPUs with 16 GB Memory

---

## 6. Implementation Reference

### Official Implementation
- **Framework:** Google Agent Development Kit (ADK)
- **Repository:** https://github.com/google/adk-samples/tree/main/python/agents/machine-learning-engineering
- **Language:** Python 3.12+
- **Package Management:** Poetry (with uv.lock)
- **Default LLM:** Gemini-2.0-Flash-001 (configurable via ROOT_AGENT_MODEL)

### Unofficial Reimplementation
- **Repository:** https://github.com/WalkingDevFlag/MLE-STAR-Open
- **Goal:** Google-free using OpenAI-compatible APIs
- **Search:** DuckDuckGo (no API key required)

---

## 7. Key Design Decisions

1. **Why web search?** LLMs have training data cutoff; web search retrieves current SOTA models
2. **Why ablation-guided targeting?** More efficient than modifying entire code; identifies highest-impact components first
3. **Why agent-proposed ensembles?** Goes beyond simple averaging; agents can propose stacking, weighted averaging, meta-learners
4. **Why separate checker agents?** LLMs systematically make two types of errors: data leakage and data underuse
5. **Why subsampling during refinement?** Faster iteration (30K sample cap); removed for final submission using Figures 26-27 prompts
6. **Why break-on-first-failure merging?** Diminishing returns from adding weaker models; prevents quality degradation
