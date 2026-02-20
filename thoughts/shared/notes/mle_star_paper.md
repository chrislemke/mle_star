# MLE-STAR: Full Paper Extraction

> Complete extraction from arXiv:2506.15692v3 (41 pages: 11 main + 30 appendix). Last updated: 2026-02-20.

---

## 1. Metadata

- **Title:** MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement
- **Authors:** Jaehyun Nam (1,2,*), Jinsung Yoon (1), Jiefeng Chen (1), Jinwoo Shin (2), Sercan O. Arik (1), Tomas Pfister (1)
- **Affiliations:** (1) Google Cloud, (2) KAIST
- **Corresponding authors:** jaehyun.nam@kaist.ac.kr, jinsunyoon@google.com
- **Note:** * Work done while Jaehyun was a student researcher at Google Cloud.
- **arXiv:** 2506.15692v3
- **Submitted:** May 27, 2025 (v1); Last Revised August 28, 2025 (v3)
- **Subject:** cs.LG (Machine Learning)
- **License:** CC BY 4.0
- **Venue:** NeurIPS 2025 (Poster), Submission #14217
- **Paper length:** 41 pages (11 main + 30 appendix)
- **GitHub:** https://github.com/jaehyun513/MLE-STAR

---

## 2. Abstract (Verbatim)

"Agents based on large language models (LLMs) for machine learning engineering (MLE) can automatically implement ML models via code generation. However, existing approaches to build such agents often rely heavily on inherent LLM knowledge and employ coarse exploration strategies that modify the entire code structure at once. This limits their ability to select effective task-specific models and perform deep exploration within specific components, such as experimenting extensively with feature engineering options. To overcome these, we propose MLE-STAR, a novel approach to build MLE agents. MLE-STAR first leverages external knowledge by using a search engine to retrieve effective models from the web, forming an initial solution, then iteratively refines it by exploring various strategies targeting specific ML components. This exploration is guided by ablation studies analyzing the impact of individual code blocks. Furthermore, we introduce a novel ensembling method using an effective strategy suggested by MLE-STAR. Our experimental results show that MLE-STAR achieves medals in 64% of the Kaggle competitions on the MLE-bench Lite, significantly outperforming the best alternative."

---

## 3. Introduction (Section 1)

**Context:** The proliferation of ML has driven high-performance applications across diverse real-world scenarios, from tabular classification to image denoising. Developing such models remains labor-intensive. Recent research focuses on using LLMs as machine learning engineering (MLE) agents that conceptualize ML tasks as code optimization problems and produce executable Python scripts.

**Problems with existing MLE agents:**
1. **Reliance on inherent LLM knowledge:** Agents are biased toward familiar and frequently used methods (e.g., scikit-learn for tabular data), neglecting potentially promising task-specific methods.
2. **Coarse exploration strategy:** Agents modify the entire code structure at once in each iteration, causing premature pivoting between steps (e.g., model selection or hyperparameter tuning) and lacking the ability for deep, iterative exploration within specific pipeline components such as feature engineering.

**Contributions (MLE-STAR = ML Engineering agent with web Search and TArgeted code block Refinement):**
- Uses Google Search to retrieve relevant and potentially state-of-the-art approaches for building a model
- Extracts a specific code block representing a distinct ML pipeline component (e.g., feature engineering or ensemble building) and concentrates on targeted refinement
- Uses ablation studies to identify the code block with the greatest impact on performance
- Introduces a novel ensembling method where the agent itself proposes ensemble strategies, iteratively refined based on performance
- Achieves a substantial gain in medal achievement, improving from 25.8% to 63.6% compared to the top-performing baseline (AIDE)
- Requires minimal human effort (only generalizable initial prompts)

---

## 4. Related Work (Section 2)

### LLM Agents
- General-purpose agents: ReAct (Yao et al., 2023), HuggingGPT (Shen et al., 2023)
- Specialized agents: Voyager (Wang et al., 2023) for Minecraft, AlphaCode (Li et al., 2022) for code generation

### Automated Machine Learning (AutoML)
- Auto-sklearn 2.0 (Feurer et al., 2022), Auto-WEKA (Kotthoff et al., 2017), TPOT (Olson and Moore, 2016), AutoGluon (Erickson et al., 2020)
- Neural architecture search (Elsken et al., 2019; Pham et al., 2018; Real et al., 2019; Zoph and Le, 2017)
- Feature engineering (Fan et al., 2010; Horn et al., 2019; Kanter and Veeramachaneni, 2015; Li et al., 2023; Zhang et al., 2023)
- These methods rely on predefined search spaces requiring domain expertise

### MLE Agents
- **AIDE** (Jiang et al., 2025): Generates candidate solutions in a tree structure. Limitations: heavy reliance on LLM internal knowledge leading to outdated/simple model choices; refinement may prematurely shift focus between pipeline stages
- **DS-Agent** (Guo et al., 2024): Uses case-based reasoning with manually curated cases from Kaggle. Limitations: scalability issues, requires manual labor, overfit to source patterns, restricted applicability to novel task types
- **MLAB** (Huang et al., 2024a), **OpenHands** (Wang et al., 2024): Take general actions by calling tools
- **Data Interpreter** (Hong et al., 2024): Graph-based approach dividing tasks into subtasks
- **DatawiseAgent** (You et al., 2025): Two-stage process with tree-structured plan then exploration
- **Agent Laboratory** (Schmidgall et al., 2025)

---

## 5. Methodology (Section 3: MLE-STAR)

### Problem Setup (Formal Definition)
- **Goal:** Find optimal solution s* = arg max_{s in S} h(s), where S is the space of possible solutions (Python scripts) and h: S -> R is a score function (e.g., validation accuracy)
- **Framework:** Multi-agent framework A which takes datasets D (multiple files) and task description T_task (task types, data modalities, score functions, etc.) as input
- **A consists of n LLM agents (A_1, ..., A_n)**, each with specific functionalities
- **MLE-STAR works across any data modalities** (tabular, image, text, audio) and task types (classification, image-to-image, sequence-to-sequence) -- not restricted to specific inputs or objectives

### 3.1 Generating an Initial Solution Using Web Search as a Tool

#### Candidate Model Search
- Uses web search as a tool to retrieve M effective, state-of-the-art models for the given task
- Formally: {T_model^i, T_code^i}_{i=1}^M = A_retriever(T_task) ... (Equation 1)
  - T_model = description of retrieved model
  - T_code = corresponding example code
- Each retrieved model i is evaluated by a candidate evaluation agent A_init that generates code:
  - s_init^i = A_init(T_task, T_model^i, T_code^i) ... (Equation 2)
- Performance is evaluated using task-specific metric h on dataset D
- Result: set of code scripts S_init = {s_init^1, ..., s_init^M} and their performance scores {h(s_init^1), ..., h(s_init^M)}

#### Merging Candidate Models for Initial Solution
- Scores sorted in descending order via permutation pi
- Initialize s_0 with top-performing script s_(1), h_best = h(s_0)
- Sequentially attempt to incorporate remaining scripts s_(k) for k=2,...,M using A_merger:
  - s_0 <- A_merger(s_0, s_(k)), h_best <- h(s_0) ... (Equation 3)
- A_merger is guided to introduce a simple average ensemble to merge multiple models
- Merging stops when validation score h_best no longer improves (break on first failure)

### 3.2 Refining a Code Block for Solution Improvement

Iterative refinement begins with initial solution s_0, proceeds for T outer loop steps (t = 0, 1, ..., T-1). Two main stages per step:

#### Targeted Code Block Extraction
1. **Ablation study generation:** Agent A_abl generates ablation study code a_t on current solution s_t, receiving summaries of previous ablation studies as input:
   - a_t = A_abl(s_t, {T_abl^i}_{i=0}^{t-1}) ... (Equation 4)
   - The script creates variations by modifying or disabling different pipeline parts

2. **Execution and summarization:** a_t is executed producing output r_t. A summarization module A_summarize processes it:
   - T_abl^t = A_summarize(a_t, r_t) ... (Equation 5)

3. **Code block extraction:** Extractor module A_extractor identifies the critical code block c_t with most significant impact. Previously refined blocks {c_i}_{i=0}^{t-1} are provided to prioritize unrefined blocks:
   - c_t, p_0 = A_extractor(T_abl^t, s_t, {c_i}_{i=0}^{t-1}) ... (Equation 6)
   - Also generates initial plan p_0 for refinement

#### Code Block Refinement (Inner Loop)
- K potential refinement strategies explored for code block c_t
- Agent A_coder implements initial plan p_0: c_t^0 = A_coder(c_t, p_0)
- Candidate solution formed by substitution: s_t^0 = s_t.replace(c_t, c_t^0) ... (Equation 7)
- Performance h(s_t^0) evaluated
- For k = 1, ..., K-1, planning agent A_planner proposes next plan using feedback from previous attempts:
  - p_k = A_planner(c_t, {(p_j, h(s_t^j))}_{j=0}^{k-1}) ... (Equation 8)
- Coding agent generates refined block: c_t^k = A_coder(c_t, p_k)
- Candidate: s_t^k = s_t.replace(c_t, c_t^k), evaluate h(s_t^k)
- Best: k* = arg max_{k in {0,...,K-1}} h(s_t^k)
- Solution s_{t+1} updated to s_t^{k*} only if improvement over s_t

### 3.3 Further Improvement by Exploring Ensemble Strategies

- Novel ensembling procedure (Figure 3)
- Given L distinct solutions {s_l}_{l=1}^L obtained from parallel runs
- Start with initial ensemble plan e_0 (e.g., simple averaging), proposed by MLE-STAR itself
- For R iterations, A_ens_planner proposes ensemble plans using history of previous plans and scores:
  - e_r = A_ens_planner({s_l}_{l=1}^L, {(e_j, h(s_ens^j))}_{j=0}^{r-1})
- Each plan implemented by A_ensembler:
  - s_ens^r = A_ensembler(e_r, {s_l}_{l=1}^L) ... (Equation 9)
- Final output: s_ens* = s_ens^{r*} where r* = arg max_{r in {0,...,R}} h(s_ens^r)

### 3.4 Additional Modules for Robust MLE Agents

#### Debugging Agent (A_debugger)
- If Python script triggers an error (traceback T_bug):
  - s <- A_debugger(s, T_bug) ... (Equation 10)
- Repeated until script executes successfully or max debugging rounds reached
- Falls back to last known executable version if unresolved

#### Data Leakage Checker (A_leakage)
- Analyzes solution script s prior to execution
- Targeted approach: extracts code block c_data where data preprocessing is done
- If leakage detected, generates corrected version c_data*: c_data* = A_leakage(c_data)
- Replaces identified segment: s <- s.replace(c_data, c_data*)
- All generated solutions pass through A_leakage before evaluation

#### Data Usage Checker (A_data)
- Checks initial solution s_0 with task description T_task to ensure all relevant provided data is utilized
- If data not adequately used, A_data revises:
  - s_0 <- A_data(s_0, T_task) ... (Equation 11)

---

## 6. All Agent Types and Their Roles

| Agent | Role | Prompt Figure |
|-------|------|---------------|
| A_retriever | Retrieves M task-specific models via web search | Figure 9 |
| A_init | Generates candidate evaluation code for each retrieved model | Figure 10 |
| A_merger | Merges retrieved model scripts into consolidated initial solution | Figure 11 |
| A_abl | Generates ablation study code to evaluate ML component contributions | Figure 12 |
| A_summarize | Summarizes ablation study raw output into concise results | Figure 13 |
| A_extractor | Identifies most impactful code block + generates initial refinement plan | Figure 14 |
| A_coder | Implements refinement plan on extracted code block | Figure 15 |
| A_planner | Suggests next refinement plan using feedback from previous attempts | Figure 16 |
| A_ens_planner | Proposes ensemble strategies using history of previous ensemble attempts | Figure 17 |
| A_ensembler | Implements ensemble plan on multiple solutions | Figure 18 |
| A_debugger | Fixes execution errors in Python scripts | Figure 19 |
| A_leakage | Detects and corrects data leakage (2-step: extract+check, then correct) | Figures 20, 21 |
| A_data | Checks and ensures all provided data is used | Figure 22 |
| A_test | Generates test submission file from final solution | Figure 25 |

---

## 7. Figures and Descriptions

**Figure 1 (p.2)** -- **Problem setup.** ML Engineering agents process a task description and datasets across various modalities (tabular, text, image, audio, etc.) to produce a solution script. Shows: Machine Learning Tasks (Task Description + Dataset) -> MLE Agent (Planning, Training, Debugging) -> Solution Script.

**Figure 2 (p.4)** -- **Overview of MLE-STAR.** Three panels:
- (a) **Initialization:** Search as tool retrieves task-specific models, generates Python scripts, evaluates them, sorts by validation score, and iteratively integrates top models. Shows scores 0.91, 0.90, 0.85 with first two integrated successfully (0.92) and third discarded (0.89).
- (b) **Target code block extraction:** Performs ablation study to find most impactful component. Example shows: Original 0.93, Median imputing 0.95, Removing feature 0.94. Identifies imputation method impacts most, extracts the corresponding code block (numeric_transformer Pipeline).
- (c) **Code block refinement:** Inner loop suggests plans, implements on target code block, evaluates, updates trajectory and feedback. Outer loop repeats with new target code blocks.

**Figure 3 (p.6)** -- **Ensembling solutions.** MLE-STAR generates solutions in parallel, iteratively proposes ensemble strategies based on previous attempts, implements plans, evaluates, and selects best ensemble. Shows: parallel generation -> A_ens_planner feedback loop -> A_ensembler -> Final Solution.

**Figure 4 (p.10)** -- **Model usage (%) on image classification competitions.** Bar chart comparing Baseline (AIDE) vs MLE-STAR (Ours). Models ordered by release year: VGG, ResNet, U-Net, MobileNet, EfficientNet, ViT. Baseline heavily uses ResNet (~70%), while MLE-STAR uses more diverse and recent models (EfficientNet ~60%, ViT ~20%). Other models (11.7%) used by MLE-STAR are omitted.

**Figure 5 (p.10)** -- **Human intervention.** Shows how a human expert can manually add a model description (e.g., RealMLP) to MLE-STAR, which then integrates its training. Code shows RealMLP_TD_Classifier usage.

**Figure 6 (p.10)** -- **Data leakage checker.** Before: improperly imputed missing values using combined train+test DataFrame. After: separate preprocessing using fit_stats from training set only.

**Figure 7 (p.10)** -- **Data usage checker.** Before: only reads train.csv. After: also processes .xyz geometry files to extract atomic volume features.

**Figure 8 (p.11)** -- **Solution refinement trajectory.** Line chart showing validation improvement (%) across refinement steps 0-4. Values: Step 0: 0%, Step 1: ~12.6%, Step 2: ~17.7%, Step 3: ~20.8%, Step 4: ~22.3%. Shows consistent improvement with most notable gains in early stages.

**Figures 9-28** -- All prompt templates (detailed in Section 14 below).

**Figure 23 (p.34)** -- **Example ablation study output.** Raw LightGBM training output for spaceship-titanic showing: Baseline: 0.8196, Ablation 1 (No StandardScaler): 0.8102, Ablation 2 (No OneHotEncoder): 0.7886, Ablation 3 (No Imputation): 0.8196.

**Figure 24 (p.35)** -- **Summarized ablation study result.** Clean summary: OneHotEncoder has most significant positive impact, followed by StandardScaler, Imputation has no significant impact.

---

## 8. All Tables and Data

### Table 1: Main Results from MLE-bench Lite

Each experiment repeated 3 seeds (except AIDE with o1-preview: 16 seeds, GPT-4o: 36 seeds). All AIDE results from GitHub repository of MLE-bench paper, except Gemini models.

| Model | Made Sub (%) | Valid Sub (%) | Above Median (%) | Bronze (%) | Silver (%) | Gold (%) | Any Medal (%) |
|-------|-------------|--------------|-------------------|-----------|-----------|---------|--------------|
| **MLE-STAR (Ours)** | | | | | | | |
| gemini-2.5-pro | 100.0+-0.0 | 100.0+-0.0 | **83.3**+-4.6 | 6.1+-3.0 | **21.2**+-5.1 | **36.4**+-6.0 | **63.6**+-6.0 |
| gemini-2.0-flash | 95.5+-2.6 | 95.5+-2.6 | 63.6+-6.0 | **9.1**+-3.6 | 4.5+-2.6 | 30.3+-5.7 | 43.9+-6.2 |
| **AIDE (Jiang et al., 2025)** | | | | | | | |
| gemini-2.0-flash | 87.9+-4.0 | 78.8+-5.0 | 39.4+-6.0 | 4.5+-2.6 | 9.1+-3.5 | 12.1+-4.0 | 25.8+-5.4 |
| o1-preview | 99.7+-0.3 | 90.3+-1.6 | 58.2+-2.6 | 4.8+-1.1 | 11.1+-1.7 | 20.7+-2.2 | 36.6+-2.6 |
| gpt-4o | 82.1+-1.4 | 65.7+-1.7 | 29.9+-1.6 | 3.4+-0.6 | 5.8+-0.8 | 9.3+-1.0 | 18.6+-1.4 |
| llama-3.1-405b-instruct | 72.7+-5.5 | 51.5+-6.2 | 18.2+-4.7 | 0.0+-0.0 | 4.5+-2.6 | 6.1+-2.9 | 10.6+-3.8 |
| claude-3-5-sonnet | 81.8+-4.7 | 66.7+-5.8 | 33.3+-5.8 | 3.0+-2.1 | 6.1+-2.9 | 10.6+-3.8 | 19.7+-4.9 |
| **MLAB (Huang et al., 2024a)** | | | | | | | |
| gpt-4o | 84.8+-4.4 | 63.6+-5.9 | 7.6+-3.3 | 3.0+-2.1 | 1.5+-1.5 | 1.5+-1.5 | 6.1+-2.9 |
| **OpenHands (Wang et al., 2024)** | | | | | | | |
| gpt-4o | 81.8+-4.7 | 71.2+-5.6 | 16.7+-4.6 | 3.0+-2.1 | 3.0+-2.1 | 6.1+-2.9 | 12.1+-4.0 |

### Table 2: Comparison with DS-Agent (4 tabular tasks, 5 seeds each)

| Task | Metric | DS-Agent | MLE-STAR |
|------|--------|----------|----------|
| WBY (wild-blueberry-yield) | MAE (down) | 213 | **166** |
| MCC (media-campaign-cost) | RMLSE (down) | 0.2964 | **0.2911** |
| ST (spaceship-titanic) | Accuracy (up) | 0.7982 | **0.8091** |
| ES (enzyme-substrate) | AUROC (up) | 0.8727 | **0.9101** |

### Table 3: Performance with Claude-Sonnet-4 (3 seeds)

| Task | Metric | Gemini-2.0-flash | Sonnet 4 |
|------|--------|-----------------|----------|
| DDD (denoising-dirty-documents) | RMSE (down) | 0.0681 | **0.0155** |
| DBI (dog-breed-identification) | Log Loss (down) | 0.4535 | **0.3114** |
| SAI (spooky-author-identification) | Log Loss (down) | 0.2797 | **0.2610** |
| WCR (the-icml-2013-whale-challenge) | AUROC (up) | **0.9903** | 0.9888 |

### Table 4: Ablation on Ensemble Strategy (MLE-bench Lite, 3 seeds, Gemini-2.0-Flash)

| Ensemble strategy | Made Sub (%) | Valid Sub (%) | Above Median (%) | Bronze (%) | Silver (%) | Gold (%) | Any Medal (%) |
|-------------------|-------------|--------------|-------------------|-----------|-----------|---------|--------------|
| **AIDE** | | | | | | | |
| None | 87.9+-4.0 | 78.8+-5.0 | 39.4+-6.0 | 4.5+-2.6 | 9.1+-3.5 | 12.1+-4.0 | 25.8+-5.4 |
| **MLE-STAR (Ours)** | | | | | | | |
| None | 95.5+-2.6 | 95.5+-2.6 | 57.6+-6.1 | 7.6+-3.3 | 4.5+-2.6 | 25.8+-5.4 | 37.9+-6.0 |
| Best-of-N | 95.5+-2.6 | 95.5+-2.6 | 62.1+-6.0 | 6.1+-3.0 | 7.6+-3.3 | 28.8+-5.6 | 42.4+-6.1 |
| Average ensemble | 95.5+-2.6 | 95.5+-2.6 | 60.6+-6.1 | 6.1+-3.0 | **12.1**+-4.0 | 25.8+-9.4 | 43.9+-6.2 |
| **Ours** | **95.5+-2.6** | **95.5+-2.6** | **63.6**+-6.0 | **9.1**+-3.6 | 4.5+-2.6 | **30.3**+-5.7 | **43.9**+-6.2 |

### Table 5: Improvement Failure Without Data Leakage Checker (spaceship-titanic)

| Metric | Accuracy (up) |
|--------|---------------|
| Validation | 0.8188 -> **0.8677** |
| Test | **0.8033** -> 0.7343 |

(Validation improves but test drops drastically without leakage checker -- the LLM performs feature engineering using target variable "Transported" not accessible in test set.)

### Table 6: Ablation of Data Usage Checker (nomad2018-predicting)

| Model | A_data | RMSLE (down) |
|-------|--------|-------------|
| MLE-STAR | X (no) | 0.0591 |
| MLE-STAR | check (yes) | **0.0559** |

### Table 7: All 22 Competitions in MLE-bench Lite

| Competition ID | Category | Dataset Size (GB) |
|----------------|----------|-------------------|
| aerial-cactus-identification | Image Classification | 0.0254 |
| aptos2019-blindness-detection | Image Classification | 10.22 |
| denoising-dirty-documents | Image To Image | 0.06 |
| detecting-insults-in-social-commentary | Text Classification | 0.002 |
| dog-breed-identification | Image Classification | 0.75 |
| dogs-vs-cats-redux-kernels-edition | Image Classification | 0.85 |
| histopathologic-cancer-detection | Image Regression | 7.76 |
| jigsaw-toxic-comment-classification-challenge | Text Classification | 0.06 |
| leaf-classification | Image Classification | 0.036 |
| mlsp-2013-birds | Audio Classification | 0.5851 |
| new-york-city-taxi-fare-prediction | Tabular | 5.7 |
| nomad2018-predict-transparent-conductors | Tabular | 0.00624 |
| plant-pathology-2020-fgvc7 | Image Classification | 0.8 |
| random-acts-of-pizza | Text Classification | 0.003 |
| ranzcr-clip-catheter-line-classification | Image Classification | 13.13 |
| siim-isic-melanoma-classification | Image Classification | 116.16 |
| spooky-author-identification | Text Classification | 0.0019 |
| tabular-playground-series-dec-2021 | Tabular | 0.7 |
| tabular-playground-series-may-2022 | Tabular | 0.57 |
| text-normalization-challenge-english-language | Seq->Seq | 0.01 |
| text-normalization-challenge-russian-language | Seq->Seq | 0.01 |
| the-icml-2013-whale-challenge-right-whale-redux | Audio Classification | 0.29314 |

### Table 8: Tabular Competitions Used in DS-Agent

| Competition ID | Category | Evaluation Metrics |
|----------------|----------|-------------------|
| media-campaign-cost | Tabular Regression | RMLSE |
| wild-blueberry-yield | Tabular Regression | MAE |
| spaceship-titanic | Tabular Classification | Accuracy |
| enzyme-substrate | Tabular Classification | AUROC |

### Table 9: Additional Comparisons with AutoGluon and DS-Agent (4 tabular tasks)

| Model | media-campaign-cost RMLSE(down) | wild-blueberry-yield MAE(down) | spaceship-titanic Accuracy(up) | enzyme-substrate AUROC(up) |
|-------|------|------|------|------|
| AutoGluon | 0.2707 | 305 | 0.8044 | 0.8683 |
| DS-Agent gpt-3.5 | **0.2702** | 291 | / | 0.5534 |
| DS-Agent gpt-4 | 0.2947 | 267 | 0.7977 | 0.8322 |
| DS-Agent gemini-2.0-flash | 0.2964 | 213 | 0.7982 | 0.8727 |
| **MLE-STAR gemini-2.0-flash** | 0.2911 | **163** | **0.8091** | **0.9101** |

---

## 9. Experimental Setup (Section 4)

### Common Setup
- **Benchmark:** 22 Kaggle competitions from MLE-bench Lite (Chan et al., 2025)
- **Seeds:** 3 random seeds, Gemini-2.0-Flash (unless otherwise specified)
- **Submission agent:** A_test takes task description + final solution as input, outputs code for loading test samples and creating submission file
- **Hyperparameters:**
  - M = 4 retrieved model candidates
  - K = 4 inner loop refinement steps
  - T = 4 outer loop steps
  - L = 2 parallel solutions for ensemble
  - R = 5 ensemble strategy rounds
- **Time limit:** 24 hours maximum per competition (following MLE-bench setup)
- **Primary baseline:** AIDE (Jiang et al., 2025) -- state-of-the-art on MLE-bench
- **Hardware (Appendix F):** 96 vCPUs with 360 GB Memory (Intel Xeon CPU), 8 NVIDIA V100 GPUs with 16 GB Memory

### Computation Time (Appendix F)
- MLE-STAR requires **14.1 hours** on average to generate a single final solution (across 22 tasks, 3 seeds = 66 experiments)
- AIDE requires **15.4 hours**
- MLE-STAR does NOT require more time than the best alternative

---

## 10. Main Results (Section 4.1)

### Quantitative Results
- MLE-STAR with Gemini-2.0-Flash improves AIDE's any medal rate from **25.8% to 43.9%** (18+ percentage point improvement)
- Above median rate improved from **39.4% to 63.6%**
- MLE-STAR with Gemini-2.0-Flash substantially outperforms AIDE using o1-preview in gold medals (10% more)
- With Gemini-2.5-Pro: medal achievement rate of over **60%** (63.6%)
- With Gemini-2.5-Pro: **36.4% gold medals**, **21.2% silver**, **6.1% bronze**

### Comparison to DS-Agent
- On four tabular tasks used in DS-Agent's development, MLE-STAR significantly outperforms DS-Agent even without human effort (Table 2)
- Notably better on wild-blueberry-yield (MAE: 213 vs 166) and enzyme-substrate (AUROC: 0.8727 vs 0.9101)

### Performance with Claude-Sonnet-4 (Table 3)
- MLE-STAR works with non-Gemini models too
- Strong results on DDD (RMSE 0.0155), DBI (Log Loss 0.3114), SAI (Log Loss 0.2610)

---

## 11. Ablation Studies (Section 4.2)

### Reasoning Models
- Gemini-2.5-Pro yields better performance than Gemini-2.0-Flash
- Example: denoising-dirty-documents -- Gemini-2.0-Flash scored above median in all 3 seeds but no medals; Gemini-2.5-Pro achieved 2 gold and 1 silver medal
- MLE-STAR designed to harness rapidly improving reasoning-based LLMs

### Claude-Sonnet-4 Compatibility
- Tested on 4 different competition types: image-to-image (DDD), image classification (DBI), text classification (SAI), audio classification (WCR)
- Results confirm framework is compatible and generalizable across LLM types

### Ensemble Strategy Effectiveness (Table 4)
- Without ensemble: MLE-STAR achieves 37.9% medals (already 12% higher than AIDE's 25.8%)
- Best-of-N: 42.4% medals
- Average ensemble: 43.9% medals
- **Proposed ensemble method (Ours): 43.9% medals** but with significantly more **gold medals (30.3%)** vs average ensemble (25.8%)
- MLE-STAR's ensemble consistently surpasses median human expert performance

### Data Leakage Checker (Table 5)
- Without leakage checker on spaceship-titanic: validation improved (0.8188 -> 0.8677) but test accuracy dropped drastically (0.8033 -> 0.7343)
- LLM performed feature engineering using target variable "Transported" which is not accessible in test set

### Data Usage Checker (Table 6)
- On nomad2018-predicting: without A_data, RMSLE = 0.0591; with A_data, RMSLE = 0.0559
- Gemini-2.0-Flash only loaded train.csv, neglecting geometry.xyz files

### Progressive Improvement (Figure 8)
- Average relative error reduction across all 22 challenges:
  - Step 0: 0%
  - Step 1: ~12.6%
  - Step 2: ~17.7%
  - Step 3: ~20.8%
  - Step 4: ~22.3%
- Most notable improvement in early refinement stages (ablation study targets most influential code blocks first)

---

## 12. Discussion (Section 5)

### Model Selection Observations
- AIDE primarily employs ResNet (2015, outdated) for image classification
- MLE-STAR uses more recent models: EfficientNet (2019), ViT (2021)
- MLE-STAR wins 37% of image classification medals vs AIDE's 26%

### Human Intervention
- Natural extension: humans can manually add model descriptions {T_model, T_code}
- Example: adding RealMLP (Holzmuller et al., 2024), a model not previously retrieved
- Users can also specify target code blocks by replacing ablation summary with manual instructions

### LLM Misbehaviors and Corrections
- Data leakage: LLM-generated code preprocesses test data using its own statistics (Figure 6)
- Data underuse: LLMs overlook some provided data sources (Figure 7)
- Both addressed by dedicated checker agents

---

## 13. Algorithms (Appendix B)

### Algorithm 1: Generating an Initial Solution
```
Input: task description T_task, datasets D, score function h, number of retrieved models M
1. {T_model^i, T_code^i}_{i=1}^M = A_retriever(T_task)
2. for i = 1 to M do
3.     s_init^i = A_init(T_task, T_model^i, T_code^i)
4.     Evaluate h(s_init^i) using D
5. end for
6. s_0 <- s_init^{pi(1)}    [best performing]
7. h_best <- h(s_0)
8. for i = 2 to M do
9.     s_candidate <- A_merger(s_0, s_init^{pi(i)})
10.    Evaluate h(s_candidate) using D
11.    if h(s_candidate) >= h_best then
12.        s_0 <- s_candidate
13.        h_best <- h(s_0)
14.    else
15.        break
16.    end if
17. end for
18. Output: initial solution s_0
```

### Algorithm 2: Refining Solution
```
Input: initial solution s_0, outer loop steps T, inner loop steps K
1. s_final <- s_0
2. h_best <- h(s_0)
3. T_abl, C = {}, {}
4. for t = 0 to T-1 do
5.     a_t = A_abl(s_t, T_abl)
6.     r_t = exec(a_t)
7.     T_abl^t = A_summarize(a_t, r_t)
8.     c_t, p_0 = A_extractor(T_abl^t, s_t, C)
9.     c_t^0 = A_coder(c_t, p_0)
10.    s_t^0 = s_t.replace(c_t, c_t^0)
11.    Evaluate h(s_t^0) using D
12.    if h(s_t^0) >= h_best then
13.        s_final <- s_t^0
14.        h_best <- h(s_t^0)
15.    end if
16.    for k = 1 to K-1 do
17.        p_k = A_planner(c_t, {p_j, h(s_t^j)}_{j=0}^{k-1})
18.        c_t^k = A_coder(c_t, p_k)
19.        s_t^k = s_t.replace(c_t, c_t^k)
20.        Evaluate h(s_t^k) using D
21.        if h(s_t^k) >= h_best then
22.            s_final <- s_t^k
23.            h_best <- h(s_t^k)
24.        end if
25.    end for
26.    T_abl <- T_abl + T_abl^t
27.    C <- C + c_t
28. end for
29. Output: final solution s_final
```

### Algorithm 3: Ensembling Final Solutions
```
Input: candidate final solutions s_final^1, ..., s_final^L, ensemble loop steps R
1. e_0 = A_ens_planner({s_final^l}_{l=1}^L)
2. s_ens^0 = A_ensembler(e_0, {s_final^l}_{l=1}^L)
3. Evaluate h(s_ens^0) using D
4. for r = 1 to R-1 do
5.     e_r = A_ens_planner({s_final^l}_{l=1}^L, {(e_j, h(s_ens^j))}_{j=0}^{r-1})
6.     s_ens^r = A_ensembler(e_r, {s_final^l}_{l=1}^L)
7.     Evaluate h(s_ens^r) using D
8. end for
9. s_ens* = s_ens^{r*} where r* = arg max_{r in {0,...,R-1}} h(s_ens^r)
10. Output: s_ens*
```

---

## 14. All Prompt Templates (Appendix A)

### Figure 9 -- Retriever Agent Prompt
```
# Competition
{task description}

# Your task
- List {M} recent effective models and their example codes to win the above competition.

# Requirement
- The example code should be concise and simple.
- You must provide an example code, i.e., do not just mention GitHubs or papers.

Use this JSON schema:
Model = {'model_name': str, 'example_code': str}
Return: list[Model]
```

### Figure 10 -- Candidate Evaluation Agent Prompt
```
# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide a task description and a model description.
- You need to implement your Python solution using the provided model.

# Task description
{task description}

# Model description
## Model name
{model description}

## Example Python code
{example code}

# Your task
- Implement the solution in Python.
- You must use the model as described in the model description.
- This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.
- Propose an evaluation metric that is reasonable for this task.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.
- Do not include other models that are not directly related to the model described.
- Use PyTorch rather than TensorFlow. Use CUDA if you need. All the necessary libraries are installed.
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.
- Only use the provided train data in the `./input` directory. Do not load test data.
- If there are more than 30,000 training samples, you must subsample to 30,000 for a faster run.

# Required
- There should be no additional headings or text in your response.
- Print out or return a final performance metric in a clear format with the exact words: 'Final Validation Performance: {final_validation_score}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior.
```

### Figure 11 -- Merging Agent Prompt
```
# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide a base solution and an additional reference solution.
- You need to implement your Python solution by integrating reference solution to the base solution.

# Base solution
{base code}

# Reference solution
{reference code}

# Your task
- Implement the solution in Python.
- You have to integrate the reference solution to the base solution.
- Your code base should be the base solution.
- Try to train additional model of the reference solution.
- When integrating, try to keep code with similar functionality in the same place (e.g., all preprocessing should be done and then all training).
- When integrating, ensemble the models.
- The solution design should be relatively simple.
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.
- Only use the provided train data in the `./input` directory.
- If there are more than 30,000 training samples, you must subsample to 30,000 for a faster run.

# Required
- There should be no additional headings or text in your response.
- Print out or return a final performance metric in a clear format with the exact words: 'Final Validation Performance: {final_validation_score}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior
```

### Figure 12 -- Ablation Study Agent Prompt
```
# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to perform an ablation study on the current Python solution to know which parts of the code contribute the most to the overall performance.
- We will now provide a current Python solution.
- We will also provide the summaries of previous ablation studies.

# Python solution
{solution script}

## Previous ablation study result {0}
{previous_ablations[0]}
...
## Previous ablation study result {t-1}
{previous_ablations[t-1]}

# Instructions
- You need to generate a simple Python code that performs an ablation study on the train.py script.
- The generated code should create variations by modifying or disabling parts (2-3 parts) of the training process.
- Your ablation study should concentrate on the other parts that have not been previously considered.
- For each ablation, print out how the modification affects the model's performance.

# Response format
- There should be no additional headings or text in your response.
- The Python code for the ablation study should not load test data. It should only focus on training and evaluating the model on the validation set.
- The code should include a printing statement that shows the performance of each ablation.
- The code should consequently print out what part of the code contributes the most to the overall performance.
```

### Figure 13 -- Ablation Study Summarization Prompt
```
# Your code for ablation study was:
{code for ablation study}

# Ablation study results after running the above code:
{raw result}

# Your task
- Summarize the result of ablation study based on the code and printed output.
```

### Figure 14 -- Extractor Prompt
```
# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to extract a code block from the current Python solution and improve the extracted block for better performance.
- Your suggestion should be based on the ablation study results of the current Python solution.
- We will now provide the current Python solution and the ablation study results.
- We also provide code blocks which you have tried to improve previously.

# Python solution
{solution script}

# Ablation study results
{summary of ablation study}

## Code block {0}
{prev_code_blocks[0]}
...
## Code block {t-1}
{prev_code_blocks[t-1]}

# Your task
- Given the ablation study results, suggest an effective next plan to improve the above Python script.
- The plan should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences).
- Please avoid plan which can make the solution's running time too long (e.g., searching hyperparameters in a very large search space).
- Try to improve the other part which was not considered before.
- Also extract the code block from the above Python script that need to be improved according to the proposed plan. You should try to extract the code block which was not improved before.

# Response format
- Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences) and a single markdown code block which is the code block that need to be improved.
- The code block can be long but should be exactly extracted from the Python script provided above.

Use this JSON schema:
Refine_Plan = {'code_block': str, 'plan': str}
Return: list[Refine_Plan]
```

### Figure 15 -- Coder Prompt
```
# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need refine the code block for better performance based on the improvement plan.
- We will now provide the code block and the improvement plan.

# Code block
{code_block}

# Improvement plan
{plan}

# Your task
- Implement the improvement plan on the above code block. But do not remove subsampling if exists.
- The code block should be improved according to the proposed plan.
- Note that all the variable including actual data is defined earlier (since you are just seeing a code block), therefore do not introduce dummy variables.

# Response format
- Your response should be a single markdown code block (wrapped in ```) which is the improved code block.
- There should be no additional headings or text in your response.
```

### Figure 16 -- Planner Prompt
```
# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you have to improve the code block for better performance.
- We will provide the code block you are improving and the improvement plans you have tried.

# Code block
{code block}

# Improvement plans you have tried

## Plan: {plans[0]}
## Score: {scores[0]}
...
## Plan: {plans[k-1]}
## Score: {scores[k-1]}

# Your task
- Suggest a better plan to improve the above code block.
- The suggested plan must be novel and effective.
- Please avoid plans which can make the solution's running time too long (e.g., searching hyperparameters in a very large search space).
- The suggested plan should be differ from the previous plans you have tried and should receive a higher score.

# Response format
- Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences).
- There should be no additional headings or text in your response.
```

### Figure 17 -- Ensemble Strategy Planner Prompt
```
# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you have to ensemble {L} Python Solutions for better performance.
- We will provide the Python Solutions and the ensemble plans you have tried.

# 1st Python Solution
{solution1}
...
# {L}th Python Solution
{solutionL}

# Ensemble plans you have tried

## Plan: {plans[0]}
## Score: {scores[0]}
...
## Plan: {plans[r-1]}
## Score: {scores[r-1]}

# Your task
- Suggest a better plan to ensemble the {L} solutions. You should concentrate how to merge, not the other parts like hyperparameters.
- The suggested plan must be easy to implement, novel, and effective.
- The suggested plan should be differ from the previous plans you have tried and should receive a higher (or lower) score.

# Response format
- Your response should be an outline/sketch of your proposed solution in natural language.
- There should be no additional headings or text in your response.
- Plan should not modify the original solutions too much since execution error can occur.
```

### Figure 18 -- Ensembler Prompt
```
# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to ensemble {L} Python Solutions for better performance based on the ensemble plan.
- We will now provide the Python Solutions and the ensemble plan.

# 1st Python Solution
{solution1}
...
# {L}th Python Solution
{solutionL}

# Ensemble Plan
{plan}

# Your task
- Implement the ensemble plan with the provided solutions.
- Unless mentioned in the ensemble plan, do not modify the original Python Solutions too much.
- All the provided data (except previous submissions; do not load submissions) is already prepared and available in the `.\input` directory. There is no need to unzip any files.
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.

# Response format required
- Your response should be a single markdown code block (wrapped in ```) which is the ensemble of {L} Python Solutions.
- There should be no additional headings or text in your response.
- Do not subsample or introduce dummy variables. You have to provide full new Python Solution using the {L} provided solutions.
- Do not forget the `./final/submission.csv` file.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {final_validation_score}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
```

### Figure 19 -- Debugging Agent Prompt
```
# Code with an error:
{code}

# Error:
{bug}

# Your task
- Please revise the code to fix the error.
- Do not remove subsampling if exists.
- Provide the improved, self-contained Python script again.
- There should be no additional headings or text in your response.
- All the provided input data is stored in "./input" directory.
- Remember to print a line in the code with 'Final Validation Performance: {final_validation_score}' so we can parse performance.
- The code should be a single-file python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the refined Python code.
```

### Figure 20 -- Data Leakage Detection Prompt
```
# Python code
{code}

# Your task
- Extract the code block where the validation and test samples are preprocessed using training samples.
- Check that the model is trained with only training samples.
- Check that before printing the final validation score, the model is not trained the validation samples.
- Also check whether the validation and test samples are preprocessed correctly, preventing information from the validation or test samples from influencing the training process (i.e., preventing data leakage).

# Requirement
- Extract a code block and also check the data leakage.
- The code block should be an exact subset of the above Python code.
- Your response for a code block should be a single markdown code block.
- If data leakage is present on validation and test samples, answer 'Yes Data Leakage'.
- If data leakage is not present on validation and test samples, answer 'No Data Leakage'.

Use this JSON schema:
Answer = {'leakage_status': str, 'code_block': str}
Return: list[Answer]
```

### Figure 21 -- Data Leakage Correction Prompt
```
# Python code
{code}

# Your task
- In the above Python code, the validation and test samples are influencing the training process, i.e., not correctly preprocessed.
- Ensure that the model is trained with only training samples.
- Ensure that before printing the final validation score, the model is not trained on the validation samples.
- Refine the code to prevent such data leakage problem.

# Requirement
- Your response should be a single markdown code block.
- Note that all the variables are defined earlier. Just modify it with the above code.
```

### Figure 22 -- Data Usage Checker Prompt
```
I have provided Python code for a machine learning task (attached below):

# Solution Code
{initial solution}

Does above solution code uses all the information provided for training? Here is task description and some guide to handle:

# Task description
{task description}

# Your task
- If the above solution code does not use the information provided, try to incorporate all. Do not bypass using try-except.
- DO NOT USE TRY AND EXCEPT; just occur error so we can debug it!
- See the task description carefully, to know how to extract unused information effectively.
- When improving the solution code by incorporating unused information, DO NOT FORGET to print out 'Final Validation Performance: {final_validation_score}' as in original solution code.

# Response format:
Option 1: If the code did not use all the provided information, your response should be a single markdown code block (wrapped in ```) which is the improved code block. There should be no additional headings or text in your response
Option 2: If the code used all the provided information, simply state that "All the provided information is used.
```

### Figure 25 -- Test Submission Agent Prompt
```
# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to come up with an excellent solution in Python.
- We will now provide a task description and a Python solution.
- What you have to do on the solution is just loading test samples and create a submission file.

# Task description
{task description}

# Python solution
{final solution}

# Your task
- Load the test samples and create a submission file.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.
- Test data is available in the `./input` directory.
- Save the test predictions in a `submission.csv` file. Put the `submission.csv` into `./final` directory.
- You should not drop any test samples. Predict the target value for all test samples.
- This is a very easy task because the only thing to do is to load test samples and then replace the validation samples with the test samples. Then you can even use the full training set!

# Required
- Do not modify the given Python solution code too much. Try to integrate test submission with minimal changes.
- There should be no additional headings or text in your response.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not forget the ./final/submission.csv file.
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior.
```

### Figure 26 -- Subsampling Extraction Prompt
```
# Introduction
- From the give Python solution, you need to extract a code block where subsampling of training samples is used. We will now provide the current Python solution.

# Current Python solution
{final solution}

# Your task
- Extract a code block where subsampling of training samples is used.

# Response format
- Your response should be a single markdown code block (wrapped in ```) which is the code block.
- The code block should be exactly extracted from the Python script provided above.
```

### Figure 27 -- Subsampling Removal Prompt
```
# Introduction
- From the give Python code block, remove the subsampling and make it to use full training samples. We will now provide the current Python code block.

# Current Python code block
{code block with subsampling}

# Your task
- Remove the subsampling and make it to use full training samples.
- Note that all the variable including actual data is defined earlier (since you are just seeing a code block), therefore do not introduce dummy variables.

# Response format
- Your response should be a single markdown code block (wrapped in ```) which is the code block.
```

### Figure 28 -- Data Contamination Check Prompt
```
Your task is to check whether the python solution is similar to the reference discussion.
Now we will give you reference discussion and our python solution.

# Reference discussion
{reference discussion}

# Python solution
{final solution}

# Your task
- Check whether the python solution just copy and pastes the reference discussion.
- If it is sufficiently novel and different, please answer 'Novel'.
- Otherwise, if you think it is too similar, please answer 'Same'.
- Your answer should be only one of 'Novel' or 'Same'.
```

---

## 15. Qualitative Examples (Appendix C)

### C.1 Generated Ablation Study Code
Example generated by A_abl available in supplementary material (example_outputs/ablation.py).

### C.2 Raw Ablation Study Output (Figure 23)
Spaceship-titanic example showing LightGBM training logs:
- Baseline Validation Performance: 0.8196
- Ablation 1 (No StandardScaler): 0.8102
- Ablation 2 (No OneHotEncoder): 0.7886
- Ablation 3 (No Imputation): 0.8196
- Final Validation Performance: 0.8196

### C.3 Summary of Ablation Study (Figure 24)
Clean summary produced by A_summarize:
- "The ablation study investigated the impact of three preprocessing steps on the performance of a LightGBM classifier: StandardScaler, OneHotEncoder, and Imputation."
- Baseline: 0.8196
- No StandardScaler: 0.8102 (small drop)
- No OneHotEncoder: 0.7886 (significant drop -- most important)
- No Imputation: 0.8196 (no change -- least important)
- Conclusion: OneHotEncoder has the most significant positive impact, followed by StandardScaler

### Example Refinement Plans (pp. 22-23, spaceship-titanic feature engineering):
1. "Since feature engineering had the biggest impact, I will focus on improving the cabin feature extraction. Instead of simply splitting the Cabin string, I will create dummy variables for each unique Deck and Side. Also, the Cabin_num will be kept as numerical, imputing missing values using a median strategy..."
2. "Instead of one-hot encoding 'Deck' and 'Side' directly, I will explore interaction features between 'Deck', 'Side', and potentially 'Cabin_num'. Specifically, I'll create combined features like 'Deck_Side' and 'Deck_Cabin_num'... Furthermore, I will impute missing 'Cabin_num' values using a more sophisticated method like k-NN imputation..."
3. "I propose a plan that focuses on a more nuanced approach to 'Cabin_num' and interaction terms. First, I'll bin 'Cabin_num' into ordinal categories (e.g., low, medium, high) based on quantiles... Then, I'll create interaction features between the binned 'Cabin_num', 'Deck', and 'Side' using one-hot encoding..."

### Example Ensemble Plans (pp. 25-26, spaceship-titanic):
1. "Averaging the predicted probabilities from both models is a straightforward and effective ensembling technique..."
2. "Stacking with a simple meta-learner: Use Logistic Regression as the meta-learner, trained on AutoGluon_Prob and LGBM_Prob as meta-features..."
3. "Weighted averaging with optimized weights determined by a simple grid search on a validation set: iterate through weights from 0.0 to 1.0 in increments of 0.1..."

---

## 16. Conclusion (Section 6)

"We propose MLE-STAR, a novel MLE agent designed for various ML tasks. Our key idea is to utilize a search engine to retrieve effective models and then explore various strategies targeting specific ML pipeline components to improve the solution. The effectiveness of MLE-STAR is validated by winning medals in 64% (where 36% are gold medals) of the MLE-bench Lite Kaggle competitions."

---

## 17. Limitations

"Since Kaggle competitions are publicly accessible, there is a potential risk that LLMs might have been trained with the relevant discussions about the challenge. Nevertheless, we show that MLE-STAR's solution is sufficiently novel (using LLM as a judge) compared to the discussions on Kaggle (see Appendix H)."

**Data contamination analysis (Appendix H):** Collected 25 discussions from 7 competitions (75 discussion-solution pairs). Using Gemini-2.5-Pro as judge with the prompt in Figure 28, all final solutions generated by MLE-STAR with Gemini-2.0-Flash were judged to be sufficiently novel compared to top Kaggle discussions.

---

## 18. Broader Impacts (Appendix I)

- MLE-STAR could lower the barrier to entry for ML, fostering innovation across sectors
- As state-of-the-art models are updated and improved, MLE-STAR's solutions are expected to automatically improve because the framework leverages search to retrieve effective models from the web
- This inherent adaptability ensures MLE-STAR continues to provide increasingly better solutions as ML advances

---

## 19. Additional Related Work on Data Science Agents (Appendix J)

- Infiagent-dabench (Hu et al., 2024)
- Dacode (Huang et al., 2024b)
- DSBench (Jing et al., 2025)
- Data Interpreter (Hong et al., 2024): graph-based approach
- DatawiseAgent (You et al., 2025): two-stage process
- These methods prioritize overall task completion rates rather than performance on specific engineering challenges

---

## 20. References (Complete List -- 50 Papers)

1. Brown et al., 2020 - Language models are few-shot learners. NeurIPS.
2. Chan et al., 2025 - MLE-bench: Evaluating machine learning agents on machine learning engineering. ICLR.
3. Chen and Guestrin, 2016 - XGBoost: A scalable tree boosting system. KDD.
4. Dosovitskiy et al., 2021 - ViT: An image is worth 16x16 words. ICLR.
5. Elsken et al., 2019 - Neural architecture search: A survey. JMLR.
6. Erickson et al., 2020 - AutoGluon-Tabular. arXiv:2003.06505.
7. Fan et al., 2019 - Brief review of image denoising techniques. Visual computing.
8. Fan et al., 2010 - Generalized and heuristic-free feature construction. SIAM.
9. Feurer et al., 2022 - Auto-sklearn 2.0. JMLR.
10. Guo et al., 2024 - DS-agent: Automated data science by empowering LLMs with case-based reasoning. ICML.
11. He et al., 2016 - Deep residual learning (ResNet). CVPR.
12. Hollmann et al., 2023 - CAAFE for context-aware automated feature engineering. NeurIPS.
13. Hollmann et al., 2025 - Accurate predictions on small data with a tabular foundation model. Nature.
14. Holzmuller et al., 2024 - Better by default: Strong pre-tuned trees on tabular data (RealMLP). NeurIPS.
15. Hong et al., 2024 - Data interpreter: An LLM agent for data science. arXiv:2402.18679.
16. Horn et al., 2019 - The autofeat python library. ECML PKDD.
17. Hu et al., 2024 - Infiagent-dabench. arXiv:2401.05507.
18. Huang et al., 2024a - MLAgentBench: Evaluating language agents on ML experimentation. ICML.
19. Huang et al., 2024b - Dacode: Agent data science code generation benchmark. arXiv:2410.07331.
20. Ichihara et al., 2025 - Evaluation of best-of-n sampling strategies for language model alignment. TMLR.
21. Jain et al., 2025 - Livecodebench. ICLR.
22. Jiang et al., 2025 - AIDE: AI-driven exploration in the space of code. arXiv:2502.13138.
23. Jimenez et al., 2024 - SWE-bench. ICLR.
24. Jin et al., 2019 - Auto-Keras. KDD.
25. Jing et al., 2025 - DSBench. ICLR.
26. Kanter and Veeramachaneni, 2015 - Deep feature synthesis. IEEE DSAA.
27. Kolodner, 1992 - An introduction to case-based reasoning. AI Review.
28. Kotthoff et al., 2017 - Auto-WEKA 2.0. JMLR.
29. LeDell and Poirier, 2020 - H2O AutoML. ICML Workshop on AutoML.
30. Li et al., 2023 - A data-driven policy network for pre-training automated feature engineering. ICLR.
31. Li et al., 2022 - Competition-level code generation with AlphaCode. Science.
32. Li et al., 2024 - AutoKaggle: A multi-agent framework. arXiv:2410.20424.
33. Nam et al., 2024 - Optimized feature generation for tabular data via LLMs with decision tree reasoning. NeurIPS.
34. Olson and Moore, 2016 - TPOT. ICML Workshop on AutoML.
35. Pedregosa et al., 2011 - Scikit-learn. JMLR.
36. Pham et al., 2018 - Efficient neural architecture search via parameter sharing. ICML.
37. Prokhorenkova et al., 2018 - CatBoost. NeurIPS.
38. Real et al., 2019 - Regularized evolution for image classifier architecture search. AAAI.
39. Schmidgall et al., 2025 - Agent Laboratory. arXiv:2501.04227.
40. Shen et al., 2023 - HuggingGPT. NeurIPS.
41. Tan and Le, 2019 - EfficientNet. ICML.
42. Team et al., 2024 - Gemini 1.5. arXiv:2403.05530.
43. Touvron et al., 2023 - LLaMA. arXiv:2302.13971.
44. Wang et al., 2023 - Voyager. arXiv:2305.16291.
45. Wang et al., 2024 - OpenHands. ICLR.
46. Watson and Marir, 1994 - Case-based reasoning: A review. Knowledge Engineering Review.
47. Yao et al., 2023 - ReAct: Synergizing reasoning and acting. ICLR.
48. You et al., 2025 - DatawiseAgent. arXiv:2503.07044.
49. Zhang et al., 2023 - OpenFE: Automated feature generation with expert-level performance. ICML.
50. Zoph and Le, 2017 - Neural architecture search with reinforcement learning. ICLR.
