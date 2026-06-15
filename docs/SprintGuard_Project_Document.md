# SprintGuard: Neuro-Fuzzy System for Agile Bug Triage and Sprint Risk Forecasting

> **Project Type:** New Generation Soft Computing / Applied ML  
> **Duration:** 4–6 Weeks (MVP)  
> **Stack:** Python · FastAPI · PyTorch · Streamlit · skfuzzy · DEAP

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [System Architecture](#3-system-architecture)
4. [Module Descriptions](#4-module-descriptions)
5. [Workflow Walkthrough](#5-workflow-walkthrough)
6. [Mathematical Foundations](#6-mathematical-foundations)
7. [Research Questions](#7-research-questions)
8. [Dataset Sources](#8-dataset-sources)
9. [Tech Stack](#9-tech-stack)
10. [MVP Roadmap (4–6 Weeks)](#10-mvp-roadmap-46-weeks)
11. [Project Report Title Ideas](#11-project-report-title-ideas)
12. [API Design Overview](#12-api-design-overview)
13. [Directory Structure](#13-directory-structure)
14. [References](#14-references)

---

## 1. Problem Statement

In Agile software development, sprint planning assumes a relatively stable workload. In practice, bugs reported mid-sprint disrupt velocity, cause developer context switching, and frequently spill over into the next sprint.

**Key Statistics:**
- ~60% of bugs reported mid-sprint either miss the current sprint deadline or get assigned to the wrong developer.
- Context switching between bug fixes and planned sprint tasks reduces developer productivity by an estimated 20–40%.
- Manual triage by Scrum Masters or leads introduces latency (avg. 2–6 hours before assignment).

**Root Causes:**
| Cause | Impact |
|---|---|
| Manual bug reading + assignment | Slow, inconsistent |
| No effort prediction at triage time | Sprint capacity unknown |
| No real-time sprint risk signal | PM reacts too late |
| Re-planning done manually | Time-consuming, error-prone |

---

## 2. Solution Overview

**SprintGuard** is a real-time, hybrid intelligent system that closes the loop between bug reporting and sprint management.

```
Bug Reported → NLP Embedding → Dev Assignment + Effort Estimate
                                          ↓
                            Sprint Risk Engine (Fuzzy)
                                          ↓
                           Auto Re-planner (Genetic Algorithm)
                                          ↓
                        PM Notification → One-click Accept
```

**Two Core Capabilities:**

1. **Auto Triage** — When a bug arrives, SprintGuard reads its description, associated file paths, and developer history to automatically assign the best developer and estimate fix time.

2. **Sprint Impact Prediction** — Using the effort estimate + current sprint state (velocity, remaining capacity, days left), SprintGuard predicts sprint risk and suggests scope adjustments.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        SprintGuard                          │
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │  Bug Input   │   │  Jira/GitHub │   │  Dev History   │  │
│  │  (webhook)   │   │  Sprint Data │   │  DB            │  │
│  └──────┬───────┘   └──────┬───────┘   └───────┬────────┘  │
│         └──────────────────┼───────────────────┘           │
│                            ▼                                │
│              ┌─────────────────────────┐                    │
│              │   1. Bug NLP Engine     │  ← BERT Embedding  │
│              │   (DistilBERT)          │                    │
│              └──────────┬──────────────┘                    │
│                         │  768-dim vector                   │
│              ┌──────────┴──────────────┐                    │
│              │                         │                    │
│    ┌─────────▼──────────┐  ┌──────────▼──────────┐         │
│    │ 2. Dev Assignment  │  │ 3. Effort Estimator  │         │
│    │    NN (MLP)        │  │    (LSTM / ANFIS)    │         │
│    └─────────┬──────────┘  └──────────┬──────────┘         │
│              │                         │                    │
│              └──────────┬──────────────┘                    │
│                         ▼                                   │
│              ┌─────────────────────────┐                    │
│              │  4. Sprint Risk Engine  │  ← Fuzzy Logic     │
│              │  (Fuzzy Inference Sys.) │                    │
│              └──────────┬──────────────┘                    │
│                         ▼                                   │
│              ┌─────────────────────────┐                    │
│              │  5. Auto Re-planner     │  ← Genetic Algo    │
│              │  (DEAP / GA)            │                    │
│              └──────────┬──────────────┘                    │
│                         ▼                                   │
│              ┌─────────────────────────┐                    │
│              │  Streamlit Dashboard    │                    │
│              │  + Slack / Jira API     │                    │
│              └─────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Module Descriptions

### Module 1: Bug NLP Engine

| Property | Detail |
|---|---|
| **Model** | `sentence-transformers/all-MiniLM-L6-v2` or DistilBERT |
| **Input** | Bug title + description + stack trace snippet |
| **Output** | 768-dimensional dense embedding vector |
| **Purpose** | Semantic understanding — "login fails on Safari" ≈ "auth error iOS WebKit" |

**Why BERT over TF-IDF:**
TF-IDF treats "null pointer exception in checkout" and "NPE during payment flow" as dissimilar. BERT's contextual embeddings capture semantic equivalence, improving triage accuracy significantly.

**Preprocessing Pipeline:**
```
Raw Bug Text → Lowercase + Strip HTML → Tokenize (WordPiece)
→ BERT Encoder → Mean Pool [CLS] token → L2 Normalize → 768-dim vector
```

---

### Module 2: Dev Assignment Neural Network

| Property | Detail |
|---|---|
| **Model** | MLP Classifier (3–4 hidden layers) |
| **Input** | Bug embedding (768) + file paths changed (encoded) + dev history features |
| **Output** | Probability distribution over all active developers |
| **Pick** | `argmax(P(dev_i))` = assigned developer |

**Input Feature Vector:**

```
[bug_embedding (768-dim)]
+ [file_module_one_hot (N modules)]
+ [dev_past_bugs_fixed_in_module (scalar per dev)]
+ [dev_current_sprint_load (hours remaining)]
+ [dev_avg_fix_time_similar_bugs (hours)]
= Total input dim ≈ 768 + N + 3D  (D = number of devs)
```

**Training:**
- Dataset: Eclipse Bugzilla / Mozilla — labeled (bug → assigned dev)
- Loss: Cross-entropy
- Metric: Top-1 Accuracy, MRR (Mean Reciprocal Rank)
- Target: ≥ 70% Top-3 Accuracy on held-out set

---

### Module 3: Effort Estimator (LSTM / ANFIS)

| Property | Detail |
|---|---|
| **Model** | LSTM for sequential patterns OR ANFIS (Adaptive Neuro-Fuzzy) |
| **Input** | Bug embedding + code churn (lines changed) + historical fix times for similar bugs |
| **Output** | Estimated fix hours ± confidence interval |
| **Example** | "3.2 hours ± 1.1h" |

**ANFIS Advantage:**
ANFIS combines fuzzy rules (handling uncertainty like "this is a vague bug description") with neural learning. It outperforms plain NN when training data is limited (< 1000 samples) — useful for small team histories.

**LSTM Advantage:**
Captures temporal patterns — if a module has seen increasing bug complexity over 3 sprints, fix times trend upward. LSTM encodes this sequence.

**Metric:** MMRE (Mean Magnitude of Relative Error), PRED(25)

```
MMRE = (1/n) * Σ |actual - predicted| / actual
PRED(25) = % of estimates within 25% of actual
```

---

### Module 4: Sprint Risk Engine (Fuzzy Inference System)

| Property | Detail |
|---|---|
| **Library** | `scikit-fuzzy (skfuzzy)` |
| **Type** | Mamdani FIS |
| **Inputs** | New bug hours, team velocity trend, sprint days remaining |
| **Output** | Sprint Risk Score → Low / Medium / High |

**Fuzzy Variables:**

```
INPUT 1: bug_hours_added
  - Low:    [0, 2, 4]
  - Medium: [3, 6, 9]
  - High:   [8, 15, 20+]

INPUT 2: velocity_trend
  - Declining: [-10, -5, 0]
  - Stable:    [-2, 0, 2]
  - Increasing:[0, 5, 10]

INPUT 3: days_remaining
  - Critical: [0, 1, 3]
  - Tight:    [2, 4, 6]
  - Comfortable: [5, 8, 14]

OUTPUT: sprint_risk
  - Low:    [0, 0.2, 0.4]
  - Medium: [0.3, 0.5, 0.7]
  - High:   [0.6, 0.8, 1.0]
```

**Sample Fuzzy Rules:**
```
IF bug_hours_added IS High AND days_remaining IS Critical THEN sprint_risk IS High
IF bug_hours_added IS Low AND velocity_trend IS Increasing THEN sprint_risk IS Low
IF bug_hours_added IS Medium AND days_remaining IS Tight THEN sprint_risk IS Medium
```

---

### Module 5: Auto Re-planner (Genetic Algorithm)

| Property | Detail |
|---|---|
| **Library** | `DEAP` (Distributed Evolutionary Algorithms in Python) |
| **Trigger** | Activated when Sprint Risk = High |
| **Input** | List of remaining sprint stories (story points, priority, assignee) |
| **Output** | Optimal subset of stories to keep / move to backlog |

**GA Setup:**

```
Chromosome: Binary string [1,0,1,1,0...] where 1=keep, 0=move to backlog

Fitness Function:
  maximize: sum(priority * story_points for kept stories)
  subject to: sum(effort_hours for kept stories) ≤ available_capacity
              must_have stories always = 1 (not removable)

Operators:
  Selection:  Tournament selection (k=3)
  Crossover:  Two-point crossover (p=0.8)
  Mutation:   Bit-flip (p=0.05 per gene)
  Generations: 100–200
  Population: 50–100 chromosomes
```

---

## 5. Workflow Walkthrough

**Scenario: Day 5 of a 2-week sprint**

```
Step 1: Tester logs JIRA-452
        Title: "Checkout crashes if coupon expired"
        Description: "Getting 500 error on /checkout POST when applying expired promo code"
        File: backend/payment/coupon_validator.py

Step 2: SprintGuard webhook fires (< 2 seconds)
        → Bug NLP Engine: embeds description
        → Module match: payment + backend = coupon_validator module
        → Dev Assignment NN: Alice (82%), Bob (11%), Charlie (7%)
        → Effort Estimator: 4.0h ± 0.8h (based on 3 similar past bugs)

Step 3: Sprint Risk Engine activates
        → bug_hours_added = 4.0 (Medium-High)
        → Alice current load = 6h remaining tasks
        → days_remaining = 3 (Critical)
        → Alice effective remaining = 6 + 4 = 10h in 3 days (only ~7.5h available)
        → Sprint Risk: HIGH (0.78)

Step 4: GA Re-planner runs
        → Sprint backlog: JIRA-440 (2pt, low priority), JIRA-445 (3pt, medium)
        → Suggestion A: Move JIRA-440 to backlog → risk drops to Medium
        → Suggestion B: Alice 2h overtime on Day 6 → risk drops to Medium
        → GA picks Suggestion A (no overtime cost)

Step 5: Slack alert fires
        "@PM: Sprint at 74% risk (HIGH). Suggested: Move JIRA-440 (2pt) to backlog.
        Alternative: Alice overtime 2h on Day 6. [Accept Suggestion A] [Accept B] [Dismiss]"

Step 6: PM clicks "Accept Suggestion A"
        → Jira API: JIRA-440 moved to backlog
        → JIRA-452 assigned to Alice, estimate 4h
        → Sprint board updated automatically
        → Risk drops to Medium (0.48)
```

---

## 6. Mathematical Foundations

### 6.1 BERT Embedding (Sentence Similarity)

Cosine similarity between two bug embeddings:

```
similarity(b1, b2) = (v1 · v2) / (||v1|| * ||v2||)
```

Where `v1, v2 ∈ ℝ^768` are L2-normalized embeddings from the transformer encoder.

### 6.2 MLP Dev Assignment (Softmax Output)

```
P(dev_k | bug) = exp(z_k) / Σ_j exp(z_j)

where z = W_L * ReLU(W_{L-1} * ... ReLU(W_1 * x + b_1) ...) + b_L
```

Loss: `L = -Σ y_k * log(P(dev_k | bug))` (Cross-entropy)

### 6.3 ANFIS Effort Estimation

ANFIS Rule i (Takagi-Sugeno form):
```
IF x1 is A_i AND x2 is B_i THEN f_i = p_i*x1 + q_i*x2 + r_i
```

Output:
```
y = Σ (w_i * f_i) / Σ w_i

where w_i = μ_A_i(x1) * μ_B_i(x2)   (firing strength)
```

### 6.4 Fuzzy Membership (Triangular)

```
μ_triangle(x; a, b, c) = max(min((x-a)/(b-a), (c-x)/(c-b)), 0)
```

### 6.5 GA Fitness Function

```
maximize F(chromosome) = Σ_{i: gene_i=1} (priority_i * story_points_i)

subject to:
  Σ_{i: gene_i=1} effort_hours_i  ≤  available_capacity
  gene_i = 1  ∀ i ∈ must_have_stories
```

---

## 7. Research Questions

Pick 1–2 for your project report / thesis:

### Q1: BERT vs TF-IDF+SVM for Bug Triage
**Question:** Does BERT-based bug triage reduce mean assignment time compared to TF-IDF + SVM?  
**Metrics:** MRR (Mean Reciprocal Rank), Top-1 Accuracy, Top-3 Accuracy  
**Dataset:** Eclipse Bugzilla, Mozilla  
**Hypothesis:** BERT's semantic understanding will improve Top-1 accuracy by ≥ 15% over TF-IDF+SVM on unseen bug descriptions involving paraphrased or abbreviated text.

### Q2: Sprint Spillover Prediction (LSTM)
**Question:** Can LSTM predict sprint spillover 3 days early using bug inflow rate and fix rate?  
**Metrics:** Precision, Recall, F1-score of binary "Sprint will fail" prediction  
**Dataset:** Apache Jira public archives  
**Hypothesis:** LSTM trained on 3-sprint rolling windows will achieve > 75% precision on spillover prediction with 3-day lead time.

### Q3: ANFIS vs Plain NN for Effort Estimation
**Question:** Does hybrid ANFIS outperform plain MLP for bug fix effort estimation on small datasets?  
**Metrics:** MMRE, PRED(25)  
**Dataset:** 500 historical bugs with known fix times  
**Hypothesis:** ANFIS will achieve lower MMRE than MLP when training data < 300 samples due to its built-in uncertainty handling.

### Q4: Developer Context-Switch Cost Reduction
**Question:** What is the measurable impact of auto triage on developer context-switch cost?  
**Metrics:** #context switches per developer per week, sprint velocity comparison  
**Method:** A/B comparison — manual triage sprints vs. SprintGuard-assisted sprints  
**Hypothesis:** Auto triage will reduce per-developer context switches by ≥ 30% in 2-week sprints.

---

## 8. Dataset Sources

| Dataset | Source | Use |
|---|---|---|
| Eclipse Bugzilla | `https://bugs.eclipse.org` | Bug descriptions + developer assignments |
| Mozilla Bugzilla | `https://bugzilla.mozilla.org` | Bug triage + fix time data |
| VSCode GitHub Issues | `https://github.com/microsoft/vscode/issues` | Modern bug language patterns |
| Apache JIRA Archives | `https://issues.apache.org/jira` | Sprint data, velocity, bug flow |
| Taiga Public Boards | `https://taiga.io` | Sprint planning data |

**Preprocessing Steps:**
1. Filter: only bugs with status `FIXED` and assigned developer
2. Remove: duplicates, security-embargoed bugs, empty descriptions
3. Label: `(bug_id, description, assigned_dev, fix_hours, module)`
4. Split: 70% train / 15% val / 15% test (stratified by developer)

---

## 9. Tech Stack

### Core ML
```
torch==2.2.0
transformers==4.40.0
sentence-transformers==2.7.0
scikit-fuzzy==0.4.2
deap==1.4.1
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
```

### Backend
```
fastapi==0.111.0
uvicorn==0.29.0
pydantic==2.7.1
httpx==0.27.0          # Jira / GitHub API calls
python-dotenv==1.0.1
```

### Frontend / Visualization
```
streamlit==1.35.0
plotly==5.22.0
altair==5.3.0
```

### Infrastructure
```
Docker + docker-compose
PostgreSQL (dev history + sprint data)
Redis (real-time sprint state cache)
```

---

## 10. MVP Roadmap (4–6 Weeks)

### Week 1–2: BERT Triage Core
- [ ] Set up FastAPI project skeleton
- [ ] Download + preprocess Eclipse Bugzilla dataset (10k bugs)
- [ ] Fine-tune `all-MiniLM-L6-v2` on bug descriptions
- [ ] Build MLP dev assignment classifier
- [ ] REST endpoint: `POST /triage` → `{ assigned_dev, confidence, top3 }`
- [ ] Unit tests + baseline metrics (TF-IDF+SVM comparison)
- **Target:** ≥ 70% Top-3 Accuracy

### Week 3: Effort Estimator
- [ ] Build LSTM effort estimator on historical fix time data
- [ ] Implement ANFIS as alternative (compare MMRE)
- [ ] REST endpoint: `POST /estimate` → `{ hours, confidence_interval }`
- [ ] Integrate with triage endpoint (single `POST /analyze-bug` response)

### Week 4: Sprint Risk Engine
- [ ] Define fuzzy variables + membership functions (skfuzzy)
- [ ] Implement Mamdani FIS rules (12–15 rules)
- [ ] Sprint state API: pull current velocity + dev loads
- [ ] REST endpoint: `GET /sprint/risk` → `{ risk_level, score, contributing_factors }`
- [ ] Basic Streamlit dashboard: burndown chart + risk gauge

### Week 5–6: GA Re-planner + Demo Polish
- [ ] Implement DEAP-based GA for story selection
- [ ] REST endpoint: `POST /replan` → `{ suggestions, projected_risk_after }`
- [ ] Streamlit demo: live bug input → full pipeline output
- [ ] Slack webhook integration (optional)
- [ ] Final evaluation: all metrics, comparison tables
- [ ] Project report / presentation

---

## 11. Project Report Title Ideas

1. **"SprintGuard: Neuro-Fuzzy System for Agile Bug Triage and Sprint Risk Forecasting"**
   *(Best for academic submission — covers all soft computing aspects)*

2. **"Reducing Sprint Spillover using Hybrid Deep Learning for Automated Bug Assignment"**
   *(Best if focusing on BERT + LSTM contribution)*

3. **"Integrating LLMs and Genetic Algorithms for Real-time Sprint Health Management"**
   *(Best if emphasizing the optimization + LLM angle)*

4. **"End-to-End Agile Intelligence: From Bug Inflow to Sprint Re-planning via Neuro-Fuzzy Hybrid Models"**
   *(Most comprehensive, suitable for conference paper)*

---

## 12. API Design Overview

### `POST /api/v1/bugs/analyze`
Accepts a new bug report and returns full triage output.

**Request:**
```json
{
  "title": "Checkout crashes if coupon expired",
  "description": "Getting 500 error on /checkout POST when applying expired promo code",
  "file_paths": ["backend/payment/coupon_validator.py"],
  "reporter": "qa_tester_1",
  "sprint_id": "SPRINT-42"
}
```

**Response:**
```json
{
  "bug_id": "JIRA-452",
  "assigned_dev": "alice",
  "assignment_confidence": 0.82,
  "top3_devs": [
    {"dev": "alice", "probability": 0.82},
    {"dev": "bob",   "probability": 0.11},
    {"dev": "charlie", "probability": 0.07}
  ],
  "effort_estimate": {
    "hours": 4.0,
    "confidence_interval": [3.2, 4.8]
  },
  "sprint_impact": {
    "risk_before": "MEDIUM",
    "risk_after":  "HIGH",
    "risk_score":  0.78,
    "reason": "Alice has 6h remaining; adding 4h in 3 days exceeds capacity"
  }
}
```

---

### `GET /api/v1/sprint/{sprint_id}/risk`
Returns current sprint risk assessment.

**Response:**
```json
{
  "sprint_id": "SPRINT-42",
  "risk_level": "HIGH",
  "risk_score": 0.78,
  "days_remaining": 3,
  "velocity_trend": "declining",
  "bug_hours_added_today": 7.5,
  "factors": [
    "3 bugs added in last 24h",
    "Alice at 133% capacity",
    "Velocity down 12% from last sprint"
  ]
}
```

---

### `POST /api/v1/sprint/{sprint_id}/replan`
Triggers GA re-planner and returns suggestions.

**Response:**
```json
{
  "sprint_id": "SPRINT-42",
  "current_risk": "HIGH",
  "suggestions": [
    {
      "id": "A",
      "action": "Move JIRA-440 to backlog",
      "story_points_removed": 2,
      "projected_risk": "MEDIUM",
      "projected_risk_score": 0.48
    },
    {
      "id": "B",
      "action": "Alice overtime 2h on Day 6",
      "projected_risk": "MEDIUM",
      "projected_risk_score": 0.44
    }
  ],
  "recommended": "A"
}
```

---

## 13. Directory Structure

```
sprintguard/
├── backend/
│   └── app/
│       ├── main.py
│       ├── models/
│       │   ├── bug_embedding.py       # BERT / MiniLM wrapper
│       │   ├── dev_assignment.py      # MLP classifier
│       │   ├── effort_estimator.py    # LSTM / ANFIS
│       │   ├── sprint_risk.py         # Fuzzy Inference System
│       │   └── replanner.py           # Genetic Algorithm (DEAP)
│       ├── routes/
│       │   ├── bugs.py                # POST /bugs/analyze
│       │   ├── sprint.py              # GET /sprint/risk, POST /sprint/replan
│       │   └── repo.py
│       └── services/
│           ├── repo_service.py
│           ├── jira_service.py        # Jira API integration
│           └── sprint_service.py     # Sprint state management
├── frontend/
│   └── dashboard.py                   # Streamlit app
├── storage/
│   ├── datasets/                      # Raw + preprocessed bug datasets
│   └── models/                        # Saved model weights (.pt files)
├── docs/
│   └── SprintGuard_Project_Document.md
├── dockers/
│   ├── Dockerfile
│   └── docker-compose.yml
└── requirements.txt
```

---

## 14. References

1. Anvik, J., Hiew, L., & Murphy, G. C. (2006). *Who should fix this bug?* ICSE 2006.
2. Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL 2019.
3. Jang, J. S. R. (1993). *ANFIS: Adaptive-network-based fuzzy inference system.* IEEE Trans. Systems, Man, Cybernetics.
4. Zimmermann, T., et al. (2010). *An empirical study of bug severity in open-source projects.* MSR 2010.
5. Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory.* Neural Computation.
6. Fortin, F. A., et al. (2012). *DEAP: Evolutionary algorithms made easy.* JMLR.
7. Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python.* JMLR.
8. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019.

---

*Document generated for SprintGuard — Agile Sprint Health + Auto Bug Triage System*  
*Version 1.0 | June 2026*
