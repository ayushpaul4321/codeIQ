# SprintGuard — Execution Plan

> **Project Duration:** 6 Weeks  
> **Team Size:** 3–4 Members  
> **Methodology:** Agile (2-week mini-sprints)  
> **Goal:** Working MVP demo + project report by Week 6

---

## Phase Overview

```
Week 1       Week 2       Week 3       Week 4       Week 5       Week 6
│────────────│────────────│────────────│────────────│────────────│
│  SETUP +   │  TRIAGE    │  EFFORT    │  RISK +    │  GA +      │  POLISH +
│  DATA PREP │  ENGINE    │  ESTIMATOR │  DASHBOARD │  INTEGRAT. │  SUBMIT
│────────────│────────────│────────────│────────────│────────────│
  Sprint 1 (Weeks 1–2)      Sprint 2 (Weeks 3–4)      Sprint 3 (Weeks 5–6)
```

---

## Sprint 1 — Foundation + Triage Engine (Weeks 1–2)

### Goals
- Environment set up and reproducible
- Dataset cleaned and ready
- BERT triage model trained and evaluated
- `/api/v1/bugs/analyze` endpoint returning dev assignment

---

### Week 1 — Environment, Data, Baseline

| Day | Task | Owner Role | Output |
|-----|------|-----------|--------|
| Mon | Repo setup: GitHub, branches, folder structure | Lead Dev | `main` + `dev` branches live |
| Mon | Docker environment: Python 3.11, FastAPI, Postgres | DevOps | `docker-compose up` works |
| Mon | Download Eclipse Bugzilla dataset (50k bugs) | Data Engineer | Raw CSV in `storage/datasets/raw/` |
| Tue | Data exploration: field distribution, missing values | Data Engineer | EDA notebook |
| Tue | Filter: keep FIXED bugs with assigned dev + description | Data Engineer | `bugs_filtered.csv` (~30k rows) |
| Wed | Preprocessing pipeline: HTML strip, tokenize, deduplicate | Data Engineer | `bugs_clean.csv` |
| Wed | Label encoding: dev names → integer IDs | Data Engineer | `dev_label_map.json` |
| Wed | Train/val/test split (70/15/15, stratified by dev) | Data Engineer | 3 split CSV files |
| Thu | TF-IDF + SVM baseline (comparison benchmark) | ML Engineer | Baseline accuracy logged |
| Thu | FastAPI skeleton: `main.py`, router structure, health check | Lead Dev | `GET /health` returns 200 |
| Fri | Database schema: `bugs`, `developers`, `sprints`, `assignments` | Lead Dev | Migrations run cleanly |
| Fri | Week 1 review: blockers, metrics check, adjust Week 2 plan | All | Review notes |

**Week 1 Exit Criteria:**
- [ ] Dataset cleaned: ≥ 25,000 usable labeled bugs
- [ ] TF-IDF+SVM baseline: Top-1 accuracy logged (expected ~45–55%)
- [ ] FastAPI app starts without errors
- [ ] Docker environment runs on all team machines

---

### Week 2 — BERT Model + Triage Endpoint

| Day | Task | Owner Role | Output |
|-----|------|-----------|--------|
| Mon | Load `all-MiniLM-L6-v2` via sentence-transformers | ML Engineer | Embeddings generated for 1k samples |
| Mon | Batch embed full training set (GPU if available) | ML Engineer | `embeddings_train.npy` |
| Tue | Build MLP classifier (3 layers, dropout 0.3, ReLU) | ML Engineer | Model architecture defined |
| Tue | Training loop: cross-entropy, Adam optimizer, LR scheduler | ML Engineer | Training loss curves |
| Wed | Evaluate: Top-1, Top-3 accuracy, MRR on val set | ML Engineer | Metrics in `results/triage_metrics.json` |
| Wed | Tune: adjust hidden dims, dropout, learning rate | ML Engineer | Best checkpoint saved |
| Thu | Build `bug_embedding.py` service wrapper | Lead Dev | Clean inference API |
| Thu | Build `dev_assignment.py` service wrapper | Lead Dev | Returns top-3 devs + probabilities |
| Thu | Wire up `POST /api/v1/bugs/analyze` endpoint | Lead Dev | Endpoint returns valid JSON |
| Fri | Integration test: send 20 real bugs, check output | All | Pass/fail log |
| Fri | Sprint 1 review + demo: live triage on 5 sample bugs | All | Demo recording |

**Week 2 Exit Criteria:**
- [ ] Top-3 Accuracy ≥ 70% on test set
- [ ] MRR ≥ 0.55
- [ ] BERT beats TF-IDF+SVM by ≥ 10 percentage points
- [ ] `/bugs/analyze` returns assignment + confidence in < 3 seconds

---

## Sprint 2 — Effort Estimation + Sprint Risk (Weeks 3–4)

### Goals
- LSTM or ANFIS effort estimator integrated
- Fuzzy sprint risk engine live
- Basic Streamlit dashboard showing burndown + risk

---

### Week 3 — Effort Estimator

| Day | Task | Owner Role | Output |
|-----|------|-----------|--------|
| Mon | Prepare effort dataset: bug → fix_hours labels | Data Engineer | `effort_dataset.csv` (500–2000 rows) |
| Mon | Feature engineering: bug embedding + code churn + module | Data Engineer | `effort_features.npy` |
| Tue | Build LSTM model: 2 layers, hidden=128, dropout=0.2 | ML Engineer | LSTM architecture |
| Tue | Train LSTM: MSE loss, Adam, 100 epochs | ML Engineer | Training loss curve |
| Wed | Build ANFIS model (alternative): 5 rules, 2 inputs | ML Engineer | ANFIS architecture via `anfis` lib |
| Wed | Train ANFIS on same dataset | ML Engineer | Comparison ready |
| Thu | Evaluate both: MMRE, PRED(25) on test set | ML Engineer | `results/effort_metrics.json` |
| Thu | Pick better model OR ensemble (weighted average) | ML Engineer | Final estimator saved |
| Thu | Build `effort_estimator.py` service wrapper | Lead Dev | Returns hours + CI |
| Fri | Extend `/bugs/analyze` to include effort estimate | Lead Dev | Updated endpoint response |
| Fri | End-to-end test: bug in → assignment + effort out | All | Integration test passes |

**Week 3 Exit Criteria:**
- [ ] MMRE < 0.40 on test set
- [ ] PRED(25) ≥ 50%
- [ ] Effort estimate returned in `/bugs/analyze` response
- [ ] ANFIS vs LSTM comparison documented

---

### Week 4 — Sprint Risk Engine + Dashboard

| Day | Task | Owner Role | Output |
|-----|------|-----------|--------|
| Mon | Define fuzzy variables + membership functions (skfuzzy) | ML Engineer | `sprint_risk.py` skeleton |
| Mon | Implement 15 Mamdani fuzzy rules | ML Engineer | Rules coded and tested |
| Tue | Unit test FIS: test 9 boundary combinations (Low/Med/High × Low/Med/High) | ML Engineer | All 9 scenarios produce expected output |
| Tue | Build `GET /api/v1/sprint/{id}/risk` endpoint | Lead Dev | Returns risk_level + score + factors |
| Wed | Sprint state service: pull dev loads, velocity from DB | Lead Dev | `sprint_service.py` |
| Wed | Connect triage output → sprint risk update (pipeline) | Lead Dev | End-to-end flow: bug → risk score updates |
| Thu | Streamlit dashboard page 1: Sprint burndown chart | Frontend Dev | Burndown renders with live data |
| Thu | Streamlit dashboard page 2: Risk gauge + risk history | Frontend Dev | Risk gauge shows Low/Med/High |
| Fri | Streamlit page 3: Triage input form → shows assignment + risk | Frontend Dev | Full demo flow in UI |
| Fri | Sprint 2 review + demo: drop a bug in UI, see risk change | All | Demo recording |

**Week 4 Exit Criteria:**
- [ ] FIS returns correct risk level for all boundary test cases
- [ ] `/sprint/risk` endpoint live and connected to real sprint state
- [ ] Streamlit dashboard shows burndown + risk gauge + triage form
- [ ] End-to-end: bug input in UI → dev assigned + risk updates on screen

---

## Sprint 3 — GA Re-planner + Integration + Polish (Weeks 5–6)

### Goals
- GA re-planner producing valid story suggestions
- Full pipeline integrated (bug → triage → effort → risk → replan)
- Final demo ready, report written

---

### Week 5 — GA Re-planner + Full Integration

| Day | Task | Owner Role | Output |
|-----|------|-----------|--------|
| Mon | Define chromosome encoding for sprint stories | ML Engineer | Binary vector schema |
| Mon | Implement fitness function with capacity constraint | ML Engineer | GA fitness function |
| Tue | DEAP setup: population=50, tournament selection, 2-pt crossover | ML Engineer | `replanner.py` skeleton |
| Tue | Run GA on sample sprint: 10 stories, find optimal subset | ML Engineer | GA converges in < 200 generations |
| Wed | Build `POST /api/v1/sprint/{id}/replan` endpoint | Lead Dev | Returns suggestions A/B + projected risk |
| Wed | Connect: if risk=HIGH → auto-trigger replan → show suggestions | Lead Dev | Full pipeline end-to-end |
| Thu | Slack webhook integration: POST suggestion to channel | Lead Dev | Slack message fires on HIGH risk |
| Thu | "Accept suggestion" button in Streamlit → calls Jira mock API | Frontend Dev | UI accept flow works |
| Fri | Full pipeline stress test: 10 bugs in sequence, check stability | All | No crashes, risk updates correctly |
| Fri | Performance profiling: < 2s end-to-end triage response | Lead Dev | Timing logged |

**Week 5 Exit Criteria:**
- [ ] GA finds valid story subset in < 5 seconds for sprints up to 30 stories
- [ ] Full pipeline (bug → replan) works end-to-end in < 5 seconds
- [ ] Slack alert fires on HIGH risk
- [ ] "Accept suggestion" updates mock sprint board

---

### Week 6 — Polish, Evaluation, Report, Demo

| Day | Task | Owner Role | Output |
|-----|------|-----------|--------|
| Mon | Final model evaluation: all metrics on held-out test set | ML Engineer | `results/final_metrics.json` |
| Mon | Comparison tables: BERT vs TF-IDF, ANFIS vs LSTM, etc. | ML Engineer | Tables for report |
| Tue | Write report: Sections 1–4 (Intro, Literature, Methodology, System Design) | All | Draft 50% complete |
| Tue | Dashboard UI polish: consistent styling, tooltips, labels | Frontend Dev | UI looks presentable |
| Wed | Write report: Sections 5–7 (Implementation, Results, Conclusion) | All | Full draft complete |
| Wed | Record demo video: 3-minute walkthrough of full pipeline | All | `demo_video.mp4` |
| Thu | Peer review report draft, fix gaps | All | Revised draft |
| Thu | Prepare presentation slides (10–12 slides) | All | Slides ready |
| Fri | Final submission: code freeze, zip, upload | Lead Dev | Submitted ✓ |
| Fri | Live demo + presentation | All | — |

**Week 6 Exit Criteria:**
- [ ] All research questions answered with metric evidence
- [ ] Report complete (≥ 3000 words + figures + tables)
- [ ] Demo video recorded
- [ ] Code pushed to final release tag on GitHub

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| GPU not available for BERT training | Medium | High | Use Google Colab Pro / Kaggle free GPU; or use pre-computed embeddings |
| Eclipse dataset access slow / API rate limit | Medium | Medium | Download full dump upfront on Day 1, cache locally |
| ANFIS library compatibility issues | High | Medium | Fallback: use LSTM only; document ANFIS as "future work" |
| Jira API credentials unavailable | Medium | Low | Use mock Jira service (`json` file) for demo |
| Team member unavailable mid-project | Low | High | Daily standups; each module has a primary + secondary owner |
| BERT model too slow for < 2s target | Low | Medium | Switch to `all-MiniLM-L6-v2` (6x faster than full BERT) |
| GA not converging for large sprints | Low | Medium | Cap sprint size at 30 stories; tune population size |

---

## Definition of Done (Per Feature)

A feature is "done" when:
- [ ] Code written and committed to feature branch
- [ ] Unit tests pass (or manual test cases documented)
- [ ] API endpoint returns correct response schema
- [ ] Metrics logged to `results/` directory
- [ ] PR reviewed by at least one other team member
- [ ] Merged to `dev` branch

---

## Milestones Summary

| Milestone | Target Date | Deliverable |
|-----------|------------|-------------|
| M1: Data Ready | End of Week 1 | Clean dataset + baseline metrics |
| M2: Triage Engine | End of Week 2 | `/bugs/analyze` live, ≥70% Top-3 |
| M3: Effort Estimator | End of Week 3 | MMRE < 0.40, endpoint integrated |
| M4: Risk Dashboard | End of Week 4 | Streamlit live, risk gauge working |
| M5: Full Pipeline | End of Week 5 | GA + Slack + end-to-end < 5s |
| M6: Final Submission | End of Week 6 | Report + demo + code submitted |

---

*SprintGuard Execution Plan v1.0 | June 2026*
