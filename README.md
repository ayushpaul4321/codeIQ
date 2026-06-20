# SprintGuard

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## Overview

SprintGuard is a hybrid neural-fuzzy-evolutionary system for Agile sprint health monitoring and automatic bug triage. It combines BERT-based semantic embeddings, an MLP classifier, LSTM effort estimation, a Mamdani Fuzzy Inference System, and a Genetic Algorithm replanner into a unified FastAPI backend with a Streamlit dashboard.

**Core capabilities:**

- Automatic bug-to-developer assignment using BERT + MLP (Top-3 Accuracy: 78.09%)
- Fix-time effort estimation with confidence intervals using LSTM + Monte Carlo Dropout (MMRE: 0.337)
- Sprint risk scoring using Mamdani Fuzzy Inference System (continuous score 0–1)
- GA-based sprint replanning when risk is HIGH (87% risk reduction rate, <2s)
- Real-time Streamlit dashboard with dark theme and animated charts

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Streamlit Dashboard  :8501                      │
│         Sprint Overview │ Bug Triage │ Re-planner               │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼──────────────────────────────────┐
│                    FastAPI Backend  :8000                        │
│                                                                  │
│  GET  /health                                                    │
│  POST /api/v1/bugs/analyze                                       │
│  GET  /api/v1/sprint/{id}/risk                                   │
│  POST /api/v1/sprint/{id}/replan                                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
         ┌─────────────────────▼──────────────────────┐
         │              Pipeline                       │
         │                                             │
         │  BugEmbeddingService  (BERT)                │
         │          │                                  │
         │  TriageService        (MLP)                 │
         │          │                                  │
         │  EffortEstimatorService (LSTM + MC Dropout) │
         │          │                                  │
         │  SprintRiskEngine     (Mamdani FIS)         │
         │          │                                  │
         │  GAReplanner          (DEAP)                │
         │          │                                  │
         └──────────┼──────────────────────────────────┘
                    │
         ┌──────────▼──────────┐
         │  PostgreSQL/Supabase │
         └─────────────────────┘
```

---

## Project Structure

```
sprintguard/
├── backend/
│   └── app/
│       ├── main.py                  # FastAPI app + lifespan model loading
│       ├── dependencies.py          # DI helpers
│       ├── models/
│       │   ├── bug_embedding.py     # BERT embedding service
│       │   ├── triage_service.py    # MLP triage wrapper
│       │   ├── effort_estimator.py  # LSTM effort estimator
│       │   ├── sprint_risk.py       # Mamdani FIS risk engine
│       │   ├── replanner.py         # DEAP genetic algorithm replanner
│       │   ├── train_triage.py      # Triage model training script
│       │   ├── train_baseline.py    # TF-IDF+SVM baseline training
│       │   └── train_effort.py      # LSTM training script
│       ├── routes/
│       │   ├── bugs.py              # POST /api/v1/bugs/analyze
│       │   └── sprint.py            # GET/POST /api/v1/sprint/{id}/...
│       ├── schemas/                 # Pydantic request/response models
│       ├── services/
│       │   └── sprint_service.py    # Sprint DB read/write
│       └── db/                      # SQLAlchemy models + session
├── frontend/
│   └── dashboard.py                 # Streamlit dashboard (3 pages)
├── storage/
│   ├── datasets/                    # Eclipse Bugzilla data + splits
│   └── models/                      # Trained model checkpoints
│       ├── bert_triage/             # mlp_classifier.pt + metrics
│       └── tfidf_svm/               # tfidf_svm_pipeline.joblib + metrics
├── docs/                            # Project documentation
├── dockers/
│   ├── Dockerfile
│   └── docker-compose.yml
├── alembic/                         # DB migrations
├── requirements.txt
└── .env.example
```

---

## Experimental Results

### RQ1 — Developer Triage (Eclipse Bugzilla, 8,847 bugs, 87 developers)

| Metric | TF-IDF + SVM (Baseline) | BERT + MLP (SprintGuard) | Target | Status |
|---|---|---|---|---|
| Top-1 Accuracy | 49.40% | 44.05% | — | — |
| Top-3 Accuracy | **79.67%** | **78.09%** | ≥ 70% | ✅ |
| MRR | 0.6595 | 0.6259 | ≥ 0.55 | ✅ |
| Training Epochs | — | 80 (best: 79) | — | — |

> **Finding:** Both models exceed the 70% Top-3 target. The TF-IDF+SVM baseline is marginally stronger (79.67% vs 78.09%) on this vocabulary-rich dataset, confirming that sparse bigram features are highly competitive with deep semantic encodings for developer assignment with 87 classes.

---

### RQ2 — Effort Estimation (LSTM)

| Metric | Achieved | Target | Status |
|---|---|---|---|
| MMRE | 0.337 | ≤ 0.40 | ✅ |
| PRED(25) | 0.534 | ≥ 0.50 | ✅ |
| Inference latency | ~42ms | < 100ms | ✅ |

---

### RQ3 — Sprint Risk Engine (Mamdani FIS)

| Test | Result | Status |
|---|---|---|
| Score range [0,1] for all valid inputs | Always satisfied | ✅ |
| Risk label determinism | 100% | ✅ |
| Monotonicity (10,000 random pairs) | 100% | ✅ |

---

### RQ4 — GA Sprint Replanner

| Metric | Achieved | Target | Status |
|---|---|---|---|
| Risk reduction (HIGH → <HIGH) | 87% of sprints | > 70% | ✅ |
| Execution time (15 stories) | 1.8s | < 5s | ✅ |
| Must-have gene invariant | 100% | 100% | ✅ |

---

### End-to-End Pipeline

| Stage | p50 latency | p95 latency |
|---|---|---|
| BERT Embedding | 38ms | 61ms |
| MLP Triage | 4ms | 7ms |
| LSTM Effort | 42ms | 68ms |
| FIS Risk | 12ms | 19ms |
| DB Persist | 48ms | 110ms |
| **Total /analyze** | **~144ms** | **~265ms** |

---

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15 (or Supabase account)
- CUDA GPU optional (RTX 4060 recommended)

### 1. Clone and install

```bash
git clone https://github.com/ayushpaul4321/codeIQ.git
cd codeIQ
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set SUPABASE_URL and SUPABASE_KEY from your Supabase project → Settings → API
```

### 3. Run the backend

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Run the dashboard

```bash
python -m streamlit run frontend/dashboard.py --server.port 8501
```

### 5. Docker (full stack)

```bash
docker-compose -f dockers/docker-compose.yml up
```

The dashboard will be available at `http://localhost:8501` with a **🎭 Demo Mode** toggle (ON by default) that shows realistic mock data without any backend connection.

---

## Dashboard

Three pages are available from the sidebar:

- **Sprint Overview** — burndown chart (Plotly), risk gauge, velocity and bug-hours metric cards
- **Bug Triage** — submit bug title + description, get developer assignment with confidence, top-3 candidates table, effort estimate with CI
- **Re-planner** — check sprint risk, add stories, run GA replanner, view/accept suggestions

Demo mode is enabled by default — click **🎭 Load Demo Data** on Sprint Overview to see a fully populated HIGH-risk sprint in one click.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check, model load status |
| POST | `/api/v1/bugs/analyze` | Full pipeline: embed → triage → effort → risk |
| GET | `/api/v1/sprint/{id}/risk` | Current sprint risk score |
| POST | `/api/v1/sprint/{id}/replan` | GA-based sprint replanning |

**Example request:**

```json
POST /api/v1/bugs/analyze
{
  "title": "NPE when submitting empty form",
  "description": "Application throws NullPointerException at FormValidator.java:142 when the user clicks submit with all fields blank.",
  "reporter": "alice.dev",
  "sprint_id": "SPRINT-42"
}
```

**Example response:**

```json
{
  "bug_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "assigned_dev": "john.smith",
  "assignment_confidence": 0.487,
  "top3_devs": [
    {"dev": "john.smith", "probability": 0.487},
    {"dev": "alice.wong", "probability": 0.312},
    {"dev": "bob.lee",   "probability": 0.201}
  ],
  "effort_estimate": {"hours": 6.5, "confidence_interval": [4.2, 9.1]},
  "sprint_impact": {
    "risk_before": "MEDIUM",
    "risk_after": "HIGH",
    "risk_score": 0.74,
    "factors": ["High bug hours added"],
    "replan_suggested": true
  }
}
```

---

## Key Formulae

| Module | Formula |
|---|---|
| Enriched text | `"product: {p} component: {c} summary: {title} {desc}"` |
| MRR | `(1/N) × Σ 1/rank(correct_dev)` |
| MMRE | `(1/N) × Σ │actual − predicted│ / actual` |
| Monte Carlo CI | `[exp(μ−σ), exp(μ+σ)]` over T=30 dropout passes |
| FIS defuzz | `risk = Σ(μ(z)·z) / Σμ(z)` (centroid) |
| GA fitness | `Σ priority_i × story_points_i` s.t. `Σ effort_i ≤ capacity` |

---

## License

MIT License — see [LICENSE](LICENSE) file.

---

## Citation

```bibtex
@software{sprintguard2024,
  title  = {SprintGuard: Hybrid Neural-Fuzzy-Evolutionary Sprint Health Monitor},
  author = {Paul, Ayush},
  year   = {2024},
  url    = {https://github.com/ayushpaul4321/codeIQ}
}
```
