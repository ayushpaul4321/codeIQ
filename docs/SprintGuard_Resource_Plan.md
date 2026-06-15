# SprintGuard — Resource Plan

> **Project Duration:** 6 Weeks  
> **Team Size:** 3–4 Members  
> **Budget Category:** Academic / Low-cost  

---

## 1. Team Roles & Responsibilities

### Role Definitions

| Role | Count | Primary Responsibilities |
|------|-------|------------------------|
| **Lead Developer** | 1 | System architecture, FastAPI backend, API design, Docker, CI/CD, code reviews |
| **ML Engineer** | 1–2 | BERT fine-tuning, MLP, LSTM, ANFIS, FIS, GA implementation, model evaluation |
| **Data Engineer** | 1 | Dataset sourcing, cleaning, preprocessing, feature engineering, EDA |
| **Frontend Developer** | 1 | Streamlit dashboard, visualizations, UX flow, Slack integration |

> In a 3-person team: Lead Dev takes DevOps duties. ML Engineer doubles as Data Engineer in Week 1.  
> In a 4-person team: roles map 1:1.

---

### Responsibility Matrix (RACI)

| Task | Lead Dev | ML Engineer | Data Engineer | Frontend Dev |
|------|----------|------------|--------------|-------------|
| Repo + Docker setup | **R/A** | I | I | I |
| Dataset download + cleaning | I | C | **R/A** | I |
| BERT embedding model | C | **R/A** | C | I |
| MLP dev assignment | C | **R/A** | I | I |
| LSTM / ANFIS effort estimator | C | **R/A** | C | I |
| Fuzzy risk engine | C | **R/A** | I | I |
| GA re-planner | C | **R/A** | I | I |
| FastAPI endpoints | **R/A** | C | I | I |
| Streamlit dashboard | I | C | I | **R/A** |
| Jira / Slack integration | **R/A** | I | I | C |
| Model evaluation + metrics | C | **R/A** | C | I |
| Project report | **A** | R | R | R |
| Demo preparation | **A** | R | R | R |

**R** = Responsible, **A** = Accountable, **C** = Consulted, **I** = Informed

---

### Weekly Effort Allocation (Hours per Role)

| Week | Lead Dev | ML Engineer | Data Engineer | Frontend Dev | Total |
|------|----------|------------|--------------|-------------|-------|
| Week 1 | 12h | 8h | 18h | 4h | **42h** |
| Week 2 | 18h | 20h | 6h | 4h | **48h** |
| Week 3 | 14h | 22h | 8h | 4h | **48h** |
| Week 4 | 16h | 14h | 4h | 16h | **50h** |
| Week 5 | 20h | 16h | 2h | 12h | **50h** |
| Week 6 | 12h | 14h | 6h | 8h | **40h** |
| **Total** | **92h** | **94h** | **44h** | **48h** | **278h** |

---

## 2. Compute Resources

### Minimum Requirements (Development)

| Resource | Specification | Purpose |
|----------|-------------|---------|
| CPU | 4-core, 8GB RAM | FastAPI, Streamlit, data preprocessing |
| GPU (optional) | NVIDIA 6GB VRAM or better | BERT fine-tuning, LSTM training |
| Storage | 20GB free disk space | Datasets, model checkpoints, Docker images |
| Network | Stable broadband | Dataset downloads, API calls |

### Free Cloud GPU Options (No cost)

| Platform | GPU Available | RAM | Storage | Limit | Best For |
|----------|-------------|-----|---------|-------|---------|
| **Google Colab Free** | T4 (15GB VRAM) | 12GB | 100GB | ~12h session | BERT embedding generation |
| **Kaggle Notebooks** | P100 / T4 | 16GB | 20GB | 30h/week | Model training |
| **Google Colab Pro** | A100 / V100 | 25GB+ | 100GB | ~$10/month | Full fine-tuning |
| **AWS Free Tier** | None (CPU only) | 1GB | 30GB | 750h/month | FastAPI deployment |

### Recommended Setup (No Budget)

```
Local machine:    Data preprocessing, FastAPI dev, Streamlit
Google Colab:     BERT fine-tuning (use free GPU, save weights to Drive)
Kaggle:           LSTM / ANFIS training
Docker Desktop:   Local full-stack testing
```

### Estimated Cloud Cost (If paid)

| Service | Usage | Monthly Cost |
|---------|-------|-------------|
| Google Colab Pro | 2 months | ~$20 |
| AWS t2.micro (FastAPI) | 6 weeks | $0 (free tier) |
| Supabase (Postgres) | 6 weeks | $0 (free tier) |
| **Total** | | **~$20 max** |

---

## 3. Software & Libraries

### All Free / Open Source

| Category | Tool | Version | License |
|----------|------|---------|---------|
| Language | Python | 3.11 | PSF |
| ML Framework | PyTorch | 2.2.0 | BSD |
| Transformers | HuggingFace Transformers | 4.40.0 | Apache 2.0 |
| Sentence Embedding | sentence-transformers | 2.7.0 | Apache 2.0 |
| Fuzzy Logic | scikit-fuzzy | 0.4.2 | BSD |
| Genetic Algorithm | DEAP | 1.4.1 | LGPL |
| Data Processing | pandas, numpy | 2.2.2, 1.26.4 | BSD |
| ML Utilities | scikit-learn | 1.4.2 | BSD |
| Backend | FastAPI + uvicorn | 0.111.0 | MIT |
| Database ORM | SQLAlchemy | 2.0.29 | MIT |
| Dashboard | Streamlit | 1.35.0 | Apache 2.0 |
| Visualization | Plotly | 5.22.0 | MIT |
| Containerization | Docker Desktop | Latest | Apache 2.0 |
| Version Control | Git + GitHub | — | Free |
| Notebook | Jupyter Lab | 4.x | BSD |

### IDE / Dev Tools (All Free)

| Tool | Purpose |
|------|---------|
| VS Code | Primary IDE |
| Postman | API testing |
| DBeaver | Database GUI |
| GitHub Actions | CI (optional) |

---

## 4. Infrastructure Plan

### Local Development Stack

```
┌─────────────────────────────────────────┐
│           docker-compose.yml            │
│                                         │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │  FastAPI App │  │   PostgreSQL DB  │ │
│  │  Port: 8000  │  │   Port: 5432     │ │
│  └──────┬───────┘  └──────────────────┘ │
│         │                               │
│  ┌──────▼───────┐  ┌──────────────────┐ │
│  │  Streamlit   │  │   Redis Cache    │ │
│  │  Port: 8501  │  │   Port: 6379     │ │
│  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────┘
```

### Services Breakdown

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `sprintguard-api` | Custom Python 3.11 | 8000 | FastAPI backend |
| `sprintguard-ui` | Custom Python 3.11 | 8501 | Streamlit frontend |
| `postgres` | postgres:15 | 5432 | Sprint + developer data |
| `redis` | redis:7-alpine | 6379 | Sprint state cache |

---

## 5. External Services & APIs

| Service | Purpose | Cost | Auth Required |
|---------|---------|------|--------------|
| Jira Cloud API | Sprint data, issue updates | Free tier (personal) | API token |
| GitHub API | Webhook for bug issues | Free | Personal Access Token |
| Slack API | Risk alerts | Free | Bot token |
| HuggingFace Hub | Model downloads | Free | Optional (public models) |
| Bugzilla REST API | Bug dataset fetch | Free | None (public) |

### API Keys Needed (Store in `.env`)

```bash
# .env (never commit to git)
JIRA_BASE_URL=https://yourteam.atlassian.net
JIRA_API_TOKEN=your_token_here
JIRA_EMAIL=your_email@example.com

GITHUB_TOKEN=ghp_...
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

POSTGRES_URL=postgresql://user:pass@localhost:5432/sprintguard
REDIS_URL=redis://localhost:6379
```

---

## 6. Team Communication & Collaboration

| Tool | Purpose | Cost |
|------|---------|------|
| **GitHub** | Code repo, PRs, issue tracking | Free |
| **GitHub Projects** | Sprint board for the project itself | Free |
| **Slack / Discord** | Daily communication | Free |
| **Google Meet / Zoom** | Weekly sync calls | Free |
| **Google Drive** | Report drafts, dataset sharing | Free (15GB) |
| **Notion / Confluence** | Meeting notes, documentation | Free tier |

### Meeting Cadence

| Meeting | Frequency | Duration | Who |
|---------|-----------|---------|-----|
| Daily Standup | Mon–Fri | 15 min | All |
| Sprint Review | Every 2 weeks (Fri) | 1 hour | All |
| Sprint Planning | Every 2 weeks (Mon) | 1 hour | All |
| Ad-hoc pair session | As needed | 30–60 min | 2 members |

---

## 7. Hardware Requirements Summary

### Per Team Member (Minimum)

| Item | Requirement |
|------|------------|
| Laptop/Desktop | ≥ 8GB RAM, ≥ 4-core CPU, 20GB free SSD |
| Operating System | Windows 10+, macOS 12+, or Ubuntu 20.04+ |
| Internet | Stable ≥ 10 Mbps for dataset downloads |
| Browser | Chrome / Firefox (for Streamlit) |

### Shared Resources (Team of 4)

| Resource | Who Owns | Shared Via |
|----------|---------|-----------|
| GPU machine (if 1 exists) | Any team member | Remote SSH / Colab |
| Bugzilla dataset dump | Data Engineer | Google Drive |
| Trained model weights | ML Engineer | Google Drive / Git LFS |
| `.env` secrets | Lead Dev | Secure message (never git) |

---

## 8. Budget Summary

| Category | Item | Estimated Cost |
|----------|------|---------------|
| Compute | Google Colab Pro (2 months, 1 account) | $20 |
| Compute | AWS Free Tier (FastAPI hosting) | $0 |
| Data | All datasets (public, open access) | $0 |
| Software | All libraries (open source) | $0 |
| APIs | Jira, GitHub, Slack (free tiers) | $0 |
| Collaboration | GitHub, Google Drive, Discord | $0 |
| Miscellaneous | Buffer | $10 |
| **Total** | | **$30 max** |

> The entire project is buildable for **$0** using Kaggle GPUs + local machines + free tiers of all services. The $20 Colab Pro is optional convenience for faster BERT training.

---

*SprintGuard Resource Plan v1.0 | June 2026*
