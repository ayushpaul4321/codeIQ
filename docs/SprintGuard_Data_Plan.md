# SprintGuard — Data Plan

> Covers all datasets used, how to acquire them, how to process them,  
> and how each feeds into the five SprintGuard ML modules.

---

## 1. Data Requirements Overview

| Module | Data Needed | Volume Target | Format |
|--------|------------|--------------|--------|
| Bug NLP Engine (BERT) | Bug title + description + module tag | ≥ 25,000 bugs | CSV |
| Dev Assignment NN | Bug + assigned developer label | ≥ 25,000 labeled pairs | CSV |
| Effort Estimator (LSTM/ANFIS) | Bug + fix time in hours | ≥ 500 rows (min), 2,000+ ideal | CSV |
| Sprint Risk Engine (Fuzzy) | Sprint velocity, bug inflow, days left | ≥ 100 sprint snapshots | CSV / JSON |
| GA Re-planner | Sprint story list + story points + priority | Sample data (mock OK for MVP) | JSON |

---

## 2. Primary Datasets

### 2.1 Eclipse Bugzilla

| Property | Detail |
|----------|--------|
| **URL** | https://bugs.eclipse.org/bugs |
| **REST API** | `https://bugs.eclipse.org/bugs/rest/bug?product=JDT&limit=500` |
| **Volume** | ~400,000 total bugs; ~50,000 FIXED with assignee |
| **Key Fields** | `id`, `summary`, `description`, `assigned_to`, `component`, `creation_time`, `cf_fixed_in` |
| **License** | Eclipse Public License (research use permitted) |
| **Use In SprintGuard** | Bug triage (Modules 1 + 2) |

**Download Script (REST API):**
```python
import requests, pandas as pd, time

BASE = "https://bugs.eclipse.org/bugs/rest/bug"
params = {
    "product": "JDT",
    "status": "RESOLVED",
    "resolution": "FIXED",
    "include_fields": "id,summary,description,assigned_to,component,creation_time",
    "limit": 500,
    "offset": 0
}

bugs = []
while True:
    resp = requests.get(BASE, params=params).json()
    batch = resp.get("bugs", [])
    if not batch:
        break
    bugs.extend(batch)
    params["offset"] += 500
    time.sleep(0.5)  # rate limit courtesy

df = pd.DataFrame(bugs)
df.to_csv("storage/datasets/raw/eclipse_bugs.csv", index=False)
print(f"Downloaded {len(df)} bugs")
```

---

### 2.2 Mozilla Bugzilla

| Property | Detail |
|----------|--------|
| **URL** | https://bugzilla.mozilla.org |
| **REST API** | `https://bugzilla.mozilla.org/rest/bug?product=Firefox&resolution=FIXED` |
| **Volume** | ~1.2M total; ~80,000 FIXED with fix time |
| **Key Fields** | `id`, `summary`, `assigned_to`, `component`, `creation_time`, `last_change_time`, `cf_crash_signature` |
| **License** | Mozilla Public License 2.0 (research use permitted) |
| **Use In SprintGuard** | Dev assignment training (Module 2); effort estimation (Module 3) |

**Effort Label Extraction:**
Fix time = `last_change_time` − `creation_time` (in hours, for FIXED bugs only).  
Filter: keep only bugs where fix time is between 0.5h and 80h (outlier removal).

---

### 2.3 VSCode GitHub Issues

| Property | Detail |
|----------|--------|
| **URL** | https://github.com/microsoft/vscode/issues |
| **API** | GitHub REST API v3: `GET /repos/microsoft/vscode/issues` |
| **Volume** | ~160,000 issues; ~30,000 closed bugs with assignee |
| **Key Fields** | `title`, `body`, `assignee`, `labels`, `closed_at`, `created_at`, `milestone` |
| **License** | MIT License (open source) |
| **Use In SprintGuard** | Modern language patterns for BERT; sprint-like milestone data |

**GitHub API Fetch:**
```python
import requests, pandas as pd

TOKEN = "ghp_your_token_here"
HEADERS = {"Authorization": f"token {TOKEN}"}
URL = "https://api.github.com/repos/microsoft/vscode/issues"

issues = []
page = 1
while True:
    params = {"state": "closed", "labels": "bug", "per_page": 100, "page": page}
    resp = requests.get(URL, headers=HEADERS, params=params).json()
    if not resp:
        break
    issues.extend(resp)
    page += 1

df = pd.DataFrame([{
    "id": i["number"],
    "title": i["title"],
    "body": i.get("body", ""),
    "assignee": i["assignee"]["login"] if i["assignee"] else None,
    "created_at": i["created_at"],
    "closed_at": i["closed_at"],
    "labels": [l["name"] for l in i["labels"]]
} for i in issues if i.get("assignee")])

df.to_csv("storage/datasets/raw/vscode_bugs.csv", index=False)
```

---

### 2.4 Apache JIRA Public Archives

| Property | Detail |
|----------|--------|
| **URL** | https://issues.apache.org/jira |
| **Projects** | HADOOP, SPARK, KAFKA, HIVE (all public) |
| **REST API** | `https://issues.apache.org/jira/rest/api/2/search?jql=project=KAFKA AND issuetype=Bug` |
| **Volume** | ~100,000 bugs across major Apache projects |
| **Key Fields** | `summary`, `description`, `assignee`, `sprint`, `story_points`, `created`, `resolutiondate` |
| **License** | Apache License 2.0 |
| **Use In SprintGuard** | Sprint velocity data (Module 4); story-level effort labels (Module 3) |

**JQL Query for Sprint Data:**
```
project = KAFKA 
AND issuetype in (Bug, Story, Task) 
AND sprint in openSprints() 
ORDER BY created DESC
```

---

### 2.5 Synthetic Sprint Data (Generated for Module 4 + 5)

Since public sprint board data is sparse, we generate realistic synthetic data for the Fuzzy Risk Engine and GA Re-planner.

**Generation Parameters:**
```python
import numpy as np, pandas as pd, json

np.random.seed(42)
n_sprints = 200

sprints = []
for i in range(n_sprints):
    velocity = np.random.normal(40, 8)          # story points per sprint
    bugs_added = np.random.poisson(4)           # mid-sprint bugs
    bug_hours = np.random.exponential(3, bugs_added).sum()
    days_remaining = np.random.randint(1, 10)
    team_size = np.random.randint(3, 7)
    completed_points = np.random.normal(velocity * 0.6, 5)
    
    # Ground truth label: did sprint spill over?
    capacity = team_size * days_remaining * 6   # 6 productive hours/day
    spillover = 1 if bug_hours > (capacity * 0.35) else 0
    
    sprints.append({
        "sprint_id": f"SPRINT-{i+1:03d}",
        "velocity_last_3": round(velocity, 1),
        "bugs_added_midprint": bugs_added,
        "bug_hours_total": round(bug_hours, 2),
        "days_remaining": days_remaining,
        "team_size": team_size,
        "completed_story_points": round(completed_points, 1),
        "spillover_label": spillover
    })

df = pd.DataFrame(sprints)
df.to_csv("storage/datasets/processed/synthetic_sprints.csv", index=False)
print(f"Spillover rate: {df.spillover_label.mean():.1%}")
```

---

## 3. Data Processing Pipeline

### 3.1 Pipeline Overview

```
Raw Data Sources
      │
      ▼
┌─────────────────────┐
│  Stage 1: Ingest    │  Download CSVs from APIs
│  (raw/)             │  Store as-is, no modification
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Stage 2: Clean     │  Remove HTML, duplicates, empties
│  (cleaned/)         │  Normalize text, fix encoding
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Stage 3: Filter    │  Keep FIXED bugs, valid assignees
│  (filtered/)        │  Effort: remove outliers (<0.5h, >80h)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Stage 4: Feature   │  Encode labels, compute features
│  Engineering        │  Generate embeddings
│  (processed/)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Stage 5: Split     │  Train/Val/Test (70/15/15)
│  (splits/)          │  Stratified by developer
└─────────────────────┘
```

---

### 3.2 Stage 2: Cleaning

```python
import pandas as pd, re
from html.parser import HTMLParser

class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
    def handle_data(self, data):
        self.text.append(data)
    def get_text(self):
        return ' '.join(self.text)

def clean_bug_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Strip HTML tags
    s = HTMLStripper()
    s.feed(text)
    text = s.get_text()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove stack traces (lines starting with "at ")
    text = re.sub(r'\n\s+at\s+[\w\.\$]+\(.*?\)', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Lowercase
    return text.lower()

df = pd.read_csv("storage/datasets/raw/eclipse_bugs.csv")
df["text_clean"] = (df["summary"] + " " + df["description"].fillna("")).apply(clean_bug_text)
df = df[df["text_clean"].str.len() > 20]  # remove near-empty descriptions
df.to_csv("storage/datasets/cleaned/eclipse_bugs_clean.csv", index=False)
```

---

### 3.3 Stage 3: Filtering

```python
df = pd.read_csv("storage/datasets/cleaned/eclipse_bugs_clean.csv")

# Keep only bugs with a real assigned developer (not auto-assign accounts)
EXCLUDE_DEVS = {"inbox@eclipse.org", "nobody@mozilla.org", "unassigned"}
df = df[~df["assigned_to"].isin(EXCLUDE_DEVS)]
df = df[df["assigned_to"].notna()]

# Keep only devs with >= 10 fixed bugs (enough training signal)
dev_counts = df["assigned_to"].value_counts()
qualified_devs = dev_counts[dev_counts >= 10].index
df = df[df["assigned_to"].isin(qualified_devs)]

# Effort: filter fix time outliers
if "fix_hours" in df.columns:
    df = df[(df["fix_hours"] >= 0.5) & (df["fix_hours"] <= 80)]

print(f"After filtering: {len(df)} bugs, {df['assigned_to'].nunique()} developers")
df.to_csv("storage/datasets/filtered/eclipse_bugs_filtered.csv", index=False)
```

---

### 3.4 Stage 4: Feature Engineering

```python
from sklearn.preprocessing import LabelEncoder
import json

df = pd.read_csv("storage/datasets/filtered/eclipse_bugs_filtered.csv")

# Encode developer labels
le = LabelEncoder()
df["dev_id"] = le.fit_transform(df["assigned_to"])

# Save label map
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
with open("storage/datasets/processed/dev_label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

# Module/component one-hot encoding
component_dummies = pd.get_dummies(df["component"], prefix="comp")
df = pd.concat([df, component_dummies], axis=1)

# Dev history features (per developer)
dev_stats = df.groupby("assigned_to").agg(
    bugs_fixed=("id", "count"),
    avg_fix_hours=("fix_hours", "mean"),
).reset_index()
df = df.merge(dev_stats, on="assigned_to", how="left")

df.to_csv("storage/datasets/processed/eclipse_bugs_features.csv", index=False)
print(f"Final dataset: {len(df)} rows, {df['dev_id'].nunique()} classes")
```

---

### 3.5 Stage 5: Train/Val/Test Split

```python
from sklearn.model_selection import train_test_split

df = pd.read_csv("storage/datasets/processed/eclipse_bugs_features.csv")

# Stratify by developer to ensure all devs appear in all splits
train_val, test = train_test_split(df, test_size=0.15, stratify=df["dev_id"], random_state=42)
train, val = train_test_split(train_val, test_size=0.176, stratify=train_val["dev_id"], random_state=42)
# 0.176 of 0.85 ≈ 0.15 of total

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

train.to_csv("storage/datasets/splits/train.csv", index=False)
val.to_csv("storage/datasets/splits/val.csv", index=False)
test.to_csv("storage/datasets/splits/test.csv", index=False)
```

---

## 4. Data Schema Reference

### 4.1 `bugs_clean.csv` (Core Training Data)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `bug_id` | int | Unique bug identifier | 452301 |
| `text_clean` | str | Cleaned title + description | "checkout crashes coupon expired 500 error payment" |
| `assigned_to` | str | Developer email / username | "alice@team.com" |
| `dev_id` | int | Encoded developer label | 3 |
| `component` | str | Module / subsystem | "payment" |
| `fix_hours` | float | Hours to fix (if available) | 4.0 |
| `created_at` | datetime | Bug report timestamp | 2024-03-15T10:23:00Z |
| `resolved_at` | datetime | Fix timestamp | 2024-03-15T14:18:00Z |

---

### 4.2 `sprint_snapshots.csv` (Risk Engine Data)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `sprint_id` | str | Sprint identifier | "SPRINT-042" |
| `snapshot_day` | int | Day of sprint (1–14) | 5 |
| `bugs_added` | int | New bugs since last snapshot | 2 |
| `bug_hours_added` | float | Effort of new bugs | 7.5 |
| `velocity_last_3` | float | Avg story points, last 3 sprints | 38.5 |
| `velocity_trend` | float | % change from prev sprint | -0.12 |
| `days_remaining` | int | Days until sprint end | 3 |
| `team_available_hours` | float | Remaining dev capacity | 42.0 |
| `spillover_label` | int | Ground truth: did sprint spill? | 1 |

---

### 4.3 `sprint_stories.json` (Re-planner Data)

```json
{
  "sprint_id": "SPRINT-042",
  "stories": [
    {
      "id": "JIRA-440",
      "title": "Update payment UI",
      "story_points": 2,
      "priority": "low",
      "effort_hours": 4.0,
      "assignee": "bob",
      "must_have": false
    },
    {
      "id": "JIRA-445",
      "title": "Fix auth token expiry",
      "story_points": 3,
      "priority": "high",
      "effort_hours": 6.5,
      "assignee": "alice",
      "must_have": true
    }
  ],
  "available_capacity_hours": 32.0
}
```

---

## 5. Dataset Size Estimates & Quality Checks

### Expected Final Dataset Sizes

| Dataset | Raw | After Filtering | Train | Val | Test |
|---------|-----|----------------|-------|-----|------|
| Eclipse bugs | ~50,000 | ~28,000 | ~19,600 | ~4,200 | ~4,200 |
| Mozilla bugs | ~80,000 | ~35,000 | ~24,500 | ~5,250 | ~5,250 |
| VSCode issues | ~30,000 | ~12,000 | ~8,400 | ~1,800 | ~1,800 |
| Effort labels | ~15,000 | ~3,500 | ~2,450 | ~525 | ~525 |
| Sprint snapshots | — | ~200 synthetic | 140 | 30 | 30 |

### Quality Checks (Run Before Training)

```python
def run_data_quality_checks(df, name):
    print(f"\n=== Quality Report: {name} ===")
    print(f"Total rows:        {len(df):,}")
    print(f"Missing text:      {df['text_clean'].isna().sum()}")
    print(f"Missing dev label: {df['dev_id'].isna().sum()}")
    print(f"Unique developers: {df['dev_id'].nunique()}")
    print(f"Avg text length:   {df['text_clean'].str.len().mean():.0f} chars")
    print(f"Min dev samples:   {df['dev_id'].value_counts().min()}")
    print(f"Max dev samples:   {df['dev_id'].value_counts().max()}")

    # Warn if class imbalance is severe
    top_dev_share = df['dev_id'].value_counts().iloc[0] / len(df)
    if top_dev_share > 0.20:
        print(f"⚠ WARNING: Top dev has {top_dev_share:.1%} of bugs — consider resampling")
    else:
        print(f"✓ Class balance OK (top dev: {top_dev_share:.1%})")
```

---

## 6. Data Storage Structure

```
storage/
├── datasets/
│   ├── raw/
│   │   ├── eclipse_bugs.csv            # Direct API download
│   │   ├── mozilla_bugs.csv
│   │   └── vscode_issues.csv
│   ├── cleaned/
│   │   ├── eclipse_bugs_clean.csv      # HTML stripped, whitespace normalized
│   │   ├── mozilla_bugs_clean.csv
│   │   └── vscode_issues_clean.csv
│   ├── filtered/
│   │   ├── eclipse_bugs_filtered.csv   # Only FIXED, qualified devs
│   │   └── effort_dataset.csv          # Bugs with fix_hours label
│   ├── processed/
│   │   ├── eclipse_bugs_features.csv   # Dev ID encoded + component features
│   │   ├── dev_label_map.json          # dev_email → int mapping
│   │   ├── synthetic_sprints.csv       # Generated sprint snapshots
│   │   └── sprint_stories_sample.json  # Sample story sets for GA
│   └── splits/
│       ├── train.csv                   # 70% stratified
│       ├── val.csv                     # 15% stratified
│       └── test.csv                    # 15% stratified (held-out)
└── models/
    ├── bert_triage/
    │   ├── mlp_classifier.pt           # Saved MLP weights
    │   └── label_encoder.pkl           # Dev label encoder
    ├── effort_estimator/
    │   ├── lstm_estimator.pt
    │   └── anfis_estimator.pkl
    └── sprint_risk/
        └── fis_config.json             # Fuzzy membership params
```

---

## 7. Data Ethics & Compliance

| Concern | Status | Notes |
|---------|--------|-------|
| PII in bug reports | ⚠ Review | Developer emails present — anonymize before sharing |
| License compliance | ✓ OK | Eclipse (EPL), Mozilla (MPL 2.0), Apache (AL 2.0) all permit research use |
| Data redistribution | ⚠ Do not redistribute raw | Use only for model training; do not publish raw dumps |
| Sensitive bug content | ✓ Filtered | Security-embargoed bugs excluded via `resolution != WONTFIX` filter |
| Synthetic data labeling | ✓ Clear | Synthetic datasets explicitly labeled as generated in all files |

### Anonymization Step (Before Publishing Results)

```python
import hashlib

def anonymize_email(email: str) -> str:
    """Replace developer email with deterministic pseudonym for publication."""
    h = hashlib.sha256(email.encode()).hexdigest()[:8]
    return f"dev_{h}"

df["assigned_to_anon"] = df["assigned_to"].apply(anonymize_email)
df.drop(columns=["assigned_to"], inplace=True)
```

---

## 8. Data Collection Timeline

| Week | Data Activity | Responsible |
|------|--------------|-------------|
| Week 1, Mon | Download Eclipse Bugzilla dump (50k bugs) | Data Engineer |
| Week 1, Tue | Download Mozilla bugs (50k FIXED) | Data Engineer |
| Week 1, Wed | Download VSCode GitHub issues | Data Engineer |
| Week 1, Thu | Run cleaning + filtering pipeline | Data Engineer |
| Week 1, Fri | Run quality checks; confirm ≥25k usable bugs | Data Engineer |
| Week 2, Mon | Generate BERT embeddings for training set | ML Engineer |
| Week 3, Mon | Extract effort labels; generate `effort_dataset.csv` | Data Engineer |
| Week 3, Tue | Generate synthetic sprint snapshots (200 rows) | Data Engineer |
| Week 4, Mon | Fetch Apache JIRA sprint samples (optional enrichment) | Data Engineer |

---

*SprintGuard Data Plan v1.0 | June 2026*
