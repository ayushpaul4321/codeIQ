"""
SprintGuard - Data Preprocessing Pipeline
Stages: clean → filter → feature engineer → split
Run after download_bugs.py
"""

import pandas as pd
import numpy as np
import re
import json
import os
from html.parser import HTMLParser
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(__file__)
RAW      = os.path.join(BASE, "raw",       "eclipse_bugs.csv")
CLEANED  = os.path.join(BASE, "cleaned",   "bugs_clean.csv")
FILTERED = os.path.join(BASE, "filtered",  "bugs_filtered.csv")
PROCESSED= os.path.join(BASE, "processed", "bugs_features.csv")
LABEL_MAP= os.path.join(BASE, "processed", "dev_label_map.json")
SPLIT_DIR= os.path.join(BASE, "splits")
CACHE_PATH = os.path.join(BASE, "..", "models", "bert_triage", "embeddings_cache.npz")

# Exclude auto-assign / bot accounts
EXCLUDE_DEVS = {
    "inbox@eclipse.org", "nobody@eclipse.org", "unassigned",
    "platform-inbox@eclipse.org", "jdt-core-inbox@eclipse.org",
    "pde-inbox@eclipse.org"
}
MIN_BUGS_PER_DEV = 10   # developer must have fixed ≥10 bugs


# ─── HTML Stripper ──────────────────────────────────────────────────────────

class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts = []

    def handle_data(self, data):
        self._parts.append(data)

    def get_text(self):
        return " ".join(self._parts)


def strip_html(text: str) -> str:
    s = _HTMLStripper()
    try:
        s.feed(text)
    except Exception:
        pass
    return s.get_text()


# ─── Text Cleaning ───────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = strip_html(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)           # remove URLs
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]+\b", " ", text)  # emails
    # Remove java stack trace lines
    text = re.sub(r"\n\s+at\s+[\w\.\$<>]+\(.*?\)", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)              # non-ascii
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


# ─── Stage 1: Clean ─────────────────────────────────────────────────────────

def stage_clean():
    print("\n[Stage 1] Cleaning raw data...")
    df = pd.read_csv(RAW)
    print(f"  Raw rows: {len(df):,}")

    df["summary"]     = df["summary"].fillna("").apply(clean_text)
    # description field may not exist if downloaded via summary-only API
    if "description" in df.columns:
        df["description"] = df["description"].fillna("").apply(clean_text)
        df["text"] = (df["summary"] + " " + df["description"]).str.strip()
    else:
        df["text"] = df["summary"].str.strip()

    # Drop rows with very short text
    df = df[df["text"].str.len() >= 20]
    print(f"  After text filter: {len(df):,}")

    df.to_csv(CLEANED, index=False)
    print(f"  Saved → {CLEANED}")
    return df


# ─── Stage 2: Filter ────────────────────────────────────────────────────────

def stage_filter():
    print("\n[Stage 2] Filtering...")
    df = pd.read_csv(CLEANED)

    # Remove bot / inbox accounts
    df = df[~df["assigned_to"].isin(EXCLUDE_DEVS)]
    df = df[df["assigned_to"].notna()]

    # Keep only devs with enough samples
    counts = df["assigned_to"].value_counts()
    qualified = counts[counts >= MIN_BUGS_PER_DEV].index
    df = df[df["assigned_to"].isin(qualified)]

    print(f"  After dev filter: {len(df):,} bugs, {df['assigned_to'].nunique()} devs")

    df.to_csv(FILTERED, index=False)
    print(f"  Saved → {FILTERED}")
    return df


# ─── Stage 3: Feature Engineering ───────────────────────────────────────────

def stage_features():
    print("\n[Stage 3] Feature engineering...")
    df = pd.read_csv(FILTERED)

    # Delete stale embedding cache before writing any new split files so that
    # subsequent training runs cannot reuse outdated pre-computed vectors.
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
        print(f"  Deleted stale embedding cache → {CACHE_PATH}")

    # Encode developer labels
    le = LabelEncoder()
    df["dev_id"] = le.fit_transform(df["assigned_to"])

    label_map = {str(cls): int(idx)
                 for idx, cls in enumerate(le.classes_)}
    with open(LABEL_MAP, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"  Saved label map → {LABEL_MAP} ({len(label_map)} devs)")

    # Component encoding
    df["component"] = df["component"].fillna("unknown").str.lower().str.strip()

    # Dev history stats
    dev_stats = df.groupby("dev_id").agg(
        dev_bugs_fixed=("id", "count"),
    ).reset_index()
    df = df.merge(dev_stats, on="dev_id", how="left")

    # Enrich text with component and product context
    # This gives BERT structural signal beyond just the summary title
    # canonical inference-time format: "product: {product} component: {component} summary: {text}"
    df["text"] = (
        "product: " + df["product"].fillna("").str.lower() + " "
        "component: " + df["component"].fillna("").str.lower() + " "
        "summary: " + df["text"]
    ).str.strip()

    # Keep only needed columns
    df = df[["id", "text", "component", "product", "assigned_to", "dev_id", "dev_bugs_fixed"]]
    df.reset_index(drop=True, inplace=True)

    df.to_csv(PROCESSED, index=False)
    print(f"  Final: {len(df):,} rows, {df['dev_id'].nunique()} classes")
    print(f"  Saved → {PROCESSED}")
    return df


# ─── Stage 4: Train/Val/Test Split ──────────────────────────────────────────

def stage_split():
    print("\n[Stage 4] Splitting...")
    df = pd.read_csv(PROCESSED)

    # Remove any dev with < 3 samples (can't stratify)
    counts = df["dev_id"].value_counts()
    df = df[df["dev_id"].isin(counts[counts >= 3].index)]

    train_val, test = train_test_split(
        df, test_size=0.15, stratify=df["dev_id"], random_state=42
    )
    train, val = train_test_split(
        train_val, test_size=0.176, stratify=train_val["dev_id"], random_state=42
    )

    train.to_csv(os.path.join(SPLIT_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(SPLIT_DIR, "val.csv"),   index=False)
    test.to_csv(os.path.join(SPLIT_DIR, "test.csv"),  index=False)

    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    print(f"  Saved splits → {SPLIT_DIR}/")


# ─── Quality Report ─────────────────────────────────────────────────────────

def quality_report():
    print("\n[Quality Report]")
    df = pd.read_csv(PROCESSED)
    counts = df["dev_id"].value_counts()
    print(f"  Total bugs      : {len(df):,}")
    print(f"  Unique devs     : {df['dev_id'].nunique()}")
    print(f"  Avg text length : {df['text'].str.len().mean():.0f} chars")
    print(f"  Min bugs/dev    : {counts.min()}")
    print(f"  Max bugs/dev    : {counts.max()}")
    top_share = counts.iloc[0] / len(df)
    flag = "⚠ Consider resampling" if top_share > 0.20 else "✓ OK"
    print(f"  Top dev share   : {top_share:.1%}  {flag}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  SprintGuard — Preprocessing Pipeline")
    print("=" * 55)

    if not os.path.exists(RAW):
        print(f"\nERROR: Raw data not found at {RAW}")
        print("Run download_bugs.py first.")
        raise SystemExit(1)

    stage_clean()
    stage_filter()
    stage_features()
    stage_split()
    quality_report()

    print("\n✓ Preprocessing complete.")
