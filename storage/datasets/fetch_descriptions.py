"""
SprintGuard - Fetch Bug Comments/Descriptions
The Bugzilla bulk API doesn't return descriptions. This script fetches
the first comment (which IS the original description) for each bug.
Enriches eclipse_bugs.csv with a 'description' column.
"""

import requests
import pandas as pd
import time
import os

BASE_URL  = "https://bugs.eclipse.org/bugs/rest/bug"
RAW_CSV   = os.path.join(os.path.dirname(__file__), "raw", "eclipse_bugs.csv")
OUT_CSV   = os.path.join(os.path.dirname(__file__), "raw", "eclipse_bugs_full.csv")

BATCH     = 20   # fetch N bugs' comments per request (API supports id list)
SLEEP     = 0.3  # seconds between batches


def fetch_comment_batch(bug_ids: list[int]) -> dict[int, str]:
    """
    Fetch the first comment (= description) for a batch of bug IDs.
    Returns {bug_id: description_text}
    """
    ids_str = ",".join(str(i) for i in bug_ids)
    url = f"{BASE_URL}/{ids_str}/comment"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("bugs", {})
    except Exception as e:
        print(f"  Warning: comment fetch failed for batch: {e}")
        return {}

    result = {}
    for bug_id_str, comments_obj in data.items():
        comments = comments_obj.get("comments", [])
        if comments:
            result[int(bug_id_str)] = comments[0].get("text", "")
    return result


def main():
    print("=" * 55)
    print("  SprintGuard — Fetch Bug Descriptions")
    print("=" * 55)

    df = pd.read_csv(RAW_CSV)
    print(f"Bugs to enrich: {len(df):,}")

    # Skip if already done
    if "description" in df.columns and df["description"].notna().sum() > 100:
        print("Descriptions already present. Skipping.")
        return

    bug_ids = df["id"].tolist()
    descriptions = {}

    total_batches = (len(bug_ids) + BATCH - 1) // BATCH
    for i in range(0, len(bug_ids), BATCH):
        batch = bug_ids[i:i + BATCH]
        batch_num = i // BATCH + 1

        desc_map = fetch_comment_batch(batch)
        descriptions.update(desc_map)

        if batch_num % 10 == 0 or batch_num == total_batches:
            print(f"  Batch {batch_num}/{total_batches}  "
                  f"({len(descriptions)} descriptions fetched)")

        time.sleep(SLEEP)

    df["description"] = df["id"].map(descriptions).fillna("")
    df.to_csv(OUT_CSV, index=False)

    filled = (df["description"].str.len() > 10).sum()
    print(f"\nDescriptions fetched : {filled:,} / {len(df):,}")
    print(f"Saved → {OUT_CSV}")


if __name__ == "__main__":
    main()
