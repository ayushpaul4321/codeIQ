"""
SprintGuard - Bug Dataset Downloader
Downloads Eclipse Bugzilla bugs (FIXED, with assignee) for triage training.
Uses the public REST API — no credentials needed.
"""

import requests
import pandas as pd
import time
import os
import sys

BASE_URL = "https://bugs.eclipse.org/bugs/rest/bug"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "raw", "eclipse_bugs.csv")

# Products to pull from (diverse modules = better generalization)
PRODUCTS = ["JDT", "Platform", "PDE", "CDT"]

def fetch_bugs(product: str, max_bugs: int = 3000) -> list:
    """Fetch FIXED bugs with assignee for a given Eclipse product."""
    bugs = []
    offset = 0
    batch_size = 500

    print(f"  Fetching {product}...", end="", flush=True)

    while len(bugs) < max_bugs:
        params = {
            "product": product,
            "status": "RESOLVED",
            "resolution": "FIXED",
            # description is not available in bulk listing — use summary only
            "include_fields": "id,summary,assigned_to,component,creation_time,last_change_time",
            "limit": batch_size,
            "offset": offset,
        }
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            batch = resp.json().get("bugs", [])
        except Exception as e:
            print(f"\n  Warning: request failed at offset {offset}: {e}")
            break

        if not batch:
            break

        bugs.extend(batch)
        offset += batch_size
        print(".", end="", flush=True)
        time.sleep(0.4)  # be polite to the API

        if len(batch) < batch_size:
            break  # last page

    print(f" {len(bugs)} bugs")
    return bugs[:max_bugs]


def main():
    print("=" * 55)
    print("  SprintGuard — Eclipse Bug Downloader")
    print("=" * 55)

    all_bugs = []
    for product in PRODUCTS:
        bugs = fetch_bugs(product, max_bugs=3000)
        for b in bugs:
            b["product"] = product
        all_bugs.extend(bugs)

    df = pd.DataFrame(all_bugs)
    print(f"\nTotal downloaded : {len(df):,} bugs")

    # Basic dedup
    df.drop_duplicates(subset="id", inplace=True)
    print(f"After dedup      : {len(df):,} bugs")

    # Filter: need summary + assigned_to
    df = df[df["summary"].notna() & df["assigned_to"].notna()]
    df = df[df["assigned_to"].str.strip() != ""]
    print(f"After null filter: {len(df):,} bugs")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
