# src/save_arxiv_metadata.py
import os
import pandas as pd
import arxiv

# =========================
# CONFIG
# =========================
QUERY = "machine learning"
MAX_RESULTS = 10000   # fetch more than 1500 to cover all PDFs

OUT_CSV = os.path.join("data", "metadata", "arxiv_metadata.csv")

def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    print(f"🔎 Fetching arXiv metadata for query='{QUERY}' (max {MAX_RESULTS})")

    search = arxiv.Search(
        query=QUERY,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    rows = []
    for i, result in enumerate(search.results(), start=1):
        arxiv_id = result.get_short_id()        # e.g. 2512.00580v2
        abstract = result.summary.strip()
        title = result.title.strip()

        rows.append({
            "arxiv_id": arxiv_id,
            "filename": f"{arxiv_id}.pdf",
            "title": title,
            "abstract": abstract
        })

        if i % 500 == 0:
            print(f"✅ Fetched {i} records")

    df = pd.DataFrame(rows).drop_duplicates(subset=["arxiv_id"])
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print("\n==============================")
    print(f"💾 Saved metadata to: {OUT_CSV}")
    print(f"Total records saved: {len(df)}")
    print("==============================")
    print(df.head(3))

if __name__ == "__main__":
    main()
