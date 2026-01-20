# src/build_jsonl_from_extracted.py
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# =========================
# PATHS (anchored to project root)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXTRACTED_DIR = PROJECT_ROOT / "data" / "processed" / "extracted"
META_CSV      = PROJECT_ROOT / "data" / "metadata" / "arxiv_metadata.csv"
OUT_JSONL     = PROJECT_ROOT / "data" / "processed" / "dataset" / "papers.jsonl"

# =========================
# CONFIG
# =========================
MAX_PAPERS = 1500        # set None to use all extracted files
LOG_EVERY = 100

MIN_ARTICLE_CHARS  = 1500
MIN_ABSTRACT_CHARS = 50

def main():
    if not EXTRACTED_DIR.exists():
        raise FileNotFoundError(f"❌ Extracted text folder not found: {EXTRACTED_DIR}")

    if not META_CSV.exists():
        raise FileNotFoundError(f"❌ Metadata CSV not found: {META_CSV}")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    print("📂 EXTRACTED_DIR:", EXTRACTED_DIR)
    print("📄 META_CSV:", META_CSV)
    print("💾 OUT_JSONL:", OUT_JSONL)

    # Load metadata
    meta = pd.read_csv(META_CSV)
    if "arxiv_id" not in meta.columns or "abstract" not in meta.columns:
        raise ValueError("❌ META_CSV must contain columns: arxiv_id, abstract")

    meta["arxiv_id"] = meta["arxiv_id"].astype(str).str.strip()
    meta["abstract"] = meta["abstract"].astype(str)

    # Map: arxiv_id -> abstract
    abs_map = dict(zip(meta["arxiv_id"], meta["abstract"]))

    # Read extracted txt files (each file name is arxiv_id.txt)
    txt_files = sorted([p for p in EXTRACTED_DIR.glob("*.txt")])

    if MAX_PAPERS is not None:
        txt_files = txt_files[:MAX_PAPERS]

    kept = 0
    skipped_no_abstract = 0
    skipped_short_article = 0

    with OUT_JSONL.open("w", encoding="utf-8") as out:
        for idx, txt_path in enumerate(tqdm(txt_files, desc="Building JSONL"), start=1):
            arxiv_id = txt_path.stem.strip()   # 2512.00580v2

            abstract = abs_map.get(arxiv_id)
            if not abstract or len(abstract) < MIN_ABSTRACT_CHARS:
                skipped_no_abstract += 1
                continue

            article = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            if len(article) < MIN_ARTICLE_CHARS:
                skipped_short_article += 1
                continue

            row = {
                "arxiv_id": arxiv_id,
                "filename": f"{arxiv_id}.pdf",
                "article": article,
                "abstract": abstract.strip(),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

            if idx % LOG_EVERY == 0:
                print(
                    f"✅ Processed {idx} | Kept {kept} | "
                    f"No-abstract {skipped_no_abstract} | Too-short {skipped_short_article}"
                )

    print("\n==============================")
    print("✅ DONE: papers.jsonl created")
    print("Kept:", kept)
    print("Skipped (no abstract):", skipped_no_abstract)
    print("Skipped (short article):", skipped_short_article)
    print("Saved to:", OUT_JSONL)
    print("==============================")

if __name__ == "__main__":
    main()
