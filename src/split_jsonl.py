# src/split_jsonl.py
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_JSONL = PROJECT_ROOT / "data" / "processed" / "dataset" / "papers.jsonl"
OUT_DIR  = PROJECT_ROOT / "data" / "processed" / "dataset"

TRAIN_OUT = OUT_DIR / "pdf_train.jsonl"
VAL_OUT   = OUT_DIR / "pdf_val.jsonl"

VAL_RATIO = 0.10
SEED = 42

def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def save_jsonl(rows, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    if not IN_JSONL.exists():
        raise FileNotFoundError(f"Missing: {IN_JSONL}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(IN_JSONL)
    train_rows, val_rows = train_test_split(rows, test_size=VAL_RATIO, random_state=SEED)

    save_jsonl(train_rows, TRAIN_OUT)
    save_jsonl(val_rows, VAL_OUT)

    print("✅ Total:", len(rows))
    print("✅ Train:", len(train_rows), "->", TRAIN_OUT)
    print("✅ Val  :", len(val_rows),   "->", VAL_OUT)

if __name__ == "__main__":
    main()
