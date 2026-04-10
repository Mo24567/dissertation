import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/qa_dataset_clean.csv")

OUTPUT_COLUMNS = ["id", "question", "answer", "source", "source_file", "page", "section"]
DEDUP_COLS = ["question", "answer", "source", "page", "section"]


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalise a Q&A DataFrame.

    Strips whitespace, removes empty rows, deduplicates, and regenerates IDs.
    Returns DataFrame with columns in canonical order.
    """
    # Strip whitespace from all object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Remove rows where question or answer is empty
    df = df[df["question"].notna() & (df["question"] != "")]
    df = df[df["answer"].notna() & (df["answer"] != "")]

    # Deduplicate on key columns (use only those present)
    dedup_subset = [c for c in DEDUP_COLS if c in df.columns]
    df = df.drop_duplicates(subset=dedup_subset)

    # Drop any existing id column and regenerate
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df = df.reset_index(drop=True)
    df.insert(0, "id", [f"qa_{i + 1}" for i in range(len(df))])

    # Return only canonical columns in canonical order
    present = [c for c in OUTPUT_COLUMNS if c in df.columns]
    return df[present]


def load_dataset():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in data/raw/")
        return

    dfs = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            if "question" not in df.columns or "answer" not in df.columns:
                print(f"Skipping {csv_path.name}: missing 'question' or 'answer' column")
                continue

            # Auto-add missing metadata columns
            if "source" not in df.columns:
                df["source"] = csv_path.stem
            if "source_file" not in df.columns:
                df["source_file"] = csv_path.name
            if "page" not in df.columns:
                df["page"] = ""
            if "section" not in df.columns:
                df["section"] = ""

            # Fill NaN in metadata columns
            df["source"] = df["source"].fillna(csv_path.stem)
            df["source_file"] = df["source_file"].fillna(csv_path.name)
            df["page"] = df["page"].fillna("")
            df["section"] = df["section"].fillna("")

            dfs.append(df)
            print(f"Loaded {csv_path.name}: {len(df)} rows")
        except Exception as e:
            print(f"Error loading {csv_path.name}: {e}")

    if not dfs:
        print("No valid CSV files loaded.")
        return

    merged = pd.concat(dfs, ignore_index=True)

    # Ensure all metadata columns exist
    for col in ["source", "source_file", "page", "section"]:
        if col not in merged.columns:
            merged[col] = ""

    merged = merged[["question", "answer", "source", "source_file", "page", "section"]]

    cleaned = clean_dataset(merged)
    cleaned = cleaned[OUTPUT_COLUMNS]
    cleaned.to_csv(OUTPUT_PATH, index=False)

    print(f"\nFiles loaded: {len(dfs)}")
    print(f"Final row count: {len(cleaned)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    load_dataset()
