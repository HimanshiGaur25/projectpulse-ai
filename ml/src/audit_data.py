from pathlib import Path

import pandas as pd


# ProjectPulseAI/ml/src/audit_data.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "ml" / "data" / "raw"

FILES = [
    "goemotions_1.csv",
    "goemotions_2.csv",
    "goemotions_3.csv",
]

# Metadata columns are not emotion labels.
METADATA_COLUMNS = {
    "text",
    "id",
    "author",
    "subreddit",
    "link_id",
    "parent_id",
    "created_utc",
    "rater_id",
    "example_very_unclear",
}


def audit_file(file_path: Path) -> pd.DataFrame:
    """Load one raw CSV and print a basic data audit."""

    print("\n" + "=" * 65)
    print(f"FILE: {file_path.name}")
    print("=" * 65)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Run download_data.py first."
        )

    df = pd.read_csv(file_path)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Unique comment IDs: {df['id'].nunique():,}")
    print(f"Repeated annotation rows: {df['id'].duplicated().sum():,}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values in important columns:")
    for column in ["text", "id", "rater_id"]:
        if column in df.columns:
            print(f"  {column}: {df[column].isna().sum():,}")

    emotion_columns = [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS
    ]

    print(f"\nEmotion label columns: {len(emotion_columns)}")
    print(emotion_columns)

    if emotion_columns:
        print("\nEmotion label counts:")
        label_counts = df[emotion_columns].sum().sort_values(
            ascending=False
        )
        print(label_counts.to_string())

    if "example_very_unclear" in df.columns:
        print("\nUnclear annotation values:")
        print(df["example_very_unclear"].value_counts(dropna=False))

    print("\nExample comments:")
    print(df["text"].head(3).to_string(index=False))

    return df


def main():
    dataframes = []

    for filename in FILES:
        path = RAW_DATA_DIR / filename
        df = audit_file(path)
        dataframes.append(df)

    combined = pd.concat(dataframes, ignore_index=True)

    print("\n" + "=" * 65)
    print("COMBINED DATASET SUMMARY")
    print("=" * 65)
    print(f"Total annotation rows: {len(combined):,}")
    print(f"Unique comment IDs: {combined['id'].nunique():,}")
    print(
        "Comments with multiple annotations: "
        f"{combined['id'].duplicated().sum():,} repeated rows"
    )

    print("\nAudit complete. No source files were modified.")


if __name__ == "__main__":
    main()