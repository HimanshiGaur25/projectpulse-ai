from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PROJECT_ROOT / "ml" / "data" / "splits"
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"

# Input files and their output names
SPLITS = {
    "train": "train.tsv",
    "validation": "dev.tsv",
    "test": "test.tsv",
}


def load_emotions():
    """Read emotion names in their official ID order."""
    emotions_file = SPLITS_DIR / "emotions.txt"

    with open(emotions_file, "r", encoding="utf-8") as file:
        emotions = [line.strip() for line in file if line.strip()]

    return emotions


def convert_label_ids(label_ids, emotions):
    """Convert numeric label IDs into emotion names."""
    ids = str(label_ids).strip().split(",")

    labels = []

    for label_id in ids:
        label_id = int(label_id.strip())

        if label_id < 0 or label_id >= len(emotions):
            raise ValueError(f"Invalid emotion ID: {label_id}")

        labels.append(emotions[label_id])

    return "|".join(labels)


def prepare_split(split_name, filename, emotions):
    """Convert one TSV split into a clean CSV."""

    input_path = SPLITS_DIR / filename
    output_path = PROCESSED_DIR / f"{split_name}.csv"

    print(f"\nPreparing {split_name} dataset...")

    # Official TSV format: text, label IDs, comment ID.
    df = pd.read_csv(
        input_path,
        sep="\t",
        header=None,
        names=["text", "label_ids", "comment_id"],
        dtype=str,
        keep_default_na=False,
    )

    # Check that the expected columns were loaded.
    if df.shape[1] != 3:
        raise ValueError(
            f"Expected 3 columns in {filename}, "
            f"but found {df.shape[1]}."
        )

    # Remove rows with empty text or labels.
    df["text"] = df["text"].str.strip()
    df["label_ids"] = df["label_ids"].str.strip()

    df = df[
        (df["text"] != "")
        & (df["label_ids"] != "")
    ].copy()

    # Convert IDs into readable labels.
    df["labels"] = df["label_ids"].apply(
        lambda value: convert_label_ids(value, emotions)
    )

    # Keep the original IDs for traceability.
    df = df[
        ["text", "labels", "comment_id"]
    ]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Rows saved: {len(df):,}")
    print(f"Output: {output_path}")

    print("\nLabel counts:")
    print(
        df["labels"]
        .value_counts()
        .head(10)
        .to_string()
    )

    return df


def main():
    print("ProjectPulse AI — Data Preparation")

    emotions = load_emotions()

    print(f"\nEmotion labels loaded: {len(emotions)}")
    print(emotions)

    for split_name, filename in SPLITS.items():
        prepare_split(split_name, filename, emotions)

    print("\nData preparation complete!")
    print(f"Processed files are in: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()