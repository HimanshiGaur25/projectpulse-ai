from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PROJECT_ROOT / "ml" / "data" / "splits"

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "google-research/google-research/master/"
    "goemotions/data/"
)

FILES = [
    "train.tsv",
    "dev.tsv",
    "test.tsv",
    "emotions.txt",
]


def download_file(filename: str) -> None:
    destination = SPLITS_DIR / filename
    url = BASE_URL + filename

    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already exists: {filename}")
        return

    print(f"Downloading {filename}...")

    try:
        urlretrieve(url, destination)

        if destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
            raise ValueError("Downloaded file is empty.")

        print(f"Saved: {destination}")

    except (HTTPError, URLError, OSError, ValueError) as error:
        destination.unlink(missing_ok=True)
        print(f"Failed: {filename} — {error}")
        raise


def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        download_file(filename)

    print("\nOfficial GoEmotions splits are ready.")


if __name__ == "__main__":
    main()