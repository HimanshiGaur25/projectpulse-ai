from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError


# Project root: ProjectPulseAI/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Where original datasets will be stored
RAW_DATA_DIR = PROJECT_ROOT / "ml" / "data" / "raw"

BASE_URL = (
    "https://storage.googleapis.com/"
    "gresearch/goemotions/data/full_dataset/"
)

FILES = [
    "goemotions_1.csv",
    "goemotions_2.csv",
    "goemotions_3.csv",
]


def download_file(filename: str) -> None:
    """Download a dataset file if it doesn't already exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    destination = RAW_DATA_DIR / filename
    url = BASE_URL + filename

    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already downloaded: {filename}")
        return

    print(f"Downloading: {filename}")

    try:
        urlretrieve(url, destination)

        if destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
            raise ValueError("Downloaded file is empty.")

        print(f"Saved: {destination}")

    except (HTTPError, URLError, OSError, ValueError) as error:
        destination.unlink(missing_ok=True)
        print(f"Failed to download {filename}: {error}")
        raise


def main() -> None:
    print("ProjectPulse AI — Dataset Download")
    print(f"Destination: {RAW_DATA_DIR}\n")

    for filename in FILES:
        download_file(filename)

    print("\nAll dataset files are ready!")


if __name__ == "__main__":
    main()