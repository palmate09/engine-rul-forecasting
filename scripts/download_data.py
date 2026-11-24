"""Download and extract the NASA C-MAPSS turbofan dataset.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --force
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEST = PROJECT_ROOT / "data" / "raw" / "CMAPSS"
URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def expected_files() -> list[Path]:
    files: list[Path] = []
    for subset in SUBSETS:
        files.extend(
            [
                DEST / f"train_{subset}.txt",
                DEST / f"test_{subset}.txt",
                DEST / f"RUL_{subset}.txt",
            ]
        )
    return files


def already_present() -> bool:
    return all(path.exists() for path in expected_files())


def download(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as response:
        with dest.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def extract_outer_and_inner(outer_zip: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cmapss_") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(outer_zip) as archive:
            archive.extractall(tmp_path)

        inner_zips = list(tmp_path.rglob("CMAPSSData.zip"))
        if not inner_zips:
            raise FileNotFoundError("Could not find the inner CMAPSSData.zip archive.")
        with zipfile.ZipFile(inner_zips[0]) as archive:
            archive.extractall(dest)

    print(f"Extracted C-MAPSS files to {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the extracted files already exist.",
    )
    args = parser.parse_args()

    if already_present() and not args.force:
        print(f"C-MAPSS files already present in {DEST}")
        return

    if args.force and DEST.exists():
        shutil.rmtree(DEST)

    archive = DEST / "CMAPSSData.zip"
    download(archive)
    extract_outer_and_inner(archive, DEST)

    missing = [str(p) for p in expected_files() if not p.exists()]
    if missing:
        raise RuntimeError("Extraction incomplete:\n  " + "\n  ".join(missing))

    print("Done. All 12 train/test/RUL files are present.")
    print(
        "Citation: A. Saxena and K. Goebel (2008), "
        "'Turbofan Engine Degradation Simulation Data Set', "
        "NASA Prognostics Data Repository, NASA Ames Research Center."
    )


if __name__ == "__main__":
    main()
