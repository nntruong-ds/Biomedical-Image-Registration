from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

from src.utils.io import ensure_dir


OASIS_URL = "https://sites.wustl.edu/oasisbrains/"
OASIS_REQUEST_URL = "https://sites.wustl.edu/oasisbrains/home/oasis-3/request-access/"
DIRLAB_URL = "https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html"


def _existing_files(out_dir: Path) -> list[Path]:
    patterns = ["*.nii", "*.nii.gz", "*.mha", "*.mhd", "*.img", "*.hdr", "*.dcm"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(out_dir.rglob(pattern))
    return sorted(files)


def _download_url(url: str, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    filename = url.rstrip("/").split("/")[-1] or "downloaded_dataset_file"
    target = out_dir / filename
    urllib.request.urlretrieve(url, target)
    return target


def oasis_instructions(out_dir: Path) -> str:
    return f"""
OASIS data were not downloaded automatically.

Reason:
  Official OASIS access is mediated through the OASIS/NITRC-IR request flow and
  requires registration, data-use terms, and staff review before project access.

Manual steps:
  1. Visit {OASIS_URL}
  2. Request access and register with NITRC when prompted.
  3. Download the OASIS-1 or other approved OASIS image files.
  4. Place the downloaded files under: {out_dir}

This helper did not create placeholder images. Re-run it after manual placement to
check that image files are visible.
""".strip()


def dirlab_instructions(out_dir: Path) -> str:
    return f"""
DIR-Lab 4DCT data were not downloaded automatically.

Reason:
  The official DIR-Lab case packets are password protected. Emory asks
  investigators to complete the request form before Dropbox access is provided.

Manual steps:
  1. Visit {DIRLAB_URL}
  2. Complete the Access Request Form.
  3. Use the provided password to download the case packets.
  4. Place extracted case files under: {out_dir}

This helper did not create placeholder images. Re-run it after manual placement to
check that image files are visible.
""".strip()


def handle_dataset(dataset: str, out: str | Path, url: str | None = None) -> None:
    out_dir = ensure_dir(out)
    existing = _existing_files(out_dir)
    if existing:
        print(f"Found {len(existing)} existing image-like files under {out_dir}.")
        for path in existing[:10]:
            print(f"  {path}")
        if len(existing) > 10:
            print("  ...")
        return

    explicit_url = url or os.environ.get(f"{dataset.upper()}_DOWNLOAD_URL")
    if explicit_url:
        target = _download_url(explicit_url, out_dir)
        print(f"Downloaded user-provided {dataset} URL to: {target}")
        return

    if dataset == "oasis":
        print(oasis_instructions(out_dir))
    elif dataset == "dirlab":
        print(dirlab_instructions(out_dir))
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset access helper.")
    parser.add_argument("--dataset", choices=["oasis", "dirlab", "all"], required=True)
    parser.add_argument("--out", default="data", help="Root directory for downloaded data.")
    parser.add_argument(
        "--url",
        default=None,
        help="Optional authorized direct download URL. Without it, manual access instructions are printed.",
    )
    args = parser.parse_args()

    root = Path(args.out)
    if args.dataset == "all":
        handle_dataset("oasis", root / "oasis", url=args.url)
        print()
        handle_dataset("dirlab", root / "dirlab", url=args.url)
    else:
        handle_dataset(args.dataset, root / args.dataset, url=args.url)


if __name__ == "__main__":
    main()

