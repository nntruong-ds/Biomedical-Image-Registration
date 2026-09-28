from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def extract_discs(raw_dir: str | Path, out_dir: str | Path, overwrite: bool = False) -> list[Path]:
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    archives = sorted(raw_path.glob("oasis_cross-sectional_disc*.tar.gz"))
    if not archives:
        raise FileNotFoundError(f"No OASIS raw disc archives found under {raw_path}")

    extracted: list[Path] = []
    for archive in archives:
        disc_name = archive.name.removeprefix("oasis_cross-sectional_").removesuffix(
            ".tar.gz"
        )
        target = out_path / disc_name
        if target.exists() and any(target.iterdir()) and not overwrite:
            print(f"Skipping {archive.name}; {target} already exists")
            extracted.append(target)
            continue
        print(f"Extracting {archive.name} -> {out_path}")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(out_path)
        extracted.append(target)
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract downloaded OASIS-1 raw discs.")
    parser.add_argument("--raw", default="data/oasis/raw")
    parser.add_argument("--out", default="data/oasis")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    extracted = extract_discs(args.raw, args.out, overwrite=args.overwrite)
    print(f"Available extracted disc directories: {len(extracted)}")
    for path in extracted:
        print(path)


if __name__ == "__main__":
    main()

