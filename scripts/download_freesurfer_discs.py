from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from extract_freesurfer_archive import extract_selected


BASE_URL = "https://download.nrg.wustl.edu/data/oasis_cs_freesurfer_disc{disc}.tar.gz"


def _has_extracted_disc(subjects_dir: Path, disc: int) -> bool:
    disc_dir = subjects_dir / f"disc{disc}"
    return disc_dir.exists() and any(disc_dir.glob("OAS*_MR*/mri/T1.mgz"))


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-L", "--fail", "-C", "-", "-o", str(out_path), url]
    subprocess.run(cmd, check=True)


def download_extract_discs(
    discs: list[int],
    raw_dir: str | Path,
    subjects_dir: str | Path,
    delete_archive: bool,
    skip_extracted: bool,
) -> None:
    raw = Path(raw_dir)
    subjects = Path(subjects_dir)
    raw.mkdir(parents=True, exist_ok=True)
    subjects.mkdir(parents=True, exist_ok=True)

    for disc in discs:
        if skip_extracted and _has_extracted_disc(subjects, disc):
            print(f"disc{disc}: already extracted, skipping download")
            continue
        url = BASE_URL.format(disc=disc)
        archive = raw / f"oasis_cs_freesurfer_disc{disc}.tar.gz"
        print(f"disc{disc}: downloading {url}")
        _download(url, archive)
        print(f"disc{disc}: extracting selected MRI/label files")
        summary = extract_selected(archive, subjects, delete_archive=delete_archive)
        print(
            f"disc{disc}: extracted {summary['files_extracted']} files for "
            f"{summary['num_subjects']} subjects"
        )
        if delete_archive:
            print(f"disc{disc}: deleted {archive}")


def _parse_discs(values: list[str] | None, start: int, end: int) -> list[int]:
    if values:
        discs: list[int] = []
        for value in values:
            if "-" in value:
                lo, hi = value.split("-", 1)
                discs.extend(range(int(lo), int(hi) + 1))
            else:
                discs.append(int(value))
        return sorted(dict.fromkeys(discs))
    return list(range(start, end + 1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download OASIS-1 FreeSurfer discs, extract only registration-relevant "
            "MRI/label files, and optionally delete each tar.gz after extraction."
        )
    )
    parser.add_argument("--raw-dir", default="data/oasis/freesurfer/raw")
    parser.add_argument("--subjects-dir", default="data/oasis/freesurfer/subjects")
    parser.add_argument("--disc", action="append", help="Disc number or range, e.g. 2 or 2-11.")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=11)
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--no-skip-extracted", action="store_true")
    args = parser.parse_args()

    discs = _parse_discs(args.disc, args.start, args.end)
    try:
        download_extract_discs(
            discs=discs,
            raw_dir=args.raw_dir,
            subjects_dir=args.subjects_dir,
            delete_archive=not args.keep_archives,
            skip_extracted=not args.no_skip_extracted,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Download command failed with exit code {exc.returncode}") from exc
    except Exception as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
