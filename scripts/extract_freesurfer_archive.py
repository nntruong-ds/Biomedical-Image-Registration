from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path, PurePosixPath

MRI_FILES = {
    "T1.mgz",
    "norm.mgz",
    "brain.mgz",
    "brainmask.mgz",
    "aseg.mgz",
    "aparc+aseg.mgz",
}


def _is_safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def _is_selected_member(name: str) -> bool:
    if not _is_safe_member(name):
        return False
    parts = PurePosixPath(name).parts
    if len(parts) < 4:
        return False
    if not parts[0].startswith("disc") or not parts[1].startswith("OAS"):
        return False
    if parts[2] == "mri" and len(parts) == 4 and parts[3] in MRI_FILES:
        return True
    if parts[2] == "label" and len(parts) == 4 and parts[3].endswith(".label"):
        return True
    return False


def extract_selected(
    archive_path: str | Path,
    out_dir: str | Path,
    delete_archive: bool = False,
) -> dict[str, object]:
    archive = Path(archive_path)
    if not archive.exists():
        raise FileNotFoundError(f"FreeSurfer archive not found: {archive}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    selected = 0
    subjects: set[str] = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not _is_selected_member(member.name):
                continue
            target = out / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            with source, target.open("wb") as file:
                file.write(source.read())
            parts = PurePosixPath(member.name).parts
            subjects.add(f"{parts[0]}/{parts[1]}")
            selected += 1

    if selected == 0:
        raise RuntimeError(
            f"No expected FreeSurfer MRI/label files were extracted from {archive}."
        )

    if delete_archive:
        archive.unlink()

    summary = {
        "archive": str(archive),
        "out_dir": str(out),
        "files_extracted": selected,
        "subjects": sorted(subjects),
        "num_subjects": len(subjects),
        "archive_deleted": delete_archive,
    }
    log_dir = out / "extraction_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{archive.name}.json"
    with log_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    summary["log"] = str(log_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the FreeSurfer MRI volumes and labels needed by this repository. "
            "The full archive is not unpacked."
        )
    )
    parser.add_argument("--archive", required=True, help="Path to an OASIS FreeSurfer tar.gz.")
    parser.add_argument(
        "--out",
        default="data/oasis/freesurfer/subjects",
        help="Directory for extracted FreeSurfer subject files.",
    )
    parser.add_argument(
        "--delete-archive",
        action="store_true",
        help="Delete the tar.gz after successful extraction.",
    )
    args = parser.parse_args()

    summary = extract_selected(args.archive, args.out, args.delete_archive)
    print(
        "Extracted "
        f"{summary['files_extracted']} files for {summary['num_subjects']} subjects "
        f"from {summary['archive']}"
    )
    if summary["archive_deleted"]:
        print(f"Deleted archive: {summary['archive']}")


if __name__ == "__main__":
    main()
