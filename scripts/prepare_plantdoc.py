"""Download the official PlantDoc repository and produce a validated split report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile

from plantdoc_data import collect_dataset, validate_for_training, write_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "plantdoc" / "raw"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "plantdoc" / "dataset-report.json"
REPOSITORY = "pratikkayal/PlantDoc-Dataset"
USER_AGENT = "FarmIQ-PlantDoc-preparation/1.0"
MAX_ARCHIVE_MEMBERS = 20_000
MAX_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024


def _request(url: str, accept: str = "application/vnd.github+json"):
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    return urlopen(request, timeout=30)


def resolve_revision(revision: str) -> tuple[str, str]:
    """Resolve a branch or tag to an immutable GitHub commit SHA."""
    url = f"https://api.github.com/repos/{REPOSITORY}/commits/{quote(revision, safe='')}"
    try:
        with _request(url) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not resolve PlantDoc revision {revision!r}. Check network access and retry."
        ) from exc
    sha = payload.get("sha")
    if not isinstance(sha, str) or len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha.lower()):
        raise RuntimeError("GitHub returned an invalid PlantDoc revision identifier.")
    return sha.lower(), payload.get("html_url", f"https://github.com/{REPOSITORY}/commit/{sha}")


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    members = archive.infolist()
    if len(members) > MAX_ARCHIVE_MEMBERS or sum(member.file_size for member in members) > MAX_EXTRACTED_BYTES:
        raise RuntimeError("PlantDoc archive exceeded the configured extraction safety limits.")
    for member in archive.infolist():
        path = Path(member.filename.replace("\\", "/"))
        mode = (member.external_attr >> 16) & 0o170000
        if path.is_absolute() or path.drive or ".." in path.parts or mode == stat.S_IFLNK:
            raise RuntimeError(f"Unsafe path in PlantDoc archive: {member.filename!r}")
        target = (root / path).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"Unsafe path in PlantDoc archive: {member.filename!r}")
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)


def download_dataset(data_root: Path = DEFAULT_DATA_ROOT, revision: str = "master") -> dict:
    data_root = Path(data_root).expanduser().resolve()
    if data_root.exists():
        if (data_root / "plantdoc-source.json").is_file():
            current = json.loads((data_root / "plantdoc-source.json").read_text(encoding="utf-8"))
            print(f"PlantDoc already downloaded at {data_root} (commit {current.get('commit', 'unknown')}).")
            return current
        raise FileExistsError(
            f"Dataset destination already exists without FarmIQ source metadata: {data_root}. "
            "Move it aside or choose another --data-root; existing files were left unchanged."
        )

    data_root.parent.mkdir(parents=True, exist_ok=True)
    commit, commit_url = resolve_revision(revision)
    archive_url = f"https://codeload.github.com/{REPOSITORY}/zip/{commit}"
    print(f"Downloading official PlantDoc repository at commit {commit}...")
    archive_hash = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix="farmiq-plantdoc-", dir=data_root.parent) as temp_name:
        temp_root = Path(temp_name)
        archive_path = temp_root / "plantdoc.zip"
        try:
            with _request(archive_url, "application/zip") as response, archive_path.open("wb") as output:
                total = int(response.headers.get("Content-Length", "0") or 0)
                received = 0
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    archive_hash.update(chunk)
                    received += len(chunk)
                    if total:
                        print(f"\rDownloaded {received / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB", end="", flush=True)
                    else:
                        print(f"\rDownloaded {received / 1024 / 1024:.1f} MB", end="", flush=True)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("PlantDoc download failed. Check network access and retry.") from exc
        print()

        extracted = temp_root / "extracted"
        extracted.mkdir()
        try:
            with zipfile.ZipFile(archive_path) as archive:
                _safe_extract(archive, extracted)
        except (zipfile.BadZipFile, OSError) as exc:
            raise RuntimeError("The downloaded PlantDoc archive was invalid or incomplete.") from exc

        candidates = [path for path in extracted.iterdir() if path.is_dir()]
        if len(candidates) != 1:
            raise RuntimeError("PlantDoc archive did not contain the expected single repository directory.")
        repository_root = candidates[0]
        metadata = {
            "repository": f"https://github.com/{REPOSITORY}",
            "revision": revision,
            "commit": commit,
            "commit_url": commit_url,
            "archive_url": archive_url,
            "archive_sha256": archive_hash.hexdigest(),
        }
        (repository_root / "plantdoc-source.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        # Validate layout before moving the download into the permanent data folder.
        collect_dataset(repository_root)
        repository_root.replace(data_root)
    print(f"PlantDoc downloaded to {data_root}; source commit {commit} is recorded.")
    return metadata


def print_summary(report: dict) -> None:
    for split in ("train", "test"):
        summary = report["splits"][split]
        print(f"\n{split.upper()} ({summary['valid_images']} readable images)")
        for label, counts in sorted(summary["class_counts"].items(), key=lambda item: item[0].casefold()):
            print(
                f"  {label}: {counts['images_valid']} valid, "
                f"{counts['images_corrupt']} corrupt, {counts['images_too_large']} oversized, {counts['images_found']} found"
            )
        if summary["corrupt_images"]:
            print(f"  Corrupt/unreadable image paths are recorded in the report.")
        if summary["empty_classes"]:
            print("  Class folders with no readable files: " + ", ".join(summary["empty_classes"]))
    print(f"\nExact duplicate groups: {len(report['duplicate_groups'])}")
    print(f"Across official splits: {len(report['cross_split_duplicate_groups'])}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--download", action="store_true", help="Download the official repository first.")
    parser.add_argument("--revision", default="master", help="Git branch, tag, or commit; resolved and recorded as a SHA.")
    parser.add_argument("--require-trainable", action="store_true", help="Fail if data-leakage checks block training.")
    args = parser.parse_args(argv)

    try:
        if args.download:
            download_dataset(args.data_root, args.revision)
        records, report = collect_dataset(args.data_root)
        write_report(args.report, report)
        print_summary(report)
        if args.require_trainable:
            labels = validate_for_training(records, report)
            report["training_classes"] = labels
            write_report(args.report, report)
        print(f"\nDataset report written to {args.report.resolve()}")
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, OSError) as exc:
        print(f"PlantDoc preparation failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
