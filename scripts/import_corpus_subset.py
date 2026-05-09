#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime
from pathlib import Path
import shutil
import sys

SELECTED_FILES = [
    "1_Daisy_Miller_Henry_James.txt",
    "2_Ethan_Frome_Edith_Wharton.txt",
    "7_The_Call_of_the_Wild_Jack_London.txt",
    "8_The_Death_of_Ivan_Ilych_Leo_Tolstoy.txt",
    "10_The_Hound_of_the_Baskervilles_Arthur_Conan_Doyle.txt",
    "11_The_Metamorphosis_Franz_Kafka.txt",
    "12_The_Time_Machine_H.G._Wells.txt",
    "13_The_Turn_of_the_Screw_Henry_James.txt",
    "14_The_Yellow_Wallpaper_Charlotte_Perkins_Gilman.txt",
    "15_The_Secret_Garden_Frances_Hodgson_Burnett.txt",
    "16_Candide_Voltaire.txt",
    "17_The_Scarlet_Letter_Nathaniel_Hawthorne.txt",
    "18_Treasure_Island_Robert_Louis_Stevenson.txt",
    "20_The_Awakening_Kate_Chopin.txt",
    "22_A_Christmas_Carol_Charles_Dickens.txt",
    "25_Mrs._Dalloway_Virginia_Woolf.txt",
]

CSV_FIELDS = ["source_path", "target_path", "status", "size_bytes", "message"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy a selected or full corpus subset from corpus/ into data/raw/ without overwriting unless forced."
    )
    parser.add_argument(
        "--source-dir",
        default="corpus",
        help="Source directory containing corpus text files (default: corpus)",
    )
    parser.add_argument(
        "--target-dir",
        default="data/raw",
        help="Target directory for raw corpus imports (default: data/raw)",
    )
    parser.add_argument(
        "--selection",
        choices=["selected", "all"],
        default="selected",
        help="Import mode: selected (default) or all corpus .txt files except *_clean.txt.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing target files when copying.",
    )
    parser.add_argument(
        "--manifest-dir",
        default="outputs/corpus_import",
        help="Directory where the import manifest will be written.",
    )
    return parser


def selected_sources(source_dir: Path) -> list[Path]:
    return [source_dir / name for name in SELECTED_FILES]


def all_sources(source_dir: Path) -> list[Path]:
    return sorted(
        [path for path in source_dir.glob("*.txt") if not path.name.endswith("_clean.txt")]
    )


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(source_path: Path, target_path: Path, force: bool) -> tuple[str, int, str]:
    if not source_path.exists():
        return "missing_source", 0, "Source file is missing"
    if not source_path.is_file():
        return "missing_source", 0, "Source path is not a file"
    ensure_directory(target_path.parent)
    if target_path.exists():
        if force:
            shutil.copy2(source_path, target_path)
            return "overwritten", source_path.stat().st_size, "Existing target overwritten"
        return "skipped_exists", source_path.stat().st_size, "Target exists, use --force to overwrite"
    shutil.copy2(source_path, target_path)
    return "copied", source_path.stat().st_size, "File copied"


def write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    ensure_directory(manifest_path.parent)
    with manifest_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    manifest_base = Path(args.manifest_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = manifest_base / timestamp / "import_manifest.csv"

    if args.selection == "selected":
        sources = selected_sources(source_dir)
    else:
        sources = all_sources(source_dir)

    rows = []
    counts = {"copied": 0, "skipped_exists": 0, "missing_source": 0, "overwritten": 0}

    for source_path in sources:
        if source_path.name.endswith("_clean.txt"):
            continue
        target_path = target_dir / source_path.name
        status, size_bytes, message = copy_file(source_path, target_path, args.force)
        counts[status] = counts.get(status, 0) + 1
        rows.append(
            {
                "source_path": str(source_path.resolve()),
                "target_path": str(target_path.resolve()),
                "status": status,
                "size_bytes": str(size_bytes),
                "message": message,
            }
        )

    write_manifest(manifest_path, rows)

    print(f"copied {counts['copied']}")
    print(f"skipped_exists {counts['skipped_exists']}")
    print(f"missing_source {counts['missing_source']}")
    print(f"overwritten {counts['overwritten']}")
    print(f"manifest path {manifest_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
