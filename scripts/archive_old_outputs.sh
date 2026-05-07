#!/usr/bin/env bash
# archive_old_outputs.sh
#
# Safely moves legacy generated output directories from data/ into
# archive/old_outputs/<timestamp>/.
#
# Usage:
#   bash scripts/archive_old_outputs.sh --dry-run   # preview only
#   bash scripts/archive_old_outputs.sh             # actually move
#
# Safe inputs that are NEVER touched:
#   data/raw/
#   data/aliases/
#   data/booknlp_models/
#
# Candidate generated outputs (moved if they exist):
#   data/books
#   data/booknlp_chapter_output
#   data/reports
#   data/ml
#   data/graphs
#   data/evaluation_logs
#   data/evaluation_runs
#   data/evaluation_bundle
#   data/evaluation_bundle.zip
#   data/booknlp_output
#   data/booknlp_only_output
#   results_christmas_carol

set -euo pipefail

DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Usage: $0 [--dry-run]" >&2
            exit 1
            ;;
    esac
done

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_DIR="archive/old_outputs/${TIMESTAMP}"

CANDIDATES=(
    "data/books"
    "data/booknlp_chapter_output"
    "data/reports"
    "data/ml"
    "data/graphs"
    "data/evaluation_logs"
    "data/evaluation_runs"
    "data/evaluation_bundle"
    "data/evaluation_bundle.zip"
    "data/booknlp_output"
    "data/booknlp_only_output"
    "results_christmas_carol"
)

if $DRY_RUN; then
    echo "[DRY RUN] The following would be moved to ${ARCHIVE_DIR}/"
    echo ""
fi

FOUND=0
for candidate in "${CANDIDATES[@]}"; do
    if [ -e "$candidate" ]; then
        FOUND=$((FOUND + 1))
        if $DRY_RUN; then
            echo "  WOULD MOVE: $candidate -> ${ARCHIVE_DIR}/$(basename "$candidate")"
        else
            mkdir -p "$ARCHIVE_DIR"
            mv "$candidate" "${ARCHIVE_DIR}/$(basename "$candidate")"
            echo "  MOVED: $candidate -> ${ARCHIVE_DIR}/$(basename "$candidate")"
        fi
    fi
done

if [ "$FOUND" -eq 0 ]; then
    echo "No candidate directories found. Nothing to archive."
    exit 0
fi

if $DRY_RUN; then
    echo ""
    echo "[DRY RUN] $FOUND item(s) would be moved."
    echo "Re-run without --dry-run to perform the archive."
else
    echo ""
    echo "Archived $FOUND item(s) to ${ARCHIVE_DIR}/"
    echo "Nothing was deleted. To undo, move items back from ${ARCHIVE_DIR}/."
fi
