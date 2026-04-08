#!/bin/bash

# Delete UBFC-Phys videos (and their BVP labels) that are in the exclusion list
# for the s1-s14 subset. Run this BEFORE preprocessing.
#
# Usage: bash scripts/inference/ubfcphys_remove_excluded.sh [DATA_DIR]
#
# After deletion, preprocessing will skip these videos automatically.
# Keep USE_EXCLUSION_LIST: True in the config as a safety net.

set -euo pipefail

DATA_DIR="${1:-/mnt/k/RawData}"

EXCLUDED=(
    # T1
    "s3_T1" "s8_T1" "s9_T1"
    # T2
    "s1_T2" "s4_T2" "s6_T2" "s8_T2" "s9_T2" "s11_T2" "s12_T2" "s13_T2" "s14_T2"
    # T3
    "s5_T3" "s8_T3" "s9_T3" "s10_T3" "s13_T3" "s14_T3"
)

echo "=== UBFC-Phys Exclude Videos ==="
echo "Data dir: $DATA_DIR"
echo "Excluding ${#EXCLUDED[@]} subject-task combinations"
echo ""

deleted=0
missing=0

for ENTRY in "${EXCLUDED[@]}"; do
    # Extract subject and task: s1_T2 -> s1, T2
    SUBJECT="${ENTRY%%_*}"
    TASK="${ENTRY##*_}"

    VID_FILE="$DATA_DIR/$SUBJECT/vid_${ENTRY}.avi"
    BVP_FILE="$DATA_DIR/$SUBJECT/bvp_${ENTRY}.csv"

    if [ -f "$VID_FILE" ]; then
        echo "DELETE: $VID_FILE ($(du -h "$VID_FILE" | cut -f1))"
        rm "$VID_FILE"
        deleted=$((deleted + 1))
    else
        echo "SKIP (not found): $VID_FILE"
        missing=$((missing + 1))
    fi

    if [ -f "$BVP_FILE" ]; then
        echo "DELETE: $BVP_FILE"
        rm "$BVP_FILE"
    fi
done

echo ""
echo "Done. Deleted: $deleted, Not found: $missing"

# Show remaining videos per subject
echo ""
echo "Remaining videos per subject:"
for SUBJECT_DIR in "$DATA_DIR"/s*/; do
    SUBJECT=$(basename "$SUBJECT_DIR")
    COUNT=$(find "$SUBJECT_DIR" -name "vid_*.avi" | wc -l)
    echo "  $SUBJECT: $COUNT videos"
done
