#!/bin/bash

# Re-compress UBFC-Phys videos from CRF0 source at a specified CRF level.
# Usage: bash scripts/inference/compress_ubfcphys_from_crf0.sh <CRF> [SRC_DIR] [DST_DIR]
#
# Example:
#   bash scripts/inference/compress_ubfcphys_from_crf0.sh 14
#   bash scripts/inference/compress_ubfcphys_from_crf0.sh 14 /mnt/h/lib/UBFC-Phys-CRF0 /mnt/h/lib/UBFC-Phys-CRF14
#
# Input structure (CRF0):
#   SRC_DIR/s1/vid_s1_T1.mp4  (H.264 CRF0 compressed)
#   SRC_DIR/s1/bvp_s1_T1.csv  (BVP labels)
#
# Output structure mirrors the source:
#   DST_DIR/s1/vid_s1_T1.mp4  (re-compressed video)
#   DST_DIR/s1/bvp_s1_T1.csv  (copied from CRF0)

set -euo pipefail

CRF="${1:?Usage: $0 <CRF> [SRC_DIR] [DST_DIR]}"
SRC_DIR="${2:-/mnt/h/lib/UBFC-Phys-CRF0}"
DST_DIR="${3:-/mnt/h/lib/UBFC-Phys-CRF${CRF}}"

echo "=== UBFC-Phys H.264 Re-Compression (from CRF0) ==="
echo "CRF:       $CRF"
echo "Source:    $SRC_DIR"
echo "Dest:      $DST_DIR"
echo ""

mkdir -p "$DST_DIR"

for SUBJECT_DIR in "$SRC_DIR"/s*/; do
    SUBJECT=$(basename "$SUBJECT_DIR")
    mkdir -p "$DST_DIR/$SUBJECT"

    for MP4_FILE in "$SUBJECT_DIR"/vid_*.mp4; do
        [ -f "$MP4_FILE" ] || continue
        BASENAME=$(basename "$MP4_FILE" .mp4)
        OUT_FILE="$DST_DIR/$SUBJECT/${BASENAME}.mp4"
        BVP_FILE="$SUBJECT_DIR/${BASENAME//vid/bvp}.csv"
        BVP_DST="$DST_DIR/$SUBJECT/${BASENAME//vid/bvp}.csv"

        if [ -f "$OUT_FILE" ]; then
            echo "SKIP (exists): $OUT_FILE"
        else
            echo "Re-compressing: $MP4_FILE -> $OUT_FILE (CRF=$CRF)"
            ffmpeg -y -i "$MP4_FILE" -c:v libx264 -crf "$CRF" -pix_fmt yuv420p -an "$OUT_FILE" -loglevel error
        fi

        # Copy BVP label if not already present
        if [ -f "$BVP_FILE" ] && [ ! -f "$BVP_DST" ]; then
            cp "$BVP_FILE" "$BVP_DST"
        fi
    done
done

echo ""
echo "Done. Re-compressed dataset at: $DST_DIR"
