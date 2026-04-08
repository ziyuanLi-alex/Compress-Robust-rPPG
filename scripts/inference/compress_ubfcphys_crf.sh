#!/bin/bash

# Compress UBFC-Phys videos with H.264 at a specified CRF level.
# Usage: bash scripts/inference/compress_ubfcphys_crf.sh <CRF> [SRC_DIR] [DST_DIR]
#
# Example:
#   bash scripts/inference/compress_ubfcphys_crf.sh 14 /mnt/k/RawData /mnt/k/UBFC-Phys-CRF14
#
# Output structure mirrors the source:
#   DST_DIR/s1/vid_s1_T1.mp4  (compressed video)
#   DST_DIR/s1/bvp_s1_T1.csv  (copied as-is)

set -euo pipefail

CRF="${1:?Usage: $0 <CRF> [SRC_DIR] [DST_DIR]}"
SRC_DIR="${2:-/mnt/k/RawData}"
DST_DIR="${3:-/mnt/k/UBFC-Phys-CRF${CRF}}"

echo "=== UBFC-Phys H.264 Compression ==="
echo "CRF:       $CRF"
echo "Source:    $SRC_DIR"
echo "Dest:      $DST_DIR"
echo ""

mkdir -p "$DST_DIR"

for SUBJECT_DIR in "$SRC_DIR"/s*/; do
    SUBJECT=$(basename "$SUBJECT_DIR")
    mkdir -p "$DST_DIR/$SUBJECT"

    for AVI_FILE in "$SUBJECT_DIR"/vid_*.avi; do
        [ -f "$AVI_FILE" ] || continue
        BASENAME=$(basename "$AVI_FILE" .avi)
        MP4_FILE="$DST_DIR/$SUBJECT/${BASENAME}.mp4"
        BVP_FILE="$SUBJECT_DIR/${BASENAME//vid/bvp}.csv"
        BVP_DST="$DST_DIR/$SUBJECT/${BASENAME//vid/bvp}.csv"

        if [ -f "$MP4_FILE" ]; then
            echo "SKIP (exists): $MP4_FILE"
        else
            echo "Compressing: $AVI_FILE -> $MP4_FILE (CRF=$CRF)"
            ffmpeg -y -i "$AVI_FILE" -c:v libx264 -crf "$CRF" -pix_fmt yuv420p -an "$MP4_FILE" -loglevel error
        fi

        # Copy BVP label if not already present
        if [ -f "$BVP_FILE" ] && [ ! -f "$BVP_DST" ]; then
            cp "$BVP_FILE" "$BVP_DST"
        fi
    done
done

echo ""
echo "Done. Compressed dataset at: $DST_DIR"
