#!/bin/bash
# Train all 9 translators in 3 parallel groups of 3
# Sensors in FL data: LeftWrist=0, RightAnkle=1, RightThigh=2

DATA_DIR="/mnt/storage/hitl_experiments/paaws_fl_tuned/DS_10"
LAB_DIR="/mnt/storage/hitl_experiments/paaws_tuned"
SUFFIX="_LeftWrist_RightAnkle_RightThigh"
OUT_BASE="output/translators"
SCRIPT="scripts/signal_translator.py"

mkdir -p \
  $OUT_BASE/wrist_to_ankle $OUT_BASE/wrist_to_thigh $OUT_BASE/ankle_to_wrist \
  $OUT_BASE/ankle_to_thigh $OUT_BASE/thigh_to_wrist $OUT_BASE/thigh_to_ankle \
  $OUT_BASE/wrist_ankle_to_thigh $OUT_BASE/wrist_thigh_to_ankle $OUT_BASE/ankle_thigh_to_wrist

COMMON="--data-dir $DATA_DIR --suffix $SUFFIX --lab-data-dir $LAB_DIR \
        --lab-participant DS_11 --epochs 50 --batch-size 512 \
        --max-train 300000 --patience 20 --viz-every 20"

run() {
    local name=$1; shift
    echo "[START] $name"
    python $SCRIPT $COMMON "$@" --out-dir $OUT_BASE/$name \
        > $OUT_BASE/$name/train.log 2>&1
    echo "[DONE]  $name (exit $?)"
}

# ── Group 1 ───────────────────────────────────────────────────────────────────
echo "=== Group 1/3 ==="
run wrist_to_ankle  --known-streams 0 --target-stream 1 \
    --initial-sensors LeftWrist --target-sensor RightAnkle --viz-activity Walking &
run wrist_to_thigh  --known-streams 0 --target-stream 2 \
    --initial-sensors LeftWrist --target-sensor RightThigh --viz-activity Walking &
run ankle_to_wrist  --known-streams 1 --target-stream 0 \
    --initial-sensors RightAnkle --target-sensor LeftWrist --viz-activity Walking &
wait
echo "=== Group 1 done ==="

# ── Group 2 ───────────────────────────────────────────────────────────────────
echo "=== Group 2/3 ==="
run ankle_to_thigh  --known-streams 1 --target-stream 2 \
    --initial-sensors RightAnkle --target-sensor RightThigh --viz-activity Walking &
run thigh_to_wrist  --known-streams 2 --target-stream 0 \
    --initial-sensors RightThigh --target-sensor LeftWrist --viz-activity Walking &
run thigh_to_ankle  --known-streams 2 --target-stream 1 \
    --initial-sensors RightThigh --target-sensor RightAnkle --viz-activity Walking &
wait
echo "=== Group 2 done ==="

# ── Group 3 ───────────────────────────────────────────────────────────────────
echo "=== Group 3/3 ==="
run wrist_ankle_to_thigh  --known-streams 0 1 --target-stream 2 \
    --initial-sensors LeftWrist RightAnkle --target-sensor RightThigh --viz-activity Walking &
run wrist_thigh_to_ankle  --known-streams 0 2 --target-stream 1 \
    --initial-sensors LeftWrist RightThigh --target-sensor RightAnkle --viz-activity Walking &
run ankle_thigh_to_wrist  --known-streams 1 2 --target-stream 0 \
    --initial-sensors RightAnkle RightThigh --target-sensor LeftWrist --viz-activity Walking &
wait
echo "=== Group 3 done ==="

echo "=== All 9 translators done ==="
echo "Check logs: tail $OUT_BASE/*/train.log"