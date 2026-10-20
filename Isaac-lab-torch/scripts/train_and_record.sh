#!/usr/bin/env bash
# =============================================================================
# train_and_record.sh — train until N_CKPT checkpoints exist, then make an MP4 (and a diagnosis)
# of each checkpoint, plus checkpoint 0 (the starting base policy) for comparison.
#
# Default (fast mode): 4096 envs x 30 steps per iteration (same samples per iteration as the original
# 200 x 600), checkpoints at 2000, 4000, 6000. Run it inside tmux:
#   tmux new -s train
#   bash ~/rma/Isaac-lab-torch/scripts/train_and_record.sh
#   (detach: Ctrl-b d, re-attach: tmux attach -t train)
#
# Settings (environment variables):
#   EXPTID=1 NUM_ENVS=4096 ROLLOUT_STEPS=30 SAVE_EVERY=2000 N_CKPT=3 DIAGNOSE=1
#   original RaiSim-style iterations (full reset every iteration): NUM_ENVS=200 ROLLOUT_STEPS=600
# If the run stops (crash, instance stop), run the same command again: it resumes from the
# newest full_<iter>.pt in the run directory (work after that checkpoint is lost).
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT_DIR="$(dirname "$SCRIPT_DIR")"
PIPELINE_DIR="${PIPELINE_DIR:-$HOME/nv-omni-tr-pipeline}"
EXPTID="${EXPTID:-1}"
NUM_ENVS="${NUM_ENVS:-4096}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-30}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
N_CKPT="${N_CKPT:-3}"
DIAGNOSE="${DIAGNOSE:-1}"
MAX_ITERS=$((SAVE_EVERY * N_CKPT))
RUN_DIR="$PORT_DIR/data/rsg_a1_task/$(printf '%04d' "$EXPTID")"
LOG_DIR="$HOME/rma_logs"
mkdir -p "$LOG_DIR"

if [ "${CONDA_DEFAULT_ENV:-}" != "isaac" ] && [ -x "$HOME/miniconda3/bin/conda" ]; then
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)" && conda activate isaac
fi
python -c "import isaaclab" 2>/dev/null || { echo "[FAIL] isaaclab not importable; run: conda activate isaac"; exit 1; }
[ "${OMNI_KIT_ACCEPT_EULA:-}" = "YES" ] || { echo "[FAIL] export OMNI_KIT_ACCEPT_EULA=YES first"; exit 1; }

# ------------------------------------------------------------------ 1. train (or resume)
last=$(ls "$RUN_DIR"/full_*.pt 2>/dev/null | sed 's/.*full_\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
if [ -n "$last" ] && [ "$last" -ge "$MAX_ITERS" ]; then
    echo ">>> checkpoint $last already exists, skipping training"
else
    if [ -n "$last" ] && [ "$last" -gt 0 ]; then
        echo ">>> resuming run $EXPTID from iteration $last"
        resume=(--loadid "$last" --overwrite)
    else
        [ -d "$RUN_DIR" ] && echo ">>> $RUN_DIR exists without a trained checkpoint, overwriting"
        resume=(--overwrite)
    fi
    echo ">>> training to iteration $MAX_ITERS with $NUM_ENVS envs x $ROLLOUT_STEPS steps, checkpoint every $SAVE_EVERY"
    echo ">>> log: $LOG_DIR/train_$EXPTID.log  (time per iteration: grep 'time elapsed' in it)"
    python -u "$PORT_DIR/runner.py" --exptid "$EXPTID" --num_envs "$NUM_ENVS" --max_iters "$MAX_ITERS" \
        --rollout_steps "$ROLLOUT_STEPS" --save_every "$SAVE_EVERY" --headless "${resume[@]}" 2>&1 | tee -a "$LOG_DIR/train_$EXPTID.log"
fi

# ------------------------------------------------------------------ 2. videos + diagnosis
FAIL=0
for it in 0 $(seq "$SAVE_EVERY" "$SAVE_EVERY" "$MAX_ITERS"); do
    if [ ! -f "$RUN_DIR/policy_$it.pt" ]; then
        echo "[FAIL] no checkpoint policy_$it.pt (training did not reach it, see $LOG_DIR/train_$EXPTID.log)"
        FAIL=1; continue
    fi
    # terrain as it was at that point of training (iteration 0 shows rough + 0.10 m stairs)
    cur=$(( it > 0 ? it : 2200 ))
    echo ">>> video of checkpoint $it"
    python -u "$PORT_DIR/record_video.py" --run_dir "$RUN_DIR" --iter "$it" --curriculum_itr "$cur" \
        --pipeline_dir "$PIPELINE_DIR" --headless > "$LOG_DIR/video_${EXPTID}_$it.log" 2>&1
    if [ -s "$RUN_DIR/video_$it.mp4" ] && ! grep -q Traceback "$LOG_DIR/video_${EXPTID}_$it.log"; then
        echo "[PASS] $RUN_DIR/video_$it.mp4"
    else
        echo "[FAIL] video $it (see $LOG_DIR/video_${EXPTID}_$it.log)"; FAIL=1
    fi
    if [ "$DIAGNOSE" = "1" ]; then
        python -u "$PORT_DIR/diagnose_policy.py" --run_dir "$RUN_DIR" --iter "$it" --envs_per_scenario 50 --headless \
            > "$LOG_DIR/diagnose_${EXPTID}_$it.log" 2>&1
        sed -n '/^=====/,/^files:/p' "$LOG_DIR/diagnose_${EXPTID}_$it.log"
    fi
done

echo
echo ">>> download the videos to your laptop with (run on the laptop):"
echo "    scp -i <key.pem> 'ubuntu@<ec2-ip>:$RUN_DIR/video_*.mp4' ."
exit $FAIL
