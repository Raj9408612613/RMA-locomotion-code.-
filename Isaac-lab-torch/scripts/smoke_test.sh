#!/usr/bin/env bash
# =============================================================================
# smoke_test.sh — checks that the Isaac Lab port actually runs. Prints [PASS]/[FAIL].
#   1. terrain unit test (CPU)
#   2. 3 PPO iterations with 64 envs (Isaac Sim, headless)
#   3. 1-episode video of checkpoint 0 through nv-omni-tr-pipeline's recorder
# Usage (conda env 'isaac' active, from anywhere):
#   bash Isaac-lab-torch/scripts/smoke_test.sh
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT_DIR="$(dirname "$SCRIPT_DIR")"
PIPELINE_DIR="${PIPELINE_DIR:-$HOME/nv-omni-tr-pipeline}"
RUN_DIR="$PORT_DIR/data/rsg_a1_task/9000"
LOG="/tmp/rma_smoke_train.log"
FAIL=0
# activate the conda env from setup_isaac.sh if this shell has not (e.g. ~/.bashrc not re-sourced)
if [ "${CONDA_DEFAULT_ENV:-}" != "isaac" ] && [ -x "$HOME/miniconda3/bin/conda" ]; then
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)" && conda activate isaac \
        && echo ">>> activated conda env 'isaac'"
fi
python -c "import isaaclab" 2>/dev/null || { echo "[FAIL] isaaclab not importable with $(command -v python); run: conda activate isaac"; exit 1; }
if [ "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" ]; then
    echo "[FAIL] OMNI_KIT_ACCEPT_EULA is not set; Isaac Sim would abort at its EULA prompt."
    echo "       If you accept the NVIDIA Omniverse EULA, run once:"
    echo "         echo 'export OMNI_KIT_ACCEPT_EULA=YES' >> ~/.bashrc && source ~/.bashrc"
    exit 1
fi
pass() { echo "[PASS] $*"; }
fail() { echo "[FAIL] $*"; FAIL=1; }

echo ">>> 1/3 terrain unit test"
python "$PORT_DIR/tests/test_terrain.py" && pass "terrain" || fail "terrain"

echo ">>> 1b/3 CPU dry run of runner.py against a stand-in Isaac Lab (catches Python errors without the GPU)"
DRY_LOG=/tmp/rma_smoke_dryrun.log
timeout 900 python "$PORT_DIR/tests/fake_isaac.py" "$PORT_DIR/runner.py" --exptid 9001 --num_envs 8 --max_iters 1 \
    --overwrite --headless --debug > "$DRY_LOG" 2>&1
if grep -q "1th iteration" "$DRY_LOG" && ! grep -q Traceback "$DRY_LOG"; then pass "dry run"
else fail "dry run (see $DRY_LOG)"; grep -A30 Traceback "$DRY_LOG" | tail -30; fi
rm -rf "$PORT_DIR/data/rsg_a1_task/9001"

echo ">>> 2/3 training smoke: 64 envs x 3 iterations (first launch compiles shaders, can take 5-10 min)"
timeout 1800 python -u "$PORT_DIR/runner.py" --exptid 9000 --num_envs 64 --max_iters 2 --overwrite --headless \
    > "$LOG" 2>&1
code=$?
[ $code -eq 0 ] && pass "runner exit code 0 (Isaac Sim reports 0 even on errors; see traceback check)" || fail "runner exit code $code (see $LOG)"
python - "$LOG" "$RUN_DIR" <<'PY' || FAIL=1
import math, os, re, sys
log, run_dir = open(sys.argv[1], errors="replace").read(), sys.argv[2]
ok = True
def check(cond, msg):
    global ok
    print(("[PASS] " if cond else "[FAIL] ") + msg)
    ok &= bool(cond)
h = [float(x) for x in re.findall(r"\[settle\] base height above terrain: mean ([-\d.]+)", log)]
check(h and 0.2 < h[0] < 0.38, f"robot stands after settle (base height {h[:1]} m, expect 0.2-0.38)")
r = [float(x) for x in re.findall(r"average ll reward:\s+([-\w.]+)", log)]
check(len(r) >= 3 and all(math.isfinite(x) for x in r), f"3 iterations with finite reward {r}")
d = [float(x) for x in re.findall(r"dones:\s+([-\w.]+)", log)]
# dones is per step. No RaiSim reference rate is known, so this is reported, not failed.
print(f"[INFO] dones per step {d} (0.01 = one episode end per env per second)")
for line in re.findall(r"\[term\].*", log)[:3]:
    print("[INFO] " + line)
check("Traceback" not in log, "no Python traceback")
check("PhysX error" not in log, "no PhysX errors")
for f in ["policy_0.pt", "mean0.csv", "var0.csv", "full_0.pt"]:
    check(os.path.isfile(os.path.join(run_dir, f)), f"checkpoint file {f}")
if "Traceback" in log:  # Isaac Sim's launcher exits 0 even on a Python exception: show it
    i = log.rfind("Traceback")
    print("------ traceback from " + sys.argv[1] + " ------\n" + log[i:i + 4000])
sys.exit(0 if ok else 1)
PY

echo ">>> 3/3 video of checkpoint 0"
VIDEO="$RUN_DIR/video_0.mp4"
rm -f "$VIDEO"
timeout 1800 python "$PORT_DIR/record_video.py" --run_dir "$RUN_DIR" --iter 0 --n_episodes 1 --max_steps 300 \
    --pipeline_dir "$PIPELINE_DIR" --headless > /tmp/rma_smoke_video.log 2>&1
[ -s "$VIDEO" ] && pass "video written: $VIDEO ($(du -h "$VIDEO" | cut -f1))" \
                || fail "no video (see /tmp/rma_smoke_video.log)"

echo "==================================================================="
if [ $FAIL -eq 0 ]; then echo "[PASS] smoke test passed"; else echo "[FAIL] smoke test failed — logs: $LOG, /tmp/rma_smoke_video.log"; fi
exit $FAIL
