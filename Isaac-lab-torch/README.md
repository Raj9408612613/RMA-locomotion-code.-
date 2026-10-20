# Isaac Lab (Isaac Sim) port of `rsg_a1_task`

Port of the RaiSim teacher-training environment (`Raisim-gym-torch/env/envs/rsg_a1_task`)
to [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 2.x on Isaac Sim. The PPO code, network
modules, `RaisimGymVecEnv` normalisation and the `Base-policy/` checkpoints are reused unchanged.

| File | Replaces |
|---|---|
| `a1_env.py` | `Environment.hpp` (GPU-vectorised `DirectRLEnv`) |
| `terrain.py` | RaiSim height maps (`add_random_terrain`, `add_stairs`, `hm_->getHeight`) |
| `isaac_wrapper.py` | compiled pybind module `rsg_a1_task.RaisimGymEnv` |
| `runner.py`, `cfg.yaml` | `rsg_a1_task/runner.py`, `cfg.yaml` |

## Run on EC2: step by step

Only the **software stack** (Isaac Sim 5.1, Isaac Lab v2.3.2, Python 3.11, torch 2.7 cu128, conda env `isaac`)
and the **MP4 helper** (`omni_spot/video_recorder.py: record_episodes`) come from nv-omni-tr-pipeline.
The networks, PPO and base policy are RMA's own (`Raisim-gym-torch/algo`, `Base-policy/`).

1. **Start the instance.** Use Ubuntu 22.04/24.04 with an NVIDIA RTX-class GPU (Isaac Sim needs RT cores, so A10G/L4/L40S
   work but T4/V100/A100 do not render), about 100 GB disk and ports for SSH.
2. **Get the code.**
   ```bash
   git clone -b claude/blissful-edison-alxpwz https://github.com/Raj9408612613/RMA-locomotion-code.- ~/rma
   ```
3. **Install everything.** `Isaac-lab-torch/scripts/setup_isaac.sh` is a copy of the pipeline's
   `isaac_run.sh` (same pinned versions). Stage 5 is changed to install `ruamel.yaml trimesh imageio[ffmpeg]`,
   clone nv-omni-tr-pipeline (for the recorder) and run the RMA smoke test.
   ```bash
   bash ~/rma/Isaac-lab-torch/scripts/setup_isaac.sh   # first run installs the driver and asks you to reboot
   sudo reboot
   bash ~/rma/Isaac-lab-torch/scripts/setup_isaac.sh   # second run does the rest and ends with the smoke test
   ```
   The torch pin is `cu128` (Blackwell). On an older GPU, edit `TORCH_CUDA`/`NVIDIA_DRIVER` at the top first.
4. **Check that it runs.** The smoke test runs automatically at the end of setup. Re-run it any time with:
   ```bash
   conda activate isaac && cd ~/rma
   bash Isaac-lab-torch/scripts/smoke_test.sh
   ```
   It must end with `[PASS] smoke test passed`. It checks the terrain test, 3 PPO iterations with 64 envs
   (clean exit, robot standing about 0.3 m after settle, finite rewards, few falls, no traceback or PhysX error,
   checkpoint files written) and a 1-episode MP4 of checkpoint 0. Logs: `/tmp/rma_smoke_train.log`,
   `/tmp/rma_smoke_video.log`.
5. **Watch the starting policy** before spending GPU hours:
   ```bash
   scp ubuntu@<ec2-ip>:~/rma/Isaac-lab-torch/data/rsg_a1_task/9000/video_0.mp4 .
   ```
   It should show the base policy walking. If it falls or twitches, stop and debug (see "Differences" below).
6. **Train** (in `tmux`):
   ```bash
   python Isaac-lab-torch/runner.py --exptid 1 --num_envs 1024 --max_iters 10000 --headless 2>&1 | tee ~/rma_train_1.log
   # continue later: add --loadid 10000 --overwrite --max_iters 20000 (same --exptid)
   ```
7. **Make a video** of any saved checkpoint (one every 2000 iterations):
   ```bash
   python Isaac-lab-torch/record_video.py --run_dir Isaac-lab-torch/data/rsg_a1_task/0001 --iter 10000 --headless
   ```

8. **Diagnose** a checkpoint per terrain (success rate, why episodes end, command tracking, energy, smoothness,
   foot impact/slip, height, tilt). Run it on `--iter 0` once for the baseline, then on each new checkpoint:
   ```bash
   python Isaac-lab-torch/diagnose_policy.py --run_dir Isaac-lab-torch/data/rsg_a1_task/0001 --iter 2000 --headless
   # options: --command random (RaiSim command sampling)  --rules eval (evaluate_policy.py termination)  --no_dr
   ```
   Writes `diagnosis_<iter>/episodes.csv` and `summary.csv` into the run directory.

`record_video.py` uses `omni_spot/video_recorder.py` from nv-omni-tr-pipeline (`--pipeline_dir`,
default `~/nv-omni-tr-pipeline`). It films env 0 with a chase camera (`viewer` in `A1RmaEnvCfg`).
The video is 100 fps (one frame per control step), so it plays in real time.

### How many iterations

The curriculum is counted in **iterations**, not samples. More envs means more data per iteration,
but the schedule stays the same:

| iteration | what happens |
|---|---|
| 0 | starts from the base policy in `Base-policy/`; checkpoint 0 is saved |
| 1200 | stairs appear (60% of terrains), lowest step height 0.10 m |
| 1200 + 1000·k | one more step-height level unlocked |
| 8200 | all levels unlocked (up to 0.23 m) |

Run to **10 000 iterations** first. That is the first checkpoint after the full curriculum. Make a video,
then continue (`--loadid 10000 --overwrite --max_iters 20000`) to about 20 000 if rewards and `dones` are still improving. The base
policy is named `_22000`, which suggests the original runs were about that long. Time per iteration is
printed in the log (`time elapsed in this iteration`). Multiply it out to estimate the total.

## Kept the same as RaiSim

- 400 Hz physics, 100 Hz control, 2-step action delay, PD target = `0.4 * a * motor_strength + default_pose`
- the 140-dim observation (two identical 70-dim halves) in RaiSim joint order (FR, FL, RR, RL), so
  `Base-policy/*` and its `mean/var` CSVs load directly
- every reward term, coefficient, NaN/clip rule and the -10 terminal reward; `getRewardInfo` layout
- termination rules, command sampling (`sample_goals`), push forces during "stop" commands
- domain randomisation: friction, PD gains, base mass + COM, motor strength, re-sampled on reset
  and every 50–59 steps
- terrain curriculum: rough only until iteration 1200, then 40% rough / 60% stairs (75% of them down),
  with the step height growing every 1000 iterations

## Differences you should know about

1. **Terrain is a tile grid, not a fresh height map per env.** Isaac Lab envs share one stage, so
   `terrain.py` pre-builds `num_variants` random variants per (kind, step height) and a terrain change
   moves the robot to another tile. The resolution is 0.05 m (RaiSim used 0.02 m for stairs), so the
   2.5 cm stair chamfer is 5 cm wide.
2. **Fractal noise is re-implemented** (Perlin, same frequency, octaves, gain and zScale). RaiSim's
   generator amplitude convention is undocumented, so rough-terrain roughness may differ. *Low confidence on exact match.*
3. **Settling**: RaiSim integrates 50 steps after every reset. Here this only happens on the full
   reset at the start of each PPO iteration. Envs that auto-reset mid-iteration start falling from 0.4 m.
4. **Torque limit** of 33.5 Nm is enforced (A1 spec). It is unclear whether RaiSim clipped torque.
5. **Contacts**: the GRF "impulse" is contact-sensor force × physics dt. Foot = `*_foot` + `*_calf` bodies.
6. **Robot asset**: Isaac Lab's `UNITREE_A1_CFG` (base link is named `trunk`). I assumed its joint axes match
   the RaiSim A1 URDF (both come from Unitree's URDF). *Medium confidence. Check the standing pose first.*
7. **Alive bonus**: `Environment.hpp` reads `cfg["aliveBonus"]`, which is missing from `cfg.yaml`, so RaiSim
   ran with 0. This is kept. Add `aliveBonus: 18` to the yaml to turn it on.
8. Friction / mass / COM changes reach PhysX in batches every 10 control steps (0.1 s), not instantly.

## Not ported yet

The DAgger student env (`dagger_a1`, 50-step observation history) and `viz_policy.py`. `diagnose_policy.py` covers `evaluate_policy.py`'s metrics.
