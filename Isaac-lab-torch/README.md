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

## Run on EC2 (same stack as nv-omni-tr-pipeline: Isaac Lab v2.3.2, Isaac Sim 5.1, conda env `isaac`)

```bash
# 1. one-time setup (reboots once for the driver, then re-run)
git clone https://github.com/Raj9408612613/nv-omni-tr-pipeline ~/nv-omni-tr-pipeline
bash ~/nv-omni-tr-pipeline/scripts/isaac_run.sh
conda activate isaac
git clone -b claude/blissful-edison-alxpwz https://github.com/Raj9408612613/RMA-locomotion-code.- ~/rma
cd ~/rma
pip install ruamel.yaml trimesh "imageio[ffmpeg]"

# 2. CPU test (seconds)
python Isaac-lab-torch/tests/test_terrain.py

# 3. smoke test: 64 envs, stop with Ctrl-C after 2-3 iterations
python Isaac-lab-torch/runner.py --exptid 0 --num_envs 64 --headless --overwrite
#    check: "[settle] base height above terrain" ~0.25-0.33 m, rewards finite, dones well below 1.0

# 4. video of the starting policy (checkpoint 0 is written by the smoke test)
python Isaac-lab-torch/record_video.py --run_dir Isaac-lab-torch/data/rsg_a1_task/0000 --iter 0 --headless
#    -> Isaac-lab-torch/data/rsg_a1_task/0000/video_0.mp4  (copy with scp)

# 5. full training (in tmux; log the output)
python Isaac-lab-torch/runner.py --exptid 1 --num_envs 1024 --headless 2>&1 | tee train_1.log

# 6. video of a checkpoint (one is written every eval_every_n = 2000 iterations)
python Isaac-lab-torch/record_video.py --run_dir Isaac-lab-torch/data/rsg_a1_task/0001 --iter 10000 --headless
```

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
then continue (`--loadid 10000`) to about 20 000 if rewards and `dones` are still improving. The base
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
8. Push forces are applied in the base frame (RaiSim: world frame). Their direction is random, so the effect is minor.

## Not ported yet

The DAgger student env (`dagger_a1`, 50-step observation history), `evaluate_policy.py` and `viz_policy.py`.
