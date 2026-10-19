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

## Run

```bash
# inside an Isaac Lab install (2.x, isaaclab + isaaclab_assets packages)
./isaaclab.sh -p -m pip install ruamel.yaml trimesh wandb
./isaaclab.sh -p /path/to/Isaac-lab-torch/runner.py --exptid 1 --headless --num_envs 4096
python Isaac-lab-torch/tests/test_terrain.py   # terrain unit test, needs only numpy + torch
```

Leave out `--headless` to watch training in the Isaac Sim GUI. Logs go to `Isaac-lab-torch/data/`.

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
