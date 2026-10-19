"""Where does env.step time go? Runs the A1 env with zero actions and prints ms per control step
and a cProfile breakdown. Run with several --num_envs to see whether the cost is per call or per robot:

    python Isaac-lab-torch/tests/profile_step.py --num_envs 200 --headless
    python Isaac-lab-torch/tests/profile_step.py --num_envs 1024 --headless
"""
import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=200)
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import cProfile
import pstats
import time

import torch
from ruamel.yaml import YAML

port_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, port_dir)
from a1_env import A1RmaEnv, A1RmaEnvCfg  # noqa: E402
from isaac_wrapper import IsaacGymEnvWrapper  # noqa: E402

cfg = YAML().load(open(os.path.join(port_dir, "cfg.yaml")))["environment"]
cfg["num_envs"] = args.num_envs
env_cfg = A1RmaEnvCfg()
env_cfg.apply_yaml(cfg)
env_cfg.sim.device = args.device
env = A1RmaEnv(env_cfg)
wrapper = IsaacGymEnvWrapper(env)
wrapper.reset()
act = torch.zeros(args.num_envs, 12, device=env.device)

for _ in range(20):  # warm-up
    env.step(act)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(args.steps):
    env.step(act)
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / args.steps * 1000
print(f"[profile] num_envs={args.num_envs}: {ms:.2f} ms per control step "
      f"-> {ms * 600 / 1000:.1f} s of env.step per 600-step iteration")

prof = cProfile.Profile()
prof.enable()
for _ in range(100):
    env.step(act)
torch.cuda.synchronize()
prof.disable()
print("[profile] top functions by cumulative time over 100 steps:")
pstats.Stats(prof).sort_stats("cumulative").print_stats(30)
print("[profile] top functions by own time:")
pstats.Stats(prof).sort_stats("tottime").print_stats(20)
env.close()
simulation_app.close()
