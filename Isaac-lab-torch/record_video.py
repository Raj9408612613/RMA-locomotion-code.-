"""Record an MP4 of a trained policy, reusing nv-omni-tr-pipeline's video recorder.

    python Isaac-lab-torch/record_video.py --run_dir Isaac-lab-torch/data/rsg_a1_task/0001 --iter 10000 \
        --pipeline_dir ~/nv-omni-tr-pipeline --headless --enable_cameras

`--iter 0` records the starting point (the blind base policy loaded by runner.py).

Only the frame-capture/MP4 helper `record_episodes` comes from nv-omni-tr-pipeline. The policy is
the RMA network trained by runner.py (Raisim-gym-torch/algo); no omni_spot network or trainer is used.
"""
import argparse
import os
import sys
sys.stdout.reconfigure(line_buffering=True)  # keep log lines when Isaac Sim exits

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", required=True, help="runner.py output dir (has cfg.yaml, policy_<iter>.pt, mean/var csv)")
parser.add_argument("--iter", type=int, required=True)
parser.add_argument("--pipeline_dir", default=os.path.expanduser("~/nv-omni-tr-pipeline"))
parser.add_argument("--output", default=None)
parser.add_argument("--n_episodes", type=int, default=4)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--curriculum_itr", type=int, default=10000,
                    help="iteration number given to the env; >=8200 unlocks all stair heights")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import torch
from ruamel.yaml import YAML

task_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, args.pipeline_dir)
from omni_spot.video_recorder import record_episodes  # noqa: E402
from a1_env import A1RmaEnv, A1RmaEnvCfg  # noqa: E402
from policy_utils import SavedPolicy  # noqa: E402


cfg = YAML().load(open(os.path.join(args.run_dir, "cfg.yaml")))
env_cfg_yaml = dict(cfg["environment"])
env_cfg_yaml["num_envs"] = 1
env_cfg_yaml["eval"] = True  # RaiSim evaluate_policy.py setting: 30% rough / 70% stairs, half down
env_cfg = A1RmaEnvCfg()
env_cfg.apply_yaml(env_cfg_yaml)
env_cfg.sim.device = args.device
env = A1RmaEnv(env_cfg, render_mode="rgb_array")
env.itr_number = args.curriculum_itr
env.resample_on_reset = True  # evaluation: new command on every reset

policy = SavedPolicy(args.run_dir, args.iter, env.device)
output = args.output or os.path.join(args.run_dir, f"video_{args.iter}.mp4")
fps = int(round(1.0 / (env_cfg.sim.dt * env_cfg.decimation)))  # 100 Hz control -> real-time video
record_episodes(env, policy, n_episodes=args.n_episodes, output_path=output,
                max_steps=args.max_steps, fps=fps)
env.close()
simulation_app.close()
