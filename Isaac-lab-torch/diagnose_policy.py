"""Diagnose how well a checkpoint walks, per terrain: success rate, why episodes end, command
tracking, energy, smoothness, foot impact/slip, height and tilt.

    python Isaac-lab-torch/diagnose_policy.py --run_dir Isaac-lab-torch/data/rsg_a1_task/0001 --iter 0 --headless

Every scenario (rough terrain, stairs up at each step height, stairs down at each step height) gets
`--envs_per_scenario` robots. Each robot contributes exactly its FIRST episode (up to 1200 steps = 12 s),
so short failed episodes are not over-counted. The policy is the deterministic RMA actor saved by runner.py.
Writes <run_dir>/diagnosis_<iter>/episodes.csv and summary.csv and prints a table.
"""
import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", required=True)
parser.add_argument("--iter", type=int, required=True)
parser.add_argument("--envs_per_scenario", type=int, default=20)
parser.add_argument("--command", default="forward", choices=["forward", "random"],
                    help="forward: fixed 0.45 m/s, no turning; random: RaiSim sample_goals")
parser.add_argument("--speed", type=float, default=0.45, help="forward command in m/s (--command forward)")
parser.add_argument("--rules", default="train", choices=["train", "eval"],
                    help="termination rules: train (as during training) or eval (evaluate_policy.py: looser)")
parser.add_argument("--no_dr", action="store_true", help="disable domain randomisation (nominal robot)")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import csv

import numpy as np
import torch
from ruamel.yaml import YAML

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from a1_env import A1RmaEnv, A1RmaEnvCfg  # noqa: E402
from policy_utils import SavedPolicy  # noqa: E402
from terrain import KIND_DOWN, KIND_ROUGH, KIND_UP, NUM_DOWN_LEVELS, STEP_HEIGHT_LIST  # noqa: E402

REASONS = ["illegal_contact", "orientation", "base_height", "yaw", "timeout"]
SCENARIOS = ([("rough", KIND_ROUGH, 0)]
             + [(f"up_{h:.2f}", KIND_UP, i) for i, h in enumerate(STEP_HEIGHT_LIST)]
             + [(f"down_{h:.2f}", KIND_DOWN, i) for i, h in enumerate(STEP_HEIGHT_LIST[:NUM_DOWN_LEVELS])])

torch.manual_seed(args.seed)
E = args.envs_per_scenario
n = E * len(SCENARIOS)
cfg = YAML().load(open(os.path.join(args.run_dir, "cfg.yaml")))
env_yaml = dict(cfg["environment"])
env_yaml["num_envs"] = n
env_yaml["eval"] = args.rules == "eval"
if args.no_dr:
    for k in ["randomize_friction", "randomize_mass", "randomize_motor_strength", "randomize_gains"]:
        env_yaml[k] = False
env_cfg = A1RmaEnvCfg()
env_cfg.apply_yaml(env_yaml)
env_cfg.sim.device = args.device
env = A1RmaEnv(env_cfg)
dev = env.device

scen_of_env = torch.arange(n, device=dev) // E
kinds = torch.tensor([k for _, k, _ in SCENARIOS], device=dev)[scen_of_env]
levels = torch.tensor([l for _, _, l in SCENARIOS], device=dev)[scen_of_env]
env.fixed_terrain = (kinds, levels)
env.fixed_command = (args.speed, 0.0) if args.command == "forward" else None
env.resample_on_reset = True
policy = SavedPolicy(args.run_dir, args.iter, dev)

env.reset()
env.settle()
obs = env.obs_buf

# per-env accumulators over the first episode
K = ["steps", "vx", "cmd_vx", "err_vx", "err_wz", "height", "max_roll", "max_pitch",
     "forward_r", "distance", "energy", "smoothness", "ground_impact", "foot_slip", "torque_sq"]
acc = {k: torch.zeros(n, device=dev) for k in K}
finished = torch.zeros(n, dtype=torch.bool, device=dev)
reason = torch.full((n,), -1, dtype=torch.long, device=dev)

print(f"[diagnose] {len(SCENARIOS)} scenarios x {E} robots, command={args.command}, rules={args.rules}, "
      f"domain randomisation={'off' if args.no_dr else 'on'}")
step = 0
while not bool(finished.all()) and step < 1300:
    obs, _, terminated, time_out, _ = env.step(policy.act(obs))
    step += 1
    live = ~finished
    ds, info = env.diag_state, env.reward_info  # pre-reset values of this step
    upd = {
        "steps": torch.ones(n, device=dev), "vx": ds[:, 0], "cmd_vx": ds[:, 3],
        "err_vx": (ds[:, 0] - ds[:, 3]).abs(), "err_wz": (ds[:, 2] - ds[:, 4]).abs(), "height": ds[:, 5],
        "forward_r": info[:, 0], "distance": info[:, 15], "energy": info[:, 11], "smoothness": info[:, 3],
        "ground_impact": info[:, 7], "foot_slip": info[:, 9], "torque_sq": info[:, 13],
    }
    for k, v in upd.items():
        acc[k] += torch.where(live, v, torch.zeros_like(v))
    acc["max_roll"] = torch.where(live, torch.maximum(acc["max_roll"], ds[:, 6]), acc["max_roll"])
    acc["max_pitch"] = torch.where(live, torch.maximum(acc["max_pitch"], ds[:, 7]), acc["max_pitch"])
    done = (terminated | time_out) & live
    if done.any():
        first_reason = torch.argmax(env.last_term.long(), -1)
        reason = torch.where(done, first_reason, reason)
        finished |= done
    if step % 200 == 0:
        print(f"[diagnose] step {step}: {int(finished.sum())}/{n} episodes finished")

# ---------------------------------------------------------------- report
out_dir = os.path.join(args.run_dir, f"diagnosis_{args.iter}")
os.makedirs(out_dir, exist_ok=True)
a = {k: v.cpu().numpy() for k, v in acc.items()}
steps = np.maximum(a["steps"], 1)
reason_np = reason.cpu().numpy()
rows = []
for i in range(n):
    s = SCENARIOS[i // E]
    rows.append({
        "scenario": s[0], "env": i, "steps": int(a["steps"][i]),
        "success": int(reason_np[i] == 4), "end_reason": REASONS[reason_np[i]] if reason_np[i] >= 0 else "unfinished",
        "vx": a["vx"][i] / steps[i], "cmd_vx": a["cmd_vx"][i] / steps[i], "err_vx": a["err_vx"][i] / steps[i],
        "err_wz": a["err_wz"][i] / steps[i], "height": a["height"][i] / steps[i],
        "max_roll": a["max_roll"][i], "max_pitch": a["max_pitch"][i],
        # evaluate_policy.py / compute_results.py metrics (sums; energy etc. per step below)
        "forward_r": a["forward_r"][i], "distance": a["distance"][i],
        "energy": a["energy"][i] / steps[i], "smoothness": a["smoothness"][i] / steps[i],
        "ground_impact": a["ground_impact"][i] / steps[i], "foot_slip": a["foot_slip"][i] / steps[i],
        "torque_sq": a["torque_sq"][i] / steps[i],
    })
with open(os.path.join(out_dir, "episodes.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

summary = []
for name, _, _ in SCENARIOS:
    r = [x for x in rows if x["scenario"] == name]
    mean = lambda k: float(np.mean([x[k] for x in r]))
    ends = {k: sum(x["end_reason"] == k for x in r) for k in REASONS[:4]}
    summary.append({
        "scenario": name, "episodes": len(r), "success_%": 100 * mean("success"),
        "survived_s": mean("steps") / 100, **{f"end_{k}": v for k, v in ends.items()},
        "vx": mean("vx"), "cmd_vx": mean("cmd_vx"), "err_vx": mean("err_vx"), "err_wz": mean("err_wz"),
        "height": mean("height"), "max_pitch": mean("max_pitch"), "distance_m": mean("distance"),
        "energy": mean("energy"), "smoothness": mean("smoothness"), "ground_impact": mean("ground_impact"),
        "foot_slip": mean("foot_slip"),
    })
with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)

hdr = f"{'scenario':<11}{'succ%':>6}{'surv s':>7}{'contact':>8}{'tilt':>5}{'low':>4}{'yaw':>4}" \
      f"{'vx':>6}{'cmd':>6}{'|e_vx|':>7}{'|e_wz|':>7}{'height':>7}{'pitch':>6}{'dist m':>7}{'slip':>7}"
print("\n" + "=" * len(hdr))
print(f"Diagnosis of {args.run_dir} iter {args.iter} ({E} robots per scenario, first episode each, max 12 s)")
print("=" * len(hdr))
print(hdr)
for s in summary:
    print(f"{s['scenario']:<11}{s['success_%']:>6.0f}{s['survived_s']:>7.1f}{s['end_illegal_contact']:>8}"
          f"{s['end_orientation']:>5}{s['end_base_height']:>4}{s['end_yaw']:>4}{s['vx']:>6.2f}{s['cmd_vx']:>6.2f}"
          f"{s['err_vx']:>7.3f}{s['err_wz']:>7.3f}{s['height']:>7.3f}{s['max_pitch']:>6.2f}"
          f"{s['distance_m']:>7.2f}{s['foot_slip']:>7.3f}")
print("-" * len(hdr))
weak = [s["scenario"] for s in summary if s["success_%"] < 50]
print("success < 50% on: " + (", ".join(weak) if weak else "none"))
print("columns: contact/tilt/low/yaw = episodes ended by illegal body contact / roll-pitch limit / base too low / "
      "yaw limit; |e_vx|,|e_wz| = mean command tracking error; pitch = mean of per-episode max |pitch| (rad)")
print(f"files: {out_dir}/episodes.csv, summary.csv")
env.close()
simulation_app.close()
