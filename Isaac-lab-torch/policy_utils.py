"""Load a runner.py checkpoint (policy_<iter>.pt + mean/var<iter>.csv) as a deterministic policy."""
import os

import numpy as np
import torch


class SavedPolicy:
    """Normalises observations exactly like RaisimGymVecEnv and runs the traced RMA actor.
    Also offers the `trainer` interface (`net`, `sample_action`) nv-omni-tr-pipeline's recorder expects."""

    def __init__(self, run_dir, itr, device):
        self.net = torch.jit.load(os.path.join(run_dir, f"policy_{itr}.pt"), map_location=device)
        mean = np.loadtxt(os.path.join(run_dir, f"mean{itr}.csv"), dtype=np.float32)
        var = np.loadtxt(os.path.join(run_dir, f"var{itr}.csv"), dtype=np.float32)
        self.mean = torch.tensor(np.atleast_2d(mean)[0], device=device)
        self.std = torch.tensor(np.sqrt(np.atleast_2d(var)[0] + 1e-8), device=device)

    @torch.no_grad()
    def act(self, obs):
        x = obs["policy"] if isinstance(obs, dict) else obs
        return self.net(((x - self.mean) / self.std).clamp(-10.0, 10.0))

    def sample_action(self, obs):
        return self.act(obs), None, None
