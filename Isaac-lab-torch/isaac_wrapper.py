"""Drop-in replacement for the pybind11 module `rsg_a1_task.RaisimGymEnv`.

Implements the methods RaisimGymVecEnv calls on its `wrapper`, so the existing vec-env,
PPO and runner code work unchanged on top of Isaac Lab.
"""

import time

import numpy as np
import torch


class IsaacGymEnvWrapper:
    def __init__(self, env):
        self.env = env
        self._obs = None
        self._step_time = 0.0

    def init(self):
        pass

    def getObDim(self):
        return self.env.cfg.observation_space

    def getActionDim(self):
        return self.env.cfg.action_space

    def getNumOfEnvs(self):
        return self.env.num_envs

    def reset(self):
        if self._step_time > 0:
            print(f"[time] env.step total since last reset: {self._step_time:.2f} s (rest of the iteration is PPO)")
            self._step_time = 0.0
        # VecEnv.reset() -> every env reset(true): resample commands, then settle.
        self.env.resample_on_reset = True
        self.env.reset()
        self.env.resample_on_reset = False
        self.env.settle()
        self._obs = self.env.obs_buf["policy"]

    def step(self, action, reward, done):
        t0 = time.perf_counter()
        act = torch.as_tensor(action, dtype=torch.float32, device=self.env.device)
        obs, rew, terminated, time_out, _ = self.env.step(act)
        self._obs = obs["policy"]
        reward[:] = rew.cpu().numpy()
        done[:] = (terminated | time_out).cpu().numpy()
        self._step_time += time.perf_counter() - t0

    def observe(self, ob):
        ob[:] = self._obs.cpu().numpy()

    def getRewardInfo(self, out):
        out[:] = self.env.reward_info.cpu().numpy()

    def getDis(self, out):
        pass

    def curriculumUpdate(self):
        self.env.curriculum_update()

    def setItrNumber(self, n):
        self.env.itr_number = int(n)

    def setSeed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def close(self):
        self.env.close()

    # visualization is handled by the Isaac Sim GUI (run without --headless)
    def turnOnVisualization(self):
        pass

    def turnOffVisualization(self):
        pass

    def startRecordingVideo(self, name):
        pass

    def stopRecordingVideo(self):
        pass
