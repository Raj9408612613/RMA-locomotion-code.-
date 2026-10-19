"""Stand-in for Isaac Lab v2.3.2 so runner.py/record_video.py run end to end on CPU.
DirectRLEnv.step/reset follow isaaclab/envs/direct_rl_env.py (v2.3.2) call order.
Physics is fake: joints track PD targets loosely, base stays near standing height."""
import sys, types, torch, numpy as np
PORT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
def mod(name, **kw):
    m = types.ModuleType(name); m.__dict__.update(kw); sys.modules[name] = m; return m
class Any_:
    def __init__(self, *a, **k): self.__dict__.update(k)
    def replace(self, **k): self.__dict__.update(k); return self
    def __call__(self, *a, **k): return Any_()
class App:
    def __init__(self, args): self.app = self
    def close(self): pass
    @staticmethod
    def add_app_launcher_args(p):
        p.add_argument("--headless", action="store_true"); p.add_argument("--enable_cameras", action="store_true")
        p.add_argument("--device", default="cpu")
class Sim:
    def __init__(self, cfg): self.cfg = cfg; self.device = "cpu"
    def step(self, render=False): FakeWorld.step()
    def forward(self): pass
    def render(self): pass
    def has_rtx_sensors(self): return False
class SimCfg:
    def __init__(self, **k): self.__dict__.update(dict(dt=0.0025, render_interval=4, device="cpu")); self.__dict__.update(k)
class SceneCfg:
    def __init__(self, **k): self.__dict__.update(k)
class Scene:
    def __init__(self): self.articulations = {}; self.sensors = {}
    def clone_environments(self, **k): pass
    def filter_collisions(self, **k): pass
    def write_data_to_sim(self): pass
    def update(self, dt): pass
    def reset(self, ids): pass
    def __getitem__(self, key): raise KeyError(f"Scene entity with key '{key}' not found.")
class EnvCfgBase:
    seed = None
class DirectRLEnv:
    def __init__(self, cfg, render_mode=None, **k):
        self.cfg = cfg; self.render_mode = render_mode
        self.sim = Sim(cfg.sim); self.num_envs = cfg.scene.num_envs; self.scene = Scene()
        FakeWorld.n = self.num_envs
        self._setup_scene()
        FakeWorld.init(self)
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool); self.reset_time_outs = self.reset_terminated.clone()
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
    device = property(lambda self: "cpu")
    physics_dt = property(lambda self: self.cfg.sim.dt)
    def step(self, action):
        self._pre_physics_step(action)
        for _ in range(self.cfg.decimation):
            self._apply_action(); self.scene.write_data_to_sim(); self.sim.step(render=False); self.scene.update(self.physics_dt)
        self.episode_length_buf += 1
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reward_buf = self._get_rewards()
        ids = (self.reset_terminated | self.reset_time_outs).nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0: self._reset_idx(ids)
        self.obs_buf = self._get_observations()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, {}
    def reset(self, seed=None, options=None):
        self._reset_idx(torch.arange(self.num_envs, dtype=torch.int64)); self.sim.forward()
        self.obs_buf = self._get_observations(); return self.obs_buf, {}
    def _reset_idx(self, ids): self.scene.reset(ids); self.episode_length_buf[ids] = 0
    def render(self, recompute=False): return np.zeros((72, 128, 3), np.uint8)
    def close(self): pass

B = 17
BODY = ["trunk"] + [f"{l}_{p}" for l in ["FR","FL","RR","RL"] for p in ("hip","thigh","calf","foot")]
JOINTS_ISAAC = [f"{l}_{j}_joint" for j in ("hip","thigh","calf") for l in ["FL","FR","RL","RR"]]
def finder(all_names):
    def f(names, preserve_order=False):
        names = [names] if isinstance(names, str) else list(names)
        return [all_names.index(n) for n in names], names
    return f
class View:
    def __init__(s, n): s.n = n
    def get_masses(s): return torch.ones(s.n, B)
    def get_coms(s): return torch.zeros(s.n, B, 7)
    def get_material_properties(s): return torch.ones(s.n, 21, 3)
    def set_masses(s, m, i): assert m.shape == (s.n, B) and i.device.type == "cpu"
    def set_coms(s, c, i): assert c.shape == (s.n, B, 7)
    def set_material_properties(s, m, i): assert m.shape[0] == s.n
class D: pass
class Robot:
    def __init__(self, cfg):
        self.cfg = cfg; self.find_joints = finder(JOINTS_ISAAC); self.find_bodies = finder(BODY)
    def setup(self, n):
        d = D(); self.data = d; self.root_physx_view = View(n); self._ALL_INDICES = torch.arange(n)
        d.joint_pos = torch.zeros(n, 12); d.joint_vel = torch.zeros(n, 12)
        d.root_pos_w = torch.zeros(n, 3); d.root_quat_w = torch.tensor([1., 0, 0, 0]).repeat(n, 1)
        d.root_lin_vel_b = torch.zeros(n, 3); d.root_ang_vel_b = torch.zeros(n, 3)
        d.body_pos_w = torch.zeros(n, B, 3); d.body_lin_vel_w = torch.zeros(n, B, 3)
        d.default_root_state = torch.zeros(n, 13); d.default_root_state[:, 3] = 1
        d.default_joint_pos = torch.zeros(n, 12); self.effort = torch.zeros(n, 12)
    def set_joint_effort_target(self, t, joint_ids=None): self.effort[:, joint_ids] = t
    def set_external_force_and_torque(self, f, t, positions=None, body_ids=None, env_ids=None, is_global=False):
        assert f.shape[-1] == 3
    def write_root_pose_to_sim(self, p, env_ids=None):
        self.data.root_pos_w[env_ids] = p[:, :3]; self.data.root_quat_w[env_ids] = p[:, 3:7]
    def write_root_velocity_to_sim(self, v, env_ids=None): self.data.root_lin_vel_b[env_ids] = v[:, :3]
    def write_joint_state_to_sim(self, p, v, joint_ids=None, env_ids=None):
        self.data.joint_pos[env_ids] = p; self.data.joint_vel[env_ids] = v
class Sensor:
    num_bodies = B
    def __init__(self, cfg): self.find_bodies = finder(BODY)
class FakeWorld:
    env = None; n = 0
    @classmethod
    def init(c, env):
        c.env = env; env.robot.setup(c.n); env.contact_sensor.data = D()
        env.contact_sensor.data.net_forces_w = torch.zeros(c.n, B, 3)
    @classmethod
    def step(c):
        e = c.env; d = e.robot.data; dt = e.physics_dt
        acc = e.robot.effort * 20.0 - 2.0 * d.joint_vel
        d.joint_vel += acc * dt; d.joint_pos += d.joint_vel * dt
        xy = d.root_pos_w[:, :2]; ground = e.terrain.heights_at(xy)
        d.root_pos_w[:, 2] += (ground + 0.3 - d.root_pos_w[:, 2]) * 0.2
        d.root_lin_vel_b = torch.randn(c.n, 3) * 0.05 + torch.tensor([0.3, 0, 0])
        d.root_pos_w[:, 0] += 0.3 * dt
        d.body_pos_w = d.root_pos_w[:, None, :] + torch.randn(c.n, B, 3) * 0.05
        d.body_pos_w[:, e._foot_ids, 2] = e.terrain.heights_at(d.body_pos_w[:, e._foot_ids, :2]) + 0.02
        f = torch.zeros(c.n, B, 3); f[:, e._sensor_foot_ids, 2] = 30.0 * (torch.rand(c.n, 4) > 0.5)
        e.contact_sensor.data.net_forces_w = f

def install():
    mod("isaaclab"); mod("isaaclab.app", AppLauncher=App)
    mod("isaaclab.sim", RigidBodyMaterialCfg=Any_, PreviewSurfaceCfg=Any_, DomeLightCfg=Any_, SimulationCfg=SimCfg)
    mod("isaaclab.actuators", IdealPDActuatorCfg=Any_)
    mod("isaaclab.assets", Articulation=Robot, ArticulationCfg=type("AC", (), {"InitialStateCfg": Any_}))
    mod("isaaclab.envs", DirectRLEnv=DirectRLEnv, DirectRLEnvCfg=EnvCfgBase, ViewerCfg=Any_)
    mod("isaaclab.scene", InteractiveSceneCfg=SceneCfg)
    mod("isaaclab.sensors", ContactSensor=Sensor, ContactSensorCfg=Any_)
    mod("isaaclab.terrains"); mod("isaaclab.terrains.utils", create_prim_from_mesh=lambda *a, **k: None)
    mod("isaaclab.utils", configclass=lambda c: c)
    mod("isaaclab_assets"); mod("isaaclab_assets.robots"); mod("isaaclab_assets.robots.unitree", UNITREE_A1_CFG=Any_())
    Any_.func = lambda self, *a: None
    mod("trimesh", Trimesh=lambda **k: None)
    sys.path.insert(0, PORT)
    sys.modules.setdefault("trimesh", sys.modules["trimesh"])

# Usage (CPU, no Isaac Sim):
#   python Isaac-lab-torch/tests/fake_isaac.py Isaac-lab-torch/runner.py --exptid 9001 --num_envs 8 --max_iters 2 \
#       --overwrite --headless --debug
# It checks the Python plumbing (config, base-policy loading, PPO, checkpoints, video) — NOT physics.
if __name__ == "__main__":
    install()
    script = sys.argv[1]; sys.argv = sys.argv[1:]
    import runpy; runpy.run_path(script, run_name="__main__")
