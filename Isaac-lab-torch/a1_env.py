"""Isaac Lab (Isaac Sim) port of Raisim-gym-torch/env/envs/rsg_a1_task/Environment.hpp.

The whole batch of environments runs on the GPU as torch tensors; every per-env scalar of
the C++ class becomes a (num_envs,) tensor. Joint-indexed tensors use the RaiSim joint order
(FR, FL, RR, RL x hip, thigh, calf) so observations stay compatible with the trained
policies in Base-policy/. Isaac Lab orders joints differently, so we map with `_joint_ids`.

Must be imported after `isaaclab.app.AppLauncher` has started the simulator.
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG

from terrain import KIND_DOWN, KIND_ROUGH, KIND_UP, NUM_DOWN_LEVELS, STEP_HEIGHT_LIST, TerrainBank

LEGS = ["FR", "FL", "RR", "RL"]
JOINT_NAMES = [f"{leg}_{j}_joint" for leg in LEGS for j in ("hip", "thigh", "calf")]
JT_MEAN_POS = [0.05, 0.8, -1.4, -0.05, 0.8, -1.4, 0.05, 0.8, -1.4, -0.05, 0.8, -1.4]


@configclass
class A1RmaEnvCfg(DirectRLEnvCfg):
    # RaiSim: simulation_dt 0.0025, control_dt 0.01
    sim: SimulationCfg = SimulationCfg(
        dt=0.0025,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )
    decimation = 4
    episode_length_s = 12.0  # termination at 1200 steps is handled in _get_dones
    action_space = 12
    observation_space = 140
    state_space = 0
    # chase camera on env 0, used by env.render() for videos (needs --enable_cameras)
    viewer: ViewerCfg = ViewerCfg(eye=(1.5, -1.5, 0.8), lookat=(0.0, 0.0, 0.0), origin_type="asset_root",
                                  asset_name="robot", env_index=0, resolution=(1280, 720))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=200, env_spacing=2.0, replicate_physics=True)

    robot: ArticulationCfg = UNITREE_A1_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            joint_pos={n: v for n, v in zip(JOINT_NAMES, JT_MEAN_POS)},
        ),
        # PD is computed in _apply_action (per-env random gains), so the actuator only
        # forwards and clips the effort.
        actuators={"legs": IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0,
                                              effort_limit=33.5, velocity_limit=21.0)},
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.0)

    base_body = "trunk"  # A1 base link is named "trunk" in the Isaac Lab asset ("base" in RaiSim)
    foot_bodies = [f"{leg}_foot" for leg in LEGS]
    calf_bodies = [f"{leg}_calf" for leg in LEGS]

    # terrain bank
    tile_size = 8.0
    horizontal_scale = 0.05
    num_variants = 5
    terrain_seed = 0

    spawn_height = 0.4  # gc_init_[2] = terrain + 0.4
    settle_steps = 50   # physics steps holding the default pose after a full reset

    # filled from the RaiSim yaml (cfg['environment']) by `apply_yaml`
    task: dict = {}

    def apply_yaml(self, env_cfg: dict):
        self.task = dict(env_cfg)
        self.scene.num_envs = int(env_cfg["num_envs"])
        self.sim.dt = float(env_cfg["simulation_dt"])
        self.decimation = int(round(env_cfg["control_dt"] / env_cfg["simulation_dt"]))
        self.sim.render_interval = self.decimation


class A1RmaEnv(DirectRLEnv):
    cfg: A1RmaEnvCfg

    def __init__(self, cfg: A1RmaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        t = cfg.task
        self.isTest = bool(t.get("test", False))
        self.isEval = bool(t.get("eval", False))
        self.privinfo = bool(t["privinfo"])
        self.includeGRF = float(bool(t["includeGRF"]))
        self.use_slope_dots = bool(t["use_slope_dots"])
        self.rand_friction = bool(t["randomize_friction"])
        self.rand_mass = bool(t["randomize_mass"])
        self.rand_motor = bool(t["randomize_motor_strength"])
        self.rand_gains = bool(t["randomize_gains"])
        self.pid_coeff = float(t["pid_coeff"])
        self.slope_th = float(t["slope_threshold"])
        self.vel_avg_w = float(t["bodyLinearVel_avg_weight"])
        self.c = {k: float(t[k]) for k in t if k.endswith("RewardCoeff")}
        self.cost_coeff = float(t["cost_coeff"])
        self.cost_decay_fac = float(t["cost_decay_fac"])
        # NOTE: Environment.hpp reads cfg["aliveBonus"], a key absent from cfg.yaml (only
        # "alive_bonus" exists); mini-yaml returns 0 for a missing key, so the RaiSim teacher
        # ran with no alive bonus. Kept faithful; add `aliveBonus: 18` to the yaml to enable it
        # (the DAgger env reads "alive_bonus" = 18).
        self.alive_bonus = float(t.get("aliveBonus", 0.0))
        self.itr_number = 0
        self.start_itr_stairs = 1200
        self.g1_position = [0.05]
        self.num_g1 = len(self.g1_position)
        self.n_slope = 4 + 2 * (self.num_g1 - 1)
        half = 42 + int(self.privinfo) * (20 + 3 * int(t["use_priv_vel"]) + 4 * int(self.use_slope_dots) + 1 + 2 * (self.num_g1 - 1))
        assert 2 * half == cfg.observation_space, f"obs dim {2 * half} != {cfg.observation_space}"

        n, d = self.num_envs, self.device
        self._joint_ids, _ = self.robot.find_joints(JOINT_NAMES, preserve_order=True)
        self._base_id = self.robot.find_bodies(cfg.base_body)[0][0]
        self._foot_ids, _ = self.robot.find_bodies(cfg.foot_bodies, preserve_order=True)
        self._sensor_foot_ids, _ = self.contact_sensor.find_bodies(cfg.foot_bodies, preserve_order=True)
        self._sensor_calf_ids, _ = self.contact_sensor.find_bodies(cfg.calf_bodies, preserve_order=True)
        allowed = set(self._sensor_foot_ids) | set(self._sensor_calf_ids)
        self._sensor_illegal_ids = [i for i in range(self.contact_sensor.num_bodies) if i not in allowed]

        z = lambda *s: torch.zeros(n, *s, device=d)
        self.action_mean = torch.tensor(JT_MEAN_POS, device=d).repeat(n, 1)
        self.action_std = 0.4
        self.act_hist = z(3, 12)          # last 3 actions, [-1] is newest
        self.p_target = self.action_mean.clone()
        self.kp, self.kd = z(), z()
        self.torque = z(12)
        self.last_torque = z(12)
        self.motor_strength = torch.ones(n, 12, device=d)
        self.mass_params = z(3)
        self.friction = torch.full((n,), 0.8, device=d)
        self.speed_vec = z(4)
        self.max_speed, self.ang_speed = z(), z()
        self.fwd_coeff, self.ang_coeff = z(), z()
        self.action_coeff = torch.full((n,), self.c["actionRewardCoeff"], device=d)
        self.upward_coeff = torch.full((n,), self.c["upwardRewardCoeff"], device=d)
        self.step_counter = torch.zeros(n, dtype=torch.long, device=d)
        self.res_goal = torch.zeros(n, dtype=torch.long, device=d)
        self.res_env = torch.zeros(n, dtype=torch.long, device=d)
        self.lin_vel_b, self.ang_vel_b, self.euler = z(3), z(3), z(3)
        self.avg_xy, self.avg_yaw = z(2), z()
        self.slope_dots = z(self.n_slope)
        self.grf_bin_obs = z(4)
        self.last_contact = z(4)
        self.last_foot_state = z(4)
        self.prev_xy = z(2)
        self.reward = z()
        self.reward_info = z(16)
        self.push_force = z(1, 3)
        self.push_pending = torch.zeros(n, dtype=torch.bool, device=d)
        # terrain state (RaiSim starts every env on random rough terrain)
        self.kind = torch.zeros(n, dtype=torch.long, device=d)
        self.level = torch.zeros(n, dtype=torch.long, device=d)
        self.variant = torch.randint(0, cfg.num_variants, (n,), device=d)
        self.isSlope = torch.zeros(n, dtype=torch.bool, device=d)
        self.sampleCmds = torch.ones(n, dtype=torch.bool, device=d)
        self.useRef = torch.ones(n, dtype=torch.bool, device=d)
        self.isDown = torch.zeros(n, dtype=torch.bool, device=d)
        self.canonical_step_height = torch.full((n,), 0.1, device=d)
        self.resample_on_reset = False

        # physx-side buffers (CPU) for mass / COM / friction randomization
        view = self.robot.root_physx_view
        self._masses = view.get_masses().clone()
        self._coms = view.get_coms().clone()
        self._materials = view.get_material_properties().clone()

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        self.terrain = TerrainBank(self.cfg.tile_size, self.cfg.horizontal_scale, num_variants=self.cfg.num_variants,
                                   stairs_noise=not bool(self.cfg.task.get("test", False)),
                                   seed=self.cfg.terrain_seed, device=self.device)
        import trimesh
        verts, tris = self.terrain.mesh()
        create_prim_from_mesh("/World/ground", trimesh.Trimesh(vertices=verts, faces=tris, process=False),
                              visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.4)),
                              physics_material=self.cfg.sim.physics_material)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------- helpers
    def _rand(self, *shape):
        """Eigen::VectorXd::Random -> U[-1, 1]"""
        return torch.rand(*shape, device=self.device) * 2 - 1

    def _joint_pos(self):
        return self.robot.data.joint_pos[:, self._joint_ids]

    def _joint_vel(self):
        return self.robot.data.joint_vel[:, self._joint_ids]

    @staticmethod
    def _quat_to_euler(q):
        w, x, y, z = q.unbind(-1)
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return torch.stack([roll, pitch, yaw], -1)

    def sample_goals(self, ids):
        m = len(ids)
        if m == 0:
            return
        stop = self._rand(m) > 0.6
        sample = self.sampleCmds[ids]
        dms = torch.where(stop, -0.5 + 0.15 * (self._rand(m) / 2 + 0.5), -0.05 + 0.1 * self._rand(m))
        das = torch.where(stop, torch.zeros(m, device=self.device), 0.4 * self._rand(m))
        self.action_coeff[ids] = torch.where(stop, -1.0, 0.0)
        self.upward_coeff[ids] = torch.where(stop, -3.0, 0.0)
        fwd, ang = self.c["forwardVelRewardCoeff"], self.c["angularVelRewardCoeff"]
        self.fwd_coeff[ids] = torch.where(stop & sample, 2 * fwd, fwd)
        self.ang_coeff[ids] = torch.where(stop, 2 * ang, ang)
        dms = torch.where(sample, dms, -0.05 + 0.05 * self._rand(m))
        das = torch.where(sample, das, 0.2 * self._rand(m))
        self.max_speed[ids] = 0.5 + dms
        self.ang_speed[ids] = das
        self.speed_vec[ids] = torch.stack([dms, das, dms, das], -1)
        if self.isTest:
            print(f"speed_vec: {(dms + 0.5).tolist()}, {das.tolist()}")

    def randomize_sim_params(self, ids):
        m = len(ids)
        if m == 0:
            return
        d = self.device
        if self.rand_friction:
            fl = torch.tensor([0.5, 0.7, 0.9, 1.2, 1.5, 1.8], device=d)
            self.friction[ids] = fl[torch.randint(0, 6, (m,), device=d)]
        else:
            self.friction[ids] = 0.8
        ids_cpu = ids.cpu()
        self._materials[ids_cpu, :, 0] = self.friction[ids].cpu()[:, None]
        self._materials[ids_cpu, :, 1] = self.friction[ids].cpu()[:, None]
        self.robot.root_physx_view.set_material_properties(self._materials, ids_cpu)

        if self.rand_gains:
            self.kp[ids] = self.pid_coeff + 5.0 * self._rand(m)
            self.kd[ids] = 0.6 + 0.2 * self._rand(m)
        else:
            self.kp[ids], self.kd[ids] = self.pid_coeff, 0.6

        if self.rand_mass:
            mass_rand = 0.5 * (1 + self._rand(m))
            com = torch.stack([0.1 * self._rand(m), 0.05 * self._rand(m), torch.zeros(m, device=d)], -1)
            self._masses[ids_cpu, self._base_id] = (6.0 + 2.0 * mass_rand).cpu()
            self._coms[ids_cpu, self._base_id, :3] = com.cpu()
            self.robot.root_physx_view.set_masses(self._masses, ids_cpu)
            self.robot.root_physx_view.set_coms(self._coms, ids_cpu)
            self.mass_params[ids] = torch.stack([mass_rand, com[:, 0], com[:, 1]], -1)
        else:
            self.mass_params[ids] = torch.tensor([0.5, 0.0, 0.0], device=d)

        self.motor_strength[ids] = (1 - 0.1 * self._rand(m, 12)) if self.rand_motor else 1.0

    def select_terrain(self, ids):
        """Equivalent of the terrain switch in reset(): pick a new tile for envs `ids`."""
        m = len(ids)
        d = self.device
        if self.itr_number > self.start_itr_stairs or self.isTest:
            sel = torch.randint(0, 100, (m,), device=d)
        else:
            sel = torch.zeros(m, dtype=torch.long, device=d)
        stairs = sel >= (30 if self.isEval else 40)
        down = (self._rand(m) > 0) if self.isEval else (self._rand(m) > -0.5)
        max_idx = min(max(1, (self.itr_number - self.start_itr_stairs) // 1000), len(STEP_HEIGHT_LIST))
        max_down = min(max_idx, NUM_DOWN_LEVELS)
        lvl_up = torch.randint(0, max_idx, (m,), device=d)
        lvl_down = torch.randint(0, max_down, (m,), device=d)
        kind = torch.where(stairs, torch.where(down, KIND_DOWN, KIND_UP), KIND_ROUGH)
        level = torch.where(kind == KIND_DOWN, lvl_down, torch.where(kind == KIND_UP, lvl_up, 0))
        self.kind[ids], self.level[ids] = kind, level
        self.variant[ids] = torch.randint(0, self.cfg.num_variants, (m,), device=d)
        self.isSlope[ids] = stairs
        self.sampleCmds[ids] = ~stairs
        self.useRef[ids] = kind != KIND_DOWN
        self.isDown[ids] = kind == KIND_DOWN
        heights = torch.tensor(STEP_HEIGHT_LIST, device=d)
        self.canonical_step_height[ids] = torch.where(stairs, heights[level], self.canonical_step_height[ids])

    def compute_slope_info(self, ids):
        """Height differences in front of the front feet and between rear and front feet."""
        dots = torch.zeros(len(ids), self.n_slope, device=self.device)
        feet = self.robot.data.body_pos_w[ids][:, self._foot_ids, :2]
        yaw = self.euler[ids, 2]
        heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)[:, None, :]
        cols = []
        for pd in self.g1_position:
            front = feet[:, :2] + pd * heading
            back = feet[:, :2] + (pd - 0.05) * heading
            cols.append(self.terrain.heights_at(front) - self.terrain.heights_at(back))
        cols.append(self.terrain.heights_at(feet[:, 2:]) - self.terrain.heights_at(feet[:, :2]))
        diff = torch.cat(cols, -1)
        diff = torch.where(diff.abs() < self.slope_th, torch.zeros_like(diff), diff)
        dots[:] = torch.where(self.isSlope[ids, None], diff, dots)
        self.slope_dots[ids] = dots

    def update_state(self, ids, force_slope=False):
        """updateObservation() minus the observation vector assembly."""
        data = self.robot.data
        self.lin_vel_b[ids] = data.root_lin_vel_b[ids]
        self.ang_vel_b[ids] = data.root_ang_vel_b[ids]
        self.euler[ids] = self._quat_to_euler(data.root_quat_w[ids])
        first = self.step_counter[ids] <= 1
        w = self.vel_avg_w
        new_xy = (1 - w) * self.avg_xy[ids] + w * self.lin_vel_b[ids, :2]
        new_yaw = (1 - w) * self.avg_yaw[ids] + w * self.ang_vel_b[ids, 2]
        self.avg_xy[ids] = torch.where(first[:, None], self.lin_vel_b[ids, :2], new_xy)
        self.avg_yaw[ids] = torch.where(first, self.ang_vel_b[ids, 2], new_yaw)
        if self.use_slope_dots:
            # RaiSim computes every 5 steps. After a reset the foot poses are only valid after
            # one physics step, so we also force a refresh at step 1.
            due = (self.step_counter[ids] % 5 == 0) | (self.step_counter[ids] == 1) | force_slope
            if due.any():
                self.compute_slope_info(ids[due])

    def settle(self):
        """Hold the default pose for `settle_steps` physics steps (RaiSim reset(): 50 integrate)."""
        self.p_target[:] = self.action_mean
        for _ in range(self.cfg.settle_steps):
            self._apply_pd()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        all_ids = torch.arange(self.num_envs, device=self.device)
        self.update_state(all_ids, force_slope=True)
        self.prev_xy[:] = self.robot.data.root_pos_w[:, :2]
        self.obs_buf = self._get_observations()
        # sanity check: A1 in the default pose stands ~0.3 m above the ground
        base_h = self.robot.data.root_pos_w[:, 2] - self.terrain.heights_at(self.robot.data.root_pos_w[:, :2])
        print(f"[settle] base height above terrain: mean {base_h.mean().item():.3f} m, "
              f"min {base_h.min().item():.3f} m (expect ~0.25-0.33 on flat ground)")

    def curriculum_update(self):
        self.cost_coeff = self.cost_coeff ** self.cost_decay_fac
        self.itr_number += 1

    def render(self, recompute: bool = False, mode: str | None = None):
        # `mode` lets nv-omni-tr-pipeline's video_recorder call env.render(mode="rgb_array")
        return super().render(recompute=recompute)

    # ------------------------------------------------------- DirectRLEnv API
    def _pre_physics_step(self, actions: torch.Tensor):
        self.step_counter += 1
        sc = self.step_counter
        self.sample_goals(((sc % (300 + self.res_goal)) == 300 + self.res_goal - 1).nonzero().squeeze(-1))
        self.randomize_sim_params(((sc % (50 + self.res_env)) == 50 + self.res_env - 1).nonzero().squeeze(-1))
        self.act_hist = torch.cat([self.act_hist[:, 1:], actions[:, None].clone()], 1)
        action = self.act_hist[:, 0]  # act_history[size - 3]: two control steps of delay
        self.p_target = action * self.motor_strength * self.action_std + self.action_mean
        self._substep = 0

    def _apply_pd(self):
        q, qd = self._joint_pos(), self._joint_vel()
        tau = self.kp[:, None] * (self.p_target - q) - self.kd[:, None] * qd
        self.torque = tau.clamp(-33.5, 33.5)
        self.robot.set_joint_effort_target(self.torque, joint_ids=self._joint_ids)

    def _apply_action(self):
        self._apply_pd()
        # RaiSim external forces only act for one integration step
        if self._substep == 0 and self.push_pending.any():
            self.robot.set_external_force_and_torque(self.push_force, torch.zeros_like(self.push_force),
                                                     body_ids=[self._base_id])
        elif self._substep == 1 and self.push_pending.any():
            self.push_force.zero_()
            self.push_pending.zero_()
            self.robot.set_external_force_and_torque(self.push_force, self.push_force.clone(), body_ids=[self._base_id])
        self._substep += 1

    def _get_dones(self):
        """Post-physics part of step(): state update, rewards (cached) and terminations."""
        n, d = self.num_envs, self.device
        all_ids = torch.arange(n, device=d)
        data = self.robot.data
        forces = self.contact_sensor.data.net_forces_w
        impulse = (forces[:, self._sensor_foot_ids].norm(dim=-1)
                   + forces[:, self._sensor_calf_ids].norm(dim=-1)) * self.physics_dt
        grf = impulse
        grf_bin = (grf > 0).float()
        swing_bin = 1 - grf_bin
        foot_vel = data.body_lin_vel_w[:, self._foot_ids].square().sum(-1)
        foot_z = data.body_pos_w[:, self._foot_ids, 2]
        foot_pos_bin = (foot_z > 0.04).float()
        foot_pos_err = (foot_z - 0.04).square()
        self.grf_bin_obs = grf_bin

        # applyExternalForceRandomly (only while the command is "stop")
        push = (self.max_speed.abs() < 0.15) & (self.step_counter >= 200) & (self.step_counter % 100 == 0)
        if push.any():
            f = 500 * self._rand(n, 3)
            f[:, :2] += torch.sign(f[:, :2]) * 400.0
            f[:, 2] = 0
            self.push_force[:, 0] = torch.where(push[:, None], f, torch.zeros_like(f))
            self.push_pending |= push

        self.update_state(all_ids)
        tau = self.torque
        qd = self._joint_vel()
        action = self.act_hist[:, 0]

        cur_foot_state = torch.where(foot_pos_bin == 1, 1.0, torch.where(swing_bin == 0, -1.0, self.last_foot_state))
        contact_changes = ((cur_foot_state + self.last_foot_state) == 0).float().sum(-1)

        vx, vy, wz = self.lin_vel_b[:, 0], self.lin_vel_b[:, 1], self.ang_vel_b[:, 2]
        forward_r = self.fwd_coeff * torch.minimum(self.max_speed, vx)
        too_fast = self.isDown & (vx > self.max_speed + 0.1)
        forward_r = forward_r - torch.where(too_fast, self.fwd_coeff * (vx - self.max_speed), torch.zeros_like(vx))
        angular_r = self.ang_coeff * (-(self.ang_speed - wz).abs() + self.ang_speed.abs())
        forward_reward = forward_r - vy.abs() + angular_r + self.alive_bonus / (2.0 - self.isSlope.float())

        dgrf = grf - self.last_contact
        delta_contact = dgrf.clamp(min=0).square().sum(-1)
        delta_release = dgrf.clamp(max=0).square().sum(-1)
        contact = grf.square().sum(-1)
        stance = swing_bin == 0
        gmax = torch.where(stance, grf, torch.zeros_like(grf)).max(-1).values.clamp(min=0)
        gmin = torch.where(stance, grf, torch.full_like(grf, 100.0)).min(-1).values
        gmin = torch.where(gmin == 100.0, torch.zeros_like(gmin), gmin)
        contact_dist = (gmax - gmin).square()
        torque_sq = tau.square().sum(-1)
        torque_r = torque_sq - tau[:, [0, 3, 6, 9]].square().sum(-1)
        delta_torque = (tau - self.last_torque).square().sum(-1)
        action_r = action.square().sum(-1)
        sideways = action[:, [0, 3, 6, 9]].square().sum(-1)
        joint_speed = qd.square().sum(-1)
        foot_slip = (foot_vel * grf_bin).sum(-1) * torch.where(self.isDown, 4.0, 1.0)
        foot_clear = (swing_bin * foot_pos_err).sum(-1)
        upward = self.euler[:, :2].square().sum(-1)
        work = (qd * tau).sum(-1)
        y_acc = self.lin_vel_b[:, 2].square()

        self.last_torque = tau.clone()
        self.last_contact = grf
        self.last_foot_state = cur_foot_state

        c = self.c
        cost = (foot_slip * c["footSlipRewardCoeff"] + torque_r * c["torqueRewardCoeff"]
                + contact * c["contactRewardCoeff"] + delta_contact * c["deltaContactRewardCoeff"]
                + delta_release * c["deltaReleaseRewardCoeff"] + delta_torque * c["deltaTorqueRewardCoeff"]
                + action_r * self.action_coeff + joint_speed * c["jointSpeedRewardCoeff"]
                + foot_clear * c["footClearenceRewardCoeff"] + upward * self.upward_coeff
                + y_acc * c["yAccRewardCoeff"] + contact_dist * c["contactDistRewardCoeff"]
                + contact_changes * c["contactChangeRewardCoeff"] + sideways * c["sidewaysRewardCoeff"])
        r = (forward_reward + work * c["workRewardCoeff"] + self.cost_coeff * cost) / 100
        r = torch.where(torch.isnan(r), torch.full_like(r, -50.0), r)
        r = torch.where((r > 100) | (r < -100), torch.full_like(r, -10.0), r)
        self.reward = r

        xy = data.root_pos_w[:, :2]
        walked = (xy - self.prev_xy).norm(dim=-1)
        self.prev_xy = xy.clone()
        zero = torch.zeros_like(r)
        self.reward_info = torch.stack([forward_reward, zero, zero, delta_torque, action_r, sideways, joint_speed,
                                        delta_contact, delta_release, foot_slip, upward, work, y_acc, torque_sq,
                                        self.canonical_step_height, walked], -1)

        # isTerminalState
        illegal = forces[:, self._sensor_illegal_ids].norm(dim=-1).max(-1).values > 0
        loose = self.isSlope | self.isTest | self.isEval
        term_pitch = torch.where(loose, 0.8, 0.2)
        bad_orient = (self.euler[:, 0].abs() > 0.6) | (self.euler[:, 1].abs() > term_pitch)
        base_h = data.root_pos_w[:, 2] - self.terrain.heights_at(xy)
        too_low = base_h < torch.where(loose, 0.1, 0.24)
        yaw_off = (~self.sampleCmds) & (not self.isEval) & (self.euler[:, 2].abs() > 0.5)
        terminated = illegal | bad_orient | too_low | yaw_off
        time_out = (self.step_counter > 1200) & ~terminated
        return terminated, time_out

    def _get_rewards(self):
        # terminalRewardCoeff_ = -10, not scaled by 1/100 (as in RaiSim)
        return self.reward + torch.where(self.reset_terminated, -10.0, 0.0)

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        ids = env_ids.to(self.device)
        m = len(ids)
        change_freq = 50 if self.itr_number > self.start_itr_stairs else 100
        if self.itr_number % change_freq == 0 or self.isTest:
            self.select_terrain(ids)
        self.randomize_sim_params(ids)
        self.avg_xy[ids], self.avg_yaw[ids] = 0.0, 0.0
        self.step_counter[ids] = 0
        if self.resample_on_reset or self.isTest:
            self.sample_goals(ids)

        xy = self.terrain.spawn_xy(self.kind[ids], self.level[ids], self.variant[ids])
        root = self.robot.data.default_root_state[ids].clone()
        root[:, :2] = xy
        root[:, 2] = self.terrain.heights_at(xy) + self.cfg.spawn_height
        root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        # RaiSim gv_init = 0.1 * U[-1,1] for all 18 dofs (angular part is in world frame)
        root[:, 7:13] = 0.1 * self._rand(m, 6)
        self.robot.write_root_pose_to_sim(root[:, :7], ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:13], ids)
        jp = self.robot.data.default_joint_pos[ids].clone()
        jv = torch.zeros_like(jp)
        jp[:, self._joint_ids] = self.action_mean[ids]
        jv[:, self._joint_ids] = 0.1 * self._rand(m, 12)
        self.robot.write_joint_state_to_sim(jp, jv, None, ids)

        self.act_hist[ids] = 0.0
        self.p_target[ids] = self.action_mean[ids]
        self.slope_dots[ids] = 0.0
        self.push_force[ids] = 0.0
        self.push_pending[ids] = False
        self.prev_xy[ids] = xy
        self.update_state(ids)
        self.res_goal[ids] = torch.randint(0, 50, (m,), device=self.device)
        self.res_env[ids] = torch.randint(0, 10, (m,), device=self.device)

    def _get_observations(self):
        half = torch.cat([
            self.euler[:, :2], self._joint_pos(), self._joint_vel(), self.act_hist[:, -1], self.speed_vec,
            self.includeGRF * self.grf_bin_obs, self.mass_params, self.motor_strength, self.friction[:, None],
            self.avg_xy, self.avg_yaw[:, None], self.slope_dots, 1.0 - self.useRef.float()[:, None],
        ], -1)
        return {"policy": torch.cat([half, half], -1)}
