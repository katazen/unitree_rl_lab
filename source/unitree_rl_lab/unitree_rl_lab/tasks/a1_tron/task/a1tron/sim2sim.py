import time
from collections import deque
import mujoco
import mujoco.viewer
import numpy as np
import torch
from pynput import keyboard


class SimToSimCfg:
    class path:
        # TODO: 填上你的 mjcf 和 policy 路径
        pos_xml_path = '/home/woan/workspace/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/A1_legs_V1/mjcf/A1_legs_V1.xml'
        tau_xml_path = '/home/woan/workspace/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/A1_legs_V1/mjcf/A1_legs_V1.xml'
        model_path = '/home/woan/workspace/unitree_rl_lab/logs/rsl_rl/a1/aliyun_dsw/05271131/policy.pt'

    class sim:
        sim_duration = 10000.0
        control_mode = "motor"  # "position" or "motor"
        action_dim = 12
        # 单帧 obs 拼接顺序 (与 PolicyCfg term 声明顺序一致):
        #   ang_vel(3) + gravity(3) + base_command(4) + joint_pos(12) + joint_vel(12) + last_action(12) + gait_phase(2) = 48
        # base_command = [vx, vy, wz, freq], 整段都进 actor 输入
        state_dim = 3 + 3 + 4 + 3 * action_dim + 2
        dt = 0.005
        decimation = 4
        his_lens = 10  # 与 PolicyCfg.history_length 对齐
        obs_slices = {
            "ang_vel":      (0,                      3),
            "gravity":      (3,                      6),
            "base_command": (6,                      10),
            "joint_pos":    (10,                     10 + action_dim),
            "joint_vel":    (10 + action_dim,        10 + 2 * action_dim),
            "actions":      (10 + 2 * action_dim,    10 + 3 * action_dim),
            "gait_phase":   (10 + 3 * action_dim,    12 + 3 * action_dim),
        }
        # PolicyCfg 里配置的 obs scale (与训练一致)
        scale_ang_vel = 0.25
        scale_gravity = 1.0
        scale_joint_pos = 1.0
        scale_joint_vel = 0.05
        scale_actions = 1.0

    class robot:
        # 与 a1leg 完全相同的本体
        default_dof_pos = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0,
                                    -0.1, 0.0, 0.0, 0.3, -0.2, 0])
        reset_dof_pos = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0,
                                  -0.1, 0.0, 0.0, 0.3, -0.2, 0])
        armature = np.array([0.01] * 12)
        effort = np.array([27, 27, 27, 27, 7, 7,
                           27, 27, 27, 27, 7, 7])
        stiffness = np.array([45., 45, 45, 45, 45, 45,
                              45., 45, 45, 45, 45, 45])
        # damping = np.array([2.5] * 12)

        damping = np.array([1.5, 1.5, 1.5, 1.5, 0.8, 0.8,
                            1.5, 1.5, 1.5, 1.5, 0.8, 0.8])
        action_scale = 0.25

        # base_command 范围 [vx, vy, wz, freq], 与训练 UniformBaseCommandCfg.Ranges 对齐
        cmd_range = [
            (-0.5, 1.0),   # vx
            (-0.5, 0.5),   # vy
            (-0.5, 0.5),   # wz
            ( 0.8, 1.6),   # freq (Hz)
        ]


class LatencySimulator:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.action_buffer = None

    def reset(self, action_delay_range: tuple):
        act_delay = np.random.randint(*action_delay_range)
        self.action_buffer = deque(
            [np.zeros(self.action_dim, dtype=np.float32)] * (act_delay + 1),
            maxlen=act_delay + 1
        )
        print(f"[Latency] Action Delay: {act_delay} steps")

    def process_action(self, new_action: np.ndarray) -> np.ndarray:
        self.action_buffer.append(new_action.copy())
        return self.action_buffer.popleft()


class MujocoRunner:
    def __init__(self, cfg: SimToSimCfg):
        self.cfg = cfg
        used_xml = self.cfg.path.pos_xml_path if self.cfg.sim.control_mode == 'position' else self.cfg.path.tau_xml_path
        self.model = mujoco.MjModel.from_xml_path(used_xml)
        self.model.opt.timestep = self.cfg.sim.dt
        self.policy = torch.jit.load(self.cfg.path.model_path, map_location="cpu")
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 4.0
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 135
        self.latency_sim = LatencySimulator(action_dim=self.cfg.sim.action_dim)
        self.init_variables()

    def init_variables(self):
        self.action_dim = self.cfg.sim.action_dim
        self.action_scale = self.cfg.robot.action_scale
        self.control_mode = self.cfg.sim.control_mode
        print(f"action_scale: {self.action_scale}, control_mode: {self.control_mode}")
        self.state_dim = self.cfg.sim.state_dim
        self.his_lens = self.cfg.sim.his_lens
        self.obs_his = np.zeros((self.his_lens, self.state_dim))
        self.sim_duration = self.cfg.sim.sim_duration
        self.decimation = self.cfg.sim.decimation
        self.dt = self.decimation * self.cfg.sim.dt   # control step (== env.step_dt)
        self.dof_pos = np.zeros(self.action_dim)
        self.dof_vel = np.zeros(self.action_dim)
        self.action = np.zeros(self.action_dim)
        self.default_dof_pos = self.cfg.robot.default_dof_pos
        self.reset_dof_pos = self.cfg.robot.reset_dof_pos
        self.episode_length_buf = 0

        # 关节顺序: a1leg 一样 (同一台机器人)
        self.isaac_joint = ['R1', 'L1', 'R2', 'L2', 'R3', 'L3',
                            'R4', 'L4', 'R5', 'L5', 'R6', 'L6']
        self.mjc_joint = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6',
                          'L1', 'L2', 'L3', 'L4', 'L5', 'L6']
        self.mjc_ctrl = list(self.mjc_joint)

        self.isaac2mjc = np.array([self.isaac_joint.index(i) for i in self.mjc_joint])
        self.mjc2isaac = np.array([self.mjc_joint.index(i) for i in self.isaac_joint])

        # base_command: [vx, vy, wz, freq], 整段进 actor obs
        # 默认: 速度 0, 频率取范围中点
        self.base_command = np.array([
            0.0,
            0.0,
            0.0,
            0.5 * sum(self.cfg.robot.cmd_range[3]),
        ], dtype=np.float32)
        self.cmd_range = self.cfg.robot.cmd_range
        self.gait_phase = 0.0  # 标量, 训练里也是标量再展开 sin/cos

        # ---- actuator config ----
        if self.control_mode == "position":
            self.model.actuator_gainprm[:, 0] = self.cfg.robot.stiffness
            self.model.actuator_biasprm[:, 1] = -self.cfg.robot.stiffness
            self.model.dof_damping[-self.action_dim:] = self.cfg.robot.damping
        elif self.control_mode == "motor":
            self.model.actuator_gainprm[:, 0] = 1.0
            self.model.actuator_biasprm[:, 1] = 0.0
            self.model.dof_damping[-self.action_dim:] = 0.0
        self.model.dof_armature[-self.action_dim:] = self.cfg.robot.armature

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos = np.concatenate([np.array([0, 0, 0.75], dtype=np.float32),
                                         np.array([1, 0, 0, 0], dtype=np.float32),
                                         self.reset_dof_pos])
        mujoco.mj_forward(self.model, self.data)
        self.latency_sim.reset(action_delay_range=(3, 4))
        self.start_time = time.time()

    def update_obs_his(self, new_obs):
        # 滚动 history buffer: 旧帧 -> 前, 新帧 -> 末
        self.obs_his[:-1] = self.obs_his[1:]
        self.obs_his[-1] = new_obs

    def get_inference_input(self):
        # 与 IsaacLab PolicyCfg + history_length=10 + flatten_history_dim=True 的布局对齐:
        #   每个 term 内部 (T=10, dim) flatten 成 [t0_term, t1_term, ..., t9_term]
        #   不同 term 之间按 obs_slices 顺序拼接
        # 总维度 = 48 * 10 = 480
        input_parts = []
        feature_names = ["ang_vel", "gravity", "base_command",
                         "joint_pos", "joint_vel", "actions", "gait_phase"]
        for name in feature_names:
            start, end = self.cfg.sim.obs_slices[name]
            feature_block = self.obs_his[:, start:end]   # (his_lens, term_dim)
            input_parts.append(feature_block.flatten())  # row-major
        return np.concatenate(input_parts)

    def get_obs(self):
        self.dof_pos = self.data.qpos[7:].astype(np.float32)
        self.dof_vel = self.data.qvel[6:].astype(np.float32)

        s = self.cfg.sim.obs_slices
        cs = self.cfg.sim
        obs = np.zeros((self.state_dim,), dtype=np.float32)

        # ang_vel
        obs[s["ang_vel"][0]:s["ang_vel"][1]] = (
            self.data.sensor("imu_gyro").data.astype(np.double) * cs.scale_ang_vel
        )
        # projected gravity
        obs[s["gravity"][0]:s["gravity"][1]] = self.quat_rotate_inverse(
            self.data.xquat[1][[1, 2, 3, 0]].astype(np.float32),
            np.array([0.0, 0.0, -1.0])
        ) * cs.scale_gravity
        # base_command: [vx, vy, wz, freq] (无 scale)
        obs[s["base_command"][0]:s["base_command"][1]] = self.base_command
        # joint pos rel
        obs[s["joint_pos"][0]:s["joint_pos"][1]] = (
            (self.dof_pos - self.default_dof_pos)[self.mjc2isaac] * cs.scale_joint_pos
        )
        # joint vel rel
        obs[s["joint_vel"][0]:s["joint_vel"][1]] = (
            self.dof_vel[self.mjc2isaac] * cs.scale_joint_vel
        )
        # last action
        obs[s["actions"][0]:s["actions"][1]] = self.action * cs.scale_actions
        # gait_phase: [sin(2π·phase), cos(2π·phase)]
        obs[s["gait_phase"][0]] = np.sin(2 * np.pi * self.gait_phase)
        obs[s["gait_phase"][0] + 1] = np.cos(2 * np.pi * self.gait_phase)

        self.update_obs_his(obs)
        return self.get_inference_input()

    def compute_target_pos(self):
        actions_scaled = self.action * self.action_scale
        return actions_scaled[self.isaac2mjc] + self.default_dof_pos

    def compute_torque(self):
        target_pos = self.compute_target_pos()
        tau = self.cfg.robot.stiffness * (target_pos - self.dof_pos) - self.cfg.robot.damping * self.dof_vel
        return np.clip(tau, -self.cfg.robot.effort, self.cfg.robot.effort)

    def run(self):
        self.setup_keyboard_listener()
        self.listener.start()
        while self.data.time < self.sim_duration:
            input_obs = self.get_obs().flatten()
            raw_policy_action = self.policy(torch.tensor(input_obs, dtype=torch.float32)).detach().numpy()
            for _ in range(self.decimation):
                self.action[:] = self.latency_sim.process_action(raw_policy_action)
                if self.control_mode == "position":
                    self.data.ctrl = self.compute_target_pos()
                else:
                    self.dof_pos = self.data.qpos[7:].astype(np.float32)
                    self.dof_vel = self.data.qvel[6:].astype(np.float32)
                    self.data.ctrl = self.compute_torque()
                mujoco.mj_step(self.model, self.data)
            cost_time = time.time() - self.start_time
            time.sleep(max(0.0, 0.02 - cost_time))
            self.start_time = time.time()
            self.viewer.cam.lookat[:] = self.data.qpos.astype(np.float32)[0:3]
            self.viewer.sync()
            self.episode_length_buf += 1
            self.calculate_gait_para()

        self.listener.stop()
        self.viewer.close()

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w ** 2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0
        return a - b + c

    def calculate_gait_para(self):
        # 与训练对齐: gait_indices = remainder(episode_length * step_dt * freq, 1.0)
        # freq = base_command[3]
        freq = float(self.base_command[3])
        self.gait_phase = (self.episode_length_buf * self.dt * freq) % 1.0

    # ---------------- keyboard ----------------
    def adjust_base_command(self, idx: int, increment: float):
        """调整 base_command 的某一维, 整段都进 actor obs."""
        self.base_command[idx] += increment
        lo, hi = self.cmd_range[idx]
        self.base_command[idx] = float(np.clip(self.base_command[idx], lo, hi))
        names = ["vx", "vy", "wz", "freq"]
        print(f"[base_cmd] {names[idx]}={self.base_command[idx]:.3f}  full={self.base_command.tolist()}")

    def setup_keyboard_listener(self):
        def on_press(key):
            try:
                # 数字小键盘: 速度三维
                if key.char == "8":
                    self.adjust_base_command(0,  0.05)   # vx +
                elif key.char == "2":
                    self.adjust_base_command(0, -0.05)   # vx -
                elif key.char == "4":
                    self.adjust_base_command(1, -0.05)   # vy -
                elif key.char == "6":
                    self.adjust_base_command(1,  0.05)   # vy +
                elif key.char == "7":
                    self.adjust_base_command(2,  0.05)   # wz +
                elif key.char == "9":
                    self.adjust_base_command(2, -0.05)   # wz -
                # 字母键: 步频
                elif key.char == "f":
                    self.adjust_base_command(3,  0.05)   # freq +
                elif key.char == "v":
                    self.adjust_base_command(3, -0.05)   # freq -
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press)


if __name__ == "__main__":
    sim_cfg = SimToSimCfg()
    runner = MujocoRunner(cfg=sim_cfg)
    runner.run()
