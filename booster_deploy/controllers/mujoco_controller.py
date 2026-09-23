from __future__ import annotations

import sys
from collections import deque
from time import sleep
import select
import numpy as np
import torch
import mujoco
import mujoco.viewer
from booster_assets import BOOSTER_ASSETS_DIR
from .base_controller import BaseController, ControllerCfg, VelocityCommand
from ..utils.remote_control_service import RemoteControlService


class MujocoController(BaseController):
    def __init__(self, cfg: ControllerCfg):
        cfg.policy.hold_start_frame = cfg.mujoco.hold_start_frame
        super().__init__(cfg)

        # A task may supply a scene that adds bodies (e.g. an object to manipulate) around the robot. The
        # robot must still come first in the model, so its free joint stays at qpos[0:7] / qvel[0:6] and its
        # joints follow contiguously; everything below slices on that assumption.
        mjcf_path = self._expand_assets_placeholder(
            self.cfg.mujoco.scene_mjcf_path or self.robot.cfg.mjcf_path)
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        # Where the robot's own generalized coordinates end. Equal to nq/nv when nothing else is in the scene,
        # so the plain single-robot tasks are untouched.
        self._nq_robot = 7 + self.robot.num_joints
        self._nv_robot = 6 + self.robot.num_joints
        if self.mj_model.nq < self._nq_robot:
            raise ValueError(
                f"{mjcf_path} has nq={self.mj_model.nq}, too small for a {self.robot.num_joints}-dof "
                "free-floating robot")
        self.mj_model.opt.timestep = self.cfg.mujoco.physics_dt
        self.decimation = self.cfg.mujoco.decimation
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        init_dof_pos = (
            np.array(self.cfg.mujoco.init_dof_pos, dtype=np.float32)
            if self.cfg.mujoco.init_dof_pos is not None
            else self.robot.default_joint_pos.numpy()
        )
        self.mj_data.qpos[: self._nq_robot] = np.concatenate(
            [
                np.array(self.cfg.mujoco.init_pos, dtype=np.float32),
                np.array(self.cfg.mujoco.init_quat, dtype=np.float32),
                init_dof_pos,
            ]
        )
        if self.cfg.mujoco.init_dof_vel is not None:
            self.mj_data.qvel[6: self._nv_robot] = np.array(
                self.cfg.mujoco.init_dof_vel, dtype=np.float32)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # A task's manipulated object is placed by its policy when the motion
        # starts. Until then park it out of the robot's way: the scene spawns
        # it right beside the robot, where the start-up would knock it away.
        object_body_name = getattr(self.cfg.policy, "object_body_name", None)
        if object_body_name is not None and self.cfg.mujoco.hold_start_frame:
            sl = self.object_qpos_slice(object_body_name)
            parked = self.mj_data.qpos[sl].copy()
            parked[:2] = self.mj_data.qpos[:2] + np.array([-2.0, 0.0])
            self.set_object_pose(object_body_name, parked[:3], parked[3:])

        self._push_body_id = None
        if self.cfg.mujoco.push_body_name is not None:
            self._push_body_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY,
                self.cfg.mujoco.push_body_name,
            )
            if self._push_body_id < 0:
                raise ValueError(
                    f"push_body_name '{self.cfg.mujoco.push_body_name}' not "
                    "found in the MJCF."
                )
        self._push_steps_per_period = max(
            1, round(self.cfg.mujoco.push_interval_s / self.cfg.policy_dt))
        self._push_steps_duration = max(
            1, round(self.cfg.mujoco.push_duration_s / self.cfg.policy_dt))

        self._push_vel_period_steps = None
        if self.cfg.mujoco.push_vel_xy is not None:
            self._push_vel_period_steps = max(
                1, round(self.cfg.mujoco.push_interval_s / self.cfg.policy_dt))

        self._gantry_body_id = None
        if self.cfg.mujoco.gantry_body_name is not None:
            self._gantry_body_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY,
                self.cfg.mujoco.gantry_body_name,
            )
            if self._gantry_body_id < 0:
                raise ValueError(
                    f"gantry_body_name '{self.cfg.mujoco.gantry_body_name}' "
                    "not found in the MJCF."
                )
            total_weight = (
                self.mj_model.body_mass.sum()
                * abs(self.mj_model.opt.gravity[2]))
            self._gantry_force = np.array([
                0.0, 0.0,
                self.cfg.mujoco.gantry_support_fraction * total_weight,
            ])
            self._gantry_offset = np.array(
                self.cfg.mujoco.gantry_attach_offset, dtype=np.float64)
            print(f"Gantry: {self._gantry_force[2]:.1f}N up on "
                  f"'{self.cfg.mujoco.gantry_body_name}' "
                  f"({self.cfg.mujoco.gantry_support_fraction:.0%} of "
                  f"{total_weight:.1f}N)")

        self._delay_min_steps = 0
        self._delay_max_steps = 0
        if self.cfg.mujoco.actuator_delay_range_s is not None:
            lo_s, hi_s = self.cfg.mujoco.actuator_delay_range_s
            physics_dt = self.cfg.mujoco.physics_dt
            self._delay_min_steps = max(0, round(lo_s / physics_dt))
            self._delay_max_steps = max(
                self._delay_min_steps, round(hi_s / physics_dt))
        self._delay_steps = 0
        self._delay_buffer: deque | None = None

        self._velocity_limit = None
        self._knee_velocity = None
        if self.cfg.mujoco.torque_speed_curve:
            robot_cfg = self.robot.cfg
            if (robot_cfg.velocity_limit is None
                    or robot_cfg.knee_point_velocity is None):
                raise ValueError(
                    "mujoco.torque_speed_curve requires both "
                    "RobotCfg.velocity_limit and "
                    "RobotCfg.knee_point_velocity to be set."
                )
            self._velocity_limit = np.array(
                robot_cfg.velocity_limit, dtype=np.float32)
            self._knee_velocity = np.clip(
                np.array(robot_cfg.knee_point_velocity, dtype=np.float32),
                0.0, self._velocity_limit,
            )
            # Guard the v_max == v_knee case, as booster_train does.
            self._tn_denom = np.maximum(
                self._velocity_limit - self._knee_velocity, 1e-6)

        # render a second "ghost" robot (kinematic only) without
        # modifying the MuJoCo XML. This uses a second MjData to compute FK from
        # generalized coordinates and draws a duplicated set of geoms via
        # viewer.user_scn.
        self._ghost_mj_data = mujoco.MjData(self.mj_model)
        # Keep ghost initialized to the current simulated pose so it is valid
        # even before any policy calls set_reference_qpos().
        self._ghost_mj_data.qpos[:] = self.mj_data.qpos
        self._ghost_mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self._ghost_mj_data)
        self._ghost_rgba = np.array(
            self.cfg.mujoco.ghost_rgba, dtype=np.float32)
        self._ghost_scene_option = mujoco.MjvOption()

        # Reference qpos can be set explicitly by the policy.
        self._reference_qpos: np.ndarray | None = None

        # Scene bodies held fixed at a pose (see pin_object): name -> (qpos
        # slice, dof start, pinned qpos).
        self._pins: dict[str, tuple[slice, int, np.ndarray]] = {}

    def start(self):
        # Clear reference; policy.reset() may set a fresh one.
        self._reference_qpos = None
        # Draw this episode's actuator lag, matching booster_train's
        # per-reset `torch.randint(min_delay, max_delay + 1)`.
        if self._delay_max_steps > 0:
            self._delay_steps = int(np.random.randint(
                self._delay_min_steps, self._delay_max_steps + 1))
            self._delay_buffer = None  # filled from the first command
            print(f"Actuator delay this episode: {self._delay_steps} physics "
                  f"steps ({self._delay_steps * self.cfg.mujoco.physics_dt * 1e3:.0f}ms)")
        return super().start()

    def _delayed_command(self, dof_targets: np.ndarray) -> np.ndarray:
        """Advance the actuator delay buffer one physics step and return the
        lagged setpoint.

        Mirrors isaaclab's `DelayBuffer.compute`: push the newest command,
        read back the entry `_delay_steps` pushes ago. Pre-filling with the
        first command reproduces its documented warm-up behaviour (return
        the oldest available entry until the buffer has filled).
        """
        if self._delay_max_steps == 0:
            return dof_targets
        if self._delay_buffer is None:
            self._delay_buffer = deque(
                [dof_targets] * (self._delay_max_steps + 1),
                maxlen=self._delay_max_steps + 1,
            )
        self._delay_buffer.append(dof_targets)
        return self._delay_buffer[-1 - self._delay_steps]

    def render_reference_robot(
        self,
        viewer,
        # mj_data: mujoco.MjData,
        *,
        rgba: np.ndarray | None = None,
    ) -> None:
        """Render a kinematic robot pose into viewer.user_scn using mj_data."""
        mujoco.mjv_updateScene(
            self.mj_model,
            self._ghost_mj_data,
            self._ghost_scene_option,
            None,
            viewer.cam,
            int(mujoco.mjtCatBit.mjCAT_DYNAMIC),
            viewer.user_scn,
        )
        if rgba is None:
            rgba = self._ghost_rgba

        for i in range(viewer.user_scn.ngeom):
            viewer.user_scn.geoms[i].rgba[:] = rgba

    def object_qpos_slice(self, body_name: str) -> slice:
        """qpos slice of `body_name`'s free joint, for a task that spawns an object into the scene."""
        bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"no body '{body_name}' in the scene model")
        jid = self.mj_model.body_jntadr[bid]
        if jid < 0 or self.mj_model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"body '{body_name}' has no free joint, so its pose cannot be set")
        adr = int(self.mj_model.jnt_qposadr[jid])
        return slice(adr, adr + 7)

    def set_object_pose(
        self, body_name: str, pos, quat, *, ghost_pos=None, ghost_quat=None
    ) -> None:
        """Teleport a free-floating scene body and zero its velocity.

        Used at reset to put a manipulated object where the reference clip says
        it starts. ghost_pos/ghost_quat optionally keep the reference overlay at
        a different pose; otherwise it follows the real object.
        """
        sl = self.object_qpos_slice(body_name)
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        quat = np.asarray(quat, dtype=np.float64).reshape(4)
        self.mj_data.qpos[sl] = np.concatenate([pos, quat])
        dof = int(self.mj_model.jnt_dofadr[
            self.mj_model.body_jntadr[
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)]])
        self.mj_data.qvel[dof: dof + 6] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)
        if hasattr(self, "_ghost_mj_data"):
            ghost_pos = pos if ghost_pos is None else np.asarray(
                ghost_pos, dtype=np.float64).reshape(3)
            ghost_quat = quat if ghost_quat is None else np.asarray(
                ghost_quat, dtype=np.float64).reshape(4)
            self._ghost_mj_data.qpos[sl] = np.concatenate(
                [ghost_pos, ghost_quat])
            mujoco.mj_forward(self.mj_model, self._ghost_mj_data)

    def pin_object(self, body_name: str) -> None:
        """Hold a free-floating scene body fixed at its current pose, like an
        object set down by hand before the motion starts. It still collides,
        but as if immovable, until unpin_object()."""
        sl = self.object_qpos_slice(body_name)
        bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        dof = int(self.mj_model.jnt_dofadr[self.mj_model.body_jntadr[bid]])
        self._pins[body_name] = (sl, dof, self.mj_data.qpos[sl].copy())

    def unpin_object(self, body_name: str) -> None:
        self._pins.pop(body_name, None)

    def _apply_pins(self) -> None:
        for sl, dof, qpos in self._pins.values():
            self.mj_data.qpos[sl] = qpos
            self.mj_data.qvel[dof: dof + 6] = 0.0

    def set_reference_qpos(
        self,
        qpos: np.ndarray | torch.Tensor | None,
    ) -> None:
        """Set the reference generalized coordinates (qpos) for ghost rendering.

        Policies should call this each step (or whenever updated). Pass None to
        clear the reference.
        """
        if qpos is None:
            self._reference_qpos = None
            return

        if isinstance(qpos, torch.Tensor):
            qpos_np = qpos.detach().cpu().numpy()
        else:
            qpos_np = np.asarray(qpos)

        qpos_np = qpos_np.astype(np.float32, copy=False).reshape(-1)
        if qpos_np.shape[0] not in (int(self.mj_model.nq), self._nq_robot):
            raise ValueError(
                f"reference qpos must have shape (nq,) or (nq_robot,), got {qpos_np.shape} "
                f"(nq={int(self.mj_model.nq)}, nq_robot={self._nq_robot})"
            )
        # A scene task passes the robot's slice only; leave any other body's ghost where it already is.
        self._reference_qpos = self._ghost_mj_data.qpos.copy()
        self._reference_qpos[: qpos_np.shape[0]] = qpos_np
        # FK + offset
        self._ghost_mj_data.qpos[:] = self._reference_qpos
        self._ghost_mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self._ghost_mj_data)

    def _expand_assets_placeholder(self, path: str) -> str:
        """Replace {BOOSTER_ASSETS_DIR} placeholder in a path string.
        """
        try:
            return path.replace("{BOOSTER_ASSETS_DIR}", str(BOOSTER_ASSETS_DIR))
        except Exception:
            return path

    def update_vel_command(self):
        cmd: VelocityCommand = self.vel_command
        if select.select([sys.stdin], [], [], 0)[0]:
            try:
                parts = sys.stdin.readline().strip().split()
                if len(parts) == 3:
                    (cmd.lin_vel_x, cmd.lin_vel_y, cmd.ang_vel_yaw) = map(float, parts)
                    print(
                        f"Updated command to: x={cmd.lin_vel_x},"
                        f"y={cmd.lin_vel_y}, yaw={cmd.ang_vel_yaw}\n"
                        "Set command (x, y, yaw): ",
                        end="",
                    )
                else:
                    raise ValueError
            except ValueError:
                print(
                    "Invalid input. Enter three numeric values. "
                    "Set command (x, y, yaw): ",
                    end="",
                )

    def update_state(self) -> None:
        dof_pos = self.mj_data.qpos.astype(np.float32)[7: self._nq_robot]
        dof_vel = self.mj_data.qvel.astype(np.float32)[6: self._nv_robot]
        dof_torque = self.mj_data.qfrc_actuator[6:].astype(np.float32)

        base_pos_w = self.mj_data.qpos.astype(np.float32)[:3]
        base_quat = self.mj_data.qpos.astype(np.float32)[3:7]
        base_lin_vel_b = self.mj_data.qvel.astype(np.float32)[:3]
        base_ang_vel_b = self.mj_data.qvel.astype(np.float32)[3:6]

        self.robot.data.joint_pos = torch.from_numpy(
            dof_pos).to(self.robot.data.device)
        self.robot.data.joint_vel = torch.from_numpy(
            dof_vel).to(self.robot.data.device)
        self.robot.data.feedback_torque = torch.from_numpy(
            dof_torque).to(self.robot.data.device)
        self.robot.data.root_pos_w = torch.from_numpy(
            base_pos_w).to(self.robot.data.device)
        self.robot.data.root_quat_w = torch.from_numpy(
            base_quat).to(self.robot.data.device)
        self.robot.data.root_lin_vel_b = torch.from_numpy(
            base_lin_vel_b).to(self.robot.data.device)
        self.robot.data.root_ang_vel_b = torch.from_numpy(
            base_ang_vel_b).to(self.robot.data.device)

    def log_states(self, dof_targets: np.ndarray) -> None:
        if self.cfg.mujoco.log_states is not None:
            if not hasattr(self, '_states'):
                self._states = {
                    'root_pos_w': [],
                    'root_quat_w': [],
                    'root_lin_vel_b': [],
                    'root_ang_vel_b': [],
                    'joint_pos': [],
                    'joint_vel': [],
                    'joint_torque': [],
                    'dof_targets': [],
                }
            base_pos_w = self.mj_data.qpos.astype(np.float32)[:3]
            base_quat = self.mj_data.qpos.astype(np.float32)[3:7]
            base_lin_vel_b = self.mj_data.qvel.astype(np.float32)[:3]
            base_ang_vel_b = self.mj_data.qvel.astype(np.float32)[3:6]
            dof_pos = self.mj_data.qpos.astype(np.float32)[7: self._nq_robot]
            dof_vel = self.mj_data.qvel.astype(np.float32)[6: self._nv_robot]
            dof_torque = self.mj_data.qfrc_actuator[6:].astype(np.float32)

            self._states['root_pos_w'].append(base_pos_w)
            self._states['root_quat_w'].append(base_quat)
            self._states['root_lin_vel_b'].append(base_lin_vel_b)
            self._states['root_ang_vel_b'].append(base_ang_vel_b)
            self._states['joint_pos'].append(dof_pos)
            self._states['joint_vel'].append(dof_vel)
            self._states['joint_torque'].append(dof_torque)
            self._states['dof_targets'].append(dof_targets)
            if len(self._states['root_pos_w']) % 100 == 0:
                _states = {k: np.stack(v) for k, v in self._states.items()}
                np.savez(f'{self.cfg.mujoco.log_states}.npz', **_states)
                print(f'saved {self.cfg.mujoco.log_states}.npz '
                      f'at {self._step_count} steps')

    def _update_push(self) -> None:
        if self._push_body_id is None:
            return
        # Skip the first period entirely (let it stabilize first), then
        # push once per period thereafter. `step_count % period` alone
        # can't distinguish "just started" from "a full period elapsed"
        # during that first cycle, so gate on step_count directly too.
        active = (
            self._step_count > self._push_steps_per_period
            and self._step_count % self._push_steps_per_period
            < self._push_steps_duration
        )
        if active:
            self.mj_data.xfrc_applied[self._push_body_id, :3] = \
                self.cfg.mujoco.push_force
        else:
            self.mj_data.xfrc_applied[self._push_body_id, :3] = 0.0

    def _update_push_vel_kick(self) -> None:
        if self._push_vel_period_steps is None:
            return
        # Single-step event at each period boundary (not a held window),
        # matching Isaac Gym/legged_gym's push_robots(): it directly
        # overwrites root qvel[0:2] rather than applying a force.
        if (self._step_count > 0
                and self._step_count % self._push_vel_period_steps == 0):
            max_vel = self.cfg.mujoco.push_vel_xy
            kick = np.random.uniform(-max_vel, max_vel, size=2)
            self.mj_data.qvel[0:2] = kick

    def _update_gantry(self) -> None:
        """Hold a constant upward support force on the gantry body.

        Written with `=` rather than `+=` so it stays constant instead of
        accumulating across steps; if the attachment point is offset from
        the CoM, the resulting moment (r x F, with r rotated into world
        frame) is applied too, which is what gives a real harness its
        self-righting behaviour as the robot tilts.
        """
        if self._gantry_body_id is None:
            return
        self.mj_data.xfrc_applied[self._gantry_body_id, :3] = self._gantry_force
        if self._gantry_offset.any():
            rot = self.mj_data.xmat[self._gantry_body_id].reshape(3, 3)
            r_world = rot @ self._gantry_offset
            self.mj_data.xfrc_applied[self._gantry_body_id, 3:] = np.cross(
                r_world, self._gantry_force)

    def _clip_effort(
        self, effort: np.ndarray, dof_vel: np.ndarray, tau_max: np.ndarray,
    ) -> np.ndarray:
        """Clip torque to the motors' piecewise-linear torque-speed curve.

        Ported from booster_train's
        `BoosterDelayedPDActuator._clip_effort`: the ceiling is `tau_max`
        while |v| <= knee_point_velocity, then falls linearly to zero at
        velocity_limit.
        """
        if self._velocity_limit is None:
            return np.clip(effort, -tau_max, tau_max)
        tau_linear = tau_max * (
            self._velocity_limit - np.abs(dof_vel)) / self._tn_denom
        max_effort = np.clip(tau_linear, 0.0, tau_max)
        return np.clip(effort, -max_effort, max_effort)

    def ctrl_step(self, dof_targets: torch.Tensor):
        dof_targets = dof_targets.cpu().numpy()  # type: ignore
        self.log_states(dof_targets)
        self._update_push()
        self._update_push_vel_kick()
        self._update_gantry()
        if self.vel_command is not None:
            self.update_vel_command()

        self._pd_step(
            dof_targets,
            self.robot.joint_stiffness.numpy(),
            self.robot.joint_damping.numpy(),
            use_delay=True,
        )

    def _pd_step(self, dof_targets: np.ndarray, kp: np.ndarray,
                 kd: np.ndarray, use_delay: bool = False) -> None:
        """Run one control step (`decimation` physics steps) of PD tracking
        towards `dof_targets`."""
        dof_pos = self.mj_data.qpos.astype(np.float32)[7: self._nq_robot]
        dof_vel = self.mj_data.qvel.astype(np.float32)[6: self._nv_robot]
        # ctrl_limit = [
        #     np.minimum(self.mj_model.actuator_forcerange[:, 0],
        #                self.mj_model.actuator_ctrlrange[:, 0]),
        #     np.maximum(self.mj_model.actuator_forcerange[:, 1],
        #                self.mj_model.actuator_ctrlrange[:, 1]),
        # ]
        ctrl_limit = self.robot.effort_limit.numpy()
        for i in range(self.decimation):
            cmd = (self._delayed_command(dof_targets) if use_delay
                   else dof_targets)
            self.mj_data.ctrl = self._clip_effort(
                kp * (cmd - dof_pos) - kd * dof_vel, dof_vel, ctrl_limit)
            self._apply_pins()
            mujoco.mj_step(self.mj_model, self.mj_data)
            self._apply_pins()
            dof_pos = self.mj_data.qpos.astype(np.float32)[7: self._nq_robot]
            dof_vel = self.mj_data.qvel.astype(np.float32)[6: self._nv_robot]

    def _start_sequence(self, remote: RemoteControlService | None):
        """Mirror BoosterRobotPortal's custom-mode start-up before the
        policy runs, yielding once per control step so the caller can render.

        Hold the spawn pose until custom mode is requested ('x'), ramp to the
        prepare pose with the prepare-state gains, and hold it until the
        policy is started ('r'). With no `remote` (headless recording) both
        waits are skipped.
        """
        prepare_state = self.robot.cfg.prepare_state
        kp = np.array(prepare_state.stiffness, dtype=np.float32)
        kd = np.array(prepare_state.damping, dtype=np.float32)
        prepare_pos = np.array(prepare_state.joint_pos, dtype=np.float32)
        spawn_pos = self.mj_data.qpos.astype(np.float32)[7: self._nq_robot]

        def hold(target):
            self._update_gantry()
            self._pd_step(target, kp, kd)

        if remote is not None:
            print(remote.get_custom_mode_operation_hint())
            while not remote.start_custom_mode():
                hold(spawn_pos)
                yield
        num = max(1, round(1.0 / self.cfg.policy_dt))  # 1s, as on the robot
        for target in np.linspace(spawn_pos, prepare_pos, num=num,
                                  dtype=np.float32):
            hold(target)
            yield
        print("Custom mode started, initialized with prepare pose")

        if remote is not None:
            hint = remote.get_rl_gait_operation_hint()
            if (self.cfg.policy.hold_start_frame
                    and self.cfg.policy.constructor.supports_start_hold):
                hint += " It first holds the motion's first frame."
            print(hint)
            while not remote.start_rl_gait():
                hold(prepare_pos)
                yield

    def _composite_ghost_geoms(self, renderer, ghost_scene, cam) -> None:
        """Copy the ghost robot's geoms into `renderer`'s scene as an overlay.

        `mujoco.Renderer` only holds a single `MjvScene`, unlike the
        interactive viewer which merges the main scene with a separate
        `user_scn`. So here we compute the ghost's geoms into a scratch
        scene, then manually copy the geom structs onto the tail of the
        renderer's scene and bump `ngeom` to include them.
        """
        n_main = renderer._scene.ngeom
        mujoco.mjv_updateScene(
            self.mj_model,
            self._ghost_mj_data,
            self._ghost_scene_option,
            None,
            cam,
            int(mujoco.mjtCatBit.mjCAT_DYNAMIC),
            ghost_scene,
        )
        n_ghost = ghost_scene.ngeom
        scalar_fields = (
            "type", "dataid", "objtype", "objid", "category", "texid",
            "texuniform", "matid", "emission", "specular", "shininess",
            "reflectance", "transparent", "camdist", "modelrbound", "label",
        )
        array_fields = ("size", "pos", "mat", "rgba")
        for i in range(n_ghost):
            src = ghost_scene.geoms[i]
            dst = renderer._scene.geoms[n_main + i]
            for f in scalar_fields:
                setattr(dst, f, getattr(src, f))
            for f in array_fields:
                getattr(dst, f)[:] = getattr(src, f)
            dst.rgba[:] = self._ghost_rgba
        renderer._scene.ngeom = n_main + n_ghost

    def _run_offscreen(self, record_path: str):
        """Headless run: render offscreen (via EGL) and encode straight to an mp4.

        Used when there's no display for the interactive `mujoco.viewer`.
        """
        import subprocess

        width = min(self.cfg.mujoco.record_video_width, self.mj_model.vis.global_.offwidth)
        height = min(self.cfg.mujoco.record_video_height, self.mj_model.vis.global_.offheight)
        fps = self.cfg.mujoco.record_video_fps or round(1.0 / self.cfg.policy_dt)

        renderer = mujoco.Renderer(self.mj_model, height=height, width=width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        mujoco.mjv_defaultFreeCamera(self.mj_model, cam)
        cam.distance = 3.0
        cam.azimuth = 180
        cam.elevation = -20

        ghost_scene = None
        if self.cfg.mujoco.visualize_reference_ghost:
            ghost_scene = mujoco.MjvScene(self.mj_model, maxgeom=1000)

        n_steps = max(1, round(self.cfg.mujoco.record_video_seconds * fps))

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            record_path,
        ]
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        def write_frame():
            cam.lookat[:] = self.mj_data.qpos.astype(np.float32)[0:3]
            renderer.update_scene(self.mj_data, camera=cam)
            if ghost_scene is not None:
                self._composite_ghost_geoms(renderer, ghost_scene, cam)
            frame = renderer.render()
            proc.stdin.write(frame.tobytes())

        try:
            # No key input when headless: the start sequence runs unattended,
            # and a first-frame hold is released after auto_release_after_s.
            # Start-sequence frames are recorded too; record_video_seconds
            # counts policy time only.
            for _ in self._start_sequence(None):
                write_frame()

            if self.vel_command is not None:
                print("\nSet command (x, y, yaw): ", end="")
            self.update_state()
            self.start()
            release_step = round(
                self.cfg.mujoco.auto_release_after_s / self.cfg.policy_dt)
            for step in range(n_steps):
                if not self.is_running:
                    break
                if step == release_step:
                    self.request_motion_release()
                self.update_state()
                dof_targets = self.policy_step()
                self.ctrl_step(dof_targets)
                write_frame()
        finally:
            proc.stdin.close()
            proc.wait()
            renderer.close()

        print(f"Saved recording to {record_path}")

    def _run_interactive(self):
        with mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data) as viewer:

            self.viewer = viewer
            viewer.cam.azimuth = 180
            viewer.cam.elevation = -20

            def sync_viewer():
                if self.cfg.mujoco.visualize_reference_ghost:
                    # Render kinematic "ghost" robot from generalized coordinates.
                    self.render_reference_robot(
                        viewer,
                        rgba=self._ghost_rgba,
                    )

                self.viewer.cam.lookat[:] = self.mj_data.qpos.astype(np.float32)[0:3]
                self.viewer.sync()

            # Same keys as the real robot (terminal 'x' / 'r', or a gamepad),
            # plus 'g' to release a policy holding its first frame. Closed as
            # soon as nothing is held, so the terminal leaves cbreak mode, as
            # update_vel_command reads whole lines from stdin.
            remote: RemoteControlService | None = RemoteControlService()
            try:
                for _ in self._start_sequence(remote):
                    if not viewer.is_running():
                        return
                    sleep(self.cfg.policy_dt)
                    sync_viewer()

                self.update_state()
                self.start()
                if self.policy.holding:
                    print(remote.get_start_motion_operation_hint())
                while viewer.is_running() and self.is_running:
                    if remote is not None and not self.policy.holding:
                        remote.close()
                        remote = None
                        if self.vel_command is not None:
                            print("\nSet command (x, y, yaw): ", end="")
                    elif remote is not None and remote.start_motion():
                        self.request_motion_release()
                    sleep(self.cfg.mujoco.physics_dt * self.cfg.mujoco.decimation)
                    self.update_state()
                    dof_targets = self.policy_step()
                    self.ctrl_step(dof_targets)
                    sync_viewer()
            finally:
                if remote is not None:
                    remote.close()

    def run(self):
        if self.cfg.mujoco.record_video_path:
            self._run_offscreen(self.cfg.mujoco.record_video_path)
        else:
            self._run_interactive()
