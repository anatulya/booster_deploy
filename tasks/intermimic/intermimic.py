from __future__ import annotations
from dataclasses import MISSING
import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg,
    MujocoControllerCfg,
    PolicyCfg
)
from booster_deploy.robots.booster import K1_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.motion_loader import RawTensorMotionLoader


class InterMimicPolicy(Policy):
    """Body-only InterMimic tracker for K1."""

    HISTORY_LEN = 10

    def __init__(self, cfg: InterMimicPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot

        self._model: torch.jit.ScriptModule = torch.jit.load(
            f"{self.task_path}/{self.cfg.checkpoint_path}",
            map_location=self.cfg.device)
        self._model.eval()

        self.action_scale = float(
            getattr(self._model, "action_scale", self.cfg.action_scale))

        self.default_joint_pos = self.robot.default_joint_pos.to(
            self.cfg.device)

        # Reference trajectory columns are already in `joint_names` order
        # (same as the network's I/O and `robot.data.joint_pos`), so no
        # sim/real remap is needed anywhere in this policy -- unlike
        # beyond_mimic, whose motion files were authored in Isaac Sim's own
        # joint order.
        self.motion = RawTensorMotionLoader(
            motion_file=f"{self.task_path}/{self.cfg.motion_path}",
            anchor_body_name=self.cfg.anchor_body_name,
            align_to_first_frame=True,
            device=self.cfg.device,
        )

    def reset(self) -> None:
        # The motion loader canonicalizes the *reference* to yaw 0 (see
        # `_align_to_first_frame`), but nothing canonicalizes the robot --
        # on hardware `imu_state.rpy[2]` is whatever heading it happens to
        # face. `rel_ori_6d` below is relative, which makes it invariant to
        # rotating robot and reference *together*, but not to rotating only
        # one; without this the clip's absolute heading leaks into the
        # command block as a constant yaw error the policy tries to correct.
        # Captured once here (not per-step) so genuine yaw drift is still
        # observable, and yaw-only so gravity-anchored roll/pitch error is
        # untouched. Same correction beyond_mimic applies.
        # Set `yaw_align=False` to reproduce the uncorrected behaviour.
        if self.cfg.yaw_align:
            self.init_root_yaw_quat_w_inv = lab_math.quat_inv(
                lab_math.yaw_quat(self.robot.data.root_quat_w))
        else:
            self.init_root_yaw_quat_w_inv = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=torch.float32,
                device=self.cfg.device)
        self.current_frame = 0
        self.last_action = torch.zeros(
            self.robot.num_joints, dtype=torch.float32, device=self.cfg.device)
        self.motion.to(self.cfg.device)
        # Per spec: on reset, all history slots hold the *current* frame,
        # not zeros -- otherwise the first HISTORY_LEN steps after a reset
        # would diverge from what the policy saw during training.
        self._set_command()
        current_frame = self._compute_proprio_frame()
        self.obs_history = current_frame.unsqueeze(0).repeat(
            self.HISTORY_LEN, 1)

    def _set_command(self) -> None:
        """Reference values for the *current* frame (k=0 lookahead) --
        used for the live tracking error, ghost visualization, and the
        safety fallback."""
        row = min(self.current_frame, self.motion.time_step_total - 1)
        self.cmd_dof_pos = self.motion.joint_pos[row]
        self.cmd_root_pos_w = self.motion.body_pos_w[row, 0]
        self.cmd_root_quat_w = self.motion.body_quat_w[row, 0]

    def _compute_proprio_frame(self) -> torch.Tensor:
        """72-dim: projected gravity(3), base ang vel(3), joint pos(22),
        joint vel(22), previous action(22). No scaling -- raw physical
        units, matching the spec (the checkpoint's normalizer handles it).
        """
        gravity_w = torch.tensor(
            [0.0, 0.0, -1.0], dtype=torch.float32, device=self.cfg.device)
        projected_gravity = lab_math.quat_apply_inverse(
            self.robot.data.root_quat_w, gravity_w)

        # NOTE: assumes root_ang_vel_b is already body-frame, matching the
        # convention every other policy in this repo relies on (mujoco's
        # free-joint qvel angular subvector). The spec's own pseudocode
        # shows an explicit frame rotation here -- if this policy tracks
        # badly in yaw/roll despite correct joint mapping, this is the
        # first thing to revisit.
        base_ang_vel = self.robot.data.root_ang_vel_b

        # q_default == 0 for this policy (home pose is zero, unlike other
        # K1 tasks' crouched stance), so this is just the raw joint angle.
        joint_pos = self.robot.data.joint_pos - self.default_joint_pos
        joint_vel = self.robot.data.joint_vel

        projected_gravity = self._add_noise(
            projected_gravity, self.cfg.noise_projected_gravity)
        base_ang_vel = self._add_noise(
            base_ang_vel, self.cfg.noise_base_ang_vel)
        joint_pos = self._add_noise(joint_pos, self.cfg.noise_joint_pos)
        joint_vel = self._add_noise(joint_vel, self.cfg.noise_joint_vel)

        return torch.cat(
            [projected_gravity, base_ang_vel, joint_pos, joint_vel,
             self.last_action],
            dim=0,
        )

    def _add_noise(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        """Additive uniform noise on ±scale, matching the spec's training
        noise (only applied to sensor-like terms -- not last_action or the
        reference command, which aren't sensor readings)."""
        if scale == 0.0:
            return x
        return x + (torch.rand_like(x) * 2.0 - 1.0) * scale

    def _compute_future_command_block(self) -> torch.Tensor:
        """500-dim: `num_future_frames` lookahead frames x (ref joint
        pos(22), ref joint vel(22), relative root orientation 6D(6)),
        spaced `future_frame_stride` sim-frames (0.1s at 50Hz) apart,
        clamped to the clip end. k=0 is the current frame (live tracking
        error), not a true future command.
        """
        n = self.cfg.num_future_frames
        last_row = self.motion.time_step_total - 1
        rows = torch.tensor(
            [min(self.current_frame + k * self.cfg.future_frame_stride, last_row)
             for k in range(n)],
            dtype=torch.long, device=self.cfg.device,
        )

        ref_joint_pos = self.motion.joint_pos[rows]        # (n, 22) absolute
        ref_joint_vel = self.motion.joint_vel[rows]         # (n, 22)
        ref_root_quat = self.motion.body_quat_w[rows, 0]    # (n, 4) wxyz

        # Robot orientation with its start-of-episode yaw removed, so it
        # shares the reference's canonical frame (see `reset`).
        cur_quat = lab_math.quat_mul(
            self.init_root_yaw_quat_w_inv, self.robot.data.root_quat_w)
        cur_quat_inv = lab_math.quat_inv(cur_quat.unsqueeze(0)).expand(n, -1)
        rel_quat = lab_math.quat_mul(cur_quat_inv, ref_root_quat)
        rel_ori_6d = lab_math.matrix_from_quat(rel_quat)[..., :2].reshape(n, 6)

        block = torch.cat([ref_joint_pos, ref_joint_vel, rel_ori_6d], dim=-1)
        return block.reshape(-1)

    def compute_observation(self) -> torch.Tensor:
        self._set_command()

        proprio = self._compute_proprio_frame()
        self.obs_history = torch.cat(
            [self.obs_history[1:], proprio.unsqueeze(0)], dim=0)

        future_command = self._compute_future_command_block()

        obs = torch.cat(
            [self.obs_history.reshape(-1), future_command], dim=0)
        return obs.reshape(1, -1)

    def inference(self) -> torch.Tensor:
        with torch.no_grad():
            obs = self.compute_observation()
            action = self._model(obs).flatten()

        # for motion visualization in Mujoco controller
        if hasattr(self.controller, "set_reference_qpos"):
            ref_qpos = torch.cat(
                [self.cmd_root_pos_w, self.cmd_root_quat_w, self.cmd_dof_pos],
                dim=0,
            )
            self.controller.set_reference_qpos(ref_qpos)    # type: ignore

        self.current_frame += 1
        self.last_action = action

        if self.cfg.enable_safety_fallback:
            gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32,
                                     device=self.cfg.device)
            projected_gravity = lab_math.quat_apply_inverse(
                self.robot.data.root_quat_w, gravity_w)
            motion_projected_gravity = lab_math.quat_apply_inverse(
                self.cmd_root_quat_w, gravity_w)

            if torch.dot(projected_gravity, motion_projected_gravity) < 0.5:
                print("\nLarge root tracking error is detected, stopping policy"
                      " for safety. You can disable safety fallback by setting "
                      f"{self.cfg.__class__.__name__}.enable_safety_fallback "
                      "to False.")
                self.controller.stop()

        # Absolute joint target: default_joint_pos is zero for this task,
        # so this reduces exactly to the spec's "action * action_scale"
        # with no offset -- same formula shape as every other policy here.
        # No sim2real remap: network output is already in `joint_names`
        # order.
        return action * self.action_scale + self.default_joint_pos


@configclass
class InterMimicPolicyCfg(PolicyCfg):
    constructor = InterMimicPolicy
    checkpoint_path: str = MISSING
    motion_path: str = MISSING
    action_scale: float = 3.0

    anchor_body_name: str = "Trunk"
    num_future_frames: int = 10
    future_frame_stride: int = 5

    # Remove the robot's start-of-episode yaw before forming the reference's
    # relative orientation, putting robot and clip in a common frame (see
    # `InterMimicPolicy.reset`). Off = the original uncorrected behaviour,
    # which is fine only when the robot happens to spawn at yaw 0.
    yaw_align: bool = True

    # Additive uniform observation noise on ±scale, matching the spec's
    # training noise table. Off (0.0) by default; a task opts in.
    noise_projected_gravity: float = 0.0
    noise_base_ang_vel: float = 0.0
    noise_joint_pos: float = 0.0
    noise_joint_vel: float = 0.0


@configclass
class K1InterMimicControllerCfg(ControllerCfg):
    robot = K1_CFG.replace(  # type: ignore
        default_joint_pos=[0.0] * 22,
        joint_stiffness=[
            4.0, 4.0,
            4.0, 4.0, 4.0, 4.0,
            4.0, 4.0, 4.0, 4.0,
            80., 80.0, 80., 80., 30., 30.,
            80., 80.0, 80., 80., 30., 30.,
        ],
        joint_damping=[
            1., 1.,
            1., 1., 1., 1.,
            1., 1., 1., 1.,
            2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2.,
        ],
        # 0.8x the spec's raw effort-limit table -- this policy's torque
        # clip is tighter than the joints' raw hardware limit.
        effort_limit=[
            4.8, 4.8,
            11.2, 11.2, 11.2, 11.2,
            11.2, 11.2, 11.2, 11.2,
            24., 28., 16., 32., 16., 16.,
            24., 28., 16., 32., 16., 16.,
        ],
    )
    enable_velocity_commands = False
    policy: InterMimicPolicyCfg = InterMimicPolicyCfg(
        # Sensor noise. joint_vel dominates jitter by a wide margin --
        # everything else is nearly negligible next to it. These are
        # legged_gym-typical magnitudes, NOT the InterMimic training
        # table (which we don't have a copy of). Set all to 0.0 for a
        # clean/noiseless run.
        # noise_projected_gravity=0.05,
        # noise_base_ang_vel=0.2,
        # noise_joint_pos=0.01,
        # noise_joint_vel=0.5,
        # TEMPORARY: disabled to watch the uncorrected failure. Set back to
        # True (the default) once you've seen it -- with init_quat at 90 deg
        # yaw this falls in ~60-95 steps on every seed.
        yaw_align=True,
    )
    mujoco = MujocoControllerCfg(
        visualize_reference_ghost=False,
        ghost_rgba=[0.6, 1.0, 0.6, 0.12],
        # `robot.default_joint_pos` must stay all-zero (this checkpoint's
        # obs/action baseline), but joint-zero on this robot *is* the
        # T-pose (arms out) -- shoulder_roll=0 is horizontal; every other
        # K1 task explicitly sets it to +-1.3 to bring the arms down.
        # So spawn MuJoCo at the real robot's actual standing pose
        # (booster.py's prepare_state.joint_pos: arms down, slightly bent
        # knees/ankles) instead, matching what other tasks look like and
        # what a real hardware handoff would actually look like. This only
        # changes MuJoCo's initial qpos, not the policy's own baseline.
        init_dof_pos=K1_CFG.prepare_state.joint_pos,
        # Root height for that pose, computed from the foot geom actually
        # touching the ground -- same pattern beyond_mimic uses to tune
        # init_pos to its own stance's real height, rather than relying
        # on the generic 0.6 default (tuned for yet another task's pose).
        init_pos=[0.0, 0.0, 0.551],
        # Spawn facing 35 deg instead of along +x, to exercise the heading
        # case that only ever occurs on hardware (MuJoCo would otherwise
        # always spawn at yaw 0, exactly where the reference clip is
        # canonicalized by the motion loader). Without the yaw correction
        # in `InterMimicPolicy.reset` the robot topples, because
        # `rel_ori_6d` lands well outside the checkpoint's own training
        # distribution (~4 sigma at 35 deg, ~30 sigma at 90).
        # Set back to [1.0, 0.0, 0.0, 0.0] for the original head-on view.
        # Caveat when using this to judge the correction: MuJoCo's contact
        # solver is exactly rotation-symmetric only at 90/180 deg
        # (open-loop delta 1e-16); at other angles it is not (1.5e-2 at
        # 45 deg under large motion, with the policy removed entirely),
        # and that solver noise alone can topple this marginally-stable
        # policy -- looking like a heading bug when it isn't. Use 90 deg
        # for a clean A/B; angles like this one are the realistic case.
        init_quat=[0.95372, 0.0, 0.0, 0.30071],
        # booster_train trains K1 with min_delay=2 / max_delay=8 physics
        # steps at its 200Hz sim, i.e. 10-40ms of actuator command lag.
        actuator_delay_range_s=[0.01, 0.04],
        torque_speed_curve=True,
        # Safety-harness support, as during hardware bring-up. Unloading
        # the legs removes ground-contact damping, which amplifies the
        # jitter driven by noise_joint_vel above (3.2x more visible motion
        # at 90% support). Disabled -- set gantry_body_name="trunk" and a
        # support fraction to re-enable.
        # gantry_body_name="trunk",
        # gantry_support_fraction=0.9,
    )
