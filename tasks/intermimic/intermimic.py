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

        return torch.cat(
            [projected_gravity, base_ang_vel, joint_pos, joint_vel,
             self.last_action],
            dim=0,
        )

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

        cur_quat_inv = lab_math.quat_inv(
            self.robot.data.root_quat_w.unsqueeze(0)).expand(n, -1)
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
    policy: InterMimicPolicyCfg = InterMimicPolicyCfg()
    mujoco = MujocoControllerCfg(
        visualize_reference_ghost=True,
    )
