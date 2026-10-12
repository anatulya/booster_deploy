from __future__ import annotations

from dataclasses import MISSING
import os

import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import PolicyCfg
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.policy_runner import create_policy_runner


class MjlabVelocityPolicy(Policy):
    """Velocity-tracking policy trained with booster_mjlab (Mjlab-Velocity-*-Booster-K1).

    Actor observation (75), no history, matching the ``actor`` group of make_velocity_env_cfg:
        base_ang_vel(3) projected_gravity(3) joint_pos_rel(22) joint_vel(22) last_action(22) command(3)
    Action: q_target = default_joint_pos + action * action_scale (per joint, unclipped).

    booster_mjlab's K1 joint order is the same as K1_CFG.joint_names, so no joint remapping is needed. The
    IMU site sits at the trunk origin with identity orientation, so root_ang_vel_b is the training gyro.
    """

    def __init__(self, cfg: MjlabVelocityPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot

        policy_path = self.cfg.checkpoint_path
        if not os.path.isabs(policy_path):
            policy_path = os.path.join(self.task_path, self.cfg.checkpoint_path)
        # .pt (TorchScript, e.g. from scripts/export_rsl_rl_jit.py) or the .onnx booster_mjlab writes.
        self._model = create_policy_runner(policy_path, torch.device("cpu"))

        self.action_scale = torch.tensor(cfg.action_scale, dtype=torch.float32)
        self.last_action = torch.zeros(len(cfg.action_scale), dtype=torch.float32)
        self._check_onnx_metadata(policy_path)

    def _check_onnx_metadata(self, policy_path: str) -> None:
        """booster_mjlab stores the training gains, home pose and action scale in each ONNX export. Refuse a
        checkpoint whose values differ from this task's config: it would run with PD gains or a baseline pose it
        was not trained for."""
        session = getattr(self._model, "session", None)
        if session is None:
            return   # TorchScript: no metadata to compare
        meta = session.get_modelmeta().custom_metadata_map
        expected = {
            "joint_stiffness": self.robot.joint_stiffness,
            "joint_damping": self.robot.joint_damping,
            "default_joint_pos": self.robot.default_joint_pos,
            "action_scale": self.action_scale,
        }
        bad = []
        for key, ours in expected.items():
            if key not in meta:
                continue
            theirs = torch.tensor([float(v) for v in meta[key].split(",")], dtype=torch.float32)
            ours = torch.as_tensor(ours, dtype=torch.float32).flatten().cpu()
            # The metadata is written with 3 decimals.
            if theirs.shape != ours.shape or not torch.allclose(theirs, ours, atol=2e-3):
                bad.append(f"  {key}:\n    checkpoint {theirs.tolist()}\n    task       {ours.tolist()}")
        if bad:
            raise ValueError(
                f"{policy_path} was trained with different settings than this task's config:\n"
                + "\n".join(bad)
                + "\nUpdate the task config to match the checkpoint, or use a matching checkpoint.")

    def reset(self) -> None:
        self.last_action.zero_()

    def compute_observation(self) -> torch.Tensor:
        base_quat = self.robot.data.root_quat_w
        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        projected_gravity = lab_math.quat_apply_inverse(base_quat, gravity_w)

        if self.cfg.enable_safety_fallback and projected_gravity[2] > -0.5:
            print("\nFalling detected, stopping policy for safety. "
                  "You can disable safety fallback by setting "
                  f"{self.cfg.__class__.__name__}.enable_safety_fallback "
                  "to False.")
            self.controller.stop()

        cmd = self.controller.vel_command
        return torch.cat([
            self.robot.data.root_ang_vel_b,
            projected_gravity,
            self.robot.data.joint_pos - self.robot.default_joint_pos,
            self.robot.data.joint_vel,
            self.last_action,
            torch.tensor(
                [cmd.lin_vel_x, cmd.lin_vel_y, cmd.ang_vel_yaw], dtype=torch.float32),
        ], dim=0)

    def inference(self) -> torch.Tensor:
        obs = self.compute_observation()
        # ONNX exports return a (1, n) buffer that is reused between calls: flatten and copy.
        action = self._model(obs).reshape(-1).clone()
        self.last_action = action
        return self.robot.default_joint_pos + action * self.action_scale


@configclass
class MjlabVelocityPolicyCfg(PolicyCfg):
    constructor = MjlabVelocityPolicy
    checkpoint_path: str = MISSING  # type: ignore
    # Per-joint 0.25 * effort_limit / stiffness from booster_mjlab's K1_ACTION_SCALE, in K1_CFG joint order.
    # Also stored as `action_scale` in the ONNX metadata of each training run.
    action_scale: list[float] = [
        0.375, 0.375,
        0.35, 0.35, 0.35, 0.35,
        0.35, 0.35, 0.35, 0.35,
        0.2125, 0.2375, 0.1196875, 0.35, 0.1915, 0.1915,
        0.2125, 0.2375, 0.1196875, 0.35, 0.1915, 0.1915,
    ]
