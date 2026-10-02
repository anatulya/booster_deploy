from __future__ import annotations
from dataclasses import MISSING
import os
import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg, MujocoControllerCfg, PolicyCfg, VelocityCommandCfg
)
from booster_deploy.robots.booster import K1_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.isaaclab import math as lab_math


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
        self._model: torch.jit.ScriptModule = torch.jit.load(
            policy_path, map_location="cpu")
        self._model.eval()

        self.action_scale = torch.tensor(cfg.action_scale, dtype=torch.float32)
        self.last_action = torch.zeros(len(cfg.action_scale), dtype=torch.float32)

    def reset(self) -> None:
        self.last_action.zero_()

    def obs_layout(self) -> list[tuple[str, int]]:
        n = self.robot.num_joints
        return [("base_ang_vel", 3), ("projected_gravity", 3), ("joint_pos", n), ("joint_vel", n),
                ("last_action", n), ("command", 3)]

    def compute_observation(self) -> torch.Tensor:
        base_quat = self.robot.data.root_quat_w
        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        projected_gravity = lab_math.quat_apply_inverse(base_quat, gravity_w)

        if self.cfg.enable_safety_fallback and projected_gravity[2] > -0.5:
            print("\nFalling detected, stopping policy for safety. "
                  "You can disable safety fallback by setting "
                  f"{self.cfg.__class__.__name__}.enable_safety_fallback "
                  "to False.")
            self.controller.stop(reason="safety_fallback")

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
        with torch.no_grad():
            action = self._model(obs)
        self.last_action = action.clone()
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


# booster_mjlab HOME_KEYFRAME: arms down, knees bent. Both the obs/action baseline and the pose the robot is
# brought to on entering custom mode, so the policy starts in-distribution.
K1_MJLAB_HOME_POS = [
    0.0, 0.0,
    0.0, -1.4, 0.0, -0.4,
    0.0, 1.4, 0.0, 0.4,
    -0.4, 0.0, 0.0, 0.8, -0.4, 0.0,
    -0.4, 0.0, 0.0, 0.8, -0.4, 0.0,
]


@configclass
class K1MjlabVelocityControllerCfg(ControllerCfg):
    # PD gains and effort limits from booster_mjlab's K1 actuators (k1_constants.py), which the policy was
    # trained against; they are also in the ONNX metadata (joint_stiffness / joint_damping).
    robot = K1_CFG.replace(  # type: ignore
        default_joint_pos=K1_MJLAB_HOME_POS,
        joint_stiffness=[
            4.0, 4.0,
            10.0, 10.0, 10.0, 10.0,
            10.0, 10.0, 10.0, 10.0,
            80.0, 80.0, 80.0, 80.0, 50.0, 50.0,
            80.0, 80.0, 80.0, 80.0, 50.0, 50.0,
        ],
        joint_damping=[
            0.25, 0.25,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            4.0, 4.0, 4.0, 4.0, 2.0, 2.0,
            4.0, 4.0, 4.0, 4.0, 2.0, 2.0,
        ],
        effort_limit=[
            6.0, 6.0,
            14.0, 14.0, 14.0, 14.0,
            14.0, 14.0, 14.0, 14.0,
            68.0, 76.0, 38.3, 112.0, 38.3, 38.3,
            68.0, 76.0, 38.3, 112.0, 38.3, 38.3,
        ],
        prepare_state=K1_CFG.prepare_state.replace(joint_pos=K1_MJLAB_HOME_POS),
    )
    # Joystick full-scale. Training's command curriculum starts at x in [-1.0, 1.2], y and yaw in [-1, 1] and
    # widens from iteration 5000; raise these only for checkpoints trained past the later stages.
    vel_command: VelocityCommandCfg = VelocityCommandCfg(
        vx_max=1.0,
        vy_max=1.0,
        vyaw_max=1.0,
    )
    policy: MjlabVelocityPolicyCfg = MjlabVelocityPolicyCfg()
    mujoco = MujocoControllerCfg(
        init_pos=[0.0, 0.0, 0.5125],
        # booster_mjlab trains K1 with 2-8 physics steps of actuator lag at 200 Hz, i.e. 10-40 ms.
        actuator_delay_range_s=[0.01, 0.04],
        torque_speed_curve=True,
    )
