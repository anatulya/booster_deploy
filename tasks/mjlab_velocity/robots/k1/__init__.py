from booster_deploy.controllers.controller_cfg import (
    ControllerCfg,
    MujocoControllerCfg,
    VelocityCommandCfg,
)
from booster_deploy.robots.k1 import K1_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from ...mjlab_velocity import MjlabVelocityPolicyCfg


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
        # X ramps to the home pose below and A starts this policy (no walking-policy hand-off).
        prepare_mode="standing",
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
    )


@configclass
class K1MjlabVelocityFlatTaskCfg(K1MjlabVelocityControllerCfg):
    '''Mjlab-Velocity-Flat-Booster-K1 policy (run 2026-10-11_14-09-28, iteration 14300).'''
    def __post_init__(self):
        super().__post_init__()
        # booster_mjlab's ONNX export (logs/rsl_rl/k1_velocity/<run>/<run>.onnx), kept in the repo's
        # checkpoints/ folder with the others; deploy.py --checkpoint picks a different one.
        self.policy.checkpoint_path = "../../checkpoints/baseline.onnx"


register_task("k1_mjlab_velocity_flat", K1MjlabVelocityFlatTaskCfg())
