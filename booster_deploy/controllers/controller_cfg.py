from typing import Callable, List, Optional
from dataclasses import MISSING
import torch

from ..utils.isaaclab.configclass import configclass


@configclass
class PrepareStateCfg:
    stiffness: List[float] = MISSING
    damping: List[float] = MISSING
    joint_pos: List[float] = MISSING


@configclass
class MujocoControllerCfg:
    init_pos: List[float] = [0.0, 0.0, 0.6]
    init_quat: List[float] = [1.0, 0.0, 0.0, 0.0]
    # Initial joint angles for MuJoCo's spawn qpos. Defaults to
    # `robot.default_joint_pos` (None); set this instead when a policy's
    # obs/action baseline pose differs from the standing pose it should
    # actually be deployed from.
    init_dof_pos: Optional[List[float]] = None
    # Initial joint velocities for MuJoCo's spawn qvel. Defaults to zero
    # (a fully at-rest spawn); set this to simulate handing off from a
    # still-settling pose rather than a static one.
    init_dof_vel: Optional[List[float]] = None
    decimation: int = 10
    # physics_dt will automatically be set by ControllerCfg
    physics_dt: float = None  # type: ignore
    log_states: Optional[str] = None
    visualize_reference_ghost: bool = False
    ghost_rgba: List[float] = [0.2, 0.8, 0.2, 0.25]

    # Periodic external-force "push" test (a standard sim2real robustness
    # check): every `push_interval_s`, apply a `push_force` world-frame
    # Cartesian force (N) to `push_body_name`'s center of mass for
    # `push_duration_s`, then release it.
    push_body_name: Optional[str] = None
    push_force: List[float] = [0.0, 0.0, 0.0]
    push_interval_s: float = 3.0
    push_duration_s: float = 0.1

    # Alternative push mechanism matching Isaac Gym/legged_gym's actual
    # domain-randomization push: instead of a sustained force, directly
    # *overwrite* the root's XY linear velocity (qvel[0:2]) with a random
    # kick every `push_interval_s`, each axis sampled uniformly in
    # [-push_vel_xy, push_vel_xy]. Independent of the force-based push
    # above -- set this instead when reproducing a specific training
    # curriculum's push strength (e.g. "50% of max_push_vel_xy").
    push_vel_xy: Optional[float] = None

    # Offscreen video recording (used when no display is available).
    record_video_path: Optional[str] = None
    record_video_seconds: float = 15.0
    record_video_fps: Optional[int] = None
    # Must not exceed the MJCF's <visual><global offwidth/offheight> (defaults
    # to 640x480 if the XML doesn't set one).
    record_video_width: int = 640
    record_video_height: int = 480


@configclass
class BoosterRobotControllerCfg:
    low_state_dt: float = 0.002
    metrics_max_events: int = 2000


@configclass
class RobotCfg:
    name: str = MISSING

    joint_names: list[str] = MISSING
    body_names: list[str] = MISSING

    sim_joint_names: list[str] = MISSING
    sim_body_names: list[str] = MISSING

    joint_stiffness: List[float] = MISSING
    joint_damping: List[float] = MISSING

    default_joint_pos: List[float] = MISSING
    effort_limit: List[float] = MISSING

    mjcf_path: str = MISSING

    prepare_state: PrepareStateCfg = MISSING

    def __post_init__(self):
        assert (
            len(self.joint_names)
            == len(self.joint_stiffness)
            == len(self.joint_damping)
            == len(self.default_joint_pos)
            == len(self.effort_limit)
        )


@configclass
class VelocityCommandCfg:
    vx_max: float = 1.0
    vy_max: float = 1.0
    vyaw_max: float = 1.0


@configclass
class PolicyCfg:
    constructor: Callable = MISSING
    checkpoint_path: str = MISSING
    enable_safety_fallback: bool = True
    device: str | torch.device = "cpu"


@configclass
class EvaluatorCfg:
    constructor: Callable = MISSING
    # Rendering
    render: bool = True


@configclass
class ControllerCfg:
    """Controller configuration class.
    """

    policy_dt: float = 0.02
    robot: RobotCfg = MISSING
    vel_command: Optional[VelocityCommandCfg] = None
    policy: PolicyCfg = MISSING

    mujoco: MujocoControllerCfg = MujocoControllerCfg()
    booster: BoosterRobotControllerCfg = BoosterRobotControllerCfg()
    evaluator: Optional[EvaluatorCfg] = None

    def __post_init__(self):
        self.mujoco.physics_dt = self.policy_dt / self.mujoco.decimation
