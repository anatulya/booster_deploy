from __future__ import annotations
from abc import abstractmethod
import inspect
import os
import time
import numpy as np
import torch

from .controller_cfg import (
    ControllerCfg, PolicyCfg, RobotCfg, VelocityCommandCfg
)


class RobotData:
    """
    The joint indexing follows the real robot,
    described in RobotCfg.joint_names
    """

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    feedback_torque: torch.Tensor
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_b: torch.Tensor
    root_ang_vel_b: torch.Tensor

    def __init__(self, cfg: RobotCfg) -> None:
        self.cfg = cfg
        num_joints = len(self.cfg.joint_names)
        self.real2sim_joint_indexes = [cfg.joint_names.index(name) for name in cfg.sim_joint_names]
        self.sim2real_joint_indexes = [cfg.sim_joint_names.index(name) for name in cfg.joint_names]
        self.device = "cpu"

        self.joint_pos: torch.Tensor = torch.zeros(num_joints, dtype=torch.float32)
        self.joint_vel: torch.Tensor = torch.zeros(num_joints, dtype=torch.float32)
        self.feedback_torque: torch.Tensor = torch.zeros(num_joints, dtype=torch.float32)
        self.root_lin_vel_b: torch.Tensor = torch.zeros(3, dtype=torch.float32)
        self.root_ang_vel_b: torch.Tensor = torch.zeros(3, dtype=torch.float32)
        self.root_pos_w: torch.Tensor = torch.zeros(3, dtype=torch.float32)
        self.root_quat_w: torch.Tensor = torch.zeros(4, dtype=torch.float32)

    def to(self, device: torch.device | str) -> None:
        self.device = device
        self.joint_pos = self.joint_pos.to(device)
        self.joint_vel = self.joint_vel.to(device)
        self.feedback_torque = self.feedback_torque.to(device)
        self.root_lin_vel_b = self.root_lin_vel_b.to(device)
        self.root_ang_vel_b = self.root_ang_vel_b.to(device)
        self.root_pos_w = self.root_pos_w.to(device)
        self.root_quat_w = self.root_quat_w.to(device)


class BoosterRobot:
    cfg: RobotCfg
    data: RobotData
    joint_stiffness: torch.Tensor
    joint_damping: torch.Tensor
    default_joint_pos: torch.Tensor

    def __init__(self, cfg: RobotCfg) -> None:
        self.cfg = cfg
        self.data = RobotData(cfg)

        self.joint_stiffness = torch.tensor(cfg.joint_stiffness, dtype=torch.float32)

        self.joint_damping = torch.tensor(cfg.joint_damping, dtype=torch.float32)

        self.default_joint_pos = torch.tensor(cfg.default_joint_pos, dtype=torch.float32)
        self.effort_limit = torch.tensor(cfg.effort_limit, dtype=torch.float32)

    @property
    def num_joints(self) -> int:
        return len(self.cfg.joint_names)

    @property
    def num_bodies(self) -> int:
        return len(self.cfg.body_names)


class Commands:
    pass


class VelocityCommand(Commands):
    lin_vel_x: float
    lin_vel_y: float
    ang_vel_yaw: float

    def __init__(self, cfg: VelocityCommandCfg) -> None:
        self.vx_max = cfg.vx_max
        self.vy_max = cfg.vy_max
        self.vyaw_max = cfg.vyaw_max

        self.lin_vel_x: float = 0.0
        self.lin_vel_y: float = 0.0
        self.ang_vel_yaw: float = 0.0


class Policy:
    # Whether this policy can hold its motion's first frame
    # (PolicyCfg.hold_start_frame). A class attribute so the robot portal can
    # read it before the policy is built in the inference process.
    supports_start_hold: bool = False
    # True while a motion-tracking policy holds its first frame.
    holding: bool = False

    def __init__(self, cfg: PolicyCfg, controller: BaseController):
        self.cfg = cfg
        self.controller = controller
        # Get the module path of the actual class (works for subclasses too)
        class_module = inspect.getmodule(self.__class__)
        self.task_path = os.path.dirname(class_module.__file__)  # type: ignore

    @abstractmethod
    def reset(self) -> None:
        """Called when the controller starts."""

    def obs_layout(self) -> list[tuple[str, int]] | None:
        """Names and widths of the observation terms, in network-input order, for the observation log.
        None logs the raw vector only."""
        return None

    def release_motion(self) -> None:
        """Stop holding the first frame and start advancing the motion."""
        if self.holding:
            self.holding = False
            print("Motion started")

    def _start_transition_remaining_s(self) -> float:
        """Seconds until a StartTransition (if any) reaches the first frame."""
        transition = getattr(self, "transition", None)
        if transition is None:
            return 0.0
        return max(0.0, transition.duration_s - self.hold_step
                   * self.controller.cfg.policy_dt)

    def _can_release(self) -> bool:
        remaining = self._start_transition_remaining_s()
        if remaining > 0.0:
            print(f"Still moving to the start pose ({remaining:.1f}s left)")
            return False
        return True

    @abstractmethod
    def inference(self) -> torch.Tensor:
        """Called each controller step to perform inference.

        Returns:
            action torch.Tensor containing the action for this step.
        """


class BaseController:
    """Simple deployment environment skeleton and execution overview.

    This class provides a minimal, dependency-light interface suitable for
    deployment scripts and controllers. It defines the method contract used by
    concrete controller implementations and documents the typical runtime
    execution order.

    Public method contract
    - `start(initial_state=None) -> obs`: prepare controller and policy for
        execution and return initial observation.
    - `policy_step() -> torch.Tensor`: invoke policy inference for one step
        and return the action tensor.
    - `ctrl_step(dof_targets: torch.Tensor) -> None`: apply action to the
        environment (send to actuators / shared buffer / simulator).
    - `update_state() -> None`: refresh internal robot state from sensors or
        shared buffers (called each control loop iteration before inference).
    - `stop() -> None`: stop the running session; should be idempotent.
    - `run() -> None`: high-level entry point for a controller process or
        thread (optional to implement for each concrete controller).

    Concrete controllers may implement `run()` to orchestrate the typical
    execution flow below:

        start()

            |
            v
    +----------------- main loop -----------------+
    |  update_state()                                |
    |      |                                         |
    |      v                                         |
    |  policy_step()  -> (action tensor)            |
    |      |                                         |
    |      v                                         |
    |  ctrl_step(action)                             |
    |      |                                         |
    +-----------------------------------------------+
            |
            v
            stop() -> cleanup()/finalize()

    Notes and recommendations
    - `update_state()` should read the latest sensor/shared-buffer data and
        populate `self.robot.data` before `policy_step()` is called.
    - `policy_step()` is responsible only for producing actions and should
        not have side-effects that interfere with `update_state()`.
    - `ctrl_step()` applies the action produced by the policy to actuators or
        publish it.
    """

    cfg: ControllerCfg
    robot: BoosterRobot
    vel_command: VelocityCommand
    policy: Policy

    def __init__(self, cfg: ControllerCfg) -> None:
        self.cfg = cfg
        self._step_count: int = 0
        self._elapsed_s: float = 0.0
        self.is_running: bool = False
        self.robot = BoosterRobot(cfg.robot)
        self.vel_command = None  # type: ignore
        if self.cfg.vel_command is not None:
            self.vel_command = VelocityCommand(cfg.vel_command)
        self.policy = self.cfg.policy.constructor(self.cfg.policy, self)
        self.obs_log = None
        if self.cfg.log_dir:
            self._start_obs_log()

    def _start_obs_log(self) -> None:
        from ..utils.obs_logger import ObsLogger, RecordingModel, file_sha256, git_state
        if not hasattr(self.policy, "_model"):
            print("[obs_logger] policy has no _model; observation logging disabled")
            return
        self.policy._model = RecordingModel(self.policy._model)
        files = {}
        for key in ("checkpoint_path", "motion_path"):
            rel = getattr(self.cfg.policy, key, None)
            if isinstance(rel, str):
                path = rel if os.path.isabs(rel) else os.path.join(self.policy.task_path, rel)
                files[key] = {"path": path, "sha256": file_sha256(path)}
        meta = {
            "task": self.cfg.task_name,
            "backend": type(self).__name__,
            "policy_class": type(self.policy).__name__,
            "policy_dt": self.cfg.policy_dt,
            "joint_names": list(self.cfg.robot.joint_names),
            "sim_joint_names": list(self.cfg.robot.sim_joint_names),
            "files": files,
            "git": git_state(),
            "t_wall_created": time.time(),
            "cfg": self.cfg.to_dict(),
        }
        self.obs_log = ObsLogger(self.cfg.log_dir, meta)
        self._obs_layout_pending = True
        print(f"[obs_logger] logging policy steps to {self.cfg.log_dir}")

    def _log_event(self, name: str, **info) -> None:
        if self.obs_log is not None:
            self.obs_log.event(name, step=self._step_count, **info)

    def _log_step(self, dof_targets: torch.Tensor) -> None:
        model = self.policy._model
        if model.last_input is None:
            return
        if self._obs_layout_pending:
            # Set on the first step, once the policy has built any lazy state the layout depends on.
            self.obs_log._meta["obs_layout"] = self.policy.obs_layout()
            self._obs_layout_pending = False
        d = self.robot.data
        quat = d.root_quat_w.detach().cpu().reshape(1, 4)
        from ..utils.isaaclab import math as lab_math
        rpy = torch.stack(lab_math.euler_xyz_from_quat(quat), dim=-1).reshape(3)
        vc = self.vel_command
        self.obs_log.record({
            "t_wall": time.time(),
            "t_mono": time.perf_counter(),
            "step": self._step_count,
            "obs": model.last_input.cpu().numpy().reshape(-1),
            "action": model.last_output.cpu().numpy().reshape(-1),
            "dof_targets": dof_targets.detach().cpu().numpy().reshape(-1),
            "holding": bool(self.policy.holding),
            "hold_step": int(getattr(self.policy, "hold_step", -1)),
            "motion_frame": int(getattr(self.policy, "current_frame", -1)),
            "joint_pos": d.joint_pos.detach().cpu().numpy(),
            "joint_vel": d.joint_vel.detach().cpu().numpy(),
            "feedback_torque": d.feedback_torque.detach().cpu().numpy(),
            "root_quat_w": quat.numpy().reshape(4),
            "root_rpy_w": rpy.numpy(),
            "root_ang_vel_b": d.root_ang_vel_b.detach().cpu().numpy(),
            "vel_cmd": np.array([vc.lin_vel_x, vc.lin_vel_y, vc.ang_vel_yaw] if vc is not None
                                else [0.0, 0.0, 0.0], dtype=np.float32),
        })

    def close_logs(self) -> None:
        if self.obs_log is not None:
            self.obs_log.close()

    def start(self):
        """Begin a deployment session.
        """
        self._step_count = 0
        self._elapsed_s = 0.0
        self.is_running = True
        self._release_requested = False
        self._release_wait_reported = False
        self.policy.reset()
        self._log_event("start", holding=bool(self.policy.holding))

    def request_motion_release(self) -> None:
        """Ask a policy holding its first frame to start the motion. Kept
        until the start cue has finished (see StartTransition), so an early
        press is remembered rather than dropped. Idempotent."""
        self._release_requested = True

    def _maybe_release_motion(self) -> None:
        if not (self._release_requested and self.policy.holding):
            return
        remaining = self.policy._start_transition_remaining_s()
        if remaining > 0:
            if not self._release_wait_reported:
                print(f"Still moving to the start pose; the motion starts "
                      f"in {remaining:.1f}s")
                self._release_wait_reported = True
            return
        self.policy.release_motion()

    def policy_step(self) -> torch.Tensor:
        """Execute one inference step and return the action.

        Returns:
            action tensor
        """
        if not self.is_running:
            raise RuntimeError("Environment.step() called before start().")

        self._step_count += 1
        self._elapsed_s = self._step_count * self.cfg.policy_dt

        was_holding = self.policy.holding
        self._maybe_release_motion()
        if was_holding and not self.policy.holding:
            self._log_event("motion_released")
        dof_targets = self.policy.inference()
        if self.obs_log is not None:
            self._log_step(dof_targets)
        return dof_targets

    def stop(self, reason: str = "stop") -> None:
        """Stop and clean up the deployment session.

        Args:
            reason: Recorded in the observation log, e.g. "safety_fallback" when a policy detects a fall.
        """
        if self.is_running:
            self._log_event("safety_stop" if reason == "safety_fallback" else "stop", reason=reason)
        self.is_running = False

    @abstractmethod
    def ctrl_step(self, dof_targets: torch.Tensor) -> None:
        """Advance the environment by one control step.

        Args:
            dof_targets: Action tensor for this step (dof targets).
        """

    @abstractmethod
    def update_state(self) -> None:
        """Update robot data from sensors or shared buffers."""

    @abstractmethod
    def run(self) -> None:
        """Main loop entry point."""
