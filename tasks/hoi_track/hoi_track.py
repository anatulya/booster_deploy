from __future__ import annotations
from dataclasses import MISSING
import numpy as np
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
from booster_deploy.utils.start_transition import StartTransition


class HoiTrackPolicy(Policy):
    """Deployable (HoiAsym) actor for the K1 suitcase pick-up.

    Two things make this different from ``BeyondMimicPolicy``.

    **The action is a residual on the reference pose**, not on the default stance::

        q_target = ref_joint_pos[frame] + action * action_scale

    ``action_scale`` and the reference come from the baked npz rather than being recomputed here, because
    ``0.25 * effort_limit / joint_stiffness`` evaluated against the *deployed* PD gains does not reproduce the
    training value -- this controller sets hip pitch stiffness to 80 where the Isaac actuator uses 30.2, which
    would rescale every action by 2.6x.

    **The reference block is pre-baked.** All ten reference observation terms except ``ref_anchor_ori_b`` are
    pure functions of (motion file, frame), so ``scripts/export_hoi_reference.py`` in booster_train evaluates
    them with the training environment's own observation functions and writes them to an npz. Nothing here
    re-derives a heading frame, which is the part that would silently disagree.

    Observation layout, 465 dims, matching ``ActorObsCfg`` term for term::

        base_ang_vel(3) joint_pos(22) joint_vel(22) last_action(22) motion_phase(1)       = 70
        for k in (0, 1, 2, 8, 16):                                                        = 5 x 79
            ref_joint_pos(22) ref_joint_vel(22) ref_anchor_ori_b(6)
            ref_object_pos(3) ref_object_ori(6) ref_object_lin_vel(3) ref_object_ang_vel(3)
            ref_contact_point_objlocal(6) ref_contact_point_refroot(6) ref_contact_flag(2)

    Everything the network sees is either a sensor reading or a table lookup, so this runs unchanged on
    hardware. It is also blind to the object: the suitcase terms are what the *reference* says, never what a
    real box is doing, so the grasp is open-loop by construction.
    """

    # Within-k block order, and which entries are looked up from the npz. ref_anchor_ori_b is spliced in at
    # index 2 at runtime because it needs the live robot orientation.
    BLOCK = (
        "ref_joint_pos",
        "ref_joint_vel",
        None,  # ref_anchor_ori_b, computed per step
        "ref_object_pos_refroot",
        "ref_object_ori_refroot",
        "ref_object_lin_vel_refroot",
        "ref_object_ang_vel_refroot",
        "ref_contact_point_objlocal",
        "ref_contact_point_refroot",
        "ref_contact_flag",
    )

    def __init__(self, cfg: HoiTrackPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot

        self._model: torch.jit.ScriptModule = torch.jit.load(
            f"{self.task_path}/{self.cfg.checkpoint_path}")
        self._model.to(self.cfg.device).eval()
        self.robot.data.to(self.cfg.device)

        dev = self.cfg.device
        data = np.load(f"{self.task_path}/{self.cfg.motion_path}")
        self.ref = {
            k: torch.as_tensor(data[k], dtype=torch.float32, device=dev)
            for k in (*[b for b in self.BLOCK if b], "ref_anchor_pos_w",
                      "ref_anchor_quat_w", "ref_object_pos_w", "ref_object_quat_w",
                      "action_scale", "default_joint_pos")
        }
        self.horizon = [int(k) for k in data["horizon"]]
        self.num_frames = int(data["num_frames"])
        self.obs_dim = int(data["obs_dim"])

        # The baked reference is in Isaac joint order; booster_deploy's K1_CFG.sim_joint_names matches it, so
        # the robot's existing index tables do the remap both ways.
        self.real2sim = self.robot.data.real2sim_joint_indexes
        self.sim2real = self.robot.data.sim2real_joint_indexes

    supports_start_hold = True

    # Zeroed while holding the first frame, so the held reference is static like the one past the clip's end.
    HOLD_ZEROED = ("ref_joint_vel", "ref_object_lin_vel_refroot", "ref_object_ang_vel_refroot")

    def reset(self) -> None:
        self.current_frame = 0
        self.holding = self.cfg.hold_start_frame
        self.hold_step = 0
        self.transition: StartTransition | None = None
        self.last_action = torch.zeros(
            self.robot.num_joints, dtype=torch.float32, device=self.cfg.device)
        # Deliberately NOT computed here. BaseController.start() calls policy.reset() before the controller
        # has copied any state into robot.data, so under MuJoCo root_quat_w is still [0,0,0,0] -- not even a
        # valid quaternion, and it turns the whole ref_anchor_ori_b block into NaN. On hardware the IMU is
        # publishing orientation continuously and this would very likely already be live, so the deferral is
        # defensive there rather than necessary. Doing it on the first inference is correct either way.
        self.align_quat: torch.Tensor | None = None

    def _lazy_init(self) -> None:
        """One-time setup that needs valid robot state, run on the first observation of an episode."""
        if self.align_quat is not None:
            return
        # Training places the robot exactly onto the reference at reset. Match that
        # orientation relationship here: align_quat maps the live robot frame into
        # the reference frame. Full alignment also preserves the object's
        # frame-zero orientation relative to the trunk. Keep yaw-only available for
        # quick comparison with the previous behavior.
        if self.cfg.full_orientation_align:
            self.align_quat = lab_math.quat_mul(
                self.ref["ref_anchor_quat_w"][0],
                lab_math.quat_inv(self.robot.data.root_quat_w),
            )
        elif self.cfg.yaw_align:
            self.align_quat = lab_math.quat_mul(
                lab_math.yaw_quat(self.ref["ref_anchor_quat_w"][0]),
                lab_math.quat_inv(lab_math.yaw_quat(self.robot.data.root_quat_w)),
            )
        else:
            self.align_quat = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.cfg.device)

        # The inverse maps the raw motion world into the spawned robot heading.
        # As in booster_train, center x/y on the live robot while retaining
        # the motion absolute z and rotating only about world Z.
        self.reference_to_robot_quat = lab_math.quat_inv(self.align_quat)
        self.reference_origin_w = self.robot.data.root_pos_w.clone()
        self.reference_origin_w[2] = self.ref["ref_anchor_pos_w"][0, 2]
        # While holding the first frame the object stays parked (see MujocoController); it is placed on release.
        if not self.holding:
            self._place_object()
        elif self.cfg.start_vel_tau_s > 0:
            # Reference is frame 0 at once; its velocity points there from where the robot is and decays.
            self.transition = StartTransition(
                start_pos=self.robot.data.joint_pos[self.real2sim],
                goal_pos=self.ref["ref_joint_pos"][0],
                tau_s=self.cfg.start_vel_tau_s,
            )

    def release_motion(self) -> None:
        """Start the motion from wherever the hold left the robot.

        The policy can drift while holding frame 0, so the reference is re-aligned to the robot's pose at
        release, and the object placed relative to that, the way training resets robot and object together.
        Clearing align_quat defers both to the next observation, after the controller has refreshed robot.data.
        """
        if self.holding and self._can_release():
            super().release_motion()
            self.align_quat = None
            self.transition = None

    def _pose_in_robot_world(
        self, pos: torch.Tensor, quat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Yaw-align a raw reference pose with the robot spawn frame."""
        rel_pos = pos - self.ref["ref_anchor_pos_w"][0]
        world_pos = self.reference_origin_w + lab_math.quat_apply(
            self.reference_to_robot_quat, rel_pos)
        world_quat = lab_math.quat_mul(self.reference_to_robot_quat, quat)
        return world_pos, world_quat

    def _place_object(self) -> None:
        """Put the scene's object where the clip starts it, in the robot's world.

        Simulation only: ``set_object_pose`` exists on the MuJoCo controller and has no hardware equivalent,
        where the box is wherever someone set it down. The policy never observes the real object either way --
        every suitcase term it reads is the reference's, so this only decides what the hands actually meet.

        ``align_object_to_robot`` rotates the object's full pose (position and orientation) into the robot's
        spawn frame with ``_pose_in_robot_world``, the same transform used for the ghost humanoid. Both need
        to move together: the object's position is expressed relative to the reference anchor, so rotating
        only the position while leaving the orientation raw put the box in the right neighborhood but facing
        the reference clip's original heading instead of the robot's -- for a non-symmetric object (a crate,
        not a round suitcase) that reads as "spawns rotated" whenever the clip's frame-0 heading isn't close
        to the robot's actual spawn heading (both captured clips so far start around -90 deg, so this fires
        every time by default).
        """
        if self.cfg.object_body_name is None or not hasattr(self.controller, "set_object_pose"):
            return
        ref_pos = self.ref["ref_object_pos_w"][0]
        ref_quat = self.ref["ref_object_quat_w"][0]
        pos = ref_pos
        quat = ref_quat
        if self.cfg.align_object_to_robot:
            pos, quat = self._pose_in_robot_world(pos, quat)
        elif self.cfg.align_object_yaw:
            quat = lab_math.quat_mul(
                lab_math.yaw_quat(self.reference_to_robot_quat), quat)
        self.controller.set_object_pose(                      # type: ignore
            self.cfg.object_body_name, pos.cpu().numpy(), quat.cpu().numpy(),
            ghost_pos=pos.cpu().numpy(), ghost_quat=quat.cpu().numpy())

    def _rows(self) -> torch.Tensor:
        """Clip-clamped frame index per horizon offset, matching ``MotionCommand._refresh_reference_frames``."""
        last = self.num_frames - 1
        if self.holding:
            # Every offset on frame 0, as they all clamp to the last frame once the clip has ended.
            return torch.zeros(len(self.horizon), dtype=torch.long, device=self.cfg.device)
        return torch.tensor(
            [min(self.current_frame + k, last) for k in self.horizon],
            dtype=torch.long, device=self.cfg.device)

    def _ref_anchor_ori_b(self, ref_quat: torch.Tensor) -> torch.Tensor:
        """Reference anchor orientation in the robot's frame, 6D, one row per horizon offset.

        The same quantity ``BeyondMimicPolicy`` forms as ``motion_anchor_ori_b``, evaluated at each offset.
        The anchor body is the Trunk, which is the K1's root, so the robot side is just ``root_quat_w``.
        """
        n = ref_quat.shape[0]
        cur = lab_math.quat_mul(self.align_quat, self.robot.data.root_quat_w)
        cur_inv = lab_math.quat_inv(cur.unsqueeze(0)).expand(n, -1)
        rel = lab_math.quat_mul(cur_inv, ref_quat)
        return lab_math.matrix_from_quat(rel)[..., :2].reshape(n, 6)

    def compute_observation(self) -> torch.Tensor:
        self._lazy_init()
        rows = self._rows()
        ref_joint_pos = self.ref["ref_joint_pos"][rows]
        ref_joint_vel = self.ref["ref_joint_vel"][rows]
        ref_anchor_quat = self.ref["ref_anchor_quat_w"][rows]
        if self.holding and self.transition is not None:
            # Each horizon offset previews the transition curve ahead, as it previews the motion otherwise.
            dt = self.controller.cfg.policy_dt
            samples = [self.transition.sample((self.hold_step + k) * dt) for k in self.horizon]
            ref_joint_pos = torch.stack([s[0] for s in samples])
            ref_joint_vel = torch.stack([s[1] for s in samples])
        self.cmd_dof_pos = ref_joint_pos[0]
        self.cmd_root_pos_w = self.ref["ref_anchor_pos_w"][rows[0]]
        self.cmd_root_quat_w = ref_anchor_quat[0]

        joint_pos = (self.robot.data.joint_pos[self.real2sim]
                     - self.ref["default_joint_pos"])
        joint_vel = self.robot.data.joint_vel[self.real2sim]
        # clip_starts is 0 for a single-clip task, so phase is just frame / length.
        motion_phase = torch.tensor(
            [self.current_frame / self.num_frames],
            dtype=torch.float32, device=self.cfg.device)

        parts = [
            self.robot.data.root_ang_vel_b,
            joint_pos,
            joint_vel,
            self.last_action,
            motion_phase,
        ]
        anchor_ori = self._ref_anchor_ori_b(ref_anchor_quat)
        for i, row in enumerate(rows):
            for j, name in enumerate(self.BLOCK):
                if name is None:
                    parts.append(anchor_ori[i])
                elif name == "ref_joint_pos":
                    parts.append(ref_joint_pos[i])
                elif name == "ref_joint_vel" and self.transition is not None:
                    parts.append(ref_joint_vel[i])
                elif self.holding and name in self.HOLD_ZEROED:
                    parts.append(torch.zeros_like(self.ref[name][row]))
                else:
                    parts.append(self.ref[name][row])

        obs = torch.cat(parts, dim=0)
        if obs.numel() != self.obs_dim:
            raise RuntimeError(
                f"observation is {obs.numel()} dims, expected {self.obs_dim} -- the baked reference and this "
                "policy's layout disagree; re-run scripts/export_hoi_reference.py")
        return obs.reshape(1, -1)

    def inference(self) -> torch.Tensor:
        with torch.no_grad():
            action = self._model(self.compute_observation()).flatten()

        if hasattr(self.controller, "set_reference_qpos"):
            ghost_root_pos = self.cmd_root_pos_w
            ghost_root_quat = self.cmd_root_quat_w
            if self.cfg.align_object_to_robot:
                ghost_root_pos, ghost_root_quat = self._pose_in_robot_world(
                    ghost_root_pos, ghost_root_quat)
            ref_qpos = torch.cat(
                [ghost_root_pos, ghost_root_quat,
                 self.cmd_dof_pos[self.sim2real]], dim=0)
            self.controller.set_reference_qpos(ref_qpos)    # type: ignore

        if self.holding:
            self.hold_step += 1
        else:
            self.current_frame += 1
        self.last_action = action

        if self.cfg.enable_safety_fallback:
            gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32,
                                     device=self.cfg.device)
            projected_gravity = lab_math.quat_apply_inverse(
                lab_math.quat_mul(self.align_quat, self.robot.data.root_quat_w),
                gravity_w)
            motion_projected_gravity = lab_math.quat_apply_inverse(
                self.cmd_root_quat_w, gravity_w)
            if torch.dot(projected_gravity, motion_projected_gravity) < 0.5:
                print("\nLarge root tracking error is detected, stopping policy"
                      " for safety. You can disable safety fallback by setting "
                      f"{self.cfg.__class__.__name__}.enable_safety_fallback "
                      "to False.")
                self.controller.stop()

        # The residual: a zero action commands the reference pose exactly.
        target = self.cmd_dof_pos + action * self.ref["action_scale"]
        return target[self.sim2real]


@configclass
class HoiTrackPolicyCfg(PolicyCfg):
    constructor = HoiTrackPolicy
    checkpoint_path: str = MISSING
    motion_path: str = MISSING

    # Optional full frame-zero root alignment. Off by default: yaw-only keeps
    # the gravity-aligned roll/pitch behavior used by the original deployment.
    full_orientation_align: bool = False
    yaw_align: bool = True

    # Rotate only the object quaternion, leaving its raw position unchanged.
    align_object_yaw: bool = False

    # Yaw-align the ghost humanoid trajectory and object position while keeping
    # the object quaternion exactly as stored. Takes precedence over the option above.
    align_object_to_robot: bool = False

    # Free-floating scene body to place at the clip's starting object pose on reset. None runs without an
    # object, which still exercises the observation layout and gait but gives the hands nothing to close on.
    object_body_name: str | None = None


@configclass
class K1HoiTrackControllerCfg(ControllerCfg):
    # Gains, damping and effort limits copied from booster_train's BOOSTER_K1_CFG actuators rather than left at
    # K1_CFG's deployment defaults. The policy was trained against this PD response, and the same override is
    # what K1LargeboxTrackerControllerCfg does for the tracker. Real joint order: head, left arm, right arm,
    # left leg, right leg.
    robot = K1_CFG.replace(     # type: ignore
        joint_stiffness=[
            3.9478, 3.9478,
            3.9478, 3.9478, 3.9478, 3.9478,
            3.9478, 3.9478, 3.9478, 3.9478,
            30.2010, 21.4480, 17.8460, 60.4020, 35.6920, 35.6920,
            30.2010, 21.4480, 17.8460, 60.4020, 35.6920, 35.6920,
        ],
        joint_damping=[
            0.2513, 0.2513,
            0.2513, 0.2513, 0.2513, 0.2513,
            0.2513, 0.2513, 0.2513, 0.2513,
            3.6050, 2.5602, 2.1302, 4.8066, 4.2604, 4.2604,
            3.6050, 2.5602, 2.1302, 4.8066, 4.2604, 4.2604,
        ],
        effort_limit=[
            6.0, 6.0,
            14.0, 14.0, 14.0, 14.0,
            14.0, 14.0, 14.0, 14.0,
            68.0, 76.0, 38.3, 112.0, 38.3, 38.3,
            68.0, 76.0, 38.3, 112.0, 38.3, 38.3,
        ],
    )
    enable_velocity_commands = False
    policy: HoiTrackPolicyCfg = HoiTrackPolicyCfg()
    mujoco = MujocoControllerCfg(
        # Scene with the captured suitcase: `python scripts/make_object_scene.py suitcase_0539923`.
        scene_mjcf_path="{BOOSTER_ASSETS_DIR}/robots/K1/K1_22dof_suitcase.xml",
        # Spawn where custom mode starts from: upright, in prepare_state.joint_pos. 0.551 puts that pose's
        # feet on the floor. The policy then brings the robot to the clip's frame-0 pose itself and holds it
        # (PolicyCfg.hold_start_frame) until the motion is released.
        init_pos=[0.0, 0.0, 0.551],
        init_dof_pos=list(K1_CFG.prepare_state.joint_pos),
        visualize_reference_ghost=False,
        # booster_train trains K1 with min_delay=2 / max_delay=8 physics steps at 200 Hz, i.e. 10-40 ms lag.
        # Off for now -- one less variable while testing the suitcase pick-up.
        actuator_delay_range_s=[0.01, 0.04],
        torque_speed_curve=True,
    )
