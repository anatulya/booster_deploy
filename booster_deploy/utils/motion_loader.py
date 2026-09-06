import os
from typing import Sequence
import numpy as np
import torch
from booster_deploy.utils.isaaclab import math as lab_math


def _align_to_first_frame(loader) -> None:
    """Re-express a loader's `_body_pos_w`/`_body_quat_w`/velocities relative
    to its own frame-0 root (xy position + yaw only), in place. Shared by
    `MotionLoader` and `RawTensorMotionLoader` -- works unchanged whether
    the loader tracks one body or many, since it only ever looks at index 0.
    """
    init_root_pos_xy = loader._body_pos_w[:1, :1].clone()
    init_root_pos_xy[:, :, 2] = 0.0
    init_root_quat_yaw = lab_math.yaw_quat(loader._body_quat_w[:1, :1])
    loader._body_pos_w, loader._body_quat_w = lab_math.subtract_frame_transforms(
        init_root_pos_xy,
        init_root_quat_yaw.repeat(*loader._body_quat_w.shape[:2], 1),
        t02=loader._body_pos_w, q02=loader._body_quat_w
    )

    q_inv = lab_math.quat_inv(init_root_quat_yaw)
    loader._body_lin_vel_w = lab_math.quat_apply(q_inv, loader._body_lin_vel_w)
    loader._body_ang_vel_w = lab_math.quat_apply(q_inv, loader._body_ang_vel_w)


class MotionLoader:
    def __init__(self, motion_file: str,
                 track_body_names: Sequence[str] | None = None,
                 track_joint_names: Sequence[str] | None = None,
                 *,
                 default_motion_body_names: Sequence[str] | None = None,
                 default_motion_joint_names: Sequence[str] | None = None,
                 align_to_first_frame: bool = False,
                 device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        self.device = device
        data = np.load(motion_file)
        self.fps = data["fps"]

        if "body_names" in data:
            self._body_names = data["body_names"].tolist()
        else:
            assert (
                track_body_names is None
                or default_motion_body_names is not None), (
                f"Motion file {motion_file} missing body_names, "
                "and no default_body_names provided, "
                "But track_body_names is not None."
            )
            self._body_names = default_motion_body_names
        if "joint_names" in data:
            self._joint_names = data["joint_names"].tolist()
        else:
            assert (
                track_joint_names is None
                or default_motion_joint_names is not None), (
                f"Motion file {motion_file} missing joint_names,"
                "and no default_joint_names provided, "
                "But track_joint_names is not None."
            )
            self._joint_names = default_motion_joint_names

        self.track_body_names = track_body_names or self._body_names
        if self.track_body_names is None:
            self._body_indexes = torch.arange(
                data['body_pos_w'].shape[1], dtype=torch.long, device=device)
        else:
            self._body_indexes = torch.tensor(
                [self._body_names.index(name)
                 for name in self.track_body_names],
                dtype=torch.long, device=device
            )
        self.track_joint_names = track_joint_names or self._joint_names
        if self.track_joint_names is None:
            self._joint_indexes = torch.arange(
                data['joint_pos'].shape[1], dtype=torch.long, device=device)
        else:
            self._joint_indexes = torch.tensor(
                [self._joint_names.index(name)
                 for name in self.track_joint_names],
                dtype=torch.long, device=device
            )
        self.joint_pos = torch.tensor(
            data["joint_pos"],
            dtype=torch.float32, device=device)[:, self._joint_indexes]
        self.joint_vel = torch.tensor(
            data["joint_vel"],
            dtype=torch.float32, device=device)[:, self._joint_indexes]
        self._body_pos_w = torch.tensor(
            data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(
            data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(
            data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(
            data["body_ang_vel_w"], dtype=torch.float32, device=device)

        if align_to_first_frame:
            _align_to_first_frame(self)

        self.time_step_total = self.joint_pos.shape[0]

    def to(self, device: str | torch.device) -> None:
        self.device = device
        self.joint_pos = self.joint_pos.to(device)
        self.joint_vel = self.joint_vel.to(device)
        self._body_pos_w = self._body_pos_w.to(device)
        self._body_quat_w = self._body_quat_w.to(device)
        self._body_lin_vel_w = self._body_lin_vel_w.to(device)
        self._body_ang_vel_w = self._body_ang_vel_w.to(device)
        self._body_indexes = self._body_indexes.to(device)
        self._joint_indexes = self._joint_indexes.to(device)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class RawTensorMotionLoader:
    """Loads a reference trajectory saved as a bare `torch.save`d `[T, 448]`
    tensor (no field names, purely positional columns), as opposed to
    `MotionLoader`'s named-array npz format.

    Column layout: `0:3` root pos, `3:7` root quat (xyzw), `7:10` root lin
    vel, `10:13` root ang vel, `13:35` dof_pos, `35:57` dof_vel, `57:` onward
    is object/per-body/contact data used only by the tracker that produced
    this rollout -- ignored here, since the deployed policy is body-only.

    Exposes the same public surface as `MotionLoader` (`joint_pos`,
    `joint_vel`, `body_pos_w`, `body_quat_w`, `time_step_total`,
    `track_body_names`, `.to()`) with a single anchor "body" (the root),
    so policy code doesn't need to care which loader it has.
    """

    def __init__(self, motion_file: str,
                 *,
                 anchor_body_name: str = "Trunk",
                 align_to_first_frame: bool = False,
                 device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        self.device = device
        raw = torch.load(motion_file, map_location=device).to(torch.float32)

        root_pos = raw[:, 0:3]
        root_quat = lab_math.convert_quat(raw[:, 3:7], to="wxyz")
        root_lin_vel = raw[:, 7:10]
        root_ang_vel = raw[:, 10:13]
        self.joint_pos = raw[:, 13:35]
        self.joint_vel = raw[:, 35:57]

        self.track_body_names = [anchor_body_name]
        self._body_indexes = torch.zeros(1, dtype=torch.long, device=device)
        self._body_pos_w = root_pos.unsqueeze(1)
        self._body_quat_w = root_quat.unsqueeze(1)
        self._body_lin_vel_w = root_lin_vel.unsqueeze(1)
        self._body_ang_vel_w = root_ang_vel.unsqueeze(1)

        if align_to_first_frame:
            _align_to_first_frame(self)

        self.time_step_total = self.joint_pos.shape[0]

    def to(self, device: str | torch.device) -> None:
        self.device = device
        self.joint_pos = self.joint_pos.to(device)
        self.joint_vel = self.joint_vel.to(device)
        self._body_pos_w = self._body_pos_w.to(device)
        self._body_quat_w = self._body_quat_w.to(device)
        self._body_lin_vel_w = self._body_lin_vel_w.to(device)
        self._body_ang_vel_w = self._body_ang_vel_w.to(device)
        self._body_indexes = self._body_indexes.to(device)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]
