"""Booster joint (motor) models.

Ported from booster_train's
`source/booster_train/booster_train/assets/robots/actuator.py` -- only the
physical motor data (torque/speed limits, torque-speed knee point,
armature), not the IsaacLab actuator classes that wrap it. The behaviour
those classes implement is reproduced elsewhere in this repo:

- command delay      -> `MujocoController._delayed_command`
- torque-speed curve -> `MujocoController._clip_effort`
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BoosterJoint:
    """One motor model's physical limits."""

    name: str
    effort_limit: float           # N*m available up to `knee_point_velocity`
    velocity_limit: float         # rad/s at which available torque hits zero
    knee_point_velocity: float    # rad/s where torque starts falling off
    armature: float               # kg*m^2 reflected rotor inertia

    def parallel(
        self,
        wrapper_name: str,
        serial_index: int,
        effort_ratio: tuple[float, float] = (1.0, 1.0),
        velocity_ratio: tuple[float, float] = (1.0, 1.0),
        armature_ratio: tuple[float, float] = (1.0, 1.0),
        knee_ratio: tuple[float, float] = (1.0, 1.0),
    ) -> BoosterJoint:
        """Effective model for one axis of a parallel mechanism driven by
        two of these motors (booster_train's `ParallelJointWrapperCfg`).
        `serial_index` is 0 for pitch, 1 for roll.
        """
        i = serial_index
        return BoosterJoint(
            name=f"{wrapper_name}({self.name})[{i}]",
            effort_limit=effort_ratio[i] * self.effort_limit,
            velocity_limit=velocity_ratio[i] * self.velocity_limit,
            knee_point_velocity=knee_ratio[i] * self.knee_point_velocity,
            armature=armature_ratio[i] * self.armature,
        )


E8116 = BoosterJoint("E8116", 130.0, 14.66, 6.28, 0.0636012)
E8112 = BoosterJoint("E8112", 96.0, 16.76, 7.54, 0.0523908)   # T1 hip pitch
E6408 = BoosterJoint("E6408", 68.0, 14.66, 1.88, 0.0478125)   # K1 hip pitch
E4315 = BoosterJoint("E4315", 76.0, 12.57, 2.62, 0.0339552)   # K1 hip roll
E4310 = BoosterJoint("E4310", 38.3, 17.59, 7.85, 0.0282528)   # K1 hip yaw
E6416 = BoosterJoint("E6416", 112.0, 12.57, 2.09, 0.095625)   # K1 knee
R14 = BoosterJoint("R14", 14.0, 33.51, 5.24, 0.001)           # K1 arm
HT4438 = BoosterJoint("HT4438", 6.0, 7.85, 10.47, 0.001)      # K1 neck
DM4310 = BoosterJoint("DM4310", 7.0, 12.57, 41.89, 0.0018)    # T1 neck

# K1's ankle is a parallel mechanism driven by two E4310s; only the
# reflected inertia differs from the base motor (armature_ratio 2.0).
_K1_ANKLE = dict(
    wrapper_name="BoosterK1AnkleParaWrapper", armature_ratio=(2.0, 2.0))
K1_ANKLE_PITCH = E4310.parallel(serial_index=0, **_K1_ANKLE)
K1_ANKLE_ROLL = E4310.parallel(serial_index=1, **_K1_ANKLE)


# Per-joint motor assignment, in `K1_CFG.joint_names` order.
K1_MOTORS: List[BoosterJoint] = [
    HT4438, HT4438,                                     # head yaw, pitch
    R14, R14, R14, R14,                                 # left arm
    R14, R14, R14, R14,                                 # right arm
    E6408, E4315, E4310, E6416,                         # left hip + knee
    K1_ANKLE_PITCH, K1_ANKLE_ROLL,                      # left ankle
    E6408, E4315, E4310, E6416,                         # right hip + knee
    K1_ANKLE_PITCH, K1_ANKLE_ROLL,                      # right ankle
]
