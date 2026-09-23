import glob
import os

from booster_deploy.utils.registry import register_task
from booster_deploy.utils.isaaclab.configclass import configclass
from .beyond_mimic import K1BeyondMimicControllerCfg


# @configclass
# class K1Smallbox047ControllerCfg(K1BeyondMimicControllerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.policy.motion_path = "motions/largebox_tracker/sub3_largebox_008_hold.npz"
#         self.policy.checkpoint_path = "models/k1_largebox_tracker_2026-09-18_01-36-50_globalw5_ar8_model_59000.pt"
#         self.robot.joint_stiffness = [
#             3.9478, 3.9478,                                              # head
#             3.9478, 3.9478, 3.9478, 3.9478,                              # left arm
#             3.9478, 3.9478, 3.9478, 3.9478,                              # right arm
#             30.2010, 21.4480, 17.8460, 60.4020, 35.6920, 35.6920,        # left leg
#             30.2010, 21.4480, 17.8460, 60.4020, 35.6920, 35.6920,        # right leg
#         ]
#         self.robot.joint_damping = [
#             0.2513, 0.2513,
#             0.2513, 0.2513, 0.2513, 0.2513,
#             0.2513, 0.2513, 0.2513, 0.2513,
#             3.6050, 2.5602, 2.1302, 4.8066, 4.2604, 4.2604,
#             3.6050, 2.5602, 2.1302, 4.8066, 4.2604, 4.2604,
#         ]
#         self.robot.effort_limit = [
#             6.0, 6.0,
#             14.0, 14.0, 14.0, 14.0,
#             14.0, 14.0, 14.0, 14.0,
#             68.0, 76.0, 38.3, 112.0, 38.3, 38.3,
#             68.0, 76.0, 38.3, 112.0, 38.3, 38.3,
#         ]

@configclass
class K1MJ2ControllerCfg(K1BeyondMimicControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = "motions/k1_mj2_seg1.npz"
        self.policy.checkpoint_path = "models/k1_mj_dance_002_2025-12-03_00-10-28.pt"
        self.robot.joint_stiffness = [
            10.0, 10.0,
            4., 4., 4., 4.,
            4., 4., 4., 4.,
            80., 80., 80., 80., 30., 30.,
            80., 80., 80., 80., 30., 30.,
        ]
        self.robot.joint_damping = [
            2., 2.,
            1., 1., 1., 1.,
            1., 1., 1., 1.,
            2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2.
        ]
        self.robot.effort_limit = [
            6, 6,
            14, 14, 14, 14,
            14, 14, 14, 14,
            30, 35, 20, 40, 20, 20,
            30, 35, 20, 40, 20, 20,
        ]


@configclass
class K1FightControllerCfg(K1BeyondMimicControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = "motions/k1_fight_final_deploy.npz"
        self.policy.checkpoint_path = "models/k1_fight_001.pt"
        self.robot.joint_stiffness = [
            10.0, 10.0,
            3.95, 3.95, 3.95, 3.95,
            3.95, 3.95, 3.95, 3.95,
            80., 80., 80., 80., 30., 30.,
            80., 80., 80., 80., 30., 30.,
        ]
        self.robot.joint_damping = [
            2., 2.,
            0.3, 0.3, 0.3, 0.3,
            0.3, 0.3, 0.3, 0.3,
            2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2.
        ]
        self.robot.effort_limit = [
            4, 4,
            12, 12, 12, 12,
            12, 12, 12, 12,
            30, 35, 20, 40, 20, 20,
            30, 35, 20, 40, 20, 20,
        ]


# # Largebox tracker motions are discovered, not listed: every *.npz in motions/largebox_tracker/ (this repo), plus every
# # *_hold*.npz in booster_train's tracker/npz folder when that checkout exists on this machine (e.g. the training box,
# # not the robot). A file in the local folder wins over a same-named one in booster_train. Each motion becomes a task
# # k1_largebox_<name>, where <name> is the file stem without "_largebox" and "_hold", e.g.
# #   sub3_largebox_008_hold.npz               -> k1_largebox_sub3_008
# #   sub10_largebox_075_hold_velinterp.npz    -> k1_largebox_sub10_075_velinterp
# #   sub10_largebox_075_ramp4lt_hold.npz      -> k1_largebox_sub10_075_ramp4lt
# # All of them run LARGEBOX_TRACKER_CHECKPOINT; per-motion policies go in LARGEBOX_TRACKER_OVERFIT below.
# _TASK_DIR = os.path.dirname(os.path.abspath(__file__))
# LARGEBOX_TRACKER_DIRS = [
#     os.path.join(_TASK_DIR, "motions", "largebox_tracker"),
#     os.environ.get(
#         "BOOSTER_TRACKER_NPZ_DIR",
#         os.path.expanduser("~/booster_train/booster_assets/motions/K1/tracker/npz"),
#     ),
# ]
# #LARGEBOX_TRACKER_CHECKPOINT = "models/k1_largebox_sub3_009_edited_100mm_300ms_iter49999.pt"
# # LARGEBOX_TRACKER_CHECKPOINT = "models/k1_largebox_tracker_sub3_009_edited_100mm_300ms_2026-09-19_13-39-10_finetune_anchor5.pt"
# # LARGEBOX_TRACKER_CHECKPOINT = "models/k1_suitcase_017_stand_2026-09-19_16-52-45_overfit_model_12000.pt"
# LARGEBOX_TRACKER_CHECKPOINT = "models/k1_largebox_tracker_sub7_002_stand_2026-09-19_19-06-08_overfit_stand_cont.pt"

# def largebox_task_name(stem: str) -> str:
#     return stem.replace("_largebox", "").replace("_hold", "")


# def discover_largebox_motions() -> dict[str, str]:
#     """{file stem: motion path relative to this task dir}, local folder first."""
#     motions: dict[str, str] = {}
#     for i, folder in enumerate(LARGEBOX_TRACKER_DIRS):
#         pattern = "*.npz" if i == 0 else "*_hold*.npz"
#         for path in sorted(glob.glob(os.path.join(folder, pattern))):
#             stem = os.path.splitext(os.path.basename(path))[0]
#             motions.setdefault(stem, os.path.relpath(path, _TASK_DIR))
#     return motions


# LARGEBOX_TRACKER_MOTIONS = discover_largebox_motions()


# @configclass
# class K1LargeboxTrackerControllerCfg(K1BeyondMimicControllerCfg):
#     """Gains and limits are the K1 actuator model the policy trained against (same as the smallbox task);
#     action_scale is derived from them as 0.25 * effort_limit / stiffness, so they must match training."""

#     motion: str = "motions/largebox_tracker/sub3_largebox_008_hold.npz"  # relative to this task dir
#     checkpoint: str = LARGEBOX_TRACKER_CHECKPOINT

#     def __post_init__(self):
#         super().__post_init__()
#         self.policy.motion_path = self.motion
#         self.policy.checkpoint_path = self.checkpoint
#         self.robot.joint_stiffness = [
#             3.9478, 3.9478,                                              # head
#             3.9478, 3.9478, 3.9478, 3.9478,                              # left arm
#             3.9478, 3.9478, 3.9478, 3.9478,                              # right arm
#             30.2010, 21.4480, 17.8460, 60.4020, 35.6920, 35.6920,        # left leg
#             30.2010, 21.4480, 17.8460, 60.4020, 35.6920, 35.6920,        # right leg
#         ]
#         self.robot.joint_damping = [
#             0.2513, 0.2513,
#             0.2513, 0.2513, 0.2513, 0.2513,
#             0.2513, 0.2513, 0.2513, 0.2513,
#             3.6050, 2.5602, 2.1302, 4.8066, 4.2604, 4.2604,
#             3.6050, 2.5602, 2.1302, 4.8066, 4.2604, 4.2604,
#         ]
#         self.robot.effort_limit = [
#             6.0, 6.0,
#             14.0, 14.0, 14.0, 14.0,
#             14.0, 14.0, 14.0, 14.0,
#             68.0, 76.0, 38.3, 112.0, 38.3, 38.3,
#             68.0, 76.0, 38.3, 112.0, 38.3, 38.3,
#         ]


register_task("k1_mj2", K1MJ2ControllerCfg())
# register_task("k1_fight", K1FightControllerCfg())
# register_task("k1_smallbox_047", K1Smallbox047ControllerCfg())
# # for _stem, _motion in LARGEBOX_TRACKER_MOTIONS.items():
# #     register_task(f"k1_largebox_{largebox_task_name(_stem)}", K1LargeboxTrackerControllerCfg(motion=_motion))

# # Single-motion (overfit) policies: booster_train task Booster-K1-Largebox-Tracker-<clip>-v0. Each only knows its own
# # motion, so it gets its own deploy task, k1_largebox_<name>_overfit, instead of replacing the shared checkpoint.
# # Keyed by motion file stem (any file discovered above).
# LARGEBOX_TRACKER_OVERFIT = {
#     "sub15_suitcase_017_stand_hold": "models/k1_largebox_tracker_sub15_suitcase_017_stand_2026-09-19_16-52-45_overfit_suitcase_model_49999.pt",
#     "sub10_largebox_075_hold_velinterp": "models/k1_largebox_tracker_sub10_075_2026-09-18_17-10-04_overfit_pos10_ori5_model_45000.pt",
#     "sub3_largebox_009_largebox_edited_100mm_300ms_hold": "models/k1_largebox_sub3_009_edited_100mm_300ms_iter49999.pt",
# }
# for _stem, _ckpt in LARGEBOX_TRACKER_OVERFIT.items():
#     if _stem not in LARGEBOX_TRACKER_MOTIONS:
#         print(f"[WARN] LARGEBOX_TRACKER_OVERFIT: no motion file named {_stem}.npz; skipping its _overfit task")
#         continue
#     register_task(
#         f"k1_largebox_{largebox_task_name(_stem)}_overfit",
#         K1LargeboxTrackerControllerCfg(motion=LARGEBOX_TRACKER_MOTIONS[_stem], checkpoint=_ckpt),
#     )

# # Second checkpoint on the sub3_009_edited_100mm_300ms motion, alongside the
# # _overfit (iter49999) task above -- same motion, different training run, to
# # compare against it directly rather than replacing it.
# register_task(
#     "k1_largebox_sub3_009_edited_100mm_300ms_anchor5",
#     K1LargeboxTrackerControllerCfg(
#         motion=LARGEBOX_TRACKER_MOTIONS["sub3_largebox_009_largebox_edited_100mm_300ms_hold"],
#         checkpoint="models/k1_largebox_tracker_sub3_009_edited_100mm_300ms_2026-09-19_13-39-10_finetune_anchor5.pt",
#     ),
# )
