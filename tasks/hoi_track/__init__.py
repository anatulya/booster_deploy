"""Auto-register deployable HOI policies from motion/model filenames.

Add a baked reference named ``motions/<clip>_ref.npz`` and a TorchScript
policy whose filename contains ``<clip>`` under ``models/``. The clip name
must contain one of the object keys in ``OBJECTS``. No per-clip config class
is needed.
"""

from __future__ import annotations

from pathlib import Path

from booster_deploy.utils.registry import register_task

from .hoi_track import K1HoiTrackControllerCfg


_TASK_DIR = Path(__file__).resolve().parent

# Adding a genuinely new object only requires an entry here and its MuJoCo
# scene (`python scripts/make_object_scene.py <folder under booster_assets/objects/>`
# writes K1_22dof_<object>.xml). Motions and checkpoints for known objects
# require no Python changes.
OBJECTS = {
    "suitcase": {
        "body_name": "suitcase",
        "scene": "{BOOSTER_ASSETS_DIR}/robots/K1/K1_22dof_suitcase.xml",
        "align_object_yaw": False,
        "yaw_align": True,
        "align_object_to_robot": True,
    },
    "largebox": {
        "body_name": "largebox",
        "scene": "{BOOSTER_ASSETS_DIR}/robots/K1/K1_22dof_largebox.xml",
        "align_object_yaw": False,
        "yaw_align": True,
        "align_object_to_robot": True,
    },
}


def _object_for_clip(clip: str) -> dict[str, str] | None:
    matches = [cfg for key, cfg in OBJECTS.items() if key in clip]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        print(
            f"[WARN] hoi_track: '{clip}' does not contain a known object "
            f"name ({', '.join(OBJECTS)}); skipping"
        )
    else:
        print(f"[WARN] hoi_track: '{clip}' names multiple objects; skipping")
    return None


def _checkpoint_for_clip(clip: str) -> Path | None:
    """Prefer ``<clip>.pt``; otherwise use the newest named export."""
    model_dir = _TASK_DIR / "models"
    exact = model_dir / f"{clip}.pt"
    if exact.is_file():
        return exact

    matches = sorted(model_dir.glob(f"*{clip}*.pt"))
    if not matches:
        print(f"[WARN] hoi_track: no model found for '{clip}'; skipping")
        return None
    if len(matches) > 1:
        print(
            f"[WARN] hoi_track: found {len(matches)} models for '{clip}'; "
            f"using newest filename '{matches[-1].name}'. Rename the desired "
            f"one to '{clip}.pt' to select it explicitly."
        )
    return matches[-1]


def discover_hoi_tasks() -> None:
    motion_dir = _TASK_DIR / "motions"
    for motion in sorted(motion_dir.glob("*_ref.npz")):
        clip = motion.name.removesuffix("_ref.npz")
        object_cfg = _object_for_clip(clip)
        if object_cfg is None:
            continue
        checkpoint = _checkpoint_for_clip(clip)
        if checkpoint is None:
            continue

        cfg = K1HoiTrackControllerCfg()
        cfg.policy.motion_path = str(motion.relative_to(_TASK_DIR))
        cfg.policy.checkpoint_path = str(checkpoint.relative_to(_TASK_DIR))
        cfg.policy.object_body_name = object_cfg["body_name"]
        cfg.policy.align_object_yaw = object_cfg["align_object_yaw"]
        cfg.policy.yaw_align = object_cfg["yaw_align"]
        cfg.policy.align_object_to_robot = object_cfg.get(
            "align_object_to_robot", False)
        cfg.mujoco.scene_mjcf_path = object_cfg["scene"]
        register_task(f"k1_hoi_{clip}", cfg)


discover_hoi_tasks()
