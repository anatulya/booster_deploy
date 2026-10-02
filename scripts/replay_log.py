"""Replay a run logged with `deploy.py --log` (sim or real robot).

    python scripts/replay_log.py <run dir or its obs/ dir> --mode open-loop [--checkpoint other.pt]
    python scripts/replay_log.py <run> --mode view   [--video out.mp4] [--speed 0.5] [--no-ghost]
    python scripts/replay_log.py <run> --mode resim  [--video out.mp4] [--seed 0] [--steps N]

open-loop  Runs the checkpoint on every logged network input and compares with the logged output. Confirms the
           deployed network saw and did exactly what the log says; with --checkpoint, tests another policy
           against the same (e.g. real-robot) observations.
view       Kinematic playback of the logged joint angles and trunk orientation. The robot doesn't measure its
           root position, so xy stays at the origin and the height keeps the lowest foot on the floor. The
           ghost shows the policy's reference joint pose from the logged observation.
resim      Closed-loop re-run in MuJoCo from the logged starting state, with the motion released at the logged
           step and the logged velocity commands. The sim run is logged next to the original and compared
           term by term. Root velocity, contacts and (in hoi tasks) the object pose are not known from a robot
           log, so this is a side-by-side comparison, not an exact reproduction: expect the runs to drift apart.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time

sys.path.append(".")

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run", help="Run directory from deploy.py --log (or its obs/ subdirectory).")
parser.add_argument("--mode", choices=["open-loop", "view", "resim"], required=True)
parser.add_argument("--checkpoint", default=None, help="open-loop: run this checkpoint instead of the logged one.")
parser.add_argument("--allow-mismatch", action="store_true",
                    help="Proceed even if the checkpoint/motion files differ from the logged sha256.")
parser.add_argument("--video", default=None, help="view/resim: render to this mp4 instead of opening a viewer.")
parser.add_argument("--speed", type=float, default=1.0, help="view/resim viewer: playback speed.")
parser.add_argument("--no-ghost", action="store_true", help="view/resim: hide the reference ghost.")
parser.add_argument("--steps", type=int, default=None, help="resim: number of steps (default: the log's length).")
parser.add_argument("--seed", type=int, default=0, help="resim: seed for the sim's random draws (actuator delay).")
args = parser.parse_args()

# Keys in a logged config that describe the original run rather than the policy, so are not re-applied.
_SKIP_CFG_KEYS = {"constructor", "log_dir", "task_name", "record_video_path", "device"}


def obs_dir(path: str) -> str:
    return path if os.path.isfile(os.path.join(path, "meta.json")) else os.path.join(path, "obs")


def apply_logged_cfg(obj, data: dict) -> None:
    """Set every attribute present in both the task config and the logged config. Lenient where
    update_class_from_dict is strict (None defaults, int/float), since a log is plain JSON."""
    for key, value in data.items():
        if key in _SKIP_CFG_KEYS or not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if isinstance(value, dict) and current is not None and not isinstance(current, dict):
            apply_logged_cfg(current, value)
        elif callable(current):
            continue
        else:
            setattr(obj, key, value)


def load_tasks():
    import pkgutil
    import tasks as tasks_pkg
    for mod_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        __import__(mod_info.name)


def rebuild_cfg(meta: dict):
    from booster_deploy.utils.registry import get_task
    if not meta.get("task"):
        raise SystemExit("log has no task name (it predates deploy.py setting task_name); cannot rebuild")
    cfg = get_task(meta["task"])
    apply_logged_cfg(cfg, meta["cfg"])
    cfg.task_name = meta["task"]
    cfg.log_dir = None
    cfg.mujoco.record_video_path = None
    # A robot log's hold setting lives under `booster`; the MuJoCo controller reads it from `mujoco`.
    cfg.mujoco.hold_start_frame = bool(meta["cfg"]["policy"].get("hold_start_frame", False))
    return cfg


def local_file(cfg, key: str) -> str | None:
    """Where this checkout keeps the policy's checkpoint/motion file. Resolved against the local task
    directory rather than the logged absolute path, so a robot log replays on another machine."""
    import inspect
    rel = getattr(cfg.policy, key, None)
    if not isinstance(rel, str):
        return None
    if os.path.isabs(rel):
        return rel
    task_dir = os.path.dirname(inspect.getfile(cfg.policy.constructor))
    return os.path.join(task_dir, rel)


def check_files(meta: dict, cfg) -> None:
    from booster_deploy.utils.obs_logger import file_sha256
    bad = []
    for key, logged in meta.get("files", {}).items():
        path = local_file(cfg, key) or logged["path"]
        sha = file_sha256(path)
        if sha != logged["sha256"]:
            bad.append(f"{key}: {path} {'missing' if sha is None else 'differs from the logged file'}")
    if bad:
        print("WARNING: files differ from the logged run:\n  " + "\n  ".join(bad))
        if not args.allow_mismatch:
            raise SystemExit("pass --allow-mismatch to replay anyway")


def make_controller(cfg):
    from booster_deploy.controllers.mujoco_controller import MujocoController
    with contextlib.redirect_stdout(io.StringIO()):
        return MujocoController(cfg)


# ---------------------------------------------------------------- pose helpers

class FootGrounding:
    """Root height that puts the lowest foot where it sits when the robot stands on the floor at spawn."""

    def __init__(self, c):
        import mujoco
        self.mujoco = mujoco
        self.c = c
        m = c.mj_model
        self.feet = [b for b in range(m.nbody)
                     if "ankle_roll" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")]
        mujoco.mj_forward(m, c.mj_data)
        self.foot_z0 = min(c.mj_data.xpos[b][2] for b in self.feet)

    def set_root(self, data, quat, joint_pos) -> None:
        data.qpos[0:3] = [0.0, 0.0, 1.0]
        data.qpos[3:7] = quat
        data.qpos[self.c._jq] = joint_pos
        self.mujoco.mj_forward(self.c.mj_model, data)
        data.qpos[2] -= min(data.xpos[b][2] for b in self.feet) - self.foot_z0
        self.mujoco.mj_forward(self.c.mj_model, data)


def ref_joint_pos_real(log, meta, i):
    terms = log["terms"]
    key = "ref_joint_pos" if "ref_joint_pos" in terms else ("k0/ref_joint_pos" if "k0/ref_joint_pos" in terms else None)
    if key is None:
        return None
    sim = terms[key][i]
    return sim[[meta["sim_joint_names"].index(n) for n in meta["joint_names"]]]


# ---------------------------------------------------------------- frame output

class FrameSink:
    """Either an interactive viewer (paced at the log rate / --speed) or an mp4 via the offscreen renderer."""

    def __init__(self, c, ghost: bool):
        import mujoco
        self.mujoco, self.c, self.ghost = mujoco, c, ghost
        self.dt = c.cfg.policy_dt / max(args.speed, 1e-3)
        self.viewer = self.proc = None
        if args.video:
            import subprocess
            m = c.mj_model
            self.w = min(c.cfg.mujoco.record_video_width, m.vis.global_.offwidth)
            self.h = min(c.cfg.mujoco.record_video_height, m.vis.global_.offheight)
            self.renderer = mujoco.Renderer(m, height=self.h, width=self.w)
            self.cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(m, self.cam)
            self.cam.distance, self.cam.azimuth, self.cam.elevation = 3.0, 180, -20
            self.ghost_scene = mujoco.MjvScene(m, maxgeom=1000) if ghost else None
            fps = round(1.0 / c.cfg.policy_dt)
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{self.w}x{self.h}", "-r", str(fps), "-i", "-", "-an", "-vcodec", "libx264",
                 "-pix_fmt", "yuv420p", args.video], stdin=subprocess.PIPE)
        else:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(c.mj_model, c.mj_data)
            self.viewer.cam.azimuth, self.viewer.cam.elevation = 180, -20

    def alive(self) -> bool:
        return self.viewer is None or self.viewer.is_running()

    def frame(self) -> None:
        c = self.c
        if self.proc is not None:
            self.cam.lookat[:] = c.mj_data.qpos[0:3]
            self.renderer.update_scene(c.mj_data, camera=self.cam)
            if self.ghost_scene is not None:
                c._composite_ghost_geoms(self.renderer, self.ghost_scene, self.cam)
            self.proc.stdin.write(self.renderer.render().tobytes())
        else:
            if self.ghost:
                c.render_reference_robot(self.viewer)
            self.viewer.cam.lookat[:] = c.mj_data.qpos[0:3]
            self.viewer.sync()
            time.sleep(self.dt)

    def close(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
            self.renderer.close()
            print(f"saved {args.video}")
        elif self.viewer is not None:
            self.viewer.close()


# ---------------------------------------------------------------- modes

def open_loop(log, meta, cfg) -> None:
    path = args.checkpoint or local_file(cfg, "checkpoint_path") or meta["files"]["checkpoint_path"]["path"]
    model = torch.jit.load(path, map_location="cpu").eval()
    obs = torch.from_numpy(log["obs"].astype(np.float32))
    with torch.no_grad():
        out = np.stack([model(o[None]).reshape(-1).numpy() for o in obs])
    diff = np.abs(out - log["action"])
    per_step = diff.max(axis=1)
    first = int(np.argmax(per_step > 1e-4)) if (per_step > 1e-4).any() else None
    print(f"open-loop {os.path.basename(path)} on {len(obs)} logged inputs: max |action diff| {diff.max():.2e}, "
          f"mean {diff.mean():.2e}; " + ("identical within 1e-4" if first is None
                                         else f"first step differing by >1e-4: {int(log['step'][first])}"))


def view(log, meta, cfg) -> None:
    cfg.mujoco.visualize_reference_ghost = not args.no_ghost
    c = make_controller(cfg)
    obj = getattr(cfg.policy, "object_body_name", None)
    if obj is not None:
        # The log doesn't say where the object was, so keep it (and its ghost) out of the picture.
        c.set_object_pose(obj, [0.0, 0.0, -10.0], [1.0, 0.0, 0.0, 0.0])
    ground = FootGrounding(c)
    sink = FrameSink(c, ghost=not args.no_ghost)
    try:
        for i in range(len(log["step"])):
            if not sink.alive():
                break
            ground.set_root(c.mj_data, log["root_quat_w"][i], log["joint_pos"][i])
            ref = ref_joint_pos_real(log, meta, i)
            if ref is not None and not args.no_ghost:
                c.set_reference_qpos(np.concatenate([c.mj_data.qpos[:7], ref]))
            sink.frame()
    finally:
        sink.close()


def resim(log, meta, cfg, run_dir: str) -> None:
    from booster_deploy.utils.obs_logger import load_obs_log
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = os.path.join(run_dir, f"resim_{time.strftime('%Y%m%d-%H%M%S')}")
    cfg.log_dir = os.path.join(out_dir, "obs")
    cfg.mujoco.visualize_reference_ghost = not args.no_ghost
    c = make_controller(cfg)

    # Start from the first logged step's state. Root linear/angular velocity aren't known from a robot log.
    FootGrounding(c).set_root(c.mj_data, log["root_quat_w"][0], log["joint_pos"][0])
    c.mj_data.qvel[:] = 0.0
    c.mj_data.qvel[c._jv] = log["joint_vel"][0]

    release = next((e["step"] for e in log["events"] if e["event"] == "motion_released"), None)
    n = min(args.steps or len(log["step"]), len(log["step"]))
    sink = FrameSink(c, ghost=not args.no_ghost)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            c.update_state()
            c.start()
        for i in range(n):
            if not (c.is_running and sink.alive()):
                break
            step = i + 1
            if release is not None and step >= release:
                c.request_motion_release()
            if c.vel_command is not None:
                c.vel_command.lin_vel_x, c.vel_command.lin_vel_y, c.vel_command.ang_vel_yaw = \
                    (float(v) for v in log["vel_cmd"][i])
            c.update_state()
            c.ctrl_step(c.policy_step())
            sink.frame()
        if c.is_running:
            c._log_event("stop", reason="resim ended")
    finally:
        c.close_logs()
        sink.close()

    sim = load_obs_log(cfg.log_dir)
    compare(log, sim, release)
    print(f"resim logged to {out_dir}")


def compare(real, sim, release) -> None:
    n = min(len(real["step"]), len(sim["step"]))
    print(f"\nresim vs log over {n} steps (release at step {release}). "
          "Root velocity, contacts and object pose are unknown from a robot log, so drift is expected.")
    jp = np.abs(sim["joint_pos"][:n] - real["joint_pos"][:n]).max(axis=1)
    over = np.nonzero(jp > 0.1)[0]
    print(f"  first step with a joint >0.1 rad apart: {int(real['step'][over[0]]) if len(over) else 'none'}")
    window = max(1, round(1.0 / real["meta"]["policy_dt"]))
    names = [k for k in ("joint_pos", "joint_vel", "base_ang_vel", "last_action") if k in real["terms"]]
    rows = [("joint_pos(state)", real["joint_pos"][:n], sim["joint_pos"][:n]),
            ("action", real["action"][:n], sim["action"][:n])]
    rows += [(f"obs:{k}", real["terms"][k][:n], sim["terms"][k][:n]) for k in names if k in sim["terms"]]
    print("  RMS difference per second:")
    print("   " + "".join(f"{'%ds' % (s // window):>9s}" for s in range(0, n, window)) + "   term")
    for label, a, b in rows:
        vals = [np.sqrt(np.mean((a[s:s + window] - b[s:s + window]) ** 2)) for s in range(0, n, window)]
        print("   " + "".join(f"{v:9.3f}" for v in vals) + f"   {label}")


def main() -> None:
    from booster_deploy.utils.obs_logger import load_obs_log
    d = obs_dir(args.run)
    log = load_obs_log(d)
    meta = log["meta"]
    if len(log["step"]) == 0:
        raise SystemExit("log has no steps")
    load_tasks()
    cfg = rebuild_cfg(meta)
    check_files(meta, cfg)
    print(f"{meta['task']}: {len(log['step'])} steps logged by {meta['backend']}")
    if args.mode == "open-loop":
        open_loop(log, meta, cfg)
    elif args.mode == "view":
        view(log, meta, cfg)
    else:
        resim(log, meta, cfg, os.path.dirname(d) if os.path.basename(d) == "obs" else d)


if __name__ == "__main__":
    if args.video:
        os.environ.setdefault("MUJOCO_GL", "egl")
    main()
