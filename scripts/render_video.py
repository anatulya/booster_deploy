"""Render a MuJoCo rollout of a task to an mp4, without a display (for headless machines).

    python scripts/render_video.py --task k1_mjlab_velocity_flat --out walk.mp4
    python scripts/render_video.py --task k1_mjlab_velocity_flat --out turn.mp4 \
        --segment 3 0 0 0 --segment 5 0.5 0 0 --segment 5 0 0 0.8

Each ``--segment SECONDS VX VY VYAW`` holds that velocity command for that long, in order (tasks without
velocity commands only use the durations). The default schedule stands, walks forward, turns, then stops.
Needs ffmpeg on PATH. Rendering uses EGL unless MUJOCO_GL is already set.
"""
import argparse
import importlib
import os
import pkgutil
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.append(".")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True, help="Registered task name (see deploy.py --list).")
parser.add_argument("--out", default="out.mp4", help="Output mp4 path.")
parser.add_argument("--segment", type=float, nargs=4, action="append", metavar=("SECONDS", "VX", "VY", "VYAW"),
                    help="Hold this velocity command for SECONDS; repeat for a schedule.")
parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="Run this checkpoint (.pt or .onnx) instead of the task's own, e.g. another iteration "
         "or run of the same task.")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--distance", type=float, default=2.5, help="Camera distance from the robot (m).")
args = parser.parse_args()

DEFAULT_SCHEDULE = [[2, 0, 0, 0], [5, 0.5, 0, 0], [5, 0, 0, 0.8], [3, 0, 0, 0]]


def main():
    import tasks as tasks_pkg
    for mod_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        importlib.import_module(mod_info.name)
    from booster_deploy.utils.registry import get_task
    from booster_deploy.controllers.mujoco_controller import MujocoController

    cfg = get_task(args.task)
    if args.checkpoint:
        cfg.policy.checkpoint_path = os.path.abspath(args.checkpoint)
        print(f"Using checkpoint {os.path.abspath(args.checkpoint)}")
    controller = MujocoController(cfg)
    controller.update_vel_command = lambda: None   # commands come from the schedule, not stdin
    model, data = controller.mj_model, controller.mj_data

    # The offscreen buffer bounds the frame size; MJCFs usually leave it at 640x480.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = args.distance, -20, 135

    fps = round(1.0 / cfg.policy_dt)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{args.width}x{args.height}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", args.out],
        stdin=subprocess.PIPE)

    schedule = args.segment or DEFAULT_SCHEDULE
    controller.update_state()
    controller.start()
    frames = 0
    for seconds, vx, vy, vyaw in schedule:
        if controller.vel_command is not None:
            cmd = controller.vel_command
            cmd.lin_vel_x, cmd.lin_vel_y, cmd.ang_vel_yaw = vx, vy, vyaw
            print(f"command x={vx} y={vy} yaw={vyaw} for {seconds}s")
        for _ in range(round(seconds * fps)):
            if not controller.is_running:
                break
            controller.update_state()
            controller.ctrl_step(controller.policy_step())
            cam.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, camera=cam)
            proc.stdin.write(renderer.render().tobytes())
            frames += 1
        if not controller.is_running:
            print("policy stopped (safety fallback); ending the video early")
            break

    proc.stdin.close()
    proc.wait()
    renderer.close()
    print(f"wrote {args.out}: {frames} frames, {frames / fps:.1f}s")


if __name__ == "__main__":
    main()
