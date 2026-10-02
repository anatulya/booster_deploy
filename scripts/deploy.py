import argparse
import sys

sys.path.append(".")

parser = argparse.ArgumentParser()
# require either --task or --list (mutually exclusive)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--task", type=str, help="Name of the configuration file.")
group.add_argument("-l", "--list", action="store_true", dest="list_tasks",
                   default=False, help="list available tasks")

parser.add_argument("--net", type=str, default="127.0.0.1",
                    help="Network interface for SDK communication.")
parser.add_argument("--mujoco", action="store_true", default=False,
                    help="deploy in mujoco simulation")
parser.add_argument("--webots", action="store_true", default=False,
                    help="deploy in webots simulation")
parser.add_argument("--record", type=str, default=None,
                    help="mujoco only: skip the interactive viewer (which needs a "
                         "display) and instead render offscreen to this .mp4 path. "
                         "Use this when running headless.")
parser.add_argument("--record-seconds", type=float, default=15.0,
                    help="mujoco --record only: length of the recorded clip in seconds.")
parser.add_argument("--scene", type=str, default=None,
                    help="mujoco only: override the task's MJCF scene, e.g. "
                         "{BOOSTER_ASSETS_DIR}/robots/K1/K1_22dof_parallel_largebox.xml for the parallel-ankle robot.")
parser.add_argument("--decimation", type=int, default=None,
                    help="mujoco only: physics steps per 20 ms policy step (default 10, i.e. 2 ms). The parallel-ankle "
                         "MJCF is tuned for 1 ms: use 20.")
parser.add_argument("--cmd", type=float, nargs=3, default=None, metavar=("VX", "VY", "VYAW"),
                    help="mujoco only: initial velocity command for velocity tasks. Needed with --record, which "
                         "has no terminal input; interactively it can still be changed by typing 'x y yaw'.")
parser.add_argument("--log", type=str, default=None, metavar="DIR",
                    help="Log every policy step (network input/output, named observation terms, robot state) "
                         "under DIR/<task>_<time>/obs. On the robot this also records a ROS bag of "
                         "ROSBAG_TOPICS into DIR/<task>_<time>/rosbag. Read logs with "
                         "booster_deploy.utils.obs_logger.load_obs_log or scripts/replay_log.py.")
parser.add_argument(
    "--device", type=str, default="cpu",
    help="Device to run the evaluation on (e.g., 'cpu', 'cuda')")
args = parser.parse_args()


# Recorded with `ros2 bag record` on robot runs with --log. /low_state is the robot's state at 500 Hz
# (IMU, motor q/dq/tau_est); joint_ctrl is what this process commands.
ROSBAG_TOPICS = ["/low_state", "/joint_ctrl"]


def start_rosbag(run_dir: str):
    """Start `ros2 bag record` as a child process; None if ros2 isn't available."""
    import shutil
    import subprocess
    if shutil.which("ros2") is None:
        print("[log] ros2 not found on PATH (source the ROS setup script first); recording observations only")
        return None
    out = f"{run_dir}/rosbag"
    print(f"[log] recording ROS bag of {' '.join(ROSBAG_TOPICS)} to {out}")
    return subprocess.Popen(["ros2", "bag", "record", "-o", out, *ROSBAG_TOPICS])


def stop_rosbag(proc) -> None:
    """SIGINT lets ros2 bag write its metadata; fall back to terminate if it doesn't exit."""
    import signal
    import subprocess
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=2.0)


def main():
    # load task registry and dispatch
    import pkgutil
    import tasks as tasks_pkg

    # auto-import all submodules under tasks (recursive) so they can register themselves
    for mod_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        full_name = mod_info.name
        try:
            __import__(full_name)
        except Exception as e:
            raise e
    from booster_deploy.utils.registry import get_task, list_tasks

    if args.list_tasks:
        print("Available tasks:")
        for task_name, cfg in list_tasks().items():
            cls = type(cfg)
            full_cls = f"{cls.__module__}.{cls.__qualname__}"
            print(f"  {task_name}\t:\t{full_cls}")
        sys.exit(0)

    try:
        task_cfg = get_task(args.task)
    except KeyError:
        print(f"Unknown task '{args.task}'. Available tasks: {list(list_tasks().keys())}")
        sys.exit(1)

    # Set device for policy
    task_cfg.policy.device = args.device
    task_cfg.task_name = args.task

    run_dir = None
    if args.log:
        import os
        import time
        run_dir = os.path.abspath(f"{args.log}/{args.task}_{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(run_dir, exist_ok=True)
        task_cfg.log_dir = f"{run_dir}/obs"

    # decide how to run based on flags
    if args.mujoco:
        if args.record:
            # Offscreen (EGL) rendering doesn't need an X display, but MuJoCo
            # picks its GL backend from this env var the first time any of its
            # rendering submodules is imported, so it must be set before that
            # happens.
            import os as _os
            _os.environ.setdefault("MUJOCO_GL", "egl")

        # run mujoco controller
        from booster_deploy.controllers.mujoco_controller import MujocoController

        if args.record:
            task_cfg.mujoco.record_video_path = args.record
            task_cfg.mujoco.record_video_seconds = args.record_seconds
        if args.scene:
            task_cfg.mujoco.scene_mjcf_path = args.scene
        if args.decimation:
            task_cfg.mujoco.decimation = args.decimation
            task_cfg.mujoco.physics_dt = task_cfg.policy_dt / args.decimation

        controller = MujocoController(task_cfg)
        if args.cmd:
            if controller.vel_command is None:
                print(f"--cmd ignored: task '{args.task}' takes no velocity command")
            else:
                cmd = controller.vel_command
                cmd.lin_vel_x, cmd.lin_vel_y, cmd.ang_vel_yaw = args.cmd
        controller.run()
        if run_dir:
            print(f"[log] run logged to {run_dir}")
    else:
        # initialize network and run robot portal
        try:
            from booster_robotics_sdk_python import ChannelFactory  # type: ignore
            ChannelFactory.Instance().Init(0, args.net)
        except ImportError as e:
            print(
                "Error: booster_robotics_sdk_python is not installed.\n"
                "Please install it to use real robot deployment.\n"
                "For MuJoCo simulation, use --mujoco flag instead."
            )
            sys.exit(1)

        # adjust ankle dampings for webots
        if args.webots:
            ankles = [-8, -7, -2, -1]  # indices of ankle joints
            for i in ankles:
                task_cfg.robot.joint_damping[i] = 0.5

        from booster_deploy.controllers.booster_robot_controller import BoosterRobotPortal
        rosbag = start_rosbag(run_dir) if run_dir else None
        try:
            with BoosterRobotPortal(task_cfg, use_sim_time=args.webots) as portal:
                portal.run()
        finally:
            stop_rosbag(rosbag)
            if run_dir:
                print(f"[log] run logged to {run_dir}")


if __name__ == "__main__":
    main()
