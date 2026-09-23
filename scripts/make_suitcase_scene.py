"""Generate a MuJoCo scene XML: the K1 plus the captured suitcase as a free body.

``mujoco_controller.py`` loads a single robot MJCF, so a pick-up task has nothing to pick up. This writes a
scene next to ``K1_22dof.xml`` by splicing the object's asset and body into a copy of it.

Splice rather than ``<include>`` on purpose. The robot model declares ``meshdir="meshes/"``, which MuJoCo
resolves relative to the file that declares it; pulling the robot into a wrapper file moves that resolution
and every STL path breaks. Writing the scene into the same directory keeps ``meshdir`` valid, and the
suitcase mesh is reached back out through it (``meshes/../../objects/...``).

Friction is taken from ``suitcase_0539923.urdf`` (lateral 0.9); mass is 3.0 kg, not the URDF's 0.1 -- see
``MASS`` below. The geom is the mesh itself, because
Isaac's URDF importer gives PhysX the convex hull and MuJoCo does the same for a mesh geom -- a box primitive
would be a poor stand-in here, since the hull fills only 53% of the mesh's bounding box.

Usage::

    python scripts/make_suitcase_scene.py            # writes K1_22dof_suitcase.xml
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re

# The URDF declares 0.1 kg with a 0.002 diagonal inertia, and that is what booster_train trains against. This
# scene uses 3.0 kg instead (6x the captured 0.5 kg that eval_tracker_with_box.py and pt_to_npz --object_mass
# default to), with the inertia scaled by the same 30x so the box does not become denser than its size implies.
# It is a deliberate mismatch with training: a heavier box than the policy ever lifted.
URDF_MASS = 0.1
URDF_DIAG_INERTIA = 0.002
MASS = 0.5
DIAG_INERTIA = URDF_DIAG_INERTIA * MASS / URDF_MASS
FRICTION = (0.9, 0.5, 0.0001)  # sliding, torsional, rolling -- lateral/rolling from the URDF's <contact>

# Perturbation applied on top of the deterministic values above -- resampled every time this script runs, so
# re-running with the same --init_pos/--seed-less invocation gives a different (but bounded) spawn each time.
# MuJoCo's geom friction is a single (sliding, torsional, rolling) triple, not PhysX's separate
# static/dynamic coefficients -- FRICTION_RANGE perturbs only the sliding component; torsional/rolling stay
# at their FRICTION values above.
POS_XY_RANGE = 0.2        # +/- m, applied to init_pos's x and y independently; z is untouched
YAW_RANGE_DEG = 30.0        # +/- deg about the world z-axis; roll/pitch are untouched
FRICTION_RANGE = (0.2, 1.2)  # sliding coefficient sampled uniformly in this range


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default=f"{here}/booster_assets/robots/K1/K1_22dof.xml")
    p.add_argument("--output", default=f"{here}/booster_assets/robots/K1/K1_22dof_suitcase.xml")
    p.add_argument("--mesh", default="../../../objects/suitcase_0539923/suitcase_0539923.obj",
                   help="Object mesh, relative to the robot model's meshdir.")
    p.add_argument("--name", default="suitcase")
    p.add_argument("--init_pos", nargs=3, type=float, default=(0.0, -0.3, 0.17),
                   help="Spawn pose; the policy overwrites it at reset from the reference.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed the position/yaw/friction perturbation for a reproducible draw.")
    p.add_argument("--no_perturb", action="store_true",
                   help="Skip POS_XY_RANGE/YAW_RANGE_DEG/FRICTION_RANGE and use the exact --init_pos/FRICTION.")
    args = p.parse_args()

    if os.path.dirname(args.output) != os.path.dirname(args.robot):
        raise SystemExit("output must sit beside the robot model, or meshdir stops resolving")

    xml = open(args.robot).read()
    if args.name in xml:
        raise SystemExit(f"'{args.name}' already present in {args.robot}")

    asset = f'    <mesh name="{args.name}" file="{args.mesh}"/>\n'
    xml = re.sub(r"(\n\s*<mesh )", asset + r"\1", xml, count=1)

    rng = random.Random(args.seed)
    x, y, z = args.init_pos
    yaw_deg = 0.0
    sliding = FRICTION[0]
    if not args.no_perturb:
        x += rng.uniform(-POS_XY_RANGE, POS_XY_RANGE)
        y += rng.uniform(-POS_XY_RANGE, POS_XY_RANGE)
        yaw_deg = rng.uniform(-YAW_RANGE_DEG, YAW_RANGE_DEG)
        sliding = rng.uniform(*FRICTION_RANGE)
    half = math.radians(yaw_deg) / 2.0
    quat = (math.cos(half), 0.0, 0.0, math.sin(half))  # yaw-only, w x y z

    body = f"""
    <!-- Captured object for the hoi_track pick-up. Free joint, so it is driven only by contact and gravity;
         the policy writes its pose at reset from the reference clip. Spawn xy/yaw and sliding friction are
         perturbed per POS_XY_RANGE/YAW_RANGE_DEG/FRICTION_RANGE above (see --no_perturb/--seed). -->
    <body name="{args.name}" pos="{x} {y} {z}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}">
      <freejoint name="{args.name}_free"/>
      <inertial pos="0 0 0" mass="{MASS}"
                diaginertia="{DIAG_INERTIA} {DIAG_INERTIA} {DIAG_INERTIA}"/>
      <geom name="{args.name}_geom" type="mesh" mesh="{args.name}"
            friction="{sliding} {FRICTION[1]} {FRICTION[2]}"
            rgba="0.45 0.45 0.5 1" condim="4"/>
    </body>
"""
    xml = xml.replace("  </worldbody>", body + "  </worldbody>", 1)
    open(args.output, "w").write(xml)
    print(f"[INFO] wrote {args.output}")

    print(f"[INFO] fixed: mass={MASS} kg (URDF {URDF_MASS} kg), "
          f"diaginertia={DIAG_INERTIA:.6f} (URDF {URDF_DIAG_INERTIA}), "
          f"friction(torsional,rolling)=({FRICTION[1]}, {FRICTION[2]})")
    if args.no_perturb:
        print(f"[INFO] perturbation off (--no_perturb): "
              f"pos=({x:.4f}, {y:.4f}, {z:.4f}), yaw=0.00 deg, sliding_friction={sliding:.4f}")
    else:
        print(f"[INFO] perturbation ranges: pos_xy=+/-{POS_XY_RANGE} m, "
              f"yaw=+/-{YAW_RANGE_DEG} deg, sliding_friction={FRICTION_RANGE}")
        print(f"[INFO] perturbation draw (seed={args.seed}):")
        print(f"         x                : init={args.init_pos[0]:+.4f}  delta={x - args.init_pos[0]:+.4f}"
              f"  ->  {x:+.4f} m")
        print(f"         y                : init={args.init_pos[1]:+.4f}  delta={y - args.init_pos[1]:+.4f}"
              f"  ->  {y:+.4f} m")
        print(f"         z                : {z:+.4f} m (unperturbed)")
        print(f"         yaw              : {yaw_deg:+.2f} deg  (roll/pitch unperturbed, "
              f"quat={tuple(round(c, 4) for c in quat)})")
        print(f"         sliding friction : init={FRICTION[0]}  ->  {sliding:.4f}")

    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(args.output)
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, args.name)
        jid = m.body_jntadr[bid]
        print(f"[INFO] compiles: nq={m.nq} nv={m.nv} nbody={m.nbody}")
        print(f"[INFO] '{args.name}' body id {bid}, qpos[{m.jnt_qposadr[jid]}:{m.jnt_qposadr[jid] + 7}],"
              f" qvel[{m.jnt_dofadr[jid]}:{m.jnt_dofadr[jid] + 6}], mass {m.body_mass[bid]:.3f} kg")
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[WARN] could not compile the scene: {exc}")


main()
