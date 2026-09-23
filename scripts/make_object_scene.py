"""Generate a MuJoCo scene XML: the K1 plus a captured object as a free body.

``mujoco_controller.py`` loads a single robot MJCF, so a pick-up task has nothing to pick up. This writes a
scene next to ``K1_22dof.xml`` by splicing the object's asset and body into a copy of it.

Everything is derived from the object's folder under ``booster_assets/objects/``, which holds
``<folder>.obj`` and ``<folder>.urdf``. The body is named after the folder minus its trailing scale suffix
(``largebox_0539923`` -> ``largebox``), and the scene is written to ``K1_22dof_<name>.xml`` -- the names
``tasks/hoi_track/__init__.py``'s ``OBJECTS`` table expects.

Splice rather than ``<include>`` on purpose. The robot model declares ``meshdir="meshes/"``, which MuJoCo
resolves relative to the file that declares it; pulling the robot into a wrapper file moves that resolution
and every STL path breaks. Writing the scene into the same directory keeps ``meshdir`` valid, and the object
mesh is reached back out through it (``meshes/../../objects/...``).

Friction comes from the URDF's ``<contact>`` block; mass is ``MASS``, not the URDF's -- see below. The geom is
the mesh itself, because Isaac's URDF importer gives PhysX the convex hull and MuJoCo does the same for a mesh
geom -- a box primitive would be a poor stand-in, since the suitcase hull fills only 53% of its bounding box.

Usage::

    python scripts/make_object_scene.py suitcase_0539923     # writes K1_22dof_suitcase.xml
    python scripts/make_object_scene.py largebox_0539923 --seed 3
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import xml.etree.ElementTree as ET

# Both captured URDFs declare 0.1 kg, and that is the nominal mass booster_train randomizes around. The scene
# uses MASS instead, with the URDF's diagonal inertia scaled by the same ratio so the object does not become
# denser than its size implies. Applies to whichever object is generated.
MASS = 0.5
ROLLING_FRICTION = 0.0001  # MuJoCo's third friction component; the URDF's rolling_friction fills the second

# Perturbation applied on top of the deterministic values above -- resampled every time this script runs, so
# re-running without --seed gives a different (but bounded) spawn each time.
# MuJoCo's geom friction is a single (sliding, torsional, rolling) triple, not PhysX's separate
# static/dynamic coefficients -- FRICTION_RANGE perturbs only the sliding component; torsional/rolling stay
# at their URDF-derived values.
POS_XY_RANGE = 0.2           # +/- m, applied to init_pos's x and y independently; z is untouched
YAW_RANGE_DEG = 30.0         # +/- deg about the world z-axis; roll/pitch are untouched
FRICTION_RANGE = (0.2, 1.2)  # sliding coefficient sampled uniformly in this range


def read_urdf(path: str) -> tuple[float, float, float, float]:
    """(mass, diagonal inertia, lateral friction, rolling friction) from a single-link object URDF."""
    link = ET.parse(path).getroot().find("link")
    inertial = link.find("inertial")
    mass = float(inertial.find("mass").get("value"))
    inertia = inertial.find("inertia")
    diag = [float(inertia.get(k)) for k in ("ixx", "iyy", "izz")]
    if len(set(diag)) != 1:
        raise SystemExit(f"{path}: non-uniform inertia {diag}; this script only writes diaginertia with one value")
    contact = link.find("contact")
    lateral = float(contact.find("lateral_friction").get("value"))
    rolling = float(contact.find("rolling_friction").get("value"))
    return mass, diag[0], lateral, rolling


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    objects_dir = f"{here}/booster_assets/objects"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("object", help="Folder name under booster_assets/objects/, e.g. largebox_0539923.")
    p.add_argument("--robot", default=f"{here}/booster_assets/robots/K1/K1_22dof.xml")
    p.add_argument("--name", default=None,
                   help="Body name; default is the folder name without its trailing _<scale> suffix.")
    p.add_argument("--init_pos", nargs=3, type=float, default=(0.0, -0.3, 0.17),
                   help="Spawn pose; the policy overwrites it at reset from the reference.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed the position/yaw/friction perturbation for a reproducible draw.")
    p.add_argument("--no_perturb", action="store_true",
                   help="Skip POS_XY_RANGE/YAW_RANGE_DEG/FRICTION_RANGE and use the exact --init_pos and URDF friction.")
    args = p.parse_args()

    folder = f"{objects_dir}/{args.object}"
    mesh_path = f"{folder}/{args.object}.obj"
    urdf_path = f"{folder}/{args.object}.urdf"
    for path in (mesh_path, urdf_path):
        if not os.path.isfile(path):
            available = ", ".join(sorted(os.listdir(objects_dir))) if os.path.isdir(objects_dir) else "none"
            raise SystemExit(f"missing {path}\navailable objects: {available}")
    name = args.name or re.sub(r"_\d+$", "", args.object)

    robot_dir = os.path.dirname(os.path.abspath(args.robot))
    robot_stem = os.path.splitext(os.path.basename(args.robot))[0]
    output = f"{robot_dir}/{robot_stem}_{name}.xml"

    xml = open(args.robot).read()
    if name in xml:
        raise SystemExit(f"'{name}' already present in {args.robot}")
    compiler = re.search(r'<compiler[^>]*\bmeshdir="([^"]*)"', xml)
    meshdir = os.path.join(robot_dir, compiler.group(1) if compiler else "")
    mesh_rel = os.path.relpath(mesh_path, meshdir)

    urdf_mass, urdf_inertia, lateral, rolling = read_urdf(urdf_path)
    diag_inertia = urdf_inertia * MASS / urdf_mass
    friction = (lateral, rolling, ROLLING_FRICTION)

    asset = f'    <mesh name="{name}" file="{mesh_rel}"/>\n'
    xml = re.sub(r"(\n\s*<mesh )", asset + r"\1", xml, count=1)

    rng = random.Random(args.seed)
    x, y, z = args.init_pos
    yaw_deg = 0.0
    sliding = friction[0]
    if not args.no_perturb:
        x += rng.uniform(-POS_XY_RANGE, POS_XY_RANGE)
        y += rng.uniform(-POS_XY_RANGE, POS_XY_RANGE)
        yaw_deg = rng.uniform(-YAW_RANGE_DEG, YAW_RANGE_DEG)
        sliding = rng.uniform(*FRICTION_RANGE)
    half = math.radians(yaw_deg) / 2.0
    quat = (math.cos(half), 0.0, 0.0, math.sin(half))  # yaw-only, w x y z

    body = f"""
    <!-- Captured object ({args.object}) for the hoi_track pick-up. Free joint, so it is driven only by contact
         and gravity; the policy writes its pose at reset from the reference clip. Spawn xy/yaw and sliding
         friction are perturbed per POS_XY_RANGE/YAW_RANGE_DEG/FRICTION_RANGE in make_object_scene.py. -->
    <body name="{name}" pos="{x} {y} {z}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}">
      <freejoint name="{name}_free"/>
      <inertial pos="0 0 0" mass="{MASS}"
                diaginertia="{diag_inertia} {diag_inertia} {diag_inertia}"/>
      <geom name="{name}_geom" type="mesh" mesh="{name}"
            friction="{sliding} {friction[1]} {friction[2]}"
            rgba="0.45 0.45 0.5 1" condim="4"/>
    </body>
"""
    xml = xml.replace("  </worldbody>", body + "  </worldbody>", 1)
    open(output, "w").write(xml)
    print(f"[INFO] wrote {output}")
    print(f"[INFO] object: {args.object} -> body '{name}', mesh {mesh_rel}")

    print(f"[INFO] fixed: mass={MASS} kg (URDF {urdf_mass} kg), "
          f"diaginertia={diag_inertia:.6f} (URDF {urdf_inertia}), "
          f"friction(torsional,rolling)=({friction[1]}, {friction[2]})")
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
        print(f"         sliding friction : init={friction[0]}  ->  {sliding:.4f}")

    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(output)
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        jid = m.body_jntadr[bid]
        print(f"[INFO] compiles: nq={m.nq} nv={m.nv} nbody={m.nbody}")
        print(f"[INFO] '{name}' body id {bid}, qpos[{m.jnt_qposadr[jid]}:{m.jnt_qposadr[jid] + 7}],"
              f" qvel[{m.jnt_dofadr[jid]}:{m.jnt_dofadr[jid] + 6}], mass {m.body_mass[bid]:.3f} kg")
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[WARN] could not compile the scene: {exc}")


main()
