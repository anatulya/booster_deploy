"""Generate a static InterMimic-format reference trajectory: a single pose
held for N frames, saved as the same `[T, 448]` raw-tensor layout
`RawTensorMotionLoader` reads (root pos/quat-xyzw/vel/angvel, dof_pos,
dof_vel; the remaining object/body/contact columns are zero-filled since
the deployed body-only policy never reads them).

Useful for testing a tracking policy against a trivial standing target
instead of a full motion clip.
"""
import argparse
import sys
import torch

sys.path.append(".")

from booster_deploy.robots.booster import K1_CFG  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--frames", type=int, default=500)
parser.add_argument("--root-height", type=float, default=0.6)
args = parser.parse_args()


def main():
    dof_pos = torch.tensor(K1_CFG.default_joint_pos, dtype=torch.float32)
    num_dof = dof_pos.shape[0]

    frame = torch.zeros(448, dtype=torch.float32)
    frame[0:3] = torch.tensor([0.0, 0.0, args.root_height])
    frame[3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])  # identity, xyzw
    frame[13:13 + num_dof] = dof_pos

    motion = frame.unsqueeze(0).repeat(args.frames, 1)
    torch.save(motion, args.output)
    print(f"Saved {tuple(motion.shape)} standing reference to {args.output}")


if __name__ == "__main__":
    main()
