"""Export an rsl_rl MLP actor checkpoint (e.g. from booster_mjlab) to TorchScript.

The deploy policies load TorchScript, while rsl_rl saves a state dict. This rebuilds the deterministic
actor -- EmpiricalNormalization followed by the MLP -- so the exported module maps a raw observation to the
action mean exactly as the training-time ONNX export does.

    python scripts/export_rsl_rl_jit.py \
        ~/booster_mjlab/logs/rsl_rl/k1_velocity/<run>/model_<iter>.pt \
        tasks/mjlab_velocity/models/k1_velocity_flat.pt

Pass ``--onnx <run>.onnx`` to check the export against the ONNX policy (needs onnxruntime).
"""

import argparse

import torch
from torch import nn

ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}


class Actor(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float, mlp: nn.Sequential):
        super().__init__()
        self.register_buffer("obs_mean", mean)
        self.register_buffer("obs_std", std)
        self.eps = eps
        self.mlp = mlp

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp((obs - self.obs_mean) / (self.obs_std + self.eps))


def build_actor(checkpoint: str, activation: str, eps: float) -> Actor:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]

    linear_ids = sorted(int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith(".weight"))
    layers: list[nn.Module] = []
    for n, i in enumerate(linear_ids):
        weight, bias = state[f"mlp.{i}.weight"], state[f"mlp.{i}.bias"]
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(bias)
        layers.append(linear)
        if n < len(linear_ids) - 1:
            layers.append(ACTIVATIONS[activation]())

    in_dim = layers[0].in_features
    if "obs_normalizer._mean" in state:
        mean = state["obs_normalizer._mean"].reshape(in_dim)
        std = state["obs_normalizer._std"].reshape(in_dim)
    else:
        mean, std, eps = torch.zeros(in_dim), torch.ones(in_dim), 0.0
    return Actor(mean, std, eps, nn.Sequential(*layers)).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="rsl_rl model_<iter>.pt")
    parser.add_argument("output", help="TorchScript file to write")
    parser.add_argument("--activation", default="elu", choices=ACTIVATIONS)
    parser.add_argument("--eps", type=float, default=1e-2, help="rsl_rl EmpiricalNormalization eps")
    parser.add_argument("--onnx", default=None, help="ONNX export of the same checkpoint to compare against")
    args = parser.parse_args()

    actor = build_actor(args.checkpoint, args.activation, args.eps)
    scripted = torch.jit.script(actor)
    scripted.save(args.output)
    in_dim = actor.obs_mean.numel()
    print(f"Saved {args.output}: obs {in_dim} -> action {actor.mlp[-1].out_features}")

    if args.onnx:
        import onnxruntime as ort

        obs = torch.randn(64, in_dim) * 2.0
        sess = ort.InferenceSession(args.onnx)
        name = sess.get_inputs()[0].name
        ref = torch.cat([torch.from_numpy(sess.run(None, {name: o[None].numpy()})[0]) for o in obs])
        with torch.no_grad():
            out = torch.jit.load(args.output)(obs)
        print(f"max |jit - onnx| = {(out - ref).abs().max().item():.2e}")


if __name__ == "__main__":
    main()
