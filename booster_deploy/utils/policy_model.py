"""Load a policy network from TorchScript (.pt) or ONNX (.onnx) behind one call interface.

Policies call the result like a TorchScript module: a float tensor in, an action tensor out, on the policy's
device. An ONNX model is run with onnxruntime on the CPU, which is plenty for these small MLPs.
"""
from __future__ import annotations

import os

import numpy as np
import torch


class OnnxPolicy:
    """onnxruntime session that takes and returns torch tensors.

    A 1-D input is batched for the model and the batch dimension dropped again on the way out, matching what a
    TorchScript MLP does with an unbatched input.
    """

    def __init__(self, path: str, device: str | torch.device = "cpu"):
        import onnxruntime as ort    # only needed for .onnx checkpoints

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1   # tiny MLP: threading only adds latency jitter
        self.session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"{path}: expected one input, got {[i.name for i in inputs]}")
        self.input_name = inputs[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.device = torch.device(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        unbatched = x.dim() == 1
        arr = x.detach().to("cpu", torch.float32).numpy()
        out = self.session.run([self.output_name], {self.input_name: arr[None] if unbatched else arr})[0]
        out = torch.from_numpy(np.asarray(out, dtype=np.float32))
        return (out[0] if unbatched else out).to(self.device)

    def to(self, device) -> "OnnxPolicy":
        self.device = torch.device(device)
        return self

    def eval(self) -> "OnnxPolicy":
        return self


def load_policy_model(path: str, device: str | torch.device = "cpu"):
    """TorchScript module or OnnxPolicy for ``path``, chosen by extension, ready for inference on ``device``."""
    if os.path.splitext(path)[1].lower() == ".onnx":
        return OnnxPolicy(path, device)
    return torch.jit.load(path, map_location=device).eval()
