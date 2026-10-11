from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task
from .mjlab_velocity import K1MjlabVelocityControllerCfg

# Register booster_mjlab velocity tasks. Checkpoints are TorchScript (export one with
# scripts/export_rsl_rl_jit.py) or the .onnx that booster_mjlab writes next to its checkpoints.


@configclass
class K1MjlabVelocityFlatControllerCfg(K1MjlabVelocityControllerCfg):
    '''Mjlab-Velocity-Flat-Booster-K1 policy.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/k1_velocity_flat.pt"


@configclass
class K1MjlabVelocityFlatOnnxControllerCfg(K1MjlabVelocityControllerCfg):
    '''Mjlab-Velocity-Flat-Booster-K1 policy, ONNX export of run 2026-09-27_14-59-54 iteration 8600.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/k1_velocity_flat_8600.onnx"


register_task(
    "k1_mjlab_velocity_flat", K1MjlabVelocityFlatControllerCfg())
register_task(
    "k1_mjlab_velocity_flat_onnx", K1MjlabVelocityFlatOnnxControllerCfg())
