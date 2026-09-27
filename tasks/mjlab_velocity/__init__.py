from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task
from .mjlab_velocity import K1MjlabVelocityControllerCfg

# Register booster_mjlab velocity tasks. Export a new checkpoint with scripts/export_rsl_rl_jit.py.


@configclass
class K1MjlabVelocityFlatControllerCfg(K1MjlabVelocityControllerCfg):
    '''Mjlab-Velocity-Flat-Booster-K1 policy.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/k1_velocity_flat.pt"


register_task(
    "k1_mjlab_velocity_flat", K1MjlabVelocityFlatControllerCfg())
