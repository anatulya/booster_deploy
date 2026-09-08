from booster_deploy.utils.registry import register_task
from booster_deploy.utils.isaaclab.configclass import configclass
from .intermimic import K1InterMimicControllerCfg


@configclass
class K1InterMimicSmallboxCfg(K1InterMimicControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/k1_smallbox_toreal_newphys_ep5001.pt"
        self.policy.motion_path = (
            "motions/sub4_smallbox_047_smallbox_intermimic_original.pt"
        )


register_task("k1_intermimic", K1InterMimicSmallboxCfg())
