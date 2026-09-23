import math

import torch


class StartTransition:
    """Joint reference for a policy holding a motion's first frame.

    Used while a policy holds its first frame (PolicyCfg.hold_start_frame).
    The position reference is the first frame from the start; only the
    velocity reference is shaped. It points from the robot's pose when the
    policy started toward the first frame, then decays exponentially to 0:

        vel(t) = (goal - start) / tau * exp(-t / tau)

    so the policy sees the target pose at once plus a fading cue of which
    way to move. After 5 tau (under 1% left) it snaps to 0, handing over to
    the static first-frame hold exactly.

    Joints only: the anchor orientation reference stays at the first frame.
    Blending it from the robot's upright stance made the tracking policies
    fall, as that upright trunk reference is far from anything they trained on.
    """

    def __init__(self, start_pos: torch.Tensor, goal_pos: torch.Tensor,
                 tau_s: float):
        self.goal_pos = goal_pos.clone()
        self.vel0 = (goal_pos - start_pos) / tau_s
        self.tau_s = tau_s
        # How long the cue lasts; the motion can't be released before this.
        self.duration_s = 5 * tau_s

    def done(self, t: float) -> bool:
        return t >= self.duration_s

    def sample(self, t: float) -> tuple[torch.Tensor, torch.Tensor]:
        """(joint pos, joint vel) of the reference at time t."""
        if self.done(t):
            return self.goal_pos, torch.zeros_like(self.vel0)
        return self.goal_pos, self.vel0 * math.exp(-max(t, 0.0) / self.tau_s)
