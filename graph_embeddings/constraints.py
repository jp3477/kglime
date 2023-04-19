# Python imports

# Third-party imports
import torch
import numpy as np

# Package imports


class Constraint():
    def __init__(self, min_val=None, max_val=None):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, inputs):

        if self.max_val is not None and self.min_val is not None:
            return torch.sigmoid(inputs) * self.max_val + self.min_val
        elif self.min_val is not None:
            # return torch.clamp(inputs, min=self.min_val)
            return torch.abs(inputs)
        elif self.max_val is not None:
            # return torch.clamp(inputs, max=self.max_val)
            return torch.abs(inputs)


class PositiveConstraint(Constraint):
    def __init__(self):
        super().__init__(min_val=0.0)


class PhaseConstraint(Constraint):
    def __init__(self):
        super().__init__(min_val=0.0, max_val=2 * np.pi)


class HAKEConstraint():
    def __init__(self):
        super().__init__()

    def __call__(self, h):
        dims = h.shape[1]
        h_mod, h_phase = torch.split(h, [int(dims / 2), int(dims / 2)], dim=-1)
        h_phase = PhaseConstraint()(h_phase)
        return torch.cat([h_mod, h_phase], dim=-1)