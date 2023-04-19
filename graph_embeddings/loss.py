# Python imports

# Third-party imports
import torch
from torch import nn

# Package imports


class AdverserialLoss():
    def __init__(self, margin=5.0, temp=0.7):
        super().__init__()

        self.margin = margin
        self.temp = temp

    def __call__(self, y_pred):
        y_pos, y_neg = y_pred[:, 0:1], y_pred[:, 1:]
        p = nn.Softmax(dim=-1)(self.temp * y_neg)

        loss = -nn.LogSigmoid()(self.margin - y_pos * -1) \
            - torch.sum(p * nn.LogSigmoid()(-1 * y_neg - self.margin), dim=-1, keepdim=True)

        return torch.mean(loss)