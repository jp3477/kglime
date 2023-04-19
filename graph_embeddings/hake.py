# Python imports

# Third-party imports
import torch
from torch import nn
import numpy as np

# Package imports


class HAKE(nn.Module):
    def __init__(self, n_rels, dim):
        super().__init__()

        self.n_rels = n_rels
        self.n_dim = dim

        self.rel_embedding = HAKEEmbedding(n_rels, dim)

        self.lam = nn.Parameter(torch.tensor([1.], requires_grad=True))
        self.lam2 = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

    def forward(self, inputs):
        h_head_m, h_tail_m, h_head_p, h_tail_p, rels = inputs

        h_rel = self.rel_embedding(rels)
        h_rel_m, h_rel_p = torch.chunk(h_rel, 2, dim=-1)

        d_m = torch.norm(torch.multiply(h_head_m, h_rel_m) - h_tail_m,
                         p=2,
                         dim=-1)
        d_p = torch.norm(torch.sin((h_head_p + h_rel_p - h_tail_p) / 2),
                         p=1,
                         dim=-1)

        score = -(self.lam2 * d_m + self.lam * d_p)
        return score


class HAKEEmbedding(nn.Embedding):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)

    def forward(self, inputs):
        embedding = super().forward(inputs)
        embedding_x, embedding_y = torch.chunk(embedding, 2, dim=-1)
        embedding_m = torch.sqrt(
            torch.square(embedding_x) + torch.square(embedding_y))
        embedding_p = torch.atan2(embedding_y, embedding_x) + np.pi

        return torch.cat([embedding_m, embedding_p], dim=-1)