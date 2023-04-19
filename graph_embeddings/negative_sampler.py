# Third-party imports
import torch


class RandomCorruptionNegativeSampler(object):
    def __init__(self, n, use_cache=False):
        self.n = n

        self.use_cache = use_cache
        self.cache = {}

    def __call__(self, g, eids):
        all_nodes = g.nodes('_N')
        src, dst = g.find_edges(eids)

        repl_src = torch.where(torch.rand((len(src), self.n)) > 0.5, 1,
                               0).to(g.device)
        repl_dst = 1 - repl_src

        repl_node = all_nodes[torch.randint(len(all_nodes), repl_src.shape)]

        new_src = torch.where(repl_src == 1, repl_node,
                              torch.reshape(src, (-1, 1)).repeat([1, self.n]))
        new_dst = torch.where(repl_dst == 1, repl_node,
                              torch.reshape(dst, (-1, 1)).repeat([1, self.n]))

        return torch.reshape(new_src, [-1]), torch.reshape(new_dst, [-1])