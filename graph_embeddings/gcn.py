# Python imports

# Third-party imports
import torch
from torch import nn
import numpy as np
import dgl
from dgl.nn import TypedLinear
import dgl.function as fn
from dgl.nn.functional import edge_softmax

# Package imports
from .hake import HAKE


class RGCNModel(nn.Module):
    def __init__(self,
                 device,
                 G,
                 n_in,
                 n_out,
                 n_rels,
                 n_layers=1,
                 basis=None,
                 dropout=0.0,
                 regularizer='basis'):
        super().__init__()

        self.G = G
        self.n_in = n_in
        self.n_out = n_out
        self.n_rels = n_rels
        self.device = device

        self.rgcn_block = RGCNBlock(G,
                                    n_in,
                                    n_out,
                                    n_rels,
                                    n_layers=n_layers,
                                    n_bases=basis,
                                    dropout=dropout,
                                    regularizer=regularizer)

        self.hake = HAKE(n_rels, n_out)

    def forward(self, inputs):
        triples, blocks = inputs
        blocks = [block.to(self.device) for block in blocks]

        out = self.rgcn_block(blocks, mode='blocks')
        h_h, r, h_t = out[triples[:, :, 0], :], triples[:, :,
                                                        1], out[triples[:, :,
                                                                        2], :]

        h_h_mod, h_h_phase = torch.chunk(h_h, 2, dim=-1)
        h_t_mod, h_t_phase = torch.chunk(h_t, 2, dim=-1)

        scores = self.hake((h_h_mod, h_t_mod, h_h_phase, h_t_phase, r))
        return scores


class RGCNBlock(nn.Module):
    def __init__(self,
                 G,
                 n_in,
                 n_out,
                 n_rels,
                 n_layers=1,
                 n_bases=None,
                 dropout=0.0,
                 regularizer='basis'):
        super().__init__()

        n_nodes = G.num_nodes()
        self.G = G

        self.embedding_layer = torch.nn.Embedding(n_nodes, n_in)

        self.rgcn_block = nn.ModuleList()
        for i in range(n_layers):
            activation = None if i == n_layers - 1 else nn.ReLU()
            rgcn = RelGraphConvWithAttn(n_in,
                                        n_out,
                                        n_rels,
                                        regularizer=regularizer,
                                        num_bases=n_bases,
                                        activation=activation,
                                        dropout=dropout,
                                        layer_norm=True)

            self.rgcn_block.append(rgcn)

        self.negatifier = nn.Parameter(torch.Tensor(n_nodes, int(n_out / 2)))
        torch.nn.init.normal_(self.negatifier)

    def forward(self, inputs, mode='graph'):
        graph = inputs
        if mode == 'blocks':
            blocks = graph

            x = self.embedding_layer(
                blocks[0].ndata[dgl.NID]['_N'][blocks[0].srcnodes()])

            for i, rgcn in enumerate(self.rgcn_block):
                block = blocks[i]
                x = rgcn(block,
                         x,
                         block.edata['rel_type'],
                         norm=block.edata['norm'])

            x_x, x_y = torch.chunk(x, 2, dim=-1)
            x_m = torch.sqrt(torch.square(x_x) + torch.square(x_y))
            x_p = torch.atan2(x_y, x_x) + np.pi

            negatifier = self.negatifier[block.ndata[dgl.NID]['_N'][
                block.dstnodes()]]
            x_m = x_m * negatifier

            x = torch.cat([x_m, x_p], dim=-1)

            return x
        elif mode == 'graph':
            x = self.embedding_layer(graph.nodes())

            for i, rgcn in enumerate(self.rgcn_block):
                x = rgcn(graph,
                         x,
                         graph.edata['rel_type'],
                         norm=graph.edata['norm'])

            x_x, x_y = torch.chunk(x, 2, dim=-1)
            x_m = torch.sqrt(torch.square(x_x) + torch.square(x_y))
            x_p = torch.atan2(x_y, x_x) + np.pi

            negatifier = self.negatifier
            x_m = x_m * negatifier

            x = torch.cat([x_m, x_p], dim=-1)

            return x


class RelGraphConvWithAttn(nn.Module):
    r"""Relational graph convolution layer from `Modeling Relational Data with Graph
    Convolutional Networks <https://arxiv.org/abs/1703.06103>`__

    It can be described in as below:

    .. math::

       h_i^{(l+1)} = \sigma(\sum_{r\in\mathcal{R}}
       \sum_{j\in\mathcal{N}^r(i)}e_{j,i}W_r^{(l)}h_j^{(l)}+W_0^{(l)}h_i^{(l)})

    where :math:`\mathcal{N}^r(i)` is the neighbor set of node :math:`i` w.r.t. relation
    :math:`r`. :math:`e_{j,i}` is the normalizer. :math:`\sigma` is an activation
    function. :math:`W_0` is the self-loop weight.

    The basis regularization decomposes :math:`W_r` by:

    .. math::

       W_r^{(l)} = \sum_{b=1}^B a_{rb}^{(l)}V_b^{(l)}

    where :math:`B` is the number of bases, :math:`V_b^{(l)}` are linearly combined
    with coefficients :math:`a_{rb}^{(l)}`.

    The block-diagonal-decomposition regularization decomposes :math:`W_r` into :math:`B`
    number of block diagonal matrices. We refer :math:`B` as the number of bases.

    The block regularization decomposes :math:`W_r` by:

    .. math::

       W_r^{(l)} = \oplus_{b=1}^B Q_{rb}^{(l)}

    where :math:`B` is the number of bases, :math:`Q_{rb}^{(l)}` are block
    bases with shape :math:`R^{(d^{(l+1)}/B)*(d^{l}/B)}`.

    Parameters
    ----------
    in_feat : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feat : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    num_rels : int
        Number of relations.
    regularizer : str, optional
        Which weight regularizer to use ("basis", "bdd" or ``None``):

         - "basis" is for basis-decomposition.
         - "bdd" is for block-diagonal-decomposition.
         - ``None`` applies no regularization.

        Default: ``None``.
    num_bases : int, optional
        Number of bases. It comes into effect when a regularizer is applied.
        If ``None``, it uses number of relations (``num_rels``). Default: ``None``.
        Note that ``in_feat`` and ``out_feat`` must be divisible by ``num_bases``
        when applying "bdd" regularizer.
    bias : bool, optional
        True if bias is added. Default: ``True``.
    activation : callable, optional
        Activation function. Default: ``None``.
    self_loop : bool, optional
        True to include self loop message. Default: ``True``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    layer_norm: bool, optional
        True to add layer norm. Default: ``False``

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import RelGraphConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = RelGraphConv(10, 2, 3, regularizer='basis', num_bases=2)
    >>> etype = th.tensor([0,1,2,0,1,2])
    >>> res = conv(g, feat, etype)
    >>> res
    tensor([[ 0.3996, -2.3303],
            [-0.4323, -0.1440],
            [ 0.3996, -2.3303],
            [ 2.1046, -2.8654],
            [-0.4323, -0.1440],
            [-0.1309, -1.0000]], grad_fn=<AddBackward0>)
    """
    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        regularizer=None,
        num_bases=None,
        bias=True,
        activation=None,
        self_loop=True,
        dropout=0.0,
        layer_norm=False,
    ):
        super().__init__()
        if regularizer is not None and num_bases is None:
            num_bases = num_rels
        self.linear_r = TypedLinear(in_feat, out_feat, num_rels, regularizer,
                                    num_bases)
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # TODO(minjie): consider remove those options in the future to make
        #   the module only about graph convolution.
        # layer norm
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feat,
                                                  elementwise_affine=True)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain("relu"))

        # weights for attention

        self.attn_weight = nn.Parameter(torch.Tensor(in_feat, in_feat))
        nn.init.xavier_uniform_(self.attn_weight)

        self.attn_vec = nn.Parameter(torch.Tensor(3 * in_feat, 1))
        nn.init.xavier_uniform_(self.attn_vec)

        self.m = nn.Parameter(torch.Tensor(num_rels, in_feat))
        nn.init.xavier_uniform_(self.m)

        self.dropout = nn.Dropout(dropout)

    def get_attn(self, edges, g):
        e = torch.matmul(
            torch.cat([
                torch.matmul(edges.src['h'], self.attn_weight),
                torch.matmul(self.m[edges.data['etype'], :], self.attn_weight),
                torch.matmul(edges.dst['h'], self.attn_weight)
            ],
                      dim=1), self.attn_vec)

        attn = edge_softmax(g, e, norm_by='dst')

        return attn

    def message(self, edges, g=None):
        """Message function."""
        m = self.linear_r(edges.src["h"], edges.data["etype"], self.presorted)
        attn = self.get_attn(edges, g)
        if "norm" in edges.data:
            m = m * edges.data["norm"] * attn
        return {"m": m}

    def forward(self, block, feat, etypes, norm=None, *, presorted=False):
        """Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
        etypes : torch.Tensor or list[int]
            An 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
        norm : torch.Tensor, optional
            An 1D tensor of edge norm value.  Shape: :math:`(|E|,)`.
        presorted : bool, optional
            Whether the edges of the input graph have been sorted by their types.
            Forward on pre-sorted graph may be faster. Graphs created
            by :func:`~dgl.to_homogeneous` automatically satisfy the condition.
            Also see :func:`~dgl.reorder_graph` for sorting edges manually.

        Returns
        -------
        torch.Tensor
            New node features. Shape: :math:`(|V|, D_{out})`.
        """
        self.presorted = presorted

        with block.local_scope():
            h_src = feat
            h_dst = feat[:block.number_of_dst_nodes()]

            block.srcdata['h'] = h_src
            block.dstdata['h'] = h_dst
            block.edata['etype'] = etypes

            if norm is not None:
                block.edata['norm'] = norm

            # message passing
            block.update_all(lambda edges: self.message(edges, block),
                             fn.sum("m", "h"))

            # apply bias and activation
            h = block.dstdata['h']
            if self.layer_norm:
                h = self.layer_norm_weight(h)
            if self.bias:
                h = h + self.h_bias
            if self.self_loop:
                h = h + feat[:block.num_dst_nodes()] @ self.loop_weight
            if self.activation:
                h = self.activation(h)

            h = self.dropout(h)
            return h