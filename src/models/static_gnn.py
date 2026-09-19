"""Baseline: static-graph GNN (fixed adjacency), reproducing the ACS-paper
approach (Wu et al., IEEE EIECT 2025) adapted to the EPS topology. Same
GCN -> GAT -> GRU -> classifier architecture and parameter budget as the
dynamic model, but the adjacency is the fixed SA/BAT/PCU/BUS/LOAD power-flow
topology for every sample and every timestep, never learned or updated.
This isolates the effect of dynamic vs. static graph structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DenseGCNLayer, DenseGATLayer


def build_fixed_adjacency(static_edge_index, n_nodes: int) -> torch.Tensor:
    """static_edge_index: (2, E) array-like of [src_row, dst_row] as used in
    the simulation topology (edges given as (src, dst) pairs). Returns a
    (N, N) binary adjacency with adj[dst, src] = 1, matching the
    edge-weight convention adj[i,j] = weight of edge j->i used by the GCN
    layer here."""
    adj = torch.zeros(n_nodes, n_nodes)
    src, dst = static_edge_index[0], static_edge_index[1]
    for s, d in zip(src, dst):
        adj[d, s] = 1.0
    return adj


class StaticGNN(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        n_features: int,
        n_classes: int,
        static_edge_index,
        gcn_dim: int = 32,
        gat_dim: int = 16,
        gat_heads: int = 2,
        gru_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        fixed_adj = build_fixed_adjacency(static_edge_index, n_nodes)
        fixed_mask = (fixed_adj > 0).float()
        idx = torch.arange(n_nodes)
        fixed_mask[idx, idx] = 1.0
        self.register_buffer("fixed_adj", fixed_adj)
        self.register_buffer("fixed_mask", fixed_mask)

        self.gcn = DenseGCNLayer(n_features, gcn_dim)
        self.gat = DenseGATLayer(gcn_dim, gat_dim, heads=gat_heads, dropout=dropout)
        gat_out_dim = gat_dim * gat_heads
        self.node_norm = nn.LayerNorm(gat_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=gat_out_dim, hidden_size=gru_hidden, batch_first=True)
        self.classifier = nn.Linear(gru_hidden, n_classes)

    def forward(self, x: torch.Tensor, return_interpret: bool = False):
        B, T, N, Fd = x.shape
        x_flat = x.reshape(B * T, N, Fd)

        adj = self.fixed_adj.unsqueeze(0).expand(B * T, -1, -1)
        mask = self.fixed_mask.unsqueeze(0).expand(B * T, -1, -1)

        h = F.relu(self.gcn(x_flat, adj))
        h, alpha = self.gat(h, mask)
        h = F.relu(h)
        h = self.node_norm(h)
        g = h.mean(dim=1)

        seq = g.view(B, T, -1)
        seq = self.dropout(seq)
        _, h_n = self.gru(seq)
        last = h_n[-1]
        logits = self.classifier(self.dropout(last))

        if return_interpret:
            extras = dict(
                a_soft=adj.view(B, T, N, N).detach(),
                attention=alpha.view(B, T, N, N).detach(),
            )
            return logits, extras
        return logits
