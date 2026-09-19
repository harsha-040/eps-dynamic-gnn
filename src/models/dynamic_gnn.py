"""Proposed model: dynamic-graph GNN for EPS fault diagnosis.

At every timestep, a learned adjacency A(t) is computed from the node
features via a small MLP + attention scorer (DynamicGraphLearner), rather
than assuming the fixed SA->PCU->BAT/BUS->LOAD topology. That dynamic graph
is propagated through a GCN layer (structural mixing) and a GAT layer
(attention-weighted node importance), pooled over nodes, and the resulting
per-timestep graph embeddings are fused over the time window with a GRU
before a final FC+softmax classifier.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DynamicGraphLearner, DenseGCNLayer, DenseGATLayer


class DynamicGNN(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        n_features: int,
        n_classes: int,
        embed_dim: int = 32,
        gcn_dim: int = 32,
        gat_dim: int = 16,
        gat_heads: int = 2,
        top_k: int = 3,
        gru_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.graph_learner = DynamicGraphLearner(n_features, embed_dim=embed_dim, top_k=top_k)
        self.gcn = DenseGCNLayer(n_features, gcn_dim)
        self.gat = DenseGATLayer(gcn_dim, gat_dim, heads=gat_heads, dropout=dropout)
        gat_out_dim = gat_dim * gat_heads
        self.node_norm = nn.LayerNorm(gat_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=gat_out_dim, hidden_size=gru_hidden, batch_first=True)
        self.classifier = nn.Linear(gru_hidden, n_classes)

    def forward(self, x: torch.Tensor, return_interpret: bool = False):
        # x: (B, T, N, F)
        B, T, N, Fd = x.shape
        x_flat = x.reshape(B * T, N, Fd)

        a_soft, mask = self.graph_learner(x_flat)             # (B*T, N, N) each
        h = F.relu(self.gcn(x_flat, a_soft))
        h, alpha = self.gat(h, mask)                            # (B*T,N,D), (B*T,N,N)
        h = F.relu(h)
        h = self.node_norm(h)
        node_embed = h.view(B, T, N, -1)                        # kept for interpretability
        g = h.mean(dim=1)                                        # mean-pool over nodes -> (B*T, D)

        seq = g.view(B, T, -1)
        seq = self.dropout(seq)
        _, h_n = self.gru(seq)
        last = h_n[-1]                                            # (B, gru_hidden)
        logits = self.classifier(self.dropout(last))

        if return_interpret:
            extras = dict(
                a_soft=a_soft.view(B, T, N, N).detach(),
                attention=alpha.view(B, T, N, N).detach(),
                node_embed=node_embed.detach(),
            )
            return logits, extras
        return logits
