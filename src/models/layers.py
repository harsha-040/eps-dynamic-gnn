"""Shared dense graph-neural-network building blocks.

Implemented from scratch (rather than using torch_geometric's sparse
edge_index API) because both the dynamic and static GNN models need dense,
per-sample (and for the dynamic model, per-timestep) adjacency matrices
batched as (B, N, N) tensors — a shape PyG's dense layers only partially
support (e.g. DenseGATConv treats `adj` purely as a 0/1 mask, discarding
learned edge weights), and because interpretability requires directly
reading off attention weights, which PyG's dense layers don't expose.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGraphLearner(nn.Module):
    """Learns a per-sample adjacency A(t) from node features via a small
    MLP node encoder + scaled dot-product attention scoring, as described
    in DyGAT-FTNet/STGLR-style dynamic graph construction.

    Returns:
        a_soft: (B, N, N) row-normalized soft adjacency (a_soft[b,i,j] =
            weight of edge j->i), fed to the GCN layer as a weighted graph.
        mask:   (B, N, N) binary top-k sparsification of a_soft (plus self
            loops), fed to the GAT layer to restrict attention to the
            learned dynamic structure.
    """

    def __init__(self, in_dim: int, embed_dim: int = 32, top_k: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, N, F)
        B, N, _ = x.shape
        h = self.encoder(x)                                   # (B, N, D)
        scores = torch.matmul(h, h.transpose(-1, -2)) / math.sqrt(self.embed_dim)
        eye = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
        scores = scores.masked_fill(eye, float("-inf"))
        a_soft = torch.softmax(scores, dim=-1)                 # (B, N, N), rows sum to 1 over j != i

        k = min(self.top_k, N - 1)
        topk_val, topk_idx = a_soft.topk(k=k, dim=-1)
        mask = torch.zeros_like(a_soft)
        mask.scatter_(-1, topk_idx, 1.0)
        idx = torch.arange(N, device=x.device)
        mask[:, idx, idx] = 1.0  # self loops always allowed
        return a_soft, mask


class DenseGCNLayer(nn.Module):
    """Row-normalized (mean-aggregation) GCN over a dense weighted adjacency
    with an added self loop, i.e. structural propagation h_i' = W * mean_j
    over {i} U N(i) of (adj[i,j] * x_j)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        eye = torch.eye(N, device=x.device, dtype=adj.dtype).unsqueeze(0)
        adj_sl = adj + eye
        deg = adj_sl.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        adj_norm = adj_sl / deg
        return self.lin(torch.matmul(adj_norm, x))


class DenseGATLayer(nn.Module):
    """Multi-head dense GAT restricted to edges allowed by a binary mask,
    exposing the (head-averaged) attention weights for interpretability."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, 1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, 1, heads, out_dim))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: (B, N, Fin), mask: (B, N, N) with mask[b,i,j] = 1 if j->i allowed
        B, N, _ = x.shape
        h = self.lin(x).view(B, N, self.heads, self.out_dim)   # (B, N, H, C)
        alpha_src = (h * self.att_src).sum(dim=-1)              # (B, N, H) -- as source j
        alpha_dst = (h * self.att_dst).sum(dim=-1)              # (B, N, H) -- as dest i

        # alpha[b, i, j, h] = score of edge j -> i
        alpha = alpha_dst.unsqueeze(2) + alpha_src.unsqueeze(1)  # (B, N_i, N_j, H)
        alpha = self.leaky(alpha)
        alpha = alpha.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        alpha = torch.softmax(alpha, dim=2)                      # normalize over src j
        alpha = self.dropout(alpha)

        out = torch.einsum("bijh,bjhc->bihc", alpha, h)
        out = out.reshape(B, N, self.heads * self.out_dim)
        return out, alpha.mean(dim=-1)                            # (B,N,D), (B,N,N)
