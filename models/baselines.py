"""Comparison baselines for ET-NAGraphSAGE — fair-protocol adaptations.

All three consume the SAME batch dict as ETNAGraphSAGE:
    node_seq       [B, T, 6]          ego (target) node feature sequence
    nbr_node_seqs  [B, K1, T, 6]      1-hop neighbor node sequences
    edge_seqs      [B, K1, T, 5]      ego→nbr edge (relative kinematic) sequences
    nbr_mask       [B, K1]
    nbr2_node_seqs [B, K1, K2, T, 6]  2-hop node sequences
    nbr2_edge_seqs [B, K1, K2, T, 5]  nbr→nbr2 edge sequences
    nbr2_mask      [B, K1, K2]
and output target-node logits [B, num_classes].

Baselines are adapted to the per-scene ego-graph classification setting while
staying faithful to each model's core mechanism:
  - STGCN  : gated temporal 1-D conv + spatial graph conv, STATIC (distance/mask) edges.
  - DCRNN  : diffusion graph conv inside a GRU over T frames, STATIC edges.
  - TGN    : per-node memory (GRU) updated by timestamped EDGE-FEATURE messages (edge-temporal).

Same data / split / 500-epoch budget / metric as the proposed model — only the
model differs. STGCN & DCRNN use edges as static structure (no edge features over
time); TGN consumes edge feature sequences (the edge-temporal counterpart to ours).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared: build the flat 2-hop ego-graph (ego + K1 + K1*K2 nodes) + adjacency
# ─────────────────────────────────────────────────────────────────────────────
def build_ego_graph(batch):
    """Return X [B,N,T,C], node_mask [B,N], A [B,N,N] (binary, symmetric, no self-loop).
    Node layout: 0 = ego, 1..K1 = 1-hop, 1+K1.. = 2-hop (nbr i's j-th at 1+K1 + i*K2 + j)."""
    node_seq       = batch['node_seq']        # [B,T,C]
    nbr_node_seqs  = batch['nbr_node_seqs']   # [B,K1,T,C]
    nbr_mask       = batch['nbr_mask']        # [B,K1]
    nbr2_node_seqs = batch['nbr2_node_seqs']  # [B,K1,K2,T,C]
    nbr2_mask      = batch['nbr2_mask']       # [B,K1,K2]

    B, K1, T, C = nbr_node_seqs.shape
    K2 = nbr2_node_seqs.shape[2]
    N = 1 + K1 + K1 * K2
    dev = node_seq.device

    X = torch.zeros(B, N, T, C, device=dev)
    X[:, 0] = node_seq
    X[:, 1:1 + K1] = nbr_node_seqs
    X[:, 1 + K1:] = nbr2_node_seqs.reshape(B, K1 * K2, T, C)

    node_mask = torch.cat([
        torch.ones(B, 1, device=dev),
        nbr_mask,
        nbr2_mask.reshape(B, K1 * K2),
    ], dim=1)                                                    # [B,N]

    A = torch.zeros(B, N, N, device=dev)
    # ego(0) — 1-hop(1..K1)
    for i in range(K1):
        w = nbr_mask[:, i]
        A[:, 0, 1 + i] = w
        A[:, 1 + i, 0] = w
    # 1-hop(i) — 2-hop(i,j)
    for i in range(K1):
        base = 1 + K1 + i * K2
        for j in range(K2):
            w = nbr2_mask[:, i, j] * nbr_mask[:, i]
            A[:, 1 + i, base + j] = w
            A[:, base + j, 1 + i] = w
    return X, node_mask, A


def _sym_norm(A):
    """D^{-1/2}(A+I)D^{-1/2}, batched. A [B,N,N]."""
    B, N, _ = A.shape
    I = torch.eye(N, device=A.device).unsqueeze(0)
    Ah = A + I
    deg = Ah.sum(-1).clamp(min=1e-6)                             # [B,N]
    dinv = deg.pow(-0.5)
    return dinv.unsqueeze(-1) * Ah * dinv.unsqueeze(1)


def _rw_norm(A):
    """D^{-1}(A+I), batched (random-walk, for diffusion)."""
    B, N, _ = A.shape
    I = torch.eye(N, device=A.device).unsqueeze(0)
    Ah = A + I
    deg = Ah.sum(-1).clamp(min=1e-6)
    return Ah / deg.unsqueeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. STGCN — Yu et al., IJCAI 2018 (adapted)
# ─────────────────────────────────────────────────────────────────────────────
class TemporalGatedConv(nn.Module):
    def __init__(self, c_in, c_out, kt=3):
        super().__init__()
        self.conv = nn.Conv1d(c_in, 2 * c_out, kt, padding=kt // 2)

    def forward(self, x):                                        # x [B,N,T,C]
        Bn, N, T, C = x.shape
        h = x.reshape(Bn * N, T, C).transpose(1, 2)             # [B*N,C,T]
        h = self.conv(h)                                        # [B*N,2c,T]
        p, q = h.chunk(2, dim=1)
        h = p * torch.sigmoid(q)                                # GLU
        return h.transpose(1, 2).reshape(Bn, N, T, -1)          # [B,N,T,c]


class STGCNBaseline(nn.Module):
    def __init__(self, node_dim=6, edge_dim=5, hidden_dim=128, T=10,
                 num_classes=3, dropout=0.3, **kw):
        super().__init__()
        h = hidden_dim
        self.t1 = TemporalGatedConv(node_dim, h)
        self.gconv1 = nn.Linear(h, h)
        self.t2 = TemporalGatedConv(h, h)
        self.gconv2 = nn.Linear(h, h)
        self.t3 = TemporalGatedConv(h, h)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, num_classes))

    def _spatial(self, x, An, lin):                             # x [B,N,T,h]
        # graph conv per time step: einsum over N
        y = torch.einsum('bij,bjth->bith', An, x)
        return F.relu(lin(y))

    def forward(self, batch):
        X, node_mask, A = build_ego_graph(batch)                # [B,N,T,C]
        An = _sym_norm(A)
        h = self.t1(X)
        h = self._spatial(h, An, self.gconv1)
        h = self.t2(h)
        h = self._spatial(h, An, self.gconv2)
        h = self.t3(h)                                          # [B,N,T,h]
        ego = h[:, 0].mean(dim=1)                               # ego, temporal mean-pool [B,h]
        return self.head(self.drop(ego))


# ─────────────────────────────────────────────────────────────────────────────
# 2. DCRNN — Li et al., ICLR 2018 (adapted, diffusion GRU)
# ─────────────────────────────────────────────────────────────────────────────
class DiffusionConv(nn.Module):
    """K-step diffusion: concat[Z, Â Z, Â² Z] → linear."""
    def __init__(self, c_in, c_out, k=2):
        super().__init__()
        self.k = k
        self.lin = nn.Linear(c_in * (k + 1), c_out)

    def forward(self, An, Z):                                    # Z [B,N,c_in]
        outs = [Z]
        z = Z
        for _ in range(self.k):
            z = torch.bmm(An, z)
            outs.append(z)
        return self.lin(torch.cat(outs, dim=-1))


class DCRNNBaseline(nn.Module):
    def __init__(self, node_dim=6, edge_dim=5, hidden_dim=128, T=10,
                 num_classes=3, dropout=0.3, **kw):
        super().__init__()
        h = hidden_dim
        self.h = h
        self.gate = DiffusionConv(node_dim + h, 2 * h)          # r,u gates
        self.cand = DiffusionConv(node_dim + h, h)              # candidate
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, num_classes))

    def forward(self, batch):
        X, node_mask, A = build_ego_graph(batch)                # [B,N,T,C]
        An = _rw_norm(A)
        B, N, T, C = X.shape
        H = torch.zeros(B, N, self.h, device=X.device)
        for t in range(T):
            x = X[:, :, t]                                      # [B,N,C]
            ru = torch.sigmoid(self.gate(An, torch.cat([x, H], -1)))
            r, u = ru.chunk(2, dim=-1)
            c = torch.tanh(self.cand(An, torch.cat([x, r * H], -1)))
            H = u * H + (1 - u) * c
        ego = H[:, 0]                                           # [B,h]
        return self.head(self.drop(ego))


# ─────────────────────────────────────────────────────────────────────────────
# 3. TGN — Rossi et al., ICML 2020 WS (adapted, edge-temporal memory)
# ─────────────────────────────────────────────────────────────────────────────
class TGNBaseline(nn.Module):
    """Per-node memory updated by timestamped edge-feature messages.
    2-hop→1-hop then 1-hop→ego message passing, sequentially over T frames."""
    def __init__(self, node_dim=6, edge_dim=5, hidden_dim=128, T=10,
                 num_classes=3, dropout=0.3, **kw):
        super().__init__()
        h = hidden_dim
        self.h = h
        # message MLP: [dst_node_feat, edge_feat, src_memory] -> message
        self.msg_lo = nn.Sequential(nn.Linear(node_dim + edge_dim + h, h), nn.ReLU())
        self.msg_hi = nn.Sequential(nn.Linear(node_dim + edge_dim + h, h), nn.ReLU())
        self.gru_nbr = nn.GRUCell(h, h)
        self.gru_ego = nn.GRUCell(h, h)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, num_classes))

    def forward(self, batch):
        node_seq       = batch['node_seq']        # [B,T,6]
        nbr_node_seqs  = batch['nbr_node_seqs']   # [B,K1,T,6]
        edge_seqs      = batch['edge_seqs']       # [B,K1,T,5]
        nbr_mask       = batch['nbr_mask']        # [B,K1]
        nbr2_node_seqs = batch['nbr2_node_seqs']  # [B,K1,K2,T,6]
        nbr2_edge_seqs = batch['nbr2_edge_seqs']  # [B,K1,K2,T,5]
        nbr2_mask      = batch['nbr2_mask']       # [B,K1,K2]
        B, K1, T, _ = nbr_node_seqs.shape
        K2 = nbr2_node_seqs.shape[2]
        dev = node_seq.device

        M_ego = torch.zeros(B, self.h, device=dev)
        M_nbr = torch.zeros(B, K1, self.h, device=dev)

        for t in range(T):
            # --- 2-hop -> 1-hop memory update ---
            if K2 > 0:
                nb2 = nbr2_node_seqs[:, :, :, t]                # [B,K1,K2,6]
                ed2 = nbr2_edge_seqs[:, :, :, t]               # [B,K1,K2,5]
                src = M_nbr.unsqueeze(2).expand(B, K1, K2, self.h)
                m = self.msg_lo(torch.cat([nb2, ed2, src], -1))# [B,K1,K2,h]
                m = m * nbr2_mask.unsqueeze(-1)
                denom = nbr2_mask.sum(-1, keepdim=True).clamp(min=1.0)
                agg = m.sum(2) / denom                         # [B,K1,h]
                newM = self.gru_nbr(agg.reshape(B * K1, self.h),
                                    M_nbr.reshape(B * K1, self.h)).reshape(B, K1, self.h)
                mask = nbr_mask.unsqueeze(-1)
                M_nbr = mask * newM + (1 - mask) * M_nbr
            # --- 1-hop -> ego memory update ---
            nb1 = nbr_node_seqs[:, :, t]                        # [B,K1,6]
            ed1 = edge_seqs[:, :, t]                            # [B,K1,5]
            m = self.msg_hi(torch.cat([nb1, ed1, M_nbr], -1))  # [B,K1,h]
            m = m * nbr_mask.unsqueeze(-1)
            denom = nbr_mask.sum(-1, keepdim=True).clamp(min=1.0)
            agg = m.sum(1) / denom                             # [B,h]
            M_ego = self.gru_ego(agg, M_ego)                   # [B,h]

        return self.head(self.drop(M_ego))


BASELINES = {'stgcn': STGCNBaseline, 'dcrnn': DCRNNBaseline, 'tgn': TGNBaseline}


def build_baseline(name, **kw):
    name = name.lower()
    if name not in BASELINES:
        raise ValueError(f"unknown baseline '{name}', choices: {list(BASELINES)}")
    return BASELINES[name](**kw)
