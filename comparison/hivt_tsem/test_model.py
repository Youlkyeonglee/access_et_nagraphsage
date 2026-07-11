"""
HiVT-adapted 재구현 검증 — 벡터화 구현(scatter_reduce_/scatter_add_ 기반) vs
완전히 독립적인 브루트포스(파이썬 for문, 노드별로 들어오는 edge만 걸러 torch.softmax로
직접 정규화) 대조.

목적: segment_softmax/scatter_sum_nodes(model.py, torch_geometric 없이 직접 짠 부분)가
"타깃 노드별로 들어오는 edge만 묶어서 softmax→가중합"이라는 원래 의도를 정확히 구현했는지
확인. 브루트포스 쪽은 이 두 헬퍼 함수를 전혀 재사용하지 않고 torch.softmax(표준 내장 함수)로
새로 계산 — 같은 버그가 양쪽에 동시에 있을 가능성을 배제하기 위함.

원본(comparison/HiVT/) 코드는 건드리지 않음 — 이 파일은 comparison/hivt_tsem/ 안에서만 동작.

실행: python comparison/hivt_tsem/test_model.py
"""
from __future__ import annotations

import torch

from model import (
    TSEMAAEncoder, TSEMGlobalInteractorLayer, MultipleInputEmbedding,
    rotate2, rotate6, segment_softmax, scatter_sum_nodes,
)

TOL = 1e-6


def check(name: str, a: torch.Tensor, b: torch.Tensor, tol: float = TOL) -> None:
    diff = (a - b).abs().max().item()
    status = 'PASS' if diff < tol else 'FAIL'
    print(f'[{status}] {name}: max|diff|={diff:.3e} (tol={tol:.0e})')
    assert diff < tol, f'{name} 불일치: max|diff|={diff:.3e}'


# ────────────────────────────────────────────────────────────────
# 1) segment_softmax 자체 — 그룹별 softmax가 torch.softmax(수동 그룹핑)과 일치하는가
# ────────────────────────────────────────────────────────────────
def test_segment_softmax():
    torch.manual_seed(0)
    E, H, N = 11, 3, 4
    alpha = torch.randn(E, H, dtype=torch.float64)
    # 각 edge를 노드 0..3 중 하나로 무작위 배정 (노드 3은 일부러 0개 배정 -> 고립 노드 케이스)
    index = torch.tensor([0, 0, 1, 1, 1, 2, 0, 2, 1, 0, 2], dtype=torch.long)
    assert (index == 3).sum() == 0  # 의도적으로 고립 노드 포함

    out = segment_softmax(alpha, index, num_nodes=N)

    # 브루트포스: 노드별로 걸러서 torch.softmax(내장) 적용
    expected = torch.zeros_like(alpha)
    for i in range(N):
        mask = index == i
        if mask.sum() == 0:
            continue  # 들어오는 edge 없음 -> 이 노드에 해당하는 alpha 자체가 없음
        expected[mask] = torch.softmax(alpha[mask], dim=0)

    check('segment_softmax vs torch.softmax(수동 그룹핑)', out, expected)

    # 그룹별 합이 1인지도 확인 (softmax 정의상 당연하지만 별도 확인)
    for i in range(N):
        mask = index == i
        if mask.sum() == 0:
            continue
        s = out[mask].sum(dim=0)
        check(f'  그룹 {i} softmax 합=1', s, torch.ones_like(s))


# ────────────────────────────────────────────────────────────────
# 2) TSEMAAEncoder 한 스텝 — 벡터화 forward vs 브루트포스(노드별 for문)
# ────────────────────────────────────────────────────────────────
def bruteforce_aa_forward(enc: TSEMAAEncoder, x, edge_index, edge_attr, bos_mask, rotate_mat):
    N = x.size(0)
    center_embed = enc.center_embed(rotate6(x, rotate_mat))
    center_embed = torch.where(bos_mask.unsqueeze(-1), enc.bos_token, center_embed)
    center_norm = enc.norm1(center_embed)

    src, tgt = edge_index[0], edge_index[1]
    tgt_rot = rotate_mat[tgt]
    x_j = rotate6(x[src], tgt_rot)
    edge_attr_rot = rotate2(edge_attr, tgt_rot)
    nbr_embed_all = enc.nbr_embed([x_j, edge_attr_rot])
    query_all = enc.lin_q(center_norm[tgt]).view(-1, enc.num_heads, enc.embed_dim // enc.num_heads)
    key_all = enc.lin_k(nbr_embed_all).view(-1, enc.num_heads, enc.embed_dim // enc.num_heads)
    value_all = enc.lin_v(nbr_embed_all).view(-1, enc.num_heads, enc.embed_dim // enc.num_heads)
    scale = (enc.embed_dim // enc.num_heads) ** 0.5
    alpha_all = (query_all * key_all).sum(dim=-1) / scale  # [E,H]

    agg = torch.zeros(N, enc.embed_dim, dtype=x.dtype)
    for i in range(N):
        idx = (tgt == i).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue  # 이 노드로 들어오는 edge 없음 -> agg[i]는 0 그대로 (벡터화 쪽 scatter_add와 동치)
        alpha_i = torch.softmax(alpha_all[idx], dim=0)  # [e_i, H] — 노드별 독립 softmax(내장함수)
        value_i = value_all[idx]  # [e_i, H, d]
        agg[i] = (value_i * alpha_i.unsqueeze(-1)).sum(dim=0).reshape(-1)

    gate = torch.sigmoid(enc.lin_ih(agg) + enc.lin_hh(center_norm))
    mha_out = enc.out_proj(agg + gate * (enc.lin_self(center_norm) - agg))

    out = center_embed + mha_out
    out = out + enc.mlp(enc.norm2(out))
    return out


def test_aa_encoder():
    torch.manual_seed(1)
    N, embed_dim, num_heads = 4, 8, 2  # ego(0) + 이웃 3
    enc = TSEMAAEncoder(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0).double().eval()

    x = torch.randn(N, 6, dtype=torch.float64)
    theta = torch.rand(N, dtype=torch.float64) * 6.28
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    rotate_mat = torch.stack([torch.stack([cos_t, -sin_t], -1), torch.stack([sin_t, cos_t], -1)], -2)
    bos_mask = torch.zeros(N, dtype=torch.bool)

    # 완전연결(자기자신 제외) + 노드 3은 고립(들어오는 edge 없음)으로 일부러 제거
    src, tgt = [], []
    for i in range(N):
        for j in range(N):
            if i != j and j != 3:  # j(=tgt)==3인 edge를 모두 제거 -> 노드3 고립
                src.append(i)
                tgt.append(j)
    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    edge_attr = torch.randn(edge_index.size(1), 2, dtype=torch.float64)

    with torch.no_grad():
        out_vec = enc(x, edge_index, edge_attr, bos_mask, rotate_mat)
        out_bf = bruteforce_aa_forward(enc, x, edge_index, edge_attr, bos_mask, rotate_mat)
    check('TSEMAAEncoder 벡터화 vs 브루트포스 (노드3 고립 포함)', out_vec, out_bf)


# ────────────────────────────────────────────────────────────────
# 3) TSEMGlobalInteractorLayer 한 스텝 — 벡터화 forward vs 브루트포스
# ────────────────────────────────────────────────────────────────
def bruteforce_global_forward(layer: TSEMGlobalInteractorLayer, x, edge_index, edge_attr):
    N = x.size(0)
    x_norm = layer.norm1(x)
    src, tgt = edge_index[0], edge_index[1]
    H, d = layer.num_heads, layer.embed_dim // layer.num_heads
    query_all = layer.lin_q_node(x_norm[tgt]).view(-1, H, d)
    key_node_all = layer.lin_k_node(x_norm[src]).view(-1, H, d)
    key_edge_all = layer.lin_k_edge(edge_attr).view(-1, H, d)
    value_node_all = layer.lin_v_node(x_norm[src]).view(-1, H, d)
    value_edge_all = layer.lin_v_edge(edge_attr).view(-1, H, d)
    scale = d ** 0.5
    alpha_all = (query_all * (key_node_all + key_edge_all)).sum(dim=-1) / scale

    agg = torch.zeros(N, layer.embed_dim, dtype=x.dtype)
    for i in range(N):
        idx = (tgt == i).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        alpha_i = torch.softmax(alpha_all[idx], dim=0)
        val_i = value_node_all[idx] + value_edge_all[idx]
        agg[i] = (val_i * alpha_i.unsqueeze(-1)).sum(dim=0).reshape(-1)

    gate = torch.sigmoid(layer.lin_ih(agg) + layer.lin_hh(x_norm))
    mha_out = layer.out_proj(agg + gate * (layer.lin_self(x_norm) - agg))
    out = x + mha_out
    out = out + layer.mlp(layer.norm2(out))
    return out


def test_global_interactor():
    torch.manual_seed(2)
    N, embed_dim, num_heads = 4, 8, 2
    layer = TSEMGlobalInteractorLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0).double().eval()

    x = torch.randn(N, embed_dim, dtype=torch.float64)
    src, tgt = [], []
    for i in range(N):
        for j in range(N):
            if i != j and j != 2:  # 노드2 고립
                src.append(i)
                tgt.append(j)
    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    edge_attr = torch.randn(edge_index.size(1), embed_dim, dtype=torch.float64)

    with torch.no_grad():
        out_vec = layer(x, edge_index, edge_attr)
        out_bf = bruteforce_global_forward(layer, x, edge_index, edge_attr)
    check('TSEMGlobalInteractorLayer 벡터화 vs 브루트포스 (노드2 고립 포함)', out_vec, out_bf)


# ────────────────────────────────────────────────────────────────
# 4) 배치 offset 로직 — HiVTTSEMAdapted._build_graph의 edge_index 벡터화 offset이
#    "샘플별로 독립적으로 완전연결 그래프를 만든 것"과 같은지 (그래프 누수 방지 확인)
# ────────────────────────────────────────────────────────────────
def test_batch_offset_no_leakage():
    from model import HiVTTSEMAdapted

    torch.manual_seed(3)
    B, W, K = 3, 5, 2
    model = HiVTTSEMAdapted(W=W, K=K, embed_dim=16, num_heads=2,
                            num_temporal_layers=1, num_global_layers=1, dropout=0.0).eval()
    node_seq = torch.randn(B, W, 6)
    nbr_node_seqs = torch.randn(B, K, W, 6)
    batch = {'node_seq': node_seq, 'nbr_node_seqs': nbr_node_seqs}

    with torch.no_grad():
        logits_batched = model(batch)
        # 샘플 1개씩 따로 넣었을 때와 동일해야 함(배치 간 edge 누수가 없어야 함)
        logits_solo = torch.cat([
            model({'node_seq': node_seq[b:b + 1], 'nbr_node_seqs': nbr_node_seqs[b:b + 1]})
            for b in range(B)
        ], dim=0)
    check('배치 처리 vs 샘플별 단독 처리 (그래프 누수 없어야 동일)', logits_batched, logits_solo, tol=1e-5)


if __name__ == '__main__':
    print('=== HiVT-adapted 재구현 검증 (원본 comparison/HiVT/는 미변경, 이 파일 내 대조만) ===\n')
    test_segment_softmax()
    print()
    test_aa_encoder()
    print()
    test_global_interactor()
    print()
    test_batch_offset_no_leakage()
    print('\n모든 검증 통과.')
