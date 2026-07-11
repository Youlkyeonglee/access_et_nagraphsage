"""
QCNet-adapted 재구현 검증 — 벡터화 구현(scatter_reduce_/scatter_add_ 기반) vs 완전히 독립적인
브루트포스(파이썬 for문, 노드별로 들어오는 edge만 걸러 torch.softmax로 직접 정규화) 대조.

comparison/hivt_tsem/test_model.py와 동일한 방법론 — segment_softmax/scatter_sum_nodes를
전혀 재사용하지 않는 별도 경로로 AttentionLayer 전체를 다시 계산해서 대조한다.

원본(comparison/QCNet/) 코드는 건드리지 않음.

실행: python comparison/qcnet_tsem/test_model.py
"""
from __future__ import annotations

import torch

from model import (
    AttentionLayer, FourierEmbedding, angle_between_2d_vectors, wrap_angle,
    segment_softmax, scatter_sum_nodes,
)

TOL = 1e-6


def check(name: str, a: torch.Tensor, b: torch.Tensor, tol: float = TOL) -> None:
    diff = (a - b).abs().max().item()
    status = 'PASS' if diff < tol else 'FAIL'
    print(f'[{status}] {name}: max|diff|={diff:.3e} (tol={tol:.0e})')
    assert diff < tol, f'{name} 불일치: max|diff|={diff:.3e}'


# ────────────────────────────────────────────────────────────────
# 1) segment_softmax — comparison/hivt_tsem 검증과 동일 로직, 여기서도 재확인
# ────────────────────────────────────────────────────────────────
def test_segment_softmax():
    torch.manual_seed(0)
    E, H, N = 13, 4, 5
    alpha = torch.randn(E, H, dtype=torch.float64)
    index = torch.tensor([0, 0, 1, 1, 1, 2, 0, 2, 1, 0, 2, 4, 4], dtype=torch.long)
    assert (index == 3).sum() == 0  # 노드 3 고립

    out = segment_softmax(alpha, index, num_nodes=N)
    expected = torch.zeros_like(alpha)
    for i in range(N):
        mask = index == i
        if mask.sum() == 0:
            continue
        expected[mask] = torch.softmax(alpha[mask], dim=0)
    check('segment_softmax vs torch.softmax(수동 그룹핑)', out, expected)


# ────────────────────────────────────────────────────────────────
# 2) angle_between_2d_vectors / wrap_angle — 알려진 값으로 정합성 확인
# ────────────────────────────────────────────────────────────────
def test_geometry_helpers():
    # ctr=(1,0)(동쪽), nbr=(0,1)(북쪽) -> ctr 기준 nbr은 반시계 90도 -> +pi/2
    ctr = torch.tensor([[1.0, 0.0]])
    nbr = torch.tensor([[0.0, 1.0]])
    angle = angle_between_2d_vectors(ctr, nbr)
    check('angle_between_2d_vectors((1,0),(0,1)) == pi/2', angle, torch.tensor([math_pi_half()]))

    # wrap_angle: 3*pi/2 -> -pi/2 로 감겨야 함
    import math
    wrapped = wrap_angle(torch.tensor([3 * math.pi / 2]))
    check('wrap_angle(3pi/2) == -pi/2', wrapped, torch.tensor([-math.pi / 2]))


def math_pi_half():
    import math
    return math.pi / 2


# ────────────────────────────────────────────────────────────────
# 3) AttentionLayer 한 스텝 — 벡터화 forward vs 브루트포스(노드별 for문)
# ────────────────────────────────────────────────────────────────
def bruteforce_attention_forward(layer: AttentionLayer, x, r, edge_index):
    N = x.size(0)
    H, d = layer.num_heads, layer.head_dim
    x_norm = layer.attn_prenorm_x(x)
    r_norm = layer.attn_prenorm_r(r) if (layer.has_pos_emb and r is not None) else r

    src, tgt = edge_index[0], edge_index[1]
    q_all = layer.to_q(x_norm).view(-1, H, d)
    k_all = layer.to_k(x_norm).view(-1, H, d)
    v_all = layer.to_v(x_norm).view(-1, H, d)

    agg = torch.zeros(N, H * d, dtype=x.dtype)
    for i in range(N):
        idx = (tgt == i).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        q_i = q_all[i].unsqueeze(0).expand(idx.numel(), -1, -1)  # [e_i,H,d]
        k_j = k_all[src[idx]]
        v_j = v_all[src[idx]]
        if layer.has_pos_emb:
            k_j = k_j + layer.to_k_r(r_norm[idx]).view(-1, H, d)
            v_j = v_j + layer.to_v_r(r_norm[idx]).view(-1, H, d)
        sim = (q_i * k_j).sum(dim=-1) * layer.scale  # [e_i,H]
        attn = torch.softmax(sim, dim=0)
        agg[i] = (v_j * attn.unsqueeze(-1)).sum(dim=0).reshape(-1)

    g = torch.sigmoid(layer.to_g(torch.cat([agg, x_norm], dim=-1)))
    gated = agg + g * (layer.to_s(x_norm) - agg)
    attn_out = layer.to_out(gated)

    out = x + layer.attn_postnorm(attn_out)
    out = out + layer.ff_postnorm(layer.ff_mlp(layer.ff_prenorm(out)))
    return out


def test_attention_layer():
    torch.manual_seed(1)
    N, hidden_dim, num_heads, head_dim = 5, 16, 2, 8
    layer = AttentionLayer(hidden_dim, num_heads, head_dim, dropout=0.0, has_pos_emb=True).double().eval()

    x = torch.randn(N, hidden_dim, dtype=torch.float64)
    src, tgt = [], []
    for i in range(N):
        for j in range(N):
            if i != j and j != 3:  # 노드3 고립(들어오는 edge 없음)
                src.append(i)
                tgt.append(j)
    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    r = torch.randn(edge_index.size(1), hidden_dim, dtype=torch.float64)

    with torch.no_grad():
        out_vec = layer(x, r, edge_index)
        out_bf = bruteforce_attention_forward(layer, x, r, edge_index)
    check('AttentionLayer 벡터화 vs 브루트포스 (노드3 고립 포함)', out_vec, out_bf)


# ────────────────────────────────────────────────────────────────
# 4) FourierEmbedding — 그냥 nn.Module이지만, 같은 입력에 결정적으로 같은 출력을 내는지
#    (dropout 없음 확인 + 두 번 호출 결과 동일한지)
# ────────────────────────────────────────────────────────────────
def test_fourier_embedding_deterministic():
    torch.manual_seed(2)
    emb = FourierEmbedding(input_dim=4, hidden_dim=16, num_freq_bands=8).double().eval()
    x = torch.randn(10, 4, dtype=torch.float64)
    with torch.no_grad():
        out1 = emb(x)
        out2 = emb(x)
    check('FourierEmbedding 결정적 재현성', out1, out2, tol=1e-12)


# ────────────────────────────────────────────────────────────────
# 5) 배치 offset 로직 — QCNetTSEMAdapted._build_flat의 edge_index 벡터화 offset이
#    "샘플별로 독립적으로 그래프를 만든 것"과 같은지 (그래프 누수 방지 확인)
# ────────────────────────────────────────────────────────────────
def test_batch_offset_no_leakage():
    from model import QCNetTSEMAdapted

    torch.manual_seed(3)
    B, W, K = 3, 6, 2
    model = QCNetTSEMAdapted(W=W, K=K, hidden_dim=16, num_heads=2, head_dim=8,
                             num_layers=1, num_freq_bands=8).eval()
    node_seq = torch.randn(B, W, 6)
    nbr_node_seqs = torch.randn(B, K, W, 6)
    batch = {'node_seq': node_seq, 'nbr_node_seqs': nbr_node_seqs}

    with torch.no_grad():
        logits_batched = model(batch)
        logits_solo = torch.cat([
            model({'node_seq': node_seq[b:b + 1], 'nbr_node_seqs': nbr_node_seqs[b:b + 1]})
            for b in range(B)
        ], dim=0)
    check('배치 처리 vs 샘플별 단독 처리 (그래프 누수 없어야 동일)', logits_batched, logits_solo, tol=1e-5)


if __name__ == '__main__':
    print('=== QCNet-adapted 재구현 검증 (원본 comparison/QCNet/는 미변경, 이 파일 내 대조만) ===\n')
    test_segment_softmax()
    print()
    test_geometry_helpers()
    print()
    test_attention_layer()
    print()
    test_fourier_embedding_deterministic()
    print()
    test_batch_offset_no_leakage()
    print('\n모든 검증 통과.')
