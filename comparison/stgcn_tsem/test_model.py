"""
STGCN-adapted 재구현 검증 — 벡터화 구현 vs 완전히 독립적인 브루트포스(파이썬 for문) 대조.

comparison/cratpred_tsem, hivt_tsem/test_model.py와 동일한 방법론. STGCN은 CGConv/attention과
달리 "1st-order GCN 근사(gso 곱)"와 "GLU 게이트 causal conv" 두 축이 핵심이라, 이 두 축을 각각
독립적으로 재검증한다: (1) gso(D^-1/2(A+I)D^-1/2) 정규화가 손으로 계산한 값과 일치하는가,
(2) 배치 gso 곱(einsum)이 노드별 for-loop 합산과 일치하는가, (3) GLU causal conv가 미래 프레임을
보지 않는가, (4) 배치 처리와 샘플별 단독 처리가 그래프 누수 없이 동일한가.

원본(comparison/STGCN/) 코드는 건드리지 않음.

실행: python comparison/stgcn_tsem/test_model.py
"""
from __future__ import annotations

import torch

from model import STGCNTSEMAdapted, TSEMGraphConv, TSEMTemporalConvLayer, _build_gso

TOL = 1e-6


def check(name: str, a: torch.Tensor, b: torch.Tensor, tol: float = TOL) -> None:
    diff = (a - b).abs().max().item()
    status = 'PASS' if diff < tol else 'FAIL'
    print(f'[{status}] {name}: max|diff|={diff:.3e} (tol={tol:.0e})')
    assert diff < tol, f'{name} 불일치: max|diff|={diff:.3e}'


# ────────────────────────────────────────────────────────────────
# 1) _build_gso — 손으로 계산한 작은 그래프의 sym_renorm_adj(D^-1/2(A+I)D^-1/2)와 일치하는가
# ────────────────────────────────────────────────────────────────
def test_build_gso_hand_computed():
    # 3개 노드, 전부 valid, 좌표를 손으로 잡아 거리를 정확히 알 수 있게 구성
    # node0=(0,0), node1=(3,0)(거리3), node2=(0,4)(거리4, node1과는 5)
    centers = torch.tensor([[[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]], dtype=torch.float64)  # [1,3,2]
    valid = torch.ones(1, 3, dtype=torch.bool)
    sigma = 5.0

    gso = _build_gso(centers, valid, sigma)[0]  # [3,3]

    import math
    d01, d02, d12 = 3.0, 4.0, 5.0
    A = torch.zeros(3, 3, dtype=torch.float64)
    A[0, 1] = A[1, 0] = math.exp(-(d01 ** 2) / sigma ** 2)
    A[0, 2] = A[2, 0] = math.exp(-(d02 ** 2) / sigma ** 2)
    A[1, 2] = A[2, 1] = math.exp(-(d12 ** 2) / sigma ** 2)
    A_hat = A + torch.eye(3, dtype=torch.float64)
    deg = A_hat.sum(-1)
    deg_inv_sqrt = deg.pow(-0.5)
    expected = deg_inv_sqrt.unsqueeze(-1) * A_hat * deg_inv_sqrt.unsqueeze(0)

    check('_build_gso vs 손 계산 sym_renorm_adj(3노드)', gso, expected)


def test_build_gso_invalid_node_isolated():
    """무효 노드(예: nbr_mask=0)는 어떤 edge weight도 갖지 않고(자기 자신 identity만 남아)
    다른 노드의 gso 값에도 영향을 주지 않아야 한다."""
    torch.manual_seed(0)
    centers = torch.randn(1, 4, 2, dtype=torch.float64) * 10
    valid = torch.tensor([[True, True, False, True]])
    sigma = 8.0

    gso_full = _build_gso(centers, valid, sigma)[0]

    # 노드2를 제거하고 3노드로 다시 계산한 gso가, 4노드 계산 결과에서 노드0,1,3 부분행렬과 같아야 함
    keep = [0, 1, 3]
    centers_sub = centers[:, keep]
    valid_sub = torch.ones(1, 3, dtype=torch.bool)
    gso_sub = _build_gso(centers_sub, valid_sub, sigma)[0]

    sub_from_full = gso_full[keep][:, keep]
    check('무효 노드 제외 시 나머지 노드 gso 불변(고립 검증)', sub_from_full, gso_sub, tol=1e-9)
    # 무효 노드(2) 행/열은 identity(자기 자신 1, 나머지 0)만 남아야 함
    check('무효 노드 자기 자신 gso=1', gso_full[2, 2], torch.tensor(1.0, dtype=torch.float64))
    check('무효 노드 나머지 gso=0', gso_full[2, keep], torch.zeros(3, dtype=torch.float64))


# ────────────────────────────────────────────────────────────────
# 2) TSEMGraphConv 배치 einsum vs 노드별 for-loop 합산
# ────────────────────────────────────────────────────────────────
def bruteforce_graph_conv(conv: TSEMGraphConv, x: torch.Tensor, gso: torch.Tensor) -> torch.Tensor:
    """x:[B,C,T,V], gso:[B,V,V] -> [B,T,V,C_out], 노드(h)별로 for문 돌며 gso[b,h,i]*x[b,t,i,:] 합산."""
    B, C, T, V = x.shape
    x_p = x.permute(0, 2, 3, 1)  # [B,T,V,C]
    C_out = conv.weight.size(1)
    out = torch.zeros(B, T, V, C_out, dtype=x.dtype)
    for b in range(B):
        for t in range(T):
            for h in range(V):
                acc = torch.zeros(C, dtype=x.dtype)
                for i in range(V):
                    acc = acc + gso[b, h, i] * x_p[b, t, i]
                out[b, t, h] = acc @ conv.weight
    if conv.bias is not None:
        out = out + conv.bias
    return out


def test_graph_conv_vs_bruteforce():
    torch.manual_seed(1)
    B, C, T, V, C_out = 2, 4, 3, 5, 6
    conv = TSEMGraphConv(C, C_out).double()
    x = torch.randn(B, C, T, V, dtype=torch.float64)
    centers = torch.randn(B, V, 2, dtype=torch.float64) * 10
    valid = torch.ones(B, V, dtype=torch.bool)
    valid[0, -1] = False  # 한 배치는 노드 하나 무효 처리(마스킹 경로도 검증)
    gso = _build_gso(centers, valid, sigma=8.0)

    with torch.no_grad():
        out_vec = conv(x, gso)
        out_bf = bruteforce_graph_conv(conv, x, gso)
    check('TSEMGraphConv einsum vs 노드별 for-loop', out_vec, out_bf)


# ────────────────────────────────────────────────────────────────
# 3) GLU causal conv — 출력 시점 t는 입력 시점 <=t만 봐야 한다(미래 프레임 미참조)
# ────────────────────────────────────────────────────────────────
def test_temporal_conv_causal():
    torch.manual_seed(2)
    Kt, c_in, c_out, T, V, B = 3, 4, 5, 10, 3, 2
    layer = TSEMTemporalConvLayer(Kt, c_in, c_out).double()
    layer.eval()

    x = torch.randn(B, c_in, T, V, dtype=torch.float64)
    with torch.no_grad():
        out_orig = layer(x)

    # 미래 프레임(t>=5)만 임의로 바꿔치기 — 앞쪽 출력(t<5)은 절대 바뀌면 안 됨
    x_perturbed = x.clone()
    x_perturbed[:, :, 5:, :] = torch.randn(B, c_in, T - 5, V, dtype=torch.float64) * 100
    with torch.no_grad():
        out_perturbed = layer(x_perturbed)

    check('causal conv: t<5 출력은 t>=5 프레임 변경에 불변', out_orig[:, :, :5, :], out_perturbed[:, :, :5, :])

    # 반대로 과거 프레임(t=0)을 바꾸면 t>=0 출력 전체(특히 t=0 이후)가 바뀔 수 있어야 함
    # (인과성 위반이 아니라, 단지 "정상적으로 과거 정보가 전파는 된다"는 걸 확인 — 항등연산이 아님을 보장)
    x_perturbed2 = x.clone()
    x_perturbed2[:, :, 0, :] = torch.randn(B, c_in, V, dtype=torch.float64) * 100
    with torch.no_grad():
        out_perturbed2 = layer(x_perturbed2)
    diff_at_0 = (out_orig[:, :, 0, :] - out_perturbed2[:, :, 0, :]).abs().max().item()
    print(f'[INFO] t=0 프레임 변경 시 t=0 출력 변화량={diff_at_0:.3e} (0이면 이상 — 과거 정보가 전파 안 됨)')
    assert diff_at_0 > 1e-6, 'causal conv가 과거 프레임 변화에도 전혀 반응하지 않음 — 게이트 상수화 등 이상'
    print('[PASS] causal conv: t=0 변경이 t=0 출력에 정상적으로 반영됨(항등연산 아님)')


# ────────────────────────────────────────────────────────────────
# 4) 배치 처리 vs 샘플별 단독 처리(그래프 누수 없어야 동일) — STGCNTSEMAdapted 전체
# ────────────────────────────────────────────────────────────────
def test_batch_offset_no_leakage():
    torch.manual_seed(3)
    B, W, K = 3, 10, 4
    model = STGCNTSEMAdapted(W=W, K=K, Kt=3, block_channels=[[8, 8, 8], [8, 8, 8]],
                              droprate=0.0, adj_sigma=10.0).eval()
    node_seq = torch.randn(B, W, 6)
    nbr_node_seqs = torch.randn(B, K, W, 6)
    nbr_mask = torch.tensor([[1, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
    batch = {'node_seq': node_seq, 'nbr_node_seqs': nbr_node_seqs, 'nbr_mask': nbr_mask}

    with torch.no_grad():
        logits_batched = model(batch)
        logits_solo = torch.cat([
            model({'node_seq': node_seq[b:b + 1], 'nbr_node_seqs': nbr_node_seqs[b:b + 1],
                  'nbr_mask': nbr_mask[b:b + 1]})
            for b in range(B)
        ], dim=0)
    check('배치 처리 vs 샘플별 단독 처리 (그래프 누수 없어야 동일)', logits_batched, logits_solo, tol=1e-5)


# ────────────────────────────────────────────────────────────────
# 5) 무효 이웃(nbr_mask=0) 노드가 gso를 통해 다른 노드의 forward 결과에 영향을 주지 않는가
#    (STGCNTSEMAdapted 전체 관점에서 재확인 — #1 test_build_gso_invalid_node_isolated의 상위 검증)
# ────────────────────────────────────────────────────────────────
def test_invalid_neighbor_no_influence_on_output():
    torch.manual_seed(4)
    W, K = 10, 4
    model = STGCNTSEMAdapted(W=W, K=K, Kt=3, block_channels=[[8, 8, 8], [8, 8, 8]],
                              droprate=0.0, adj_sigma=10.0).eval()
    node_seq = torch.randn(1, W, 6)
    nbr_node_seqs = torch.randn(1, K, W, 6)
    nbr_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)  # slot1,3 무효

    with torch.no_grad():
        out1 = model({'node_seq': node_seq, 'nbr_node_seqs': nbr_node_seqs, 'nbr_mask': nbr_mask})
        nbr_node_seqs2 = nbr_node_seqs.clone()
        nbr_node_seqs2[:, [1, 3]] = torch.randn(1, 2, W, 6) * 100  # 무효 슬롯 값을 마구 바꿔도
        out2 = model({'node_seq': node_seq, 'nbr_node_seqs': nbr_node_seqs2, 'nbr_mask': nbr_mask})
    check('무효 이웃 슬롯 값 변경이 logits에 영향 없음', out1, out2, tol=1e-5)


if __name__ == '__main__':
    print('=== STGCN-adapted 재구현 검증 (원본 comparison/STGCN/는 미변경, 이 파일 내 대조만) ===\n')
    test_build_gso_hand_computed()
    print()
    test_build_gso_invalid_node_isolated()
    print()
    test_graph_conv_vs_bruteforce()
    print()
    test_temporal_conv_causal()
    print()
    test_batch_offset_no_leakage()
    print()
    test_invalid_neighbor_no_influence_on_output()
    print('\n모든 검증 통과.')
