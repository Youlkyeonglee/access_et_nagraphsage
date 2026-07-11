"""
CRAT-Pred-adapted 재구현 검증 — 벡터화 구현(scatter_add_ 기반) vs 완전히 독립적인 브루트포스
(파이썬 for문으로 노드별 들어오는 edge만 걸러 직접 합산) 대조.

comparison/hivt_tsem, qcnet_tsem/test_model.py와 동일한 방법론. CRAT-Pred는 softmax가 없는
gated-sum 집계(CGConv)라 HiVT/QCNet의 attention보다 검증할 표면적은 작지만, scatter_add_ 인덱싱
실수(예: src/tgt 뒤바뀜)는 여전히 조용히 틀린 결과를 낼 수 있어 동일한 방식으로 대조한다.

원본(comparison/crat-pred/) 코드는 건드리지 않음.

실행: python comparison/cratpred_tsem/test_model.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from model import TSEMCGConv, scatter_sum_nodes

TOL = 1e-6


def check(name: str, a: torch.Tensor, b: torch.Tensor, tol: float = TOL) -> None:
    diff = (a - b).abs().max().item()
    status = 'PASS' if diff < tol else 'FAIL'
    print(f'[{status}] {name}: max|diff|={diff:.3e} (tol={tol:.0e})')
    assert diff < tol, f'{name} 불일치: max|diff|={diff:.3e}'


# ────────────────────────────────────────────────────────────────
# 1) scatter_sum_nodes 자체 — 노드별 for-loop 합산과 일치하는가
# ────────────────────────────────────────────────────────────────
def test_scatter_sum_nodes():
    torch.manual_seed(0)
    E, D, N = 11, 5, 4
    messages = torch.randn(E, D, dtype=torch.float64)
    index = torch.tensor([0, 0, 1, 1, 1, 2, 0, 2, 1, 0, 2], dtype=torch.long)
    assert (index == 3).sum() == 0  # 노드3 고립

    out = scatter_sum_nodes(messages, index, num_nodes=N)
    expected = torch.zeros(N, D, dtype=torch.float64)
    for i in range(N):
        mask = index == i
        if mask.sum() == 0:
            continue
        expected[i] = messages[mask].sum(dim=0)
    check('scatter_sum_nodes vs 노드별 for-loop 합산', out, expected)


# ────────────────────────────────────────────────────────────────
# 2) TSEMCGConv 한 스텝 — 벡터화 forward vs 브루트포스(노드별 for문, message 재계산)
# ────────────────────────────────────────────────────────────────
def bruteforce_cgconv_forward(conv: TSEMCGConv, x, edge_index, edge_attr):
    N = x.size(0)
    src, tgt = edge_index[0], edge_index[1]
    D = x.size(-1)

    agg = torch.zeros(N, D, dtype=x.dtype)
    for i in range(N):
        idx = (tgt == i).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue  # 벡터화 쪽 scatter_add도 이 노드엔 0을 유지 -> 동치
        for e in idx.tolist():
            j = int(src[e])
            z = torch.cat([x[i], x[j], edge_attr[e]], dim=-1)
            msg = torch.sigmoid(conv.lin_f(z)) * F.softplus(conv.lin_s(z))
            agg[i] = agg[i] + msg

    agg = conv.bn(agg)
    return agg + x


def test_cgconv():
    torch.manual_seed(1)
    N, channels, edge_dim = 5, 8, 2
    conv = TSEMCGConv(channels, edge_dim).double()
    conv.eval()  # BatchNorm1d를 running stats 모드로 고정 — 배치 통계 의존성 없이 대조하기 위함

    # BatchNorm의 running stats를 임의 값으로 채워 "학습된 상태"를 흉내(초기값 0/1 그대로면
    # bn이 사실상 identity라 검증 의미가 약해짐)
    with torch.no_grad():
        conv.bn.running_mean.copy_(torch.randn(channels, dtype=torch.float64) * 0.1)
        conv.bn.running_var.copy_(torch.rand(channels, dtype=torch.float64) + 0.5)

    x = torch.randn(N, channels, dtype=torch.float64)
    src, tgt = [], []
    for i in range(N):
        for j in range(N):
            if i != j and j != 2:  # 노드2 고립(들어오는 edge 없음)
                src.append(i)
                tgt.append(j)
    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    edge_attr = torch.randn(edge_index.size(1), edge_dim, dtype=torch.float64)

    with torch.no_grad():
        out_vec = conv(x, edge_index, edge_attr)
        out_bf = bruteforce_cgconv_forward(conv, x, edge_index, edge_attr)
    check('TSEMCGConv 벡터화 vs 브루트포스 (노드2 고립 포함, eval 모드)', out_vec, out_bf)


# ────────────────────────────────────────────────────────────────
# 3) 배치 offset 로직 — CratPredTSEMAdapted._build_graph의 edge_index 벡터화 offset이
#    "샘플별로 독립적으로 그래프를 만든 것"과 같은지 (그래프 누수 방지 확인)
# ────────────────────────────────────────────────────────────────
def test_batch_offset_no_leakage():
    from model import CratPredTSEMAdapted

    torch.manual_seed(2)
    B, W, K = 3, 6, 2
    model = CratPredTSEMAdapted(W=W, K=K, latent_size=16, num_heads=2).eval()
    node_seq = torch.randn(B, W, 6)
    nbr_node_seqs = torch.randn(B, K, W, 6)
    nbr_mask = torch.ones(B, K)
    batch = {'node_seq': node_seq, 'nbr_node_seqs': nbr_node_seqs, 'nbr_mask': nbr_mask}

    with torch.no_grad():
        logits_batched = model(batch)
        logits_solo = torch.cat([
            model({'node_seq': node_seq[b:b + 1], 'nbr_node_seqs': nbr_node_seqs[b:b + 1],
                  'nbr_mask': nbr_mask[b:b + 1]})
            for b in range(B)
        ], dim=0)
    check('배치 처리 vs 샘플별 단독 처리 (그래프 누수 없어야 동일)', logits_batched, logits_solo, tol=1e-5)


if __name__ == '__main__':
    print('=== CRAT-Pred-adapted 재구현 검증 (원본 comparison/crat-pred/는 미변경, 이 파일 내 대조만) ===\n')
    test_scatter_sum_nodes()
    print()
    test_cgconv()
    print()
    test_batch_offset_no_leakage()
    print('\n모든 검증 통과.')
