"""
SIMPL-adapted 재구현 검증 — 배치 벡터화 구현 vs 완전히 독립적인 브루트포스(파이썬 for문) 대조.

comparison/hivt_tsem, qcnet_tsem, cratpred_tsem/test_model.py와 동일한 방법론이지만, SIMPL은
attention 자체를 재구현하지 않고 PyTorch 내장 nn.MultiheadAttention에 위임하므로(model.py
docstring 참조), 검증 대상은 (1) RPE 수식, (2) SftLayer의 memory 텐서 브로드캐스트 로직이다 —
attention 수식 자체(softmax/스케일링 등)는 내장 모듈이라 별도 검증하지 않는다.

원본(comparison/SIMPL/) 코드는 건드리지 않음.

실행: python comparison/simpl_tsem/test_model.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from model import TSEMSftLayer, build_rpe, build_rpe_batched

TOL = 1e-6


def check(name: str, a: torch.Tensor, b: torch.Tensor, tol: float = TOL) -> None:
    diff = (a - b).abs().max().item()
    status = 'PASS' if diff < tol else 'FAIL'
    print(f'[{status}] {name}: max|diff|={diff:.3e} (tol={tol:.0e})')
    assert diff < tol, f'{name} 불일치: max|diff|={diff:.3e}'


# ────────────────────────────────────────────────────────────────
# 1) build_rpe_batched vs build_rpe(단일 그래프, 배치 없음) — 배치 차원을 추가해도
#    각 샘플별로 독립적으로 계산한 것과 같은 값이 나오는지
# ────────────────────────────────────────────────────────────────
def test_rpe_batched_matches_single():
    torch.manual_seed(0)
    B, N = 4, 6
    ctrs = torch.randn(B, N, 2, dtype=torch.float64) * 10
    vecs = torch.randn(B, N, 2, dtype=torch.float64)
    vecs = vecs / vecs.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # 단위벡터화(현실적 heading)

    rpe_batched = build_rpe_batched(ctrs, vecs, radius=100.0)  # [B,N,N,5]
    for b in range(B):
        rpe_single = build_rpe(ctrs[b], vecs[b], radius=100.0)  # [N,N,5]
        check(f'build_rpe_batched[{b}] vs build_rpe(단일)', rpe_batched[b], rpe_single)


# ────────────────────────────────────────────────────────────────
# 2) RPE 알려진 값 검산 — 두 토큰이 서로 마주보는 단순 배치에서 손계산값과 비교
# ────────────────────────────────────────────────────────────────
def test_rpe_known_values():
    import math
    # 토큰0: 원점, 동쪽(1,0) 바라봄. 토큰1: (0,10)(정북 10m), 남쪽(0,-1) 바라봄(토큰0을 마주봄).
    ctrs = torch.tensor([[0.0, 0.0], [0.0, 10.0]], dtype=torch.float64)
    vecs = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
    rpe = build_rpe(ctrs, vecs, radius=100.0)  # [2,2,5]

    # d_pos[0,1] = |ctrs[1]-ctrs[0]| * 2/100 = 10*2/100 = 0.2
    check('d_pos[0,1] == 0.2', rpe[0, 1, 4:5], torch.tensor([0.2], dtype=torch.float64))
    # cos_a1[0,1] = cos(angle between vecs[1]=(0,-1) and vecs[0]=(1,0)) = 0 (수직)
    check('cos_a1[0,1] == 0 (수직 헤딩)', rpe[0, 1, 0:1], torch.tensor([0.0], dtype=torch.float64))


# ────────────────────────────────────────────────────────────────
# 3) TSEMSftLayer — 배치 벡터화 vs 브루트포스(샘플별·타깃별 for문, 각 타깃마다 개별
#    nn.MultiheadAttention 호출로 memory 시퀀스 재계산)
# ────────────────────────────────────────────────────────────────
def bruteforce_sft_forward(layer: TSEMSftLayer, node, edge, key_padding_mask):
    """node:[B,N,D], edge:[B,N,N,d_edge], key_padding_mask:[B,N] -> node_new:[B,N,D]
    (edge_new는 벡터화 쪽과 별도 비교하므로 여기선 node만 정밀 대조)"""
    B, N, D = node.shape
    node_new = torch.zeros_like(node)
    edge_new_bf = torch.zeros_like(edge) if layer.update_edge else None

    for b in range(B):
        for i in range(N):  # target(=쿼리) 인덱스
            # 원본 정의: memory[p,q] = proj_memory(cat(edge[p,q], node[q]=src, node[p]=tar)).
            # 타깃 q=i 고정, 시퀀스 p를 0..N-1로 순회 — src는 node[i](고정), tar는 node[p](가변).
            src_x = node[b, i].unsqueeze(0).expand(N, -1)  # [N,D] : src_x[p] = node[b,i] (고정)
            tar_x = node[b]  # [N,D] : tar_x[p] = node[b,p] (가변)
            mem_i = layer.proj_memory(torch.cat([edge[b, :, i, :], src_x, tar_x], dim=-1))  # [N,D], seq=p
            if layer.update_edge:
                edge_new_bf[b, :, i, :] = layer.norm_edge(
                    edge[b, :, i, :] + layer.proj_edge(mem_i))

            q = node[b, i].view(1, 1, D)  # [seq=1,batch=1,D]
            kv = mem_i.view(N, 1, D)  # [seq=N,batch=1,D]
            kpm = key_padding_mask[b].view(1, N)
            attn_out, _ = layer.mha(q, kv, kv, key_padding_mask=kpm, need_weights=False)
            attn_out = layer.dropout2(attn_out).view(D)
            x = layer.norm2(node[b, i] + attn_out)
            ff = layer.linear2(layer.dropout(F.relu(layer.linear1(x))))
            node_new[b, i] = layer.norm3(x + layer.dropout3(ff))

    return node_new, edge_new_bf


def test_sft_layer():
    torch.manual_seed(1)
    B, N, D, d_edge, n_head = 2, 5, 16, 8, 2
    layer = TSEMSftLayer(D, d_edge, n_head, dropout=0.0, update_edge=True).double().eval()

    node = torch.randn(B, N, D, dtype=torch.float64)
    edge = torch.randn(B, N, N, d_edge, dtype=torch.float64)
    key_padding_mask = torch.zeros(B, N, dtype=torch.bool)
    key_padding_mask[1, 3] = True  # 샘플1의 토큰3은 무효(마스킹) — 고립 케이스 대응

    with torch.no_grad():
        node_vec, edge_vec = layer(node, edge, key_padding_mask)
        node_bf, edge_bf = bruteforce_sft_forward(layer, node, edge, key_padding_mask)
    check('TSEMSftLayer node 출력 벡터화 vs 브루트포스', node_vec, node_bf)
    check('TSEMSftLayer edge 업데이트 벡터화 vs 브루트포스', edge_vec, edge_bf)


# ────────────────────────────────────────────────────────────────
# 4) 배치 offset 로직 — SIMPLTSEMAdapted 전체가 배치 처리와 샘플별 단독 처리에서 동일한지
# ────────────────────────────────────────────────────────────────
def test_batch_offset_no_leakage():
    from model import SIMPLTSEMAdapted

    torch.manual_seed(2)
    B, W, K = 3, 6, 2
    model = SIMPLTSEMAdapted(W=W, K=K, hidden_size=16, n_fpn_scale=2, d_rpe=8,
                             n_layer=1, n_head=2).eval()
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
    print('=== SIMPL-adapted 재구현 검증 (원본 comparison/SIMPL/는 미변경, 이 파일 내 대조만) ===\n')
    test_rpe_batched_matches_single()
    print()
    test_rpe_known_values()
    print()
    test_sft_layer()
    print()
    test_batch_offset_no_leakage()
    print('\n모든 검증 통과.')
