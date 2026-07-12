"""
DCRNN-adapted 재구현 검증 — 벡터화 구현(bmm 기반) vs 완전히 독립적인 브루트포스(파이썬 for문/직접
손계산) 대조. comparison/cratpred_tsem, hivt_tsem, qcnet_tsem/test_model.py와 동일한 방법론.

원본(comparison/DCRNN/)은 건드리지 않음.

실행: python comparison/dcrnn_tsem/test_model.py
"""
from __future__ import annotations

import torch

from model import DCRNNTSEMAdapted, TSEMDCGRUCell, build_diffusion_supports

TOL = 1e-6


def check(name: str, a: torch.Tensor, b: torch.Tensor, tol: float = TOL) -> None:
    diff = (a - b).abs().max().item()
    status = 'PASS' if diff < tol else 'FAIL'
    print(f'[{status}] {name}: max|diff|={diff:.3e} (tol={tol:.0e})')
    assert diff < tol, f'{name} 불일치: max|diff|={diff:.3e}'


# ────────────────────────────────────────────────────────────────
# 1) build_diffusion_supports — 작은 합성 그래프로 P_f=D_O^-1 A, P_b=D_I^-1 A^T를 손계산과 대조
# ────────────────────────────────────────────────────────────────
def test_diffusion_supports_hand_computed():
    torch.manual_seed(0)
    # 4개 노드, 좌표를 직접 지정해 pairwise distance를 손으로 검산 가능하게 함
    # node0=(0,0), node1=(3,0), node2=(0,4), node3=무효(마스킹)
    pos = torch.tensor([[[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [10.0, 10.0]]])  # [1,4,2]
    valid = torch.tensor([[True, True, True, False]])

    P_f, P_b = build_diffusion_supports(pos, valid)

    # 손계산: dist(0,1)=3, dist(0,2)=4, dist(1,2)=5 (노드3은 무효라 전부 0)
    d01, d02, d12 = 3.0, 4.0, 5.0
    valid_dists = torch.tensor([d01, d02, d12, d01, d02, d12])  # off-diag 6쌍(3노드, 대칭)
    sigma = valid_dists.std().clamp(min=1.0).item()

    def k(d):
        return torch.exp(torch.tensor(-(d ** 2) / (sigma ** 2))).item()

    A = torch.zeros(4, 4)
    A[0, 1] = A[1, 0] = k(d01)
    A[0, 2] = A[2, 0] = k(d02)
    A[1, 2] = A[2, 1] = k(d12)
    # 노드3(무효)은 전부 0으로 유지

    d_out = A.sum(dim=1, keepdim=True).clamp(min=1e-6)
    P_f_expected = (A / d_out).unsqueeze(0)
    A_t = A.t()
    d_in = A_t.sum(dim=1, keepdim=True).clamp(min=1e-6)
    P_b_expected = (A_t / d_in).unsqueeze(0)

    check('P_f 손계산 대조 (D_O^-1 A)', P_f, P_f_expected, tol=1e-5)
    check('P_b 손계산 대조 (D_I^-1 A^T)', P_b, P_b_expected, tol=1e-5)
    # 무효 노드(3) 관련 행/열은 전부 0이어야 함
    check('무효 노드 행 전부 0 (P_f)', P_f[0, 3], torch.zeros(4))
    check('무효 노드 열 전부 0 (P_f)', P_f[0, :, 3], torch.zeros(4))


# ────────────────────────────────────────────────────────────────
# 2) _diffuse — 벡터화 K-hop bmm 반복 vs 브루트포스 for-loop P^k 반복행렬곱
# ────────────────────────────────────────────────────────────────
def bruteforce_diffuse(z: torch.Tensor, P_f: torch.Tensor, P_b: torch.Tensor, K: int) -> torch.Tensor:
    """comparison/dcrnn_tsem/model.py::TSEMDCGRUCell._diffuse 와 동일 결과를, bmm이 아닌
    노드별 for문으로 P^k x = sum_j P[i,j] x[j]를 직접 계산해 K번 반복 적용."""
    B, N, D = z.shape

    def matvec(P, x):
        out = torch.zeros_like(x)
        for b in range(B):
            for i in range(N):
                acc = torch.zeros(D, dtype=x.dtype)
                for j in range(N):
                    acc = acc + P[b, i, j] * x[b, j]
                out[b, i] = acc
        return out

    feats = [z]
    if K > 0:
        x = z
        for _ in range(K):
            x = matvec(P_f, x)
            feats.append(x)
        x = z
        for _ in range(K):
            x = matvec(P_b, x)
            feats.append(x)
    return torch.cat(feats, dim=-1)


def test_diffuse_vs_bruteforce():
    torch.manual_seed(1)
    B, N, D, K = 2, 4, 3, 3
    z = torch.randn(B, N, D, dtype=torch.float64)
    # 임의의 (행 정규화된 확률적) P_f, P_b — 실제 build_diffusion_supports 출력 형태와 동일 성질
    raw_f = torch.rand(B, N, N, dtype=torch.float64)
    raw_f[:, torch.arange(N), torch.arange(N)] = 0.0  # 자기 자신 제외(대각 0, A와 동일 관례)
    P_f = raw_f / raw_f.sum(dim=2, keepdim=True).clamp(min=1e-9)
    raw_b = torch.rand(B, N, N, dtype=torch.float64)
    raw_b[:, torch.arange(N), torch.arange(N)] = 0.0
    P_b = raw_b / raw_b.sum(dim=2, keepdim=True).clamp(min=1e-9)

    cell = TSEMDCGRUCell(input_dim=D, hidden_dim=5, K=K).double()
    out_vec = cell._diffuse(z, P_f, P_b)
    out_bf = bruteforce_diffuse(z, P_f, P_b, K)
    check('_diffuse 벡터화(bmm) vs 브루트포스(for-loop P^k)', out_vec, out_bf, tol=1e-9)


# ────────────────────────────────────────────────────────────────
# 3) DCGRU 게이트 수식 sanity check — K=0(diffusion 없음, gconv가 순수 fc로 퇴화)일 때
#    표준 GRU 수식(r,u,c,new_h)과 셀 자신의 선형 가중치를 이용한 수동 계산이 정확히 일치하는지.
#    (diffuse가 K=0이면 concat 없이 z 그대로이므로, DCGRUCell.forward는 원본
#     dcrnn_cell.py::DCGRUCell._fc 경로(use_gc_for_ru=False)와 동치가 되어야 함 — 표준 fc-GRU 수식)
# ────────────────────────────────────────────────────────────────
def test_dcgru_reduces_to_standard_gru_when_k0():
    torch.manual_seed(2)
    B, N, input_dim, hidden_dim = 2, 3, 4, 6
    cell = TSEMDCGRUCell(input_dim, hidden_dim, K=0).double()
    x = torch.randn(B, N, input_dim, dtype=torch.float64)
    h = torch.randn(B, N, hidden_dim, dtype=torch.float64)
    # K=0이면 P_f/P_b는 아예 사용되지 않아야 함 -> 아무 값이나 넣어도 결과가 같아야 함(사용되지 않는지도 확인)
    P_f_dummy = torch.randn(B, N, N, dtype=torch.float64)
    P_b_dummy = torch.randn(B, N, N, dtype=torch.float64)

    with torch.no_grad():
        out_cell = cell(x, h, P_f_dummy, P_b_dummy)

        # 표준 fc-GRU 수식을 셀 자신의 gate_lin/cand_lin 가중치로 수동 재현
        xh = torch.cat([x, h], dim=-1)
        ru = torch.sigmoid(cell.gate_lin(xh))
        r, u = ru.chunk(2, dim=-1)
        xh_r = torch.cat([x, r * h], dim=-1)
        c = torch.tanh(cell.cand_lin(xh_r))
        expected = u * h + (1.0 - u) * c

    check('K=0일 때 DCGRUCell == 표준 fc-GRU 수식(r,u,c,new_h) 수동 재현', out_cell, expected, tol=1e-9)

    # P_f/P_b를 다른 무작위 값으로 바꿔도 결과가 그대로인지(=diffusion 항이 실제로 안 쓰이는지) 확인
    with torch.no_grad():
        out_cell2 = cell(x, h, torch.randn_like(P_f_dummy), torch.randn_like(P_b_dummy))
    check('K=0일 때 P_f/P_b를 바꿔도 출력 불변(diffusion 항 미사용 확인)', out_cell, out_cell2, tol=1e-12)


# ────────────────────────────────────────────────────────────────
# 4) 배치 처리 vs 샘플별 단독 처리 (그래프 누수 검사) — DCRNNTSEMAdapted._build_graph의
#    scene 정렬 + 인접행렬 구성이 배치 내 다른 샘플과 섞이지 않는지
# ────────────────────────────────────────────────────────────────
def test_batch_vs_solo_no_leakage():
    torch.manual_seed(3)
    B, W, K_nbr = 3, 6, 2
    model = DCRNNTSEMAdapted(W=W, K_nbr=K_nbr, hidden_dim=8, K_diffusion=2, num_layers=2).eval()
    node_seq = torch.randn(B, W, 6)
    nbr_node_seqs = torch.randn(B, K_nbr, W, 6)
    nbr_mask = torch.ones(B, K_nbr)
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
    print('=== DCRNN-adapted 재구현 검증 (원본 comparison/DCRNN/는 미변경, 이 파일 내 대조만) ===\n')
    test_diffusion_supports_hand_computed()
    print()
    test_diffuse_vs_bruteforce()
    print()
    test_dcgru_reduces_to_standard_gru_when_k0()
    print()
    test_batch_vs_solo_no_leakage()
    print('\n모든 검증 통과.')
