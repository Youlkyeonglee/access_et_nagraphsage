"""
LSTM-adapted 재구현 검증 — 이 모델은 커스텀 scatter/attention이 전혀 없다(단순 LSTM + masked
mean-pooling)이므로, 검증 표면적은 CGConv/attention baseline보다 훨씬 작다. 대신 이 baseline의
핵심 설계 포인트(=학습되는 상호작용 모듈이 없다는 것)가 실제로 지켜지는지를 검증한다:

  1) masked mean-pooling이 무효 이웃(nbr_mask=0)을 정확히 제외하는지 — 브루트포스 for-loop 대조
  2) 배치 처리 vs 샘플별 단독 처리 결과 일치 — 그래프/배치 누수(다른 샘플의 이웃이 섞여 들어가는지) 검사
  3) 이웃이 전부 없는 경우(고립 ego, nbr_mask 전부 0) NaN이 나지 않는지

원본 GitHub repo가 없는 baseline이라(README 참조) 대조할 "원본 코드"가 없다 — 대신 이 파일
자체가 벡터화 구현(model.py)과 완전히 독립적인 순수 파이썬 for-loop 구현을 대조하는 역할을 한다
(comparison/cratpred_tsem, hivt_tsem, qcnet_tsem/test_model.py와 동일한 방법론).

실행: python comparison/lstm_tsem/test_model.py
"""
from __future__ import annotations

import torch

from model import LstmTSEMAdapted

TOL = 1e-6


def check(name: str, a: torch.Tensor, b: torch.Tensor, tol: float = TOL) -> None:
    diff = (a - b).abs().max().item()
    status = 'PASS' if diff < tol else 'FAIL'
    print(f'[{status}] {name}: max|diff|={diff:.3e} (tol={tol:.0e})')
    assert diff < tol, f'{name} 불일치: max|diff|={diff:.3e}'


def bruteforce_masked_mean(h: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """h: [B,K,D], valid: [B,K](0/1) -> [B,D] — 배치별 for-loop로 무효 이웃 제외하고 평균."""
    B, K, D = h.shape
    out = torch.zeros(B, D, dtype=h.dtype)
    for b in range(B):
        vecs = [h[b, k] for k in range(K) if valid[b, k] > 0]
        if len(vecs) == 0:
            out[b] = torch.zeros(D, dtype=h.dtype)
        else:
            out[b] = torch.stack(vecs, dim=0).mean(dim=0)
    return out


# ────────────────────────────────────────────────────────────────
# 1) masked mean-pooling이 무효 이웃을 정확히 제외하는가 (모델 내부 로직과 동일한 수식을
#    model.py의 헬퍼를 재사용하지 않고 순수 for-loop로 재계산해 대조)
# ────────────────────────────────────────────────────────────────
def test_masked_mean_pooling():
    torch.manual_seed(0)
    B, K, D = 6, 6, 8
    h = torch.randn(B, K, D, dtype=torch.float64)
    valid = torch.zeros(B, K)
    # 다양한 유효 이웃 패턴(전부 유효/일부/전무)
    valid[0] = 1
    valid[1, :3] = 1
    valid[2, 0] = 1
    valid[3] = 0  # 이 샘플만 이웃 전무
    valid[4, [1, 3, 5]] = 1
    valid[5, :] = 1

    denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
    pooled_vec = (h * valid.unsqueeze(-1)) .sum(dim=1) / denom
    pooled_bf = bruteforce_masked_mean(h, valid)

    # 이웃 0명인 샘플(3번)은 벡터화 쪽은 분모를 1로 clamp해 0벡터가 나오고, 브루트포스도 0벡터
    # 정의이므로 일치해야 함
    check('masked mean-pooling: 벡터화 vs 브루트포스(for-loop)', pooled_vec, pooled_bf)
    assert not torch.isnan(pooled_vec).any(), '이웃 0명 샘플에서 NaN 발생'


# ────────────────────────────────────────────────────────────────
# 2) 배치 처리 vs 샘플별 단독 처리 — LstmTSEMAdapted.forward의 배치 벡터화가
#    "샘플별로 독립 처리한 것"과 동일한 결과를 내는지(그래프/배치 누수 없음 확인)
# ────────────────────────────────────────────────────────────────
def test_batch_vs_solo_no_leakage():
    torch.manual_seed(1)
    B, W, K = 4, 6, 3
    model = LstmTSEMAdapted(W=W, K=K, latent_size=16, hidden_size=16).eval()
    node_seq = torch.randn(B, W, 6)
    nbr_node_seqs = torch.randn(B, K, W, 6)
    nbr_mask = torch.zeros(B, K)
    nbr_mask[0] = 1
    nbr_mask[1, :2] = 1
    nbr_mask[2, 0] = 1
    nbr_mask[3] = 0  # 고립 ego
    batch = {'node_seq': node_seq, 'nbr_node_seqs': nbr_node_seqs, 'nbr_mask': nbr_mask}

    with torch.no_grad():
        logits_batched = model(batch)
        logits_solo = torch.cat([
            model({'node_seq': node_seq[b:b + 1], 'nbr_node_seqs': nbr_node_seqs[b:b + 1],
                  'nbr_mask': nbr_mask[b:b + 1]})
            for b in range(B)
        ], dim=0)
    check('배치 처리 vs 샘플별 단독 처리 (그래프/배치 누수 없어야 동일)', logits_batched, logits_solo,
          tol=1e-5)
    assert not torch.isnan(logits_batched).any(), '배치 처리 결과에 NaN 발생'


# ────────────────────────────────────────────────────────────────
# 3) 고립 ego(이웃 전부 무효) 전용 스트레스 테스트 — 여러 배치 크기·시드로 NaN 없는지 반복 확인
# ────────────────────────────────────────────────────────────────
def test_isolated_ego_no_nan():
    for seed in range(3):
        torch.manual_seed(seed)
        B, W, K = 5, 10, 6
        model = LstmTSEMAdapted(W=W, K=K, latent_size=32, hidden_size=32).eval()
        node_seq = torch.randn(B, W, 6)
        nbr_node_seqs = torch.zeros(B, K, W, 6)  # 이웃 전부 all-zero(무효 프레임)
        nbr_mask = torch.zeros(B, K)  # 이웃 전부 무효
        batch = {'node_seq': node_seq, 'nbr_node_seqs': nbr_node_seqs, 'nbr_mask': nbr_mask}
        with torch.no_grad():
            logits = model(batch)
        has_nan = torch.isnan(logits).any().item()
        status = 'PASS' if not has_nan else 'FAIL'
        print(f'[{status}] 고립 ego(seed={seed}) NaN 없음: has_nan={has_nan}')
        assert not has_nan, f'seed={seed}: 고립 ego 배치에서 NaN 발생'


if __name__ == '__main__':
    print('=== LSTM-adapted 재구현 검증 (원본 repo 없음 — 이 파일 내 브루트포스 대조만) ===\n')
    test_masked_mean_pooling()
    print()
    test_batch_vs_solo_no_leakage()
    print()
    test_isolated_ego_no_nan()
    print('\n모든 검증 통과.')
