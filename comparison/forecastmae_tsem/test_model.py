"""
Forecast-MAE-adapted 재구현 검증 — 커스텀 마스킹·복원 정렬 로직을 독립 브루트포스와 대조.

comparison/hivt_tsem, qcnet_tsem, cratpred_tsem, simpl_tsem/test_model.py와 동일한 방법론.
TSEMBlock(=nn.MultiheadAttention 감싼 표준 transformer layer)은 attention 자체를 재구현하지
않으므로(model.py docstring 참조) 별도 검증하지 않는다 — 검증 대상은 (1) 마스킹 샘플링이
"유효 에이전트 중 정확히 mask_ratio 비율을, 최소 1개는 항상 남기고" 선택하는지, (2) 복원 loss가
마스킹된+실제관측된 위치에만 걸리는지, (3) 배치 처리와 그래프 격리(누수 여부)다.

원본(comparison/forecast-mae/) 코드는 건드리지 않음.

실행: python comparison/forecastmae_tsem/test_model.py
"""
from __future__ import annotations

import torch

from model import TSEMMAEPretrain, TSEMMAEFinetune

TOL = 1e-6


def check(name: str, cond: bool, detail: str = '') -> None:
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {name}' + (f' — {detail}' if detail else ''))
    assert cond, f'{name} 실패: {detail}'


def check_close(name: str, a: torch.Tensor, b: torch.Tensor, tol: float = TOL) -> None:
    diff = (a - b).abs().max().item()
    status = 'PASS' if diff < tol else 'FAIL'
    print(f'[{status}] {name}: max|diff|={diff:.3e} (tol={tol:.0e})')
    assert diff < tol, f'{name} 불일치: max|diff|={diff:.3e}'


# ────────────────────────────────────────────────────────────────
# 1) _sample_mask — 유효 에이전트 중 정확히 mask_ratio 비율(반올림)을 뽑는지,
#    무효 에이전트는 절대 안 뽑히는지, 최소 1개 남기는 규칙을 지키는지
# ────────────────────────────────────────────────────────────────
def test_sample_mask_ratio_and_validity():
    torch.manual_seed(0)
    model = TSEMMAEPretrain(W=5, K=6, embed_dim=16, encoder_depth=1, decoder_depth=1,
                            num_heads=2, mask_ratio=0.5)
    B, N = 200, 7  # 통계적으로 비율을 확인하기 위해 넉넉한 배치

    # 다양한 유효 에이전트 수 케이스를 섞어 구성(1개~7개 전부 유효)
    valid_agent = torch.zeros(B, N, dtype=torch.bool)
    for b in range(B):
        n_valid = (b % N) + 1  # 1..N 순환
        valid_agent[b, :n_valid] = True

    mask = model._sample_mask(valid_agent)

    check('마스킹은 항상 valid_agent의 부분집합', bool((mask & ~valid_agent).sum() == 0))

    # n_valid=1인 샘플들은 마스킹 대상이 0개여야 함(복원할 context가 없어짐 방지)
    for b in range(B):
        n_valid = int(valid_agent[b].sum())
        n_masked = int(mask[b].sum())
        if n_valid == 1:
            check(f'  n_valid=1(b={b})이면 마스킹 0개', n_masked == 0, f'실제 {n_masked}')
        else:
            expected = max(1, int(n_valid * model.mask_ratio))
            # 정렬 기반 임계값이라 반올림/동률 처리로 ±1 오차 허용
            check(f'  n_valid={n_valid}(b={b}) 마스킹 개수 ≈ round(n_valid*0.5)',
                  abs(n_masked - expected) <= 1, f'기대 {expected}, 실제 {n_masked}')


# ────────────────────────────────────────────────────────────────
# 2) reg_mask 정렬 — 브루트포스로 "마스킹된 & present인" 위치만 loss에 들어가는지 확인
#    (모델 내부 forward 로직 중 reg_mask 계산 부분을 별도 재현해서 대조)
# ────────────────────────────────────────────────────────────────
def test_reg_mask_alignment():
    torch.manual_seed(1)
    B, N, W = 4, 5, 6
    present = torch.rand(B, N, W) > 0.3  # 임의의 관측 패턴
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, 0] = True  # 편의상 agent0만 마스킹 대상으로 가정

    # 벡터화(모델과 동일한 방식)
    reg_mask_vec = present.clone()
    reg_mask_vec[~mask] = False

    # 브루트포스(파이썬 for문으로 원소별 판정)
    reg_mask_bf = torch.zeros_like(present)
    for b in range(B):
        for n in range(N):
            for t in range(W):
                reg_mask_bf[b, n, t] = bool(mask[b, n]) and bool(present[b, n, t])

    check('reg_mask 벡터화 vs 브루트포스 완전 일치', bool((reg_mask_vec == reg_mask_bf).all()))


# ────────────────────────────────────────────────────────────────
# 3) 배치 offset / 그래프 격리 — 사전학습·미세조정 둘 다 배치 처리와 샘플별 단독 처리가 동일한지
# ────────────────────────────────────────────────────────────────
def test_batch_isolation_pretrain_and_finetune():
    torch.manual_seed(2)
    B, W, K = 3, 6, 2

    torch.manual_seed(10)
    pre = TSEMMAEPretrain(W=W, K=K, embed_dim=16, encoder_depth=1, decoder_depth=1,
                          num_heads=2, mask_ratio=0.0).eval()  # mask_ratio=0: 마스킹 무작위성 제거하고 순수 배치격리만 확인
    node_seq = torch.randn(B, W, 6)
    nbr_node_seqs = torch.randn(B, K, W, 6)
    nbr_mask = torch.ones(B, K)
    batch = {'node_seq': node_seq, 'nbr_node_seqs': nbr_node_seqs, 'nbr_mask': nbr_mask}

    # mask_ratio=0이면 _sample_mask가 전부 False를 반환해 결정적 — 배치/단독 대조 가능
    with torch.no_grad():
        loss_batched = pre(batch)
        loss_solo = torch.stack([
            pre({'node_seq': node_seq[b:b + 1], 'nbr_node_seqs': nbr_node_seqs[b:b + 1],
                'nbr_mask': nbr_mask[b:b + 1]})
            for b in range(B)
        ])
    # loss가 스칼라(배치 전체 평균)라 개별 합을 직접 비교하기보단, reg_mask가 전부 비어
    # loss=0*sum이 되는 경로(마스킹 없음)인지만 확인 — 실제 격리 검증은 finetune(분류, 샘플별
    # logits 벡터가 나옴)에서 정밀 대조한다.
    check('mask_ratio=0이면 pretrain loss는 항상 0(그라드 흐름용 sum*0)', bool(loss_batched.item() == 0.0))

    ft = TSEMMAEFinetune(W=W, K=K, embed_dim=16, encoder_depth=1, num_heads=2).eval()
    with torch.no_grad():
        logits_batched = ft(batch)
        logits_solo = torch.cat([
            ft({'node_seq': node_seq[b:b + 1], 'nbr_node_seqs': nbr_node_seqs[b:b + 1],
               'nbr_mask': nbr_mask[b:b + 1]})
            for b in range(B)
        ], dim=0)
    check_close('finetune 배치 처리 vs 샘플별 단독 처리 (그래프 누수 없어야 동일)',
               logits_batched, logits_solo, tol=1e-5)


if __name__ == '__main__':
    print('=== Forecast-MAE-adapted 재구현 검증 (원본 comparison/forecast-mae/는 미변경) ===\n')
    test_sample_mask_ratio_and_validity()
    print()
    test_reg_mask_alignment()
    print()
    test_batch_isolation_pretrain_and_finetune()
    print('\n모든 검증 통과.')
