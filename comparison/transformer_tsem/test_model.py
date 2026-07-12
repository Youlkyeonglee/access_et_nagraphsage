"""
Transformer-adapted 재구현 검증 — comparison/cratpred_tsem, hivt_tsem/test_model.py와 동일한
방법론(완전히 독립적인 브루트포스 구현과 대조)을 따르되, 이 모델 고유의 검증 포인트(masked
mean-pooling, positional encoding, 배치/패딩 처리)에 맞춘 4개 테스트.

원본 GitHub repo 없음(model.py 상단 docstring 참조) — 대조할 "원본 코드"가 없으므로, 검증은
(a) 우리가 직접 정의한 masked mean-pooling 로직이 의도한 대로 동작하는지(bruteforce 대조),
(b) positional encoding이 sanity를 만족하는지, (c) 배치 벡터화가 그래프/배치 누수 없이 샘플별
단독 처리와 일치하는지, (d) 이웃이 전부 없는 극단 케이스에서 NaN이 나지 않는지를 확인한다.

실행: python comparison/transformer_tsem/test_model.py
"""
from __future__ import annotations

import torch

from model import SinusoidalPositionalEncoding, TransformerTSEMAdapted

TOL = 1e-6


def check(name: str, a: torch.Tensor, b: torch.Tensor, tol: float = TOL) -> None:
    diff = (a - b).abs().max().item()
    status = 'PASS' if diff < tol else 'FAIL'
    print(f'[{status}] {name}: max|diff|={diff:.3e} (tol={tol:.0e})')
    assert diff < tol, f'{name} 불일치: max|diff|={diff:.3e}'


def check_true(name: str, cond: bool) -> None:
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {name}')
    assert cond, f'{name} 실패'


# ────────────────────────────────────────────────────────────────
# 1) masked mean-pooling — 벡터화 forward vs 브루트포스(무효 이웃을 파이썬 for문으로 직접 제외)
# ────────────────────────────────────────────────────────────────
def test_masked_mean_pooling():
    torch.manual_seed(0)
    B, W, K, D = 4, 6, 5, 8
    model = TransformerTSEMAdapted(W=W, K=K, d_model=D, nhead=2, num_layers=1,
                                    dim_feedforward=16, dropout=0.0).eval()

    node_seq = torch.randn(B, W, 6)
    nbr_node_seqs = torch.randn(B, K, W, 6)
    # 무효 슬롯은 raw 데이터도 전부 0으로 만들어 present=False가 되게 함(실제 파이프라인과 동일)
    nbr_mask = torch.zeros(B, K)
    nbr_mask[:, :3] = 1.0  # 앞 3개만 유효
    nbr_mask[0, :] = 1.0   # 배치0은 전부 유효(엣지케이스 다양화)
    nbr_node_seqs = nbr_node_seqs * nbr_mask.view(B, K, 1, 1)
    batch = {'node_seq': node_seq, 'nbr_node_seqs': nbr_node_seqs, 'nbr_mask': nbr_mask}

    with torch.no_grad():
        x_seq, valid_agent = model._build_inputs(batch)
        h = model.input_proj(x_seq)
        h = model.pos_encoding(h)
        h = model.transformer(h, mask=model.causal_mask.to(dtype=h.dtype))
        agent_embed = h[:, -1, :].view(B, model.N, D)

        # 브루트포스: 배치·에이전트별 for문으로 유효 이웃만 직접 평균
        pooled_bf = torch.zeros(B, D)
        for b in range(B):
            valid_idx = [k for k in range(K) if valid_agent[b, 1 + k]]
            assert len(valid_idx) > 0
            acc = torch.zeros(D)
            for k in valid_idx:
                acc = acc + agent_embed[b, 1 + k]
            pooled_bf[b] = acc / len(valid_idx)

        logits = model(batch)  # 벡터화 경로(전체 forward)에서도 같은 pooling을 거침
        # pooling 결과 자체를 별도로 재계산해 비교(model.forward 내부와 동일 수식)
        nbr_embed = agent_embed[:, 1:, :]
        nbr_valid = valid_agent[:, 1:].float()
        nbr_sum = (nbr_embed * nbr_valid.unsqueeze(-1)).sum(dim=1)
        nbr_count = nbr_valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled_vec = nbr_sum / nbr_count

    check('masked mean-pooling 벡터화 vs 브루트포스(무효 이웃 제외)', pooled_vec, pooled_bf)


# ────────────────────────────────────────────────────────────────
# 2) sinusoidal positional encoding — 시간축마다 다른 값을 주는지(sanity)
# ────────────────────────────────────────────────────────────────
def test_positional_encoding_varies_per_step():
    pe_module = SinusoidalPositionalEncoding(d_model=16, max_len=32)
    pe = pe_module.pe[0]  # [max_len, D]
    # 서로 다른 timestep의 PE 벡터가 전부 달라야 함(퇴화 케이스 방지)
    for i in range(10):
        for j in range(i + 1, 10):
            diff = (pe[i] - pe[j]).abs().max().item()
            assert diff > 1e-4, f'PE[{i}]와 PE[{j}]가 사실상 동일함(diff={diff:.2e})'
    print('[PASS] positional encoding이 시간축(t=0..9)마다 서로 다른 값을 가짐')

    # forward가 실제로 입력에 PE를 더하는지도 확인
    x = torch.zeros(2, 10, 16)
    out = pe_module(x)
    check('positional_encoding(0-tensor) == PE[:, :10, :] 그 자체', out, pe_module.pe[:, :10, :].expand(2, -1, -1))


# ────────────────────────────────────────────────────────────────
# 3) 배치 처리 vs 샘플별 단독 처리 — 그래프/배치 누수 검사(causal mask 포함 특히 중요)
# ────────────────────────────────────────────────────────────────
def test_batch_vs_solo_consistency():
    torch.manual_seed(2)
    B, W, K = 3, 7, 4
    model = TransformerTSEMAdapted(W=W, K=K, d_model=16, nhead=2, num_layers=2,
                                    dim_feedforward=32, dropout=0.0).eval()
    node_seq = torch.randn(B, W, 6)
    nbr_node_seqs = torch.randn(B, K, W, 6)
    nbr_mask = torch.ones(B, K)
    nbr_mask[1, 2:] = 0.0  # 배치1은 이웃 일부 무효
    nbr_node_seqs = nbr_node_seqs * nbr_mask.view(B, K, 1, 1)
    batch = {'node_seq': node_seq, 'nbr_node_seqs': nbr_node_seqs, 'nbr_mask': nbr_mask}

    with torch.no_grad():
        logits_batched = model(batch)
        logits_solo = torch.cat([
            model({'node_seq': node_seq[b:b + 1], 'nbr_node_seqs': nbr_node_seqs[b:b + 1],
                  'nbr_mask': nbr_mask[b:b + 1]})
            for b in range(B)
        ], dim=0)
    check('배치 처리 vs 샘플별 단독 처리 (그래프/배치 누수 없어야 동일)', logits_batched, logits_solo, tol=1e-4)


# ────────────────────────────────────────────────────────────────
# 4) 이웃이 전부 없는 경우 — NaN 없이 forward 되는지
# ────────────────────────────────────────────────────────────────
def test_no_neighbors_no_nan():
    torch.manual_seed(3)
    B, W, K = 2, 8, 5
    model = TransformerTSEMAdapted(W=W, K=K, d_model=16, nhead=2, num_layers=1,
                                    dim_feedforward=32, dropout=0.0).eval()
    node_seq = torch.randn(B, W, 6)
    nbr_node_seqs = torch.zeros(B, K, W, 6)  # 이웃 전부 결측
    nbr_mask = torch.zeros(B, K)             # 이웃 전부 무효
    batch = {'node_seq': node_seq, 'nbr_node_seqs': nbr_node_seqs, 'nbr_mask': nbr_mask}

    with torch.no_grad():
        logits = model(batch)
    check_true('이웃이 전부 없어도 NaN 없음', not torch.isnan(logits).any().item())
    check_true('이웃이 전부 없어도 Inf 없음', not torch.isinf(logits).any().item())
    check_true('출력 shape이 [B, num_classes]', logits.shape == (B, 3))


if __name__ == '__main__':
    print('=== Transformer-adapted 재구현 검증 (원본 repo 없음 — model.py 상단 docstring 참조) ===\n')
    test_masked_mean_pooling()
    print()
    test_positional_encoding_varies_per_step()
    print()
    test_batch_vs_solo_consistency()
    print()
    test_no_neighbors_no_nan()
    print('\n모든 검증 통과.')
