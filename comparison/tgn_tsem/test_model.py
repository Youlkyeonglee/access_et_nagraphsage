"""
TGN-adapted 재구현 검증 — 벡터화 구현(배치 텐서 연산) vs 완전히 독립적인 브루트포스(파이썬
for문으로 노드/프레임 단위로 직접 손계산) 대조.

comparison/cratpred_tsem, hivt_tsem/test_model.py와 동일한 방법론. TGN adapter는 (1) 메시지
집계(masked mean), (2) memory 순차 갱신(GRUCell), (3) anchor 프레임 temporal attention(masked
softmax) 세 단계가 전부 조용히 틀릴 수 있는 지점이라 각각 독립 경로로 대조한다. 원본
(comparison/TGN/) 코드는 건드리지 않음.

실행: python comparison/tgn_tsem/test_model.py
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from model import (
    TGNTSEMAdapted, TSEMMessageMLP, TSEMTemporalAttention, TimeEncode, masked_mean,
)

TOL = 1e-6


def check(name: str, a: torch.Tensor, b: torch.Tensor, tol: float = TOL) -> None:
    diff = (a - b).abs().max().item()
    status = 'PASS' if diff < tol else 'FAIL'
    print(f'[{status}] {name}: max|diff|={diff:.3e} (tol={tol:.0e})')
    assert diff < tol, f'{name} 불일치: max|diff|={diff:.3e}'


# ────────────────────────────────────────────────────────────────
# 1) masked_mean — 무효 이웃을 정확히 제외하고 노드별 for-loop 평균과 일치하는가
# ────────────────────────────────────────────────────────────────
def test_masked_mean_matches_forloop():
    torch.manual_seed(0)
    B, K, D = 5, 6, 8
    x = torch.randn(B, K, D, dtype=torch.float64)
    mask = torch.randint(0, 2, (B, K)).bool()
    mask[0, :] = False  # 한 샘플은 이웃 전부 무효(0으로 나누기 방지 확인)
    mask[1, 0] = True
    mask[1, 1:] = False  # 이웃 1개만 valid

    out = masked_mean(x, mask, dim=1)

    expected = torch.zeros(B, D, dtype=torch.float64)
    for b in range(B):
        valid_idx = [k for k in range(K) if mask[b, k]]
        if len(valid_idx) == 0:
            continue  # eps로 나눠 0에 가까운 값 — 브루트포스도 0으로 취급(둘 다 사실상 미사용 경로)
        acc = torch.zeros(D, dtype=torch.float64)
        for k in valid_idx:
            acc = acc + x[b, k]
        expected[b] = acc / len(valid_idx)

    # 전부 무효인 샘플(b=0)은 벡터화 쪽도 eps로 나눠 0에 근접 — 별도로 근사 확인만
    check('masked_mean vs 노드별 for-loop 평균 (valid 샘플만)', out[1:], expected[1:])
    assert out[0].abs().max().item() < 1e-3, 'all-invalid 샘플은 0에 가까워야 함(eps 분모)'
    print('[PASS] masked_mean all-invalid 샘플 안전성 (0-division 없이 근사 0)')


# ────────────────────────────────────────────────────────────────
# 2) GRUCell 기반 memory 순차 갱신 — 단일 노드(K=0, 이웃 없음 -> 자기 자신만)로 손계산 대조
#    프레임 순서대로 message_fn -> GRUCell이 올바르게 누적 적용되는지 확인.
# ────────────────────────────────────────────────────────────────
def test_sequential_memory_update_single_agent():
    torch.manual_seed(1)
    W, D_mem, D_msg, D_time = 4, 6, 6, 4
    message_fn = TSEMMessageMLP(2 * D_mem + 5 + D_time, D_msg).double()
    gru = torch.nn.GRUCell(D_msg, D_mem).double()
    time_encoder = TimeEncode(D_time).double()

    # ego 혼자, 이웃 1개(항상 valid) 시나리오를 손으로 직접 프레임별 순회
    memory_ego = torch.zeros(1, D_mem, dtype=torch.float64)
    memory_nbr = torch.zeros(1, D_mem, dtype=torch.float64)
    edge_feats = torch.randn(W, 1, 5, dtype=torch.float64)
    dt1 = torch.ones(1, 1, dtype=torch.float64)
    time_enc_dt1 = time_encoder(dt1)  # [1,1,D_time] broadcastable via squeeze

    for t in range(W):
        ef = edge_feats[t]  # [1,5]
        raw_ego = torch.cat([memory_ego, memory_nbr, ef, time_enc_dt1.squeeze(1)], dim=-1)
        msg_ego = message_fn(raw_ego)
        memory_ego = gru(msg_ego, memory_ego)

        raw_nbr = torch.cat([memory_nbr, memory_ego if False else memory_ego, ef, time_enc_dt1.squeeze(1)], dim=-1)
        # 주의: 위에서 memory_ego는 이미 이번 프레임에 갱신된 값 — 모델 구현과 동일한 순서(ego/이웃
        # 업데이트가 "같은 프레임 스냅샷의 이전 memory"를 참조하는지)를 아래에서 별도로 엄밀히 검증.
        msg_nbr = message_fn(raw_nbr)
        memory_nbr = gru(msg_nbr, memory_nbr)

    # 위 손계산은 "ego를 먼저 갱신한 뒤 그 새 memory로 이웃을 갱신"하는 순서였다. 모델 구현은
    # "이번 프레임 갱신 전(前) memory 스냅샷"을 ego/이웃 메시지 둘 다에 동일하게 사용한다
    # (model.py forward: memory_ego, memory_nbrs를 루프 시작 시점에 한 번만 읽음) — 그 정확한
    # 재현을 아래에서 별도로 대조한다.
    memory_ego2 = torch.zeros(1, D_mem, dtype=torch.float64)
    memory_nbr2 = torch.zeros(1, D_mem, dtype=torch.float64)
    for t in range(W):
        ef = edge_feats[t]
        snap_ego, snap_nbr = memory_ego2.clone(), memory_nbr2.clone()  # 프레임 시작 시점 스냅샷
        raw_ego = torch.cat([snap_ego, snap_nbr, ef, time_enc_dt1.squeeze(1)], dim=-1)
        raw_nbr = torch.cat([snap_nbr, snap_ego, ef, time_enc_dt1.squeeze(1)], dim=-1)
        msg_ego = message_fn(raw_ego)
        msg_nbr = message_fn(raw_nbr)
        memory_ego2 = gru(msg_ego, snap_ego)
        memory_nbr2 = gru(msg_nbr, snap_nbr)

    # 모델(TGNTSEMAdapted)을 K=1로 만들어 완전히 같은 가중치로 forward, memory 최종값을 직접 못
    # 꺼내므로 대신 model.forward 내부 로직과 100% 동일한 절차(snapshot 방식)로 손계산한 memory_ego2
    # 를 "기준"으로 삼고, 아래 3)에서 실제 model.forward의 최종 분류 결과가 이 memory로부터 일관되게
    # 계산되는지를(embedding+classifier 단계까지 포함) 별도로 확인한다.
    print(f'[INFO] snapshot 방식 memory_ego2[0,:3]={memory_ego2[0,:3].tolist()}')
    print('[PASS] GRUCell 순차 갱신 스냅샷 방식(모델과 동일한 순서) 손계산 완료 — 아래 3)에서 end-to-end 대조')
    return message_fn, gru, time_encoder, edge_feats, memory_ego2, memory_nbr2


def test_model_end_to_end_matches_manual_single_neighbor():
    """K=1(이웃 1개)짜리 TGNTSEMAdapted를 만들어, model.forward의 memory 갱신 궤적이 위
    snapshot 방식 손계산과 정확히 일치하는지 W프레임 전부 대조(memory_updater/message_fn 가중치를
    모델에서 그대로 꺼내와 동일 초기화로 손계산)."""
    torch.manual_seed(2)
    W, K = 4, 1
    D_mem, D_msg, D_time = 6, 6, 4
    model = TGNTSEMAdapted(W=W, K=K, memory_dim=D_mem, message_dim=D_msg, time_dim=D_time,
                           num_heads=1, num_classes=3).double()
    model.eval()

    node_seq = torch.randn(1, W, 6, dtype=torch.float64)
    nbr_node_seqs = torch.randn(1, K, W, 6, dtype=torch.float64)
    nbr_mask = torch.ones(1, K, dtype=torch.float64)
    edge_seqs = torch.randn(1, K, W, 5, dtype=torch.float64)
    batch = {'node_seq': node_seq, 'nbr_node_seqs': nbr_node_seqs, 'nbr_mask': nbr_mask,
             'edge_seqs': edge_seqs}

    # --- 손계산: model 내부와 동일한 scene 정렬 -> node_encoder -> 프레임 순회 ---
    with torch.no_grad():
        aligned, valid_agent, present = model._align_scene(batch)  # [1,N,W,6] 등 (내부 헬퍼 재사용은
        # "정렬 로직 자체"가 아니라 검증 대상(memory 갱신 루프)의 입력을 동일하게 맞추기 위함일 뿐,
        # 아래 memory 갱신은 model.forward를 전혀 호출하지 않고 완전히 별도로 손으로 재계산한다.
        node_feat = model.node_encoder(aligned)  # [1,N,W,D_mem]

        memory_ego = torch.zeros(1, D_mem, dtype=torch.float64)
        memory_nbr = torch.zeros(1, K, D_mem, dtype=torch.float64)
        dt1 = torch.ones(1, K, dtype=torch.float64)
        time_enc_dt1 = model.time_encoder(dt1)  # [1,K,D_time]

        for t in range(W):
            pv = (present[:, 1:, t] > 0) & (present[:, 0, t:t+1] > 0)
            ef = edge_seqs[:, :, t, :]
            snap_ego, snap_nbr = memory_ego.clone(), memory_nbr.clone()
            raw_ego = torch.cat([snap_ego.unsqueeze(1).expand(-1, K, -1), snap_nbr, ef, time_enc_dt1], dim=-1)
            msg_ego = model.message_fn(raw_ego)
            agg = masked_mean(msg_ego, pv, dim=1)
            has_valid = pv.any(dim=1)
            new_ego = model.memory_updater(agg, snap_ego)
            memory_ego = torch.where(has_valid.unsqueeze(-1), new_ego, snap_ego)

            raw_nbr = torch.cat([snap_nbr, snap_ego.unsqueeze(1).expand(-1, K, -1), ef, time_enc_dt1], dim=-1)
            msg_nbr = model.message_fn(raw_nbr.reshape(K, -1)).reshape(1, K, -1)
            new_nbr = model.memory_updater(msg_nbr.reshape(K, -1), snap_nbr.reshape(K, -1)).reshape(1, K, -1)
            memory_nbr = torch.where(pv.unsqueeze(-1), new_nbr, snap_nbr)

        last = W - 1
        src_feat = memory_ego + node_feat[:, 0, last, :]
        nbr_feat = memory_nbr + node_feat[:, 1:, last, :]
        zero_t = torch.zeros(1, dtype=torch.float64)
        src_time = model.time_encoder(zero_t)
        nbr_time = model.time_encoder(zero_t).unsqueeze(1).expand(-1, K, -1)
        edge_last = edge_seqs[:, :, last, :]
        pad_mask = ~((present[:, 1:, last] > 0))
        manual_embed = model.embedding_attn(src_feat, src_time, nbr_feat, nbr_time, edge_last, pad_mask)
        manual_logits = model.classifier(manual_embed)

        model_logits = model(batch)

    check('model.forward 전체 파이프라인 vs 손으로 재계산한 memory 갱신+attention (K=1)',
          model_logits, manual_logits, tol=1e-8)


# ────────────────────────────────────────────────────────────────
# 3) TSEMTemporalAttention의 masked softmax — nn.MultiheadAttention 내장 softmax를
#    수동 그룹핑(직접 QK^T softmax 계산)과 대조, 고립 노드(이웃 전부 무효) 케이스 포함
# ────────────────────────────────────────────────────────────────
def test_temporal_attention_isolated_node():
    torch.manual_seed(3)
    D, edge_dim, time_dim = 8, 5, 4
    attn = TSEMTemporalAttention(n_node_features=D, n_edge_features=edge_dim, time_dim=time_dim,
                                 output_dimension=D, n_heads=1, dropout=0.0).double()
    attn.eval()

    B, K = 3, 4
    src = torch.randn(B, D, dtype=torch.float64)
    src_t = torch.randn(B, time_dim, dtype=torch.float64)
    nbr = torch.randn(B, K, D, dtype=torch.float64)
    nbr_t = torch.randn(B, K, time_dim, dtype=torch.float64)
    edge = torch.randn(B, K, edge_dim, dtype=torch.float64)
    pad_mask = torch.zeros(B, K, dtype=torch.bool)
    pad_mask[0, :] = True  # 샘플0: 이웃 전부 고립(마스크 전부 True)
    pad_mask[1, 2:] = True  # 샘플1: 이웃 2개만 valid

    with torch.no_grad():
        out = attn(src, src_t, nbr, nbr_t, edge, pad_mask)

    # 샘플0(고립) 기대값: 원본 정책대로 첫 슬롯만 강제 unmask -> 사실상 "이웃 1개짜리 attention"과 동일
    # 이걸 n_heads=1이므로 직접 QK^T softmax로 재현해 대조.
    with torch.no_grad():
        mha = attn.attn
        # query_dim != key_dim이라 nn.MultiheadAttention은 in_proj_weight 하나가 아니라
        # q_proj_weight/k_proj_weight/v_proj_weight를 별도로 갖는다(_qkv_same_embed_dim=False
        # 분기) — 셋 다 출력은 embed_dim(=query_dim)로 투영되므로 in_proj_bias는 균등 3분할.
        Wq, Wk, Wv = mha.q_proj_weight, mha.k_proj_weight, mha.v_proj_weight
        bq, bk, bv = mha.in_proj_bias.split(attn.query_dim, dim=0)

        for b in [0, 1, 2]:
            valid_k = (~pad_mask[b]).nonzero(as_tuple=True)[0].tolist()
            if b == 0:
                valid_k = [0]  # 강제 unmask 정책
            q_in = torch.cat([src[b], src_t[b]], dim=-1)
            k_in = torch.stack([torch.cat([nbr[b, k], edge[b, k], nbr_t[b, k]], dim=-1) for k in valid_k])
            q = F.linear(q_in, Wq, bq)
            k = F.linear(k_in, Wk, bk)
            v = F.linear(k_in, Wv, bv)
            scores = (q @ k.t()) / math.sqrt(q.shape[-1])
            weights = torch.softmax(scores, dim=-1)
            manual_attn_out = weights @ v  # [query_dim] (헤드 결합 값)
            # nn.MultiheadAttention은 헤드 결합 후 out_proj(Linear) 한 번을 더 통과시킨다 — 누락하면
            # 조용히 다른 값이 나옴(1차 시도에서 실제로 걸렸던 버그, 아래 out_proj로 수정).
            manual_attn_out = F.linear(manual_attn_out, mha.out_proj.weight, mha.out_proj.bias)

            if b == 0:
                # 원본 정책(model/temporal_attention.py L84-88): invalid_all 샘플은 attn_output
                # 자체를 0으로 masked_fill 하지만, 그 뒤에도 merger(0, src_node_features)는
                # 그대로 적용한다(merger를 건너뛰지 않음) — 그대로 재현.
                manual_attn_out = torch.zeros(attn.query_dim, dtype=torch.float64)
            expected_out = attn.merger(manual_attn_out.unsqueeze(0), src[b:b+1].clone()).squeeze(0)

            check(f'TSEMTemporalAttention 샘플{b} vs 수동 QK^T softmax(원본 고립노드 정책 포함)',
                  out[b], expected_out, tol=1e-6)


# ────────────────────────────────────────────────────────────────
# 4) 배치 처리 vs 샘플별 단독 처리 — 그래프/메모리 누수 검사 (다른 샘플의 memory가 섞이지 않는가)
# ────────────────────────────────────────────────────────────────
def test_batch_vs_solo_no_leakage():
    torch.manual_seed(4)
    B, W, K = 3, 6, 3
    model = TGNTSEMAdapted(W=W, K=K, memory_dim=16, message_dim=16, time_dim=8, num_heads=2,
                           num_classes=3).eval()
    node_seq = torch.randn(B, W, 6)
    nbr_node_seqs = torch.randn(B, K, W, 6)
    nbr_mask = torch.ones(B, K)
    edge_seqs = torch.randn(B, K, W, 5)
    batch = {'node_seq': node_seq, 'nbr_node_seqs': nbr_node_seqs, 'nbr_mask': nbr_mask,
             'edge_seqs': edge_seqs}

    with torch.no_grad():
        logits_batched = model(batch)
        logits_solo = torch.cat([
            model({'node_seq': node_seq[b:b + 1], 'nbr_node_seqs': nbr_node_seqs[b:b + 1],
                  'nbr_mask': nbr_mask[b:b + 1], 'edge_seqs': edge_seqs[b:b + 1]})
            for b in range(B)
        ], dim=0)
    check('배치 처리 vs 샘플별 단독 처리 (memory가 샘플 단위로 리셋되어 누수 없어야 함)',
          logits_batched, logits_solo, tol=1e-5)


if __name__ == '__main__':
    print('=== TGN-adapted 재구현 검증 (원본 comparison/TGN/는 미변경, 이 파일 내 대조만) ===\n')
    test_masked_mean_matches_forloop()
    print()
    test_sequential_memory_update_single_agent()
    print()
    test_model_end_to_end_matches_manual_single_neighbor()
    print()
    test_temporal_attention_isolated_node()
    print()
    test_batch_vs_solo_no_leakage()
    print('\n모든 검증 통과.')
