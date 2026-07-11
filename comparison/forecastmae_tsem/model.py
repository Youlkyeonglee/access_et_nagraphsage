"""
Forecast-MAE-adapted — TSEM-SAGE 비교용 재구현
=========================================
원본: comparison/forecast-mae/ (Cheng, Mei & Liu, ICCV 2023)
  - src/model/layers/transformer_blocks.py::Block -> TSEMBlock (수식 동일, 아래 재구현)
  - src/model/model_mae.py                        -> TSEMMAEPretrain (마스킹+복원 사전학습)
  - src/model/model_forecast.py                   -> TSEMMAEFinetune (사전학습 인코더 재사용 + 분류 헤드)
  - src/model/layers/agent_embedding.py(NATTEN 기반) -> TSEMAgentEmbed (아래 "주요 변경점" 참조 — 대체)

이 환경(tna_research)에는 `natten`(Neighborhood Attention, CUDA 커널 필요)과 `timm`이 모두
설치돼 있지 않다. `transformer_blocks.py::Block`은 순수 PyTorch(`nn.MultiheadAttention`)라
DropPath(timm 유래, 아래 재구현)만 대체하면 그대로 재구현 가능하지만, `agent_embedding.py`의
`AgentEmbeddingLayer`는 NATTEN의 1D 지역(windowed) attention에 강하게 의존해 CUDA 커널 없이는
동작 자체가 불가능하다 — 이건 HiVT의 `torch_geometric`처럼 "얇은 wrapper만 걷어내면 되는"
의존성이 아니라 아예 대체 커널이 없으면 재현 불가능한 경우라, README.md에 상세히 밝힌 근거로
**Conv1d 토크나이저 + 표준 global self-attention(TSEMBlock 재사용)**으로 대체했다 — 우리
W=10(원 논문 historical_steps=50)처럼 짧은 시�퀀스에서는 지역attention과 전역attention의 실질
차이가 작다는 점도 이 대체를 뒷받침한다.

Forecast-MAE의 진짜 검증 대상(README.md 참조)은 백본 종류가 아니라 **"마스킹된 에이전트를
복원하도록 사전학습하면 다운스트림 성능이 좋아지는가"**라는 자기지도 사전학습 프로토콜 자체이므로,
백본을 교체해도 이 핵심 비교 포인트는 그대로 유지된다.

주요 변경점(README.md 참조):
  - AgentEmbeddingLayer(NATTEN) → Conv1d + TSEMBlock(전역 self-attention) 대체 — natten 미설치.
  - LaneEmbeddingLayer·차선 마스킹·차선 복원loss 제거 — 맵 데이터 없음.
  - future_embed/future_pred(미래 궤적 복원) 제거 — 우리 데이터엔 미래 좌표 정답이 없음(3클래스
    state 라벨만 있음). 사전학습 복원 대상은 **과거 궤적(history)만**.
  - MAE 원조의 비대칭 encoder/decoder(마스킹된 토큰을 인코더 입력에서 아예 제거해 계산량을
    줄이는 트릭)는 생략 — 우리 그래프가 에이전트 7개뿐이라 그 최적화의 이득이 미미하다. 대신
    BERT류 방식(마스킹 위치를 mask_token으로 치환해 전부 함께 인코딩)으로 단순화 — 마스킹+복원
    학습 신호 자체는 동일하게 유지.
  - MultimodalDecoder(다중모달 궤적 회귀) 제거 — 사전학습 후 분류 헤드로 교체.
  - 2단계 프로토콜은 원본 그대로 유지: (1) 사전학습(레이블 미사용, 마스킹 복원 loss만) →
    (2) 미세조정(사전학습 인코더 가중치 로드 + 분류 헤드, 지도학습).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """timm.models.layers.DropPath 와 동일 동작(stochastic depth) — timm 미설치라 직접 구현."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class TSEMBlock(nn.Module):
    """comparison/forecast-mae/src/model/layers/transformer_blocks.py::Block 과 동일
    (pre-norm MHA+MLP, PyTorch 내장 nn.MultiheadAttention 사용 — natten/timm 의존성 없음,
    DropPath만 위 재구현으로 교체)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, drop_path: float = 0.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, add_bias_kv=qkv_bias,
                                          dropout=dropout, batch_first=True)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout))
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, src: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src2 = self.norm1(src)
        src2 = self.attn(query=src2, key=src2, value=src2, key_padding_mask=key_padding_mask)[0]
        src = src + self.drop_path1(src2)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))
        return src


class TSEMAgentEmbed(nn.Module):
    """AgentEmbeddingLayer(NATTEN) 대체 — Conv1d 토크나이저 + TSEMBlock(전역 self-attention)
    2개, 마지막 timestep을 pooling해 에이전트당 단일 임베딩으로 축약. 파일 상단 docstring 참조."""

    def __init__(self, in_chans: int, embed_dim: int, num_heads: int = 4, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = nn.Conv1d(in_chans, embed_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            TSEMBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [num_agents, in_chans, T] -> [num_agents, embed_dim]"""
        h = self.tokenizer(x).transpose(1, 2)  # [num_agents, T, embed_dim]
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return h[:, -1, :]  # 마지막(anchor) timestep을 요약 토큰으로 사용


class TSEMMAEEncoder(nn.Module):
    """사전학습·미세조정 공유 인코더 — 에이전트별 임베딩 + 위치임베딩 + scene TSEMBlock 스택.
    사전학습 시에는 forward(..., mask=...)로 마스킹된 에이전트를 mask_token으로 치환한다."""

    def __init__(self, W: int, K: int, embed_dim: int = 64, depth: int = 4, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.W = W
        self.K = K
        self.N = 1 + K
        self.embed_dim = embed_dim

        self.agent_embed = TSEMAgentEmbed(in_chans=4, embed_dim=embed_dim, dropout=dropout)  # 4=dx,dy,speed,valid
        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))  # [cx,cz,cos,sin]
        self.blocks = nn.ModuleList([
            TSEMBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def embed_agents(self, actor_in: torch.Tensor, centers: torch.Tensor,
                     head_vecs: torch.Tensor) -> torch.Tensor:
        """actor_in:[B*N,4,W], centers/head_vecs:[B,N,2] -> tokens[B,N,D] (위치임베딩 포함, mask 적용 전)"""
        B, N = centers.shape[:2]
        agent_tok = self.agent_embed(actor_in).view(B, N, self.embed_dim)
        angle = torch.atan2(head_vecs[..., 1], head_vecs[..., 0])
        pos_feat = torch.cat([centers, torch.cos(angle).unsqueeze(-1), torch.sin(angle).unsqueeze(-1)], dim=-1)
        agent_tok = agent_tok + self.pos_embed(pos_feat)
        return agent_tok

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """tokens:[B,N,D], key_padding_mask:[B,N](True=완전 무효라 attention에서 제외),
        mask:[B,N](True=재구성 대상으로 마스킹, mask_token으로 치환) -> encoded[B,N,D]"""
        if mask is not None:
            tokens = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens)
        for blk in self.blocks:
            tokens = blk(tokens, key_padding_mask=key_padding_mask)
        return self.norm(tokens)


def _build_inputs(batch: dict, N: int, W: int):
    """네 어댑터(hivt/qcnet/cratpred/simpl) 공통 패턴 — scene을 ego anchor 헤딩 기준 1회 정렬."""
    device = batch['node_seq'].device
    B = batch['node_seq'].size(0)

    raw = torch.cat([batch['node_seq'].unsqueeze(1), batch['nbr_node_seqs']], dim=1)  # [B,N,W,6]
    nbr_mask = batch.get('nbr_mask')
    if nbr_mask is not None:
        valid_agent = torch.cat(
            [torch.ones(B, 1, device=device, dtype=torch.bool), nbr_mask.bool()], dim=1)
    else:
        valid_agent = raw.abs().sum(dim=(-1, -2)) > 0

    present = raw.abs().sum(dim=-1) > 0  # [B,N,W]

    ego_pos_anchor = batch['node_seq'][:, -1, 0:2]
    ego_dir_anchor = batch['node_seq'][:, -1, 3:5]
    theta = torch.atan2(ego_dir_anchor[:, 1], ego_dir_anchor[:, 0])
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    rot = torch.stack([torch.stack([cos_t, -sin_t], -1), torch.stack([sin_t, cos_t], -1)], -2)
    rot_flat = rot.repeat_interleave(N, dim=0)
    origin_flat = ego_pos_anchor.repeat_interleave(N, dim=0)

    pos_raw = raw[..., 0:2].reshape(B * N, W, 2)
    pos_centered = pos_raw - origin_flat.unsqueeze(1)
    pos = torch.bmm(pos_centered.reshape(-1, 1, 2),
                    rot_flat.unsqueeze(1).expand(-1, W, -1, -1).reshape(-1, 2, 2)).reshape(B * N, W, 2)
    dir_raw = raw[..., 3:5].reshape(B * N, W, 2)
    dir_rot = torch.bmm(dir_raw.reshape(-1, 1, 2),
                        rot_flat.unsqueeze(1).expand(-1, W, -1, -1).reshape(-1, 2, 2)).reshape(B * N, W, 2)
    speed = raw[..., 2].reshape(B * N, W)
    present_flat = present.reshape(B * N, W)
    pos = torch.where(present_flat.unsqueeze(-1), pos, torch.zeros_like(pos))
    speed = torch.where(present_flat, speed, torch.zeros_like(speed))

    actor_in = torch.stack([pos[..., 0], pos[..., 1], speed, present_flat.float()], dim=-1)  # [B*N,W,4]
    actor_in = actor_in.transpose(1, 2)  # [B*N,4,W]

    centers = pos[:, -1, :].reshape(B, N, 2)
    head_vecs = dir_rot[:, -1, :].reshape(B, N, 2)

    return actor_in, pos.reshape(B, N, W, 2), present_flat.reshape(B, N, W), centers, head_vecs, valid_agent


class TSEMMAEPretrain(nn.Module):
    """1단계: 마스킹된 에이전트의 과거 궤적을 복원하도록 자기지도 사전학습.
    배치 dict -> loss(스칼라). 레이블(y) 미사용."""

    def __init__(self, W: int = 10, K: int = 6, embed_dim: int = 64, encoder_depth: int = 4,
                 decoder_depth: int = 2, num_heads: int = 8, mask_ratio: float = 0.5, dropout: float = 0.1):
        super().__init__()
        self.W = W
        self.K = K
        self.N = 1 + K
        self.mask_ratio = mask_ratio

        self.encoder = TSEMMAEEncoder(W, K, embed_dim, encoder_depth, num_heads, dropout)
        self.decoder_embed = nn.Linear(embed_dim, embed_dim)
        self.decoder_blocks = nn.ModuleList([
            TSEMBlock(embed_dim, num_heads, dropout=dropout) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.recon_head = nn.Linear(embed_dim, W * 2)  # 과거 궤적(정규화된 x,z) 복원

    def _sample_mask(self, valid_agent: torch.Tensor) -> torch.Tensor:
        """valid_agent:[B,N] -> mask:[B,N] (True=복원 대상). 유효 에이전트 중 mask_ratio 비율을
        무작위 선택 — 최소 1개는 항상 유지(전부 마스킹되면 복원할 context가 없어짐)."""
        B, N = valid_agent.shape
        noise = torch.rand(B, N, device=valid_agent.device)
        noise = noise.masked_fill(~valid_agent, 2.0)  # 무효 에이전트는 절대 마스킹 대상으로 안 뽑히게
        n_valid = valid_agent.sum(dim=1)
        n_mask = (n_valid.float() * self.mask_ratio).long().clamp(max=(n_valid - 1).clamp(min=0))
        thresh_idx = n_mask.clamp(min=1) - 1
        sorted_noise, _ = noise.sort(dim=1)
        thresh = sorted_noise.gather(1, thresh_idx.unsqueeze(1)).squeeze(1)
        mask = (noise <= thresh.unsqueeze(1)) & valid_agent
        mask = mask & (n_mask.unsqueeze(1) > 0)
        return mask

    def forward(self, batch: dict) -> torch.Tensor:
        actor_in, pos, present, centers, head_vecs, valid_agent = _build_inputs(batch, self.N, self.W)
        B, N = valid_agent.shape

        tokens = self.encoder.embed_agents(actor_in, centers, head_vecs)
        mask = self._sample_mask(valid_agent)
        key_padding_mask = ~valid_agent
        encoded = self.encoder(tokens, key_padding_mask, mask=mask)

        x = self.decoder_embed(encoded)
        for blk in self.decoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.decoder_norm(x)

        recon = self.recon_head(x).view(B, N, self.W, 2)
        target = pos  # [B,N,W,2] (ego 앵커 프레임 기준 상대좌표, 이미 scene 정렬됨)

        reg_mask = present.clone()
        reg_mask[~mask] = False  # 마스킹된(복원 대상) + 실제 관측된 프레임만 loss에 포함
        if reg_mask.sum() == 0:
            return recon.sum() * 0.0  # 배치 전체에 마스킹 대상이 없는 극단 케이스 방지(그라드 흐름 유지)
        loss = F.l1_loss(recon[reg_mask], target[reg_mask])
        return loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TSEMMAEFinetune(nn.Module):
    """2단계: 사전학습 인코더(TSEMMAEEncoder) 재사용 + 분류 헤드. 마스킹 없이 전체 에이전트 사용."""

    def __init__(self, W: int = 10, K: int = 6, embed_dim: int = 64, encoder_depth: int = 4,
                 num_heads: int = 8, dropout: float = 0.1, num_classes: int = 3):
        super().__init__()
        self.W = W
        self.K = K
        self.N = 1 + K
        self.encoder = TSEMMAEEncoder(W, K, embed_dim, encoder_depth, num_heads, dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def load_pretrained_encoder(self, ckpt_path: str, map_location='cpu'):
        ckpt = torch.load(ckpt_path, map_location=map_location)
        state = ckpt['encoder'] if 'encoder' in ckpt else ckpt
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        return missing, unexpected

    def forward(self, batch: dict) -> torch.Tensor:
        actor_in, pos, present, centers, head_vecs, valid_agent = _build_inputs(batch, self.N, self.W)
        tokens = self.encoder.embed_agents(actor_in, centers, head_vecs)
        key_padding_mask = ~valid_agent
        encoded = self.encoder(tokens, key_padding_mask, mask=None)  # 미세조정: 마스킹 없음
        ego_embed = encoded[:, 0, :]
        return self.classifier(ego_embed)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
