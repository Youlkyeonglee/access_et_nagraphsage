"""
TSEM-SAGE — Semantic Temporal + NAGraphSAGE Spatial @ t
========================================================
Stage A: SemanticDerivation (raw 6D → semantic 6D)
Stage B: DecompTemporalEncoder per node (ego + neighbors)
Stage C: ETSAGELayer @ anchor t — edge는 기본적으로 마지막 프레임만 사용
         (opt-in `edge_temporal=True`면 edge_seqs 전체 W프레임을 TemporalEncoder로 인코딩,
         2026-07-12 hypothesis 1 저비용 검증용 — 기본값은 여전히 anchor 프레임만)
Stage D: 3-class state(t+H) classifier

ET-NAGraphSAGE(`et_nagraphsage.py`)와 별도 파일. 공간 레이어만 import 재사용.
"""
import torch
import torch.nn as nn

from .decomp_encoder import DecompTemporalEncoder
from .et_nagraphsage import ETSAGELayer
from .temporal_encoder import TemporalEncoder
from .tsem_semantic_derivation import LearnableSemanticResidual, SemanticDerivation


class TSEMSAGE(nn.Module):
  def __init__(
      self,
      node_dim: int = 6,
      edge_dim: int = 5,
      hidden_dim: int = 128,
      d_e: int = 32,
      T: int = 10,
      encoder_type: str = 'gru',
      use_attention: bool = True,
      use_2hop: bool = True,
      use_spatial: bool = True,
      use_semantic: bool = True,
      semantic_variant: str = 'full',
      raw_append: str = 'none',
      num_classes: int = 3,
      dropout: float = 0.3,
      decomp_kernel: int = 5,
      decomp_learnable: bool = True,
      use_speed_head: bool = False,
      edge_temporal: bool = False,
      learnable_residual_dim: int = 0,
  ):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.d_e = d_e
    self.T = T
    self.use_2hop = use_2hop
    self.use_spatial = use_spatial
    self.use_semantic = use_semantic
    # ① edge-temporal (2026-07-12, hypothesis 1 저비용 검증 — 학습은 아직 미실행):
    # 기존엔 edge_proj(Linear)가 anchor(t=W) 프레임만 봤는데, edge_temporal=True면
    # et_nagraphsage.py와 동일한 TemporalEncoder(GRU)로 edge_seqs 전체 W프레임을 인코딩한다.
    # 구조 순서(시간 먼저 → anchor에서 spatial 1회)는 그대로 — edge 표현만 시간 인지로 바뀐다.
    # comparison/README.md "semantic 8D 채널별 site-specific 여부 재점검" 및
    # docs/edge_temporal_report.html §5(2×2 절제: 약한 표현 base에서만 엣지가 도움)의 재검증용.
    self.edge_temporal = edge_temporal

    # 10차 (2026-07-08): NAGraphSAGE-adapted 베이스라인 — use_semantic=False면 Stage A(의미 분해)를
    # 건너뛰고 raw 6D를 그대로 시계열 인코더에 넣는다(§구조 "베이스라인" 페이지 정의: "과거 W raw +
    # 프레임 t 이웃 GNN"). 공간 집계(ETSAGELayer, Stage C)는 그대로 유지 — 그래야 "semantic 분해 자체의
    # 기여"만 격리해서 볼 수 있다(공간 유무는 TSEMSemanticOnly가 이미 격리).
    # semantic_variant='invariant' (2026-07-11): Δρ·접선(로터리 중심 world 상수 참조, site-specific)
    # 을 제외한 6D(v,a,j,ω,d_lat,κ)만 반환 — comparison/ 5개 baseline과의 apples-to-apples 비교용.
    # raw_append='none'과 함께 써야 완전 invariant(position도 안 씀)가 된다.
    if use_semantic:
      self.semantic = SemanticDerivation(variant=semantic_variant)
      sem_dim = self.semantic.SEM_DIM
    else:
      self.semantic = None
      sem_dim = node_dim

    # ② 학습 가능한 잔차 브랜치 (2026-07-12, §실험현황 "20260712 추가 실험계획" ②): 고정
    # 손공학 SemanticDerivation이 버릴 수 있는 미세한 궤적 정보를 raw speed·accel(위치·절대방향
    # 무관, invariant-safe)에서 소형 MLP로 보강. opt-in — 0이면 self.residual=None, 기존
    # 체크포인트 strict load에 영향 없음.
    self.learnable_residual_dim = learnable_residual_dim
    if learnable_residual_dim > 0:
      self.residual = LearnableSemanticResidual(residual_dim=learnable_residual_dim, dropout=dropout)
      sem_dim += learnable_residual_dim
    else:
      self.residual = None

    # 10차-2/-3 (2026-07-08): 10차에서 "raw 6D > semantic 8D"가 나온 원인을 격리하기 위한 입력 조합.
    # semantic 8D에 raw 채널을 추가로 concat — 'pos'면 position_x/z(→10D), 'pos_dir'이면
    # position_x/z + direction_x/z(→12D). NODE_COLS 순서 [pos_x, pos_z, speed, dir_x, dir_z, accel] 기준.
    # 'polar' (2026-07-09, 학교서버 방법1 실험): 절대좌표 대신 **구조 참조 좌표** [ρ, sinθ, cosθ]를
    # 추가(→11D) — 로터리 중심 기준 반경·각도라 다른 로터리에서는 중심만 재측정하면 같은 의미로
    # 재사용 가능(raw 절대좌표의 위치 이득을 이식 가능한 형태로 재표현하는 가설 검증용).
    assert raw_append in ('none', 'pos', 'pos_dir', 'polar')
    self.raw_append = raw_append
    self._raw_idx = {'none': [], 'pos': [0, 1], 'pos_dir': [0, 1, 3, 4], 'polar': []}[raw_append]
    sem_dim += {'none': 0, 'pos': 2, 'pos_dir': 4, 'polar': 3}[raw_append]

    enc_kw = dict(
        encoder_type=encoder_type,
        use_attention=use_attention,
        kernel=decomp_kernel,
        learnable=decomp_learnable,
        dropout=dropout,
    )
    self.node_encoder = DecompTemporalEncoder(sem_dim, hidden_dim, **enc_kw)
    if edge_temporal:
      # edge_seqs 전체 W프레임을 시간 인코딩(GRU) — node_encoder와 같은 encoder_type/use_attention
      self.edge_encoder = TemporalEncoder(
          edge_dim, d_e, encoder_type=encoder_type, use_attention=use_attention, dropout=dropout)
    else:
      # spatial @ t: edge 마지막 프레임만 → 단일 벡터 인코딩 (기본, 기존 동작 그대로)
      self.edge_proj = nn.Sequential(
          nn.Linear(edge_dim, d_e),
          nn.ReLU(),
          nn.Dropout(dropout),
      )

    if use_2hop:
      self.layer_2hop = ETSAGELayer(hidden_dim, hidden_dim, d_e, dropout)
    self.layer_1hop = ETSAGELayer(hidden_dim, hidden_dim, d_e, dropout)
    self.classifier = nn.Linear(hidden_dim, num_classes)
    # 보조 헤드: temporal(h_ego)·spatial(h_N) 각각 단독으로도 분류 가능해야 한다는
    # 명제 H1·H2를 훈련 중 직접 감독하기 위함 (2026-07-07, 단일 최종 loss가 표현학습을
    # 왜곡하는 문제 완화).
    self.classifier_temporal = nn.Linear(hidden_dim, num_classes)
    self.classifier_spatial = nn.Linear(hidden_dim, num_classes)
    # 미래 속도 회귀 보조헤드 (2026-07-09, 제안1): v(t+H)/10 을 회귀 — "출발 임박 normal"
    # (v(t)≈0에서 1초 뒤 출발, recall 19.5%) 경계에 연속 신호로 gradient를 공급하기 위함.
    # opt-in(config model.use_speed_head) — False면 모듈 자체가 없어서 기존 체크포인트
    # strict load에 영향 없음.
    self.head_speed = nn.Linear(hidden_dim, 1) if use_speed_head else None

  def _encode_nodes(self, raw_seq: torch.Tensor) -> torch.Tensor:
    """raw [..., T, 6] → hidden [..., d]"""
    sem = self.semantic(raw_seq) if self.use_semantic else raw_seq
    if self.residual is not None:
      sem = torch.cat([sem, self.residual(raw_seq)], dim=-1)
    if self._raw_idx:
      sem = torch.cat([sem, raw_seq[..., self._raw_idx]], dim=-1)
    elif self.raw_append == 'polar':
      # 구조 참조 좌표 [ρ, sinθ, cosθ] — 로터리 중심(SemanticDerivation 상수) 기준.
      # 결측(all-zero) 프레임은 0 유지 (0 패딩 위치를 실좌표로 오인 방지).
      present = (raw_seq.abs().sum(dim=-1, keepdim=True) > 0).float()
      cx = raw_seq[..., 0:1] - SemanticDerivation.CENTER_X
      cz = raw_seq[..., 1:2] - SemanticDerivation.CENTER_Z
      rho = torch.sqrt(cx * cx + cz * cz + 1e-8)
      theta = torch.atan2(cz, cx)
      polar = torch.cat([rho, torch.sin(theta), torch.cos(theta)], dim=-1) * present
      sem = torch.cat([sem, polar], dim=-1)
    shape = sem.shape[:-2]
    T = sem.shape[-2]
    flat = sem.reshape(-1, T, sem.shape[-1])
    h = self.node_encoder(flat)
    return h.reshape(*shape, -1)

  def _encode_edges(self, edge_seq: torch.Tensor) -> torch.Tensor:
    """edge_temporal=False(기본): anchor(t=W) 프레임만 Linear로 투영, 기존 동작 그대로.
    edge_temporal=True: edge [..., T, edge_dim] 전체를 TemporalEncoder(GRU)로 인코딩."""
    if not self.edge_temporal:
      return self.edge_proj(edge_seq[..., -1, :])
    shape = edge_seq.shape[:-2]
    T = edge_seq.shape[-2]
    flat = edge_seq.reshape(-1, T, edge_seq.shape[-1])
    h = self.edge_encoder(flat)
    return h.reshape(*shape, -1)

  def forward(self, batch: dict, return_aux: bool = False):
    """return_aux=False(기본): 기존과 동일하게 최종 logits 텐서 하나만 반환 (평가·추론용, 하위호환).
    return_aux=True: 학습용 — {'logits','logits_temporal','logits_spatial','beta_1hop','nbr_mask'} dict 반환."""
    node_seq = batch['node_seq']
    nbr_node_seqs = batch['nbr_node_seqs']
    edge_seqs = batch['edge_seqs']
    nbr_mask = batch['nbr_mask']
    nbr2_node_seqs = batch['nbr2_node_seqs']
    nbr2_edge_seqs = batch.get('nbr2_edge_seqs')
    nbr2_mask = batch['nbr2_mask']

    B, K1, T, _ = nbr_node_seqs.shape

    h_ego = self._encode_nodes(node_seq)
    h_nbr = self._encode_nodes(nbr_node_seqs)
    h_nbr = h_nbr * nbr_mask.unsqueeze(-1)

    if not self.use_spatial:
      logits = self.classifier(h_ego)
      if return_aux:
        out = {
            'logits': logits, 'logits_temporal': logits,
            'logits_spatial': None, 'beta_1hop': None, 'nbr_mask': nbr_mask,
        }
        if self.head_speed is not None:
          out['v_pred'] = self.head_speed(h_ego)
        return out
      return logits

    e1 = self._encode_edges(edge_seqs)
    e1 = e1 * nbr_mask.unsqueeze(-1)

    if self.use_2hop and nbr2_node_seqs is not None and nbr2_node_seqs.shape[2] > 0:
      K2 = nbr2_node_seqs.shape[2]
      h_nbr2 = self._encode_nodes(nbr2_node_seqs)
      h_nbr2 = h_nbr2 * nbr2_mask.unsqueeze(-1)
      e2 = self._encode_edges(nbr2_edge_seqs)
      e2 = e2 * nbr2_mask.unsqueeze(-1)

      h_nbr_updated = self.layer_2hop(
          h_ego=h_nbr.reshape(B * K1, -1),
          h_nbr=h_nbr2.reshape(B * K1, K2, -1),
          e_temp=e2.reshape(B * K1, K2, -1),
          nbr_mask=nbr2_mask.reshape(B * K1, K2),
      ).reshape(B, K1, -1)
      h_nbr_updated = h_nbr_updated * nbr_mask.unsqueeze(-1)
    else:
      h_nbr_updated = h_nbr

    if not return_aux:
      h_out = self.layer_1hop(h_ego, h_nbr_updated, e1, nbr_mask)
      return self.classifier(h_out)

    h_out, h_N, beta = self.layer_1hop(h_ego, h_nbr_updated, e1, nbr_mask, return_extra=True)
    out = {
        'logits': self.classifier(h_out),
        'logits_temporal': self.classifier_temporal(h_ego),
        'logits_spatial': self.classifier_spatial(h_N),
        'beta_1hop': beta,
        'nbr_mask': nbr_mask,
    }
    if self.head_speed is not None:
      out['v_pred'] = self.head_speed(h_out)
    return out

  def count_parameters(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TSEMSemanticOnly(TSEMSAGE):
  """Spatial off ablation — ego semantic temporal만."""

  def __init__(self, **kwargs):
    kwargs['use_spatial'] = False
    kwargs['use_2hop'] = False
    super().__init__(**kwargs)


class TSEMNAGraphSAGEAdapted(TSEMSAGE):
  """10차 — NAGraphSAGE-adapted 베이스라인. Semantic 분해(Stage A) off, 공간 집계(Stage C)는
  유지 — raw 6D를 그대로 시계열 인코더에 넣고 이웃 GNN을 그대로 적용한다."""

  def __init__(self, **kwargs):
    kwargs['use_semantic'] = False
    super().__init__(**kwargs)


class TSEMSAGEInterleaved(nn.Module):
  """2026-07-12 hypothesis 1(공간 집계 시점) 검증용 — spatial-temporal 순서를 뒤집은 변형.

  기존 TSEMSAGE: "시간 먼저 요약(GRU, 노드별 독립) → anchor 시점에 딱 1번 ETSAGELayer".
  이 클래스   : "매 프레임 t마다 먼저 ETSAGELayer(ego+1-hop 이웃, t시점)로 공간인지 벡터 s_t를
                만들고, 시퀀스 s_1..s_W를 GRU(시간 인코더)에 넣어 anchor 표현을 얻는다."
  HiVT의 AAEncoder(매 timestep spatial)→TemporalEncoder, QCNet의 시간·공간 attention 교대
  반복과 원리적으로 동일 — 6D invariant(76.21%)가 HiVT(77.48%)·QCNet(77.97%)에 뒤지는 이유가
  "공간 집계를 anchor 1회만 하는 구조" 때문인지를 직접 검증한다(comparison/README.md
  §semantic 8D 채널별 site-specific 여부 재점검, 도식은 Obsidian
  `Drawing 2026-07-12 spatial-temporal 검증.excalidraw.md` 참조).

  주의(근사): SemanticDerivation은 결측(0-padding) 프레임에서 v,a,j,ω,d_lat,κ,Δρ,접선이 전부
  0에 가깝게 나오도록 이미 설계돼 있어(valid_diff 게이팅) 별도 프레임별 마스킹 없이도 결측
  프레임이 과도한 신호를 만들진 않는다 — 그러나 TSEMSAGE(anchor 1회, 항상 유효한 프레임만 사용)
  와 달리 이 클래스는 결측 프레임에서도 ETSAGELayer를 호출하므로 완전히 동일한 보장은 아니다.
  """

  def __init__(
      self,
      node_dim: int = 6,
      edge_dim: int = 5,
      hidden_dim: int = 128,
      d_e: int = 32,
      T: int = 10,
      semantic_variant: str = 'full',
      raw_append: str = 'none',
      use_2hop: bool = True,
      num_classes: int = 3,
      dropout: float = 0.3,
      use_speed_head: bool = False,
  ):
    super().__init__()
    self.T = T
    self.use_2hop = use_2hop

    self.semantic = SemanticDerivation(variant=semantic_variant)
    sem_dim = self.semantic.SEM_DIM

    assert raw_append in ('none', 'pos', 'pos_dir', 'polar')
    self.raw_append = raw_append
    self._raw_idx = {'none': [], 'pos': [0, 1], 'pos_dir': [0, 1, 3, 4], 'polar': []}[raw_append]
    sem_dim += {'none': 0, 'pos': 2, 'pos_dir': 4, 'polar': 3}[raw_append]

    # 매 timestep 공유(shared)로 적용되는 입력 투영 — TSEMSAGE의 node_encoder(GRU, 시퀀스 전체
    # 요약)와 달리 여기선 프레임 하나씩 hidden_dim으로 투영만 한다(시간 요약은 뒤쪽 GRU가 담당).
    self.node_proj = nn.Sequential(
        nn.Linear(sem_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
    self.edge_proj = nn.Sequential(
        nn.Linear(edge_dim, d_e), nn.ReLU(), nn.Dropout(dropout))

    if use_2hop:
      self.layer_2hop = ETSAGELayer(hidden_dim, hidden_dim, d_e, dropout)
    self.layer_1hop = ETSAGELayer(hidden_dim, hidden_dim, d_e, dropout)

    # 프레임별 공간인지 벡터 s_1..s_W를 시간축으로 요약 — 프로젝트 확정 결론(GRU 채택)을 따름.
    self.temporal_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    self.classifier = nn.Linear(hidden_dim, num_classes)
    self.head_speed = nn.Linear(hidden_dim, 1) if use_speed_head else None

  def _sem_feat(self, raw_seq: torch.Tensor) -> torch.Tensor:
    """raw [..., T, 6] -> semantic(+raw_append) [..., T, sem_dim] — 프레임별 concat, 시간 요약 없음."""
    sem = self.semantic(raw_seq)
    if self._raw_idx:
      sem = torch.cat([sem, raw_seq[..., self._raw_idx]], dim=-1)
    elif self.raw_append == 'polar':
      present = (raw_seq.abs().sum(dim=-1, keepdim=True) > 0).float()
      cx = raw_seq[..., 0:1] - SemanticDerivation.CENTER_X
      cz = raw_seq[..., 1:2] - SemanticDerivation.CENTER_Z
      rho = torch.sqrt(cx * cx + cz * cz + 1e-8)
      theta = torch.atan2(cz, cx)
      polar = torch.cat([rho, torch.sin(theta), torch.cos(theta)], dim=-1) * present
      sem = torch.cat([sem, polar], dim=-1)
    return sem

  def forward(self, batch: dict, return_aux: bool = False):
    node_seq = batch['node_seq']              # [B,T,6]
    nbr_node_seqs = batch['nbr_node_seqs']    # [B,K1,T,6]
    edge_seqs = batch['edge_seqs']             # [B,K1,T,5]
    nbr_mask = batch['nbr_mask']               # [B,K1]
    nbr2_node_seqs = batch['nbr2_node_seqs']  # [B,K1,K2,T,6]
    nbr2_edge_seqs = batch.get('nbr2_edge_seqs')
    nbr2_mask = batch['nbr2_mask']             # [B,K1,K2]

    B, K1, T, _ = nbr_node_seqs.shape
    use_2hop = self.use_2hop and nbr2_node_seqs is not None and nbr2_node_seqs.shape[2] > 0
    K2 = nbr2_node_seqs.shape[2] if use_2hop else 0

    sem_ego = self._sem_feat(node_seq)              # [B,T,sem_dim]
    sem_nbr = self._sem_feat(nbr_node_seqs)          # [B,K1,T,sem_dim]
    if use_2hop:
      sem_nbr2 = self._sem_feat(nbr2_node_seqs)      # [B,K1,K2,T,sem_dim]

    s_list = []
    for t in range(T):
      h_ego_t = self.node_proj(sem_ego[:, t])                 # [B,hidden]
      h_nbr_t = self.node_proj(sem_nbr[:, :, t]) * nbr_mask.unsqueeze(-1)  # [B,K1,hidden]
      e1_t = self.edge_proj(edge_seqs[:, :, t]) * nbr_mask.unsqueeze(-1)   # [B,K1,d_e]

      if use_2hop:
        h_nbr2_t = self.node_proj(sem_nbr2[:, :, :, t]) * nbr2_mask.unsqueeze(-1)  # [B,K1,K2,hidden]
        e2_t = self.edge_proj(nbr2_edge_seqs[:, :, :, t]) * nbr2_mask.unsqueeze(-1)  # [B,K1,K2,d_e]
        h_nbr_t = self.layer_2hop(
            h_ego=h_nbr_t.reshape(B * K1, -1),
            h_nbr=h_nbr2_t.reshape(B * K1, K2, -1),
            e_temp=e2_t.reshape(B * K1, K2, -1),
            nbr_mask=nbr2_mask.reshape(B * K1, K2),
        ).reshape(B, K1, -1) * nbr_mask.unsqueeze(-1)

      s_t = self.layer_1hop(h_ego_t, h_nbr_t, e1_t, nbr_mask)  # [B,hidden] — spatial @ t
      s_list.append(s_t)

    s_seq = torch.stack(s_list, dim=1)          # [B,T,hidden]
    _, h_n = self.temporal_encoder(s_seq)       # h_n: [1,B,hidden]
    h_final = h_n.squeeze(0)                    # [B,hidden] — anchor 표현

    logits = self.classifier(h_final)
    if not return_aux:
      return logits
    out = {
        'logits': logits, 'logits_temporal': logits, 'logits_spatial': logits,
        'beta_1hop': None, 'nbr_mask': nbr_mask,
    }
    if self.head_speed is not None:
      out['v_pred'] = self.head_speed(h_final)
    return out

  def count_parameters(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
