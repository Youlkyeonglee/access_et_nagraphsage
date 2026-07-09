"""
TSEM-SAGE — Semantic Temporal + NAGraphSAGE Spatial @ t
========================================================
Stage A: SemanticDerivation (raw 6D → semantic 6D)
Stage B: DecompTemporalEncoder per node (ego + neighbors)
Stage C: ETSAGELayer @ anchor t — edge는 마지막 프레임만 (edge-temporal 없음)
Stage D: 3-class state(t+H) classifier

ET-NAGraphSAGE(`et_nagraphsage.py`)와 별도 파일. 공간 레이어만 import 재사용.
"""
import torch
import torch.nn as nn

from .decomp_encoder import DecompTemporalEncoder
from .et_nagraphsage import ETSAGELayer
from .tsem_semantic_derivation import SemanticDerivation


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
      raw_append: str = 'none',
      num_classes: int = 3,
      dropout: float = 0.3,
      decomp_kernel: int = 5,
      decomp_learnable: bool = True,
  ):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.d_e = d_e
    self.T = T
    self.use_2hop = use_2hop
    self.use_spatial = use_spatial
    self.use_semantic = use_semantic

    # 10차 (2026-07-08): NAGraphSAGE-adapted 베이스라인 — use_semantic=False면 Stage A(의미 분해)를
    # 건너뛰고 raw 6D를 그대로 시계열 인코더에 넣는다(§구조 "베이스라인" 페이지 정의: "과거 W raw +
    # 프레임 t 이웃 GNN"). 공간 집계(ETSAGELayer, Stage C)는 그대로 유지 — 그래야 "semantic 분해 자체의
    # 기여"만 격리해서 볼 수 있다(공간 유무는 TSEMSemanticOnly가 이미 격리).
    if use_semantic:
      self.semantic = SemanticDerivation()
      sem_dim = SemanticDerivation.SEM_DIM
    else:
      self.semantic = None
      sem_dim = node_dim

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
    # spatial @ t: edge 마지막 프레임만 → 단일 벡터 인코딩
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

  def _encode_nodes(self, raw_seq: torch.Tensor) -> torch.Tensor:
    """raw [..., T, 6] → hidden [..., d]"""
    sem = self.semantic(raw_seq) if self.use_semantic else raw_seq
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
        return {
            'logits': logits, 'logits_temporal': logits,
            'logits_spatial': None, 'beta_1hop': None, 'nbr_mask': nbr_mask,
        }
      return logits

    e1 = self.edge_proj(edge_seqs[:, :, -1, :])
    e1 = e1 * nbr_mask.unsqueeze(-1)

    if self.use_2hop and nbr2_node_seqs is not None and nbr2_node_seqs.shape[2] > 0:
      K2 = nbr2_node_seqs.shape[2]
      h_nbr2 = self._encode_nodes(nbr2_node_seqs)
      h_nbr2 = h_nbr2 * nbr2_mask.unsqueeze(-1)
      e2 = self.edge_proj(nbr2_edge_seqs[:, :, :, -1, :])
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
    return {
        'logits': self.classifier(h_out),
        'logits_temporal': self.classifier_temporal(h_ego),
        'logits_spatial': self.classifier_spatial(h_N),
        'beta_1hop': beta,
        'nbr_mask': nbr_mask,
    }

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
