"""
ET-NAGraphSAGE-Decomp — 주파수 분해 시간인코더 적용 (기여 ①).
기존 et_nagraphsage.py 불변. base 모델을 서브클래싱해 node/edge 인코더만
DecompTemporalEncoder로 교체 (forward·공간집계는 base 그대로 재사용).
"""
from .et_nagraphsage import ETNAGraphSAGE
from .decomp_encoder import DecompTemporalEncoder


class ETNAGraphSAGEDecomp(ETNAGraphSAGE):
    def __init__(self, node_dim=6, edge_dim=5, hidden_dim=128, d_e=32, T=10,
                 encoder_type='gru', use_attention=True, use_2hop=True,
                 num_classes=3, dropout=0.3, temporal_target='both',
                 decomp_kernel=5, decomp_learnable=True):
        super().__init__(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim,
                         d_e=d_e, T=T, encoder_type=encoder_type,
                         use_attention=use_attention, use_2hop=use_2hop,
                         num_classes=num_classes, dropout=dropout,
                         temporal_target=temporal_target)
        # 인코더만 분해형으로 교체 (나머지 구조·forward 동일)
        self.node_encoder = DecompTemporalEncoder(
            node_dim, hidden_dim, encoder_type=encoder_type, use_attention=use_attention,
            kernel=decomp_kernel, learnable=decomp_learnable)
        self.edge_encoder = DecompTemporalEncoder(
            edge_dim, d_e, encoder_type=encoder_type, use_attention=use_attention,
            kernel=decomp_kernel, learnable=decomp_learnable)
