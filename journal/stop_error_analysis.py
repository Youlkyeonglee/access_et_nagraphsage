"""
[신규] stop 과다예측(normal→stop 오분류) 원인 가설 검증.
가설: normal→stop 오분류는 무작위가 아니라 ego 속도(및 인접 밀도)가 낮은 저속 구간에
     체계적으로 몰려있다 (로터리 서행/정체 구간이 정지와 혼동됨).
사용: python journal/stop_error_analysis.py tsem_sage_w10_h10_labelA_multiloss_polar
"""
import sys, glob, os
import numpy as np
import torch

sys.path.insert(0, '/home/oem/TNA_research')
from modules.data_manager_tsem import build_tsem_dataloaders
from modules.tsem_instant_label import CLASS_NAMES, STOP, LANE_CHANGE, NORMAL
from models.tsem_sage import TSEMSAGE, TSEMSemanticOnly

DATA_DIR = '/home/oem/data/TII_data/'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def csvs(cfg):
    ds = cfg['data']['dataset']
    sub = 'Gongeoptap' if ds == 'gongeoptap' else ''
    return sorted(glob.glob(os.path.join(DATA_DIR, sub, '*.csv')))


def load_model(exp):
    ck_path = f'/home/oem/TNA_research/checkpoints/tsem/{exp}/best.pt'
    c = torch.load(ck_path, map_location=dev, weights_only=False)
    cfg = c['cfg']
    m = cfg['model']
    cls = TSEMSemanticOnly if m.get('name') == 'tsem_semantic_only' else TSEMSAGE
    model = cls(
        node_dim=6, edge_dim=5,
        hidden_dim=m['hidden_dim'], d_e=m['d_e'], T=cfg['tsem']['W'],
        encoder_type=m['encoder_type'], use_attention=m['use_attention'],
        use_2hop=m['use_2hop'], use_spatial=m['use_spatial'],
        num_classes=m['num_classes'], dropout=m['dropout'],
        decomp_kernel=m['decomp_kernel'], decomp_learnable=m['decomp_learnable'],
    ).to(dev)
    model.load_state_dict(c['model'])
    model.eval()
    return model, cfg


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else 'tsem_sage_w10_h10_labelA_multiloss_polar'
    model, cfg = load_model(exp)
    gcfg, tcfg = cfg['graph'], cfg['tsem']

    _, _, test_loader = build_tsem_dataloaders(
        csv_files=csvs(cfg), W=tcfg['W'], H=tcfg['H'],
        radius=gcfg['radius'], K_max=gcfg['K_max'], K_max2=gcfg['K_max2'],
        batch_size=4096, train_ratio=cfg['data']['train_ratio'],
        val_ratio=cfg['data']['val_ratio'], num_workers=6, verbose=False,
    )

    preds, ys, speeds, accels, n_nbr, rel_speed, min_dist = [], [], [], [], [], [], []
    with torch.no_grad():
        for b in test_loader:
            bg = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            logits = model(bg)
            preds.append(logits.argmax(-1).cpu().numpy())
            ys.append(b['y'].numpy())

            node_seq = b['node_seq']           # [B,T,6] cols: x,z,speed,dx,dz,accel
            nbr_mask = b['nbr_mask']            # [B,K]
            edge_seqs = b['edge_seqs']          # [B,K,T,5] rel_speed,rel_accel,rel_dx,rel_dz,dist

            speeds.append(node_seq[:, -1, 2].numpy())
            accels.append(node_seq[:, -1, 5].numpy())
            n_nbr.append(nbr_mask.sum(-1).numpy())

            last_edge = edge_seqs[:, :, -1, :]                  # [B,K,5]
            mask = nbr_mask.unsqueeze(-1)                        # [B,K,1]
            denom = nbr_mask.sum(-1).clamp(min=1)                # [B]
            mean_rel_speed = (last_edge[..., 0].abs() * nbr_mask).sum(-1) / denom
            rel_speed.append(mean_rel_speed.numpy())

            dist = last_edge[..., 4].masked_fill(nbr_mask == 0, float('inf'))
            min_d = dist.min(dim=-1).values
            min_d[torch.isinf(min_d)] = float('nan')
            min_dist.append(min_d.numpy())

    P = np.concatenate(preds); Y = np.concatenate(ys)
    speed = np.concatenate(speeds); accel = np.concatenate(accels)
    n_nbr = np.concatenate(n_nbr); rel_speed = np.concatenate(rel_speed)
    min_dist = np.concatenate(min_dist)

    print(f'[{exp}] test N={len(Y):,}  Acc={(P==Y).mean():.4f}')

    # 핵심 관심군: 정답=normal인데 stop으로 오분류된 샘플 vs 정답=normal이고 맞춘 샘플
    is_normal = (Y == NORMAL)
    fp_stop = is_normal & (P == STOP)          # normal → stop 오분류 (가설 대상)
    tp_normal = is_normal & (P == NORMAL)      # normal 정분류

    print(f'\n정답=normal 전체: {is_normal.sum():,}  '
          f'→stop 오분류: {fp_stop.sum():,} ({fp_stop.sum()/is_normal.sum()*100:.1f}%)  '
          f'정분류: {tp_normal.sum():,} ({tp_normal.sum()/is_normal.sum()*100:.1f}%)')

    def stats(name, arr):
        a, b = arr[fp_stop], arr[tp_normal]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        print(f'  {name:14s}  FP(normal→stop) mean={a.mean():7.3f} median={np.median(a):7.3f}  |  '
              f'TP(normal 정분류) mean={b.mean():7.3f} median={np.median(b):7.3f}')

    print('\n--- 프록시별 분포 비교 (가설: FP-stop 군은 저속·저밀도에 몰려있다) ---')
    stats('ego_speed', speed)
    stats('ego_accel(abs)', np.abs(accel))
    stats('n_neighbors', n_nbr)
    stats('mean|rel_speed|', rel_speed)
    stats('min_dist(m)', min_dist)

    # 속도 구간별 normal 오분류율(=stop으로 샐 확률) — heteroscedasticity 직접 확인
    print('\n--- ego_speed 구간별 normal 표본의 stop 오분류율 ---')
    bins = [0, 0.5, 1, 2, 3, 5, 8, 100]
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = is_normal & (speed >= lo) & (speed < hi)
        n = sel.sum()
        if n == 0:
            continue
        fp_rate = (P[sel] == STOP).mean()
        print(f'  speed∈[{lo:>5},{hi:>5})  n={n:7,}  stop오분류율={fp_rate*100:5.1f}%')

    # 참고: 전체 confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y, P, labels=[STOP, LANE_CHANGE, NORMAL])
    print('\n혼동행렬 (행=정답, 열=예측) [stop, lane_change, normal]:')
    print(cm)


if __name__ == '__main__':
    main()
