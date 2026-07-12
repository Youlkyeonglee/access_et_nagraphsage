"""
[신규 2026-07-09] 논문용 정성적 사례 시각화 — 클래스당 5개, 개별 고해상도 PNG.
각 이미지: (좌) 지도 위 과거 W궤적 + 앵커 + 이웃 + 실제 미래 경로(t→t+H),
          (우) 10D 입력 전 채널 시계열 + 3-class 예측 확률 막대.
모델: 10차-2 (semantic+position 10D, 최종 채택). 라벨/텍스트 전부 영문(논문용).
사례 선별: 정답 + 고확신 우선, 가능하면 상태 전이 사례(persist가 틀리는, 즉 진짜 예측이 필요한 경우).
출력: journal/paper_figs/qualitative/{class}_{rank}.png (dpi=300)
"""
import glob
import os
import sys

sys.path.insert(0, '/home/oem/TNA_research')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models.tsem_sage import TSEMSAGE
from models.tsem_semantic_derivation import SemanticDerivation
from modules.data_manager_tsem import TSEMFutureStateDataset, _collate_fn, _get_file_data
from modules.tsem_instant_label import CLASS_NAMES

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = '/home/oem/TNA_research/journal/paper_figs/gongeoptap/qualitative'
CKPT = '/home/oem/TNA_research/checkpoints/tsem/tsem_sage_w10_h10_sem_pos_10d/best.pt'
DATA_DIR = '/home/oem/data/TII_data/Gongeoptap'
N_PER_CLASS = 5
CLASS_COLORS = {'stop': '#d62728', 'lane_change': '#9467bd', 'normal': '#2ca02c'}
CHANNELS = ['v', 'a', 'j', 'omega', 'd_lat', 'kappa', 'delta_rho', 'tangential', 'pos_x', 'pos_z']
CH_LABELS = {
    'v': 'v (speed)', 'a': 'a (accel)', 'j': 'j (jerk)', 'omega': 'omega (heading rate)',
    'd_lat': 'd_lat (lateral disp.)', 'kappa': 'kappa (curvature)',
    'delta_rho': 'delta-rho (radius change)', 'tangential': 'tangential (d-theta x rho)',
    'pos_x': 'position_x (world)', 'pos_z': 'position_z (world)',
}
Y_LABELS = {
    'v': 'speed', 'a': '|acceleration|', 'j': 'delta |a| per frame',
    'omega': 'rad / frame', 'd_lat': 'meters', 'kappa': 'rad / m',
    'delta_rho': 'm / frame', 'tangential': 'm / frame',
    'pos_x': 'meters (world x)', 'pos_z': 'meters (world z)',
}


def load_model():
    ck = torch.load(CKPT, map_location=DEV, weights_only=False)
    cfg = ck['cfg']
    m = cfg['model']
    model = TSEMSAGE(
        node_dim=6, edge_dim=5, hidden_dim=m['hidden_dim'], d_e=m['d_e'], T=cfg['tsem']['W'],
        encoder_type=m['encoder_type'], use_attention=m['use_attention'], use_2hop=m['use_2hop'],
        use_spatial=m['use_spatial'], num_classes=m['num_classes'], dropout=m['dropout'],
        decomp_kernel=m['decomp_kernel'], decomp_learnable=m['decomp_learnable'],
        raw_append=m.get('raw_append', 'pos'),
    ).to(DEV)
    model.load_state_dict(ck['model'])
    model.eval()
    return model


def density_background():
    csvs = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))[:4]
    df = pd.concat(pd.read_csv(f, usecols=['position_x', 'position_z']) for f in csvs)
    H, xe, ze = np.histogram2d(df.position_x, df.position_z, bins=400)
    return np.log1p(H).T, [xe[0], xe[-1], ze[0], ze[-1]]


def lane_polylines_world():
    """HD 차선 annotation(road_data/lane_annotations.json)을 world 좌표 폴리라인 목록으로 반환.
    호모그래피는 첫 CSV의 bbox↔position 쌍으로 fit (journal/map_features.py 재사용)."""
    from map_features import build_map_world
    csv = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))[0]
    df = pd.read_csv(csv)
    _, _, lanes = build_map_world(df)
    return [l['cw'] for l in lanes]      # list of [M, 2] world polylines


LANE_COLOR = '#3b6bb0'   # 차선 폴리라인 색 (밀도 회색과 구분되는 청회색)
LANES = []               # main()에서 lane_polylines_world()로 채움


def draw_map_background(ax, style, bg, extent, lanes):
    """style: 'density' | 'lanes' | 'overlay' — 지도 배경을 ax에 그림."""
    if style in ('density', 'overlay'):
        ax.imshow(bg, extent=extent, origin='lower', cmap='Greys', alpha=0.55, aspect='equal')
    if style in ('lanes', 'overlay'):
        for cl in lanes:
            ax.plot(cl[:, 0], cl[:, 1], '-', color=LANE_COLOR,
                    lw=1.0, alpha=0.55 if style == 'overlay' else 0.8, zorder=1)


@torch.no_grad()
def collect_predictions(model, ds):
    from torch.utils.data import DataLoader
    # GPU가 sweep 실험들로 점유 중이라 배치를 작게 (여유 ~1.3GB 내에서 동작)
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=_collate_fn, num_workers=6)
    probs, ys, ypers = [], [], []
    for b in loader:
        bg = {k: v.to(DEV) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        p = torch.softmax(model(bg), dim=-1).cpu()
        probs.append(p)
        ys.append(b['y'])
        ypers.append(b['y_persist'])
    return torch.cat(probs).numpy(), torch.cat(ys).numpy(), torch.cat(ypers).numpy()


def select_cases(probs, ys, ypers, ds):
    """클래스별 5개: 정답 + 고확신, 파일·차량 중복 배제.
    - stop/LC: 전이 사례(persist가 틀림) 우선 — '지금 상태 유지'로는 못 맞히는 진짜 예측이 인상적.
    - normal (2026-07-09 수정): 순항(steady, persist=normal) + 고속(v(t) 상위) 우선 —
      전이 우선으로 뽑으면 '정지→출발' 케이스만 나와서 거의 안 움직이는 차량들만 보이는
      선별 편향이 생김(사용자 지적). 대표적 normal은 '계속 잘 달리는 차'가 맞음.
      다양성을 위해 전이(출발) 사례도 마지막 1개로 포함."""
    preds = probs.argmax(1)
    conf = probs.max(1)
    # 창 전체 이동량(첫 유효 프레임 → 마지막 프레임 변위) — normal 순항 사례 선별용.
    # 앵커 순간 속도(v(t))로 거르면 "창 안에서 막 출발해 앵커에만 빠른" 차가 통과하는 함정이 있어
    # (2026-07-09 사용자 지적) 창 전체 변위 ≥ 8m 로 판정한다.
    disp = None
    if ds._cache is not None:
        ns = np.asarray(ds._cache['node_seq'], dtype=np.float32)          # [N, T, 6]
        present = np.abs(ns).sum(-1) > 0                                   # [N, T]
        first_i = present.argmax(1)                                        # 첫 유효 프레임
        idx_n = np.arange(len(ns))
        p_first = ns[idx_n, first_i, :2]
        p_last = ns[:, -1, :2]                                             # 앵커는 항상 유효
        disp = np.hypot(*(p_last - p_first).T)
    out = {}
    for c in range(3):
        cand = np.where((ys == c) & (preds == c))[0]
        transition = cand[ypers[cand] != c]
        steady = cand[ypers[cand] == c]
        if c == 2 and disp is not None:  # NORMAL: 창 내내 실제 주행한(변위 큰) 순항 사례만
            cruising = steady[disp[steady] >= 8.0]
            ordered = list(cruising[np.argsort(-conf[cruising])]) + \
                      list(steady[np.argsort(-disp[steady])])
        else:
            ordered = list(transition[np.argsort(-conf[transition])]) + \
                      list(steady[np.argsort(-conf[steady])])
        picked, seen = [], set()
        for i in ordered:
            csv_path, ego_id, wf, _ = ds.samples[i]
            key = (csv_path, ego_id)
            if key in seen:
                continue
            seen.add(key)
            picked.append(i)
            if len(picked) == N_PER_CLASS:
                break
        out[c] = picked
    return out


def compute_channels(node_seq_np):
    """raw [T,6] → 10채널 dict (semantic 8 + position 2)."""
    x = torch.from_numpy(node_seq_np).unsqueeze(0)
    sem = SemanticDerivation()(x)[0].numpy()          # [T, 8]
    return {
        'v': sem[:, 0], 'a': sem[:, 1], 'j': sem[:, 2], 'omega': sem[:, 3],
        'd_lat': sem[:, 4], 'kappa': sem[:, 5], 'delta_rho': sem[:, 6], 'tangential': sem[:, 7],
        'pos_x': node_seq_np[:, 0], 'pos_z': node_seq_np[:, 1],
    }


def render_case(idx, ds, probs, ys, ypers, bg, extent, rank):
    sample = ds[idx]
    meta = sample['meta']
    csv_path_base, ego_id = meta['csv_path'], meta['object_id']
    wf, fut = meta['window_frames'], meta['future_frame']
    gt, pred = CLASS_NAMES[ys[idx]], CLASS_NAMES[probs[idx].argmax()]
    persist = CLASS_NAMES[ypers[idx]]

    csv_path = os.path.join(DATA_DIR, csv_path_base)
    fd = _get_file_data(csv_path)

    node_seq = sample['node_seq'].numpy()             # [T, 6]
    past = node_seq[np.abs(node_seq).sum(1) > 0][:, :2]
    # 실제 미래 경로 t→t+H
    frames_obj = fd.obj_frames[ego_id]
    fut_frames = [f for f in frames_obj if wf[-1] <= f <= fut]
    fut_path = np.array([fd.frame_node[f][ego_id][:2] for f in fut_frames if ego_id in fd.frame_node.get(f, {})])
    # 이웃 위치 @ t
    nbr = sample['nbr_node_seqs'].numpy()[:, -1, :2]
    mask = sample['nbr_mask'].numpy() > 0
    nbr = nbr[mask]

    # ── 채널 계산: 과거 W + 미래 H를 이어붙여 한 번에 계산 (차분 채널의 경계 연속성 유지) ──
    fut_frames_only = [f for f in frames_obj if wf[-1] < f <= fut and f in fd.frame_node
                       and ego_id in fd.frame_node[f]]
    fut_raw = np.array([fd.frame_node[f][ego_id] for f in fut_frames_only], dtype=np.float32) \
        if fut_frames_only else np.zeros((0, 6), dtype=np.float32)
    full_seq = np.concatenate([node_seq, fut_raw], axis=0) if len(fut_raw) else node_seq
    ch = compute_channels(full_seq)
    T_past = len(node_seq)
    step = max(fd.frame_step, 1)
    x_past = np.arange(-T_past + 1, 1)
    x_fut = np.array([(f - wf[-1]) / step for f in fut_frames_only])

    case_dir = os.path.join(OUT_DIR, f'{gt}_{rank}')
    os.makedirs(case_dir, exist_ok=True)
    correct = 'CORRECT' if gt == pred else 'WRONG'

    # ── ① 지도 (단독 이미지, 배경 3버전: density / lanes / overlay) ──
    allpts = np.vstack([past, fut_path]) if len(fut_path) else past
    cx0, cz0 = allpts.mean(0)
    span = max(28.0, np.abs(allpts - [cx0, cz0]).max() * 1.7)
    for style in ('density', 'lanes', 'overlay'):
        fig, ax = plt.subplots(figsize=(8, 8))
        draw_map_background(ax, style, bg, extent, LANES)
        ax.plot(past[:, 0], past[:, 1], '-', color='#1f77b4', lw=2.5, label=f'past W={len(past)} frames')
        ax.scatter(past[:, 0], past[:, 1], c=np.linspace(0.3, 1, len(past)), cmap='Blues', s=32, zorder=3)
        if len(fut_path) > 1:
            ax.plot(fut_path[:, 0], fut_path[:, 1], '--', color=CLASS_COLORS[gt], lw=2.4,
                    label='actual future path (t to t+H)')
            ax.scatter(*fut_path[-1], marker='*', s=300, color=CLASS_COLORS[gt], zorder=5,
                       edgecolors='k', linewidths=0.6, label=f't+H: GT={gt}')
        ax.scatter(*past[-1], marker='o', s=150, color='#1f77b4', edgecolors='k', zorder=5, label='anchor t')
        if len(nbr):
            ax.scatter(nbr[:, 0], nbr[:, 1], marker='s', s=60, color='orange',
                       edgecolors='k', zorder=4, label=f'neighbors at t ({len(nbr)})')
        ax.set_xlim(cx0 - span, cx0 + span)
        ax.set_ylim(cz0 - span, cz0 + span)
        ax.set_aspect('equal')
        ax.set_xlabel('position_x (m)', fontsize=12)
        ax.set_ylabel('position_z (m)', fontsize=12)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax.set_title(f'GT: {gt} | Pred: {pred} (p={probs[idx].max():.2f}) [{correct}]', fontsize=13)

        # 전체 맵 미니맵(인셋): 같은 배경 스타일 + 확대 영역 빨간 사각형
        axins = ax.inset_axes([0.66, 0.02, 0.32, 0.32])
        draw_map_background(axins, style, bg, extent, LANES)
        from matplotlib.patches import Rectangle
        axins.add_patch(Rectangle((cx0 - span, cz0 - span), 2 * span, 2 * span,
                                  fill=False, edgecolor='red', lw=1.8))
        axins.plot(past[:, 0], past[:, 1], '-', color='#1f77b4', lw=1.0)
        axins.set_xticks([]); axins.set_yticks([])
        axins.set_title('full site (zoom area in red)', fontsize=8)
        for spine in axins.spines.values():
            spine.set_edgecolor('#555')

        fig.savefig(os.path.join(case_dir, f'00_map_{style}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # ── ② 채널 10개 (각각 단독 이미지) — 과거(검정 실선) + 예측 구간(클래스 색 점선) ──
    for k, name in enumerate(CHANNELS, 1):
        fig, axc = plt.subplots(figsize=(4.5, 3.2))
        past_vals = ch[name][:T_past]
        axc.plot(x_past, past_vals, '-o', ms=4, lw=1.8, color='#333',
                 label='observed (past W)')
        if len(x_fut):
            fut_vals = ch[name][T_past:]
            # 앵커(t) 값에서 이어지도록 경계점을 포함해 그림
            axc.plot(np.concatenate([[0], x_fut]), np.concatenate([[past_vals[-1]], fut_vals]),
                     '--o', ms=4, lw=1.8, color=CLASS_COLORS[gt], alpha=0.9,
                     label=f'future t to t+H (GT={gt})')
        axc.axvline(0, color='#999', lw=0.9, ls=':')
        axc.axhline(0, color='#bbb', lw=0.8)
        axc.set_title(CH_LABELS[name], fontsize=12)
        axc.set_xlabel('frame offset from t', fontsize=10)
        axc.set_ylabel(Y_LABELS[name], fontsize=10)
        axc.tick_params(labelsize=9)
        axc.legend(fontsize=7.5, framealpha=0.85, loc='best')
        fig.savefig(os.path.join(case_dir, f'{k:02d}_{name}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # ── ③ 예측 확률 막대 (단독 이미지) ──
    fig, axp = plt.subplots(figsize=(6, 3.2))
    colors = [CLASS_COLORS[n] for n in CLASS_NAMES]
    axp.barh(range(3), probs[idx], color=colors, alpha=0.85)
    axp.set_yticks(range(3))
    axp.set_yticklabels(list(CLASS_NAMES), fontsize=11)
    for i, v in enumerate(probs[idx]):
        axp.text(min(v + 0.02, 0.86), i, f'{v:.3f}', va='center', fontsize=11)
    axp.set_xlim(0, 1)
    axp.set_title('predicted class probabilities (softmax)', fontsize=12)
    axp.tick_params(labelsize=10)
    fig.savefig(os.path.join(case_dir, '11_probs.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ── ④ 케이스 메타데이터 (텍스트 파일) ──
    with open(os.path.join(case_dir, 'summary.txt'), 'w') as f:
        f.write(
            f'GT: {gt}\nPred: {pred} (p={probs[idx].max():.3f}) [{correct}]\n'
            f'persist(state@t): {persist}\n'
            f'transition case: {"YES (persist wrong)" if persist != gt else "no (steady state)"}\n'
            f'file: {csv_path_base}\nobject_id: {ego_id}\n'
            f'window W: {len(node_seq)} frames (past only)\nhorizon H: {meta["H"]} frames\n'
            f'anchor frame: {meta["frame"]}\nfuture frame: {fut}\n'
            f'probs: ' + ', '.join(f'{n}={v:.3f}' for n, v in zip(CLASS_NAMES, probs[idx])) + '\n'
            f'model: 10D final (semantic 8 + position 2)\n'
        )
    return case_dir


def main():
    csvs = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    ds = TSEMFutureStateDataset(csvs, W=10, H=10, split='test', verbose=False)
    print(f'test samples: {len(ds):,}')
    pred_cache = os.path.join(OUT_DIR, '_preds.npz')
    if os.path.exists(pred_cache):
        z = np.load(pred_cache)
        probs, ys, ypers = z['probs'], z['ys'], z['ypers']
        print('예측 캐시 로드:', pred_cache)
    else:
        model = load_model()
        probs, ys, ypers = collect_predictions(model, ds)
        os.makedirs(OUT_DIR, exist_ok=True)
        np.savez(pred_cache, probs=probs, ys=ys, ypers=ypers)
    picked = select_cases(probs, ys, ypers, ds)
    bg, extent = density_background()
    global LANES
    LANES = lane_polylines_world()
    print(f'lane polylines: {len(LANES)}개')
    for c, idxs in picked.items():
        for rank, idx in enumerate(idxs, 1):
            out = render_case(idx, ds, probs, ys, ypers, bg, extent, rank)
            print('saved:', out)


if __name__ == '__main__':
    main()
