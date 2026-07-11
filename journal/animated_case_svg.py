"""
[2026-07-09 v2] 케이스 애니메이션 SVG — 무한 반복 재생.
구성: (좌) 확대 지도 + 전체 사이트 미니맵(확대 영역 빨간 사각형), (우) 10D 전 채널 시계열 2열.
확률 막대는 제외(사용자 요청). 모든 애니메이션이 전체 주기의 keyframe(values/keyTimes)으로
정의되고 repeatCount="indefinite"라 자동으로 깨끗하게 무한 반복된다.
타임라인: 과거 W프레임(0.5s/frame) → 미래 H프레임 드로잉(0.5s/frame) → 1.5s 정지 → 반복.
출력: journal/paper_figs/qualitative/{class}_{rank}/animation.svg
"""
import base64
import glob
import io
import os
import sys

sys.path.insert(0, '/home/oem/TNA_research')
sys.path.insert(0, '/home/oem/TNA_research/journal')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from qualitative_cases import (
    CLASS_COLORS, CLASS_NAMES, DATA_DIR, OUT_DIR, compute_channels,
    density_background, select_cases,
)
from modules.data_manager_tsem import TSEMFutureStateDataset, _get_file_data

SEC_PER_FRAME = 0.5
HOLD_SEC = 1.5           # 마지막 상태 유지 시간 (그 후 처음부터 반복)
CHANNELS = ['v', 'a', 'j', 'omega', 'd_lat', 'kappa', 'delta_rho', 'tangential', 'pos_x', 'pos_z']
CH_LABELS = {
    'v': 'v (speed)', 'a': '|a| (accel)', 'j': 'j (jerk)', 'omega': 'omega (heading rate)',
    'd_lat': 'd_lat (lateral disp.)', 'kappa': 'kappa (curvature)', 'delta_rho': 'delta-rho (radius change)',
    'tangential': 'tangential', 'pos_x': 'position_x', 'pos_z': 'position_z',
}
Y_UNITS = {
    'v': 'speed', 'a': '|accel|', 'j': 'jerk', 'omega': 'rad/frame',
    'd_lat': 'm', 'kappa': 'rad/m', 'delta_rho': 'm/frame',
    'tangential': 'm/frame', 'pos_x': 'm (world)', 'pos_z': 'm (world)',
}


def render_png_b64(bg, extent, xlim=None, ylim=None, size=5, alpha=0.55,
                   style='density', lanes=None):
    """지도 배경 PNG(base64). style: density | lanes | overlay."""
    fig, ax = plt.subplots(figsize=(size, size), dpi=150)
    if style in ('density', 'overlay'):
        ax.imshow(bg, extent=extent, origin='lower', cmap='Greys', alpha=alpha, aspect='equal')
    if style in ('lanes', 'overlay') and lanes:
        for cl in lanes:
            ax.plot(cl[:, 0], cl[:, 1], '-', color='#3b6bb0', lw=1.0,
                    alpha=0.55 if style == 'overlay' else 0.8)
    if xlim:
        ax.set_xlim(xlim); ax.set_ylim(ylim)
    else:
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def kt(t, T):
    """keyTime 값 — [0.0005, 0.9995]로 클램프 (경계 중복 방지)."""
    return min(max(t / T, 0.0005), 0.9995)


def anim(attr, pairs, T, extra=''):
    """전체 주기 keyframe 애니메이션. pairs = [(time_sec, value), ...] (시각 오름차순).
    자동으로 t=0과 t=T 앵커를 보강하고 repeatCount=indefinite를 붙인다."""
    if pairs[0][0] > 0:
        pairs = [(0.0, pairs[0][1])] + pairs
    if pairs[-1][0] < T:
        pairs = pairs + [(T, pairs[-1][1])]
    keys, vals = [], []
    for i, (t, v) in enumerate(pairs):
        k = 0.0 if i == 0 else (1.0 if i == len(pairs) - 1 else kt(t, T))
        if keys and k <= keys[-1]:
            k = min(keys[-1] + 0.0005, 0.9995)
        keys.append(k)
        vals.append(v)
    return (f'<animate attributeName="{attr}" values="{";".join(str(v) for v in vals)}" '
            f'keyTimes="{";".join(f"{k:.4f}" for k in keys)}" dur="{T}s" '
            f'repeatCount="indefinite" {extra}/>')


def make_svg(case, style='density', lanes=None):
    (past, fut_path, ch, T_past, x_fut, gt, pred, conf, bg, extent,
     nbr_seqs, nbr_mask, nbr_fut) = case
    n_fut = len(x_fut)
    t_past_end = T_past * SEC_PER_FRAME
    t_fut_end = t_past_end + max(n_fut, 1) * SEC_PER_FRAME
    T = t_fut_end + HOLD_SEC                             # 전체 주기

    MAP = dict(x=52, y=50, w=520, h=520)
    allpts = np.vstack([past, fut_path]) if len(fut_path) else past
    cx0, cz0 = allpts.mean(0)
    span = max(28.0, np.abs(allpts - [cx0, cz0]).max() * 1.6)
    xlim, ylim = (cx0 - span, cx0 + span), (cz0 - span, cz0 + span)

    def mx(x):
        return MAP['x'] + (x - xlim[0]) / (2 * span) * MAP['w']

    def mz(z):
        return MAP['y'] + (ylim[1] - z) / (2 * span) * MAP['h']

    W_SVG, H_SVG = 1310, 720
    P = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_SVG}" height="{H_SVG}" '
         f'viewBox="0 0 {W_SVG} {H_SVG}" font-family="Helvetica, Arial, sans-serif">',
         f'<rect width="{W_SVG}" height="{H_SVG}" fill="white"/>']
    correct = 'CORRECT' if gt == pred else 'WRONG'
    P.append(f'<text x="20" y="30" font-size="17" font-weight="bold">GT: {gt}  |  Pred: {pred} '
             f'(p={conf:.2f})  [{correct}]</text>')

    # ── 확대 지도 배경 ──
    b64 = render_png_b64(bg, extent, xlim, ylim, style=style, lanes=lanes)
    P.append(f'<image x="{MAP["x"]}" y="{MAP["y"]}" width="{MAP["w"]}" height="{MAP["h"]}" '
             f'href="data:image/png;base64,{b64}"/>')
    P.append(f'<rect x="{MAP["x"]}" y="{MAP["y"]}" width="{MAP["w"]}" height="{MAP["h"]}" '
             f'fill="none" stroke="#888"/>')

    # ── 지도 x·y축 눈금 + 라벨 ──
    for frac in (0.0, 0.5, 1.0):
        # x축 눈금 (world x)
        wx = xlim[0] + frac * (xlim[1] - xlim[0])
        px = MAP['x'] + frac * MAP['w']
        P.append(f'<line x1="{px:.1f}" y1="{MAP["y"] + MAP["h"]}" x2="{px:.1f}" '
                 f'y2="{MAP["y"] + MAP["h"] + 4}" stroke="#555"/>')
        P.append(f'<text x="{px:.1f}" y="{MAP["y"] + MAP["h"] + 15}" font-size="10" fill="#333" '
                 f'text-anchor="middle">{wx:.0f}</text>')
        # y축 눈금 (world z) — SVG y는 아래로 증가하므로 상단=zlim 최대
        wz = ylim[1] - frac * (ylim[1] - ylim[0])
        py = MAP['y'] + frac * MAP['h']
        P.append(f'<line x1="{MAP["x"] - 4}" y1="{py:.1f}" x2="{MAP["x"]}" y2="{py:.1f}" stroke="#555"/>')
        P.append(f'<text x="{MAP["x"] - 7}" y="{py + 3:.1f}" font-size="10" fill="#333" '
                 f'text-anchor="end">{wz:.0f}</text>')
    P.append(f'<text x="{MAP["x"] + MAP["w"] / 2:.1f}" y="{MAP["y"] + MAP["h"] + 30}" font-size="12" '
             f'fill="#333" text-anchor="middle">position_x (m)</text>')
    ylab_mx, ylab_my = MAP['x'] - 34, MAP['y'] + MAP['h'] / 2
    P.append(f'<text x="{ylab_mx}" y="{ylab_my:.1f}" font-size="12" fill="#333" text-anchor="middle" '
             f'transform="rotate(-90 {ylab_mx} {ylab_my:.1f})">position_z (m)</text>')

    # ── 과거 궤적 점 (프레임 순서대로 등장, 주기마다 리셋) ──
    for i, (x, z) in enumerate(past):
        t0 = i * SEC_PER_FRAME
        P.append(f'<circle cx="{mx(x):.1f}" cy="{mz(z):.1f}" r="4" fill="#1f77b4">'
                 + anim('opacity', [(0, 0), (t0, 0), (t0 + 0.05, 0.85)], T) + '</circle>')

    # ── 이웃 차량: 과거 W프레임 동안의 움직임 (ego와 동기화, 주황 사각형 + 옅은 궤적 점) ──
    NB = 11  # 사각형 한 변
    for k in range(len(nbr_mask)):
        if nbr_mask[k] <= 0:
            continue
        seq = nbr_seqs[k]                                   # [T, 6]
        present = np.abs(seq).sum(1) > 0
        if not present.any():
            continue
        # 궤적 점 (프레임 순서대로 등장)
        for i in np.where(present)[0]:
            x, z = seq[i, 0], seq[i, 1]
            t0 = i * SEC_PER_FRAME
            P.append(f'<circle cx="{mx(x):.1f}" cy="{mz(z):.1f}" r="2.2" fill="orange">'
                     + anim('opacity', [(0, 0), (t0, 0), (t0 + 0.05, 0.45)], T) + '</circle>')
        # 이동 마커: 결측 프레임은 직전 위치 유지, 첫 등장 시점부터 표시.
        # 미래 구간(t→t+H)에서도 실제 위치를 따라 계속 이동 (nbr_fut, 원본 파일에서 id 복원).
        first = int(np.where(present)[0][0])
        last_pos = seq[first, :2]
        xp, yp = [], []
        for i in range(T_past):
            if present[i]:
                last_pos = seq[i, :2]
            xp.append((i * SEC_PER_FRAME, f'{mx(last_pos[0]) - NB / 2:.1f}'))
            yp.append((i * SEC_PER_FRAME, f'{mz(last_pos[1]) - NB / 2:.1f}'))
        t_past_end_k = T_past * SEC_PER_FRAME
        for d, (fx_, fz_) in nbr_fut.get(k, []):
            t_sec = t_past_end_k + (d - 1) * SEC_PER_FRAME
            xp.append((t_sec, f'{mx(fx_) - NB / 2:.1f}'))
            yp.append((t_sec, f'{mz(fz_) - NB / 2:.1f}'))
            # 미래 구간 이웃 궤적 점 (과거보다 더 옅게)
            P.append(f'<circle cx="{mx(fx_):.1f}" cy="{mz(fz_):.1f}" r="2.2" fill="orange">'
                     + anim('opacity', [(0, 0), (t_sec, 0), (t_sec + 0.05, 0.3)], T) + '</circle>')
        P.append(f'<rect width="{NB}" height="{NB}" fill="orange" stroke="black" stroke-width="1">'
                 + anim('x', xp, T) + anim('y', yp, T)
                 + anim('opacity', [(0, 0), (first * SEC_PER_FRAME, 0),
                                    (first * SEC_PER_FRAME + 0.05, 1)], T) + '</rect>')

    # ── 이동 차량(ego) 마커 ──
    cx_pairs = [(i * SEC_PER_FRAME, f'{mx(x):.1f}') for i, (x, _) in enumerate(past)]
    cy_pairs = [(i * SEC_PER_FRAME, f'{mz(z):.1f}') for i, (_, z) in enumerate(past)]
    P.append('<circle r="9" fill="#1f77b4" stroke="black" stroke-width="1.5">'
             + anim('cx', cx_pairs, T) + anim('cy', cy_pairs, T) + '</circle>')

    # ── 미래 경로 점선 드로잉 + 별표 ──
    if len(fut_path) > 1:
        pts = ' '.join(f'{mx(x):.1f},{mz(z):.1f}' for x, z in fut_path)
        seg = np.diff(np.array([[mx(x), mz(z)] for x, z in fut_path]), axis=0)
        L = float(np.sum(np.hypot(seg[:, 0], seg[:, 1]))) + 1
        col = CLASS_COLORS[gt]
        P.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="3" '
                 f'stroke-dasharray="{L:.0f}" opacity="0.9">'
                 + anim('stroke-dashoffset', [(0, f'{L:.0f}'), (t_past_end, f'{L:.0f}'),
                                              (t_fut_end, '0')], T) + '</polyline>')
        fx, fz = mx(fut_path[-1][0]), mz(fut_path[-1][1])
        P.append(f'<text x="{fx:.1f}" y="{fz + 8:.1f}" font-size="34" text-anchor="middle" fill="{col}">'
                 + anim('opacity', [(0, 0), (t_fut_end - 0.1, 0), (t_fut_end, 1)], T) + '&#9733;</text>')
        P.append(f'<text x="{fx:.1f}" y="{fz + 26:.1f}" font-size="12" text-anchor="middle" fill="{col}">'
                 + anim('opacity', [(0, 0), (t_fut_end - 0.1, 0), (t_fut_end, 1)], T)
                 + f't+H: GT={gt}</text>')

    # ── 시각 라벨 (한 번에 하나만 표시) ──
    total_frames = T_past + n_fut
    for i in range(total_frames):
        off = i - (T_past - 1)
        lab = f't{off:+d}' if off != 0 else 't (anchor)'
        t0, t1 = i * SEC_PER_FRAME, (i + 1) * SEC_PER_FRAME
        end = T if i == total_frames - 1 else t1          # 마지막 라벨은 hold 동안 유지
        P.append(f'<text x="{MAP["x"] + 8}" y="{MAP["y"] + MAP["h"] - 10}" font-size="15" '
                 f'font-weight="bold" fill="#333">'
                 + anim('opacity', [(0, 0), (t0, 0), (t0 + 0.02, 1), (end - 0.02, 1), (end, 0)], T)
                 + f'{lab}</text>')

    # ── 전체 사이트 미니맵 (확대 영역 빨간 사각형) ──
    MM = dict(w=150, h=150)
    MM['x'], MM['y'] = MAP['x'] + MAP['w'] - MM['w'] - 8, MAP['y'] + MAP['h'] - MM['h'] - 8
    mm64 = render_png_b64(bg, extent, size=3, alpha=0.85, style=style, lanes=lanes)
    P.append(f'<rect x="{MM["x"] - 2}" y="{MM["y"] - 16}" width="{MM["w"] + 4}" height="{MM["h"] + 20}" '
             f'fill="white" opacity="0.85" stroke="#555"/>')
    P.append(f'<text x="{MM["x"]}" y="{MM["y"] - 4}" font-size="10" fill="#333">full site '
             f'(zoom area in red)</text>')
    P.append(f'<image x="{MM["x"]}" y="{MM["y"]}" width="{MM["w"]}" height="{MM["h"]}" '
             f'href="data:image/png;base64,{mm64}" preserveAspectRatio="none"/>')

    def mmx(x):
        return MM['x'] + (x - extent[0]) / (extent[1] - extent[0]) * MM['w']

    def mmz(z):
        return MM['y'] + (extent[3] - z) / (extent[3] - extent[2]) * MM['h']

    rx, ry = mmx(xlim[0]), mmz(ylim[1])
    rw, rh = mmx(xlim[1]) - rx, mmz(ylim[0]) - ry
    P.append(f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{rw:.1f}" height="{rh:.1f}" '
             f'fill="none" stroke="red" stroke-width="1.6"/>')

    # ── 10채널 시계열 (2열 × 5행, 프레임 동기화 드로잉) ──
    CH = dict(x=600, y=58, w=290, h=78, gap=44, col_gap=62)
    t_all = np.arange(-T_past + 1, 1).tolist() + list(x_fut)
    for k, name in enumerate(CHANNELS):
        col, row = divmod(k, 5)
        gx0 = CH['x'] + col * (CH['w'] + CH['col_gap'])
        gy0 = CH['y'] + row * (CH['h'] + CH['gap'])
        vals = ch[name][:T_past + n_fut]
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        pad = (vmax - vmin) * 0.15 or 0.5
        vmin, vmax = vmin - pad, vmax + pad
        xmin, xmax = t_all[0], max(t_all[-1], 1)

        def gx(t):
            return gx0 + (t - xmin) / (xmax - xmin) * CH['w']

        def gy(v):
            return gy0 + (vmax - v) / (vmax - vmin) * CH['h']

        P.append(f'<rect x="{gx0}" y="{gy0}" width="{CH["w"]}" height="{CH["h"]}" '
                 f'fill="#fafafa" stroke="#ccc"/>')
        P.append(f'<text x="{gx0}" y="{gy0 - 5}" font-size="11.5" font-weight="bold">{CH_LABELS[name]}</text>')
        ax_x = gx(0)
        P.append(f'<line x1="{ax_x:.1f}" y1="{gy0}" x2="{ax_x:.1f}" y2="{gy0 + CH["h"]}" '
                 f'stroke="#999" stroke-dasharray="3 3"/>')
        # y축: 회전 라벨(단위) + min/max 눈금값 (좌측)
        ylab_x, ylab_y = gx0 - 26, gy0 + CH['h'] / 2
        P.append(f'<text x="{ylab_x:.1f}" y="{ylab_y:.1f}" font-size="9.5" fill="#333" '
                 f'text-anchor="middle" transform="rotate(-90 {ylab_x:.1f} {ylab_y:.1f})">'
                 f'{Y_UNITS[name]}</text>')
        P.append(f'<text x="{gx0 - 4}" y="{gy0 + 9}" font-size="8.5" fill="#666" '
                 f'text-anchor="end">{vmax:.2f}</text>')
        P.append(f'<text x="{gx0 - 4}" y="{gy0 + CH["h"]}" font-size="8.5" fill="#666" '
                 f'text-anchor="end">{vmin:.2f}</text>')
        # x축: 눈금값(-9, 0, +10) + 축 라벨
        for tv, tl in [(t_all[0], str(int(t_all[0]))), (0, '0'),
                       (t_all[-1], f'+{int(t_all[-1])}' if t_all[-1] > 0 else str(int(t_all[-1])))]:
            P.append(f'<text x="{gx(tv):.1f}" y="{gy0 + CH["h"] + 11}" font-size="8.5" fill="#666" '
                     f'text-anchor="middle">{tl}</text>')
        P.append(f'<text x="{gx0 + CH["w"] / 2:.1f}" y="{gy0 + CH["h"] + 23}" font-size="9.5" '
                 f'fill="#333" text-anchor="middle">frame offset from t</text>')
        col_f = CLASS_COLORS[gt]
        for i in range(len(t_all) - 1):
            x1, y1 = gx(t_all[i]), gy(vals[i])
            x2, y2 = gx(t_all[i + 1]), gy(vals[i + 1])
            t0 = (i + 1) * SEC_PER_FRAME
            style = ('stroke="#333" stroke-width="1.8"' if i < T_past - 1
                     else f'stroke="{col_f}" stroke-width="1.8" stroke-dasharray="5 3"')
            P.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {style}>'
                     + anim('opacity', [(0, 0), (t0, 0), (t0 + 0.05, 1)], T) + '</line>')

    # 하단 범례
    ly = H_SVG - 12
    P.append(f'<line x1="600" y1="{ly - 4}" x2="630" y2="{ly - 4}" stroke="#333" stroke-width="2"/>')
    P.append(f'<text x="636" y="{ly}" font-size="11">observed (past W)</text>')
    P.append(f'<line x1="770" y1="{ly - 4}" x2="800" y2="{ly - 4}" stroke="{CLASS_COLORS[gt]}" '
             f'stroke-width="2" stroke-dasharray="5 3"/>')
    P.append(f'<text x="806" y="{ly}" font-size="11">future t to t+H (GT={gt})</text>')
    P.append(f'<circle cx="985" cy="{ly - 4}" r="5" fill="#1f77b4" stroke="black"/>')
    P.append(f'<text x="994" y="{ly}" font-size="11">ego</text>')
    P.append(f'<rect x="1035" y="{ly - 9}" width="10" height="10" fill="orange" stroke="black"/>')
    P.append(f'<text x="1050" y="{ly}" font-size="11">neighbors</text>')

    P.append('</svg>')
    return '\n'.join(P)


def main():
    csvs = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    ds = TSEMFutureStateDataset(csvs, W=10, H=10, split='test', verbose=False)
    z = np.load(os.path.join(OUT_DIR, '_preds.npz'))
    probs, ys, ypers = z['probs'], z['ys'], z['ypers']
    picked = select_cases(probs, ys, ypers, ds)
    bg, extent = density_background()
    from qualitative_cases import lane_polylines_world
    LANES = lane_polylines_world()
    print(f'lane polylines: {len(LANES)}개')

    for c, idxs in picked.items():
        for rank, idx in enumerate(idxs, 1):
            sample = ds[idx]
            meta = sample['meta']
            csv_path = os.path.join(DATA_DIR, meta['csv_path'])
            fd = _get_file_data(csv_path)
            ego_id, wf, fut = meta['object_id'], meta['window_frames'], meta['future_frame']
            node_seq = sample['node_seq'].numpy()
            past = node_seq[np.abs(node_seq).sum(1) > 0][:, :2]
            frames_obj = fd.obj_frames[ego_id]
            fut_frames = [f for f in frames_obj if wf[-1] < f <= fut and ego_id in fd.frame_node.get(f, {})]
            fut_path = np.array([fd.frame_node[f][ego_id][:2] for f in fut_frames]) \
                if fut_frames else np.zeros((0, 2))
            fut_raw = np.array([fd.frame_node[f][ego_id] for f in fut_frames], dtype=np.float32) \
                if fut_frames else np.zeros((0, 6), dtype=np.float32)
            full = np.concatenate([node_seq, fut_raw]) if len(fut_raw) else node_seq
            ch = compute_channels(full)
            step = max(fd.frame_step, 1)
            x_fut = [(f - wf[-1]) / step for f in fut_frames]
            gt, pred = CLASS_NAMES[ys[idx]], CLASS_NAMES[probs[idx].argmax()]

            # 이웃 id 복원 (앵커 프레임 위치 대조) → 미래 구간 위치 수집
            nbr_seqs_np = sample['nbr_node_seqs'].numpy()
            nbr_mask_np = sample['nbr_mask'].numpy()
            anchor = wf[-1]
            anchor_nodes = fd.frame_node.get(anchor, {})
            nbr_fut = {}
            for kk in range(len(nbr_mask_np)):
                if nbr_mask_np[kk] <= 0:
                    continue
                pos_t = nbr_seqs_np[kk, -1, :2]
                # 캐시가 float16이라 위치가 ~0.01m 반올림됨 → 최근접 매칭(0.5m 이내)으로 id 복원
                cands = {o: float(np.hypot(*(v[:2] - pos_t)))
                         for o, v in anchor_nodes.items() if o != ego_id}
                oid_match = min(cands, key=cands.get) if cands else None
                if oid_match is None or cands[oid_match] > 0.5:
                    continue
                moves = []
                for f in fut_frames:
                    fn = fd.frame_node.get(f, {})
                    if oid_match in fn:
                        d = (f - anchor) / step
                        moves.append((d, (float(fn[oid_match][0]), float(fn[oid_match][1]))))
                if moves:
                    nbr_fut[kk] = moves

            case = (past, fut_path, ch, len(node_seq), x_fut, gt, pred,
                    float(probs[idx].max()), bg, extent,
                    nbr_seqs_np, nbr_mask_np, nbr_fut)
            for style in ('density', 'lanes', 'overlay'):
                svg = make_svg(case, style=style, lanes=LANES)
                out = os.path.join(OUT_DIR, f'{gt}_{rank}', f'animation_{style}.svg')
                with open(out, 'w') as f:
                    f.write(svg)
                print('saved:', out)


if __name__ == '__main__':
    main()
