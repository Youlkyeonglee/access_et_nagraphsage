"""
openDD 케이스 애니메이션 SVG — animated_case_svg.py(공업탑)의 openDD 버전 (2026-07-12)
==============================================================================
공업탑 버전은 학습된 모델(10차-2 체크포인트)의 GT vs Pred를 보여주지만, openDD는 아직
학습 파이프라인이 없다(§데이터 설계 openDD 참조) — 이 스크립트는 **GT-only**로,
opendd_position_memorization_entropy.py에서 이미 검증한 라벨 산출 로직(nearest-lane +
hold filter + B안 stop persistence + A안 LC window)을 그대로 재사용해 클래스별 대표 사례를
시각화한다. 모델 예측/확률 패널은 없다.

범위 제한: rdb1, 처음 N_RECORDINGS_SCAN개 레코딩에서 사례를 탐색(전체 153개가 아님).
채널도 공업탑의 semantic 8D(Δρ·접선 등, 로터리 중심 상수 필요)가 아니라 openDD 원시 필드로
계산 가능한 v(speed), dist_to_center(로터리 중심까지 거리)만 사용한다.

출력: journal/paper_figs/opendd/qualitative/{class}_{rank}/animation.svg
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/oem/TNA_research')
sys.path.insert(0, '/home/oem/TNA_research/journal')

from opendd_position_memorization_entropy import (  # noqa: E402
    FPS, H_FRAMES, LANE_DIST_THRESH_M, LANE_HOLD_FRAMES, ROOT, SITE,
    SPEED_STOP_THRESH, STOP_PERSIST_DELTA, VEHICLE_CLASSES,
    assign_lane_ids, hold_filter, load_lanes,
)

N_RECORDINGS_SCAN = 8       # 사례 탐색에 쓸 레코딩 수 (범위 제한)
W_FRAMES = round(FPS * 1.0)  # 과거 1초 (공업탑 W=10프레임@10fps와 동일 시간 지평)
SEC_PER_FRAME = 1.0 / FPS
HOLD_SEC = 1.5
N_PER_CLASS = 3
CLASS_NAMES = ('stop', 'lane_change', 'normal')
CLASS_COLORS = ['#2563eb', '#dc2626', '#16a34a']
OUT_DIR = '/home/oem/TNA_research/journal/paper_figs/opendd/qualitative'
CENTER_X, CENTER_Y = 619314.6, 5809163.6  # rdb1 lane centroid (§데이터 설계 1단계 분석과 동일 값)


def kt(t, T):
    return min(max(t / T, 0.0005), 0.9995)


def anim(attr, pairs, T, extra=''):
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


def find_cases():
    """레코딩을 스캔하며 클래스별 대표 (table, obj_id, anchor_idx) 사례를 수집."""
    lane_ids, polys = load_lanes()
    d = os.path.join(ROOT, f'opendd_v3-{SITE}', SITE)
    tcon = sqlite3.connect(os.path.join(d, f'trajectories_{SITE}_v3.sqlite'))
    tables = [r[0] for r in tcon.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    tables = tables[:N_RECORDINGS_SCAN]

    found = {c: [] for c in range(3)}
    for t in tables:
        if all(len(v) >= N_PER_CLASS for v in found.values()):
            break
        df = pd.read_sql(f"SELECT TIMESTAMP, OBJID, UTM_X, UTM_Y, V, CLASS FROM {t}", tcon)
        df = df[df['CLASS'].isin(VEHICLE_CLASSES)].sort_values(['OBJID', 'TIMESTAMP'])
        if df.empty:
            continue
        pts = df[['UTM_X', 'UTM_Y']].to_numpy()
        lane_assign, _ = assign_lane_ids(pts, lane_ids, polys, LANE_DIST_THRESH_M)
        df = df.assign(lane=lane_assign)

        for oid, g in df.groupby('OBJID', sort=False):
            g = g.reset_index(drop=True)
            n = len(g)
            if n < W_FRAMES + H_FRAMES + STOP_PERSIST_DELTA + 1:
                continue
            v = g['V'].to_numpy()
            lane = hold_filter(g['lane'].to_numpy(), LANE_HOLD_FRAMES)
            is_stop_instant = v <= SPEED_STOP_THRESH
            stop_persist = np.zeros(n, dtype=bool)
            for i in range(n):
                lo, hi = max(0, i - STOP_PERSIST_DELTA), min(n, i + STOP_PERSIST_DELTA + 1)
                stop_persist[i] = is_stop_instant[lo:hi].mean() >= 0.5
            valid_lane = lane >= 0
            lc_window = np.zeros(n, dtype=bool)
            for i in range(W_FRAMES, n - H_FRAMES):
                hi = min(n, i + 1 + H_FRAMES)
                seg, segv = lane[i:hi], valid_lane[i:hi]
                base = lane[i] if valid_lane[i] else -1
                lc_window[i] = ((seg != base) & segv & (base >= 0)).any()

            # 다양성: 같은 (recording, obj)는 클래스당 최대 1개 사례만 채택(연속 프레임 중복 방지),
            # 이 객체에서 각 클래스별로 가장 먼저 나오는 anchor 한 번씩만 사용.
            for c in range(3):
                if len(found[c]) >= N_PER_CLASS:
                    continue
                if (t, oid) in {(tt, oo) for tt, oo, _, _ in found[c]}:
                    continue
                for i in range(W_FRAMES, n - H_FRAMES):
                    label = 0 if stop_persist[i] else (1 if lc_window[i] else 2)
                    if label == c:
                        found[c].append((t, oid, i, g))
                        break
        print(f'  [{t}] 스캔 완료 — 현재 stop={len(found[0])} lc={len(found[1])} normal={len(found[2])}')
    tcon.close()
    return found, polys


def make_svg(g, anchor_i, gt_c):
    past = g[['UTM_X', 'UTM_Y']].to_numpy()[anchor_i - W_FRAMES + 1: anchor_i + 1]
    fut = g[['UTM_X', 'UTM_Y']].to_numpy()[anchor_i + 1: anchor_i + 1 + H_FRAMES]
    v_all = g['V'].to_numpy()[anchor_i - W_FRAMES + 1: anchor_i + 1 + H_FRAMES]
    dist_all = np.hypot(
        g['UTM_X'].to_numpy()[anchor_i - W_FRAMES + 1: anchor_i + 1 + H_FRAMES] - CENTER_X,
        g['UTM_Y'].to_numpy()[anchor_i - W_FRAMES + 1: anchor_i + 1 + H_FRAMES] - CENTER_Y,
    )
    n_fut = len(fut)
    t_past_end = W_FRAMES * SEC_PER_FRAME
    t_fut_end = t_past_end + max(n_fut, 1) * SEC_PER_FRAME
    T = t_fut_end + HOLD_SEC

    allpts = np.vstack([past, fut]) if len(fut) else past
    cx0, cy0 = allpts.mean(0)
    span = max(15.0, np.abs(allpts - [cx0, cy0]).max() * 1.6)
    xlim, ylim = (cx0 - span, cx0 + span), (cy0 - span, cy0 + span)
    MAP = dict(x=40, y=40, w=480, h=480)

    def mx(x):
        return MAP['x'] + (x - xlim[0]) / (2 * span) * MAP['w']

    def my(y):
        return MAP['y'] + (ylim[1] - y) / (2 * span) * MAP['h']

    W_SVG, H_SVG = 980, 560
    col = CLASS_COLORS[gt_c]
    P = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_SVG}" height="{H_SVG}" '
         f'viewBox="0 0 {W_SVG} {H_SVG}" font-family="Helvetica, Arial, sans-serif">',
         f'<rect width="{W_SVG}" height="{H_SVG}" fill="white"/>',
         f'<text x="20" y="24" font-size="16" font-weight="bold" fill="{col}">'
         f'openDD {SITE} — GT: {CLASS_NAMES[gt_c]} (no trained model, GT-only)</text>']

    P.append(f'<rect x="{MAP["x"]}" y="{MAP["y"]}" width="{MAP["w"]}" height="{MAP["h"]}" '
             f'fill="#f7f7f7" stroke="#888"/>')

    for i, (x, y) in enumerate(past):
        t0 = i * SEC_PER_FRAME
        P.append(f'<circle cx="{mx(x):.1f}" cy="{my(y):.1f}" r="4" fill="#1f77b4">'
                 + anim('opacity', [(0, 0), (t0, 0), (t0 + 0.02, 0.85)], T) + '</circle>')

    cx_pairs = [(i * SEC_PER_FRAME, f'{mx(x):.1f}') for i, (x, _) in enumerate(past)]
    cy_pairs = [(i * SEC_PER_FRAME, f'{my(y):.1f}') for i, (_, y) in enumerate(past)]
    P.append('<circle r="8" fill="#1f77b4" stroke="black" stroke-width="1.5">'
             + anim('cx', cx_pairs, T) + anim('cy', cy_pairs, T) + '</circle>')

    if len(fut) > 1:
        pts = ' '.join(f'{mx(x):.1f},{my(y):.1f}' for x, y in fut)
        seg = np.diff(np.array([[mx(x), my(y)] for x, y in fut]), axis=0)
        L = float(np.sum(np.hypot(seg[:, 0], seg[:, 1]))) + 1
        P.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="3" '
                 f'stroke-dasharray="{L:.0f}" opacity="0.9">'
                 + anim('stroke-dashoffset', [(0, f'{L:.0f}'), (t_past_end, f'{L:.0f}'),
                                              (t_fut_end, '0')], T) + '</polyline>')
        fx, fy = mx(fut[-1][0]), my(fut[-1][1])
        P.append(f'<text x="{fx:.1f}" y="{fy + 8:.1f}" font-size="28" text-anchor="middle" fill="{col}">'
                 + anim('opacity', [(0, 0), (t_fut_end - 0.1, 0), (t_fut_end, 1)], T) + '&#9733;</text>')

    P.append(f'<text x="{MAP["x"] + MAP["w"] / 2:.1f}" y="{MAP["y"] + MAP["h"] + 20}" font-size="12" '
             f'text-anchor="middle" fill="#333">UTM_X (m)</text>')
    ylab_x, ylab_y = MAP['x'] - 24, MAP['y'] + MAP['h'] / 2
    P.append(f'<text x="{ylab_x}" y="{ylab_y:.1f}" font-size="12" fill="#333" text-anchor="middle" '
             f'transform="rotate(-90 {ylab_x} {ylab_y:.1f})">UTM_Y (m)</text>')

    # 시각 라벨
    total = W_FRAMES + n_fut
    for i in range(total):
        off = i - (W_FRAMES - 1)
        lab = f't{off:+d}' if off != 0 else 't (anchor)'
        t0, t1 = i * SEC_PER_FRAME, (i + 1) * SEC_PER_FRAME
        end = T if i == total - 1 else t1
        P.append(f'<text x="{MAP["x"] + 8}" y="{MAP["y"] + MAP["h"] - 10}" font-size="13" '
                 f'font-weight="bold" fill="#333">'
                 + anim('opacity', [(0, 0), (t0, 0), (t0 + 0.02, 1), (end - 0.02, 1), (end, 0)], T)
                 + f'{lab}</text>')

    # ── 채널 시계열: speed, dist_to_center ──
    t_all = (np.arange(-W_FRAMES + 1, 1 + n_fut)).tolist()
    channels = [('v (speed)', v_all, 'm/s'), ('dist_to_center', dist_all, 'm')]
    CH = dict(x=560, y=60, w=380, h=180, gap=60)
    for k, (name, vals, unit) in enumerate(channels):
        gy0 = CH['y'] + k * (CH['h'] + CH['gap'])
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        pad = (vmax - vmin) * 0.15 or 0.5
        vmin, vmax = vmin - pad, vmax + pad
        xmin_, xmax_ = t_all[0], max(t_all[-1], 1)

        def gx(t, xmin_=xmin_, xmax_=xmax_):
            return CH['x'] + (t - xmin_) / (xmax_ - xmin_) * CH['w']

        def gy(v, gy0=gy0, vmin=vmin, vmax=vmax):
            return gy0 + (vmax - v) / (vmax - vmin) * CH['h']

        P.append(f'<rect x="{CH["x"]}" y="{gy0}" width="{CH["w"]}" height="{CH["h"]}" fill="#fafafa" stroke="#ccc"/>')
        P.append(f'<text x="{CH["x"]}" y="{gy0 - 6}" font-size="13" font-weight="bold">{name} ({unit})</text>')
        ax_x = gx(0)
        P.append(f'<line x1="{ax_x:.1f}" y1="{gy0}" x2="{ax_x:.1f}" y2="{gy0+CH["h"]}" stroke="#999" stroke-dasharray="3 3"/>')
        P.append(f'<text x="{CH["x"]-4}" y="{gy0+9}" font-size="9" fill="#666" text-anchor="end">{vmax:.2f}</text>')
        P.append(f'<text x="{CH["x"]-4}" y="{gy0+CH["h"]}" font-size="9" fill="#666" text-anchor="end">{vmin:.2f}</text>')
        for i in range(len(t_all) - 1):
            x1, y1 = gx(t_all[i]), gy(vals[i])
            x2, y2 = gx(t_all[i + 1]), gy(vals[i + 1])
            t0 = (i + 1) * SEC_PER_FRAME
            style = ('stroke="#333" stroke-width="1.8"' if i < W_FRAMES - 1
                     else f'stroke="{col}" stroke-width="1.8" stroke-dasharray="5 3"')
            P.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {style}>'
                     + anim('opacity', [(0, 0), (t0, 0), (t0 + 0.02, 1)], T) + '</line>')

    P.append('</svg>')
    return '\n'.join(P)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'openDD {SITE} 케이스 탐색 중 (레코딩 {N_RECORDINGS_SCAN}개, 범위 제한)...')
    found, _ = find_cases()
    for c, cases in found.items():
        for rank, (t, oid, anchor_i, g) in enumerate(cases, 1):
            svg = make_svg(g, anchor_i, c)
            case_dir = os.path.join(OUT_DIR, f'{CLASS_NAMES[c]}_{rank}')
            os.makedirs(case_dir, exist_ok=True)
            out = os.path.join(case_dir, 'animation.svg')
            with open(out, 'w') as f:
                f.write(svg)
            print(f'saved: {out}  (recording={t}, obj={oid}, anchor_frame_idx={anchor_i})')
    for c in range(3):
        if len(found[c]) < N_PER_CLASS:
            print(f'경고: {CLASS_NAMES[c]} 클래스 사례가 {len(found[c])}/{N_PER_CLASS}개만 발견됨 '
                  f'(N_RECORDINGS_SCAN을 늘리면 더 찾을 수 있음)')


if __name__ == '__main__':
    main()
