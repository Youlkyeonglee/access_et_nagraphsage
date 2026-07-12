"""
위치 암기 분석 ④ — 추론 시 position 채널 고정(freeze) ablation (2026-07-12)
========================================================================
이미 학습된 10차-2(semantic 8D + raw position 2D = 10D) 체크포인트를 그대로 두고,
"테스트 시점에만" 입력의 위치 관련 채널을 상수로 바꿔치기해서 정확도 하락폭을 잰다.
재학습 없음 — best.pt 그대로 forward만 다르게 한다.

세 가지 개입 강도를 비교한다:
  (A) raw_append 채널만 고정 — 10D 중 마지막 2D(raw position_x,z)만 로터리 평균좌표로 치환.
      semantic 8D 자체(Δρ·접선 포함)는 원래 좌표로 정상 계산 — "명시적 raw position 채널"만 죽인다.
  (B) raw 위치 자체를 Stage A 이전 단계에서 고정 — ego+이웃+2-hop 전부의 raw pos_x,pos_z를
      로터리 평균좌표로 치환한 뒤 semantic 8D를 계산 → raw_append 2D는 물론 Δρ·접선·d_lat까지
      전부 그 여파로 무너진다(diff 기반이라 상수 위치에서는 자동으로 0이 됨). "위치가 모델
      추론에 관여하는 모든 경로"를 차단하는 가장 포괄적인 개입.
  (C) 대조군 — semantic 8D 중 딱 한 채널만(v,a,j,ω,d_lat,κ,Δρ,접선 각각) 그 채널의 train 평균으로
      고정. "채널 하나를 없앴을 때" 통상적인 하락폭이 얼마인지 분포를 만들어, (A)/(B)가 유별나게
      큰 하락인지 비교한다.

고정 상수: 사용자 제안대로 "로터리 전체 평균 좌표"(train split anchor 위치 평균)를 사용.
semantic 채널 고정값도 각 채널의 train split 평균.

사용: python journal/position_freeze_ablation.py [--gpus 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, '/home/oem/TNA_research')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

import yaml  # noqa: E402

from modules.data_manager_tsem import build_tsem_dataloaders  # noqa: E402
from modules.tsem_eval import confusion_matrix, metrics_from_confusion  # noqa: E402
from modules.tsem_instant_label import CLASS_NAMES  # noqa: E402
from train_tsem import build_model, get_csv_files  # noqa: E402

CONFIG_PATH = '/home/oem/TNA_research/configs/tsem_sage_10th_2.yaml'
CKPT_PATH = '/home/oem/TNA_research/checkpoints/tsem/tsem_sage_w10_h10_sem_pos_10d/best.pt'
OUT_DIR = '/home/oem/TNA_research/journal/paper_figs/gongeoptap'
SEM_CHANNEL_NAMES = ['v', 'a', 'j', 'omega', 'd_lat', 'kappa', 'drho', 'tangent']
SITE_SPECIFIC = {'drho', 'tangent'}  # comparison/README.md 재점검 결론


@torch.no_grad()
def compute_train_means(model, loader, device):
    """semantic 8D 채널별 평균 + anchor raw position 평균(ego, present 프레임만)을
    train split을 한 번 순회하며 계산."""
    sem_sum = torch.zeros(8, dtype=torch.float64)
    sem_cnt = torch.zeros(8, dtype=torch.float64)
    pos_sum = torch.zeros(2, dtype=torch.float64)
    pos_cnt = 0.0
    model.eval()
    for batch in loader:
        node_seq = batch['node_seq'].to(device)  # [B,T,6]
        sem = model.semantic(node_seq)  # [B,T,8]
        present = (node_seq.abs().sum(dim=-1) > 0)  # [B,T]
        sem_sum += sem[present].double().sum(dim=0).cpu()
        sem_cnt += present.sum().item()
        anchor_pos = node_seq[:, -1, 0:2]  # [B,2] — anchor(t) 항상 present(ego 정의상)
        pos_sum += anchor_pos.double().sum(dim=0).cpu()
        pos_cnt += anchor_pos.shape[0]
    sem_mean = (sem_sum / sem_cnt.clamp(min=1)).float()
    pos_mean = (pos_sum / max(pos_cnt, 1)).float()
    return sem_mean, pos_mean


def make_encode_nodes(freeze_raw_append_const=None, freeze_sem_idx=None, freeze_sem_const=None):
    """TSEMSAGE._encode_nodes 몽키패치 팩토리 — raw_append='pos' 체크포인트 전용
    (원본 로직에서 polar 분기는 이 체크포인트에 해당 없어 생략)."""

    def _encode_nodes(self, raw_seq: torch.Tensor) -> torch.Tensor:
        sem = self.semantic(raw_seq) if self.use_semantic else raw_seq
        if freeze_sem_idx is not None:
            present = (raw_seq.abs().sum(dim=-1) > 0).float()
            sem = sem.clone()
            sem[..., freeze_sem_idx] = freeze_sem_const * present
        if self._raw_idx:
            if freeze_raw_append_const is not None:
                present = (raw_seq.abs().sum(dim=-1, keepdim=True) > 0).float()
                const = freeze_raw_append_const.to(raw_seq.device, raw_seq.dtype)
                raw_part = const.expand(*raw_seq.shape[:-1], len(self._raw_idx)) * present
            else:
                raw_part = raw_seq[..., self._raw_idx]
            sem = torch.cat([sem, raw_part], dim=-1)
        shape = sem.shape[:-2]
        T = sem.shape[-2]
        flat = sem.reshape(-1, T, sem.shape[-1])
        h = self.node_encoder(flat)
        return h.reshape(*shape, -1)

    return _encode_nodes


def freeze_position_in_batch(batch_gpu: dict, pos_const: torch.Tensor) -> dict:
    """(B) raw 위치 자체를 Stage A 이전에 고정 — ego/1-hop/2-hop 전부, present 프레임만."""
    out = dict(batch_gpu)
    for key in ('node_seq', 'nbr_node_seqs', 'nbr2_node_seqs'):
        t = batch_gpu.get(key)
        if t is None or t.numel() == 0:
            continue
        present = (t.abs().sum(dim=-1, keepdim=True) > 0).float()
        const = pos_const.to(t.device, t.dtype)
        t2 = t.clone()
        t2[..., 0:2] = const.expand(*t.shape[:-1], 2) * present
        out[key] = t2
    return out


@torch.no_grad()
def run_eval(model, loader, device, batch_transform=None):
    cm = torch.zeros(3, 3, dtype=torch.int64)
    model.eval()
    for batch in loader:
        targets = batch['y']
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if batch_transform is not None:
            batch_gpu = batch_transform(batch_gpu)
        preds = model(batch_gpu).argmax(dim=-1).cpu()
        cm += confusion_matrix(preds, targets, 3)
    return metrics_from_confusion(cm, list(CLASS_NAMES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', type=str, default=None)
    args = ap.parse_args()
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    W, H = cfg['tsem']['W'], cfg['tsem']['H']
    gcfg = cfg['graph']
    csv_files = get_csv_files(cfg)

    model = build_model(cfg, T=W).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    orig_encode_nodes = model._encode_nodes.__func__
    print(f'체크포인트 로드: {CKPT_PATH} (epoch {ckpt.get("epoch")})')
    print(f'모델: raw_append={model.raw_append}  semantic_variant={model.semantic.variant}  '
          f'params={model.count_parameters():,}')

    train_loader, _, test_loader = build_tsem_dataloaders(
        csv_files, W=W, H=H, radius=gcfg['radius'], K_max=gcfg['K_max'], K_max2=gcfg['K_max2'],
        batch_size=4096, train_ratio=cfg['data']['train_ratio'], val_ratio=cfg['data']['val_ratio'],
        num_workers=4, augment=None, stop_persist_delta=cfg['tsem'].get('stop_persist_delta', 2),
        verbose=True,
    )

    print('\n=== train 평균값 계산 (semantic 8D 채널별 + anchor raw position) ===')
    sem_mean, pos_mean = compute_train_means(model, train_loader, device)
    for name, v in zip(SEM_CHANNEL_NAMES, sem_mean.tolist()):
        print(f'  {name:>8s} 평균 = {v:.4f}')
    print(f'  anchor position 평균(로터리 평균좌표) = ({pos_mean[0].item():.3f}, {pos_mean[1].item():.3f})')

    results = {}

    # ---- baseline (원본 forward, 개입 없음) ----
    model._encode_nodes = types.MethodType(orig_encode_nodes, model)
    r = run_eval(model, test_loader, device)
    results['baseline'] = r
    print(f'\n[baseline]            acc={r["accuracy"]*100:.2f}%  macro_f1={r["macro_f1"]*100:.2f}%  '
          f'(체크포인트 results.json 81.91%/81.95%와 대조)')

    # ---- (A) raw_append 2D만 고정 ----
    model._encode_nodes = types.MethodType(
        make_encode_nodes(freeze_raw_append_const=pos_mean), model)
    r = run_eval(model, test_loader, device)
    results['freeze_raw_append_pos'] = r
    print(f'[A: raw position 2D만 고정]  acc={r["accuracy"]*100:.2f}%  macro_f1={r["macro_f1"]*100:.2f}%  '
          f'(Δ acc={100*(r["accuracy"]-results["baseline"]["accuracy"]):+.2f}%p)')

    # ---- (B) raw 위치 자체를 Stage A 이전에 고정 (ego+이웃+2hop 전부) ----
    model._encode_nodes = types.MethodType(orig_encode_nodes, model)  # 원본 encode로 되돌림
    r = run_eval(model, test_loader, device,
                 batch_transform=lambda b: freeze_position_in_batch(b, pos_mean))
    results['freeze_raw_position_all_nodes'] = r
    print(f'[B: 전체 노드 raw 위치 고정]  acc={r["accuracy"]*100:.2f}%  macro_f1={r["macro_f1"]*100:.2f}%  '
          f'(Δ acc={100*(r["accuracy"]-results["baseline"]["accuracy"]):+.2f}%p)')

    # ---- (C) semantic 채널 하나씩 고정 (대조군) ----
    per_channel = {}
    for idx, name in enumerate(SEM_CHANNEL_NAMES):
        model._encode_nodes = types.MethodType(
            make_encode_nodes(freeze_sem_idx=idx, freeze_sem_const=float(sem_mean[idx])), model)
        r = run_eval(model, test_loader, device)
        per_channel[name] = r
        tag = ' (site-specific)' if name in SITE_SPECIFIC else ''
        print(f'[C: semantic「{name}」고정]{tag}  acc={r["accuracy"]*100:.2f}%  macro_f1={r["macro_f1"]*100:.2f}%  '
              f'(Δ acc={100*(r["accuracy"]-results["baseline"]["accuracy"]):+.2f}%p)')
    results['freeze_one_semantic_channel'] = per_channel
    model._encode_nodes = types.MethodType(orig_encode_nodes, model)

    # ---- 요약 ----
    base_acc = results['baseline']['accuracy'] * 100
    base_f1 = results['baseline']['macro_f1'] * 100
    a_drop = base_acc - results['freeze_raw_append_pos']['accuracy'] * 100
    b_drop = base_acc - results['freeze_raw_position_all_nodes']['accuracy'] * 100
    c_drops = {n: base_acc - per_channel[n]['accuracy'] * 100 for n in SEM_CHANNEL_NAMES}
    c_drops_non_site = {n: d for n, d in c_drops.items() if n not in SITE_SPECIFIC}

    print('\n=== 요약: accuracy 하락폭(%p) ===')
    print(f'(A) raw position 2D만 고정        : {a_drop:.2f}%p')
    print(f'(B) 전체 노드 raw 위치 고정(연쇄) : {b_drop:.2f}%p')
    print(f'(C) semantic 채널 1개 고정 — 범위 : '
          f'{min(c_drops.values()):.2f}~{max(c_drops.values()):.2f}%p '
          f'(평균 {np.mean(list(c_drops.values())):.2f}%p)')
    print(f'    └ site-specific 제외(v,a,j,ω,d_lat,κ만): '
          f'{min(c_drops_non_site.values()):.2f}~{max(c_drops_non_site.values()):.2f}%p '
          f'(평균 {np.mean(list(c_drops_non_site.values())):.2f}%p)')
    print(f'\n(A)가 순수 운동학 채널 평균 하락폭의 {a_drop/np.mean(list(c_drops_non_site.values())):.1f}배, '
          f'(B)는 {b_drop/np.mean(list(c_drops_non_site.values())):.1f}배')

    summary = {
        'checkpoint': CKPT_PATH,
        'baseline_accuracy_pct': base_acc, 'baseline_macro_f1_pct': base_f1,
        'A_freeze_raw_append_pos_accuracy_pct': results['freeze_raw_append_pos']['accuracy'] * 100,
        'A_drop_pct_points': a_drop,
        'B_freeze_all_raw_position_accuracy_pct': results['freeze_raw_position_all_nodes']['accuracy'] * 100,
        'B_drop_pct_points': b_drop,
        'C_per_channel_drop_pct_points': c_drops,
        'C_non_site_specific_mean_drop': float(np.mean(list(c_drops_non_site.values()))),
        'C_non_site_specific_max_drop': float(max(c_drops_non_site.values())),
        'rotary_mean_position': [float(pos_mean[0]), float(pos_mean[1])],
        'per_class': {
            'baseline': results['baseline']['per_class'],
            'A_freeze_raw_append_pos': results['freeze_raw_append_pos']['per_class'],
            'B_freeze_all_raw_position': results['freeze_raw_position_all_nodes']['per_class'],
        },
    }
    json_path = os.path.join(OUT_DIR, 'position_freeze_ablation_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n요약 JSON: {json_path}')

    # ---- 시각화 (막대그래프, ko/en) ----
    _visualize(base_acc, a_drop, b_drop, c_drops, SITE_SPECIFIC, 'ko')
    _visualize(base_acc, a_drop, b_drop, c_drops, SITE_SPECIFIC, 'en')


def _visualize(base_acc, a_drop, b_drop, c_drops, site_specific, lang):
    labels_ko = {
        'A': '(A) raw position\n2D만 고정', 'B': '(B) 전체 노드\nraw 위치 고정(연쇄)',
        'title': '위치 암기 분석④ — 추론 시 위치 채널 고정 시 accuracy 하락폭\n(10차-2 체크포인트, 재학습 없음, test split)',
        'ylabel': 'accuracy 하락폭 (%p, baseline 대비)', 'site': '(랜드마크 참조)',
    }
    labels_en = {
        'A': '(A) freeze raw\nposition 2D only', 'B': '(B) freeze raw position\nat all nodes (cascaded)',
        'title': 'Position-memorization analysis 4 -- accuracy drop when position is frozen at inference\n'
                 '(10D checkpoint, no retraining, test split)',
        'ylabel': 'Accuracy drop (pct. points vs baseline)', 'site': '(landmark-referenced)',
    }
    t = labels_ko if lang == 'ko' else labels_en

    names = list(c_drops.keys())
    drops = [c_drops[n] for n in names]
    colors = ['#f97316' if n in site_specific else '#94a3b8' for n in names]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(names, drops, color=colors, label=('semantic 채널 1개' if lang == 'ko' else 'single semantic channel'))
    ax.bar(['A'], [a_drop], color='#dc2626', label=t['A'].replace('\n', ' '))
    ax.bar(['B'], [b_drop], color='#7c1d1d', label=t['B'].replace('\n', ' '))
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel(t['ylabel'])
    ax.set_title(t['title'], fontsize=11)
    for i, n in enumerate(names):
        if n in site_specific:
            ax.text(i, drops[i], t['site'], ha='center', va='bottom', fontsize=8, color='#f97316')
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f'position_freeze_ablation_{lang}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'그림 저장[{lang}]: {out_path}')


if __name__ == '__main__':
    main()
