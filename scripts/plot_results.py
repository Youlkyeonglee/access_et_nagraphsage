"""
ET-NAGraphSAGE 실험 결과 시각화
================================
task output 로그를 파싱해 학습 곡선과 결과 비교 그림을 생성한다.
출력: docs/figures/*.png
"""

import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TASK_DIR = "/tmp/claude-1000/-home-oem-TNA-research/9ceaeaa7-0e93-44b3-bb42-75e6b520789d/tasks"
LOG_DIR  = "/home/oem/TNA_research/logs"
OUT_DIR  = "/home/oem/TNA_research/docs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s*\|\s*Train Acc\s+([\d.]+)\s*\|\s*Val Acc\s+([\d.]+)")


def parse_log(path):
    """로그에서 (epochs, train_acc, val_acc) 추출."""
    eps, tr, va = [], [], []
    try:
        with open(path) as f:
            for line in f:
                m = EPOCH_RE.search(line)
                if m:
                    eps.append(int(m.group(1)))
                    tr.append(float(m.group(2)))
                    va.append(float(m.group(3)))
    except FileNotFoundError:
        pass
    return eps, tr, va


def t(name):
    return os.path.join(TASK_DIR, name)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: 핵심 실험 val accuracy 곡선 비교
# ─────────────────────────────────────────────────────────────────────────────

FIG1 = [
    ("1-hop GRU (93.85%)",       t("t10_v2.output"),      "#94a3b8", "-"),
    ("2-hop (94.15%)",           t("2hop_reg.output"),    "#3b82f6", "-"),
    ("2-hop+SupCon (94.32%)",    t("2hop_supcon.output"), "#8b5cf6", "-"),
    ("K1=10 (94.33%)",           t("2hop_k10.output"),    "#10b981", "-"),
]

plt.figure(figsize=(9, 5.5))
for label, path, color, ls in FIG1:
    eps, _, va = parse_log(path)
    if eps:
        plt.plot(eps, va, label=label, color=color, linestyle=ls, linewidth=1.8)
plt.axhline(0.9454, color="#ef4444", linestyle="--", linewidth=1.5,
            label="NAGraphSAGE (94.54%)")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy — Key Models (150 epoch)")
plt.ylim(0.90, 0.955)
plt.legend(loc="lower right", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig1_val_curves.png"), dpi=130)
plt.close()
print("saved fig1_val_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Test Acc + 클래스별 정확도 비교 (막대)
# ─────────────────────────────────────────────────────────────────────────────

RESULTS = [
    ("T=1",            0.9251, 0.9985, 0.7415, 0.9417),
    ("1-hop GRU",      0.9385, 0.9978, 0.7952, 0.9475),
    ("SupCon",         0.9367, 0.9977, 0.8051, 0.9378),
    ("2-hop",          0.9415, 0.9992, 0.8153, 0.9435),
    ("2-hop+SupCon",   0.9432, 0.9986, 0.8115, 0.9505),
    ("K1=10",          0.9433, 0.9987, 0.8126, 0.9500),
]
labels   = [r[0] for r in RESULTS]
state    = [r[1] for r in RESULTS]
lc       = [r[3] for r in RESULTS]

x = range(len(labels))
fig, ax1 = plt.subplots(figsize=(10, 5.5))

bars = ax1.bar([i - 0.2 for i in x], state, width=0.4,
               label="State Acc", color="#3b82f6")
ax1.bar([i + 0.2 for i in x], lc, width=0.4,
        label="LaneChange Acc", color="#f59e0b")
ax1.axhline(0.9454, color="#ef4444", linestyle="--", linewidth=1.4,
            label="NAGraphSAGE 94.54%")

for i, v in zip(x, state):
    ax1.text(i - 0.2, v + 0.002, f"{v*100:.1f}", ha="center", fontsize=8)
for i, v in zip(x, lc):
    ax1.text(i + 0.2, v + 0.002, f"{v*100:.1f}", ha="center", fontsize=8)

ax1.set_ylabel("Accuracy")
ax1.set_ylim(0.70, 1.0)
ax1.set_xticks(list(x))
ax1.set_xticklabels(labels, rotation=15, fontsize=9)
ax1.set_title("Test State_Acc & LaneChange Accuracy")
ax1.legend(loc="lower center", fontsize=9)
ax1.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig2_test_bars.png"), dpi=130)
plt.close()
print("saved fig2_test_bars.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: 과적합 점검 — Train vs Val (최고 모델 K1=10)
# ─────────────────────────────────────────────────────────────────────────────

plt.figure(figsize=(9, 5.5))
for label, path, color in [
    ("K1=10", t("2hop_k10.output"), "#10b981"),
    ("2-hop+SupCon", t("2hop_supcon.output"), "#8b5cf6"),
]:
    eps, tr, va = parse_log(path)
    if eps:
        plt.plot(eps, tr, color=color, linestyle="--", linewidth=1.3,
                 label=f"{label} Train")
        plt.plot(eps, va, color=color, linestyle="-", linewidth=1.8,
                 label=f"{label} Val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Val — Overfitting Check (small gap)")
plt.ylim(0.90, 0.97)
plt.legend(loc="lower right", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig3_overfit.png"), dpi=130)
plt.close()
print("saved fig3_overfit.png")

print("\n모든 그림 저장 완료:", OUT_DIR)
