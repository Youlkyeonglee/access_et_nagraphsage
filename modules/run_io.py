"""
학습 산출물 저장 유틸
=====================
- Tee         : stdout을 콘솔 + train.log 파일에 동시 기록
- save_results: test 결과를 실험별 results.json + 전체 비교용 summary.csv에 기록

세 학습 스크립트(train.py / train_supcon.py / train_feat.py)가 공용으로 사용한다.
"""
import csv
import json
import sys
from pathlib import Path

# 프로젝트 루트 (modules/ 의 부모)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# summary.csv 에 기록하는 고정 컬럼 순서 (스크립트마다 달라도 정렬 유지)
SUMMARY_FIELDS = [
    'timestamp', 'experiment', 'script', 'test_acc',
    'acc_stop', 'acc_lanechange', 'acc_normal', 'best_val_acc',
    'encoder', 'T', 'hidden_dim', 'temporal_target',
    'neighbor_mode', 'num_epochs', 'seed',
]


class Tee:
    """stdout을 콘솔과 로그 파일에 동시 출력. 학습 로그 보존용."""

    def __init__(self, log_path):
        self.stdout = sys.stdout
        self.file = open(str(log_path), 'a', buffering=1)

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        """stdout 복원 후 파일 닫기."""
        sys.stdout = self.stdout
        self.file.close()


def save_results(save_dir, result: dict, summary_csv=None):
    """
    실험 결과 저장.
      - {save_dir}/results.json : 해당 실험 전체 결과 (체크포인트와 함께 자체 보존)
      - {summary_csv}           : 전체 실험 비교용 한 줄 append (기본 results/summary.csv)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if summary_csv is None:
        summary_csv = PROJECT_ROOT / 'results' / 'summary.csv'
    summary_csv = Path(summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    write_header = not summary_csv.exists()
    with open(summary_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(SUMMARY_FIELDS)
        w.writerow([result.get(k, '') for k in SUMMARY_FIELDS])
