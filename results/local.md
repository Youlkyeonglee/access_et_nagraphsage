# 로컬 서버 (LOCAL) 실험 결과 로그

> 로컬 서버(원본, Mamba 전용 포함) 결과 기록. 학교 서버는 school.md에 기록.

## 완료 (150 epoch, 예산 부족 — 참고용)
- 2-hop + SupCon: Test 94.32% (Stop 99.86 / LC 81.15 / Normal 95.05)
- K1=10 K2=6: Test 94.33% (Stop 99.87 / LC 81.26 / Normal 95.00)
- 2-hop (CE+LS): Test 94.15% ← C·E ablation 기준 비교값

## 진행중 (500 epoch)
- D-supcon-ep500 (2-hop+SupCon): 진행중
- D-k10-ep500 (K1=10, 대조군): 진행중
- D-k10-supcon-ep500 (K1=10+SupCon, 본命): 진행중
