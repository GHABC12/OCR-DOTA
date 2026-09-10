# OCR-DOTA V3 Core2 Ablation (10 datasets)

本报告由隔离 runner 自动生成；标签仅用于完整 replay 后的评测与 paired diagnostic。

## 实验身份

- version: ocr-dota-v3-core2-ablation10-v1
- identity_sha256: e02c4164f0b865ecd2b4331bd6e3c8d31ffc21000721b9f52c8a9454d5588798
- precision: fp32
- selection: global / per-dataset oracle / LOOCV are reported separately

## Rank-free Base

| Dataset | Correct | N | Accuracy | Last50 accuracy |
|---|---:|---:|---:|---:|
| Aircraft | 908 | 3333 | 27.242724% | 28.974205% |
| Caltech101 | 2326 | 2465 | 94.361055% | 95.133820% |
| Cars | 5605 | 8041 | 69.705261% | 71.375280% |
| DTD | 863 | 1692 | 51.004728% | 54.255319% |
| EuroSAT | 5490 | 8100 | 67.777778% | 68.543210% |
| Flower102 | 1895 | 2463 | 76.938693% | 80.681818% |
| Food101 | 26373 | 30300 | 87.039604% | 87.207921% |
| Pets | 3409 | 3669 | 92.913600% | 93.950954% |
| SUN397 | 13681 | 19850 | 68.921914% | 70.035264% |
| UCF101 | 2801 | 3783 | 74.041766% | 76.744186% |

## Posterior global best

- candidate: `P1`
- prediction_strength: `0.0075`
- macro accuracy: `71.005493%`
- micro accuracy: `75.701348%`
- positive datasets: `1/10`

## Update responsibility global best

- candidate: `U1_t0.15_p2`
- macro accuracy: `71.616265%`
- last50 macro accuracy: `73.395822%`
- micro accuracy: `76.470799%`
- positive datasets: `4/10`
- positive last50 datasets: `5/10`

## Full V3 confirmation

- datasets completed: 10/10
- micro accuracy: 75.543634%

## 说明

Per-dataset oracle 仅用于机制诊断；LOOCV 在选择 held-out 数据集参数时不使用 held-out 结果。Update/Full 阶段由同一 sidecar 的后续 phase 写入相应表。
