# OCR-DOTA V3 Core2 Ablation (10 datasets)

本报告由隔离 runner 自动生成；标签仅用于完整 replay 后的评测与 paired diagnostic。

## 实验身份

- version: ocr-dota-v3-core2-ablation10-v1
- identity_sha256: 32cd1a26b11c38b4235d77208a1e0d882ae795d1829105b6a6457f6d58382c90
- precision: fp32
- selection: global / per-dataset oracle / LOOCV are reported separately

## Rank-free Base

| Dataset | Correct | N | Accuracy | Last50 accuracy |
|---|---:|---:|---:|---:|
| DTD | 23 | 64 | 35.937500% | 40.625000% |

## Posterior global best

- candidate: `P0`
- prediction_strength: `0.0`
- macro accuracy: `35.937500%`
- micro accuracy: `35.937500%`
- positive datasets: `0/1`

## Update responsibility global best

- candidate: `U2_t0.075_p0.25_mix05`
- macro accuracy: `37.500000%`
- last50 macro accuracy: `40.625000%`
- micro accuracy: `37.500000%`
- positive datasets: `1/1`
- positive last50 datasets: `0/1`

## 说明

Per-dataset oracle 仅用于机制诊断；LOOCV 在选择 held-out 数据集参数时不使用 held-out 结果。Update/Full 阶段由同一 sidecar 的后续 phase 写入相应表。
