# OCR-DOTA（V3）

本仓库是 **OCR-DOTA V3** 版本代码及其消融实验（ablation sidecar）的整理发布。
从本地开发仓库 `~/shiyunxiao/OCR-DOTA`（上游 DOTA baseline：[skylineeeeen/DOTA](https://github.com/skylineeeeen/DOTA)）
抽取整理，2026-09-09 发布至 GitHub。

V3 的核心改动：排序机制只保留一个核心量 **rank compatibility**，同时用于预测
（作为显式 rank prior 加入 posterior）与更新（衡量样本对分布更新的适配度），
删除无显著贡献的 semantic leakage 与旧 order violation 项。详见
[OCR-DOTA-V3/README.md](OCR-DOTA-V3/README.md)。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| [OCR-DOTA-PaperCore/](OCR-DOTA-PaperCore/) | **论文核心实验**：基于 LegacyState DOTA 的 OCR-calibrated posterior + OCR-calibrated update responsibility，四个正式臂（Base / Posterior Only / Responsibility Only / Full），预注册 dev/holdout 划分与选择协议。含 `ocr_dota_paper/` 模块、tests、exact 复现验证、phase0 smoke 结果 |
| [OCR-DOTA-V3/](OCR-DOTA-V3/) | **V3 主实现**：`ocr_dota_v3/`（rank_compatibility.py、model.py）、configs、单元测试、run.py、smoke_cache.py、tune_\*/supervise_\* 调参驱动、BEST\_\*.json/.txt 冠军结果 |
| [V3-P0-ablation/](V3-P0-ablation/) | **P0 机制消融**：冻结 21 个非 ImageNet 数据流冠军，真实 GPU 逐样本运行各机制臂，记录纠错、compatibility 可靠性、更新污染与时间窗口统计 |
| [V3-compat-risk-search/](V3-compat-risk-search/) | **compatibility 来源搜索**：严格比较五个候选来源 D / E / O / EO / DO（`compatibility = exp(-risk/0.15)`），同一张量同时供 prediction prior 与 update gate |
| [V3-layered-xiaorong/](V3-layered-xiaorong/) | **分层消融**：Original DOTA → Tuned-base → Enhanced core → +Residual → +prediction rank prior → Full V3，另含 Residual/Prediction-rank/Update-rank 的 2³ 全组合 + 精确 Shapley 归因 |
| [V3-risk-xiaorong/](V3-risk-xiaorong/) | **三风险全因子消融**：对三种历史 risk（D: residual magnitude、E: semantic leakage、O: order violation）做 2³ 全因子开关，Shapley 值分配贡献 |

各 sidecar 均为隔离的真实在线 GPU 实验：不修改 `OCR-DOTA-V3/` 或其他实验目录，
支持断点恢复、STOP 安全停止、PidLock、JSONL fsync 与冷启动复验。完整实验输出
（summary、state、candidate results、per-sample traces、verification）保存在各自的
`results/` 目录中。

## 依赖说明

- V3 主实现 `ocr_dota_v3/model.py` 复用上游 DOTA 的状态与数据流实现；
- 消融 sidecar 及调参脚本依赖本地父仓库的 baseline 代码（`dota.py`、`datasets/` 等）
  与数据集缓存（`log/all_dataset_perf/cache/`）。本仓库仅包含 V3 实现、实验驱动脚本
  与实验结果，父仓库代码请参见 [skylineeeeen/DOTA](https://github.com/skylineeeeen/DOTA)。
- `DOTA-Rank-Only/`（legacy baseline 上的 rank-only 消融）整理时全量实验仍在运行，本次未包含。
- `OCR-DOTA-PaperCore/results/paper_core_v1_20260909/`（论文核心 campaign）已补传最终结果：
  campaign 按预注册协议以**负结果**结束（两个 18 点 posterior 网格均未达到验收标准），
  因此**未产生** `FROZEN_PAPER_CONFIG.yaml` 与 held-out 评估——协议要求此时停止、不使用 held-out
  标签，也不重新定义 Base。最终产物为 `posterior_search.json`、
  `posterior_development_correction.csv`、`research_negative.md`、完整 `candidate_results.jsonl`
  与逐候选 traces（`runs/`）。

## 运行测试

```bash
python3 -m unittest discover -s OCR-DOTA-V3 -p 'test_*.py' -v
```

32 样本 GPU 冷启动复现（需父仓库数据集缓存）：

```bash
python3 OCR-DOTA-V3/smoke_cache.py \
  --cache log/all_dataset_perf/cache/dtd_vitb16.pt --device cuda --max-samples 32
```

## License

[MIT License](LICENSE)，沿用上游 DOTA 项目（Copyright (c) 2024 Adilbek Karmanov）。
