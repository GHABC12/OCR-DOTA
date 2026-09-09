# V3 三风险真实在线消融

此目录是完全独立的诊断 sidecar，不修改或覆盖 `OCR-DOTA-V3/`。

固定当前 V3 的 base、rank compatibility、更新规则、cache、样本顺序和随机种子，
仅对三种历史 risk 做完整 2^3 全因子开关：

- D：residual magnitude；
- E：semantic leakage；
- O：旧 top-margin order violation。

三项 risk 在诊断模型中以历史公式进入 Gaussian geometry logits；V3 的同一个 rank
compatibility 仍同时用于 prediction prior 和 update gate。每个实验臂均从全新状态开始，
在 GPU 上逐样本预测、更新，不读取离线轨迹。

默认使用经典十数据集全量流。贡献用精确三因子 Shapley 值分配，同时报告 signed share、
absolute share 和 positive-only share；这些结果属于 full-stream diagnostic oracle，不是独立
测试集泛化结果。

运行：

```bash
/home/user/.virtualenvs/CLIP-main/bin/python3 \
  V3-risk-xiaorong/run_risk_ablation.py --device cuda
```

恢复时增加 `--resume`。创建 `V3-risk-xiaorong/STOP` 可在样本检查点安全停止。
