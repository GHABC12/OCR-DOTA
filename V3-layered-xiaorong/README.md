# OCR-DOTA-V3 分层消融

本目录是独立实验 sidecar，不修改 `OCR-DOTA-V3/` 或主线代码。

每个数据集冻结当前 V3 冠军参数、cache、样本顺序、FP32 和 global seed，真实 GPU
逐样本冷启动运行以下层级：

1. Original DOTA；
2. Tuned-base DOTA；
3. Enhanced core（Residual、prediction rank、update rank均关闭）；
4. Enhanced core + Residual；
5. Enhanced core + Residual + prediction rank prior；
6. Full V3（再加入update rank gate）。

此外完整运行 Residual/Prediction-rank/Update-rank 的 2^3 组合，用精确 Shapley 值
消除顺序归因偏差。每个变体都独立冷启动复跑，验证 correct、prediction、state、
trajectory 和 compatibility SHA。

该实验使用完整评测流和已经在完整流上选择的冠军参数，属于机制诊断，不是无偏泛化评估。
