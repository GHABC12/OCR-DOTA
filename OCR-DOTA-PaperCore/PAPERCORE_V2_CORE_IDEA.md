# OCR-DOTA PaperCore V2 核心思路

PaperCore V2 将 OCR 信息用于两个可独立消融的位置：当前预测的 posterior 校正，以及在线 Gaussian 更新中的样本责任度。两者共享同一份由稳定文本排序和动态 Gaussian 排序计算出的类别兼容度。

## 共享兼容度

对输入样本计算文本相似度排序与 Gaussian 分数排序。类别兼容度定义为：

```text
C_k = exp(-|r_text(k) - r_gaussian(k)| / tau_rank)
```

以 DOTA Gaussian posterior 得到样本级 clean mass：

```text
m(x) = sum_k softmax(g_DOTA)_k * C_k
```

同一组 `C_k` 和 `m(x)` 供预测侧与更新侧使用。

## 两个模块

- **OCR Posterior**：通过 `g_OCR,k = g_DOTA,k + gamma * log(C_k)` 调整 Gaussian posterior。`gamma=0` 时严格退化为 DOTA。V2 的预注册开发集搜索没有找到满足正向验收条件的设置，因此冻结结果采用 `gamma=0`，保留模块作为消融接口，不宣称它带来提升。
- **OCR Responsibility**：将 DOTA 原始更新权重 `p_zs(k|x)` 乘以样本级 clean mass：`omega_k = m(x)^delta * p_zs(k|x)`。`delta=1` 的冻结设置降低不可靠样本对在线 Gaussian 状态的写入强度。

## 2×2 消融

使用同一缓存、样本顺序、FP32 精度、随机种子和 DOTA 基础参数，比较 Base、Posterior、Responsibility、Full 四个设置。标签只用于预测和状态更新完成后的评估。Posterior Only 与 Base 共用完全相同的更新轨迹；Responsibility Only 改变更新责任度；Full 同时启用两者。

本说明只记录方法定义和实验约束，不包含逐样本结果、轨迹或缓存数据。
