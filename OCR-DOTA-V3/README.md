# OCR-DOTA-V3

该目录是独立实现，不修改上级 OCR-DOTA 的现有文件。

排序机制只保留一个核心量：`rank compatibility`。

1. 预测用途：把逐类 compatibility 作为显式 rank prior 加入 posterior；
2. 更新用途：用同一 compatibility 在基础 posterior 下的期望，衡量样本是否适合更新分布。

删除没有显著贡献的 semantic leakage 和旧 order violation。保留唯一有历史增益证据的 residual magnitude 作为几何可靠度，但不再把多个 risk 相加。显式 rank discrepancy 只用于构造 compatibility：

```text
d_k = abs(rank_stable,k - rank_dynamic,k) / (K - 1)
c_k = exp(-d_k / tau_rank)
```

预测校正：

```text
rank_logits_k = base_logits_k + gamma_pred * log(c_k)
p_rank = softmax(rank_logits)
```

更新适配度与责任：

```text
c_sample = sum_k p_base(k) * c_k
omega_k = c_sample^gamma_update * p_geometry(k)
```

文件：

- `ocr_dota_v3/rank_compatibility.py`：唯一 rank compatibility 的定义与两种用途；
- `ocr_dota_v3/model.py`：复用现有 DOTA 状态实现，替换 posterior/responsibility；
- `tests/`：公式、退化点、同源性和状态更新测试；
- `configs/default.yaml`：最小配置示例。

运行测试：

```bash
cd /home/user/shiyunxiao/OCR-DOTA
/home/user/.virtualenvs/CLIP-main/bin/python3 -m unittest discover -s OCR-DOTA-V3 -p 'test_*.py' -v
```

32 样本 GPU 冷启动复现：

```bash
/home/user/.virtualenvs/CLIP-main/bin/python3 OCR-DOTA-V3/smoke_cache.py \
  --cache log/all_dataset_perf/cache/dtd_vitb16.pt --device cuda --max-samples 32
```
