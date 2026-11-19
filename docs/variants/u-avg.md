# U-Avg（代號 `S`）

原版 UnifiedLM，也就是本倉庫 `model.py` 的標準模型：同一組格子既當 FFN 又當上下文記憶，格子裡存“先驗 + 上下文證據”的後驗均值。

- **類別**：主線（統一 block，不含 attention）
- **基於**：—
- **程式碼**：`model.py` 的 `UnifiedBlock`（`kernel.py` 的 `unified_memory`），訓練時 `--variant=S`

## 怎麼做的

- 每個 block：`h = RMSNorm(x)`，`u` = 寬 4 的因果深度卷積(`h`)，對 m 個格子的位址 `r_t = relu(K u_t + b)² / (eps + Σ relu(K u_t + b)²)`（`eps` 是可學的空槽質量）。
- 當前位置的值 `v_t = W_v h_t` 寫到前一位置的位址 `r_{t−1}` 上，讀的時候用當前位址 `r_t`。
- 每個格子存後驗均值 `m = (w·μ + S) / (w + n)`：`μ` 是靜態內容（就是 FFN 的輸出權重），`w` 是每格的先驗質量，`S`、`n` 是按遺忘率 `λ` 衰減累加的證據。`λ` 每層一個標量，學出來的。
- 同一個位址 `r` 既選 FFN 的格子也選記憶的格子，這就是“統一”。代價是證據越多，靜態 `μ` 的比重越小（證據稀釋先驗）。
- 快記憶那一項是以格子為 key 維的因果線性注意力，訓練時由 `kernel.py` 的 Triton kernel 計算，推理用逐 token 的遞推（與 `run.c` 相同）。

```text
r_t = relu(K u_t + b)^2 / (eps + sum_a relu(K u_t + b)_a^2)
S_t = lam S_{t-1} + v_t r_{t-1}^T,   n_t = lam n_{t-1} + r_{t-1}
y_t = ((w mu + S_t) / (w + n_t)) r_t,   x <- x + y_t
```

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fw15m_s`（+2 個 seed） | 3 | 15.17M | 4.112（4.107–4.116） | 5.658 | 4.392 | 4.087 |
| 42M | `fw42m_s` | 1 | 41.63M | 3.423 | 8.708 | 3.767 | 3.383 |
| 110M | `fw110m_s` | 1 | 109.40M | 2.994 | 6.635 | 3.375 | 2.949 |

## 結論

15M 4.112（3 個 seed，4.107–4.116），比 Transformer（3.945）差 0.167；42M 3.423 對 3.214；110M 2.994 對 2.822。過了訓練長度會崩：長度 4,096 時 5.026–6.175。記憶任務 m2kv 在間隔 1,024 只有 18.5%，4,096 是 0%。後面所有 U- 變體都從它改起；最大的改進是把“平均”換成 delta rule（[U-Delta](u-delta.md)）。

## 相關實驗

- [記憶的寫法：平均、遺忘門、delta rule](../experiments/write-rule.md)
- [規模：15M → 42M → 110M](../experiments/scale.md)
- [統一的對照：共享位址 vs 分開位址](../experiments/unification.md)
- [記憶任務 m2kv](../experiments/memory-tasks.md)
- [多 seed 的穩定性](../experiments/seeds.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fw15m.txt）
python scripts/train_variant.py --out_dir=outputs/fw15m_s --variant=S --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=32 --gradient_accumulation_steps=4 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fw15m_s
# 42M（scripts/nrp/fw42m.txt）
python scripts/train_variant.py --out_dir=outputs/fw42m_s --variant=S --hidden_dim=1280 --dataset=fineweb --dim=512 --n_layers=16 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=32 --gradient_accumulation_steps=4 --learning_rate=5e-4 --warmup_iters=1000 --max_iters=6400 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fw42m_s
# 110M（scripts/nrp/fw110m.txt）
python scripts/train_variant.py --out_dir=outputs/fw110m_s --variant=S --hidden_dim=1912 --dataset=fineweb4 --dim=768 --n_layers=24 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=2600 --max_iters=16700 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=1000 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fw110m_s
```
