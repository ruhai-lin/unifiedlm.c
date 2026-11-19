# U-Delta（代號 `SD`）

把格子裡的“後驗均值”換成以 FFN 為錨的 delta rule 記憶：記憶只存 FFN 沒預測到的部分，同一個位址的新值按比例覆蓋舊值。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[U-Avg](u-avg.md)
- **程式碼**：`scripts/delta.py` 的 `DeltaBlock`、`delta_memory`，訓練時 `--variant=SD`

## 怎麼做的

- 位址、卷積、`μ` 跟 U-Avg 一樣；讀出變成 `y_t = μ r_t + E_tᵀ q_t`，`q_t = r_t / |r_t|`。`E`（m × d）是格子相對 `μ` 的偏差，所以記憶衰減時回到 FFN。
- 寫入目標 `u_t = v_t − μ k_t`，`k_t = r_{t−1}`：只寫 FFN 在這個位址預測不到的殘差。
- delta rule：先用寫位址在記憶裡查（`Λ Eᵀ k̂`，記憶在這裡已經有什麼），只寫 `β` 倍的誤差，所以已經存好的值不會被重寫。`β_t = sigmoid(W_β h_t + b)` 是寫入強度。
- 遺忘率 `Λ_t` 沿用 U-Gate 的每格、每 token 遺忘門（由位址 `r_t` 經過低秩 16 算出），但去掉 U-Gate 的 `β·k` 項（delta rule 做了這件事）。
- 遞推按塊計算（PyTorch autograd，fp32 + TF32；塊內解單位下三角系統，塊間傳狀態）。

```text
y_t = mu r_t + E_t^T q_t,   q_t = r_t / |r_t|
u_t = v_t - mu k_t,   k_t = r_{t-1},   kh_t = k_t / |k_t|
E_t = (I - beta_t kh_t kh_t^T) Lam_t E_{t-1} + beta_t kh_t u_t^T
log lam_{t,a} = expm1(-rate_a * g_{t,a}),   g_t = softplus(P Q rms(r_t) + b) / ln 2
```

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsd15m`（+2 個 seed） | 3 | 15.45M | 3.898（3.895–3.901） | 3.857 | 4.261 | 3.852 |
| 42M | `fw42m_sd` | 1 | 41.77M | 3.350 | 3.303 | 3.757 | 3.299 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.816（3 個 seed）。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：7 / 373 / 315 / 434 / 259 / 23 / 76 / 95 / 86 / 84 / 65 / 314。

15M 按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量（第 0 → 11 層，[每個格子記多久](../experiments/slot-retention.md)）：0% / 51% / 42% / 48% / 35% / 4% / 0% / 1% / 1% / 1% / 1% / 38%。

## 結論

15M 3.898（3 個 seed，3.895–3.901），三個 seed 都贏 Transformer（3.945（3 個 seed，3.928–3.957）），比 U-Avg 好 0.214，比 U-Gate 好 0.100。42M 卻輸 Transformer 0.136（3.350 對 3.214），每個位置都差，主因是 FFN 讀了歸一化的位址，見 [U-Delta-rawFFN](u-delta-rawffn.md)。弱點在開頭：前 64 個 token 4.258，Transformer 4.224，GDN 4.166。讀到 20K 不崩（16K–20K 段 3.810–3.820）；記憶任務 m2kv 間隔 4,096 100.0%，不重訓拉到 8,192 還有 95.4%。每層自己學出記多久：有效記憶長度從第 0 層的約 7 個 token 到第 1–4、11 層的幾百個。

## 相關實驗

- [記憶的寫法：平均、遺忘門、delta rule](../experiments/write-rule.md)
- [規模：15M → 42M → 110M](../experiments/scale.md)
- [記憶設計：快記憶、慢記憶、遺忘率](../experiments/memory-design.md)
- [內容是被遺忘還是被覆寫](../experiments/forgetting-vs-overwrite.md)
- [寫入量與 loss](../experiments/write-vs-loss.md)
- [長上下文與外推（讀到 20K）](../experiments/extrapolation.md)
- [記憶任務 m2kv](../experiments/memory-tasks.md)
- [U-Delta 為什麼贏 GDN](../experiments/vs-gdn.md)
- [多 seed 的穩定性](../experiments/seeds.md)
- [逐 token 差距：輸在哪些 token](../experiments/token-gap.md)
- [拿掉記憶：記憶值多少、模型多依賴記憶](../experiments/memory-ablation.md)
- [遠處的前文值多少](../experiments/reach.md)
- [每個格子記多久（按寫入量加權）](../experiments/slot-retention.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsd.txt）
python scripts/train_variant.py --out_dir=outputs/fwsd15m --variant=SD --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsd15m
# 42M（scripts/nrp/fwnext.txt）
python scripts/train_variant.py --out_dir=outputs/fw42m_sd --variant=SD --hidden_dim=1248 --dim=512 --n_layers=16 --batch_size=8 --gradient_accumulation_steps=16 --warmup_iters=1000 --max_iters=6400 --dataset=fineweb --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --learning_rate=5e-4 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fw42m_sd
```
