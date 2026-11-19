# U-Delta+slow（代號 `SD2`）

U-Delta 再加一份慢記憶（每格固定速率，起點 1/(16L)）和讀出混合 ρ。

- **類別**：記憶設計
- **基於**：[U-Delta](u-delta.md)
- **程式碼**：`scripts/delta.py` 的 `TwoDeltaBlock`，訓練時 `--variant=SD2`

## 怎麼做的

- 快記憶和 U-Delta 一樣。另加一份慢記憶：每格常數遺忘率（U-Gate-nogate 的形式，起點 1/(16L)，約 16,000 個 token），有自己的寫入強度 β_slow。
- 讀出混合：`y = μ r + (1 − ρ_t) E_fastᵀ q + ρ_t E_slowᵀ q`，`ρ_t = sigmoid(W_ρ h_t)`。
- 想法是寫入門判斷為重要的 token 可以待在慢記憶裡，不被快記憶的週轉沖掉。狀態是 U-Delta 的 2 倍。

```text
y_t = mu r_t + (1 - rho_t) E_fast,t^T q_t + rho_t E_slow,t^T q_t,   rho_t = sigmoid(W_rho h_t)
E_slow: per-slot constant rates (start 1/(16 L)), its own beta
```

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsd2_15m` | 1 | 15.47M | 3.888 | 3.851 | 4.258 | 3.841 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.856。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：9 / 327 / 161 / 323 / 491 / 159 / 96 / 93 / 109 / 224 / 83 / 76。

## 結論

15M 3.888，比 U-Delta 三個 seed（3.895–3.901）好約 0.01（1 個 seed）。同容量對照 [U-Delta+fast2](u-delta-plus-fast2.md)（兩份都是快記憶）3.897，跟 U-Delta 一樣，所以進步不是來自容量。但慢記憶其實沒學：速率停在初始值附近（幾乎不忘）；第 3、4 層 ρ 高但慢記憶幾乎不寫，等於關掉記憶；第 11 層只用快記憶。只用快記憶時 NLL 6.110，只用慢記憶 5.007，混合 3.874。讀到 20K 時 4K 以後反而比 U-Delta 差（16K–20K 3.856）。真的需要記很久的記憶任務才用得上：m2kv 不重訓拉到間隔 16,384 還有 77.0%，20,000 66.9%（U-Delta 55.1%、33.9%）。單一記憶讓中間層忘快一點（U-Delta-spreadλ、U-Delta-λlr10）就追平它，見[記憶設計](../experiments/memory-design.md)。

## 相關實驗

- [記憶設計：快記憶、慢記憶、遺忘率](../experiments/memory-design.md)
- [記憶任務 m2kv](../experiments/memory-tasks.md)
- [長上下文與外推（讀到 20K）](../experiments/extrapolation.md)
- [遠處的前文值多少](../experiments/reach.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsd.txt）
python scripts/train_variant.py --out_dir=outputs/fwsd2_15m --variant=SD2 --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=8 --gradient_accumulation_steps=16 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsd2_15m
```
