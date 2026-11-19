# UD2（代號 `SDula`）

**新的基底 UD2。** U-Delta-rawFFN+λlr1000 的寫入目標改錨在 FFN 實際給出的 μz（原本是歸一化的 μr）。不加參數。之後的變體都是 UD2 加後綴，後綴表示讀出端的修改。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[U-Delta-rawFFN+λlr1000](u-delta-rawffn-lambda-lr1000.md)
- **程式碼**：`scripts/delta.py` 的 `AnchorRawFFNDeltaBlock`，訓練時 `--variant=SDula`

## 怎麼做的

- U-Delta-rawFFN+λlr1000：寫入目標 u = v − μ r_{t−1}，r = z ／ (eps + Σ z) 是歸一化的位址；FFN 輸出的卻是 μ z。
- UD2：寫入目標 u = v − μ z_{t−1}，跟 FFN 那一半實際輸出的一樣。
- 其他（λ 學快 1,000 倍、位址、讀法）跟 U-Delta-rawFFN+λlr1000 一樣；參數、狀態一樣。

**問題**：U-Delta 的寫入以 FFN 為錨：記憶只存 FFN 猜錯的部分。U-Delta-rawFFN 改了 FFN 讀的東西，錨卻沒跟著改，兩半對不上。對齊之後會不會更好？

```text
y = μ z_t + E_tᵀ q_t，z = relu(K u + b)²
E_t = (I − β k̂ k̂ᵀ) Λ E_{t−1} + β k̂ uᵀ，u = v_t − μ z_{t−1}（U-Delta-rawFFN+λlr1000：v_t − μ r_{t−1}）
```

**設定**：15M（12 層、2,300 步），跟 U-Delta-rawFFN+λlr1000 15M 同設定；1 個 seed（1337），A40 1.4 小時。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsdula15m` | 1 | 15.45M | 3.778 | 3.736 | 4.133 | 3.734 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.695。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：2 / 23 / 1 / 1 / 1 / 1 / 1 / 1 / 1 / 1 / 2 / 1。

15M 按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量（第 0 → 11 層，[每個格子記多久](../experiments/slot-retention.md)）：0% / 36% / 0% / 0% / 0% / 0% / 0% / 2% / 3% / 16% / 20% / 2%。

## 結論

15M（1 個 seed）：NLL 3.778（長度 1,024）、3.736（4,096），比同一個 seed 的 U-Delta-rawFFN+λlr1000（3.793）好 0.015，也低於 U-Delta-rawFFN+λlr1000 三個 seed 最好的那個，U-Delta-rawFFN+λlr1000 的 seed 標準差只有 0.003，所以大概是真的。贏 Transformer（3 個 seed 平均 3.945）0.167（之後加讀出門的 UD2-gate 再好 0.012）。每個位置都好 0.013–0.020（前 64 個 token 4.133 對 4.153，512–1,024 3.734 對 3.748），所以是整體變好，不是記憶的距離變了。按格子平均的記憶長度跟 U-Delta-rawFFN+λlr1000 差不多（12 層裡有 11 層只有 1.1–1.7 個 token，第 1 層 23 個）；按寫入量加權，第 1 層過了 256 個 token 還留 36%（628/712 個格子留超過 1/e），第 9、10 層留 16%、20%，其他層最多 3%。讀到 20K：每一段都比 U-Delta-rawFFN+λlr1000 好 0.013–0.020，16K–20K 3.695（U-Delta-rawFFN+λlr1000 3.713、U-Delta-λlr1000 3.752、滑動窗口的 Transformer 3.874），越讀越遠也沒有縮小。

## 相關實驗

- [讀出端：門、縮放、RMSNorm、讀位址修正、旋轉相對位置](../experiments/read-side.md)
- [長上下文與外推（讀到 20K）](../experiments/extrapolation.md)
- [逐 token 差距：輸在哪些 token](../experiments/token-gap.md)
- [遠處的前文值多少](../experiments/reach.md)
- [每個格子記多久（按寫入量加權）](../experiments/slot-retention.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdfk.txt）
python scripts/train_variant.py --out_dir=outputs/fwsdula15m --variant=SDula --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsdula15m
```
