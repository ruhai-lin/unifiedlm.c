# UD2-scale（代號 `SDulas`）

UD2-gate 的對照：每個 token 的門換成每層一個可學的純量 2·sigmoid(b)。每層多 1 個參數。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[UD2-gate](ud2-gate.md)
- **程式碼**：`scripts/delta.py` 的 `ReadScaleAnchorDeltaBlock`，訓練時 `--variant=SDulas`

## 怎麼做的

- UD2-gate：y = x + μ z_t + o_t · E_tᵀ q_t，o_t = 2·sigmoid(w·h_t + b)，每個 token 一個值。
- UD2-scale：y = x + μ z_t + s · E_tᵀ q_t，s = 2·sigmoid(b)，每層一個純量，b = 0 起步（s = 1）。
- 每層只多 1 個參數（UD2 15,452,760 → 15,452,772）；其他跟 UD2 一樣。

**問題**：UD2-gate 的門每層差很多、同一層裡幾乎不變。它的進步是來自每層把讀出調大調小，還是需要跟著 token 變？

```text
y = μ z_t + s · E_tᵀ q_t，s = 2·sigmoid(b)（每層一個）
```

**設定**：15M（12 層、2,300 步），跟 UD2 15M 同設定；1 個 seed（1337），2.3 小時（8,318 秒）；fork session 跑（2026-11-15）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsdulas15m` | 1 | 15.45M | 3.776 | 3.734 | 4.130 | 3.733 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.695。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：2 / 20 / 1 / 1 / 1 / 1 / 1 / 2 / 1 / 2 / 1 / 1。

15M 按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量（第 0 → 11 層，[每個格子記多久](../experiments/slot-retention.md)）：0% / 36% / 0% / 0% / 0% / 0% / 6% / 4% / 22% / 0% / 7% / 3%。

## 結論

15M（1 個 seed）：NLL 3.777（長度 1,024）、3.734（4,096），跟 UD2（3.778、3.736）一樣，UD2-gate（3.766）的進步沒有出現。讀到 20K 的 16K–20K 3.695，也跟 UD2 一樣。學到的縮放每層都在 0.88–1.04，幾乎沒離開 1；UD2-gate 的門卻落在 0.15–1.13，而且它的讀出本身比較大（每層平均 0.38–2.31，UD2-scale 0.25–1.30）。所以 UD2-gate 的進步不是「每層一個固定縮放」就拿得到；是門跟著輸入變的那一點點，還是門讓模型學出更大的讀出再縮小，這次分不開。按寫入量加權，過了 256 個 token 還留下的：第 1 層 36%、第 8 層 22%，其他層最多 7%，跟 UD2 差不多。

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
python scripts/train_variant.py --out_dir=outputs/fwsdulas15m --variant=SDulas --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsdulas15m
```
