# UD2-gate-norm（代號 `SDulagn`）

UD2-gate 的讀出先 RMSNorm 再過門（GDN 的做法）。每層多 288 個參數。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[UD2-gate](ud2-gate.md)
- **程式碼**：`scripts/delta.py` 的 `ReadNormGateAnchorDeltaBlock`，訓練時 `--variant=SDulagn`

## 怎麼做的

- UD2-gate：y = x + μ z_t + o_t · E_tᵀ q_t。
- UD2-gate-norm：y = x + μ z_t + o_t · RMSNorm(E_tᵀ q_t)；每個信道的增益從 1/√dim 起步，所以一開始讀出的長度是 1。
- 每層多 288 個參數（RMSNorm 的增益），共 15,459,684；其他跟 UD2-gate 一樣。

**問題**：記憶越裝越滿，讀出就越大，後面的位置被放大。先把讀出歸一化，會不會讓後面的位置更好？

```text
y = μ z_t + o_t · RMSNorm(E_tᵀ q_t)，o_t = 2·sigmoid(w·h_t + b)
```

**設定**：15M（12 層、2,300 步），跟 UD2-gate 15M 同設定；1 個 seed（1337），1.7 小時（5,944 秒）；fork session 跑（2026-11-15）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsdulagn15m` | 1 | 15.46M | 3.739 | 3.696 | 4.122 | 3.691 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.661。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：87 / 2 / 2 / 1 / 3 / 1 / 1 / 2 / 2 / 2 / 2 / 1。

15M 按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量（第 0 → 11 層，[每個格子記多久](../experiments/slot-retention.md)）：39% / 1% / 11% / 0% / 23% / 0% / 0% / 1% / 45% / 58% / 16% / 0%。

## 結論

15M（1 個 seed）：NLL 3.739（長度 1,024）、3.696（4,096），比同一個 seed 的 UD2-gate（3.766）好 0.028。進步越後面越大：前 64 個 token 4.122 對 4.119（打平），512–1,024 3.691 對 3.722（好 0.030），符合「讀出隨位置變大」的假設。讀到 20K 的 16K–20K 3.661（UD2-gate 3.685）。逐 token 對 Transformer：全部 −0.182（UD2-gate −0.156），多出來的一半以上在見過的 token（−0.118 對 −0.079），其次是照抄前文（−0.623 對 −0.587）和位置 256 以後的新 token（−0.031 對 −0.017）。截掉遠處前文，新 token 還是變好 0.010，干擾沒有消除。按寫入量加權，記得久的層換了位置：第 0、2、4、8、9、10 層過了 256 個 token 還留 39%／11%／23%／45%／58%／16%，UD2-gate 記最久的第 1 層只剩 1%。

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
python scripts/train_variant.py --out_dir=outputs/fwsdulagn15m --variant=SDulagn --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsdulagn15m
```
