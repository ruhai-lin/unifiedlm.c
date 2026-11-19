# UD2-gate-q32（代號 `SDulagq`）

UD2-gate 的讀位址加一個秩 32 的修正，讓讀和寫的比對不對稱；hidden_dim 降到 662 對齊參數。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[UD2-gate](ud2-gate.md)
- **程式碼**：`scripts/delta.py` 的 `ReadQueryGateAnchorDeltaBlock`，訓練時 `--variant=SDulagq`

## 怎麼做的

- UD2-gate：讀位址 q_t = unit(r_t)，寫入鑰匙 k_t = unit(r_{t−1})，兩個都是同一個 FFN 位址 relu(K u + b)²。
- UD2-gate-q32：q_t = unit(relu(K u_t + b + U V u_t)²)，V：dim → 32，U：32 → 格子數；寫入鑰匙不變。U 從 0 起步，所以一開始就是 UD2-gate。
- hidden_dim 從 712 降到 662 來對齊參數（15,454,428；UD2-gate 15,456,228），所以格子少 7%。

**問題**：42M 輸在窗口裡沒出現過的新 token。統一 block 的位址是 FFN 的位址，比對只能找「像」的，不能找「相關」的。讓讀位址跟寫入鑰匙不一樣，會不會補上新 token？

```text
q_t = unit(relu(K u_t + b + U V u_t)²)，k_s = unit(r_{s−1})
y = μ z_t + o_t · E_tᵀ q_t
```

**設定**：15M（12 層、2,300 步），hidden_dim 662，其他跟 UD2-gate 15M 同設定；1 個 seed（1337），1.9 小時（6,966 秒）；fork session 跑（2026-11-15）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsdulagq15m` | 1 | 15.45M | 3.746 | 3.701 | 4.111 | 3.699 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.663。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：2 / 81 / 84 / 5 / 1 / 1 / 2 / 1 / 2 / 2 / 1 / 2。

15M 按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量（第 0 → 11 層，[每個格子記多久](../experiments/slot-retention.md)）：0% / 42% / 38% / 7% / 1% / 0% / 1% / 43% / 21% / 1% / 14% / 41%。

## 結論

15M（1 個 seed）：NLL 3.746（長度 1,024）、3.701（4,096），比同一個 seed 的 UD2-gate（3.766）好 0.021，格子還少 7%。前 64 個 token 4.111 對 4.119，512–1,024 3.699 對 3.722。讀到 20K 的 16K–20K 3.663（UD2-gate 3.685）。逐 token 對 Transformer：位置 64–256 的新 token −0.029，UD2-gate −0.010、UD2-gate-norm −0.022，是三個裡補最多的，符合「位址只能找相似」的假設，但 UD2-gate-norm 不改位址也補了一大半，還不能說都是位址的功勞。全部 −0.177，見過的 token −0.106。截掉遠處前文，新 token 還是變好 0.014。按寫入量加權，第 1、2、7、8、10、11 層過了 256 個 token 還留 42%／38%／43%／21%／14%／41%。

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
python scripts/train_variant.py --out_dir=outputs/fwsdulagq15m --variant=SDulagq --hidden_dim=662 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsdulagq15m
```
