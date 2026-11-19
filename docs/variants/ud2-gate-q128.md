# UD2-gate-q128（代號 `UD2-gate-q128`）

UD2-gate-q32 的讀位址修正從秩 32 加到 128；hidden_dim 降到 538 對齊參數（格子少 19%）。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[UD2-gate-q32](ud2-gate-q32.md)
- **程式碼**：`scripts/delta.py` 的 `ReadQuery128GateAnchorDeltaBlock`，訓練時 `--variant=UD2-gate-q128`

## 怎麼做的

- UD2-gate-q32：q_t = unit(relu(K u_t + b + U V u_t)²)，V：dim → 32，U：32 → 格子數；hidden_dim 662。
- UD2-gate-q128：同樣的修正，秩 128；為了對齊參數，hidden_dim 從 662 降到 538（格子少 19%，比 UD2-gate 的 712 少 24%）。
- 參數 15,449,196；其他跟 UD2-gate-q32 一樣。

**問題**：讀位址修正的秩加大，新 token 會不會補更多？

```text
q_t = unit(relu(K u_t + b + U V u_t)²)，V：dim → 128，U：128 → 538
```

**設定**：15M（12 層、2,300 步），hidden_dim 538；1 個 seed（1337），3090 1.3 小時（4,643 秒）；fork session 跑（2026-11-15）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `ud2_gate_q128_15m` | 1 | 15.45M | 3.760 | 3.715 | 4.130 | 3.713 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.673。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：3 / 102 / 45 / 3 / 2 / 1 / 2 / 3 / 3 / 2 / 2 / 3。

15M 按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量（第 0 → 11 層，[每個格子記多久](../experiments/slot-retention.md)）：0% / 37% / 37% / 1% / 0% / 1% / 1% / 19% / 28% / 2% / 2% / 34%。

## 結論

15M（1 個 seed）：NLL 3.760（長度 1,024）、3.715（4,096），比秩 32 的 UD2-gate-q32（3.746）差 0.015，只比沒有修正的 UD2-gate（3.766）好 0.006。前 64 個 token 4.130（UD2-gate-q32 4.111、UD2-gate 4.119），512–1,024 3.713（UD2-gate-q32 3.699、UD2-gate 3.722）。秩加大要拿格子換（662 → 538），看起來格子少的代價大過修正多出的容量；但兩個改變是綁在一起的，這次分不開。讀到 20K 的 16K–20K 3.673（UD2-gate-q32 3.663）。逐 token 每一類都比 UD2-gate-q32 差：全部對 Transformer −0.161（UD2-gate-q32 −0.177），位置 64–256 的新 token −0.006（UD2-gate-q32 −0.029），秩 32 補的那一塊幾乎沒了。截掉遠處前文，新 token 變好 0.009。按寫入量加權，第 1、2、7、8、11 層過了 256 個 token 還留 37%／37%／19%／28%／34%。

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
python scripts/train_variant.py --out_dir=outputs/ud2_gate_q128_15m --variant=UD2-gate-q128 --hidden_dim=538 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/ud2_gate_q128_15m
```
