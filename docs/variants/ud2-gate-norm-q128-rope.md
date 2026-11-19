# UD2-gate-norm-q128-rope（代號 `UD2-gate-norm-q128-rope`）

三個讀出端的修改一起：讀出 RMSNorm、秩 128 的讀位址修正、一半格子加旋轉相對位置；hidden_dim 538。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[UD2-gate-norm-q32](ud2-gate-norm-q32.md)
- **程式碼**：`scripts/delta.py` 的 `RopeReadNormQuery128GateAnchorDeltaBlock`，訓練時 `--variant=UD2-gate-norm-q128-rope`

## 怎麼做的

- UD2-gate-norm-q32：讀出先 RMSNorm，讀位址加秩 32 的修正，hidden_dim 662。
- UD2-gate-norm-q128-rope：修正的秩加到 128（hidden_dim 538 對齊參數）；再把一半的格子兩兩成對，寫入鑰匙按寫入的位置旋轉、讀位址按讀的位置旋轉（週期 4–256 個 token，對數間隔），同一對的衰減綁在一起，所以比對只看相對距離。旋轉不加參數。
- 參數 15,452,652。

**問題**：三個讀出端的修改合在一起，會不會比 UD2-gate-norm-q32 更好？

```text
q_t = unit(relu(K u_t + b + U V u_t)²)，秩 128
一半格子：k_s ← R(s) k_s，q_t ← R(t) q_t（成對旋轉），同一對共用衰減
y = μ z_t + o_t · RMSNorm(E_tᵀ q_t)
```

**設定**：15M（12 層、2,300 步），hidden_dim 538；1 個 seed（1337），3090 1.6 小時（5,751 秒）；fork session 跑（2026-11-15）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `ud2_gate_norm_q128_rope_15m` | 1 | 15.45M | 3.732 | 3.688 | 4.122 | 3.684 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.651。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：3 / 3 / 2 / 1 / 2 / 1 / 2 / 5 / 5 / 5 / 2 / 2。

15M 按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量（第 0 → 11 層，[每個格子記多久](../experiments/slot-retention.md)）：18% / 8% / 0% / 0% / 1% / 0% / 2% / 17% / 50% / 26% / 1% / 1%。

## 結論

15M（1 個 seed）：NLL 3.732（長度 1,024）、3.688（4,096），比 UD2-gate-norm-q32（3.727、3.683）差 0.005，每一段都差 0.002–0.005（前 64 個 token 4.122 對 4.120，512–1,024 3.684 對 3.678），沒有超過目前 15M 最好的。比 UD2-gate-q128（3.760）好 0.029，那是 RMSNorm 加旋轉一起的效果；旋轉單獨值多少要看 UD2-gate-norm-q32-rope（跟 UD2-gate-norm-q32 只差旋轉，跑中）。秩 128 本身在 UD2-gate-q128 上比秩 32 差，所以這個組合也揹著那個代價。讀到 20K 的 16K–20K 3.651（UD2-gate-norm-q32 3.645）。逐 token 對 Transformer 全部 −0.184，每一類都比 UD2-gate-norm-q32 差，位置 64–256 的新 token −0.021（UD2-gate-norm-q32 −0.039），跟單獨的秩 128 一樣，秩 128 吃掉了新 token 的進步。

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
python scripts/train_variant.py --out_dir=outputs/ud2_gate_norm_q128_rope_15m --variant=UD2-gate-norm-q128-rope --hidden_dim=538 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/ud2_gate_norm_q128_rope_15m
```
