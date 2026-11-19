# UD2-gate-norm-q32-decay-ev（代號 `UD2-gate-norm-q32-decay-ev`）

UD2-gate-norm-q32-decay 加 embedding 寫入值；兩個進步完全疊加。每層同一種 block、不用旋轉相對位置、沒有 ×1000 學習率倍數的版本中 15M 最好（1 個 seed）。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[UD2-gate-norm-q32-decay](ud2-gate-norm-q32-decay.md)
- **程式碼**：`scripts/delta.py` 的 `EmbedValueSpectrumReadNormQueryGateAnchorDeltaBlock`（`scripts/variants.py` 裡的 `UD2-gate-norm-q32-decay-ev`），訓練時 `--variant=UD2-gate-norm-q32-decay-ev`

## 怎麼做的

- UD2-gate-norm-q32-decay 加 embedding 寫入值；兩個進步完全疊加。每層同一種 block、不用旋轉相對位置、沒有 ×1000 學習率倍數的版本中 15M 最好（1 個 seed）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `ud2_gate_norm_q32_decay_ev_15m` | 1 | 15.45M | 3.712 | 3.668 | 4.106 | 3.662 |
| 42M | `ud2_gate_norm_q32_decay_ev_42m` | 1 | 41.76M | 3.167 | 3.117 | 3.584 | 3.113 |

15M 按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量（第 0 → 11 層，[每個格子記多久](../experiments/slot-retention.md)）：28% / 15% / 1% / 21% / 20% / 4% / 49% / 8% / 43% / 52% / 29% / 37%。

42M 按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量（第 0 → 15 層，[每個格子記多久](../experiments/slot-retention.md)）：9% / 29% / 2% / 7% / 3% / 10% / 32% / 14% / 5% / 35% / 29% / 29% / 49% / 45% / 48% / 44%。

## 結論

UD2-gate-norm-q32-decay 加 embedding 寫入值；兩個進步完全疊加。每層同一種 block、不用旋轉相對位置、沒有 ×1000 學習率倍數的版本中 15M 最好（1 個 seed）。

## 相關實驗

- [讀出端：門、縮放、RMSNorm、讀位址修正、旋轉相對位置](../experiments/read-side.md)
- [逐 token 差距：輸在哪些 token](../experiments/token-gap.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdfk.txt）
python scripts/train_variant.py --out_dir=outputs/ud2_gate_norm_q32_decay_ev_15m --variant=UD2-gate-norm-q32-decay-ev --hidden_dim=662 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/ud2_gate_norm_q32_decay_ev_15m
# 42M（scripts/nrp/fwsdfk.txt）
python scripts/train_variant.py --out_dir=outputs/ud2_gate_norm_q32_decay_ev_42m --variant=UD2-gate-norm-q32-decay-ev --hidden_dim=1195 --dim=512 --n_layers=16 --batch_size=8 --gradient_accumulation_steps=16 --warmup_iters=1000 --max_iters=6400 --dataset=fineweb --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --learning_rate=5e-4 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/ud2_gate_norm_q32_decay_ev_42m
```
