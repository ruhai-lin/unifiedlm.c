# UD2-gate-norm-q32-decay（代號 `UD2-gate-norm-q32-decay`）

不加旋轉相對位置；每個格子的衰減速率固定不學（buffer），g=1 時半衰期從 2 到 8,192 個 token 等比分佈，門改成 2σ（最多加速 2 倍）。哪些格子還記得就編碼了過了多久，又不會把內容轉掉。參數少 7,944 個。15M 好 0.006（1 個 seed），照抄是所有版本中最好的。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[UD2-gate-norm-q32](ud2-gate-norm-q32.md)
- **程式碼**：`scripts/delta.py` 的 `SpectrumReadNormQueryGateAnchorDeltaBlock`（`scripts/variants.py` 裡的 `UD2-gate-norm-q32-decay`），訓練時 `--variant=UD2-gate-norm-q32-decay`

## 怎麼做的

- 不加旋轉相對位置；每個格子的衰減速率固定不學（buffer），g=1 時半衰期從 2 到 8,192 個 token 等比分佈，門改成 2σ（最多加速 2 倍）。哪些格子還記得就編碼了過了多久，又不會把內容轉掉。參數少 7,944 個。15M 好 0.006（1 個 seed），照抄是所有版本中最好的。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `ud2_gate_norm_q32_decay_15m` | 1 | 15.45M | 3.721 | 3.679 | 4.113 | 3.673 |

15M 按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量（第 0 → 11 層，[每個格子記多久](../experiments/slot-retention.md)）：27% / 15% / 1% / 20% / 20% / 4% / 49% / 8% / 43% / 61% / 34% / 38%。

## 結論

不加旋轉相對位置；每個格子的衰減速率固定不學（buffer），g=1 時半衰期從 2 到 8,192 個 token 等比分佈，門改成 2σ（最多加速 2 倍）。哪些格子還記得就編碼了過了多久，又不會把內容轉掉。參數少 7,944 個。15M 好 0.006（1 個 seed），照抄是所有版本中最好的。

## 相關實驗

- [讀出端：門、縮放、RMSNorm、讀位址修正、旋轉相對位置](../experiments/read-side.md)
- [逐 token 差距：輸在哪些 token](../experiments/token-gap.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdfk.txt）
python scripts/train_variant.py --out_dir=outputs/ud2_gate_norm_q32_decay_15m --variant=UD2-gate-norm-q32-decay --hidden_dim=662 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/ud2_gate_norm_q32_decay_15m
```
