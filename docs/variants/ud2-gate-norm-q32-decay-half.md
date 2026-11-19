# UD2-gate-norm-q32-decay-half（代號 `UD2-gate-norm-q32-decay-half`）

每個 block 一半格子用 decay 的固定半衰期、一半照原本可學（×1000、softplus 門）。15M 跟 decay 一樣（1 個 seed）：全部固定就夠。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[UD2-gate-norm-q32-decay](ud2-gate-norm-q32-decay.md)
- **程式碼**：`scripts/delta.py` 的 `HalfSpectrumReadNormQueryGateAnchorDeltaBlock`（`scripts/variants.py` 裡的 `UD2-gate-norm-q32-decay-half`），訓練時 `--variant=UD2-gate-norm-q32-decay-half`

## 怎麼做的

- 每個 block 一半格子用 decay 的固定半衰期、一半照原本可學（×1000、softplus 門）。15M 跟 decay 一樣（1 個 seed）：全部固定就夠。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `ud2_gate_norm_q32_decay_half_15m` | 1 | 15.45M | 3.722 | 3.680 | 4.113 | 3.674 |

15M 按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量（第 0 → 11 層，[每個格子記多久](../experiments/slot-retention.md)）：30% / 17% / 1% / 11% / 3% / 3% / 48% / 12% / 4% / 44% / 37% / 34%。

## 結論

每個 block 一半格子用 decay 的固定半衰期、一半照原本可學（×1000、softplus 門）。15M 跟 decay 一樣（1 個 seed）：全部固定就夠。

## 相關實驗

- [讀出端：門、縮放、RMSNorm、讀位址修正、旋轉相對位置](../experiments/read-side.md)
- [逐 token 差距：輸在哪些 token](../experiments/token-gap.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdfk.txt）
python scripts/train_variant.py --out_dir=outputs/ud2_gate_norm_q32_decay_half_15m --variant=UD2-gate-norm-q32-decay-half --hidden_dim=662 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/ud2_gate_norm_q32_decay_half_15m
```
