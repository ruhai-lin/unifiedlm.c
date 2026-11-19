# UD2-gate-norm-q32-ropelo（代號 `UD2-gate-norm-q32-ropelo`）

UD2-gate-norm-q32-rope 只在下半的 block（15M 12 個裡的 6 個）加旋轉相對位置，上半不加；參數一樣。15M 比全部加好 0.006（1 個 seed），進步在遠處照抄；42M 跟全部加、完全不加一樣（3.171）。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[UD2-gate-norm-q32-rope](ud2-gate-norm-q32-rope.md)
- **程式碼**：`scripts/delta.py` 的 `RopeReadNormQueryGateAnchorDeltaBlock`（`scripts/variants.py` 裡的 `UD2-gate-norm-q32-ropelo`），訓練時 `--variant=UD2-gate-norm-q32-ropelo`

## 怎麼做的

- UD2-gate-norm-q32-rope 只在下半的 block（15M 12 個裡的 6 個）加旋轉相對位置，上半不加；參數一樣。15M 比全部加好 0.006（1 個 seed），進步在遠處照抄；42M 跟全部加、完全不加一樣（3.171）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `ud2_gate_norm_q32_ropelo_15m` | 1 | 15.46M | 3.711 | 3.666 | 4.106 | 3.661 |
| 42M | `ud2_gate_norm_q32_ropelo_42m` | 1 | 41.77M | 3.171 | 3.121 | 3.584 | 3.118 |

## 結論

UD2-gate-norm-q32-rope 只在下半的 block（15M 12 個裡的 6 個）加旋轉相對位置，上半不加；參數一樣。15M 比全部加好 0.006（1 個 seed），進步在遠處照抄；42M 跟全部加、完全不加一樣（3.171）。

## 相關實驗

- [讀出端：門、縮放、RMSNorm、讀位址修正、旋轉相對位置](../experiments/read-side.md)
- [逐 token 差距：輸在哪些 token](../experiments/token-gap.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdfk.txt）
python scripts/train_variant.py --out_dir=outputs/ud2_gate_norm_q32_ropelo_15m --variant=UD2-gate-norm-q32-ropelo --hidden_dim=662 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/ud2_gate_norm_q32_ropelo_15m
# 42M（scripts/nrp/fwsdfk.txt）
python scripts/train_variant.py --out_dir=outputs/ud2_gate_norm_q32_ropelo_42m --variant=UD2-gate-norm-q32-ropelo --hidden_dim=1195 --dim=512 --n_layers=16 --batch_size=8 --gradient_accumulation_steps=16 --warmup_iters=1000 --max_iters=6400 --dataset=fineweb --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --learning_rate=5e-4 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/ud2_gate_norm_q32_ropelo_42m
```
