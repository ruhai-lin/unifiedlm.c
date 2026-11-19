# UD2-gate-norm-q32-ev（代號 `UD2-gate-norm-q32-ev`）

不加旋轉相對位置的 UD2-gate-norm-q32 加上 embedding 寫入值（同 UD2-gate-norm-q32-rope-ev）。15M 好 0.0095（1 個 seed），跟加在有旋轉相對位置的版本上一樣。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[UD2-gate-norm-q32](ud2-gate-norm-q32.md)
- **程式碼**：`scripts/delta.py` 的 `EmbedValueReadNormQueryGateAnchorDeltaBlock`（`scripts/variants.py` 裡的 `UD2-gate-norm-q32-ev`），訓練時 `--variant=UD2-gate-norm-q32-ev`

## 怎麼做的

- 不加旋轉相對位置的 UD2-gate-norm-q32 加上 embedding 寫入值（同 UD2-gate-norm-q32-rope-ev）。15M 好 0.0095（1 個 seed），跟加在有旋轉相對位置的版本上一樣。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `ud2_gate_norm_q32_ev_15m` | 1 | 15.46M | 3.718 | 3.672 | 4.117 | 3.668 |

## 結論

不加旋轉相對位置的 UD2-gate-norm-q32 加上 embedding 寫入值（同 UD2-gate-norm-q32-rope-ev）。15M 好 0.0095（1 個 seed），跟加在有旋轉相對位置的版本上一樣。

## 相關實驗

- [讀出端：門、縮放、RMSNorm、讀位址修正、旋轉相對位置](../experiments/read-side.md)
- [逐 token 差距：輸在哪些 token](../experiments/token-gap.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdfk.txt）
python scripts/train_variant.py --out_dir=outputs/ud2_gate_norm_q32_ev_15m --variant=UD2-gate-norm-q32-ev --hidden_dim=662 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/ud2_gate_norm_q32_ev_15m
```
