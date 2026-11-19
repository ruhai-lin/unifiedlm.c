# UD2-gate-norm-q32-conv8（代號 `UD2-gate-norm-q32-conv8`）

不加旋轉相對位置，卷積從看 4 個 token 改成 8 個；多 13,824 個參數。15M 跟不改一樣（1 個 seed）：旋轉相對位置補的不是 4–8 個 token 內的詞序。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[UD2-gate-norm-q32](ud2-gate-norm-q32.md)
- **程式碼**：`scripts/delta.py` 的 `ReadNormQueryGateAnchorDeltaBlock`（`scripts/variants.py` 裡的 `UD2-gate-norm-q32-conv8`），訓練時 `--variant=UD2-gate-norm-q32-conv8`

## 怎麼做的

- 不加旋轉相對位置，卷積從看 4 個 token 改成 8 個；多 13,824 個參數。15M 跟不改一樣（1 個 seed）：旋轉相對位置補的不是 4–8 個 token 內的詞序。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `ud2_gate_norm_q32_conv8_15m` | 1 | 15.47M | 3.728 | 3.682 | 4.121 | 3.678 |

## 結論

不加旋轉相對位置，卷積從看 4 個 token 改成 8 個；多 13,824 個參數。15M 跟不改一樣（1 個 seed）：旋轉相對位置補的不是 4–8 個 token 內的詞序。

## 相關實驗

- [讀出端：門、縮放、RMSNorm、讀位址修正、旋轉相對位置](../experiments/read-side.md)
- [逐 token 差距：輸在哪些 token](../experiments/token-gap.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdfk.txt）
python scripts/train_variant.py --out_dir=outputs/ud2_gate_norm_q32_conv8_15m --variant=UD2-gate-norm-q32-conv8 --conv_size=8 --hidden_dim=662 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/ud2_gate_norm_q32_conv8_15m
```
