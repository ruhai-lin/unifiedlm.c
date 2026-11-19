# U-Gate-fromh（代號 `SGh`）

U-Gate 的門改從 block 輸入 h 讀（原本從位址 r 讀）。

- **類別**：遺忘門
- **基於**：[U-Gate](u-gate.md)
- **程式碼**：`scripts/gated.py` 的 `GatedBlock(gate_from="h")`，訓練時 `--variant=SGh`

## 怎麼做的

- `g_t = softplus(P Q rms(h_t) + b) / ln 2`，其他不變。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsgh_15m` | 1 | 15.38M | 4.012 | 3.978 | 4.325 | 3.974 |

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：56 / 263 / 74 / 65 / 143 / 253 / 88 / 147 / 69 / 221 / 56 / 117。

## 結論

15M 4.012，比 U-Gate（3.998）略差（1 個 seed）：從位址讀門就夠了。

## 相關實驗

- [記憶的寫法：平均、遺忘門、delta rule](../experiments/write-rule.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwlam.txt）
python scripts/train_variant.py --out_dir=outputs/fwsgh_15m --variant=SGh --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsgh_15m
```
