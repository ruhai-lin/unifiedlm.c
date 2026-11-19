# U-Gate-rank64（代號 `SG64`）

U-Gate 的門低秩從 16 加到 64。

- **類別**：遺忘門
- **基於**：[U-Gate](u-gate.md)
- **程式碼**：`scripts/gated.py` 的 `GatedBlock(rank=64)`，訓練時 `--variant=SG64`

## 怎麼做的

- `Q`: m→64、`P`: 64→m，其他不變。參數多約 0.8M。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsg64_15m` | 1 | 16.28M | 3.977 | 3.942 | 4.299 | 3.938 |

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：29 / 335 / 24 / 22 / 17 / 203 / 58 / 51 / 38 / 35 / 117 / 39。

## 結論

15M 3.977，比 U-Gate 三個 seed 平均（3.998）好 0.021（1 個 seed，參數 16.28M 對 15.46M）。

## 相關實驗

- [記憶的寫法：平均、遺忘門、delta rule](../experiments/write-rule.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwlam.txt）
python scripts/train_variant.py --out_dir=outputs/fwsg64_15m --variant=SG64 --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsg64_15m
```
