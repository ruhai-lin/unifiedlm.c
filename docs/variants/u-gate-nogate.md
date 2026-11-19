# U-Gate-nogate（代號 `SR`）

U-Gate 的消融：每格各自的遺忘率，沒有門，也沒有寫入時的遺忘項。

- **類別**：遺忘門
- **基於**：[U-Gate](u-gate.md)
- **程式碼**：`scripts/gated.py` 的 `RateBlock`，訓練時 `--variant=SR`

## 怎麼做的

- `log λ_a = expm1(−rate_a)`：每格一個可學的常數速率，不看內容、不隨 token 變。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsr15m` | 1 | 15.18M | 4.124 | 5.886 | 4.399 | 4.101 |

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：2822 / 3108 / 3206 / 3632 / 3099 / 3193 / 3160 / 3348 / 3186 / 3045 / 3076 / 3242。

## 結論

15M 4.124，跟 U-Avg（4.112）一樣（1 個 seed），長度 4,096 照樣崩（5.886）。每格的速率幾乎沒離開起點（有效記憶長度每層約 2,800–3,600 個 token）。U-Gate 的進步來自按內容的門，不是每格各自的速率。

## 相關實驗

- [記憶的寫法：平均、遺忘門、delta rule](../experiments/write-rule.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsr.txt）
python scripts/train_variant.py --out_dir=outputs/fwsr15m --variant=SR --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsr15m
```
