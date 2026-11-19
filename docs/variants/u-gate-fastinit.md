# U-Gate-fastinit（代號 `SH`）

U-Gate 的對照：U-Avg 只把遺忘率起點從 1/(4L) 調快到 1/64（U-Gate 的門最後停的位置附近），仍是每層一個、可學。

- **類別**：遺忘門
- **基於**：[U-Avg](u-avg.md)
- **程式碼**：`scripts/gated.py` 的 `ShortBlock`，訓練時 `--variant=SH`

## 怎麼做的

- 跟 U-Avg 完全一樣，只改遺忘率的起始值：1/(4L) → 1/64。
- 用來量 U-Gate 的進步有多少只是因為“忘得比較快”。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsh15m`（+2 個 seed） | 3 | 15.17M | 4.051（4.049–4.052） | 4.029 | 4.332 | 4.024 |
| 42M | `fwsh42m` | 1 | 41.63M | 3.361 | 3.327 | 3.713 | 3.320 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.993（3 個 seed）。

## 結論

15M 4.051（3 個 seed，4.049–4.052），拿到 U-Gate 相對 U-Avg 進步的 53.8%；42M 3.361（U-Gate 3.330）。也不再過了訓練長度就崩（長度 4,096 4.030）。記憶任務 m2kv 間隔 1,024 只剩 44.0%：忘得快就記不久。

## 相關實驗

- [記憶的寫法：平均、遺忘門、delta rule](../experiments/write-rule.md)
- [規模：15M → 42M → 110M](../experiments/scale.md)
- [記憶任務 m2kv](../experiments/memory-tasks.md)
- [多 seed 的穩定性](../experiments/seeds.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsh.txt）
python scripts/train_variant.py --out_dir=outputs/fwsh15m --variant=SH --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=32 --gradient_accumulation_steps=4 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsh15m
# 42M（scripts/nrp/fwsh2.txt）
python scripts/train_variant.py --out_dir=outputs/fwsh42m --variant=SH --hidden_dim=1280 --dataset=fineweb --dim=512 --n_layers=16 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=32 --gradient_accumulation_steps=4 --learning_rate=5e-4 --warmup_iters=1000 --max_iters=6400 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsh42m
```
