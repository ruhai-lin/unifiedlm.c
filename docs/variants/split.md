# Split（代號 `P`）

統一的對照：靜態格子和上下文格子用各自的位址，兩組格子不共享。

- **類別**：統一的對照
- **基於**：[U-Avg](u-avg.md)
- **程式碼**：`scripts/variants.py` 的 `SeparateBlock`，訓練時 `--variant=P`

## 怎麼做的

- 每個 block 有兩組格子：`m_s` 個靜態格子（FFN）和 `m_f` 個上下文格子（記憶），各有自己的位址投影 `r_s`、`r_f`。
- `y = μ(r_s) + fast(r_f)`；記憶的寫法跟 U-Avg 一樣（累計平均、每層一個遺忘率）。
- 這裡用 `m_s = m_f = 712`（跟 U-Avg 的格子數一樣），所以參數比 U-Avg 多：15M 17.64M 對 15.17M，42M 52.1M 對 41.6M。

```text
y_t = mu r_s,t + fast(r_f,t)   (two address projections, two slot sets)
```

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwp15m`（+2 個 seed） | 3 | 17.64M | 4.038（4.034–4.040） | 6.233 | 4.346 | 4.003 |
| 42M | `fwp42m` | 1 | 52.14M | 3.368 | 7.561 | 3.723 | 3.324 |

## 結論

15M 4.038（3 個 seed，4.034–4.040），比 U-Avg（4.112）好 0.074；42M 3.368 對 3.423。但參數多 16–25%，所以這不是同參數的比較。兩者過了訓練長度都崩（長度 4,096：Split 5.944–6.721）。U-Delta 系列還沒做同樣的拆分對照，所以“共享位址不吃虧”在 U-Delta 上還沒有直接證據。

## 相關實驗

- [統一的對照：共享位址 vs 分開位址](../experiments/unification.md)
- [多 seed 的穩定性](../experiments/seeds.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwp.txt）
python scripts/train_variant.py --out_dir=outputs/fwp15m --variant=P --m_s=712 --m_f=712 --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=32 --gradient_accumulation_steps=4 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwp15m
# 42M（scripts/nrp/fwp.txt）
python scripts/train_variant.py --out_dir=outputs/fwp42m --variant=P --m_s=1280 --m_f=1280 --hidden_dim=1280 --dataset=fineweb --dim=512 --n_layers=16 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=1000 --max_iters=6400 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwp42m
```
