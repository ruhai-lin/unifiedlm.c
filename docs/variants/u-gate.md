# U-Gate（代號 `SG`）

U-Avg 加上按內容決定的遺忘門：每個格子、每個 token 的遺忘率由位址算出。

- **類別**：遺忘門
- **基於**：[U-Avg](u-avg.md)
- **程式碼**：`scripts/gated.py` 的 `GatedBlock`、`gated_memory`，訓練時 `--variant=SG`

## 怎麼做的

- U-Avg 每層只有一個遺忘率。U-Gate 改成 `log λ_{t,a} = expm1(−x_{t,a})`，`x_{t,a} = rate_a · g_{t,a} + β · k_{t,a}`。
- `rate_a`：每格一個可學的基礎速率，起點等於 U-Avg 的速率。
- `g_t = softplus(P Q rms(r_t) + b) / ln 2`：從位址 `r_t` 經過低秩 16 的瓶頸（`Q`: m→16，`P`: 16→m）讀出的門，所以選格子的位址也決定每格忘多少；初始化時 `g ≈ 1`，起點幾乎就是 U-Avg。
- `β · k_t`：正在被寫的格子先忘一點（delta rule 的味道），`β` 起點 e⁻⁷。
- `expm1(−x)` 把 `log λ` 限制在 (−1, 0]：每個 token 最多忘 63%。狀態更新跟 U-Avg 一樣，只是 `λ` 變成每格、每 token。

```text
log lam_{t,a} = expm1(-x_{t,a}),   x_{t,a} = rate_a g_{t,a} + beta k_{t,a}
g_t = softplus(P Q rms(r_t) + b) / ln 2
U_t = lam_t * U_{t-1} + (1 - lam_t) * mu w + v_t k_t^T,   c_t = lam_t * c_{t-1} + (1 - lam_t) * w + k_t
y_t = U_t (r_t / c_t)
```

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsg15m`（+2 個 seed） | 3 | 15.46M | 3.998（3.988–4.003） | 3.963 | 4.315 | 3.959 |
| 42M | `fwsg42m`（+2 個 seed） | 3 | 42.33M | 3.330（3.328–3.333） | 3.288 | 3.694 | 3.285 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.925（3 個 seed）。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：39 / 342 / 39 / 48 / 48 / 121 / 45 / 52 / 131 / 191 / 39 / 82。

## 結論

15M 3.998（3 個 seed，3.988–4.003），比 U-Avg 好 0.114，但還輸 Transformer 0.053；42M 3.330（3 個 seed，3.328–3.333），輸 0.116。一半以上只是“忘得比較快”：只把 U-Avg 的遺忘率起點調快的 [U-Gate-fastinit](u-gate-fastinit.md) 拿到 53.8% 的進步。過了訓練長度不崩（長度 4,096 3.968）。記憶任務 m2kv 間隔 1,024 98.5%，4,096 39.6%。門的形狀（低秩 64、從 h 讀、加 SiLU）都只差一點，見 [記憶的寫法](../experiments/write-rule.md)。

## 相關實驗

- [記憶的寫法：平均、遺忘門、delta rule](../experiments/write-rule.md)
- [規模：15M → 42M → 110M](../experiments/scale.md)
- [記憶任務 m2kv](../experiments/memory-tasks.md)
- [長上下文與外推（讀到 20K）](../experiments/extrapolation.md)
- [多 seed 的穩定性](../experiments/seeds.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsg.txt）
python scripts/train_variant.py --out_dir=outputs/fwsg15m --variant=SG --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsg15m
# 42M（scripts/nrp/fwsg.txt）
python scripts/train_variant.py --out_dir=outputs/fwsg42m --variant=SG --hidden_dim=1280 --dataset=fineweb --dim=512 --n_layers=16 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=8 --gradient_accumulation_steps=16 --learning_rate=5e-4 --warmup_iters=1000 --max_iters=6400 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsg42m
```
