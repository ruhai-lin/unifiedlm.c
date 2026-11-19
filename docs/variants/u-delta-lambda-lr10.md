# U-Delta-λlr10（代號 `SDl`）

U-Delta 的每格基礎遺忘率 `log_rate` 有效學習率 ×10（重參數化，起點不變）。

- **類別**：記憶設計
- **基於**：[U-Delta](u-delta.md)
- **程式碼**：`scripts/delta.py` 的 `FastRateDeltaBlock`，訓練時 `--variant=SDl`

## 怎麼做的

- log_rate 改存成 θ＝log_rate／10，用的時候乘回 10。Adam 每步大約移動 lr 個單位（1 維參數沒有 weight decay），所以 log_rate 每步移動變成 10 倍；已在 CPU 上量過，正好 10 倍。
- 初始值跟 U-Delta 一樣：全部 slot 從記 4,096 個 token 開始。初始輸出跟 U-Delta 完全相同。
- 參數、狀態跟 U-Delta 一樣。

**問題**：U-Delta-spreadλ 追平 U-Delta+slow，但它的基礎遺忘率在 2,300 步內幾乎沒動，等於靠初始值。U-Delta 的 λ 本來就能學，是學不出慢 slot，還是只是學得太慢？

```text
λ = exp(−exp(10·θ) · g(u))，θ 初始 = log(1/(4L))／10
其他同 U-Delta
```

**設定**：參數：15.45M（同 U-Delta）；狀態：2.46M（同 U-Delta）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsdl15m` | 1 | 15.45M | 3.887 | 3.846 | 4.251 | 3.841 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.806。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：4 / 254 / 209 / 296 / 173 / 6 / 18 / 31 / 25 / 22 / 23 / 254。

## 結論

學得出來，但方向跟預期相反：U-Delta 沒有學出慢 slot，而是把記憶縮短。NLL 3.887（長度 1,024）、3.846（4,096），比 U-Delta 三個 seed（3.895–3.901）都好，跟 U-Delta-spreadλ（3.890）、U-Delta+slow（3.888）同級（1 個 seed）。基礎記憶長度中位數從 4,096 學成 667–3,332 個 token，p90 都低於 4,040；有效記憶長度在中間層（第 6–10 層）從 U-Delta 的 65–95 降到 18–31，跟 U-Delta-spreadλ 一樣短。所以 U-Delta-spreadλ 的進步可能主要來自記得較短，不一定是慢 slot；讀到 20K 確認了這點：U-Delta-λlr10 跟 U-Delta-spreadλ 從第 128 個 token 起每段差不到 0.004（16K–20K 3.806 對 3.807，U-Delta 三個 seed 3.810–3.820，U-Delta+slow 3.856）。U-Delta-spreadλ 有記上萬 token 的 slot、U-Delta-λlr10 沒有，結果一樣，所以慢 slot 沒用上。

## 相關實驗

- [記憶設計：快記憶、慢記憶、遺忘率](../experiments/memory-design.md)
- [內容是被遺忘還是被覆寫](../experiments/forgetting-vs-overwrite.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdl.txt）
python scripts/train_variant.py --out_dir=outputs/fwsdl15m --variant=SDl --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsdl15m
```
