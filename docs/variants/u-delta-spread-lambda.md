# U-Delta-spreadλ（代號 `SDw`）

U-Delta 的遺忘率初始值分散在 1/(L/4)～1/(16L)：同一份記憶裡有快有慢。

- **類別**：記憶設計
- **基於**：[U-Delta](u-delta.md)
- **程式碼**：`scripts/delta.py` 的 `WideRateDeltaBlock`，訓練時 `--variant=SDw`

## 怎麼做的

- U-Delta 的每個 slot 基礎遺忘率都從 1/(4L) 開始（記憶約 4,096 個 token）。
- U-Delta-spreadλ 改成在 slot 之間以對數均勻分散，從 1/(L/4) 到 1/(16L)：最快的 slot 約 256 個 token，最慢的約 16,384 個（跟 U-Delta+slow 慢記憶的初始值一樣）。
- 其他（依內容的 gate、delta rule、β）完全不變，參數和狀態都跟 U-Delta 一樣。

**問題**：U-Delta+fast2 顯示 U-Delta+slow 的進步不是容量，而是那份「幾乎不忘、很少寫」的慢記憶；U-Delta+slow 診斷也顯示慢記憶的速率停在初始值沒動。如果 U-Delta-spreadλ 追上 U-Delta+slow（3.888），就代表統一 block 維持單一記憶、只要把 λ 初始值設好就夠了。

```text
log_rate_a = linspace(log(4/L), log(1/(16L)), m)   # U-Delta：全部 = log(1/(4L))
λ_{t,a} = exp(−exp(log_rate_a) · g_{t,a})，g 是依內容決定的 gate
```

**設定**：參數：15.45M（同 U-Delta）；狀態：2.46M（同 U-Delta，U-Delta+slow 是 4.92M）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsdw15m` | 1 | 15.45M | 3.890 | 3.848 | 4.254 | 3.843 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.807。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：7 / 252 / 145 / 331 / 17 / 15 / 19 / 23 / 26 / 15 / 20 / 285。

## 結論

單一記憶就追平 U-Delta+slow：U-Delta-spreadλ 3.890，比 U-Delta（3.897，同一個 seed；U-Delta 三個 seed 3.895–3.901）好 0.007，跟 U-Delta+slow（3.888）只差 0.001，而 U-Delta+slow 多一份記憶、狀態是兩倍。4,096 評估 2K–4K 3.833；讀到 20K 16K–20K 段 3.807，U-Delta+slow 在這段變差到 3.856。但進步不是來自慢 slot：U-Delta-spreadλ 訓練完基礎記憶長度 p10／中位數／p90 約 330／1,900／10,000，跟初始值差不多，慢 slot 一直都在；對照組 U-Delta-λlr10（從 U-Delta 的單一初始值出發、log_rate 學快 10 倍）把記憶縮短、沒有慢 slot，結果在 1,024 內和讀到 20K 都跟 U-Delta-spreadλ 一樣。兩者共同的地方是中間層（第 6–10 層）記得較短：有效記憶長度約 15–31 個 token，U-Delta 是 65–95。1 個 seed；要不要補 seed 在「待決定」。

## 相關實驗

- [記憶設計：快記憶、慢記憶、遺忘率](../experiments/memory-design.md)
- [內容是被遺忘還是被覆寫](../experiments/forgetting-vs-overwrite.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdx.txt）
python scripts/train_variant.py --out_dir=outputs/fwsdw15m --variant=SDw --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsdw15m
```
