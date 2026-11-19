# U-Delta-diverseλ（代號 `SDv`）

U-Delta 的基礎記憶長度從 4 到 65,536 個 token 分散，門最多只能讓遺忘快 2 倍，所以慢格子留得住。

- **類別**：記憶設計
- **基於**：[U-Delta](u-delta.md)
- **程式碼**：`scripts/delta.py` 的 `DiverseRateDeltaBlock`，訓練時 `--variant=SDv`

## 怎麼做的

- 初始記憶長度範圍從 U-Delta-spreadλ 的 256–16,384 放寬到 4–65,536 個 token。
- gate 從 softplus（沒有上限）改成 2·sigmoid，值在 0 到 2 之間：可以讓 slot 記得更久（g→0 就完全不忘），但最多只能忘快 2 倍。初始 g＝1，跟 U-Delta 一樣。
- 原因：U-Delta-spreadλ 的 gate 平均把慢 slot 的遺忘加快約 25–30 倍，初始記上萬 token 的 slot 實際只記約 350 個，分散被 gate 收回去了。
- 代價：慢 slot 在文件開頭也沒辦法很快清空。
- 參數、狀態跟 U-Delta 一樣。

**問題**：λ 真的很分散（從記幾個 token 到記幾萬個）時，單一記憶能不能同時拿到 U-Delta-spreadλ 在一般文本上的好處和 U-Delta+slow 在記憶題上撐得久的好處？還是 gate 被限制住反而變差？

```text
λ_{t,a} = exp(−rate_a · g_{t,a})，g = 2·sigmoid(Split Q rms(r_t)) ∈ (0, 2)
rate_a 初始從 1/4 到 1/(64 L) 對數均勻（U-Delta：全部 1/(4L)，g = softplus(·)/ln 2）
```

**設定**：參數：15.45M（同 U-Delta）；狀態：2.46M（同 U-Delta）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsdv15m` | 1 | 15.45M | 3.877 | 3.845 | 4.245 | 3.832 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.859。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：17 / 29 / 47 / 18 / 18 / 18 / 19 / 20 / 20 / 20 / 39 / 29。

## 結論

分散留下來了，NLL 小幅變好：3.877（長度 1,024）、3.845（4,096），比 U-Delta 三個 seed（3.895–3.901）好約 0.021，比 U-Delta-spreadλ（3.890）、U-Delta-λlr10（3.887）好，但不如把記憶縮短的 U-Delta-λlr100（3.851）、U-Delta-λlr1000（3.843）（1 個 seed）。訓練後每層的基礎記憶長度幾乎沒動：p10 約 9、中位數約 430–510、p90 約 21,000–23,000 個 token，跟初始一樣分散；gate 最多只能忘快 2 倍，所以慢 slot 真的留著。有效記憶長度每層都是 17–47 個 token（被快 slot 主導），中間層跟 U-Delta-λlr1000 一樣短，所以進步可能來自一開始就有很快的 slot，而不是慢 slot。 讀到 20K：2K 以內每段比 U-Delta 好約 0.02，2K–4K 打平，4K 之後變差，16K–20K 3.859，比 U-Delta 三個 seed（3.810–3.820）差約 0.04，跟完全不忘的 U-Delta+slow（3.856）一樣。兩個保留慢 slot 的做法（U-Delta+slow、U-Delta-diverseλ）讀超過 4K 都變差，把記憶縮短的 U-Delta-λlr100、U-Delta-λlr1000 反而越讀越好。

## 相關實驗

- [記憶設計：快記憶、慢記憶、遺忘率](../experiments/memory-design.md)
- [長上下文與外推（讀到 20K）](../experiments/extrapolation.md)
- [遠處的前文值多少](../experiments/reach.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdv.txt）
python scripts/train_variant.py --out_dir=outputs/fwsdv15m --variant=SDv --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsdv15m
```
