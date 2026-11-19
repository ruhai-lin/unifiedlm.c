# U-Delta-docreset（代號 `SDb`）

U-Delta 遇到文件開頭（BOS）就清空記憶。

- **類別**：記憶設計
- **基於**：[U-Delta](u-delta.md)
- **程式碼**：`scripts/delta.py` 的 `DocResetDeltaBlock`、`doc_reset_forward`，訓練時 `--variant=SDb`

## 怎麼做的

- U-Delta 的遺忘 gate 只學會在文件開頭多忘一點（大部分層是平常的 1–4 倍），慢 slot 會把上一篇帶到下一篇。
- U-Delta-docreset 在 BOS 那一步讓記憶歸零：λ＝0，而且那一步不寫入（β＝0），從下一個 token 開始寫新文件。
- 卷積（看前 3 個 token）沒有清空。
- 實作：分塊計算裡把跨過 BOS 的項遮掉；已跟逐 token 的參考算法比過，誤差 4e-16。
- 沒有新參數，參數、狀態跟 U-Delta 一樣。

**問題**：跨文件的記憶是雜訊嗎？明確清空之後，gate 不必學這件事，NLL 會不會變好？之後如果要用很慢的 slot（U-Delta-diverseλ），這個清空是不是必要的？

```text
E_t = 0（x_t = BOS）
否則 E_t = (I − β_t k_t k_tᵀ) diag(λ_t) E_{t−1} + β_t k_t u_tᵀ（同 U-Delta）
```

**設定**：參數：15.45M（同 U-Delta）；狀態：2.46M（同 U-Delta）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsdb15m` | 1 | 15.45M | 3.890 | 3.843 | 4.261 | 3.842 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.797。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：12 / 572 / 244 / 650 / 33 / 51 / 70 / 91 / 92 / 78 / 66 / 344。

## 結論

小幅變好：NLL 3.890（長度 1,024）、3.843（4,096）；U-Delta 三個 seed 是 3.895–3.901、3.854–3.860，所以好約 0.008、0.014，跟 U-Delta-spreadλ、U-Delta-λlr10 同級（1 個 seed）。長度 4,096 的 1K–4K 段好 0.014–0.016，比 1,024 內多，表示上一篇文件留在記憶裡的東西確實有點干擾。清空之後模型讓少數層記更久：第 1、3 層 572、650 個 token（U-Delta 373、434），第 4 層反而變短（33，U-Delta 259）。讀到 20K 時好得更多：1,024 內跟 U-Delta 三個 seed 重疊，2K 之後每段都比三個 seed 好 0.01–0.02，16K–20K 3.797（U-Delta 3.810–3.820，U-Delta-λlr10 3.806，U-Delta-spreadλ 3.807），是不改 FFN 的 U-Delta 變體裡最好的。文件越多、越遠，清空越有用。

## 相關實驗

- [記憶設計：快記憶、慢記憶、遺忘率](../experiments/memory-design.md)
- [長上下文與外推（讀到 20K）](../experiments/extrapolation.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdu.txt）
python scripts/train_variant.py --out_dir=outputs/fwsdb15m --variant=SDb --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsdb15m
```
