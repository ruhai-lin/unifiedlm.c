# U-Delta+fast2（代號 `SD2c`）

U-Delta+slow 的同容量對照：第二份記憶也是普通的快記憶，狀態大小相同。

- **類別**：記憶設計
- **基於**：[U-Delta+slow](u-delta-plus-slow.md)
- **程式碼**：`scripts/delta.py` 的 `TwoFastDeltaBlock`，訓練時 `--variant=SD2c`

## 怎麼做的

- U-Delta+slow：一份 U-Delta 記憶（依內容的 gate、從 1/(4L) 開始）＋一份慢記憶（固定速率、從 1/(16L) 開始），用 ρ 混合。
- U-Delta+fast2：兩份都是 U-Delta 記憶，各自有 gate 和寫入強度 β，一樣用 ρ 混合。
- 狀態都是 4.92M；U-Delta+fast2 參數多 1.8%（第二份 gate）。

**問題**：U-Delta+slow 比 U-Delta 好 0.009，但它同時多了一份狀態。同容量對照能分開兩種解釋。

```text
y = μ r + (1 − ρ) E₁ᵀ q + ρ E₂ᵀ q,   ρ = sigmoid(W_ρ h)
U-Delta+slow：E₂ 的遺忘率固定、很慢；U-Delta+fast2：E₂ 跟 E₁ 一樣由內容決定
```

**設定**：參數：15.75M（U-Delta+slow 15.47M）；狀態：4.92M（同 U-Delta+slow）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsd2c_15m` | 1 | 15.75M | 3.897 | 3.853 | 4.270 | 3.849 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.814。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：9 / 213 / 256 / 1643 / 255 / 99 / 175 / 313 / 254 / 117 / 91 / 126。

## 結論

多一份容量沒有用：U-Delta+fast2（3.897）跟 U-Delta 一樣，U-Delta+slow（3.888）比它好 0.009（約 U-Delta seed 標準差的 3 倍，各 1 個 seed）。所以 U-Delta+slow 的進步不是容量。不是容量的話，原本推測是那份幾乎不忘的慢記憶；但 U-Delta-spreadλ（3.890）和 U-Delta-λlr10（3.887）用單一記憶就追平 U-Delta+slow，而且 U-Delta-λlr10 沒有慢 slot，所以慢記憶是不是原因還不確定。 每層記憶長度：第 3 層拉到約 1,600 個 token，其他層跟 U-Delta 差不多。

## 相關實驗

- [記憶設計：快記憶、慢記憶、遺忘率](../experiments/memory-design.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwnext2.txt）
python scripts/train_variant.py --out_dir=outputs/fwsd2c_15m --variant=SD2c --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=8 --gradient_accumulation_steps=16 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsd2c_15m
```
