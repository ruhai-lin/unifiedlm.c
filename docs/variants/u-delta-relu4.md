# U-Delta-ReLU4（代號 `SDr`）

U-Delta 前 4 個 block 用 ReLU 位址（p = 1），上面照舊 relu²。

- **類別**：位址尖銳度
- **基於**：[U-Delta](u-delta.md)
- **程式碼**：`scripts/delta.py` 的 `FixedPowerDeltaBlock`（前 4 個 block，p = 1），訓練時 `--variant=SDr`

## 怎麼做的

- 第 0–3 個 block：r = ReLU(Ku + b) ÷ (ε + Σ)；第 4–11 個 block 跟 U-Delta 一樣是 ReLU²。
- p 是固定的，不學。參數跟 U-Delta 完全一樣。
- U-Delta 裡第 0 層的記憶最短（約 7 個 token），是最像「快記憶」的一層。

**問題**：U-Delta-learnP 學到的 p 都 ≥ 2，前幾層也接近 2。直接把前幾層壓成 p＝1 是反方向的檢查：鈍的位址會讓更多 slot 同時被讀寫，快記憶會變好（容易混合近處資訊）還是變差（干擾變多）？

```text
block 0–3：r = ReLU(K u + b) / (ε + Σ)
block 4–11：r = ReLU(K u + b)² / (ε + Σ)（同 U-Delta）
```

**設定**：參數：15.45M（同 U-Delta）；狀態：2.46M（同 U-Delta）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsdr15m` | 1 | 15.45M | 3.894 | 3.852 | 4.264 | 3.847 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.812。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：6 / 405 / 376 / 22 / 330 / 239 / 29 / 80 / 104 / 104 / 400 / 111。

## 結論

跟 U-Delta 打平：3.894 對 3.897（同一個 seed，在 U-Delta 三個 seed 的差距內）。近處略差（前 64 個 token 4.264 對 4.258），遠處略好（512–1K 3.847 對 3.851）。改成 ReLU 的第 3 層記憶從約 430 個 token 掉到 22，長記憶改由第 4、5、10 層負責；第 0 層本來就是快記憶（約 6 個 token）。所以鈍的位址沒有讓快記憶變好，模型只是把長記憶搬到 ReLU² 的層。跟 U-Delta-learnP 學出來的 p 都 ≥ 2 一致：前幾層不需要比較鈍的位址。

## 相關實驗

- [位址尖銳度](../experiments/sharpness.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdx.txt）
python scripts/train_variant.py --out_dir=outputs/fwsdr15m --variant=SDr --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsdr15m
```
