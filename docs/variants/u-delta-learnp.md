# U-Delta-learnP（代號 `SDp`）

U-Delta 每層的位址次方 p 可學（從 2 開始）。

- **類別**：位址尖銳度
- **基於**：[U-Delta](u-delta.md)
- **程式碼**：`scripts/delta.py` 的 `PowerDeltaBlock`，訓練時 `--variant=SDp`

## 怎麼做的

- 每層多一個純量 a，p = 1 + eᵃ，從 0 開始（p＝2，跟 U-Delta 完全一樣）；a 限制在 [−5, 2]，所以 p 在 1.007 到 8.4 之間。
- 位址 r = ReLU(Ku + b)^p ÷ (ε + Σ)，同一個 r 同時給 FFN、寫入鑰匙和讀取查找，所以還是統一 block。
- p 的梯度：∂s^p/∂p = s^p · log s，整層所有 slot 和 token 加總到同一個 a；AdamW、學習率 5e-4、不做 weight decay。

**問題**：blog（alexwang.ai/posts/associative-memory）說回想函數越尖，干擾越少、容量越大（線性約 d，ReLU^p 約 d^p，softmax 指數級）。U-Delta 的尖銳度是寫死的，讓它可學就知道模型想不想要更尖。

```text
r = ReLU(K u + b)^p / (ε + Σ_a ReLU(K u + b)_a^p),   p = 1 + exp(a)
ReLU² 除以總和對 K 的尺度不變：K 放大 c 倍，結果不變，所以原本學不到尖銳度
```

**設定**：參數：跟 U-Delta 一樣（每層只多 1 個數）；狀態：2.46M（同 U-Delta）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsdp15m` | 1 | 15.45M | 3.892 | 3.851 | 4.255 | 3.846 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.812。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：7 / 366 / 279 / 491 / 274 / 24 / 76 / 94 / 83 / 78 / 61 / 337。

## 結論

每一層都學得比 2 尖一點（2.02–2.35），沒有一層想變鈍；記憶長的層（第 1–4、11 層）比較尖，第 10 層是例外。NLL 只好 0.005（約 U-Delta seed 標準差的 1.5 倍），前 64 個 token 沒有改善。尖銳度不是瓶頸，改成 softmax 的必要性不高。

## 相關實驗

- [位址尖銳度](../experiments/sharpness.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdp.txt）
python scripts/train_variant.py --out_dir=outputs/fwsdp15m --variant=SDp --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsdp15m
```
