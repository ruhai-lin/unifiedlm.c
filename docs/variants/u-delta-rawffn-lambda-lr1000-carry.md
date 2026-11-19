# U-Delta-rawFFN+λlr1000+carry（代號 `SDulc`）

U-Delta-rawFFN+λlr1000 訓練時記憶跨窗口接續：每個窗口從同一串上一個窗口結束的記憶開始（梯度不跨窗口），評測照舊從空記憶開始。

- **類別**：記憶設計
- **基於**：[U-Delta-rawFFN+λlr1000](u-delta-rawffn-lambda-lr1000.md)
- **程式碼**：`scripts/delta.py` 的 `CarryRawFFNDeltaBlock`，訓練時 `--variant=SDulc`

## 怎麼做的

- U-Delta-rawFFN+λlr1000：每個訓練窗口的記憶從 0 開始，窗口之間互不相干。
- U-Delta-rawFFN+λlr1000+carry：訓練資料（shard 首尾相接）切成 A × B 段等長的連續串，批量裡的每一列一直讀同一串；每個窗口的記憶從這一列上一個窗口結束的狀態開始，梯度不往前一個窗口傳（stop-gradient）。
- 評測照舊：每個窗口從空記憶開始。其他跟 U-Delta-rawFFN+λlr1000 一樣；因為狀態跨步傳遞，訓練不能用 torch.compile。

**問題**：U-Delta-rawFFN+λlr1000 外推不崩、越讀越好，但訓練時記憶從沒看過 1,024 以後的東西。讓它在訓練時就接著長串讀，長距離會不會更好？

```text
E_0(窗口 n) = stopgrad(E_T(窗口 n−1))，同一列；U-Delta-rawFFN+λlr1000：E_0 = 0
```

**設定**：15M（12 層、2,300 步），跟 U-Delta-rawFFN+λlr1000 15M 同設定；1 個 seed（1337），A6000 3.2 小時（compile 關掉）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsdulc15m` | 1 | 15.45M | 3.815 | 3.759 | 4.240 | 3.760 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.710。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：2 / 56 / 10 / 1 / 2 / 1 / 1 / 1 / 1 / 1 / 1 / 1。

## 結論

15M（1 個 seed）：NLL 3.815（長度 1,024）、3.759（4,096），比同一個 seed 的 U-Delta-rawFFN+λlr1000（3.793）差 0.022。差距在開頭最大、越後面越小：前 64 個 token 差 0.087，512–1,024 差 0.012。評測時每個窗口從空記憶開始，訓練時卻幾乎都接著上一個窗口，所以開頭吃虧可能來自訓練和評測不一致。讀到 20K 沒有長距離的好處：1K–4K 還差 0.006–0.008，4K 之後打平（16K–20K 3.710 對 3.713）。記憶也沒有變長（按格子平均）：有效記憶長度第 1 層 56 個 token（U-Delta-rawFFN+λlr1000 38），第 2 層反而從 22 縮到 10，其他層一樣只有 1–2 個 token；U-Delta-rawFFN+λlr1000 第 2 層那批基本記憶長度約 20 萬 token 的慢 slot 不見了（p90 9）。限制：梯度不往前一個窗口傳，寫入得不到「幫到下一個窗口」的獎勵，所以這只測了「看過長串」，沒有測「跨窗口反向傳播」。

## 相關實驗

- [長上下文與外推（讀到 20K）](../experiments/extrapolation.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdfk.txt）
python scripts/train_variant.py --out_dir=outputs/fwsdulc15m --variant=SDulc --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50 --compile=False
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsdulc15m
```
