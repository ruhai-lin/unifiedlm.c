# U-Delta-λlr1000（代號 `SDl1000`）

U-Delta 的 `log_rate` 有效學習率 ×1000：按格子平均，幾乎每層都把記憶縮到上限（約 1 個 token）；按寫入量加權，第 1–3、11 層仍記得過 256 個 token。

- **類別**：記憶設計
- **基於**：[U-Delta](u-delta.md)
- **程式碼**：`scripts/delta.py` 的 `FastRate1000DeltaBlock`，訓練時 `--variant=SDl1000`

## 怎麼做的

- log_rate 改存成 θ＝log_rate／100（或／1000），用的時候乘回去；Adam 每步讓 log_rate 移動約 100（1000）個學習率。學習率 5e-4，所以每步最多約 0.05（0.5），在 log 尺度上，2,300 步可以走很遠。CPU 上量過：每步平均移動 0.050 和 0.500，是 U-Delta 的 100 和 1000 倍。
- 初始值、初始輸出、參數、狀態都跟 U-Delta 一樣。
- 遺忘率有上限：每個 token 最多忘 63%（expm1 的形式），所以 log_rate 很大也不會出數值問題。

**問題**：U-Delta-λlr10（10 倍）把記憶縮短，沒學出慢 slot。再放大 100、1000 倍，記憶長度會停在同一個地方（代表那是它想要的），還是一路往短（或長）衝、NLL 變差？

```text
λ = exp(−exp(c·θ) · g(u))，c＝100 或 1000，θ 初始 = log(1/(4L))／c
其他同 U-Delta
```

**設定**：參數：15.45M（同 U-Delta）；狀態：2.46M（同 U-Delta）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsdl1000_15m` | 1 | 15.45M | 3.843 | 3.796 | 4.220 | 3.794 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.752。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：1 / 53 / 14 / 9 / 1 / 1 / 1 / 1 / 1 / 1 / 1 / 2。

15M 按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量（第 0 → 11 層，[每個格子記多久](../experiments/slot-retention.md)）：0% / 34% / 18% / 19% / 0% / 0% / 0% / 0% / 0% / 0% / 0% / 21%。

## 結論

U-Delta-λlr100：一路往短衝，而且變好很多。NLL 3.851（長度 1,024）、3.809（4,096），比 U-Delta 三個 seed（3.895–3.901）好約 0.047，比 U-Delta-λlr10（3.887）好 0.036，每個位置都好 0.04–0.05（1 個 seed）。第 5–10 層的基礎記憶長度中位數學到 1–5 個 token，有效記憶長度 1–2，已經碰到上限（λ ≥ e⁻¹，每個 token 最多忘 63%）；第 0 層 2，第 3、4、11 層 19–34，只有第 1–2 層還記 83–139 個。所以記憶長度沒有停在 U-Delta-λlr10 的位置：中間層想要的是「幾乎不記」，短到上限還在變好，那幾層的記憶等於只看前一兩個 token 的局部運算。讀到 20K 也不崩，每一段都比 U-Delta 好約 0.048（前 64 個 token 0.033），而且不隨距離縮小，16K–20K 3.769（U-Delta 3.817，U-Delta-rawFFN 3.760，U-Delta+attn 3.762）：中間層幾乎不記，遠距離靠第 1–2 層就夠。 U-Delta-λlr1000：再往短衝，又更好。NLL 3.843（長度 1,024）、3.796（4,096），比 U-Delta-λlr100 再好 0.008、0.013；長度 4,096 跟含注意力的 U-Delta+attn（3.798）一樣，每一段差不到 0.006。除了第 1–3 層（有效記憶長度 53、14、9 個 token），每層都忘到上限（1.0–1.9）。所以記憶長度沒有停在某個想要的值：給它多少自由，它就把中間層的記憶縮短多少，NLL 一路變好。可是位置越後面 NLL 還是越低，長距離的資訊可能靠第 2–3 層少數慢 slot（基礎記憶長度 p90 417、983）。U-Delta-λlr1000 讀到 20K 越遠越好：比 U-Delta 好的幅度從前 64 個 token 的 0.029 變大到 8K 之後的 0.065，16K–20K 3.752，是所有模型裡這段最好的（U-Delta-rawFFN 3.760，U-Delta+attn 3.762，U-Delta 3.817）。中間層幾乎不記，外推反而最好。 更正（2026-11-14，按寫入量加權的 slotlen）：上面的「忘到上限」是按格子平均的有效記憶長度，會被快格子拉低。按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量：U-Delta-λlr1000 第 1、2、3、11 層 34%／18%／19%／21%，第 4–10 層 0%；U-Delta-λlr100 第 1、2、3、11 層 48%／31%／24%／30%，第 4 層 8%，其他層最多 1%。所以是少數層慢、中間層幾乎不記，不是長記憶被關掉；長距離的資訊靠第 1–3、11 層。

## 相關實驗

- [記憶設計：快記憶、慢記憶、遺忘率](../experiments/memory-design.md)
- [長上下文與外推（讀到 20K）](../experiments/extrapolation.md)
- [拿掉記憶：記憶值多少、模型多依賴記憶](../experiments/memory-ablation.md)
- [每個格子記多久（按寫入量加權）](../experiments/slot-retention.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdl2.txt）
python scripts/train_variant.py --out_dir=outputs/fwsdl1000_15m --variant=SDl1000 --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsdl1000_15m
```
