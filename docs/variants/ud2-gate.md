# UD2-gate（代號 `SDulag`）

UD2 的記憶讀出乘上每個 token 一個門 o = 2·sigmoid(w·h + b)（0 到 2，起點約 1）。每層多 289 個參數。3 個 seed 都比 UD2 好。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[UD2](ud2.md)
- **程式碼**：`scripts/delta.py` 的 `ReadGateAnchorDeltaBlock`，訓練時 `--variant=SDulag`

## 怎麼做的

- UD2：y = x + μ z_t + E_tᵀ q_t，每個 token 的記憶讀出都全額加上。
- UD2-gate：y = x + μ z_t + o_t · E_tᵀ q_t，o_t = 2·sigmoid(w·h_t + b)，h_t 是歸一化後的輸入；b = 0、w 標準差 0.02 起步，一開始 o_t 約 0.65–1.35。
- 每層多 289 個參數（w 288 個、b 1 個），12 層共 3,468 個；其他跟 UD2 一樣。

**問題**：上一批診斷看到：把遠處前文截掉，記憶模型的新 token 反而變好，像是舊內容在干擾；但讀記憶時沒有開關可以關掉。給它一個讀出門，干擾會不會消失、整體會不會更好？

```text
y = μ z_t + o_t · E_tᵀ q_t，o_t = 2·sigmoid(w·h_t + b)
E_t = (I − β k̂ k̂ᵀ) Λ E_{t−1} + β k̂ uᵀ，u = v_t − μ z_{t−1}
```

**設定**：15M（12 層、2,300 步），跟 UD2 15M 同設定；3 個 seed（1337、1338、1339），A40 等 1–1.4 小時；fork session 跑（2026-11-14／15）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsdulag15m`（+2 個 seed） | 3 | 15.46M | 3.765（3.761–3.767） | 3.722 | 4.120 | 3.720 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.685。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：2 / 65 / 6 / 3 / 2 / 1 / 2 / 1 / 1 / 2 / 1 / 3。

15M 按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量（第 0 → 11 層，[每個格子記多久](../experiments/slot-retention.md)）：0% / 51% / 7% / 19% / 0% / 0% / 1% / 0% / 13% / 53% / 27% / 41%。

## 結論

15M 3 個 seed：3.766／3.767／3.761，平均 3.765 ± 0.003（長度 4,096 平均 3.722），三個都比 UD2（3.778，1 個 seed）好 0.011–0.017，所以讀出門的進步是真的；贏 Transformer（3 個 seed 平均 3.945）0.181。以下是 seed 1337 的細節。每個位置都好 0.009–0.014（前 64 個 token 4.119 對 4.133，512–1,024 3.722 對 3.734）；讀到 20K，前 64 個 token 打平，之後每段好 0.009–0.034，16K–20K 3.685（UD2 3.695）。加讀出門原本想擋掉舊內容的干擾，這點沒有證實：把前文截到最近 65–128 個 token，新 token 還是變好（UD2-gate −0.016、UD2 −0.013，Transformer 反而差 0.011）。跟 Transformer 比，進步落在照抄前文的前 64 個 token（UD2-gate −0.508、UD2 −0.433）、位置 256–1,024 的新 token（−0.017、−0.005）和見過的 token（−0.079、−0.074）；常見的新 token（頻率前 100、位置 ≥ 128）反而比 Transformer 差 0.024（UD2 跟 Transformer 打平）。按寫入量加權、只算衰減，過了 256 個 token 還留下的：第 1、3、8、9、10、11 層 51%／19%／13%／53%／27%／41%，UD2 只有第 1、9、10 層 36%／16%／20%，加了讀出門，慢層變多。推理時關掉記憶 NLL 3.737 → 5.387（+1.65），最依賴第 0、5 層（關掉單層各 +0.39、+0.40）。 門實際學到什麼（readstats）：每層的門平均差很多（第 11 層 0.15、第 0 層 1.13），但同一層裡不同類別、不同位置的 token 幾乎一樣（例如第 1 層 0.54–0.59），所以它主要像一個每層的縮放；讀出本身在記得久的層隨位置變大（第 1 層從前 64 個 token 的 0.35 到 256–1,024 的 0.93），門沒有跟著調。這兩點引出兩個對照：每層一個固定縮放（UD2-scale），以及讀出先 RMSNorm（UD2-gate-norm）。

## 相關實驗

- [讀出端：門、縮放、RMSNorm、讀位址修正、旋轉相對位置](../experiments/read-side.md)
- [多 seed 的穩定性](../experiments/seeds.md)
- [長上下文與外推（讀到 20K）](../experiments/extrapolation.md)
- [逐 token 差距：輸在哪些 token](../experiments/token-gap.md)
- [拿掉記憶：記憶值多少、模型多依賴記憶](../experiments/memory-ablation.md)
- [遠處的前文值多少](../experiments/reach.md)
- [每個格子記多久（按寫入量加權）](../experiments/slot-retention.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdfk.txt）
python scripts/train_variant.py --out_dir=outputs/fwsdulag15m --variant=SDulag --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsdulag15m
```
