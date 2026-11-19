# U-Delta-rawFFN（代號 `SDu`）

U-Delta 的 FFN 那一半改讀未歸一化的 relu²（就是標準的 ReLU² FFN），記憶那一半不變。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[U-Delta](u-delta.md)
- **程式碼**：`scripts/delta.py` 的 `RawFFNDeltaBlock`，訓練時 `--variant=SDu`

## 怎麼做的

- U-Delta：y = μ r + Eᵀq，r = relu(K u + b)² ／ (ε + Σ)，FFN 和記憶共用歸一化過的位址。除以總和讓 FFN 輸出跟「這個 token 有多強烈地觸發 slot」無關，一般的 FFN（ReLU²、SwiGLU）都保留這個大小。
- U-Delta-rawFFN：y = μ z + Eᵀq，z = relu(K u + b)²（不除總和）。記憶的讀寫位址、寫入目標 v − μ k（k = r_{t−1}）都跟 U-Delta 一樣。
- 還是同一組 slot、同一個 K：只改 FFN 讀的是 z 還是 r。
- 參數、狀態跟 U-Delta 一樣；初始時 FFN 輸出比 U-Delta 大（約跟一般 FFN 一樣）。

**問題**：U-Delta 15M 贏 Transformer 0.047，42M 卻輸 0.136，而且每個位置都輸一樣多，像是基本建模能力（FFN）落後。FFN 那一半讀歸一化過的位址，是不是放大後的瓶頸？

```text
z = relu(K u + b)²，r = z ／ (ε + Σ_a z_a)
y = μ z + E_tᵀ q_t（U-Delta：μ r + E_tᵀ q_t）
```

**設定**：15M 先跑，沒變差（NLL 不高於 U-Delta 三個 seed 的範圍太多）就接著跑 42M（16 層、6,400 步，約 12 小時）；參數：15.45M（同 U-Delta）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsdu15m` | 1 | 15.45M | 3.812 | 3.774 | 4.170 | 3.768 |
| 42M | `fw42m_sdu` | 1 | 41.77M | 3.228 | 3.183 | 3.623 | 3.179 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.760。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：5 / 509 / 464 / 29 / 13 / 19 / 13 / 138 / 135 / 17 / 14 / 68。

42M 按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量（第 0 → 15 層，[每個格子記多久](../experiments/slot-retention.md)）：0% / 39% / 64% / 23% / 0% / 6% / 90% / 40% / 37% / 0% / 48% / 32% / 60% / 59% / 55% / 55%。

## 結論

15M 大幅變好：NLL 3.812（長度 1,024）、3.774（4,096）。U-Delta 三個 seed 是 3.895–3.901，所以好了 0.086，約是 seed 間差距的 10 倍（1 個 seed）。它也贏 Transformer（3 個 seed 平均 3.945）0.133、贏含注意力的 U-Delta+attn（3.840）0.028，是目前 15M 最好的模型。進步在每個位置都差不多（0.08–0.09），前 64 個 token 4.170，追平 GDN（4.166；U-Delta 4.258），所以是基本建模能力變好，不是記憶，符合假設：FFN 讀歸一化過的位址會讓它變弱。學到的記憶長度也變了：長記憶層變少，第 1–2 層記約 460–510 個 token，第 7–8 層約 135，其他層 5–68（U-Delta 是第 1–4、11 層記 259–434）。讀到 20K 也不崩，每一段都最好，連 U-Delta+attn 都贏（16K–20K 3.760 對 3.762，打平；U-Delta 3.817），但比 U-Delta 好的幅度越遠越小：1,024 內約 0.09，16K–20K 0.056，可能因為記得久的層變少。42M：NLL 3.228（1,024）、3.183（4,096），Transformer 42M 3.214，只輸 0.014；U-Delta 42M 3.350，好 0.122；U-Gate 42M 3.328（各 1 個 seed）。U-Delta 從 15M 贏 Transformer 0.047 變成 42M 輸 0.136，U-Delta-rawFFN 從贏 0.133 變成輸 0.014，FFN 是 U-Delta 在 42M 落後的主因。對 Transformer：前 64 個 token 略贏（3.623 對 3.628），之後每段落後 0.013–0.016，剩下的差距在比較遠的位置。讀到 20K（42M）：每一段都比 U-Delta 42M 好 0.10–0.14，但都輸只看最近 1,024 個 token 的 Transformer（滑動窗口）0.01–0.03，16K–20K 3.166 對 3.142（U-Delta 42M 3.269）；原本的 Transformer 過了 1,024 就崩（16K–20K 6.251）。

## 相關實驗

- [FFN 那一半讀哪個位址](../experiments/ffn-address.md)
- [規模：15M → 42M → 110M](../experiments/scale.md)
- [長上下文與外推（讀到 20K）](../experiments/extrapolation.md)
- [寫入量與 loss](../experiments/write-vs-loss.md)
- [U-Delta 為什麼贏 GDN](../experiments/vs-gdn.md)
- [逐 token 差距：輸在哪些 token](../experiments/token-gap.md)
- [拿掉記憶：記憶值多少、模型多依賴記憶](../experiments/memory-ablation.md)
- [遠處的前文值多少](../experiments/reach.md)
- [每個格子記多久（按寫入量加權）](../experiments/slot-retention.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdu.txt）
python scripts/train_variant.py --out_dir=outputs/fwsdu15m --variant=SDu --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsdu15m
# 42M（scripts/nrp/fwsdu42m.txt）
python scripts/train_variant.py --out_dir=outputs/fw42m_sdu --variant=SDu --hidden_dim=1248 --dim=512 --n_layers=16 --batch_size=8 --gradient_accumulation_steps=16 --warmup_iters=1000 --max_iters=6400 --dataset=fineweb --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --learning_rate=5e-4 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fw42m_sdu
```
