# U-Delta-rawFFN+λlr1000（代號 `SDul`）

**3 個 seed、有 42M 的統一 block 裡最好的。** U-Delta-rawFFN 再加上 U-Delta-λlr1000：每格基礎遺忘率學快 1,000 倍。不加參數。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[U-Delta-rawFFN](u-delta-rawffn.md)
- **程式碼**：`scripts/delta.py` 的 `RawFFNFastRate1000DeltaBlock`，訓練時 `--variant=SDul`

## 怎麼做的

- U-Delta-rawFFN：y = μ z + Eᵀq，FFN 讀沒除總和的 relu²，記憶跟 U-Delta 一樣。
- U-Delta-λlr1000：log_rate 存成 θ = log_rate ／ 1000，Adam 的步伐等於讓 log_rate 學快 1,000 倍；起點一樣。
- U-Delta-rawFFN+λlr1000：兩個都做，其他跟 U-Delta-rawFFN 一樣。參數、狀態跟 U-Delta 一樣。

**問題**：U-Delta-rawFFN（15M 好 0.086，每個位置差不多）和 U-Delta-λlr1000（好 0.055，越後面越多）進步的位置不一樣。合在一起能不能疊加？U-Delta-rawFFN 42M 只輸 Transformer 0.014，而且差距在 64 之後、越後面越大，正好是 U-Delta-λlr1000 進步的地方。

```text
y = μ z + E_tᵀ q_t，z = relu(K u + b)²
log λ = expm1(−exp(1000 θ) · g)
```

**設定**：15M（12 層、2,300 步），跟 U-Delta-rawFFN 15M 同設定；比 U-Delta-rawFFN 好 0.01 以上就接著跑 42M（16 層、6,400 步，約 9.5 小時）；參數：15.45M（同 U-Delta）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsdul15m`（+2 個 seed） | 3 | 15.45M | 3.796（3.793–3.799） | 3.753 | 4.154 | 3.751 |
| 42M | `fw42m_sdul` | 1 | 41.77M | 3.223 | 3.175 | 3.624 | 3.174 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.713。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：2 / 38 / 22 / 1 / 2 / 1 / 1 / 1 / 2 / 1 / 2 / 1。

15M 按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量（第 0 → 11 層，[每個格子記多久](../experiments/slot-retention.md)）：0% / 40% / 44% / 3% / 0% / 0% / 0% / 1% / 5% / 0% / 20% / 0%。

42M 按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量（第 0 → 15 層，[每個格子記多久](../experiments/slot-retention.md)）：0% / 13% / 27% / 50% / 0% / 2% / 0% / 0% / 24% / 0% / 0% / 0% / 22% / 0% / 2% / 27%。

## 結論

15M 三個 seed：NLL 3.793、3.795、3.799（平均 3.796，標準差 0.003；長度 4,096 平均 3.753），比 U-Delta-rawFFN（3.812，1 個 seed）好約 0.016，是目前 15M 最好的統一 block，也贏 U-Delta+attn（3.840）0.044、Transformer（3 個 seed 平均 3.945，標準差 0.016）0.149。U-Delta-rawFFN+λlr1000 最差的 seed 也比 Transformer 最好的 seed（3.928）好 0.128，同一個 seed 配對每組差 0.134–0.161；每個位置都贏 Transformer，越後面贏越多（前 64 個 token 0.083，512–1,024 0.167）。以下是 seed 1337 的細節。但只疊加了一部分：U-Delta-λlr1000 單獨比 U-Delta 好 0.055，加在 U-Delta-rawFFN 上只再好 0.019。每個位置都比 U-Delta-rawFFN 好 0.017–0.020（前 64 個 token 4.153 對 4.170，512–1,024 3.748 對 3.768）。按格子平均，有效記憶長度 12 層裡有 10 層只有 1.1–2.2 個 token，只剩第 1、2 層記 38、22 個 token（U-Delta-rawFFN 是 5–509）；但這個平均被快格子拉低，按寫入量加權（slotlen），第 1、2、10 層過了 256 個 token 還留 40%、44%、20% 的寫入量，其他層最多 5%：記憶沒有被關掉，是少數層慢、其他層快的分工。比 U-Delta-rawFFN 好 0.01 以上，照規則接著跑 42M（結果在最後）。讀到 20K：每一段都是 15M 最好的，16K–20K 3.713（U-Delta-λlr1000 3.752、U-Delta-rawFFN 3.760、U-Delta+attn 3.762、U-Delta 3.817、滑動窗口的 Transformer 3.874）。比 U-Delta-rawFFN 好的幅度越讀越大：前 64 個 token 0.017、512–1,024 0.026、16K–20K 0.047，U-Delta-λlr1000 那一半帶來的外推好處留住了。42M（1 個 seed）：NLL 3.223（長度 1,024）、3.175（4,096），Transformer 42M 3.214，輸 0.009；U-Delta-rawFFN 42M 3.228，只好 0.004（15M 好 0.016），所以 log_rate 學快在 1,024 內幾乎沒幫助（讀到 20K 才拉開，見最後）。輸 Transformer 的 0.009 比 15M Transformer 三個 seed 的標準差（0.016）小，算追平，但沒有贏。15M 到 42M，U-Delta 從贏 Transformer 0.047 變成輸 0.136，U-Delta-rawFFN 從贏 0.133 變成輸 0.014，U-Delta-rawFFN+λlr1000 從贏 0.149 變成輸 0.009：15M 的大幅領先都沒有帶到 42M。每個位置：前 64 個 token 略贏 Transformer（0.005），之後每段輸 0.009–0.012，形狀跟 U-Delta-rawFFN 42M 一樣，只是每段少輸 0.003–0.005。訓練曲線：第 1,000 步以前領先 Transformer（第 500 步 4.866 對 4.922），第 1,500 步起落後。按格子平均的有效記憶長度只有 1.2–20 個 token（U-Delta-rawFFN 42M 4–375），但按寫入量加權，第 2、3、8、12、15 層過了 256 個 token 還留 22–50%（第 3 層 1,171/1,248 個格子），其他層最多 13%；U-Delta-rawFFN 42M 有 12 層留 23–90%，所以 log_rate 學快讓慢的層變少，但沒有消失；4,096 比 1,024 低 0.048，過了訓練長度還在用前文。A6000 訓練 8.3 小時。讀到 20K（42M）：追平只看最近 1,024 個 token 的 Transformer（滑動窗口），16K–20K 3.142 對 3.142，1K 之後每段相差不到 0.006（4K–8K、8K–16K 略贏 0.004–0.005）；1,024 以內有輸有贏（前 64 個 token 贏 0.021，64–256 輸 0.017–0.019）。比 U-Delta-rawFFN 42M 好的幅度在 2K 之後越讀越大（2K–4K 0.010，16K–20K 0.024；U-Delta-rawFFN 42M 3.166、U-Delta 42M 3.269），跟 15M 一樣，U-Delta-λlr1000 那一半帶來的外推好處在 42M 也留住了。原本的 Transformer 過了 1,024 就崩（16K–20K 6.251）。只有 64 個窗口、1 個 seed。

## 相關實驗

- [規模：15M → 42M → 110M](../experiments/scale.md)
- [記憶設計：快記憶、慢記憶、遺忘率](../experiments/memory-design.md)
- [長上下文與外推（讀到 20K）](../experiments/extrapolation.md)
- [多 seed 的穩定性](../experiments/seeds.md)
- [逐 token 差距：輸在哪些 token](../experiments/token-gap.md)
- [拿掉記憶：記憶值多少、模型多依賴記憶](../experiments/memory-ablation.md)
- [遠處的前文值多少](../experiments/reach.md)
- [每個格子記多久（按寫入量加權）](../experiments/slot-retention.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdul.txt）
python scripts/train_variant.py --out_dir=outputs/fwsdul15m --variant=SDul --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsdul15m
# 42M（scripts/nrp/fwsdul42m.txt）
python scripts/train_variant.py --out_dir=outputs/fw42m_sdul --variant=SDul --hidden_dim=1248 --dim=512 --n_layers=16 --batch_size=8 --gradient_accumulation_steps=16 --warmup_iters=1000 --max_iters=6400 --dataset=fineweb --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --learning_rate=5e-4 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fw42m_sdul
```
