# UD2-gate-norm-q32（代號 `SDulagnq`）

UD2-gate-norm 和 UD2-gate-q32 合在一起：讀出先 RMSNorm，讀位址加秩 32 的修正。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[UD2-gate](ud2-gate.md)
- **程式碼**：`scripts/delta.py` 的 `ReadNormQueryGateAnchorDeltaBlock`，訓練時 `--variant=SDulagnq`

## 怎麼做的

- UD2-gate-norm：讀出先 RMSNorm 再過門。
- UD2-gate-q32：讀位址加秩 32 的修正，hidden_dim 662。
- UD2-gate-norm-q32：兩個都加，hidden_dim 662，參數 15,457,884。

**問題**：兩個修改各好 0.028、0.021，加起來會不會疊加？

```text
q_t = unit(relu(K u_t + b + U V u_t)²)
y = μ z_t + o_t · RMSNorm(E_tᵀ q_t)
```

**設定**：15M（12 層、2,300 步），hidden_dim 662；1 個 seed（1337），1.3 小時（4,857 秒）；fork session 跑（2026-11-15）；讀到 20K 和診斷用修正過的程式（5f2e380）重跑。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsdulagnq15m` | 1 | 15.46M | 3.727 | 3.683 | 4.120 | 3.678 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.645。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：104 / 2 / 2 / 2 / 1 / 2 / 3 / 2 / 7 / 4 / 4 / 2。

15M 按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量（第 0 → 11 層，[每個格子記多久](../experiments/slot-retention.md)）：35% / 7% / 0% / 5% / 0% / 1% / 12% / 1% / 32% / 42% / 29% / 26%。

## 結論

15M（1 個 seed）：NLL 3.727（長度 1,024）、3.683（4,096），是目前 15M 最好的統一 block，贏 Transformer（3 個 seed 平均 3.945）0.218。比 UD2-gate（3.766）好 0.039，比 UD2-gate-norm 好 0.011、比 UD2-gate-q32 好 0.018：兩個修改各自的 0.028、0.021 大約疊加了八成。前 64 個 token 4.120（UD2-gate 4.119，打平），512–1,024 3.678（UD2-gate 3.722），進步全在 64 以後。讀到 20K 的 16K–20K 3.645，是 15M 模型裡這段最好的（UD2-gate-norm 3.661、UD2-gate-q32 3.663、UD2-gate 3.685）。逐 token 對 Transformer 全部 −0.197（UD2-gate-q32 −0.177），每一類都領先：照抄 −0.637、見過 −0.145、新 token −0.040，位置 64–256 的新 token −0.039。截掉遠處前文，新 token 只變好 0.004（UD2-gate-q32 0.014、Transformer 變差 0.011）：讀出端的修改裡干擾最小，但還沒消失。按寫入量加權，第 0、6、8、9、10、11 層過了 256 個 token 還留 35%／12%／32%／42%／29%／26%。 42M（hidden 1195，參數跟 UD2-gate-norm-q32-rope 42M 一樣）：3.172（1,024）、3.122（4,096），對 Transformer 42M 3.214 贏 0.042；比 UD2-gate-norm-q32-rope 42M 3.170 只差 0.002，所以 42M 的進步不是來自旋轉相對位置（各 1 個 seed）。 42M 診斷（減掉 Transformer 42M，128 個窗口）：逐 token 整體 −0.045，跟 UD2-gate-norm-q32-rope 42M 一樣；照抄 −0.044（rope 版 −0.038）、見過 −0.060（−0.065）、新 token −0.033（−0.031）。讀到 20K 的 16K–20K 3.094（rope 版 3.087）。沒有旋轉相對位置時第 9、14 層過了 256 個 token 還留 54%、58% 的寫入量，rope 版只留 1%、10%。

## 相關實驗

- [讀出端：門、縮放、RMSNorm、讀位址修正、旋轉相對位置](../experiments/read-side.md)
- [長上下文與外推（讀到 20K）](../experiments/extrapolation.md)
- [逐 token 差距：輸在哪些 token](../experiments/token-gap.md)
- [遠處的前文值多少](../experiments/reach.md)
- [每個格子記多久（按寫入量加權）](../experiments/slot-retention.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdfk.txt）
python scripts/train_variant.py --out_dir=outputs/fwsdulagnq15m --variant=SDulagnq --hidden_dim=662 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsdulagnq15m
```
