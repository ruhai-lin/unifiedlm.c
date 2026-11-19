# UD2-gate-mass-q32（代號 `UD2-gate-mass-q32`）

UD2-gate-q32 的讀出除以位址碰到的記憶量（像 softmax 的分母）。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[UD2-gate-q32](ud2-gate-q32.md)
- **程式碼**：`scripts/delta.py` 的 `MassQueryGateAnchorDeltaBlock`，訓練時 `--variant=UD2-gate-mass-q32`

## 怎麼做的

- UD2-gate-q32：y = x + μ z_t + o_t · E_tᵀ q_t，讀位址加秩 32 的修正，hidden_dim 662。
- UD2-gate-mass-q32：每個 token 也用同一個 delta rule 寫一個常數 1 到一組計量 f，讀出 m_t = q_t · f_t 量的是位址蓋到多少最近寫過的格子；y = x + μ z_t + o_t · E_tᵀ q_t ／ (τ + m_t)，τ = softplus(可學)，起點 1。
- 參數 15,454,440；其他跟 UD2-gate-q32 一樣。

**問題**：讀出大小會隨記憶變滿而變大。用「碰到多少記憶」當分母（像 softmax），會不會比 RMSNorm 更好？

```text
f_t = 用同一個 delta rule 寫入常數 1，m_t = q_t · f_t
y = μ z_t + o_t · E_tᵀ q_t ／ (τ + m_t)
```

**設定**：15M（12 層、2,300 步），hidden_dim 662；1 個 seed（1337），3090 1.3 小時（4,653 秒）；fork session 跑（2026-11-15）；第一次跑時程式有 bug（讀位址修正沒作用），這是修正後從頭重跑的。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `ud2_gate_mass_q32_15m` | 1 | 15.45M | 3.749 | 3.705 | 4.114 | 3.702 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.669。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：2 / 115 / 194 / 155 / 13 / 2 / 1 / 2 / 2 / 2 / 2 / 2。

15M 按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量（第 0 → 11 層，[每個格子記多久](../experiments/slot-retention.md)）：0% / 44% / 40% / 47% / 13% / 1% / 0% / 1% / 21% / 1% / 4% / 20%。

## 結論

15M（1 個 seed）：NLL 3.749（長度 1,024）、3.705（4,096），比直接的基底 UD2-gate-q32（3.746、3.701）差 0.003，每一段都差 0.003–0.005（前 64 個 token 4.114 對 4.111，512–1,024 3.702 對 3.699）。除以記憶量沒有幫助；同樣是處理讀出大小，RMSNorm（UD2-gate-norm-q32 3.727）好得多。讀到 20K 的 16K–20K 3.669（UD2-gate-q32 3.663）。逐 token 對 Transformer 全部 −0.177；照抄 −0.642 跟 UD2-gate-norm-q32 差不多，見過的 token 最差（−0.093，UD2-gate-norm-q32 −0.145）。

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
python scripts/train_variant.py --out_dir=outputs/ud2_gate_mass_q32_15m --variant=UD2-gate-mass-q32 --hidden_dim=662 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/ud2_gate_mass_q32_15m
```
