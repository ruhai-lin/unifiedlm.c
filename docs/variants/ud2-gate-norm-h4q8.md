# UD2-gate-norm-h4q8（代號 `UD2-gate-norm-h4q8`）

UD2-gate-norm-q32 的讀出拆成 4 個 head，每個 head 有自己秩 8 的讀位址修正、各自 RMSNorm；參數一樣。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[UD2-gate-norm-q32](ud2-gate-norm-q32.md)
- **程式碼**：`scripts/delta.py` 的 `MultiHeadReadNormGateAnchorDeltaBlock`，訓練時 `--variant=UD2-gate-norm-h4q8`

## 怎麼做的

- UD2-gate-norm-q32：一個讀位址，加秩 32 的修正；讀出先 RMSNorm 再過門。
- UD2-gate-norm-h4q8：值的寬度分成 4 段，每個 head 用自己的讀位址 q_h = unit(relu(K u + b + U_h V_h u)²)（秩 8，4 × 8 = 32）讀自己那段，各自 RMSNorm；寫入對所有 head 一樣（同一個鑰匙、β、衰減），所以狀態大小不變。
- 參數 15,457,884，跟 UD2-gate-norm-q32 一樣；head 折進 batch 計算，記憶那一步大約貴 4 倍（A10 每步 5.3 秒）。

**問題**：同樣 32 個秩的修正，分給 4 個 head 各查各的，會不會比一個 head 好？

```text
q_h = unit(relu(K u_t + b + U_h V_h u_t)²)，h = 1…4，秩 8
y = μ z_t + o_t · [RMSNorm(E_{t,h}ᵀ q_h)]_h
```

**設定**：15M（12 層、2,300 步），hidden_dim 662，跟 UD2-gate-norm-q32 同設定；1 個 seed（1337），A10 3.6 小時（13,000 秒）；fork session 跑（2026-11-15）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `ud2_gate_norm_h4q8_15m` | 1 | 15.46M | 3.732 | 3.688 | 4.124 | 3.683 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.654。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：112 / 2 / 1 / 2 / 1 / 6 / 2 / 2 / 2 / 3 / 4 / 1。

15M 按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量（第 0 → 11 層，[每個格子記多久](../experiments/slot-retention.md)）：36% / 5% / 0% / 1% / 0% / 57% / 9% / 1% / 1% / 39% / 21% / 16%。

## 結論

15M（1 個 seed）：NLL 3.732（長度 1,024）、3.688（4,096），比 UD2-gate-norm-q32（3.727、3.683）差 0.005，每一段都差：長度 1,024 差 0.004–0.006，4,096 差 0.005–0.009。同樣 32 個秩，4 個秩 8 的 head 輸給 1 個秩 32 的 head，而且記憶那一步貴約 4 倍。比目前最好的 UD2-gate-norm-q32-rope（3.716）差 0.016。讀到 20K 的 16K–20K 3.654（UD2-gate-norm-q32 3.645）。逐 token 對 Transformer 全部 −0.193：照抄是所有模型裡最好的（−0.651，UD2-gate-norm-q32 −0.637），但新 token 變弱（−0.032 對 −0.040），位置 64–256 只剩 −0.018。

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
python scripts/train_variant.py --out_dir=outputs/ud2_gate_norm_h4q8_15m --variant=UD2-gate-norm-h4q8 --hidden_dim=662 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/ud2_gate_norm_h4q8_15m
```
