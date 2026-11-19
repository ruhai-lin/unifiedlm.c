# Qwen 式混合（代號 `QHybrid`）

對照，含 attention，不算統一 block：照 Qwen3-Next 的排法，每 4 層裡 3 層是 Gated DeltaNet、1 層是 gated attention，每層後面接 SwiGLU。

- **類別**：基線（含 attention 的對照）
- **基於**：[GDN](gdn.md)
- **程式碼**：`scripts/variants.py` 的 `GatedAttentionHalf` 和 `GatedDeltaHalf`（`variant == "QHybrid"` 那一段），訓練時 `--variant=QHybrid`

## 怎麼做的

- 每 4 層：3 層 GDN 半層、1 層 gated attention 半層，每個半層後面接 SwiGLU。
- attention 層：GQA（2 個 kv head）、q 和 k 先 RMSNorm、輸出乘 sigmoid 門、完整的 RoPE。
- hidden_dim 1262，參數量對齊 Transformer 42M。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42M | `fw42m_qhybrid` | 1 | 41.69M | 3.166 | 3.164 | 3.558 | 3.117 |

## 結論

42M 跟 UD2-gate-norm-q32-decay-ev（3.167）打平，比 Transformer 好 0.048、比純 GDN 好 0.063。贏在開頭和新 token，照抄（隨機 token 512 個重複 8.15，decay-ev 5.06）和記憶檢索（N = 256 時 0.03，decay-ev 0.13）都比統一 block 差；長度 4,096 時兩層 attention 過了訓練長度變差（3.164，decay-ev 3.117）。1 個 seed。

## 相關實驗

- [逐 token 差距：輸在哪些 token](../experiments/token-gap.md)

## 重現

`scripts/nrp/fwsdfk.txt` 的 `fw42m_qhybrid` 那一行（Gary 2026-11-18 核准）。
