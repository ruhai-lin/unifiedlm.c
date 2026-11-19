# 記憶的寫法：平均、遺忘門、delta rule

**問題**：格子裡的記憶該怎麼寫？

**簡短回答**：delta rule 遠好於平均：4.112 → 3.998 → 3.898。

## 怎麼做的

- 從原版 U-Avg 出發，一次只改記憶的寫法，其餘（位址、卷積、FFN、超參數）不變。15M，dim 288，12 個 block，長度 1,024，每步 131,072 tokens，2,300 步（約 3.0 億 tokens），lr 5e-4，warmup 360，cosine 到 0，weight decay 0.1，β₂ 0.95。
- U-Avg：累計平均，每層一個遺忘率。U-Gate-fastinit：只把遺忘率起點調快（對照）。U-Gate-nogate：每格各自的速率、沒有門（消融）。U-Gate：每格、每 token 的遺忘門。U-Delta：delta rule + U-Gate 的門。
- 門的形狀另試三種：低秩 64、從 block 輸入 h 讀、門裡加 SiLU。

## 結果

### 驗證 NLL（15M）

| 模型 | seed 數 | 參數量 | NLL @1024 | @4096 | 前 64 | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [U-Avg](../variants/u-avg.md) | 3 | 15.17M | 4.112（4.107–4.116） | 5.658 | 4.392 | 4.087 |
| [U-Gate-nogate](../variants/u-gate-nogate.md) | 1 | 15.18M | 4.124 | 5.886 | 4.399 | 4.101 |
| [U-Gate-fastinit](../variants/u-gate-fastinit.md) | 3 | 15.17M | 4.051（4.049–4.052） | 4.029 | 4.332 | 4.024 |
| [U-Gate](../variants/u-gate.md) | 3 | 15.46M | 3.998（3.988–4.003） | 3.963 | 4.315 | 3.959 |
| [U-Gate-fromh](../variants/u-gate-fromh.md) | 1 | 15.38M | 4.012 | 3.978 | 4.325 | 3.974 |
| [U-Gate-silu](../variants/u-gate-silu.md) | 1 | 15.46M | 4.002 | 3.966 | 4.324 | 3.962 |
| [U-Gate-rank64](../variants/u-gate-rank64.md) | 1 | 16.28M | 3.977 | 3.942 | 4.299 | 3.938 |
| [U-Delta](../variants/u-delta.md) | 3 | 15.45M | 3.898（3.895–3.901） | 3.857 | 4.261 | 3.852 |
| [Transformer](../variants/transformer.md) | 3 | 15.19M | 3.945（3.928–3.957） | 5.016 | 4.237 | 3.918 |

## 結論

- **delta rule 遠好於平均**：U-Avg 4.112 → U-Gate 3.998 → U-Delta 3.898（各 3 個 seed）。只有 U-Delta 贏 Transformer（3.945）。
- U-Gate 的進步有 53.8% 只是“忘得比較快”：U-Gate-fastinit 只調了起點就到 4.051。
- 每格各自的速率沒用：U-Gate-nogate 4.124 跟 U-Avg 一樣，速率幾乎沒離開起點；進步來自按內容的門。
- 門的形狀影響小：低秩 64 好 0.021（多 0.8M 參數），從 h 讀、加 SiLU 都沒幫助。
- 過了訓練長度：U-Avg、U-Gate-nogate 在長度 4,096 崩（5.0–6.2）；忘得快的 U-Gate-fastinit 和有門的都不崩。

## 注意

- U-Gate-nogate、門的三種形狀各 1 個 seed。

## 涉及的變體

[U-Avg](../variants/u-avg.md)、[U-Gate-nogate](../variants/u-gate-nogate.md)、[U-Gate-fastinit](../variants/u-gate-fastinit.md)、[U-Gate](../variants/u-gate.md)、[U-Gate-fromh](../variants/u-gate-fromh.md)、[U-Gate-silu](../variants/u-gate-silu.md)、[U-Gate-rank64](../variants/u-gate-rank64.md)、[U-Delta](../variants/u-delta.md)
