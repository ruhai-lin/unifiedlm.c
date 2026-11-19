# 寫入量與 loss

**問題**：記憶是不是只寫沒預測到的 token？

**簡短回答**：只是偏向，相關 0.14–0.21；自己預測到的也照寫。

## 怎麼做的

- 每個 token、每層的寫入量 `|δ_t| = β_t |u_t − E_{t−1}ᵀ diag(λ_t) k_t|`（記憶裡還沒有的那部分），對上該 token 的 loss `−log p(x_t | x_<t)`。第 t 步的寫入用 t−1 的位址存 token t 的值，所以對的是 t−1 位置的 logits。
- 報 Spearman 相關（寫入量、寫入目標 |u|、寫入強度 β 各對 loss），以及按 loss 十分位的寫入量。
- 生成測試：512 個 token 的真實前文之後，模型自己生成 256 個 token（按概率採樣、greedy 各一次），跟真實後文同位置比寫入量。
- 逐 token 重算記憶，跟 kernel 的輸出比對（相對誤差 2×10⁻⁶）。16 段驗證文字。`scripts/fineweb/writeloss.py`。

## 結果

### 寫入量與 loss

| 模型 | 規模 | 寫入量 vs loss | |u| vs loss | β vs loss | loss 最低一成的寫入量 | 最高一成 | 自己採樣 / 真實 | 自己 greedy / 真實 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [U-Delta](../variants/u-delta.md) | 15M | 0.147 | 0.056 | 0.108 | 0.83 | 1.11 | 1.04 | 0.63 |
| [U-Delta-rawFFN](../variants/u-delta-rawffn.md) | 15M | 0.204 | 0.135 | 0.162 | 0.67 | 1.26 | 1.10 | 0.52 |
| [U-Delta](../variants/u-delta.md) | 42M | 0.135 | 0.120 | 0.096 | 0.80 | 1.18 | 1.09 | 0.74 |
| [U-Delta-rawFFN](../variants/u-delta-rawffn.md) | 42M | 0.208 | 0.148 | 0.166 | 0.67 | 1.35 | 1.06 | 0.68 |

## 結論

- **偏向寫意外的 token，但很弱**：寫入量和 loss 的 Spearman 相關只有 0.14–0.21；loss 最低的一成 token 寫入量是平均的 0.67–0.83 倍，最高的一成 1.11–1.35 倍。
- **預測到的也照寫**：模型自己 greedy 生成的文字（幾乎都是自己預測到的），寫入量還有真實文字的 0.52–0.74 倍；按概率採樣時反而多 4–10%。
- U-Delta-rawFFN 比 U-Delta 更挑一點（0.20–0.21 對 0.14–0.15）。寫入目標 |u| 和寫入強度 β 跟 loss 的相關也都很弱（0.06–0.17）。

## 注意

- 各 1 個 seed，16 段文字、8 段生成。

## 涉及的變體

[U-Delta](../variants/u-delta.md)、[U-Delta-rawFFN](../variants/u-delta-rawffn.md)
