# 含 attention 的對照

**問題**：加回局部 attention 能好多少？

**簡短回答**：好 0.058，但不含 attention 的最新版本已經贏它。

## 怎麼做的

- 含 attention 的版本只當參考，不進主線。U-Delta+attn：每 4 個 U-Delta block 後插一個窗口 128 的局部 attention 半層，`hidden_dim` 降到 576 讓參數一樣。
- Transformer+conv、Transformer+SSMax 是 Transformer 的改法，用來排除卷積和外推的解釋。

## 結果

### 驗證 NLL（15M）

| 模型 | seed 數 | 參數量 | NLL @1024 | @4096 | 前 64 | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [U-Delta+attn](../variants/u-delta-plus-attn.md) | 1 | 15.45M | 3.840 | 3.798 | 4.212 | 3.793 |
| [U-Delta](../variants/u-delta.md) | 3 | 15.45M | 3.898（3.895–3.901） | 3.857 | 4.261 | 3.852 |
| [U-Delta-rawFFN](../variants/u-delta-rawffn.md) | 1 | 15.45M | 3.812 | 3.774 | 4.170 | 3.768 |
| [U-Delta-rawFFN+λlr1000](../variants/u-delta-rawffn-lambda-lr1000.md) | 3 | 15.45M | 3.796（3.793–3.799） | 3.753 | 4.154 | 3.751 |
| [Transformer+conv](../variants/transformer-conv.md) | 1 | 15.20M | 3.921 | 4.910 | 4.237 | 3.889 |
| [Transformer+SSMax](../variants/transformer-ssmax.md) | 1 | 15.19M | 3.913 | 5.101 | 4.230 | 3.880 |
| [Transformer](../variants/transformer.md) | 3 | 15.19M | 3.945（3.928–3.957） | 5.016 | 4.237 | 3.918 |

## 結論

- 局部 attention 讓 U-Delta 好 0.057，補的是開頭（前 64 個 token 4.212 對 4.258）。
- 不含 attention 的 U-Delta-rawFFN（3.812）和 U-Delta-rawFFN+λlr1000（3.796）都已經贏 U-Delta+attn，所以主線不需要加 attention。
- Transformer+SSMax 在 1,024 內好 0.015（1 個 seed），但過了 1,024 照樣崩。

## 注意

- 各 1 個 seed。

## 涉及的變體

[U-Delta+attn](../variants/u-delta-plus-attn.md)、[Transformer+conv](../variants/transformer-conv.md)、[Transformer+SSMax](../variants/transformer-ssmax.md)
