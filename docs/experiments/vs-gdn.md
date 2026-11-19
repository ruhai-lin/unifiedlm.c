# U-Delta 為什麼贏 GDN

**問題**：同樣是 delta rule，差在哪？

**簡短回答**：不是卷積、狀態大小或寫入規則，是以 FFN 為錨的格子。

## 怎麼做的

- U-Delta 和 GDN 都用 delta rule，為什麼 U-Delta 贏？一個一個排除：
- 卷積：Transformer+conv 在 attention 前加同樣的寬 4 因果卷積。狀態大小：GDN-bigstate 用 2 個 144 維的大 head，狀態約 3 倍。寫入規則：GDN-2（抹除和寫入門分開、按通道衰減）和 Mesa（每個 token 精確解最小平方）。
- 全部 15M，同一組超參數；GDN、GDN-2、Mesa 都是把 Transformer 的 attention 半層換掉，FFN 半層不變。

## 結果

### 驗證 NLL（15M）

| 模型 | seed 數 | 參數量 | NLL @1024 | @4096 | 前 64 | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Transformer](../variants/transformer.md) | 3 | 15.19M | 3.945（3.928–3.957） | 5.016 | 4.237 | 3.918 |
| [Transformer+conv](../variants/transformer-conv.md) | 1 | 15.20M | 3.921 | 4.910 | 4.237 | 3.889 |
| [GDN](../variants/gdn.md) | 1 | 15.23M | 3.964 | 3.948 | 4.166 | 3.946 |
| [GDN-bigstate](../variants/gdn-bigstate.md) | 1 | 15.22M | 3.964 | 3.948 | 4.160 | 3.947 |
| [GDN-2](../variants/gdn-2.md) | 1 | 15.21M | 3.970 | 3.954 | 4.171 | 3.952 |
| [Mesa](../variants/mesa.md) | 1 | 15.24M | 3.969 | 3.951 | 4.188 | 3.949 |
| [U-Delta](../variants/u-delta.md) | 3 | 15.45M | 3.898（3.895–3.901） | 3.857 | 4.261 | 3.852 |
| [U-Delta-rawFFN](../variants/u-delta-rawffn.md) | 1 | 15.45M | 3.812 | 3.774 | 4.170 | 3.768 |
| [U-Delta+attn](../variants/u-delta-plus-attn.md) | 1 | 15.45M | 3.840 | 3.798 | 4.212 | 3.793 |

## 結論

- **不是卷積**：Transformer+conv 3.921，只比同 seed 的 Transformer（3.928）好 0.007。
- **不是狀態大小或寫入規則**：GDN 3.964、GDN-bigstate 3.964、GDN-2 3.970、Mesa 3.969 都在同一檔，都輸 U-Delta（3.898）。
- **差別在格子本身**：U-Delta 的記憶以 FFN 為錨、用同一組格子的位址，贏在 64 之後的位置；GDN 贏在開頭 64 個 token（4.166 對 4.258）。這個弱點來自 FFN 讀歸一化位址，U-Delta-rawFFN 補平了（4.170）。

## 注意

- 各 1 個 seed（U-Delta、Transformer 3 個）。

## 涉及的變體

[Transformer](../variants/transformer.md)、[Transformer+conv](../variants/transformer-conv.md)、[GDN](../variants/gdn.md)、[GDN-bigstate](../variants/gdn-bigstate.md)、[GDN-2](../variants/gdn-2.md)、[Mesa](../variants/mesa.md)、[U-Delta](../variants/u-delta.md)、[U-Delta-rawFFN](../variants/u-delta-rawffn.md)、[U-Delta+attn](../variants/u-delta-plus-attn.md)
