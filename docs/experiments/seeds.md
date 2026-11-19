# 多 seed 的穩定性

**問題**：差距是不是 seed 的噪聲？

**簡短回答**：統一 block 的 seed 標準差約 0.003，Transformer 約 0.016；最新的統一 block 三個 seed 都贏。

## 怎麼做的

- seed 1337、1338、1339。換 seed 同時換初始化和資料順序（`fineweb_subset.DATA_SEED = seed − 1337`）。
- 15M，同各模型的設定。

## 結果

### seed 間的差距（驗證 NLL，長度 1,024）

| 模型 | 規模 | seed 數 | 平均 | 標準差 | 範圍 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Transformer](../variants/transformer.md) | 15M | 3 | 3.945 | 0.016 | 3.928–3.957 |
| [UD2-gate](../variants/ud2-gate.md) | 15M | 3 | 3.765 | 0.003 | 3.761–3.767 |
| [U-Delta-rawFFN+λlr1000](../variants/u-delta-rawffn-lambda-lr1000.md) | 15M | 3 | 3.796 | 0.003 | 3.793–3.799 |
| [U-Delta](../variants/u-delta.md) | 15M | 3 | 3.898 | 0.003 | 3.895–3.901 |
| [U-Gate](../variants/u-gate.md) | 15M | 3 | 3.998 | 0.008 | 3.988–4.003 |
| [U-Gate-fastinit](../variants/u-gate-fastinit.md) | 15M | 3 | 4.051 | 0.002 | 4.049–4.052 |
| [Split](../variants/split.md) | 15M | 3 | 4.038 | 0.003 | 4.034–4.040 |
| [U-Avg](../variants/u-avg.md) | 15M | 3 | 4.112 | 0.005 | 4.107–4.116 |
| [U-Gate](../variants/u-gate.md) | 42M | 3 | 3.330 | 0.003 | 3.328–3.333 |

## 結論

- **統一 block 的 seed 差距很小**（標準差 0.002–0.008），Transformer 大約 0.016。
- **U-Delta-rawFFN+λlr1000 三個 seed 都贏 Transformer**：同 seed 配對每組差 -0.161 到 -0.134。
- **UD2-gate 三個 seed 都比 UD2 好**：3.766, 3.767, 3.761 對 3.778（UD2 只有 1 個 seed）。
- 只有 1 個 seed 的結果，差距小於約 0.02 時要小心。

## 涉及的變體

[Transformer](../variants/transformer.md)、[UD2-gate](../variants/ud2-gate.md)、[U-Delta-rawFFN+λlr1000](../variants/u-delta-rawffn-lambda-lr1000.md)、[U-Delta](../variants/u-delta.md)、[U-Gate](../variants/u-gate.md)、[U-Gate-fastinit](../variants/u-gate-fastinit.md)、[Split](../variants/split.md)、[U-Avg](../variants/u-avg.md)
