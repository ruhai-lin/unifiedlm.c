# 記憶任務 m2kv

**問題**：純記憶題能記多遠？

**簡短回答**：只有 U-Delta 系列外推：間隔 4,096 仍有 99.95%。

## 怎麼做的

- m2kv：開頭給 16 組“鑰匙 → 值”，中間塞 G 個填充 token（從鑰匙和值的範圍裡抽，但不是被問的鑰匙，所以會干擾），最後逐一問鑰匙。只看答案位置的正確率。
- 寬度 32（記憶很緊），訓練時 G 最多 256；表中 G = 1,024、4,096 都沒訓練過，8,192 以上是不重訓直接拉長（`scripts/memory/mqar.py`、`scripts/memory/mem_long.py`）。1 個 seed。

## 結果

### 正確率

| 模型 | 間隔 0 | 間隔 64 | 間隔 256 | 間隔 1024 | 間隔 4096 | 間隔 8192 | 間隔 16384 | 間隔 20000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Transformer](../variants/transformer.md) | 99.6% | 99.8% | 99.7% | 0.0% | 0.0% | — | — | — |
| [U-Avg](../variants/u-avg.md) | 99.9% | 100.0% | 99.9% | 18.5% | 0.0% | — | — | — |
| [U-Gate-fastinit](../variants/u-gate-fastinit.md) | 90.2% | 90.8% | 90.7% | 44.0% | 0.0% | 0.0% | — | — |
| [U-Gate](../variants/u-gate.md) | 99.6% | 99.7% | 99.5% | 98.5% | 39.6% | 3.7% | 0.5% | 0.3% |
| [U-Delta](../variants/u-delta.md) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 95.4% | 55.1% | 33.9% |
| [U-Delta+slow](../variants/u-delta-plus-slow.md) | 100.0% | 100.0% | 100.0% | 100.0% | 98.3% | 95.3% | 77.0% | 66.9% |

## 結論

- **只有 U-Delta 系列外推**：間隔 4,096 時 U-Delta 100.0%，U-Gate 39.6%，Transformer 和 U-Avg 0%。
- 不重訓再拉長：U-Delta 8,192 95.4%、16,384 55.1%、20,000 33.9%；U-Delta+slow 16,384 77.0%、20,000 66.9%。慢記憶在真的需要記很久時有用，只是自然文字上用不到（見[記憶設計](memory-design.md)）。
- U-Gate-fastinit 忘得快，間隔 1,024 就只剩 44%。

## 注意

- 1 個 seed，寬度 32。GPU 訓練不完全可重現，同一 seed 重跑會有差異。

## 涉及的變體

[Transformer](../variants/transformer.md)、[U-Avg](../variants/u-avg.md)、[U-Gate-fastinit](../variants/u-gate-fastinit.md)、[U-Gate](../variants/u-gate.md)、[U-Delta](../variants/u-delta.md)、[U-Delta+slow](../variants/u-delta-plus-slow.md)
