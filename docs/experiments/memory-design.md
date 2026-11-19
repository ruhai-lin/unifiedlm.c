# 記憶設計：快記憶、慢記憶、遺忘率

**問題**：需要慢記憶嗎？遺忘率怎麼設？

**簡短回答**：不需要慢記憶；記憶越短越好，短到上限還在變好。

## 怎麼做的

- 都從 U-Delta 出發，只改記憶的遺忘率或記憶份數，15M，dim 288，12 個 block，長度 1,024，每步 131,072 tokens，2,300 步（約 3.0 億 tokens），lr 5e-4，warmup 360，cosine 到 0，weight decay 0.1，β₂ 0.95。
- 兩份記憶：U-Delta+slow（快 + 慢，讀出混合 ρ）；同容量對照 U-Delta+fast2（兩份都是快的）。
- 單一記憶、改遺忘率：初始值分散（spreadλ）、`log_rate` 學快 10 / 100 / 1000 倍（λlr10 / λlr100 / λlr1000，重參數化，起點不變）、基礎記憶長度 4–65,536 分散且門有上限（diverseλ）、文件開頭清空（docreset）。
- “有效記憶長度”：遺忘門實際衰減到 1/e 的 token 數，64 段驗證文字平均（`scripts/fineweb/gates.py`），只算衰減、不算覆寫。
- 讀到 20K：64 個長 20,480 的窗口（`scripts/fineweb/long_eval.py`）。

## 結果

### 驗證 NLL（15M）

| 模型 | seed 數 | 參數量 | NLL @1024 | @4096 | 前 64 | 512–1,024 | 16K–20K（讀到 20K） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [U-Delta](../variants/u-delta.md) | 3 | 15.45M | 3.898（3.895–3.901） | 3.857 | 4.261 | 3.852 | 3.816 |
| [U-Delta+slow](../variants/u-delta-plus-slow.md) | 1 | 15.47M | 3.888 | 3.851 | 4.258 | 3.841 | 3.856 |
| [U-Delta+fast2](../variants/u-delta-plus-fast2.md) | 1 | 15.75M | 3.897 | 3.853 | 4.270 | 3.849 | 3.814 |
| [U-Delta-spreadλ](../variants/u-delta-spread-lambda.md) | 1 | 15.45M | 3.890 | 3.848 | 4.254 | 3.843 | 3.807 |
| [U-Delta-λlr10](../variants/u-delta-lambda-lr10.md) | 1 | 15.45M | 3.887 | 3.846 | 4.251 | 3.841 | 3.806 |
| [U-Delta-diverseλ](../variants/u-delta-diverse-lambda.md) | 1 | 15.45M | 3.877 | 3.845 | 4.245 | 3.832 | 3.859 |
| [U-Delta-docreset](../variants/u-delta-docreset.md) | 1 | 15.45M | 3.890 | 3.843 | 4.261 | 3.842 | 3.797 |
| [U-Delta-λlr100](../variants/u-delta-lambda-lr100.md) | 1 | 15.45M | 3.851 | 3.809 | 4.219 | 3.805 | 3.769 |
| [U-Delta-λlr1000](../variants/u-delta-lambda-lr1000.md) | 1 | 15.45M | 3.843 | 3.796 | 4.220 | 3.794 | 3.752 |
| [U-Delta-rawFFN](../variants/u-delta-rawffn.md) | 1 | 15.45M | 3.812 | 3.774 | 4.170 | 3.768 | 3.760 |
| [U-Delta-rawFFN+λlr1000](../variants/u-delta-rawffn-lambda-lr1000.md) | 3 | 15.45M | 3.796（3.793–3.799） | 3.753 | 4.154 | 3.751 | 3.713 |

### 每層有效記憶長度（token，第 0 → 11 層）

| 模型 | 每層 |
| --- | --- |
| [U-Delta](../variants/u-delta.md) | 7 / 373 / 315 / 434 / 259 / 23 / 76 / 95 / 86 / 84 / 65 / 314 |
| [U-Delta+slow](../variants/u-delta-plus-slow.md) | 9 / 327 / 161 / 323 / 491 / 159 / 96 / 93 / 109 / 224 / 83 / 76 |
| [U-Delta+fast2](../variants/u-delta-plus-fast2.md) | 9 / 213 / 256 / 1643 / 255 / 99 / 175 / 313 / 254 / 117 / 91 / 126 |
| [U-Delta-spreadλ](../variants/u-delta-spread-lambda.md) | 7 / 252 / 145 / 331 / 17 / 15 / 19 / 23 / 26 / 15 / 20 / 285 |
| [U-Delta-λlr10](../variants/u-delta-lambda-lr10.md) | 4 / 254 / 209 / 296 / 173 / 6 / 18 / 31 / 25 / 22 / 23 / 254 |
| [U-Delta-diverseλ](../variants/u-delta-diverse-lambda.md) | 17 / 29 / 47 / 18 / 18 / 18 / 19 / 20 / 20 / 20 / 39 / 29 |
| [U-Delta-docreset](../variants/u-delta-docreset.md) | 12 / 572 / 244 / 650 / 33 / 51 / 70 / 91 / 92 / 78 / 66 / 344 |
| [U-Delta-λlr100](../variants/u-delta-lambda-lr100.md) | 2 / 139 / 83 / 30 / 19 / 2 / 1 / 1 / 1 / 1 / 2 / 34 |
| [U-Delta-λlr1000](../variants/u-delta-lambda-lr1000.md) | 1 / 53 / 14 / 9 / 1 / 1 / 1 / 1 / 1 / 1 / 1 / 2 |
| [U-Delta-rawFFN](../variants/u-delta-rawffn.md) | 5 / 509 / 464 / 29 / 13 / 19 / 13 / 138 / 135 / 17 / 14 / 68 |
| [U-Delta-rawFFN+λlr1000](../variants/u-delta-rawffn-lambda-lr1000.md) | 2 / 38 / 22 / 1 / 2 / 1 / 1 / 1 / 2 / 1 / 2 / 1 |

## 結論

- **不是容量**：同容量、兩份都是快記憶的 U-Delta+fast2 跟 U-Delta 一樣。
- **單一記憶就追平兩份記憶**：分散初始值（spreadλ）或遺忘率學快 10 倍（λlr10），都比 U-Delta 好約 0.01，跟 U-Delta+slow 一樣，參數不變、狀態減半。兩者學到幾乎一樣的分工：中間層（第 6–10 層）從 65–95 降到 15–31 個 token。
- **記憶短一點更好，短到上限還在變好**：λlr100 3.851，λlr1000 3.843；λlr1000 按格子平均，除了第 1–3 層，每層都忘到上限（每個 token 最多忘 63%，約 1 個 token；按寫入量加權，第 1、2、3、11 層過了 256 個 token 還留 34/18/19/21%，第 4–10 層不留，見[每個格子記多久](slot-retention.md)），長度 4,096 跟含 attention 的 U-Delta+attn 一樣。
- **強迫留住慢記憶沒用**：U-Delta+slow 的慢記憶沒學（速率停在初始值），diverseλ 的慢格子留住了；兩者讀到 4K 以後都比 U-Delta 差。把記憶縮短的 λlr100、λlr1000 反而越讀越好。
- **和 rawFFN 合併只疊加一部分**：λlr1000 單獨好 0.055，加在 U-Delta-rawFFN 上只再好 0.016；但外推的好處留住了，讀到 20K 比 U-Delta-rawFFN 好的幅度從開頭的 0.017 增加到 16K–20K 的 0.047。
- **文件開頭清空記憶小幅有用**：docreset 好約 0.008，長度 4,096 和讀到 20K 時好得更多。

## 注意

- 除 U-Delta 和 U-Delta-rawFFN+λlr1000 外都只有 1 個 seed；0.01 左右的差距要小心（U-Delta 的 seed 標準差約 0.003）。

## 涉及的變體

[U-Delta](../variants/u-delta.md)、[U-Delta+slow](../variants/u-delta-plus-slow.md)、[U-Delta+fast2](../variants/u-delta-plus-fast2.md)、[U-Delta-spreadλ](../variants/u-delta-spread-lambda.md)、[U-Delta-λlr10](../variants/u-delta-lambda-lr10.md)、[U-Delta-diverseλ](../variants/u-delta-diverse-lambda.md)、[U-Delta-docreset](../variants/u-delta-docreset.md)、[U-Delta-λlr100](../variants/u-delta-lambda-lr100.md)、[U-Delta-λlr1000](../variants/u-delta-lambda-lr1000.md)、[U-Delta-rawFFN](../variants/u-delta-rawffn.md)、[U-Delta-rawFFN+λlr1000](../variants/u-delta-rawffn-lambda-lr1000.md)
