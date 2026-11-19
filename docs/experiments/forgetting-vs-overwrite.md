# 內容是被遺忘還是被覆寫

**問題**：格子丟掉內容，是衰減還是被新寫入覆蓋？

**簡短回答**：典型格子是衰減（覆寫不到 1%）；少數常被寫的格子覆寫較重。

## 怎麼做的

- delta 更新 `E_t = (I − β_t k_t k_tᵀ) Λ_t E_{t−1} + …` 在衰減 `1 − λ_{t,a}` 之外，從格子 a 再拿走約 `β_t k_{t,a}²`（k 單位化，忽略格子之間的混合）。
- 每格對 token 平均：衰減 `d_a = mean −log λ_{t,a}`，覆寫 `o_a = mean −log(1 − β_t k_{t,a}²)`；時間常數 1/d、1/o、1/(d+o)，覆寫占比 o/(d+o)。
- 按每格的基礎記憶長度（1/rate_a）分組報中位數，也按層報平均。16 段驗證文字。`scripts/fineweb/overwrite.py`。

## 結果

### 按基礎記憶長度分組（每組格子的中位數）

| 模型 | 基礎記憶長度（token） | 格子數 | 衰減時間常數 | 覆寫時間常數 | 覆寫占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [U-Delta](../variants/u-delta.md) | 1,024–4,096 | 8,357 | 113 | 16,524 | 0.6% |
| [U-Delta](../variants/u-delta.md) | 4,096–8,192 | 187 | 2,880 | 18,894 | 14.2% |
| [U-Delta-spreadλ](../variants/u-delta-spread-lambda.md) | 0–1,024 | 3,017 | 15 | 9,558 | 0.2% |
| [U-Delta-spreadλ](../variants/u-delta-spread-lambda.md) | 1,024–4,096 | 2,858 | 78 | 15,026 | 0.5% |
| [U-Delta-spreadλ](../variants/u-delta-spread-lambda.md) | 4,096–8,192 | 1,423 | 195 | 20,654 | 0.9% |
| [U-Delta-spreadλ](../variants/u-delta-spread-lambda.md) | ≥ 8,192 | 1,246 | 354 | 24,612 | 1.3% |
| [U-Delta-λlr10](../variants/u-delta-lambda-lr10.md) | 0–1,024 | 2,220 | 10 | 8,102 | 0.1% |
| [U-Delta-λlr10](../variants/u-delta-lambda-lr10.md) | 1,024–4,096 | 6,208 | 164 | 16,742 | 0.7% |
| [U-Delta-λlr10](../variants/u-delta-lambda-lr10.md) | 4,096–8,192 | 113 | 2,833 | 16,595 | 14.4% |
| [U-Delta-λlr10](../variants/u-delta-lambda-lr10.md) | ≥ 8,192 | 3 | 150,700 | 7,373 | 88.5% |

### 按層平均的覆寫占比（%，第 0 → 11 層）

| 模型 | 每層 |
| --- | --- |
| [U-Delta](../variants/u-delta.md) | 0.4 / 5.9 / 1.2 / 4.3 / 0.4 / 0.8 / 22.0 / 18.4 / 20.4 / 13.1 / 5.0 / 1.9 |
| [U-Delta-spreadλ](../variants/u-delta-spread-lambda.md) | 0.4 / 4.2 / 0.6 / 2.6 / 0.2 / 1.6 / 6.7 / 4.6 / 8.0 / 0.9 / 0.8 / 2.9 |
| [U-Delta-λlr10](../variants/u-delta-lambda-lr10.md) | 0.2 / 4.1 / 0.9 / 2.7 / 0.4 / 0.2 / 5.7 / 5.3 / 5.5 / 3.0 / 1.5 / 1.6 |

## 結論

- **典型格子的內容是被遺忘掉的，不是被覆寫掉的**：絕大多數格子（基礎記憶長度 4,096 以下，U-Delta 8,544 格中有 8,357 格）覆寫占比中位數不到 1%。
- **遺忘門讓實際遺忘比基礎速率快約 10–35 倍**：U-Delta 基礎記 1K–4K 個 token 的格子，實際約 113 個 token 就衰減到 1/e。
- **少數常被寫的格子覆寫比較重**：按層平均（被常寫的格子主導），U-Delta 第 6–9 層有 13–22% 來自覆寫；U-Delta-spreadλ、U-Delta-λlr10 每層都不超過 8%。基礎速率很慢的格子覆寫占比較高（14%，λlr10 裡 3 個最慢的格子 89%）。
- 所以“新內容把舊記憶洗掉”主要靠遺忘門決定，delta rule 的覆寫只在少數格子上起作用。

## 注意

- 覆寫量用 `β k²` 近似，忽略格子之間的混合。
- 各 1 個 seed、16 段文字。

## 涉及的變體

[U-Delta](../variants/u-delta.md)、[U-Delta-spreadλ](../variants/u-delta-spread-lambda.md)、[U-Delta-λlr10](../variants/u-delta-lambda-lr10.md)
