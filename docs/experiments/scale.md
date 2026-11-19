# 規模：15M → 42M → 110M

**問題**：15M 贏 Transformer 會不會只是小模型的現象？

**簡短回答**：15M 贏；42M 打平到略輸：最好的 U-Delta-rawFFN+λlr1000 輸 0.009（1 個 seed）。

## 怎麼做的

- 15M：dim 288，12 個 block，長度 1,024，每步 131,072 tokens，2,300 步（約 3.0 億 tokens），lr 5e-4，warmup 360，cosine 到 0，weight decay 0.1，β₂ 0.95。
- 42M：dim 512，16 個 block，6,400 步（約 8.4 億 tokens），warmup 1,000，其餘同 15M。110M：dim 768，24 個 block，16,700 步，用 4 個 parquet 的子集（`FW_SUBSET=4of14`）。
- 每個規模用同一組超參數，學習率都是 5e-4，沒有為任何模型調。參數量對齊到 ±2%（Split、U-Gate-rank64 例外）。
- 驗證 NLL：長度 1,024（2,048 個窗口，約 210 萬 tokens）和 4,096（512 個窗口），按位置分段；`scripts/fineweb/eval_fineweb.py`。

## 結果

### 驗證 NLL（長度 1,024）

| 模型 | 15M | 42M | 110M |
| --- | ---: | ---: | ---: |
| [Transformer](../variants/transformer.md) | 3.945（3 個 seed，3.928–3.957） | 3.214（3 個 seed，3.214–3.215） | 2.822 |
| [UD2-gate-norm-q32-rope](../variants/ud2-gate-norm-q32-rope.md) | 3.722（3 個 seed，3.716–3.727） | 3.171（3 個 seed，3.166–3.176） | — |
| [U-Delta-rawFFN+λlr1000](../variants/u-delta-rawffn-lambda-lr1000.md) | 3.796（3 個 seed，3.793–3.799） | 3.223 | — |
| [U-Delta-rawFFN](../variants/u-delta-rawffn.md) | 3.812 | 3.228 | — |
| [U-Delta](../variants/u-delta.md) | 3.898（3 個 seed，3.895–3.901） | 3.350 | — |
| [U-Gate](../variants/u-gate.md) | 3.998（3 個 seed，3.988–4.003） | 3.330（3 個 seed，3.328–3.333） | — |
| [U-Gate-fastinit](../variants/u-gate-fastinit.md) | 4.051（3 個 seed，4.049–4.052） | 3.361 | — |
| [Split](../variants/split.md) | 4.038（3 個 seed，4.034–4.040） | 3.368 | — |
| [U-Avg](../variants/u-avg.md) | 4.112（3 個 seed，4.107–4.116） | 3.423 | 2.994 |

### 42M 按位置（長度 1,024）

| 模型 | 0–64 | 64–128 | 128–256 | 256–512 | 512–1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Transformer](../variants/transformer.md) | 3.627 | 3.294 | 3.233 | 3.182 | 3.164 |
| [UD2-gate-norm-q32-rope](../variants/ud2-gate-norm-q32-rope.md) | 3.584 | 3.255 | 3.191 | 3.139 | 3.119 |
| [U-Delta-rawFFN+λlr1000](../variants/u-delta-rawffn-lambda-lr1000.md) | 3.624 | 3.305 | 3.241 | 3.193 | 3.174 |
| [U-Delta-rawFFN](../variants/u-delta-rawffn.md) | 3.623 | 3.307 | 3.245 | 3.196 | 3.179 |
| [U-Delta](../variants/u-delta.md) | 3.757 | 3.437 | 3.369 | 3.317 | 3.299 |
| U-Delta-rawFFN+λlr1000 − Transformer | -0.004 | +0.011 | +0.008 | +0.010 | +0.010 |

### 15M 按位置（長度 1,024，各 3 個 seed 平均）

| 模型 | 0–64 | 64–128 | 128–256 | 256–512 | 512–1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Transformer](../variants/transformer.md) | 4.237 | 3.979 | 3.937 | 3.923 | 3.918 |
| [U-Delta-rawFFN+λlr1000](../variants/u-delta-rawffn-lambda-lr1000.md) | 4.154 | 3.872 | 3.812 | 3.770 | 3.751 |
| [U-Delta](../variants/u-delta.md) | 4.261 | 3.978 | 3.914 | 3.870 | 3.852 |
| U-Delta-rawFFN+λlr1000 − Transformer | -0.083 | -0.107 | -0.125 | -0.154 | -0.167 |

## 結論

- **15M 統一 block 贏 Transformer**：U-Delta-rawFFN+λlr1000 3.796（3 個 seed，3.793–3.799） 對 Transformer 3.945（3 個 seed，3.928–3.957），最差的 seed 也比 Transformer 最好的 seed 好 0.128。U-Delta 本身 3.898（3 個 seed，3.895–3.901） 也贏。
- **42M：UD2-gate-norm-q32-rope 贏 Transformer 0.044（各 1 個 seed）**：3.170 對 3.214，每一段位置都贏 0.038–0.045，前 64 個 token 也贏；長度 4,096 是 3.121（Transformer 過了訓練長度升到 4.871）。比 U-Delta-rawFFN+λlr1000 42M 好 0.054。這個模型是在 15M、同一個 seed 從十個讀出端變體裡挑出來的，要補 seed 才能確定。
- **U-Delta 系列 42M 打平到略輸**：U-Delta 3.350 輸 0.136，FFN 改讀未歸一化位址的 U-Delta-rawFFN 3.228 輸 0.014，再加遺忘率學快 1,000 倍的 U-Delta-rawFFN+λlr1000 3.223 輸 0.009（各 1 個 seed，比 15M Transformer 的 seed 標準差 0.016 小）。遺忘率學快在 15M 好 0.016，在 42M 只好 0.004。前 64 個 token 還略贏，之後每段輸 0.009–0.012。15M 的領先（U-Delta 0.047、U-Delta-rawFFN 0.133、U-Delta-rawFFN+λlr1000 0.149）到 42M 都沒有了。
- **15M → 42M 變的是 Transformer**：15M 的 Transformer 從 64–128 段到 512–1,024 段只降 0.064，42M 降 0.129；U-Delta-rawFFN 兩個規模都約 0.12–0.13。15M 越往後贏越多，有一部分是小 Transformer 不太會用前文。
- **110M 只跑了原版**：U-Avg 2.994 對 Transformer 2.822。U-Delta 系列還沒跑 110M。

## 注意

- 42M 和 110M 只有 1 個 seed（U-Gate 42M 例外，3 個 seed 3.328–3.333）。
- UD2-gate-norm-q32-rope 42M 的 hidden_dim 是 1,195（41.77M 參數），其他設定跟 U-Delta-rawFFN+λlr1000 42M 一樣。
- 學習率沒調，Transformer 在 1e-3 可能更好，還沒試。

## 涉及的變體

[Transformer](../variants/transformer.md)、[UD2-gate-norm-q32-rope](../variants/ud2-gate-norm-q32-rope.md)、[U-Delta-rawFFN+λlr1000](../variants/u-delta-rawffn-lambda-lr1000.md)、[U-Delta-rawFFN](../variants/u-delta-rawffn.md)、[U-Delta](../variants/u-delta.md)、[U-Gate](../variants/u-gate.md)、[U-Gate-fastinit](../variants/u-gate-fastinit.md)、[Split](../variants/split.md)、[U-Avg](../variants/u-avg.md)
