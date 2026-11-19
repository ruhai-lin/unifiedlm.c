# FFN 那一半讀哪個位址

**問題**：U-Delta 到 42M 為什麼輸 Transformer？

**簡短回答**：FFN 讀歸一化位址是瓶頸；改讀 relu² 後 15M 好 0.086、42M 好 0.122。

## 怎麼做的

- U-Delta 的 FFN 那一半讀的是歸一化位址 `r = z / (eps + Σz)`，`z = relu(K u + b)²`：除以總和之後，token 激活多強的資訊就沒了。
- U-Delta-rawFFN 只改這一處：FFN 讀 `z`（就是標準的 ReLU² FFN），記憶照舊用歸一化的 `r`。不加參數。
- 15M 和 42M 各跑一次，同 U-Delta 的設定。

## 結果

### 15M

| 模型 | seed 數 | 參數量 | NLL @1024 | @4096 | 前 64 | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [U-Delta](../variants/u-delta.md) | 3 | 15.45M | 3.898（3.895–3.901） | 3.857 | 4.261 | 3.852 |
| [U-Delta-rawFFN](../variants/u-delta-rawffn.md) | 1 | 15.45M | 3.812 | 3.774 | 4.170 | 3.768 |
| [Transformer](../variants/transformer.md) | 3 | 15.19M | 3.945（3.928–3.957） | 5.016 | 4.237 | 3.918 |

### 42M

| 模型 | seed 數 | 參數量 | NLL @1024 | @4096 | 前 64 | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [U-Delta](../variants/u-delta.md) | 1 | 41.77M | 3.350 | 3.303 | 3.757 | 3.299 |
| [U-Delta-rawFFN](../variants/u-delta-rawffn.md) | 1 | 41.77M | 3.228 | 3.183 | 3.623 | 3.179 |
| [Transformer](../variants/transformer.md) | 3 | 41.69M | 3.214（3.214–3.215） | 4.860 | 3.627 | 3.164 |

### 42M 按位置

| 模型 | 0–64 | 64–128 | 128–256 | 256–512 | 512–1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [U-Delta](../variants/u-delta.md) | 3.757 | 3.437 | 3.369 | 3.317 | 3.299 |
| [U-Delta-rawFFN](../variants/u-delta-rawffn.md) | 3.623 | 3.307 | 3.245 | 3.196 | 3.179 |
| [Transformer](../variants/transformer.md) | 3.627 | 3.294 | 3.233 | 3.182 | 3.164 |
| U-Delta-rawFFN − U-Delta | -0.134 | -0.130 | -0.124 | -0.121 | -0.120 |

## 結論

- FFN 讀歸一化位址是瓶頸：15M 好 0.086（U-Delta 三個 seed 才相差 0.006），42M 好 0.122。
- 每個位置都好差不多的量，所以是基本建模能力，不是記憶。
- 統一 block 原本最弱的開頭 64 個 token 補平了：15M 4.170，GDN 4.166，Transformer 4.224。
- 統一性沒變：FFN 和記憶仍然由同一個 `z`（同一組格子、同一個投影）選格子，只是 FFN 不除以總和。

## 注意

- 各 1 個 seed。

## 涉及的變體

[U-Delta](../variants/u-delta.md)、[U-Delta-rawFFN](../variants/u-delta-rawffn.md)、[U-Delta-rawFFN+λlr1000](../variants/u-delta-rawffn-lambda-lr1000.md)
