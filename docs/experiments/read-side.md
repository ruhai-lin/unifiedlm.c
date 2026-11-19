# 讀出端：門、縮放、RMSNorm、讀位址修正、旋轉相對位置

**問題**：記憶的讀出怎麼改？

**簡短回答**：讀出門（3 個 seed）、讀出 RMSNorm、讀位址修正、記憶裡的旋轉相對位置都有用，合起來 3.716；每層固定縮放、秩 128、多 head、除以記憶量都沒用。

## 怎麼做的

- UD2（`SDula`）每個 token 把記憶讀出 E_tᵀ q_t 全額加上：y = x + μ z_t + E_tᵀ q_t。這組只改讀出端，寫入不變；各 15M、同一個 seed（1337），UD2-gate 補到 3 個 seed。
- UD2-scale：讀出乘上每層一個可學的純量 2·sigmoid(b)（起點 1）。UD2-gate：乘上每個 token 一個門 o_t = 2·sigmoid(w·h_t + b)。UD2-gate-norm：讀出先 RMSNorm 再過門（GDN 的做法）。
- UD2-gate-q32：讀位址自己多一個秩 32 的修正，q_t = unit(relu(K u_t + b + U V u_t)²)，寫入鑰匙不變，所以比對不再是同一個位址跟自己比；hidden_dim 降到 662 對齊參數。UD2-gate-norm-q32：兩個一起。
- UD2-gate-norm-q32-rope：再把一半格子兩兩成對旋轉，寫入鑰匙按寫入的位置、讀位址按讀的位置轉（週期 4–256 個 token），這些格子的比對只看相對距離，同一對共用衰減；不加參數。UD2-gate-norm-q128-rope：同樣加旋轉，修正用秩 128（hidden_dim 538）。UD2-gate-norm-h4q8：UD2-gate-norm-q32 的讀出拆成 4 個 head，每個 head 有自己秩 8 的修正。UD2-gate-mass-q32：UD2-gate-q32 的讀出除以位址碰到的記憶量。
- 讀出大小：`scripts/fineweb/readstats.py`，32 個長 1,024 的窗口，每層的 |E_tᵀ q_t| 和門的平均，按 token 類別（照抄、見過、新）和位置分組。

## 結果

### 驗證 NLL（15M）

| 模型 | seed 數 | 參數量 | NLL @1024 | @4096 | 前 64 | 512–1,024 | 讀到 20K 的 16K–20K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [UD2](../variants/ud2.md) | 1 | 15.45M | 3.778 | 3.736 | 4.133 | 3.734 | 3.695 |
| [UD2-scale](../variants/ud2-scale.md) | 1 | 15.45M | 3.776 | 3.734 | 4.130 | 3.733 | 3.695 |
| [UD2-gate](../variants/ud2-gate.md) | 3 | 15.46M | 3.765（3.761–3.767） | 3.722 | 4.120 | 3.720 | 3.685 |
| [UD2-gate-norm](../variants/ud2-gate-norm.md) | 1 | 15.46M | 3.739 | 3.696 | 4.122 | 3.691 | 3.661 |
| [UD2-gate-q32](../variants/ud2-gate-q32.md) | 1 | 15.45M | 3.746 | 3.701 | 4.111 | 3.699 | 3.663 |
| [UD2-gate-norm-q32](../variants/ud2-gate-norm-q32.md) | 1 | 15.46M | 3.727 | 3.683 | 4.120 | 3.678 | 3.645 |
| [UD2-gate-q128](../variants/ud2-gate-q128.md) | 1 | 15.45M | 3.760 | 3.715 | 4.130 | 3.713 | 3.673 |
| [UD2-gate-norm-q32-rope](../variants/ud2-gate-norm-q32-rope.md) | 3 | 15.46M | 3.722（3.716–3.727） | 3.678 | 4.108 | 3.675 | 3.632 |
| [UD2-gate-norm-q128-rope](../variants/ud2-gate-norm-q128-rope.md) | 1 | 15.45M | 3.732 | 3.688 | 4.122 | 3.684 | 3.651 |
| [UD2-gate-norm-h4q8](../variants/ud2-gate-norm-h4q8.md) | 1 | 15.46M | 3.732 | 3.688 | 4.124 | 3.683 | 3.654 |
| [UD2-gate-mass-q32](../variants/ud2-gate-mass-q32.md) | 1 | 15.45M | 3.749 | 3.705 | 4.114 | 3.702 | 3.669 |
| [U-Delta-rawFFN+λlr1000](../variants/u-delta-rawffn-lambda-lr1000.md) | 3 | 15.45M | 3.796（3.793–3.799） | 3.753 | 4.154 | 3.751 | 3.713 |
| [Transformer](../variants/transformer.md) | 3 | 15.19M | 3.945（3.928–3.957） | 5.016 | 4.237 | 3.918 | 6.088 |

### 讀出大小和門（每層平均）

| 層 | UD2 讀出大小 | UD2-scale 學到的縮放 | UD2-gate 的門（各類別、位置的範圍） | UD2-gate 讀出大小：0–64 → 256–1,024 | UD2-gate-norm 的門 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.40 | 1.01 | 1.13（1.06–1.25） | 0.37 → 0.38 | 1.13 |
| 1 | 0.62 | 0.88 | 0.56（0.54–0.59） | 0.35 → 0.93 | 0.84 |
| 2 | 0.38 | 0.95 | 0.34（0.31–0.38） | 0.85 → 1.18 | 0.46 |
| 3 | 0.22 | 1.00 | 0.34（0.32–0.41） | 0.80 → 0.83 | 0.76 |
| 4 | 1.43 | 1.04 | 0.70（0.66–0.76） | 0.49 → 0.59 | 0.39 |
| 5 | 0.49 | 0.97 | 0.81（0.62–0.89） | 1.49 → 1.54 | 0.96 |
| 6 | 0.96 | 0.94 | 0.51（0.38–0.54） | 1.13 → 1.16 | 0.71 |
| 7 | 0.25 | 0.94 | 0.29（0.26–0.45） | 1.43 → 1.49 | 0.52 |
| 8 | 0.45 | 0.98 | 0.29（0.25–0.33） | 0.48 → 0.96 | 1.00 |
| 9 | 0.88 | 0.99 | 0.49（0.36–0.64） | 1.86 → 2.03 | 0.71 |
| 10 | 0.63 | 0.93 | 0.78（0.57–0.89） | 0.66 → 1.00 | 0.94 |
| 11 | 0.25 | 0.97 | 0.15（0.12–0.21） | 1.77 → 2.37 | 0.56 |

## 結論

- **讀出門的進步是真的**：UD2-gate 三個 seed 3.766, 3.767, 3.761，平均 3.765，都比 UD2（3.778，1 個 seed）好 0.011–0.017。
- **不是每層一個縮放就夠**：UD2-scale 3.776，跟 UD2 一樣；學到的縮放都在 0.88–1.04，沒離開 1。UD2-gate 的門每層平均 0.15–1.13，同一層裡不同類別、位置幾乎一樣（第 1 層 0.54–0.59），但它的讀出本身比較大（每層平均 0.38–2.31，UD2-scale 0.25–1.30）。是門跟著輸入變的那一點點有用，還是門讓模型學出更大的讀出再縮小，這次分不開。
- **讀出 RMSNorm 的進步越後面越大**：UD2-gate-norm 3.739，前 64 個 token +0.003，512–1,024 -0.030（對 UD2-gate）。UD2-gate 記得久的層讀出隨位置變大（第 1 層 0.35 → 0.93），門沒有跟著調。
- **讀位址修正**：UD2-gate-q32 3.746，格子少 7% 還好 0.021；補最多的是位置 64–256 的新 token（見[逐 token 差距](token-gap.md)）。修正加到秩 128（UD2-gate-q128，hidden_dim 538）反而是 3.760，比秩 32 差 0.015：秩加大要拿格子換，格子少的代價看起來大過修正多出的容量，但兩者綁在一起分不開。
- **兩個一起大約疊加八成**：UD2-gate-norm-q32 3.727，比 UD2-gate 好 0.039（各自 0.028、0.021），贏 Transformer（三個 seed 平均）0.218；前 64 個 token 打平（+0.001），進步都在 64 以後。讀到 20K 的 16K–20K 3.645。
- **記憶加上旋轉相對位置又好 0.011**：UD2-gate-norm-q32-rope 3.716，不加參數，是目前 15M 最好的統一 block，贏 Transformer 0.229。每一段都好（對 UD2-gate-norm-q32，前 64 個 token -0.016，512–1,024 -0.010）；讀到 20K 的 16K–20K 3.632，是 15M 模型裡這段最好的；進步在新 token（見[逐 token 差距](token-gap.md)）。42M 是 3.170，贏 Transformer 42M 0.044（各 1 個 seed，見[規模](scale.md)）。
- **其他三個沒有用**：旋轉加秩 128 的 UD2-gate-norm-q128-rope 3.732；讀出拆成 4 個 head 的 UD2-gate-norm-h4q8 3.732（參數一樣，記憶那一步貴約 4 倍）；讀出除以記憶量的 UD2-gate-mass-q32 3.749（UD2-gate-q32 3.746）。
- **都沒有擋掉舊內容的干擾**：截掉遠處前文，新 token 還是變好（見[遠處的前文值多少](reach.md)）。

## 注意

- 除了 UD2-gate，都只有 1 個 seed（UD2-gate 的 seed 標準差 0.003）。
- UD2-gate-q32、UD2-gate-norm-q32 的 hidden_dim 降到 662 來對齊參數，記憶格子少 7%；UD2-gate-q128、UD2-gate-norm-q128-rope 降到 538，少 24%；UD2-gate-norm-q32-rope、UD2-gate-norm-h4q8、UD2-gate-mass-q32 是 662。

## 涉及的變體

[UD2](../variants/ud2.md)、[UD2-scale](../variants/ud2-scale.md)、[UD2-gate](../variants/ud2-gate.md)、[UD2-gate-norm](../variants/ud2-gate-norm.md)、[UD2-gate-q32](../variants/ud2-gate-q32.md)、[UD2-gate-norm-q32](../variants/ud2-gate-norm-q32.md)、[UD2-gate-q128](../variants/ud2-gate-q128.md)、[UD2-gate-norm-q32-rope](../variants/ud2-gate-norm-q32-rope.md)、[UD2-gate-norm-q128-rope](../variants/ud2-gate-norm-q128-rope.md)、[UD2-gate-norm-h4q8](../variants/ud2-gate-norm-h4q8.md)、[UD2-gate-mass-q32](../variants/ud2-gate-mass-q32.md)
