# 長上下文與外推（讀到 20K）

**問題**：過了訓練長度會不會崩？

**簡短回答**：統一 block 都不崩；Transformer 只有開滑動窗口才不崩。

## 怎麼做的

- 訓練長度 1,024，評估讀到 20,480（20 倍）：64 個長 20,480 的驗證窗口，按位置分段（`scripts/fineweb/long_eval.py`、`scripts/fineweb/sgvt.sh`）。
- Transformer 超過訓練長度時試三種：完整上下文（RoPE 直接外推）、推理時滑動窗口 1,024、推理時 log-n 縮放（query 乘 log n / log L，Qwen 的做法，不重訓）。另訓練一個 Transformer+SSMax（Scalable-Softmax）。
- 長窗口訓練：Transformer 用 4,096、16,384 訓練，U-Gate、U-Gate-fastinit 用 4,096 訓練，每步 token 數相同。

## 結果

### 15M 讀到 20K（NLL，按位置）

| 模型 | 0–64 | 512–1024 | 1024–2048 | 2048–4096 | 4096–8192 | 8192–16384 | 16384–20480 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer（完整上下文） | 4.106 | 3.842 | 4.784 | 5.603 | 5.875 | 6.074 | 6.088 |
| Transformer（滑動窗口 1,024） | 4.106 | 3.842 | 3.886 | 3.903 | 3.923 | 3.911 | 3.874 |
| Transformer（log-n 縮放） | 4.106 | 3.842 | 4.774 | 5.608 | 5.942 | 6.194 | 6.236 |
| Transformer+SSMax | 4.100 | 3.824 | 4.948 | 5.769 | 6.159 | 6.453 | 6.515 |
| GDN | 4.052 | 3.872 | 3.927 | 3.948 | 3.965 | 3.960 | 3.922 |
| U-Gate | 4.188 | 3.924 | 3.946 | 3.982 | 3.985 | 3.954 | 3.931 |
| U-Delta | 4.114 | 3.812 | 3.836 | 3.876 | 3.875 | 3.841 | 3.817 |
| U-Delta+slow | 4.125 | 3.796 | 3.826 | 3.876 | 3.889 | 3.870 | 3.856 |
| U-Delta-λlr1000 | 4.085 | 3.753 | 3.776 | 3.814 | 3.813 | 3.776 | 3.752 |
| U-Delta-rawFFN | 4.024 | 3.725 | 3.748 | 3.796 | 3.803 | 3.780 | 3.760 |
| U-Delta-rawFFN+λlr1000 | 4.007 | 3.699 | 3.725 | 3.766 | 3.768 | 3.736 | 3.713 |
| UD2 | 3.987 | 3.684 | 3.712 | 3.752 | 3.753 | 3.719 | 3.695 |
| UD2-scale | 3.979 | 3.678 | 3.706 | 3.749 | 3.752 | 3.718 | 3.695 |
| UD2-gate | 3.988 | 3.665 | 3.702 | 3.739 | 3.741 | 3.711 | 3.685 |
| UD2-gate-norm | 3.979 | 3.644 | 3.670 | 3.711 | 3.717 | 3.683 | 3.661 |
| UD2-gate-q32 | 3.976 | 3.651 | 3.672 | 3.715 | 3.719 | 3.684 | 3.663 |
| UD2-gate-norm-q32 | 3.984 | 3.630 | 3.656 | 3.700 | 3.703 | 3.670 | 3.645 |
| UD2-gate-q128 | 3.994 | 3.671 | 3.689 | 3.728 | 3.734 | 3.698 | 3.673 |
| UD2-gate-norm-q32-rope | 3.954 | 3.619 | 3.645 | 3.687 | 3.689 | 3.655 | 3.632 |
| UD2-gate-norm-q128-rope | 3.977 | 3.638 | 3.658 | 3.704 | 3.708 | 3.673 | 3.651 |
| UD2-gate-mass-q32 | 3.964 | 3.657 | 3.682 | 3.722 | 3.723 | 3.690 | 3.669 |
| UD2-gate-norm-h4q8 | 3.967 | 3.632 | 3.661 | 3.703 | 3.709 | 3.675 | 3.654 |
| U-Delta-rawFFN+λlr1000+carry | 4.126 | 3.711 | 3.731 | 3.774 | 3.769 | 3.735 | 3.710 |
| U-Delta+attn（含 attention） | 4.061 | 3.747 | 3.778 | 3.814 | 3.819 | 3.785 | 3.762 |

### 42M 讀到 20K

| 模型 | 0–64 | 512–1024 | 1024–2048 | 2048–4096 | 4096–8192 | 8192–16384 | 16384–20480 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer（完整上下文） | 3.533 | 3.120 | 4.703 | 5.810 | 5.991 | 6.221 | 6.251 |
| Transformer（滑動窗口 1,024） | 3.533 | 3.120 | 3.148 | 3.198 | 3.196 | 3.161 | 3.142 |
| UD2-gate-norm-q32-rope | 3.479 | 3.074 | 3.101 | 3.145 | 3.139 | 3.103 | 3.087 |
| U-Delta | 3.663 | 3.262 | 3.280 | 3.325 | 3.322 | 3.286 | 3.269 |
| U-Delta-rawFFN | 3.521 | 3.139 | 3.158 | 3.208 | 3.206 | 3.176 | 3.166 |
| U-Delta-rawFFN+λlr1000 | 3.512 | 3.129 | 3.154 | 3.198 | 3.192 | 3.156 | 3.142 |

### 用更長的窗口訓練（15M，在自己的訓練長度評估）

| 模型 | 訓練長度 | NLL | 512–1,024 段 |
| --- | ---: | ---: | ---: |
| Transformer | 1,024 | 3.928 | 3.899 |
| Transformer | 4,096 | 3.983 | 3.979 |
| Transformer | 16,384 | 4.157 | 4.169 |
| U-Gate | 1,024 | 4.003 | 3.964 |
| U-Gate | 4,096 | 3.999 | 3.998 |
| U-Gate-fastinit | 1,024 | 4.052 | 4.024 |
| U-Gate-fastinit | 4,096 | 4.020 | 4.017 |

## 結論

- **統一 block 過了訓練長度都不崩**：16K–20K 段 15M 在 3.7–4.0，42M 3.142–3.269。完整上下文的 Transformer 升到 6.1（15M）、6.3（42M）。
- **Transformer 只有推理時開滑動窗口才不崩**；log-n 縮放和 SSMax 都救不了（SSMax 過了 1,024 照樣崩）。
- **15M 統一 block 贏滑動窗口的 Transformer**：U-Delta-rawFFN+λlr1000 3.713，滑動窗口的 Transformer 三個 seed 3.874–3.903。
- **42M 的 U-Delta-rawFFN+λlr1000 追平滑動窗口的 Transformer**：16K–20K 3.142 對 3.142，1K 之後每段相差不到 0.006；1,024 以內有輸有贏（開頭 64 個 token 贏 0.021，64–256 輸 0.017–0.019）。U-Delta-rawFFN 還輸 0.01–0.03（3.166）。比 U-Delta-rawFFN 好的幅度在 2K 之後越讀越大（2K–4K 0.010，16K–20K 0.024），跟 15M 一樣。1 個 seed。
- **訓練時讓記憶跨窗口接續沒有幫助**：U-Delta-rawFFN+λlr1000+carry 每個訓練窗口從同一串上一個窗口結束的記憶開始（梯度不跨窗口），讀到 20K 只在 4K 之後跟 U-Delta-rawFFN+λlr1000 打平（16K–20K 3.710 對 3.713），開頭 64 個 token 反而差 0.119（評測時每個窗口從空記憶開始，訓練時不是）。1 個 seed。
- **記憶縮短的版本越讀越好，保留慢記憶的越讀越差**：λlr1000 和 rawFFN+λlr1000 在 8K 之後繼續下降；U-Delta+slow、U-Delta-diverseλ 在 4K 之後比 U-Delta 差。
- **直接用長窗口訓練反而變差**：同樣的 token 數，Transformer 用 4K、16K 訓練，在 512–1,024 段比用 1,024 訓練還差；U-Gate 用 4K 訓練差約 0.03，U-Gate-fastinit 差不多（512–1,024 段好 0.007）。

## 注意

- 讀到 20K 只有 64 個窗口，每段的 token 數從 4,096 到 26 萬不等，開頭幾段比較吵。

## 涉及的變體

[Transformer](../variants/transformer.md)、[Transformer+SSMax](../variants/transformer-ssmax.md)、[GDN](../variants/gdn.md)、[U-Gate](../variants/u-gate.md)、[U-Gate-fastinit](../variants/u-gate-fastinit.md)、[U-Delta](../variants/u-delta.md)、[U-Delta+slow](../variants/u-delta-plus-slow.md)、[U-Delta-λlr1000](../variants/u-delta-lambda-lr1000.md)、[U-Delta-rawFFN](../variants/u-delta-rawffn.md)、[U-Delta-rawFFN+λlr1000](../variants/u-delta-rawffn-lambda-lr1000.md)、[UD2](../variants/ud2.md)、[UD2-scale](../variants/ud2-scale.md)、[UD2-gate](../variants/ud2-gate.md)、[UD2-gate-norm](../variants/ud2-gate-norm.md)、[UD2-gate-q32](../variants/ud2-gate-q32.md)、[UD2-gate-norm-q32](../variants/ud2-gate-norm-q32.md)、[UD2-gate-q128](../variants/ud2-gate-q128.md)、[UD2-gate-norm-q32-rope](../variants/ud2-gate-norm-q32-rope.md)、[UD2-gate-norm-q128-rope](../variants/ud2-gate-norm-q128-rope.md)、[UD2-gate-mass-q32](../variants/ud2-gate-mass-q32.md)、[UD2-gate-norm-h4q8](../variants/ud2-gate-norm-h4q8.md)、[U-Delta-rawFFN+λlr1000+carry](../variants/u-delta-rawffn-lambda-lr1000-carry.md)、[U-Delta+attn](../variants/u-delta-plus-attn.md)
