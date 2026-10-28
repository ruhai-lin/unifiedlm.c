# 训练 kernel

`model.py` 里每个 `UnifiedBlock` 的快记忆由 `kernel.py` 的 `unified_memory(r, v, g, w)` 计算（Triton，前向 + 反向）。本文记录它的设计，以及与纯 PyTorch 实现和 [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)（fla）的对比。

## 要算的东西

对每个 batch 行、每个位置 `t`（`lam = exp(g)`）：

```text
k_t    = r_{t-1}                                  写地址（k_0 = 0）
c_t    = w + sum_{s<=t} lam^(t-s) k_s             每个格子的证据质量
q_t    = r_t / c_t
fast_t = sum_{s<=t} lam^(t-s) (q_t . k_s) v_s
```

`fast` 是一个因果线性注意力：key 维 = 格子数 `hidden_dim`（140–1912），value 维 = `dim`（64–768），只有一个标量衰减。块的输出是 `x + mu(q * w) + fast`，其中 `mu(q * w)` 是普通 matmul，留在 PyTorch。

## 设计

- **一个通用 kernel `_la`**：`o_t = sum_{s<=t} lam^(t-s) (a_t . b_s) c_s`，按长度 64 分块，分三步——块状态 `h_n`（`BK x BV` 分片，按块顺序推进）、块内矩阵 `A_n = (a b^T) * lam^(i-j)`、输出 `lam^(i+1) a h_n + A_n c`。前向和反向需要的四个乘积都是它，只交换角色、或把时间反过来：

  | | a | b | c | 方向 |
  | --- | --- | --- | --- | --- |
  | fast | q | k | v | 正向 |
  | dq | dfast | v | k | 正向 |
  | dk | v | dfast | q | 反向 |
  | dv | k | q | dfast | 反向 |

- **分母**：`c` 和 `q = r / c` 在一个 kernel 里算，块内用 `tl.associative_scan` 做衰减累加（`(a1, x1), (a2, x2) -> (a1 a2, x1 a2 + x2)`），块间传递状态；`k = shift(r)` 在 kernel 里按位置偏移读取，不单独存。
- **反向的分母部分**一个 kernel 完成：`dr = dq / c`、`dc = -dq q / c`、`dc` 的反向衰减累加、`dk` 合并，以及 `dw`、`dg` 的原子累加。`dg` 用 `d/dg sum_{s<=t} lam^(t-s) (...) = sum_t t (...)_t - sum_s s (...)_s` 从已有的 `dq`、`dk` 直接得到。
- **精度**：matmul 操作数 bf16、fp32 累加；`c`、`q` 及它们的梯度全程 fp32。分块大小由 `triton.autotune` 选。
- **与参考实现的误差**（对 float64 参考，`test_all.py::test_kernel` 与开发时的算子级对拍）：模型 loss 与全部参数梯度的最大相对误差 fp32 下约 3e-3、bf16 autocast 下约 8e-3；算子级 `q` 2e-7、`fast` 2.3e-3、`dr` 5e-4、`dv` 2.3e-3、`dw` 5e-4、`dlam` 1e-3–1.7e-2。260K checkpoint 用 kernel 复评的 FP32 验证 NLL 为 1.2904748，纯 PyTorch 参考为 1.2904758。

## 与 fla 的对比

fla 的 `chunk_simple_gla` 正好是"标量衰减的线性注意力"，可以直接承担 `fast`，但要做三处适配：

1. fla 每个 head 的 key 维有上限（约 256），所以把格子分组当作 head（格子数补零到组大小的倍数），`v` 复制到每一组，输出再按组求和；
2. 分母 `c` 要再调用一次（value = `k`、key 维为 1 的线性注意力）；
3. 常数衰减参数 `g_gamma` 没有反向梯度，衰减要按位置展开成 `g` 传入，`lam` 才学得动。

### 端到端训练吞吐

bf16 autocast + `torch.compile`，完整训练步（前向、反向、AdamW），单张 RTX 3090 Ti；每档按 2 的幂增加 micro-batch，表中是放得下的最大值。纯 PyTorch 为同一算子的分块实现（块内二次、块间传递状态，`torch.compile`）；fla 的组大小在 64 / 128 / 256 中 128 最好（15M、B32 时分别为 124K / 136K / 141K tokens/s）。

| 规模 | 纯 PyTorch | fla | Triton（`kernel.py`） |
| --- | --- | --- | --- |
| 15M，T=256 | 157.0K tokens/s，B128，16.5 GiB | 145.0K，B128，18.8 GiB | **216.0K**，B128，12.6 GiB |
| 15M，T=1024 | 133.9K，B32，18.2 GiB | 144.7K，B32，18.8 GiB | **212.4K**，B32，12.6 GiB |
| 42M，T=1024 | 50.6K，B16，21.7 GiB | 55.6K，B16，16.8 GiB | **87.0K**，B16，11.3 GiB |
| 110M，T=1024 | 14.3K，B4，14.9 GiB（B8 放不下） | 20.2K，B8，18.0 GiB | **30.3K**，B8，11.7 GiB |

RTX 3060 12G，15M，T=256：Triton 最大 B64，75.9K tokens/s，6.4 GiB；纯 PyTorch 在 B32 时 54.0K。

### 只看快记忆这一项

单个 block 的 `unified_memory` 前向 + 反向（bf16，3090 Ti）：

| | 15M 尺寸，B32，T=256 | 42M 尺寸，B8，T=1024 |
| --- | ---: | ---: |
| 纯 PyTorch 分块（`torch.compile`） | 4.17 ms | 12.76 ms |
| Triton | 1.56 ms | 3.18 ms |

fla 在单个 block 的 eager 对比里比纯 PyTorch 快 1.9×（15M）/ 3.0×（42M），但放进 `torch.compile` 后的整模型几乎没有收益：`v` 的复制、额外的分母调用和组求和吃掉了 kernel 本身的优势，而且 compile 无法跨 fla 的调用融合前后的逐元素运算。

### 优化过程中的教训

- 第一版 Triton（fp32 / TF32 matmul、每个 value 分片重复计算块内矩阵 `A`、分母在 PyTorch 里算）比纯 PyTorch 还慢；改成 bf16 matmul、`A` 单独算一次之后，单次 `_la` 从 0.63 ms 降到 0.24 ms。
- 分母最初用 fp32 精确（`ieee`）点积做块内累加，单个 kernel 就要 1.6 ms（Ampere 上 fp32 精确点积是软件模拟的）；换成 `tl.associative_scan` 后降到 0.12 ms，精度不变。
- 现在快记忆已不是瓶颈：15M、B32 时整步约 42 ms，其中去掉快记忆后约 22 ms（key 矩阵乘、relu² 与归一化、卷积、输出头与 cross-entropy）。
