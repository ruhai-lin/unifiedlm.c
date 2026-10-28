# unifiedlm.c

UnifiedLM用同一种块同时做 token mixing 和 channel mixing 的语言模型。每个块是一个稀疏寻址的关联记忆，格子里存的是"先验 + 上下文证据"的后验均值：先验很强的格子就是一个静态 FFN 槽，先验很弱的格子就是一个上下文寄存器。没有 attention、没有 RoPE、没有 KV cache，推理时每个块的状态大小与上下文长度无关。训练栈沿用 [llama2.c](https://github.com/karpathy/llama2.c)，并配有纯 C 推理 `run.c`以简化硬件部署。

研究目标见 [`docs/research_goal.md`](docs/research_goal.md)，各规模的设置与训练命令见 [`docs/models.md`](docs/models.md)。

## 模型

每个 `UnifiedBlock` 对 RMSNorm 后的隐状态 `h_t` 做宽度 4 的因果深度卷积得到 `u_t`，用 relu² 产生对格子的寻址并归一化（分母里有一个可学习的空槽质量 `eps`）：

```text
r_t = relu(K u_t + b)^2 / (eps + sum_a relu(K u_t + b)_a^2)
```

当前位置的值 `v_t = W_v h_t` 写到前一位置的地址 `r_{t-1}` 上，然后用当前地址读出：

```text
S_t = lam * S_{t-1} + v_t r_{t-1}^T          n_t = lam * n_{t-1} + r_{t-1}
m_t = (w * mu + S_t) / (w + n_t)              # 每个格子的后验均值（逐格）
y_t = m_t r_t,   x_t <- x_t + y_t
```

`mu`（慢内容）、每格先验质量 `w`、遗忘率 `lam` 都是学出来的。

快记忆这一项是一个以格子为 key 维、以 `dim` 为 value 维、带一个标量衰减的因果线性注意力。训练时它由 `kernel.py` 里的 Triton kernel 计算（分块：块内二次、块间传递状态，块长 64；前向和反向都在 kernel 里，matmul 用 bf16、统计量用 fp32），在 15M–110M 各档比 `torch.compile` 后的纯 PyTorch 实现快 1.4–2.1 倍、显存更省；设计和与 fla 的对比见 [`docs/kernel.md`](docs/kernel.md)。推理用逐 token 的递推形式（`forward_recurrent`，与 `run.c` 完全相同）。训练和 `sample.py` 需要 CUDA；`run.c` 在 CPU 上运行。

## 环境

本项目使用 [uv](https://docs.astral.sh/uv/)（Python 3.12，PyTorch 2.10.0 + CUDA 12.8，版本锁在 `uv.lock`）：

```bash
uv sync
source .venv/bin/activate
```

## 数据

TinyStories，与 llama2.c 相同的流程：每篇故事加 BOS、不加 EOS，存成 uint16 的 `.bin`；按文件名排序后第一个 shard 是验证集。

```bash
python tinystories.py download        # data/TinyStories_all_data/*.json
python tinystories.py pretokenize     # Llama 2 32K 词表（15M 及以上）

# 512 词表（260K）：用仓库里的 tok512.model（llama2.c 官方 260K 模型的词表）
cp tok512.model data/
python tinystories.py pretokenize --vocab_size=512
```

更大的语料用 `fineweb.py`（FineWeb-Edu sample-10BT）或 `climbmix.py`（ClimbMix），接口相同，最后一个 shard 是验证集；换数据集就改 `train.py` 顶部 `from tinystories import Task` 这一行，见 [`docs/models.md`](docs/models.md#数据集)。

```bash
python fineweb.py download            # HuggingFace 缓存，约 27 GB parquet
python fineweb.py pretokenize         # data/FineWebEdu_10BT/*.bin，Llama 2 32K 词表
```

## 训练

```bash
python train.py                       # 默认 15M，单卡
torchrun --standalone --nproc_per_node=4 train.py
```

其他规模（260K / 42M / 110M）的完整命令在 [`docs/models.md`](docs/models.md)。所有超参都可以用 `--key=value` 覆盖。

## 采样与 C 推理

```bash
python sample.py --checkpoint=outputs/15M/best.pt

make run
python export.py --checkpoint outputs/15M/best.pt --output outputs/15M/model.bin
./run outputs/15M/model.bin -t 0.8 -p 0.9 -n 256 -i "Once upon a time"
./run outputs/260K/model.bin -z tok512.bin -t 0.8 -p 0.9 -i "Once upon a time"   # 512 词表
```

`train.py` 在验证集变好时会自动导出 `model.bin`。`run` 的参数与 llama2.c 相同：`-z` tokenizer，`-t` 温度，`-p` top-p，`-s` 随机种子，`-n` 步数，`-i` 提示词。

## 测试

```bash
make test        # 编译 ./run，再运行 python test_all.py
```

`test_all.py` 检查：纯 PyTorch 参考实现的分块形式与二次形式一致、并行形式与逐 token 递推一致、因果性、梯度有限（double 精度，误差 < 1e-8）；Triton kernel 的 loss 与全部参数梯度对 double 参考的相对误差 < 1e-2（需要 CUDA）；以及 `./run` 的贪心生成与 Python 参考逐字相同。

## 评测

```bash
python eval/core_eval.py --checkpoint outputs/15M/best.pt
```

[CORE](https://arxiv.org/abs/2406.11717)（DCLM 论文指标），移植自 gdn.c；首次运行会下载 `data/eval_bundle/`。

## 结果

**260K，TinyStories，512 词表**（100K 次更新，6.55B tokens，单张 RTX 3090 Ti，约 70 分钟）。验证 NLL 在同一组 1,048,576 个验证 token 上用 FP32 计算：

| 模型 | 参数量 | 验证 NLL |
| --- | ---: | ---: |
| UnifiedLM | 259,012 | **1.2905** |
| llama2.c 官方 `stories260K.pt` | 260,032 | 1.2965 |

`run.c` 生成的样本（`-t 0.8 -p 0.9 -s 2`）：

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big box in the sunshine. She asked her mommy, "What's wrong, Lily?" Her mommy said, "I lost my toy ice cream!"
> Lily was sad and wanted to help her mommy. She asked her mommy if they could borrow it. They both smiled and said, "Thank you for my favorite toy is an easy secret." Lily was happy to help her mommy, and they all had a good friend.

15M 及以上尚未完整训练。

## 试过的其他做法

和现在的标准配方（快写入 + relu² 地址 + 先验 4）做过对照的：

- **去掉快写入**（每个块只剩对静态格子 `mu` 的归一化 relu² 读出，没有上下文写入和后验分母）：260K、327.68M tokens 的短预算下，等参数量另训的模型验证 NLL 为 1.664，标准配方为 1.511。在训练好的 260K 模型上推理时直接去掉快写入，验证 NLL 从 1.31 升到 1.78。
- **更弱的先验（prior 0.4）**：15M、24.6M tokens 的短预算下验证 NLL 为 2.057，prior 4 为 1.962。

另外几种变体（固定写入率、hash n-gram 格子、二值 spike / k-WTA 地址、有限 fan-in、共享权重循环）已经实现并通过了一致性检查，但还没有在文本上与标准配方比较。

## 目录

```text
model.py        UnifiedLM（唯一的模型）
kernel.py       快记忆的 Triton 训练 kernel（前向 + 反向）
train.py        训练（默认 15M）
sample.py       PyTorch 采样
export.py       导出 run.c 权重
run.c           纯 C 推理
tinystories.py  TinyStories 下载、分词、加载
fineweb.py      FineWeb-Edu sample-10BT（同接口）
climbmix.py     ClimbMix（同接口，来自 gdn.c）
tokenizer.py    sentencepiece 封装与 .bin 导出
test_all.py     一致性测试
eval/           评测（CORE）
docs/           研究目标、各规模设置、训练 kernel
```

`references/`（llama2.c、gdn.c 等上游与 baseline 仓库）、`scripts/`（开发脚本）、`data/`、`outputs/` 都在 `.gitignore` 里。baseline 在 `references/` 下用各自的代码训练和评测，例如：

```bash
git clone https://github.com/karpathy/llama2.c references/llama2.c
```
