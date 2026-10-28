# 模型规模与训练设置

四档规模的宽度和深度对齐 llama2.c 的 260K / 15M / 42M / 110M：一个 llama2.c 层（attention + FFN）对应两个 UnifiedBlock，每个 block 的格子数 `hidden_dim` 取为与 llama2.c 同档参数量相等。`n_layers` 是 UnifiedBlock 的个数。`conv_size=4`、`prior=4.0` 在各档相同，命令里不再重复。

所有命令都在仓库根目录、用 `.venv` 的 Python 运行（`uv sync` 之后 `source .venv/bin/activate`，或把 `python` 换成 `uv run python`）。训练输出写入 `outputs/<size>/`：`best.pt`（验证集最好）、`ckpt.pt`（最近一次评估，可续训）、`model.bin`（`best.pt` 导出的 run.c 权重）。

| | 260K | 15M | 42M | 110M |
| --- | ---: | ---: | ---: | ---: |
| dim | 64 | 288 | 512 | 768 |
| n_layers（block 数） | 10 | 12 | 16 | 24 |
| hidden_dim（每 block 格子数） | 140 | 712 | 1280 | 1912 |
| 参数量 | 259,012 | 15,167,352 | 41,632,288 | 109,400,496 |
| 词表 | 512（custom） | 32000（Llama 2） | 32000 | 32000 |
| max_seq_len | 512 | 256 | 1024 | 1024 |
| batch_size × grad_accum | 128 × 1 | 128 × 4 | 待调参 | 待调参 |
| tokens / iter | 65,536 | 131,072 | 待调参 | 待调参 |
| learning_rate | 1e-3 | 5e-4 | 待调参 | 待调参 |
| warmup_iters | 1000 | 1000 | 待调参 | 待调参 |
| weight_decay | 0.01 | 0.1 | 待调参 | 待调参 |
| betas | (0.9, 0.99) | (0.9, 0.95) | 待调参 | 待调参 |
| grad_clip | 1.0 | 1.0 | 1.0 | 1.0 |
| dropout | 0.05 | 0.0 | 0.0 | 0.0 |
| max_iters | 100,000 | 100,000 | 待调参 | 待调参 |
| 训练 tokens | 6.55B | 13.1B | 待调参 | 待调参 |
| 状态 | 已训练，val 1.2905 | 默认配置，冒烟测试通过 | 未训练 | 未训练 |

学习率计划在各档相同：线性 warmup，再 cosine 衰减到 0；AdamW（fused），dim ≥ 2 的参数做 weight decay；bf16 autocast，`torch.compile`。

## 单卡吞吐与可用 batch

bf16 + `torch.compile`，一次完整的训练步（前向、反向、AdamW），按 2 的幂测到放不下为止：

| 规模 | GPU | 最大 micro-batch | tokens/s | 峰值显存 |
| --- | --- | ---: | ---: | ---: |
| 15M，T=256 | RTX 3090 Ti 24G | 128 | 216K | 12.6 GiB |
| 15M，T=1024 | RTX 3090 Ti 24G | 32 | 212K | 12.6 GiB |
| 42M，T=1024 | RTX 3090 Ti 24G | 16 | 87K | 11.3 GiB |
| 110M，T=1024 | RTX 3090 Ti 24G | 8 | 30K | 11.7 GiB |
| 15M，T=256 | RTX 3060 12G | 64 | 76K | 6.4 GiB |

micro-batch 再翻倍就会超出显存；需要更大的有效 batch 时增大 `--gradient_accumulation_steps`。在 3060 上跑 15M 默认配置用 `--batch_size=64 --gradient_accumulation_steps=8`（每步 tokens 不变）。

## 数据集

`train.py` 顶部的 `from tinystories import Task` 决定训练数据。换数据集就手动改这一行；三个数据模块的 `Task` 接口相同，其余不用改：

| 模块 | 数据 | 准备命令 | 验证集 |
| --- | --- | --- | --- |
| `tinystories.py` | TinyStories（默认） | `python tinystories.py download && python tinystories.py pretokenize` | 第一个 shard |
| `fineweb.py` | FineWeb-Edu sample-10BT，11.4B tokens（120 个 shard，22 GB） | `python fineweb.py download && python fineweb.py pretokenize` | 最后一个 shard |
| `climbmix.py` | ClimbMix-400B（同 gdn.c） | `python climbmix.py download -n 170 && python climbmix.py pretokenize` | 最后一个 shard |

FineWeb-Edu 和 ClimbMix 只用 Llama 2 32K 词表（`vocab_source=llama2`）。

## 260K

TinyStories，512 词表；与 llama2.c 的 `doc/stories260K.md` 相同的预算。单张 RTX 3090 Ti 约 70 分钟。

```bash
python train.py --out_dir=outputs/260K \
  --vocab_source=custom --vocab_size=512 \
  --dim=64 --n_layers=10 --hidden_dim=140 --max_seq_len=512 \
  --batch_size=128 --gradient_accumulation_steps=1 \
  --learning_rate=1e-3 --warmup_iters=1000 --max_iters=100000 \
  --weight_decay=0.01 --beta2=0.99 --dropout=0.05 \
  --eval_interval=2000 --eval_iters=100
```

## 15M

`train.py` 的默认值，与 llama2.c 15M 的训练配方相同（TinyStories，Llama 2 32K 词表）。单张 RTX 3090 Ti 约 216K tokens/s，100K 步约 17 小时。

```bash
python train.py
```

## 42M

宽度和深度已定，batch、学习率等训练超参待调参；下面的优化器参数暂时沿用 15M，micro-batch 用上表测得的最大值。

```bash
python train.py --out_dir=outputs/42M \
  --dim=512 --n_layers=16 --hidden_dim=1280 --max_seq_len=1024 \
  --batch_size=16 --gradient_accumulation_steps=32 \
  --learning_rate=5e-4 --warmup_iters=1000 --max_iters=100000
```

## 110M

同 42M：宽度和深度已定，训练超参待调参。

```bash
python train.py --out_dir=outputs/110M \
  --dim=768 --n_layers=24 --hidden_dim=1912 --max_seq_len=1024 \
  --batch_size=8 --gradient_accumulation_steps=64 \
  --learning_rate=5e-4 --warmup_iters=1000 --max_iters=100000
```
