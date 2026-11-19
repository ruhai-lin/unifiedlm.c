# Transformer（代號 `T`）

llama2.c 式 Transformer 基線：RoPE attention + SwiGLU FFN，參數量跟統一 block 對齊。

- **類別**：基線
- **基於**：—
- **程式碼**：`scripts/variants.py` 的 `AttentionHalf`、`FeedForwardHalf`，訓練時 `--variant=T`

## 怎麼做的

- `n_layers` 個 block 交替為 attention 半層和 FFN 半層（15M：12 個 = 6 層 attention + 6 層 FFN），跟統一 block 的計數一致。
- attention：RoPE，head 維度 48（15M）/ 64（42M），因果遮罩；FFN：SwiGLU，`hidden_dim` 768（15M）/ 1376（42M），讓參數跟統一 block 一樣。
- 超過訓練長度時報兩種結果：完整上下文（RoPE 直接外推）和推理時滑動窗口 1,024。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fw15m_t`（+2 個 seed） | 3 | 15.19M | 3.945（3.928–3.957） | 5.016 | 4.237 | 3.918 |
| 42M | `fw42m_t`（+2 個 seed） | 3 | 41.69M | 3.214（3.214–3.215） | 4.860 | 3.627 | 3.164 |
| 110M | `fw110m_t` | 1 | 109.53M | 2.822 | 4.614 | 3.263 | 2.768 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：6.049（3 個 seed）；推理時滑動窗口 1,024：3.893。

## 結論

15M 3.945（3 個 seed，3.928–3.957），42M 3.214，110M 2.822。seed 間的差距（標準差 0.016）比統一 block（0.003）大約 5 倍。過了訓練長度就崩：完整上下文 16K–20K 段 5.999–6.088；推理時開滑動窗口 1,024 才穩（3.874–3.903）。15M 的 Transformer 不太會用前文：64–128 段到 512–1,024 段只降 0.064；42M 時降 0.129。所有模型都用同一個學習率 5e-4，Transformer 沒有單獨調。

## 相關實驗

- [規模：15M → 42M → 110M](../experiments/scale.md)
- [長上下文與外推（讀到 20K）](../experiments/extrapolation.md)
- [記憶任務 m2kv](../experiments/memory-tasks.md)
- [U-Delta 為什麼贏 GDN](../experiments/vs-gdn.md)
- [多 seed 的穩定性](../experiments/seeds.md)
- [逐 token 差距：輸在哪些 token](../experiments/token-gap.md)
- [遠處的前文值多少](../experiments/reach.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fw15m.txt）
python scripts/train_variant.py --out_dir=outputs/fw15m_t --variant=T --hidden_dim=768 --head_dim=48 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=32 --gradient_accumulation_steps=4 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fw15m_t
# 42M（scripts/nrp/fw42m.txt）
python scripts/train_variant.py --out_dir=outputs/fw42m_t --variant=T --hidden_dim=1376 --head_dim=64 --dataset=fineweb --dim=512 --n_layers=16 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=32 --gradient_accumulation_steps=4 --learning_rate=5e-4 --warmup_iters=1000 --max_iters=6400 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fw42m_t
# 110M（scripts/nrp/fw110m.txt）
python scripts/train_variant.py --out_dir=outputs/fw110m_t --variant=T --hidden_dim=2048 --head_dim=64 --dataset=fineweb4 --dim=768 --n_layers=24 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=2600 --max_iters=16700 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=1000 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fw110m_t
```
