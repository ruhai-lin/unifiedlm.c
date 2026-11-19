# GDN（代號 `GDN`）

線性遞歸（SSM 類）基線：Transformer 的 attention 換成 Gated DeltaNet（6 個 head × 48）。

- **類別**：基線
- **基於**：[Transformer](transformer.md)
- **程式碼**：`scripts/variants.py` 的 `GatedDeltaHalf`，訓練時 `--variant=GDN`

## 怎麼做的

- 每個 head：`S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ`，`o_t = S_tᵀ q_t`；`α_t = exp(−exp(A) softplus(W_a h + dt))`，`β_t = sigmoid(W_b h)`。
- q、k、v 先過寬 4 的因果卷積和 SiLU，q、k 單位化；輸出過每 head 的 RMSNorm、乘 SiLU(W_g h)、再投影，跟論文和 fla 的 GatedDeltaNet 一樣。
- 跟 FFN 半層交替，跟 Transformer 一樣的層數計法。

```text
S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T,   o_t = S_t^T q_t
```

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fw15m_gdn` | 1 | 15.23M | 3.964 | 3.948 | 4.166 | 3.946 |
| 42M | `fw42m_gdn` | 1 | 41.69M | 3.229 | 3.196 | 3.578 | 3.192 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.922。

## 結論

15M 3.964，輸 U-Delta（3.898），但開頭最強：前 64 個 token 4.166（U-Delta 4.258，Transformer 4.224）。讀到 20K 不崩（16K–20K 3.922）。U-Delta 贏它的原因見 [U-Delta 為什麼贏 GDN](../experiments/vs-gdn.md)。

## 相關實驗

- [U-Delta 為什麼贏 GDN](../experiments/vs-gdn.md)
- [長上下文與外推（讀到 20K）](../experiments/extrapolation.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwgdn.txt）
python scripts/train_variant.py --out_dir=outputs/fw15m_gdn --variant=GDN --hidden_dim=672 --head_dim=48 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fw15m_gdn
```
