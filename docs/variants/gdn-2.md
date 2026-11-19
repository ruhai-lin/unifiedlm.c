# GDN-2（代號 `GDN2`）

NVIDIA Gated DeltaNet-2：抹除門和寫入門分開，衰減按通道。

- **類別**：基線
- **基於**：[GDN](gdn.md)
- **程式碼**：`scripts/variants.py` 的 `GatedDeltaNet2Half`、`erase_write_memory`，訓練時 `--variant=GDN2`

## 怎麼做的

- GDN：一個純量 β 同時決定抹掉多少、寫進多少；衰減 α 是每個 head 一個純量。
- GDN-2：抹除 gate b（鑰匙的每一維）、寫入 gate w（值的每一維）分開；衰減 α 是每個鑰匙維度一個（沿用 KDA）。
- 跟論文不同：三個 gate 用低秩投影（秩 48，照 fla 的 KDA）；FFN 寬度 672→576 讓參數跟 GDN 對齊。

**問題**：論文（Hatamizadeh、Choi、Kautz，NeurIPS 2026）在 1.3B、100B token 贏 GDN 和 KDA，長文檢索進步最多。在我們的規模是否也成立，它會不會是最強的基準。

```text
S_t = (I − k_t (b_t ⊙ k_t)ᵀ) diag(α_t) S_{t−1} + k_t (w_t ⊙ v_t)ᵀ
b_t = σ(W_b h), w_t = σ(W_w h), log α_t = −exp(a) · softplus(W_f h + δ)
```

**設定**：參數：15.21M（GDN 15.23M）；狀態：83K（同 GDN）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwgdn2_15m` | 1 | 15.21M | 3.970 | 3.954 | 4.171 | 3.952 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.926。

## 結論

在 15M 沒有進步：3.970 對 GDN 3.964，在雜訊內，各位置幾乎重疊。規模（15M、3 億 token 對 1.3B、100B）、低秩 gate、FFN 變窄都可能是原因，所以不代表論文錯，只說明在這個規模把抹除和寫入拆開沒有幫助。對 U-Delta 的意義：近距離的弱點不是因為 β 只有一個純量。

## 相關實驗

- [U-Delta 為什麼贏 GDN](../experiments/vs-gdn.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwnext2.txt）
python scripts/train_variant.py --out_dir=outputs/fwgdn2_15m --variant=GDN2 --hidden_dim=576 --head_dim=48 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwgdn2_15m
```
