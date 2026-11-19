# GDN-bigstate（代號 `GDNs`）

我們自己的 GDN 變體：2 個大 head（各 144 維），狀態約 3 倍，衰減用平滑上下限。

- **類別**：基線
- **基於**：[GDN](gdn.md)
- **程式碼**：`scripts/variants.py` 的 `GatedDeltaHalf(soft=True)`，`--head_dim=144`，訓練時 `--variant=GDNs`

## 怎麼做的

- head 數 6→2，每個 head 的維度 48→144。每個 head 的狀態是 d_k × d_v，所以總狀態 6×48² → 2×144²，變 3 倍。
- 衰減 log α 的下限從硬性截斷（−2）改成平滑的 2·expm1(log α / 2)，讓靠近下限的 head 還有梯度。
- FFN 寬度 672，參數 15.22M，跟 GDN 對齊。

**問題**：U-Delta 每條串行的狀態約 2.46M 個數，GDN 只有 83K。U-Delta 贏 GDN 可能只是因為能存的東西多 30 倍。

```text
S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ,   o_t = S_tᵀ q_t
每個 head：U-Avg 是 144 × 144（GDN 是 48 × 48）
```

**設定**：參數：15.22M（GDN 15.23M）；狀態：249K（GDN 83K，U-Delta 2.46M）。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fw15m_gdn144` | 1 | 15.22M | 3.964 | 3.948 | 4.160 | 3.947 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.923。

## 結論

狀態放大 3 倍完全沒進步（3.964 對 3.964，各位置差都在 0.01 內）。所以「GDN 缺狀態」的說法變弱了，但還不能排除：U-Delta 的總狀態仍是 GDN-bigstate 的 10 倍。

## 相關實驗

- [U-Delta 為什麼贏 GDN](../experiments/vs-gdn.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwnext.txt）
python scripts/train_variant.py --out_dir=outputs/fw15m_gdn144 --variant=GDNs --hidden_dim=672 --head_dim=144 --dim=288 --n_layers=12 --batch_size=16 --gradient_accumulation_steps=8 --warmup_iters=360 --max_iters=2300 --dataset=fineweb --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --learning_rate=5e-4 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fw15m_gdn144
```
