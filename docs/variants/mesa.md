# Mesa（代號 `MESA`）

Mesa layer：每個 token 精確解一次加權最小平方（GDN 和 U-Delta 的 delta rule 每步只往這個解走一步）。

- **類別**：基線
- **基於**：[GDN](gdn.md)
- **程式碼**：`scripts/mesa.py` 的 `MesaHalf`，訓練時 `--variant=MESA`

## 怎麼做的

- 跟 GDN 一樣是把 Transformer 的注意力換掉，q、k、v、卷積、輸出 gate 都相同。
- 記憶不是一步步更新 U-Avg，而是存 H（kkᵀ 的加權和）和 G（vkᵀ 的加權和），讀取時解 (H + λI) x = q。
- 用共軛梯度法解 24 步（相關鑰匙的難例誤差約 1e-3），梯度用隱式微分，不需要存下每個 token 的 H。

**問題**：blog 說最小平方是「最佳線性記憶」，容量約 2d（一般外積記憶約 d）。如果 Mesa 明顯贏 GDN，代表 delta rule 一步近似浪費了容量，U-Delta 也可能從精確解受益。

```text
H_t = γ_t H_{t−1} + β_t k_t k_tᵀ,   G_t = γ_t G_{t−1} + β_t v_t k_tᵀ
o_t = G_t (H_t + diag(λ))⁻¹ q_t
```

**設定**：參數：15.24M（GDN 15.23M）；狀態：166K（H 和 G 各一份，GDN 83K）；速度：一步約 10 秒，是 GDN 的約 4 倍。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwmesa15m` | 1 | 15.24M | 3.969 | 3.951 | 4.188 | 3.949 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.925。

## 結論

沒有比 GDN 好：Mesa 3.969，GDN 3.964、GDN-2 3.970，都輸 Transformer（3.928）和 U-Delta（3.897）。近處（前 64 個 token）4.188 比 GDN 4.166 還差；訓練時間約 GDN 的 4 倍。在 15M，把 delta rule 的單步近似換成每個 token 精確解最小平方沒有幫助，U-Delta 贏 GDN 的原因不在寫入規則多精確。1 個 seed。

## 相關實驗

- [U-Delta 為什麼贏 GDN](../experiments/vs-gdn.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwnext2.txt）
python scripts/train_variant.py --out_dir=outputs/fwmesa15m --variant=MESA --hidden_dim=672 --head_dim=48 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwmesa15m
```
