# U-Delta+attn（代號 `SDA`）

對照組，含 attention，不算統一 block：每 4 個 U-Delta block 後面插一個窗口 128 的局部 attention 半層。

- **類別**：含 attention 的對照（只當參考，不算統一 block）
- **基於**：[U-Delta](u-delta.md)
- **程式碼**：`scripts/variants.py` 的 `WindowAttentionHalf` + `scripts/delta.py` 的 `DeltaBlock`，訓練時 `--variant=SDA`

## 怎麼做的

- `n_layers = 15`：每第 5 個 block 是窗口 128 的 attention 半層（RoPE，每個 query 只看最近 128 個 key），其餘 12 個是 U-Delta block。
- `hidden_dim` 從 712 降到 576，讓參數跟 U-Delta 一樣（15.45M）。
- 只當參考上限，不進主線：目標是不含 attention 的統一 block。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsda15m` | 1 | 15.45M | 3.840 | 3.798 | 4.212 | 3.793 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.762。

## 結論

15M 3.840，比 U-Delta 好 0.057，開頭 64 個 token 4.212（U-Delta 4.258）：局部 attention 補的是近距離。不含 attention 的 [U-Delta-rawFFN](u-delta-rawffn.md)（3.812）和 [U-Delta-rawFFN+λlr1000](u-delta-rawffn-lambda-lr1000.md)（3.796（3 個 seed，3.793–3.799））都已經贏它。

## 相關實驗

- [含 attention 的對照](../experiments/attention-control.md)
- [U-Delta 為什麼贏 GDN](../experiments/vs-gdn.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwhy.txt）
python scripts/train_variant.py --out_dir=outputs/fwsda15m --variant=SDA --hidden_dim=576 --dataset=fineweb --dim=288 --n_layers=15 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsda15m
```
