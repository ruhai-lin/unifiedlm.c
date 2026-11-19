# UD2-gate-norm-q32-rope-ev（代號 `UD2-gate-norm-q32-rope-ev`）

UD2-gate-norm-q32-rope 的記憶寫入值加上 token embedding：v = W_v(h + g ⊙ e)，e 是 RMSNorm 過的 embedding，g 每個 channel 一個、從 0 開始；多 3,456 個參數。15M 好 0.009（1 個 seed），進步在照抄、見過的 token 和少見的詞。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[UD2-gate-norm-q32-rope](ud2-gate-norm-q32-rope.md)
- **程式碼**：`scripts/delta.py` 的 `EmbedValueRopeReadNormQueryGateAnchorDeltaBlock`（`scripts/variants.py` 裡的 `UD2-gate-norm-q32-rope-ev`），訓練時 `--variant=UD2-gate-norm-q32-rope-ev`

## 怎麼做的

- UD2-gate-norm-q32-rope 的記憶寫入值加上 token embedding：v = W_v(h + g ⊙ e)，e 是 RMSNorm 過的 embedding，g 每個 channel 一個、從 0 開始；多 3,456 個參數。15M 好 0.009（1 個 seed），進步在照抄、見過的 token 和少見的詞。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `ud2_gate_norm_q32_rope_ev_15m` | 1 | 15.46M | 3.707 | 3.661 | 4.098 | 3.659 |

## 結論

UD2-gate-norm-q32-rope 的記憶寫入值加上 token embedding：v = W_v(h + g ⊙ e)，e 是 RMSNorm 過的 embedding，g 每個 channel 一個、從 0 開始；多 3,456 個參數。15M 好 0.009（1 個 seed），進步在照抄、見過的 token 和少見的詞。

## 相關實驗

- [讀出端：門、縮放、RMSNorm、讀位址修正、旋轉相對位置](../experiments/read-side.md)
- [逐 token 差距：輸在哪些 token](../experiments/token-gap.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdfk.txt）
python scripts/train_variant.py --out_dir=outputs/ud2_gate_norm_q32_rope_ev_15m --variant=UD2-gate-norm-q32-rope-ev --hidden_dim=662 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/ud2_gate_norm_q32_rope_ev_15m
```
