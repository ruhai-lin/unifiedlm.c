# Transformer+conv（代號 `TC`）

Transformer 在 attention 前加上跟 U-Delta 一樣的寬 4 因果卷積：看 U-Delta 的優勢是不是卷積帶來的。

- **類別**：含 attention 的對照（只當參考，不算統一 block）
- **基於**：[Transformer](transformer.md)
- **程式碼**：`scripts/variants.py` 的 `AttentionHalf(conv=True)`，訓練時 `--variant=TC`

## 怎麼做的

- 每個注意力半層：歸一化後先過一個寬 4 的因果 depthwise 卷積，再算 q、k、v。
- 卷積初始化成恆等（一開始等於沒有卷積），沒有激活函數，跟 U-Delta 的卷積完全相同。
- 其他（RoPE、FFN、寬度）都跟 Transformer 一樣，參數只多 6.9K。

**問題**：U-Delta 的位址前面有卷積，Transformer 沒有。卷積本身讓每個 token 混到前幾個 token 的資訊，可能就足以解釋 U-Delta 的優勢。

```text
a_t = conv4(RMSNorm(x))_t        # 看前 3 個 token 加自己
q, k, v = W_q a_t, W_k a_t, W_v a_t  # 之後跟 Transformer 一樣
```

**設定**：參數：15.20M（Transformer 15.19M）；狀態：跟 Transformer 一樣，KV cache 隨長度增加。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fw15m_tc` | 1 | 15.20M | 3.921 | 4.910 | 4.237 | 3.889 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：5.887。

## 結論

卷積解釋不了 U-Delta 的優勢。Transformer+conv 只比 Transformer 好 0.007，大約是 U-Delta 領先幅度（0.031）的四分之一，而且在 Transformer 三個 seed 的標準差（0.016）之內；前 64 個 token 完全沒改善。

## 相關實驗

- [U-Delta 為什麼贏 GDN](../experiments/vs-gdn.md)
- [含 attention 的對照](../experiments/attention-control.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwnext.txt）
python scripts/train_variant.py --out_dir=outputs/fw15m_tc --variant=TC --hidden_dim=768 --head_dim=48 --dim=288 --n_layers=12 --batch_size=32 --gradient_accumulation_steps=4 --warmup_iters=360 --max_iters=2300 --dataset=fineweb --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --learning_rate=5e-4 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fw15m_tc
```
