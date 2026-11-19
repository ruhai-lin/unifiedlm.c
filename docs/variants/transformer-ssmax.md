# Transformer+SSMax（代號 `TS`）

Transformer 的 attention 換成 Scalable-Softmax（logits 乘 s·log n），想讓它讀長文時不變鈍。

- **類別**：含 attention 的對照（只當參考，不算統一 block）
- **基於**：[Transformer](transformer.md)
- **程式碼**：`scripts/variants.py` 的 `AttentionHalf(ssmax=True)`，訓練時 `--variant=TS`

## 怎麼做的

- Transformer+SSMax（要重訓）：一個看得到 n 個 key 的 query，logits 乘上 s·log n；s 每個 head 可學，從 1/log 1,024 開始，所以在 n＝1,024 時等於普通 softmax。
- log-n 縮放（不用重訓，Qwen 的做法）：只在評估時，超過訓練長度 L 之後 query 乘上 log n / log L；在 L 內輸出完全不變。
- 兩者都不改參數量（Transformer+SSMax 每層多 6 個數）。

**問題**：純 RoPE 的 Transformer 讀到 2 倍長度以上就崩（20K 時 NLL 6.05），拿它跟 U-Delta 比是稻草人。滑動窗口已經是一個公平的對照；SSMax 和 log-n 測的是能不能讓 Transformer 用完整上下文也不崩。

```text
Transformer+SSMax：softmax(s · log n · q kᵀ / √d)
log-n：q_t ← q_t · max(1, log n_t / log L)
```

**設定**：參數：跟 Transformer 一樣；狀態：跟 Transformer 一樣，KV cache 隨長度增加。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fw15m_ts` | 1 | 15.19M | 3.913 | 5.101 | 4.230 | 3.880 |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：6.515；推理時滑動窗口 1,024：3.857。

## 結論

訓練長度內 Transformer+SSMax 比 Transformer 好 0.015（3.913 對 3.928，在 Transformer 的 seed 標準差附近，1 個 seed），仍輸 U-Delta（3.897）。超過訓練長度照樣崩，比 Transformer 更差：讀到 20K 的 16K–20K 段 Transformer+SSMax 6.52、Transformer 6.05（3 個 seed）、U-Delta 3.82。SSMax 解決的是注意力被攤平，不是 RoPE 沒看過的位置。Transformer 的 log-n 縮放也沒用（16K–20K 6.19）。Transformer+SSMax 加窗口 1,024 是 3.857，比 Transformer 加窗口（3.893）好一點。

## 相關實驗

- [長上下文與外推（讀到 20K）](../experiments/extrapolation.md)
- [含 attention 的對照](../experiments/attention-control.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwnext3.txt）
python scripts/train_variant.py --out_dir=outputs/fw15m_ts --variant=TS --hidden_dim=768 --head_dim=48 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=32 --gradient_accumulation_steps=4 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fw15m_ts
```
