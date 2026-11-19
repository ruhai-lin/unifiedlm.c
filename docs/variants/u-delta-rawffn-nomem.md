# U-Delta-rawFFN-nomem（代號 `SDn`）

沒有記憶的對照：U-Delta-rawFFN 把記憶那一半整個拿掉、從頭訓練，只剩 FFN 和位址前的短卷積。

- **類別**：基線
- **基於**：[U-Delta-rawFFN](u-delta-rawffn.md)
- **程式碼**：`scripts/delta.py` 的 `NoMemoryDeltaBlock`，訓練時 `--variant=SDn`

## 怎麼做的

- U-Delta-rawFFN：y = μ z + Eᵀq，記憶 E 用 delta rule 寫。
- U-Delta-rawFFN-nomem：y = μ z，沒有 E；記憶的參數（值投影、β、遺忘門、log_rate、log_eps）都拿掉，所以參數少 1.3M（14.16M 對 15.45M）。
- 每層位址前寬 4 的因果卷積留著，所以每層還看得到最近 4 個 token，12 層疊起來最多約 36 個 token。

**問題**：關掉記憶的診斷只量得到「帶著記憶訓練的模型多依賴記憶」。沒有記憶、從頭訓練的模型能做到多好？記憶真正值多少？

```text
y = μ z_t，z = relu(K u_t + b)²，u = conv4(RMSNorm(x))
```

**設定**：15M（12 層、2,300 步），跟 U-Delta-rawFFN 15M 同設定；1 個 seed（1337），A40 16 分鐘。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `fwsdn15m` | 1 | 14.16M | 4.388 | 4.386 | 4.421 | 4.387 |

## 結論

15M（1 個 seed）：NLL 4.388（長度 1,024）、4.386（4,096），比 U-Delta-rawFFN（3.812）差 0.576，也比 Transformer（3 個 seed 平均 3.945）差 0.443，跟 U-Avg（4.112）比也差。前文完全沒有幫助：每個位置都是 4.37–4.42（前 64 個 token 4.421，512–1,024 4.387，2K–4K 4.379），所以卷積以外讀前文全靠記憶。記憶的價值越後面越大：前 64 個 token 值 0.251（U-Delta-rawFFN 4.170），64–128 值 0.485，512–1,024 值 0.619（U-Delta-rawFFN 3.768）。從頭訓練的 U-Delta-rawFFN-nomem 比推理時把 U-Delta-rawFFN 的記憶關掉（5.248，128 個窗口）好約 0.86，所以關掉記憶那個診斷高估了記憶的作用：模型沒學過沒有記憶的情況。參數少 1.3M，不完全對齊。

## 相關實驗

- [拿掉記憶：記憶值多少、模型多依賴記憶](../experiments/memory-ablation.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdfk.txt）
python scripts/train_variant.py --out_dir=outputs/fwsdn15m --variant=SDn --hidden_dim=712 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/fwsdn15m
```
