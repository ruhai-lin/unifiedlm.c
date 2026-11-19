# FineWeb-Edu 42M：S 對 T（PoC，2026-11-09）

Gary 決定先做 42M，資料用 FineWeb-Edu，跑在 A6000 上。這是 PoC：每個模型只跑一次，一組 LR、一個 seed，不 sweep。

## 要回答的問題

- 42M 的 S 能不能在大一點的真實語料上正常學（loss 一路下降、沒有發散）。
- S 和 T 的 val loss 差多少。BabyLM 15M 時 S 比 T 高 0.10，但那是不同的資料和詞表，不能直接當成放大的趨勢。要看趨勢，同一份資料上也補 15M 的 S 和 T（Gary 2026-11-10 同意，`scripts/nrp/fw15m.txt`：dim 288、12 個 block，S 712 格，T SwiGLU 768、head dim 48；2300 步 = 3.01 億 token，約參數量 20 倍；warmup 360 步，和 42M 一樣佔總步數約 16%；其他同 42M）。
- 超過訓練長度（1024 → 4096）時 S 還撐不撐得住，T 是否照樣崩。

## 資料

- FineWeb-Edu sample-10BT 14 個 parquet 中的前 2 個（`000_00000`、`001_00000`），由沒改過的 `fineweb.py` 用 Llama 2 32K 詞表切成約 1 億 token 一個的 uint16 shard。
- 照 `fineweb.py` 的規則，排序後最後一個 shard 是驗證集，訓練不會讀到。
- 程式在 `scripts/fineweb/fineweb_subset.py`，在 NRP 的 CPU Job `ulm-fw-prep` 上建好，放在 PVC `ulm-storage` 的 `/workspace/data/FineWebEdu_2of14`，附 `SHA256SUMS`。

## 設定（`scripts/nrp/fw42m.txt`）

| | S | T |
| --- | --- | --- |
| 結構 | 16 個 UnifiedBlock，dim 512，每 block 1280 格 | 16 個 block = 8 個 attention + 8 個 FFN 交替，dim 512，SwiGLU 1376，head dim 64（8 個 head，同 llama2.c 42M） |
| 參數量 | 41,632,288 | 41,689,600 |

兩者共用：

- 序列長度 1024，Llama 2 詞表 32000。
- 每步 32 × 4（梯度累積）× 1024 = 131,072 token，練 6400 步，共 8.39 億 token，約參數量的 20 倍。資料不重複。
- LR 5e-4（沿用 15M 的配方，`docs/models.md` 42M 欄寫「待調參」），warmup 1000 步，再 cosine 衰減到 0。weight decay 0.1，β2 0.95，dropout 0，bf16 + `torch.compile`，seed 1337。
- 每 500 步用 50 個 batch 估一次 val loss。

風險：LR 沒調。如果 S 和 T 對 LR 的敏感度不同，比較可能不公平。結果只當作單次 PoC 觀察。

## 評估（`scripts/fineweb/eval_fineweb.py`）

- 最後一個 checkpoint 在驗證 shard 上的逐位置 NLL：長度 1024 取 2048 個窗口（約 210 萬 token），長度 4096 取 512 個窗口。
- 沒有 BLiMP：手上的 BLiMP 是用 BabyLM 詞表切的。

## 預估成本

- A6000 測速（`scripts/bench/train_speed.py`，bf16，T=1024）：
  - 42M 的 S 在 batch 32 時每秒 8.0 萬 token，顯存 22GB，練完約 2.9 小時。
  - 42M 的 T 在 batch 32 時每秒 21.4 萬 token，約 1.1 小時。
- 兩個 pod 各一張 A6000，每個 pod 都在 6 小時上限內。
- A6000 節點的空閒 CPU 只有 0.1–0.3 顆，所以 pod 只要 150m CPU，裝環境和編譯會比較慢。

## 排程（2026-11-10）

- 每種允許的卡各送一個 Job 排隊（`scripts/nrp/fw-run.sh`），依序做 `fw42m.txt`、`fw15m.txt`，靠 `run_queue.sh` 的 claim 分工：只有一張卡時它一路做下去，有第二張就自動分攤。
- 24GB / 32GB 的卡把 42M 改成 micro-batch 16 × 梯度累積 8，每步 token 數不變。
- 第一張排上的是 RTX 5000 Ada（32GB，k8s-gpu-6.ucsc.edu），領走 42M S。它在開跑後才加 15M，所以是在 pod 裡的 `run_index.sh` 結尾補上一段：發布完就接著跑兩份計畫裡還沒人領的項目。

## 疊上去的下一步：P-free 15M 和 42M（2026-11-10）

- Gary 要卡不閒著，先把下一步排上去再一起分析。下一步是統一版真正的對照組 P-free：分開兩組位址，格子數和 S 一樣（42M：m_s = m_f = 1280；15M：712），參數多出 K_f、b_f、ε_f。
- 問題：BabyLM 上 S 和 P-free 打平、P-free 超過訓練長度會崩；在 FineWeb 上、到 42M 是否還是這樣。
- `scripts/nrp/fwp.txt`，其他設定同 S。三個在跑的 pod 都在 `run_index.sh` 結尾接上這份計畫；15M 排前面，讓最早空出來的 A10 做短的，3090 接長的 42M。
- 三張卡都小於 40GB，pod 裡的 `fwp.txt` 把 42M 改成 micro-batch 16 × 梯度累積 8（P-free 比 S 多吃顯存，S 在 batch 32 已經用到 22GB）。repo 裡的 `fwp.txt` 就是 pod 裡那份。
- P-free 在 d=512 沒有跑過 kernel 檢查（`check_kernel.py` 只測到 d=288），也沒有 preflight。

## 再疊上去：110M 的 S 和 T、15M 三個 seed（2026-11-10）

- **110M**（`scripts/nrp/fw110m.txt`）：dim 768、24 個 block，S 1912 格，T SwiGLU 2048、head dim 64。16,700 步 × 131,072 = 21.9 億 token，約參數量 20 倍。warmup 2600 步，約 16%，LR 5e-4，每 1000 步評估一次。micro-batch 16 × 梯度累積 8，小於 40GB 的卡改成 8 × 16。
- **110M 的資料要多一倍**：2of14 只有 16.7 億訓練 token，不夠。`scripts/fineweb/prep_4of14.sh` 再切 parquet 002、003，和 2of14 的訓練 shard 合成 `FineWebEdu_4of14`，約 34 億訓練 token。驗證 shard 沿用 2of14 的 `001_00000_008.bin`，改名為 `val_001_00000_008.bin`，讓它排序在最後，所以 15M、42M、110M 用的是同一個驗證集，110M 也不會練到它。
- **15M 三個 seed**（`scripts/nrp/fw15m_seeds.txt`）：Gary 允許的。S、T、P-free 各加 seed 1338、1339，跟 1337 合起來共三個。跟 BabyLM 一樣，seed 同時改變初始化和資料順序：`fineweb_subset.DATA_SEED = seed − 1337` 會平移 `fineweb.py` 原本固定的亂數 42，所以 1337 的順序不變。
- **排程**：
  - 在跑的三個 pod 用 `run_index.sh` 結尾的第三段接 `scripts/nrp/fw-next.sh`。
  - A10 比較慢，先做 seed，110M 只做 T（它那份 `fw110m.txt` 只有 T 那一行）。
  - 5000 Ada 和 3090 先做 110M 再做 seed；4of14 還沒好的話，先跳過 110M。
  - 新的 Job（`scripts/nrp/fw-run2.sh`）等資料好了，再送到較快的卡上排隊。
- **110M 的 S 在 A6000 上估計要約 22 小時**（3090 Ti 實測 30K tok/s），T 約 9 小時。
- **還沒驗證的風險**：在 Mac 上 forward 一次，110M 的 S 剛初始化時 loss 是 15.3。
