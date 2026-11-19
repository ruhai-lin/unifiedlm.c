# 統一性實驗報告（FineWeb-Edu，15M / 42M）

截至 2026-11-17。目標是**不含 attention 的統一 block**（同一組格子同時當 FFN 和上下文記憶），在同樣的資料、步數和學習率下達到同規模 Transformer 的語言建模質量；含 attention 的版本只作對照。主線問題見 [`research_goal.md`](research_goal.md)。結果、主張和主要觀察在 [README](../README.md)。

每個變體一份文件（怎麼做的、程式碼在哪、結果、怎麼重現），每個實驗一份文件（問題、怎麼做的、結果表、結論）。

## 設定

- **資料**：FineWeb-Edu sample-10BT 的前 2 個 parquet（`FW_SUBSET=2of14`，見 `scripts/fineweb/fineweb_subset.py`），Llama 2 32K 詞表；驗證集是 `001_00000_008.bin`。110M 用前 4 個（`4of14`），驗證集相同。
- **15M**：dim 288，12 個 block，`hidden_dim` 712（U- 系列），長度 1,024，每步 131,072 tokens，2,300 步（約 3.0 億 tokens）；lr 5e-4，warmup 360，cosine 到 0，weight decay 0.1，β₂ 0.95，dropout 0。
- **42M**：dim 512，16 個 block，`hidden_dim` 1248（U-Delta 系列；U-Avg 1280），6,400 步（約 8.4 億 tokens），warmup 1,000，其餘同 15M。**110M**：dim 768，24 個 block，16,700 步。
- **所有模型同一個學習率，沒有調參**；參數量對齊到 ±2%（Split、U-Gate-rank64 例外）。換 seed 同時換初始化和資料順序。
- **評測**：驗證 NLL，長度 1,024（2,048 個窗口）和 4,096（512 個窗口），按位置分段（`scripts/fineweb/eval_fineweb.py`）。讀到 20K：64 個長 20,480 的窗口（`scripts/fineweb/long_eval.py`）。Transformer 超過訓練長度時報完整上下文和滑動窗口 1,024 兩種。
- **怎麼跑**：`scripts/nrp/*.txt` 每行一個 run（run 名 + `scripts/train_variant.py` 的參數），`scripts/nrp/run_index.sh` 訓練、評測併發布到 PVC；FineWeb 的 run 每次 eval 都把 checkpoint 複製到 PVC，被中斷時由下一個 pod 接著跑。在 NRP Nautilus 上一次一張卡（A40、A6000、A10、L40 等）。

## 實驗

| 實驗 | 問題 | 簡短回答 |
| --- | --- | --- |
| [規模：15M → 42M → 110M](experiments/scale.md) | 15M 贏 Transformer 會不會只是小模型的現象？ | 15M 贏；42M 打平到略輸：最好的 U-Delta-rawFFN+λlr1000 輸 0.009（1 個 seed）。 |
| [讀出端：門、縮放、RMSNorm、讀位址修正、旋轉相對位置](experiments/read-side.md) | 記憶的讀出怎麼改？ | 讀出門（3 個 seed）、讀出 RMSNorm、讀位址修正、記憶裡的旋轉相對位置都有用，合起來 3.716；每層固定縮放、秩 128、多 head、除以記憶量都沒用。 |
| [逐 token 差距：輸在哪些 token](experiments/token-gap.md) | 統一 block 跟 Transformer 的差距落在哪些 token 上？ | 15M 贏的近九成是照抄前文；42M 輸的全是窗口裡沒出現過的新 token。 |
| [遠處的前文值多少](experiments/reach.md) | 把前文截到最近 128 或 512 個 token，會差多少？42M 輸在新 token 是因為遠距記憶嗎？ | 不是：約七成的差距在最近 128 個 token 內就有；遠處前文主要用來照抄，記憶模型的新 token 還會被舊內容干擾，讀出端的修改都沒擋掉。 |
| [多 seed 的穩定性](experiments/seeds.md) | 差距是不是 seed 的噪聲？ | 統一 block 的 seed 標準差約 0.003，Transformer 約 0.016；最新的統一 block 三個 seed 都贏。 |
| [統一的對照：共享位址 vs 分開位址](experiments/unification.md) | 同一組格子同時當 FFN 和記憶，會不會比分開差？ | 原版上分開的好 0.074，但多 16% 參數；U-Delta 上還沒做。 |
| [記憶的寫法：平均、遺忘門、delta rule](experiments/write-rule.md) | 格子裡的記憶該怎麼寫？ | delta rule 遠好於平均：4.112 → 3.998 → 3.898。 |
| [FFN 那一半讀哪個位址](experiments/ffn-address.md) | U-Delta 到 42M 為什麼輸 Transformer？ | FFN 讀歸一化位址是瓶頸；改讀 relu² 後 15M 好 0.086、42M 好 0.122。 |
| [記憶設計：快記憶、慢記憶、遺忘率](experiments/memory-design.md) | 需要慢記憶嗎？遺忘率怎麼設？ | 不需要慢記憶；記憶越短越好，短到上限還在變好。 |
| [拿掉記憶：記憶值多少、模型多依賴記憶](experiments/memory-ablation.md) | 沒有記憶會差多少？推理時關掉記憶會怎樣？哪一層最重要？ | 記憶真正值 0.58（沒有記憶、從頭訓練的版本 4.388）；推理時關掉記憶 NLL 升 1.5–2.0，按位置變平：讀前文全靠記憶。 |
| [每個格子記多久（按寫入量加權）](experiments/slot-retention.md) | 遺忘門的“有效記憶長度”只有 1–2 個 token，記憶真的這麼短嗎？ | 不是：按寫入量加權，U-Delta-rawFFN+λlr1000 有 3 層（42M 有 5 層）過了 256 個 token 還留 20–50%；U-Delta-λlr1000 也有 4 層。 |
| [內容是被遺忘還是被覆寫](experiments/forgetting-vs-overwrite.md) | 格子丟掉內容，是衰減還是被新寫入覆蓋？ | 典型格子是衰減（覆寫不到 1%）；少數常被寫的格子覆寫較重。 |
| [寫入量與 loss](experiments/write-vs-loss.md) | 記憶是不是只寫沒預測到的 token？ | 只是偏向，相關 0.14–0.21；自己預測到的也照寫。 |
| [位址尖銳度](experiments/sharpness.md) | 位址更尖（更大的次方）會不會更好？ | 不會：可學的次方停在 2.0–2.35，NLL 不變。 |
| [U-Delta 為什麼贏 GDN](experiments/vs-gdn.md) | 同樣是 delta rule，差在哪？ | 不是卷積、狀態大小或寫入規則，是以 FFN 為錨的格子。 |
| [長上下文與外推（讀到 20K）](experiments/extrapolation.md) | 過了訓練長度會不會崩？ | 統一 block 都不崩；Transformer 只有開滑動窗口才不崩。 |
| [記憶任務 m2kv](experiments/memory-tasks.md) | 純記憶題能記多遠？ | 只有 U-Delta 系列外推：間隔 4,096 仍有 99.95%。 |
| [含 attention 的對照](experiments/attention-control.md) | 加回局部 attention 能好多少？ | 好 0.058，但不含 attention 的最新版本已經贏它。 |

## 變體

### 主線（統一 block，不含 attention）

| 變體 | 代號 | 是什麼 | 15M 驗證 NLL |
| --- | --- | --- | ---: |
| [U-Avg](variants/u-avg.md) | `S` | 原版 UnifiedLM，也就是本倉庫 `model.py` 的標準模型：同一組格子既當 FFN 又當上下文記憶，格子裡存“先驗 + 上下文證據”的後驗均值。 | 4.112（3 個 seed，4.107–4.116） |
| [U-Delta](variants/u-delta.md) | `SD` | 把格子裡的“後驗均值”換成以 FFN 為錨的 delta rule 記憶：記憶只存 FFN 沒預測到的部分，同一個位址的新值按比例覆蓋舊值。 | 3.898（3 個 seed，3.895–3.901） |
| [U-Delta-rawFFN](variants/u-delta-rawffn.md) | `SDu` | U-Delta 的 FFN 那一半改讀未歸一化的 relu²（就是標準的 ReLU² FFN），記憶那一半不變。 | 3.812 |
| [U-Delta-rawFFN+λlr1000](variants/u-delta-rawffn-lambda-lr1000.md) | `SDul` | 3 個 seed、有 42M 的統一 block 裡最好的。 U-Delta-rawFFN 再加上 U-Delta-λlr1000：每格基礎遺忘率學快 1,000 倍。不加參數。 | 3.796（3 個 seed，3.793–3.799） |
| [UD2](variants/ud2.md) | `SDula` | 新的基底 UD2。 U-Delta-rawFFN+λlr1000 的寫入目標改錨在 FFN 實際給出的 μz（原本是歸一化的 μr）。不加參數。之後的變體都是 UD2 加後綴，後綴表示讀出端的修改。 | 3.778 |
| [UD2-gate](variants/ud2-gate.md) | `SDulag` | UD2 的記憶讀出乘上每個 token 一個門 o = 2·sigmoid(w·h + b)（0 到 2，起點約 1）。每層多 289 個參數。3 個 seed 都比 UD2 好。 | 3.765（3 個 seed，3.761–3.767） |
| [UD2-scale](variants/ud2-scale.md) | `SDulas` | UD2-gate 的對照：每個 token 的門換成每層一個可學的純量 2·sigmoid(b)。每層多 1 個參數。 | 3.776 |
| [UD2-gate-norm](variants/ud2-gate-norm.md) | `SDulagn` | UD2-gate 的讀出先 RMSNorm 再過門（GDN 的做法）。每層多 288 個參數。 | 3.739 |
| [UD2-gate-q32](variants/ud2-gate-q32.md) | `SDulagq` | UD2-gate 的讀位址加一個秩 32 的修正，讓讀和寫的比對不對稱；hidden_dim 降到 662 對齊參數。 | 3.746 |
| [UD2-gate-norm-q32](variants/ud2-gate-norm-q32.md) | `SDulagnq` | UD2-gate-norm 和 UD2-gate-q32 合在一起：讀出先 RMSNorm，讀位址加秩 32 的修正。 | 3.727 |
| [UD2-gate-q128](variants/ud2-gate-q128.md) | `UD2-gate-q128` | UD2-gate-q32 的讀位址修正從秩 32 加到 128；hidden_dim 降到 538 對齊參數（格子少 19%）。 | 3.760 |
| [UD2-gate-norm-q32-rope](variants/ud2-gate-norm-q32-rope.md) | `UD2-gate-norm-q32-rope` | 15M、42M 目前最好（各 1 個 seed），42M 贏 Transformer 0.044。 UD2-gate-norm-q32 的一半格子兩兩成對旋轉（週期 4–256 個 token），讓記憶的比對看相對距離；不加參數。 | 3.722（3 個 seed，3.716–3.727） |
| [UD2-gate-norm-q128-rope](variants/ud2-gate-norm-q128-rope.md) | `UD2-gate-norm-q128-rope` | 三個讀出端的修改一起：讀出 RMSNorm、秩 128 的讀位址修正、一半格子加旋轉相對位置；hidden_dim 538。 | 3.732 |
| [UD2-gate-norm-h4q8](variants/ud2-gate-norm-h4q8.md) | `UD2-gate-norm-h4q8` | UD2-gate-norm-q32 的讀出拆成 4 個 head，每個 head 有自己秩 8 的讀位址修正、各自 RMSNorm；參數一樣。 | 3.732 |
| [UD2-gate-mass-q32](variants/ud2-gate-mass-q32.md) | `UD2-gate-mass-q32` | UD2-gate-q32 的讀出除以位址碰到的記憶量（像 softmax 的分母）。 | 3.749 |
| [UD2-gate-norm-q32-ropelo](variants/ud2-gate-norm-q32-ropelo.md) | `UD2-gate-norm-q32-ropelo` | UD2-gate-norm-q32-rope 只在下半的 block（15M 12 個裡的 6 個）加旋轉相對位置，上半不加；參數一樣。15M 比全部加好 0.006（1 個 seed），進步在遠處照抄；42M 跟全部加、完全不加一樣（3.171）。 | 3.711 |
| [UD2-gate-norm-q32-rope-ev](variants/ud2-gate-norm-q32-rope-ev.md) | `UD2-gate-norm-q32-rope-ev` | UD2-gate-norm-q32-rope 的記憶寫入值加上 token embedding：v = W_v(h + g ⊙ e)，e 是 RMSNorm 過的 embedding，g 每個 channel 一個、從 0 開始；多 3,456 個參數。15M 好 0.009（1 個 seed），進步在照抄、見過的 token 和少見的詞。 | 3.707 |
| [UD2-gate-norm-q32-ev](variants/ud2-gate-norm-q32-ev.md) | `UD2-gate-norm-q32-ev` | 不加旋轉相對位置的 UD2-gate-norm-q32 加上 embedding 寫入值（同 UD2-gate-norm-q32-rope-ev）。15M 好 0.0095（1 個 seed），跟加在有旋轉相對位置的版本上一樣。 | 3.718 |
| [UD2-gate-norm-q32-ropelo-ev](variants/ud2-gate-norm-q32-ropelo-ev.md) | `UD2-gate-norm-q32-ropelo-ev` | 15M 目前最好（1 個 seed）。 旋轉相對位置只放下半的 block，每層都加 embedding 寫入值；兩個進步約八成六疊加。 | 3.703 |
| [UD2-gate-norm-q32-decay](variants/ud2-gate-norm-q32-decay.md) | `UD2-gate-norm-q32-decay` | 不加旋轉相對位置；每個格子的衰減速率固定不學（buffer），g=1 時半衰期從 2 到 8,192 個 token 等比分佈，門改成 2σ（最多加速 2 倍）。哪些格子還記得就編碼了過了多久，又不會把內容轉掉。參數少 7,944 個。15M 好 0.006（1 個 seed），照抄是所有版本中最好的。 | 3.721 |
| [UD2-gate-norm-q32-decay-ev](variants/ud2-gate-norm-q32-decay-ev.md) | `UD2-gate-norm-q32-decay-ev` | UD2-gate-norm-q32-decay 加 embedding 寫入值；兩個進步完全疊加。每層同一種 block、不用旋轉相對位置、沒有 ×1000 學習率倍數的版本中 15M 最好（1 個 seed）。 | 3.712 |
| [UD2-gate-norm-q32-decay-half](variants/ud2-gate-norm-q32-decay-half.md) | `UD2-gate-norm-q32-decay-half` | 每個 block 一半格子用 decay 的固定半衰期、一半照原本可學（×1000、softplus 門）。15M 跟 decay 一樣（1 個 seed）：全部固定就夠。 | 3.722 |
| [UD2-gate-norm-q32-conv8](variants/ud2-gate-norm-q32-conv8.md) | `UD2-gate-norm-q32-conv8` | 不加旋轉相對位置，卷積從看 4 個 token 改成 8 個；多 13,824 個參數。15M 跟不改一樣（1 個 seed）：旋轉相對位置補的不是 4–8 個 token 內的詞序。 | 3.728 |

### 記憶設計

| 變體 | 代號 | 是什麼 | 15M 驗證 NLL |
| --- | --- | --- | ---: |
| [U-Delta-rawFFN+λlr1000+carry](variants/u-delta-rawffn-lambda-lr1000-carry.md) | `SDulc` | U-Delta-rawFFN+λlr1000 訓練時記憶跨窗口接續：每個窗口從同一串上一個窗口結束的記憶開始（梯度不跨窗口），評測照舊從空記憶開始。 | 3.815 |
| [U-Delta-λlr10](variants/u-delta-lambda-lr10.md) | `SDl` | U-Delta 的每格基礎遺忘率 `log_rate` 有效學習率 ×10（重參數化，起點不變）。 | 3.887 |
| [U-Delta-λlr100](variants/u-delta-lambda-lr100.md) | `SDl100` | U-Delta 的 `log_rate` 有效學習率 ×100。 | 3.851 |
| [U-Delta-λlr1000](variants/u-delta-lambda-lr1000.md) | `SDl1000` | U-Delta 的 `log_rate` 有效學習率 ×1000：按格子平均，幾乎每層都把記憶縮到上限（約 1 個 token）；按寫入量加權，第 1–3、11 層仍記得過 256 個 token。 | 3.843 |
| [U-Delta-spreadλ](variants/u-delta-spread-lambda.md) | `SDw` | U-Delta 的遺忘率初始值分散在 1/(L/4)～1/(16L)：同一份記憶裡有快有慢。 | 3.890 |
| [U-Delta-diverseλ](variants/u-delta-diverse-lambda.md) | `SDv` | U-Delta 的基礎記憶長度從 4 到 65,536 個 token 分散，門最多只能讓遺忘快 2 倍，所以慢格子留得住。 | 3.877 |
| [U-Delta-docreset](variants/u-delta-docreset.md) | `SDb` | U-Delta 遇到文件開頭（BOS）就清空記憶。 | 3.890 |
| [U-Delta+slow](variants/u-delta-plus-slow.md) | `SD2` | U-Delta 再加一份慢記憶（每格固定速率，起點 1/(16L)）和讀出混合 ρ。 | 3.888 |
| [U-Delta+fast2](variants/u-delta-plus-fast2.md) | `SD2c` | U-Delta+slow 的同容量對照：第二份記憶也是普通的快記憶，狀態大小相同。 | 3.897 |

### 位址尖銳度

| 變體 | 代號 | 是什麼 | 15M 驗證 NLL |
| --- | --- | --- | ---: |
| [U-Delta-learnP](variants/u-delta-learnp.md) | `SDp` | U-Delta 每層的位址次方 p 可學（從 2 開始）。 | 3.892 |
| [U-Delta-ReLU4](variants/u-delta-relu4.md) | `SDr` | U-Delta 前 4 個 block 用 ReLU 位址（p = 1），上面照舊 relu²。 | 3.894 |

### 遺忘門

| 變體 | 代號 | 是什麼 | 15M 驗證 NLL |
| --- | --- | --- | ---: |
| [U-Gate](variants/u-gate.md) | `SG` | U-Avg 加上按內容決定的遺忘門：每個格子、每個 token 的遺忘率由位址算出。 | 3.998（3 個 seed，3.988–4.003） |
| [U-Gate-fastinit](variants/u-gate-fastinit.md) | `SH` | U-Gate 的對照：U-Avg 只把遺忘率起點從 1/(4L) 調快到 1/64（U-Gate 的門最後停的位置附近），仍是每層一個、可學。 | 4.051（3 個 seed，4.049–4.052） |
| [U-Gate-nogate](variants/u-gate-nogate.md) | `SR` | U-Gate 的消融：每格各自的遺忘率，沒有門，也沒有寫入時的遺忘項。 | 4.124 |
| [U-Gate-rank64](variants/u-gate-rank64.md) | `SG64` | U-Gate 的門低秩從 16 加到 64。 | 3.977 |
| [U-Gate-fromh](variants/u-gate-fromh.md) | `SGh` | U-Gate 的門改從 block 輸入 h 讀（原本從位址 r 讀）。 | 4.012 |
| [U-Gate-silu](variants/u-gate-silu.md) | `SGs` | U-Gate 的門中間加一層 SiLU（兩層 MLP 的門）。 | 4.002 |

### 統一的對照

| 變體 | 代號 | 是什麼 | 15M 驗證 NLL |
| --- | --- | --- | ---: |
| [Split](variants/split.md) | `P` | 統一的對照：靜態格子和上下文格子用各自的位址，兩組格子不共享。 | 4.038（3 個 seed，4.034–4.040） |

### 基線

| 變體 | 代號 | 是什麼 | 15M 驗證 NLL |
| --- | --- | --- | ---: |
| [U-Delta-rawFFN-nomem](variants/u-delta-rawffn-nomem.md) | `SDn` | 沒有記憶的對照：U-Delta-rawFFN 把記憶那一半整個拿掉、從頭訓練，只剩 FFN 和位址前的短卷積。 | 4.388 |
| [Transformer](variants/transformer.md) | `T` | llama2.c 式 Transformer 基線：RoPE attention + SwiGLU FFN，參數量跟統一 block 對齊。 | 3.945（3 個 seed，3.928–3.957） |
| [GDN](variants/gdn.md) | `GDN` | 線性遞歸（SSM 類）基線：Transformer 的 attention 換成 Gated DeltaNet（6 個 head × 48）。 | 3.964 |
| [GDN-bigstate](variants/gdn-bigstate.md) | `GDNs` | 我們自己的 GDN 變體：2 個大 head（各 144 維），狀態約 3 倍，衰減用平滑上下限。 | 3.964 |
| [GDN-2](variants/gdn-2.md) | `GDN2` | NVIDIA Gated DeltaNet-2：抹除門和寫入門分開，衰減按通道。 | 3.970 |
| [Mesa](variants/mesa.md) | `MESA` | Mesa layer：每個 token 精確解一次加權最小平方（GDN 和 U-Delta 的 delta rule 每步只往這個解走一步）。 | 3.969 |

### 含 attention 的對照（只當參考，不算統一 block）

| 變體 | 代號 | 是什麼 | 15M 驗證 NLL |
| --- | --- | --- | ---: |
| [U-Delta+attn](variants/u-delta-plus-attn.md) | `SDA` | 對照組，含 attention，不算統一 block：每 4 個 U-Delta block 後面插一個窗口 128 的局部 attention 半層。 | 3.840 |
| [Transformer+conv](variants/transformer-conv.md) | `TC` | Transformer 在 attention 前加上跟 U-Delta 一樣的寬 4 因果卷積：看 U-Delta 的優勢是不是卷積帶來的。 | 3.921 |
| [Transformer+SSMax](variants/transformer-ssmax.md) | `TS` | Transformer 的 attention 換成 Scalable-Softmax（logits 乘 s·log n），想讓它讀長文時不變鈍。 | 3.913 |

## 全部結果

### 15M（驗證 NLL，越低越好）

| 模型 | 代號 | 參數量 | seed 數 | 驗證 NLL @1024（範圍） | @4096 |
| --- | --- | ---: | ---: | ---: | ---: |
| **統一 block（主線）** | | | | | |
| UD2-gate-norm-q32 | `SDulagnq` | 15.46M | 1 | 3.727 | 3.683 |
| UD2-gate-norm-q32-rope | `UD2-gate-norm-q32-rope` | 15.46M | 1 | 3.716 | 3.671 |
| UD2-gate-norm-h4q8 | `UD2-gate-norm-h4q8` | 15.46M | 1 | 3.732 | 3.688 |
| UD2-gate-norm-q128-rope | `UD2-gate-norm-q128-rope` | 15.45M | 1 | 3.732 | 3.688 |
| UD2-gate-mass-q32 | `UD2-gate-mass-q32` | 15.45M | 1 | 3.749 | 3.705 |
| UD2-gate-q128 | `UD2-gate-q128` | 15.45M | 1 | 3.760 | 3.715 |
| UD2-gate-norm | `SDulagn` | 15.46M | 1 | 3.739 | 3.696 |
| UD2-gate-q32 | `SDulagq` | 15.45M | 1 | 3.746 | 3.701 |
| UD2-gate | `SDulag` | 15.46M | 3 | 3.765（3.761–3.767） | 3.722 |
| UD2-scale | `SDulas` | 15.45M | 1 | 3.776 | 3.734 |
| UD2 | `SDula` | 15.45M | 1 | 3.778 | 3.736 |
| U-Delta-rawFFN+λlr1000 | `SDul` | 15.45M | 3 | 3.796（3.793–3.799） | 3.753 |
| U-Delta-rawFFN+λlr1000+carry | `SDulc` | 15.45M | 1 | 3.815 | 3.759 |
| U-Delta-rawFFN | `SDu` | 15.45M | 1 | 3.812 | 3.774 |
| U-Delta-λlr1000 | `SDl1000` | 15.45M | 1 | 3.843 | 3.796 |
| U-Delta-λlr100 | `SDl100` | 15.45M | 1 | 3.851 | 3.809 |
| U-Delta-diverseλ | `SDv` | 15.45M | 1 | 3.877 | 3.845 |
| U-Delta-λlr10 | `SDl` | 15.45M | 1 | 3.887 | 3.846 |
| U-Delta+slow | `SD2` | 15.47M | 1 | 3.888 | 3.851 |
| U-Delta-docreset | `SDb` | 15.45M | 1 | 3.890 | 3.843 |
| U-Delta-spreadλ | `SDw` | 15.45M | 1 | 3.890 | 3.848 |
| U-Delta-learnP | `SDp` | 15.45M | 1 | 3.892 | 3.851 |
| U-Delta-ReLU4 | `SDr` | 15.45M | 1 | 3.894 | 3.852 |
| U-Delta | `SD` | 15.45M | 3 | 3.898（3.895–3.901） | 3.857 |
| U-Delta+fast2 | `SD2c` | 15.75M | 1 | 3.897 | 3.853 |
| U-Gate-rank64 | `SG64` | 16.28M | 1 | 3.977 | 3.942 |
| U-Gate | `SG` | 15.46M | 3 | 3.998（3.988–4.003） | 3.963 |
| U-Gate-silu | `SGs` | 15.46M | 1 | 4.002 | 3.966 |
| U-Gate-fromh | `SGh` | 15.38M | 1 | 4.012 | 3.978 |
| U-Gate-fastinit | `SH` | 15.17M | 3 | 4.051（4.049–4.052） | 4.029 |
| U-Avg | `S` | 15.17M | 3 | 4.112（4.107–4.116） | 5.658 |
| U-Gate-nogate（SR） | `SR` | 15.18M | 1 | 4.124 | 5.886 |
| **基線** | | | | | |
| Transformer | `T` | 15.19M | 3 | 3.945（3.928–3.957） | 5.016 |
| GDN | `GDN` | 15.23M | 1 | 3.964 | 3.948 |
| GDN-bigstate | `GDNs` | 15.22M | 1 | 3.964 | 3.948 |
| Mesa | `MESA` | 15.24M | 1 | 3.969 | 3.951 |
| GDN-2 | `GDN2` | 15.21M | 1 | 3.970 | 3.954 |
| Split（P） | `P` | 17.64M | 3 | 4.038（4.034–4.040） | 6.233 |
| U-Delta-rawFFN-nomem | `SDn` | 14.16M | 1 | 4.388 | 4.386 |
| **含 attention 的對照（不算統一 block）** | | | | | |
| U-Delta+attn | `SDA` | 15.45M | 1 | 3.840 | 3.798 |
| Transformer+SSMax | `TS` | 15.19M | 1 | 3.913 | 5.101 |
| Transformer+conv | `TC` | 15.20M | 1 | 3.921 | 4.910 |

按位置（長度 1,024，各 3 個 seed 平均）：

| 位置（3 個 seed 平均） | 0-64 | 64-128 | 128-256 | 256-512 | 512-1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | 4.237 | 3.979 | 3.937 | 3.923 | 3.918 |
| U-Delta-rawFFN+λlr1000 | 4.154 | 3.872 | 3.812 | 3.770 | 3.751 |
| U-Delta | 4.261 | 3.978 | 3.914 | 3.870 | 3.852 |
| U-Delta-rawFFN+λlr1000 − Transformer | -0.083 | -0.107 | -0.125 | -0.154 | -0.167 |

### 42M

| 模型 | 代號 | 參數量 | seed 數 | 驗證 NLL @1024（範圍） | @4096 |
| --- | --- | ---: | ---: | ---: | ---: |
| Transformer | `T` | 41.69M | 1 | 3.214 | 4.871 |
| UD2-gate-norm-q32-rope | `UD2-gate-norm-q32-rope` | 41.77M | 1 | 3.170 | 3.121 |
| U-Delta-rawFFN+λlr1000 | `SDul` | 41.77M | 1 | 3.223 | 3.175 |
| U-Delta-rawFFN | `SDu` | 41.77M | 1 | 3.228 | 3.183 |
| U-Gate | `SG` | 42.33M | 3 | 3.330（3.328–3.333） | 3.288 |
| U-Delta | `SD` | 41.77M | 1 | 3.350 | 3.303 |
| U-Gate-fastinit | `SH` | 41.63M | 1 | 3.361 | 3.327 |
| Split（P） | `P` | 52.14M | 1 | 3.368 | 7.561 |
| U-Avg | `S` | 41.63M | 1 | 3.423 | 8.708 |

按位置（長度 1,024）：

| 位置 | 0-64 | 64-128 | 128-256 | 256-512 | 512-1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | 3.628 | 3.293 | 3.233 | 3.182 | 3.164 |
| UD2-gate-norm-q32-rope | 3.584 | 3.255 | 3.189 | 3.139 | 3.118 |
| U-Delta-rawFFN+λlr1000 | 3.624 | 3.305 | 3.241 | 3.193 | 3.174 |
| U-Delta-rawFFN | 3.623 | 3.307 | 3.245 | 3.196 | 3.179 |
| U-Delta | 3.757 | 3.437 | 3.369 | 3.317 | 3.299 |
| UD2-gate-norm-q32-rope − Transformer | -0.044 | -0.038 | -0.043 | -0.044 | -0.045 |
| U-Delta-rawFFN+λlr1000 − Transformer | -0.005 | +0.012 | +0.009 | +0.010 | +0.011 |

### 110M（FineWeb-Edu 4of14，16,700 步）

| 模型 | 代號 | 參數量 | seed 數 | 驗證 NLL @1024 | @4096 |
| --- | --- | ---: | ---: | ---: | ---: |
| Transformer | `T` | 109.53M | 1 | 2.822 | 4.614 |
| U-Avg | `S` | 109.40M | 1 | 2.994 | 6.635 |

### 讀到 20K（按位置分段的 NLL）

| 模型（15M） | 0-64 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-20480 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T（完整上下文） | 4.106 | 3.842 | 4.784 | 5.603 | 5.875 | 6.074 | 6.088 |
| T（滑動窗口 1,024） | 4.106 | 3.842 | 3.886 | 3.903 | 3.923 | 3.911 | 3.874 |
| GDN | 4.052 | 3.872 | 3.927 | 3.948 | 3.965 | 3.960 | 3.922 |
| U-Delta | 4.114 | 3.812 | 3.836 | 3.876 | 3.875 | 3.841 | 3.817 |
| U-Delta-spreadλ | 4.111 | 3.800 | 3.827 | 3.865 | 3.870 | 3.832 | 3.807 |
| U-Delta+slow | 4.125 | 3.796 | 3.826 | 3.876 | 3.889 | 3.870 | 3.856 |
| U-Delta-diverseλ | 4.107 | 3.789 | 3.818 | 3.869 | 3.886 | 3.873 | 3.859 |
| U-Delta-docreset | 4.129 | 3.806 | 3.825 | 3.861 | 3.858 | 3.822 | 3.797 |
| U-Delta-λlr100 | 4.081 | 3.763 | 3.786 | 3.830 | 3.828 | 3.793 | 3.769 |
| U-Delta-λlr1000 | 4.085 | 3.753 | 3.776 | 3.814 | 3.813 | 3.776 | 3.752 |
| U-Delta-rawFFN | 4.024 | 3.725 | 3.748 | 3.796 | 3.803 | 3.780 | 3.760 |
| U-Delta-rawFFN+λlr1000 | 4.007 | 3.699 | 3.725 | 3.766 | 3.768 | 3.736 | 3.713 |
| UD2 | 3.987 | 3.684 | 3.712 | 3.752 | 3.753 | 3.719 | 3.695 |
| UD2-scale | 3.979 | 3.678 | 3.706 | 3.749 | 3.752 | 3.718 | 3.695 |
| UD2-gate | 3.988 | 3.665 | 3.702 | 3.739 | 3.741 | 3.711 | 3.685 |
| UD2-gate-norm | 3.979 | 3.644 | 3.670 | 3.711 | 3.717 | 3.683 | 3.661 |
| UD2-gate-q32 | 3.976 | 3.651 | 3.672 | 3.715 | 3.719 | 3.684 | 3.663 |
| UD2-gate-norm-q32-rope | 3.954 | 3.619 | 3.645 | 3.687 | 3.689 | 3.655 | 3.632 |
| UD2-gate-norm-q128-rope | 3.977 | 3.638 | 3.658 | 3.704 | 3.708 | 3.673 | 3.651 |
| UD2-gate-mass-q32 | 3.964 | 3.657 | 3.682 | 3.722 | 3.723 | 3.690 | 3.669 |
| UD2-gate-norm-h4q8 | 3.967 | 3.632 | 3.661 | 3.703 | 3.709 | 3.675 | 3.654 |
| UD2-gate-norm-q32 | 3.984 | 3.630 | 3.656 | 3.700 | 3.703 | 3.670 | 3.645 |
| UD2-gate-q128 | 3.994 | 3.671 | 3.689 | 3.728 | 3.734 | 3.698 | 3.673 |
| U-Delta-rawFFN+λlr1000+carry | 4.126 | 3.711 | 3.731 | 3.774 | 3.769 | 3.735 | 3.710 |
| U-Delta+attn | 4.061 | 3.747 | 3.778 | 3.814 | 3.819 | 3.785 | 3.762 |

| 模型（42M） | 0-64 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-20480 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T（完整上下文） | 3.533 | 3.120 | 4.703 | 5.810 | 5.991 | 6.221 | 6.251 |
| T（滑動窗口 1,024） | 3.533 | 3.120 | 3.148 | 3.198 | 3.196 | 3.161 | 3.142 |
| UD2-gate-norm-q32-rope | 3.479 | 3.074 | 3.101 | 3.145 | 3.139 | 3.103 | 3.087 |
| U-Delta | 3.663 | 3.262 | 3.280 | 3.325 | 3.322 | 3.286 | 3.269 |
| U-Delta-rawFFN | 3.521 | 3.139 | 3.158 | 3.208 | 3.206 | 3.176 | 3.166 |
| U-Delta-rawFFN+λlr1000 | 3.512 | 3.129 | 3.154 | 3.198 | 3.192 | 3.156 | 3.142 |

### 記憶任務 m2kv（寬度 32，正確率；訓練時間隔 ≤ 256）

| 模型 | 間隔 0 | 間隔 64 | 間隔 256 | 間隔 1024 | 間隔 4096 |
| --- | ---: | ---: | ---: | ---: | ---: |
| U-Avg | 99.9% | 100.0% | 99.9% | 18.5% | 0.0% |
| U-Gate | 99.6% | 99.7% | 99.5% | 98.5% | 39.6% |
| U-Gate-fastinit | 90.2% | 90.8% | 90.7% | 44.0% | 0.0% |
| Transformer | 99.6% | 99.8% | 99.7% | 0.0% | 0.0% |
| U-Delta | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| U-Delta+slow | 100.0% | 100.0% | 100.0% | 100.0% | 98.3% |

## 進行中與待決定

- **進行中**：沒有。
- **待決定**：U-Delta 系列的共享 vs 分開位址對照；42M 的學習率（U-Delta-rawFFN 只差 0.014）；U-Delta-λlr10、U-Delta-spreadλ 補 seed；Transformer 加 NoPE 層作完整上下文的外推對照；U-Delta 的兩個歸因實驗（寫入目標去掉 −μk 錨、μ 和記憶用各自的位址）；Transformer 學習率 1e-3；Transformer 和 U-Delta 各跑 2 倍步數。

## 資料與程式碼

- 結果資料：`scripts/reports/fineweb/`（`runs.json` 每個 run 的評測、曲線和遺忘門統計，`sgvt20.json` 讀到 20K，`mem.json`、`mem_long.json` 記憶任務，`overwrite.json`、`writeloss.json` 記憶機制分析）。
- 變體註冊表 `scripts/variants.py`，U-Gate 系列 `scripts/gated.py`，U-Delta 系列 `scripts/delta.py`，Mesa `scripts/mesa.py`。依照 `AGENTS.md`，變體在贏之前不進 `model.py`，所以 `model.py` 仍然是 U-Avg。
