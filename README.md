# unifiedlm.c

UnifiedLM用同一种块同时做 token mixing 和 channel mixing 的语言模型。每个块是一个稀疏寻址的关联记忆，格子里存的是"先验 + 上下文证据"的后验均值：先验很强的格子就是一个静态 FFN 槽，先验很弱的格子就是一个上下文寄存器。没有 attention、没有 RoPE、没有 KV cache，推理时每个块的状态大小与上下文长度无关。训练栈沿用 [llama2.c](https://github.com/karpathy/llama2.c)，并配有纯 C 推理 `run.c`以简化硬件部署。

研究目标见 [`docs/research_goal.md`](docs/research_goal.md)，各规模的设置与训练命令见 [`docs/models.md`](docs/models.md)。

## 當前進展（統一性實驗，截至 2026-11-19）

目標：**不含 attention 的統一 block**——同一組格子同時當 FFN 和上下文記憶——在同樣的資料、步數和學習率下，達到同規模 Transformer 的語言建模質量。FineWeb-Edu，Llama 2 32K 詞表，所有模型用同一個學習率 5e-4，沒有調參。完整報告（設定、全部結果表）見 [`docs/experiments.md`](docs/experiments.md)。

### 結果

| 模型 | 是什麼 | 15M | 42M | 讀到 20K（15M，16K–20K 段） |
| --- | --- | ---: | ---: | ---: |
| [Transformer](docs/variants/transformer.md) | llama2.c 式基線 | 3.945（3 個 seed） | 3.214（3 個 seed） | 6.049（完整上下文）/ 3.893（滑動窗口 1,024） |
| **[UD2-gate-norm-q32-rope](docs/variants/ud2-gate-norm-q32-rope.md)** | 15M、42M 目前最好的統一 block | **3.722**（3 個 seed） | **3.171**（3 個 seed） | **3.632** |
| [UD2-gate-norm-q32-decay-ev](docs/variants/ud2-gate-norm-q32-decay-ev.md) | 不加旋轉相對位置、衰減速率固定 2–8,192、記憶寫入值加 token embedding；42M 數字最低的統一 block，但在 seed 差距內 | 3.712 | 3.167（1 個 seed） | — |
| [GDN](docs/variants/gdn.md) | Transformer 的 attention 換成 Gated DeltaNet（SSM 類基線） | 3.964 | 3.229 | 3.922 |
| [Qwen 式混合](docs/variants/qhybrid.md) | 對照，含 attention：每 4 層 3 層 GDN、1 層 gated attention（Qwen3-Next 的排法） | — | 3.166（1 個 seed） | — |
| [UD2-gate](docs/variants/ud2-gate.md) | 15M 有 3 個 seed 的最好統一 block | 3.765（3 個 seed） | 還沒跑 | 3.685 |
| [U-Delta-rawFFN+λlr1000](docs/variants/u-delta-rawffn-lambda-lr1000.md) | U-Delta 系列裡 15M 有 3 個 seed、也有 42M 的 | 3.796（3 個 seed） | 3.223 | 3.713 |
| [U-Avg](docs/variants/u-avg.md) | 原版統一 block（`model.py`） | 4.112（3 個 seed） | 3.423 | — |

驗證 NLL，長度 1,024，越低越好。15M 和 42M 的參數量與 Transformer 對齊到 ±2%。

名稱：UD2 = U-Delta-rawFFN+λlr1000 再把寫入目標錨在 FFN 實際的輸出（代號 `SDula`），是 2026-11-15 起的基底；之後的變體用“UD2-後綴”，後綴依序表示讀出端的修改：gate＝每個 token 的讀出門，scale＝每層一個固定縮放，norm＝讀出先 RMSNorm，qN＝讀位址加秩 N 的修正，rope＝記憶裡的旋轉相對位置。

從原版到現在（15M）：[U-Avg](docs/variants/u-avg.md) 4.112 → 記憶改用 delta rule（[U-Delta](docs/variants/u-delta.md)）3.898 → FFN 那一半讀未歸一化的位址（[U-Delta-rawFFN](docs/variants/u-delta-rawffn.md)）3.812 → 遺忘率學快 1,000 倍（[U-Delta-rawFFN+λlr1000](docs/variants/u-delta-rawffn-lambda-lr1000.md)）3.796 → 寫入目標改錨在 FFN 實際的輸出 μz（[UD2](docs/variants/ud2.md)）3.778 → 記憶讀出加每個 token 一個門（[UD2-gate](docs/variants/ud2-gate.md)）3.765（3 個 seed）→ 讀出先 RMSNorm、讀位址加秩 32 的修正（[UD2-gate-norm-q32](docs/variants/ud2-gate-norm-q32.md)）3.727 → 一半格子加旋轉相對位置（[UD2-gate-norm-q32-rope](docs/variants/ud2-gate-norm-q32-rope.md)）3.722（3 個 seed；UD2 和中間兩步各 1 個 seed）；最後一個的 42M 是 3.171（3 個 seed），贏 Transformer 42M 0.044（3 個 seed）。

### 主張

1. **15M：統一 block 贏 Transformer，三個 seed 都贏。** 3.796 對 3.945；同 seed 配對每組差 -0.161 到 -0.134，最差的統一 block seed 也比最好的 Transformer seed 好 0.128。要打折扣：15M 的 Transformer 不太會用前文（64–128 段到 512–1,024 段只降 0.064），這一點到 42M 消失。逐 token 看，15M 贏的 88.9% 來自照抄前文就猜得到的 token（Transformer 2.017 對 1.416）。見[規模](docs/experiments/scale.md)、[多 seed](docs/experiments/seeds.md)、[逐 token 差距](docs/experiments/token-gap.md)。
2. **42M：讀出端改好的統一 block 贏 Transformer 0.044，三個 seed 都贏。** [UD2-gate-norm-q32-rope](docs/variants/ud2-gate-norm-q32-rope.md) 三個 seed 3.171 ± 0.005（3.170 / 3.176 / 3.166）對 Transformer 三個 seed 3.214 ± 0.0003；同 seed 配對差 -0.044、-0.038、-0.048，每個 seed 每一段位置都贏，最差的統一 block seed 也比最好的 Transformer seed 低 0.038；長度 4,096 是 3.122（Transformer 過了訓練長度升到 4.871）。贏的原因不是旋轉相對位置：拿掉它的 UD2-gate-norm-q32 42M 是 3.172，只差 0.002（1 個 seed）。要打折扣：這個模型是在 15M、seed 1337 上從十個讀出端變體裡挑出來的，seed 1338、1339 是確認用；學習率兩邊都沒調。下面的逐 token 和讀到 20K 是 seed 1337。逐 token 看每一類都贏：全部 -0.045，贏的 52.2% 來自窗口裡見過的 token（-0.065），沒出現過的新 token 也從輸變贏（-0.031）；讀到 20K 的 16K–20K 是 3.087（滑動窗口的 Transformer 3.142）。之前的 [U-Delta-rawFFN+λlr1000](docs/variants/u-delta-rawffn-lambda-lr1000.md) 42M 3.223 輸 0.009；[U-Delta-rawFFN](docs/variants/u-delta-rawffn.md) 輸 0.014。U-Delta-rawFFN+λlr1000 開頭 64 個 token 還略贏，之後每段輸 0.009–0.012；逐 token 看，差距全在窗口裡沒出現過的新 token（+0.020），照抄前文反而略贏。15M 的大幅領先（0.149）它在 42M 沒有保住。讀到 20K 追平滑動窗口的 Transformer（16K–20K 3.142 對 3.142）。見[規模](docs/experiments/scale.md)、[外推](docs/experiments/extrapolation.md)。
3. **過了訓練長度不會崩。** 訓練長度 1,024、讀到 20,480 時，統一 block 的 16K–20K 段在 3.7–4.0（15M）；完整上下文的 Transformer 升到 5.999–6.088，只有推理時開滑動窗口才穩。見[外推](docs/experiments/extrapolation.md)。
4. **比原版好 0.316，最大的一步是記憶的寫法。** 把格子裡的累計平均換成以 FFN 為錨的 delta rule 就好 0.214，見[記憶的寫法](docs/experiments/write-rule.md)；FFN 讀未歸一化位址再好 0.086，見 [FFN 讀哪個位址](docs/experiments/ffn-address.md)。
5. **還沒證明的：共享位址不吃虧。** 在原版上，靜態和記憶格子分開位址（Split）比共享好 0.074，但多 16% 參數；U-Delta 系列還沒做這個對照。見[統一的對照](docs/experiments/unification.md)。

### 42M 診斷：UD2-gate-norm-q32-rope 贏在哪裡

**結論：0.044 是全面的贏。** 每一類 token 都贏 Transformer，最大一份來自窗口裡見過的 token；[U-Delta-rawFFN+λlr1000](docs/variants/u-delta-rawffn-lambda-lr1000.md) 42M 輸掉的位置 64 之後的新 token，到這裡也變成贏。但新 token 變好，是因為每種上下文長度都猜得更準，不是遠處的前文用得更好。下面的數字是減掉 Transformer 42M，128 個窗口，括號裡是 U-Delta-rawFFN+λlr1000 42M。

- **逐 token**：全部 -0.045（+0.005）；照抄前文就猜得到的 -0.038（-0.005）；窗口裡見過的 -0.065（-0.008）；沒出現過的新 token -0.031（+0.020）；其中位置 64–256 的新 token -0.021（+0.025）；256–1,024 的新 token -0.033（+0.022）。贏的 52.2% 來自見過的 token。見[逐 token 差距](docs/experiments/token-gap.md)。
- **遠處前文值多少**（把前文截到最近 65–128 個 token，NLL 升多少；Transformer 對 UD2-gate-norm-q32-rope）：全部 +0.146 對 +0.150；照抄 +0.499 對 +0.507；見過 +0.087 對 +0.100；新 token +0.014 對 +0.007。重複的內容，它從遠處拿到的比 Transformer 多一點；新 token 則是 Transformer 從遠處拿得多，但截掉遠處之後它仍然贏（4.665 對 4.702）。15M 時遠處前文反而讓新 token 變差（截掉遠處反而好 0.006），42M 變成有幫助（截掉遠處變差 0.007）。見[遠處前文值多少](docs/experiments/reach.md)。
- **讀到 20K**（16K–20K 段）：3.087；滑動窗口的 Transformer 3.142，U-Delta-rawFFN+λlr1000 3.142，完整上下文的 Transformer 6.251。包括 Transformer 在內，所有模型在 2K–8K 都先升一點再降回來，所以那個起伏來自文本，不是模型。見[外推](docs/experiments/extrapolation.md)。
- **每個格子記多久**（按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量，第 0 → 15 層）：3% / 22% / 19% / 0% / 0% / 1% / 1% / 1% / 0% / 1% / 24% / 12% / 34% / 27% / 10% / 8%。只有第 1–2 層、第 10–13 層和最後兩層留得住，其他層約 0；跟 15M 一樣，少數層慢、其他層很快忘。見[每個格子記多久](docs/experiments/slot-retention.md)。
- **讀出**：慢的第 1 層，原始讀出從位置 0–64 的 2.3 長到 256 之後的 4.6，經過 RMSNorm 的輸出一直在 1.0 左右（0.95 → 1.03）：記憶越寫越滿造成的增長被 norm 吸收掉。各類 token 的門值差別不大，多數層相差 0.3 以內。
- **要打折扣**：逐 token 和截前文只用 128 個窗口，都是 seed 1337；這個模型是在 15M 從十個讀出端變體裡挑出來的。

### 主要觀察

- **少數層記得久，其他層很快忘。** 放開遺忘率的學習率，按格子平均的記憶長度大多縮到約 1 個 token，NLL 反而變好，讀到 20K 也越讀越好；但按寫入量加權，U-Delta-rawFFN+λlr1000 仍有 3 層過了 256 個 token 還留 20–44% 的寫入量（42M 有 5 層留 22–50%），U-Delta-λlr1000 也有 4 層留 18–34%。強迫保留慢記憶（U-Delta+slow、U-Delta-diverseλ）讀到 4K 以後變差。見[記憶設計](docs/experiments/memory-design.md)、[每個格子記多久](docs/experiments/slot-retention.md)。
- **U-Delta-rawFFN+λlr1000 42M 輸在新 token 的差距不是遠距記憶。** 把前文截到最近 65–128 個 token，差距還有 +0.016（完整 +0.022），約七成在最近 128 個 token 內就有；記憶模型的新 token 甚至被舊內容干擾（截掉遠處反而變好），讀出端的修改（門、RMSNorm、讀位址修正、旋轉相對位置）整體都更好，干擾卻沒變少。見[遠處前文值多少](docs/experiments/reach.md)。
- **記憶值 0.58，讀前文全靠它。** 把記憶拿掉、從頭訓練的 [U-Delta-rawFFN-nomem](docs/variants/u-delta-rawffn-nomem.md) 是 4.388（U-Delta-rawFFN 3.812），每個位置都一樣，前文完全沒有幫助。推理時關掉記憶 NLL 升 1.5–2.0（高估了作用，模型沒學過沒有記憶的情況）；U-Delta-rawFFN+λlr1000 按格子平均大部分層只記 1–2 個 token，卻比 U-Delta-rawFFN 更依賴記憶。見[拿掉記憶](docs/experiments/memory-ablation.md)。
- **讀出端是目前最有效的方向。** 每層一個固定的讀出縮放沒有用（[UD2-scale](docs/variants/ud2-scale.md) 3.776，跟 UD2 一樣）；每個 token 的門有用（3 個 seed 都好）；讀出先 RMSNorm 讓後面的位置變好（[UD2-gate-norm](docs/variants/ud2-gate-norm.md) 3.739，512–1,024 好 0.030、前 64 個 token 打平）；讀位址加秩 32 的修正補最多的是位置 64–256 的新 token（[UD2-gate-q32](docs/variants/ud2-gate-q32.md) 3.746），加到秩 128（格子跟著變少）反而變差（[UD2-gate-q128](docs/variants/ud2-gate-q128.md) 3.760）；記憶加上旋轉相對位置再好 0.011，進步在新 token（[UD2-gate-norm-q32-rope](docs/variants/ud2-gate-norm-q32-rope.md) 3.716）；讀出拆成多個 head、除以記憶量都沒用。見[讀出端](docs/experiments/read-side.md)。
- **照抄的能力：短的、遠的抄得比 Transformer 好，長的抄得差；42M 的旋轉相對位置反而讓照抄變差。** 同一段內容出現第二次時的 NLL（64 條序列，seed 1337）：隨機 token 32 個重複一次，42M UD2-gate-norm-q32-rope 2.04、Transformer 3.35（第一次出現約 12.4）；隨機 512 個則是 6.98 對 5.61；一段 128 個 token 的文字隔 768 個 token 再出現，3.20 對 3.36。512 個 token 的原文完全重複，42M 也只從約 3.2 降到約 2.5，所有模型都抄得不好；15M 的 Transformer 完全抄不了隨機 token（12.7 → 11.2）。拿掉旋轉相對位置的 UD2-gate-norm-q32 42M 每一項都抄得更好（隨機 128 個 2.51 對 3.31，原文 512 個 2.391 對 2.535），跟它多兩層留住長記憶一致；整體 NLL 卻差 0.002，旋轉相對位置的好處在別處。各 1 個 seed。腳本 `scripts/fineweb/copyprobe.py`，結果 `scripts/reports/fineweb/copyprobe.json`。
- **旋轉相對位置的替代，和 embedding 寫入值（15M，各 1 個 seed）。** [UD2-gate-norm-q32-ropelo-ev](docs/variants/ud2-gate-norm-q32-ropelo-ev.md) 3.703；[UD2-gate-norm-q32-rope-ev](docs/variants/ud2-gate-norm-q32-rope-ev.md) 3.707；[UD2-gate-norm-q32-ropelo](docs/variants/ud2-gate-norm-q32-ropelo.md) 3.711；[UD2-gate-norm-q32-decay-ev](docs/variants/ud2-gate-norm-q32-decay-ev.md) 3.712；[UD2-gate-norm-q32-rope](docs/variants/ud2-gate-norm-q32-rope.md) 3.716；[UD2-gate-norm-q32-ev](docs/variants/ud2-gate-norm-q32-ev.md) 3.718；[UD2-gate-norm-q32-decay](docs/variants/ud2-gate-norm-q32-decay.md) 3.721；[UD2-gate-norm-q32-decay-half](docs/variants/ud2-gate-norm-q32-decay-half.md) 3.722；[UD2-gate-norm-q32](docs/variants/ud2-gate-norm-q32.md) 3.727；[UD2-gate-norm-q32-conv8](docs/variants/ud2-gate-norm-q32-conv8.md) 3.728（都是 seed 1337）。旋轉相對位置在 15M 幫的是新 token，卻讓照抄變差；只放下半的 block（ropelo）把照抄補回來。記憶寫入值加 token embedding（ev）在有沒有旋轉相對位置的基底上都好約 0.009，幫的是照抄、見過的 token 和少見的詞，對新 token 沒用；ropelo-ev 兩樣都留住，是 15M 最好的。不用旋轉相對位置、把衰減速率固定成 2–8,192 個 token 的光譜（decay）照抄最好，補回旋轉相對位置約一半的進步，加上 ev 完全疊加：decay 補新 token、ev 補見過的 token，decay-ev 是 15M 最會照抄的（隨機 token 512 個重複一次 6.85，ropelo-ev 7.85），也是唯一三類 token 都比基底好的不加旋轉相對位置模型。一半格子改回可學速率（decay-half）沒用：可學的那一半又縮成短記憶。卷積加寬到 8 沒用。42M 時旋轉相對位置放哪裡都一樣（3.170 / 3.171 / 3.172），不加的照抄最好，所以 ropelo-ev 在 15M 的領先到 42M 不一定還在。42M 的 decay-ev 是 3.167（1 個 seed），比不加的好 0.004、比 rope 同 seed 好 0.003，在 seed 差距內；15M 的進步縮到約四分之一。它每一項照抄都最好，是 42M 第一個在隨機 token 512 個重複上贏 Transformer 的統一 block（5.06 對 5.61），但新 token 比不加的差 0.012，大多在位置 256–1,024。腳本 `scripts/fineweb/copyprobe.py`，結果 `copyprobe.json`、`tokgap.json`。
- **42M 跟其他架構比：統一 block 跟 Qwen 式混合打平，遠贏純 GDN；照抄和記憶檢索統一 block 最好。** 驗證 NLL（長度 1,024，各 1 個 seed）：[Qwen 式混合](docs/variants/qhybrid.md) 3.166、[UD2-gate-norm-q32-decay-ev](docs/variants/ud2-gate-norm-q32-decay-ev.md) 3.167、UD2-gate-norm-q32-rope 3.170、UD2-gate-norm-q32 3.172、Transformer 3.214、[GDN](docs/variants/gdn.md) 3.229。Qwen 式混合贏在開頭（前 64 個 token 比 UD2-gate-norm-q32 好 0.027），decay-ev 到 512–1,024 反而好 0.004；長度 4,096 時 Qwen 式混合的兩層 attention 過了訓練長度變差（3.164，decay-ev 3.117）。逐 token（減掉 Transformer 42M；GDN／Qwen 式混合／decay-ev）：全部 +0.021／−0.048／−0.045，照抄 +0.243／−0.024／−0.056，見過的 −0.037／−0.048／−0.068，新 token −0.033／−0.058／−0.021，最少見的詞 +0.187／−0.053／−0.118：Qwen 式混合贏新 token，decay-ev 贏照抄、見過的和少見的詞。照抄測試（隨機 token 512 個重複一次，第二次的 NLL）：GDN 11.25、Qwen 式混合 8.15、Transformer 5.61、decay-ev 5.06。記憶檢索（MQAR 格式，1,024 個 token 內放 N 組 key-value 再問，top-1 正確率，N = 8／64／256）：Transformer 0.31／0.28／0.11，UD2-gate-norm-q32 0.54／0.44／0.09，UD2-gate-norm-q32-rope 0.49／0.32／0.05，decay-ev 0.67／0.51／0.13，GDN 0.06／0.01／0.00，Qwen 式混合 0.32／0.25／0.03。腳本 `scripts/fineweb/kvrecall.py`、`copyprobe.py`，結果 `kvrecall.json`、`copyprobe.json`、`tokgap.json`。
- **單一記憶就夠，不需要第二份慢記憶。** U-Delta+slow 的慢記憶沒學（速率停在初始值）；同容量的兩份快記憶跟一份一樣；單一記憶讓中間層忘快一點就追平它。
- **內容主要是被遺忘掉的，不是被新寫入覆蓋掉的。** 絕大多數格子覆寫占比不到 1%，遺忘門讓實際遺忘比基礎速率快約 10–35 倍；只有少數常被寫的格子覆寫較重（U-Delta 第 6–9 層按層平均 13–22%）。見[遺忘還是覆寫](docs/experiments/forgetting-vs-overwrite.md)。
- **記憶偏向寫意外的 token，但很弱。** 寫入量和 token loss 的 Spearman 相關只有 0.14–0.21；模型自己 greedy 生成（幾乎都是預測到的）時寫入量還有真實文字的 0.52–0.74 倍。見[寫入量與 loss](docs/experiments/write-vs-loss.md)。
- **每一層自己決定記多久。** 從同一個起點出發，訓練完第 0 層只記幾個 token，第 1–4、11 層記幾百個。
- **U-Delta 贏 GDN 不是因為卷積、狀態大小或寫入規則**，而是以 FFN 為錨的格子；統一 block 原本弱在開頭 64 個 token，來自 FFN 讀歸一化位址，U-Delta-rawFFN 補平了。見 [U-Delta 為什麼贏 GDN](docs/experiments/vs-gdn.md)。
- **純記憶題上只有 U-Delta 系列外推**：訓練時間隔最多 256，間隔 4,096 時 U-Delta 仍有 99.95%，Transformer 和原版 0%。見[記憶任務](docs/experiments/memory-tasks.md)。

### 變體

每個變體一份文件：改了什麼、公式、程式碼位置、結果、重現命令。15M 驗證 NLL（長度 1,024）。

**主線**（統一 block，不含 attention）

| 變體 | 改了什麼 | 15M |
| --- | --- | ---: |
| [U-Avg](docs/variants/u-avg.md) | 原版 UnifiedLM，也就是本倉庫 `model.py` 的標準模型：同一組格子既當 FFN 又當上下文記憶，格子裡存“先驗 + 上下文證據”的後驗均值。 | 4.112（3 個 seed，4.107–4.116） |
| [U-Delta](docs/variants/u-delta.md) | 把格子裡的“後驗均值”換成以 FFN 為錨的 delta rule 記憶：記憶只存 FFN 沒預測到的部分，同一個位址的新值按比例覆蓋舊值。 | 3.898（3 個 seed，3.895–3.901） |
| [U-Delta-rawFFN](docs/variants/u-delta-rawffn.md) | U-Delta 的 FFN 那一半改讀未歸一化的 relu²（就是標準的 ReLU² FFN），記憶那一半不變。 | 3.812 |
| [U-Delta-rawFFN+λlr1000](docs/variants/u-delta-rawffn-lambda-lr1000.md) | U-Delta-rawFFN 再加上 U-Delta-λlr1000：每格基礎遺忘率學快 1,000 倍。不加參數。 | 3.796（3 個 seed，3.793–3.799） |
| [UD2](docs/variants/ud2.md) | U-Delta-rawFFN+λlr1000 的寫入目標改錨在 FFN 實際給出的 μz（原本是歸一化的 μr）。不加參數。之後的變體都是 UD2 加後綴，後綴表示讀出端的修改。 | 3.778 |
| [UD2-gate](docs/variants/ud2-gate.md) | UD2 的記憶讀出乘上每個 token 一個門 o = 2·sigmoid(w·h + b)（0 到 2，起點約 1）。每層多 289 個參數。3 個 seed 都比 UD2 好。 | 3.765（3 個 seed，3.761–3.767） |
| [UD2-scale](docs/variants/ud2-scale.md) | UD2-gate 的對照：每個 token 的門換成每層一個可學的純量 2·sigmoid(b)。每層多 1 個參數。 | 3.776 |
| [UD2-gate-norm](docs/variants/ud2-gate-norm.md) | UD2-gate 的讀出先 RMSNorm 再過門（GDN 的做法）。每層多 288 個參數。 | 3.739 |
| [UD2-gate-q32](docs/variants/ud2-gate-q32.md) | UD2-gate 的讀位址加一個秩 32 的修正，讓讀和寫的比對不對稱；hidden_dim 降到 662 對齊參數。 | 3.746 |
| [UD2-gate-norm-q32](docs/variants/ud2-gate-norm-q32.md) | UD2-gate-norm 和 UD2-gate-q32 合在一起：讀出先 RMSNorm，讀位址加秩 32 的修正。 | 3.727 |
| [UD2-gate-q128](docs/variants/ud2-gate-q128.md) | UD2-gate-q32 的讀位址修正從秩 32 加到 128；hidden_dim 降到 538 對齊參數（格子少 19%）。 | 3.760 |
| [UD2-gate-norm-q32-rope](docs/variants/ud2-gate-norm-q32-rope.md) | UD2-gate-norm-q32 的一半格子兩兩成對旋轉（週期 4–256 個 token），讓記憶的比對看相對距離；不加參數。 | 3.722（3 個 seed，3.716–3.727） |
| [UD2-gate-norm-q128-rope](docs/variants/ud2-gate-norm-q128-rope.md) | 三個讀出端的修改一起：讀出 RMSNorm、秩 128 的讀位址修正、一半格子加旋轉相對位置；hidden_dim 538。 | 3.732 |
| [UD2-gate-norm-h4q8](docs/variants/ud2-gate-norm-h4q8.md) | UD2-gate-norm-q32 的讀出拆成 4 個 head，每個 head 有自己秩 8 的讀位址修正、各自 RMSNorm；參數一樣。 | 3.732 |
| [UD2-gate-mass-q32](docs/variants/ud2-gate-mass-q32.md) | UD2-gate-q32 的讀出除以位址碰到的記憶量（像 softmax 的分母）。 | 3.749 |
| [UD2-gate-norm-q32-ropelo](docs/variants/ud2-gate-norm-q32-ropelo.md) | UD2-gate-norm-q32-rope 只在下半的 block（15M 12 個裡的 6 個）加旋轉相對位置，上半不加；參數一樣。15M 比全部加好 0.006（1 個 seed），進步在遠處照抄；42M 跟全部加、完全不加一樣（3.171）。 | 3.711 |
| [UD2-gate-norm-q32-rope-ev](docs/variants/ud2-gate-norm-q32-rope-ev.md) | UD2-gate-norm-q32-rope 的記憶寫入值加上 token embedding：v = W_v(h + g ⊙ e)，e 是 RMSNorm 過的 embedding，g 每個 channel 一個、從 0 開始；多 3,456 個參數。15M 好 0.009（1 個 seed），進步在照抄、見過的 token 和少見的詞。 | 3.707 |
| [UD2-gate-norm-q32-ev](docs/variants/ud2-gate-norm-q32-ev.md) | 不加旋轉相對位置的 UD2-gate-norm-q32 加上 embedding 寫入值（同 UD2-gate-norm-q32-rope-ev）。15M 好 0.0095（1 個 seed），跟加在有旋轉相對位置的版本上一樣。 | 3.718 |
| [UD2-gate-norm-q32-ropelo-ev](docs/variants/ud2-gate-norm-q32-ropelo-ev.md) | 旋轉相對位置只放下半的 block，每層都加 embedding 寫入值；兩個進步約八成六疊加。 | 3.703 |
| [UD2-gate-norm-q32-decay](docs/variants/ud2-gate-norm-q32-decay.md) | 不加旋轉相對位置；每個格子的衰減速率固定不學（buffer），g=1 時半衰期從 2 到 8,192 個 token 等比分佈，門改成 2σ（最多加速 2 倍）。哪些格子還記得就編碼了過了多久，又不會把內容轉掉。參數少 7,944 個。15M 好 0.006（1 個 seed），照抄是所有版本中最好的。 | 3.721 |
| [UD2-gate-norm-q32-decay-ev](docs/variants/ud2-gate-norm-q32-decay-ev.md) | UD2-gate-norm-q32-decay 加 embedding 寫入值；兩個進步完全疊加。每層同一種 block、不用旋轉相對位置、沒有 ×1000 學習率倍數的版本中 15M 最好（1 個 seed）。 | 3.712 |
| [UD2-gate-norm-q32-decay-half](docs/variants/ud2-gate-norm-q32-decay-half.md) | 每個 block 一半格子用 decay 的固定半衰期、一半照原本可學（×1000、softplus 門）。15M 跟 decay 一樣（1 個 seed）：全部固定就夠。 | 3.722 |
| [UD2-gate-norm-q32-conv8](docs/variants/ud2-gate-norm-q32-conv8.md) | 不加旋轉相對位置，卷積從看 4 個 token 改成 8 個；多 13,824 個參數。15M 跟不改一樣（1 個 seed）：旋轉相對位置補的不是 4–8 個 token 內的詞序。 | 3.728 |

**記憶設計**

| 變體 | 改了什麼 | 15M |
| --- | --- | ---: |
| [U-Delta-rawFFN+λlr1000+carry](docs/variants/u-delta-rawffn-lambda-lr1000-carry.md) | U-Delta-rawFFN+λlr1000 訓練時記憶跨窗口接續：每個窗口從同一串上一個窗口結束的記憶開始（梯度不跨窗口），評測照舊從空記憶開始。 | 3.815 |
| [U-Delta-λlr10](docs/variants/u-delta-lambda-lr10.md) | U-Delta 的每格基礎遺忘率 `log_rate` 有效學習率 ×10（重參數化，起點不變）。 | 3.887 |
| [U-Delta-λlr100](docs/variants/u-delta-lambda-lr100.md) | U-Delta 的 `log_rate` 有效學習率 ×100。 | 3.851 |
| [U-Delta-λlr1000](docs/variants/u-delta-lambda-lr1000.md) | U-Delta 的 `log_rate` 有效學習率 ×1000：按格子平均，幾乎每層都把記憶縮到上限（約 1 個 token）；按寫入量加權，第 1–3、11 層仍記得過 256 個 token。 | 3.843 |
| [U-Delta-spreadλ](docs/variants/u-delta-spread-lambda.md) | U-Delta 的遺忘率初始值分散在 1/(L/4)～1/(16L)：同一份記憶裡有快有慢。 | 3.890 |
| [U-Delta-diverseλ](docs/variants/u-delta-diverse-lambda.md) | U-Delta 的基礎記憶長度從 4 到 65,536 個 token 分散，門最多只能讓遺忘快 2 倍，所以慢格子留得住。 | 3.877 |
| [U-Delta-docreset](docs/variants/u-delta-docreset.md) | U-Delta 遇到文件開頭（BOS）就清空記憶。 | 3.890 |
| [U-Delta+slow](docs/variants/u-delta-plus-slow.md) | U-Delta 再加一份慢記憶（每格固定速率，起點 1/(16L)）和讀出混合 ρ。 | 3.888 |
| [U-Delta+fast2](docs/variants/u-delta-plus-fast2.md) | U-Delta+slow 的同容量對照：第二份記憶也是普通的快記憶，狀態大小相同。 | 3.897 |

**位址尖銳度**

| 變體 | 改了什麼 | 15M |
| --- | --- | ---: |
| [U-Delta-learnP](docs/variants/u-delta-learnp.md) | U-Delta 每層的位址次方 p 可學（從 2 開始）。 | 3.892 |
| [U-Delta-ReLU4](docs/variants/u-delta-relu4.md) | U-Delta 前 4 個 block 用 ReLU 位址（p = 1），上面照舊 relu²。 | 3.894 |

**遺忘門**

| 變體 | 改了什麼 | 15M |
| --- | --- | ---: |
| [U-Gate](docs/variants/u-gate.md) | U-Avg 加上按內容決定的遺忘門：每個格子、每個 token 的遺忘率由位址算出。 | 3.998（3 個 seed，3.988–4.003） |
| [U-Gate-fastinit](docs/variants/u-gate-fastinit.md) | U-Gate 的對照：U-Avg 只把遺忘率起點從 1/(4L) 調快到 1/64（U-Gate 的門最後停的位置附近），仍是每層一個、可學。 | 4.051（3 個 seed，4.049–4.052） |
| [U-Gate-nogate](docs/variants/u-gate-nogate.md) | U-Gate 的消融：每格各自的遺忘率，沒有門，也沒有寫入時的遺忘項。 | 4.124 |
| [U-Gate-rank64](docs/variants/u-gate-rank64.md) | U-Gate 的門低秩從 16 加到 64。 | 3.977 |
| [U-Gate-fromh](docs/variants/u-gate-fromh.md) | U-Gate 的門改從 block 輸入 h 讀（原本從位址 r 讀）。 | 4.012 |
| [U-Gate-silu](docs/variants/u-gate-silu.md) | U-Gate 的門中間加一層 SiLU（兩層 MLP 的門）。 | 4.002 |

**統一的對照**

| 變體 | 改了什麼 | 15M |
| --- | --- | ---: |
| [Split](docs/variants/split.md) | 統一的對照：靜態格子和上下文格子用各自的位址，兩組格子不共享。 | 4.038（3 個 seed，4.034–4.040） |

**基線**

| 變體 | 改了什麼 | 15M |
| --- | --- | ---: |
| [U-Delta-rawFFN-nomem](docs/variants/u-delta-rawffn-nomem.md) | 沒有記憶的對照：U-Delta-rawFFN 把記憶那一半整個拿掉、從頭訓練，只剩 FFN 和位址前的短卷積。 | 4.388 |
| [Transformer](docs/variants/transformer.md) | llama2.c 式 Transformer 基線：RoPE attention + SwiGLU FFN，參數量跟統一 block 對齊。 | 3.945（3 個 seed，3.928–3.957） |
| [GDN](docs/variants/gdn.md) | 線性遞歸（SSM 類）基線：Transformer 的 attention 換成 Gated DeltaNet（6 個 head × 48）。 | 3.964 |
| [GDN-bigstate](docs/variants/gdn-bigstate.md) | 我們自己的 GDN 變體：2 個大 head（各 144 維），狀態約 3 倍，衰減用平滑上下限。 | 3.964 |
| [GDN-2](docs/variants/gdn-2.md) | NVIDIA Gated DeltaNet-2：抹除門和寫入門分開，衰減按通道。 | 3.970 |
| [Mesa](docs/variants/mesa.md) | Mesa layer：每個 token 精確解一次加權最小平方（GDN 和 U-Delta 的 delta rule 每步只往這個解走一步）。 | 3.969 |

**含 attention 的對照**（只當參考，不算統一 block）

| 變體 | 改了什麼 | 15M |
| --- | --- | ---: |
| [U-Delta+attn](docs/variants/u-delta-plus-attn.md) | 對照組，含 attention，不算統一 block：每 4 個 U-Delta block 後面插一個窗口 128 的局部 attention 半層。 | 3.840 |
| [Transformer+conv](docs/variants/transformer-conv.md) | Transformer 在 attention 前加上跟 U-Delta 一樣的寬 4 因果卷積：看 U-Delta 的優勢是不是卷積帶來的。 | 3.921 |
| [Transformer+SSMax](docs/variants/transformer-ssmax.md) | Transformer 的 attention 換成 Scalable-Softmax（logits 乘 s·log n），想讓它讀長文時不變鈍。 | 3.913 |

### 實驗

每個實驗一份文件：問題、怎麼做的、結果表、結論、注意事項。

| 實驗 | 問題 | 簡短回答 |
| --- | --- | --- |
| [規模：15M → 42M → 110M](docs/experiments/scale.md) | 15M 贏 Transformer 會不會只是小模型的現象？ | 15M 贏；42M 打平到略輸：最好的 U-Delta-rawFFN+λlr1000 輸 0.009（1 個 seed）。 |
| [讀出端：門、縮放、RMSNorm、讀位址修正、旋轉相對位置](docs/experiments/read-side.md) | 記憶的讀出怎麼改？ | 讀出門（3 個 seed）、讀出 RMSNorm、讀位址修正、記憶裡的旋轉相對位置都有用，合起來 3.716；每層固定縮放、秩 128、多 head、除以記憶量都沒用。 |
| [逐 token 差距：輸在哪些 token](docs/experiments/token-gap.md) | 統一 block 跟 Transformer 的差距落在哪些 token 上？ | 15M 贏的近九成是照抄前文；42M 輸的全是窗口裡沒出現過的新 token。 |
| [遠處的前文值多少](docs/experiments/reach.md) | 把前文截到最近 128 或 512 個 token，會差多少？42M 輸在新 token 是因為遠距記憶嗎？ | 不是：約七成的差距在最近 128 個 token 內就有；遠處前文主要用來照抄，記憶模型的新 token 還會被舊內容干擾，讀出端的修改都沒擋掉。 |
| [多 seed 的穩定性](docs/experiments/seeds.md) | 差距是不是 seed 的噪聲？ | 統一 block 的 seed 標準差約 0.003，Transformer 約 0.016；最新的統一 block 三個 seed 都贏。 |
| [統一的對照：共享位址 vs 分開位址](docs/experiments/unification.md) | 同一組格子同時當 FFN 和記憶，會不會比分開差？ | 原版上分開的好 0.074，但多 16% 參數；U-Delta 上還沒做。 |
| [記憶的寫法：平均、遺忘門、delta rule](docs/experiments/write-rule.md) | 格子裡的記憶該怎麼寫？ | delta rule 遠好於平均：4.112 → 3.998 → 3.898。 |
| [FFN 那一半讀哪個位址](docs/experiments/ffn-address.md) | U-Delta 到 42M 為什麼輸 Transformer？ | FFN 讀歸一化位址是瓶頸；改讀 relu² 後 15M 好 0.086、42M 好 0.122。 |
| [記憶設計：快記憶、慢記憶、遺忘率](docs/experiments/memory-design.md) | 需要慢記憶嗎？遺忘率怎麼設？ | 不需要慢記憶；記憶越短越好，短到上限還在變好。 |
| [拿掉記憶：記憶值多少、模型多依賴記憶](docs/experiments/memory-ablation.md) | 沒有記憶會差多少？推理時關掉記憶會怎樣？哪一層最重要？ | 記憶真正值 0.58（沒有記憶、從頭訓練的版本 4.388）；推理時關掉記憶 NLL 升 1.5–2.0，按位置變平：讀前文全靠記憶。 |
| [每個格子記多久（按寫入量加權）](docs/experiments/slot-retention.md) | 遺忘門的“有效記憶長度”只有 1–2 個 token，記憶真的這麼短嗎？ | 不是：按寫入量加權，U-Delta-rawFFN+λlr1000 有 3 層（42M 有 5 層）過了 256 個 token 還留 20–50%；U-Delta-λlr1000 也有 4 層。 |
| [內容是被遺忘還是被覆寫](docs/experiments/forgetting-vs-overwrite.md) | 格子丟掉內容，是衰減還是被新寫入覆蓋？ | 典型格子是衰減（覆寫不到 1%）；少數常被寫的格子覆寫較重。 |
| [寫入量與 loss](docs/experiments/write-vs-loss.md) | 記憶是不是只寫沒預測到的 token？ | 只是偏向，相關 0.14–0.21；自己預測到的也照寫。 |
| [位址尖銳度](docs/experiments/sharpness.md) | 位址更尖（更大的次方）會不會更好？ | 不會：可學的次方停在 2.0–2.35，NLL 不變。 |
| [U-Delta 為什麼贏 GDN](docs/experiments/vs-gdn.md) | 同樣是 delta rule，差在哪？ | 不是卷積、狀態大小或寫入規則，是以 FFN 為錨的格子。 |
| [長上下文與外推（讀到 20K）](docs/experiments/extrapolation.md) | 過了訓練長度會不會崩？ | 統一 block 都不崩；Transformer 只有開滑動窗口才不崩。 |
| [記憶任務 m2kv](docs/experiments/memory-tasks.md) | 純記憶題能記多遠？ | 只有 U-Delta 系列外推：間隔 4,096 仍有 99.95%。 |
| [含 attention 的對照](docs/experiments/attention-control.md) | 加回局部 attention 能好多少？ | 好 0.058，但不含 attention 的最新版本已經贏它。 |

實驗程式碼在 `scripts/`（變體註冊表 `scripts/variants.py`，U-Delta 系列在 `scripts/delta.py`），結果資料在 `scripts/reports/fineweb/`。依照 `AGENTS.md`，變體在贏之前不進 `model.py`，所以 `model.py` 仍然是 U-Avg。

## 模型

每个 `UnifiedBlock` 对 RMSNorm 后的隐状态 `h_t` 做宽度 4 的因果深度卷积得到 `u_t`，用 relu² 产生对格子的寻址并归一化（分母里有一个可学习的空槽质量 `eps`）：

```text
r_t = relu(K u_t + b)^2 / (eps + sum_a relu(K u_t + b)_a^2)
```

当前位置的值 `v_t = W_v h_t` 写到前一位置的地址 `r_{t-1}` 上，然后用当前地址读出：

```text
S_t = lam * S_{t-1} + v_t r_{t-1}^T          n_t = lam * n_{t-1} + r_{t-1}
m_t = (w * mu + S_t) / (w + n_t)              # 每个格子的后验均值（逐格）
y_t = m_t r_t,   x_t <- x_t + y_t
```

`mu`（慢内容）、每格先验质量 `w`、遗忘率 `lam` 都是学出来的。

快记忆这一项是一个以格子为 key 维、以 `dim` 为 value 维、带一个标量衰减的因果线性注意力。训练时它由 `kernel.py` 里的 Triton kernel 计算（分块：块内二次、块间传递状态，块长 64；前向和反向都在 kernel 里，matmul 用 bf16、统计量用 fp32），在 15M–110M 各档比 `torch.compile` 后的纯 PyTorch 实现快 1.4–2.1 倍、显存更省；设计和与 fla 的对比见 [`docs/kernel.md`](docs/kernel.md)。推理用逐 token 的递推形式（`forward_recurrent`，与 `run.c` 完全相同）。训练和 `sample.py` 需要 CUDA；`run.c` 在 CPU 上运行。

## 环境

本项目使用 [uv](https://docs.astral.sh/uv/)（Python 3.12，PyTorch 2.10.0 + CUDA 12.8，版本锁在 `uv.lock`）：

```bash
uv sync
source .venv/bin/activate
```

## 数据

TinyStories，与 llama2.c 相同的流程：每篇故事加 BOS、不加 EOS，存成 uint16 的 `.bin`；按文件名排序后第一个 shard 是验证集。

```bash
python tinystories.py download        # data/TinyStories_all_data/*.json
python tinystories.py pretokenize     # Llama 2 32K 词表（15M 及以上）

# 512 词表（260K）：用仓库里的 tok512.model（llama2.c 官方 260K 模型的词表）
cp tok512.model data/
python tinystories.py pretokenize --vocab_size=512
```

更大的语料用 `fineweb.py`（FineWeb-Edu sample-10BT）或 `climbmix.py`（ClimbMix），接口相同，最后一个 shard 是验证集；换数据集就改 `train.py` 顶部 `from tinystories import Task` 这一行，见 [`docs/models.md`](docs/models.md#数据集)。

```bash
python fineweb.py download            # HuggingFace 缓存，约 27 GB parquet
python fineweb.py pretokenize         # data/FineWebEdu_10BT/*.bin，Llama 2 32K 词表
```

## 训练

```bash
python train.py                       # 默认 15M，单卡
torchrun --standalone --nproc_per_node=4 train.py
```

其他规模（260K / 42M / 110M）的完整命令在 [`docs/models.md`](docs/models.md)。所有超参都可以用 `--key=value` 覆盖。

## 采样与 C 推理

```bash
python sample.py --checkpoint=outputs/15M/best.pt

make run
python export.py --checkpoint outputs/15M/best.pt --output outputs/15M/model.bin
./run outputs/15M/model.bin -t 0.8 -p 0.9 -n 256 -i "Once upon a time"
./run outputs/260K/model.bin -z tok512.bin -t 0.8 -p 0.9 -i "Once upon a time"   # 512 词表
```

`train.py` 在验证集变好时会自动导出 `model.bin`。`run` 的参数与 llama2.c 相同：`-z` tokenizer，`-t` 温度，`-p` top-p，`-s` 随机种子，`-n` 步数，`-i` 提示词。

## 测试

```bash
make test        # 编译 ./run，再运行 python test_all.py
```

`test_all.py` 检查：纯 PyTorch 参考实现的分块形式与二次形式一致、并行形式与逐 token 递推一致、因果性、梯度有限（double 精度，误差 < 1e-8）；Triton kernel 的 loss 与全部参数梯度对 double 参考的相对误差 < 1e-2（需要 CUDA）；以及 `./run` 的贪心生成与 Python 参考逐字相同。

## 评测

```bash
python eval/core_eval.py --checkpoint outputs/15M/best.pt
```

[CORE](https://arxiv.org/abs/2406.11717)（DCLM 论文指标），移植自 gdn.c；首次运行会下载 `data/eval_bundle/`。

## 结果

**260K，TinyStories，512 词表**（100K 次更新，6.55B tokens，单张 RTX 3090 Ti，约 70 分钟）。验证 NLL 在同一组 1,048,576 个验证 token 上用 FP32 计算：

| 模型 | 参数量 | 验证 NLL |
| --- | ---: | ---: |
| UnifiedLM | 259,012 | **1.2905** |
| llama2.c 官方 `stories260K.pt` | 260,032 | 1.2965 |

`run.c` 生成的样本（`-t 0.8 -p 0.9 -s 2`）：

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big box in the sunshine. She asked her mommy, "What's wrong, Lily?" Her mommy said, "I lost my toy ice cream!"
> Lily was sad and wanted to help her mommy. She asked her mommy if they could borrow it. They both smiled and said, "Thank you for my favorite toy is an easy secret." Lily was happy to help her mommy, and they all had a good friend.

15M、42M、110M 在 FineWeb-Edu 上的结果见上面的「当前进展」和 [`docs/experiments.md`](docs/experiments.md)。

## 试过的其他做法

和现在的标准配方（快写入 + relu² 地址 + 先验 4）做过对照的：

- **去掉快写入**（每个块只剩对静态格子 `mu` 的归一化 relu² 读出，没有上下文写入和后验分母）：260K、327.68M tokens 的短预算下，等参数量另训的模型验证 NLL 为 1.664，标准配方为 1.511。在训练好的 260K 模型上推理时直接去掉快写入，验证 NLL 从 1.31 升到 1.78。
- **更弱的先验（prior 0.4）**：15M、24.6M tokens 的短预算下验证 NLL 为 2.057，prior 4 为 1.962。

另外几种变体（固定写入率、hash n-gram 格子、二值 spike / k-WTA 地址、有限 fan-in、共享权重循环）已经实现并通过了一致性检查，但还没有在文本上与标准配方比较。

FineWeb-Edu 上的统一性实验（遗忘门、delta rule、FFN 地址、遗忘率、快慢记忆、外推、记忆任务，以及 Transformer、GDN、Mesa 等基线）见 [`docs/experiments.md`](docs/experiments.md)。

## 目录

```text
model.py        UnifiedLM（唯一的模型）
kernel.py       快记忆的 Triton 训练 kernel（前向 + 反向）
train.py        训练（默认 15M）
sample.py       PyTorch 采样
export.py       导出 run.c 权重
run.c           纯 C 推理
tinystories.py  TinyStories 下载、分词、加载
fineweb.py      FineWeb-Edu sample-10BT（同接口）
climbmix.py     ClimbMix（同接口，来自 gdn.c）
tokenizer.py    sentencepiece 封装与 .bin 导出
test_all.py     一致性测试
eval/           评测（CORE）
docs/           研究目标、各规模设置、训练 kernel、统一性实验报告
```

`references/`（llama2.c、gdn.c 等上游与 baseline 仓库）、`scripts/`（开发脚本）、`data/`、`outputs/` 都在 `.gitignore` 里。这个分支例外地提交了 `scripts/` 里的实验代码和结果数据（`git add -f`），方便复现报告里的数字。baseline 在 `references/` 下用各自的代码训练和评测，例如：

```bash
git clone https://github.com/karpathy/llama2.c references/llama2.c
```
