# 要不要用 BabyLM 的 tokenizer（2026-11-07 查證）

## 結論

- BabyLM 沒有規定要用哪個 tokenizer。所謂「BabyLM 的 tokenizer」，其實是每一版、每個賽道的官方 baseline
  各自在自己那份語料上練的 byte-level BPE，詞表都是 16384。
- 如果自然文字階段用 BabyLM 的資料，**建議直接用 2026 版對應賽道的官方 tokenizer**
  （10M 字用 `BabyLM-2026-Baseline-GPT2-Strict-Small`，100M 字用 `...-Strict`）：
  - 符合規則：它只在 BabyLM 的語料上練過，不會帶入外部資料。
  - 跟官方 GPT-2 baseline 同一個 tokenizer、同一份資料，每 token 的 loss 可以直接比較。
  - 詞表 16384 比我們現在的 Llama 2 32000 小。15M 那檔（dim 288，embedding 和輸出共用）的詞表參數
    會從 920 萬降到 470 萬，更多參數留給 block。
- 代價：
  - 它是 Hugging Face `tokenizers` 的格式，不是 SentencePiece。要加 `tokenizers` 套件和一個 `babylm.py` 資料模組。
  - `run.c` 和 `tokenizer.bin` 的解碼只支援 SentencePiece 的詞片，不支援 byte-level BPE。這只影響 C 推論，PoC 用不到。
- 另一個做法：用現有的 `train_vocab` 流程，在 BabyLM 文字上自己練一個 SentencePiece。
  - 一樣符合規則，`run.c` 也照常能用，詞表大小可以自己選。
  - 但每 token 的 loss 不能跟官方 baseline 直接比。BLiMP 這類測驗比的是整句的機率，換 tokenizer 也照樣能比。
- 不要用 Llama 2 32K，也不要用 TinyStories 的 tokenizer：
  - 它們在 BabyLM 以外的文字上練過。
  - 規則說「學過語言的元件，它看過的字算進字數預算」，FAQ 也不准用現成的 POS tagger。
  - 預訓練 tokenizer 算不算違規，官方沒有明講，但不值得冒這個險。

## 查到的事實

**規則（2026 版，第 4 屆，EMNLP 2026）：**
- 模型架構隨意，可以試 tokenization。Strict 和 Strict-Small 不一定要用官方語料，只要不超過字數預算。
  字數是用空白切開算的，跟 subword 詞表大小無關。
- 輔助模型看過的訓練資料算進預算。同一份文字可以同時拿來練 tokenizer、parser 和 LM。
- 2026 的投稿已截止（7 月 20 日），workshop 是 10 月。我們只拿它當資料集和評測，不投稿。

**官方 baseline 的 tokenizer**（直接下載 `tokenizer.json` 檢查）：

| baseline | 類型 | 詞表 | 跟 2026 Strict 的詞重疊 |
|---|---|---|---|
| 2026 GPT2-Strict（100M） | byte-level BPE | 16384 | 1.00 |
| 2026 GPT2-Strict-Small（10M） | byte-level BPE | 16384 | 0.85 |
| 2025 baseline-100m-gpt2 | byte-level BPE | 16384 | 0.90 |
| 2025 baseline-10m-gpt2 | byte-level BPE | 16384 | 0.82 |

- 同一年、同一個字數的 GPT-2 和 GPT-BERT 用同一個 tokenizer。不同年、不同字數之間都不一樣。
- 詞表裡有 CHILDES 的說話者標記（例如 `*MOT`），所以 tokenizer 是在 BabyLM 語料上練的。
- 正規化依序是：前面加空白、NFKC、換行處理。特殊 token 有 `<unk> <s> </s> <pad> <mask>`。
- baseline 的模型大小：12 層、寬度 768、位置長度 1024。

**資料（2026，MIT 授權，不用申請）：**
- `BabyLM-community/BabyLM-2026-Strict-Small` 有 54 MB 文字，`BabyLM-2026-Strict` 有 543 MB。
- 兩者都只有 train，由 6 個來源組成：CHILDES、Gutenberg、OpenSubtitles、Simple Wiki、BNC spoken、Switchboard。

**評測：**
- 2026 要求模型能給一句話打分（log-likelihood），以 Hugging Face transformers 的形式提供。
- GPT-BERT baseline 本身就是自訂架構，用 `auto_map` 和 `trust_remote_code` 載入，所以 ULM 包成 HF 模型就能接上。
- Strict 賽道的測驗包括：BLiMP、BLiMP Supplement、EWoK、Entity Tracking、COMPS、GlobalPIQA、閱讀時間，以及 (Super)GLUE 微調。

## 來源

- https://babylm.github.io/ 、https://babylm.github.io/guidelines.html 、https://babylm.github.io/faqs.html
- 2026 call for papers：https://arxiv.org/html/2602.20092（4.1、4.2、4.5 節）
- https://huggingface.co/collections/BabyLM-community/babylm-2026-baselines-69fa0ae445279c6a55e884ed
- https://huggingface.co/BabyLM-community/babylm-baseline-10m-gpt2 （和同系列的 100m、gpt-bert）
- https://github.com/babylm/evaluation-pipeline-2025 、https://github.com/babylm-org/babylm-eval
