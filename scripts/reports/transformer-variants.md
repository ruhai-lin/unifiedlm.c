# Transformer architecture variants (2023–2026): survey for ULM

Scope: attention/block variants. The previous survey's topics (RoPE scaling, ALiBi, sinks via StreamingLLM, FoX itself,
stick-breaking, log-length scaling, QK-norm, doc masking, progressive length, SWA+global, linear RNN/SSM family) are not repeated.
"(v)" = arXiv id confirmed by web search in this session. No "(v)" = from memory, likely right but unchecked.

## Read this first: what the mapping is mapped onto

From `/Users/gary/code/ULM/model.py` on `main` (UnifiedBlock):
- address `r_t = normalize(relu(K u_t + b)^2)` over m slots, `u_t = causal depthwise conv4(RMSNorm(x_t))`; one address shared by all d value dims (effectively **one head**).
- write: key `r_{t-1}`, value `v_t = W_v h_t`; read with `r_t` (so query and key are the **same projection, shifted by one** — a hard-wired bigram/induction lookup; no separate q/k).
- read is a normalised posterior mean: `(w_a mu_a + Σ λ^(t-s) r_{s-1,a} v_s) / (w_a + Σ λ^(t-s) r_{s-1,a})`; `log_eps` is an implicit null slot (= a learned "off-by-one" sink), `w_a` a prior mass.
- **On main, λ is one scalar per layer** (`log_rate`, init 1/(4·max_seq_len)), not per-slot data-dependent. The brief says per-slot data-dependent λ; that presumably lives in a branch/scripts. Mappings below assume either.
- No output gate, no write-strength gate, every token writes mass Σr ≤ 1.

Plausible reasons for the 64–1024 deficit, which the ideas below target: (1) slot collisions — each slot averages every write that hit it within the decay horizon, so a specific token at distance 300 is blurred with others; (2) one address per token = one lookup per layer per token, vs H heads in the Transformer; (3) the value written is a deep hidden state, so exact token identity for copying is weak; (4) every token writes, including uninformative ones.

---

## A. Attention-score / normalisation changes

**1. Differential Transformer** — Ye et al., 2024, ICLR 2025, arXiv 2410.05258 (v).
Attention = softmax(Q1K1) − λ·softmax(Q2K2), per-head GroupNorm. Fixes "attention noise" on irrelevant context. Scaling curves: ~60–65% of params/tokens for equal loss (830M–13B, ctx 2048); better NIAH and in-context retrieval.
- 15M @1K: unclear; smallest reported model is 830M. Halves head count at equal params, which hurts at d=288. Low confidence.
- (a) Baseline: maybe, not a priority; untested at our scale.
- (b) Transfer: **differential read** — two addresses r¹, r² from the same conv'd input; `y = read(r¹) − β·read(r²)`, β learned per layer (init ~0.2). It cancels the common-mode "average of everything written" that a collided slot returns. Diff-Mamba (Schneider et al., 2025, arXiv 2507.06204 (v)) reports that naive transfer to an SSM is not enough; it needed extra normalisation. Medium priority.

**2. Gated Attention (output sigmoid gate)** — Qiu et al. (Qwen), 2025, NeurIPS 2025 oral, arXiv 2505.06708 (v).
`o = σ(W_g x) ⊙ SDPA(q,k,v)`, head-specific/elementwise, before W_o. Adds non-linearity to the low-rank V·O path and query-dependent sparsity. Removes attention sinks (first-token mass 46.7%→4.8%), cuts loss spikes, allows larger LR, helps length extrapolation. 30 variants at 1.7B dense and 15B MoE, 3.5T tokens. Used in Qwen3-Next.
- 15M @1K: likely a small, robust gain (cost: one d×d matrix per layer). No small-scale evidence; mark as expectation.
- (a) Baseline: **yes**, top-3. Cheap, stable, widely adopted.
- (b) Transfer: **direct**. `y = σ(W_g h_t) ⊙ fast_t` (or gate fast and slow read separately). GLA, GDN, Mamba2 and HGRN2 all have this output gate; ULM does not. The gate lets a token reject a collided or irrelevant memory read. That job is done now only by `log_eps` and the prior mass w, which are not query-elementwise. Top-5.

**3. Learned attention sinks (softmax off-by-one)** — gpt-oss model card (OpenAI, 2025, arXiv 2508.10925, not confirmed); Miller's "attention is off by one" (2023 blog).
A per-head learnable logit in the softmax denominator, so heads can attend to "nothing".
- 15M @1K: tiny effect; mostly matters for streaming and quantisation.
- (a) Baseline: optional; overlaps with the output gate. Don't add both.
- (b) Transfer: **ULM already has it** (`log_eps` null slot plus prior mass w). Nothing new; it does show that ULM's normalisation is already sink-aware.

**4. Softpick** — Zuhri, Fuadi, Aji, 2025, arXiv 2504.20966 (v).
ReLU(e^x − 1)/Σ|e^x − 1|: rectified, not sum-to-one. 0% sink rate, no massive activations, better low-bit quantisation (340M, 1.8B). The authors flag under-scaled scores at long context.
- 15M @1K: neutral to slightly negative on loss.
- (a) Baseline: no.
- (b) Transfer: weak. ULM's relu² address is already rectified. The analogue would be dropping the sum-to-one in `code()`, which the null slot already approximates.

**5. Scalable-Softmax (SSMax)** — Nakanishi, 2025, arXiv 2501.19399 (v). Logits × s·log n. Same family as log-length scaling (already covered). Skip.

**6. Selective Attention** — Leviathan, Kalman, Matias (Google), 2024, ICLR 2025, arXiv 2410.02703 (v).
Parameter-free. Head 0's logits, ReLU'd and cumulatively summed, are subtracted from all heads' logits, so token c can "mask" earlier token a for all future tokens. On C4 it matches a Transformer with ~2× heads/params, and needs 16×/25×/47× less attention memory at ctx 512/1K/2K for equal perplexity.
- 15M @1K: **one of the few results measured at small scale and at 512–2K context**, so likely positive. It is a softmax-side change, so it needs a custom FlashAttention mask (cost).
- (a) Baseline: yes, plausibly a real gain at our scale. It is closer to FoX (data-dependent decay inside softmax) than to a free lunch, though. Include it if FoX is in the comparison; otherwise optional.
- (b) Transfer: **concrete, and parameter-free**. Selective attention is a forget gate whose erase signal comes from an existing projection. ULM mapping: per-slot erase from the address logits the block already computes: `λ_{t,a} = λ · exp(−relu(z_{t,a} − τ))`, where z = K u_t (pre-square). A token that strongly addresses slot a clears stale content in that slot before writing: a "replace, don't average" rule. It cuts slot collisions, cause (1), at no parameter cost. If the per-slot λ branch already exists, this suggests deriving it from K rather than from a separate projection. Top-5.

**7. Multi-Token Attention (MTA)** — Golovneva, Wang, Weston, Sukhbaatar (Meta), 2025, COLM 2025, arXiv 2504.00927 (v).
2-D conv over the attention score map (query × key) plus mixing across heads, so a weight can depend on several neighbouring q/k pairs. Better on NIAH and BabiLong and slightly better perplexity (880M scale, I believe).
- 15M @1K: unknown. Breaks FlashAttention (needs materialised scores), so it is expensive.
- (a) Baseline: no (cost and kernel complexity).
- (b) Transfer: partly present. ULM's conv on the address path is the key-side half of MTA (address = f(a window of tokens)). Missing: **conv on the write key separately from the read query**. ULM ties them (read r_t, write r_{t-1}); untying them (a separate conv/projection for the write key) gives "store under context A, retrieve with context B". Medium.

**8. Sigmoid attention** — Ramapuram et al. (Apple), 2024, arXiv 2409.04431 (not confirmed). Elementwise sigmoid with a −log n bias. Equal to softmax with care, no gain. Skip for both.

**9. 2-simplicial / higher-order attention** — Roy et al. (Meta), 2025, arXiv 2507.02754 (not confirmed). Trilinear attention. Gains reported on reasoning at large scale. Too expensive and irrelevant at 15M. Skip.

## B. Positional encoding

**10. CoPE (Contextual Position Encoding)** — Golovneva et al. (Meta), 2024, arXiv 2405.18719 (v).
Position = Σ of sigmoid gates σ(q·k) between the two tokens: count only tokens the model chooses, interpolating position embeddings. Solves selective copy, counting and Flip-Flop; better perplexity on Wikipedia and code (small models, ~100M range I believe).
- 15M @1K: plausible small gain, but O(T²) gate sums and a custom kernel.
- (a) Baseline: no (non-standard, slow).
- (b) Transfer: **content-dependent clock**. In recurrent form the decay advances only when a gate fires: `λ_t = λ^{g_t}`, g_t = σ(w·h_t + b) ∈ (0,1), one scalar per token per layer, shared by all slots. This is Mamba's Δ_t (covered before); the CoPE framing adds only that g_t should count *units* (words, sentences), so the bias init should make most sub-word tokens near-free (g ≈ 0.3). It is the cheapest data-dependent λ if main stays scalar. Medium.

**11. NoPE and partial-NoPE hybrids** — Kazemnejad et al., 2023, arXiv 2305.19466 (NoPE length generalisation, small models); Yang et al. (Cohere), "Rope to Nope and Back Again", 2025, NeurIPS 2025, arXiv 2501.18795 (v) — RNoPE-SWA: local RoPE layers and global NoPE layers beat full RoPE on short and long tasks; Command-A uses 3:1 SWA-RoPE : full-NoPE. Llama 4 "iRoPE" (Meta blog, April 2025, no paper): NoPE every 4th layer with temperature scaling. SmolLM3 (HF blog, 2025, (v)): drops RoPE in every 4th layer, with no short-context loss. DroPE (2025, arXiv 2512.12167 (v)): remove PE after pretraining, for long context.
- 15M @1K: in-context gains are small or neutral; the benefit is extrapolation, which ULM already wins.
- (a) Baseline: a "NoPE every 4th layer" baseline extrapolates better and is free. It is a fairer *long-context* baseline, but does little to the 64–1024 in-context gap we are trying to close. Worth one run if extrapolation curves go in the paper.
- (b) Transfer: none directly (ULM has no PE). The lesson is that position-free global layers plus local positional layers is the winning split; ULM's conv is the "local positional" part.

**12. p-RoPE / partial RoPE** — Barbero et al. (GDM/Oxford), 2024, ICLR 2025, arXiv 2410.06205 (v). Gemma uses high frequencies for positional heads and low frequencies as semantic channels; p-RoPE removes the lowest fraction of frequencies (rotates only part of the dims) and improves 2B-scale perplexity. Partial RoPE (GPT-NeoX/GPT-J rotary_pct=0.25, DeepSeek MLA decoupled RoPE) is a long-standing practice. Canon (below) finds partial RoPE + Canon > full RoPE.
- 15M @1K: small gain is plausible; it costs nothing.
- (a) Baseline: yes, as a free knob (e.g. RoPE on 50% of head dims). Lower priority than the top-3.
- (b) Transfer: none.

**13. PaTH attention** — Yang, Shen, Wen, Tan, Mishra, Ren, Panda, Kim, 2025, NeurIPS 2025, arXiv 2505.16381 (v).
Data-dependent Householder (I − β w wᵀ) transitions accumulated between positions replace RoPE. Solves state-tracking; beats RoPE and FoX at 760M; "PaTH-FoX" combines both (I believe it is in the same paper).
- 15M @1K: likely positive, but needs flash-linear-attention kernels.
- (a) Baseline: an "upgraded Transformer" rather than a fair baseline. Skip unless the comparison targets SOTA attention.
- (b) Transfer: this is the delta rule inside softmax — the GDN family, already covered. Nothing new.

**14. FoX follow-ups** — Adaptive Computation Pruning, Lin et al., 2025, COLM 2025, arXiv 2504.06949 (v). It prunes attention entries the forget gate has decayed: ~70% fewer attention FLOPs, 2–3× attention speed, no loss. An efficiency-only result. It does show that **trained FoX heads mostly have short effective spans**, and only a few heads keep long memory. For ULM this suggests that a learned spread of λ across slots or heads (most fast, a few slow) is what the gate converges to. Initialise λ that way (see the multi-timescale idea, item 19).

## C. Interleaved local-global

**15. Gemma 2 / Gemma 3 / Command A / Llama 4** — Gemma 2 (2024, arXiv 2408.00118): 1:1 local (4K window) : global, logit soft-capping. Gemma 3 (2025, arXiv 2503.19786): 5:1 local (1024 window) : global, QK-norm instead of soft-cap. Command A (Cohere, 2025, report arXiv 2504.00698, not confirmed): 3:1 SWA-RoPE : NoPE global. Llama 4: chunked local attention plus NoPE global.
- 15M @1K: at ctx 1024 a 1K window is full attention, so nothing changes. It only matters with small windows (64–256), which would *weaken* in-context quality.
- (a) Baseline: no for the in-context fight (it lowers quality at equal context); useful as a long-context baseline only.
- (b) Transfer: the ratio lesson (most layers local, a few global) maps to ULM as **layer-wise λ schedules**: most layers short λ (span ~32–128), every 4th–6th layer λ ≈ 1 (span ≥ 4K). At present all layers share one λ init (1/(4·max_seq_len)). Cheap to test: init only. It is part of top-5 item 4.

## D. KV-compressed and sparse attention

**16. Multi-head Latent Attention (MLA)** — DeepSeek-V2, 2024, arXiv 2405.04434. Low-rank joint KV latent (c_KV) with decoupled RoPE dims. ~93% KV-cache cut with quality ≥ MHA at DeepSeek scale.
- 15M @1K: none; it is a KV-memory optimisation, and at small d it may lose a little.
- (a) Baseline: no.
- (b) Transfer: the decoupled-RoPE idea (a small positional sub-space next to a content sub-space) is mirrored by ULM's conv. Otherwise none.

**17. Tensor Product Attention (TPA / T6)** — Zhang et al., 2025, arXiv 2501.06425 (v). Contextual low-rank factorisation of Q/K/V. Reports lower validation loss than MHA/GQA/MLA with ~10× smaller KV cache.
- 15M @1K: unclear; gains are reported mostly at 124M–1.5B (I believe).
- (a) Baseline: no.
- (b) Transfer: weak. The idea is "per-token, contextual rank-r factorisation of heads". The ULM analogue is cheap multi-head addressing: H addresses over m slots built from rank-r factors.

**18. Native Sparse Attention (NSA)** — Yuan et al. (DeepSeek/PKU/UW), 2025, ACL 2025 best paper, arXiv 2502.11089 (v). Three gated branches: compressed block tokens (coarse), top-k selected blocks (fine), and a sliding window. Matches or beats full attention (27B), with large 64K speedups.
**19. MoBA** — Lu et al. (Moonshot), 2025, arXiv 2502.13189 (v). MoE-style top-k routing of each query to KV blocks via mean-pooled block keys; serves Kimi.
**20. DeepSeek Sparse Attention (DSA)** — DeepSeek-V3.2(-Exp), 2025, arXiv 2512.02556 (v). A lightning indexer (few FP8 heads) picks the top-k (2048) tokens per query; the indexer is warmed up by KL to dense attention.
**21. Mixture of Sparse Attention (MoSA)** — Piękos, Csordás, Schmidhuber, 2025, NeurIPS 2025, arXiv 2505.00315 (v). Expert-choice: each head picks k tokens. Beats dense at equal compute (up to 27% perplexity), measured at **small scales**.
- 15M @1K: NSA, MoBA and DSA are efficiency methods for ≥32K; at 1K they cost quality or do nothing. MoSA is the exception: positive at small scale and short context, but non-standard.
- (a) Baseline: no. Dense attention at 1K is the strongest and fairest baseline.
- (b) Transfer, the shared idea: **(i) multi-resolution reads** (NSA: compressed + selected + local, mixed by a learned gate) and **(ii) top-k sparsity to avoid dilution**. ULM mapping: (i) keep 2–3 groups of slots (or 2–3 copies of the fast store) with different fixed λ (local ≈ 32 tokens, mid ≈ 256, global ≈ 4K) and a per-token sigmoid gate over the three reads, as NSA's branch gate does. This targets the 64–1024 band directly. State grows 2–3× but stays fixed. (ii) top-k masking of the address r_t (e.g. k = 16 of 712) before writing, to reduce collisions; relu² is already sparse, so measure its effective support first. MoSA's expert-choice (a slot picks the tokens it stores) maps to a **write gate per slot**; see item 23 (MoD).

## E. Memory tokens and segment recurrence

**22. Memorizing Transformer** (Wu et al., 2022, arXiv 2203.08913; kNN over a non-differentiable external KV memory); **Landmark Attention** (Mohtashami & Jaggi, 2023, arXiv 2305.16300; a landmark token per block gates block retrieval); **RMT** (Bulatov et al., 2022, arXiv 2207.06881; memory tokens passed between segments); **Infini-attention** (Munkhdalai et al., 2024, arXiv 2404.07143 (v); a linear-attention compressive memory plus local softmax, mixed by a learned scalar gate); a **Hugging Face reproduction failed** (blog, Aug 2024 (v)): the gate did not learn to use the memory and quality degraded; **Activation Beacon** (Zhang et al., 2024, arXiv 2401.03462; condensed beacon activations).
- 15M @1K: none; these are built for contexts far beyond the training window.
- (a) Baseline: no.
- (b) Transfer: Infini-attention is in effect "ULM-style memory plus local softmax", and its failure to reproduce is a warning that a compressive linear memory next to exact local attention tends to get ignored. Landmark's lesson, **store a block summary as its own retrievable key**, is the same as NSA's compressed branch (item 18(i)). Nothing additional.

**23. Log-Linear Attention** — Guo, Yang, et al., 2025, arXiv 2506.04761 (v). Fenwick-tree buckets: the prefix is split into O(log T) segments of geometrically growing size, each with its own linear-attention state, mixed by data-dependent per-level weights. Applied to Mamba-2 and GDN, it improves associative recall over lengths where the base model collapses.
- (a) Baseline: no (it is a linear-model upgrade).
- (b) Transfer: **the principled version of multi-timescale memory**. A fixed-state approximation for ULM: L = 3–4 states per layer with λ spans {16, 128, 1024, ∞}, read with per-token softmax or sigmoid level weights `α_{t,ℓ} = f(h_t)`. This gives the "high resolution recent, coarse distant" profile, and the 64–1024 band gets its own state instead of sharing one blurred store. Top-5.

## F. Block and residual changes

**24. Canon layers** — Allen-Zhu, "Physics of LMs Part 4.1", 2025, NeurIPS 2025, arXiv 2512.17351 (v).
A residual causal depthwise conv, kernel 4 (`h + conv1d(h)`), at positions A (before attention), B (on Q/K/V), C (before MLP) and D (inside MLP). Lifts NoPE to the level of RoPE, and GLA to the level of Mamba2/GDN. Most of Mamba2's gain is its conv1d (a "partial Canon-B" without residual). Synthetic results use GPT-2-small size (8–12 layers, d = 512–768); 1.3B/100B-token academic runs are called noisy by the author.
- 15M @1K: **yes, the evidence is at small scale**. It mostly helps short-range composition and reasoning depth.
- (a) Baseline: **yes**, top-3, and a fairness argument. ULM has a conv4 on its address path and the llama2.c baseline has none, so a Canon-A/B Transformer removes that confound.
- (b) Transfer: ULM's conv is Canon-A on the address path only, with no residual (it is initialised to identity, which is close). Missing: **Canon-B on the value path** (`v_t = W_v conv(h)_t`, so the stored value is an n-gram, not a single token) and the residual form. Cheap. Lower in-context gain than the top-3 because ULM already has the main conv. Top-5 item 5.

**25. Value residual (ResFormer) and value embeddings** — Zhou et al., 2024, arXiv 2410.17897 (v): `V_n ← λ₁V_n + λ₂V_1` (layer-1 values mixed into every layer). Same loss with ~16% fewer params or ~20% less data (v5 numbers; v4 reported 13%/15%), 82M–468M models. modded-nanogpt speedrun (124M, FineWeb, (v)): **value embeddings** (a separate token-embedding table added to V in some layers, U-net pattern) gave one of the largest record drops (Dec 2024).
- 15M @1K: **yes; this has the strongest small-scale evidence of anything here** (124M speedrun, ResFormer from 82M). Cost: a vocab × d table (32000 × 288 = 9.2M params, large at 15M, so share it across layers or count it honestly), or a free layer-1 value residual.
- (a) Baseline: **yes**, top-3. Use the parameter-free ResFormer version for the 15M baseline so parameter counts stay matched.
- (b) Transfer: **direct, and aimed at cause (3)**. Write `v_t = W_v h_t + α_ℓ·v_t^{(1)}` (layer-1 value) or `+ E_v[token_t]` into the fast store. The memory can then return the *exact identity* of the token that followed the matching context, which is what induction and copying at distance 64–1024 need. In Transformers this is what V/OV circuits do in early layers. Top-5 item 1.

**26. Hyper-Connections / mHC** — Zhu et al. (ByteDance), 2024, ICLR 2025, arXiv 2409.19606 (v): n parallel residual streams with learned and dynamic depth/width mixing, gains on OLMo 1B and 7B MoE. mHC (DeepSeek, 2025/26, arXiv 2512.24880 (v)) constrains the mixing matrices to be doubly stochastic (Sinkhorn) for stability: 6.7% overhead at n = 4.
- 15M @1K: unknown; plausibly small. It touches the residual stream, not the mixer.
- (a) Baseline: no (it changes the comparison to "both models plus HC").
- (b) Transfer: architecture-agnostic; it would help or hurt both equally. Not a fix for the mid-range gap.

**27. Mixture-of-Depths (MoD)** — Raposo et al. (GDM), 2024, arXiv 2404.02258 (not confirmed). Top-k router picks which tokens pass through a block. Equal quality at fewer FLOPs.
- 15M @1K: no (an efficiency method; routing is noisy at small scale).
- (a) Baseline: no.
- (b) Transfer: **write gating**, the key idea for cause (4): not every token should write. ULM: `U += β_t v_t r_{t-1}ᵀ, c += β_t r_{t-1}` with `β_t = σ(w_β·h_t + b)` per token (or per token and slot). Uninformative tokens stop diluting the posterior means, which raises effective capacity at long distance. Like DeltaNet's β, but in ULM's averaging (not delta) form it is a *weighted* posterior mean, so it stays consistent with the slot-statistics view. Pairs with Selective-style erase (item 6). Top-5 item 2.

**28. Memory layers at scale** — Berges et al. (Meta), 2024, arXiv 2412.09764 (not confirmed). Product-key memory replaces some FFNs; large factual gains. Relevant only because ULM's slots *are* the FFN. A product-key address (two √m codebooks) would let m grow to 10⁴–10⁵ slots at the same cost, which directly cuts collisions. A larger redesign; mention only.

**29. nGPT** (Loshchilov et al., NVIDIA, 2024, arXiv 2410.01131, not confirmed): everything on the hypersphere, 4–20× faster convergence claimed and disputed in replications. Skip.

---

## Ranked: transferable ideas for ULM (mid-range 64–1024)

1. **Value embeddings / layer-1 value residual into the fast store** (ResFormer, modded-nanogpt). `v_t += α_ℓ v_t^{(1)}` (free) or `+ E_v[token]`. Gives retrieval exact token identity. Strong small-scale evidence on the Transformer side.
2. **Write-strength gate β_t** (MoD, MoSA expert-choice, NSA selection). Weighted posterior mean: uninformative tokens don't write. Targets slot dilution. Tiny parameter cost; the kernel needs β in both the numerator and c.
3. **Output gate** `σ(W_g h) ⊙ read` (Gated Attention; standard in GLA/GDN/Qwen3-Next). Lets a token reject a collided or irrelevant read. The most proven single change.
4. **Multi-timescale state with per-token level mixing** (Log-linear attention, NSA's three branches, Gemma's local:global ratio, ACP's finding that most FoX heads are short-span). 2–4 fixed λ spans per layer, or a layer-wise λ init schedule as the zero-cost first step.
5. **Selective-style erase from existing address logits** (Selective Attention): a parameter-free per-slot "replace, don't average". Or, if a per-slot λ already exists, **Canon-B on the value path** (Canon).
   Honourable mentions: differential read (Diff Transformer, Diff-Mamba); untying the write-key conv from the read query (MTA); multi-head addressing (TPA/MLA framing).

## Ranked: strengthening the Transformer baseline (fairness)

1. **Canon-A/B short conv** (Canon layers): evidence at GPT-2-small scale, and it removes the "ULM has a conv, the baseline doesn't" confound.
2. **Value residual (ResFormer, parameter-free)**, or value embeddings if parameters are accounted for: the strongest small-scale (82M–124M) evidence.
3. **Gated attention output gate**: cheap and stable, allows a higher LR. Evidence is at 1.7B+, so treat the gain at 15M as a hypothesis.
   (Partial RoPE and NoPE-every-4th-layer are free knobs, for extrapolation fairness rather than in-context gains. Selective attention is a likely real gain at 1K but needs a custom kernel; include it only if FoX is also a comparator.)

Per the project's one-LR/one-seed rule, each of these is a single paired run. Small effects (<0.01 loss) will not be distinguishable from noise; judge by the 64–1024 distance-binned loss rather than aggregate loss.
