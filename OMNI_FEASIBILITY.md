# Omni-backend feasibility study

Can the frozen `Qwen/Qwen3-VL-8B-Instruct` backend in `async_omni_v2/` be swapped for
`Qwen/Qwen3-Omni-30B-A3B-Instruct` (audio+vision+text)? Fallback: `Qwen/Qwen2.5-Omni-7B`.

Feasibility study only — no pipeline code was modified, nothing was committed.

Date: 2026-07-30 · env `/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env`
(torch 2.12.1+cu130, transformers 5.12.1) · `HF_HOME=/iopsstor/scratch/cscs/dbartaula/hf_cache`

---

## VERDICT

### `Qwen/Qwen3-Omni-30B-A3B-Instruct` — **GO WITH CAVEATS**

The architectural blocker everyone expects **does not exist**. The thinker's text decoder is
API-identical to the one the project already drives, and I verified the full cache-surgery
contract (splice / evict / crop / deepcopy) against it on GPU.

It is a *memory and speed* trade, not a compatibility one:

| Caveat | Number | Fix |
|---|---|---|
| Does **not** fit at `kv_budget=262144` on one GPU | needs 110.08 GiB, GPU has 95.58 GiB | drop `kv_budget` to **≈160K** (max 182,929) |
| **Cannot** host 2 LLM copies on one GPU | 2 × 56.87 GiB = 113.74 GiB alone | single-copy mode only (which is what `run.py` actually does today) |
| Trained context is 4× shorter than the KV budget | `max_position_embeddings` 65,536 vs `next_pos` unbounded | re-base the RoPE clock on eviction (see §7) |
| MoE is **slower**, not faster, despite 3.36 B active | measured ~5–7 % more per layer than the dense 8B, and 48 layers vs 36 | none — accept ≈60–64 ms/tok (estimate) |

The A3B "3 B active params" buys nothing here: batch-1 decode in this system is
kernel-launch-bound, not memory-bandwidth-bound, and that is precisely the axis MoE does not help.

### `Qwen/Qwen2.5-Omni-7B` — **GO**

Fits the *full* stated spec — 2 LLM decoder copies + vision encoder + audio encoder +
primary KV + snapshot clone at 262144 tokens — in **61.82 GiB on a single GPU**, with 33 GiB
spare. Same decoder-interface verification passed. Its one real weakness is
`max_position_embeddings = 32,768` (2.5 min of stream), the worst of the three.

### Recommendation

If you want audio at minimum risk: **Qwen2.5-Omni-7B**. It is the only candidate that satisfies
the memory spec as written, and it is *smaller* per token of KV than the 8B you run today.

If you want the strongest model and can accept ~160K context and ~1.35× slower decode:
**Qwen3-Omni-30B-A3B**, single-copy, single-GPU.

Do **not** split either across GPUs: the snapshot clone would have to cross NVLink every writer
tick (24 GiB for Qwen3-Omni), which is exactly the failure the project already measured at
107–407 ms/tok.

---

## 0. Method, and what is measured vs estimated

The node was **busy throughout**: 4 processes (PIDs 251779–251782, another user) each held
~83 GB on the 4 GH200s at 100 % utilisation, leaving ~13 GiB free per GPU. No large model was
loaded and no process was touched.

**MEASURED** (small GPU footprint, ≤7 GiB, on `cuda:3`):
- exact parameter counts, read from safetensors headers
- the audio encoder loaded standalone and run on real audio
- the decoder-interface contract, on real-width / reduced-depth models
- relative decode cost, MoE vs dense, at equal layer count

**ESTIMATED** (clearly labelled at each use):
- full-depth ms/token, extrapolated from equal-depth measurements plus the project's own
  measured 45 ms/tok anchor for the 36-layer 8B

**NOT MEASURED** (see §8 Uncertainty):
- end-to-end load of either full model
- uncontended absolute latency
- generation quality

> One important environment correction: `nvidia-smi` reports **97,871 MiB = 95.58 GiB** per GH200,
> not 120 GB. All budgets below use 95.58 GiB. This matters — several configs land in the
> 96–115 GiB band where the distinction decides the verdict.

---

## 1. Model facts — Qwen3-Omni-30B-A3B-Instruct

Weights **already fully cached**, no download needed:
`hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695`
— 15 shards, 70.52 GB per the index, all present, zero `.incomplete` blobs, 28,010 tensors, all BF16.

The repo holds three stacked models. Only the **thinker** is relevant.

| Component | Params (B) | bf16 GiB | Keep? |
|---|---|---|---|
| thinker text decoder, 48 layers | 29.9098 | 55.71 | yes |
| thinker `embed_tokens` | 0.3114 | 0.58 | yes |
| thinker `lm_head` | 0.3114 | 0.58 | yes |
| **= one LLM decoder copy** | **30.5326** | **56.87** | |
| thinker `visual` (ViT, depth 27 + mergers) | 0.5388 | 1.00 | yes |
| thinker `audio_tower` (AuT, 32 layers) | 0.6479 | 1.21 | yes |
| **= thinker total** | **31.7193** | **59.08** | |
| `talker` (MoE, 20 layers, speech) | 3.3246 | 6.19 | **drop** |
| `code2wav` (vocoder) | 0.2160 | 0.40 | **drop** |
| **repo total** | **35.2598** | **65.68** | |

MEASURED by summing safetensors header shapes; the "30B" name refers to the thinker (30.5 B decoder).

Text decoder config (`thinker_config.text_config`):

| Field | Value |
|---|---|
| `num_hidden_layers` | **48** |
| `hidden_size` | **2048** |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | **4** |
| `head_dim` | **128** |
| `num_experts` / `num_experts_per_tok` | **128 / 8** |
| `moe_intermediate_size` | 768 (`shared_expert_intermediate_size` 0 — no shared expert) |
| `vocab_size` | 152064 |
| `rope_scaling` | `mrope_section [24,20,20]`, `mrope_interleaved: true` |
| `max_position_embeddings` | **65536** |
| `dtype` | bfloat16 |

**Active params per token — 3.36 B** (computed, matches the "A3B" name):
per layer = attention 18.87 M + router 0.26 M + 8 × 4.718 M experts 37.75 M = 56.88 M;
× 48 = 2.73 B; plus `embed_tokens` 0.311 B + `lm_head` 0.311 B.

---

## 2. Memory

### KV cache per token

`2 (K,V) × num_key_value_heads × head_dim × 2 bytes × num_layers`

| Model | per layer | × layers | **per token** | @ 262144 |
|---|---|---|---|---|
| Qwen3-VL-8B (current) | 2×8×128×2 = 4096 B | ×36 | **147,456 B = 144 KiB** | 38.65 GB = **36.00 GiB** |
| Qwen3-Omni-30B-A3B | 2×4×128×2 = 2048 B | ×48 | **98,304 B = 96 KiB** | 25.77 GB = **24.00 GiB** |
| Qwen2.5-Omni-7B | 2×4×128×2 = 2048 B | ×28 | **57,344 B = 56 KiB** | 15.03 GB = **14.00 GiB** |

The 144 KiB/token figure reproduces the reference exactly, which validates the formula.

**Good news that is easy to miss: both Omni models have a _cheaper_ KV cache than the 8B you run
today** — 4 KV heads instead of 8. Qwen3-Omni is 0.67× and Qwen2.5-Omni is 0.39× the bytes per token.

### Per 300 s of stream

At the project's ~196 vision tokens/s (`max_pixels=200704` → 196 tokens/frame at ~1 fps),
plus audio:

| Model | tok/s | KV per 300 s | 262144 tokens = |
|---|---|---|---|
| Qwen3-VL-8B | 196 | 8.67 GB *(reference: 8.2 GB)* | 22.3 min |
| Qwen3-Omni-30B-A3B | 196 + **13** audio = 209 | **6.16 GB** | 20.9 min |
| Qwen2.5-Omni-7B | 196 + **25** audio = 221 | **3.80 GB** | 19.8 min |

Audio is nearly free: 13 tok/s means 262144 tokens holds 5.6 hours of pure audio.

### Does it fit on one GPU (95.58 GiB)?

Budget = *n* × decoder + ViT + AuT + primary KV + snapshot clone + 3 GiB workspace/context.
The snapshot clone is **not optional** — `manager.snapshot_clone()` deepcopies the whole cache,
so the cache is doubly resident whenever the writer is generating.

| Model | 1 LLM copy | 2 LLM copies |
|---|---|---|
| Qwen3-VL-8B (current) | 15.26+1.07+0+72.00+3 = **91.33 GiB** FITS | 106.59 GiB — does not fit |
| **Qwen3-Omni-30B-A3B** | 56.87+1.00+1.21+48.00+3 = **110.08 GiB — DOES NOT FIT** | 166.95 GiB — no |
| **Qwen2.5-Omni-7B** | 14.19+1.26+1.19+28.00+3 = **47.64 GiB FITS** | **61.82 GiB FITS** |

Max `kv_budget` that does fit (primary + clone both counted):

| Model | 1 copy | 2 copies |
|---|---|---|
| Qwen3-VL-8B | 277,607 | 222,060 (needs 2 GPUs anyway) |
| **Qwen3-Omni-30B-A3B** | **182,929** | impossible on 1 GPU |
| **Qwen2.5-Omni-7B** | 710,975 | **578,169** |

So Qwen3-Omni works on one GPU at `kv_budget ≈ 160K` (≈13 min of A/V stream), and
Qwen2.5-Omni works at the full 262144 with either 1 or 2 copies.

**A note on the "2 copies" requirement.** The code as written builds exactly **one** backend —
`run.py:48`, the only construction site. `writer_device` and `encoder_device` are declared in
`config.py:213-214` but **never read anywhere in the codebase**; the `role="vision"/"language"`
split in `backend.py:_apply_role` is currently dead code. So single-copy is the real
configuration, and the 2-copy budget above is for the multi-GPU mode that is not yet wired up.

### Across 4 GPUs — technically yes, practically no

Qwen3-Omni would fit as: GPU0 = decoder + primary KV + both encoders (87.4 GiB),
GPU1 = decoder + clone KV (83.9 GiB). But the snapshot then crosses NVLink on **every writer
tick**: 24 GiB over the NV6 link (`nvidia-smi topo -m` shows NV6 between all pairs) is on the
order of **170 ms per snapshot**. That is the same wall the project already hit at 107–407 ms/tok,
and it is 4× worse here because the cache is bigger than a per-token transfer. Single-GPU only.

---

## 3. Transformers support — the blocking question, answered

**All classes exist in transformers 5.12.1**, no `trust_remote_code`, no version bump:
`Qwen3OmniMoeForConditionalGeneration`, `Qwen3OmniMoeThinkerForConditionalGeneration`,
`Qwen3OmniMoeThinkerTextModel`, `Qwen3OmniMoeProcessor` — all import cleanly
(`transformers/models/qwen3_omni_moe/`).

### The decoder interface is byte-identical to the one already in use

I diffed `Qwen3OmniMoeThinkerTextModel.forward` against `Qwen3VLTextModel.forward`. Same
signature, same body structure:

```python
def forward(self, input_ids=None, attention_mask=None, position_ids=None,
            past_key_values: Cache|None = None, inputs_embeds=None, use_cache=None,
            visual_pos_masks=None, deepstack_visual_embeds=None,
            **kwargs: Unpack[FlashAttentionKwargs])
```

Point by point against `backend.py:forward()`:

| Requirement | Qwen3-Omni | Notes |
|---|---|---|
| callable with `inputs_embeds` | ✅ | named parameter |
| mutable `past_key_values` | ✅ | `Cache`, returned in `past_key_values` |
| `position_ids` as 3D mRoPE | ✅ | `[3,1,L]` accepted; same `mrope_section [24,20,20]`, same `mrope_interleaved` as Qwen3-VL-8B |
| `cache_position` | ✅ | absorbed by `**kwargs` — **exactly as it already is for Qwen3-VL today** |
| DynamicCache-compatible | ✅ | defaults to `DynamicCache(config=...)`; has `.layers[i].keys/.values` and `.crop()` |
| deepcopy-able | ✅ | verified |

The `cache_position` detail is worth spelling out because it looks like a risk and is not: the
causal mask is built by `create_causal_mask(..., past_key_values=..., position_ids=text_position_ids)`
from the cache's **physical** length, not from `cache_position`. That is precisely what the
`pos_start` (logical RoPE clock) vs `phys_start` (physical write index) skew in `manager.py`
needs. Passing `[3,1,L]` (rather than `[4,1,L]`) leaves `text_position_ids=None`, so the mask
falls back to pure physical causal — correct for this append-only usage.

### Verified end-to-end on GPU

I built real-width, 4-layer instances of both Omni text decoders and ran the *actual* sequence
`backend.py` + `manager.py` perform. **Every step passed for both models:**

```
Qwen3-Omni-30B-A3B thinker text decoder
  seed ok, cache len 64, hidden (1, 64, 2048)
  ingest ok (pos != phys), cache len 96
  evict ok  -> phys 48, logical pos still 96      # manager._evict_locked, transformers-5 .layers branch
  post-evict forward ok -> phys 56, pos 104       # the pos/phys skew works
  probe splice + cache.crop(phys0) ok -> phys 56  # manager.probe leaves no trace
  copy.deepcopy ok, clone independent: True       # manager.snapshot_clone (MVCC)
  generate-on-clone ok; primary untouched: True
  KV bytes/token/layer = 2048
```

Qwen2.5-Omni-7B produced identical output. **This is the finding that decides the study: there is
no silent killer.** `manager.py` needs no changes at all.

### Loading — the talker drops cleanly

VERIFIED on meta device (no weights loaded): `enable_audio_output=False` →
`has_talker: False`, no `talker` attribute, no `code2wav` attribute, model = **31.719 B params**
(= 59.08 GiB). The 8,037 talker + 230 code2wav checkpoint tensors are simply skipped as
unexpected keys. `Qwen3OmniMoeThinkerForConditionalGeneration` (`base_model_prefix="thinker"`)
also works as a direct load path — 1311/1407 of its keys match the prefix-stripped checkpoint.

The 96 keys reported "missing" are exactly 48 layers × `{gate_up_proj, down_proj}`: transformers
5.x stores experts **fused** (`gate_up_proj [128, 1536, 2048]`) while the checkpoint has 384
per-expert tensors per layer. The fusion is registered — `conversion_mapping.py` maps
`"qwen3_omni_moe" → "qwen2_moe"`, which carries
`mlp.experts.*.{gate,up}_proj.weight → mlp.experts.gate_up_proj`. So `from_pretrained` handles it,
at the cost of extra load time and host RAM while 18,432 tensors are fused (node has 856 GB, fine).

---

## 4. Decode speed — MoE is slower, not faster

MEASURED, equal depth (4 layers, real widths, bf16, sdpa, ctx=32768, median of 30 single-token
decodes, all on the same contended `cuda:3` back-to-back):

| Decoder | ms/token @ 4 layers |
|---|---|
| Qwen3-VL-8B dense (baseline) | **34.98** |
| Qwen2.5-Omni-7B dense | 34.79 |
| Qwen3-Omni-30B-A3B MoE, `eager` experts | **36.64** |
| Qwen3-Omni-30B-A3B MoE, `grouped_mm` experts | 37.28 |

**Per layer the MoE costs ~5–7 % _more_ than the dense 8B layer.** It has 48 layers to the 8B's 36.

**ESTIMATE for full depth:** anchoring on the project's own measured 45 ms/tok for the real
36-layer 8B, and applying the measured per-layer parity plus the layer-count ratio:

> **Qwen3-Omni-30B-A3B ≈ 45 × (48/36) × 1.00–1.07 = 60–64 ms/token** — about **1.35× slower**
> than the current 8B. Qwen2.5-Omni-7B ≈ 45 × (28/36) × 0.99 ≈ **35 ms/token**, slightly faster.

Why the A3B active-parameter count does not help: the project already established this decode is
**kernel-launch-bound**, not memory-bound. MoE reduces *bytes fetched per token*, which is the
bottleneck it does not have. Meanwhile it adds 48 router + gather/scatter stages and 12 extra layers.

Two practical notes on the expert kernels:
- `experts_implementation="batched_mm"` is **unusable** — it tried to allocate **192 GiB** at
  batch=1 (it materialises a dense `[tokens × 128 experts]` tensor). Do not use it.
- `experts_implementation="grouped_mm"` works (torch 2.12 has `F.grouped_mm`; GH200 is SM90) and
  showed no measurable difference under contention. It is still probably the right default,
  because the `eager` path calls `.nonzero()` and then **iterates a CUDA tensor in Python**
  (`for expert_idx in expert_hit`) once per layer — 48 forced device→host syncs per token. That
  is the exact class of cost the project already fought when it moved logits off the D2H path.
  Worth re-measuring on an idle GPU.

---

## 5. The audio encoder — MEASURED, and cleanly separable

`Qwen3OmniMoeAudioEncoder` ("AuT"): 32 transformer layers, `d_model` 1280, `encoder_ffn_dim` 5120,
128 mel bins, `output_dim` **2048** = thinker hidden size (so its output drops straight into the
same KV cache as vision and text, no projection needed). 0.6479 B params.

I loaded it **standalone on GPU**, by pulling only the 525 `thinker.audio_tower.*` tensors out of
the checkpoint — no other part of the model was instantiated:

```
audio_tower tensors in ckpt: 525
loaded standalone in 4.2 s; missing=[] unexpected=[]
params 0.6479 B, GPU mem 2.48 GiB
FE WhisperFeatureExtractor sr=16000 hop=160 n_fft=400

   1.0s ->  100 mel frames ->   13 tokens (13.00 tok/s)  out (13, 2048)   811.6 ms
   2.0s ->  200 mel frames ->   26 tokens (13.00 tok/s)  out (26, 2048)   167.2 ms
   5.0s ->  500 mel frames ->   65 tokens (13.00 tok/s)  out (65, 2048)   168.6 ms
  10.0s -> 1000 mel frames ->  130 tokens (13.00 tok/s)  out (130, 2048)  161.8 ms
  30.0s -> 3000 mel frames ->  390 tokens (13.00 tok/s)  out (390, 2048)  157.9 ms
```

- **13.0 audio tokens per second of audio**, exactly. (16 kHz / hop 160 = 100 mel frames/s;
  three stride-2 conv2d stages → /8 → 12.5, rounded up to 13 by the chunking.)
- **Yes, it runs fully independently** — a `role="audio"` split analogous to `role="vision"` is
  straightforward, and cheaper: 1.21 GiB of weights (2.48 GiB resident incl. the fp32 sinusoid
  buffer).
- Encoder latency is ~160 ms and **flat in input length** (kernel-launch-bound, 32 layers).
  Note this was measured on a GPU at 100 % utilisation from another job, so it is an upper bound.
  At the config's `seconds_per_chunk: 2` it is ~160 ms per 2 s chunk — comfortably real-time,
  but it is a serialised 160 ms that would land on the ingester thread.

---

## 6. Fallback — Qwen2.5-Omni-7B

Not in the local cache (would need a ~22 GB download). All facts below from the model card
config plus safetensors headers fetched by HTTP range request.

| Component | Params (B) | bf16 GiB | Keep? |
|---|---|---|---|
| thinker text decoder, 28 layers (incl. embed) | 7.0706 | 13.17 | yes |
| thinker `lm_head` (untied) | 0.5450 | 1.02 | yes |
| **= one LLM decoder copy** | **7.6156** | **14.19** | |
| thinker `visual` (ViT depth 32, patch 14) | 0.6766 | 1.26 | yes |
| thinker `audio_tower` (32 layers) | 0.6396 | 1.19 | yes |
| **= thinker total** | **8.9318** | **16.64** | |
| `talker` | 1.3514 | 2.52 | **drop** |
| `token2wav` (stored F32) | 0.4491 | 0.84 | **drop** |
| **repo total** | **10.7322** | **19.99** | |

Text config: 28 layers, `hidden_size` 3584, 28 heads, **4 KV heads**, `head_dim` 128
(derived — the config has no `head_dim` field), `intermediate_size` 18944 (**dense**, not MoE),
`vocab_size` 152064, `mrope_section [16,24,24]` (chunked, **not** interleaved),
`max_position_embeddings` **32768**.

1. **Interface**: `Qwen2_5OmniThinkerTextModel.forward` has the same shape as the Qwen3-VL one
   (`inputs_embeds` / `past_key_values` / `position_ids` / `use_cache` / `**kwargs`).
   **Full cache-surgery contract VERIFIED on GPU — all steps passed** (§3).
2. **Memory**: KV **56 KiB/token** — the cheapest of the three. Full spec (2 copies + both
   encoders + primary + clone @ 262144) = **61.82 GiB on one GPU**. Could run `kv_budget` up to
   578K. This is the only candidate that meets the requirement as literally stated.
3. **Speed**: measured at parity per layer with the 8B; **ESTIMATE ≈ 35 ms/token** full depth
   — slightly faster than today, because it is dense and has only 28 layers.
4. **Audio**: `Qwen2_5OmniAudioEncoder` — `conv1` (stride 1) → `conv2` (stride 2) → `avg_pooler`
   (stride 2) → 100/4 = **25 audio tokens/s**, nearly 2× Qwen3-Omni's rate, though each token is
   cheaper. `output_dim` 3584 = thinker hidden. Separable the same way.
5. **Talker drops** the same way (`enable_audio_output=False`).
6. **Its weakness**: `max_position_embeddings` **32768** — only ~2.5 min of stream before the
   logical clock leaves trained territory. Also a 2023-era vision tower (patch 14, depth 32) and a
   generally weaker model than Qwen3-Omni.

---

## 7. Engineering work required

Ordered by necessity. `manager.py`, `controller.py`, `writer.py`, `proactivity.py` need **no
changes** — the cache contract is satisfied as-is.

1. **New `Qwen3OmniBackend` in `backend.py`** (the only model-specific file, as designed).
   Load with `enable_audio_output=False`, `dtype=bfloat16`, `experts_implementation="grouped_mm"`.
2. **`_resolve_modules` works unchanged.** The thinker layout is `m.audio_tower` / `m.visual` /
   `m.model` (text decoder) / `m.lm_head`. `cand_lm`'s third candidate `m.model` and `cand_vis`'s
   second candidate `m.visual` already resolve correctly — the defensive design pays off here.
3. **`_apply_role` must be rewritten** (~15 lines). It assumes `inner = m.model` then
   `inner.visual` / `inner.language_model`. For the thinker, `m.model` **is** the text decoder, so
   the role split must operate on `m.visual` / `m.audio_tower` / `m.model` directly. Currently
   dead code (nothing reads `writer_device`/`encoder_device`), so this is only needed if/when
   multi-GPU is wired up — which §2 argues against.
4. **Add `embed_audio(waveform)`** mirroring `embed_frame`: `WhisperFeatureExtractor` →
   `audio_tower(input_features, feature_lens=...)` → `[1, N, 2048]`. Note the input packing —
   `input_features` must be permuted and masked to `[num_mel_bins, valid_frames]` before the
   tower (see `get_audio_features` in the modeling file). Plus an audio ingest path in
   `input_ingester.py` / a new `audio_stream.py` alongside `vision_stream.py`.
5. **Lower `kv_budget`** 262144 → **160000** for Qwen3-Omni (no change needed for Qwen2.5-Omni).
6. **`_cap_image_resolution` works unchanged** — the processor is `Qwen2VLImageProcessor` with
   `patch_size` 16 / `merge_size` 2, identical to Qwen3-VL, so `max_pixels=200704` still yields
   196 tokens/frame. `get_image_features` returns an object with `pooler_output`, which
   `embed_frame` already unwraps. Deepstack features are discarded exactly as they are today.
7. **RoPE clock re-basing (recommended, and a genuine improvement).** `manager.next_pos` is
   monotonic and unbounded, but Qwen3-Omni was trained to 65,536 positions (Qwen2.5-Omni: 32,768).
   At 209 tok/s that is crossed after **5.2 minutes**. This is a *new* problem — Qwen3-VL-8B has
   `max_position_embeddings=262144`, exactly matching `kv_budget`, so today the clock never
   escapes. The standard StreamingLLM fix is to assign positions by **position-within-cache**
   rather than original position, which bounds the clock by `kv_budget` and removes the limit
   entirely. Roughly: on eviction, reset `next_pos` to the post-eviction physical length.
8. `yes_ids` / `no_ids` / `newline_ids` are derived at runtime by `_word_ids`, so the vocab change
   (151936 → 152064) needs no manual work.

---

## 8. Uncertainty — what I could not measure

The node was fully occupied by another user's job for the whole study (~13 GiB free per GPU), so:

- **Neither model was ever loaded end-to-end.** The `from_pretrained` path — in particular the
  fusion of 18,432 per-expert tensors into 96 fused parameters — is verified only structurally
  (conversion mapping registered, meta-device key match). Load time and peak host RAM are unknown.
  This is the largest remaining unknown for Qwen3-Omni, and it is a *cost* risk, not a
  *feasibility* risk.
- **All absolute latencies are inflated by contention.** The equal-depth comparison in §4 is
  reliable because all four runs shared identical conditions back-to-back, but the absolute
  ms/token figures are not usable. My attempt to isolate a per-layer slope by fitting across
  layer counts (2,3,4,5) produced non-physical negative slopes — pure contention noise — and is
  discarded rather than reported. **The full-depth ms/token figures in the verdict are
  ESTIMATES**, anchored on the project's own measured 45 ms/tok.
- **Generation quality was never assessed.** In particular the existing deliberate simplification
  — feeding linear positions to the 3D mRoPE instead of true `get_rope_index` geometry — is
  untested for audio tokens, which carry a temporal position rate (`position_id_per_seconds: 13`)
  that linear positions ignore. Interleaved vs chunked mRoPE may also behave differently under
  this simplification.
- **The 3 GiB workspace allowance** in the memory tables is an assumption, not a measurement.
  For Qwen3-Omni, which lands 14.5 GiB over budget at 262144, the verdict is insensitive to it;
  for the recommended ~160K setting it is worth confirming before committing.
- The audio encoder's ~160 ms is an upper bound measured under 100 % contention.

Re-run the §4 benchmark and attempt one real load when the node frees up.

---

## Appendix — reproduction

Scratch scripts (not part of the repo):
`/tmp/claude-1370/-iopsstor-scratch-cscs-dbartaula-system-3/849041e4-33db-4fbd-8488-df4fff67ee29/scratchpad/`
— `params.py` (safetensors param census), `audiotest2.py` (standalone audio tower),
`archtest.py` (cache-surgery contract), `perftest.py` (equal-depth decode), `metakeys.py`
(talker drop + key mapping).

Key source locations:
- `async_omni_v2/backend.py:214` — `forward()`, the call pattern that must be satisfied
- `async_omni_v2/manager.py:83,140,161,177` — evict / probe / snapshot_clone / deepcopy
- `async_omni_v2/run.py:48` — the single backend construction site
- `async_omni_v2/config.py:213-214` — `writer_device` / `encoder_device`, declared but unused
- `transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py:1666` — `Qwen3OmniMoeThinkerTextModel`
- `transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py:744` — `Qwen3OmniMoeAudioEncoder`
- `transformers/conversion_mapping.py:58` — `"qwen3_omni_moe" → "qwen2_moe"` expert fusion
- `transformers/integrations/moe.py:487` — expert implementation registry
