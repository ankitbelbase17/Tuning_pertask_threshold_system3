# Open-Weight Omni Model Survey — Frozen Backbone Candidates

**Date:** 2026-07-30 · **Branch:** `icl_ingester_writer` · **Status:** research only, no pipeline code changed, nothing committed, no weights downloaded.

**Purpose.** Find open-weight omni models (audio + video in → text out) that can serve as a **drop-in frozen backbone** for `async_omni_v2`, replacing the current vision-only `Qwen/Qwen3-VL-8B-Instruct`.

**Scope note.** `Qwen/Qwen3-Omni-30B-A3B-Instruct` and `Qwen/Qwen2.5-Omni-7B` are covered in depth by a separate agent. They appear here **only as comparison rows** in the ranking table.

**Hardware ground truth (measured, not from spec sheets):**
```
$ nvidia-smi --query-gpu=index,name,memory.total --format=csv
0, NVIDIA GH200 120GB, 97871 MiB
1, NVIDIA GH200 120GB, 97871 MiB
2, NVIDIA GH200 120GB, 97871 MiB
3, NVIDIA GH200 120GB, 97871 MiB
```
The card is branded "GH200 120GB" but **usable HBM is 97871 MiB ≈ 95.6 GB**, not 120 GB. Every fit calculation below uses 95.6 GB. This matters: several "fits in 120GB" conclusions become false at 95.6 GB.

**Environment (verified against the real interpreter, `/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env/bin/python`):** transformers 5.12.1, torch 2.12.1+cu130, Python 3.12, **aarch64**. flash-attention-2 unavailable → sdpa only.

---

## 1. VERDICT / RANKING

| # | Model | Params | Fits 1 GPU (95.6 GB)? | Splice-able decoder? | Encoders separable? | Audio tok/s | Vision tok/frame | Licence | Cached? | Verdict |
|---|-------|--------|----------------------|---------------------|--------------------|-----------|-----------------|---------|---------|---------|
| **1** | **openbmb/MiniCPM-o-4_5** | **9.37B** (8.19B decoder) | **YES** @ ≤193K KV | **YES** — decoder is stock `Qwen3ForCausalLM` | **YES** — `vpm`/`apm`/`tts` are separate top-level modules | **10** | **64** | **Apache-2.0** | No (empty stub) | **GO — TOP PICK** |
| 2 | nvidia/omnivinci | 9.2B (7.6B decoder) | YES @ ≤262K KV | YES — decoder is stock `Qwen2ForCausalLM` | YES — shipped as separate HF subfolders | ~25 | ~64–256 | Apache-2.0 | No | **CAVEATS** — 32K RoPE limit |
| — | *Qwen/Qwen3-Omni-30B-A3B-Instruct* | *30B-A3B MoE* | *tight* | *yes (Thinker)* | *yes (talker droppable)* | *~12.5* | *~185* | *Apache-2.0* | **YES (66 GB)** | *other agent's remit* |
| — | *Qwen/Qwen2.5-Omni-7B* | *~10.7B* | *yes* | *yes (Thinker)* | *yes (talker droppable)* | *25* | *~185* | *Apache-2.0* | *No* | *other agent's remit* |
| 3 | microsoft/Phi-4-multimodal-instruct | 5.6B | YES, easily | Probably (native `phi4_multimodal` in tf 5.12.1) | YES | ~12.5 | n/a | MIT | No | **CAVEATS — no video** |
| 4 | google/gemma-3n-E4B-it | ~8B raw / 4B eff. | YES | Partial — no `cache_position`, PLE needs `input_ids` | YES | ~6.25 | n/a | Gemma ToU | No | **CAVEATS → NO-GO** |
| 5 | nvidia/Nemotron-3-Nano-Omni-30B-A3B | 31B-A3B | Yes on memory | **NO — Mamba2 SSM state, uncroppable** | YES | ? | ~var | NVIDIA OMA | No | **NO-GO** |
| 6 | inclusionAI/Ming-flash-omni-Preview | ~100B MoE (52 shards) | **NO** | unknown | — | — | — | MIT | No | **NO-GO — size** |
| 7 | `cosmos3_omni` (transformers native) | — | — | yes | — | **no audio config** | — | — | — | **NO-GO — no audio input** |
| 8 | stepfun-ai/Step-Audio-2-mini | ~8B | yes | unknown | — | — | **no video** | Apache-2.0 | No | **NO-GO — no video** |
| 9 | VITA-1.5 / Ola-7b / Baichuan-Omni-1.5 / Megrez-3B-Omni / InternOmni / Ming-Lite-Omni | 3–8B | yes | unknown | — | — | — | mixed | No | **NO-GO — abandoned, tf 4.3x remote code** |
| 10 | video-SALMONN 2 / Freeze-Omni | — | — | — | — | — | — | — | No | **NO-GO — not a general omni backbone** |
| — | *Qwen/Qwen3-VL-8B-Instruct (incumbent)* | *8B* | *yes* | *yes* | *yes* | ***no audio*** | *185* | *Apache-2.0* | **YES (17 GB)** | *current baseline* |

### Budget burn per second of stream @ 1 fps (the number that decides horizon)

| Model | vis tok/frame | audio tok/s | **total tok/s** | Stream horizon @ 262144 KV | Horizon @ a memory-feasible budget |
|---|---|---|---|---|---|
| **MiniCPM-o 4.5** | **64** | **10** | **74** | **59 min** | **43 min** @193K |
| Qwen3-VL-8B (incumbent) | 185 | — (none) | 185 | 23.6 min | 17 min @193K |
| OmniVinci | ~64–256 | ~25 | ~89–281 | n/a (32K RoPE cap) | ~6 min @32K |
| Qwen2.5-Omni-7B | ~185 | 25 | ~210 | 20.8 min | — |

The 23.6 min figure for the incumbent independently reproduces MISSION.md's own statement that eviction begins at *"under ~23 min at 1 fps"* — which validates the arithmetic used for every other row.

**MiniCPM-o 4.5 buys ~2.5× the wall-clock horizon at identical KV memory.**

---

## 2. TOP RECOMMENDATION — `openbmb/MiniCPM-o-4_5`

- Model: <https://huggingface.co/openbmb/MiniCPM-o-4_5>
- Paper: arXiv **2604.27393** (the rival baseline, per MISSION.md §2)
- Licence: **Apache-2.0** (weights *and* code) — permits research publication and commercial use.
- Latest omni release from OpenBMB (`lastModified` 2026-07-06, **612,000 downloads**). There is no MiniCPM-o 4.6; `MiniCPM-V-4.6` is the vision-only line. Verified by listing the whole `openbmb` org via the HF API.

### 2.1 Why this one specifically — the experiment it enables

This is the **rival baseline paper's own model, and it is open-weight**. MISSION.md §2 states their key claim: *"the model first predicts a binary listen/speak control token before content generation"*, trained in via speech pretraining → omni pretraining → SFT → GRPO + RLAIF-V.

Using MiniCPM-o 4.5 frozen as our backbone gives the cleanest experiment available in this project: **same weights, same modalities, same encoders — their TRAINED listen/speak proactivity vs our training-free ICL controller architecture.** Every confound (backbone capability, pretraining data, modality coverage) is held constant. If our frozen architecture matches or beats their trained control token on their own model, the result is directly damaging to the paper's central claim. No other candidate offers this.

It also directly serves INVARIANT 4 (frozen model): we are not competing on training, we are removing training from the comparison entirely.

### 2.2 Requirement #1 — splice-able decoder: **VERIFIED, strongest of any candidate**

`modeling_minicpmo.py` line 116:

```python
self.llm = Qwen3ForCausalLM(config)
```

The text decoder is a **stock `transformers.Qwen3ForCausalLM`** — not a fork, not a custom class. So `model.llm.model` is a plain `Qwen3Model` and `model.llm.lm_head` is a plain `nn.Linear`. This is exactly the object `backend.forward()` needs.

I verified the full `manager.py` contract end-to-end against a stock Qwen3 on transformers 5.12.1 (tiny random config, CPU — no weights downloaded):

```
type(m.model)= Qwen3Model  lm_head Linear
after c1 len 8                       # inputs_embeds + DynamicCache append
after c2 len 12                      # position_ids=1000..1003 while cache_position=8..11  <- TWO CLOCKS
logits torch.Size([1, 100])          # lm_head on last position only
deepcopy len 12                      # MVCC snapshot_clone()
crop-> 8   snap untouched 12         # probe erase; snapshot genuinely independent
after evict 5                        # StreamingLLM eviction via cache.layers[i].keys/.values
post-evict forward OK len 7          # forward still works after manual eviction
SPLICE TEST: PASS
```

Every primitive `manager.py` relies on — `inputs_embeds`, mutable `past_key_values`, divergent `position_ids`/`cache_position` (the two-clock design), `crop()`, `deepcopy()` snapshots, manual `.layers[].keys/.values` surgery — works.

**Better still, the model already implements our architecture's core primitives itself**, which is independent evidence that this design is sound on these weights:

| `async_omni_v2` concept | MiniCPM-o 4.5 equivalent | Location |
|---|---|---|
| `manager._evict_locked()` StreamingLLM eviction | `drop_tokens_from_cache(cache, length, preserve=...)` — keeps a sink prefix, drops a middle span | `utils.py:1046` |
| two clocks (`next_pos` vs physical length) | `streaming_position_offset` + `realign_rotary_suffix()` (they *re-rotate* the suffix keys instead) | `utils.py:1004`, `modeling_minicpmo.py:1601` |
| `manager.probe()` splice-then-erase | `save_speculative_snapshot()` / restore-by-truncation | `modeling_minicpmo.py:1605+` |
| `manager.snapshot_clone()` | deep-clones `audio_past_key_values` into a fresh `DynamicCache` | `modeling_minicpmo.py:1636` |
| ingester single-writer prefill | `streaming_prefill()` → `self.llm(past_key_values=..., inputs_embeds=..., use_cache=True)` | `modeling_minicpmo.py:1959` |

`streaming_prefill` calls the LLM exactly the way `backend.forward()` does.

### 2.3 Architecture, and what we can drop

From `config.json` (<https://huggingface.co/openbmb/MiniCPM-o-4_5/raw/main/config.json>) and the safetensors index:

| Component | Attribute | What it is | Config flag |
|---|---|---|---|
| Vision | `vpm` + `resampler` | SigLIP2 (`siglip_vision_model`, 27 layers, h=1152, patch 14) + perceiver `Resampler(num_queries=64)` | `init_vision` |
| Audio | `apm` + `audio_projection_layer` + `audio_avg_pooler` | `MiniCPMWhisperEncoder` from `openai/whisper-medium` (24 layers, d=1024) | `init_audio` |
| Text decoder | `llm` | **stock `Qwen3ForCausalLM`** — 36 layers, h=4096, 32 heads, **8 KV heads, head_dim 128**, vocab 151748, `max_position_embeddings=40960` | — |
| Speech out | `tts` | CosyVoice2 (h=768, 20 layers) + `assets/token2wav/*.onnx` | **`init_tts`** |

Total from the safetensors index: `total_size = 18,743,575,332 bytes` → **18.74 GB bf16 → 9.37B params**. Top-level module tensor counts: `vpm` 437, `llm` 399, `apm` 367, `tts` 194, `resampler` 13, `audio_projection_layer` 4.

**The talker is cleanly droppable.** `config.init_tts = False` skips `init_tts_module()` entirely (`modeling_minicpmo.py:136`), saving ~0.8 GB of weights *and* avoiding the ONNX/`token2wav` asset loading path (`campplus.onnx`, `flow.pt`, `hift.pt`, `speech_tokenizer_v2_25hz.onnx`). We need text output only, so this is pure profit — exactly the "talker is separable = PLUS" case.

**Encoders are separable, and the split is cleaner than the incumbent's.** `vpm`, `apm`, `llm`, `tts` are four independent top-level `nn.Module` attributes, each toggled by its own config flag. The `role=` mechanism in `backend.py::_apply_role()` maps onto this directly and *more* simply than it does onto Qwen3-VL (where the layout has "moved between transformers versions", per the backend's own docstring):

- `role="vision"` → keep `vpm` + `resampler`, set `init_audio=False, init_tts=False`, drop `llm`
- `role="audio"` → keep `apm` + `audio_projection_layer`, drop the rest (a genuinely new role, which the pipeline needs anyway for the audio encoder thread)
- `role="language"` → keep `llm` only, all three `init_*` flags off

### 2.4 Token rates — derived from source, not marketing

**Audio = 10 tokens/sec.** From `_get_feat_extract_output_lengths` (`modeling_minicpmo.py:404`):
```python
input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
input_lengths_after_pooling = (input_lengths_after_cnn - self.config.audio_pool_step) // self.config.audio_pool_step + 1
```
Whisper mel hop is 160 samples @ 16 kHz = **100 mel frames/s** → conv stride 2 → **50/s** → `audio_pool_step = 5` (`config.json`) → **10 tokens/s**. This is the *lowest audio token rate of any candidate* — 2.5× cheaper than Qwen2.5-Omni / OmniVinci (25/s).

**Vision = 64 tokens/frame.** `config.query_num = 64`, and `streaming_prefill` calls the processor with `max_slice_nums=1` (`modeling_minicpmo.py:1922`), so a streaming frame yields the single global view = **64 tokens**. I checked for the MiniCPM-V 4.5 "3D-Resampler" (6 frames → 64 tokens, 96×): **it is not in MiniCPM-o 4.5's modeling file** — no `temporal_ids`, and `Resampler` is the plain 2D perceiver. This is *good news for us*: ingestion is genuinely frame-by-frame with no 6-frame grouping, so there is no lookahead buffer and **INVARIANT 1 is respected natively**.

**Combined: 64 + 10 = 74 tokens per second of stream at 1 fps** vs the incumbent's 185. Same KV memory buys 2.5× the horizon.

### 2.5 Single-GPU fit (95.6 GB)

KV bytes/token = `2 (K,V) × 36 layers × 8 kv_heads × 128 head_dim × 2 bytes` = **147,456 B = 144 KiB/token**.

Critically, **this is byte-identical to the incumbent Qwen3-VL-8B** (verified from its cached `config.json`: 36 layers, h=4096, 8 KV heads, head_dim 128). MiniCPM-o 4.5 is a **KV-cost-neutral swap** — no memory regression, only a 2.5× token-rate improvement.

| Item | Size |
|---|---|
| `llm` decoder × 2 replicas (ingester + writer), 8.19B bf16 | 32.8 GB |
| `vpm` SigLIP2 + `resampler` | ~0.9 GB |
| `apm` Whisper-medium encoder + projector | ~0.7 GB |
| `tts` **dropped** | 0 (saves ~0.8 GB) |
| **Static subtotal** | **~34.4 GB** |
| Headroom for activations / fragmentation | ~4 GB |
| **Remaining for primary KV + MVCC snapshot** | **~57 GB** |

Since `snapshot_clone()` doubles the cache transiently, max budget ≈ `57e9 / (2 × 147456)` ≈ **193,000 tokens**, i.e. **~43 minutes of stream at 74 tok/s**.

Note the honest consequence: **`kv_budget = 262144` with a live full-length snapshot does not fit on one GH200** (it would need 32.8 + 1.6 + 2×38.7 = **111.8 GB > 95.6 GB**). But this is *already true today* for Qwen3-VL-8B, identical arithmetic — it is not a regression introduced by this swap. At the incumbent's actual horizon (23.6 min), MiniCPM-o needs only ~105K tokens = 15.5 GB primary + 15.5 GB snapshot, total ~65 GB — comfortable, with ~30 GB spare.

Given the project's measured finding that multi-GPU splitting is **worse** (107–407 ms/tok vs 94), staying on one GPU is the whole game, and MiniCPM-o's lower token rate is what buys the room to do it.

### 2.6 transformers 5.12.1 compatibility — the one real risk, and it is contained

The repo is `custom_code` (`trust_remote_code=True`) and its README pins `transformers==4.51.0`. I tested every top-level import in `modeling_minicpmo.py` against the real 5.12.1 interpreter:

```
OK   transformers            LlamaConfig / LlamaModel / PreTrainedModel
OK   transformers            Qwen3ForCausalLM / Qwen3PreTrainedModel / TextIteratorStreamer
OK   transformers.cache_utils Cache / DynamicCache / EncoderDecoderCache / StaticCache
OK   transformers.generation.logits_process  TopKLogitsWarper / TopPLogitsWarper
OK   transformers.integrations               is_deepspeed_zero3_enabled
OK   transformers.modeling_outputs           BaseModelOutputWithPast
OK   transformers.models.whisper.modeling_whisper  WhisperEncoder
OK   transformers.activations                ACT2FN
```

**All 16 imports resolve on 5.12.1.** No import-level breakage.

The one genuine incompatibility I found:

```
$ python -c "DynamicCache().key_cache"
AttributeError: 'DynamicCache' object has no attribute 'key_cache'
```

`key_cache` / `value_cache` / `_seen_tokens` were **removed** in transformers 5.x (replaced by `cache.layers[i].keys/.values`). MiniCPM-o's `utils.drop_tokens_from_cache` and several `modeling_minicpmo.py` snapshot paths use `cache.key_cache` directly, so **their own streaming helpers will break on 5.12.1**.

**Why this does not block us:** we do not call their streaming helpers. We call `model.llm.model(...)` and `model.llm.lm_head(...)` from our own `backend.forward()`, and eviction/snapshot/crop are done by *our* `manager.py`, which already handles both the 4.x and 5.x cache layouts (`manager.py:88` branches on `hasattr(self.cache, "layers")`). We need their `__init__` to run and their `vpm`/`apm`/`llm` submodules to load — nothing more.

**Where it does bite:** to run *their* trained listen/speak proactivity as the head-to-head baseline, we need their `streaming_prefill`/`streaming_generate` path, which needs `transformers==4.51`. **Plan for a second conda env pinned to transformers 4.51 for the baseline arm of the experiment.** This is a real, schedule-relevant finding.

**flash-attn:** not required. `init_audio_module()` explicitly *avoids* flash-attn (`modeling_minicpmo.py:228`: *"using flash_attention_2 will cause: RuntimeError: cu_seqlens_q must have shape (batch_size + 1)"*) and selects `sdpa`. The vision module falls back to `eager` unless flash-attn is requested. Loading with `attn_implementation="sdpa"` is a supported path — **safe on aarch64.**

### 2.7 Cache status — NOT downloaded

```
$ du -sh $HF_HOME/hub/models--openbmb--MiniCPM-o-4_5
8.0K   .../models--openbmb--MiniCPM-o-4_5     # only a refs/ stub — no blobs
```
Both `models--openbmb--MiniCPM-o-4_5` (8 KB) and `models--openbmb--MiniCPM-V-4_5` (48 KB) are **empty stubs**. Someone started a download that never completed. Real download ≈ **18.7 GB** — about 20 minutes on a good link, trivial against a 7-week deadline, and 3.5× smaller than the already-cached Qwen3-Omni-30B (66 GB).

---

## 3. SECOND CHOICE — `nvidia/omnivinci`

- Model: <https://huggingface.co/nvidia/omnivinci> · Paper: <https://huggingface.co/papers/2510.15870>
- Licence: **Apache-2.0**. Last modified 2026-02-23. Downloads: **491** (low adoption — a risk signal).

**Its standout property: components ship as independent, standard HF folders.**
```
subdirs: ['asset', 'llm', 'mm_projector', 'sound_mm_projector', 'sound_tower', 'vision_tower', 'transformers']
llm/config.json          -> architectures: ['Qwen2ForCausalLM'],  28 layers, h=3584, 4 KV heads
vision_tower/config.json -> architectures: ['SiglipVisionModel']
sound_tower/config.json  -> architectures: ['Qwen2AudioEncoder']
```
All three are **stock transformers classes**, loadable via `from_pretrained(repo, subfolder="llm")`. This means we can **bypass the entire VILA remote-code framework** (`builder.py`, `media.py`, `distributed.py`, …) and never call `VILAForCausalLM` at all. The `role=` mechanism becomes trivial — each role loads a different subfolder. That is arguably an even cleaner separation than MiniCPM-o's.

It is genuinely omni for our use case: `config.json` has `load_audio_in_video = True` and `interleaved_vis_aud_in_video = True`, i.e. audio and video are interleaved along a shared timeline — the exact ingestion pattern our ingester wants.

**Why it is second, not first:**
1. **`max_position_embeddings = 32768`.** Hard blocker against our design. `manager.py` keeps a *monotonic* logical RoPE clock (`next_pos`) that keeps climbing after eviction. On a 32K-trained model that clock leaves the trained RoPE range within ~6 minutes of stream. Fixing it means adopting MiniCPM-o-style *position re-indexing* (re-rotating suffix keys on eviction) rather than our two-clock scheme — a real change to `manager.py`, which is core, invariant-bearing code.
2. **~25 audio tokens/s** (Qwen2-Audio-style encoder: 100 mel fps → /2 → /2), 2.5× MiniCPM-o's burn.
3. It is **not the rival paper's model**, so it loses the entire "same weights, their training vs our architecture" experimental argument.
4. 491 downloads and a heavyweight research framework around it — low community validation, higher chance of undiscovered friction.

KV cost is attractive though: `2 × 28 × 4 × 128 × 2` = **57,344 B = 56 KiB/token**, only 39% of MiniCPM-o's — so memory is never the binding constraint here. The 32K RoPE ceiling is.

---

## 4. Disqualified candidates — exactly why each failed

### 4.1 `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` — **NO-GO (requirement #1)**
<https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16> · 601,951 downloads, NVIDIA Open Model Agreement.

On paper this is the most attractive new model: 256K context, CRADIO-v4-H vision + Parakeet audio, 31B-A3B (only ~3B active). It fails hard on the decisive requirement.

From `config.json`:
```
llm_config.hybrid_override_pattern = MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME
llm_config.num_hidden_layers = 52     ssm_state_size = 128     mamba_num_heads = 64
```
Decoded: **23 Mamba2 layers, 23 MLP layers, and only 6 attention layers** out of 52.

`modeling_nemotron_h.py` defines its own `NemotronHHybridDynamicCache` (line 59) holding `conv_states` and `ssm_states` alongside `key_cache`/`value_cache`. This breaks us in three independent ways:

1. **SSM state is not croppable.** A Mamba2 recurrent state is a fixed-size *summary* of all history, not a per-token record. `manager._truncate(phys_len)` and StreamingLLM eviction (keep sink + recent window) are undefined operations on it. There is no `crop`.
2. **The probe cannot be erased.** `manager.probe()` splices a question, reads logits, then truncates back to leave *no trace*. Running the probe through Mamba layers **irreversibly advances the recurrent state**; restoring it requires a full save/restore of all 23 SSM states on *every probe*, every tick.
3. **The two-clock design does not apply.** There is no RoPE position for an SSM layer; logical-vs-physical position separation is meaningless there.

Additionally it needs compiled `causal-conv1d` and `mamba-ssm` kernels (`modeling_nemotron_h.py:308–316`), neither of which ships aarch64 wheels — they would need source builds on GH200.

Ironically its KV cache would be tiny (6 attention layers × 2 KV heads = **6 KiB/token**, 1.6 GB at 262K). Doesn't matter: the architecture is incompatible with a mutable, croppable, snapshot-able cache. **This is exactly the silent killer the brief warned about.**

### 4.2 `google/gemma-3n-E4B-it` — **NO-GO**
<https://huggingface.co/google/gemma-3n-E4B-it>. Natively supported (`transformers/models/gemma3n`), so splice-ability looked plausible. Three problems, from the 5.12.1 source:

1. **Per-Layer Embeddings need `input_ids`.** `Gemma3nTextModel.forward` (line 1701) derives `per_layer_inputs` from `input_ids`. Our pipeline feeds `inputs_embeds` for vision/audio tokens with **no corresponding token ids**, so the PLE contribution silently degrades for every non-text token — a frozen model quietly running off-distribution on exactly the tokens we care about.
2. **Sliding-window attention + KV sharing.** `create_sliding_window_causal_mask` (line 35), `is_kv_shared_layer` (line 1187) — most layers attend only within a short local window, and the last N layers reuse earlier layers' KV. A long-horizon streaming *memory* thesis on a mostly-local-attention model is self-defeating.
3. `Gemma3nTextModel.forward` does **not** declare `cache_position`; it would pass only via `**kwargs`. Combined with 32K context and no real video path (frames-as-images only), this is not a viable backbone.

Licence is the Gemma Terms of Use — permits research publication, but is not OSI-open, unlike Apache-2.0 alternatives.

### 4.3 `microsoft/Phi-4-multimodal-instruct` — **CAVEATS, effectively NO-GO**
<https://huggingface.co/microsoft/Phi-4-multimodal-instruct> · MIT, 553,615 downloads, 5.6B (32 layers, h=3072, 8 KV heads → **128 KiB/token**). Attractive size, permissive licence, and **natively implemented** in transformers 5.12.1 as `phi4_multimodal`.

Blockers:
1. **No video.** It is an image + audio model; the processor has no video path and the model has no temporal modelling. Frames-as-images loses all temporal structure — the core of this project.
2. **Mixture-of-LoRAs.** `config.json` declares `vision_lora` (r=256) and `speech_lora` (r=320) applied to the *decoder's* attention and MLP projections. Per the Phi-4-Mini report (<https://arxiv.org/html/2503.01743v1>), joint vision-speech input activates **only LoRA_V**. Usable if we pin one adapter for the whole stream — but it means the decoder we splice into is adapter-modified, and the audio path is then running under the vision adapter.
3. **Checkpoint/implementation mismatch.** The Hub repo carries `auto_map` → remote `modeling_phi4mm.py` written for transformers 4.46.1, while the *native* `Phi4MultimodalModel` in 5.12.1 contains **no LoRA code at all** (grep for `lora` returns nothing) — it expects pre-merged weights. Which checkpoint pairs with which implementation is unresolved and would need empirical work.
4. Its audio stack is speech-centric (ASR/AST/SQA), typically trained on ≤30 s clips — not continuous ambient stream audio.

### 4.4 `cosmos3_omni` — **NO-GO (no audio)**
Native in transformers 5.12.1. Its config declares only `sub_configs = {"vision_config": ..., "text_config": ...}` and vision/video token ids — **there is no audio config**. "Omni" here means video+text. Fails requirement #3.

### 4.5 `inclusionAI/Ming-flash-omni-Preview` — **NO-GO (size)**
<https://huggingface.co/inclusionAI/Ming-flash-omni-Preview> · MIT, 824 downloads, **52 safetensors shards** (~100B-class MoE). Cannot hold two decoder replicas + encoders + a 262K KV cache on one 95.6 GB GH200, and the project has measured that multi-GPU splitting is worse (107–407 ms/tok vs 94).

### 4.6 `stepfun-ai/Step-Audio-2-mini` — **NO-GO (no video)**
Apache-2.0, 20,890 downloads, actively maintained (2026-02-14) — but audio+text only. Fails requirement #3.

### 4.7 The 2025-era cohort — **NO-GO (abandoned)**
Checked via HF API; all are stale with negligible adoption:

| Model | Licence | Downloads | Last modified |
|---|---|---|---|
| VITA-MLLM/VITA-1.5 | none declared | 90 | 2025-01-16 |
| THUdyh/Ola-7b | apache-2.0 | 73 | 2025-06-23 |
| baichuan-inc/Baichuan-Omni-1d5 | apache-2.0 | 151 | 2025-02-08 |
| Infinigence/Megrez-3B-Omni | apache-2.0 | 77 | 2025-02-14 |
| inclusionAI/Ming-Lite-Omni | mit | 84 | 2025-10-27 |
| OpenGVLab/InternOmni | mit | 77 | 2025-01-20 |

Double-digit download counts 6–18 months after release means effectively zero community validation. All are `custom_code` targeting transformers 4.37–4.44; the `key_cache`/`value_cache` removal alone (demonstrated in §2.6) will break their cache handling on 5.12.1, and a `VITA-1.5` with no declared licence cannot be used for a publication. `video-SALMONN 2` and `Freeze-Omni` are research systems for video captioning and duplex speech respectively, not general omni backbones — neither offers a video+audio→text decoder we can splice.

---

## 5. Engineering work required

### 5.1 MiniCPM-o 4.5 (top pick) — a genuinely small diff, confined to `backend.py`

Nothing outside `backend.py` needs to change. `manager.py`, the ingester, controller and writer are model-agnostic and already satisfy this model's cache contract.

**New `MiniCPMO45Backend(ModelBackend)`:**

1. **Load.** `AutoModel.from_pretrained("openbmb/MiniCPM-o-4_5", trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16, init_tts=False)` — `init_tts=False` drops CosyVoice2 and the ONNX token2wav assets. Set `init_vision=False` / `init_audio=False` per role.

2. **`_resolve_modules()` — simpler than the incumbent.** Fixed, stable attribute names; no defensive `_first()` probing needed:
   ```
   self.language_model = m.llm.model      # stock Qwen3Model
   self.lm_head        = m.llm.lm_head
   self.embed_tokens   = m.llm.model.embed_tokens
   self.visual         = m.vpm            # + m.resampler
   self.audio          = m.apm            # + m.audio_projection_layer, m.audio_avg_pooler
   ```

3. **`forward()` — a simplification, not a complication.** MiniCPM-o's decoder uses ordinary **1D RoPE**, not Qwen3-VL's 3D mRoPE. The current code's deliberate hack disappears:
   ```python
   # Qwen3-VL (current): shape-correct fudge for 3D mRoPE, ignores image geometry
   position_ids = lin.view(1, 1, L).expand(3, 1, L).contiguous()
   # MiniCPM-o 4.5: just the honest thing
   position_ids = lin.view(1, L)
   ```
   This **removes** the one documented "deliberate simplification" in the backend docstring — a correctness *gain*, not just a port.

4. **`embed_frame()`.** Replace the Qwen2VL image-processor path with `vpm` → `resampler`, which returns exactly `query_num = 64` tokens. `_cap_image_resolution()` is no longer needed for token control (the resampler fixes the count structurally); keep a resolution cap only for encoder speed.

5. **`embed_audio()` — the one genuinely new method.** `apm` (Whisper encoder, chunked with `audio_past_key_values`) → `audio_projection_layer` → `audio_avg_pooler` → `[1, N, 4096]` at 10 tokens/s. This becomes the audio encoder thread's output, feeding the same single-writer ingester — precisely the "audio becomes another encoder node" story MISSION.md §3 already commits to.

6. **`_apply_role()`.** Extend from two roles to three (`vision` / `audio` / `language`) using the `init_*` flags, which is cleaner than the current `inner.visual = None` surgery.

7. **`yes_ids`/`no_ids`.** Re-derive for this tokenizer (vocab 151748, Qwen3 tokenizer + MiniCPM special tokens) — `_word_ids()` already handles this generically.

**Config changes** (`config.py`): `model_id`, and lower `kv_budget` from 262144 to ~193000 to fit one GPU with a live snapshot (which still *increases* horizon from ~17 min to ~43 min because the token rate drops 2.5×).

**Separately, for the baseline arm:** build a second env pinned to `transformers==4.51.0` to run MiniCPM-o's *own* `streaming_prefill`/`streaming_generate` listen/speak proactivity — their code path needs the removed `key_cache` API. Budget this; it is the other half of the head-to-head.

### 5.2 OmniVinci (second choice) — more work, in riskier places

1. **Load components independently**, bypassing VILA entirely: `Qwen2ForCausalLM.from_pretrained(repo, subfolder="llm")`, `SiglipVisionModel.from_pretrained(repo, subfolder="vision_tower")`, `Qwen2AudioEncoder.from_pretrained(repo, subfolder="sound_tower")`, plus the two projectors from `mm_projector/` and `sound_mm_projector/`. All stock classes → `forward()` is the same shape as MiniCPM-o's, 1D RoPE.
2. **Reimplement the projectors' glue** — `mm_projector` / `sound_mm_projector` weights are shipped but their exact forward logic lives in the VILA `builder.py`; we would need to read it and reproduce it faithfully.
3. **The expensive part: change `manager.py`'s position handling.** With `max_position_embeddings = 32768`, the monotonic `next_pos` clock must be replaced by position **re-indexing on eviction** (re-rotate the surviving suffix keys, as `utils.realign_rotary_suffix` does in MiniCPM-o). This touches `_evict_locked`, `probe`, and `snapshot_clone` — INVARIANT-bearing code the brief explicitly protects. This is the main reason it ranks second.

---

## 6. UNCERTAINTY — what I could not verify without loading weights

The node is currently saturated (4 processes × ~83 GB, ~13 GB free per GPU), so **no model was loaded and no weights were downloaded**. Everything above comes from `config.json`, `modeling_*.py` source, safetensors indices, HF API metadata, model cards and papers — plus one CPU-only splice test on a *randomly initialised* tiny Qwen3.

Open items, roughly in order of risk:

1. **Real MiniCPM-o weights have never been loaded here.** The splice test used a random tiny `Qwen3Config`. It proves the *API contract* on transformers 5.12.1; it does not prove that `AutoModel.from_pretrained(..., trust_remote_code=True)` **completes** on 5.12.1. `__init__` and `from_pretrained` may touch removed 4.x APIs on the loading path (not just the streaming path I audited). **This is the single most important thing to test first** — it is a ~20 min download and a 5 min load, and it converts the top recommendation from "very likely" to "confirmed".
2. **Quality of frozen MiniCPM-o 4.5 under our ICL controller is entirely unmeasured.** Its instruction-following on our control-JSON DSL, and whether Qwen3-8B-based weights respond to the prompts tuned for Qwen3-VL-8B, are unknown. Prompts in `config.py` may need retuning.
3. **Vision token count of 64/frame is inferred** from `query_num=64` + `max_slice_nums=1` in `streaming_prefill`. Confirm empirically the way `_cap_image_resolution`'s docstring says the current count was confirmed — via the profiler's `vis_tokens_per_frame` sample.
4. **Audio 10 tok/s is derived from source arithmetic**, not measured. Verify against a known-length wav.
5. **Memory figures are analytic.** Component weight sizes are apportioned from the 18.74 GB total by parameter count, not measured with `torch.cuda.memory_allocated()`. Real usage includes activations, the ~150K-vocab lm_head, allocator fragmentation, and Whisper's `audio_past_key_values`. **The ~193K token budget should be treated as an estimate to validate, not a setting to trust.** Expect to tune it down.
6. **Latency is completely unknown.** The whole project's justification for single-GPU rests on measured ms/tok (94 vs 107–407). Whether MiniCPM-o's SigLIP2+resampler encoder and Whisper chunked encoder hit the tick budget at 1–3 fps has not been measured. SigLIP2 at `image_size: 980` could be slower than Qwen3-VL's ViT per frame even though it emits fewer tokens.
7. **`init_tts=False` is read from `modeling_minicpmo.py:136`** but not exercised. Confirm it doesn't break `from_pretrained` on a checkpoint whose safetensors index *contains* 194 `tts.*` tensors (may warn about unexpected keys, or may error under strict loading).
8. **The transformers 4.51 baseline env is assumed workable but untested** — including whether `flash_attn` is a hard requirement anywhere in their full duplex path on aarch64.
9. **OmniVinci's projector logic is unread.** I confirmed the *layout* (standalone subfolders, stock classes) but not that the projectors can be driven without the VILA framework.
10. **Nemotron's disqualification is architectural and I am confident in it**, but I did not attempt an SSM-state save/restore prototype. If a future variant exposed a checkpointable SSM state with cheap save/restore, the probe-erase objection could soften (the crop/eviction objections would not).
11. **Qwen3.5-Omni-Light could not be verified.** Reported by one secondary source as the only open-weight variant of Qwen3.5-Omni, but `Qwen/Qwen3.5-Omni-Light` returns "Invalid username or password" from the HF API (gated or nonexistent). If it exists and is open, it deserves evaluation — it would likely inherit the splice-able Thinker structure of the Qwen omni line.
12. **Licence review is from HF `cardData` metadata**, not a lawyer's read of full text. Apache-2.0 (MiniCPM-o, OmniVinci) is unambiguous for research publication. NVIDIA Open Model Agreement (Nemotron) and Gemma Terms of Use carry additional conditions that would need checking if either were revived.

---

## 7. Bottom line

**Adopt `openbmb/MiniCPM-o-4_5` as the omni backbone.** It is the only candidate that is simultaneously: the rival paper's own open-weight model (enabling the cleanest possible experiment), Apache-2.0, splice-able through a *stock* `Qwen3ForCausalLM`, KV-cost-identical to the incumbent, 2.5× cheaper per second of stream, cleanly separable into vision/audio/language/talker modules, droppable-TTS, sdpa-safe on aarch64, and only an 18.7 GB download.

Its decoder is more splice-able than what the pipeline runs today, its 1D RoPE *removes* the backend's one documented approximation, and the model's own source independently implements StreamingLLM eviction, RoPE re-indexing, cache snapshots and probe-erase — the very primitives `manager.py` was built around.

**First action:** download the 18.7 GB and confirm `from_pretrained(..., trust_remote_code=True, init_tts=False)` completes under transformers 5.12.1. That single test resolves the largest open risk. **Second action:** stand up the `transformers==4.51` baseline env, since the head-to-head needs their code path and that dependency is on the critical path.
