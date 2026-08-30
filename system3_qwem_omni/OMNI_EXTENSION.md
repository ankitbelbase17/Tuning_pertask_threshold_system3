# OMNI EXTENSION — decision report

**Date:** 2026-07-30 · **Status:** decision made, execution deferred to Phase 5
**Question:** should System 3 move from vision-only (Qwen3-VL-8B) to an audio+video
backbone, and if so, which one?
**Sources:** `OMNI_FEASIBILITY.md` (Qwen-Omni deep dive), `OMNI_MODEL_SURVEY.md`
(10+ candidate survey). This document is the consolidated decision.

---

## 1. Verdict

| | |
|---|---|
| **Do it?** | **Yes — but in Phase 5, not now.** Prove the method on vision-only first. |
| **Which model?** | **`openbmb/MiniCPM-o-4_5`** (9.37 B, Apache-2.0) |
| **Second choice** | `nvidia/omnivinci` (9.2 B, Apache-2.0) |
| **Fine-tune?** | **No. Never.** It deletes the thesis (INVARIANT 4). |
| **Blocking prerequisite** | RoPE position re-basing (ROADMAP 1.5) |

**The blocker everyone expected does not exist.** The concern was that an omni model
would hide its KV cache behind a monolithic `.generate()` and be un-splice-able. It
doesn't. Verified on transformers 5.12.1 for all three serious candidates.

---

## 2. Why extend at all

We ingest no audio, and **84% of OmniPro depends on it**. Measured from
`benchmark.json` (2,700 samples):

| audio_dependency | samples | share |
|---|---:|---:|
| none | 432 | **16.0%** |
| helpful | 500 | 18.5% |
| required | 1,768 | 65.5% |

Vision-only caps us at **16% of the benchmark**, and reviewers will notice. Audio is
not a feature request — it is the difference between a subset result and a benchmark
result.

**But the extension must not cost us the thesis.** MiniCPM-o 4.5 reached 20.9% F1 with
full pretraining + SFT + GRPO + RLAIF-V. Our claim is that a *frozen* model already
knows when to speak. A frozen backbone swap keeps that claim; fine-tuning discards it
and puts us on their turf with less data, less compute and less time.

---

## 3. The recommendation: MiniCPM-o 4.5

### 3.1 The decisive technical fact

```python
# modeling_minicpmo.py:116
self.llm = Qwen3ForCausalLM(config)
```

A **stock transformers class**, not a fork. The full `manager.py` contract was verified
against it on transformers 5.12.1 — `inputs_embeds` + mutable `DynamicCache`, divergent
`position_ids` / `cache_position` (our two-clock case), `crop()`, `deepcopy()` snapshots,
manual `.layers[].keys/.values` eviction. All pass.

### 3.2 It already implements our primitives

| MiniCPM-o has | our equivalent |
|---|---|
| `drop_tokens_from_cache` | sink-preserving StreamingLLM eviction |
| `realign_rotary_suffix` + `streaming_position_offset` | **the two-clock fix we are missing** (ROADMAP 1.5) |
| speculative-snapshot-by-truncation | our probe splice-and-erase |
| `streaming_prefill` | `backend.forward()` |

That third row matters beyond convenience: **a shipped streaming system independently
needed the exact position fix our RoPE bug describes.** It is confirmation, and a
reference implementation.

### 3.3 It beats the incumbent on our own axes

| | Qwen3-VL-8B (now) | **MiniCPM-o 4.5** |
|---|---|---|
| stream token rate | 185 tok/s | **74 tok/s** (64 vision/frame + 10 audio/s) |
| horizon at equal KV | 1× | **~2.5×** |
| decoder KV dims | 36 L / 8 KV / d128 | **identical** → KV-cost-neutral swap |
| RoPE | 3D mRoPE, fed *linear* positions (a documented approximation in `backend.py`) | **1D RoPE — the approximation disappears** |
| modalities | vision | vision + audio |
| licence | — | Apache-2.0 |

### 3.4 The strategic argument

It is **the rival paper's own model**, open-weight under Apache-2.0. Using it frozen
gives the cleanest experiment available to this project:

> **Same weights. Same modalities. Their *trained* listen/speak control token vs our
> *training-free* architecture.**

If our frozen architecture matches or beats their trained proactivity on their own
model, the "proactivity is architecture, not training" claim stops being an argument
and becomes a controlled result. No other candidate offers this.

---

## 4. Full ranking

| # | Model | Verdict | Reason |
|---|---|---|---|
| **1** | **MiniCPM-o-4_5** (9.37 B, Apache-2.0) | **GO** | stock `Qwen3ForCausalLM`; 74 tok/s; rival's own model |
| 2 | omnivinci (9.2 B, Apache-2.0) | CAVEATS | standalone `llm/`+`vision_tower/`+`sound_tower/` with stock classes; but `max_position_embeddings=32768` |
| 3 | Qwen2.5-Omni-7B | CAVEATS | fits easily (61.8 GiB for 2 copies); **but 32 768 positions ≈ 3 min horizon** |
| 4 | Qwen3-Omni-30B-A3B | CAVEATS | one decoder copy alone = 110 GiB > 95.6 GiB available; MoE *slower* here |
| 5 | Phi-4-multimodal (5.6 B, MIT) | NO-GO | **no video** (image+audio only); Mixture-of-LoRAs modifies the decoder we splice into |
| 6 | Gemma 3n | NO-GO | per-layer embeddings derived from `input_ids` we don't have for vision/audio tokens; sliding-window + KV-shared layers defeat a long-memory thesis |
| 7 | Nemotron-3-Nano-Omni-30B | NO-GO | **23 Mamba2 / 6 attention layers of 52.** SSM state is a fixed-size summary: not croppable, no StreamingLLM eviction, and the probe **irreversibly mutates it**. Needs `mamba-ssm` kernels with no aarch64 wheels. |
| 8 | Ming-flash-omni | NO-GO | ~100 B MoE, 52 shards — cannot hold 2 decoder replicas on one GPU |
| 9 | cosmos3_omni / Step-Audio-2-mini | NO-GO | no audio / no video respectively |
| 10 | VITA-1.5, Ola, Baichuan-Omni-1.5, Megrez-Omni, InternOmni, Ming-Lite-Omni | NO-GO | all `custom_code` for tf 4.37–4.44, broken by the `DynamicCache.key_cache` removal. **VITA-1.5 ships no licence** → unusable for publication. |

**Row 7 is the one to remember.** Nemotron is the silent killer this survey was designed
to catch: a hybrid Mamba/attention model looks fine on every spec sheet, but an SSM's
fixed-size state cannot be cropped, snapshotted or erased — every primitive this
architecture is built on. Any future candidate must be checked for this first.

---

## 5. Hard constraints discovered

### 5.1 These GH200s expose **95.6 GiB**, not 120 GB
`nvidia-smi` reports 97,871 MiB. Several candidates land in the 96–115 GiB band where
this single fact decides the verdict.

### 5.2 `kv_budget = 262144` is **not reachable today** — already true for the incumbent
Primary cache 37.7 GiB + live MVCC snapshot clone 37.7 GiB + weights 15.3 GiB ≈
**112 GiB > 95.6 GiB.** Max feasible ≈ 193 K tokens.

> **The current system's real horizon is ~17 minutes, not unbounded.** Memory binds
> before the RoPE limit (23.6 min) does. Both must be fixed before the unbounded-stream
> claim is defensible. On MiniCPM-o's 74 tok/s the same 193 K tokens buys **~43 min**.

### 5.3 Position horizon is a first-class selection criterion

| model | `max_position_embeddings` | horizon @ its own token rate |
|---|---:|---|
| Qwen3-VL-8B (now) | 262,144 | 23.6 min |
| Qwen3-Omni-30B | 65,536 | ~5.9 min |
| Qwen2.5-Omni-7B | 32,768 | ~3.0 min |
| omnivinci | 32,768 | ~3 min |

**Every omni candidate has a shorter position horizon than the incumbent.** For a paper
whose thesis is *unbounded* streaming, ROADMAP 1.5 is not optional — it gates the entire
extension. MiniCPM-o is the only candidate that already solves it internally.

### 5.4 MoE buys nothing here — I was wrong about this
I predicted Qwen3-Omni-30B-**A3B**'s ~3 B active params might decode *faster* than the
dense 8 B. Measured back-to-back at equal depth on the same contended GPU:
**36.64 ms/layer (MoE) vs 34.98 ms (dense)** — ~5% *slower*, with 48 layers to 36.
Estimated full depth **60–64 ms/tok vs the measured 45**.

The reason is already in our own docs: decode here is **kernel-launch-bound**, and that
is precisely the axis MoE does not help. Fewer active parameters buy nothing when the GPU
is idle waiting on Python. (Also: `batched_mm` experts tried to allocate **192 GiB** at
batch = 1.)

### 5.5 Running the rival's baseline needs a second environment
transformers 5.x removed `DynamicCache.key_cache`/`value_cache`. Harmless for *us* — we
drive `model.llm.model` ourselves and `manager.py` already branches on
`hasattr(cache, "layers")`. But MiniCPM-o's *own* streaming helpers use the removed API,
so reproducing **their** listen/speak baseline needs an env pinned to
`transformers==4.51`. **This is on the critical path for the head-to-head.**

### 5.6 Weights
- MiniCPM-o-4_5: **not cached** — 18.7 GB download (the HF dirs are empty ref stubs)
- Qwen3-Omni-30B-A3B: **cached**, 66 GB
- Audio tower (Qwen): separable, 1.21 GiB, **13 tok/s**; talker+vocoder drop cleanly via
  `enable_audio_output=False` (8,267 checkpoint tensors skipped)

---

## 6. Engineering work required

`backend.py` is the only model-specific file — this is exactly the seam it was built for.

1. **`MiniCPMoBackend(ModelBackend)`** — `embed_text` / `embed_frame` / `embed_audio` /
   `forward`. `forward` targets `model.llm.model`, same signature as today.
2. **Drop the talker** — `init_tts=False` / `enable_audio_output=False`.
3. **New encoder role** — `role="audio"` alongside the existing `"vision"` / `"language"`
   split in `_apply_role`.
4. **Audio ingestion path** — an audio encoder thread feeding the same `vis_q`; the
   ingester stays the sole writer, unchanged. Audio embeds are 2048-dim = thinker hidden,
   so they drop straight into the shared cache.
5. **Delete the mRoPE workaround** — MiniCPM-o is 1D RoPE, so `backend.py`'s documented
   "linear positions into a 3D rotary" approximation disappears.
6. **`manager.py`: no changes.** Verified.
7. **Second env** at `transformers==4.51` for the rival baseline only.

---

## 7. Uncertainties — what is NOT yet proven

Both studies ran while the node was fully occupied (~13 GiB free/GPU), so **neither model
was loaded end-to-end.** Specifically:

- The MiniCPM-o splice test used a **randomly initialised tiny Qwen3**. It proves the API
  contract on 5.12.1; it does **not** prove `from_pretrained(..., trust_remote_code=True)`
  completes on real weights under 5.12.1. The loading path may touch removed 4.x APIs.
  **→ First action: ~20 min download + 5 min load. Cheap, and it is the one thing that
  could still sink the recommendation.**
- Memory figures are **analytic** (apportioned by parameter count), not measured.
- All full-depth ms/tok figures are **estimates**, clearly labelled as such in the source
  reports. An attempt to isolate a per-layer slope produced non-physical negative values
  from contention noise and was discarded rather than reported.
- **Latency is entirely unmeasured for MiniCPM-o.** SigLIP2 at `image_size: 980` could be
  slower per frame than Qwen3-VL's ViT despite emitting fewer tokens.

---

## 8. Recommended sequence

**Not now.** The method is not yet proven on the easy case, and adding a modality to an
unproven method doubles the debugging surface. Order:

| when | action |
|---|---|
| Phase 1–4 | **Vision-only.** Fix the method: schema decoder, dense metric, trigger split, memory, the high-sample tasks. |
| now, cheap | **Download MiniCPM-o (18.7 GB) and confirm it loads under 5.12.1.** The one open question that could invalidate the plan; costs ~25 min and no research time. |
| before any swap | **ROADMAP 1.5** — position re-basing. Every omni candidate has a *shorter* horizon than the incumbent; without this the extension makes the unbounded claim worse, not better. |
| Phase 5 | Backend swap, audio ingestion, re-run on `audio=none+helpful` (932 samples, 34.5%). |
| Phase 5 | Second env at tf 4.51; run **their** listen/speak baseline on **their** weights for the controlled head-to-head. |

**Fallback if the swap fails:** the vision-only paper stands on its own. The extension is
additive, never a replacement — that is why it is scheduled late and behind a kill
criterion rather than built into the critical path.
