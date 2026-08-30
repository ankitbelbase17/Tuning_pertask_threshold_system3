# FORK: system3_qwem_omni — Qwen2.5-Omni-7B backend + audio + TMRoPE

**Base:** `system_3` @ `ab49ce9` (git history preserved; `git log` is identical
up to this point). **Date:** 2026-08-24. **Status:** implemented and
structurally verified against the real `transformers==5.12.1` source in
`prosync_env`; **NOT** yet run end-to-end on real downloaded weights (see
Uncertainty section — this is consistent with how big a first action that
still is per `OMNI_EXTENSION.md`).

This document is the single place that says what changed and why. Read it
before touching `backend.py`, `manager.py`, or `input_ingester.py` in this
checkout — those three carry the load-bearing changes.

---

## 1. What changed, file by file

| file | change |
|---|---|
| `config.py` | `model_id` → `Qwen/Qwen2.5-Omni-7B`; new `backend` selector; new AUDIO block (`use_audio`, `audio_sampling_rate`, `audio_position_id_per_seconds=25`, `audio_seconds_per_chunk=2.0`); new TMRoPE block (`tmrope_positions`); new `model_max_position_embeddings=32768` + its rationale (see §3). |
| `backend.py` | New `Qwen2_5OmniBackend(ModelBackend)` class (mirrors `Qwen3VLBackend`'s shape, does not inherit from it). New module-level `tmrope_position_ids()` — pure, independently testable position-tensor builder. `forward()` (on **both** backend classes) gained an optional `position_ids=` override, default-`None` behaviour unchanged. `embed_frame()` (on **both** classes) now returns `(embeds, grid_hw)` instead of just `embeds`. |
| `manager.py` | `KVCacheManager` gained `chunk_anchor_pos` / `chunk_anchor_vt` state. `ingest()` / `_forward_primary()` gained optional `token_kind=`/`real_seconds=`/`grid_hw=` (default `"text"`/`None`/`None` — every pre-existing call site is a no-op change). New loud RoPE-horizon guard (see §3). |
| `input_ingester.py` | Implements **Option A** synchronous audio ingestion (see §2). Unpacks the new `(vt, embeds, grid_hw)` tuple from `vis_q`. Raises loudly if `use_audio=True` but the loaded backend has no `embed_audio`. Flushes a final partial audio chunk on stream end. |
| `vision_stream.py` | Unpacks `embeds, grid_hw = backend.embed_frame(img)`; threads `grid_hw` through `vis_q` as a 3-tuple. |
| `audio_io.py` | **New file.** `AudioChunkReader` — on-demand raw-audio chunk extraction via `av`, called synchronously by the ingester. Not a thread. |
| `run.py` | Backend selection via `cfg.backend` (`"qwen2_5_omni"` default, `"qwen3_vl"` kept for an explicit A/B). |

Everything else (`controller.py`, `writer.py`, `proactivity.py`, `prompts.py`,
`util.py`) is **byte-identical** to the parent — confirmed by `py_compile`
over the whole package after the edits above.

---

## 2. Audio ingestion: "Option A", and why

Two designs were considered for feeding a second modality into the single
shared cache:

- **A separate free-running audio-encoder thread** (symmetric with
  `vision_stream.py`). Rejected: audio (~160ms/chunk, fixed cadence) and
  vision (variable per-frame latency) are two independent producers with no
  ordering guarantee between their queues. Without a real timestamp-merge,
  audio can land in the cache out of true-time order relative to video —
  which breaks the exact property the project's own deterministic-lockstep
  design exists to guarantee ("every tick sees exactly frames `[0..vt]`,
  identically on every run").
- **Option A (implemented):** no second thread. The ingester — already the
  sole writer of the primary cache — is also the *only* caller of
  `backend.embed_audio()`, invoked synchronously every
  `cfg.audio_seconds_per_chunk` (2s) of video time, in the same control flow
  as frame ingestion. There is nothing to race, because there is no second
  producer. The cost is that the audio encoder's latency lands on the
  per-tick critical path rather than being hidden in a parallel thread — a
  deliberate throughput-for-correctness trade, not an oversight.

`audio_seconds_per_chunk=2` and `audio_position_id_per_seconds=25` are **not
free parameters** — they are `Qwen2_5OmniConfig`'s own defaults, confirmed
from the real `transformers` source, not the model card. Changing them
desyncs ingestion from what the frozen model was actually trained on.

---

## 3. Positions (TMRoPE): what's exact, what's a documented simplification

Verified directly against `Qwen2_5OmniThinkerForConditionalGeneration
.get_rope_index` in the real modeling file (not guessed):

- Text block: `temporal = height = width = arange(L) + start_idx` — this was
  **already** what the parent's flat-linear hack does; no change needed.
- Vision block (one frame): temporal is **constant** across the frame (one
  frame = one instant), height/width are the true 2D patch grid.
- Audio block: `temporal = height = width = arange(L) + start_idx` — audio
  already emits at exactly `position_id_per_seconds` (25) tokens/sec, so a
  plain per-token counter *is* the correct time axis.

**Where this implementation deliberately diverges from the reference, and
why:** `get_rope_index` is a *batch* function — it assumes the whole video's
`video_grid_thw` and full `audio_seqlens` are known up front, and it
re-derives its `start_idx` baseline by scanning the complete position list
built so far. This pipeline ingests one frame, and separately one ~2s audio
chunk, at a time, online, with no lookahead. Instead of one global scan,
`KVCacheManager` keeps a *rolling* anchor (`chunk_anchor_pos`,
`chunk_anchor_vt`) reset only at each audio-chunk boundary — walking the same
mechanism forward incrementally rather than computing it from a complete
sequence in one shot. **Verified numerically** (see `git log` for the
validation transcript, reproducible with the snippet in `backend.py`'s
`tmrope_position_ids` docstring): a second frame later in the same open
chunk lands at a strictly greater temporal position than the first; an audio
chunk closing a window starting at `chunk_anchor_pos=8` for 50 tokens (2s ×
25/s) produces exactly the range `8..57`.

**The one correctness risk this surfaced that the parent didn't have:**
Qwen2.5-Omni-7B's real `max_position_embeddings` is **32768** — confirmed
from `Qwen2_5OmniTextConfig()`'s default, not the model card — vs. the
parent's 262144, which not-coincidentally equals its own `kv_budget`. Since
`kv_budget` is left at 262144 here (Qwen2.5-Omni is the one candidate that
fits the *full* spec at that budget, per `OMNI_FEASIBILITY.md`), **the
position clock will very plausibly outlive the model's trained RoPE range
long before eviction ever fires to bound it.** `manager.py` now carries a
loud, one-time warning for exactly this (mirroring the existing ROADMAP-1.5
eviction guard's own pattern) — it does not fix it. The real fix is the same
one `OMNI_EXTENSION.md` §5.3 already names as a blocking prerequisite for the
*whole* omni extension: RoPE position re-basing on eviction (what MiniCPM-o
does internally via `realign_rotary_suffix`). **Not implemented in this
fork** — flagged loudly instead of silently, per this project's own standing
rule.

---

## 4. What was verified, and how (no weights downloaded)

Mirrors the methodology `OMNI_FEASIBILITY.md` and `OMNI_MODEL_SURVEY.md`
themselves used — tiny random-weight instances of the *real* transformers
classes, on CPU, no download:

1. **Full audio pipeline, end to end:** `WhisperFeatureExtractor(sampling_rate=
   16000, feature_size=128)` on a real 2-second waveform → reshaped per
   `get_audio_features`'s own convention → run through a real (tiny-weight)
   `Qwen2_5OmniAudioEncoder` → **exactly 50 output tokens**, i.e.
   `25 tok/s × 2s`, confirmed arithmetically.
2. **The manager.py cache-surgery contract, with real TMRoPE positions,**
   through a real (tiny-weight) `Qwen2_5OmniThinkerTextModel`: seed (text) →
   ingest a fake vision chunk (4×4 grid, real 3D positions) → a second frame
   in the same open chunk (temporal position verified strictly greater) →
   ingest a fake audio chunk (50 tokens, position range verified exact) →
   probe-style splice + `cache.crop()` erase → MVCC `copy.deepcopy()`
   snapshot. All six steps passed under `torch.no_grad()`.
3. **Real audio demuxing** (`audio_io.AudioChunkReader`) against three actual
   dataset videos (`omnipro_data/raw_videos/*.mp4`), each confirmed (via
   `av` directly) to carry a real audio stream: extracted 2-second chunks
   with plausible, varying RMS (0.10–0.30, not silence, not clipped noise).
4. **A real, live bug this validation caught in itself:** `AudioChunkReader`'s
   first draft silently treated `import av` failing (a broken
   `prosync_env` — `libXau` is missing from its link path right now, a
   known-fragile env per this project's own `LEARNINGS.md`/`EXPERIMENTS.md`)
   identically to "this video has no audio track." Fixed to log the two
   cases distinctly and loudly — the exact failure class this project's
   whole documentation culture exists to catch. **`prosync_env`'s `av` import
   is currently broken on this node**; this affects `vision_stream.py` too
   (pre-existing, not introduced by this fork) — workaround confirmed:
   `LD_LIBRARY_PATH` must include
   `.../prosync_env/lib/python3.12/site-packages/pillow.libs` until the env
   itself is repaired.

## 5. What is NOT verified (be honest about this before trusting a run)

- **No real weights have been loaded.** `Qwen2_5OmniForConditionalGeneration
  .from_pretrained("Qwen/Qwen2.5-Omni-7B", ...)` has never been executed —
  per `OMNI_EXTENSION.md`'s own recommended sequence, that ~20GB download +
  load is the correct *first* action before trusting anything else, and it
  was intentionally left for a deliberate, explicit step rather than done
  silently inside this change.
- **Generation quality is entirely unmeasured.** Whether the existing
  per-task ICL prompts (tuned for Qwen3-VL-8B) transfer to this backbone is
  unknown, and every task prompt still carries the "VISUAL-ONLY, no audio"
  disclaimer from the parent — see `prompts.py`'s module docstring. Two tasks
  (`semantic_condition_alert`, `instant_event_alert`) are defined by OmniPro
  as audio-first; their ICL blocks were NOT rewritten by this fork and still
  actively tell the model to ignore audio content it will now actually
  receive. Fixing this is prompt work, not code, and is the natural next
  step once a real run is possible.
- **The RoPE-horizon risk in §3 is flagged, not fixed.** A long enough
  stream (roughly `32768 / (74 tok/s combined vision+audio rate)` ≈ 7-8
  minutes, before the model's own attention starts operating outside its
  trained range) will silently degrade in quality past that point — "loud"
  here means a log line, not a correctness fix.
- **`_apply_role`'s 3-way vision/audio/language split is extended but
  untested on real weights**, same caveat the parent's own vision/language
  split already carried before this fork (multi-GPU replicas are not wired
  into `run.py`'s single-manager pipeline).
- **`AudioChunkReader`'s per-chunk re-seek-and-decode** is a real but
  bounded imprecision (see its own docstring) — cheap and simple, not
  byte-exact windowing. Flagged there, not silently assumed exact.

## 6. First next action

Exactly what `OMNI_EXTENSION.md` already recommends, unchanged by this fork:
download the ~18-20GB of weights and confirm
`Qwen2_5OmniForConditionalGeneration.from_pretrained(cfg.model_id,
enable_audio_output=False)` completes under transformers 5.12.1 — no
`trust_remote_code` needed, confirmed: `Qwen2_5OmniForConditionalGeneration`
is a NATIVE top-level `transformers` class in 5.12.1, not custom/remote code
(unlike MiniCPM-o, which is `trust_remote_code=True`). That is the one thing
everything above could not check without a real download, and it is cheap
(network + disk time, no GPU-hours).
