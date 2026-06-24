# async-omni v2 — modular, Qwen3-VL-8B

Same idea as the Mobile-O prototype, rewritten **modularly** and pointed at a
bigger model (`Qwen/Qwen3-VL-8B-Instruct`) so the gating signal and the
commentary are actually good. One model, one linear KV cache, three async
threads. **Training-free.**

> ⚠️ **UNTESTED on the author's machine** (GPU was full + model not downloaded +
> needs newer `transformers`). The async machinery is identical to the
> Mobile-O version we verified end-to-end; the only new/unverified code is
> `backend.py` (the Qwen3-VL adapter). See **Validation** below.

## The parts (one responsibility per file)

| file | what it owns |
|------|--------------|
| `config.py` | every knob + all prompts, one dataclass |
| `backend.py` | **the only model-specific file** — Qwen3-VL load, text/frame embedding, decoder forward. Swap models = new backend. |
| `manager.py` | the ONE shared KV cache + the single GPU-owning worker thread; priority queue; eviction; self-erasing probe. Model-agnostic. |
| `proactivity.py` | the yes/no proactivity score `P(yes)/(P(yes)+P(no))` |
| `vision_stream.py` | PyAV decode thread + real-time pacing clock (the pacemaker) |
| `orchestrator.py` | the thinker: ingest, vision gate, evict, writer gate, trigger |
| `writer.py` | the mouth: token-by-token commentary at low priority + anti-repeat |
| `run.py` | CLI + wires the three threads |

## Architecture recap (why it's shaped like this)

- **One model, one linear KV cache.** Orchestrator (thinker) and writer are two
  roles writing into the *same* growing sequence. No dual-view RoPE kernel — we
  keep a single linear timeline (see Mobile-O `lessons_learned.md` §D).
- **A single manager thread** is the only code that touches the GPU/cache.
  Everyone else `submit(prio, op)`s and waits. `PRIO_ORCH < PRIO_WRITER` ⇒ the
  writer can never block the thinker.
- **Two clocks:** `next_pos` (logical RoPE position, monotonic) vs physical
  cache length. Eviction (StreamingLLM-style: keep a system-prompt sink + a
  recent window) shrinks the physical cache but `next_pos` keeps running, so the
  recent window stays positionally contiguous with new tokens. → bounded memory
  on an unbounded video.
- **Proactivity = a self-erasing yes/no probe** (AsyncReasoning §3.2): splice a
  question, read the lean, truncate it back out of the cache. Two gates: vision
  ("look closer?") and writer ("did a goal happen?").

## Run (on the GPU server)

```bash
pip install -r requirements.txt
# edit PY / VIDEO in run.sh, then:
./run.sh            # real-time, 2x
./run.sh batch      # as-fast-as-GPU
# or call run.py directly; every config field is a --flag:
python run.py --model_id Qwen/Qwen3-VL-8B-Instruct --video_path "<clip>.webm" \
  --dtype float16 --fps 1.5 --max_seconds 120 --kv_budget 4096 \
  --goal_threshold 0.5 --realtime --speed 2.0 --log_gate_every 1
```

float16 (~16 GB for 8B weights) fits on a 24 GB+ GPU; the KV cache adds a bit.

## Validation (do this first, in order)

The risk is concentrated in `backend.py`. Validate it standalone before trusting
the full run:

1. **Module resolution + a forward.** Tiny script: build `Qwen3VLBackend(cfg)`,
   then `mgr.op_seed("hi")`, `mgr.op_probe(".. yes or no: ")`,
   `mgr.op_ingest_frame(some_PIL)`, `mgr.op_append_text(" The")`. If
   `_resolve_modules` raises, fix the attribute paths for your transformers
   version (it lists what it tried).
2. **Positions.** We feed *linear* mRoPE positions (`[3,1,L]`, all axes equal).
   If understanding is weak, switch to true positions via the model's
   `get_rope_index` per chunk — this is the most likely thing to need work.
3. **Frame token count.** Print `embed_frame(img).shape[1]`; set `kv_budget` to
   hold ~4–6 frames (`budget ≈ tokens_per_frame * 5`).
4. Then run the full pipeline and watch `orch.ggate yes_share` — it should
   *spike* at goals, not sit flat.

## Known simplifications / next levers
- Linear mRoPE positions (see Validation #2).
- Writer uses temperature sampling + a crude anti-repeat; tune in `config.py`.
- Goal gate is absolute-thresholded; a running-baseline / spike detector would
  be more robust (port from the Mobile-O `lessons_learned.md` "next levers").
