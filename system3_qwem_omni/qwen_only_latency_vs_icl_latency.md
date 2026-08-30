Raw Qwen3-VL language-only decode (isolated, idle GH200, KV cache preloaded)

┌────────────────────────────────────────────────────────┬──────────┬───────┐
│                          what                          │ ms/token │ tok/s │
├────────────────────────────────────────────────────────┼──────────┼───────┤
│ trunk only (36 decoder layers, no lm_head)             │ 43–53    │ ~21   │
├────────────────────────────────────────────────────────┼──────────┼───────┤
│ + lm_head + GPU argmax                                 │ 45–49    │ ~21   │
├────────────────────────────────────────────────────────┼──────────┼───────┤
│ + our old CPU-logits round-trip (the bug)              │ 78–91    │ ~12   │
├────────────────────────────────────────────────────────┼──────────┼───────┤
│ theoretical memory-bound floor (16GB weights / ~3TB/s) │ ~5       │ ~200  │
└────────────────────────────────────────────────────────┴──────────┴───────┘

Flat across cache lengths 1k/8k/24k → not attention/memory-bound. The raw model's ~45 ms/tok is ~9× above the hardware floor — it's per-call kernel-launch/framework overhead (36 layers of small kernels through the HF Python stack, every token).

Our ICL pipeline's decode (same model, in the live 3-thread system)

┌───────────────────────────────────────────────────┬──────────────────────────────────────────────────┐
│                       stage                       │                     measured                     │
├───────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
│ controller decode in-pipeline (after the GPU fix) │ ~105–120 ms/tok (mean, telemetry over full runs) │
├───────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
│ its prompt prefill per tick                       │ ~0.2 s (negligible)                              │
├───────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
│ KV snapshot per tick                              │ 5–18 ms (negligible)                             │
└───────────────────────────────────────────────────┴──────────────────────────────────────────────────┘

So the ICL version pays the raw ~45–94 ms/tok plus a contention tax from the encoder/ingester sharing the GPU (and the multi-GPU split didn't remove it — the per-tick cross-GPU cache copy ate the gain).

What we already made faster vs what's left

┌─────────────────────────────────────────┬────────────────────────┬────────────────────────────────────────────────────────────────────────────────┐
│                  lever                  │          gain          │                                     status                                     │
├─────────────────────────────────────────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
│ kill CPU logit round-trip → GPU         │ 149→94 ms/tok (1.6×)   │ ✅ implemented, committed                                                      │
│ sampling                                │                        │                                                                                │
├─────────────────────────────────────────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
│ skip lm_head on ingest                  │ free win on every      │ ✅ implemented                                                                 │
│                                         │ frame                  │                                                                                │
├─────────────────────────────────────────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
│ diff-decoding (your idea) — quiet ticks │ ~4× fewer tokens/tick  │ ✅ implemented                                                                 │
│  ~14 tokens                             │                        │                                                                                │
├─────────────────────────────────────────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
│ reuse position tensors                  │ 94→84 ms/tok (+10%)    │ not yet (trivial)                                                              │
├─────────────────────────────────────────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
│ StaticCache + CUDA graphs               │ est. 45→~10–15 ms/tok  │ ❌ blocked: torch.compile failed on our growing DynamicCache (verified) —      │
│                                         │ (3–4×)                 │ needs a preallocated cache rewrite                                             │
├─────────────────────────────────────────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
│ flash-attention-2                       │ —                      │ ❌ not built for this aarch64 env; sdpa is our best                            │
└─────────────────────────────────────────┴────────────────────────┴────────────────────────────────────────────────────────────────────────────────┘

Bottom line: raw language-only Qwen3-VL on this stack decodes at ~45 ms/tok (best achievable without CUDA graphs); our ICL controller runs at ~105–120 ms/tok in-pipeline. The remaining big win (~3–4×, to ~15 ms/tok → a full control JSON in <0.5s) is the StaticCache + CUDA-graphs rewrite — real engineering on the cache manager, and the only untapped lever left.
