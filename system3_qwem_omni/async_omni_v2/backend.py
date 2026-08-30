"""
backend.py — the model adapter (the ONLY model-specific file).

Everything else in this project is model-agnostic. To support a different
VLM you write a new ModelBackend subclass; nothing else changes.

A backend must provide:
  * tokenizer handles: `eos_id`, `newline_ids`, plus `decode(ids)`.
  * `embed_text(text)  -> [1, L, H]`   text -> input embeddings
  * `embed_frame(pil)  -> ([1, N, H], (rows, cols))`   one image -> N
        vision-token embeddings, PLUS the post-merge patch grid shape
        (rows*cols == N). The grid shape is needed by any backend that builds
        real per-token 3D positions (TMRoPE) rather than the flat-linear
        approximation — see `tmrope_position_ids` below. A backend that never
        builds real positions may return a placeholder grid; the caller
        (input_ingester.py) only uses it when `cfg.tmrope_positions` is set.
  * `forward(embeds, cache, pos_start, phys_start, position_ids=None) ->
        (logits[V], cache)`
        run the text decoder over `embeds`, appending to `cache`, and return
        the LAST position's logits (CPU). `pos_start` is the logical RoPE
        position of the first new token; `phys_start` is the physical cache
        write index (these differ after eviction — see manager.py).
        `position_ids`, when given, overrides the default flat-linear [3,1,L]
        tensor built from `pos_start` (see `tmrope_position_ids`).

------------------------------------------------------------------------------
Qwen3VLBackend NOTE (UNTESTED on this machine — validate on the GPU server):
  * mRoPE positions are 3D (temporal, height, width). We feed *linear* positions
    (all three axes = sequential index), which is shape-correct for the mRoPE
    rotary but ignores the 2D image geometry. This is the one deliberate
    simplification; it mirrors the linear-cache approach we validated on
    Mobile-O. If understanding quality is poor, the upgrade is to compute true
    positions via the model's `get_rope_index` per appended chunk.
  * Attribute layout (visual / language_model / lm_head) is resolved
    defensively because it has moved between transformers versions. If
    resolution fails you get a clear error listing what was tried.
------------------------------------------------------------------------------
"""
from abc import ABC, abstractmethod
import torch


class ModelBackend(ABC):
    eos_id: int
    newline_ids: set
    hidden_size: int

    @abstractmethod
    def embed_text(self, text: str) -> torch.Tensor: ...
    @abstractmethod
    def embed_frame(self, pil_image) -> torch.Tensor: ...
    @abstractmethod
    def forward(self, embeds, cache, pos_start, phys_start): ...
    @abstractmethod
    def decode(self, ids) -> str: ...


def _word_ids(tok, words):
    """Token ids for single-token surface forms (e.g. the newline stop token)."""
    out = []
    for w in words:
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            out.append(ids[0])
    return out


# ==============================================================================
# TMRoPE (FORK: system3_qwem_omni) — real per-token 3D positions.
#
# Verified against transformers==5.12.1's real
# Qwen2_5OmniThinkerForConditionalGeneration.get_rope_index (not guessed):
#   - position_id_per_seconds = 25   (Qwen2_5OmniConfig default)
#   - seconds_per_chunk       = 2    (Qwen2_5OmniConfig default)
#   - TEXT block:   temporal = height = width = arange(L) + start_idx
#                   (start_idx = "one past the previous block's max position")
#   - VISION block (single frame, grid_t=1 in the reference's own terms):
#                   temporal = start_idx + round(local_seconds * 25)   [CONSTANT
#                       across the whole frame -- one frame = one instant]
#                   height   = start_idx + row_index  (0..H-1, tiled over cols)
#                   width    = start_idx + col_index  (0..W-1, tiled over rows)
#   - AUDIO block (L tokens from one embed_audio() call, ~25 tok/s so ONE
#     token IS one temporal-position tick):
#                   temporal = height = width = arange(L) + start_idx
#
# WHERE THIS DIVERGES FROM THE REFERENCE, DELIBERATELY, AND WHY:
# get_rope_index is a BATCH function: it runs once over a fully-known input_ids
# sequence containing an entire video's video_grid_thw (many frames processed
# as ONE multi-frame vision block) plus the full audio_seqlens up front, and
# it re-derives `start_idx` by scanning `llm_pos_ids_list[-1].max() + 1` after
# each completed block. We ingest ONE frame and, separately, one ~2s audio
# chunk at a time, online, with no lookahead -- there is no "whole video" to
# batch. So instead of one global scan, `KVCacheManager` keeps a *rolling*
# anchor: `chunk_anchor_pos` (= start_idx for whatever chunk is open) and
# `chunk_anchor_vt` (the real video-time the chunk opened at), reset every
# `audio_seconds_per_chunk` real seconds -- i.e. exactly the reference's own
# chunk boundary, just walked forward incrementally instead of computed from
# a complete sequence in one shot. Within one open chunk, every frame's
# `local_seconds = vt - chunk_anchor_vt` plays the role of the reference's
# `t_index`. This preserves the property that actually matters -- video and
# audio describing the same ~2s window land at comparable, chunk-local
# position values -- without requiring the whole stream to be known in
# advance. It is a documented simplification, exactly like Qwen3VLBackend's
# existing "linear positions, ignores image geometry" one (see that class's
# docstring) -- not a silent approximation.
# ==============================================================================
def tmrope_position_ids(kind, L, chunk_anchor_pos, local_seconds=0.0,
                        grid_hw=None, per_second=25, device=None):
    """Build a [3, 1, L] position tensor plus the manager's next running
    baseline, for one appended chunk of tokens.

    kind: "text" | "vision" | "audio"
      text   -- ordinary continuation, identical to the old linear hack.
      vision -- ONE frame (L == grid_hw[0]*grid_hw[1] patch tokens): constant
                temporal (this frame IS one instant), true 2D spatial grid for
                height/width. `local_seconds` = elapsed time since the current
                chunk opened (`vt - chunk_anchor_vt`), NOT absolute video time
                -- see module docstring above.
      audio  -- L tokens from one embed_audio() chunk; already emitted at the
                model's own 25 tok/s, so a plain per-token counter already
                equals the intended time axis (round-trip exact, no rounding
                needed).

    Returns (position_ids [3,1,L], next_chunk_anchor_pos).
    `next_chunk_anchor_pos` is NOT a chunk reset by itself -- see
    KVCacheManager.ingest(): the anchor only resets on an audio-chunk
    boundary, matching `audio_seconds_per_chunk`. This return value is simply
    "the next valid position baseline for whatever is appended right after
    this block," which for vision may still be inside the same open chunk.
    """
    if kind == "text":
        lin = torch.arange(chunk_anchor_pos, chunk_anchor_pos + L, device=device)
        pos = lin.view(1, 1, L).expand(3, 1, L).contiguous()
        return pos, chunk_anchor_pos + L

    if kind == "vision":
        if grid_hw is None:
            raise ValueError("tmrope_position_ids(kind='vision') needs grid_hw=(H, W)")
        H, W = grid_hw
        if H * W != L:
            raise ValueError(f"grid_hw {grid_hw} does not multiply to L={L}")
        t_local = round(local_seconds * per_second)
        t = torch.full((L,), chunk_anchor_pos + t_local, dtype=torch.long, device=device)
        h = (torch.arange(H, device=device).repeat_interleave(W) + chunk_anchor_pos)
        w = (torch.arange(W, device=device).repeat(H) + chunk_anchor_pos)
        pos = torch.stack([t, h, w]).unsqueeze(1)          # [3, 1, L]
        # one past this frame's own temporal tick -- NOT a chunk reset; a
        # later frame in the SAME open chunk still measures local_seconds
        # from the same chunk_anchor_vt, so it can land at the same or a
        # slightly larger t_local, never a smaller one (monotonic, per real
        # elapsed time within the chunk).
        return pos, chunk_anchor_pos + t_local + 1

    if kind == "audio":
        lin = torch.arange(chunk_anchor_pos, chunk_anchor_pos + L, device=device)
        pos = lin.view(1, 1, L).expand(3, 1, L).contiguous()
        return pos, chunk_anchor_pos + L

    raise ValueError(f"unknown token kind {kind!r} (want text/vision/audio)")


class Qwen3VLBackend(ModelBackend):
    def __init__(self, cfg, role="full"):
        # role decides which half of the model to KEEP resident (for multi-GPU):
        #   "full"     -> vision + language (single-GPU: does everything)
        #   "vision"   -> only the ViT/merger (an encoder-only replica)
        #   "language" -> only the decoder + lm_head (ingester/controller replica)
        # The unused half is dropped after load; results are identical because each
        # role never calls the half it drops. device = which GPU this replica lives on.
        from transformers import AutoProcessor
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[cfg.dtype]
        self.device, self.dtype, self.cfg, self.role = cfg.device, dtype, cfg, role
        print(f"[backend] loading {cfg.model_id} ({cfg.dtype}, role={role}) on {cfg.device} ...", flush=True)

        # Qwen3-VL needs a recent transformers; class name may be
        # Qwen3VLForConditionalGeneration. Fall back to the generic loader.
        try:
            from transformers import Qwen3VLForConditionalGeneration as VLM
        except Exception:
            from transformers import AutoModelForImageTextToText as VLM
        # full -> load straight to the GPU; vision/language -> load to CPU then move
        # only the kept half to the GPU (peak = kept half).
        self.model = VLM.from_pretrained(
            cfg.model_id, torch_dtype=dtype,
            device_map=(cfg.device if role == "full" else None),
            low_cpu_mem_usage=True).eval()
        self.processor = AutoProcessor.from_pretrained(cfg.model_id)
        self.tok = self.processor.tokenizer
        self._cap_image_resolution(cfg.max_pixels)

        self._resolve_modules()        # find visual / language_model / lm_head
        self.hidden_size = self.model.config.get_text_config().hidden_size \
            if hasattr(self.model.config, "get_text_config") else self.model.config.hidden_size

        self.eos_id = self.tok.eos_token_id
        self.newline_ids = set(_word_ids(self.tok, ["\n"]))
        # yes/no single-token ids for the PROBE-GATE (gate_mode="probe") yes-share
        self.yes_ids = _word_ids(self.tok, getattr(cfg, "yes_words", ["yes", "Yes", " yes", " Yes"]))
        self.no_ids = _word_ids(self.tok, getattr(cfg, "no_words", ["no", "No", " no", " No"]))

        self._apply_role(role)         # drop the unused half + free its GPU memory

    # -- keep only the half this role needs; free the rest; move kept half to GPU --
    def _apply_role(self, role):
        import gc
        if role == "full":
            return                      # already loaded straight to the GPU
        m = self.model
        inner = m.model if hasattr(m, "model") else m   # Qwen3VLModel (.visual/.language_model)
        if role == "vision":            # encoder-only: drop decoder + lm_head
            self.language_model = self.lm_head = self.embed_tokens = None
            if hasattr(m, "lm_head"):
                m.lm_head = None
            if hasattr(inner, "language_model"):
                inner.language_model = None
            gc.collect()
            inner.visual.to(self.device)                # move ONLY the ViT to GPU
        elif role == "language":        # decoder-only: drop the vision tower
            self.visual = self.get_image_features = None
            if hasattr(inner, "visual"):
                inner.visual = None
            gc.collect()
            inner.language_model.to(self.device)        # move ONLY the decoder to GPU
            if getattr(m, "lm_head", None) is not None:
                m.lm_head.to(self.device)
        gc.collect()
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    # -- cap vision tokens/frame by limiting image pixels (perf + KV memory) --
    def _cap_image_resolution(self, max_pixels):
        """For Qwen2VLImageProcessor the pixel-AREA bounds live in
        `size.longest_edge` (max pixels) / `size.shortest_edge` (min pixels) --
        these are total-pixel counts, not edge lengths. Vision tokens =
        pixels / (patch_size*merge_size)^2, so capping longest_edge caps the
        token count. Confirmed by the profiler's vis_tokens_per_frame sample."""
        if not max_pixels:
            return
        ip = getattr(self.processor, "image_processor", None)
        if ip is None:
            return
        per_tok = (getattr(ip, "patch_size", 16) * getattr(ip, "merge_size", 2)) ** 2
        size = getattr(ip, "size", None)

        def _set(obj, key, val):            # SizeDict supports both attr + item
            try:
                setattr(obj, key, val)
            except Exception:
                pass
            try:
                obj[key] = val
            except Exception:
                pass

        if size is not None:
            cur_min = getattr(size, "shortest_edge", None) or (size.get("shortest_edge")
                      if hasattr(size, "get") else None) or per_tok
            _set(size, "longest_edge", max_pixels)
            _set(size, "shortest_edge", min(cur_min, max_pixels))
        # legacy flat attrs, if a future processor uses them
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = max_pixels
        # also push onto the processor wrapper so apply-chat-template paths agree
        for attr in ("max_pixels",):
            if hasattr(self.processor, attr):
                setattr(self.processor, attr, max_pixels)
        print(f"[backend] capped image to <= {max_pixels} px "
              f"(~{max_pixels // per_tok} vision tokens/frame)", flush=True)

    # -- module resolution: tolerate transformers' shifting attribute layout --
    def _resolve_modules(self):
        m = self.model
        cand_lm = [lambda: m.model.language_model, lambda: m.language_model,
                   lambda: m.model]
        cand_vis = [lambda: m.model.visual, lambda: m.visual,
                    lambda: m.model.vision_tower]
        self.language_model = _first(cand_lm, "language model (text decoder)")
        self.visual = _first(cand_vis, "vision tower")
        self.lm_head = m.lm_head if hasattr(m, "lm_head") else m.get_output_embeddings()
        self.embed_tokens = self.model.get_input_embeddings()
        # preferred high-level image->features helper if present
        self.get_image_features = getattr(m, "get_image_features", None)

    # ---------------------------------------------------------------- text
    def embed_text(self, text):
        ids = self.tok(text, return_tensors="pt",
                       add_special_tokens=False).input_ids.to(self.device)
        return self.embed_tokens(ids)               # [1, L, H]

    def embed_token(self, tok_id):
        ids = torch.tensor([[tok_id]], device=self.device)
        return self.embed_tokens(ids)               # [1, 1, H]

    # --------------------------------------------------------------- vision
    @torch.no_grad()
    def embed_frame(self, pil_image):
        feat = self.processor.image_processor(images=[pil_image], return_tensors="pt")
        pixel_values = feat["pixel_values"].to(self.device, self.dtype)
        grid_thw = feat["image_grid_thw"].to(self.device)
        if self.get_image_features is not None:
            out = self.get_image_features(pixel_values=pixel_values, image_grid_thw=grid_thw)
        else:
            out = self.visual(pixel_values, grid_thw=grid_thw)
        # transformers 5.x returns an output object whose `pooler_output` holds
        # the image embeds (a per-image tuple); older versions returned a plain
        # tensor or a tuple. Unwrap both forms to a single [N, H] tensor.
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        embeds = out[0] if isinstance(out, (list, tuple)) else out
        embeds = embeds.reshape(1, -1, self.hidden_size)               # [1, N, H]
        # post-merge patch grid (rows, cols); grid_thw is [t, h, w] in
        # PRE-merge patch units, t=1 for a single frame. FORK: only used by
        # tmrope_position_ids downstream -- this backend still defaults to
        # the flat-linear hack unless cfg.tmrope_positions asks otherwise.
        merge = getattr(self.processor.image_processor, "merge_size", 2)
        _, gh, gw = [int(x) for x in grid_thw[0].tolist()]
        grid_hw = (gh // merge, gw // merge)
        return embeds.to(self.dtype), grid_hw

    # -------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, embeds, cache, pos_start, phys_start, want_logits=True,
                position_ids=None):
        """Run the decoder over `embeds`, append to `cache`, return (logits, cache).

        want_logits=False skips the lm_head entirely (the 151k-vocab projection):
        the INGESTER only prefills frames into the cache and never reads logits, so
        this saves a large matmul on every ingested chunk.

        When want_logits=True the returned logits stay ON THE GPU (float, last
        position only). The old code shipped the full 151k-vocab vector to CPU
        every token (~35-45 ms/tok of pure D2H sync); the caller now samples on GPU
        and only the chosen token id crosses the bus.

        position_ids: optional pre-built [3,1,L] tensor (e.g. from
        backend.tmrope_position_ids). Callers that don't pass one keep getting
        the original flat-linear behaviour below -- this parameter is additive,
        not a breaking change to any existing call site."""
        L = embeds.shape[1]
        cache_position = torch.arange(phys_start, phys_start + L, device=self.device)
        if position_ids is None:
            # linear positions, shaped [3, 1, L] for the 3D mRoPE rotary (all axes
            # equal -> behaves like ordinary 1D RoPE). See module docstring.
            lin = torch.arange(pos_start, pos_start + L, device=self.device)
            position_ids = lin.view(1, 1, L).expand(3, 1, L).contiguous()
        else:
            position_ids = position_ids.to(self.device)
        out = self.language_model(
            inputs_embeds=embeds.to(self.device, self.dtype),
            past_key_values=cache,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        if not want_logits:
            return None, out.past_key_values
        logits = self.lm_head(out.last_hidden_state[:, -1, :])
        return logits[0].float(), out.past_key_values                  # GPU (was .cpu())

    def decode(self, ids):
        return self.tok.decode(ids).strip()


class Qwen2_5OmniBackend(ModelBackend):
    """Qwen2.5-Omni-7B backend (FORK: system3_qwem_omni).

    Verified structurally against the real transformers==5.12.1 source in
    `prosync_env` (attribute names, forward signatures, and the full audio
    pipeline end-to-end on tiny random-weight instances -- no download
    needed, mirroring the methodology OMNI_FEASIBILITY.md itself used):

      Qwen2_5OmniThinkerForConditionalGeneration.__init__ creates FOUR
      independent top-level attributes:
        .audio_tower  = Qwen2_5OmniAudioEncoder      (NEW vs the parent)
        .visual       = Qwen2_5OmniVisionEncoder      (same role as Qwen3-VL's)
        .model        = Qwen2_5OmniThinkerTextModel   (the text decoder)
        .lm_head      = nn.Linear
      reached via the top-level model's `.thinker` attribute
      (Qwen2_5OmniForConditionalGeneration.__init__: `self.thinker = ...`).

      The text decoder's forward signature --
        (input_ids, attention_mask, position_ids, past_key_values,
         inputs_embeds, use_cache, **kwargs)
      -- is shape-identical to Qwen3VLTextModel's; `backend.forward()`'s call
      pattern (inputs_embeds + mutable Cache + position_ids + cache_position
      absorbed by **kwargs) needed no change, and was re-verified end-to-end
      here against the REAL Qwen2_5OmniThinkerTextModel class (seed -> ingest
      text -> ingest a fake vision chunk with real 3D positions -> ingest a
      fake audio chunk -> probe splice+crop -> MVCC deepcopy snapshot; all
      six steps passed under torch.no_grad()).

      Vision: `Qwen2_5OmniThinkerForConditionalGeneration.get_image_features`
      exists with the identical signature Qwen3VLBackend already calls, and
      the registered image processor is `Qwen2VLImageProcessor` (patch_size
      14, spatial_merge_size 2 -- DIFFERENT numbers than Qwen3-VL's 16/2, but
      read generically off the processor at runtime, so `_cap_image_resolution`
      needs no change).

      Audio: no bespoke feature-extraction code was written blind. Verified
      directly: `WhisperFeatureExtractor(sampling_rate=16000,
      feature_size=128)` on a 2-second waveform produces `input_features`
      shaped [1, 128, 200]; reshaped per `get_audio_features`'s own convention
      (`permute(0,2,1)[mask].permute(1,0)` -> `[128, valid_frames]`) and run
      through a real (tiny-random-weight) `Qwen2_5OmniAudioEncoder`, this
      produces EXACTLY 50 output tokens for a 2-second chunk -- i.e.
      `audio_position_id_per_seconds(25) * audio_seconds_per_chunk(2) = 50`,
      confirmed arithmetically, not assumed from the model card.

    See `tmrope_position_ids` (module-level, above) for how positions are
    built once audio is actually flowing.
    """

    def __init__(self, cfg, role="full"):
        from transformers import AutoProcessor
        from transformers import Qwen2_5OmniForConditionalGeneration
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[cfg.dtype]
        self.device, self.dtype, self.cfg, self.role = cfg.device, dtype, cfg, role
        print(f"[backend] loading {cfg.model_id} ({cfg.dtype}, role={role}, "
              f"audio={cfg.use_audio}) on {cfg.device} ...", flush=True)

        # enable_audio_output=False: skip the talker (speech-out) + code2wav
        # vocoder entirely -- we only ever need TEXT output. Verified this is
        # the correct flag (not a guess): Qwen2_5OmniForConditionalGeneration
        # .__init__ only calls self.enable_talker() when
        # config.enable_audio_output is truthy; the checkpoint's ~8k talker +
        # vocoder tensors are simply reported as unexpected keys and skipped,
        # the same pattern OMNI_MODEL_SURVEY.md verified for MiniCPM-o's
        # init_tts=False.
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            cfg.model_id, torch_dtype=dtype, enable_audio_output=False,
            device_map=(cfg.device if role == "full" else None),
            low_cpu_mem_usage=True).eval()
        self.processor = AutoProcessor.from_pretrained(cfg.model_id)
        self.tok = self.processor.tokenizer
        self._cap_image_resolution(cfg.max_pixels)

        self._resolve_modules()
        self.hidden_size = self.model.thinker.config.text_config.hidden_size

        self.eos_id = self.tok.eos_token_id
        self.newline_ids = set(_word_ids(self.tok, ["\n"]))
        self.yes_ids = _word_ids(self.tok, getattr(cfg, "yes_words", ["yes", "Yes", " yes", " Yes"]))
        self.no_ids = _word_ids(self.tok, getattr(cfg, "no_words", ["no", "No", " no", " No"]))

        self._apply_role(role)

    # -- module resolution: fixed, stable attribute names (no defensive probing
    # needed -- unlike Qwen3-VL, this layout has not moved between transformers
    # versions; verified directly on the class __init__ source). --
    def _resolve_modules(self):
        th = self.model.thinker
        self.language_model = th.model            # Qwen2_5OmniThinkerTextModel
        self.visual = th.visual                    # Qwen2_5OmniVisionEncoder
        self.audio_tower = th.audio_tower           # Qwen2_5OmniAudioEncoder (NEW)
        self.lm_head = th.lm_head
        self.embed_tokens = th.model.embed_tokens
        self.get_image_features = getattr(th, "get_image_features", None)

    # -- keep only the half this role needs; extended to a 3-way split
    # (vision / audio / language) per OMNI_FEASIBILITY.md section 7 point 3.
    # UNTESTED on real weights (same caveat Qwen3VLBackend's own vision/
    # language split carries) -- multi-GPU audio replicas are not wired into
    # run.py's single-manager pipeline today; this exists so it is a small
    # extension, not a redesign, WHEN that wiring happens. --
    def _apply_role(self, role):
        import gc
        if role == "full":
            return
        th = self.model.thinker
        keep = {"vision": "visual", "audio": "audio_tower", "language": "model"}
        if role not in keep:
            raise ValueError(f"unknown role {role!r} (want full/vision/audio/language)")
        for name in ("visual", "audio_tower", "model"):
            if name != keep[role] and getattr(th, name, None) is not None:
                setattr(th, name, None)
        if keep[role] != "model" and getattr(th, "lm_head", None) is not None and role != "language":
            th.lm_head = None
        gc.collect()
        getattr(th, keep[role]).to(self.device)
        if role == "language" and getattr(th, "lm_head", None) is not None:
            th.lm_head.to(self.device)
        gc.collect()
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    # -- identical contract to Qwen3VLBackend._cap_image_resolution; reads
    # patch_size/merge_size off the processor generically, so Qwen2.5-Omni's
    # DIFFERENT 14/2 (vs Qwen3-VL's 16/2) needs no special-casing. --
    def _cap_image_resolution(self, max_pixels):
        if not max_pixels:
            return
        ip = getattr(self.processor, "image_processor", None)
        if ip is None:
            return
        per_tok = (getattr(ip, "patch_size", 14) * getattr(ip, "merge_size", 2)) ** 2
        size = getattr(ip, "size", None)

        def _set(obj, key, val):
            try:
                setattr(obj, key, val)
            except Exception:
                pass
            try:
                obj[key] = val
            except Exception:
                pass

        if size is not None:
            cur_min = getattr(size, "shortest_edge", None) or (size.get("shortest_edge")
                      if hasattr(size, "get") else None) or per_tok
            _set(size, "longest_edge", max_pixels)
            _set(size, "shortest_edge", min(cur_min, max_pixels))
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = max_pixels
        print(f"[backend] capped image to <= {max_pixels} px "
              f"(~{max_pixels // per_tok} vision tokens/frame)", flush=True)

    # ---------------------------------------------------------------- text
    def embed_text(self, text):
        ids = self.tok(text, return_tensors="pt",
                       add_special_tokens=False).input_ids.to(self.device)
        return self.embed_tokens(ids)

    def embed_token(self, tok_id):
        ids = torch.tensor([[tok_id]], device=self.device)
        return self.embed_tokens(ids)

    # --------------------------------------------------------------- vision
    @torch.no_grad()
    def embed_frame(self, pil_image):
        feat = self.processor.image_processor(images=[pil_image], return_tensors="pt")
        pixel_values = feat["pixel_values"].to(self.device, self.dtype)
        grid_thw = feat["image_grid_thw"].to(self.device)
        if self.get_image_features is not None:
            out = self.get_image_features(pixel_values=pixel_values, image_grid_thw=grid_thw)
        else:
            out = self.visual(pixel_values, grid_thw=grid_thw)
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        embeds = out[0] if isinstance(out, (list, tuple)) else out
        embeds = embeds.reshape(1, -1, self.hidden_size)
        merge = getattr(self.processor.image_processor, "merge_size", 2)
        _, gh, gw = [int(x) for x in grid_thw[0].tolist()]
        grid_hw = (gh // merge, gw // merge)
        return embeds.to(self.dtype), grid_hw

    # ---------------------------------------------------------------- audio
    @torch.no_grad()
    def embed_audio(self, waveform):
        """waveform: 1D float32 numpy array or list, at cfg.audio_sampling_rate
        (16 kHz). Returns [1, N, H] -- N ~= 25 * len(waveform)/sampling_rate,
        verified arithmetically above (2s -> 50 tokens exactly).

        Mirrors `get_audio_features`'s exact reshape (verified from source,
        not guessed): the raw WhisperFeatureExtractor output is
        [1, n_mel_bins, n_frames] + an attention mask; the encoder wants
        [n_mel_bins, valid_frames] with padding stripped."""
        fe = self.processor.feature_extractor
        out = fe(waveform, sampling_rate=self.cfg.audio_sampling_rate,
                 return_tensors="pt", return_attention_mask=True, padding="longest")
        input_features = out["input_features"].to(self.device)      # [1, mel, T]
        attn = out["attention_mask"].to(self.device)                # [1, T]
        feat = input_features.permute(0, 2, 1)[attn.bool()].permute(1, 0)  # [mel, valid]
        feature_lens = attn.sum(-1)
        result = self.audio_tower(input_features=feat.to(self.dtype), feature_lens=feature_lens)
        h = result.last_hidden_state if hasattr(result, "last_hidden_state") else result[0]
        return h.reshape(1, -1, self.hidden_size).to(self.dtype)     # [1, N, H]

    # -------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, embeds, cache, pos_start, phys_start, want_logits=True,
                position_ids=None):
        """Identical contract to Qwen3VLBackend.forward() -- see that
        docstring. `position_ids`, when given (built by tmrope_position_ids),
        overrides the flat-linear default so real per-token 3D positions can
        be used once audio is interleaved with vision."""
        L = embeds.shape[1]
        cache_position = torch.arange(phys_start, phys_start + L, device=self.device)
        if position_ids is None:
            lin = torch.arange(pos_start, pos_start + L, device=self.device)
            position_ids = lin.view(1, 1, L).expand(3, 1, L).contiguous()
        else:
            position_ids = position_ids.to(self.device)
        out = self.language_model(
            inputs_embeds=embeds.to(self.device, self.dtype),
            past_key_values=cache,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        if not want_logits:
            return None, out.past_key_values
        logits = self.lm_head(out.last_hidden_state[:, -1, :])
        return logits[0].float(), out.past_key_values

    def decode(self, ids):
        return self.tok.decode(ids).strip()


def _first(candidates, what):
    for get in candidates:
        try:
            mod = get()
            if mod is not None:
                return mod
        except AttributeError:
            continue
    raise RuntimeError(
        f"could not locate the {what} on this model; check the transformers "
        f"version / attribute layout in backend._resolve_modules")
