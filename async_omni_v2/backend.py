"""
backend.py — the model adapter (the ONLY model-specific file).

Everything else in this project is model-agnostic. To support a different
VLM you write a new ModelBackend subclass; nothing else changes.

A backend must provide:
  * tokenizer handles: `eos_id`, `newline_ids`, `yes_ids`, `no_ids`, plus
    `decode(ids)`.
  * `embed_text(text)  -> [1, L, H]`   text -> input embeddings
  * `embed_frame(pil)  -> [1, N, H]`   one image -> N vision-token embeddings
  * `forward(embeds, cache, pos_start, phys_start) -> (logits[V], cache)`
        run the text decoder over `embeds`, appending to `cache`, and return
        the LAST position's logits (CPU). `pos_start` is the logical RoPE
        position of the first new token; `phys_start` is the physical cache
        write index (these differ after eviction — see manager.py).

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
    yes_ids: list
    no_ids: list
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
    """Token ids for single-token surface forms (probe scores compare these)."""
    out = []
    for w in words:
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            out.append(ids[0])
    return out


class Qwen3VLBackend(ModelBackend):
    def __init__(self, cfg):
        from transformers import AutoProcessor
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[cfg.dtype]
        self.device, self.dtype = cfg.device, dtype
        print(f"[backend] loading {cfg.model_id} ({cfg.dtype}) ...", flush=True)

        # Qwen3-VL needs a recent transformers; class name may be
        # Qwen3VLForConditionalGeneration. Fall back to the generic loader.
        try:
            from transformers import Qwen3VLForConditionalGeneration as VLM
        except Exception:
            from transformers import AutoModelForImageTextToText as VLM
        self.model = VLM.from_pretrained(
            cfg.model_id, torch_dtype=dtype, device_map=cfg.device,
            low_cpu_mem_usage=True).eval()
        self.processor = AutoProcessor.from_pretrained(cfg.model_id)
        self.tok = self.processor.tokenizer

        self._resolve_modules()        # find visual / language_model / lm_head
        self.hidden_size = self.model.config.get_text_config().hidden_size \
            if hasattr(self.model.config, "get_text_config") else self.model.config.hidden_size

        self.eos_id = self.tok.eos_token_id
        self.newline_ids = set(_word_ids(self.tok, ["\n"]))
        self.yes_ids = _word_ids(self.tok, cfg.yes_words)
        self.no_ids = _word_ids(self.tok, cfg.no_words)
        if not self.yes_ids or not self.no_ids:
            raise RuntimeError("could not map yes/no to single tokens for this tokenizer")

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
        return embeds.reshape(1, -1, self.hidden_size).to(self.dtype)   # [1, N, H]

    # -------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, embeds, cache, pos_start, phys_start):
        L = embeds.shape[1]
        cache_position = torch.arange(phys_start, phys_start + L, device=self.device)
        # linear positions, shaped [3, 1, L] for the 3D mRoPE rotary (all axes
        # equal -> behaves like ordinary 1D RoPE). See module docstring.
        lin = torch.arange(pos_start, pos_start + L, device=self.device)
        position_ids = lin.view(1, 1, L).expand(3, 1, L).contiguous()
        out = self.language_model(
            inputs_embeds=embeds.to(self.device, self.dtype),
            past_key_values=cache,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        logits = self.lm_head(out.last_hidden_state[:, -1, :])
        return logits[0].float().cpu(), out.past_key_values

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
