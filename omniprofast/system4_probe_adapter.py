"""
system4_probe_adapter.py — Probe-mode runner for system4, backed by system4_probe.

Mirrors system4's own probe_run.py (audio-grounded GT-probe): seed the system
prompt, stream the video as 1 s AUDIO chunks (+ optional timestamp marks) AND
1 fps FRAMES into one growing KV cache, and at each trigger's pre/post offset run
the self-erasing yes/no event probe; generate a short content answer at the post
probe. Audio + vision together — system4's Qwen2.5-Omni handles both.

Imports backend/manager from SYSTEM4_PROBE_DIR (editable copy), so this runs in
its own process (package isolation from the online system4 engine).
"""
from __future__ import annotations

import dataclasses
import sys

from utils import SYSTEM4_PROBE_DIR, log
from prompts import build_probe_content_question, probe_offsets
from dataset import iter_frames

if SYSTEM4_PROBE_DIR not in sys.path:
    sys.path.insert(0, SYSTEM4_PROBE_DIR)


class System4ProbeRunner:
    def __init__(self, *, model_id: str | None = None, dtype: str = "bfloat16",
                 device: str = "cuda", kv_budget: int | None = None,
                 content_max_tokens: int = 32):
        from config import AsyncOmniConfig
        from backend import build_backend
        from manager import KVCacheManager
        from audio_io import load_audio_16k

        self._cfg_cls = AsyncOmniConfig
        self._KVCacheManager = KVCacheManager
        self._load_audio_16k = load_audio_16k
        self.content_max_tokens = content_max_tokens

        base_kwargs = dict(mode="video_audio", dtype=dtype, device=device, profile=False)
        if model_id:
            base_kwargs["model_id"] = model_id
        if kv_budget:
            base_kwargs["kv_budget"] = kv_budget
        self.base_cfg = AsyncOmniConfig(**base_kwargs)

        import torch
        self.torch = torch
        log(f"[s4-probe] loading backend {self.base_cfg.model_id} ({dtype}) from "
            f"system4_probe ...", tag="s4probe")
        self.backend = build_backend(self.base_cfg)
        log("[s4-probe] backend loaded.", tag="s4probe")

    def _generate(self, mgr, content_q: str) -> str:
        b = self.backend
        cache, pos, phys = mgr.snapshot_clone()

        def step(emb):
            nonlocal pos, phys, cache
            logits, cache = b.forward(emb, cache, pos_start=pos, phys_start=phys)
            pos += emb.shape[1]; phys += emb.shape[1]
            return logits

        eos = getattr(b, "eos_id", None)
        nl = getattr(b, "newline_ids", set())
        logits = step(b.embed_text(content_q))
        ids = []
        for _ in range(self.content_max_tokens):
            tok = int(logits.argmax().item())
            if eos is not None and tok == eos:
                break
            piece = b.decode([tok])
            if tok in nl or "\n" in piece:
                break
            ids.append(tok)
            logits = step(b.embed_token(tok))
        return b.decode(ids).strip()

    def run_sample(self, sample, prompt_fields: dict, *, fps: float = 1.0,
                   max_seconds: float | None = None, content: bool = True) -> list[dict]:
        cfg = dataclasses.replace(
            self.base_cfg,
            system_prompt=prompt_fields["system_prompt"],
            event_question=prompt_fields["goal_question"],
            video_path=sample.video_path, prompt_mode="task", realtime=False,
        )
        b = self.backend
        mgr = self._KVCacheManager(self.backend, kv_budget=cfg.kv_budget, prof=None)
        mgr.seed(cfg.system_prompt)
        content_q = build_probe_content_question(sample.task, sample)
        event_q = cfg.event_question

        points = []
        for idx, g in enumerate(sample.ground_truth):
            pre_off, post_off = probe_offsets(sample.task, seed_idx=idx)
            points.append((max(0.0, g["t_sec"] + pre_off), "pre", idx))
            points.append((max(0.0, g["t_sec"] + post_off), "post", idx))
        points.sort(key=lambda x: x[0])
        rec = {idx: {"task": sample.task, "question": content_q, "pre_share": 1.0,
                     "post_share": 0.0, "post_text": "", "gt_item": g}
               for idx, g in enumerate(sample.ground_truth)}

        cap = max_seconds if max_seconds else (max(p[0] for p in points) + 5 if points else 30)

        # pre-decode frames at fps (cheap to hold ~cap PILs); align to integer second
        frames = {}
        for t, img in iter_frames(sample.video_path, fps=fps, max_seconds=cap):
            frames[int(round(t))] = img

        # audio chunks (mirrors probe_run.py)
        sr = cfg.audio_sr
        chunk = int(cfg.audio_chunk_s * sr)
        try:
            y = self._load_audio_16k(sample.video_path, sr)
        except Exception as e:
            log(f"[s4-probe] no audio for {sample.video_id} ({e}); vision-only probe",
                tag="s4probe")
            y = None
        n_chunks = int(min(len(y) if y is not None else 0,
                           int(cap * sr)) // chunk) if y is not None else int(cap)

        pi = 0
        last_ts = -1e9

        def resolve(role, idx):
            share = mgr.probe(event_q, f"probe.{role}")
            if role == "pre":
                rec[idx]["pre_share"] = float(share)
            else:
                rec[idx]["post_share"] = float(share)
                if content:
                    rec[idx]["post_text"] = self._generate(mgr, content_q)

        for c in range(int(n_chunks)):
            vt = (c + 1) * cfg.audio_chunk_s
            if cfg.timestamps and (vt - last_ts) >= (cfg.timestamp_every_s - 1e-6):
                last_ts = vt
                mgr.ingest(b.embed_text(cfg.timestamp_fmt.format(vt)))
            if y is not None:
                wav = y[c * chunk:(c + 1) * chunk]
                if len(wav) > 0:
                    mgr.ingest(b.embed_audio(wav))
            img = frames.get(int(round(vt)))
            if img is not None and not cfg.vision_off:
                mgr.ingest(b.embed_frame(img))
            while pi < len(points) and points[pi][0] <= vt:
                _, role, idx = points[pi]
                resolve(role, idx)
                pi += 1
        while pi < len(points):
            _, role, idx = points[pi]
            resolve(role, idx)
            pi += 1

        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return [dict(id=sample.id, trig_idx=i, **rec[i]) for i in sorted(rec)]
