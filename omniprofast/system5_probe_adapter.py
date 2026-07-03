"""
system5_probe_adapter.py — Probe-mode runner, backed by the system_5_probe COPY.

OmniPro Probe mode is non-streaming by design ("does not require streaming
capability"): for each ground-truth trigger at time t we ask the model a
pre-probe (t+pre_off, expect NO) and a post-probe (t+post_off, expect YES /
structured answer). This is a different protocol from the async pipeline, so it
legitimately uses system_5's PRIMITIVES synchronously rather than its threads.

Crucially this imports backend/manager/etc. from SYSTEM5_PROBE_DIR (the editable
copy), NOT the original system_5 — satisfying "use system_5_probe only for the
probe metrics". Because the two packages share flat module names, this runner
must execute in its own process (evaluate.py runs probe as a separate phase).

Efficiency: each (sample, variant) streams the video ONCE; the growing KV cache
is probed in place (mgr.probe is self-erasing) and snapshot-cloned for content
generation, so all probe points for a sample share one ingest pass.
"""
from __future__ import annotations

import dataclasses
import sys

from utils import SYSTEM5_PROBE_DIR, log
from prompts import build_probe_content_question, probe_offsets
from dataset import iter_frames

if SYSTEM5_PROBE_DIR not in sys.path:
    sys.path.insert(0, SYSTEM5_PROBE_DIR)


class ProbeRunner:
    def __init__(self, *, model_id: str | None = None, dtype: str = "bfloat16",
                 device: str = "cuda", kv_budget: int | None = None,
                 content_max_tokens: int = 32):
        from config import AsyncOmniConfig
        from backend import Qwen3VLBackend
        from manager import KVCacheManager

        self._AsyncOmniConfig = AsyncOmniConfig
        self._KVCacheManager = KVCacheManager
        self.content_max_tokens = content_max_tokens

        base_kwargs = dict(mode="video", dtype=dtype, device=device, profile=False)
        if model_id:
            base_kwargs["model_id"] = model_id
        if kv_budget:
            base_kwargs["kv_budget"] = kv_budget
        self.base_cfg = AsyncOmniConfig(**base_kwargs)

        import torch
        self.torch = torch
        log(f"[probe] loading backend {self.base_cfg.model_id} ({dtype}) from "
            f"system_5_probe ...", tag="probe")
        self.backend = Qwen3VLBackend(self.base_cfg)
        log("[probe] backend loaded.", tag="probe")

    # greedy, deterministic content generation off an MVCC snapshot
    def _generate(self, mgr, content_q: str) -> str:
        b = self.backend
        cache, pos, phys = mgr.snapshot_clone()

        def step(emb):
            nonlocal pos, phys, cache
            logits, cache = b.forward(emb, cache, pos_start=pos, phys_start=phys)
            pos += emb.shape[1]; phys += emb.shape[1]
            return logits

        logits = step(b.embed_text(content_q))
        ids: list[int] = []
        for _ in range(self.content_max_tokens):
            tok = int(logits.argmax().item())
            if tok == b.eos_id:
                break
            piece = b.decode([tok])
            if tok in b.newline_ids or "\n" in piece:
                break
            ids.append(tok)
            logits = step(b.embed_token(tok))
        return b.decode(ids).strip()

    def run_sample(self, sample, prompt_fields: dict, *, fps: float = 1.0,
                   max_seconds: float | None = None, content: bool = True) -> list[dict]:
        cfg = dataclasses.replace(
            self.base_cfg,
            system_prompt=prompt_fields["system_prompt"],
            event_question=prompt_fields["goal_question"],   # map goal_-> event_ (prosync)
            video_path=sample.video_path, groundtruth=False, realtime=False,
            **({"fps": fps} if fps else {}),
        )
        mgr = self._KVCacheManager(self.backend, kv_budget=cfg.kv_budget, prof=None)
        mgr.seed(cfg.system_prompt)
        content_q = build_probe_content_question(sample.task, sample)
        goal_q = cfg.event_question

        # build probe points (time, role, trig_idx)
        points = []
        for idx, g in enumerate(sample.ground_truth):
            pre_off, post_off = probe_offsets(sample.task, seed_idx=idx)
            points.append((max(0.0, g["t_sec"] + pre_off), "pre", idx))
            points.append((max(0.0, g["t_sec"] + post_off), "post", idx))
        points.sort(key=lambda x: x[0])

        rec = {idx: {"task": sample.task, "question": content_q,
                     "pre_share": 1.0, "post_share": 0.0, "post_text": "",
                     "gt_item": g} for idx, g in enumerate(sample.ground_truth)}

        cap = max_seconds if max_seconds else (max(p[0] for p in points) + 5 if points else 30)
        pi = 0

        def resolve(role, idx):
            share = mgr.probe(goal_q, f"probe.{role}")
            if role == "pre":
                rec[idx]["pre_share"] = float(share)
            else:
                rec[idx]["post_share"] = float(share)
                if content:
                    rec[idx]["post_text"] = self._generate(mgr, content_q)

        for t, img in iter_frames(sample.video_path, fps=fps, max_seconds=cap):
            mgr.ingest(self.backend.embed_frame(img))
            while pi < len(points) and points[pi][0] <= t:
                _, role, idx = points[pi]
                resolve(role, idx)
                pi += 1
        # resolve any trailing points against the final context
        while pi < len(points):
            _, role, idx = points[pi]
            resolve(role, idx)
            pi += 1

        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return [dict(id=sample.id, trig_idx=i, **rec[i]) for i in sorted(rec)]
