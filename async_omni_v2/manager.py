"""
manager.py — the shared KV cache + the single GPU/cache owner thread.

This file is model-agnostic: it knows nothing about Qwen vs Mobile-O, only the
ModelBackend contract. It implements the core architecture:

  * ONE linear KV cache (`DynamicCache`) for the whole system.
  * a SINGLE worker thread that is the only one allowed to run `backend.forward`
    and mutate the cache. Other threads `submit(prio, op)` and wait. Requests
    are served by priority (PRIO_ORCH < PRIO_WRITER) so the writer can never
    block the orchestrator.
  * TWO clocks: `next_pos` (logical RoPE position, monotonic, survives eviction)
    vs physical cache length (write index). StreamingLLM-style eviction keeps a
    sink prefix + a recent window and leaves `next_pos` running.
  * proactivity probe (AsyncReasoning §3.2): splice a yes/no question, read the
    answer, then ERASE it from the cache (truncate back) so it leaves no trace.

`op_*` methods return a closure that the worker thread executes with exclusive
cache access.
"""
import heapq
import itertools
import threading
import torch
from transformers import DynamicCache

from proactivity import yes_share

PRIO_ORCH = 0      # orchestrator: thinking, probes, vision ingest, eviction
PRIO_WRITER = 2    # writer: commentary -- never blocks the orchestrator


class _Req:
    __slots__ = ("prio", "seq", "fn", "done", "result", "error")

    def __init__(self, prio, seq, fn):
        self.prio, self.seq, self.fn = prio, seq, fn
        self.done = threading.Event()
        self.result = self.error = None

    def __lt__(self, other):              # heap order: (priority, FIFO)
        return (self.prio, self.seq) < (other.prio, other.seq)


class KVCacheManager:
    def __init__(self, backend):
        self.b = backend
        self.cache = DynamicCache()
        self.next_pos = 0                 # logical RoPE clock (see module doc)
        self._q, self._counter = [], itertools.count()
        self._cv = threading.Condition()
        self._stop = False
        self._worker = threading.Thread(target=self._run, daemon=True, name="kv-manager")
        self._worker.start()

    # ---- client API (any thread) ----
    def submit(self, prio, fn):
        req = _Req(prio, next(self._counter), fn)
        with self._cv:
            heapq.heappush(self._q, req)
            self._cv.notify()
        req.done.wait()
        if req.error:
            raise req.error
        return req.result

    def shutdown(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._worker.join(timeout=5)      # clean exit (no "terminate called")

    # ---- worker (the only thread touching cache/GPU) ----
    def _run(self):
        while True:
            with self._cv:
                while not self._q and not self._stop:
                    self._cv.wait()
                if self._stop and not self._q:
                    return
                req = heapq.heappop(self._q)
            try:
                req.result = req.fn(self)
            except Exception as e:        # surface to the caller
                req.error = e
            req.done.set()

    # ===== primitives (run inside the worker only) =====
    def _cache_len(self):
        return self.cache.get_seq_length()

    def _forward(self, embeds):
        """Append embeds, advance both clocks, return last-position logits."""
        logits, self.cache = self.b.forward(
            embeds, self.cache, pos_start=self.next_pos, phys_start=self._cache_len())
        self.next_pos += embeds.shape[1]
        return logits

    def _truncate(self, phys_len):
        # transformers 5.x: DynamicCache exposes `.layers[i].keys/.values` and a
        # `crop()` helper; 4.x exposed `.key_cache/.value_cache` lists.
        if hasattr(self.cache, "crop"):
            self.cache.crop(phys_len)
        else:
            for i in range(len(self.cache.key_cache)):
                self.cache.key_cache[i] = self.cache.key_cache[i][:, :, :phys_len, :]
                self.cache.value_cache[i] = self.cache.value_cache[i][:, :, :phys_len, :]
            self.cache._seen_tokens = phys_len

    def _evict(self, budget, sink):
        n = self._cache_len()
        if n <= budget:
            return 0
        keep_recent = budget - sink
        if hasattr(self.cache, "layers"):                 # transformers 5.x
            for layer in self.cache.layers:
                if getattr(layer, "keys", None) is None:
                    continue
                k, v = layer.keys, layer.values
                layer.keys = torch.cat([k[:, :, :sink], k[:, :, n - keep_recent:]], dim=2)
                layer.values = torch.cat([v[:, :, :sink], v[:, :, n - keep_recent:]], dim=2)
        else:                                             # transformers 4.x
            for i in range(len(self.cache.key_cache)):
                k, v = self.cache.key_cache[i], self.cache.value_cache[i]
                self.cache.key_cache[i] = torch.cat([k[:, :, :sink], k[:, :, n - keep_recent:]], dim=2)
                self.cache.value_cache[i] = torch.cat([v[:, :, :sink], v[:, :, n - keep_recent:]], dim=2)
            self.cache._seen_tokens = sink + keep_recent
        return n - budget

    # ===== ops (closures handed to submit) =====
    def op_seed(self, system_text):
        """Seed the cache; return its length (used as the eviction sink)."""
        def fn(m):
            m._forward(m.b.embed_text(system_text))
            return m._cache_len()
        return fn

    def op_ingest_text(self, text):
        return lambda m: m._forward(m.b.embed_text(text))

    def op_ingest_frame(self, pil):
        return lambda m: m._forward(m.b.embed_frame(pil))

    def op_append_token(self, tok_id):
        return lambda m: m._forward(m.b.embed_token(tok_id))

    def op_append_text(self, text):
        return lambda m: m._forward(m.b.embed_text(text))

    def op_evict(self, budget, sink):
        return lambda m: m._evict(budget, sink)

    def op_probe(self, question):
        """Yes/no proactivity probe that self-erases. Returns yes_share in [0,1]."""
        def fn(m):
            phys0, pos0 = m._cache_len(), m.next_pos
            logits = m._forward(m.b.embed_text(question))
            share = yes_share(logits, m.b.yes_ids, m.b.no_ids)
            m._truncate(phys0)
            m.next_pos = pos0
            return share
        return fn
