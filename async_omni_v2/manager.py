"""
manager.py — the shared KV cache as a CONCURRENT, DB-like resource.

Design goals (v3, the "nothing blocks anyone" rewrite):

  * There is no longer a single GPU worker thread that serializes everything.
    Each component thread (encoder / orchestrator / writer) issues its own GPU
    work. The GPU still time-shares kernels (one device), but there is no
    *logical* head-of-line blocking anymore.

  * Concurrency control, MVCC-style:
      - The PRIMARY cache has exactly ONE writer: the orchestrator thread. So
        there is never a write-write conflict on it (the cache is a linear
        sequence; two appenders would be incoherent anyway).
      - READERS that must not be disturbed mid-flight (the writer thread) take a
        SNAPSHOT: `snapshot_clone()` returns an independent deep copy of the
        cache + the logical position. The reader then generates on its private
        copy, holding NO lock — so the orchestrator keeps mutating the primary
        concurrently and neither blocks the other. This is the read-snapshot
        isolation you get from an MVCC database.
      - The internal `_lock` is held only for the *brief* primary mutations
        (append / probe / evict) and for the clone copy. Read-read is free;
        a snapshot waits at most one in-flight op.

  * Two clocks survive eviction: `next_pos` (logical RoPE position, monotonic)
    vs physical cache length (write index). StreamingLLM eviction keeps a sink
    prefix + recent window; `next_pos` keeps running.

  * proactivity probe: splice a yes/no question, read the answer, ERASE it
    (truncate back) so it leaves no trace in the primary cache.

Honest limit: on ONE GPU the forwards still serialize on the SMs. Put the
writer (or encoder) on a second GPU with a model replica and this same code
becomes truly parallel — the snapshot already removes the cache coupling.
"""
import copy
import threading
import time
import torch
from transformers import DynamicCache


class KVCacheManager:
    def __init__(self, backend, kv_budget, prof=None, sync=True):
        self.b = backend
        self.kv_budget = kv_budget
        self.prof = prof
        # sync: torch.cuda.synchronize() around each op so per-op timing is true
        # GPU compute, not just kernel-launch wall time.
        self._sync = sync and torch.cuda.is_available()
        self.cache = DynamicCache()
        self.next_pos = 0                 # logical RoPE clock
        self.sink = 0                     # protected prefix length (system prompt)
        self._lock = threading.Lock()     # guards PRIMARY mutations + snapshot

    # ---- profiling helper ----
    def _rec(self, label, dur, wait):
        if self.prof is not None:
            self.prof.record_op(label, dur, wait)

    # ---- primary-cache primitives (call only while holding self._lock) ----
    def _len(self):
        return self.cache.get_seq_length()

    def _forward_primary(self, embeds):
        logits, self.cache = self.b.forward(
            embeds, self.cache, pos_start=self.next_pos, phys_start=self._len())
        self.next_pos += embeds.shape[1]
        return logits

    def _evict_locked(self):
        n = self._len()
        if n <= self.kv_budget:
            return 0
        keep_recent = self.kv_budget - self.sink
        if hasattr(self.cache, "layers"):                 # transformers 5.x
            for layer in self.cache.layers:
                if getattr(layer, "keys", None) is None:
                    continue
                k, v = layer.keys, layer.values
                layer.keys = torch.cat([k[:, :, :self.sink], k[:, :, n - keep_recent:]], dim=2)
                layer.values = torch.cat([v[:, :, :self.sink], v[:, :, n - keep_recent:]], dim=2)
        else:                                             # transformers 4.x
            for i in range(len(self.cache.key_cache)):
                k, v = self.cache.key_cache[i], self.cache.value_cache[i]
                self.cache.key_cache[i] = torch.cat([k[:, :, :self.sink], k[:, :, n - keep_recent:]], dim=2)
                self.cache.value_cache[i] = torch.cat([v[:, :, :self.sink], v[:, :, n - keep_recent:]], dim=2)
            self.cache._seen_tokens = self.sink + keep_recent
        return n - self.kv_budget

    # ===== public API (thread-safe) =====
    def seed(self, system_text):
        """Seed the cache once; the resulting length becomes the eviction sink."""
        t0 = time.time()
        with self._lock:
            t1 = time.time()
            self._forward_primary(self.b.embed_text(system_text))
            self.sink = self._len()
            if self._sync:
                torch.cuda.synchronize()
            t2 = time.time()
        self._rec("seed", t2 - t1, t1 - t0)
        return self.sink

    def ingest(self, embeds):
        """Append projected visual (or text) tokens to the primary cache.
        ONLY the orchestrator thread calls this -> single-writer, no conflict."""
        t0 = time.time()
        with self._lock:
            t1 = time.time()
            self._forward_primary(embeds)
            if self._sync:
                torch.cuda.synchronize()
            t2 = time.time()
        self._rec("ingest.frame", t2 - t1, t1 - t0)

    def evict(self):
        t0 = time.time()
        with self._lock:
            t1 = time.time()
            dropped = self._evict_locked()
            if self._sync:
                torch.cuda.synchronize()
            t2 = time.time()
        self._rec("evict", t2 - t1, t1 - t0)
        return dropped

    def snapshot_clone(self):
        """MVCC read snapshot: return an INDEPENDENT clone of the primary cache
        plus its logical position + physical length. The caller (writer) then
        generates on the clone holding no lock, fully concurrent with the
        orchestrator's ongoing mutations of the primary."""
        t0 = time.time()
        with self._lock:
            t1 = time.time()
            clone = self._clone_cache()
            pos, phys = self.next_pos, self._len()
            if self._sync:
                torch.cuda.synchronize()
            t2 = time.time()
        self._rec("snapshot_clone", t2 - t1, t1 - t0)
        return clone, pos, phys

    def _clone_cache(self):
        try:
            return copy.deepcopy(self.cache)
        except Exception:
            # manual fallback: clone the K/V tensors layer by layer
            dst = DynamicCache()
            src = self.cache
            if hasattr(src, "layers"):
                for layer in src.layers:
                    k, v = getattr(layer, "keys", None), getattr(layer, "values", None)
                    if k is None:
                        continue
                    dst.update(k.clone(), v.clone(), len(dst.layers))
            else:
                for i in range(len(src.key_cache)):
                    dst.update(src.key_cache[i].clone(), src.value_cache[i].clone(), i)
            return dst
