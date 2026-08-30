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
        # (label, phys0, pos0) while a reader is generating IN PLACE on the
        # primary; None otherwise. See borrow_begin().
        self._borrowed = None

        # ---- TMRoPE chunk anchor (FORK: system3_qwem_omni) -----------------
        # "start_idx" in the reference Qwen2_5Omni.get_rope_index, walked
        # forward incrementally instead of computed from a fully-known
        # sequence -- see backend.tmrope_position_ids's module docstring for
        # the full derivation. Reset only at an audio-chunk boundary (i.e. by
        # ingest(token_kind="audio")), so every vision frame ingested while
        # one ~2s audio window is still open shares the same coordinate
        # system as the audio that will close it.
        self.chunk_anchor_pos = 0
        self.chunk_anchor_vt = 0.0

    # ---- profiling helper ----
    def _rec(self, label, dur, wait):
        if self.prof is not None:
            self.prof.record_op(label, dur, wait)

    # ---- primary-cache primitives (call only while holding self._lock) ----
    def _len(self):
        return self.cache.get_seq_length()

    def _forward_primary(self, embeds, token_kind="text", real_seconds=None, grid_hw=None):
        # ingest/seed only build the KV cache; they never read logits -> skip the
        # lm_head (want_logits=False) so every frame prefill is a bit cheaper.
        #
        # FORK (system3_qwem_omni): when TMRoPE is on and this chunk carries
        # real-world timing (vision/audio), build actual 3D positions instead
        # of letting backend.forward() fall back to the flat-linear default.
        # token_kind="text" (the default -- every call site EXCEPT the new
        # audio/vision ones in input_ingester.py is unaffected) always takes
        # the old path.
        position_ids = None
        if getattr(self.b.cfg, "tmrope_positions", False) and token_kind != "text":
            from backend import tmrope_position_ids
            L = embeds.shape[1]
            local_seconds = 0.0 if real_seconds is None else (real_seconds - self.chunk_anchor_vt)
            position_ids, new_anchor = tmrope_position_ids(
                token_kind, L, self.chunk_anchor_pos, local_seconds=local_seconds,
                grid_hw=grid_hw, per_second=self.b.cfg.audio_position_id_per_seconds,
                device=self.b.device)
            _, self.cache = self.b.forward(
                embeds, self.cache, pos_start=self.next_pos, phys_start=self._len(),
                want_logits=False, position_ids=position_ids)
            # audio always advances the running counter AND closes the chunk
            # (next frame opens a fresh coordinate window); vision advances
            # only "one past this frame's own tick" -- more frames may still
            # land in the SAME open chunk. See tmrope_position_ids docstring.
            self.next_pos = new_anchor
            # LOUD, NOT SILENT (config.py's model_max_position_embeddings
            # comment explains why this fires much sooner on this backend
            # than the parent's identically-shaped eviction guard below):
            # the position clock can exceed the model's TRAINED RoPE range
            # long before kv_budget forces an eviction. A frozen model asked
            # to attend past its trained positions does not error -- it just
            # degrades, silently, which is precisely the failure mode this
            # project's own LEARNINGS.md spent two weeks hunting instances of.
            max_pos = getattr(self.b.cfg, "model_max_position_embeddings", None)
            if max_pos and self.next_pos > max_pos and not getattr(self, "_rope_warned", False):
                self._rope_warned = True
                print(f"[manager] WARNING: position clock ({self.next_pos}) has exceeded "
                      f"model_max_position_embeddings ({max_pos}). The model is now "
                      f"attending at RoPE positions it was never trained on. This is "
                      f"NOT a crash -- generation will continue and look plausible --  "
                      f"but results from this point in the stream onward are suspect. "
                      f"See config.py's model_max_position_embeddings comment / "
                      f"OMNI_EXTENSION.md sec 5.3 (ROADMAP 1.5 position re-basing is "
                      f"the real fix, not implemented here).", flush=True)
            if token_kind == "audio":
                self.chunk_anchor_pos = new_anchor
                # chunk_anchor_vt is set by the caller (input_ingester.py)
                # right before the NEXT chunk's first frame, not here --
                # the manager does not track wall-clock time on its own.
            return None

        _, self.cache = self.b.forward(
            embeds, self.cache, pos_start=self.next_pos, phys_start=self._len(),
            want_logits=False)
        self.next_pos += embeds.shape[1]
        return None

    def _truncate(self, phys_len):
        if hasattr(self.cache, "crop"):
            self.cache.crop(phys_len)
        else:
            for i in range(len(self.cache.key_cache)):
                self.cache.key_cache[i] = self.cache.key_cache[i][:, :, :phys_len, :]
                self.cache.value_cache[i] = self.cache.value_cache[i][:, :, :phys_len, :]
            self.cache._seen_tokens = phys_len

    def _evict_locked(self):
        n = self._len()
        if n <= self.kv_budget:
            return 0
        # ---- ROADMAP 1.5 GUARD (no numeric effect; loud instead of silent) ----
        # Until the first eviction, next_pos and the physical length advance
        # together on every path, so they are EQUAL and RoPE positions are exact.
        # This is the instant they diverge: from here on `next_pos` keeps climbing
        # past the trained range (Qwen3-VL-8B: max_position_embeddings=262144),
        # while the cache holds only `kv_budget` tokens. StreamingLLM requires
        # positions assigned WITHIN the cache window; we do not do that yet, and
        # fixing it properly means re-rotating the cached keys (cf. MiniCPM-o's
        # `realign_rotary_suffix`), not just changing pos_start.
        # Every eval so far runs at max_seconds=300 (~55k tokens), so this has
        # never fired. If it ever does, results past this point are suspect.
        if not getattr(self, "_evict_warned", False):
            self._evict_warned = True
            print(f"[manager] WARNING: FIRST EVICTION at len={n} (budget="
                  f"{self.kv_budget}). next_pos ({self.next_pos}) now diverges from "
                  f"the physical window; RoPE positions will exceed the trained "
                  f"range. See ROADMAP.md 1.5 — results beyond this point are "
                  f"NOT trustworthy until position re-basing is implemented.",
                  flush=True)
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

    def ingest(self, embeds, token_kind="text", real_seconds=None, grid_hw=None):
        """Append projected visual / audio (or text) tokens to the primary
        cache. ONLY the ingester thread calls this -> single-writer, no
        conflict.

        token_kind/real_seconds/grid_hw (FORK: system3_qwem_omni, all
        optional, default preserves the old text-only behaviour exactly):
        see backend.tmrope_position_ids. token_kind="audio" additionally
        closes the current TMRoPE chunk (see _forward_primary)."""
        t0 = time.time()
        self._assert_not_borrowed("ingest")
        with self._lock:
            t1 = time.time()
            self._forward_primary(embeds, token_kind=token_kind,
                                  real_seconds=real_seconds, grid_hw=grid_hw)
            if self._sync:
                torch.cuda.synchronize()
            t2 = time.time()
        self._rec("ingest.frame", t2 - t1, t1 - t0)

    def evict(self):
        t0 = time.time()
        # eviction re-lays the physical window, so a borrow's saved phys0 would no
        # longer mean what it meant. Never evict mid-borrow.
        self._assert_not_borrowed("evict")
        with self._lock:
            t1 = time.time()
            dropped = self._evict_locked()
            if self._sync:
                torch.cuda.synchronize()
            t2 = time.time()
        self._rec("evict", t2 - t1, t1 - t0)
        return dropped

    def probe(self, question, label):
        """PROBE-GATE (gate_mode='probe'): splice a yes/no question onto the PRIMARY
        cache, read one forward pass of logits, compute the yes-share, then ERASE
        the probe (truncate + restore the logical clock) so it leaves no trace."""
        from proactivity import yes_share
        t0 = time.time()
        self._assert_not_borrowed("probe")
        with self._lock:
            t1 = time.time()
            phys0, pos0 = self._len(), self.next_pos
            logits, self.cache = self.b.forward(
                self.b.embed_text(question), self.cache,
                pos_start=pos0, phys_start=phys0, want_logits=True)
            share = yes_share(logits, self.b.yes_ids, self.b.no_ids)
            self._truncate(phys0)
            self.next_pos = pos0
            if self._sync:
                torch.cuda.synchronize()
            t2 = time.time()
        self._rec(label, t2 - t1, t1 - t0)
        return share

    # ---- IN-PLACE READ (the snapshot-free path) -------------------------------
    # `probe()` above already proves the mechanism: splice onto the PRIMARY, read,
    # truncate back. borrow_begin/borrow_end is the same trick opened up so a
    # caller can run a whole multi-token generation between the two, instead of a
    # single forward. It exists because snapshot_clone() copies the ENTIRE cache
    # every tick -- 144 KB/token, so ~8 GB on a 300 s clip -- purely to protect the
    # reader from a concurrent writer.
    #
    # WHEN THAT PROTECTION IS WORTH NOTHING: in lockstep (cfg.deterministic=True,
    # which is every benchmark number -- MISSION §6) the ingester is parked in
    # `while clock.get_next_check() <= vt: sleep(0.002)` for the whole duration of
    # a controller tick. There is no concurrent writer to protect against. The copy
    # is pure cost.
    #
    # We do NOT hold the lock across the generation. Holding it for the 2-4 s of a
    # tick would make a free-running ingester block on the cache, which breaks
    # INVARIANT 2. Instead the borrow is DECLARED: `_borrowed` is set, and every
    # primary mutation refuses loudly while it is. A silent corruption (an ingested
    # frame landing inside the borrow, then being truncated away by borrow_end)
    # becomes an immediate, named exception.
    def borrow_begin(self, label="borrow"):
        """Lend the PRIMARY cache to a reader. Returns (pos, phys) to generate at.

        The reader appends to `self.cache` exactly as it would to a clone, using
        the same pos_start/phys_start -- identical prefix, identical positions,
        therefore identical logits. `borrow_end()` erases every appended token."""
        t0 = time.time()
        with self._lock:
            if self._borrowed is not None:
                raise RuntimeError(
                    f"borrow_begin({label}) while cache is already borrowed by "
                    f"{self._borrowed[0]!r} -- two readers cannot share the primary")
            self._borrowed = (label, self._len(), self.next_pos)
        self._rec(f"{label}.begin", 0.0, time.time() - t0)
        return self._borrowed[2], self._borrowed[1]      # (pos, phys)

    def borrow_end(self):
        """Erase the borrow: truncate back to the pre-borrow length and restore the
        logical clock. Idempotent -- safe to call when nothing is borrowed."""
        if self._borrowed is None:
            return
        label, phys0, pos0 = self._borrowed
        t0 = time.time()
        with self._lock:
            t1 = time.time()
            self._truncate(phys0)
            self.next_pos = pos0
            if self._sync:
                torch.cuda.synchronize()
            t2 = time.time()
            self._borrowed = None
        self._rec(f"{label}.end", t2 - t1, t1 - t0)

    def _assert_not_borrowed(self, op):
        if self._borrowed is not None:
            raise RuntimeError(
                f"{op}() on the primary cache while it is borrowed by "
                f"{self._borrowed[0]!r}. In lockstep this cannot happen (the "
                f"ingester waits on the clock); if you see it, the caller is "
                f"running free and must use snapshot_clone() instead.")

    def snapshot_clone(self):
        """MVCC read snapshot: return an INDEPENDENT clone of the primary cache
        plus its logical position + physical length. The caller (writer) then
        generates on the clone holding no lock, fully concurrent with the
        orchestrator's ongoing mutations of the primary.

        Costs a full deep copy of the cache on EVERY call. See borrow_begin() for
        the snapshot-free path used when there is provably no concurrent writer."""
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
