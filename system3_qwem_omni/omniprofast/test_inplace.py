"""
test_inplace.py — prove the controller can read the shared KV cache WITHOUT the
per-tick MVCC snapshot, and get bit-identical numbers.

The claim under test
--------------------
`controller.py` calls `mgr.snapshot_clone()` once per tick: a full deep copy of
the KV cache. At 144 KB/token that is ~8 GB on a 300 s clip, copied every second
of video, and then thrown away. The copy exists to isolate the reader from a
concurrent writer -- but in lockstep (`deterministic=True`, which is every
benchmark number, MISSION §6) the ingester is parked in
`while clock.get_next_check() <= vt: sleep(0.002)` for the whole tick. There is no
concurrent writer. The copy protects against nothing.

`mgr.borrow_begin()/borrow_end()` replaces it: generate directly on the primary at
the same `(pos_start, phys_start)`, then truncate the appended tokens away. Same
key/value prefix + same positions => the same logits, exactly.

What this file asserts, in increasing strength
----------------------------------------------
  T1 restore     borrow_end() returns the primary to a BIT-IDENTICAL state --
                 every layer's K and V tensor equal element-for-element, plus the
                 logical clock and physical length.
  T2 logits      a forward on the borrowed primary produces EXACTLY the same
                 logits as the same forward on a snapshot clone. Not "close" --
                 `torch.equal`. Anything less and the two modes could diverge
                 after enough greedy argmax ties.
  T3 generation  a full multi-token schema-style walk (the real access pattern:
                 prefill a prompt, then step token by token) matches between the
                 two modes, token id for token id, and STILL restores.
  T4 isolation   an exception raised mid-borrow still leaves the cache restored,
                 and a concurrent ingest during a borrow is REFUSED LOUDLY rather
                 than silently corrupting the primary.
  T5 cost        peak GPU memory and wall time for one snapshot vs one borrow, at
                 a realistic cache length. This is the number the change is for.

T1-T4 are correctness and must all pass. T5 is a measurement and only prints.

This is a mechanism test: it drives the manager directly with text tokens, so it
needs the model but no video and no dataset. The end-to-end proof (same emissions
on real videos) is a separate A/B run -- see `ab_inplace.sh`.

Usage:
    python test_inplace.py                 # all tests, default cache length
    python test_inplace.py --tokens 4000   # bigger cache => bigger T5 numbers
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "async_omni_v2"))

import torch  # noqa: E402


PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok)))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  {detail}" if detail else ""),
          flush=True)
    return ok


def cache_tensors(cache):
    """Every (K, V) pair in the cache, transformers 4.x and 5.x."""
    if hasattr(cache, "layers"):
        return [(l.keys, l.values) for l in cache.layers
                if getattr(l, "keys", None) is not None]
    return list(zip(cache.key_cache, cache.value_cache))


def fingerprint(cache):
    """Cheap content hash of the whole cache, for a fast inequality check."""
    out = []
    for k, v in cache_tensors(cache):
        out.append((tuple(k.shape), tuple(v.shape),
                    float(k.float().sum()), float(v.float().sum())))
    return out


def deep_equal(a, b):
    """Element-for-element equality of two caches. Not allclose -- equal."""
    ta, tb = cache_tensors(a), cache_tensors(b)
    if len(ta) != len(tb):
        return False, f"layer count {len(ta)} != {len(tb)}"
    for i, ((ka, va), (kb, vb)) in enumerate(zip(ta, tb)):
        if ka.shape != kb.shape:
            return False, f"layer {i} key shape {tuple(ka.shape)} != {tuple(kb.shape)}"
        if not torch.equal(ka, kb):
            d = (ka.float() - kb.float()).abs().max().item()
            return False, f"layer {i} keys differ, max|d|={d:.3e}"
        if not torch.equal(va, vb):
            d = (va.float() - vb.float()).abs().max().item()
            return False, f"layer {i} values differ, max|d|={d:.3e}"
    return True, ""


def build(cfg_overrides=None):
    from config import AsyncOmniConfig
    from backend import Qwen3VLBackend
    from manager import KVCacheManager
    cfg = AsyncOmniConfig(**(cfg_overrides or {}))
    b = Qwen3VLBackend(cfg)
    mgr = KVCacheManager(b, kv_budget=cfg.kv_budget, prof=None)
    return cfg, b, mgr


def fill(mgr, b, n_tokens):
    """Seed the cache to roughly `n_tokens`, the way the ingester would.

    Ingested in big chunks rather than one frame at a time: we only need a cache
    of a REALISTIC LENGTH to measure against (a 300 s clip at ~185 tok/frame is
    ~55k tokens), and prefill is parallel, so a 1000-token chunk costs one forward
    instead of thirty. What is in the cache does not affect the equality claim --
    the two modes read the same prefix either way."""
    mgr.seed("You are a streaming video assistant. Report events as they happen.")
    unit = ("At time 1s the scene shows a wide shot of a street with cars and "
            "pedestrians moving through the frame in both directions. ")
    chunk = unit * 40                      # ~1000 tokens per forward
    emb = b.embed_text(chunk)
    step = emb.shape[1]
    while mgr.cache.get_seq_length() + step <= n_tokens:
        mgr.ingest(emb)
    while mgr.cache.get_seq_length() < n_tokens:
        mgr.ingest(b.embed_text(unit))
    return mgr.cache.get_seq_length()


# ---------------------------------------------------------------------------
def t1_restore(mgr, b):
    print("\nT1  borrow_end() restores the primary bit-for-bit")
    before_fp = fingerprint(mgr.cache)
    import copy
    before = copy.deepcopy(mgr.cache)
    pos0, phys0 = mgr.next_pos, mgr.cache.get_seq_length()

    pos, phys = mgr.borrow_begin("t1")
    check("borrow_begin returns the live (pos, phys)", (pos, phys) == (pos0, phys0),
          f"got ({pos}, {phys}) want ({pos0}, {phys0})")
    # append a realistic controller splice
    logits, mgr.cache = b.forward(b.embed_text('{"seen":"a street with cars"'),
                                  mgr.cache, pos_start=pos, phys_start=phys)
    grew = mgr.cache.get_seq_length() > phys0
    check("the borrow actually appended to the PRIMARY", grew,
          f"len {phys0} -> {mgr.cache.get_seq_length()}")
    check("the primary is observably dirty mid-borrow",
          fingerprint(mgr.cache) != before_fp)
    mgr.borrow_end()

    ok, why = deep_equal(mgr.cache, before)
    check("every K/V tensor restored exactly", ok, why)
    check("physical length restored", mgr.cache.get_seq_length() == phys0,
          f"{mgr.cache.get_seq_length()} vs {phys0}")
    check("logical clock (next_pos) restored", mgr.next_pos == pos0,
          f"{mgr.next_pos} vs {pos0}")
    check("borrow_end is idempotent", (mgr.borrow_end() is None))
    del before


def t2_logits(mgr, b):
    print("\nT2  borrowed forward == snapshot forward, exactly")
    text = ('\nNow emit ONLY your control JSON for the current stream:\n'
            '{"seen":"')
    emb = b.embed_text(text)

    clone, pos_c, phys_c = mgr.snapshot_clone()
    lo_snap, _ = b.forward(emb, clone, pos_start=pos_c, phys_start=phys_c)
    lo_snap = lo_snap.clone()
    del clone

    pos, phys = mgr.borrow_begin("t2")
    lo_inp, mgr.cache = b.forward(emb, mgr.cache, pos_start=pos, phys_start=phys)
    lo_inp = lo_inp.clone()
    mgr.borrow_end()

    check("same logit shape", lo_snap.shape == lo_inp.shape,
          f"{tuple(lo_snap.shape)} vs {tuple(lo_inp.shape)}")
    same = torch.equal(lo_snap, lo_inp)
    d = (lo_snap.float() - lo_inp.float()).abs().max().item()
    check("logits BIT-IDENTICAL (torch.equal)", same, f"max|d|={d:.3e}")
    check("argmax token identical",
          int(lo_snap.argmax()) == int(lo_inp.argmax()),
          f"{int(lo_snap.argmax())} vs {int(lo_inp.argmax())}")
    # a p_hit-style logit read is what the gate actually consumes
    tid = b.tok.encode("true", add_special_tokens=False)
    fid = b.tok.encode("false", add_special_tokens=False)
    if len(tid) == 1 and len(fid) == 1:
        def p_hit(l):
            pair = torch.stack([l[tid[0]], l[fid[0]]]).float()
            return float(torch.softmax(pair, 0)[0])
        check("p_hit identical to the last bit",
              repr(p_hit(lo_snap)) == repr(p_hit(lo_inp)),
              f"{p_hit(lo_snap)!r} vs {p_hit(lo_inp)!r}")


def _walk(b, cache, pos, phys, n, on_cache=None):
    """Prefill a prompt then greedily step `n` tokens -- the controller's pattern."""
    emb = b.embed_text('{"seen":"')
    logits, cache = b.forward(emb, cache, pos_start=pos, phys_start=phys)
    if on_cache:
        on_cache(cache)
    L = emb.shape[1]
    pos, phys = pos + L, phys + L
    ids = []
    for _ in range(n):
        tid = int(torch.argmax(logits))
        ids.append(tid)
        logits, cache = b.forward(b.embed_token(tid), cache,
                                  pos_start=pos, phys_start=phys)
        if on_cache:
            on_cache(cache)
        pos, phys = pos + 1, phys + 1
    return ids, cache


def t3_generation(mgr, b, n=16):
    print(f"\nT3  a {n}-token greedy walk matches, and still restores")
    import copy
    before = copy.deepcopy(mgr.cache)
    phys0, pos0 = mgr.cache.get_seq_length(), mgr.next_pos

    clone, pos_c, phys_c = mgr.snapshot_clone()
    ids_snap, _ = _walk(b, clone, pos_c, phys_c, n)
    del clone

    pos, phys = mgr.borrow_begin("t3")

    def keep(c):
        mgr.cache = c
    ids_inp, _ = _walk(b, mgr.cache, pos, phys, n, on_cache=keep)
    mgr.borrow_end()

    check("token ids identical", ids_snap == ids_inp,
          f"\n      snapshot: {b.decode(ids_snap)!r}"
          f"\n      inplace : {b.decode(ids_inp)!r}")
    ok, why = deep_equal(mgr.cache, before)
    check("primary restored after a full generation", ok, why)
    check("clock restored after a full generation",
          (mgr.cache.get_seq_length(), mgr.next_pos) == (phys0, pos0))
    del before


def t4_isolation(mgr, b):
    print("\nT4  a borrow is isolated, and violations are LOUD")
    import copy
    before = copy.deepcopy(mgr.cache)
    phys0 = mgr.cache.get_seq_length()

    # (a) an exception mid-borrow must still restore -- this is why controller.py
    #     wraps the whole tick in try/finally.
    try:
        pos, phys = mgr.borrow_begin("t4a")
        try:
            _, mgr.cache = b.forward(b.embed_text("partial splice"), mgr.cache,
                                     pos_start=pos, phys_start=phys)
            raise RuntimeError("simulated mid-tick failure")
        finally:
            mgr.borrow_end()
    except RuntimeError as e:
        check("exception propagates", "simulated" in str(e))
    ok, why = deep_equal(mgr.cache, before)
    check("primary restored despite the exception", ok, why)

    # (b) a writer touching the primary mid-borrow must RAISE, not corrupt.
    pos, phys = mgr.borrow_begin("t4b")
    try:
        raised = False
        try:
            mgr.ingest(b.embed_text("a frame arriving mid-borrow"))
        except RuntimeError as e:
            raised = "borrowed" in str(e)
        check("ingest() during a borrow is refused", raised)
        raised = False
        try:
            mgr.evict()
        except RuntimeError as e:
            raised = "borrowed" in str(e)
        check("evict() during a borrow is refused", raised)
        raised = False
        try:
            mgr.borrow_begin("t4c")
        except RuntimeError as e:
            raised = "already borrowed" in str(e)
        check("a second borrow is refused", raised)
    finally:
        mgr.borrow_end()
    ok, why = deep_equal(mgr.cache, before)
    check("primary intact after the refusals", ok, why)
    check("length unchanged", mgr.cache.get_seq_length() == phys0)
    del before


def t5_cost(mgr, b, reps=5):
    print("\nT5  what the snapshot actually costs (measurement, not a pass/fail)")
    dev = b.device
    n_tok = mgr.cache.get_seq_length()
    per_tok = sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                  for k, v in cache_tensors(mgr.cache)) / max(n_tok, 1)
    emb = b.embed_text('{"seen":"')

    def timeit(fn):
        fn()                                   # warm
        torch.cuda.synchronize(dev)
        t = time.time()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize(dev)
        return (time.time() - t) / reps

    def do_snapshot():
        clone, p, ph = mgr.snapshot_clone()
        _, clone = b.forward(emb, clone, pos_start=p, phys_start=ph)
        del clone

    def do_inplace():
        p, ph = mgr.borrow_begin("t5")
        _, mgr.cache = b.forward(emb, mgr.cache, pos_start=p, phys_start=ph)
        mgr.borrow_end()

    torch.cuda.reset_peak_memory_stats(dev)
    base = torch.cuda.memory_allocated(dev)
    t_snap = timeit(do_snapshot)
    peak_snap = torch.cuda.max_memory_allocated(dev) - base

    torch.cuda.reset_peak_memory_stats(dev)
    t_inp = timeit(do_inplace)
    peak_inp = torch.cuda.max_memory_allocated(dev) - base

    mb = 1024 ** 2
    print(f"      cache: {n_tok} tokens, {per_tok / 1024:.1f} KB/token, "
          f"{n_tok * per_tok / mb:.0f} MB resident")
    print(f"      one tick, snapshot : {1000 * t_snap:8.1f} ms   "
          f"peak extra {peak_snap / mb:8.1f} MB")
    print(f"      one tick, inplace  : {1000 * t_inp:8.1f} ms   "
          f"peak extra {peak_inp / mb:8.1f} MB")
    if t_inp > 0:
        print(f"      saving             : {1000 * (t_snap - t_inp):8.1f} ms/tick  "
              f"({100 * (1 - t_inp / t_snap):.1f}%)   "
              f"{(peak_snap - peak_inp) / mb:.0f} MB/tick")
    # a 300 s clip at 1 fps ticks ~300 times
    print(f"      extrapolated over a 300-tick clip: "
          f"{300 * (t_snap - t_inp):.1f} s of wall time not spent copying")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=2000,
                    help="cache length to test at (a 300s clip is ~55k)")
    ap.add_argument("--gen", type=int, default=16, help="tokens for the T3 walk")
    ap.add_argument("--skip-cost", action="store_true")
    ap.add_argument("--only-cost", action="store_true",
                    help="T5 only. Needed at a realistic cache length: T1/T3/T4 "
                         "each hold a reference copy for the equality check, so "
                         "the TEST needs 3x the cache while the system needs 2x.")
    a = ap.parse_args()

    print(f"loading backend ...", flush=True)
    cfg, b, mgr = build()
    n = fill(mgr, b, a.tokens)
    print(f"cache filled to {n} tokens on {b.device}")

    if not a.only_cost:
        t1_restore(mgr, b)
        t2_logits(mgr, b)
        t3_generation(mgr, b, a.gen)
        t4_isolation(mgr, b)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    if not a.skip_cost and torch.cuda.is_available():
        t5_cost(mgr, b)

    n_fail = sum(1 for _, ok in _results if not ok)
    print(f"\n{'-' * 60}\n{len(_results) - n_fail}/{len(_results)} checks passed")
    if n_fail:
        print(f"{n_fail} FAILED -- inplace is NOT equivalent, do not use it")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
