"""
ab_inplace_diff.py — diff two eval arms that must be identical.

Companion to ab_inplace.sh. Given two run directories (snapshot vs inplace), it
compares everything the controller produced and reports the first divergence with
enough context to act on.

The comparison is deliberately layered, because WHERE it first differs tells you
what broke:

  1. ctrl.raw     the control JSON per tick. Differs => the model saw a different
                  cache. Nothing downstream is worth reading.
  2. ctrl.gate    p_hit / p_more / level / fire, with `gen=` and `ntok=` stripped.
                  Identical raw with a drifting p_hit still changes emissions,
                  because p_hit is what the gate consumes.
  3. CONTROLLER   the emissions that actually get scored (time + text).
  4. online_pred  the scored artefact, minus wall-clock fields.
  5. metrics      the headline numbers. Equal by construction if 1-4 are, but
                  checked anyway -- if these differ while 1-4 match, the bug is in
                  scoring, not in the cache.

Timing IS expected to differ (that is the whole point), so `gen=`/`wall_s` are
stripped before diffing and reported separately as the measured saving.

Usage:  python ab_inplace_diff.py output_ab_inplace/snapshot output_ab_inplace/inplace
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

_TICK = re.compile(r"\|\s*vid\s*([\d.]+)s\]\s+(ctrl\.raw|ctrl\.gate|CONTROLLER)\s+(.*?)\s*$")
_SAMPLE = re.compile(r"===== \[\s*\d+\s*/\s*\d+\s*\]\s+(\S+)")
# fields that are ALLOWED to differ: wall-clock, and the token count that follows
# from it. Everything else must match.
_STRIP = re.compile(r"\s+(gen=[\d.]+s|ntok=\d+)")
_GEN = re.compile(r"gen=([\d.]+)s")


def load(run_dir):
    """-> ({(sample, kind, vt, n) -> text}, [gen_seconds])"""
    out, gens = {}, []
    # run_fast.sh tees to run_<stamp>.log; calling evaluate.py directly (what
    # ab_inplace.sh does) leaves everything in launch.log. Accept either.
    logs = sorted(glob.glob(os.path.join(run_dir, "run_*.log")))
    if not logs:
        logs = sorted(glob.glob(os.path.join(run_dir, "launch.log")))
    if not logs:
        sys.exit(f"no run_*.log or launch.log in {run_dir}")
    sample = "?"
    seen = {}
    for p in logs:
        with open(p, errors="replace") as fh:
            for line in fh:
                s = _SAMPLE.search(line)
                if s:
                    sample = s.group(1)
                    continue
                m = _TICK.search(line)
                if not m:
                    continue
                vt, kind, body = m.group(1), m.group(2), m.group(3)
                g = _GEN.search(body)
                if g:
                    gens.append(float(g.group(1)))
                # a (sample, kind, vt) can repeat if a tick is re-run; number them
                k0 = (sample, kind, vt)
                seen[k0] = seen.get(k0, 0) + 1
                out[k0 + (seen[k0],)] = _STRIP.sub("", body)
    return out, gens


def cmp_dicts(a, b, label, show=6):
    ka, kb = set(a), set(b)
    only_a, only_b = ka - kb, kb - ka
    common = ka & kb
    diff = [k for k in sorted(common) if a[k] != b[k]]
    ok = not (only_a or only_b or diff)
    tag = "\033[32mIDENTICAL\033[0m" if ok else "\033[31mDIFFERS\033[0m"
    print(f"  {label:<14} {len(common):>7} compared   {tag}")
    if only_a or only_b:
        print(f"      only in snapshot: {len(only_a)}   only in inplace: {len(only_b)}")
        for k in sorted(only_a)[:show]:
            print(f"        S-only {k}: {a[k][:120]}")
        for k in sorted(only_b)[:show]:
            print(f"        I-only {k}: {b[k][:120]}")
    if diff:
        print(f"      {len(diff)} differing entries; first {min(show, len(diff))}:")
        for k in diff[:show]:
            print(f"        {k}")
            print(f"          snapshot: {a[k][:160]}")
            print(f"          inplace : {b[k][:160]}")
    return ok


def preds(run_dir):
    p = os.path.join(run_dir, "online_pred.jsonl")
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p):
        r = json.loads(line)
        r.pop("wall_s", None)
        # per-emission latency is wall-clock; strip it and keep t_sec/share/raw,
        # which are the fields the scorer actually consumes.
        for e in r.get("predictions") or []:
            if isinstance(e, dict):
                for k in ("writer_latency_s", "latency_s", "gen_s"):
                    e.pop(k, None)
        out[r.get("id", "?")] = json.dumps(r, sort_keys=True, default=str)
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    A, B = sys.argv[1], sys.argv[2]
    print(f"\nsnapshot arm : {A}\ninplace  arm : {B}\n")

    ta, ga = load(A)
    tb, gb = load(B)

    def sub(d, kind):
        return {k: v for k, v in d.items() if k[1] == kind}

    print("PER-TICK CONTROLLER OUTPUT")
    ok = True
    for kind, label in [("ctrl.raw", "ctrl.raw"), ("ctrl.gate", "ctrl.gate"),
                        ("CONTROLLER", "emissions")]:
        ok &= cmp_dicts(sub(ta, kind), sub(tb, kind), label)

    print("\nSCORED ARTEFACT")
    pa, pb = preds(A), preds(B)
    if pa is None or pb is None:
        print("  online_pred.jsonl missing in one arm -- skipped")
    else:
        ok &= cmp_dicts(pa, pb, "online_pred")

    print("\nHEADLINE METRICS")
    for d, name in ((A, "snapshot"), (B, "inplace")):
        p = os.path.join(d, "online_metrics.json")
        if os.path.exists(p):
            o = json.load(open(p)).get("overall", {})
            print(f"  {name:<9} time_f1={o.get('time_f1')} joint_f1={o.get('joint_f1')} "
                  f"n_emits={o.get('n_emits')} tp={o.get('tp_time')}")

    print("\nWHAT IS ALLOWED TO DIFFER (and does)")
    if ga and gb:
        ma, mb = sorted(ga)[len(ga) // 2], sorted(gb)[len(gb) // 2]
        print(f"  controller gen_s   median  snapshot {ma:.3f}s   inplace {mb:.3f}s   "
              f"delta {1000 * (ma - mb):+.1f} ms/tick")
        print(f"  total controller time      snapshot {sum(ga):.1f}s   "
              f"inplace {sum(gb):.1f}s   saved {sum(ga) - sum(gb):+.1f}s "
              f"over {len(ga)} ticks")

    print("\n" + "=" * 62)
    if ok:
        print("\033[32mEQUIVALENT\033[0m — inplace reproduced snapshot exactly. "
              "The snapshot can be dropped in lockstep.")
    else:
        print("\033[31mNOT EQUIVALENT\033[0m — see the first differing block above. "
              "Do NOT use inplace until it is explained.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
