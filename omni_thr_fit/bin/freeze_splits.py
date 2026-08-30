#!/usr/bin/env python
"""Freeze the 5% fitting subset and the disjoint 5% validation slice (sec.2).

Pass 1 and pass 2 must see IDENTICAL samples or the comparison is meaningless,
so the draw is written to disk once and every cell reuses it.

It is frozen by IMPORTING the harness's own load_samples rather than
reimplementing the filter -- a second implementation is a second set of bugs,
and the stride is index-based so any disagreement about which samples exist
silently lands on different ids.
"""
import json, os, sys, collections

REPO = os.environ["REPO"]
sys.path.insert(0, os.path.join(REPO, "omniprofast"))
sys.path.insert(0, os.path.join(REPO, "async_omni_v2"))
from dataset import load_samples, ALL_TASKS      # noqa: E402

OUT = os.path.join(os.environ["THR_ROOT"], "splits_thr_fit")
EVERY = 20
os.makedirs(OUT, exist_ok=True)

manifest = {"subset_every": EVERY, "audio": "all", "tasks": {}}
for task in ALL_TASKS:
    # exactly what `evaluate.py --tasks <task> --audio all` loads, with NO
    # max_duration -- the cap is a per-generation window filter applied after
    # the stride, never part of the frozen draw (sec.2).
    s = load_samples(tasks=[task], audio="all", max_duration=None,
                     benchmark_json=os.environ["OMNIPRO_BENCHMARK_JSON"],
                     dataset_dir=os.environ["OMNIPRO_DATASET_DIR"])
    s.sort(key=lambda x: x.id)                    # evaluate.py:85
    row = {"n_eligible": len(s)}
    for off in (0, 1):
        sel = [x for i, x in enumerate(s) if i % EVERY == off]
        mix = collections.Counter(x.audio_dependency for x in sel)
        rec = {
            "subset_off": off,
            "n": len(sel),
            "audio_mix": dict(mix),
            "duration_s": {"min": round(min(x.duration for x in sel), 1),
                           "max": round(max(x.duration for x in sel), 1),
                           "sum": round(sum(x.duration for x in sel), 1)},
            "n_gt": sum(len(x.ground_truth) for x in sel),
            "ids": [x.id for x in sel],
        }
        row[f"off{off}"] = rec
        with open(os.path.join(OUT, f"{task}.off{off}.json"), "w") as f:
            json.dump(rec, f, indent=1)
    manifest["tasks"][task] = {k: v for k, v in row.items() if k != "ids"}
    a, b = row["off0"], row["off1"]
    assert not (set(a["ids"]) & set(b["ids"])), f"{task}: off0/off1 overlap"
    print(f"{task:<30} elig={row['n_eligible']:>4}  fit n={a['n']:>3} "
          f"gt={a['n_gt']:>4} {dict(a['audio_mix'])}  "
          f"dur {a['duration_s']['min']:.0f}-{a['duration_s']['max']:.0f}s")

with open(os.path.join(os.environ["THR_ROOT"], "splits_thr_fit", "MANIFEST.json"), "w") as f:
    json.dump(manifest, f, indent=1)
tot = sum(v["off0"]["n"] for v in manifest["tasks"].values())
print(f"\nfrozen -> {OUT}   total fitting samples = {tot}")
