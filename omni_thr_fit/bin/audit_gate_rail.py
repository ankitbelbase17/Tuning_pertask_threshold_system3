#!/usr/bin/env python
"""Per-task p_hit-at-fire audit of a completed run's ctrl.gate logs.

WHY. THRESHOLD_FIT_v2 sec.1 assumes OMNIPRO_HIT_THRESHOLD is THE firing knob.
It is not the only one: controller.py:608 runs gate_strategy="hysteresis", whose
fire test is `p_hit >= cfg.gate_high_thr` (a fixed global 0.5), while
cfg.hit_threshold only decides whether an `answer` is decoded at all. The
effective rail is therefore max(hit_threshold, 0.5). This script measures that
rail directly off a finished run instead of arguing about it.
"""
import re, sys, glob, os, collections

SAMPLE_RE = re.compile(r"=====\s*\[\d+/\d+\]\s+(\S+)")
GATE_RE   = re.compile(r"ctrl\.gate .*?fire=(True|False).*?p_hit=([0-9.]+)")

def main(logs):
    per = collections.defaultdict(lambda: {"fire": [], "nofire": []})
    task = None
    for lp in logs:
        with open(lp, errors="replace") as f:
            for line in f:
                m = SAMPLE_RE.search(line)
                if m:
                    task = m.group(1).split("::")[0]
                    continue
                g = GATE_RE.search(line)
                if g and task:
                    per[task]["fire" if g.group(1) == "True" else "nofire"].append(float(g.group(2)))
    SHIPPED = {"cumulative_counting":0.925,"dedup_counting":0.992,"event_narration":0.10,
               "explicit_target_grounding":0.50,"instant_event_alert":0.45,
               "realtime_state_monitor":0.80,"semantic_condition_alert":0.98,
               "sequential_step_instruction":0.01,"snapshot_counting":0.985}
    print(f"{'task':<30}{'shipped':>8}{'min p_hit@fire':>16}{'n_fire':>9}{'n_tick':>9}"
          f"{'rail':>8}  verdict")
    for t in sorted(per):
        fr, nf = per[t]["fire"], per[t]["nofire"]
        if not fr:
            print(f"{t:<30}{SHIPPED.get(t,float('nan')):>8.3f}{'-':>16}{0:>9}{len(nf):>9}")
            continue
        mn = min(fr)
        shipped = SHIPPED.get(t, float("nan"))
        rail = max(shipped, 0.5)
        # the rail is CONFIRMED if the lowest firing p_hit sits at it, not at `shipped`
        verdict = ("CLAMPED by gate_high_thr=0.5" if shipped < 0.5 - 1e-9 and mn >= 0.5 - 1e-3
                   else "threshold binds")
        print(f"{t:<30}{shipped:>8.3f}{mn:>16.4f}{len(fr):>9}{len(fr)+len(nf):>9}"
              f"{rail:>8.3f}  {verdict}")

if __name__ == "__main__":
    logs = sys.argv[1:] or sorted(glob.glob(
        "/iopsstor/scratch/cscs/dbartaula/omni_s3_eval/results/phaseA_omni_full/lane*.log"))
    print(f"# {len(logs)} lane logs")
    main(logs)
