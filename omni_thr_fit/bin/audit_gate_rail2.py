#!/usr/bin/env python
"""How much of the firing actually happens BELOW the task's own hit_threshold?

If task_hit_thresholds were the binding gate, no task could fire at p_hit below
its own threshold. Measure the fraction that does.
"""
import re, sys, glob, collections
SAMPLE_RE = re.compile(r"=====\s*\[\d+/\d+\]\s+(\S+)")
GATE_RE   = re.compile(r"ctrl\.gate .*?fire=(True|False).*?p_hit=([0-9.]+)")
SHIPPED = {"cumulative_counting":0.925,"dedup_counting":0.992,"event_narration":0.10,
           "explicit_target_grounding":0.50,"instant_event_alert":0.45,
           "realtime_state_monitor":0.80,"semantic_condition_alert":0.98,
           "sequential_step_instruction":0.01,"snapshot_counting":0.985}

per = collections.defaultdict(list)
logs = sorted(glob.glob("/iopsstor/scratch/cscs/dbartaula/omni_s3_eval/results/"
                        "phaseA_omni_full/lane*.log"))
task = None
for lp in logs:
    with open(lp, errors="replace") as f:
        for line in f:
            m = SAMPLE_RE.search(line)
            if m:
                task = m.group(1).split("::")[0]; continue
            g = GATE_RE.search(line)
            if g and task and g.group(1) == "True":
                per[task].append(float(g.group(2)))

print(f"{'task':<30}{'thr':>7}{'n_fire':>8}{'below thr':>11}{'%below':>8}"
      f"{'in[0.5,thr)':>13}{'p10':>7}{'p50':>7}")
tot_below = tot = 0
for t in sorted(per):
    v = sorted(per[t]); thr = SHIPPED[t]; n = len(v)
    below = sum(1 for x in v if x < thr - 1e-9)
    band  = sum(1 for x in v if 0.5 - 1e-9 <= x < thr - 1e-9)
    tot_below += below; tot += n
    p10 = v[int(0.10*(n-1))]; p50 = v[int(0.50*(n-1))]
    print(f"{t:<30}{thr:>7.3f}{n:>8}{below:>11}{100*below/n:>7.1f}%{band:>13}"
          f"{p10:>7.3f}{p50:>7.3f}")
print(f"\nTOTAL fires {tot}, of which {tot_below} ({100*tot_below/tot:.1f}%) are BELOW "
      f"the task's own configured hit_threshold.")
