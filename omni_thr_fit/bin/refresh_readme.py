#!/usr/bin/env python3
"""refresh_readme.py -- regenerate the README's status block from the records.

The README carries numbers that go stale the moment a generation lands: how much
of each stage is banked, how many prediction records there are, what they cost.
Hand-editing them is how a README ends up claiming "Pass 2 in progress" a day
after pass 2 finished, so the block between the AUTOSYNC markers belongs to this
script and nothing else edits it.

Every number is RECOUNTED from the committed records, never incremented. An
incremented counter drifts silently the first time a cycle is missed or a file is
re-synced; a recount cannot.

  python3 bin/refresh_readme.py             rewrite the block in $PUSH_STAGE/README.md
  python3 bin/refresh_readme.py --summary   one line, for a commit subject
"""
# No `from __future__ import annotations`: the login nodes' system python3 is
# older than 3.7 and rejects it outright. This script has to run from the
# unattended cycle, where whatever python3 is on PATH is the one it gets.
import glob, json, os, sys

BEGIN = "<!-- AUTOSYNC:BEGIN -->"
END = "<!-- AUTOSYNC:END -->"

# (directory under results/, label, denominator). The denominators are the run
# sizes fixed by the spec, not derived from the files -- a stage that is missing
# records must read as incomplete, not as complete against a smaller total.
STAGES = [
    ("p1", "pass 1 (wide grid, 94 cells)", 1410),
    ("p2", "pass 2 (refined grid, 45 cells)", 675),
    ("full2700", "stage 3 arm 1 (`fitted`, per-task)", 2700),
]


def scan(root):
    """Counts per stage, deduplicated PER CELL.

    A cell is the directory holding the lane*/ dirs: results/p1/<task>/thr_X for
    the grid passes, results/full2700 for stage 3. The unit of work is one
    (cell, id) pair, NOT an id -- pass 1 evaluates the same 15 videos in all 94
    cells on purpose, so counting distinct ids across the tree would report
    thousands of "duplicates" that are the experiment working as designed. That
    is the shape of the bug this docstring exists to stop someone re-introducing.
    """
    tot, uniq, dups, gpu_s, arms = {}, {}, {}, {}, {}
    files = sorted(glob.glob(os.path.join(root, "omni_thr_fit/results/**/online_pred.jsonl"),
                             recursive=True))
    torn = 0
    cell_ids = {}
    for fp in files:
        rel = os.path.relpath(fp, root)
        top = rel.split("/")[2]
        cell = os.path.dirname(os.path.dirname(rel))
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                torn += 1
                continue
            tot[top] = tot.get(top, 0) + 1
            cell_ids.setdefault((top, cell), []).append(r.get("id"))
            if r.get("wall_s"):
                gpu_s[top] = gpu_s.get(top, 0.0) + float(r["wall_s"])
            if top == "full2700":
                arms[r.get("arm") or ""] = arms.get(r.get("arm") or "", 0) + 1
    for (top, _), ids in cell_ids.items():
        uniq[top] = uniq.get(top, 0) + len(set(ids))
        dups[top] = dups.get(top, 0) + len(ids) - len(set(ids))
    nbytes = sum(os.path.getsize(f) for f in files)
    return files, tot, uniq, dups, gpu_s, arms, nbytes, torn


def summary(uniq, gpu_s, nbytes, files):
    done = uniq.get("full2700", 0)
    return ("arm 1 at %s/2,700 (%.0f%%); %s sample-evals, %.0f GPU-h, %.1f MB in %s files"
            % (f"{done:,}", 100.0 * done / 2700, f"{sum(uniq.values()):,}",
               sum(gpu_s.values()) / 3600, nbytes / 1e6, f"{len(files):,}"))


def block(files, tot, uniq, dups, gpu_s, arms, nbytes, torn):
    L = ["| stage | banked | state |", "|---|---|---|"]
    for key, label, denom in STAGES:
        n = uniq.get(key, 0)
        state = "**complete**" if n >= denom else "**running** — %.0f%%" % (100.0 * n / denom)
        L.append("| %s | %s/%s | %s |" % (label, f"{n:,}", f"{denom:,}", state))
    # Arm 2 is reported as CANCELLED, not omitted. A control that was designed and
    # then dropped is a limitation of the study; a table that simply stops listing
    # it reads as though the study never needed one.
    L.append("| stage 3 arm 2 (`g015`, flat global 0.15) | — | **not run** — cancelled "
             "2026-09-01, before launch (see RUNBOOK §2.10) |")
    arm_txt = ", ".join("`arm=%s` %s" % (k or '""', f"{v:,}") for k, v in sorted(arms.items()))
    dup_txt = ", ".join("%s %d" % (k, dups.get(k, 0)) for k, _, _ in STAGES)
    return "\n".join([
        BEGIN,
        "",
        "\n".join(L),
        "",
        "**Banked at the last sync:** %s prediction files, %s sample-evals (%s raw "
        "records), %.1f MB, ≈ %.0f GPU-hours." % (
            f"{len(files):,}", f"{sum(uniq.values()):,}", f"{sum(tot.values()):,}",
            nbytes / 1e6, sum(gpu_s.values()) / 3600),
        "",
        "**Integrity: %d torn lines** across every generation and lane reshape, despite a "
        "SIGKILL at each 22-minute wall. Re-evaluations, counted per cell — %s. The grid "
        "passes double-evaluated a handful (two lanes claiming the same unit before either "
        "banked it); stage 3's shard split with global `--done_glob` resume has produced "
        "**none**. Either way no score moves: `lib/score_cells.py` keys its records by id "
        "before scoring, so a repeat replaces rather than double-counts." % (torn, dup_txt),
        "",
        "Stage 3 arm labels: %s. The empty label predates the 2026-08-31 fix that stamps "
        "`OMNIPRO_ARM` into each record; those are all `fitted`, the only arm that had run "
        "while the field was empty." % arm_txt,
        "",
        "`content_acc` and `joint_f1` stay **`WITHHELD`, never guessed**, until an LLM "
        "judge is reachable. `time_f1` is judge-free and is what every result is selected on.",
        "",
        "<sub>This block is regenerated by `omni_thr_fit/bin/refresh_readme.py`; every "
        "number is recounted from the records, never incremented.</sub>",
        "",
        END,
    ])


def main():
    root = os.environ.get("PUSH_STAGE")
    if not root:
        sys.exit("PUSH_STAGE must name the staging repo")
    files, tot, uniq, dups, gpu_s, arms, nbytes, torn = scan(root)
    if "--summary" in sys.argv:
        print(summary(uniq, gpu_s, nbytes, files))
        return
    p = os.path.join(root, "README.md")
    s = open(p).read()
    if BEGIN not in s or END not in s:
        sys.exit("README has no AUTOSYNC block; refusing to guess where it goes")
    head, rest = s.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    open(p, "w").write(head + block(files, tot, uniq, dups, gpu_s, arms, nbytes, torn) + tail)
    print(summary(uniq, gpu_s, nbytes, files))


if __name__ == "__main__":
    main()
