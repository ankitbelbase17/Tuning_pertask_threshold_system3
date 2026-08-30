# Per-task gate threshold fit — Qwen2.5-Omni-7B on OmniPro Online

This repository mirrors the working directory `system3_qwem_omni_8_28/`, which
holds two trees that are deliberately kept separate:

| folder | what it is |
|---|---|
| **`system3_qwem_omni/`** | the system under study — a fork of `system_3` that swaps the vision-only Qwen3-VL-8B backbone for **Qwen2.5-Omni-7B** (vision **+ audio**). Start at [`system3_qwem_omni/CLAUDE.md`](system3_qwem_omni/CLAUDE.md). |
| **`omni_thr_fit/`** | the threshold-fit *experiment* — fleets, worklists, banked results, audits, figures and the paper. Kept out of the system tree so that **a run never mutates the code it is measuring**. |

## Where to start

- **What the system is, and the findings so far** → `system3_qwem_omni/CLAUDE.md`
- **How the experiment is run** → `system3_qwem_omni/THRESHOLD_FIT_RUNBOOK.md`
  (§2.5–2.7 carry the three substantive results)
- **The spec** → `system3_qwem_omni/THRESHOLD_FIT_v2.md`
- **The paper** → `omni_thr_fit/paper/` (nine sections, builds with `lib/build_pdf.py`)

## The headline result

`p_hit` does not rank event-adjacent ticks above quiet ones (per-task AUC
0.431–0.551, every 95 % CI containing 0.5). A threshold on such a score buys
emission *volume*, not *timing* — so pooled over all tasks a **single global**
operating point is identified and stable (re-selected in 94 % of bootstrap
resamples), while **per-task** thresholds are not (25–61 %, and no task's fitted
value beats its own best rival outside noise). Details and the falsifiable
predictions recorded in advance are in the runbook.

## What is NOT in this repository, and why

The rule is: commit anything expensive to **regenerate**, exclude anything large
that a re-run reproduces.

| excluded | size | why |
|---|---|---|
| `omni_thr_fit/repo/` | 2.4 GB | a full copy of `system3_qwem_omni/`, made so a run never mutates the source tree. **Recreate it by copying that folder** — do not version it twice. |
| `omni_thr_fit/**/run.log` | 444 MB | verbose per-tick logs; any re-run reproduces them |
| core dumps | ~7 GB | torch crash artefacts |
| `.env` | — | **contains live API keys**; recreate locally with `GEMINI_API_KEY` and `OPENAI_API_KEY` |

The banked predictions (`omni_thr_fit/results/**/online_pred.jsonl`, ~6.6 MB
across 576 files) **are** committed. They represent roughly 150 GPU-hours, and
every number in the study is re-scorable from them offline with no GPU at all:

```bash
export THR_ROOT=$PWD/omni_thr_fit REPO=$PWD/system3_qwem_omni
python omni_thr_fit/lib/score_cells.py --pass p1
python omni_thr_fit/lib/pick.py        --pass p1
python omni_thr_fit/bin/audit_fit_noise.py --pass p1
python omni_thr_fit/lib/ablation.py    --pass p1
```

## Status

Pass 1 complete (1410/1410 samples, 94/94 cells). Pass 2 in progress. Stage 3
(the full 2,700-sample evaluation) not yet run — content-acc and joint-F1 are
`WITHHELD` throughout until a judge is reachable, never guessed.
