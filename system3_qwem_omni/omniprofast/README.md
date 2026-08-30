# omniprofast — fast, portable OmniPro subset eval (system_5, online)

A self-contained duplicate of `eval/omniproeval` trimmed for **quick iteration**:
it scores the **N shortest videos per task** instead of the full 2,700-sample
sweep, so you get near-faithful metrics in minutes. Drop this whole folder into
another repo, point three env vars at your setup, and run.

Nothing here is a wrapper — every module (`evaluate.py`, `dataset.py`,
`metrics.py`, `prompts.py`, `utils.py`, the adapters) is a real copy and can be
edited independently of the original.

## What's different from `omniproeval`
- **Portable paths** (`utils.py`): `OMNIPRO_DATASET_DIR`, `OMNIPRO_BENCHMARK_JSON`,
  `OMNIPRO_OUTPUT_DIR` are env-overridable. Defaults: score the bundled
  `benchmark_mini.json`, write to `./output`. No `$SCRATCH/eval` writes.
- **Shortest-video selection** (`dataset.py`): `load_samples(..., shortest=True)`
  ranks each task's candidates by duration before applying the per-task limit.
- **New CLI flags** (`evaluate.py`): `--shortest`, `--benchmark_json`, `--dataset_dir`.
- **Bundled subset** (`make_subset.py` → `benchmark_mini.json`): the manifest of
  chosen samples travels with the code, so you don't need the full benchmark in
  the target repo — only the video files.

## Quick start
```bash
# (optional) regenerate the subset: 3 shortest per task, no-audio samples
python make_subset.py --per_task 3 --audio none

# run system_5 online on the bundled subset, one prompt variant
bash run_fast.sh
# -> writes output/<variant>/online_pred.jsonl + online_metrics.json
```

### Moving to another repo
Copy the folder, then set:
```bash
export PROSYNC_DIR=/path/to/prosync            # the model pipeline package
export OMNIPRO_DATASET_DIR=/path/to/omni_pro/dataset   # where raw_videos/*.mp4 live
export PY=/path/to/python-with-prosync-deps
bash run_fast.sh
```
The bundled `benchmark_mini.json` keeps relative `video_path`s (e.g.
`raw_videos/<id>.mp4`), so they resolve against `OMNIPRO_DATASET_DIR` wherever
you run.

## The subset currently bundled
27 samples = 9 tasks × 3 shortest, `audio_dependency == none` (system_5 is
vision-only, so no-audio samples are the fair, faithful choice). ~31 min of
footage at 1 fps. Regenerate smaller/larger with `--per_task`.

> Note: system_5's proactive writer only emits meaningfully on its **native
> tasks** — `instant_event_alert`, `semantic_condition_alert`, `event_narration`.
> The counting/position/state tasks need structured outputs it isn't built to
> produce, so it mostly stays silent on them (which is faithful to the full run).
> For the fastest signal, restrict with e.g.
> `python make_subset.py --tasks instant_event_alert,semantic_condition_alert,event_narration`.

## LLM-as-judge (free-text tasks)
Only `event_narration` and `sequential_step_instruction` are judged by an LLM
(`metrics.py`, `JUDGE_TASKS`). `ContentJudge` decides its mode at construction:
- If `OPENAI_API_KEY` **or** `GEMINI_API_KEY` is set **and** OmniPro's judge is
  importable from `$SCRATCH/omni_pro/repo/metrics/llm_judge.py`, it uses the LLM:
  score 1–5, **correct iff ≥ 4** (OmniPro protocol).
- Otherwise it falls back to a **lexical** Jaccard-overlap proxy (correct iff
  ≥ 0.3) and labels the run `content_judge: "lexical"` so it's never silently
  mislabeled. It never crashes for a missing key.

In another repo, LLM mode only activates if that `metrics/llm_judge.py` path
exists — otherwise you get the lexical proxy even with a key set. The timing
metric (`time_f1`) never touches the judge.
