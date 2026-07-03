A folder omniprofast/ has been copied into this repo. It's a read-only evaluation harness for the OmniPro benchmark — it measures system_5's proactive streaming behavior (does the model fire an alert at the right video-time?). It does not reimplement system_5; it spins up your real pipeline (encoder_thread → orchestrator_thread → writer_thread over a shared KVCacheManager) unmodified and captures emissions. Your job is to wire it to this repo's prosync and confirm the interface matches. Only ONLINE mode + system5 matters; ignore the probe/system4 adapter files.

1. Point it at this repo's system_5. Edit omniprofast/run_fast.sh, the hardcoded paths near the top:
- PROSYNC_DIR → must point to this repo's prosync package (the system_5 code). This is the only path that's truly repo-specific.
- PY → a python with prosync's deps (torch, transformers, av, requests). Reuse the same conda env if deps match.
- OMNIPRO_DATASET_DIR, HF_HOME, SCRATCH → same machine, so likely unchanged (they point at shared scratch). Verify they exist.

The harness resolves system_5 via utils.py (SYSTEM5_DIR = $PROSYNC_DIR, fallback ../../prosync) and prepends it to sys.path.

2. Verify the integration contract — this is the important part. The adapter (omniprofast/system5_adapter.py) calls exactly this surface of your code. Grep your prosync and confirm each still matches; if a name/signature differs, adapt the adapter (that's the intended seam), not your system_5:

- config.AsyncOmniConfig(mode="video", dtype, device, profile, model_id, kv_budget), and it does dataclasses.replace(cfg, ...) setting these fields — all must exist by these names or replace raises: system_prompt, event_question, writer_prompt, event_threshold, prompt_mode, gate_mode, video_path, max_seconds, realtime, groundtruth, fps. It also reads cfg.kv_budget, cfg.frame_q_size, cfg.audio_q_size, cfg.fps, cfg.encoder_idle_fps, cfg.encoder_focus_fps. (Note: it maps the old goal_question/goal_threshold → event_question/event_threshold. If your config uses yet another name, fix the mapping at system5_adapter.py:119-121.)
- prompt_mode="task" and gate_mode="fixed" must be accepted values (use injected prompts verbatim + honor a fixed threshold).
- backend.build_backend(cfg), manager.KVCacheManager(backend, kv_budget=, prof=None), util.EncoderControl(fps, idle_fps, focus_fps).
- Thread signatures (system5_adapter.py:139-149):
  - vision_stream.encoder_thread(cfg, backend, in_q, ctrl, stop, None)
  - orchestrator.orchestrator_thread(cfg, mgr, in_q, writer_q, ctrl, stop, done, None, ev)
  - writer.writer_thread(cfg, mgr, writer_q, done, None, ev, backend)
- Evaluator hooks — your threads call these on the ev object, and all three must exist or you crash after model load: ev.record_gate(vt, share, thr) (called every gate tick), ev.record_trigger(vt, share) (on threshold crossing), ev.record_write(vt, text, wall_latency). The harness's CaptureEvaluator already implements all three. If your orchestrator calls any other hook name, add it to CaptureEvaluator (system5_adapter.py:41). (I already had to add record_gate in the source repo — check whether your version calls something else too.)

3. Two gotchas already handled — don't undo them:
- metrics.py loads the OmniPro LLM judge via importlib by file path (not from metrics.llm_judge import), because this file is itself named metrics and shadows the package. Leave that as is.
- The LLM judge (only for event_narration / sequential_step_instruction) needs both GEMINI_API_KEY and GEMINI_API_BASE in the repo-root .env (base defaults to a placeholder). run_fast.sh sources the nearest .env and warns if the base is missing.

4. Run it:
cd omniprofast
CUDA_VISIBLE_DEVICES=0 bash run_fast.sh    # 27 videos, system_5 online, one prompt variant
Smoke test one video first:
CUDA_VISIBLE_DEVICES=0 bash run_fast.sh --tasks semantic_condition_alert --limit 1 --shortest

5. What a healthy result looks like (reference from the source repo, same 41s clip): after "backend loaded", you should see a line like semantic_condition_alert 1/1 emits=17 gt=3 tp=3 fp=14 fn=0 and time_f1=0.3. Outputs land in omniprofast/output/<variant>/online_pred.jsonl + online_metrics.json. Key metrics: time_f1 (did it fire within ±3s of GT triggers), content_acc, and per-task breakdown. High fp = model over-fires; that's a real model/threshold signal, not a harness bug.

6. If you change your system_5 and want to compare versions: re-run and diff online_metrics.json. Timing (time_f1) is deterministic enough for version comparison in batch mode; treat small emission-time jitter as noise since the pipeline is multi-threaded.

Read omniprofast/README.md for the subset details and make_subset.py if you want to change how many videos per task (default 3 shortest, audio=none).