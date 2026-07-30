# Eval report — 40 samples, 15392 ticks (tolerance ±3s)

| task | vids | GT | emits | time-F1 | content-F1 | content-acc | prec | rec | AUC | ticks | pos% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dedup_counting | 12 | 45 | 25 | 0.143 | 0.032 | 0.200 | 0.200 | 0.111 | 0.476 | 5240 | 6.2% |
| semantic_condition_alert | 12 | 33 | 5 | 0.053 | 0.053 | 1.000 | 0.200 | 0.030 | 0.550 | 1733 | 12.1% |
| realtime_state_monitor | 7 | 30 | 17 | 0.043 | 0.000 | 0.000 | 0.059 | 0.033 | 0.591 | 1801 | 6.4% |
| event_narration | 5 | 25 | 8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.550 | 4384 | 6.8% |
| snapshot_counting | 2 | 2 | 5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.511 | 584 | 1.2% |
| cumulative_counting | 1 | 4 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.616 | 351 | 6.8% |
| instant_event_alert | 1 | 1 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.574 | 394 | 1.5% |
| **OVERALL** | 34 | 140 | 61 | **0.070** | **0.021** | 0.286 | 0.115 | 0.050 | 0.494 | 14487 | 6.8% |

`time-F1` = ±tol temporal match only. `content-F1` = match must ALSO be content-correct (OmniPro joint). `content-acc` = of temporally matched emits, the fraction also content-correct. `AUC` = ROC of per-tick `p_hit` against a ±tol positive label; **0.5 = chance**.

## Emit times vs ground-truth times

| task | GT times (s) | emit times (s) |
|---|---|---|
| dedup_counting | 1, 1, 10, 10, 14, 19 …+39 | 0, 0, 0, 1, 1, 2 …+19 |
| semantic_condition_alert | 17, 18, 18, 25, 28, 32 …+27 | 1, 4, 17, 27, 81 |
| realtime_state_monitor | 7, 12, 20, 21, 30, 40 …+24 | 0, 0, 0, 0, 0, 2 …+11 |
| event_narration | 60, 75, 98, 135, 135, 160 …+19 | 0, 1, 1, 1, 289, 302 …+2 |
| snapshot_counting | 386, 387 | 204, 381, 417, 427, 430 |
| cumulative_counting | 156, 200, 236, 295 | 240 |
| instant_event_alert | 334 | — |

## Per sample

| task | video | GT | emits | tp_time | tp_content | fp | fn |
|---|---|---:|---:|---:|---:|---:|---:|
| cumulative_counting | cVsCAgSOHJw | 4 | 1 | 0 | 0 | 1 | 4 |
| dedup_counting | 0dJSPsXCujc | 3 | 3 | 1 | 1 | 2 | 2 |
| dedup_counting | 2C1PWbzKIn0 | 2 | 2 | 1 | 0 | 1 | 1 |
| dedup_counting | 3HIvIoG-P1s | 4 | 1 | 0 | 0 | 1 | 4 |
| dedup_counting | CR55TVLjTzc | 4 | 0 | 0 | 0 | 0 | 4 |
| dedup_counting | I1ECITk7WTY | 4 | 1 | 0 | 0 | 1 | 4 |
| dedup_counting | J881Moimbyc | 2 | 2 | 1 | 0 | 1 | 1 |
| dedup_counting | JJPTYTswPCI | 4 | 1 | 0 | 0 | 1 | 4 |
| dedup_counting | KPWKVLDh01Y | 3 | 6 | 1 | 0 | 5 | 2 |
| dedup_counting | TJMrW-vgNz0 | 3 | 3 | 0 | 0 | 3 | 3 |
| dedup_counting | Wtnw9ooAW98 | 5 | 3 | 0 | 0 | 3 | 5 |
| dedup_counting | le0B8XH-W1I | 8 | 2 | 1 | 0 | 1 | 7 |
| dedup_counting | y9XYO9d9H94 | 3 | 1 | 0 | 0 | 1 | 3 |
| event_narration | CR55TVLjTzc | 4 | 3 | 0 | 0 | 3 | 4 |
| event_narration | JJPTYTswPCI | 6 | 3 | 0 | 0 | 3 | 6 |
| event_narration | KE1RZcZvWMw | 5 | 0 | 0 | 0 | 0 | 5 |
| event_narration | LqZBXB5HP9A | 5 | 0 | 0 | 0 | 0 | 5 |
| event_narration | Uaa-Mz84vC8 | 5 | 2 | 0 | 0 | 2 | 5 |
| instant_event_alert | 1SwvQVV5fug | 1 | 0 | 0 | 0 | 0 | 1 |
| realtime_state_monitor | 4QLU8CNu6GQ | 6 | 1 | 0 | 0 | 1 | 6 |
| realtime_state_monitor | 5_LRMz5WxUk | 4 | 1 | 0 | 0 | 1 | 4 |
| realtime_state_monitor | CR55TVLjTzc | 4 | 8 | 1 | 0 | 7 | 3 |
| realtime_state_monitor | TJMrW-vgNz0 | 5 | 1 | 0 | 0 | 1 | 5 |
| realtime_state_monitor | Uaa-Mz84vC8 | 5 | 1 | 0 | 0 | 1 | 5 |
| realtime_state_monitor | noOM42oLy_s | 2 | 4 | 0 | 0 | 4 | 2 |
| realtime_state_monitor | v2Tlscx1Rzc | 4 | 1 | 0 | 0 | 1 | 4 |
| semantic_condition_alert | 0kpiM7TWDZY | 1 | 1 | 0 | 0 | 1 | 1 |
| semantic_condition_alert | 2RgzTn6vV54 | 4 | 0 | 0 | 0 | 0 | 4 |
| semantic_condition_alert | 4JwMlFMzh5Y | 3 | 3 | 0 | 0 | 3 | 3 |
| semantic_condition_alert | DLBbSgu_JdM | 1 | 0 | 0 | 0 | 0 | 1 |
| semantic_condition_alert | L0RIDVNu39s | 3 | 0 | 0 | 0 | 0 | 3 |
| semantic_condition_alert | MEqYM_4UbRs | 4 | 0 | 0 | 0 | 0 | 4 |
| semantic_condition_alert | Wnx9R5oq8ys | 2 | 0 | 0 | 0 | 0 | 2 |
| semantic_condition_alert | c0fPidVkbds | 5 | 0 | 0 | 0 | 0 | 5 |
| semantic_condition_alert | dA2PeQKIWWI | 2 | 1 | 1 | 1 | 0 | 1 |
| semantic_condition_alert | pvSVznXAbug | 2 | 0 | 0 | 0 | 0 | 2 |
| semantic_condition_alert | vkfKi0Ui3gU | 4 | 0 | 0 | 0 | 0 | 4 |
| semantic_condition_alert | y7moDAUZ0Is | 2 | 0 | 0 | 0 | 0 | 2 |
| snapshot_counting | 0dJSPsXCujc | 1 | 0 | 0 | 0 | 0 | 1 |
| snapshot_counting | UZQZ2fAoTn4 | 1 | 5 | 0 | 0 | 5 | 1 |

---

## Diagnosis (2026-07-30, 40 samples / 15,627 ticks / 43 videos)

### 1. The rising-edge gate is a hard recall ceiling

| | measured |
|---|---:|
| ticks | 15,627 |
| `level=True` | 4,014 (25.7%) |
| **rising edges (the only thing that can fire)** | **61** |
| falling edges (needed to re-arm) | 61 |
| videos latched `True` and never fell | 6 / 43 |
| **mean rising edges per video** | **1.42** |
| mean ground-truth events per video | **3.5** |

The gate can fire at most 1.42 times per video against 3.5 expected events, so
**maximum achievable recall is 61/140 = 0.44 no matter how good the model is.**
Measured recall is 0.050, so the gate costs us ~9x and the signal costs the rest.

### 2. `p_hit` behaves like a scene descriptor, not an event detector

`level` is TRUE 25.7% of the time but changes only 122 times in 15,627 ticks —
long sticky runs, not event-shaped spikes. That is exactly the profile of a signal
tracking *topic/scene* rather than *"the condition is satisfied at this instant"*,
and it explains AUC ≈ 0.5 against point-in-time labels.

### 3. Fires cluster at video start

| video-time bucket | fires | share |
|---|---:|---:|
| 0–2 s | 15 | **19.5%** |
| 2–5 s | 3 | 3.9% |
| 5–15 s | 13 | 16.9% |
| 15–60 s | 15 | 19.5% |
| 60 s+ | 31 | 40.3% |

For a ~300 s video the first 2 s is 0.7% of the timeline but takes 19.5% of alerts:
the level goes TRUE on the opening ticks, fires, then latches. Visible directly in
the emit times — `realtime_state_monitor` emits at 0,0,0,0,0,2 against GT 7,12,20,21,30,40.

### 4. Under-emission overall
61 emits against 140 ground-truth triggers. The system is mostly silent.

### What this rules out
- **Not** the logit read: verified 647/647 against free decode.
- **Not** JSON/format: 100% valid, 100% field compliance.
- **Not** threshold choice alone: at AUC ≈ 0.5 no threshold helps, and the
  per-task thresholds in `config.py` were fitted on 3 videos of noise.

### Ranked next steps
1. **Replace the rising-edge gate.** 1.42 firings/video cannot cover 3.5 events.
2. **Re-ask the question.** `have_enough_info` is producing a sticky scene signal;
   an event-shaped question ("did X *just start* in the last second?") is the
   experiment.
3. **Delete `task_hit_thresholds`.** Fitted on noise.
