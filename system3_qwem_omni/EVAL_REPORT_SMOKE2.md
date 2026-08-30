# Eval report — 38 samples, 4241 ticks (tolerance ±3s)

| task | vids | GT | emits | time-F1 | content-F1 | content-acc | prec | rec | AUC | ticks | pos% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic_condition_alert | 20 | 57 | 93 | 0.200 | 0.062 | 0.267 | 0.161 | 0.263 | 0.554 | 2212 | 16.6% |
| instant_event_alert | 17 | 23 | 113 | 0.118 | 0.118 | 1.000 | 0.071 | 0.348 | 0.561 | 1728 | 6.9% |
| event_narration | 1 | 4 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0 | 0.0% |
| **OVERALL** | 35 | 84 | 206 | **0.159** | **0.090** | 0.522 | 0.112 | 0.274 | 0.553 | 3940 | 12.4% |

`time-F1` = ±tol temporal match only. `content-F1` = match must ALSO be content-correct (OmniPro joint). `content-acc` = of temporally matched emits, the fraction also content-correct. `AUC` = ROC of per-tick `p_hit` against a ±tol positive label; **0.5 = chance**.

## Emit times vs ground-truth times

| task | GT times (s) | emit times (s) |
|---|---|---|
| semantic_condition_alert | 8, 10, 13, 15, 16, 17 …+51 | 0, 0, 1, 4, 7, 9 …+87 |
| instant_event_alert | 3, 7, 8, 10, 17, 25 …+17 | 1, 1, 1, 1, 2, 3 …+107 |
| event_narration | 25, 50, 75, 108 | — |

## Per sample

| task | video | GT | emits | tp_time | tp_content | fp | fn |
|---|---|---:|---:|---:|---:|---:|---:|
| event_narration | ln-yFLZ5wcc | 4 | 0 | 0 | 0 | 0 | 4 |
| instant_event_alert | 28bX9kzKMys | 1 | 12 | 1 | 1 | 11 | 0 |
| instant_event_alert | 2Vh-xCGMLzg | 1 | 1 | 0 | 0 | 1 | 1 |
| instant_event_alert | 2llGbAybvB0 | 1 | 1 | 1 | 1 | 0 | 0 |
| instant_event_alert | 4JwMlFMzh5Y | 1 | 26 | 0 | 0 | 26 | 1 |
| instant_event_alert | 7shRl0Lxycc | 1 | 3 | 0 | 0 | 3 | 1 |
| instant_event_alert | EUXY75I2qr0 | 2 | 0 | 0 | 0 | 0 | 2 |
| instant_event_alert | HdbBkWjBQWw | 2 | 26 | 1 | 1 | 25 | 1 |
| instant_event_alert | IWquNO7pJ_Q | 1 | 5 | 0 | 0 | 5 | 1 |
| instant_event_alert | JVuXCJWfc1M | 2 | 14 | 1 | 1 | 13 | 1 |
| instant_event_alert | L0RIDVNu39s | 1 | 3 | 0 | 0 | 3 | 1 |
| instant_event_alert | LFelGo9gA_I | 2 | 1 | 1 | 1 | 0 | 1 |
| instant_event_alert | L_ZmMIIdg0A | 1 | 3 | 1 | 1 | 2 | 0 |
| instant_event_alert | Lno78zPUEXs | 1 | 9 | 1 | 1 | 8 | 0 |
| instant_event_alert | TXeVf42sQdM | 1 | 5 | 0 | 0 | 5 | 1 |
| instant_event_alert | Ull7qP303ds | 1 | 1 | 1 | 1 | 0 | 0 |
| instant_event_alert | V9Kl1FK8fUU | 1 | 0 | 0 | 0 | 0 | 1 |
| instant_event_alert | Ws4T24vp6rI | 3 | 3 | 0 | 0 | 3 | 3 |
| semantic_condition_alert | 0kpiM7TWDZY | 1 | 6 | 1 | 0 | 5 | 0 |
| semantic_condition_alert | 2a0O3-5Fpyw | 5 | 0 | 0 | 0 | 0 | 5 |
| semantic_condition_alert | 36Ush_U7z-c | 1 | 15 | 0 | 0 | 15 | 1 |
| semantic_condition_alert | 3ezGUOPrk9c | 4 | 0 | 0 | 0 | 0 | 4 |
| semantic_condition_alert | 6Hb8uzHLA2Q | 1 | 5 | 1 | 0 | 4 | 0 |
| semantic_condition_alert | DLBbSgu_JdM | 1 | 0 | 0 | 0 | 0 | 1 |
| semantic_condition_alert | GEmaF4_W0RE | 2 | 0 | 0 | 0 | 0 | 2 |
| semantic_condition_alert | L0RIDVNu39s | 3 | 0 | 0 | 0 | 0 | 3 |
| semantic_condition_alert | MEqYM_4UbRs | 4 | 0 | 0 | 0 | 0 | 4 |
| semantic_condition_alert | MqarNjlLyvk | 4 | 2 | 1 | 0 | 1 | 3 |
| semantic_condition_alert | N56253QAq_o | 2 | 0 | 0 | 0 | 0 | 2 |
| semantic_condition_alert | RWpTUgVDOB0 | 4 | 2 | 1 | 0 | 1 | 3 |
| semantic_condition_alert | TXeVf42sQdM | 1 | 7 | 0 | 0 | 7 | 1 |
| semantic_condition_alert | Ull7qP303ds | 2 | 4 | 2 | 1 | 2 | 0 |
| semantic_condition_alert | Wnx9R5oq8ys | 2 | 19 | 1 | 0 | 18 | 1 |
| semantic_condition_alert | YnUdAck_oM0 | 6 | 15 | 6 | 1 | 9 | 0 |
| semantic_condition_alert | c0fPidVkbds | 5 | 3 | 0 | 0 | 3 | 5 |
| semantic_condition_alert | n3fCfTKfWOM | 5 | 14 | 2 | 2 | 12 | 3 |
| semantic_condition_alert | pvSVznXAbug | 2 | 1 | 0 | 0 | 1 | 2 |
| semantic_condition_alert | y7moDAUZ0Is | 2 | 0 | 0 | 0 | 0 | 2 |
