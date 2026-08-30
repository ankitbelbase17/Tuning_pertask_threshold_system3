# Eval report — 53 samples, 17260 ticks (tolerance ±3s)

| task | vids | GT | emits | time-F1 | content-F1 | content-acc | prec | rec | AUC | ticks | pos% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic_condition_alert | 25 | 77 | 25 | 0.118 | 0.062 | 0.500 | 0.240 | 0.078 | 0.570 | 4024 | 12.1% |
| dedup_counting | 12 | 45 | 25 | 0.143 | 0.032 | 0.200 | 0.200 | 0.111 | 0.476 | 5240 | 6.2% |
| realtime_state_monitor | 7 | 30 | 17 | 0.043 | 0.000 | 0.000 | 0.059 | 0.033 | 0.591 | 1801 | 6.4% |
| event_narration | 5 | 25 | 8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.550 | 4384 | 6.8% |
| snapshot_counting | 2 | 2 | 5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.511 | 584 | 1.2% |
| cumulative_counting | 1 | 4 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.616 | 351 | 6.8% |
| instant_event_alert | 1 | 1 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.574 | 394 | 1.5% |
| **OVERALL** | 47 | 184 | 81 | **0.091** | **0.032** | 0.333 | 0.148 | 0.065 | 0.506 | 16778 | 7.5% |

`time-F1` = ±tol temporal match only. `content-F1` = match must ALSO be content-correct (OmniPro joint). `content-acc` = of temporally matched emits, the fraction also content-correct. `AUC` = ROC of per-tick `p_hit` against a ±tol positive label; **0.5 = chance**.

## Emit times vs ground-truth times

| task | GT times (s) | emit times (s) |
|---|---|---|
| semantic_condition_alert | 8, 10, 10, 13, 16, 17 …+71 | 0, 0, 1, 2, 4, 6 …+19 |
| dedup_counting | 1, 1, 10, 10, 14, 19 …+39 | 0, 0, 0, 1, 1, 2 …+19 |
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
| semantic_condition_alert | -IiTYqQKXlg | 2 | 0 | 0 | 0 | 0 | 2 |
| semantic_condition_alert | 0kpiM7TWDZY | 1 | 1 | 0 | 0 | 1 | 1 |
| semantic_condition_alert | 2RgzTn6vV54 | 4 | 0 | 0 | 0 | 0 | 4 |
| semantic_condition_alert | 36Ush_U7z-c | 1 | 1 | 1 | 1 | 0 | 0 |
| semantic_condition_alert | 4JwMlFMzh5Y | 3 | 3 | 0 | 0 | 3 | 3 |
| semantic_condition_alert | DLBbSgu_JdM | 1 | 0 | 0 | 0 | 0 | 1 |
| semantic_condition_alert | FL9j4DUagn4 | 8 | 1 | 1 | 0 | 0 | 7 |
| semantic_condition_alert | GvBRNnsucao | 3 | 0 | 0 | 0 | 0 | 3 |
| semantic_condition_alert | IW7FuSyFdIs | 3 | 0 | 0 | 0 | 0 | 3 |
| semantic_condition_alert | L0RIDVNu39s | 3 | 0 | 0 | 0 | 0 | 3 |
| semantic_condition_alert | MEqYM_4UbRs | 4 | 0 | 0 | 0 | 0 | 4 |
| semantic_condition_alert | MnnT5w7Uey8 | 6 | 0 | 0 | 0 | 0 | 6 |
| semantic_condition_alert | MqarNjlLyvk | 4 | 2 | 0 | 0 | 2 | 4 |
| semantic_condition_alert | SpJdncoK-QY | 1 | 0 | 0 | 0 | 0 | 1 |
| semantic_condition_alert | Ull7qP303ds | 2 | 5 | 2 | 1 | 3 | 0 |
| semantic_condition_alert | Wnx9R5oq8ys | 2 | 0 | 0 | 0 | 0 | 2 |
| semantic_condition_alert | XUGBvAVZ8CI | 3 | 0 | 0 | 0 | 0 | 3 |
| semantic_condition_alert | YJV5yWfrB98 | 2 | 1 | 0 | 0 | 1 | 2 |
| semantic_condition_alert | YnUdAck_oM0 | 6 | 2 | 0 | 0 | 2 | 6 |
| semantic_condition_alert | c0fPidVkbds | 5 | 0 | 0 | 0 | 0 | 5 |
| semantic_condition_alert | dA2PeQKIWWI | 2 | 1 | 1 | 1 | 0 | 1 |
| semantic_condition_alert | pvSVznXAbug | 2 | 0 | 0 | 0 | 0 | 2 |
| semantic_condition_alert | qHgcvGUjyJE | 3 | 8 | 1 | 0 | 7 | 2 |
| semantic_condition_alert | vkfKi0Ui3gU | 4 | 0 | 0 | 0 | 0 | 4 |
| semantic_condition_alert | y7moDAUZ0Is | 2 | 0 | 0 | 0 | 0 | 2 |
| snapshot_counting | 0dJSPsXCujc | 1 | 0 | 0 | 0 | 0 | 1 |
| snapshot_counting | UZQZ2fAoTn4 | 1 | 5 | 0 | 0 | 5 | 1 |
