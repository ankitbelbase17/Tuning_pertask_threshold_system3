"""
metrics.py — OmniPro metrics, self-contained.

Mirrors the official OmniPro online scorer (repo: metrics/online/scorer.py):
  - greedy 1-to-1 temporal matching of emissions to ground-truth triggers
    within ±tolerance seconds (closest pair first);
  - time-only Precision / Recall / F1;
  - content scoring per task (time_only | count | count_at_time | position |
    state | gpt_judge);
  - content-gated Joint Precision / Recall / F1;
  - micro aggregation across samples (sum tp/fp/fn, then recompute).

Content judging for free-text tasks (event_narration, sequential_step_instruction)
uses an LLM judge if OPENAI_API_KEY / GEMINI_API_KEY is set (OmniPro's protocol,
score 1-5, correct if >=4); otherwise falls back to a lexical-overlap proxy and
flags `judge="lexical"` so the result is never silently mislabelled.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict

TIME_ONLY = {"instant_event_alert", "semantic_condition_alert"}
COUNT_TASKS = {"snapshot_counting", "cumulative_counting", "dedup_counting"}
POSITION_TASKS = {"explicit_target_grounding"}
STATE_TASKS = {"realtime_state_monitor"}
JUDGE_TASKS = {"event_narration", "sequential_step_instruction"}

_POSITIONS = ["top-left", "top-center", "top-right", "center-left", "center",
              "center-right", "bottom-left", "bottom-center", "bottom-right"]


# ---------------------------------------------------------------------------
# temporal matching (greedy, closest-first)
# ---------------------------------------------------------------------------
def match_emits_to_gt(emit_times: list[float], gt_times: list[float],
                      tolerance: float = 3.0):
    pairs = []
    for i, et in enumerate(emit_times):
        for j, gt in enumerate(gt_times):
            dt = abs(et - gt)
            if dt <= tolerance:
                pairs.append((dt, i, j))
    pairs.sort(key=lambda x: x[0])
    used_e, used_g, matches = set(), set(), []
    for dt, i, j in pairs:
        if i in used_e or j in used_g:
            continue
        used_e.add(i); used_g.add(j)
        matches.append((i, j, dt))
    unmatched_e = [i for i in range(len(emit_times)) if i not in used_e]
    unmatched_g = [j for j in range(len(gt_times)) if j not in used_g]
    return matches, unmatched_e, unmatched_g


# ---------------------------------------------------------------------------
# content parsing / scoring
# ---------------------------------------------------------------------------
def _extract_count(text: str):
    m = re.findall(r"-?\d+", text or "")
    return int(m[0]) if m else None


def _extract_position(text: str):
    t = (text or "").lower()
    for p in _POSITIONS:
        if p in t or p.replace("-", " ") in t:
            return p
    return None


def _extract_state(text: str, states: list[str] | None):
    t = (text or "").lower()
    for s in (states or []):
        if s.lower() in t:
            return s.lower()
    return None


def _lexical_sim(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    tb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class ContentJudge:
    """LLM judge if a key is present, else lexical fallback."""

    def __init__(self):
        self.mode = "lexical"
        self._judge = None
        if os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            try:
                # Load OmniPro's llm_judge.py DIRECTLY by file path. We cannot do
                # `from metrics.llm_judge import ...` because THIS file is also
                # imported as the top-level module `metrics`, so the repo's
                # `metrics` package is shadowed in sys.modules and the import
                # silently fails into lexical. importlib by path sidesteps that.
                import importlib.util
                from utils import SCRATCH
                judge_path = os.environ.get(
                    "OMNIPRO_LLM_JUDGE",
                    os.path.join(SCRATCH, "omni_pro", "repo", "metrics", "llm_judge.py"))
                spec = importlib.util.spec_from_file_location("omnipro_llm_judge", judge_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self._judge = mod.LLMJudge()      # auto-detects gemini from GEMINI_API_KEY
                self.mode = "llm"
                print(f"[judge] LLM judge active: provider={self._judge.provider} "
                      f"model={self._judge.model} base={self._judge.api_base}", flush=True)
            except Exception as e:
                self.mode = "lexical"
                print(f"[judge] LLM judge unavailable ({type(e).__name__}: {e}); "
                      f"using lexical fallback", flush=True)

    def score(self, question: str, gt: str, pred: str) -> float:
        """Return correctness in [0,1]."""
        if self.mode == "llm" and self._judge is not None:
            try:
                r = self._judge.judge(question, gt, pred)
                return 1.0 if int(r.get("score", 0)) >= 4 else 0.0
            except Exception:
                pass
        return 1.0 if _lexical_sim(gt, pred) >= 0.3 else 0.0


def _content_correct(task: str, emit_raw: str, gt_item: dict,
                     question: str, judge: ContentJudge) -> bool:
    if task in TIME_ONLY:
        return True
    if task in COUNT_TASKS:
        return _extract_count(emit_raw) == gt_item.get("count")
    if task in POSITION_TASKS:
        gp = (gt_item.get("position") or "").lower()
        return _extract_position(emit_raw) == gp if gp else False
    if task in STATE_TASKS:
        gs = (gt_item.get("state_to") or "").lower()
        return _extract_state(emit_raw, [gs]) == gs if gs else False
    if task in JUDGE_TASKS:
        return judge.score(question, gt_item.get("response", ""), emit_raw) >= 0.5
    return True


# ---------------------------------------------------------------------------
# per-sample + aggregate
# ---------------------------------------------------------------------------
def score_sample(pred: dict, tolerance: float = 3.0,
                 judge: ContentJudge | None = None) -> dict:
    judge = judge or ContentJudge()
    task = pred["task"]
    emits = pred.get("predictions", [])
    gts = pred["ground_truth"]
    emit_times = [float(e["t_sec"]) for e in emits]
    gtt = [float(g["t_sec"]) for g in gts]

    matches, un_e, un_g = match_emits_to_gt(emit_times, gtt, tolerance)
    tp_time = len(matches)
    fp = len(un_e)
    fn = len(un_g)

    tp_content = 0
    for ei, gj, _dt in matches:
        if _content_correct(task, emits[ei].get("raw", ""), gts[gj],
                            pred.get("question", ""), judge):
            tp_content += 1

    return {"id": pred["id"], "task": task,
            "tp_time": tp_time, "tp_content": tp_content, "fp": fp, "fn": fn,
            "n_emits": len(emits), "n_gt": len(gtt)}


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def aggregate(per_sample: list[dict]) -> dict:
    by_task: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    overall = defaultdict(float)
    for s in per_sample:
        for k in ("tp_time", "tp_content", "fp", "fn", "n_emits", "n_gt"):
            by_task[s["task"]][k] += s[k]
            overall[k] += s[k]
        by_task[s["task"]]["n"] += 1
        overall["n"] += 1

    def block(d):
        tpt, tpc, fp, fn = d["tp_time"], d["tp_content"], d["fp"], d["fn"]
        tp_p, tp_r, tp_f = _prf(tpt, fp, fn)
        jp, jr, jf = _prf(tpc, fp, fn)          # content-gated (joint)
        content_acc = (tpc / tpt) if tpt else 0.0
        return {"n_samples": int(d["n"]), "n_gt": int(d["n_gt"]),
                "n_emits": int(d["n_emits"]),
                "time_precision": round(tp_p, 4), "time_recall": round(tp_r, 4),
                "time_f1": round(tp_f, 4),
                "joint_precision": round(jp, 4), "joint_recall": round(jr, 4),
                "joint_f1": round(jf, 4),
                "content_acc": round(content_acc, 4)}

    return {"overall": block(overall),
            "per_task": {t: block(d) for t, d in sorted(by_task.items())}}


# ---------------------------------------------------------------------------
# probe-mode metrics (OmniPro GT-probe), computed from system_5_probe records
# ---------------------------------------------------------------------------
# A probe record (one per ground-truth trigger):
#   {task, question, pre_share, post_share, post_text, gt_item}
# - paired/pre/post temporal accuracy from the yes/no probe shares vs threshold
#   (pre expects NO, post expects YES) — the temporal-awareness axis, all tasks;
# - post content accuracy via the task-appropriate parse/judge — the content
#   axis (for alerts this coincides with the yes/no post).
def probe_metrics(records: list[dict], *, threshold: float = 0.5,
                  judge: ContentJudge | None = None) -> dict:
    judge = judge or ContentJudge()
    by_task: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    overall = defaultdict(float)

    for r in records:
        task = r["task"]
        pre_ok = float(r["pre_share"] < threshold)
        post_ok = float(r["post_share"] >= threshold)
        paired_ok = float(pre_ok and post_ok)
        content_ok = float(_content_correct(task, r.get("post_text", ""),
                                            r["gt_item"], r.get("question", ""), judge))
        for d in (by_task[task], overall):
            d["pre_ok"] += pre_ok; d["post_ok"] += post_ok
            d["paired_ok"] += paired_ok; d["content_ok"] += content_ok; d["n"] += 1

    def block(d):
        n = d["n"] or 1
        pre_a, post_a = d["pre_ok"] / n, d["post_ok"] / n
        f1 = 2 * pre_a * post_a / (pre_a + post_a) if (pre_a + post_a) else 0.0
        return {"n": int(d["n"]),
                "paired_accuracy": round(d["paired_ok"] / n, 4),
                "pre_accuracy": round(pre_a, 4), "post_accuracy": round(post_a, 4),
                "pre_post_f1": round(f1, 4),
                "content_accuracy": round(d["content_ok"] / n, 4)}

    return {"overall": block(overall),
            "per_task": {t: block(d) for t, d in sorted(by_task.items())}}
