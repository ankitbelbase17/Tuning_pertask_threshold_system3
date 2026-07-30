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

import json
import os
import re
from collections import defaultdict

TIME_ONLY = {"instant_event_alert"}
COUNT_TASKS = {"snapshot_counting", "cumulative_counting", "dedup_counting"}
POSITION_TASKS = {"explicit_target_grounding"}
STATE_TASKS = {"realtime_state_monitor"}
# Per the OmniPro paper, semantic_condition_alert responses are content-judged by
# an LLM: a temporal match counts toward the joint metric ONLY if the response is
# also content-correct (state WHAT happened AND WHY it meets the condition).
JUDGE_TASKS = {"event_narration", "sequential_step_instruction", "semantic_condition_alert"}

# ORDER MATTERS: _extract_position returns the FIRST substring hit, so every
# compound name must be tested before the bare "center" it contains. With the old
# alphabetical-ish order, "center-right" and "bottom-center" both matched "center"
# first and scored as the wrong cell -> 2 of the 9 cells were unwinnable, and both
# occur in the saved ETG ground truth. Sorted longest-first so this cannot regress.
_POSITIONS = sorted(
    ["top-left", "top-center", "top-right", "center-left", "center",
     "center-right", "bottom-left", "bottom-center", "bottom-right"],
    key=len, reverse=True)


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


_GENAI_JUDGE_PROMPT = (
    "You are evaluating whether a model's response correctly describes an event in a "
    "video, compared to the ground-truth answer.\n\n"
    "Question/Instruction: {question}\n"
    "Ground-truth answer: {gt}\n"
    "Model response: {pred}\n\n"
    "Score 1-5:\n"
    "- 5: Perfect - same event/information, accurate\n"
    "- 4: Good - mostly correct, minor differences\n"
    "- 3: Acceptable - right event, notable inaccuracies\n"
    "- 2: Poor - partially relevant but significantly wrong/vague\n"
    "- 1: Wrong - irrelevant or wrong event\n\n"
    'Respond with ONLY a JSON object: {{"score": <int 1-5>, "explanation": "<brief>"}}')


def _parse_judge_score(text: str):
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return int(json.loads(m.group()).get("score", 0))
    except Exception:
        m2 = re.search(r'"?score"?\s*[:=]\s*(\d)', text)
        return int(m2.group(1)) if m2 else None


class ContentJudge:
    """LLM-as-judge for free-text content. Priority: (1) google-genai SDK (works
    with a bare GEMINI_API_KEY, no base URL needed); (2) the OmniPro REST llm_judge
    (needs GEMINI_API_BASE); (3) lexical fallback. Correct if score >= 3 (paper).

    REPRODUCIBLE: the Gemini call is pinned with a fixed `seed`, and every verdict
    is persisted to a JSON cache keyed by (question, gt, pred) — re-scoring the
    same predictions always returns the same joint-F1 and costs zero API calls."""

    CACHE_PATH = os.environ.get(
        "OMNIPRO_JUDGE_CACHE",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_cache.json"))

    def __init__(self):
        self.mode = "lexical"
        self._judge = None
        self._genai = None
        self._model = None
        self._seed = int(os.environ.get("OMNIPRO_JUDGE_SEED", "1234"))
        self._cache = {}
        try:
            with open(self.CACHE_PATH) as f:
                self._cache = json.load(f)
        except Exception:
            self._cache = {}
        key = os.environ.get("GEMINI_API_KEY")

        # (1) preferred: google-genai SDK with the key passed explicitly
        if key:
            try:
                from google import genai
                self._genai = genai.Client(api_key=key)
                self._model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
                self.mode = "genai"
                print(f"[judge] google-genai judge active: model={self._model}", flush=True)
                return
            except Exception as e:
                print(f"[judge] genai SDK unavailable ({type(e).__name__}: {e}); "
                      f"trying REST judge", flush=True)

        # (2) fallback: OmniPro's REST llm_judge (loaded by path to avoid the
        # metrics-package shadowing; needs GEMINI_API_BASE/OPENAI_API_BASE set).
        if os.environ.get("OPENAI_API_KEY") or key:
            try:
                import importlib.util
                from utils import SCRATCH
                judge_path = os.environ.get(
                    "OMNIPRO_LLM_JUDGE",
                    os.path.join(SCRATCH, "omni_pro", "repo", "metrics", "llm_judge.py"))
                spec = importlib.util.spec_from_file_location("omnipro_llm_judge", judge_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self._judge = mod.LLMJudge()
                self.mode = "llm"
                print(f"[judge] REST LLM judge active: provider={self._judge.provider} "
                      f"model={self._judge.model} base={self._judge.api_base}", flush=True)
            except Exception as e:
                print(f"[judge] REST judge unavailable ({type(e).__name__}: {e}); "
                      f"using lexical fallback", flush=True)

    def _cache_key(self, question: str, gt: str, pred: str) -> str:
        import hashlib
        return hashlib.sha256(f"{question}\x1f{gt}\x1f{pred}".encode()).hexdigest()[:24]

    def _cache_put(self, key: str, score: float):
        self._cache[key] = score
        try:
            with open(self.CACHE_PATH, "w") as f:
                json.dump(self._cache, f)
        except Exception:
            pass

    def score(self, question: str, gt: str, pred: str) -> float:
        """Return correctness in [0,1] (1.0 if judged score >= 3, per the paper)."""
        k = self._cache_key(question, gt, pred)
        if k in self._cache:                        # persisted verdict -> stable re-scores
            return float(self._cache[k])
        if self.mode == "genai" and self._genai is not None:
            try:
                prompt = _GENAI_JUDGE_PROMPT.format(question=question, gt=gt, pred=pred)
                # pin `seed` so the judge itself is reproducible (per-user choice:
                # seed rather than temperature=0)
                try:
                    from google.genai import types
                    gcfg = types.GenerateContentConfig(seed=self._seed)
                except Exception:
                    gcfg = None
                r = self._genai.models.generate_content(
                    model=self._model, contents=prompt,
                    **({"config": gcfg} if gcfg is not None else {}))
                sc = _parse_judge_score(getattr(r, "text", "") or "")
                if sc is not None:
                    out = 1.0 if sc >= 3 else 0.0   # paper: score>=3 is correct
                    self._cache_put(k, out)
                    return out
            except Exception:
                pass
            return 1.0 if _lexical_sim(gt, pred) >= 0.3 else 0.0
        if self.mode == "llm" and self._judge is not None:
            try:
                r = self._judge.judge(question, gt, pred)
                out = 1.0 if int(r.get("score", 0)) >= 3 else 0.0   # paper: >=3
                self._cache_put(k, out)
                return out
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
