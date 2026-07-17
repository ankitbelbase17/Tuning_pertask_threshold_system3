import json, sys
from metrics import ContentJudge, aggregate, score_sample
judge = ContentJudge()
runs = {
 "V1 edge": "/iopsstor/scratch/cscs/dbartaula/system_3_wt_edge/omniprofast/output_sca_v1edge/online_pred.jsonl",
 "V2 evidence": "/iopsstor/scratch/cscs/dbartaula/system_3_wt_evidence/omniprofast/output_sca_v2evid/online_pred.jsonl",
 "V3 (+cal+2ex)": "/iopsstor/scratch/cscs/dbartaula/system_3_wt_evidence/omniprofast/output_sca_v3/online_pred.jsonl",
 "v2best(finegrid)": "/iopsstor/scratch/cscs/dbartaula/system_3_wt_evidence/omniprofast/output_v2best/online_pred.jsonl",
}
print(f"{'run':<18}{'emits':>6}{'tp':>4}{'fp':>4}{'fn':>4}{'timeP':>7}{'timeR':>7}{'timeF1':>8}{'jointF1':>8}{'contAcc':>8}")
for name, path in runs.items():
    rows = {}
    for l in open(path):
        r = json.loads(l); rows[r["id"]] = r
    per = [score_sample(r, tolerance=3.0, judge=judge) for r in rows.values()]
    o = aggregate(per)["overall"]
    tp = int(sum(p["tp_time"] for p in per)); fp = int(sum(p["fp"] for p in per)); fn = int(sum(p["fn"] for p in per))
    print(f"{name:<18}{int(o['n_emits']):>6}{tp:>4}{fp:>4}{fn:>4}{o['time_precision']:>7.3f}{o['time_recall']:>7.3f}{o['time_f1']:>8.3f}{o['joint_f1']:>8.3f}{o['content_acc']:>8.3f}")
