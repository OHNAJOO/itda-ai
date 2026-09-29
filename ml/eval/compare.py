"""ml/eval/results/의 평가 결과를 한 표로 모음. 지표를 건별 기록(records)에서 다시 계산하므로
노트북에서 잰 결과(eval_before_test_*.json 등)와 Ollama 결과를 같은 기준으로 비교할 수 있음.
실행 (레포 맨 위): python ml/eval/compare.py
결과: 화면 표 + ml/eval/results/compare.csv, compare.md
"""
import csv, glob, json
from collections import Counter
from pathlib import Path

def parse_events(text):
    import re
    t = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.S).replace("```json", "").replace("```", "")
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b < 0: return None
    try:
        obj = json.loads(t[a:b + 1]); ev = obj.get("events") if isinstance(obj, dict) else None
        return ev if isinstance(ev, list) else None
    except Exception:
        return None

def score(records):
    tp = fp = fn = ok = exact = e_tot = e_ok = 0
    for r in records:
        pred = parse_events(r["pred_raw"]); good = pred is not None; ok += good
        pred = [e for e in (pred or []) if isinstance(e, dict)]
        g = Counter((e["type"], e["status"]) for e in r["gold"]); p = Counter((e.get("type"), e.get("status")) for e in pred)
        h = sum((g & p).values()); tp += h; fp += sum(p.values()) - h; fn += sum(g.values()) - h
        exact += int(good and g == p)
        if not r["gold"]: e_tot += 1; e_ok += int(good and not pred)
    n = len(records); P = tp / (tp + fp) if tp + fp else 0; R = tp / (tp + fn) if tp + fn else 0
    return {"precision": P, "recall": R, "f1": 2 * P * R / (P + R) if P + R else 0,
            "exact_match_rate": exact / n, "json_ok_rate": ok / n, "empty_correct_rate": e_ok / e_tot if e_tot else None}

rows = []
for path in sorted(glob.glob("ml/eval/results/eval_*.json")):
    d = json.load(open(path, encoding="utf-8")); s = d["summary"]; recs = d.get("records") or []
    m = score(recs) if recs and "pred_raw" in recs[0] else {}
    rows.append({
        "결과 파일": Path(path).stem.removeprefix("eval_"),
        "모델": s.get("model", "(노트북)"), "장치": s.get("device", "gpu(노트북)"), "스키마": s.get("schema", "none"),
        "건수": s.get("n"), "GPU비율": s.get("gpu_fraction"),
        "정밀도": m.get("precision", s.get("precision")), "재현율": m.get("recall", s.get("recall")),
        "F1": m.get("f1", s.get("f1")), "메모 완전일치": m.get("exact_match_rate", s.get("exact_match_rate")),
        "JSON 통과": m.get("json_ok_rate", s.get("json_ok_rate")),
        "빈 결과 정확": m.get("empty_correct_rate", s.get("empty_correct_rate")),
        "시간표현 일치": s.get("time_expr_match_rate"), "근거 원문 포함": s.get("evidence_in_memo_rate"),
        "초/건 평균": s.get("sec_per_memo"), "초/건 최대": s.get("sec_max"),
    })

def fmt(v):
    if v is None: return "-"
    if isinstance(v, float): return f"{v:.4f}" if v <= 1 else f"{v:.2f}"
    return str(v)

cols = list(rows[0].keys()) if rows else []
md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
md += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
print("\n".join(md))
Path("ml/eval/results/compare.md").write_text("\n".join(md) + "\n", encoding="utf-8")
with open("ml/eval/results/compare.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
print("\n저장: ml/eval/results/compare.md, compare.csv")
