"""[전원] 문장화 결과를 코드로 검수하고, 통과한 것만 정답 형식으로 저장함. 눈 검사용 50건도 뽑음.
실행 예: python ml/data_gen/check.py --slot S1 --personas T01,T05 --target 500
결과: ok/S1.jsonl (통과한 것 중 앞에서부터 target개), ok/S1_report.txt, ok/S1_review.txt (눈 검사용 50건),
      ok/S1_failed.txt (떨어진 계획 번호)
"""
import argparse, json, random
from collections import Counter
from common import HERE, TYPES, KO, REDUCED_WORDS, assign_persona, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--slot", required=True)
ap.add_argument("--personas", required=True)
ap.add_argument("--target", type=int, required=True)   # S1~S4: 500, W_val: 200, W_test: 300
args = ap.parse_args()
personas = args.personas.split(",")

plans = json.load(open(HERE / "plans" / f"{args.slot}.json", encoding="utf-8"))
index = {p["id"]: i for i, p in enumerate(plans)}
raw = {}
for r in read_jsonl(HERE / "raw" / f"{args.slot}.jsonl"):
    raw.setdefault(r.get("id"), r)                     # 같은 id가 두 번 있으면 먼저 것을 씀

def check(plan, r):
    """통과하면 None, 아니면 떨어진 이유"""
    memo, ev, events = r.get("memo"), r.get("evidence"), plan["events"]
    if not isinstance(memo, str) or len(memo.strip()) < 5: return "메모 없음"
    if not isinstance(ev, list) or len(ev) != len(events): return "근거 개수 불일치"
    for e, span in zip(events, ev):
        if e["type"] not in TYPES: return "유형 코드 오류"
        if not isinstance(span, str) or not span.strip(): return "빈 근거"
        if span not in memo: return "근거가 메모에 없음"
        if e["time_expr"] and e["time_expr"] not in memo: return "시간 표현 누락"
        if e["type"] == "reduced_intake" and e["status"] == "present" and not any(w in span for w in REDUCED_WORDS):
            return "식사량 감소에 줄었다는 표현 없음"
    return None

def to_gold(plan, r):
    events = [dict(e, evidence=span) for e, span in zip(plan["events"], r["evidence"])]
    events.sort(key=lambda e: (r["memo"].find(e["evidence"]), TYPES.index(e["type"])))   # 원문 순서 (기획안 4-4)
    return {"events": events}

ok, reasons, seen, failed = [], Counter(), set(), []
for plan in plans:                                     # 계획 순서대로 확인
    r = raw.get(plan["id"])
    if r is None: reasons["아직 문장화 안 함"] += 1; continue
    why = check(plan, r)
    if why is None and r["memo"] in seen: why = "같은 메모 중복"
    if why:
        reasons[why] += 1
        if why != "아직 문장화 안 함": failed.append(plan["id"])
        continue
    seen.add(r["memo"])
    ok.append({"id": plan["id"], "persona": assign_persona(index[plan["id"]], personas),
               "memo": r["memo"], "gold": to_gold(plan, r)})

final = ok[:args.target]
out = HERE / "ok"; out.mkdir(exist_ok=True)
with open(out / f"{args.slot}.jsonl", "w", encoding="utf-8") as f:
    for row in final: f.write(json.dumps(row, ensure_ascii=False) + "\n")

types = Counter(e["type"] for row in final for e in row["gold"]["events"])
lines = [f"[{args.slot}] 통과 {len(ok)}개 / 저장 {len(final)}개 / 목표 {args.target}개",
         "떨어진 이유: " + (", ".join(f"{k} {v}" for k, v in reasons.most_common()) or "없음"),
         "유형별 사건 수: " + ", ".join(f"{KO[t]} {types[t]}" for t in TYPES),
         "문체별 메모 수: " + ", ".join(f"{k} {v}" for k, v in Counter(r['persona'] for r in final).items())]
if len(final) < args.target:
    lines.append(f"목표에 {args.target - len(final)}개 모자람 → 방법 A는 verbalize.py에 --redo를 붙여 떨어진 것만 다시 문장화, 방법 B는 남은 배치 파일을 계속 진행")
(out / f"{args.slot}_failed.txt").write_text("\n".join(failed), encoding="utf-8")   # verbalize.py --redo가 읽음
(out / f"{args.slot}_report.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

rng = random.Random(1)
review = rng.sample(final, min(50, len(final)))
with open(out / f"{args.slot}_review.txt", "w", encoding="utf-8") as f:
    for row in review:
        f.write(f"[{row['id']} / {row['persona']}]\n메모: {row['memo']}\n")
        for e in row["gold"]["events"]:
            st = "있었음" if e["status"] == "present" else "없었음"
            f.write(f"  - {KO[e['type']]} / {st} / {e['time_expr']} / {e['count']}회 / 근거: {e['evidence']}\n")
        if not row["gold"]["events"]: f.write("  - (사건 없음)\n")
        f.write("  판정: [ ] 맞음  [ ] 틀림 → 이유:\n\n")
print(f"눈 검사 파일: {out / (args.slot + '_review.txt')}")
