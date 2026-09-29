"""[김우진] 데모 시나리오 생성 (상세개발가이드 2-9). 같은 시드로 돌리면 항상 같은 파일이 나옴.
실행 (레포 맨 위): python ml/demo_gen/make_demo.py
결과: demo/demo_memos.json, demo/demo_events.json
확인: python ml/demo_gen/check_scenarios.py  (가이드 3-5 함수로 표시·주간 추이 top3를 미리 계산)

설계 (가상 환자 1명, 2026-05-21 ~ 2026-09-26)
- 진료일 5/21, 8/20 → 기준 구간 5/21~8/19(91일), 이번 구간 8/20~9/26(38일)
- 배회: 기준 구간에 드묾(4일) → 이번 구간 9일          → "증가"
- 환각: 9/7에 처음 나타나 9/12, 9/20까지 3일            → "새로 나타남"
- 야간 각성: 약 시작(9/10) 전까지 늘었다가 이후 줄어듦  → 주간 추이에서 약 시작선 뒤로 떨어짐
- 과민·짜증: 조금 늘지만 표시 기준 아래                 → 주간 추이 top3에만 들어감
- 낙상 1회(9/3), 약 시작 1건(9/10), 날짜 특정 불가("요즘") 사건 1건
- 기록하지 않은 날: 기준 구간 14일, 이번 구간 7일(9/14~9/23에는 9/17, 9/20, 9/23만 기록 → 회색 주)
- 메모는 T01 문체(50대 딸, "~하심"). 대부분 다음 날 아침에 전날 일을 쓰고, 사건 없는 날은 그날 저녁에 "별일 없었음"을 씀
"""
import json, random, re, datetime as dt
from pathlib import Path

D = dt.date
SEED = 20260929
rng = random.Random(SEED)
ROOT = Path.cwd()

START, VISIT2, END = D(2026, 5, 21), D(2026, 8, 20), D(2026, 9, 26)
MED_DATE = D(2026, 9, 10)
def drange(a, b): return [a + dt.timedelta(days=i) for i in range((b - a).days + 1)]

BASE = drange(START, VISIT2 - dt.timedelta(days=1))           # 91일
CUR = drange(VISIT2, END)                                       # 38일
CUR_MISSING = {D(2026, 9, d) for d in (14, 15, 16, 18, 19, 21, 22)}
BASE_MISSING = set(rng.sample([d for d in BASE if d != START], 14))
BASE_REC = [d for d in BASE if d not in BASE_MISSING]           # 77일
CUR_REC = [d for d in CUR if d not in CUR_MISSING]              # 31일
CUR_PRE = [d for d in CUR_REC if d < MED_DATE]                  # 21일
CUR_POST = [d for d in CUR_REC if d >= MED_DATE]                # 10일

# ---------- 날짜별 사건 배치 ----------
plan = {d: [] for d in BASE_REC + CUR_REC}                      # 날짜 → [(type, status, count)]
def put(days, t, k, status="present", counts=(1, 1, 1, 2)):
    for d in rng.sample(days, k):
        plan[d].append((t, status, rng.choice(counts) if status == "present" else 1))

# 기준 구간 (발생일 수)
put([d for d in BASE_REC if d < D(2026, 7, 31)], "wandering_exit", 4)   # 기준 구간 앞쪽에만 드물게
put(BASE_REC, "irritability", 8)
put(BASE_REC, "night_waking", 19, counts=(1, 1, 2, 2, 3))
put([d for d in BASE_REC if not any(t == "night_waking" for t, _, _ in plan[d])], "night_waking", 2, status="absent")
put(BASE_REC, "reduced_intake", 6)
put(BASE_REC, "anxiety", 5)
put(BASE_REC, "confusion", 6)
put(BASE_REC, "medication_refusal", 3, counts=(1,))
put(BASE_REC, "low_mood_apathy", 4, counts=(1,))
put(BASE_REC, "delusion", 3, counts=(1,))
put(BASE_REC, "agitation", 2, counts=(1,))

# 이번 구간
last = CUR_REC[-1]                                              # 9/26은 그날 저녁에 쓰므로 밤 사건을 두지 않음
put([d for d in CUR_REC if d != last], "wandering_exit", 9, counts=(1, 1, 2))
put(CUR_REC, "irritability", 7)
put(CUR_PRE, "night_waking", 11, counts=(1, 2, 2, 3))
plan[D(2026, 9, 11)].append(("night_waking", "present", 1))
for d in (D(2026, 9, 13), D(2026, 9, 17), D(2026, 9, 24)):
    plan[d].append(("night_waking", "absent", 1))
for d in (D(2026, 9, 7), D(2026, 9, 12), D(2026, 9, 20)):
    plan[d].append(("hallucination", "present", 1))
plan[D(2026, 9, 3)].append(("fall", "present", 1))
put(CUR_REC, "reduced_intake", 2)
put(CUR_REC, "anxiety", 2)
put(CUR_REC, "confusion", 3)
put(CUR_REC, "medication_refusal", 1, counts=(1,))
put(CUR_REC, "low_mood_apathy", 1, counts=(1,))
put(CUR_REC, "delusion", 1, counts=(1,))
put(CUR_REC, "agitation", 1, counts=(1,))
for d in plan:                                                  # 같은 날 같은 유형이 두 번 뽑히면 하나만 둠
    seen, uniq = set(), []
    for e in plan[d]:
        if e[0] not in seen: seen.add(e[0]); uniq.append(e)
    plan[d] = uniq

# ---------- 문장 (T01 문체) ----------
COUNT_WORD = {2: "두 번", 3: "세 번이나"}
BODIES = {
    "night_waking": {1: ["깨셔서 한참 못 주무심", "깨셔서 다시 주무시기까지 한 시간 넘게 걸림", "깨셔서 불 켜 놓고 앉아 계심"],
                     "n": ["{c} 깨심", "{c} 깨셔서 나도 잠을 설침"],
                     "absent": ["엔 한 번도 안 깨고 푹 주무심", "엔 안 깨시고 아침까지 주무심"]},
    "wandering_exit": {1: ["현관문 열고 나가시려는 걸 말림", "신발 신고 나가신다고 해서 한참 달램", "집 안을 계속 서성이심",
                           "아파트 복도로 나가셔서 모시고 들어옴"],
                       "n": ["현관문 쪽으로 {c} 나가시려 함"]},
    "irritability": {1: ["옷 갈아입혀 드리는데 괜히 짜증 내심", "리모컨 못 찾으신다고 버럭 화내심", "국이 짜다고 역정 내심",
                         "사소한 걸로 계속 신경질 내심"],
                     "n": ["별것 아닌 일로 {c} 짜증 내심"]},
    "anxiety": {1: ["내가 잠깐 나가려니 혼자 있기 싫다고 불안해하심", "누가 문 두드릴까 봐 무섭다고 하심", "통장 걱정을 계속 하심"],
                "n": ["혼자 있기 무섭다고 {c} 전화하심"]},
    "low_mood_apathy": {1: ["하루 종일 누워만 계시고 아무것도 안 하려고 하심", "좋아하시던 트로트 틀어 드려도 관심 없으심",
                            "기운 없이 멍하니 앉아 계심"]},
    "delusion": {1: ["지갑을 누가 가져갔다고 하심", "며느리가 반찬에 뭘 넣었다고 의심하심"]},
    "hallucination": {1: ["창밖에 누가 서 있다고 하심", "방에 모르는 아이들이 있다고 하심", "벽에 벌레가 기어다닌다고 하심"]},
    "reduced_intake": {1: ["평소보다 식사를 절반도 안 드심", "죽을 몇 숟갈만 드시고 마심", "밥 안 드신다고 숟가락 내려놓으심"],
                       "n": ["평소보다 적게 드시고 두 끼를 거의 남기심"]},
    "medication_refusal": {1: ["약 안 드신다고 뱉으심", "약은 나중에 먹겠다고 계속 미루심"]},
    "confusion": {1: ["나를 언니라고 부르심", "여기가 어디냐고 집에 가자고 하심", "지금이 아침인지 저녁인지 헷갈려하심"],
                  "n": ["나를 {c} 언니라고 부르심"]},
    "agitation": {1: ["목욕시켜 드리는데 소리 지르시고 밀치심"]},
    "fall": {1: ["화장실 가시다가 미끄러져 넘어지심"]},
}
NIGHT_TIME = ["어젯밤", "어젯밤", "새벽에", "새벽 3시쯤", "새벽 두 시쯤"]
# "어제 낮에"처럼 날짜 표현 두 개가 겹치는 말은 쓰지 않음 (같은 길이면 "낮에"가 먼저 걸려 당일로 계산됨)
DAY_TIME_NEXT = {"default": ["어제", "어제", "어제 저녁", "어제 오후", "어제 아침", "어제 점심때"],
                 "hallucination": ["어젯밤", "어제 저녁"], "fall": ["어제", "어제 오후"]}
DAY_TIME_SAME = [None, None, "오늘", "아까", "저녁에", "낮에"]
QUIET = ["오늘은 별일 없었음", "오늘은 무난하게 지나감. 점심도 잘 드심", "주간보호센터 다녀오심. 별일 없으셨음",
         "오늘은 산책 30분 같이 함. 별일 없음", "오늘은 조용히 지나감", "오늘은 컨디션 괜찮으셨음",
         "저녁 잘 드시고 일찍 주무심", "오늘은 복지관 프로그램 다녀오심. 별일 없음", "손주 전화 받고 좋아하심. 오늘은 무난함",
         "오늘은 빨래 개는 거 같이 하심. 별일 없음"]
EXTRA = ["약은 잘 챙겨 드심", "나도 좀 지친다", "오후엔 기분 좋으셨음", "낮에 소파에서 한참 주무심", "오빠네가 다녀감"]

def clause(t, status, count, time_expr):
    b = BODIES[t]
    if status == "absent":
        body = rng.choice(b["absent"]); return f"{time_expr}{body}"      # "어젯밤" + "엔 한 번도 ..."
    if count > 1 and "n" in b:
        body = rng.choice(b["n"]).format(c=COUNT_WORD[count])
    else:
        body = rng.choice(b[1])
    return f"{time_expr} {body}" if time_expr else body

# ---------- 날짜 계산 (가이드 3-3과 같은 규칙) ----------
DATE_EXPR = {"오늘 새벽": -1, "오늘": 0, "아까": 0, "낮에": 0, "저녁에": 0, "어제": -1, "어젯밤": -1, "어제 저녁": -1,
             "그저께": -2, "그제": -2, "새벽": -1}
def resolve(time_expr, written):
    if not time_expr: return written, False
    if "요즘" in time_expr: return None, True
    m = re.search(r"(\d+)일 ?전", time_expr)
    if m: return written - dt.timedelta(days=int(m.group(1))), False
    for e in sorted(DATE_EXPR, key=len, reverse=True):
        if e in time_expr: return written + dt.timedelta(days=DATE_EXPR[e]), False
    return None, True

# ---------- 메모 만들기 ----------
memos, events = [], []
def add_memo(written, parts):
    """parts: [(clause_text, event_or_None)]"""
    mid = len(memos) + 1
    text = ". ".join(p for p, _ in parts) + "."
    memos.append({"id": mid, "written_date": written.isoformat(), "text": text})
    for p, ev in parts:
        if ev is None: continue
        t, status, count, time_expr = ev
        ed, unknown = resolve(time_expr, written)
        events.append({"memo_id": mid, "event_date": ed.isoformat() if ed else None, "date_unknown": unknown,
                       "type": t, "status": status, "time_expr": time_expr, "count": count, "evidence": p})
    return mid

undated_done = False
for d in sorted(plan):
    evs = plan[d]
    if not evs:
        add_memo(d, [(rng.choice(QUIET), None)])
        continue
    night = any(t == "night_waking" for t, _, _ in evs)
    next_morning = d != last and (night or rng.random() < 0.8)
    written = d + dt.timedelta(days=1) if next_morning else d
    parts = []
    order = sorted(evs, key=lambda e: e[0] != "night_waking")    # 밤 일을 먼저 씀
    for t, status, count in order:
        if t == "night_waking":
            te = "어젯밤" if status == "absent" else rng.choice(NIGHT_TIME if count == 1 else ["어젯밤", "새벽에"])  # 여러 번 깬 날은 시각을 적지 않음
        elif next_morning:
            te = rng.choice(DAY_TIME_NEXT.get(t, DAY_TIME_NEXT["default"]))
        else:
            te = rng.choice(DAY_TIME_SAME)
        parts.append((clause(t, status, count, te), (t, status, count, te)))
    if written == D(2026, 9, 25) and not undated_done:            # 날짜 특정 불가 사건 1건
        parts.append(("요즘 부쩍 기운이 없으시고 말씀도 잘 안 하심", ("low_mood_apathy", "present", 1, "요즘")))
        undated_done = True
    if rng.random() < 0.2:
        parts.append((rng.choice(EXTRA), None))
    add_memo(written, parts)

memos.sort(key=lambda m: (m["written_date"], m["id"]))              # 작성일 순으로 번호를 다시 매김
remap = {m["id"]: i + 1 for i, m in enumerate(memos)}
for m in memos: m["id"] = remap[m["id"]]
for e in events: e["memo_id"] = remap[e["memo_id"]]
events.sort(key=lambda e: (e["memo_id"], e["event_date"] or ""))

# 검산: evidence가 메모 원문에 그대로 있고, 사건 날짜가 계획과 같아야 함
by_id = {m["id"]: m for m in memos}
for e in events:
    assert e["evidence"] in by_id[e["memo_id"]]["text"], e
    if e["time_expr"]: assert e["time_expr"] in e["evidence"], e
planned = sorted((d.isoformat(), t, s) for d, evs in plan.items() for t, s, _ in evs)
made = sorted((e["event_date"], e["type"], e["status"]) for e in events if not e["date_unknown"])
assert planned == made, "계획과 사건 날짜가 다름"

demo_events = {
    "patient": {"alias": "어머니"},
    "visits": [START.isoformat(), VISIT2.isoformat()],
    "medications": [{"name": "트라조돈 25mg", "change_type": "start", "change_date": MED_DATE.isoformat()}],
    "questions": [{"text": "약 시작하고 밤에 깨시는 게 줄었는데, 계속 같은 용량으로 드셔도 되는지", "created_at": "2026-09-18"},
                  {"text": "창밖에 사람이 있다고 하실 때 어떻게 대하면 좋은지", "created_at": "2026-09-21"}],
    "as_of": END.isoformat(),
    "events": events,
}
(ROOT / "demo").mkdir(exist_ok=True)
json.dump(memos, open(ROOT / "demo" / "demo_memos.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(demo_events, open(ROOT / "demo" / "demo_events.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"메모 {len(memos)}건, 사건 {len(events)}건 (기준 기록 {len(BASE_REC)}일, 이번 기록 {len(CUR_REC)}일)")
