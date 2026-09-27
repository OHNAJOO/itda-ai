"""[김우진만 실행] 모든 사람의 정답 계획을 한 번에 만들어 plans/ 폴더에 나눠 저장함.
실행: python ml/data_gen/make_plans.py
결과: plans/S1.json ~ S4.json (학습용, 각 600개), plans/W_val.json (240개), plans/W_test.json (360개)
      목표 500 / 200 / 300보다 20% 넉넉히 만듦. 검수에서 떨어지는 것을 채우기 위함
"""
import json, random
from collections import Counter
from common import TYPES, HERE

TIME_EXPRS = ["어젯밤", "어제", "어제 저녁", "그저께", "오늘", "아까", "낮에", "저녁에", "새벽에",
              "새벽 3시쯤", "2일 전", "요즘", "며칠 전"]
DISTRACTORS = [
    "같은 질문을 여러 번 되풀이하심 (반복 질문)",
    "소변 실수를 하심 (실금)",
    "낮에 소파에서 한참 주무심 (낮잠)",
    "밥 양만 적음. 줄었다는 말 없이 (예: 저녁은 반 공기 드심)",
    "증상이 있었는지 스스로 묻는 의문문 (예: 밤에 좀 부스럭거리셨나?)",
    "보호자 자신이 힘들다는 말 (예: 나도 너무 지친다)",
    "약을 잘 챙겨 드심 (정상 복용)",
    "오늘 가족이 다녀감",
    "의사 선생님이 한 말을 옮김",
    "예전에 잘하시던 일을 떠올림 (오래된 과거 회상)",
]
SLOTS = [("S1", 600), ("S2", 600), ("S3", 600), ("S4", 600), ("W_val", 240), ("W_test", 360)]

def pick_types(rng, n, used):
    """적게 나온 유형이 먼저 뽑히도록 가중치를 줌 (유형 균형)"""
    pool = TYPES[:]
    chosen = []
    for _ in range(n):
        low = min(used[t] for t in pool)
        weights = [3.0 if used[t] <= low + 2 else 1.0 for t in pool]
        t = rng.choices(pool, weights=weights)[0]
        chosen.append(t); pool.remove(t); used[t] += 1
    return chosen

def make_plan(rng, used):
    r = rng.random()
    if r < 0.10:   n = 0                      # 잡담만 (빈 결과) 약 10%
    elif r < 0.40: n = 1                      # 사건 1개 약 30%
    elif r < 0.80: n = rng.choice([2, 3])     # 사건 2~3개 약 40%
    else:          n = rng.choice([2, 3, 4])  # 복합
    events = []
    for t in pick_types(rng, n, used):
        status = "absent" if rng.random() < 0.12 else "present"
        time_expr = rng.choice(TIME_EXPRS) if rng.random() < 0.35 else None
        count = rng.choice([1, 1, 1, 2, 3]) if status == "present" else 1
        events.append({"type": t, "status": status, "time_expr": time_expr, "count": count})
    distractors = [rng.choice(DISTRACTORS)] if rng.random() < 0.20 else []
    return {"events": events, "distractors": distractors}

if __name__ == "__main__":
    rng = random.Random(20260927)
    out = HERE / "plans"; out.mkdir(exist_ok=True)
    for slot, n in SLOTS:
        used = Counter({t: 0 for t in TYPES})
        plans = []
        for i in range(n):
            p = make_plan(rng, used)
            p["id"] = f"{slot}-{i:04d}"
            plans.append(p)
        json.dump(plans, open(out / f"{slot}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(slot, n, "유형별 등장:", dict(used))
