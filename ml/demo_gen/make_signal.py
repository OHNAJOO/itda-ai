"""[김우진] 평가 ③(증가 표시 탐지율·오경보율)용 시나리오 생성 (상세개발가이드 2-9, 3-8). 시드 고정.
실행 (레포 맨 위): python ml/demo_gen/make_signal.py
결과: ml/eval/signal_scenarios.json  → itda-backend 레포 eval/ 로 전달

- 환자 40명: 변화를 심은 20명(SIG-C01~20), 심지 않은 20명(SIG-N01~20). 메모 문장은 만들지 않고 사건과 기록일만 둠
- 환자마다 기준 구간 70~100일, 이번 구간 28~45일, 기록률 70~95%(날마다 무작위)
- 평소 나타나는 유형 3~5개, 유형별 발생일 비율 5~35% (날마다 무작위라 실제 비율은 흔들림)
- 변화를 심은 환자: 유형 하나를 바꿈
    increase(약 70%): 이번 구간 비율 = 기준 비율 + 15~30%p
    new(약 30%): 기준 구간에 없던 유형이 이번 구간에 10~25% 비율로 나타남
- 낙상은 넣지 않음 (비율과 관계없이 목록으로 보여 주는 유형이라 평가 ③ 대상이 아님)

백엔드 사용법: 메모가 없으므로 기록일은 recorded_days를 그대로 씀. visits로 두 구간을 나누고,
유형마다 mark_for(3-5)를 돌려 planted_changes의 유형에 increase/new가 붙은 비율(탐지율),
N 환자의 유형(어느 구간에서든 한 번이라도 나온 유형)에 표시가 붙은 비율(오경보율)을 셈. sigma 3과 2로 두 번.
"""
import json, random, datetime as dt
from pathlib import Path

rng = random.Random(20260930)
TYPES = ["night_waking", "wandering_exit", "agitation", "irritability", "anxiety", "low_mood_apathy",
         "delusion", "hallucination", "reduced_intake", "medication_refusal", "confusion"]

def make_patient(pid, planted):
    base_len, cur_len = rng.randint(70, 100), rng.randint(28, 45)
    start = dt.date(2026, 1, 5) + dt.timedelta(days=rng.randint(0, 60))
    visit2 = start + dt.timedelta(days=base_len)
    end = visit2 + dt.timedelta(days=cur_len - 1)
    cov = rng.uniform(0.70, 0.95)
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    rec = [d for d in days if rng.random() < cov]
    active = rng.sample(TYPES, rng.randint(3, 5))
    base_rate = {t: round(rng.uniform(0.05, 0.35), 3) for t in active}
    cur_rate = dict(base_rate)
    changes = []
    if planted:
        if rng.random() < 0.7:
            t = rng.choice(active)
            cur_rate[t] = round(min(0.8, base_rate[t] + rng.uniform(0.15, 0.30)), 3)
            changes.append({"type": t, "kind": "increase", "baseline_rate": base_rate[t], "current_rate": cur_rate[t]})
        else:
            t = rng.choice([x for x in TYPES if x not in active])
            base_rate[t] = 0.0; cur_rate[t] = round(rng.uniform(0.10, 0.25), 3)
            changes.append({"type": t, "kind": "new", "baseline_rate": 0.0, "current_rate": cur_rate[t]})
    events = []
    for d in rec:
        rates = base_rate if d < visit2 else cur_rate
        for t, r in rates.items():
            if rng.random() < r:
                events.append({"type": t, "event_date": d.isoformat(), "status": "present", "count": 1})
    return {"patient_id": pid, "planted": planted, "planted_changes": changes,
            "visits": [start.isoformat(), visit2.isoformat()], "as_of": end.isoformat(),
            "recorded_days": [d.isoformat() for d in rec], "events": events,
            "generator_rates": {"baseline": base_rate, "current": cur_rate}}

pts = [make_patient(f"SIG-C{i:02d}", True) for i in range(1, 21)] + \
      [make_patient(f"SIG-N{i:02d}", False) for i in range(1, 21)]
out = Path.cwd() / "ml" / "eval"; out.mkdir(parents=True, exist_ok=True)
json.dump(pts, open(out / "signal_scenarios.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"환자 {len(pts)}명, 사건 {sum(len(p['events']) for p in pts)}건 → ml/eval/signal_scenarios.json")
