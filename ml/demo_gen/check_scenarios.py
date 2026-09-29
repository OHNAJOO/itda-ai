"""[김우진] 2-9 시나리오 검산. 가이드 3-5의 함수를 그대로 옮겨 와서 백엔드가 낼 결과를 미리 계산함.
실행 (레포 맨 위): python ml/demo_gen/check_scenarios.py
- 데모: 유형별 기준/이번 비율, 표시(mark), 주간 추이 top3, 회색 주
- 평가 ③: sigma 3과 2에서 탐지율·오경보율 (백엔드 결과와 대조용 참고값)
"""
import json, math, datetime as dt
from collections import namedtuple
from pathlib import Path

ROOT = Path.cwd()
CFG = {"min_recorded_days": 14, "min_event_days": 3, "sigma": 3, "trend_top_n": 3, "trend_min_days": 3,
       "low_coverage_week_days": 4}
P = lambda s: dt.date.fromisoformat(s)

# ---- 가이드 3-5 함수 (그대로) ----
def recorded_days(memos, events, start, end):
    days = {e.event_date for e in events if e.event_date}
    memos_with_events = {e.memo_id for e in events}
    days |= {m.written_date for m in memos if m.status == "confirmed" and m.id not in memos_with_events}
    return {d for d in days if start <= d <= end}

def occurrence_days(events, t, start, end):
    return {e.event_date for e in events
            if e.type == t and e.status == "present" and e.event_date and start <= e.event_date <= end}

def mark_for(t, base_rate, cur_days, n_cur, has_baseline, cfg):
    if not has_baseline: return "not_comparable"
    if n_cur < cfg["min_recorded_days"]: return "insufficient"
    if base_rate == 0 and cur_days > 0: return "new"
    upper = base_rate + cfg["sigma"] * math.sqrt(base_rate * (1 - base_rate) / n_cur)
    if cur_days / n_cur > upper and cur_days >= cfg["min_event_days"]: return "increase"
    return None

def pick_trend_types(rows, cfg):
    cand = [r for r in rows if max(r["cur_days"], r.get("base_days") or 0) >= cfg["trend_min_days"]]
    def score(r):
        if r["baseline_rate"] is None: return r["current_rate"]
        return abs(r["current_rate"] - r["baseline_rate"])
    return [r["type"] for r in sorted(cand, key=score, reverse=True)[:cfg["trend_top_n"]]]

# ---- 데모 ----
M = namedtuple("M", "id written_date status"); E = namedtuple("E", "memo_id event_date type status count")
memos_raw = json.load(open(ROOT / "demo" / "demo_memos.json", encoding="utf-8"))
dem = json.load(open(ROOT / "demo" / "demo_events.json", encoding="utf-8"))
memos = [M(m["id"], P(m["written_date"]), "confirmed") for m in memos_raw]
events = [E(e["memo_id"], P(e["event_date"]) if e["event_date"] else None, e["type"], e["status"], e["count"])
          for e in dem["events"]]
v1, v2 = map(P, dem["visits"]); as_of = P(dem["as_of"])
bs, be, cs, ce = v1, v2 - dt.timedelta(days=1), v2, as_of
rb, rc = recorded_days(memos, events, bs, be), recorded_days(memos, events, cs, ce)
print(f"[데모] 메모 {len(memos)}건, 사건 {len(events)}건 | 기준 {bs}~{be} 기록 {len(rb)}/{(be - bs).days + 1}일"
      f" | 이번 {cs}~{ce} 기록 {len(rc)}/{(ce - cs).days + 1}일")
rows = []
for t in sorted({e.type for e in events}):
    b, c = len(occurrence_days(events, t, bs, be)), len(occurrence_days(events, t, cs, ce))
    br, cr = b / len(rb), c / len(rc)
    wk = sum(e.count for e in events if e.type == t and e.status == "present" and e.event_date and cs <= e.event_date <= ce) / len(rc) * 7
    rows.append({"type": t, "baseline_rate": br, "current_rate": cr, "base_days": b, "cur_days": c,
                 "mark": mark_for(t, br, c, len(rc), True, CFG), "weekly": round(wk, 1)})
print(f"  {'유형':20}{'기준':>8}{'이번':>8}{'주당':>6}  표시")
for r in sorted(rows, key=lambda r: -r["current_rate"]):
    print(f"  {r['type']:20}{r['baseline_rate']:>8.1%}{r['current_rate']:>8.1%}{r['weekly']:>6}  {r['mark'] or ''}")
top = pick_trend_types(rows, CFG)
print("  주간 추이 top3:", top)
print("  낙상:", sorted(str(e.event_date) for e in events if e.type == "fall"),
      "| 날짜 특정 불가 사건:", sum(1 for e in dem["events"] if e["date_unknown"]))
alld = rb | rc
for anchor, name in [(bs, "기준 구간 시작 기준 주"), (bs - dt.timedelta(days=bs.weekday()), "월요일 시작 주")]:
    gray, k = [], 0
    while anchor + dt.timedelta(days=7 * k) <= ce:
        ws = anchor + dt.timedelta(days=7 * k); we = ws + dt.timedelta(days=6)
        n = sum(1 for d in alld if ws <= d <= we)
        if n < CFG["low_coverage_week_days"] and ws >= bs and we <= ce: gray.append(f"{ws}({n}일)")
        k += 1
    print(f"  회색 주 ({name}):", gray)
for t in top:
    s, line = bs, []
    while s <= ce:
        e_ = s + dt.timedelta(days=6)
        n = sum(1 for d in alld if s <= d <= e_); o = len(occurrence_days(events, t, s, e_))
        line.append(f"{s.month}/{s.day}:{(o / n if n else 0):.0%}" + ("*" if n < 4 else ""))
        s += dt.timedelta(days=7)
    print(f"  {t} 주별 (*=회색): " + " ".join(line[-10:]))

# ---- 평가 ③ ----
pts = json.load(open(ROOT / "ml" / "eval" / "signal_scenarios.json", encoding="utf-8"))
for sigma in (3, 2):
    cfg = dict(CFG, sigma=sigma); hit = tot = fa = fa_tot = 0
    for p in pts:
        v1, v2 = map(P, p["visits"]); end = P(p["as_of"])
        rec = {P(d) for d in p["recorded_days"]}
        evs = [E(None, P(e["event_date"]), e["type"], e["status"], e["count"]) for e in p["events"]]
        rb = {d for d in rec if v1 <= d < v2}; rc = {d for d in rec if v2 <= d <= end}
        planted = {c["type"] for c in p["planted_changes"]}
        for t in sorted({e.type for e in evs} | planted):
            b = len(occurrence_days(evs, t, v1, v2 - dt.timedelta(days=1)))
            c = len(occurrence_days(evs, t, v2, end))
            m = mark_for(t, b / len(rb), c, len(rc), True, cfg)
            flagged = m in ("increase", "new")
            if t in planted: tot += 1; hit += flagged
            elif not p["planted"]: fa_tot += 1; fa += flagged
    print(f"[평가 ③ 참고값] sigma {sigma}: 탐지율 {hit}/{tot} = {hit / tot:.0%}, 오경보율 {fa}/{fa_tot} = {fa / fa_tot:.1%}")
