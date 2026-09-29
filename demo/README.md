# 데모·평가 시나리오 (상세개발가이드 2-9)

itda-ai의 `ml/demo_gen/`에서 만든 파일임. 시드가 고정되어 있어 다시 돌려도 같은 내용이 나옴.

## demo/demo_memos.json

메모 108건. `[{id, written_date, text}]`

- /demo/load에서 memos 테이블에 그대로 넣음. status는 "confirmed", written_date는 파일 값 그대로 씀
- 대부분 다음 날 아침에 전날 일을 쓴 메모임("어젯밤", "어제 저녁" 등). 사건이 없는 날은 그날 쓴 "별일 없었음" 메모가 있음

## demo/demo_events.json

```
{ "patient": {"alias"}, "visits": [날짜...], "medications": [{name, change_type, change_date}],
  "questions": [{text, created_at}], "as_of": "2026-09-26",
  "events": [{memo_id, event_date, date_unknown, type, status, time_expr, count, evidence}] }
```

- events는 events 테이블에 source="model"로 넣음. memo_id는 demo_memos.json의 id임
- event_date는 가이드 3-3 규칙으로 계산해 둔 값임. 다시 계산하지 않고 그대로 넣음
- visits, medications, questions도 각 테이블에 넣음
- 요약지 기준일(as_of)은 2026-09-26으로 두고 봄

## 가이드 3-5 함수로 미리 계산한 결과 (ml/demo_gen/check_scenarios.py)

| 항목 | 값 |
|---|---|
| 기록일 | 기준 구간 77/91일, 이번 구간 31/38일 |
| 배회·출입문 시도 | 기준 5.2% → 이번 29.0%, **증가** |
| 환각 | 기준 0% → 이번 9.7% (9/7 처음), **새로 나타남** |
| 주간 추이 top3 | 배회, 야간 각성(약 시작 9/10 뒤로 떨어짐), 과민·짜증 |
| 회색 주 | 9/17~9/23 (기록 3일). 월요일 시작으로 나누면 9/14~9/20 (2일) |
| 낙상 | 9/3 |
| 날짜 특정 불가 사건 | 1건 ("요즘", 9/25 메모) |

- 확인 필요: 낙상도 기준 구간에 0건이라 mark_for를 그대로 돌리면 "새로 나타남"이 붙음. 낙상은 목록으로 따로 보여 주는 유형이므로 표시 계산에서 빼는지 백엔드에서 정해야 함

## ml/eval/signal_scenarios.json (평가 ③)

환자 40명. 변화를 심은 SIG-C01~20, 심지 않은 SIG-N01~20.

```
[{patient_id, planted, planted_changes: [{type, kind: increase|new, baseline_rate, current_rate}],
  visits: [기준 시작, 이번 시작], as_of, recorded_days: [날짜...],
  events: [{type, event_date, status, count}], generator_rates}]
```

- 메모가 없으므로 기록일은 recorded_days를 그대로 씀
- 탐지율: planted_changes의 유형에 increase 또는 new가 붙은 비율
- 오경보율: SIG-N 환자에게서 한 번이라도 나온 유형에 표시가 붙은 비율
- generator_rates는 생성할 때 쓴 확률임. 실제 발생 비율은 날마다 무작위로 뽑아서 조금씩 다름
- 참고값 (같은 함수로 계산): sigma 3에서 탐지율 13/20(65%), 오경보율 3/76(3.9%). sigma 2에서 탐지율 17/20(85%), 오경보율 5/76(6.6%)
